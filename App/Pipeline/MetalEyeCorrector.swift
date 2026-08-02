import Metal
import CoreVideo
import AspectusKit

/// resamples the original eye pixels through the geometric eyeball warp on the GPU
///
/// every failure path returns the untouched input frame rather than a degraded one, so the worst
/// case is always original-frame passthrough; timestamps and the frame header travel unchanged
actor MetalEyeCorrector: EyeCorrector {
    typealias Pixels = CVReadyFrame

    private struct Uniforms {
        var leftCenter: SIMD2<Float>
        var rightCenter: SIMD2<Float>
        var leftRadius: SIMD2<Float>
        var rightRadius: SIMD2<Float>
        var leftDisplacement: SIMD2<Float>
        var rightDisplacement: SIMD2<Float>
        var irisPlateau: Float
    }

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState
    private let textureCache: CVMetalTextureCache
    private let tuning: EyeWarpTuning

    private var pool: CVPixelBufferPool?
    private var poolWidth = 0
    private var poolHeight = 0

    /// headroom over the single in-flight frame plus whatever the preview still holds
    private static let maxPooledBuffers = 6

    /// keeps Metal out of the orchestrator, which only needs to know whether a corrector exists
    init?(tuning: EyeWarpTuning = .init()) {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        self.init(device: device, tuning: tuning)
    }

    init?(device: MTLDevice, tuning: EyeWarpTuning = .init()) {
        guard let commandQueue = device.makeCommandQueue(),
              let library = try? device.makeDefaultLibrary(bundle: .main),
              let vfn = library.makeFunction(name: "warp_vertex"),
              let ffn = library.makeFunction(name: "eye_warp_fragment") else { return nil }

        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = vfn
        desc.fragmentFunction = ffn
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        guard let pso = try? device.makeRenderPipelineState(descriptor: desc) else { return nil }

        var cache: CVMetalTextureCache?
        guard CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache) == kCVReturnSuccess,
              let cache else { return nil }

        self.device = device
        self.commandQueue = commandQueue
        self.pipeline = pso
        self.textureCache = cache
        self.tuning = tuning
    }

    func correct(_ pixels: CVReadyFrame,
                 tracking: TrackingResult,
                 request: CorrectionRequest,
                 header: FrameHeader) async -> CVReadyFrame {
        guard request.strength > 0, header.width > 0, header.height > 0 else { return pixels }

        let aspect = Double(header.width) / Double(header.height)
        guard let warps = GazeGeometry.warps(tracking, imageAspect: aspect,
                                             request: request,
                                             tuning: tuning) else { return pixels }

        guard let output = makeOutputBuffer(width: header.width, height: header.height),
              let source = makeTexture(from: pixels.pixelBuffer),
              let destination = makeTexture(from: output),
              let sourceTexture = CVMetalTextureGetTexture(source),
              let destinationTexture = CVMetalTextureGetTexture(destination) else {
            return pixels
        }

        // carries colour space and any other propagating attachments to the corrected frame
        CVBufferPropagateAttachments(pixels.pixelBuffer, output)

        var uniforms = Uniforms(
            leftCenter: simd(warps.left.center), rightCenter: simd(warps.right.center),
            leftRadius: simd(warps.left.radius), rightRadius: simd(warps.right.radius),
            leftDisplacement: simd(warps.left.displacement),
            rightDisplacement: simd(warps.right.displacement),
            irisPlateau: Float(tuning.irisPlateau))

        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = destinationTexture
        rpd.colorAttachments[0].loadAction = .dontCare      // every texel is written by the shader
        rpd.colorAttachments[0].storeAction = .store

        guard let cmd = commandQueue.makeCommandBuffer(),
              let enc = cmd.makeRenderCommandEncoder(descriptor: rpd) else { return pixels }
        enc.setRenderPipelineState(pipeline)
        enc.setFragmentTexture(sourceTexture, index: 0)
        enc.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
        enc.endEncoding()

        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            cmd.addCompletedHandler { _ in cont.resume() }
            cmd.commit()
        }

        // an MTLTexture is only valid while its CVMetalTexture lives, and the GPU read is not
        // finished until the await above returns
        withExtendedLifetime(source) {}
        withExtendedLifetime(destination) {}

        return CVReadyFrame(header: header, pixelBuffer: output)
    }

    private func simd(_ p: NormPoint) -> SIMD2<Float> { SIMD2(Float(p.x), Float(p.y)) }

    private func makeOutputBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        if pool == nil || poolWidth != width || poolHeight != height {
            let attributes: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferMetalCompatibilityKey as String: true,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:] as CFDictionary,
            ]
            var created: CVPixelBufferPool?
            guard CVPixelBufferPoolCreate(kCFAllocatorDefault, nil,
                                          attributes as CFDictionary, &created) == kCVReturnSuccess else {
                return nil
            }
            pool = created
            poolWidth = width
            poolHeight = height
        }
        guard let pool else { return nil }
        var buffer: CVPixelBuffer?
        // bounded by construction: with one frame in flight the pool settles far below the
        // threshold, so hitting it means a leak, which must degrade to passthrough rather than grow
        let auxAttributes = [
            kCVPixelBufferPoolAllocationThresholdKey as String: Self.maxPooledBuffers
        ] as CFDictionary
        guard CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(
            kCFAllocatorDefault, pool, auxAttributes, &buffer) == kCVReturnSuccess else { return nil }
        return buffer
    }

    private func makeTexture(from pb: CVPixelBuffer) -> CVMetalTexture? {
        var texture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, textureCache, pb, nil, .bgra8Unorm,
            CVPixelBufferGetWidth(pb), CVPixelBufferGetHeight(pb), 0, &texture)
        return status == kCVReturnSuccess ? texture : nil
    }
}
