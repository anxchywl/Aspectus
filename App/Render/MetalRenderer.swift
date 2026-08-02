import Metal
import MetalKit
import CoreVideo
import simd
import AspectusKit

/// draws a CVPixelBuffer to an MTKView as a zero-copy Metal texture via CVMetalTextureCache
/// aspect-fill and mirror happen in the shader, no per-frame allocations beyond command buffers
@MainActor
final class MetalRenderer: NSObject {
    private struct Uniforms { var uvScale: SIMD2<Float>; var uvOffset: SIMD2<Float>; var mirror: UInt32 }

    /// timing travels with the pixels so latency is attributed to the frame that was actually
    /// presented, not to whichever frame happened to be newest when the GPU finished
    private struct PendingFrame {
        let pixelBuffer: CVPixelBuffer
        let timing: FrameTiming
    }

    let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState
    private var textureCache: CVMetalTextureCache!

    // display choice for self-view, does not affect what the virtual camera outputs
    var mirror = true

    private var pending: PendingFrame?

    // an MTLTexture is only valid while its CVMetalTexture lives, and the GPU reads it until the
    // command buffer completes; the drawable pool is 3 deep so holding 3 outlives any in-flight read
    private var retainedTextures: [CVMetalTexture] = []
    private static let drawablePoolDepth = 3

    /// fires off the main actor once the drawable is on screen, carrying that frame's own timing
    /// and the present timestamp in the HostClock base
    var onPresented: (@Sendable (FrameTiming, Double) -> Void)?

    // weak so the view can be driven without owning the frame source
    weak var attachedView: MTKView?

    init?(mtkView: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.queue = queue

        mtkView.device = device
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.framebufferOnly = true
        mtkView.isPaused = true              // draw on demand, driven by frames
        mtkView.enableSetNeedsDisplay = false

        guard let library = try? device.makeDefaultLibrary(bundle: .main),
              let vfn = library.makeFunction(name: "preview_vertex"),
              let ffn = library.makeFunction(name: "preview_fragment") else { return nil }

        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = vfn
        desc.fragmentFunction = ffn
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        guard let pso = try? device.makeRenderPipelineState(descriptor: desc) else { return nil }
        self.pipeline = pso

        super.init()

        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)
        mtkView.delegate = self
        self.attachedView = mtkView
    }

    func enqueue(_ pixelBuffer: CVPixelBuffer, timing: FrameTiming, view: MTKView) {
        pending = PendingFrame(pixelBuffer: pixelBuffer, timing: timing)
        view.draw()
    }

    /// releases GPU resources held across a stop so a restart starts clean
    func flush() {
        pending = nil
        retainedTextures.removeAll()
        CVMetalTextureCacheFlush(textureCache, 0)
    }

    private func render(in view: MTKView) {
        // take-once, so a redraw that is not driven by a new frame cannot double-count latency
        guard let frame = pending else { return }
        pending = nil

        guard let drawable = view.currentDrawable,
              let rpd = view.currentRenderPassDescriptor,
              let cmd = queue.makeCommandBuffer(),
              let cvTexture = makeTexture(from: frame.pixelBuffer),
              let texture = CVMetalTextureGetTexture(cvTexture) else { return }

        retainedTextures.append(cvTexture)
        if retainedTextures.count > Self.drawablePoolDepth { retainedTextures.removeFirst() }

        let w = Float(CVPixelBufferGetWidth(frame.pixelBuffer))
        let h = Float(CVPixelBufferGetHeight(frame.pixelBuffer))
        let dw = Float(view.drawableSize.width)
        let dh = Float(view.drawableSize.height)
        var u = aspectFill(texW: w, texH: h, viewW: dw, viewH: dh)

        guard let enc = cmd.makeRenderCommandEncoder(descriptor: rpd) else { return }
        enc.setRenderPipelineState(pipeline)
        enc.setVertexBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 0)
        enc.setFragmentTexture(texture, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
        enc.endEncoding()

        // presentedTime shares the mach base with HostClock, so present latency needs no conversion
        let timing = frame.timing
        let notify = onPresented
        drawable.addPresentedHandler { presented in
            // presentedTime is 0 when the frame never reached the screen, e.g. the window was
            // occluded; counting those yields a latency of minus the machine's uptime
            guard presented.presentedTime > 0 else { return }
            notify?(timing, presented.presentedTime)
        }
        cmd.present(drawable)
        cmd.commit()
    }

    private func aspectFill(texW: Float, texH: Float, viewW: Float, viewH: Float) -> Uniforms {
        guard texW > 0, texH > 0, viewW > 0, viewH > 0 else {
            return Uniforms(uvScale: .init(1, 1), uvOffset: .zero, mirror: mirror ? 1 : 0)
        }
        let texAspect = texW / texH
        let viewAspect = viewW / viewH
        // scale the sampled uv region to fill the view, cropping the longer dimension
        var scale = SIMD2<Float>(1, 1)
        if texAspect > viewAspect {
            scale.x = viewAspect / texAspect   // crop sides
        } else {
            scale.y = texAspect / viewAspect   // crop top/bottom
        }
        return Uniforms(uvScale: scale, uvOffset: .zero, mirror: mirror ? 1 : 0)
    }

    private func makeTexture(from pb: CVPixelBuffer) -> CVMetalTexture? {
        let width = CVPixelBufferGetWidth(pb)
        let height = CVPixelBufferGetHeight(pb)
        var cvTexture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, textureCache, pb, nil,
            .bgra8Unorm, width, height, 0, &cvTexture)
        guard status == kCVReturnSuccess else { return nil }
        return cvTexture
    }
}

extension MetalRenderer: MTKViewDelegate {
    nonisolated func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    // MTKViewDelegate is not actor-annotated, but the view is paused with setNeedsDisplay off, so
    // the only draws come from enqueue on the main actor
    nonisolated func draw(in view: MTKView) {
        MainActor.assumeIsolated { render(in: view) }
    }
}
