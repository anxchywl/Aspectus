import Metal
import MetalKit
import CoreVideo
import simd

/// draws a CVPixelBuffer to an MTKView as a zero-copy Metal texture via CVMetalTextureCache
/// aspect-fill and mirror happen in the shader, no per-frame allocations beyond command buffers
final class MetalRenderer: NSObject, MTKViewDelegate {
    private struct Uniforms { var uvScale: SIMD2<Float>; var uvOffset: SIMD2<Float>; var mirror: UInt32 }

    let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState
    private var textureCache: CVMetalTextureCache!

    // display choice for self-view, does not affect what the virtual camera outputs
    var mirror = true

    private let frameLock = NSLock()
    private var pending: CVPixelBuffer?

    var onPresented: (() -> Void)?

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

    func enqueue(_ pixelBuffer: CVPixelBuffer, view: MTKView) {
        frameLock.lock(); pending = pixelBuffer; frameLock.unlock()
        view.draw()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        frameLock.lock(); let pb = pending; frameLock.unlock()
        guard let pb,
              let drawable = view.currentDrawable,
              let rpd = view.currentRenderPassDescriptor,
              let cmd = queue.makeCommandBuffer() else { return }

        guard let texture = makeTexture(from: pb) else { return }

        let w = Float(CVPixelBufferGetWidth(pb))
        let h = Float(CVPixelBufferGetHeight(pb))
        let dw = Float(view.drawableSize.width)
        let dh = Float(view.drawableSize.height)
        var u = aspectFill(texW: w, texH: h, viewW: dw, viewH: dh)

        guard let enc = cmd.makeRenderCommandEncoder(descriptor: rpd) else { return }
        enc.setRenderPipelineState(pipeline)
        enc.setVertexBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 0)
        enc.setFragmentTexture(texture, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
        enc.endEncoding()
        cmd.present(drawable)
        cmd.addCompletedHandler { [weak self] _ in self?.onPresented?() }
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

    private func makeTexture(from pb: CVPixelBuffer) -> MTLTexture? {
        let width = CVPixelBufferGetWidth(pb)
        let height = CVPixelBufferGetHeight(pb)
        var cvTexture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, textureCache, pb, nil,
            .bgra8Unorm, width, height, 0, &cvTexture)
        guard status == kCVReturnSuccess, let cvTexture,
              let tex = CVMetalTextureGetTexture(cvTexture) else { return nil }
        return tex
    }
}
