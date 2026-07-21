import SwiftUI
import MetalKit

/// SwiftUI bridge to the MTKView driven by MetalRenderer
struct MetalPreviewView: NSViewRepresentable {
    let controller: PipelineController

    func makeNSView(context: Context) -> MTKView {
        let view = MTKView()
        if let renderer = MetalRenderer(mtkView: view) {
            context.coordinator.renderer = renderer
            controller.attach(renderer: renderer)
        }
        return view
    }

    func updateNSView(_ nsView: MTKView, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        // strong owner of the renderer for the view's lifetime
        var renderer: MetalRenderer?
    }
}
