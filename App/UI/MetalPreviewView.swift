import SwiftUI
import MetalKit

/// SwiftUI bridge to the MTKView driven by MetalRenderer
///
/// `secondary` marks a preview that borrows the frame stream from the main window's preview and
/// returns it when the view goes away; without that the pipeline's single renderer reference would
/// be left pointing at a deallocated view.
struct MetalPreviewView: NSViewRepresentable {
    let controller: PipelineController
    var secondary = false

    func makeNSView(context: Context) -> MTKView {
        let view = MTKView()
        context.coordinator.controller = controller
        context.coordinator.secondary = secondary
        if let renderer = MetalRenderer(mtkView: view) {
            context.coordinator.renderer = renderer
            if secondary {
                controller.attachSecondary(renderer: renderer)
            } else {
                controller.attach(renderer: renderer)
            }
        }
        return view
    }

    func updateNSView(_ nsView: MTKView, context: Context) {}

    static func dismantleNSView(_ nsView: MTKView, coordinator: Coordinator) {
        guard coordinator.secondary else { return }
        coordinator.controller?.detachSecondary()
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        // strong owner of the renderer for the view's lifetime
        var renderer: MetalRenderer?
        weak var controller: PipelineController?
        var secondary = false
    }
}
