import SwiftUI

struct ContentView: View {
    @StateObject private var controller = PipelineController()

    var body: some View {
        ZStack(alignment: .topLeading) {
            Color.black.ignoresSafeArea()

            MetalPreviewView(controller: controller)
                .ignoresSafeArea()

            TrackingOverlay(controller: controller)
                .ignoresSafeArea()

            DiagnosticsHUD(controller: controller)
                .padding(12)

            if controller.permissionDenied {
                overlayMessage("Camera access denied.\nEnable it in System Settings ▸ Privacy & Security ▸ Camera.")
            } else if !controller.isRunning {
                overlayMessage("Starting camera…")
            }
        }
        .frame(minWidth: 640, minHeight: 480)
        .toolbar {
            ToolbarItem(placement: .automatic) {
                Toggle("Mirror", isOn: Binding(
                    get: { controller.mirrorPreview },
                    set: { controller.mirrorPreview = $0 }))
            }
            ToolbarItem(placement: .automatic) {
                Toggle("Overlay", isOn: Binding(
                    get: { controller.showOverlay },
                    set: { controller.showOverlay = $0 }))
            }
            ToolbarItem(placement: .automatic) {
                Button(controller.isRunning ? "Stop" : "Start") {
                    if controller.isRunning { controller.stop() }
                    else { Task { await controller.start() } }
                }
            }
        }
        .task { await controller.start() }
    }

    private func overlayMessage(_ text: String) -> some View {
        Text(text)
            .multilineTextAlignment(.center)
            .foregroundStyle(.white)
            .padding(24)
            .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 12))
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
