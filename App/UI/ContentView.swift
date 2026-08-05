import SwiftUI

struct ContentView: View {
    @ObservedObject var controller: PipelineController
    @ObservedObject var virtualCamera: SystemExtensionInstaller
    @ObservedObject var ui: UIState

    var body: some View {
        ZStack(alignment: .topLeading) {
            Color.black.ignoresSafeArea()

            MetalPreviewView(controller: controller)
                .ignoresSafeArea()

            TrackingOverlay(controller: controller)
                .ignoresSafeArea()

            if ui.showDiagnostics {
                DiagnosticsHUD(controller: controller)
                    .padding(12)
            }

            if case let .failed(message) = virtualCamera.state {
                overlayMessage("Virtual camera could not install.\n\(message)")
            } else if virtualCamera.state == .needsApproval {
                overlayMessage(virtualCamera.state.summary)
            } else if controller.permissionDenied {
                overlayMessage("Camera access denied.\nEnable it in System Settings ▸ Privacy & Security ▸ Camera.")
            } else if let interruption = controller.captureInterruption {
                overlayMessage(interruption)
            } else if !controller.isRunning {
                overlayMessage("Starting camera…")
            }
        }
        .frame(minWidth: 640, minHeight: 480)
        // only the controls worth reaching for mid-call; the rest is in settings and the menus
        .toolbar {
            ToolbarItem(placement: .automatic) {
                Toggle("Correct", isOn: Binding(
                    get: { controller.correctionEnabled },
                    set: { controller.correctionEnabled = $0 }))
                .help("Toggle gaze correction to compare against the original frame")
            }
            ToolbarItem(placement: .automatic) {
                Button(controller.calibration == nil ? "Calibrate" : "Calibrated") {
                    ui.showCalibration = true
                }
                .help(controller.calibration == nil
                      ? "Measure how your eyes read relative to the camera"
                      : "Recalibrate or reset the stored calibration")
                .disabled(!controller.isRunning)
            }
            ToolbarItem(placement: .automatic) {
                Button(controller.isRunning ? "Stop" : "Start") {
                    if controller.isRunning { controller.stop() }
                    else { Task { await controller.start() } }
                }
            }
        }
        .sheet(isPresented: $ui.showCalibration) {
            CalibrationView(controller: controller)
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
