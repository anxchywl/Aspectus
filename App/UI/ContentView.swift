import SwiftUI

struct ContentView: View {
    @StateObject private var controller = PipelineController()
    @StateObject private var virtualCamera = SystemExtensionInstaller()
    @State private var showCalibration = false

    var body: some View {
        ZStack(alignment: .topLeading) {
            Color.black.ignoresSafeArea()

            MetalPreviewView(controller: controller)
                .ignoresSafeArea()

            TrackingOverlay(controller: controller)
                .ignoresSafeArea()

            DiagnosticsHUD(controller: controller)
                .padding(12)

            if case let .failed(message) = virtualCamera.state {
                overlayMessage("Virtual camera could not install.\n\(message)")
            } else if virtualCamera.state == .needsApproval {
                overlayMessage(virtualCamera.state.summary)
            } else if controller.permissionDenied {
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
                Toggle("Correct", isOn: Binding(
                    get: { controller.correctionEnabled },
                    set: { controller.correctionEnabled = $0 }))
                .help("Toggle gaze correction to compare against the original frame")
            }
            ToolbarItem(placement: .automatic) {
                HStack(spacing: 6) {
                    Text("Amount")
                    Slider(value: Binding(get: { controller.redirectDegrees },
                                          set: { controller.redirectDegrees = $0 }),
                           in: 0...18)
                        .frame(width: 110)
                        .accessibilityLabel("Gaze correction amount in degrees")
                    Text(String(format: "%.0f°", controller.redirectDegrees))
                        .monospacedDigit()
                }
                .help("Angle between your screen and the camera lens")
            }
            ToolbarItem(placement: .automatic) {
                Button(controller.calibration == nil ? "Calibrate" : "Calibrated") {
                    showCalibration = true
                }
                .help(controller.calibration == nil
                      ? "Measure how your eyes read relative to the camera"
                      : "Recalibrate or reset the stored calibration")
                .disabled(!controller.isRunning)
            }
            ToolbarItem(placement: .automatic) {
                Button(virtualCamera.state == .activated ? "Camera installed" : "Install camera") {
                    virtualCamera.activate()
                }
                .help(virtualCamera.state.summary)
                .disabled(virtualCamera.state == .requesting)
            }
            ToolbarItem(placement: .automatic) {
                Button(controller.isRunning ? "Stop" : "Start") {
                    if controller.isRunning { controller.stop() }
                    else { Task { await controller.start() } }
                }
            }
        }
        .sheet(isPresented: $showCalibration) {
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
