import SwiftUI
import AspectusKit

/// the preferences window, holding everything that is chosen once rather than watched live
///
/// the toolbar keeps only the controls worth reaching for mid-call; anything set and forgotten —
/// the redirect angle, the camera, the virtual camera's install state — lives here and is saved
struct SettingsView: View {
    @Environment(\.openWindow) private var openWindow
    @ObservedObject var controller: PipelineController
    @ObservedObject var virtualCamera: SystemExtensionInstaller
    @ObservedObject var ui: UIState

    var body: some View {
        TabView {
            correction.tabItem { Label("Correction", systemImage: "eye") }
            camera.tabItem { Label("Camera", systemImage: "video") }
            output.tabItem { Label("Virtual camera", systemImage: "arrow.up.right.video") }
            appearance.tabItem { Label("View", systemImage: "rectangle.on.rectangle") }
        }
        // fixed rather than fitted: the tab view sizes to whichever pane opened first, which cut
        // the calibration buttons off the bottom of the correction pane
        .frame(width: 480, height: 570)
    }

    private var correction: some View {
        Form {
            Section {
                Toggle("Correct gaze", isOn: Binding(get: { controller.correctionEnabled },
                                                     set: { controller.correctionEnabled = $0 }))
                if controller.calibration == nil {
                    LabeledContent("Screen-to-lens angle") {
                        HStack(spacing: 8) {
                            Slider(value: Binding(get: { controller.redirectDegrees },
                                                  set: { controller.redirectDegrees = $0 }),
                                   in: 0...controller.maxRedirectDegrees)
                                .accessibilityLabel("Uncalibrated gaze correction amount in degrees")
                            Text(String(format: "%.0f°", controller.redirectDegrees))
                                .monospacedDigit().frame(width: 34, alignment: .trailing)
                        }
                    }
                    Text("Used only until calibration establishes where the camera lens is relative "
                         + "to your gaze. The safety gate still refuses anything past "
                         + "\(Int(controller.maxRedirectDegrees))°.")
                        .font(.caption).foregroundStyle(.secondary)
                } else {
                    LabeledContent("Correction target") {
                        Text("camera lens").foregroundStyle(.secondary)
                    }
                    Text("Calibration already measures gaze relative to the lens, so no additional "
                         + "screen offset is added.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }

            Section("Calibration") {
                LabeledContent("Status") { Text(calibrationStatus).foregroundStyle(.secondary) }
                LabeledContent("Viewing distance") {
                    HStack(spacing: 4) {
                        TextField("", value: $controller.viewingDistanceMM, format: .number)
                            .frame(width: 60).multilineTextAlignment(.trailing)
                        Text("mm")
                        Stepper("", value: $controller.viewingDistanceMM, in: 250...1200, step: 25)
                            .labelsHidden()
                    }
                }
                Text("macOS reports no camera field of view, so the scale fit cannot be measured "
                     + "and is only as good as this number.")
                    .font(.caption).foregroundStyle(.secondary)
                HStack {
                    Button("Calibrate…") { ui.showCalibration = true }
                        .disabled(!controller.isRunning)
                    Button("Reset") { controller.resetCalibration() }
                        .disabled(controller.calibration == nil)
                }
            }

            Section("Gaze model development") {
                Button("Collect model data…") { openWindow(id: "gaze-dataset") }
                    .disabled(!controller.isRunning || controller.calibrationProgress != nil)
                Text("Records labelled 60 × 60 eye crops and head pose on this Mac. This is "
                     + "biometric training data: it is never uploaded and can be deleted here.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }

    private var calibrationStatus: String {
        guard let calibration = controller.calibration else { return "not calibrated" }
        return calibration.createdAt.formatted(date: .abbreviated, time: .shortened)
    }

    private var camera: some View {
        Form {
            Section {
                Picker("Camera", selection: Binding(get: { controller.preferredCameraID },
                                                    set: { controller.selectCamera($0) })) {
                    Text("System default").tag(String?.none)
                    ForEach(controller.cameras) { device in
                        Text(device.name).tag(String?.some(device.id))
                    }
                }
                Text("Aspectus never offers its own virtual camera here — selecting it would feed "
                     + "the pipeline its own output.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section("Active") {
                LabeledContent("Format") {
                    Text(controller.formatDescription).font(.caption.monospaced())
                        .foregroundStyle(.secondary).multilineTextAlignment(.trailing)
                }
                LabeledContent("Session") {
                    Text(controller.captureStateLabel).foregroundStyle(.secondary)
                }
                if let interruption = controller.captureInterruption {
                    Text(interruption).font(.caption).foregroundStyle(.orange)
                }
            }
        }
        .formStyle(.grouped)
    }

    private var output: some View {
        Form {
            Section {
                LabeledContent("Status") {
                    Text(extensionStatus).foregroundStyle(.secondary)
                        .multilineTextAlignment(.trailing)
                }
                HStack {
                    Button("Install") { virtualCamera.activate() }
                        .disabled(virtualCamera.state == .requesting)
                    Button("Remove") { virtualCamera.deactivate() }
                        .disabled(virtualCamera.state == .requesting)
                }
                Text("Installing and removing both need your approval in System Settings ▸ General "
                     + "▸ Login Items & Extensions. Removing it takes the Aspectus camera away from "
                     + "every app; the preview here carries on either way.")
                    .font(.caption).foregroundStyle(.secondary)

                if controller.virtualCameraLost, virtualCamera.state != .removed {
                    Text("Installing over a running extension replaces it, and macOS never shows the "
                         + "new camera to an app that was already open. Aspectus has to be relaunched "
                         + "before it can publish again.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }

            Section("Published") {
                LabeledContent("Stream") {
                    Text(controller.virtualCameraState).foregroundStyle(.secondary)
                }
                LabeledContent("Frames") {
                    Text("\(controller.virtualCameraSent) sent · "
                         + "\(controller.virtualCameraPaced) paced · "
                         + "\(controller.virtualCameraDropped) dropped")
                        .font(.caption.monospaced()).foregroundStyle(.secondary)
                }
                Text("The extension advertises \(VirtualCameraFormat.width)×"
                     + "\(VirtualCameraFormat.height) at \(VirtualCameraFormat.frameRate) fps. A "
                     + "camera that runs faster is paced down to it rather than published at a rate "
                     + "hosts were not told about.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }

    /// the installer only knows what it did this launch, so an extension installed on an earlier
    /// one reads as idle; a stream we are publishing into is the honest signal in that case
    private var extensionStatus: String {
        guard virtualCamera.state == .idle else { return virtualCamera.state.summary }
        return controller.virtualCameraConnected ? "installed" : "not detected"
    }

    private var appearance: some View {
        Form {
            Section {
                Toggle("Mirror preview", isOn: $controller.mirrorPreview)
                Text("A display choice only. What the virtual camera publishes is never mirrored.")
                    .font(.caption).foregroundStyle(.secondary)
                Toggle("Tracking overlay", isOn: $controller.showOverlay)
                Toggle("Diagnostics", isOn: $ui.showDiagnostics)
            }
        }
        .formStyle(.grouped)
    }
}
