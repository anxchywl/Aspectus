import SwiftUI
import AspectusKit

struct ContentView: View {
    @ObservedObject var controller: PipelineController
    @ObservedObject var virtualCamera: SystemExtensionInstaller
    @ObservedObject var ui: UIState

    var body: some View {
        preview
            .frame(minWidth: 560, minHeight: 380)
            .toolbar { toolbar }
            .inspector(isPresented: $ui.showDiagnostics) {
                DiagnosticsHUD(controller: controller)
                    .inspectorColumnWidth(min: 240, ideal: 280, max: 340)
            }
            .sheet(isPresented: $ui.showCalibration) {
                CalibrationView(controller: controller)
            }
            .task { await controller.start() }
    }

    /// everything drawn over the picture is a ZStack child rather than an `.overlay`: the preview is
    /// an NSView, and an overlay applied to its container composites underneath it
    private var preview: some View {
        ZStack(alignment: .topLeading) {
            Color.black

            MetalPreviewView(controller: controller)

            if controller.showOverlay {
                TrackingOverlay(model: controller.overlay, mirror: controller.mirrorPreview)
            }

            unavailable

            StatusPill(controller: controller).padding(14)

            extensionNotice
                .padding(14)
                .frame(maxWidth: .infinity, alignment: .top)
        }
        // the picture is always dark, so the things drawn over it are too, whatever the system theme
        .environment(\.colorScheme, .dark)
    }

    /// only the camera can make the preview unusable — an extension that will not install is
    /// reported without taking the picture away, because the virtual camera is an output and never
    /// a dependency
    @ViewBuilder
    private var unavailable: some View {
        if controller.permissionDenied {
            blocking("Camera access denied", systemImage: "video.slash",
                     detail: "Allow Aspectus in System Settings ▸ Privacy & Security ▸ Camera.")
        } else if let interruption = controller.captureInterruption {
            blocking("Camera unavailable", systemImage: "video.badge.ellipsis",
                     detail: interruption)
        } else if !controller.isRunning {
            blocking("Camera stopped", systemImage: "video.slash",
                     detail: "Start the camera to preview and publish a corrected picture.") {
                Button("Start") { Task { await controller.start() } }
                    .buttonStyle(.borderedProminent)
            }
        }
    }

    private func blocking(_ title: String, systemImage: String, detail: String,
                          @ViewBuilder actions: () -> some View = { EmptyView() }) -> some View {
        ContentUnavailableView {
            Label(title, systemImage: systemImage)
        } description: {
            Text(detail)
        } actions: {
            actions()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.black.opacity(0.7))
    }

    @ViewBuilder
    private var extensionNotice: some View {
        switch virtualCamera.state {
        case .needsApproval:
            notice("Allow the camera extension in System Settings ▸ General ▸ Login Items & Extensions",
                   systemImage: "exclamationmark.shield", tint: .orange)
        case let .failed(message):
            notice("Virtual camera could not install — \(message)",
                   systemImage: "exclamationmark.triangle", tint: .orange)
        default:
            EmptyView()
        }
    }

    private func notice(_ text: String, systemImage: String, tint: Color) -> some View {
        Label(text, systemImage: systemImage)
            .font(.callout)
            .foregroundStyle(tint)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(.regularMaterial, in: Capsule())
    }

    @ToolbarContentBuilder
    private var toolbar: some ToolbarContent {
        ToolbarItemGroup(placement: .primaryAction) {
            Toggle(isOn: Binding(get: { controller.correctionEnabled },
                                 set: { controller.correctionEnabled = $0 })) {
                Label("Correct", systemImage: controller.correctionEnabled ? "eye" : "eye.slash")
            }
            .toggleStyle(.button)
            .help("Turn gaze correction on and off to compare against the original frame")

            Button {
                ui.showCalibration = true
            } label: {
                Label("Calibrate", systemImage: "scope")
            }
            .disabled(!controller.isRunning)
            .help(controller.calibration == nil
                  ? "Measure how your eyes read relative to the camera"
                  : "Recalibrate or reset the stored calibration")

            Button {
                if controller.isRunning { controller.stop() }
                else { Task { await controller.start() } }
            } label: {
                Label(controller.isRunning ? "Stop" : "Start",
                      systemImage: controller.isRunning ? "stop.fill" : "play.fill")
            }
            .help(controller.isRunning ? "Stop the camera" : "Start the camera")

            Button {
                ui.showDiagnostics.toggle()
            } label: {
                Label("Diagnostics", systemImage: "sidebar.trailing")
            }
            .help("Show or hide the diagnostics panel")
        }
    }
}

/// what the pipeline is doing right now, in words, so the answer is not "read thirty rows of
/// telemetry and work it out"
private struct StatusPill: View {
    @ObservedObject var controller: PipelineController

    var body: some View {
        HStack(spacing: 7) {
            Circle().fill(tint).frame(width: 7, height: 7)
            Text(title).font(.callout.weight(.medium))
            if let detail {
                Text(detail).font(.callout).foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 11)
        .padding(.vertical, 7)
        .background(.regularMaterial, in: Capsule())
    }

    private var title: String {
        guard controller.isRunning else { return "Stopped" }
        guard controller.captureInterruption == nil else { return "Reconnecting" }
        switch controller.gaze.latest?.fallback ?? .noTracking {
        case .none: return "Correcting"
        case .disabled: return "Passthrough"
        case .noTracking: return "Looking for you"
        case .eyesClosed: return "Blinking"
        case .headPose: return "Head turned too far"
        case .angleLimit: return "Beyond the trusted angle"
        case .lowConfidence: return "Not sure enough"
        case .engaging: return "Easing in"
        case .degenerateGeometry, .correctorFailed: return "Passing the frame through"
        }
    }

    private var detail: String? {
        guard controller.isRunning, let blend = controller.gaze.latest?.blendStrength, blend > 0
        else { return nil }
        return String(format: "%.0f%%", blend * 100)
    }

    private var tint: Color {
        guard controller.isRunning else { return .secondary }
        guard controller.captureInterruption == nil else { return .orange }
        switch controller.gaze.latest?.fallback ?? .noTracking {
        case .none: return .green
        case .disabled: return .secondary
        default: return .yellow
        }
    }
}
