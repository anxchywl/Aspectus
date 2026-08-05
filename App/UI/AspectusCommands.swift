import SwiftUI

/// menu commands for the controls the toolbar no longer carries, so moving them into settings did
/// not put any of them out of reach
struct AspectusCommands: Commands {
    let controller: PipelineController
    let ui: UIState

    var body: some Commands {
        // nothing here opens or creates documents
        CommandGroup(replacing: .newItem) { }
        CommandMenu("Camera") { CameraMenu(controller: controller, ui: ui) }
        CommandGroup(after: .toolbar) { ViewMenu(controller: controller, ui: ui) }
    }
}

/// the menu items are views rather than commands so they observe the controller and re-render when
/// it changes; a `Commands` value on its own would show whatever state it was built with
private struct CameraMenu: View {
    @ObservedObject var controller: PipelineController
    @ObservedObject var ui: UIState

    var body: some View {
        Button(controller.isRunning ? "Stop" : "Start") {
            if controller.isRunning { controller.stop() } else { Task { await controller.start() } }
        }
        .keyboardShortcut("r")

        Divider()

        Button("Calibrate…") { ui.showCalibration = true }
            .keyboardShortcut("k")
            .disabled(!controller.isRunning)
        Button("Reset calibration") { controller.resetCalibration() }
            .disabled(controller.calibration == nil)
    }
}

private struct ViewMenu: View {
    @ObservedObject var controller: PipelineController
    @ObservedObject var ui: UIState

    var body: some View {
        Toggle("Correct gaze", isOn: Binding(get: { controller.correctionEnabled },
                                             set: { controller.correctionEnabled = $0 }))
            .keyboardShortcut("c", modifiers: [.command, .shift])
        Toggle("Mirror preview", isOn: $controller.mirrorPreview)
            .keyboardShortcut("m", modifiers: [.command, .shift])
        Toggle("Tracking overlay", isOn: $controller.showOverlay)
            .keyboardShortcut("o", modifiers: [.command, .shift])
        Toggle("Diagnostics", isOn: $ui.showDiagnostics)
            .keyboardShortcut("d", modifiers: [.command, .shift])
        Divider()
    }
}
