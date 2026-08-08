import SwiftUI

@main
@MainActor
struct Aspectus: App {
    // owned here rather than by the window, so the settings scene and the menu commands act on the
    // same pipeline the preview is showing
    @StateObject private var controller = PipelineController()
    @StateObject private var virtualCamera = SystemExtensionInstaller()
    @StateObject private var ui = UIState()

    var body: some Scene {
        WindowGroup("Aspectus") {
            ContentView(controller: controller, virtualCamera: virtualCamera, ui: ui)
        }
        // wide enough for a 16:9 preview with the diagnostics inspector open
        .defaultSize(width: 1100, height: 660)
        .commands { AspectusCommands(controller: controller, ui: ui) }

        Settings {
            SettingsView(controller: controller, virtualCamera: virtualCamera, ui: ui)
        }

        Window("Gaze model data", id: "gaze-dataset") {
            GazeDatasetView(controller: controller)
        }
        .defaultSize(width: 900, height: 650)
    }
}
