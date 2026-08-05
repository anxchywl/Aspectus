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
        .windowResizability(.contentSize)
        .commands { AspectusCommands(controller: controller, ui: ui) }

        Settings {
            SettingsView(controller: controller, virtualCamera: virtualCamera, ui: ui)
        }
    }
}
