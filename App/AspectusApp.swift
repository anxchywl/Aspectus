import SwiftUI

@main
struct Aspectus: App {
    var body: some Scene {
        WindowGroup("Aspectus") {
            ContentView()
        }
        .windowResizability(.contentSize)
    }
}
