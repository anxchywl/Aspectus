import SwiftUI

/// presentation state that belongs to no pipeline stage: which sheet is showing, and whether the
/// diagnostics HUD is up. kept out of `PipelineController` so the orchestrator stays free of
/// SwiftUI concerns
@MainActor
final class UIState: ObservableObject {
    @Published var showCalibration = false
    @Published var showDiagnostics = true {
        didSet { preferences.showDiagnostics = showDiagnostics }
    }

    private let preferences = Preferences()

    init() { showDiagnostics = preferences.showDiagnostics }
}
