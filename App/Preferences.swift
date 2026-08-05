import Foundation

/// the user choices that must survive a relaunch
///
/// `UserDefaults` rather than a file: these are preferences, chosen rather than measured. The
/// calibration is the one piece of user state that *is* measured, and it keeps its own store
struct Preferences {
    private let store: UserDefaults

    init(store: UserDefaults = .standard) { self.store = store }

    var mirrorPreview: Bool {
        get { bool("mirrorPreview", default: true) }
        nonmutating set { store.set(newValue, forKey: "mirrorPreview") }
    }

    var showOverlay: Bool {
        get { bool("showOverlay", default: true) }
        nonmutating set { store.set(newValue, forKey: "showOverlay") }
    }

    /// the HUD carries the diagnostics AGENTS §7 requires, so it stays on until asked otherwise
    var showDiagnostics: Bool {
        get { bool("showDiagnostics", default: true) }
        nonmutating set { store.set(newValue, forKey: "showDiagnostics") }
    }

    var correctionEnabled: Bool {
        get { bool("correctionEnabled", default: true) }
        nonmutating set { store.set(newValue, forKey: "correctionEnabled") }
    }

    var redirectDegrees: Double {
        get { double("redirectDegrees", default: 12) }
        nonmutating set { store.set(newValue, forKey: "redirectDegrees") }
    }

    var viewingDistanceMM: Double {
        get { double("viewingDistanceMM", default: 550) }
        nonmutating set { store.set(newValue, forKey: "viewingDistanceMM") }
    }

    /// nil means whatever the system offers as default, which is also what a camera that has since
    /// been unplugged falls back to
    var preferredCameraID: String? {
        get { store.string(forKey: "preferredCameraID") }
        nonmutating set { store.set(newValue, forKey: "preferredCameraID") }
    }

    // an absent key must read as the default rather than as false or zero
    private func bool(_ key: String, default fallback: Bool) -> Bool {
        store.object(forKey: key) as? Bool ?? fallback
    }

    private func double(_ key: String, default fallback: Double) -> Double {
        store.object(forKey: key) as? Double ?? fallback
    }
}
