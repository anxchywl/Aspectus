import AppKit

/// system sleep and wake, which capture does not survive: macOS releases the camera on the way into
/// sleep, and a session left running comes back producing nothing
final class SystemSleepObserver {
    /// held rather than reached through NSWorkspace on teardown, so deinit touches no shared state
    private let center = NSWorkspace.shared.notificationCenter
    private var tokens: [any NSObjectProtocol] = []

    /// both callbacks are delivered on the main queue, where the pipeline is driven from
    func start(willSleep: @escaping @Sendable () -> Void, didWake: @escaping @Sendable () -> Void) {
        stop()
        tokens = [
            center.addObserver(forName: NSWorkspace.willSleepNotification,
                               object: nil, queue: .main) { _ in willSleep() },
            center.addObserver(forName: NSWorkspace.didWakeNotification,
                               object: nil, queue: .main) { _ in didWake() },
        ]
    }

    func stop() {
        tokens.forEach { center.removeObserver($0) }
        tokens = []
    }

    deinit { stop() }
}
