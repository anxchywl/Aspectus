/// decides how capture recovers from a disconnect, a session runtime error or a sleep/wake cycle
///
/// the policy is separated from the session so it can be unit-tested without a camera: it takes
/// events and answers with what to do, never touching AVFoundation itself
public struct CaptureRecovery: Sendable, Equatable {
    public enum State: Sendable, Equatable {
        case running
        /// the session stopped itself and AVFoundation is expected to bring it back
        case interrupted
        case retrying(attempt: Int)
        /// there is no camera to open, so retrying is pointless until one appears
        case waitingForDevice
        case asleep
        /// the ladder ran out, only a device arriving or an explicit restart moves this
        case exhausted
    }

    public enum Event: Sendable, Equatable {
        case runtimeError
        case interrupted
        case interruptionEnded
        case deviceDisconnected
        case deviceConnected
        case willSleep
        case didWake
        case restartSucceeded
        /// deviceMissing separates "no camera exists" from "a camera is there and opening it failed"
        case restartFailed(deviceMissing: Bool)
    }

    /// stopping the session and scheduling a restart are independent: a disconnect needs both, a
    /// sleep needs only the first
    public struct Action: Sendable, Equatable {
        public var suspendsSession = false
        public var restartAfter: Double?

        public static let none = Action()

        public init(suspendsSession: Bool = false, restartAfter: Double? = nil) {
            self.suspendsSession = suspendsSession
            self.restartAfter = restartAfter
        }
    }

    public private(set) var state: State = .running

    private let backoff: [Double]
    /// how many restarts have already failed, indexing the ladder for the next delay
    private var failures = 0

    /// the ladder is bounded so a camera that never returns stops burning restarts, and wide enough
    /// to cover a wake, where the device re-enumerates seconds after the app is running again
    public init(backoffSeconds: [Double] = [0.5, 1, 2, 4, 8]) {
        precondition(!backoffSeconds.isEmpty, "recovery needs at least one delay")
        backoff = backoffSeconds
    }

    public mutating func handle(_ event: Event) -> Action {
        switch event {
        case .willSleep:
            state = .asleep
            failures = 0
            return Action(suspendsSession: true)

        case .didWake:
            guard state == .asleep else { return .none }
            return scheduleRestart(suspendsSession: false)

        // asleep outranks everything: the camera is expected to fail on the way down, and waking is
        // the only thing that should restart it
        case .runtimeError:
            guard state != .asleep else { return .none }
            // a session that fails again while recovering is a failed recovery, not a fresh fault:
            // resetting the ladder here would reopen a permanently broken camera forever
            guard case .retrying = state else { return scheduleRestart(suspendsSession: true) }
            var action = retryOrGiveUp()
            action.suspendsSession = true
            return action

        case .deviceDisconnected:
            guard state != .asleep else { return .none }
            // another camera may still be attached, so this is a restart attempt rather than a wait
            return scheduleRestart(suspendsSession: true)

        case .deviceConnected:
            switch state {
            case .waitingForDevice, .retrying, .exhausted:
                return scheduleRestart(suspendsSession: false)
            case .running, .interrupted, .asleep:
                return .none
            }

        case .interrupted:
            guard state != .asleep else { return .none }
            state = .interrupted
            return .none

        case .interruptionEnded:
            guard state == .interrupted else { return .none }
            state = .running
            return .none

        case .restartSucceeded:
            state = .running
            failures = 0
            return .none

        case let .restartFailed(deviceMissing):
            guard deviceMissing else { return retryOrGiveUp() }
            // a connect notification is the only signal worth waiting for, so the ladder stops here
            state = .waitingForDevice
            failures = 0
            return .none
        }
    }

    private mutating func scheduleRestart(suspendsSession: Bool) -> Action {
        failures = 0
        state = .retrying(attempt: 1)
        return Action(suspendsSession: suspendsSession, restartAfter: backoff[0])
    }

    private mutating func retryOrGiveUp() -> Action {
        failures += 1
        guard failures < backoff.count else {
            state = .exhausted
            return .none
        }
        state = .retrying(attempt: failures + 1)
        return Action(restartAfter: backoff[failures])
    }
}
