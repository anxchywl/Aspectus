import XCTest
@testable import AspectusKit

final class CaptureRecoveryTests: XCTestCase {
    func testDisconnectStopsTheSessionAndSchedulesARestart() {
        var recovery = CaptureRecovery(backoffSeconds: [0.5, 1])
        let action = recovery.handle(.deviceDisconnected)
        XCTAssertTrue(action.suspendsSession, "a dead device must not be left running")
        XCTAssertEqual(action.restartAfter, 0.5, "another camera may be attached, so it retries")
        XCTAssertEqual(recovery.state, .retrying(attempt: 1))
    }

    func testAMissingDeviceStopsRetryingAndWaits() {
        var recovery = CaptureRecovery(backoffSeconds: [0.5, 1, 2])
        _ = recovery.handle(.deviceDisconnected)
        let action = recovery.handle(.restartFailed(deviceMissing: true))
        XCTAssertNil(action.restartAfter, "retrying against no camera at all is pointless")
        XCTAssertEqual(recovery.state, .waitingForDevice)
    }

    func testReconnectRestartsFromTheWaitingState() {
        var recovery = CaptureRecovery(backoffSeconds: [0.5, 1])
        _ = recovery.handle(.deviceDisconnected)
        _ = recovery.handle(.restartFailed(deviceMissing: true))
        let action = recovery.handle(.deviceConnected)
        XCTAssertEqual(action.restartAfter, 0.5, "a camera coming back is the signal to reopen")
        XCTAssertFalse(action.suspendsSession, "nothing is running to suspend")
        XCTAssertEqual(recovery.state, .retrying(attempt: 1))
    }

    func testRestartSuccessReturnsToRunning() {
        var recovery = CaptureRecovery(backoffSeconds: [0.5, 1])
        _ = recovery.handle(.runtimeError)
        let action = recovery.handle(.restartSucceeded)
        XCTAssertEqual(action, .none)
        XCTAssertEqual(recovery.state, .running)
    }

    func testFailedRestartsWalkTheBackoffLadderThenGiveUp() {
        var recovery = CaptureRecovery(backoffSeconds: [0.5, 1, 2])
        XCTAssertEqual(recovery.handle(.runtimeError).restartAfter, 0.5)
        XCTAssertEqual(recovery.handle(.restartFailed(deviceMissing: false)).restartAfter, 1)
        XCTAssertEqual(recovery.state, .retrying(attempt: 2))
        XCTAssertEqual(recovery.handle(.restartFailed(deviceMissing: false)).restartAfter, 2)
        XCTAssertEqual(recovery.state, .retrying(attempt: 3))
        let exhausted = recovery.handle(.restartFailed(deviceMissing: false))
        XCTAssertNil(exhausted.restartAfter, "the ladder is bounded, it must not retry forever")
        XCTAssertEqual(recovery.state, .exhausted)
    }

    func testARuntimeErrorWhileRecoveringEscalatesInsteadOfResetting() {
        var recovery = CaptureRecovery(backoffSeconds: [0.5, 1])
        XCTAssertEqual(recovery.handle(.runtimeError).restartAfter, 0.5)
        let second = recovery.handle(.runtimeError)
        XCTAssertTrue(second.suspendsSession, "the reopened session is broken and must be stopped")
        XCTAssertEqual(second.restartAfter, 1, "a camera that keeps failing must back off")
        XCTAssertNil(recovery.handle(.runtimeError).restartAfter, "and eventually stop trying")
        XCTAssertEqual(recovery.state, .exhausted)
    }

    func testTheLadderResetsAfterARecovery() {
        var recovery = CaptureRecovery(backoffSeconds: [0.5, 1])
        _ = recovery.handle(.runtimeError)
        _ = recovery.handle(.restartFailed(deviceMissing: false))
        _ = recovery.handle(.restartSucceeded)
        XCTAssertEqual(recovery.handle(.runtimeError).restartAfter, 0.5,
                       "a later failure must start from the shortest delay again")
    }

    func testExhaustedStillRecoversWhenACameraArrives() {
        var recovery = CaptureRecovery(backoffSeconds: [0.5])
        _ = recovery.handle(.runtimeError)
        _ = recovery.handle(.restartFailed(deviceMissing: false))
        XCTAssertEqual(recovery.state, .exhausted)
        XCTAssertEqual(recovery.handle(.deviceConnected).restartAfter, 0.5)
        XCTAssertEqual(recovery.state, .retrying(attempt: 1))
    }

    func testSleepSuspendsWithoutScheduling() {
        var recovery = CaptureRecovery()
        let action = recovery.handle(.willSleep)
        XCTAssertTrue(action.suspendsSession)
        XCTAssertNil(action.restartAfter, "nothing should run while the machine is asleep")
        XCTAssertEqual(recovery.state, .asleep)
    }

    func testWakeRestarts() {
        var recovery = CaptureRecovery(backoffSeconds: [0.5, 1])
        _ = recovery.handle(.willSleep)
        let action = recovery.handle(.didWake)
        XCTAssertEqual(action.restartAfter, 0.5)
        XCTAssertEqual(recovery.state, .retrying(attempt: 1))
    }

    func testEventsDuringSleepAreIgnored() {
        var recovery = CaptureRecovery()
        _ = recovery.handle(.willSleep)
        // the camera tears down on the way into sleep, so its complaints are expected noise
        XCTAssertEqual(recovery.handle(.runtimeError), .none)
        XCTAssertEqual(recovery.handle(.deviceDisconnected), .none)
        XCTAssertEqual(recovery.handle(.deviceConnected), .none)
        XCTAssertEqual(recovery.state, .asleep, "only waking may restart capture")
        XCTAssertEqual(recovery.handle(.didWake).restartAfter, 0.5)
    }

    func testWakeWithoutSleepDoesNothing() {
        var recovery = CaptureRecovery()
        XCTAssertEqual(recovery.handle(.didWake), .none)
        XCTAssertEqual(recovery.state, .running)
    }

    func testAnInterruptionIsReportedButLeftToAVFoundation() {
        var recovery = CaptureRecovery()
        XCTAssertEqual(recovery.handle(.interrupted), .none,
                       "the session restarts itself when the interruption ends")
        XCTAssertEqual(recovery.state, .interrupted)
        XCTAssertEqual(recovery.handle(.interruptionEnded), .none)
        XCTAssertEqual(recovery.state, .running)
    }

    func testInterruptionEndedWithoutAnInterruptionIsIgnored() {
        var recovery = CaptureRecovery(backoffSeconds: [0.5])
        _ = recovery.handle(.runtimeError)
        _ = recovery.handle(.interruptionEnded)
        XCTAssertEqual(recovery.state, .retrying(attempt: 1),
                       "a stray interruption end must not claim the session is healthy")
    }
}
