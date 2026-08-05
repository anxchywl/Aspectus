import XCTest
@testable import AspectusKit

final class PublishPacerTests: XCTestCase {
    private func published(_ pacer: inout PublishPacer, times: [Double]) -> Int {
        times.reduce(into: 0) { count, t in if pacer.shouldPublish(at: t) { count += 1 } }
    }

    /// frames per second actually let through, which is the number the advertised format promises
    private func publishedRate(_ pacer: inout PublishPacer, times: [Double]) -> Double {
        let span = times[times.count - 1] - times[0]
        return Double(published(&pacer, times: times)) / span
    }

    func testASourceAtTheAdvertisedRatePassesEveryFrame() {
        var pacer = PublishPacer(frameRate: 30)
        let times = (0..<300).map { Double($0) / 30.0 }
        XCTAssertEqual(published(&pacer, times: times), 300,
                       "pacing must cost nothing when the camera already runs at the advertised rate")
    }

    func testJitterAroundTheIntervalIsNotDropped() {
        var pacer = PublishPacer(frameRate: 30)
        // ±1 ms either side of a 33.3 ms interval, which is what a real capture clock looks like
        let times = (0..<300).map { (i: Int) -> Double in
            let jitter: Double = i.isMultiple(of: 2) ? 0.001 : -0.001
            return Double(i) / 30.0 + jitter
        }
        XCTAssertEqual(published(&pacer, times: times), 300, "jitter must not be mistaken for excess rate")
    }

    func testAFasterSourceIsHalvedRatherThanForwardedWhole() {
        var pacer = PublishPacer(frameRate: 30)
        let times = (0..<600).map { Double($0) / 60.0 }
        XCTAssertEqual(published(&pacer, times: times), 300,
                       "a 60 fps camera must reach a 30 fps virtual camera as every other frame")
    }

    func testASourceSlightlyOverTheCapLosesOnlyItsExcess() {
        var pacer = PublishPacer(frameRate: 30)
        // 33.3 fps against a 30 fps cap: only the excess tenth should go
        let times = (0..<300).map { Double($0) * 0.9 / 30.0 }
        XCTAssertEqual(publishedRate(&pacer, times: times), 30, accuracy: 0.5,
                       "a source just over the cap must be paced to the cap, not beaten down below it")
    }

    func testALongGapCannotReleaseABurst() {
        var pacer = PublishPacer(frameRate: 30)
        XCTAssertTrue(pacer.shouldPublish(at: 0))
        // two seconds of silence, then frames arriving far faster than the cap
        var published = 0
        for i in 0..<10 where pacer.shouldPublish(at: 2 + Double(i) * 0.001) { published += 1 }
        XCTAssertLessThanOrEqual(published, 2, "banked credit must not be spendable as a burst")
    }

    func testATimestampRestartInThePastRepublishesImmediately() {
        var pacer = PublishPacer(frameRate: 30)
        XCTAssertTrue(pacer.shouldPublish(at: 1000))
        XCTAssertFalse(pacer.shouldPublish(at: 1000.001))
        XCTAssertTrue(pacer.shouldPublish(at: 0),
                      "a reopened camera restarting its clock must not stall publishing")
        XCTAssertFalse(pacer.shouldPublish(at: 0.001))
        XCTAssertTrue(pacer.shouldPublish(at: 1.0 / 30.0))
    }

    func testAnUnsetRatePacesNothing() {
        var pacer = PublishPacer(frameRate: 0)
        XCTAssertTrue(pacer.shouldPublish(at: 0))
        XCTAssertTrue(pacer.shouldPublish(at: 0))
    }
}
