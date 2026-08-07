import XCTest
@testable import AspectusKit

final class RateMeterTests: XCTestCase {
    private let interval = 1.0 / 30.0

    private func steady(_ meter: inout RateMeter, frames: Int, from start: Double = 0) -> Double {
        var last: Double = 0
        for i in 0..<frames { last = meter.tick(at: start + Double(i) * interval) }
        return last
    }

    func testReportsTheSourceRate() {
        var meter = RateMeter()
        let rate = steady(&meter, frames: 60)
        XCTAssertEqual(rate, 30, accuracy: 0.5, "a steady 30 fps source must read 30 fps")
    }

    func testFewerThanTwoSamplesReadsZero() {
        var meter = RateMeter()
        XCTAssertEqual(meter.tick(at: 0), 0, "one sample cannot establish a rate")
    }

    func testDecaysWhileNothingTicks() {
        var meter = RateMeter()
        let last = 59 * interval
        _ = steady(&meter, frames: 60)

        // this is the occluded-window case: nothing presents, so nothing ticks
        let half = meter.rate(asOf: last + 0.5)
        XCTAssertLessThan(half, 20, "a half-second stall must pull the rate down, not latch it")
        XCTAssertGreaterThan(half, 0)
        XCTAssertEqual(meter.rate(asOf: last + 1.5), 0,
                       "past the window every sample has expired and the rate is zero")
    }

    /// a source that stalls and then catches up delivers a second's worth of frames in less than a
    /// second. measured on screen: output read 58 fps and capture 78 fps on a camera that cannot
    /// exceed 30, because the rate was taken across the burst rather than across the window
    func testABurstAfterAStallCannotReadAboveTheSourceRate() {
        var meter = RateMeter()
        _ = steady(&meter, frames: 60)

        // the stall plus the burst has to outlast the window, or a surviving pre-stall sample keeps
        // the span wide and hides the fault
        let stallEnded = 59 * interval + 0.6
        var last: Double = 0
        for i in 0..<30 { last = meter.tick(at: stallEnded + Double(i) / 60.0) }

        // thirty events fell in the last second, so no reading above ~30 is defensible
        XCTAssertLessThanOrEqual(last, 32, "a catch-up burst must not read as a higher frame rate")
    }

    /// the window is the denominator, but only once there has been a window of it to measure
    func testRampsUpHonestlyBeforeAFullWindowHasElapsed() {
        var meter = RateMeter()
        let last = steady(&meter, frames: 10)
        XCTAssertEqual(last, 30, accuracy: 1,
                       "a young meter must report the rate it has seen, not divide by a window "
                       + "that has not elapsed")
    }

    func testResumesAfterAStall() {
        var meter = RateMeter()
        _ = steady(&meter, frames: 60)
        _ = meter.rate(asOf: 59 * interval + 2)
        let resumed = steady(&meter, frames: 60, from: 100)
        XCTAssertEqual(resumed, 30, accuracy: 0.5, "the meter must recover its reading, not stay at zero")
    }
}
