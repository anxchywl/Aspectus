import XCTest
@testable import AspectusKit

final class LatestValueBoxTests: XCTestCase {
    func testDropsStaleInsteadOfQueuing() async {
        let box = LatestValueBox<Int>()
        // offer faster than take, only the latest survives
        box.offer(1)
        box.offer(2)
        box.offer(3)
        let v = await box.take()
        XCTAssertEqual(v, 3, "take must return the most recent value")
        XCTAssertEqual(box.dropped, 2, "two stale values must be counted as drops")
    }

    func testAwaitsWhenEmptyThenResumes() async {
        let box = LatestValueBox<Int>()
        let waiter = Task { await box.take() }
        // let the waiter park
        try? await Task.sleep(nanoseconds: 5_000_000)
        box.offer(42)
        let v = await waiter.value
        XCTAssertEqual(v, 42)
        XCTAssertEqual(box.dropped, 0, "direct hand-off to a waiter is not a drop")
    }

    func testFinishUnblocksWaiter() async {
        let box = LatestValueBox<Int>()
        let waiter = Task { await box.take() }
        try? await Task.sleep(nanoseconds: 5_000_000)
        box.finish()
        let v = await waiter.value
        XCTAssertNil(v, "finish must resume a parked waiter with nil")
    }
}

final class OneEuroFilterTests: XCTestCase {
    func testReducesJitterOnNoisyConstant() {
        var f = OneEuroFilter(minCutoff: 1.0, beta: 0.0)
        var raw: [Double] = []
        var filt: [Double] = []
        // 100 Hz noisy signal around 0.5
        var seed: UInt64 = 88172645463325252
        func rnd() -> Double { // deterministic xorshift
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17
            return Double(seed % 1000) / 1000.0 - 0.5
        }
        for i in 0..<200 {
            let t = Double(i) / 100.0
            let x = 0.5 + 0.05 * rnd()
            raw.append(x)
            filt.append(f.filter(x, t: t))
        }
        func variance(_ a: ArraySlice<Double>) -> Double {
            let m = a.reduce(0, +) / Double(a.count)
            return a.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Double(a.count)
        }
        // compare the settled portion
        let vr = variance(raw[100...])
        let vf = variance(filt[100...])
        XCTAssertLessThan(vf, vr * 0.6, "filtered variance should be well below raw")
    }

    func testTracksRampWithBoundedLag() {
        var f = OneEuroFilter(minCutoff: 1.0, beta: 0.5)
        var last = 0.0
        for i in 0..<200 {
            let t = Double(i) / 100.0
            last = f.filter(t, t: t) // signal is the ramp itself
        }
        XCTAssertEqual(last, 2.0, accuracy: 0.1, "high-beta filter must not lag a steady ramp much")
    }

    func testResetClearsHistory() {
        var f = OneEuroFilter()
        _ = f.filter(10, t: 0.0)
        _ = f.filter(10, t: 0.1)
        f.reset()
        let v = f.filter(-5, t: 0.2)
        XCTAssertEqual(v, -5, accuracy: 1e-9, "first sample after reset passes through")
    }
}

final class CorrectionGateTests: XCTestCase {
    func testHysteresisPreventsToggle() {
        var gate = CorrectionGate(config: .init(enterConfidence: 0.6, exitConfidence: 0.4,
                                                slewPerSecond: 1000))
        // below enter, stays off
        _ = gate.update(confidence: 0.5, requestedCorrectionDegrees: 5, t: 0.0)
        XCTAssertFalse(gate.isEngaged)
        // crossing enter engages
        _ = gate.update(confidence: 0.65, requestedCorrectionDegrees: 5, t: 0.1)
        XCTAssertTrue(gate.isEngaged)
        // dipping into the hysteresis band stays engaged
        _ = gate.update(confidence: 0.5, requestedCorrectionDegrees: 5, t: 0.2)
        XCTAssertTrue(gate.isEngaged)
        // below exit disengages
        _ = gate.update(confidence: 0.35, requestedCorrectionDegrees: 5, t: 0.3)
        XCTAssertFalse(gate.isEngaged)
    }

    func testAngleLimitForcesFallback() {
        var gate = CorrectionGate(config: .init(maxCorrectionDegrees: 18, slewPerSecond: 1000))
        // high confidence but way past the trusted angle, weight decays to 0
        var w = 0.0
        for i in 0..<10 { w = gate.update(confidence: 0.9, requestedCorrectionDegrees: 30, t: Double(i) * 0.1) }
        XCTAssertEqual(w, 0.0, accuracy: 1e-6, "beyond guard band correction must be fully off")
    }

    func testSlewRateLimitsRamp() {
        var gate = CorrectionGate(config: .init(enterConfidence: 0.6, slewPerSecond: 2.0))
        // dt 0.1s at slew 2/s allows at most +0.2 per step
        let w = gate.update(confidence: 0.9, requestedCorrectionDegrees: 5, t: 0.1)
        XCTAssertLessThanOrEqual(w, 0.001, "first step from t=nil sets baseline")
        let w2 = gate.update(confidence: 0.9, requestedCorrectionDegrees: 5, t: 0.2)
        XCTAssertEqual(w2, 0.2, accuracy: 1e-6, "blend must ramp, not pop")
    }
}

final class GeometryTests: XCTestCase {
    func testNormRectExpandClampsToUnitSquare() {
        let r = NormRect(x: 0.9, y: 0.9, width: 0.08, height: 0.08)
        let e = r.expanded(by: 1.0)
        XCTAssertGreaterThanOrEqual(e.x, 0)
        XCTAssertLessThanOrEqual(e.x + e.width, 1.0 + 1e-9)
        XCTAssertLessThanOrEqual(e.y + e.height, 1.0 + 1e-9)
    }
}

final class StageMetricsTests: XCTestCase {
    func testPercentileAndCounts() {
        let m = StageMetrics(name: "x", window: 100)
        for v in 1...100 { m.record(ms: Double(v)) }
        let s = m.snapshot()
        XCTAssertEqual(s.processed, 100)
        XCTAssertEqual(s.maxMs, 100)
        XCTAssertEqual(s.p95Ms, 96, accuracy: 1.0)
        XCTAssertEqual(s.meanMs, 50.5, accuracy: 0.5)
    }
}
