import XCTest
@testable import AspectusKit

private let aspect = 1280.0 / 720.0

private func observation(pupilX: Double, pupilY: Double = 0.405, openness: Double = 1.0,
                         cornerMidpointY: Double? = nil) -> EyeObservation {
    EyeObservation(region: NormRect(x: 0.4, y: 0.4, width: 0.06, height: 0.02),
                   pupilCenter: NormPoint(x: pupilX, y: pupilY),
                   openness: openness,
                   cornerMidpointY: cornerMidpointY)
}

private func result(pupilX: Double, openness: Double = 1.0,
                    confidence: Double = 0.9) -> TrackingResult {
    TrackingResult(faceBounds: NormRect(x: 0.3, y: 0.2, width: 0.4, height: 0.5),
                   leftEye: observation(pupilX: pupilX, openness: openness),
                   rightEye: observation(pupilX: pupilX, openness: openness),
                   headPose: HeadPose(yaw: 0, pitch: 0, roll: 0),
                   confidence: confidence)
}

private func variance(_ a: [Double]) -> Double {
    let m = a.reduce(0, +) / Double(a.count)
    return a.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Double(a.count)
}

/// deterministic xorshift so jitter tests never flake
private struct Noise {
    var seed: UInt64 = 0x9E3779B97F4A7C15
    mutating func next() -> Double {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17
        return Double(seed % 2000) / 1000.0 - 1.0
    }
}

final class TemporalStabilizerTests: XCTestCase {
    func testSmoothingReducesPupilJitterOnAStillFace() {
        var s = TemporalStabilizer()
        var noise = Noise()
        var raw: [Double] = [], smoothed: [Double] = []
        for i in 0..<120 {
            let t = Double(i) / 30.0
            let x = 0.43 + 0.002 * noise.next()
            raw.append(x)
            smoothed.append(s.stabilize(result(pupilX: x), t: t).leftEye.pupilCenter.x)
        }
        // compare the settled tail, the first samples are the filter warming up
        XCTAssertLessThan(variance(Array(smoothed[60...])), variance(Array(raw[60...])) * 0.5,
                          "a still face must produce a still warp centre")
    }

    func testOpennessIsNeverSmoothed() {
        var s = TemporalStabilizer()
        for i in 0..<10 { _ = s.stabilize(result(pupilX: 0.43), t: Double(i) / 30.0) }
        // a blink is a one-frame transient, it must survive intact
        let blink = s.stabilize(result(pupilX: 0.43, openness: 0.0), t: 10.0 / 30.0)
        XCTAssertEqual(blink.leftEye.openness, 0.0, accuracy: 1e-12,
                       "smoothing openness would slur the blink and delay the safety gate")
        XCTAssertEqual(blink.rightEye.openness, 0.0, accuracy: 1e-12)
    }

    func testCornerMidpointIsSmoothedWithTheLandmarks() {
        var s = TemporalStabilizer()
        var first = result(pupilX: 0.43)
        first.leftEye.cornerMidpointY = 0.41
        _ = s.stabilize(first, t: 0)
        first.leftEye.cornerMidpointY = 0.51
        let filtered = s.stabilize(first, t: 1.0 / 30.0)
        XCTAssertLessThan(filtered.leftEye.cornerMidpointY ?? 1, 0.51)
    }

    func testEyeAxisEvidenceSurvivesLandmarkSmoothing() throws {
        var s = TemporalStabilizer()
        var first = result(pupilX: 0.43)
        first.leftEye.contourPointCount = 8
        first.leftEye.imageAxisStart = .init(x: 0.40, y: 0.40)
        first.leftEye.imageAxisEnd = .init(x: 0.46, y: 0.42)
        _ = s.stabilize(first, t: 0)

        first.leftEye.imageAxisStart = .init(x: 0.42, y: 0.40)
        first.leftEye.imageAxisEnd = .init(x: 0.48, y: 0.42)
        let filtered = s.stabilize(first, t: 1.0 / 30.0)

        XCTAssertEqual(filtered.leftEye.contourPointCount, 8)
        XCTAssertLessThan(try XCTUnwrap(filtered.leftEye.imageAxisStart).x, 0.42)
        XCTAssertLessThan(try XCTUnwrap(filtered.leftEye.imageAxisEnd).x, 0.48)
    }

    func testResetMakesTheNextSamplePassThrough() {
        var s = TemporalStabilizer()
        for i in 0..<30 { _ = s.stabilize(result(pupilX: 0.43), t: Double(i) / 30.0) }
        s.reset()
        let after = s.stabilize(result(pupilX: 0.50), t: 30.0 / 30.0)
        XCTAssertEqual(after.leftEye.pupilCenter.x, 0.50, accuracy: 1e-12,
                       "recovery must not blend against pre-loss history")
    }

    func testForwardTimestampJumpResetsAutomatically() {
        var s = TemporalStabilizer(config: .init(maxFrameGap: 0.25))
        for i in 0..<30 { _ = s.stabilize(result(pupilX: 0.43), t: Double(i) / 30.0) }
        XCTAssertTrue(s.isDiscontinuous(at: 1.0 + 5.0))
        let after = s.stabilize(result(pupilX: 0.50), t: 1.0 + 5.0)
        XCTAssertEqual(after.leftEye.pupilCenter.x, 0.50, accuracy: 1e-12,
                       "a five second gap means the history is stale, e.g. after sleep/wake")
    }

    func testBackwardTimestampResets() {
        var s = TemporalStabilizer()
        for i in 0..<30 { _ = s.stabilize(result(pupilX: 0.43), t: Double(i) / 30.0) }
        XCTAssertTrue(s.isDiscontinuous(at: 0.1))
        let after = s.stabilize(result(pupilX: 0.50), t: 0.1)
        XCTAssertEqual(after.leftEye.pupilCenter.x, 0.50, accuracy: 1e-12,
                       "a clock that goes backwards must not produce a negative dt")
    }

    func testSmallGapsDoNotReset() {
        var s = TemporalStabilizer(config: .init(maxFrameGap: 0.25))
        for i in 0..<30 { _ = s.stabilize(result(pupilX: 0.43), t: Double(i) / 30.0) }
        XCTAssertFalse(s.isDiscontinuous(at: 1.0 + 1.0 / 30.0),
                       "an ordinary frame interval is not a discontinuity")
        let after = s.stabilize(result(pupilX: 0.50), t: 1.0 + 1.0 / 30.0)
        XCTAssertLessThan(after.leftEye.pupilCenter.x, 0.50,
                          "a normal frame must still be smoothed")
    }
}

final class TemporalPipelineBehaviourTests: XCTestCase {
    private func request(_ gaze: GazeEstimate, redirectDegrees: Double = 12) -> CorrectionRequest {
        CorrectionRequest(yawOffset: -gaze.yaw,
                          pitchOffset: -gaze.pitch + redirectDegrees * .pi / 180,
                          strength: 1)
    }

    func testBlinkSuppressesCorrectionOnTheVeryNextFrame() {
        var s = TemporalStabilizer()
        var t = 0.0
        for _ in 0..<30 {
            _ = s.stabilize(result(pupilX: 0.43), t: t)
            t += 1.0 / 30.0
        }
        let blinking = s.stabilize(result(pupilX: 0.43, openness: 0.0), t: t)
        let gaze = GazeEstimate(yaw: 0, pitch: 0, confidence: 0.9)
        XCTAssertNil(GazeGeometry.warps(blinking, imageAspect: aspect, request: request(gaze)),
                     "correction must stop the instant the eye closes, with no smoothing lag")
    }

    func testStrengthRecoversWithinTheDesignBudgetAfterTrackingLoss() {
        var gate = CorrectionGate(config: .init(slewPerSecond: 6.0))
        var t = 0.0
        // engaged and fully ramped
        for _ in 0..<120 {
            _ = gate.update(confidence: 0.9, requestedCorrectionDegrees: 12, t: t)
            t += 1.0 / 30.0
        }
        XCTAssertEqual(gate.currentWeight, 1.0, accuracy: 1e-6)

        // tracking lost
        gate.forceFallback()
        for _ in 0..<30 {
            _ = gate.update(confidence: 0, requestedCorrectionDegrees: 0, t: t)
            t += 1.0 / 30.0
        }
        XCTAssertEqual(gate.currentWeight, 0.0, accuracy: 1e-6, "fallback must fade fully out")

        // tracking regained, measure how long full strength takes to come back
        let recoveryStart = t
        var recovered = 0.0
        while t - recoveryStart <= 0.30 {
            recovered = gate.update(confidence: 0.9, requestedCorrectionDegrees: 12, t: t)
            t += 1.0 / 30.0
        }
        XCTAssertEqual(recovered, 1.0, accuracy: 1e-6,
                       "design target is smooth recovery within about 300 ms")
    }

    func testRecoveryRampsRatherThanPopping() {
        var gate = CorrectionGate(config: .init(slewPerSecond: 6.0))
        var t = 0.0
        _ = gate.update(confidence: 0, requestedCorrectionDegrees: 0, t: t)
        t += 1.0 / 30.0
        var previous = 0.0
        var framesToFull = 0
        while previous < 1.0, framesToFull < 100 {
            let w = gate.update(confidence: 0.9, requestedCorrectionDegrees: 12, t: t)
            XCTAssertLessThanOrEqual(w - previous, 6.0 / 30.0 + 1e-9,
                                     "no frame may jump more than the slew limit allows")
            previous = w
            framesToFull += 1
            t += 1.0 / 30.0
        }
        XCTAssertGreaterThanOrEqual(framesToFull, 5,
                                    "engaging must span several frames rather than popping on")
    }

    func testConfidenceDipInsideHysteresisBandDoesNotDisengage() {
        var gate = CorrectionGate(config: .init(enterConfidence: 0.6, exitConfidence: 0.4,
                                                slewPerSecond: 6.0))
        var t = 0.0
        for _ in 0..<60 {
            _ = gate.update(confidence: 0.9, requestedCorrectionDegrees: 12, t: t)
            t += 1.0 / 30.0
        }
        // a dip that stays above the exit threshold must not flicker correction off
        for _ in 0..<10 {
            _ = gate.update(confidence: 0.5, requestedCorrectionDegrees: 12, t: t)
            t += 1.0 / 30.0
        }
        XCTAssertTrue(gate.isEngaged)
        XCTAssertEqual(gate.currentWeight, 1.0, accuracy: 1e-6,
                       "hysteresis exists so ordinary confidence noise cannot cause flicker")
    }
}
