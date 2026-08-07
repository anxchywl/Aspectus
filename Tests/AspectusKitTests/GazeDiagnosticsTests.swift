import XCTest
@testable import AspectusKit

private let aspect = 1280.0 / 720.0

private func eye(pupilDX: Double = 0, pupilDY: Double = 0,
                 openness: Double = 1.0,
                 width: Double = 0.06, height: Double = 0.02,
                 source: PupilSource = .visionLandmark,
                 points: Int = 1) -> EyeObservation {
    let region = NormRect(x: 0.4, y: 0.4, width: width, height: height)
    let c = region.center
    return EyeObservation(region: region,
                          pupilCenter: NormPoint(x: c.x + pupilDX, y: c.y + pupilDY),
                          openness: openness,
                          pupilSource: source,
                          pupilPointCount: points)
}

private func tracking(left: EyeObservation, right: EyeObservation,
                      yaw: Double = 0, pitch: Double = 0,
                      confidence: Double = 0.9) -> TrackingResult {
    TrackingResult(faceBounds: NormRect(x: 0.3, y: 0.2, width: 0.4, height: 0.5),
                   leftEye: left, rightEye: right,
                   headPose: HeadPose(yaw: yaw, pitch: pitch, roll: 0),
                   confidence: confidence)
}

final class PupilOffsetTests: XCTestCase {
    func testOffsetFallsBackToTheApertureCentreWithoutCorners() {
        let e = eye(pupilDX: 0.007, pupilDY: -0.003)
        XCTAssertEqual(e.pupilOffset.x, 0.007, accuracy: 1e-12)
        XCTAssertEqual(e.pupilOffset.y, -0.003, accuracy: 1e-12)
    }

    func testVerticalOffsetUsesTheCornerLineWhileHorizontalUsesTheApertureCentre() {
        var e = eye(pupilDX: 0.007, pupilDY: -0.003)
        e.cornerMidpointY = e.region.center.y - 0.001
        XCTAssertEqual(e.pupilOffset.x, 0.007, accuracy: 1e-12)
        XCTAssertEqual(e.pupilOffset.y, -0.002, accuracy: 1e-12)
    }

    func testSmoothingPreservesThePupilSource() {
        var s = TemporalStabilizer()
        let contour = eye(source: .contourCentroid, points: 0)
        let filtered = s.stabilize(tracking(left: contour, right: contour), t: 0)
        XCTAssertEqual(filtered.leftEye.pupilSource, .contourCentroid,
                       "filtering geometry must not erase where the pupil came from")
        XCTAssertEqual(filtered.leftEye.pupilPointCount, 0)
    }

    func testCornerMidpointUsesTheHorizontalExtremes() {
        let contour = [NormPoint(x: 0.40, y: 0.44),
                       NormPoint(x: 0.42, y: 0.39),
                       NormPoint(x: 0.46, y: 0.42),
                       NormPoint(x: 0.43, y: 0.47)]
        let midpoint = EyeObservation.cornerMidpointY(of: contour)
        XCTAssertEqual(midpoint ?? 0, 0.43, accuracy: 1e-12)
    }

    func testCornerRelativePupilPositionIgnoresWholeEyeTranslation() {
        let contour = [NormPoint(x: 0.40, y: 0.44), NormPoint(x: 0.46, y: 0.42)]
        let translated = contour.map { NormPoint(x: $0.x + 0.1, y: $0.y + 0.2) }
        let anchor = EyeObservation.cornerMidpointY(of: contour)!
        let translatedAnchor = EyeObservation.cornerMidpointY(of: translated)!
        XCTAssertEqual(0.41 - anchor, 0.61 - translatedAnchor, accuracy: 1e-12)
    }
}

final class GazeRejectionTests: XCTestCase {
    func testNoRejectionWhenGeometryIsUsable() {
        let t = tracking(left: eye(), right: eye())
        XCTAssertEqual(GazeGeometry.rejection(t, imageAspect: aspect, requiringBothEyes: true), .none)
    }

    func testExcessiveHeadPoseIsNamed() {
        let t = tracking(left: eye(), right: eye(), yaw: 40 * .pi / 180)
        XCTAssertEqual(GazeGeometry.rejection(t, imageAspect: aspect, requiringBothEyes: false),
                       .headPose)
    }

    func testClosedEyesAreNamed() {
        let shut = eye(openness: 0.0)
        XCTAssertEqual(GazeGeometry.rejection(tracking(left: shut, right: shut),
                                              imageAspect: aspect, requiringBothEyes: false),
                       .eyesClosed)
    }

    func testDegenerateRegionIsNamed() {
        let degenerate = EyeObservation(region: NormRect(x: 0.4, y: 0.4, width: 0, height: 0),
                                        pupilCenter: NormPoint(x: 0.4, y: 0.4), openness: 1)
        XCTAssertEqual(GazeGeometry.rejection(tracking(left: degenerate, right: degenerate),
                                              imageAspect: aspect, requiringBothEyes: false),
                       .degenerateGeometry)
    }

    // the estimate survives on one eye and the warp does not, so the reported reason must follow
    // whichever threshold the caller is actually asking about
    func testOneClosedEyeRejectsOnlyTheWarp() {
        let t = tracking(left: eye(), right: eye(openness: 0.0))
        XCTAssertEqual(GazeGeometry.rejection(t, imageAspect: aspect, requiringBothEyes: false),
                       .none)
        XCTAssertEqual(GazeGeometry.rejection(t, imageAspect: aspect, requiringBothEyes: true),
                       .eyesClosed)
    }

    func testRejectionAgreesWithEstimate() {
        let cases = [tracking(left: eye(), right: eye()),
                     tracking(left: eye(openness: 0), right: eye(openness: 0)),
                     tracking(left: eye(), right: eye(), yaw: 40 * .pi / 180)]
        for t in cases {
            let rejected = GazeGeometry.rejection(t, imageAspect: aspect,
                                                  requiringBothEyes: false) != .none
            let estimated = GazeGeometry.estimate(t, imageAspect: aspect) == nil
            XCTAssertEqual(rejected, estimated,
                           "the diagnostic must never disagree with the gate it describes")
        }
    }

    func testRejectionAgreesWithWarps() {
        let cases = [tracking(left: eye(), right: eye()),
                     tracking(left: eye(), right: eye(openness: 0)),
                     tracking(left: eye(openness: 0), right: eye(openness: 0))]
        let request = CorrectionRequest(yawOffset: 0.1, pitchOffset: 0.1, strength: 1)
        for t in cases {
            let rejected = GazeGeometry.rejection(t, imageAspect: aspect,
                                                  requiringBothEyes: true) != .none
            let warped = GazeGeometry.warps(t, imageAspect: aspect, request: request) == nil
            XCTAssertEqual(rejected, warped)
        }
    }
}

final class CorrectionGateAngleFactorTests: XCTestCase {
    func testAngleFactorIsOneInsideTheLimit() {
        var gate = CorrectionGate()
        _ = gate.update(confidence: 0.9, requestedCorrectionDegrees: 12, t: 0)
        XCTAssertEqual(gate.lastAngleFactor, 1.0, accuracy: 1e-12)
    }

    // the silent decay: a request past the limit weakens correction with nothing else to show it
    func testAngleFactorDecaysPastTheLimit() {
        var gate = CorrectionGate(config: .init(maxCorrectionDegrees: 18))
        _ = gate.update(confidence: 0.9, requestedCorrectionDegrees: 21, t: 0)
        XCTAssertEqual(gate.lastAngleFactor, 0.5, accuracy: 1e-12,
                       "half way through the six degree guard band")
    }

    func testAngleFactorReachesZeroAtTheEndOfTheGuardBand() {
        var gate = CorrectionGate(config: .init(maxCorrectionDegrees: 18))
        _ = gate.update(confidence: 0.9, requestedCorrectionDegrees: 24, t: 0)
        XCTAssertEqual(gate.lastAngleFactor, 0.0, accuracy: 1e-12)
    }

    func testAngleFactorIsSignIndependent() {
        var gate = CorrectionGate(config: .init(maxCorrectionDegrees: 18))
        _ = gate.update(confidence: 0.9, requestedCorrectionDegrees: -21, t: 0)
        XCTAssertEqual(gate.lastAngleFactor, 0.5, accuracy: 1e-12)
    }
}

final class ValueStatsTests: XCTestCase {
    func testEmptyStatsReadAsZeroRatherThanInfinity() {
        let s = ValueStats()
        XCTAssertEqual(s.count, 0)
        XCTAssertEqual(s.minimum, 0)
        XCTAssertEqual(s.maximum, 0)
        XCTAssertEqual(s.mean, 0)
    }

    func testTracksMinMeanMax() {
        var s = ValueStats()
        for v in [0.2, 0.8, 0.5] { s.add(v) }
        XCTAssertEqual(s.count, 3)
        XCTAssertEqual(s.minimum, 0.2, accuracy: 1e-12)
        XCTAssertEqual(s.maximum, 0.8, accuracy: 1e-12)
        XCTAssertEqual(s.mean, 0.5, accuracy: 1e-12)
    }

    func testNonFiniteSamplesAreIgnored() {
        var s = ValueStats()
        s.add(0.5)
        s.add(.nan)
        s.add(.infinity)
        XCTAssertEqual(s.count, 1, "a bad sample must not poison the whole distribution")
        XCTAssertEqual(s.mean, 0.5, accuracy: 1e-12)
    }
}

final class DiagnosticsCollectorTests: XCTestCase {
    private func sample(id: UInt64 = 0,
                        confidence: Double = 0.9,
                        offsetY: Double = 0,
                        source: PupilSource = .visionLandmark,
                        ageMs: Double = 33,
                        fallback: FallbackReason = .none) -> GazeSample {
        let e = eye(pupilDY: offsetY, source: source)
        return GazeSample(frameID: FrameID(id),
                          left: EyeSample(e), right: EyeSample(e),
                          faceConfidence: confidence,
                          gazeConfidence: confidence,
                          correctionAgeMs: ageMs,
                          fallback: fallback)
    }

    func testEmptySnapshotIsSafeToDisplay() {
        let s = DiagnosticsCollector().snapshot()
        XCTAssertNil(s.latest)
        XCTAssertEqual(s.frames, 0)
        XCTAssertEqual(s.visionPupilShare, 0)
        XCTAssertEqual(s.dominantFallback, .none)
    }

    func testLatestSampleWins() {
        let c = DiagnosticsCollector()
        c.record(sample(id: 1))
        c.record(sample(id: 2))
        XCTAssertEqual(c.snapshot().latest?.frameID, FrameID(2),
                       "the HUD shows the newest frame, never an average of positions")
    }

    func testVisionPupilShareCountsEyesNotFrames() {
        let c = DiagnosticsCollector()
        c.record(sample(source: .visionLandmark))
        c.record(sample(source: .contourCentroid))
        XCTAssertEqual(c.snapshot().visionPupilShare, 0.5, accuracy: 1e-12,
                       "two eyes per frame, half of them from a real landmark")
    }

    func testVerticalOffsetPoolsBothEyes() {
        let c = DiagnosticsCollector()
        c.record(sample(offsetY: -0.004))
        c.record(sample(offsetY: -0.002))
        let s = c.snapshot()
        XCTAssertEqual(s.verticalPupilOffset.count, 4)
        XCTAssertEqual(s.verticalPupilOffset.mean, -0.003, accuracy: 1e-12)
        XCTAssertEqual(s.verticalPupilOffset.minimum, -0.004, accuracy: 1e-12)
    }

    func testDominantFallbackIgnoresSuccessfulFrames() {
        let c = DiagnosticsCollector()
        for _ in 0..<10 { c.record(sample(fallback: .none)) }
        for _ in 0..<3 { c.record(sample(fallback: .lowConfidence)) }
        for _ in 0..<1 { c.record(sample(fallback: .headPose)) }
        let s = c.snapshot()
        XCTAssertEqual(s.dominantFallback, .lowConfidence,
                       "the gate worth fixing is the most frequent one that fired, not the mode")
        XCTAssertEqual(s.correctedFrames, 10)
        XCTAssertEqual(s.frames, 14)
    }

    func testZeroAgeIsNotRecordedAsALatency() {
        let c = DiagnosticsCollector()
        c.record(sample(ageMs: 0))
        XCTAssertEqual(c.snapshot().correctionAgeMeanMs, 0,
                       "the first frame has no previous frame to be stale against")
        c.record(sample(ageMs: 33))
        XCTAssertEqual(c.snapshot().correctionAgeMeanMs, 33, accuracy: 1e-9)
    }

    func testResetClearsDistributionsAndAge() {
        let c = DiagnosticsCollector()
        for _ in 0..<5 { c.record(sample(fallback: .lowConfidence)) }
        c.reset()
        let s = c.snapshot()
        XCTAssertEqual(s.frames, 0)
        XCTAssertNil(s.latest)
        XCTAssertEqual(s.correctionAgeMeanMs, 0)
        XCTAssertEqual(s.faceConfidence.count, 0)
        XCTAssertEqual(s.dominantFallback, .none,
                       "a restart must not report the previous session's gates")
    }

    func testConcurrentRecordingIsSafe() async {
        let c = DiagnosticsCollector()
        let one = sample()
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<8 {
                group.addTask { for _ in 0..<200 { c.record(one) } }
            }
        }
        XCTAssertEqual(c.snapshot().frames, 1600,
                       "the collector is written from the detached loop and read from the timer")
    }
}
