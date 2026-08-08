import XCTest
@testable import AspectusKit

private let aspect = 1280.0 / 720.0
private let toRadians = Double.pi / 180

private func eye(pupilDX: Double = 0, pupilDY: Double = 0,
                 openness: Double = 1.0,
                 width: Double = 0.06, height: Double = 0.02) -> EyeObservation {
    let region = NormRect(x: 0.4, y: 0.4, width: width, height: height)
    let c = region.center
    return EyeObservation(region: region,
                          pupilCenter: NormPoint(x: c.x + pupilDX, y: c.y + pupilDY),
                          openness: openness,
                          pupilSource: .visionLandmark, pupilPointCount: 1)
}

private func tracking(dx: Double = 0, dy: Double = 0,
                      openness: Double = 1.0,
                      headYaw: Double = 0, headPitch: Double = 0, headRoll: Double = 0,
                      confidence: Double = 1.0,
                      rightDX: Double? = nil) -> TrackingResult {
    TrackingResult(faceBounds: NormRect(x: 0.3, y: 0.2, width: 0.4, height: 0.5),
                   leftEye: eye(pupilDX: dx, pupilDY: dy, openness: openness),
                   rightEye: eye(pupilDX: rightDX ?? dx, pupilDY: dy, openness: openness),
                   headPose: HeadPose(yaw: headYaw, pitch: headPitch, roll: headRoll),
                   confidence: confidence)
}

private func sample(_ target: CalibrationTarget, yaw: Double, pitch: Double) -> CalibrationSample {
    CalibrationSample(target: target, yawDegrees: yaw, pitchDegrees: pitch,
                      leftYawDegrees: yaw, leftPitchDegrees: pitch,
                      rightYawDegrees: yaw, rightPitchDegrees: pitch,
                      headYawDegrees: 0, headPitchDegrees: 0, headRollDegrees: 0,
                      confidence: 1)
}

/// a well-formed set: a +4 degree neutral bias on pitch, clear separation on both axes
private func goodSamples(count: Int = 12,
                         yawBias: Double = 0,
                         pitchBias: Double = 4) -> [CalibrationSample] {
    var out: [CalibrationSample] = []
    for _ in 0..<count {
        out.append(sample(.center, yaw: yawBias, pitch: pitchBias))
        out.append(sample(.up, yaw: yawBias, pitch: pitchBias + 6))
        out.append(sample(.down, yaw: yawBias, pitch: pitchBias - 6))
        out.append(sample(.left, yaw: yawBias - 6, pitch: pitchBias))
        out.append(sample(.right, yaw: yawBias + 6, pitch: pitchBias))
    }
    return out
}

final class GazeCalibrationModelTests: XCTestCase {
    func testApplyRemovesTheNeutralBias() {
        let c = GazeCalibration(yawOffsetDegrees: -1.5, pitchOffsetDegrees: 4.0)
        let raw = GazeEstimate(yaw: -1.5 * toRadians, pitch: 4.0 * toRadians, confidence: 0.9)
        let out = c.apply(raw)
        XCTAssertEqual(out.yaw, 0, accuracy: 1e-12,
                       "looking at the lens must read as zero after calibration")
        XCTAssertEqual(out.pitch, 0, accuracy: 1e-12)
    }

    func testApplyPreservesConfidence() {
        let c = GazeCalibration(yawOffsetDegrees: 1, pitchOffsetDegrees: 1)
        XCTAssertEqual(c.apply(GazeEstimate(yaw: 0, pitch: 0, confidence: 0.42)).confidence, 0.42,
                       accuracy: 1e-12, "calibration corrects angles, it does not judge trust")
    }

    func testApplyIsIdentityForAZeroCalibration() {
        let c = GazeCalibration(yawOffsetDegrees: 0, pitchOffsetDegrees: 0)
        let raw = GazeEstimate(yaw: 0.1, pitch: -0.05, confidence: 1)
        XCTAssertEqual(c.apply(raw).yaw, 0.1, accuracy: 1e-12)
        XCTAssertEqual(c.apply(raw).pitch, -0.05, accuracy: 1e-12)
    }

    func testFutureVersionIsRejected() {
        var c = GazeCalibration(yawOffsetDegrees: 0, pitchOffsetDegrees: 0)
        c.version = GazeCalibration.currentVersion + 1
        XCTAssertFalse(c.isUsable, "a file from a newer build must be ignored, not guessed at")
    }

    func testVersionTwoCalibrationRemainsUsableDuringMigration() {
        XCTAssertTrue(GazeCalibration(version: 2, yawOffsetDegrees: 0,
                                      pitchOffsetDegrees: 0).isUsable)
    }

    func testApertureRelativeCalibrationIsRejected() {
        let c = GazeCalibration(version: 1, yawOffsetDegrees: 0, pitchOffsetDegrees: 0)
        XCTAssertFalse(c.isUsable,
                       "version 1 was fitted against the incompatible eyelid-centre vertical signal")
    }

    func testImplausibleOffsetIsRejected() {
        XCTAssertFalse(GazeCalibration(yawOffsetDegrees: 0, pitchOffsetDegrees: 45).isUsable)
        XCTAssertFalse(GazeCalibration(yawOffsetDegrees: .nan, pitchOffsetDegrees: 0).isUsable)
        XCTAssertFalse(GazeCalibration(yawOffsetDegrees: 0, pitchOffsetDegrees: 0, pitchGain: 0).isUsable)
    }

    func testRoundTripsThroughJSON() throws {
        let original = try CalibrationFit.fit(goodSamples())
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(GazeCalibration.self, from: data)
        XCTAssertEqual(decoded.yawOffsetDegrees, original.yawOffsetDegrees, accuracy: 1e-9)
        XCTAssertEqual(decoded.pitchOffsetDegrees, original.pitchOffsetDegrees, accuracy: 1e-9)
        XCTAssertTrue(decoded.isUsable)
    }
}

final class CalibrationFitTests: XCTestCase {
    func testFitsTheNeutralBiasFromTheCentreTargetOnly() throws {
        // the off-axis targets are deliberately asymmetric: only the centre may set the baseline
        var samples = goodSamples(pitchBias: 4)
        samples.append(contentsOf: (0..<12).map { _ in sample(.up, yaw: 0, pitch: 40) })
        let fitted = try CalibrationFit.fit(samples)
        XCTAssertEqual(fitted.pitchOffsetDegrees, 4, accuracy: 1e-9,
                       "looking at the lens is the only target with known ground truth")
    }

    func testMeasuresTheBiasSeenOnHardware() throws {
        let fitted = try CalibrationFit.fit(goodSamples(yawBias: -1.2, pitchBias: 4.3))
        XCTAssertEqual(fitted.yawOffsetDegrees, -1.2, accuracy: 1e-9)
        XCTAssertEqual(fitted.pitchOffsetDegrees, 4.3, accuracy: 1e-9)
        XCTAssertEqual(fitted.verticalSeparationDegrees, 12, accuracy: 1e-9)
        XCTAssertEqual(fitted.horizontalSeparationDegrees, 12, accuracy: 1e-9)
    }

    func testGainIsPinnedWhenNoGeometryIsSupplied() throws {
        let fitted = try CalibrationFit.fit(goodSamples())
        XCTAssertEqual(fitted.yawGain, 1.0, accuracy: 1e-12)
        XCTAssertEqual(fitted.pitchGain, 1.0, accuracy: 1e-12,
                       "slope needs the true angle of the off-axis targets, so without display "
                       + "geometry and a distance the only honest slope is one")
    }

    func testMissingTargetFails() {
        let samples = goodSamples().filter { $0.target != .left }
        XCTAssertThrowsError(try CalibrationFit.fit(samples)) { error in
            guard case .notEnoughSamples(let target, _, _) = error as? CalibrationFit.Failure else {
                return XCTFail("expected a missing-target failure, got \(error)")
            }
            XCTAssertEqual(target, .left)
        }
    }

    func testTooFewSamplesFails() {
        XCTAssertThrowsError(try CalibrationFit.fit(goodSamples(count: 2)))
    }

    // a flat vertical axis must fail loudly rather than produce a confident-looking calibration
    func testNoVerticalSeparationFails() {
        let flat = (0..<12).flatMap { _ in
            [sample(.center, yaw: 0, pitch: 4), sample(.up, yaw: 0, pitch: 4),
             sample(.down, yaw: 0, pitch: 4), sample(.left, yaw: -6, pitch: 4),
             sample(.right, yaw: 6, pitch: 4)]
        }
        XCTAssertThrowsError(try CalibrationFit.fit(flat)) { error in
            guard case .verticalNotSeparated = error as? CalibrationFit.Failure else {
                return XCTFail("expected vertical separation failure, got \(error)")
            }
        }
    }

    func testInvertedVerticalSignFails() {
        let inverted = (0..<12).flatMap { _ in
            [sample(.center, yaw: 0, pitch: 0), sample(.up, yaw: 0, pitch: -6),
             sample(.down, yaw: 0, pitch: 6), sample(.left, yaw: -6, pitch: 0),
             sample(.right, yaw: 6, pitch: 0)]
        }
        XCTAssertThrowsError(try CalibrationFit.fit(inverted)) { error in
            guard case .verticalSignInverted = error as? CalibrationFit.Failure else {
                return XCTFail("expected an inverted-sign failure, got \(error)")
            }
        }
    }

    func testInvertedHorizontalSignFails() {
        let inverted = (0..<12).flatMap { _ in
            [sample(.center, yaw: 0, pitch: 0), sample(.up, yaw: 0, pitch: 6),
             sample(.down, yaw: 0, pitch: -6), sample(.left, yaw: 6, pitch: 0),
             sample(.right, yaw: -6, pitch: 0)]
        }
        XCTAssertThrowsError(try CalibrationFit.fit(inverted)) { error in
            guard case .horizontalSignInverted = error as? CalibrationFit.Failure else {
                return XCTFail("expected an inverted-sign failure, got \(error)")
            }
        }
    }

    func testImplausibleBiasFails() {
        XCTAssertThrowsError(try CalibrationFit.fit(goodSamples(pitchBias: 35))) { error in
            guard case .implausibleOffset = error as? CalibrationFit.Failure else {
                return XCTFail("expected an implausible-offset failure, got \(error)")
            }
        }
    }
}

final class CalibrationSessionTests: XCTestCase {
    /// settle is disabled here so each test drives exactly the rule it is about; the settle
    /// window has its own tests below
    private func config(_ perTarget: Int = 3,
                        settle: Double = 0) -> CalibrationSession.Config {
        CalibrationSession.Config(samplesPerTarget: perTarget, settleSeconds: settle)
    }

    /// steady, well-separated observations for whichever target the session is currently on
    private func feed(_ session: inout CalibrationSession, frames: Int,
                      dx: Double = 0, dy: Double = 0, from t0: Double = 0) -> [CalibrationSession.Outcome] {
        var out: [CalibrationSession.Outcome] = []
        for i in 0..<frames {
            out.append(session.offer(tracking(dx: dx, dy: dy), imageAspect: aspect,
                                     t: t0 + Double(i) / 30.0))
        }
        return out
    }

    func testStartsOnTheCentreTarget() {
        XCTAssertEqual(CalibrationSession().target, .center)
    }

    func testSettleWindowCollectsNothing() {
        var s = CalibrationSession(config: config(3, settle: 2.0))
        let outcome = s.offer(tracking(), imageAspect: aspect, t: 0)
        guard case let .settling(remaining) = outcome else {
            return XCTFail("expected settling, got \(outcome)")
        }
        XCTAssertEqual(remaining, 2.0, accuracy: 1e-9)
        XCTAssertTrue(s.samples.isEmpty, "the move to a target must never be measured")
    }

    func testCollectionStartsOnceTheSettleWindowExpires() {
        var s = CalibrationSession(config: config(3, settle: 0.5))
        var t = 0.0
        while t < 0.5 {
            _ = s.offer(tracking(), imageAspect: aspect, t: t)
            t += 1.0 / 30.0
        }
        XCTAssertTrue(s.samples.isEmpty)
        XCTAssertEqual(s.offer(tracking(), imageAspect: aspect, t: t), .accepted)
    }

    func testEachTargetGetsItsOwnSettleWindow() {
        var s = CalibrationSession(config: config(1, settle: 0.2))
        var t = 0.0
        // first target: settle, then one sample completes it
        while s.target == .center, t < 5 {
            _ = s.offer(tracking(), imageAspect: aspect, t: t)
            t += 1.0 / 30.0
        }
        XCTAssertEqual(s.target, .up)
        guard case .settling = s.offer(tracking(), imageAspect: aspect, t: t) else {
            return XCTFail("the second target must settle before it measures")
        }
    }

    func testMeansReportWhatEachTargetRead() {
        var s = CalibrationSession(config: config(3))
        _ = feed(&s, frames: 3, dy: -0.002)
        let means = s.means(for: .center)
        XCTAssertEqual(means?.count, 3)
        XCTAssertGreaterThan(means?.pitchDegrees ?? 0, 0,
                             "a pupil above the aperture centre reads as looking up")
        XCTAssertNil(s.means(for: .right), "a target with no samples has no reading")
    }

    func testSamplesCarryThePupilOffsetTheAnglesCameFrom() {
        var s = CalibrationSession(config: config(3))
        _ = feed(&s, frames: 3, dy: -0.002)
        let recorded = s.samples.first
        XCTAssertEqual(recorded?.leftOffsetY ?? 0, -0.002, accuracy: 1e-9)
        XCTAssertEqual(recorded?.rightOffsetY ?? 0, -0.002, accuracy: 1e-9)
        XCTAssertEqual(s.means(for: .center)?.offsetY ?? 0, -0.002, accuracy: 1e-9,
                       "the per-target mean is what makes eye travel comparable across runs")
    }

    func testRecordedOffsetIsIndependentOfEyeWidth() {
        func offsetY(width: Double) -> Double {
            var s = CalibrationSession(config: config(3))
            var t = 3.0
            for _ in 0..<3 {
                let e = eye(pupilDY: -0.002, width: width)
                let r = TrackingResult(faceBounds: NormRect(x: 0.3, y: 0.2, width: 0.4, height: 0.5),
                                       leftEye: e, rightEye: e,
                                       headPose: HeadPose(yaw: 0, pitch: 0, roll: 0),
                                       confidence: 1.0)
                _ = s.offer(r, imageAspect: aspect, t: t)
                t += 1.0 / 30.0
            }
            return s.means(for: .center)?.offsetY ?? 0
        }
        XCTAssertEqual(offsetY(width: 0.06), offsetY(width: 0.03), accuracy: 1e-9)
    }

    func testCalibrationSampleDecodesFilesWrittenBeforeTheOffsetFields() throws {
        let legacy = """
        {"target":"center","yawDegrees":1,"pitchDegrees":2,"leftYawDegrees":1,\
        "leftPitchDegrees":2,"rightYawDegrees":1,"rightPitchDegrees":2,"headYawDegrees":0,\
        "headPitchDegrees":0,"headRollDegrees":0,"confidence":0.9}
        """
        let decoded = try JSONDecoder().decode(CalibrationSample.self,
                                               from: Data(legacy.utf8))
        XCTAssertEqual(decoded.yawDegrees, 1)
        XCTAssertEqual(decoded.offsetX, 0, "a missing offset reads as zero rather than failing")
        XCTAssertEqual(decoded.offsetY, 0)
    }

    /// drives the five fixation targets and stops at the head-motion sweep
    private func runTargets(_ s: inout CalibrationSession, from t0: Double = 0) -> [CalibrationTarget] {
        var seen: [CalibrationTarget] = []
        var t = t0
        while s.phase == .targets, t - t0 < 100 {
            seen.append(s.target)
            _ = s.offer(tracking(), imageAspect: aspect, t: t)
            t += 1.0 / 30.0
        }
        return seen
    }

    func testAdvancesThroughEveryTargetInOrder() {
        var s = CalibrationSession(config: config(3))
        let seen = runTargets(&s)
        XCTAssertEqual(Array(Set(seen)).count, CalibrationTarget.allCases.count)
        XCTAssertEqual(s.samples.count, 3 * CalibrationTarget.allCases.count)
        XCTAssertEqual(s.phase, .headMotion,
                       "the fixation targets hand over to the head sweep rather than finishing")
        XCTAssertFalse(s.isFinished)
    }

    func testSkippingTheSweepFinishesWithoutCompensation() throws {
        var s = CalibrationSession(config: config(12))
        // each target has to read differently or the fit rightly refuses the whole calibration
        let offsets: [CalibrationTarget: (Double, Double)] = [
            .center: (0, -0.0018), .up: (0, -0.0058), .down: (0, 0.0022),
            .left: (0.004, -0.0018), .right: (-0.004, -0.0018),
        ]
        var t = 0.0
        while s.phase == .targets, t < 100 {
            let (dx, dy) = offsets[s.target]!
            _ = s.offer(tracking(dx: dx, dy: dy), imageAspect: aspect, t: t)
            t += 1.0 / 30.0
        }
        s.skipHeadMotion()
        XCTAssertTrue(s.isFinished)
        XCTAssertTrue(s.headMotionSamples.isEmpty)
        XCTAssertNil(try s.fit().headCoupling,
                     "skipping the sweep must leave behaviour exactly as it was")
    }

    func testSweepCollectsOnlyWithHeadPose() {
        var s = CalibrationSession(config: config(3))
        _ = runTargets(&s)
        XCTAssertEqual(s.phase, .headMotion)
        let noPose = TrackingResult(faceBounds: NormRect(x: 0.3, y: 0.2, width: 0.4, height: 0.5),
                                    leftEye: eye(), rightEye: eye(),
                                    headPose: HeadPose(yaw: 0.2, pitch: 0.1, roll: 0),
                                    confidence: 1, headPoseAvailable: false)
        XCTAssertEqual(s.offer(noPose, imageAspect: aspect, t: 50), .rejected(.headPose),
                       "a sweep sample without a pose carries no information at all")
    }

    func testSweepAcceptsWideHeadRotationThatTheFixationStepsReject() {
        var s = CalibrationSession(config: config(3))
        _ = runTargets(&s)
        let turned = tracking(headYaw: 25 * toRadians)
        XCTAssertEqual(s.offer(turned, imageAspect: aspect, t: 50), .acceptedHeadMotion,
                       "turning the head is the point of this step, not a reason to reject")
    }

    func testRejectsBlinks() {
        var s = CalibrationSession(config: config())
        let outcome = s.offer(tracking(openness: 0.1), imageAspect: aspect, t: 0)
        XCTAssertEqual(outcome, .rejected(.eyesClosed))
        XCTAssertTrue(s.samples.isEmpty)
    }

    func testRejectsLowConfidence() {
        var s = CalibrationSession(config: config())
        XCTAssertEqual(s.offer(tracking(confidence: 0.2), imageAspect: aspect, t: 0),
                       .rejected(.lowConfidence))
    }

    func testRejectsLostTracking() {
        var s = CalibrationSession(config: config())
        XCTAssertEqual(s.offer(nil, imageAspect: aspect, t: 0), .rejected(.noFace))
    }

    func testRejectsLargeHeadPose() {
        var s = CalibrationSession(config: config())
        XCTAssertEqual(s.offer(tracking(headYaw: 30 * toRadians), imageAspect: aspect, t: 0),
                       .rejected(.headPose))
    }

    func testRejectsFastHeadMotion() {
        var s = CalibrationSession(config: config(10))
        _ = s.offer(tracking(headYaw: 0), imageAspect: aspect, t: 0)
        // ten degrees in a thirtieth of a second is three hundred degrees a second
        XCTAssertEqual(s.offer(tracking(headYaw: 10 * toRadians), imageAspect: aspect, t: 1.0 / 30),
                       .rejected(.headMotion))
    }

    func testRejectsMotionBlurProxy() {
        var s = CalibrationSession(config: config(10))
        _ = s.offer(tracking(dx: 0), imageAspect: aspect, t: 0)
        XCTAssertEqual(s.offer(tracking(dx: 0.02), imageAspect: aspect, t: 1.0 / 30),
                       .rejected(.motionBlur))
    }

    func testRejectsEyeDisagreement() {
        var s = CalibrationSession(config: config(10))
        // one eye reading far from the other means at least one pupil landmark is wrong
        XCTAssertEqual(s.offer(tracking(dx: 0, rightDX: 0.012), imageAspect: aspect, t: 0),
                       .rejected(.eyeDisagreement))
    }

    func testRejectsBackwardTimestamps() {
        var s = CalibrationSession(config: config(10))
        _ = s.offer(tracking(), imageAspect: aspect, t: 1.0)
        XCTAssertEqual(s.offer(tracking(), imageAspect: aspect, t: 0.5),
                       .rejected(.timestampDiscontinuity))
    }

    func testRejectsLongTimestampGaps() {
        var s = CalibrationSession(config: config(10))
        _ = s.offer(tracking(), imageAspect: aspect, t: 1.0)
        XCTAssertEqual(s.offer(tracking(), imageAspect: aspect, t: 6.0),
                       .rejected(.timestampDiscontinuity))
    }

    func testMotionIsJudgedAgainstTheLastOfferedFrameNotTheLastAccepted() {
        var s = CalibrationSession(config: config(10))
        _ = s.offer(tracking(headYaw: 0), imageAspect: aspect, t: 0)
        // a rejected frame still moves the baseline, so a fast move cannot be laundered by
        // hiding it behind rejections
        _ = s.offer(tracking(headYaw: 10 * toRadians), imageAspect: aspect, t: 1.0 / 30)
        let next = s.offer(tracking(headYaw: 10 * toRadians), imageAspect: aspect, t: 2.0 / 30)
        XCTAssertEqual(next, .accepted, "once the head stops moving, sampling resumes")
    }

    func testTargetSwitchClearsTheMotionBaseline() {
        var s = CalibrationSession(config: config(2))
        XCTAssertEqual(s.offer(tracking(), imageAspect: aspect, t: 0), .accepted)
        XCTAssertEqual(s.offer(tracking(), imageAspect: aspect, t: 1.0 / 30),
                       .targetComplete(.center))
        XCTAssertEqual(s.target, .up)
        // the user is about to look somewhere else, and that jump would trip the motion proxy if
        // the baseline carried over from the previous target
        XCTAssertEqual(s.offer(tracking(dy: -0.01), imageAspect: aspect, t: 2.0 / 30), .accepted)
    }

    func testRejectionIsRemembered() {
        var s = CalibrationSession(config: config())
        _ = s.offer(tracking(openness: 0), imageAspect: aspect, t: 0)
        XCTAssertEqual(s.lastRejection, .eyesClosed)
        _ = s.offer(tracking(), imageAspect: aspect, t: 1.0 / 30)
        XCTAssertNil(s.lastRejection, "an accepted sample clears the guidance message")
    }

    func testResetDiscardsEverything() {
        var s = CalibrationSession(config: config(3))
        _ = feed(&s, frames: 5)
        s.reset()
        XCTAssertTrue(s.samples.isEmpty)
        XCTAssertEqual(s.target, .center)
        XCTAssertFalse(s.isFinished)
        XCTAssertNil(s.lastRejection)
    }

    func testFinishedSessionStopsCollecting() {
        var s = CalibrationSession(config: config(1))
        _ = runTargets(&s)
        s.skipHeadMotion()
        let t = 10.0
        let before = s.samples.count
        XCTAssertEqual(s.offer(tracking(), imageAspect: aspect, t: t + 1), .finished)
        XCTAssertEqual(s.samples.count, before, "a finished session must not keep appending")
    }

    func testProgressReachesOneOnlyWhenComplete() {
        var s = CalibrationSession(config: config(2))
        XCTAssertEqual(s.overallProgress, 0, accuracy: 1e-12)
        _ = runTargets(&s)
        XCTAssertEqual(s.overallProgress, 1, accuracy: 1e-12)
    }

    // the end-to-end shape: a session driven with a real vertical bias fits that bias back out
    func testSessionFeedsAFitThatRecoversTheBias() throws {
        var s = CalibrationSession(config: config(12, settle: 0))
        var t = 0.0
        // pupil sits above the aperture centre on every target, exactly as measured on hardware
        let bias = -0.0018
        let offsets: [CalibrationTarget: (Double, Double)] = [
            .center: (0, bias), .up: (0, bias - 0.004), .down: (0, bias + 0.004),
            .left: (0.004, bias), .right: (-0.004, bias),
        ]
        while s.phase == .targets, t < 100 {
            let (dx, dy) = offsets[s.target]!
            _ = s.offer(tracking(dx: dx, dy: dy), imageAspect: aspect, t: t)
            t += 1.0 / 30.0
        }
        s.skipHeadMotion()
        let fitted = try s.fit()
        XCTAssertGreaterThan(fitted.pitchOffsetDegrees, 0,
                             "a pupil above the aperture centre must fit a positive upward bias")
        XCTAssertGreaterThan(fitted.verticalSeparationDegrees, CalibrationFit.minimumSeparationDegrees)
        XCTAssertGreaterThan(fitted.horizontalSeparationDegrees, CalibrationFit.minimumSeparationDegrees)

        // and applying it puts steady direct gaze back at zero
        let centred = GazeGeometry.estimate(tracking(dx: 0, dy: bias), imageAspect: aspect)!
        let calibrated = fitted.apply(centred)
        XCTAssertEqual(calibrated.pitch * 180 / .pi, 0, accuracy: 1e-6,
                       "after calibration, looking at the lens must read as looking at the lens")
    }
}

/// a 302 × 189 mm display at 550 mm, the geometry of a 14-inch laptop at a normal viewing distance
private func laptopGeometry(distanceMM: Double = 550) -> CalibrationGeometry {
    CalibrationGeometry(viewingDistanceMM: distanceMM, offsets: [
        .center: TargetOffsetMM(right: 0, up: 0),
        .down: TargetOffsetMM(right: 0, up: -189),
        .left: TargetOffsetMM(right: -151, up: -94.5),
        .right: TargetOffsetMM(right: 151, up: -94.5),
    ])
}

final class CalibrationGeometryTests: XCTestCase {
    func testTrueAnglesFollowTheDisplayLayout() {
        let g = laptopGeometry()
        XCTAssertEqual(g.trueAngles(.center)!.yaw, 0, accuracy: 1e-12)
        XCTAssertEqual(g.trueAngles(.center)!.pitch, 0, accuracy: 1e-12,
                       "the lens is the one target whose true angle is exactly zero")
        XCTAssertGreaterThan(g.trueAngles(.right)!.yaw, 0)
        XCTAssertLessThan(g.trueAngles(.left)!.yaw, 0)
        XCTAssertLessThan(g.trueAngles(.down)!.pitch, 0, "the bottom edge is below the lens")
    }

    func testAnglesShrinkAsTheViewerSitsFurtherBack() {
        let near = laptopGeometry(distanceMM: 400).trueAngles(.right)!.yaw
        let far = laptopGeometry(distanceMM: 900).trueAngles(.right)!.yaw
        XCTAssertGreaterThan(near, far)
    }

    func testTheUpTargetHasNoGeometryAndIsExcluded() {
        XCTAssertNil(laptopGeometry().trueAngles(.up),
                     "above the camera is off the display, so it has no measurable position")
    }

    func testImplausibleDistanceIsRejected() {
        XCTAssertFalse(laptopGeometry(distanceMM: 20).isUsable)
        XCTAssertFalse(laptopGeometry(distanceMM: 5000).isUsable)
        XCTAssertTrue(laptopGeometry().isUsable)
    }
}

final class CalibrationGainFitTests: XCTestCase {
    /// samples from an estimator that over-reads yaw by 2x and under-reads pitch by 2x, which is
    /// the shape of the error measured on hardware
    private func skewedSamples(yawSkew: Double, pitchSkew: Double,
                               geometry: CalibrationGeometry,
                               pitchBias: Double = 4, count: Int = 12) -> [CalibrationSample] {
        var out: [CalibrationSample] = []
        for _ in 0..<count {
            for target in CalibrationTarget.allCases {
                let truth = geometry.trueAngles(target) ?? (yaw: 0, pitch: 12)
                out.append(sample(target,
                                  yaw: truth.yaw * yawSkew,
                                  pitch: truth.pitch * pitchSkew + pitchBias))
            }
        }
        return out
    }

    func testGainIsPinnedWithoutGeometry() throws {
        let fitted = try CalibrationFit.fit(goodSamples())
        XCTAssertEqual(fitted.yawGain, 1.0, accuracy: 1e-12)
        XCTAssertEqual(fitted.gainFitted, false)
        XCTAssertNil(fitted.viewingDistanceMM)
    }

    func testGainInvertsAMeasuredOverRead() throws {
        let g = laptopGeometry()
        let fitted = try CalibrationFit.fit(skewedSamples(yawSkew: 2.0, pitchSkew: 0.5, geometry: g),
                                            geometry: g)
        XCTAssertEqual(fitted.yawGain, 0.5, accuracy: 1e-6,
                       "an estimate twice too large must be scaled back by half")
        XCTAssertEqual(fitted.pitchGain, 2.0, accuracy: 1e-6,
                       "an estimate half the truth must be doubled")
        XCTAssertEqual(fitted.gainFitted, true)
        XCTAssertEqual(fitted.viewingDistanceMM, 550)
    }

    func testFittedGainRecoversTheTrueAngle() throws {
        let g = laptopGeometry()
        let fitted = try CalibrationFit.fit(skewedSamples(yawSkew: 2.0, pitchSkew: 0.5, geometry: g),
                                            geometry: g)
        // the right-edge target, put back through the calibration, must land on its true angle
        let truth = g.trueAngles(.right)!
        let raw = GazeEstimate(yaw: truth.yaw * 2.0 * toRadians,
                               pitch: (truth.pitch * 0.5 + 4) * toRadians, confidence: 1)
        let out = fitted.apply(raw)
        XCTAssertEqual(out.yaw * 180 / .pi, truth.yaw, accuracy: 1e-6)
        XCTAssertEqual(out.pitch * 180 / .pi, truth.pitch, accuracy: 1e-6)
    }

    func testOffsetIsStillFittedFromTheLensAlone() throws {
        let g = laptopGeometry()
        let fitted = try CalibrationFit.fit(skewedSamples(yawSkew: 2.0, pitchSkew: 0.5,
                                                          geometry: g, pitchBias: 5.38),
                                            geometry: g)
        XCTAssertEqual(fitted.pitchOffsetDegrees, 5.38, accuracy: 1e-9)
    }

    // an implausible slope must be dropped on its own axis without taking the rest of a sound
    // calibration down with it — amplifying noise by 20x is worse than not scaling at all
    func testImplausibleGainDegradesToUnityOnThatAxisOnly() throws {
        let g = laptopGeometry(distanceMM: 550)
        let fitted = try CalibrationFit.fit(skewedSamples(yawSkew: 0.10, pitchSkew: 0.5, geometry: g),
                                            geometry: g)
        XCTAssertEqual(fitted.yawGain, 1.0, accuracy: 1e-12, "the refused axis falls back to unity")
        XCTAssertEqual(fitted.yawGainFitted, false)
        XCTAssertEqual(fitted.pitchGain, 2.0, accuracy: 1e-6, "the sound axis is still fitted")
        XCTAssertEqual(fitted.pitchGainFitted, true)
    }

    func testAnImplausibleAxisNeverDiscardsTheNeutralBias() throws {
        let g = laptopGeometry(distanceMM: 550)
        let fitted = try CalibrationFit.fit(skewedSamples(yawSkew: 0.10, pitchSkew: 0.5,
                                                          geometry: g, pitchBias: 5.0),
                                            geometry: g)
        XCTAssertEqual(fitted.pitchOffsetDegrees, 5.0, accuracy: 1e-9,
                       "the bias does not depend on geometry and must survive a refused slope")
    }

    func testUnusableGeometryFallsBackToPinnedGain() throws {
        let broken = CalibrationGeometry(viewingDistanceMM: 10, offsets: [:])
        let fitted = try CalibrationFit.fit(goodSamples(), geometry: broken)
        XCTAssertEqual(fitted.yawGain, 1.0, accuracy: 1e-12,
                       "bad geometry must degrade to offset-only, never to a bogus slope")
        XCTAssertEqual(fitted.gainFitted, false)
    }
}

final class HeadCouplingTests: XCTestCase {
    /// a sweep where true gaze is zero and the estimator leaks a known fraction of head rotation
    private func sweep(yawFromYaw: Double, yawFromPitch: Double,
                       pitchFromYaw: Double, pitchFromPitch: Double,
                       count: Int = 60, span: Double = 20) -> [HeadMotionSample] {
        var out: [HeadMotionSample] = []
        for i in 0..<count {
            // an ellipse so the two axes are not collinear and the split is identifiable
            let t = Double(i) / Double(count) * 2 * .pi
            let hy = span * cos(t)
            let hp = span * sin(t)
            out.append(HeadMotionSample(headYawDegrees: hy, headPitchDegrees: hp,
                                        rawYawDegrees: yawFromYaw * hy + yawFromPitch * hp,
                                        rawPitchDegrees: pitchFromYaw * hy + pitchFromPitch * hp))
        }
        return out
    }

    func testRecoversAKnownCoupling() throws {
        let c = try HeadCouplingFit.fit(sweep(yawFromYaw: 0.30, yawFromPitch: -0.10,
                                              pitchFromYaw: 0.05, pitchFromPitch: 0.40))
        XCTAssertEqual(c.yawFromYaw, 0.30, accuracy: 1e-6)
        XCTAssertEqual(c.yawFromPitch, -0.10, accuracy: 1e-6)
        XCTAssertEqual(c.pitchFromYaw, 0.05, accuracy: 1e-6)
        XCTAssertEqual(c.pitchFromPitch, 0.40, accuracy: 1e-6,
                       "cross terms must be recovered separately, not folded into the same axis")
    }

    func testRecoversNegativeCouplingSoSignIsNeverAssumed() throws {
        let c = try HeadCouplingFit.fit(sweep(yawFromYaw: -0.5, yawFromPitch: 0.2,
                                              pitchFromYaw: -0.15, pitchFromPitch: -0.6))
        XCTAssertEqual(c.yawFromYaw, -0.5, accuracy: 1e-6)
        XCTAssertEqual(c.pitchFromPitch, -0.6, accuracy: 1e-6,
                       "Vision's pitch is positive nodding down while ours is positive looking up, "
                       + "so a negative slope is expected and must survive the fit")
    }

    func testTooFewSamplesFails() {
        XCTAssertThrowsError(try HeadCouplingFit.fit(sweep(yawFromYaw: 0.3, yawFromPitch: 0,
                                                           pitchFromYaw: 0, pitchFromPitch: 0.3,
                                                           count: 10)))
    }

    func testInsufficientHeadMotionFails() {
        let barelyMoving = sweep(yawFromYaw: 0.3, yawFromPitch: 0,
                                 pitchFromYaw: 0, pitchFromPitch: 0.3, span: 2)
        XCTAssertThrowsError(try HeadCouplingFit.fit(barelyMoving)) { error in
            guard case .insufficientHeadMotion = error as? HeadCouplingFit.Failure else {
                return XCTFail("expected insufficient-motion failure, got \(error)")
            }
        }
    }

    // a head that only moved diagonally cannot tell the two axes apart
    func testCollinearHeadMotionIsRefused() {
        var samples: [HeadMotionSample] = []
        for i in 0..<80 {
            let v = Double(i - 40)
            samples.append(HeadMotionSample(headYawDegrees: v, headPitchDegrees: v,
                                            rawYawDegrees: 0.3 * v, rawPitchDegrees: 0.3 * v))
        }
        XCTAssertThrowsError(try HeadCouplingFit.fit(samples),
                             "a single direction of motion leaves the split arbitrary")
    }

    func testImplausibleCouplingIsRefused() {
        XCTAssertThrowsError(try HeadCouplingFit.fit(sweep(yawFromYaw: 9, yawFromPitch: 0,
                                                           pitchFromYaw: 0, pitchFromPitch: 0.3)))
    }

    func testApplyRemovesTheHeadContribution() {
        let coupling = HeadCoupling(yawFromYaw: 0.3, yawFromPitch: -0.1,
                                    pitchFromYaw: 0.05, pitchFromPitch: 0.4)
        let c = GazeCalibration(yawOffsetDegrees: 0, pitchOffsetDegrees: 0,
                                headCoupling: coupling)
        // a reading that is entirely head contamination must calibrate back to zero gaze
        let raw = GazeEstimate(yaw: (0.3 * 20 - 0.1 * 10) * toRadians,
                               pitch: (0.05 * 20 + 0.4 * 10) * toRadians, confidence: 1)
        let out = c.apply(raw, headYawDegrees: 20, headPitchDegrees: 10, headPoseAvailable: true)
        XCTAssertEqual(out.yaw, 0, accuracy: 1e-9)
        XCTAssertEqual(out.pitch, 0, accuracy: 1e-9)
    }

    func testApplyMeasuresHeadContributionRelativeToTheLensPose() {
        let coupling = HeadCoupling(yawFromYaw: 0.3, yawFromPitch: -0.1,
                                    pitchFromYaw: 0.05, pitchFromPitch: 0.4)
        let c = GazeCalibration(yawOffsetDegrees: 2, pitchOffsetDegrees: 5,
                                headCoupling: coupling,
                                headReferenceYawDegrees: 4,
                                headReferencePitchDegrees: 10)
        let raw = GazeEstimate(yaw: 2 * toRadians, pitch: 5 * toRadians, confidence: 1)
        let out = c.apply(raw, headYawDegrees: 4, headPitchDegrees: 10,
                          headPoseAvailable: true)
        XCTAssertEqual(out.yaw, 0, accuracy: 1e-12)
        XCTAssertEqual(out.pitch, 0, accuracy: 1e-12,
                       "the centre target must stay neutral at the pose where it was measured")
    }

    func testCompensationIsSkippedWhenPoseIsUnavailable() {
        let c = GazeCalibration(yawOffsetDegrees: 0, pitchOffsetDegrees: 0,
                                headCoupling: HeadCoupling(yawFromYaw: 0.3, yawFromPitch: 0,
                                                           pitchFromYaw: 0, pitchFromPitch: 0.4))
        let raw = GazeEstimate(yaw: 0.1, pitch: 0.05, confidence: 1)
        let out = c.apply(raw, headYawDegrees: 20, headPitchDegrees: 10, headPoseAvailable: false)
        XCTAssertEqual(out.yaw, 0.1, accuracy: 1e-12,
                       "subtracting a coupling against a fabricated zero pose would invent an error")
        XCTAssertEqual(out.pitch, 0.05, accuracy: 1e-12)
    }

    func testNoCouplingLeavesTheEstimateAlone() {
        let c = GazeCalibration(yawOffsetDegrees: 0, pitchOffsetDegrees: 0)
        let raw = GazeEstimate(yaw: 0.1, pitch: 0.05, confidence: 1)
        let out = c.apply(raw, headYawDegrees: 30, headPitchDegrees: 20, headPoseAvailable: true)
        XCTAssertEqual(out.yaw, 0.1, accuracy: 1e-12,
                       "skipping the sweep must leave behaviour exactly as it was")
    }

    func testStoredCalibrationRoundTripsWithCoupling() throws {
        let original = GazeCalibration(yawOffsetDegrees: 0.7, pitchOffsetDegrees: 5.4,
                                       headCoupling: HeadCoupling(yawFromYaw: 0.3, yawFromPitch: -0.1,
                                                                  pitchFromYaw: 0.05, pitchFromPitch: 0.4))
        let decoded = try JSONDecoder().decode(GazeCalibration.self,
                                               from: JSONEncoder().encode(original))
        XCTAssertEqual(decoded.headCoupling, original.headCoupling)
    }

    // an older file must still decode so its incompatibility is handled as a version decision
    func testOlderFileWithoutCouplingDecodesButIsNotUsable() throws {
        let json = """
        {"version":1,"yawOffsetDegrees":0.68,"pitchOffsetDegrees":5.38,"yawGain":1,"pitchGain":1,
         "verticalSeparationDegrees":9.7,"horizontalSeparationDegrees":61.8,"sampleCount":300,
         "createdAt":"2026-08-03T09:45:35Z"}
        """
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(GazeCalibration.self, from: Data(json.utf8))
        XCTAssertNil(decoded.headCoupling)
        XCTAssertFalse(decoded.isUsable)
    }
}
