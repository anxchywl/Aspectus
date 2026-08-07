import XCTest
@testable import AspectusKit

/// the phase 3 estimator matrix: one test per behaviour the brief calls out, each pinning a
/// convention that a future change could silently invert
private let aspect = 1280.0 / 720.0
private let toRadians = Double.pi / 180

private func eye(dx: Double = 0, dy: Double = 0,
                 openness: Double = 1.0,
                 width: Double = 0.06, height: Double = 0.02) -> EyeObservation {
    let region = NormRect(x: 0.4, y: 0.4, width: width, height: height)
    let c = region.center
    return EyeObservation(region: region,
                          pupilCenter: NormPoint(x: c.x + dx, y: c.y + dy),
                          openness: openness,
                          pupilSource: .visionLandmark, pupilPointCount: 1)
}

private func face(dx: Double = 0, dy: Double = 0,
                  rightDX: Double? = nil, rightDY: Double? = nil,
                  openness: Double = 1.0,
                  headYaw: Double = 0, headPitch: Double = 0, headRoll: Double = 0,
                  poseAvailable: Bool = true,
                  confidence: Double = 1.0) -> TrackingResult {
    TrackingResult(faceBounds: NormRect(x: 0.3, y: 0.2, width: 0.4, height: 0.5),
                   leftEye: eye(dx: dx, dy: dy, openness: openness),
                   rightEye: eye(dx: rightDX ?? dx, dy: rightDY ?? dy, openness: openness),
                   headPose: HeadPose(yaw: headYaw, pitch: headPitch, roll: headRoll),
                   confidence: confidence,
                   headPoseAvailable: poseAvailable)
}

final class GazeSignConventionTests: XCTestCase {
    func testNeutralGazeReadsAsZero() {
        let e = GazeGeometry.estimate(face(), imageAspect: aspect)!
        XCTAssertEqual(e.yaw, 0, accuracy: 1e-12)
        XCTAssertEqual(e.pitch, 0, accuracy: 1e-12)
    }

    // image x grows right and an unmirrored camera shows the subject's right on the image left,
    // so a pupil sitting left in the image means the subject is looking to their own right
    func testLookingRightIsPositiveYaw() {
        XCTAssertGreaterThan(GazeGeometry.estimate(face(dx: -0.006), imageAspect: aspect)!.yaw, 0)
    }

    func testLookingLeftIsNegativeYaw() {
        XCTAssertLessThan(GazeGeometry.estimate(face(dx: 0.006), imageAspect: aspect)!.yaw, 0)
    }

    // image y grows downward, so a pupil above the aperture centre is a gaze above the lens
    func testLookingAboveTheCameraIsPositivePitch() {
        XCTAssertGreaterThan(GazeGeometry.estimate(face(dy: -0.003), imageAspect: aspect)!.pitch, 0)
    }

    func testLookingBelowTheCameraIsNegativePitch() {
        XCTAssertLessThan(GazeGeometry.estimate(face(dy: 0.003), imageAspect: aspect)!.pitch, 0)
    }

    func testVerticalEstimateUsesTheEyeCornerLine() {
        let region = NormRect(x: 0.4, y: 0.4, width: 0.06, height: 0.02)
        let pupil = NormPoint(x: region.center.x, y: region.center.y - 0.003)
        let first = EyeObservation(region: region, pupilCenter: pupil, openness: 1,
                                   cornerMidpointY: region.center.y)
        let shiftedAperture = EyeObservation(
            region: NormRect(x: region.x, y: region.y - 0.002,
                             width: region.width, height: region.height),
            pupilCenter: pupil, openness: 1, cornerMidpointY: region.center.y)

        let firstAngles = GazeGeometry.eyeAngles(first, imageAspect: aspect)!
        let shiftedAngles = GazeGeometry.eyeAngles(shiftedAperture, imageAspect: aspect)!
        XCTAssertEqual(firstAngles.yaw, shiftedAngles.yaw, accuracy: 1e-12)
        XCTAssertEqual(firstAngles.pitch, shiftedAngles.pitch, accuracy: 1e-12)
    }

    func testCornerRelativeEstimateIgnoresWholeEyeTranslation() {
        let first = EyeObservation(
            region: NormRect(x: 0.4, y: 0.4, width: 0.06, height: 0.02),
            pupilCenter: NormPoint(x: 0.427, y: 0.407), openness: 1,
            cornerMidpointY: 0.410)
        let translated = EyeObservation(
            region: NormRect(x: 0.5, y: 0.6, width: 0.06, height: 0.02),
            pupilCenter: NormPoint(x: 0.527, y: 0.607), openness: 1,
            cornerMidpointY: 0.610)

        let firstAngles = GazeGeometry.eyeAngles(first, imageAspect: aspect)!
        let translatedAngles = GazeGeometry.eyeAngles(translated, imageAspect: aspect)!
        XCTAssertEqual(firstAngles.yaw, translatedAngles.yaw, accuracy: 1e-12)
        XCTAssertEqual(firstAngles.pitch, translatedAngles.pitch, accuracy: 1e-12)
    }

    // the whole point of correction: removing the observed gaze must send the iris back to centre
    func testCorrectionOpposesTheObservedGazeOnBothAxes() {
        for (dx, dy) in [(0.006, 0.003), (-0.006, -0.003), (0.004, -0.002), (-0.004, 0.002)] {
            let tracking = face(dx: dx, dy: dy)
            let gaze = GazeGeometry.estimate(tracking, imageAspect: aspect)!
            let request = CorrectionRequest(yawOffset: -gaze.yaw, pitchOffset: -gaze.pitch,
                                            strength: 1)
            let w = GazeGeometry.warps(tracking, imageAspect: aspect, request: request,
                                       tuning: EyeWarpTuning(recenterGain: 1.0))!
            XCTAssertEqual(w.left.displacement.x, dx, accuracy: 1e-9)
            XCTAssertEqual(w.left.displacement.y, dy, accuracy: 1e-9)
        }
    }
}

final class GazeHeadPoseTests: XCTestCase {
    // the eyes are centred in their sockets, so the estimate is zero no matter where the head points
    func testHeadRotationWithoutEyeRotationYieldsNoGaze() {
        let e = GazeGeometry.estimate(face(headYaw: 10 * toRadians, headPitch: 5 * toRadians),
                                      imageAspect: aspect)!
        XCTAssertEqual(e.yaw, 0, accuracy: 1e-12)
        XCTAssertEqual(e.pitch, 0, accuracy: 1e-12)
    }

    func testEyeRotationWithoutHeadRotationIsMeasured() {
        let e = GazeGeometry.estimate(face(dx: -0.006), imageAspect: aspect)!
        XCTAssertGreaterThan(abs(e.yaw), 0)
    }

    func testExcessiveHeadPoseSuppressesTheEstimate() {
        XCTAssertNil(GazeGeometry.estimate(face(dx: -0.006, headYaw: 40 * toRadians),
                                           imageAspect: aspect))
        XCTAssertTrue(GazeGeometry.exceedsHeadPoseLimit(face(headYaw: 40 * toRadians)))
    }

    func testHeadPoseLimitAppliesToPitchAsWellAsYaw() {
        XCTAssertTrue(GazeGeometry.exceedsHeadPoseLimit(face(headPitch: 40 * toRadians)))
    }

    func testHeadPoseLimitIsSignIndependent() {
        XCTAssertTrue(GazeGeometry.exceedsHeadPoseLimit(face(headYaw: 40 * toRadians)))
        XCTAssertTrue(GazeGeometry.exceedsHeadPoseLimit(face(headYaw: -40 * toRadians)))
    }

    // measured on hardware: a landmarks-only request reports no pose at all, and zero would then
    // masquerade as a perfectly square head
    func testMissingPoseDoesNotFabricateASquareHead() {
        let noPose = face(headYaw: 0, headPitch: 0, poseAvailable: false)
        XCTAssertFalse(GazeGeometry.exceedsHeadPoseLimit(noPose),
                       "the limit cannot be enforced without a pose, and must not pretend it can")
        XCTAssertNotNil(GazeGeometry.estimate(noPose, imageAspect: aspect),
                        "refusing every frame when pose is unavailable would disable the product")
    }

    func testPoseAvailabilitySurvivesSmoothing() {
        var s = TemporalStabilizer()
        let filtered = s.stabilize(face(poseAvailable: false), t: 0)
        XCTAssertFalse(filtered.headPoseAvailable)
    }
}

final class GazePerEyeTests: XCTestCase {
    func testPerEyeAnglesAreReportedSeparately() {
        let tracking = face(dx: -0.006, rightDX: 0.006)
        let l = GazeGeometry.eyeAngles(tracking.leftEye, imageAspect: aspect)!
        let r = GazeGeometry.eyeAngles(tracking.rightEye, imageAspect: aspect)!
        XCTAssertGreaterThan(l.yaw, 0)
        XCTAssertLessThan(r.yaw, 0)
    }

    func testCombinedEstimateAveragesTheTwoEyes() {
        let tracking = face(dx: -0.006, rightDX: 0.002)
        let l = GazeGeometry.eyeAngles(tracking.leftEye, imageAspect: aspect)!
        let r = GazeGeometry.eyeAngles(tracking.rightEye, imageAspect: aspect)!
        let combined = GazeGeometry.estimate(tracking, imageAspect: aspect)!
        XCTAssertEqual(combined.yaw, (l.yaw + r.yaw) / 2, accuracy: 1e-12)
    }

    func testDisagreeingEyesStillProduceAnEstimateButCorrectionUsesBoth() {
        let tracking = face(dx: -0.010, rightDX: 0.010)
        XCTAssertNotNil(GazeGeometry.estimate(tracking, imageAspect: aspect))
        XCTAssertNotNil(GazeGeometry.warps(tracking, imageAspect: aspect,
                                           request: CorrectionRequest(yawOffset: 0.1, pitchOffset: 0,
                                                                      strength: 1)))
    }

    func testOneUsableEyeHalvesConfidence() {
        let tracking = face(openness: 1.0)
        let oneShut = TrackingResult(faceBounds: tracking.faceBounds,
                                     leftEye: tracking.leftEye,
                                     rightEye: eye(openness: 0),
                                     headPose: tracking.headPose, confidence: 1.0)
        XCTAssertEqual(GazeGeometry.estimate(oneShut, imageAspect: aspect)!.confidence, 0.5,
                       accuracy: 1e-12)
    }
}

final class GazeRobustnessTests: XCTestCase {
    func testBlinkSuppressesEstimateAndCorrection() {
        let blinking = face(dx: -0.006, openness: 0.0)
        XCTAssertNil(GazeGeometry.estimate(blinking, imageAspect: aspect))
        XCTAssertNil(GazeGeometry.warps(blinking, imageAspect: aspect,
                                        request: CorrectionRequest(yawOffset: 0.1, pitchOffset: 0.1,
                                                                   strength: 1)))
    }

    func testInvalidLandmarksAreRejected() {
        let degenerate = EyeObservation(region: NormRect(x: 0.4, y: 0.4, width: 0, height: 0),
                                        pupilCenter: NormPoint(x: 0.4, y: 0.4), openness: 1)
        let tracking = TrackingResult(faceBounds: NormRect(x: 0, y: 0, width: 1, height: 1),
                                      leftEye: degenerate, rightEye: degenerate,
                                      headPose: HeadPose(yaw: 0, pitch: 0, roll: 0), confidence: 1)
        XCTAssertNil(GazeGeometry.estimate(tracking, imageAspect: aspect))
    }

    func testNonPositiveAspectIsRejected() {
        XCTAssertNil(GazeGeometry.estimate(face(dx: 0.004), imageAspect: 0))
        XCTAssertNil(GazeGeometry.eyeAngles(eye(dx: 0.004), imageAspect: 0))
    }

    // the estimate is normalized against the frame, so the same look must read the same whether
    // the face is near or far
    func testEstimateIsInvariantToFaceDistance() {
        // a face twice as far has an eye half as wide and a pupil offset half as large
        let near = GazeGeometry.estimate(face(dx: -0.008), imageAspect: aspect)!
        let farTracking = TrackingResult(
            faceBounds: NormRect(x: 0.4, y: 0.35, width: 0.2, height: 0.25),
            leftEye: eye(dx: -0.004, width: 0.03, height: 0.01),
            rightEye: eye(dx: -0.004, width: 0.03, height: 0.01),
            headPose: HeadPose(yaw: 0, pitch: 0, roll: 0), confidence: 1)
        let far = GazeGeometry.estimate(farTracking, imageAspect: aspect)!
        XCTAssertEqual(near.yaw, far.yaw, accuracy: 1e-9,
                       "distance changes the pixels, not the angle the eye is turned through")
    }

    // mirroring is a preview-only choice; the correction path never sees it
    func testMirroringIsNotPartOfTheEstimateOrTheWarp() {
        let tracking = face(dx: -0.006)
        let straight = GazeGeometry.estimate(tracking, imageAspect: aspect)!
        // a mirrored image would place the pupil on the other side of the aperture
        let mirroredTracking = face(dx: 0.006)
        let mirrored = GazeGeometry.estimate(mirroredTracking, imageAspect: aspect)!
        XCTAssertEqual(straight.yaw, -mirrored.yaw, accuracy: 1e-12,
                       "if mirroring ever reached the correction path it would invert the redirect")
    }

    func testWarpNeverTouchesAnythingOutsideTheAperture() {
        for (width, height) in [(0.06, 0.02), (0.10, 0.015), (0.03, 0.012)] {
            let tracking = TrackingResult(
                faceBounds: NormRect(x: 0.3, y: 0.2, width: 0.4, height: 0.5),
                leftEye: eye(width: width, height: height),
                rightEye: eye(width: width, height: height),
                headPose: HeadPose(yaw: 0, pitch: 0, roll: 0), confidence: 1)
            let w = GazeGeometry.warps(tracking, imageAspect: aspect,
                                       request: CorrectionRequest(yawOffset: 0.2, pitchOffset: 0.2,
                                                                  strength: 1))!
            XCTAssertLessThanOrEqual(w.left.radius.x, width / 2)
            XCTAssertLessThanOrEqual(w.left.radius.y, height / 2)
        }
    }
}

final class GazeStalenessTests: XCTestCase {
    // correction runs on the previous frame's landmarks, so a warp built from stale geometry must
    // still be anchored on that geometry rather than drifting toward the current frame
    func testWarpIsAnchoredOnTheTrackingItWasGiven() {
        let stale = face(dx: -0.006)
        let w = GazeGeometry.warps(stale, imageAspect: aspect,
                                   request: CorrectionRequest(yawOffset: 0.1, pitchOffset: 0,
                                                              strength: 1))!
        XCTAssertEqual(w.left.center.x, stale.leftEye.pupilCenter.x, accuracy: 1e-12)
        XCTAssertEqual(w.left.center.y, stale.leftEye.pupilCenter.y, accuracy: 1e-12)
    }

    func testTimestampDiscontinuityResetsSmoothingSoStaleGeometryIsNotBlended() {
        var s = TemporalStabilizer(config: .init(maxFrameGap: 0.25))
        for i in 0..<30 { _ = s.stabilize(face(dx: 0.0), t: Double(i) / 30.0) }
        XCTAssertTrue(s.isDiscontinuous(at: 10.0))
        let after = s.stabilize(face(dx: -0.006), t: 10.0)
        let expected = eye(dx: -0.006).pupilCenter.x
        XCTAssertEqual(after.leftEye.pupilCenter.x, expected, accuracy: 1e-12)
    }

    func testLostTrackingClearsHistory() {
        var s = TemporalStabilizer()
        for i in 0..<30 { _ = s.stabilize(face(dx: 0.006), t: Double(i) / 30.0) }
        s.reset()
        let reacquired = s.stabilize(face(dx: -0.006), t: 2.0)
        XCTAssertEqual(reacquired.leftEye.pupilCenter.x, eye(dx: -0.006).pupilCenter.x,
                       accuracy: 1e-12)
    }
}

final class CameraToScreenOffsetTests: XCTestCase {
    // the screen-to-lens angle is not visible in the image, so it is added on top of the measured
    // gaze rather than being something the estimator could ever recover
    func testRedirectAddsToTheMeasuredGaze() {
        let redirect = 12 * toRadians
        let tracking = face(dy: 0.002)
        let gaze = GazeGeometry.estimate(tracking, imageAspect: aspect)!
        let request = CorrectionRequest(yawOffset: -gaze.yaw,
                                        pitchOffset: -gaze.pitch + redirect, strength: 1)
        XCTAssertGreaterThan(request.pitchOffset, redirect,
                             "looking below the lens must ask for more lift than the offset alone")
    }

    func testRedirectAloneStillProducesVisibleTravel() {
        let w = GazeGeometry.warps(face(), imageAspect: aspect,
                                   request: CorrectionRequest(yawOffset: 0,
                                                              pitchOffset: 12 * toRadians,
                                                              strength: 1))!
        XCTAssertGreaterThan(GazeGeometry.displacementPixels(w.left, width: 1280, height: 720), 3.0)
    }
}
