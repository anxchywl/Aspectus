import XCTest
@testable import AspectusKit

private let aspect = 1280.0 / 720.0

private func eye(pupilDX: Double = 0, pupilDY: Double = 0,
                 openness: Double = 1.0,
                 width: Double = 0.06, height: Double = 0.02) -> EyeObservation {
    let region = NormRect(x: 0.4, y: 0.4, width: width, height: height)
    let c = region.center
    return EyeObservation(region: region,
                          pupilCenter: NormPoint(x: c.x + pupilDX, y: c.y + pupilDY),
                          openness: openness)
}

private func req(yaw: Double = 0, pitch: Double = 0, strength: Double = 1.0) -> CorrectionRequest {
    CorrectionRequest(yawOffset: yaw, pitchOffset: pitch, strength: strength)
}

private func tracking(left: EyeObservation, right: EyeObservation,
                      yaw: Double = 0, pitch: Double = 0,
                      confidence: Double = 0.9) -> TrackingResult {
    TrackingResult(faceBounds: NormRect(x: 0.3, y: 0.2, width: 0.4, height: 0.5),
                   leftEye: left, rightEye: right,
                   headPose: HeadPose(yaw: yaw, pitch: pitch, roll: 0),
                   confidence: confidence)
}

final class GazeGeometryWarpTests: XCTestCase {
    func testZeroRequestProducesNoDisplacement() {
        let t = tracking(left: eye(pupilDX: 0.01), right: eye(pupilDX: 0.01))
        let w = GazeGeometry.warps(t, imageAspect: aspect, request: req())
        XCTAssertNotNil(w)
        XCTAssertEqual(w!.left.displacement.x, 0, accuracy: 1e-12,
                       "asking for no redirection must move nothing")
        XCTAssertEqual(w!.left.displacement.y, 0, accuracy: 1e-12)
    }

    func testDisplacementScalesWithRequestAndStrength() {
        let t = tracking(left: eye(), right: eye())
        let full = GazeGeometry.warps(t, imageAspect: aspect, request: req(yaw: 0.1, pitch: 0.1))!
        let half = GazeGeometry.warps(t, imageAspect: aspect,
                                      request: req(yaw: 0.1, pitch: 0.1, strength: 0.5))!
        XCTAssertGreaterThan(full.left.displacement.x, 0)
        XCTAssertGreaterThan(full.left.displacement.y, 0)
        XCTAssertEqual(half.left.displacement.x, full.left.displacement.x / 2, accuracy: 1e-12,
                       "strength must scale the warp linearly so the blend cannot pop")
        XCTAssertEqual(half.left.displacement.y, full.left.displacement.y / 2, accuracy: 1e-12)
    }

    // the one test that pins the sign convention end to end: an observed gaze, negated into a
    // request, must produce a displacement that lands the iris back on the aperture centre
    func testRemovingObservedGazeRecentersTheIris() {
        let tuning = EyeWarpTuning(recenterGain: 1.0)
        for (dx, dy) in [(0.01, 0.004), (-0.01, -0.004), (0.008, -0.003)] {
            let t = tracking(left: eye(pupilDX: dx, pupilDY: dy),
                             right: eye(pupilDX: dx, pupilDY: dy))
            let gaze = GazeGeometry.estimate(t, imageAspect: aspect, tuning: tuning)!
            let request = CorrectionRequest(yawOffset: -gaze.yaw, pitchOffset: -gaze.pitch,
                                            strength: 1.0)
            let w = GazeGeometry.warps(t, imageAspect: aspect, request: request, tuning: tuning)!
            XCTAssertEqual(w.left.displacement.x, dx, accuracy: 1e-9,
                           "x displacement must cancel the observed pupil offset exactly")
            XCTAssertEqual(w.left.displacement.y, dy, accuracy: 1e-9,
                           "y displacement must cancel the observed pupil offset exactly")
        }
    }

    // the failure that made the first prototype invisible: a real screen-to-lens angle has to
    // move the iris by pixels a human can actually see
    func testRedirectAngleProducesVisibleTravel() {
        let t = tracking(left: eye(width: 0.06), right: eye(width: 0.06))
        let request = req(pitch: 12 * .pi / 180)
        let w = GazeGeometry.warps(t, imageAspect: aspect, request: request)!
        let px = GazeGeometry.displacementPixels(w.left, width: 1280, height: 720)
        XCTAssertGreaterThan(px, 3.0,
                             "a 12 degree redirect must move the iris more than a few pixels")
    }

    func testUpwardRedirectLiftsTheIris() {
        let t = tracking(left: eye(), right: eye())
        let w = GazeGeometry.warps(t, imageAspect: aspect, request: req(pitch: 0.2))!
        // image y grows downward and content moves by -displacement, so a positive y sampling
        // offset lifts the iris toward a lens sitting above the screen
        XCTAssertGreaterThan(w.left.displacement.y, 0)
    }

    func testZeroStrengthYieldsNoWarp() {
        let t = tracking(left: eye(pupilDX: 0.01), right: eye(pupilDX: 0.01))
        XCTAssertNil(GazeGeometry.warps(t, imageAspect: aspect, request: req(yaw: 0.1, strength: 0)),
                     "zero strength must fall through to the original frame, not a no-op warp")
    }

    func testInfluenceIsApertureShapedNotCircular() {
        let width = 0.06, height = 0.02
        let t = tracking(left: eye(width: width, height: height),
                         right: eye(width: width, height: height))
        let w = GazeGeometry.warps(t, imageAspect: aspect, request: req(yaw: 0.1, pitch: 0.1))!
        XCTAssertEqual(w.left.radius.y / w.left.radius.x, height / width, accuracy: 1e-12,
                       "the influence ellipse must follow the eye's own proportions")
    }

    func testInfluenceNeverReachesTheEyelids() {
        // an eye is far wider than tall, so a pixel-space circle would drag the lids vertically
        for (width, height) in [(0.06, 0.02), (0.10, 0.015), (0.03, 0.012)] {
            let t = tracking(left: eye(width: width, height: height),
                             right: eye(width: width, height: height))
            let w = GazeGeometry.warps(t, imageAspect: aspect, request: req(yaw: 0.1, pitch: 0.1))!
            XCTAssertLessThanOrEqual(w.left.radius.x, width / 2,
                                     "falloff must reach zero inside the aperture horizontally")
            XCTAssertLessThanOrEqual(w.left.radius.y, height / 2,
                                     "falloff must reach zero inside the aperture vertically, or "
                                     + "lids, lashes and brows would be displaced")
        }
    }

    func testClosedEyeIsNotWarped() {
        let closed = eye(pupilDX: 0.01, openness: 0.1)
        XCTAssertNil(GazeGeometry.warps(tracking(left: closed, right: eye(pupilDX: 0.01)), imageAspect: aspect, request: req(pitch: 0.1)),
                     "a blinking eye must suppress correction on both eyes")
    }

    func testOneClosedEyeSuppressesBothEyes() {
        let t = tracking(left: eye(pupilDX: 0.01), right: eye(pupilDX: 0.01, openness: 0.0))
        XCTAssertNil(GazeGeometry.warps(t, imageAspect: aspect, request: req(yaw: 0.1, pitch: 0.1)),
                     "correcting a single eye would look worse than not correcting at all")
    }

    func testDegenerateEyeRegionIsRejected() {
        let degenerate = EyeObservation(region: NormRect(x: 0.4, y: 0.4, width: 0, height: 0),
                                        pupilCenter: NormPoint(x: 0.4, y: 0.4), openness: 1)
        XCTAssertNil(GazeGeometry.warps(tracking(left: degenerate, right: eye()), imageAspect: aspect, request: req(pitch: 0.1)))
    }
}

final class GazeGeometryEstimateTests: XCTestCase {
    func testCenteredPupilsMeanLookingAtLens() {
        let e = GazeGeometry.estimate(tracking(left: eye(), right: eye()), imageAspect: aspect)
        XCTAssertNotNil(e)
        XCTAssertEqual(e!.magnitudeDegrees, 0, accuracy: 1e-9)
    }

    func testAngleGrowsMonotonicallyWithPupilOffset() {
        var previous = -1.0
        for dx in stride(from: 0.0, through: 0.02, by: 0.004) {
            let t = tracking(left: eye(pupilDX: dx), right: eye(pupilDX: dx))
            let m = GazeGeometry.estimate(t, imageAspect: aspect)!.magnitudeDegrees
            XCTAssertGreaterThan(m, previous)
            previous = m
        }
    }

    func testExcessiveHeadPoseRejectsTheEstimate() {
        let t = tracking(left: eye(), right: eye(), yaw: 40 * .pi / 180)
        XCTAssertNil(GazeGeometry.estimate(t, imageAspect: aspect),
                     "past the head-pose limit a centred pupil no longer implies looking at the lens")
    }

    func testHeadPoseLimitIsSignIndependent() {
        let limit = 40 * Double.pi / 180
        XCTAssertNil(GazeGeometry.estimate(tracking(left: eye(), right: eye(), yaw: limit),
                                           imageAspect: aspect))
        XCTAssertNil(GazeGeometry.estimate(tracking(left: eye(), right: eye(), yaw: -limit),
                                           imageAspect: aspect),
                     "gating on magnitude must not depend on Vision's handedness")
    }

    func testClosedEyesYieldNoEstimate() {
        let shut = eye(openness: 0.0)
        XCTAssertNil(GazeGeometry.estimate(tracking(left: shut, right: shut), imageAspect: aspect))
    }

    func testSingleUsableEyeLowersConfidence() {
        let t = tracking(left: eye(), right: eye(openness: 0.0), confidence: 1.0)
        let e = GazeGeometry.estimate(t, imageAspect: aspect)
        XCTAssertNotNil(e)
        XCTAssertEqual(e!.confidence, 0.5, accuracy: 1e-12,
                       "one eye is a weaker signal and must gate down")
    }

    func testEstimateIsAspectCorrected() {
        let t = tracking(left: eye(pupilDY: 0.004), right: eye(pupilDY: 0.004))
        let wide = GazeGeometry.estimate(t, imageAspect: 16.0 / 9.0)!
        let square = GazeGeometry.estimate(t, imageAspect: 1.0)!
        XCTAssertLessThan(abs(wide.pitch), abs(square.pitch),
                          "the same normalized dy is fewer pixels on a wide frame")
    }
}
