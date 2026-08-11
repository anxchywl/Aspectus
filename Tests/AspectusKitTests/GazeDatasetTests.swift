import XCTest
@testable import AspectusKit

final class GazeDatasetPlanTests: XCTestCase {
    func testSchemaFourManifestContractIsStableAndUnambiguous() {
        XCTAssertEqual(GazeDatasetSchema4.version, 4)
        XCTAssertEqual(GazeDatasetSchema4.manifestColumns.count, 41)
        XCTAssertEqual(Set(GazeDatasetSchema4.manifestColumns).count, 41)
        XCTAssertEqual(GazeDatasetSchema4.manifestColumns.first, "schema_version")
        XCTAssertEqual(GazeDatasetSchema4.manifestColumns.last, "crop_clipped_fraction_r")
    }

    func testPlanCoversEveryGridPointAndLensForEveryPose() {
        let targets = GazeDatasetPlan.targets(for: .training)
        XCTAssertEqual(targets.count, 5 * 27)

        for pose in GazePosePrompt.allCases {
            let block = targets.filter { $0.pose == pose }
            XCTAssertEqual(block.filter { $0.kind == .screen }.count, 25)
            XCTAssertEqual(block.filter { $0.kind == .lens }.count, 2)
            XCTAssertEqual(Set(block.filter { $0.kind == .screen }.map {
                "\($0.xFraction),\($0.yFraction)"
            }).count, 25)
        }
    }

    func testTrainingAndValidationUseDifferentOrdersWithTheSameTargets() {
        let training = GazeDatasetPlan.targets(for: .training)
        let validation = GazeDatasetPlan.targets(for: .validation)
        XCTAssertNotEqual(training.map { [$0.xFraction, $0.yFraction] },
                          validation.map { [$0.xFraction, $0.yFraction] })
        XCTAssertEqual(Set(training.map { "\($0.pose.rawValue),\($0.kind.rawValue),\($0.xFraction),\($0.yFraction)" }),
                       Set(validation.map { "\($0.pose.rawValue),\($0.kind.rawValue),\($0.xFraction),\($0.yFraction)" }))
    }

    func testTargetOrderMatchesTheOfflineIntegrityFixtures() {
        let training = GazeDatasetPlan.targets(for: .training)
        let validation = GazeDatasetPlan.targets(for: .validation)

        XCTAssertEqual(training[0], .init(id: 0, kind: .lens, xFraction: 0.5,
                                          yFraction: 0, pose: .neutral))
        XCTAssertEqual(training[1], .init(id: 1, kind: .screen, xFraction: 0.5,
                                          yFraction: 0.5, pose: .neutral))
        XCTAssertEqual(validation[1], .init(id: 1, kind: .screen, xFraction: 0.7,
                                            yFraction: 0.5, pose: .neutral))
        XCTAssertEqual(training[28], .init(id: 28, kind: .screen, xFraction: 0.1,
                                           yFraction: 0.9, pose: .turnLeft))
    }

    func testGeometryUsesThePhysicalLensAsItsOrigin() throws {
        let geometry = GazeDatasetGeometry(displayWidthMM: 300, displayHeightMM: 200,
                                           viewingDistanceMM: 500)
        let lens = GazeDatasetTarget(id: 0, kind: .lens, xFraction: 0.5,
                                     yFraction: 0, pose: .neutral)
        let bottomRight = GazeDatasetTarget(id: 1, kind: .screen, xFraction: 1,
                                            yFraction: 1, pose: .neutral)
        let lensAngles = try XCTUnwrap(geometry.angles(for: lens))
        XCTAssertEqual(lensAngles.yaw, 0, accuracy: 1e-12)
        XCTAssertEqual(lensAngles.pitch, 0, accuracy: 1e-12)

        let screenAngles = try XCTUnwrap(geometry.angles(for: bottomRight))
        XCTAssertEqual(screenAngles.yaw, atan2(150.0, 500.0) * 180 / .pi,
                       accuracy: 1e-12)
        XCTAssertEqual(screenAngles.pitch, atan2(-200.0, 500.0) * 180 / .pi,
                       accuracy: 1e-12)
    }

    func testUnusableGeometryProducesNoLabel() {
        let geometry = GazeDatasetGeometry(displayWidthMM: 0, displayHeightMM: 200,
                                           viewingDistanceMM: 500)
        XCTAssertNil(geometry.angles(for: GazeDatasetPlan.targets(for: .training)[1]))
    }
}

final class GazeDatasetAcceptanceTests: XCTestCase {
    private func tracking(confidence: Double = 0.9, openness: Double = 1,
                          yawDegrees: Double = 0, poseAvailable: Bool = true,
                          pitchDegrees: Double = 0,
                          rollDegrees: Double = 0,
                          eyeX: Double = 0.2, eyeWidth: Double = 0.08) -> TrackingResult {
        let eye = EyeObservation(region: .init(x: eyeX, y: 0.2,
                                               width: eyeWidth, height: 0.03),
                                 pupilCenter: .init(x: 0.24, y: 0.215),
                                 openness: openness)
        return TrackingResult(faceBounds: .init(x: 0.1, y: 0.1, width: 0.8, height: 0.8),
                              leftEye: eye, rightEye: eye,
                              headPose: .init(yaw: yawDegrees * .pi / 180,
                                              pitch: pitchDegrees * .pi / 180,
                                              roll: rollDegrees * .pi / 180),
                              confidence: confidence,
                              headPoseAvailable: poseAvailable)
    }

    func testAcceptsAStableOpenFaceInsideThePoseLimit() {
        XCTAssertNil(GazeDatasetAcceptance.rejection(tracking()))
    }

    func testNamesEveryRejectedInput() {
        XCTAssertEqual(GazeDatasetAcceptance.rejection(nil), .noTracking)
        XCTAssertEqual(GazeDatasetAcceptance.rejection(tracking(confidence: 0.4)), .lowConfidence)
        XCTAssertEqual(GazeDatasetAcceptance.rejection(tracking(openness: 0.2)), .eyesClosed)
        XCTAssertEqual(GazeDatasetAcceptance.rejection(tracking(poseAvailable: false)),
                       .headPoseUnavailable)
        XCTAssertEqual(GazeDatasetAcceptance.rejection(tracking(yawDegrees: 30)), .headPose)
        XCTAssertEqual(GazeDatasetAcceptance.rejection(tracking(rollDegrees: 21)), .headPose)
        XCTAssertEqual(GazeDatasetAcceptance.rejection(tracking(pitchDegrees: .nan)), .headPose)
        XCTAssertEqual(GazeDatasetAcceptance.rejection(tracking(eyeWidth: 0)), .degenerateEyes)
        XCTAssertEqual(GazeDatasetAcceptance.rejection(tracking(eyeX: 1.1)), .degenerateEyes)
        XCTAssertEqual(GazeDatasetAcceptance.rejection(tracking(eyeX: .nan)), .degenerateEyes)
    }

    func testSchemaFourRequiresObservableEyeAxes() {
        XCTAssertEqual(
            GazeDatasetAcceptance.rejection(tracking(), imageWidth: 1280, imageHeight: 720),
            .eyeAlignment)
    }

    func testSchemaFourAcceptsFinitePixelSpaceEyeAxes() {
        var value = tracking()
        value.leftEye.contourPointCount = 8
        value.leftEye.imageAxisStart = .init(x: 0.20, y: 0.20)
        value.leftEye.imageAxisEnd = .init(x: 0.28, y: 0.20)
        value.rightEye = value.leftEye

        XCTAssertNil(GazeDatasetAcceptance.rejection(
            value, imageWidth: 1280, imageHeight: 720))
    }
}

final class GazeDatasetCanonicalAlignmentTests: XCTestCase {
    private func eye(centerX: Double, centerY: Double, length: Double,
                     angleDegrees: Double, width: Double = 1280,
                     height: Double = 720) -> EyeObservation {
        let angle = angleDegrees * .pi / 180
        let dx = cos(angle) * length / 2
        let dy = sin(angle) * length / 2
        return EyeObservation(
            region: .init(x: (centerX - length / 2) / width,
                          y: (centerY - length / 4) / height,
                          width: length / width, height: length / 2 / height),
            pupilCenter: .init(x: centerX / width, y: centerY / height),
            openness: 1,
            pupilSource: .visionLandmark,
            pupilPointCount: 1,
            contourPointCount: 8,
            imageAxisStart: .init(x: (centerX - dx) / width,
                                  y: (centerY - dy) / height),
            imageAxisEnd: .init(x: (centerX + dx) / width,
                                y: (centerY + dy) / height))
    }

    func testFarthestContourPairDefinesTheImageOrderedAxis() throws {
        let points = [
            NormPoint(x: 0.2, y: 0.8),
            NormPoint(x: 0.8, y: 0.8),
            NormPoint(x: 0.5, y: 0.2),
        ]

        let axis = try XCTUnwrap(EyeObservation.imageAxis(
            of: points, imageWidth: 1280, imageHeight: 720))

        XCTAssertEqual(axis.start, points[0])
        XCTAssertEqual(axis.end, points[1])
    }

    func testSerializedCropContractUsesTheCanonicalGeometryConstants() {
        let contract = GazeDatasetCropContract.canonicalPairedEyesV1

        XCTAssertEqual(contract.scale,
                       "\(GazeDatasetCanonicalAlignment.cropScale)x-maximum-eye-axis-length-pixels")
        XCTAssertEqual(contract.outputWidth, GazeDatasetCanonicalAlignment.outputWidth)
        XCTAssertEqual(contract.outputHeight, GazeDatasetCanonicalAlignment.outputHeight)
        XCTAssertEqual(contract.sampling, "core-image-affine-hq-downsample-edge-clamp")
    }

    func testSchemaFourFreezesTheVisionHeadPitchConvention() {
        let contract = GazeDatasetHeadPoseContract.visionRevision3DegreesV1

        XCTAssertEqual(contract.source,
                       "Vision.VNFaceObservation.face-rectangles-revision-3")
        XCTAssertEqual(contract.order, "yaw-pitch-roll")
        XCTAssertEqual(contract.pitchPositive, "head-down")
    }

    func testBothEyesUseOnePixelSpaceRotationAndScale() throws {
        let alignment = try XCTUnwrap(GazeDatasetCanonicalAlignment(
            left: eye(centerX: 480, centerY: 300, length: 40, angleDegrees: 10),
            right: eye(centerX: 800, centerY: 302, length: 50, angleDegrees: 14),
            imageWidth: 1280, imageHeight: 720))

        XCTAssertEqual(alignment.rotationRadians * 180 / .pi, 12, accuracy: 1e-9)
        XCTAssertEqual(alignment.disagreementDegrees, 4, accuracy: 1e-9)
        XCTAssertEqual(alignment.cropSidePixels, 90, accuracy: 1e-9)
        XCTAssertEqual(alignment.left.centerX, 480, accuracy: 1e-9)
        XCTAssertEqual(alignment.right.centerY, 302, accuracy: 1e-9)
        XCTAssertEqual(alignment.left.clippedFraction, 0, accuracy: 1e-12)
        XCTAssertEqual(alignment.right.clippedFraction, 0, accuracy: 1e-12)
    }

    func testClippingUsesTheRequestedRotatedSquareArea() throws {
        let alignment = try XCTUnwrap(GazeDatasetCanonicalAlignment(
            left: eye(centerX: 5, centerY: 300, length: 20, angleDegrees: 0),
            right: eye(centerX: 800, centerY: 300, length: 20, angleDegrees: 0),
            imageWidth: 1280, imageHeight: 720))

        XCTAssertEqual(alignment.cropSidePixels, 36, accuracy: 1e-9)
        XCTAssertEqual(alignment.left.clippedFraction, 13.0 / 36.0, accuracy: 1e-9)
        XCTAssertEqual(alignment.right.clippedFraction, 0, accuracy: 1e-12)
    }

    func testRotatedAndFullyOutsideCropsHaveExactClippingFractions() throws {
        let alignment = try XCTUnwrap(GazeDatasetCanonicalAlignment(
            left: eye(centerX: 0, centerY: 0, length: 20, angleDegrees: 45),
            right: eye(centerX: -100, centerY: 300, length: 20, angleDegrees: 45),
            imageWidth: 1280, imageHeight: 720))

        XCTAssertEqual(alignment.left.clippedFraction, 0.75, accuracy: 1e-9)
        XCTAssertEqual(alignment.right.clippedFraction, 1, accuracy: 1e-12)
    }

    func testAlignmentRejectsMissingAndDegenerateAxes() {
        let missing = eye(centerX: 480, centerY: 300, length: 40, angleDegrees: 0)
        var degenerate = missing
        degenerate.imageAxisEnd = degenerate.imageAxisStart

        XCTAssertNil(GazeDatasetCanonicalAlignment(
            left: missing, right: EyeObservation(
                region: missing.region, pupilCenter: missing.pupilCenter, openness: 1),
            imageWidth: 1280, imageHeight: 720))
        XCTAssertNil(GazeDatasetCanonicalAlignment(
            left: degenerate, right: missing, imageWidth: 1280, imageHeight: 720))
    }

    func testAlignmentRejectsNonFiniteAndOpposingAxes() {
        var nonFinite = eye(centerX: 480, centerY: 300, length: 40, angleDegrees: 0)
        nonFinite.imageAxisStart?.x = .nan

        XCTAssertNil(GazeDatasetCanonicalAlignment(
            left: nonFinite,
            right: eye(centerX: 800, centerY: 300, length: 40, angleDegrees: 0),
            imageWidth: 1280, imageHeight: 720))
        XCTAssertNil(GazeDatasetCanonicalAlignment(
            left: eye(centerX: 480, centerY: 300, length: 40, angleDegrees: 0),
            right: eye(centerX: 800, centerY: 300, length: 40, angleDegrees: 180),
            imageWidth: 1280, imageHeight: 720))
    }
}

final class GazePosePromptGateTests: XCTestCase {
    func testHorizontalPromptsRequireRealOppositeMotion() {
        var gate = GazePosePromptGate()
        XCTAssertTrue(gate.accepts(.neutral, yawDegrees: 2, pitchDegrees: 8))
        XCTAssertFalse(gate.accepts(.turnLeft, yawDegrees: 5, pitchDegrees: 8))
        XCTAssertTrue(gate.accepts(.turnLeft, yawDegrees: -5, pitchDegrees: 8))
        XCTAssertFalse(gate.accepts(.turnRight, yawDegrees: -6, pitchDegrees: 8))
        XCTAssertTrue(gate.accepts(.turnRight, yawDegrees: 9, pitchDegrees: 8))
    }

    func testVerticalPromptsRequireRealOppositeMotion() {
        var gate = GazePosePromptGate()
        XCTAssertTrue(gate.accepts(.neutral, yawDegrees: 0, pitchDegrees: 10))
        XCTAssertFalse(gate.accepts(.lookUp, yawDegrees: 0, pitchDegrees: 13))
        XCTAssertTrue(gate.accepts(.lookUp, yawDegrees: 0, pitchDegrees: 4))
        XCTAssertFalse(gate.accepts(.lookDown, yawDegrees: 0, pitchDegrees: 2))
        XCTAssertTrue(gate.accepts(.lookDown, yawDegrees: 0, pitchDegrees: 17))
    }

    func testVerticalPromptSignCannotFlipBetweenSessions() {
        var gate = GazePosePromptGate()
        XCTAssertTrue(gate.accepts(.neutral, yawDegrees: 0, pitchDegrees: 0))
        XCTAssertFalse(gate.accepts(.lookUp, yawDegrees: 0, pitchDegrees: 6))
        XCTAssertFalse(gate.accepts(.lookDown, yawDegrees: 0, pitchDegrees: -6))
    }
}

final class GazeDatasetSettleGateTests: XCTestCase {
    func testRequiresContinuousAcceptanceForTheWholeSettleWindow() {
        var gate = GazeDatasetSettleGate()
        XCTAssertFalse(gate.isReady(targetID: 4, accepted: true,
                                    now: 10, settleSeconds: 2))
        XCTAssertFalse(gate.isReady(targetID: 4, accepted: false,
                                    now: 11.5, settleSeconds: 2))
        XCTAssertFalse(gate.isReady(targetID: 4, accepted: true,
                                    now: 12, settleSeconds: 2))
        XCTAssertTrue(gate.isReady(targetID: 4, accepted: true,
                                   now: 14, settleSeconds: 2))
    }

    func testChangingTargetStartsANewSettleWindow() {
        var gate = GazeDatasetSettleGate()
        XCTAssertFalse(gate.isReady(targetID: 4, accepted: true,
                                    now: 10, settleSeconds: 1))
        XCTAssertTrue(gate.isReady(targetID: 4, accepted: true,
                                   now: 11, settleSeconds: 1))
        XCTAssertFalse(gate.isReady(targetID: 5, accepted: true,
                                    now: 11.1, settleSeconds: 1))
    }
}
