import XCTest
@testable import AspectusKit

final class GazeDatasetPlanTests: XCTestCase {
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
                          eyeWidth: Double = 0.08) -> TrackingResult {
        let eye = EyeObservation(region: .init(x: 0.2, y: 0.2,
                                               width: eyeWidth, height: 0.03),
                                 pupilCenter: .init(x: 0.24, y: 0.215),
                                 openness: openness)
        return TrackingResult(faceBounds: .init(x: 0.1, y: 0.1, width: 0.8, height: 0.8),
                              leftEye: eye, rightEye: eye,
                              headPose: .init(yaw: yawDegrees * .pi / 180,
                                              pitch: 0, roll: 0),
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
        XCTAssertEqual(GazeDatasetAcceptance.rejection(tracking(eyeWidth: 0)), .degenerateEyes)
    }
}
