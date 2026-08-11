import Foundation

public enum GazeDatasetSplit: String, Codable, Sendable, CaseIterable {
    case training
    case validation
}

/// Schema 5 differs from schema 4 in one declared factor: the crop side is per eye rather than one
/// shared side taken from the longer axis, so rendered eye scale no longer varies with head yaw.
public enum GazeDatasetSchema5 {
    public static let version = 5
    public static let manifestColumns = [
        "schema_version", "participant_id", "session_id", "split", "sample", "frame_id",
        "elapsed_s", "target_id", "target_kind", "target_x", "target_y", "target_yaw_deg",
        "target_pitch_deg", "pose_prompt", "head_yaw_deg", "head_pitch_deg", "head_roll_deg",
        "face_conf", "open_l", "open_r", "left_image", "right_image", "contour_points_l",
        "contour_points_r", "pupil_source_l", "pupil_source_r", "pupil_points_l",
        "pupil_points_r", "axis_start_x_l", "axis_start_y_l", "axis_end_x_l", "axis_end_y_l",
        "axis_start_x_r", "axis_start_y_r", "axis_end_x_r", "axis_end_y_r",
        "alignment_rotation_deg", "alignment_disagreement_deg", "crop_side_px_l",
        "crop_side_px_r", "crop_clipped_fraction_l", "crop_clipped_fraction_r",
    ]
}

public enum GazePosePrompt: String, Codable, Sendable, CaseIterable {
    case neutral
    case turnLeft
    case turnRight
    case lookUp
    case lookDown
}

public enum GazeDatasetTargetKind: String, Codable, Sendable {
    case screen
    case lens
}

/// one labelled fixation in the deterministic collection sequence
public struct GazeDatasetTarget: Codable, Sendable, Equatable, Identifiable {
    public var id: Int
    public var kind: GazeDatasetTargetKind
    public var xFraction: Double
    public var yFraction: Double
    public var pose: GazePosePrompt

    public init(id: Int, kind: GazeDatasetTargetKind,
                xFraction: Double, yFraction: Double,
                pose: GazePosePrompt) {
        self.id = id
        self.kind = kind
        self.xFraction = xFraction
        self.yFraction = yFraction
        self.pose = pose
    }
}

/// fixed before collection so train and validation sessions are comparable and reproducible
public enum GazeDatasetPlan {
    public static let gridSize = 5
    public static let samplesPerTarget = 6
    public static let captureIntervalSeconds = 0.12
    public static let settleSeconds = 1.0
    public static let lensSettleSeconds = 2.0
    public static let poseTransitionSettleSeconds = 1.5

    private static let fractions = [0.10, 0.30, 0.50, 0.70, 0.90]
    private static let trainingOrder = [
        12, 0, 24, 4, 20, 6, 18, 8, 16, 2, 22, 10, 14,
        1, 23, 5, 19, 9, 15, 3, 21, 7, 17, 11, 13,
    ]
    private static let validationOrder = Array(trainingOrder.reversed())

    public static func targets(for split: GazeDatasetSplit) -> [GazeDatasetTarget] {
        let baseOrder = split == .training ? trainingOrder : validationOrder
        var targets: [GazeDatasetTarget] = []
        for (poseIndex, pose) in GazePosePrompt.allCases.enumerated() {
            let shift = poseIndex * 4
            let order = Array(baseOrder[shift...] + baseOrder[..<shift])
            targets.append(.init(id: targets.count, kind: .lens,
                                 xFraction: 0.5, yFraction: 0, pose: pose))
            for gridIndex in order {
                let row = gridIndex / gridSize
                let column = gridIndex % gridSize
                targets.append(.init(id: targets.count, kind: .screen,
                                     xFraction: fractions[column], yFraction: fractions[row],
                                     pose: pose))
            }
            targets.append(.init(id: targets.count, kind: .lens,
                                 xFraction: 0.5, yFraction: 0, pose: pose))
        }
        return targets
    }
}

/// camera-relative label geometry for a point on the physical display
public struct GazeDatasetGeometry: Codable, Sendable, Equatable {
    public var displayWidthMM: Double
    public var displayHeightMM: Double
    public var viewingDistanceMM: Double

    public init(displayWidthMM: Double, displayHeightMM: Double,
                viewingDistanceMM: Double) {
        self.displayWidthMM = displayWidthMM
        self.displayHeightMM = displayHeightMM
        self.viewingDistanceMM = viewingDistanceMM
    }

    public var isUsable: Bool {
        displayWidthMM > 1 && displayHeightMM > 1
            && viewingDistanceMM >= 150 && viewingDistanceMM <= 2000
    }

    /// the lens sits at the top centre; positive yaw is right and positive pitch is up
    public func angles(for target: GazeDatasetTarget) -> (yaw: Double, pitch: Double)? {
        guard isUsable else { return nil }
        guard target.kind == .screen else { return (0, 0) }
        let right = (target.xFraction - 0.5) * displayWidthMM
        let up = -target.yFraction * displayHeightMM
        let degrees = 180.0 / Double.pi
        return (atan2(right, viewingDistanceMM) * degrees,
                atan2(up, viewingDistanceMM) * degrees)
    }
}

public struct GazeDatasetCropContract: Codable, Sendable, Equatable {
    public var version: Int
    public var coordinateSpace: String
    public var axisExtractor: String
    public var alignment: String
    public var center: String
    public var scale: String
    public var outputWidth: Int
    public var outputHeight: Int
    public var sampling: String
    public var colorSpace: String

    public static let canonicalPairedEyesV2 = GazeDatasetCropContract(
        version: 2,
        coordinateSpace: "source-image-fraction-top-left",
        axisExtractor: "farthest-contour-pair-ordered-image-x",
        alignment: "circular-mean-paired-eye-axes",
        center: "per-eye-axis-midpoint",
        scale: "\(GazeDatasetCanonicalAlignment.cropScale)x-own-eye-axis-length-pixels",
        outputWidth: GazeDatasetCanonicalAlignment.outputWidth,
        outputHeight: GazeDatasetCanonicalAlignment.outputHeight,
        sampling: "core-image-affine-hq-downsample-edge-clamp",
        colorSpace: "sRGB")
}

public struct GazeDatasetLabelContract: Codable, Sendable, Equatable {
    public var version: Int
    public var units: String
    public var origin: String
    public var yawPositive: String
    public var pitchPositive: String

    public static let lensAngularV1 = GazeDatasetLabelContract(
        version: 1,
        units: "degrees",
        origin: "physical-lens",
        yawPositive: "subject-right",
        pitchPositive: "up")
}

public struct GazeDatasetHeadPoseContract: Codable, Sendable, Equatable {
    public var version: Int
    public var source: String
    public var units: String
    public var order: String
    public var yawPositive: String
    public var pitchPositive: String
    public var rollPositive: String

    public static let visionRevision3DegreesV1 = GazeDatasetHeadPoseContract(
        version: 1,
        source: "Vision.VNFaceObservation.face-rectangles-revision-3",
        units: "degrees",
        order: "yaw-pitch-roll",
        yawPositive: "counterclockwise",
        pitchPositive: "head-down",
        rollPositive: "counterclockwise")
}

public struct GazeDatasetCanonicalAlignment: Sendable, Equatable {
    public struct EyeCrop: Sendable, Equatable {
        public var centerX: Double
        public var centerY: Double
        /// `cropScale ×` this eye's own axis length. Deriving the side per eye rather than from the
        /// longer of the pair keeps rendered eye scale invariant: under head yaw the far eye is
        /// foreshortened, and a shared side would render it smaller in proportion to that yaw.
        public var cropSidePixels: Double
        public var clippedFraction: Double
    }

    public static let cropScale = 1.8
    public static let outputWidth = 60
    public static let outputHeight = 60

    public var rotationRadians: Double
    public var disagreementDegrees: Double
    public var left: EyeCrop
    public var right: EyeCrop

    public init?(left: EyeObservation, right: EyeObservation,
                 imageWidth: Int, imageHeight: Int) {
        guard imageWidth > 0, imageHeight > 0,
              left.contourPointCount >= 2, right.contourPointCount >= 2,
              let leftStart = left.imageAxisStart, let leftEnd = left.imageAxisEnd,
              let rightStart = right.imageAxisStart, let rightEnd = right.imageAxisEnd else {
            return nil
        }
        let width = Double(imageWidth)
        let height = Double(imageHeight)
        let values = [
            leftStart.x, leftStart.y, leftEnd.x, leftEnd.y,
            rightStart.x, rightStart.y, rightEnd.x, rightEnd.y,
        ]
        guard values.allSatisfy(\.isFinite) else { return nil }

        let leftAxis = Self.axis(start: leftStart, end: leftEnd,
                                 width: width, height: height)
        let rightAxis = Self.axis(start: rightStart, end: rightEnd,
                                  width: width, height: height)
        guard leftAxis.length > 1e-6, rightAxis.length > 1e-6 else { return nil }
        let sine = sin(leftAxis.angle) + sin(rightAxis.angle)
        let cosine = cos(leftAxis.angle) + cos(rightAxis.angle)
        guard hypot(sine, cosine) > 1e-6 else { return nil }

        rotationRadians = atan2(sine, cosine)
        disagreementDegrees = abs(atan2(sin(leftAxis.angle - rightAxis.angle),
                                        cos(leftAxis.angle - rightAxis.angle))) * 180 / .pi
        let leftSide = Self.cropScale * leftAxis.length
        let rightSide = Self.cropScale * rightAxis.length
        guard leftSide.isFinite, leftSide > 1e-6,
              rightSide.isFinite, rightSide > 1e-6 else { return nil }
        self.left = EyeCrop(
            centerX: leftAxis.center.x,
            centerY: leftAxis.center.y,
            cropSidePixels: leftSide,
            clippedFraction: Self.clippedFraction(
                center: leftAxis.center, rotation: rotationRadians,
                side: leftSide, width: width, height: height))
        self.right = EyeCrop(
            centerX: rightAxis.center.x,
            centerY: rightAxis.center.y,
            cropSidePixels: rightSide,
            clippedFraction: Self.clippedFraction(
                center: rightAxis.center, rotation: rotationRadians,
                side: rightSide, width: width, height: height))
    }

    private struct Point {
        var x: Double
        var y: Double
    }

    private static func axis(start: NormPoint, end: NormPoint,
                             width: Double, height: Double)
        -> (center: Point, length: Double, angle: Double) {
        let first = Point(x: start.x * width, y: start.y * height)
        let second = Point(x: end.x * width, y: end.y * height)
        let dx = second.x - first.x
        let dy = second.y - first.y
        return (Point(x: (first.x + second.x) / 2,
                      y: (first.y + second.y) / 2),
                hypot(dx, dy), atan2(dy, dx))
    }

    private static func clippedFraction(center: Point, rotation: Double,
                                        side: Double, width: Double, height: Double) -> Double {
        let half = side / 2
        let axis = Point(x: cos(rotation), y: sin(rotation))
        let perpendicular = Point(x: -axis.y, y: axis.x)
        var polygon = [
            Point(x: center.x - half * axis.x - half * perpendicular.x,
                  y: center.y - half * axis.y - half * perpendicular.y),
            Point(x: center.x + half * axis.x - half * perpendicular.x,
                  y: center.y + half * axis.y - half * perpendicular.y),
            Point(x: center.x + half * axis.x + half * perpendicular.x,
                  y: center.y + half * axis.y + half * perpendicular.y),
            Point(x: center.x - half * axis.x + half * perpendicular.x,
                  y: center.y - half * axis.y + half * perpendicular.y),
        ]
        polygon = clip(polygon, inside: { $0.x >= 0 }, vertical: 0)
        polygon = clip(polygon, inside: { $0.x <= width }, vertical: width)
        polygon = clip(polygon, inside: { $0.y >= 0 }, horizontal: 0)
        polygon = clip(polygon, inside: { $0.y <= height }, horizontal: height)
        let visible = polygonArea(polygon)
        return min(1, max(0, 1 - visible / (side * side)))
    }

    private static func clip(_ polygon: [Point], inside: (Point) -> Bool,
                             vertical boundary: Double) -> [Point] {
        clip(polygon, inside: inside) { first, second in
            let dx = second.x - first.x
            let amount = abs(dx) > 1e-12 ? (boundary - first.x) / dx : 0
            return Point(x: boundary, y: first.y + amount * (second.y - first.y))
        }
    }

    private static func clip(_ polygon: [Point], inside: (Point) -> Bool,
                             horizontal boundary: Double) -> [Point] {
        clip(polygon, inside: inside) { first, second in
            let dy = second.y - first.y
            let amount = abs(dy) > 1e-12 ? (boundary - first.y) / dy : 0
            return Point(x: first.x + amount * (second.x - first.x), y: boundary)
        }
    }

    private static func clip(_ polygon: [Point], inside: (Point) -> Bool,
                             intersection: (Point, Point) -> Point) -> [Point] {
        guard var previous = polygon.last else { return [] }
        var result: [Point] = []
        for current in polygon {
            if inside(current) {
                if !inside(previous) { result.append(intersection(previous, current)) }
                result.append(current)
            } else if inside(previous) {
                result.append(intersection(previous, current))
            }
            previous = current
        }
        return result
    }

    private static func polygonArea(_ polygon: [Point]) -> Double {
        guard polygon.count >= 3 else { return 0 }
        return abs(polygon.indices.reduce(0.0) { total, index in
            let next = polygon[(index + 1) % polygon.count]
            return total + polygon[index].x * next.y - next.x * polygon[index].y
        }) / 2
    }
}

public enum GazeDatasetRejection: String, Codable, Sendable, Equatable, CaseIterable {
    case noTracking
    case lowConfidence
    case eyesClosed
    case headPoseUnavailable
    case headPose
    case posePrompt
    case degenerateEyes
    case eyeAlignment
}

/// verifies that prompted head positions add real pose coverage instead of relying on instructions
public struct GazePosePromptGate: Sendable, Equatable {
    public struct Config: Sendable, Equatable {
        public var minimumHorizontalChangeDegrees: Double
        public var minimumVerticalChangeDegrees: Double

        public init(minimumHorizontalChangeDegrees: Double = 6,
                    minimumVerticalChangeDegrees: Double = 5) {
            self.minimumHorizontalChangeDegrees = minimumHorizontalChangeDegrees
            self.minimumVerticalChangeDegrees = minimumVerticalChangeDegrees
        }
    }

    private var config: Config
    private var neutralYawTotal = 0.0
    private var neutralPitchTotal = 0.0
    private var neutralSamples = 0
    private var leftYawSign: Double?

    public init(config: Config = .init()) { self.config = config }

    public mutating func accepts(_ prompt: GazePosePrompt,
                                 yawDegrees: Double, pitchDegrees: Double) -> Bool {
        switch prompt {
        case .neutral:
            neutralYawTotal += yawDegrees
            neutralPitchTotal += pitchDegrees
            neutralSamples += 1
            return true
        case .turnLeft:
            guard let baseline = neutralYaw else { return false }
            let change = yawDegrees - baseline
            guard abs(change) >= config.minimumHorizontalChangeDegrees else { return false }
            if leftYawSign == nil { leftYawSign = change.sign == .minus ? -1 : 1 }
            return change * (leftYawSign ?? 0) > 0
        case .turnRight:
            guard let baseline = neutralYaw, let leftYawSign else { return false }
            let change = yawDegrees - baseline
            return abs(change) >= config.minimumHorizontalChangeDegrees
                && change * leftYawSign < 0
        case .lookUp:
            guard let baseline = neutralPitch else { return false }
            let change = pitchDegrees - baseline
            return change <= -config.minimumVerticalChangeDegrees
        case .lookDown:
            guard let baseline = neutralPitch else { return false }
            let change = pitchDegrees - baseline
            return change >= config.minimumVerticalChangeDegrees
        }
    }

    private var neutralYaw: Double? {
        neutralSamples > 0 ? neutralYawTotal / Double(neutralSamples) : nil
    }

    private var neutralPitch: Double? {
        neutralSamples > 0 ? neutralPitchTotal / Double(neutralSamples) : nil
    }
}

/// requires a continuously valid target before collection resumes
public struct GazeDatasetSettleGate: Sendable, Equatable {
    private var targetID: Int?
    private var validSince: Double?

    public init() {}

    public mutating func isReady(targetID: Int, accepted: Bool,
                                 now: Double, settleSeconds: Double) -> Bool {
        if self.targetID != targetID {
            self.targetID = targetID
            validSince = nil
        }
        guard accepted else {
            validSince = nil
            return false
        }
        guard let validSince else {
            self.validSince = now
            return settleSeconds <= 0
        }
        return now - validSince >= settleSeconds
    }
}

/// collection gate kept in the core so labelled images cannot bypass it silently
public enum GazeDatasetAcceptance {
    public struct Config: Sendable, Equatable {
        public var minimumFaceConfidence: Double
        public var minimumOpenness: Double
        public var maximumHeadPoseDegrees: Double
        public var maximumHeadRollDegrees: Double

        public init(minimumFaceConfidence: Double = 0.70,
                    minimumOpenness: Double = 0.40,
                    maximumHeadPoseDegrees: Double = 25.0,
                    maximumHeadRollDegrees: Double = 20.0) {
            self.minimumFaceConfidence = minimumFaceConfidence
            self.minimumOpenness = minimumOpenness
            self.maximumHeadPoseDegrees = maximumHeadPoseDegrees
            self.maximumHeadRollDegrees = maximumHeadRollDegrees
        }
    }

    public static func rejection(_ tracking: TrackingResult?,
                                 imageWidth: Int? = nil, imageHeight: Int? = nil,
                                 config: Config = .init()) -> GazeDatasetRejection? {
        guard let tracking else { return .noTracking }
        guard tracking.confidence >= config.minimumFaceConfidence else { return .lowConfidence }
        guard tracking.leftEye.openness >= config.minimumOpenness,
              tracking.rightEye.openness >= config.minimumOpenness else { return .eyesClosed }
        guard tracking.headPoseAvailable else { return .headPoseUnavailable }
        let pose = [tracking.headPose.yaw, tracking.headPose.pitch, tracking.headPose.roll]
        guard pose.allSatisfy(\.isFinite) else { return .headPose }
        let degrees = 180.0 / Double.pi
        let worst = max(abs(tracking.headPose.yaw), abs(tracking.headPose.pitch)) * degrees
        let roll = abs(tracking.headPose.roll) * degrees
        guard worst <= config.maximumHeadPoseDegrees,
              roll <= config.maximumHeadRollDegrees else { return .headPose }
        guard usableImageRegion(tracking.leftEye.region),
              usableImageRegion(tracking.rightEye.region) else { return .degenerateEyes }
        if imageWidth != nil || imageHeight != nil {
            guard let imageWidth, let imageHeight,
                  GazeDatasetCanonicalAlignment(
                    left: tracking.leftEye, right: tracking.rightEye,
                    imageWidth: imageWidth, imageHeight: imageHeight) != nil else {
                return .eyeAlignment
            }
        }
        return nil
    }

    private static func usableImageRegion(_ region: NormRect) -> Bool {
        let values = [region.x, region.y, region.width, region.height]
        guard values.allSatisfy(\.isFinite),
              region.width > 1e-6, region.height > 1e-6 else { return false }
        return region.x < 1 && region.y < 1
            && region.x + region.width > 0
            && region.y + region.height > 0
    }
}
