import Foundation

public enum GazeDatasetSplit: String, Codable, Sendable, CaseIterable {
    case training
    case validation
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
    public static let captureIntervalSeconds = 0.08
    public static let settleSeconds = 0.65
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

public enum GazeDatasetRejection: String, Codable, Sendable, Equatable, CaseIterable {
    case noTracking
    case lowConfidence
    case eyesClosed
    case headPoseUnavailable
    case headPose
    case posePrompt
    case degenerateEyes
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
    private var upPitchSign: Double?

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
            guard abs(change) >= config.minimumVerticalChangeDegrees else { return false }
            if upPitchSign == nil { upPitchSign = change.sign == .minus ? -1 : 1 }
            return change * (upPitchSign ?? 0) > 0
        case .lookDown:
            guard let baseline = neutralPitch, let upPitchSign else { return false }
            let change = pitchDegrees - baseline
            return abs(change) >= config.minimumVerticalChangeDegrees
                && change * upPitchSign < 0
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

        public init(minimumFaceConfidence: Double = 0.60,
                    minimumOpenness: Double = 0.40,
                    maximumHeadPoseDegrees: Double = 25.0) {
            self.minimumFaceConfidence = minimumFaceConfidence
            self.minimumOpenness = minimumOpenness
            self.maximumHeadPoseDegrees = maximumHeadPoseDegrees
        }
    }

    public static func rejection(_ tracking: TrackingResult?,
                                 config: Config = .init()) -> GazeDatasetRejection? {
        guard let tracking else { return .noTracking }
        guard tracking.confidence >= config.minimumFaceConfidence else { return .lowConfidence }
        guard tracking.leftEye.openness >= config.minimumOpenness,
              tracking.rightEye.openness >= config.minimumOpenness else { return .eyesClosed }
        guard tracking.headPoseAvailable else { return .headPoseUnavailable }
        let degrees = 180.0 / Double.pi
        let worst = max(abs(tracking.headPose.yaw), abs(tracking.headPose.pitch)) * degrees
        guard worst <= config.maximumHeadPoseDegrees else { return .headPose }
        guard usableImageRegion(tracking.leftEye.region),
              usableImageRegion(tracking.rightEye.region) else { return .degenerateEyes }
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
