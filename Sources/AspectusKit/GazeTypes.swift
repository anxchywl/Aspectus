import Foundation

/// normalized image-space point, origin top-left, range [0,1] so landmarks survive format changes
public struct NormPoint: Sendable, Equatable {
    public var x: Double
    public var y: Double
    public init(x: Double, y: Double) { self.x = x; self.y = y }
}

/// axis-aligned normalized rectangle in image space
public struct NormRect: Sendable, Equatable {
    public var x: Double, y: Double, width: Double, height: Double
    public init(x: Double, y: Double, width: Double, height: Double) {
        self.x = x; self.y = y; self.width = width; self.height = height
    }
    public var center: NormPoint { NormPoint(x: x + width / 2, y: y + height / 2) }
    public func expanded(by f: Double) -> NormRect {
        let nx = max(0, x - width * f)
        let ny = max(0, y - height * f)
        let nw = min(1 - nx, width * (1 + 2 * f))
        let nh = min(1 - ny, height * (1 + 2 * f))
        return NormRect(x: nx, y: ny, width: nw, height: nh)
    }
}

/// where the pupil centre actually came from, since a contour centroid and a real pupil landmark
/// have very different accuracy and the difference is otherwise invisible downstream
public enum PupilSource: String, Sendable, Equatable, CaseIterable {
    case visionLandmark
    case contourCentroid
    case none
}

/// per-eye geometry from the tracking stage
public struct EyeObservation: Sendable {
    public var region: NormRect          // tight bbox of the eye opening
    public var pupilCenter: NormPoint    // detected pupil / iris center
    public var cornerMidpointY: Double?
    public var openness: Double          // 0 closed (blink), 1 fully open
    public var pupilSource: PupilSource
    public var pupilPointCount: Int      // landmark points behind pupilCenter, 0 for a fallback
    public init(region: NormRect, pupilCenter: NormPoint, openness: Double,
                pupilSource: PupilSource = .none, pupilPointCount: Int = 0,
                cornerMidpointY: Double? = nil) {
        self.region = region; self.pupilCenter = pupilCenter; self.openness = openness
        self.pupilSource = pupilSource; self.pupilPointCount = pupilPointCount
        self.cornerMidpointY = cornerMidpointY
    }

    public static func cornerMidpointY(of contour: [NormPoint]) -> Double? {
        guard let left = contour.min(by: { $0.x < $1.x }),
              let right = contour.max(by: { $0.x < $1.x }) else { return nil }
        return (left.y + right.y) / 2
    }

    /// pupil displacement from the horizontal aperture centre and vertical corner line
    public var pupilOffset: NormPoint {
        let c = region.center
        return NormPoint(x: pupilCenter.x - c.x,
                         y: pupilCenter.y - (cornerMidpointY ?? c.y))
    }
}

/// head orientation in radians, right-handed, camera-facing
public struct HeadPose: Sendable {
    public var yaw: Double, pitch: Double, roll: Double
    public init(yaw: Double, pitch: Double, roll: Double) {
        self.yaw = yaw; self.pitch = pitch; self.roll = roll
    }
}

/// tracking output for the primary face, nil TrackingResult means no valid face
public struct TrackingResult: Sendable {
    public var faceBounds: NormRect
    public var leftEye: EyeObservation
    public var rightEye: EyeObservation
    public var headPose: HeadPose
    public var confidence: Double
    /// false when the tracker had no yaw/pitch to report and substituted zero, which would make
    /// every head-pose gate silently inert rather than protective
    public var headPoseAvailable: Bool
    public init(faceBounds: NormRect, leftEye: EyeObservation, rightEye: EyeObservation,
                headPose: HeadPose, confidence: Double, headPoseAvailable: Bool = true) {
        self.faceBounds = faceBounds; self.leftEye = leftEye
        self.rightEye = rightEye; self.headPose = headPose; self.confidence = confidence
        self.headPoseAvailable = headPoseAvailable
    }
}

/// gaze relative to the camera axis in radians, (0,0) means already looking at the lens
public struct GazeEstimate: Sendable {
    public var yaw: Double        // + looking to subject's right
    public var pitch: Double      // + looking up
    public var confidence: Double
    public init(yaw: Double, pitch: Double, confidence: Double) {
        self.yaw = yaw; self.pitch = pitch; self.confidence = confidence
    }
    public var magnitudeDegrees: Double {
        (Foundation.sqrt(yaw * yaw + pitch * pitch)) * 180.0 / .pi
    }
}

/// the redirection asked of the corrector, angles are the offset to remove in radians
public struct CorrectionRequest: Sendable {
    public var yawOffset: Double
    public var pitchOffset: Double
    public var strength: Double   // 0…1 user/gate scaling
    public init(yawOffset: Double, pitchOffset: Double, strength: Double) {
        self.yawOffset = yawOffset; self.pitchOffset = pitchOffset; self.strength = strength
    }

    /// aims a gaze estimate at the lens without counting the screen offset twice
    ///
    /// a calibrated estimate is already lens-relative, so both measured axes are simply removed
    /// without calibration the vertical anatomical bias is unknown, so only yaw is measured and
    /// the explicit screen-to-lens lift supplies pitch
    public static func aimingAtLens(from gaze: GazeEstimate,
                                    calibrated: Bool,
                                    uncalibratedPitchOffset: Double,
                                    strength: Double = 1) -> CorrectionRequest {
        CorrectionRequest(yawOffset: -gaze.yaw,
                          pitchOffset: calibrated ? -gaze.pitch : uncalibratedPitchOffset,
                          strength: strength)
    }

    public var magnitudeDegrees: Double {
        (Foundation.sqrt(yawOffset * yawOffset + pitchOffset * pitchOffset)) * 180.0 / .pi
    }
}
