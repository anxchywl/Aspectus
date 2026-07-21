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

/// per-eye geometry from the tracking stage
public struct EyeObservation: Sendable {
    public var region: NormRect          // tight bbox of the eye opening
    public var pupilCenter: NormPoint    // detected pupil / iris center
    public var openness: Double          // 0 closed (blink), 1 fully open
    public init(region: NormRect, pupilCenter: NormPoint, openness: Double) {
        self.region = region; self.pupilCenter = pupilCenter; self.openness = openness
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
    public init(faceBounds: NormRect, leftEye: EyeObservation, rightEye: EyeObservation,
                headPose: HeadPose, confidence: Double) {
        self.faceBounds = faceBounds; self.leftEye = leftEye
        self.rightEye = rightEye; self.headPose = headPose; self.confidence = confidence
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
    public var magnitudeDegrees: Double {
        (Foundation.sqrt(yawOffset * yawOffset + pitchOffset * pitchOffset)) * 180.0 / .pi
    }
}
