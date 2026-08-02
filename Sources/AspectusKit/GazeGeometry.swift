import Foundation

/// tuning for the geometric eyeball model, all ratios rather than absolute sizes so the same
/// values hold at any capture resolution or face distance
public struct EyeWarpTuning: Sendable {
    /// eyeball radius as a fraction of palpebral fissure width, from adult anthropometry
    /// (eyeball radius ~12 mm, fissure width ~30 mm) — an approximation, not a per-user measurement
    public var eyeballRadiusPerEyeWidth: Double
    /// how much of the pupil offset is removed, below 1 deliberately undercorrects
    public var recenterGain: Double
    /// influence radii as a fraction of the eye aperture's own extent
    /// deliberately aperture-shaped, not circular: an eye is roughly five times wider than tall,
    /// so a pixel-space circle covering the iris would also drag the lids, lashes, and brows
    public var influenceRadiusPerEyeExtent: Double
    /// fraction of the influence radius that moves rigidly before the falloff begins, sized to the
    /// iris (radius ~0.2 of eye width) so the iris slides intact instead of stretching
    public var irisPlateau: Double

    /// below this openness the eye is blinking or nearly closed and must not be touched
    public var minOpenness: Double
    /// past this head turn a centered pupil no longer means looking at the lens
    public var maxHeadPoseDegrees: Double

    public init(eyeballRadiusPerEyeWidth: Double = 0.40,
                recenterGain: Double = 0.85,
                influenceRadiusPerEyeExtent: Double = 0.50,
                irisPlateau: Double = 0.45,
                minOpenness: Double = 0.35,
                maxHeadPoseDegrees: Double = 25.0) {
        self.eyeballRadiusPerEyeWidth = eyeballRadiusPerEyeWidth
        self.recenterGain = recenterGain
        self.influenceRadiusPerEyeExtent = influenceRadiusPerEyeExtent
        self.irisPlateau = irisPlateau
        self.minOpenness = minOpenness
        self.maxHeadPoseDegrees = maxHeadPoseDegrees
    }
}

/// one eye's resampling instruction in normalized image space
/// the resampler reads the source at `uv + displacement * falloff(uv)`, so content moves by
/// `-displacement`; everything outside the influence ellipse is left bit-identical
public struct EyeWarp: Sendable, Equatable {
    /// centre of the influence ellipse, the detected pupil
    public var center: NormPoint
    /// ellipse radii in normalized x and y, shaped like the eye aperture so lids stay put
    public var radius: NormPoint
    /// sampling offset applied at the centre, decaying to zero at the ellipse boundary
    public var displacement: NormPoint

    public init(center: NormPoint, radius: NormPoint, displacement: NormPoint) {
        self.center = center
        self.radius = radius
        self.displacement = displacement
    }
}

/// geometric gaze model: the iris is a disc on a sphere, so redirecting gaze toward the lens is
/// re-centering the iris in its aperture rather than synthesizing anything
///
/// deliberately free of any left/right or yaw-sign convention on the correction path — the
/// displacement is derived purely from observed image geometry, so it cannot be inverted by a
/// disagreement with Vision's or the camera's handedness
public enum GazeGeometry {
    /// pupil offset from the aperture centre, in units of eye widths
    private static func pupilOffset(_ eye: EyeObservation) -> (x: Double, y: Double)? {
        guard eye.region.width > 1e-6, eye.region.height > 1e-6 else { return nil }
        let c = eye.region.center
        return (eye.pupilCenter.x - c.x, eye.pupilCenter.y - c.y)
    }

    /// nil when the geometry cannot be trusted: degenerate eye region, closed eyes, or a head
    /// turn large enough to invalidate the centred-pupil assumption
    ///
    /// sign convention follows GazeEstimate (+yaw looks to the subject's right, +pitch looks up);
    /// only the magnitude drives gating, so a handedness disagreement cannot misdirect correction
    public static func estimate(_ tracking: TrackingResult,
                                imageAspect: Double,
                                tuning: EyeWarpTuning = .init()) -> GazeEstimate? {
        guard imageAspect > 0 else { return nil }
        let degrees = 180.0 / Double.pi
        let pose = tracking.headPose
        let worstPose = max(abs(pose.yaw), abs(pose.pitch)) * degrees
        guard worstPose <= tuning.maxHeadPoseDegrees else { return nil }

        var yaws: [Double] = []
        var pitches: [Double] = []
        for eye in [tracking.leftEye, tracking.rightEye] {
            guard eye.openness >= tuning.minOpenness,
                  let offset = pupilOffset(eye) else { continue }
            let radiusInEyeWidths = tuning.eyeballRadiusPerEyeWidth * eye.region.width
            guard radiusInEyeWidths > 1e-6 else { continue }
            // image y is normalized against height, so it needs the aspect to become a pixel ratio
            let sinYaw = offset.x / radiusInEyeWidths
            let sinPitch = offset.y / (radiusInEyeWidths * imageAspect)
            yaws.append(-asin(max(-1, min(1, sinYaw))))
            pitches.append(-asin(max(-1, min(1, sinPitch))))
        }
        guard !yaws.isEmpty else { return nil }

        let yaw = yaws.reduce(0, +) / Double(yaws.count)
        let pitch = pitches.reduce(0, +) / Double(pitches.count)
        // both eyes agreeing is the strongest signal that the landmarks are real
        let agreement = yaws.count == 2 ? 1.0 : 0.5
        return GazeEstimate(yaw: yaw, pitch: pitch, confidence: tracking.confidence * agreement)
    }

    /// nil when either eye must be left untouched, which the caller turns into original-frame
    /// passthrough rather than correcting one eye and not the other
    /// the request carries the total angular redirection to apply: removing the observed gaze plus
    /// whatever screen-to-lens offset the layout demands, which is not visible in the image at all
    public static func warps(_ tracking: TrackingResult,
                             imageAspect: Double,
                             request: CorrectionRequest,
                             tuning: EyeWarpTuning = .init()) -> (left: EyeWarp, right: EyeWarp)? {
        guard request.strength > 0, imageAspect > 0 else { return nil }
        guard let left = warp(for: tracking.leftEye, imageAspect: imageAspect,
                              request: request, tuning: tuning),
              let right = warp(for: tracking.rightEye, imageAspect: imageAspect,
                               request: request, tuning: tuning)
        else { return nil }
        return (left, right)
    }

    private static func warp(for eye: EyeObservation,
                             imageAspect: Double,
                             request: CorrectionRequest,
                             tuning: EyeWarpTuning) -> EyeWarp? {
        guard eye.openness >= tuning.minOpenness, eye.region.width > 1e-6,
              eye.region.height > 1e-6 else { return nil }

        let scale = tuning.recenterGain * max(0, min(1, request.strength))

        // rotating the eyeball sweeps the iris across the sphere, so the image-space travel is
        // radius * sin(angle); eyeballRadius is in normalized x, hence the aspect term on y only
        let eyeballRadius = tuning.eyeballRadiusPerEyeWidth * eye.region.width
        let displacement = NormPoint(
            x: eyeballRadius * sin(request.yawOffset) * scale,
            y: eyeballRadius * sin(request.pitchOffset) * imageAspect * scale)

        // radii track the aperture's own extent, so the falloff reaches zero inside the eye
        let radius = NormPoint(x: tuning.influenceRadiusPerEyeExtent * eye.region.width,
                               y: tuning.influenceRadiusPerEyeExtent * eye.region.height)

        return EyeWarp(center: eye.pupilCenter, radius: radius, displacement: displacement)
    }

    /// pixel travel of the iris for a warp, the number that must be non-trivial for correction to
    /// be visible at all — a couple of pixels is invisible and was the first prototype's failure
    public static func displacementPixels(_ warp: EyeWarp, width: Int, height: Int) -> Double {
        let dx = warp.displacement.x * Double(width)
        let dy = warp.displacement.y * Double(height)
        return (dx * dx + dy * dy).squareRoot()
    }
}
