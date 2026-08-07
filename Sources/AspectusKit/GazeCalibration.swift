import Foundation

/// where the user was asked to look while a sample was taken, all relative to the camera lens
/// rather than the screen, because the lens is what the correction aims the eyes at
public enum CalibrationTarget: String, Codable, Sendable, Equatable, CaseIterable {
    case center
    case up
    case down
    case left
    case right
}

/// one accepted observation, angles in degrees so a stored file is readable without conversion
///
/// deliberately carries no imagery and no landmark coordinates — only derived angles — so the
/// persisted file cannot reconstruct a picture of anyone
public struct CalibrationSample: Codable, Sendable, Equatable {
    public var target: CalibrationTarget
    public var yawDegrees: Double
    public var pitchDegrees: Double
    public var leftYawDegrees: Double
    public var leftPitchDegrees: Double
    public var rightYawDegrees: Double
    public var rightPitchDegrees: Double
    public var headYawDegrees: Double
    public var headPitchDegrees: Double
    public var headRollDegrees: Double
    public var confidence: Double
    /// raw pupil displacement behind each angle, retained to diagnose suspicious fits
    public var leftOffsetX: Double
    public var leftOffsetY: Double
    public var rightOffsetX: Double
    public var rightOffsetY: Double

    public init(target: CalibrationTarget,
                yawDegrees: Double, pitchDegrees: Double,
                leftYawDegrees: Double, leftPitchDegrees: Double,
                rightYawDegrees: Double, rightPitchDegrees: Double,
                headYawDegrees: Double, headPitchDegrees: Double, headRollDegrees: Double,
                confidence: Double,
                leftOffsetX: Double = 0, leftOffsetY: Double = 0,
                rightOffsetX: Double = 0, rightOffsetY: Double = 0) {
        self.leftOffsetX = leftOffsetX
        self.leftOffsetY = leftOffsetY
        self.rightOffsetX = rightOffsetX
        self.rightOffsetY = rightOffsetY
        self.target = target
        self.yawDegrees = yawDegrees
        self.pitchDegrees = pitchDegrees
        self.leftYawDegrees = leftYawDegrees
        self.leftPitchDegrees = leftPitchDegrees
        self.rightYawDegrees = rightYawDegrees
        self.rightPitchDegrees = rightPitchDegrees
        self.headYawDegrees = headYawDegrees
        self.headPitchDegrees = headPitchDegrees
        self.headRollDegrees = headRollDegrees
        self.confidence = confidence
    }

    /// preserves decoding for samples written before raw offsets were stored
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        target = try c.decode(CalibrationTarget.self, forKey: .target)
        yawDegrees = try c.decode(Double.self, forKey: .yawDegrees)
        pitchDegrees = try c.decode(Double.self, forKey: .pitchDegrees)
        leftYawDegrees = try c.decode(Double.self, forKey: .leftYawDegrees)
        leftPitchDegrees = try c.decode(Double.self, forKey: .leftPitchDegrees)
        rightYawDegrees = try c.decode(Double.self, forKey: .rightYawDegrees)
        rightPitchDegrees = try c.decode(Double.self, forKey: .rightPitchDegrees)
        headYawDegrees = try c.decode(Double.self, forKey: .headYawDegrees)
        headPitchDegrees = try c.decode(Double.self, forKey: .headPitchDegrees)
        headRollDegrees = try c.decode(Double.self, forKey: .headRollDegrees)
        confidence = try c.decode(Double.self, forKey: .confidence)
        leftOffsetX = try c.decodeIfPresent(Double.self, forKey: .leftOffsetX) ?? 0
        leftOffsetY = try c.decodeIfPresent(Double.self, forKey: .leftOffsetY) ?? 0
        rightOffsetX = try c.decodeIfPresent(Double.self, forKey: .rightOffsetX) ?? 0
        rightOffsetY = try c.decodeIfPresent(Double.self, forKey: .rightOffsetY) ?? 0
    }

    /// mean of the two eyes, which is the quantity `GazeGeometry.estimate` effectively averages
    public var offsetX: Double { (leftOffsetX + rightOffsetX) / 2 }
    public var offsetY: Double { (leftOffsetY + rightOffsetY) / 2 }
}

/// per-axis affine map from raw geometric gaze to calibrated gaze
///
/// offset comes from the one target with exact ground truth — looking at the lens is 0° by
/// definition. gain needs the true angle of the off-axis targets, which needs display geometry and
/// a viewing distance, so it is fitted only when that geometry is supplied and pinned to 1 otherwise
public struct GazeCalibration: Codable, Sendable, Equatable {
    public static let currentVersion = 2

    public var version: Int
    public var yawOffsetDegrees: Double
    public var pitchOffsetDegrees: Double
    public var yawGain: Double
    public var pitchGain: Double
    public var verticalSeparationDegrees: Double
    public var horizontalSeparationDegrees: Double
    public var sampleCount: Int
    public var createdAt: Date
    /// nil when the gains were pinned rather than fitted, which the UI has to be able to say
    public var viewingDistanceMM: Double?
    public var gainFitted: Bool?
    /// per axis, because one axis can be fitted while the other is refused
    public var yawGainFitted: Bool?
    public var pitchGainFitted: Bool?
    /// nil when the head-motion step was skipped, which means no compensation at all
    public var headCoupling: HeadCoupling?

    public init(version: Int = GazeCalibration.currentVersion,
                yawOffsetDegrees: Double,
                pitchOffsetDegrees: Double,
                yawGain: Double = 1.0,
                pitchGain: Double = 1.0,
                verticalSeparationDegrees: Double = 0,
                horizontalSeparationDegrees: Double = 0,
                sampleCount: Int = 0,
                createdAt: Date = Date(),
                viewingDistanceMM: Double? = nil,
                gainFitted: Bool? = nil,
                yawGainFitted: Bool? = nil,
                pitchGainFitted: Bool? = nil,
                headCoupling: HeadCoupling? = nil) {
        self.version = version
        self.yawOffsetDegrees = yawOffsetDegrees
        self.pitchOffsetDegrees = pitchOffsetDegrees
        self.yawGain = yawGain
        self.pitchGain = pitchGain
        self.verticalSeparationDegrees = verticalSeparationDegrees
        self.horizontalSeparationDegrees = horizontalSeparationDegrees
        self.sampleCount = sampleCount
        self.createdAt = createdAt
        self.viewingDistanceMM = viewingDistanceMM
        self.gainFitted = gainFitted
        self.yawGainFitted = yawGainFitted
        self.pitchGainFitted = pitchGainFitted
        self.headCoupling = headCoupling
    }

    /// raw → remove head contamination → remove the neutral bias → scale to true angles
    ///
    /// head compensation is skipped when the tracker reported no pose: subtracting a coupling term
    /// against a fabricated zero would be worse than leaving the contamination in
    public func apply(_ gaze: GazeEstimate,
                      headYawDegrees: Double = 0,
                      headPitchDegrees: Double = 0,
                      headPoseAvailable: Bool = false) -> GazeEstimate {
        let toRadians = Double.pi / 180
        var yawDegrees = gaze.yaw / toRadians
        var pitchDegrees = gaze.pitch / toRadians

        if headPoseAvailable, let headCoupling {
            let contribution = headCoupling.contribution(headYawDegrees: headYawDegrees,
                                                         headPitchDegrees: headPitchDegrees)
            yawDegrees -= contribution.yaw
            pitchDegrees -= contribution.pitch
        }

        return GazeEstimate(yaw: yawGain * (yawDegrees - yawOffsetDegrees) * toRadians,
                            pitch: pitchGain * (pitchDegrees - pitchOffsetDegrees) * toRadians,
                            confidence: gaze.confidence)
    }

    /// a file written by a future build, or one whose numbers are physically implausible, must be
    /// ignored rather than silently steering correction with nonsense
    public var isUsable: Bool {
        guard version == Self.currentVersion else { return false }
        let finite = [yawOffsetDegrees, pitchOffsetDegrees, yawGain, pitchGain].allSatisfy(\.isFinite)
        return finite
            && abs(yawOffsetDegrees) <= Self.maxPlausibleOffsetDegrees
            && abs(pitchOffsetDegrees) <= Self.maxPlausibleOffsetDegrees
            && yawGain > 0 && yawGain <= 4 && pitchGain > 0 && pitchGain <= 4
    }

    /// a neutral bias larger than this is a botched calibration, not a face
    public static let maxPlausibleOffsetDegrees = 20.0
}

/// fits a calibration from accepted samples, or explains precisely why it cannot
public enum CalibrationFit {
    public enum Failure: Error, Equatable, Sendable {
        case notEnoughSamples(CalibrationTarget, have: Int, need: Int)
        /// looking up and looking down produced the same reading, so the vertical axis is unusable
        case verticalNotSeparated(measuredDegrees: Double, needDegrees: Double)
        case horizontalNotSeparated(measuredDegrees: Double, needDegrees: Double)
        /// the axis responds backwards, which a fitted gain would hide instead of surfacing
        case verticalSignInverted(measuredDegrees: Double)
        case horizontalSignInverted(measuredDegrees: Double)
        case implausibleOffset(yawDegrees: Double, pitchDegrees: Double)
        /// a fitted slope this far from unity means the geometry or the distance is wrong, and
        /// storing it would scale every later frame by a wrong constant
        case implausibleGain(yaw: Double, pitch: Double)
    }

    /// outside this band the fit is not believable, whatever the arithmetic says
    public static let gainBounds = 0.2...4.0

    /// the off-axis targets must move the estimate by at least this much to count as a response
    public static let minimumSeparationDegrees = 2.0
    public static let minimumSamplesPerTarget = 10

    public static func fit(_ samples: [CalibrationSample],
                           geometry: CalibrationGeometry? = nil,
                           headCoupling: HeadCoupling? = nil,
                           minimumPerTarget: Int = minimumSamplesPerTarget,
                           minimumSeparation: Double = minimumSeparationDegrees,
                           now: Date = Date()) throws -> GazeCalibration {
        for target in CalibrationTarget.allCases {
            let count = samples.count { $0.target == target }
            guard count >= minimumPerTarget else {
                throw Failure.notEnoughSamples(target, have: count, need: minimumPerTarget)
            }
        }

        func mean(_ target: CalibrationTarget, _ value: (CalibrationSample) -> Double) -> Double {
            let matching = samples.filter { $0.target == target }
            return matching.reduce(0) { $0 + value($1) } / Double(matching.count)
        }

        // +pitch is up and +yaw is the subject's right, so both separations must come out positive
        let vertical = mean(.up, \.pitchDegrees) - mean(.down, \.pitchDegrees)
        let horizontal = mean(.right, \.yawDegrees) - mean(.left, \.yawDegrees)

        if vertical <= -minimumSeparation { throw Failure.verticalSignInverted(measuredDegrees: vertical) }
        guard vertical >= minimumSeparation else {
            throw Failure.verticalNotSeparated(measuredDegrees: vertical, needDegrees: minimumSeparation)
        }
        if horizontal <= -minimumSeparation {
            throw Failure.horizontalSignInverted(measuredDegrees: horizontal)
        }
        guard horizontal >= minimumSeparation else {
            throw Failure.horizontalNotSeparated(measuredDegrees: horizontal, needDegrees: minimumSeparation)
        }

        // looking at the lens is the one target whose true angle is known exactly, so it and only
        // it sets the neutral baseline
        let yawOffset = mean(.center, \.yawDegrees)
        let pitchOffset = mean(.center, \.pitchDegrees)

        // slope through the origin on the offset-corrected readings: gain = Σ(true·measured)/Σ(measured²)
        //
        // the axes use different targets because their geometry differs in quality. yaw uses the
        // two side targets, whose positions on the display are exact. pitch uses the lens itself
        // (0° by definition) and the bottom edge; the "look above the camera" target has no
        // well-defined physical position, so it stays a sign check and never enters the fit
        func gain(_ targets: [CalibrationTarget],
                  offset: Double,
                  measured: (CalibrationSample) -> Double,
                  truth: (CalibrationTarget) -> Double?) -> Double? {
            var numerator = 0.0, denominator = 0.0
            for sample in samples where targets.contains(sample.target) {
                guard let trueAngle = truth(sample.target) else { return nil }
                let m = measured(sample) - offset
                numerator += trueAngle * m
                denominator += m * m
            }
            guard denominator > 1e-9 else { return nil }
            return numerator / denominator
        }

        // each axis stands or falls on its own: an implausible slope on one is no reason to throw
        // away a sound slope on the other, still less the neutral bias, which does not depend on
        // geometry at all. an unusable axis degrades to 1.0 and is reported as unfitted
        var yawGain = 1.0, pitchGain = 1.0
        var yawFitted = false, pitchFitted = false
        if let geometry, geometry.isUsable {
            if let y = gain([.left, .right], offset: yawOffset, measured: \.yawDegrees,
                            truth: { geometry.trueAngles($0)?.yaw }), gainBounds.contains(y) {
                yawGain = y
                yawFitted = true
            }
            if let p = gain([.center, .down], offset: pitchOffset, measured: \.pitchDegrees,
                            truth: { geometry.trueAngles($0)?.pitch }), gainBounds.contains(p) {
                pitchGain = p
                pitchFitted = true
            }
        }
        let fitted = yawFitted || pitchFitted

        let calibration = GazeCalibration(yawOffsetDegrees: yawOffset,
                                          pitchOffsetDegrees: pitchOffset,
                                          yawGain: yawGain,
                                          pitchGain: pitchGain,
                                          verticalSeparationDegrees: vertical,
                                          horizontalSeparationDegrees: horizontal,
                                          sampleCount: samples.count,
                                          createdAt: now,
                                          viewingDistanceMM: fitted ? geometry?.viewingDistanceMM : nil,
                                          gainFitted: fitted,
                                          yawGainFitted: yawFitted,
                                          pitchGainFitted: pitchFitted,
                                          headCoupling: headCoupling)
        guard calibration.isUsable else {
            throw Failure.implausibleOffset(yawDegrees: yawOffset, pitchDegrees: pitchOffset)
        }
        return calibration
    }
}

extension CalibrationFit.Failure: CustomStringConvertible {
    public var description: String {
        switch self {
        case let .notEnoughSamples(target, have, need):
            return "not enough usable samples looking \(target.rawValue) (\(have) of \(need))"
        case let .verticalNotSeparated(measured, need):
            return String(format: "looking up and down read almost the same (%.1f°, need %.1f°)",
                          measured, need)
        case let .horizontalNotSeparated(measured, need):
            return String(format: "looking left and right read almost the same (%.1f°, need %.1f°)",
                          measured, need)
        case let .verticalSignInverted(measured):
            return String(format: "the vertical estimate responds backwards (%.1f°)", measured)
        case let .horizontalSignInverted(measured):
            return String(format: "the horizontal estimate responds backwards (%.1f°)", measured)
        case let .implausibleOffset(yaw, pitch):
            return String(format: "neutral bias is implausible (yaw %.1f°, pitch %.1f°)", yaw, pitch)
        case let .implausibleGain(yaw, pitch):
            return String(format: "fitted scale is implausible (yaw ×%.2f, pitch ×%.2f) — check the "
                          + "viewing distance", yaw, pitch)
        }
    }
}

/// physical position of a calibration target relative to the camera lens, in millimetres
public struct TargetOffsetMM: Sendable, Codable, Equatable {
    public var right: Double   // + toward the subject's right
    public var up: Double      // + above the lens
    public init(right: Double, up: Double) { self.right = right; self.up = up }
}

/// the physical layout a gain fit needs: where each target sits relative to the lens, and how far
/// away the viewer is
///
/// distance has to be supplied rather than measured — macOS exposes no camera field of view
/// (`videoFieldOfView` is API_UNAVAILABLE on macOS), so there is no way to recover scale from the
/// image. the fitted gain is therefore only as accurate as the distance it was given
public struct CalibrationGeometry: Sendable, Codable, Equatable {
    public var viewingDistanceMM: Double
    public var offsets: [CalibrationTarget: TargetOffsetMM]

    public init(viewingDistanceMM: Double, offsets: [CalibrationTarget: TargetOffsetMM]) {
        self.viewingDistanceMM = viewingDistanceMM
        self.offsets = offsets
    }

    /// the angles the eyes genuinely turn through to land on a target, in degrees
    public func trueAngles(_ target: CalibrationTarget) -> (yaw: Double, pitch: Double)? {
        guard viewingDistanceMM > 0, let o = offsets[target] else { return nil }
        let degrees = 180.0 / Double.pi
        return (atan2(o.right, viewingDistanceMM) * degrees,
                atan2(o.up, viewingDistanceMM) * degrees)
    }

    public var isUsable: Bool {
        viewingDistanceMM >= 150 && viewingDistanceMM <= 2000
            && offsets[.left] != nil && offsets[.right] != nil && offsets[.down] != nil
    }
}

/// one observation taken while the user fixates the lens and rotates their head
///
/// true gaze is zero by construction for every one of these, so whatever the estimator reports is
/// entirely head contamination — which is what makes the slope identifiable
public struct HeadMotionSample: Codable, Sendable, Equatable {
    public var headYawDegrees: Double
    public var headPitchDegrees: Double
    public var rawYawDegrees: Double
    public var rawPitchDegrees: Double

    public init(headYawDegrees: Double, headPitchDegrees: Double,
                rawYawDegrees: Double, rawPitchDegrees: Double) {
        self.headYawDegrees = headYawDegrees
        self.headPitchDegrees = headPitchDegrees
        self.rawYawDegrees = rawYawDegrees
        self.rawPitchDegrees = rawPitchDegrees
    }
}

/// how much of the raw reading head rotation alone explains, as a 2x2 linear map in degrees
///
/// the cross terms are kept because they are not obviously zero: Vision reports pitch as positive
/// when nodding down while this codebase treats positive pitch as looking up, and yaw's
/// "counterclockwise" is ambiguous for a face, so every sign here is measured rather than assumed
public struct HeadCoupling: Sendable, Codable, Equatable {
    public var yawFromYaw: Double
    public var yawFromPitch: Double
    public var pitchFromYaw: Double
    public var pitchFromPitch: Double

    public init(yawFromYaw: Double, yawFromPitch: Double,
                pitchFromYaw: Double, pitchFromPitch: Double) {
        self.yawFromYaw = yawFromYaw
        self.yawFromPitch = yawFromPitch
        self.pitchFromYaw = pitchFromYaw
        self.pitchFromPitch = pitchFromPitch
    }

    public static let none = HeadCoupling(yawFromYaw: 0, yawFromPitch: 0,
                                          pitchFromYaw: 0, pitchFromPitch: 0)

    public func contribution(headYawDegrees: Double,
                             headPitchDegrees: Double) -> (yaw: Double, pitch: Double) {
        (yawFromYaw * headYawDegrees + yawFromPitch * headPitchDegrees,
         pitchFromYaw * headYawDegrees + pitchFromPitch * headPitchDegrees)
    }

    /// a slope this steep would subtract more than the head could plausibly contribute, and would
    /// amplify head-tracking noise straight into the correction
    public static let bounds = -2.0...2.0
    public var isPlausible: Bool {
        [yawFromYaw, yawFromPitch, pitchFromYaw, pitchFromPitch]
            .allSatisfy { $0.isFinite && Self.bounds.contains($0) }
    }
}

public enum HeadCouplingFit {
    public enum Failure: Error, Equatable, Sendable {
        case notEnoughSamples(have: Int, need: Int)
        /// without spread on an axis the slope is unidentifiable, and fitting it anyway would just
        /// be reading noise
        case insufficientHeadMotion(yawRange: Double, pitchRange: Double, need: Double)
        case implausibleCoupling
    }

    public static let minimumSamples = 40
    public static let minimumRangeDegrees = 12.0

    /// two independent least-squares regressions, each raw axis on both head axes plus an intercept
    ///
    /// the intercept is discarded: it is the same neutral bias the centre target already measures,
    /// and taking it from here instead would let a head-motion step overwrite a cleaner number
    public static func fit(_ samples: [HeadMotionSample],
                           minimumSamples: Int = minimumSamples,
                           minimumRange: Double = minimumRangeDegrees) throws -> HeadCoupling {
        guard samples.count >= minimumSamples else {
            throw Failure.notEnoughSamples(have: samples.count, need: minimumSamples)
        }
        let yaws = samples.map(\.headYawDegrees)
        let pitches = samples.map(\.headPitchDegrees)
        let yawRange = (yaws.max() ?? 0) - (yaws.min() ?? 0)
        let pitchRange = (pitches.max() ?? 0) - (pitches.min() ?? 0)
        guard yawRange >= minimumRange, pitchRange >= minimumRange else {
            throw Failure.insufficientHeadMotion(yawRange: yawRange, pitchRange: pitchRange,
                                                 need: minimumRange)
        }

        guard let yaw = solve(samples, target: \.rawYawDegrees),
              let pitch = solve(samples, target: \.rawPitchDegrees) else {
            throw Failure.implausibleCoupling
        }
        let coupling = HeadCoupling(yawFromYaw: yaw.a, yawFromPitch: yaw.b,
                                    pitchFromYaw: pitch.a, pitchFromPitch: pitch.b)
        guard coupling.isPlausible else { throw Failure.implausibleCoupling }
        return coupling
    }

    /// ordinary least squares for target = a·headYaw + b·headPitch + c, solved on the 2x2 normal
    /// equations of the mean-centred predictors so the intercept falls out
    private static func solve(_ samples: [HeadMotionSample],
                              target: KeyPath<HeadMotionSample, Double>) -> (a: Double, b: Double)? {
        let n = Double(samples.count)
        let mYaw = samples.reduce(0) { $0 + $1.headYawDegrees } / n
        let mPitch = samples.reduce(0) { $0 + $1.headPitchDegrees } / n
        let mTarget = samples.reduce(0) { $0 + $1[keyPath: target] } / n

        var sYY = 0.0, sPP = 0.0, sYP = 0.0, sYT = 0.0, sPT = 0.0
        for s in samples {
            let y = s.headYawDegrees - mYaw
            let p = s.headPitchDegrees - mPitch
            let t = s[keyPath: target] - mTarget
            sYY += y * y; sPP += p * p; sYP += y * p
            sYT += y * t; sPT += p * t
        }
        let determinant = sYY * sPP - sYP * sYP
        // a near-singular system means the head moved along one line only, so the two axes cannot
        // be told apart and the split between them would be arbitrary
        guard abs(determinant) > 1e-6 else { return nil }
        return ((sPP * sYT - sYP * sPT) / determinant,
                (sYY * sPT - sYP * sYT) / determinant)
    }
}
