import Foundation

/// why one frame was not usable as a calibration sample, shown to the user so a stalled step is
/// self-explaining rather than silently never filling up
public enum CalibrationRejection: String, Sendable, Equatable, CaseIterable {
    case noFace
    case noEstimate
    case eyesClosed
    case lowConfidence
    case headPose
    case headMotion
    case motionBlur
    case eyeDisagreement
    case timestampDiscontinuity
}

extension CalibrationRejection: CustomStringConvertible {
    public var description: String {
        switch self {
        case .noFace: return "no face detected"
        case .noEstimate: return "eyes not readable"
        case .eyesClosed: return "keep your eyes open"
        case .lowConfidence: return "tracking not confident"
        case .headPose: return "face the camera more squarely"
        case .headMotion: return "hold your head still"
        case .motionBlur: return "hold still, too much motion"
        case .eyeDisagreement: return "the two eyes disagree"
        case .timestampDiscontinuity: return "dropped frames, resuming"
        }
    }
}

/// collects validated samples for each calibration target in turn
///
/// every threshold here is deliberately stricter than the runtime gate: a bad calibration is worse
/// than none at all, because it biases every later frame, whereas a bad runtime frame only costs
/// that one frame's correction
public struct CalibrationSession: Sendable {
    public struct Config: Sendable {
        public var samplesPerTarget: Int
        public var minOpenness: Double
        public var minConfidence: Double
        public var maxHeadPoseDegrees: Double
        public var maxHeadMotionDegreesPerSecond: Double
        /// proxy for motion blur: we have no blur metric, so pupil travel per second stands in
        public var maxPupilSpeedPerSecond: Double
        public var maxEyeDisagreementDegrees: Double
        public var maxFrameGap: Double
        /// head rotation is the point of the last step, so its own gate has to be far wider than
        /// the one that protects the five fixation targets
        public var headMotionMaxPoseDegrees: Double
        public var headMotionSamples: Int
        /// grace period after each target change, so the eye movement itself is never sampled and
        /// the user has time to find the new target before anything is measured
        public var settleSeconds: Double

        public init(samplesPerTarget: Int = 60,
                    minOpenness: Double = 0.5,
                    minConfidence: Double = 0.6,
                    maxHeadPoseDegrees: Double = 15.0,
                    maxHeadMotionDegreesPerSecond: Double = 20.0,
                    maxPupilSpeedPerSecond: Double = 0.15,
                    maxEyeDisagreementDegrees: Double = 8.0,
                    maxFrameGap: Double = 0.25,
                    settleSeconds: Double = 2.0,
                    headMotionMaxPoseDegrees: Double = 35.0,
                    headMotionSamples: Int = 150) {
            self.settleSeconds = settleSeconds
            self.headMotionMaxPoseDegrees = headMotionMaxPoseDegrees
            self.headMotionSamples = headMotionSamples
            self.samplesPerTarget = samplesPerTarget
            self.minOpenness = minOpenness
            self.minConfidence = minConfidence
            self.maxHeadPoseDegrees = maxHeadPoseDegrees
            self.maxHeadMotionDegreesPerSecond = maxHeadMotionDegreesPerSecond
            self.maxPupilSpeedPerSecond = maxPupilSpeedPerSecond
            self.maxEyeDisagreementDegrees = maxEyeDisagreementDegrees
            self.maxFrameGap = maxFrameGap
        }
    }

    /// the five fixation targets first, then the head-rotation sweep that makes head contamination
    /// identifiable; the sweep is optional and skipping it simply means no compensation
    public enum Phase: Sendable, Equatable {
        case targets
        case headMotion
        case finished
    }

    public enum Outcome: Sendable, Equatable {
        /// still inside the grace period after a target change, nothing is being measured yet
        case settling(remainingSeconds: Double)
        case accepted
        case rejected(CalibrationRejection)
        case targetComplete(CalibrationTarget)
        /// accepted into the head-motion sweep rather than into a fixation target
        case acceptedHeadMotion
        case finished
    }

    /// what one target actually read, for live feedback and for explaining a failed fit
    public struct TargetMeans: Sendable, Equatable {
        public var yawDegrees: Double
        public var pitchDegrees: Double
        public var count: Int
        /// raw pupil displacement behind the angles, averaged over both eyes and target samples
        public var offsetX: Double = 0
        public var offsetY: Double = 0
    }

    public let config: Config
    public private(set) var target: CalibrationTarget
    public private(set) var samples: [CalibrationSample] = []
    public private(set) var headMotionSamples: [HeadMotionSample] = []
    public private(set) var phase: Phase = .targets
    public private(set) var lastRejection: CalibrationRejection?
    /// the most recent verdict, so a UI polling on its own timer can render the live state without
    /// having to be on the frame path itself
    public private(set) var lastOutcome: Outcome?
    public private(set) var isFinished = false

    private var lastHeadPose: HeadPose?
    private var lastPupils: (left: NormPoint, right: NormPoint)?
    private var lastTime: Double?
    private var targetStarted: Double?

    public init(config: Config = Config()) {
        self.config = config
        self.target = CalibrationTarget.allCases[0]
    }

    public var acceptedForTarget: Int { samples.filter { $0.target == target }.count }

    /// how much of the head sweep has been covered, on the axis that has moved least
    public var headMotionProgress: Double {
        guard !headMotionSamples.isEmpty else { return 0 }
        let count = min(1.0, Double(headMotionSamples.count) / Double(config.headMotionSamples))
        let yaws = headMotionSamples.map(\.headYawDegrees)
        let pitches = headMotionSamples.map(\.headPitchDegrees)
        let yawSpan = ((yaws.max() ?? 0) - (yaws.min() ?? 0)) / HeadCouplingFit.minimumRangeDegrees
        let pitchSpan = ((pitches.max() ?? 0) - (pitches.min() ?? 0)) / HeadCouplingFit.minimumRangeDegrees
        return min(count, min(1.0, yawSpan), min(1.0, pitchSpan))
    }

    public var headMotionSpan: (yaw: Double, pitch: Double) {
        let yaws = headMotionSamples.map(\.headYawDegrees)
        let pitches = headMotionSamples.map(\.headPitchDegrees)
        return ((yaws.max() ?? 0) - (yaws.min() ?? 0), (pitches.max() ?? 0) - (pitches.min() ?? 0))
    }
    public var progressForTarget: Double {
        config.samplesPerTarget == 0 ? 1
            : min(1, Double(acceptedForTarget) / Double(config.samplesPerTarget))
    }
    public var overallProgress: Double {
        let total = config.samplesPerTarget * CalibrationTarget.allCases.count
        return total == 0 ? 1 : min(1, Double(samples.count) / Double(total))
    }

    public func means(for target: CalibrationTarget) -> TargetMeans? {
        let matching = samples.filter { $0.target == target }
        guard !matching.isEmpty else { return nil }
        let n = Double(matching.count)
        return TargetMeans(yawDegrees: matching.reduce(0) { $0 + $1.yawDegrees } / n,
                           pitchDegrees: matching.reduce(0) { $0 + $1.pitchDegrees } / n,
                           count: matching.count,
                           offsetX: matching.reduce(0) { $0 + $1.offsetX } / n,
                           offsetY: matching.reduce(0) { $0 + $1.offsetY } / n)
    }

    /// every target's reading so far, which is what makes a separation failure explainable
    public var allMeans: [CalibrationTarget: TargetMeans] {
        var out: [CalibrationTarget: TargetMeans] = [:]
        for target in CalibrationTarget.allCases { out[target] = means(for: target) }
        return out
    }

    /// offers one tracked frame as a candidate sample
    ///
    /// motion is judged against the previous *offered* frame, not the previous accepted one, so a
    /// long run of rejections cannot make a fast movement look slow
    public mutating func offer(_ tracking: TrackingResult?,
                               imageAspect: Double,
                               tuning: EyeWarpTuning = .init(),
                               t: Double) -> Outcome {
        let outcome = evaluate(tracking, imageAspect: imageAspect, tuning: tuning, t: t)
        lastOutcome = outcome
        return outcome
    }

    private mutating func evaluate(_ tracking: TrackingResult?,
                                   imageAspect: Double,
                                   tuning: EyeWarpTuning,
                                   t: Double) -> Outcome {
        guard !isFinished else { return .finished }
        if targetStarted == nil { targetStarted = t }

        let previousTime = lastTime
        let previousPose = lastHeadPose
        let previousPupils = lastPupils
        lastTime = t

        guard let tracking else {
            lastHeadPose = nil
            lastPupils = nil
            return reject(.noFace)
        }

        lastHeadPose = tracking.headPose
        lastPupils = (tracking.leftEye.pupilCenter, tracking.rightEye.pupilCenter)

        if let previousTime, t <= previousTime || t - previousTime > config.maxFrameGap {
            return reject(.timestampDiscontinuity)
        }

        // the saccade to a new target is itself a fast eye movement, so measuring during it would
        // mix the journey into the destination
        if let started = targetStarted {
            let remaining = config.settleSeconds - (t - started)
            if remaining > 0 {
                lastRejection = nil
                return .settling(remainingSeconds: remaining)
            }
        }

        let degrees = 180.0 / Double.pi
        guard tracking.leftEye.openness >= config.minOpenness,
              tracking.rightEye.openness >= config.minOpenness else { return reject(.eyesClosed) }
        guard tracking.confidence >= config.minConfidence else { return reject(.lowConfidence) }

        if phase == .headMotion { return offerHeadMotion(tracking, imageAspect: imageAspect,
                                                         tuning: tuning, degrees: degrees) }

        guard max(abs(tracking.headPose.yaw), abs(tracking.headPose.pitch)) * degrees
                <= config.maxHeadPoseDegrees else { return reject(.headPose) }

        if let previousTime, let previousPose {
            let dt = t - previousTime
            if dt > 0 {
                let dYaw = abs(tracking.headPose.yaw - previousPose.yaw) * degrees
                let dPitch = abs(tracking.headPose.pitch - previousPose.pitch) * degrees
                let dRoll = abs(tracking.headPose.roll - previousPose.roll) * degrees
                let speed = max(dYaw, max(dPitch, dRoll)) / dt
                if speed > config.maxHeadMotionDegreesPerSecond { return reject(.headMotion) }
            }
        }

        if let previousTime, let previousPupils {
            let dt = t - previousTime
            if dt > 0 {
                let l = hypot(tracking.leftEye.pupilCenter.x - previousPupils.left.x,
                              tracking.leftEye.pupilCenter.y - previousPupils.left.y)
                let r = hypot(tracking.rightEye.pupilCenter.x - previousPupils.right.x,
                              tracking.rightEye.pupilCenter.y - previousPupils.right.y)
                if max(l, r) / dt > config.maxPupilSpeedPerSecond { return reject(.motionBlur) }
            }
        }

        guard let left = GazeGeometry.eyeAngles(tracking.leftEye, imageAspect: imageAspect,
                                                tuning: tuning),
              let right = GazeGeometry.eyeAngles(tracking.rightEye, imageAspect: imageAspect,
                                                 tuning: tuning),
              let combined = GazeGeometry.estimate(tracking, imageAspect: imageAspect,
                                                   tuning: tuning) else {
            return reject(.noEstimate)
        }

        let yawGap = abs(left.yaw - right.yaw) * degrees
        let pitchGap = abs(left.pitch - right.pitch) * degrees
        guard max(yawGap, pitchGap) <= config.maxEyeDisagreementDegrees else {
            return reject(.eyeDisagreement)
        }

        samples.append(CalibrationSample(
            target: target,
            yawDegrees: combined.yaw * degrees,
            pitchDegrees: combined.pitch * degrees,
            leftYawDegrees: left.yaw * degrees,
            leftPitchDegrees: left.pitch * degrees,
            rightYawDegrees: right.yaw * degrees,
            rightPitchDegrees: right.pitch * degrees,
            headYawDegrees: tracking.headPose.yaw * degrees,
            headPitchDegrees: tracking.headPose.pitch * degrees,
            headRollDegrees: tracking.headPose.roll * degrees,
            confidence: combined.confidence,
            leftOffsetX: tracking.leftEye.pupilOffset.x,
            leftOffsetY: tracking.leftEye.pupilOffset.y,
            rightOffsetX: tracking.rightEye.pupilOffset.x,
            rightOffsetY: tracking.rightEye.pupilOffset.y))
        lastRejection = nil

        guard acceptedForTarget >= config.samplesPerTarget else { return .accepted }

        let completed = target
        let all = CalibrationTarget.allCases
        if let index = all.firstIndex(of: target), index + 1 < all.count {
            target = all[index + 1]
            // the next target needs a fresh motion baseline and a fresh settle window, the user
            // is about to look somewhere else
            lastHeadPose = nil
            lastPupils = nil
            lastTime = nil
            targetStarted = nil
            return .targetComplete(completed)
        }
        // the fixation targets are done; the sweep follows, and the head gate widens for it
        phase = .headMotion
        lastHeadPose = nil
        lastPupils = nil
        lastTime = nil
        targetStarted = nil
        return .targetComplete(completed)
    }

    /// the sweep: the user holds their gaze on the lens while turning their head, so true gaze is
    /// zero on every accepted frame and the raw reading is pure contamination
    ///
    /// head motion is expected here, so the motion gates that protect the fixation targets are not
    /// applied — only the openness, confidence and estimate checks, plus a wide pose limit
    private mutating func offerHeadMotion(_ tracking: TrackingResult,
                                          imageAspect: Double,
                                          tuning: EyeWarpTuning,
                                          degrees: Double) -> Outcome {
        guard tracking.headPoseAvailable else { return reject(.headPose) }
        guard max(abs(tracking.headPose.yaw), abs(tracking.headPose.pitch)) * degrees
                <= config.headMotionMaxPoseDegrees else { return reject(.headPose) }
        guard let gaze = GazeGeometry.estimate(tracking, imageAspect: imageAspect,
                                               tuning: tuning) else { return reject(.noEstimate) }

        headMotionSamples.append(HeadMotionSample(
            headYawDegrees: tracking.headPose.yaw * degrees,
            headPitchDegrees: tracking.headPose.pitch * degrees,
            rawYawDegrees: gaze.yaw * degrees,
            rawPitchDegrees: gaze.pitch * degrees))
        lastRejection = nil

        let span = headMotionSpan
        guard headMotionSamples.count >= config.headMotionSamples,
              span.yaw >= HeadCouplingFit.minimumRangeDegrees,
              span.pitch >= HeadCouplingFit.minimumRangeDegrees else { return .acceptedHeadMotion }

        phase = .finished
        isFinished = true
        return .finished
    }

    /// ends the sweep early, keeping the five-target calibration and giving up compensation
    public mutating func skipHeadMotion() {
        guard phase == .headMotion else { return }
        headMotionSamples = []
        phase = .finished
        isFinished = true
    }

    /// discards everything and starts over at the first target
    public mutating func reset() {
        samples = []
        headMotionSamples = []
        phase = .targets
        target = CalibrationTarget.allCases[0]
        lastRejection = nil
        lastOutcome = nil
        isFinished = false
        lastHeadPose = nil
        lastPupils = nil
        lastTime = nil
        targetStarted = nil
    }

    /// the sweep is optional: if it was skipped or is unusable, the calibration is still fitted,
    /// just without head compensation
    public func fit(geometry: CalibrationGeometry? = nil,
                    now: Date = Date()) throws -> GazeCalibration {
        let coupling = try? HeadCouplingFit.fit(headMotionSamples)
        return try CalibrationFit.fit(samples, geometry: geometry,
                                      headCoupling: coupling, now: now)
    }

    /// surfaced separately so the UI can say why compensation was not fitted
    public func headCouplingFailure() -> HeadCouplingFit.Failure? {
        do { _ = try HeadCouplingFit.fit(headMotionSamples); return nil }
        catch let failure as HeadCouplingFit.Failure { return failure }
        catch { return nil }
    }

    private mutating func reject(_ reason: CalibrationRejection) -> Outcome {
        lastRejection = reason
        return .rejected(reason)
    }
}
