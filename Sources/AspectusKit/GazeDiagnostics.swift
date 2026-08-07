import Foundation

/// why a frame was not corrected, so a zero blend names the gate that fired instead of being a
/// silent passthrough that looks identical to correction having no effect
public enum FallbackReason: String, Sendable, Equatable, CaseIterable {
    case none
    case disabled              // correction switched off by the user
    case noTracking            // no face, or landmarks not available yet
    case headPose              // head turned past the trusted limit
    case eyesClosed            // blink or squint below the openness floor
    case degenerateGeometry    // eye region too small to be a real aperture
    case lowConfidence         // gate not engaged
    case angleLimit            // requested redirection outside the trusted angle
    case engaging              // engaged but the blend is still slewing up from zero
    case correctorFailed       // the corrector returned its input frame
}

/// min/mean/max over a run, enough to answer "does this value ever reach the threshold"
/// without keeping every sample; percentiles live in StageMetrics where the ring buffer already is
public struct ValueStats: Sendable, Equatable {
    public private(set) var count: Int = 0
    private var sum: Double = 0
    private var low: Double = .infinity
    private var high: Double = -.infinity

    public init() {}

    public var mean: Double { count == 0 ? 0 : sum / Double(count) }
    public var minimum: Double { count == 0 ? 0 : low }
    public var maximum: Double { count == 0 ? 0 : high }

    public mutating func add(_ v: Double) {
        guard v.isFinite else { return }
        count += 1
        sum += v
        low = Swift.min(low, v)
        high = Swift.max(high, v)
    }

    public mutating func reset() { self = ValueStats() }
}

/// one eye's tracked geometry as the diagnostics see it
public struct EyeSample: Sendable, Equatable {
    public var region: NormRect
    public var pupil: NormPoint
    public var pupilOffset: NormPoint
    public var cornerMidpointY: Double?
    public var openness: Double
    public var source: PupilSource
    public var pupilPointCount: Int

    public init(_ e: EyeObservation) {
        region = e.region
        pupil = e.pupilCenter
        pupilOffset = e.pupilOffset
        cornerMidpointY = e.cornerMidpointY
        openness = e.openness
        source = e.pupilSource
        pupilPointCount = e.pupilPointCount
    }
}

/// everything the gaze path decided for one frame, in degrees so it reads without conversion
///
/// calibrated angles are optional and stay nil until a calibration exists — reporting the raw
/// value in both slots would make an uncalibrated run look calibrated
public struct GazeSample: Sendable {
    public var frameID: FrameID
    public var left: EyeSample?
    public var right: EyeSample?

    public var headYawDegrees: Double
    public var headPitchDegrees: Double
    public var headRollDegrees: Double
    public var faceConfidence: Double
    /// false when the tracker had no head yaw/pitch to report
    public var headPoseAvailable: Bool

    public var rawYawDegrees: Double
    public var rawPitchDegrees: Double
    /// nil when no estimate was attempted at all, so switching correction off cannot drag the
    /// measured confidence distribution down to zero
    public var gazeConfidence: Double?
    public var calibratedYawDegrees: Double?
    public var calibratedPitchDegrees: Double?

    public var requestedYawDegrees: Double
    public var requestedPitchDegrees: Double
    public var requestedMagnitudeDegrees: Double
    public var angleFactor: Double
    public var blendStrength: Double

    /// capture-time gap between the frame being corrected and the frame its landmarks came from
    public var correctionAgeMs: Double
    public var irisTravelPixels: Double
    public var fallback: FallbackReason

    public init(frameID: FrameID,
                left: EyeSample? = nil,
                right: EyeSample? = nil,
                headYawDegrees: Double = 0,
                headPitchDegrees: Double = 0,
                headRollDegrees: Double = 0,
                faceConfidence: Double = 0,
                headPoseAvailable: Bool = true,
                rawYawDegrees: Double = 0,
                rawPitchDegrees: Double = 0,
                gazeConfidence: Double? = nil,
                calibratedYawDegrees: Double? = nil,
                calibratedPitchDegrees: Double? = nil,
                requestedYawDegrees: Double = 0,
                requestedPitchDegrees: Double = 0,
                requestedMagnitudeDegrees: Double = 0,
                angleFactor: Double = 0,
                blendStrength: Double = 0,
                correctionAgeMs: Double = 0,
                irisTravelPixels: Double = 0,
                fallback: FallbackReason = .none) {
        self.frameID = frameID
        self.left = left
        self.right = right
        self.headYawDegrees = headYawDegrees
        self.headPitchDegrees = headPitchDegrees
        self.headRollDegrees = headRollDegrees
        self.faceConfidence = faceConfidence
        self.headPoseAvailable = headPoseAvailable
        self.rawYawDegrees = rawYawDegrees
        self.rawPitchDegrees = rawPitchDegrees
        self.gazeConfidence = gazeConfidence
        self.calibratedYawDegrees = calibratedYawDegrees
        self.calibratedPitchDegrees = calibratedPitchDegrees
        self.requestedYawDegrees = requestedYawDegrees
        self.requestedPitchDegrees = requestedPitchDegrees
        self.requestedMagnitudeDegrees = requestedMagnitudeDegrees
        self.angleFactor = angleFactor
        self.blendStrength = blendStrength
        self.correctionAgeMs = correctionAgeMs
        self.irisTravelPixels = irisTravelPixels
        self.fallback = fallback
    }
}

/// accumulates per-frame gaze samples off the main actor so the HUD can read them on its timer
///
/// this exists because the diagnostics that matter are distributions, not instants: whether the
/// confidence gate ever actually fires, how often a real pupil landmark is available, and how
/// large the eyelid-induced vertical offset is on this particular face
public final class DiagnosticsCollector: @unchecked Sendable {
    public struct Snapshot: Sendable {
        public let latest: GazeSample?
        public let frames: UInt64
        public let faceConfidence: ValueStats
        public let gazeConfidence: ValueStats
        /// pooled across both eyes, the phantom-pitch term from the eyelid bias
        public let verticalPupilOffset: ValueStats
        public let horizontalPupilOffset: ValueStats
        public let correctionAgeMeanMs: Double
        public let correctionAgeP95Ms: Double
        /// share of eye observations whose pupil came from a real Vision landmark, 0…1
        public let visionPupilShare: Double
        /// share of tracked frames on which the tracker actually had a head yaw and pitch
        public let headPoseShare: Double
        public let fallbackCounts: [FallbackReason: Int]

        public static let empty = Snapshot(latest: nil, frames: 0,
                                           faceConfidence: ValueStats(),
                                           gazeConfidence: ValueStats(),
                                           verticalPupilOffset: ValueStats(),
                                           horizontalPupilOffset: ValueStats(),
                                           correctionAgeMeanMs: 0, correctionAgeP95Ms: 0,
                                           visionPupilShare: 0, headPoseShare: 0,
                                           fallbackCounts: [:])

        /// the most frequent reason correction did not run, which is the one worth fixing
        public var dominantFallback: FallbackReason {
            fallbackCounts.filter { $0.key != .none }
                .max { $0.value < $1.value }?.key ?? .none
        }

        public var correctedFrames: Int { fallbackCounts[.none] ?? 0 }
    }

    private let lock = NSLock()
    /// replaced wholesale on reset, so it is guarded by our own lock rather than only its own
    private var age = StageMetrics(name: "correction age", window: 240)

    private var latest: GazeSample?
    private var frames: UInt64 = 0
    private var faceConfidence = ValueStats()
    private var gazeConfidence = ValueStats()
    private var verticalOffset = ValueStats()
    private var horizontalOffset = ValueStats()
    private var eyeObservations = 0
    private var visionPupilObservations = 0
    private var trackedFrames = 0
    private var headPoseFrames = 0
    private var fallbackCounts: [FallbackReason: Int] = [:]

    public init() {}

    public func record(_ sample: GazeSample) {
        lock.lock()
        defer { lock.unlock() }
        // age is only meaningful once there is a previous frame to be stale against
        if sample.correctionAgeMs > 0 { age.record(ms: sample.correctionAgeMs) }
        latest = sample
        frames &+= 1
        fallbackCounts[sample.fallback, default: 0] += 1
        // confidences are only meaningful on frames that had a face, otherwise the zeros from
        // untracked frames drag the distribution down and hide what the gate actually sees
        if sample.left != nil || sample.right != nil {
            faceConfidence.add(sample.faceConfidence)
            trackedFrames += 1
            if sample.headPoseAvailable { headPoseFrames += 1 }
        }
        if let c = sample.gazeConfidence { gazeConfidence.add(c) }
        for eye in [sample.left, sample.right].compactMap({ $0 }) {
            eyeObservations += 1
            if eye.source == .visionLandmark { visionPupilObservations += 1 }
            verticalOffset.add(eye.pupilOffset.y)
            horizontalOffset.add(eye.pupilOffset.x)
        }
    }

    public func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        let a = age.snapshot()
        return Snapshot(latest: latest,
                        frames: frames,
                        faceConfidence: faceConfidence,
                        gazeConfidence: gazeConfidence,
                        verticalPupilOffset: verticalOffset,
                        horizontalPupilOffset: horizontalOffset,
                        correctionAgeMeanMs: a.meanMs,
                        correctionAgeP95Ms: a.p95Ms,
                        visionPupilShare: eyeObservations == 0
                            ? 0 : Double(visionPupilObservations) / Double(eyeObservations),
                        headPoseShare: trackedFrames == 0
                            ? 0 : Double(headPoseFrames) / Double(trackedFrames),
                        fallbackCounts: fallbackCounts)
    }

    /// a stop/start cycle must not report the previous session's distributions
    public func reset() {
        lock.lock()
        defer { lock.unlock() }
        latest = nil
        frames = 0
        age = StageMetrics(name: "correction age", window: 240)
        faceConfidence.reset()
        gazeConfidence.reset()
        verticalOffset.reset()
        horizontalOffset.reset()
        eyeObservations = 0
        visionPupilObservations = 0
        trackedFrames = 0
        headPoseFrames = 0
        fallbackCounts = [:]
    }
}
