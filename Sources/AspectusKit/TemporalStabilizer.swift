import Foundation

/// low-lag temporal smoothing for the tracking geometry, so a still face produces a still warp
///
/// openness is deliberately never filtered. a blink is a fast transient, and smoothing it would
/// both slur the blink itself and delay the gate that suppresses correction during one
///
/// history is dropped on tracking loss or a timestamp discontinuity, because blending a fresh
/// observation against stale state is exactly what makes recovery pop
public struct TemporalStabilizer: Sendable {
    public struct Config: Sendable {
        public var landmark: OneEuroTuning
        public var gaze: OneEuroTuning
        /// a frame gap longer than this means the stream jumped and history no longer applies
        public var maxFrameGap: Double

        public init(landmark: OneEuroTuning = .init(minCutoff: 1.2, beta: 0.02),
                    gaze: OneEuroTuning = .init(minCutoff: 0.8, beta: 0.01),
                    maxFrameGap: Double = 0.25) {
            self.landmark = landmark
            self.gaze = gaze
            self.maxFrameGap = maxFrameGap
        }
    }

    private struct RectFilter: Sendable {
        var origin: OneEuroPointFilter
        var size: OneEuroPointFilter

        init(_ t: OneEuroTuning) {
            origin = OneEuroPointFilter(minCutoff: t.minCutoff, beta: t.beta, dCutoff: t.dCutoff)
            size = OneEuroPointFilter(minCutoff: t.minCutoff, beta: t.beta, dCutoff: t.dCutoff)
        }

        mutating func filter(_ r: NormRect, t: Double) -> NormRect {
            let o = origin.filter(x: r.x, y: r.y, t: t)
            let s = size.filter(x: r.width, y: r.height, t: t)
            return NormRect(x: o.x, y: o.y, width: s.x, height: s.y)
        }

        mutating func reset() { origin.reset(); size.reset() }
    }

    private struct EyeFilter: Sendable {
        var region: RectFilter
        var pupil: OneEuroPointFilter

        init(_ t: OneEuroTuning) {
            region = RectFilter(t)
            pupil = OneEuroPointFilter(minCutoff: t.minCutoff, beta: t.beta, dCutoff: t.dCutoff)
        }

        mutating func filter(_ e: EyeObservation, t: Double) -> EyeObservation {
            let p = pupil.filter(x: e.pupilCenter.x, y: e.pupilCenter.y, t: t)
            return EyeObservation(region: region.filter(e.region, t: t),
                                  pupilCenter: NormPoint(x: p.x, y: p.y),
                                  openness: e.openness)
        }

        mutating func reset() { region.reset(); pupil.reset() }
    }

    public private(set) var config: Config

    private var face: RectFilter
    private var left: EyeFilter
    private var right: EyeFilter
    private var yaw: OneEuroFilter
    private var pitch: OneEuroFilter
    private var roll: OneEuroFilter
    private var gazeYaw: OneEuroFilter
    private var gazePitch: OneEuroFilter
    private var lastTime: Double?

    public init(config: Config = Config()) {
        self.config = config
        face = RectFilter(config.landmark)
        left = EyeFilter(config.landmark)
        right = EyeFilter(config.landmark)
        yaw = OneEuroFilter(minCutoff: config.landmark.minCutoff, beta: config.landmark.beta)
        pitch = OneEuroFilter(minCutoff: config.landmark.minCutoff, beta: config.landmark.beta)
        roll = OneEuroFilter(minCutoff: config.landmark.minCutoff, beta: config.landmark.beta)
        gazeYaw = OneEuroFilter(minCutoff: config.gaze.minCutoff, beta: config.gaze.beta)
        gazePitch = OneEuroFilter(minCutoff: config.gaze.minCutoff, beta: config.gaze.beta)
    }

    /// true when the incoming timestamp cannot be reconciled with the filter history
    public func isDiscontinuous(at t: Double) -> Bool {
        guard let last = lastTime else { return false }
        return t <= last || t - last > config.maxFrameGap
    }

    public mutating func stabilize(_ tracking: TrackingResult, t: Double) -> TrackingResult {
        if isDiscontinuous(at: t) { reset() }
        lastTime = t
        return TrackingResult(faceBounds: face.filter(tracking.faceBounds, t: t),
                              leftEye: left.filter(tracking.leftEye, t: t),
                              rightEye: right.filter(tracking.rightEye, t: t),
                              headPose: HeadPose(yaw: yaw.filter(tracking.headPose.yaw, t: t),
                                                 pitch: pitch.filter(tracking.headPose.pitch, t: t),
                                                 roll: roll.filter(tracking.headPose.roll, t: t)),
                              confidence: tracking.confidence)
    }

    /// gaze is filtered separately because it is derived, and its own cutoff is lower
    public mutating func stabilize(_ gaze: GazeEstimate, t: Double) -> GazeEstimate {
        GazeEstimate(yaw: gazeYaw.filter(gaze.yaw, t: t),
                     pitch: gazePitch.filter(gaze.pitch, t: t),
                     confidence: gaze.confidence)
    }

    /// the next sample passes through unsmoothed instead of blending with stale history
    public mutating func reset() {
        face.reset()
        left.reset()
        right.reset()
        yaw.reset()
        pitch.reset()
        roll.reset()
        gazeYaw.reset()
        gazePitch.reset()
        lastTime = nil
    }
}
