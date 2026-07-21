import Foundation

/// per-frame blend weight for the corrected eye region, 0 means original passes through
/// hysteresis and slew-limiting prevent flicker; correction fades out past the trusted angle
public struct CorrectionGate: Sendable {
    public struct Config: Sendable {
        public var enterConfidence: Double
        public var exitConfidence: Double
        public var maxCorrectionDegrees: Double
        public var slewPerSecond: Double
        public var maxStrength: Double
        public init(enterConfidence: Double = 0.6,
                    exitConfidence: Double = 0.4,
                    maxCorrectionDegrees: Double = 18.0,
                    slewPerSecond: Double = 6.0,
                    maxStrength: Double = 1.0) {
            self.enterConfidence = enterConfidence
            self.exitConfidence = exitConfidence
            self.maxCorrectionDegrees = maxCorrectionDegrees
            self.slewPerSecond = slewPerSecond
            self.maxStrength = maxStrength
        }
    }

    public private(set) var config: Config
    private var engaged = false
    private var weight: Double = 0
    private var lastTime: Double?

    public init(config: Config = Config()) { self.config = config }

    public var currentWeight: Double { weight }
    public var isEngaged: Bool { engaged }

    public mutating func update(confidence: Double,
                                requestedCorrectionDegrees: Double,
                                t: Double) -> Double {
        if engaged {
            if confidence < config.exitConfidence { engaged = false }
        } else {
            if confidence >= config.enterConfidence { engaged = true }
        }

        let angle = abs(requestedCorrectionDegrees)
        let angleFactor: Double
        if angle <= config.maxCorrectionDegrees {
            angleFactor = 1.0
        } else {
            // linear decay over a 6° guard band past the limit, then hard zero
            let over = angle - config.maxCorrectionDegrees
            angleFactor = max(0.0, 1.0 - over / 6.0)
        }

        let target = (engaged ? config.maxStrength : 0.0) * angleFactor

        // first frame only sets the time baseline, else engaging pops to full strength
        defer { lastTime = t }
        guard let lt = lastTime, t > lt else { return weight }
        let dt = t - lt
        let maxDelta = config.slewPerSecond * dt
        let delta = target - weight
        weight += max(-maxDelta, min(maxDelta, delta))
        weight = max(0.0, min(1.0, weight))
        return weight
    }

    // weight still slews down afterwards so the fallback fades in rather than pops
    public mutating func forceFallback() { engaged = false }
}
