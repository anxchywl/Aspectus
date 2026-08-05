import Foundation

/// rolling rate over a fixed time window, cheap enough to call every frame
///
/// reading is separate from ticking because the events that feed a meter can stop arriving — an
/// occluded window presents nothing — and a meter that only recomputes on a tick reports the last
/// rate it saw forever
public struct RateMeter: Sendable {
    private var times: [Double] = []
    private let window: Double

    public init(window: Double = 1.0) { self.window = window }

    @discardableResult
    public mutating func tick(at t: Double) -> Double {
        times.append(t)
        return rate(asOf: t)
    }

    /// the span is measured to `t` rather than to the newest sample, so a stall decays towards zero
    /// instead of latching at whatever was last seen
    public mutating func rate(asOf t: Double) -> Double {
        let cutoff = t - window
        while let first = times.first, first < cutoff { times.removeFirst() }
        guard times.count >= 2, let first = times.first, t > first else { return 0 }
        return Double(times.count - 1) / (t - first)
    }
}
