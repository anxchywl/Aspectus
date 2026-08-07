import Foundation

/// rolling rate over a fixed time window, cheap enough to call every frame
///
/// reading is separate from ticking because the events that feed a meter can stop arriving — an
/// occluded window presents nothing — and a meter that only recomputes on a tick reports the last
/// rate it saw forever
public struct RateMeter: Sendable {
    private var times: [Double] = []
    private var firstEver: Double?
    private let window: Double

    public init(window: Double = 1.0) { self.window = window }

    @discardableResult
    public mutating func tick(at t: Double) -> Double {
        if firstEver == nil { firstEver = t }
        times.append(t)
        return rate(asOf: t)
    }

    /// the span is measured to `t` rather than to the newest sample, so a stall decays towards zero
    /// instead of latching at whatever was last seen
    public mutating func rate(asOf t: Double) -> Double {
        let cutoff = t - window
        while let first = times.first, first < cutoff { times.removeFirst() }
        guard times.count >= 2, let firstEver else { return 0 }

        // divided by the window rather than by the spread of the samples inside it. a source that
        // stalls and then catches up puts a whole window's frames into part of one, and dividing by
        // that part reports the burst instead of the rate — measured on screen as 58 fps of output
        // and 78 fps of capture from a camera that cannot exceed 30. until the meter has lived a
        // whole window there is not one to divide by, so its own age stands in
        let observed = min(window, t - firstEver)
        guard observed > 0 else { return 0 }
        return Double(times.count - 1) / observed
    }
}
