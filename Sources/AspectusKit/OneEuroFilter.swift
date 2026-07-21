import Foundation

/// 1€ filter (Casiez, Roussel, Vogel 2012), adaptive smoothing that trades jitter against lag
/// by signal speed, so landmarks/gaze/strength settle without the visible lag of a fixed low-pass
public struct OneEuroFilter: Sendable {
    public var minCutoff: Double   // lower smooths more at rest
    public var beta: Double        // higher cuts lag during fast motion
    public var dCutoff: Double     // cutoff for the derivative estimate

    private var xPrev: Double?
    private var dxPrev: Double = 0
    private var tPrev: Double?

    public init(minCutoff: Double = 1.0, beta: Double = 0.007, dCutoff: Double = 1.0) {
        self.minCutoff = minCutoff
        self.beta = beta
        self.dCutoff = dCutoff
    }

    private static func alpha(cutoff: Double, dt: Double) -> Double {
        let tau = 1.0 / (2.0 * Double.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
    }

    public mutating func filter(_ x: Double, t: Double) -> Double {
        defer { tPrev = t }
        guard let xp = xPrev, let tp = tPrev, t > tp else {
            xPrev = x
            dxPrev = 0
            return x
        }
        let dt = t - tp
        let dx = (x - xp) / dt
        let aD = Self.alpha(cutoff: dCutoff, dt: dt)
        let dxHat = aD * dx + (1 - aD) * dxPrev
        let cutoff = minCutoff + beta * abs(dxHat)
        let a = Self.alpha(cutoff: cutoff, dt: dt)
        let xHat = a * x + (1 - a) * xp
        xPrev = xHat
        dxPrev = dxHat
        return xHat
    }

    /// next sample passes through unsmoothed rather than blending with stale history
    public mutating func reset() {
        xPrev = nil
        dxPrev = 0
        tPrev = nil
    }
}

/// filters an (x, y) point with two independent 1€ filters
public struct OneEuroPointFilter: Sendable {
    private var fx: OneEuroFilter
    private var fy: OneEuroFilter
    public init(minCutoff: Double = 1.0, beta: Double = 0.007, dCutoff: Double = 1.0) {
        fx = OneEuroFilter(minCutoff: minCutoff, beta: beta, dCutoff: dCutoff)
        fy = OneEuroFilter(minCutoff: minCutoff, beta: beta, dCutoff: dCutoff)
    }
    public mutating func filter(x: Double, y: Double, t: Double) -> (x: Double, y: Double) {
        (fx.filter(x, t: t), fy.filter(y, t: t))
    }
    public mutating func reset() { fx.reset(); fy.reset() }
}
