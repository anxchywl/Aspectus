import Foundation

/// rolling fps over a fixed time window, cheap enough to call every frame
struct RateMeter {
    private var times: [Double] = []
    private let window: Double

    init(window: Double = 1.0) { self.window = window }

    mutating func tick(at t: Double) -> Double {
        times.append(t)
        let cutoff = t - window
        while let first = times.first, first < cutoff { times.removeFirst() }
        guard times.count >= 2, let span = times.last.map({ $0 - times[0] }), span > 0 else {
            return 0
        }
        return Double(times.count - 1) / span
    }
}
