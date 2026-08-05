import Foundation

/// caps publication at a fixed advertised rate, dropping the frames that would exceed it
///
/// a CMIO extension declares its frame duration before any frame exists, and forwards a faster
/// stream to hosts without complaint — so a camera that outruns the advertised rate has to be
/// paced on our side or hosts are told one cadence and given another
///
/// the decision is a credit balance rather than a fixed schedule: a source a little over the cap
/// then loses only its excess frames, where a schedule with the same tolerance would beat against
/// the source and drop far more than the excess
public struct PublishPacer: Sendable {
    private let interval: Double
    private var credit: Double = 0
    private var last: Double?

    public init(frameRate: Double) { interval = frameRate > 0 ? 1 / frameRate : 0 }

    /// the shortfall a frame may still be published on; without it the jitter of a source running
    /// at exactly the advertised rate reads as excess rate and half the frames are dropped
    private static let tolerance = 0.1

    public mutating func shouldPublish(at t: Double) -> Bool {
        guard interval > 0 else { return true }
        // a reopened camera can restart its presentation clock in the past, which must not stall
        // publishing until the old timeline catches up
        guard let previous = last, t >= previous else {
            last = t
            credit = 0
            return true
        }
        last = t
        credit += (t - previous) / interval
        guard credit >= 1 - Self.tolerance else { return false }
        // capped after spending, so a long gap cannot bank credit and release a burst
        credit = min(credit - 1, 1)
        return true
    }
}
