import Foundation

/// monotonic per-frame id assigned at the source, lets stages detect drops without wall-clock time
public struct FrameID: Hashable, Comparable, Sendable, CustomStringConvertible {
    public let value: UInt64
    public init(_ value: UInt64) { self.value = value }
    public static func < (l: FrameID, r: FrameID) -> Bool { l.value < r.value }
    public var description: String { "#\(value)" }
    public func next() -> FrameID { FrameID(value &+ 1) }
}

/// host-time stamps carried with every frame so latency is measured, not estimated
/// captureHostTime is the source presentation time, ingestHostTime is when we first saw it
public struct FrameTiming: Sendable {
    public let captureHostTime: Double
    public let ingestHostTime: Double
    public init(captureHostTime: Double, ingestHostTime: Double) {
        self.captureHostTime = captureHostTime
        self.ingestHostTime = ingestHostTime
    }
    public func age(now: Double) -> Double { now - captureHostTime }
}

/// frame metadata without the pixel payload, so this core stays framework-free
/// the app layer pairs this header with the actual buffer via FrameID
public struct FrameHeader: Sendable {
    public let id: FrameID
    public let timing: FrameTiming
    public let width: Int
    public let height: Int

    public init(id: FrameID, timing: FrameTiming, width: Int, height: Int) {
        self.id = id
        self.timing = timing
        self.width = width
        self.height = height
    }
}
