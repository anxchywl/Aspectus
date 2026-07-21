import Foundation

/// rolling latency stats for one stage, sampled cheaply by the HUD without touching the hot path
public final class StageMetrics: @unchecked Sendable {
    public let name: String
    private let lock = NSLock()
    private var samples: [Double] = []       // milliseconds, ring buffer
    private let capacity: Int
    private var head = 0
    private var count = 0
    private var processedCount: UInt64 = 0
    private var droppedCount: UInt64 = 0

    public init(name: String, window: Int = 240) {
        self.name = name
        self.capacity = max(1, window)
        self.samples = Array(repeating: 0, count: capacity)
    }

    public func record(ms: Double) {
        lock.lock(); defer { lock.unlock() }
        samples[head] = ms
        head = (head + 1) % capacity
        if count < capacity { count += 1 }
        processedCount &+= 1
    }

    public func recordDrop() {
        lock.lock(); defer { lock.unlock() }
        droppedCount &+= 1
    }

    public struct Snapshot: Sendable {
        public let name: String
        public let meanMs: Double
        public let p95Ms: Double
        public let maxMs: Double
        public let processed: UInt64
        public let dropped: UInt64
    }

    public func snapshot() -> Snapshot {
        lock.lock()
        let slice = Array(samples.prefix(count))
        let processed = processedCount
        let dropped = droppedCount
        lock.unlock()

        guard !slice.isEmpty else {
            return Snapshot(name: name, meanMs: 0, p95Ms: 0, maxMs: 0,
                            processed: processed, dropped: dropped)
        }
        let mean = slice.reduce(0, +) / Double(slice.count)
        let sorted = slice.sorted()
        let p95Index = min(sorted.count - 1, Int((Double(sorted.count) * 0.95).rounded(.down)))
        return Snapshot(name: name, meanMs: mean, p95Ms: sorted[p95Index],
                        maxMs: sorted.last ?? 0, processed: processed, dropped: dropped)
    }

    @discardableResult
    public func measure<T>(_ body: () throws -> T) rethrows -> T {
        let start = DispatchTime.now().uptimeNanoseconds
        let result = try body()
        let end = DispatchTime.now().uptimeNanoseconds
        record(ms: Double(end - start) / 1_000_000.0)
        return result
    }
}
