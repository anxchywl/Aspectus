import Foundation

/// single-slot drop-stale hand-off from capture to processing, capacity is one in-flight frame
/// offer overwrites and counts a drop rather than queuing; this is how we keep latency low
public final class LatestValueBox<Value: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Value?
    private var finished = false
    private var waiter: CheckedContinuation<Value?, Never>?
    private var droppedCount = 0
    private var deliveredCount = 0

    public init() {}

    public var dropped: Int { lock.withLock { droppedCount } }
    public var delivered: Int { lock.withLock { deliveredCount } }

    public func offer(_ value: Value) {
        let resume: CheckedContinuation<Value?, Never>?
        lock.lock()
        if finished { lock.unlock(); return }
        if let w = waiter {
            waiter = nil
            deliveredCount += 1
            resume = w
            lock.unlock()
            resume?.resume(returning: value)
            return
        }
        if stored != nil { droppedCount += 1 } // overwrite drops the stale one
        stored = value
        lock.unlock()
    }

    /// returns nil once finished and no value remains
    public func take() async -> Value? {
        await withCheckedContinuation { (cont: CheckedContinuation<Value?, Never>) in
            lock.lock()
            if let v = stored {
                stored = nil
                deliveredCount += 1
                lock.unlock()
                cont.resume(returning: v)
                return
            }
            if finished {
                lock.unlock()
                cont.resume(returning: nil)
                return
            }
            waiter = cont
            lock.unlock()
        }
    }

    /// closes the box and resumes any parked waiter with nil, keeps shutdown prompt
    public func finish() {
        let resume: CheckedContinuation<Value?, Never>?
        lock.lock()
        finished = true
        resume = waiter
        waiter = nil
        lock.unlock()
        resume?.resume(returning: nil)
    }
}

private extension NSLock {
    func withLock<T>(_ body: () -> T) -> T {
        lock(); defer { unlock() }; return body()
    }
}
