import CoreMediaIO
import CoreMedia
import CoreVideo
import AspectusKit
import os

/// pushes corrected frames into the camera extension's sink stream
///
/// the extension is another process, so frames cross through the sink stream's CMSimpleQueue; its
/// fixed capacity carries the drop-stale invariant over the boundary. every failure is non-fatal
final class VirtualCameraSink: @unchecked Sendable {
    private let lock = NSLock()
    private var deviceID: CMIODeviceID = 0
    private var streamID: CMIOStreamID = 0
    private var queue: CMSimpleQueue?
    private var formatDescription: CMFormatDescription?
    private var started = false
    private var sequence: UInt64 = 0
    private var warnedAboutSize = false
    /// back-to-back full-queue drops, which is what a far side that has gone away looks like
    private var consecutiveDrops = 0

    /// two seconds at 30 fps; a merely slow host drains again well inside this
    private static let stallThreshold = 60

    private(set) var dropped = 0
    private(set) var sent = 0
    /// frames withheld to hold the advertised rate, counted apart from `dropped` because they are
    /// the pacer working rather than the far side failing
    private(set) var paced = 0

    /// capture may run faster than the format the extension declared; hosts are told one cadence
    /// and must not be given another
    private var pacer = PublishPacer(frameRate: Double(VirtualCameraFormat.frameRate))

    private let log = Logger(subsystem: "com.aspectus.app", category: "sink")

    /// the localized name the extension vends; matching on it avoids hard-coding a device id, and
    /// capture uses it to make sure a reopen never selects the camera we publish into
    static let deviceName = VirtualCameraIdentity.name

    /// whether the current run of failed connects has already been reported
    private var reportedFailure = false
    /// before the user installs the extension there is nothing to connect to, and saying so would be
    /// noise rather than news; only a camera we have actually held can be reported as lost
    private var everConnected = false
    /// when a camera we held stopped being reachable, which is what `isLost` is measured from
    private var lostAt: Date?

    /// how long a camera stays merely missing before it is called lost
    ///
    /// generous next to the 0.5 s retry, because the point is to be sure rather than quick: nothing
    /// the user could do about it gets better by being told sooner
    private static let lostAfter: TimeInterval = 3

    /// a camera this process held, lost, and cannot get back
    ///
    /// measured on macOS 26.6: when the extension is replaced, this process is told the old device
    /// went away and never that the new one arrived. no lookup finds it again — not the device list,
    /// not uid translation, not AVFoundation — while a process started afterwards sees it at once.
    /// so this state is terminal for the process, and the only thing that clears it is a relaunch
    var isLost: Bool {
        lock.withLock {
            guard everConnected, !started, let lostAt else { return false }
            return Date().timeIntervalSince(lostAt) >= Self.lostAfter
        }
    }

    var isConnected: Bool { lock.withLock { started } }

    // MARK: - connection

    /// looks for the virtual camera and opens its sink stream
    ///
    /// returns false when the extension is not installed or not yet activated, which is the normal
    /// state before the user installs it and must not be treated as an error
    @discardableResult
    func connect() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !started else { return true }

        guard let device = findDevice(named: Self.deviceName) else {
            reportFailure("it is not in this process's device list")
            return false
        }

        guard let sink = findSinkStream(of: device) else {
            reportFailure("it has no sink stream")
            return false
        }

        var cmQueue: Unmanaged<CMSimpleQueue>?
        let status = CMIOStreamCopyBufferQueue(sink, { _, _, _ in }, nil, &cmQueue)
        guard status == noErr, let cmQueue else {
            reportFailure("its queue could not be taken: \(status)")
            return false
        }

        guard CMIODeviceStartStream(device, sink) == noErr else {
            reportFailure("its sink stream would not start")
            return false
        }

        deviceID = device
        streamID = sink
        queue = cmQueue.takeUnretainedValue()
        started = true
        sequence = 0
        consecutiveDrops = 0
        reportedFailure = false
        everConnected = true
        lostAt = nil
        log.info("connected to the virtual camera")
        return true
    }

    /// reports the first failure of a run and stays quiet for the rest of it
    ///
    /// `revalidate` retries on the stats timer, so a failure that persists is the same failure twice
    /// a second; a run of them is one fact, and this process has already flooded the unified log once
    private func reportFailure(_ reason: String) {
        guard everConnected else { return }
        if lostAt == nil { lostAt = Date() }
        guard !reportedFailure else { return }
        reportedFailure = true
        log.error("""
            could not connect to the virtual camera: \(reason, privacy: .public). \
            retrying on the stats timer
            """)
    }

    func disconnect() {
        lock.lock()
        defer { lock.unlock() }
        guard started else { return }
        CMIODeviceStopStream(deviceID, streamID)
        queue = nil
        started = false
        consecutiveDrops = 0
    }

    /// rebuilds the connection when the extension process has been replaced under us
    ///
    /// a draining queue proves the far side is alive, so nothing is polled while frames flow; the
    /// extension consumes whether or not a host is attached, so a queue that stays full means gone
    @discardableResult
    func revalidate() -> Bool {
        lock.lock()
        let connected = started
        let stalled = started && consecutiveDrops >= Self.stallThreshold
        lock.unlock()

        guard connected else { return connect() }
        guard stalled else { return true }

        log.error("""
            the virtual camera stopped draining after \(Self.stallThreshold, privacy: .public) \
            consecutive frames, reconnecting
            """)
        disconnect()
        return connect()
    }

    // MARK: - publishing

    /// enqueues one finished frame, dropping it if the far side has not drained
    ///
    /// the timing carried here is the frame's own capture time, so the extension can present with
    /// the same cadence the camera produced rather than inventing one
    func publish(_ pixelBuffer: CVPixelBuffer, timing: FrameTiming) {
        lock.lock()
        defer { lock.unlock() }
        guard started, let queue else { return }

        // paced before the queue is consulted: withholding a frame we were never entitled to send
        // is not the far side falling behind and must not read as one
        guard pacer.shouldPublish(at: timing.captureHostTime) else {
            paced += 1
            return
        }

        // the consumer paces us, so a full queue is dropped and counted rather than newest-wins;
        // the run length is what tells a slow consumer from an absent one
        guard CMSimpleQueueGetCount(queue) < CMSimpleQueueGetCapacity(queue) else {
            dropped += 1
            consecutiveDrops += 1
            return
        }
        consecutiveDrops = 0

        guard let format = format(for: pixelBuffer) else { return }
        warnOnceIfFormatDiffers(from: pixelBuffer)

        let presentation = CMTime(value: CMTimeValue(timing.captureHostTime * 1_000_000_000),
                                  timescale: 1_000_000_000)
        var info = CMSampleTimingInfo(duration: .invalid,
                                      presentationTimeStamp: presentation,
                                      decodeTimeStamp: .invalid)
        var sampleBuffer: CMSampleBuffer?
        let status = CMSampleBufferCreateReadyWithImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescription: format,
            sampleTiming: &info,
            sampleBufferOut: &sampleBuffer)
        guard status == noErr, let sampleBuffer else {
            dropped += 1
            return
        }

        CMSimpleQueueEnqueue(queue, element: Unmanaged.passRetained(sampleBuffer).toOpaque())
        sequence &+= 1
        sent += 1
    }

    /// a mismatch here is invisible from inside the app: CMIO forwards it and hosts render it,
    /// having been told the wrong geometry. once only, this is the hot path
    private func warnOnceIfFormatDiffers(from pixelBuffer: CVPixelBuffer) {
        guard !warnedAboutSize else { return }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        guard width != Int(VirtualCameraFormat.width) || height != Int(VirtualCameraFormat.height)
        else { return }
        warnedAboutSize = true
        log.error("""
            publishing \(width, privacy: .public)x\(height, privacy: .public) into a virtual camera \
            that advertises \(VirtualCameraFormat.width, privacy: .public)x\
            \(VirtualCameraFormat.height, privacy: .public); hosts will be told the wrong geometry
            """)
    }

    /// the format description is cached because building one per frame would allocate on the hot path
    private func format(for pixelBuffer: CVPixelBuffer) -> CMFormatDescription? {
        if let formatDescription,
           CMVideoFormatDescriptionMatchesImageBuffer(formatDescription, imageBuffer: pixelBuffer) {
            return formatDescription
        }
        var created: CMFormatDescription?
        guard CMVideoFormatDescriptionCreateForImageBuffer(
            allocator: kCFAllocatorDefault, imageBuffer: pixelBuffer,
            formatDescriptionOut: &created) == noErr else { return nil }
        formatDescription = created
        return created
    }

    // MARK: - device discovery

    private func findDevice(named name: String) -> CMIODeviceID? {
        var address = CMIOObjectPropertyAddress(
            mSelector: CMIOObjectPropertySelector(kCMIOHardwarePropertyDevices),
            mScope: CMIOObjectPropertyScope(kCMIOObjectPropertyScopeGlobal),
            mElement: CMIOObjectPropertyElement(kCMIOObjectPropertyElementMain))

        var size: UInt32 = 0
        guard CMIOObjectGetPropertyDataSize(CMIOObjectID(kCMIOObjectSystemObject),
                                            &address, 0, nil, &size) == noErr, size > 0 else {
            return nil
        }
        let count = Int(size) / MemoryLayout<CMIODeviceID>.size
        var devices = [CMIODeviceID](repeating: 0, count: count)
        var used: UInt32 = 0
        guard CMIOObjectGetPropertyData(CMIOObjectID(kCMIOObjectSystemObject), &address, 0, nil,
                                        size, &used, &devices) == noErr else { return nil }

        return devices.first { deviceName(of: $0) == name }
    }

    private func deviceName(of device: CMIODeviceID) -> String? {
        var address = CMIOObjectPropertyAddress(
            mSelector: CMIOObjectPropertySelector(kCMIOObjectPropertyName),
            mScope: CMIOObjectPropertyScope(kCMIOObjectPropertyScopeGlobal),
            mElement: CMIOObjectPropertyElement(kCMIOObjectPropertyElementMain))
        var name: Unmanaged<CFString>?
        var used: UInt32 = 0
        let size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        guard CMIOObjectGetPropertyData(device, &address, 0, nil, size, &used, &name) == noErr else {
            return nil
        }
        return name?.takeRetainedValue() as String?
    }

    /// the extension vends both a source and a sink; only the sink accepts frames from us
    private func findSinkStream(of device: CMIODeviceID) -> CMIOStreamID? {
        var address = CMIOObjectPropertyAddress(
            mSelector: CMIOObjectPropertySelector(kCMIODevicePropertyStreams),
            mScope: CMIOObjectPropertyScope(kCMIOObjectPropertyScopeGlobal),
            mElement: CMIOObjectPropertyElement(kCMIOObjectPropertyElementMain))

        var size: UInt32 = 0
        guard CMIOObjectGetPropertyDataSize(device, &address, 0, nil, &size) == noErr,
              size > 0 else { return nil }
        let count = Int(size) / MemoryLayout<CMIOStreamID>.size
        var streams = [CMIOStreamID](repeating: 0, count: count)
        var used: UInt32 = 0
        guard CMIOObjectGetPropertyData(device, &address, 0, nil, size, &used, &streams) == noErr
        else { return nil }

        // measured on macOS 26.6 against our own extension: kCMIOStreamPropertyDirection reports
        // 1 for the source stream hosts read from, and 0 for the sink we write into. the naming is
        // from the device's point of view, not ours, and getting it the wrong way round silently
        // pushes every frame into the stream nobody reads
        return streams.first { direction(of: $0) == Self.sinkDirection }
            ?? streams.dropFirst().first
    }

    /// the value kCMIOStreamPropertyDirection reports for a sink, established by measurement
    private static let sinkDirection: UInt32 = 0

    private func direction(of stream: CMIOStreamID) -> UInt32? {
        var address = CMIOObjectPropertyAddress(
            mSelector: CMIOObjectPropertySelector(kCMIOStreamPropertyDirection),
            mScope: CMIOObjectPropertyScope(kCMIOObjectPropertyScopeGlobal),
            mElement: CMIOObjectPropertyElement(kCMIOObjectPropertyElementMain))
        var value: UInt32 = 0
        var used: UInt32 = 0
        guard CMIOObjectGetPropertyData(stream, &address, 0, nil,
                                        UInt32(MemoryLayout<UInt32>.size), &used, &value) == noErr
        else { return nil }
        return value
    }
}

private extension NSLock {
    func withLock<T>(_ body: () -> T) -> T {
        lock(); defer { unlock() }; return body()
    }
}
