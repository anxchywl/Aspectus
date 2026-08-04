import Foundation
import CoreMediaIO
import CoreMedia
import os

/// the unified log shows nothing for this process, so state goes to a file in the shared app group
/// container where both the extension and anyone debugging can read it
enum ExtensionTrace {
    private static let lock = NSLock()
    private static let url: URL? = FileManager.default
        .containerURL(forSecurityApplicationGroupIdentifier: "group.com.aspectus")?
        .appendingPathComponent("extension.log")

    static func write(_ message: String) {
        guard let url else { return }
        lock.lock(); defer { lock.unlock() }
        let line = "\(Date().timeIntervalSince1970) \(message)\n"
        if let handle = try? FileHandle(forWritingTo: url) {
            handle.seekToEndOfFile()
            handle.write(Data(line.utf8))
            try? handle.close()
        } else {
            try? line.write(to: url, atomically: true, encoding: .utf8)
        }
    }
}

/// the virtual camera the host applications see
///
/// deliberately thin: it owns no capture, no tracking and no correction. the app does all of that
/// and pushes finished frames into the sink stream; this process only forwards them to the source
/// stream that Zoom, Meet, OBS and friends read from. keeping the pipeline in the app means the
/// extension cannot fall behind it, and calibration keeps a UI to live in
///
/// the two streams are the whole design:
///   app  → sink stream   (CMIOStreamCopyBufferQueue, fixed capacity, drops when full)
///   sink → source stream (sendSampleBuffer)
final class ExtensionDeviceSource: NSObject, CMIOExtensionDeviceSource {
    private(set) var device: CMIOExtensionDevice!
    private var sourceStream: CMIOExtensionStream!
    private var sinkStream: CMIOExtensionStream!
    private var sourceStreamSource: SourceStreamSource!
    private var sinkStreamSource: SinkStreamSource!

    private let log = Logger(subsystem: "com.aspectus.app.cameraextension", category: "device")

    /// matches what the app's correction path produces, so no format conversion happens in transit
    static let width: Int32 = 1280
    static let height: Int32 = 720
    static let frameRate: Int32 = 30

    init(localizedName: String, deviceID: UUID) {
        super.init()
        device = CMIOExtensionDevice(localizedName: localizedName,
                                     deviceID: deviceID,
                                     legacyDeviceID: deviceID.uuidString,
                                     source: self)

        var format: CMFormatDescription?
        CMVideoFormatDescriptionCreate(allocator: kCFAllocatorDefault,
                                       codecType: kCVPixelFormatType_32BGRA,
                                       width: Self.width, height: Self.height,
                                       extensions: nil, formatDescriptionOut: &format)
        guard let format else { return }

        let streamFormat = CMIOExtensionStreamFormat(
            formatDescription: format,
            maxFrameDuration: CMTime(value: 1, timescale: Self.frameRate),
            minFrameDuration: CMTime(value: 1, timescale: Self.frameRate),
            validFrameDurations: nil)

        sourceStreamSource = SourceStreamSource(formats: [streamFormat])
        sinkStreamSource = SinkStreamSource(formats: [streamFormat])

        do {
            sourceStream = CMIOExtensionStream(localizedName: "Aspectus",
                                               streamID: UUID(),
                                               direction: .source,
                                               clockType: .hostTime,
                                               source: sourceStreamSource)
            sinkStream = CMIOExtensionStream(localizedName: "Aspectus Input",
                                             streamID: UUID(),
                                             direction: .sink,
                                             clockType: .hostTime,
                                             source: sinkStreamSource)
            sourceStreamSource.stream = sourceStream
            sinkStreamSource.stream = sinkStream
            sinkStreamSource.output = sourceStreamSource

            try device.addStream(sourceStream)
            try device.addStream(sinkStream)
        } catch {
            log.error("could not add streams: \(error.localizedDescription, privacy: .public)")
        }
    }

    var availableProperties: Set<CMIOExtensionProperty> { [.deviceTransportType, .deviceModel] }

    func deviceProperties(forProperties properties: Set<CMIOExtensionProperty>) throws
        -> CMIOExtensionDeviceProperties {
        let state = CMIOExtensionDeviceProperties(dictionary: [:])
        if properties.contains(.deviceTransportType) {
            // the four-char code IOKit uses for a virtual transport; the audio constant that spells
            // it is not exposed to Swift
            state.transportType = 0x76697274 // 'virt'
        }
        if properties.contains(.deviceModel) {
            state.model = "Aspectus Virtual Camera"
        }
        return state
    }

    func setDeviceProperties(_ deviceProperties: CMIOExtensionDeviceProperties) throws {}
}

/// what hosts read from
final class SourceStreamSource: NSObject, CMIOExtensionStreamSource {
    private(set) var formats: [CMIOExtensionStreamFormat]
    weak var stream: CMIOExtensionStream?

    /// a host is attached and expecting frames
    private let streaming = OSAllocatedUnfairLock(initialState: false)
    var isStreaming: Bool { streaming.withLock { $0 } }

    init(formats: [CMIOExtensionStreamFormat]) {
        self.formats = formats
        super.init()
    }

    var availableProperties: Set<CMIOExtensionProperty> { [.streamActiveFormatIndex, .streamFrameDuration] }

    func streamProperties(forProperties properties: Set<CMIOExtensionProperty>) throws
        -> CMIOExtensionStreamProperties {
        let state = CMIOExtensionStreamProperties(dictionary: [:])
        if properties.contains(.streamActiveFormatIndex) { state.activeFormatIndex = 0 }
        if properties.contains(.streamFrameDuration) {
            state.frameDuration = CMTime(value: 1, timescale: ExtensionDeviceSource.frameRate)
        }
        return state
    }

    func setStreamProperties(_ streamProperties: CMIOExtensionStreamProperties) throws {}

    func authorizedToStartStream(for client: CMIOExtensionClient) -> Bool { true }

    func startStream() throws {
        ExtensionTrace.write("source startStream - a host attached")
        streaming.withLock { $0 = true }
    }

    func stopStream() throws {
        ExtensionTrace.write("source stopStream")
        streaming.withLock { $0 = false }
    }

    /// forwards one finished frame to whichever host is attached
    func send(_ sampleBuffer: CMSampleBuffer, hostTimeInNanoseconds: UInt64) {
        guard isStreaming, let stream else { return }
        stream.send(sampleBuffer, discontinuity: [], hostTimeInNanoseconds: hostTimeInNanoseconds)
    }
}

/// what the app writes into
///
/// the consume loop is self-rescheduling rather than a queue: exactly one outstanding request at a
/// time, so a slow host can never make this process accumulate frames
final class SinkStreamSource: NSObject, CMIOExtensionStreamSource {
    private(set) var formats: [CMIOExtensionStreamFormat]
    weak var stream: CMIOExtensionStream?
    weak var output: SourceStreamSource?

    /// CMIOExtensionClient is not Sendable-audited, but it is only ever touched from the stream's
    /// own callbacks on the provider's client queue, so the confinement is real
    private nonisolated(unsafe) var client: CMIOExtensionClient?
    private let clientLock = NSLock()
    private let consuming = OSAllocatedUnfairLock(initialState: false)
    private nonisolated(unsafe) var consumed: UInt64 = 0
    private let log = Logger(subsystem: "com.aspectus.app.cameraextension", category: "sink")

    init(formats: [CMIOExtensionStreamFormat]) {
        self.formats = formats
        super.init()
    }

    var availableProperties: Set<CMIOExtensionProperty> { [.streamActiveFormatIndex, .streamFrameDuration] }

    func streamProperties(forProperties properties: Set<CMIOExtensionProperty>) throws
        -> CMIOExtensionStreamProperties {
        let state = CMIOExtensionStreamProperties(dictionary: [:])
        if properties.contains(.streamActiveFormatIndex) { state.activeFormatIndex = 0 }
        if properties.contains(.streamFrameDuration) {
            state.frameDuration = CMTime(value: 1, timescale: ExtensionDeviceSource.frameRate)
        }
        return state
    }

    func setStreamProperties(_ streamProperties: CMIOExtensionStreamProperties) throws {}

    func authorizedToStartStream(for client: CMIOExtensionClient) -> Bool {
        ExtensionTrace.write("sink authorized for a client")
        clientLock.lock()
        self.client = client
        clientLock.unlock()
        return true
    }

    func startStream() throws {
        ExtensionTrace.write("sink startStream")
        consuming.withLock { $0 = true }
        consumeNext()
    }

    func stopStream() throws {
        ExtensionTrace.write("sink stopStream")
        consuming.withLock { $0 = false }
        clientLock.lock()
        client = nil
        clientLock.unlock()
    }

    private func consumeNext() {
        clientLock.lock()
        let client = self.client
        clientLock.unlock()
        guard consuming.withLock({ $0 }), let stream else { return }
        // the client can arrive after the stream starts; returning without rescheduling used to
        // kill the loop permanently, so retry instead of giving up
        guard let client else {
            DispatchQueue.global().asyncAfter(deadline: .now() + 0.1) { [weak self] in
                self?.consumeNext()
            }
            return
        }

        stream.consumeSampleBuffer(from: client) { [weak self] sampleBuffer, sequence, discontinuity,
                                                    hasMoreSampleBuffers, error in
            guard let self else { return }
            if let sampleBuffer {
                let hostTime = UInt64(CMTimeGetSeconds(
                    CMSampleBufferGetPresentationTimeStamp(sampleBuffer)) * 1_000_000_000)
                self.consumed &+= 1
                if self.consumed % 60 == 1 {
                    ExtensionTrace.write("consumed \(self.consumed) forwarding=\(self.output?.isStreaming == true)")
                }
                self.output?.send(sampleBuffer, hostTimeInNanoseconds: hostTime)
                // the app is owed an acknowledgement or its queue never drains
                self.stream?.notifyScheduledOutputChanged(
                    CMIOExtensionScheduledOutput(sequenceNumber: sequence, hostTimeInNanoseconds: hostTime))
            } else if let error {
                self.log.error("consume failed: \(error.localizedDescription, privacy: .public)")
            }
            // one outstanding request at a time keeps this process bounded by construction
            self.consumeNext()
        }
    }
}

/// owns the single virtual device for the lifetime of the process
final class ExtensionProviderSource: NSObject, CMIOExtensionProviderSource {
    private(set) var provider: CMIOExtensionProvider!
    private var deviceSource: ExtensionDeviceSource!

    /// stable across launches so hosts remember the device instead of re-prompting every time
    private static let deviceID = UUID(uuidString: "6B0B9E3C-0F7A-4B0B-9E3C-0F7A4B0B9E3C")!

    init(clientQueue: DispatchQueue?) {
        super.init()
        provider = CMIOExtensionProvider(source: self, clientQueue: clientQueue)
        deviceSource = ExtensionDeviceSource(localizedName: "Aspectus", deviceID: Self.deviceID)
        do {
            try provider.addDevice(deviceSource.device)
        } catch {
            fatalError("could not add the virtual camera device: \(error)")
        }
    }

    func connect(to client: CMIOExtensionClient) throws {}
    func disconnect(from client: CMIOExtensionClient) {}

    var availableProperties: Set<CMIOExtensionProperty> { [.providerManufacturer] }

    func providerProperties(forProperties properties: Set<CMIOExtensionProperty>) throws
        -> CMIOExtensionProviderProperties {
        let state = CMIOExtensionProviderProperties(dictionary: [:])
        if properties.contains(.providerManufacturer) { state.manufacturer = "Aspectus" }
        return state
    }

    func setProviderProperties(_ providerProperties: CMIOExtensionProviderProperties) throws {}
}
