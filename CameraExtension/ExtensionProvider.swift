import Foundation
import CoreMediaIO
import CoreMedia
import CoreVideo
import os

// lifecycle traces are `.error` for retention, not severity: only error and above persist without
// a live `log stream`. read them with
// `log show --predicate 'subsystem == "com.aspectus.app.cameraextension"' --info --debug`

final class ExtensionDeviceSource: NSObject, CMIOExtensionDeviceSource {
    private(set) var device: CMIOExtensionDevice!
    private var sourceStream: CMIOExtensionStream!
    private var sinkStream: CMIOExtensionStream!
    private var sourceStreamSource: SourceStreamSource!
    private var sinkStreamSource: SinkStreamSource!

    private let log = Logger(subsystem: "com.aspectus.app.cameraextension", category: "device")

    /// shared with the app so the advertised format and the published one cannot drift
    static let width = VirtualCameraFormat.width
    static let height = VirtualCameraFormat.height
    static let frameRate = VirtualCameraFormat.frameRate

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

    private let log = Logger(subsystem: "com.aspectus.app.cameraextension", category: "source")
    private nonisolated(unsafe) var sends: UInt64 = 0

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
        log.error("source startStream - a host attached")
        streaming.withLock { $0 = true }
    }

    func stopStream() throws {
        log.error("source stopStream")
        streaming.withLock { $0 = false }
    }

    /// forwards one finished frame to whichever host is attached
    func send(_ sampleBuffer: CMSampleBuffer, hostTimeInNanoseconds: UInt64) {
        guard isStreaming else { return }
        // separated from the `isStreaming` gate: a nil weak stream is otherwise indistinguishable
        // from a working forward
        guard let stream else {
            log.error("forward dropped: the source stream reference is nil")
            return
        }
        sends &+= 1
        if sends % 300 == 1 {
            log.debug("forwarded \(self.sends, privacy: .public) buffers into the host stream")
        }
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
        log.error("sink authorized for a client")
        clientLock.lock()
        self.client = client
        clientLock.unlock()
        return true
    }

    func startStream() throws {
        log.error("sink startStream")
        consuming.withLock { $0 = true }
        consumeNext()
    }

    func stopStream() throws {
        log.error("sink stopStream")
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
                if self.consumed % 300 == 1 {
                    self.log.debug("""
                        consumed \(self.consumed, privacy: .public) \
                        forwarding=\(self.output?.isStreaming == true, privacy: .public) \
                        hostTime=\(hostTime, privacy: .public)
                        """)
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
