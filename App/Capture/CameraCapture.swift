// AVFoundation is not Sendable-audited, the session and device are confined to this type
@preconcurrency import AVFoundation
import CoreVideo
import AspectusKit
import os

/// owns the AVCaptureSession and delivers frames into a drop-stale box
/// alwaysDiscardsLateVideoFrames plus the single-slot box keep at most one frame in flight
final class CameraCapture: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    let output = LatestValueBox<CVReadyFrame>()

    private let session = AVCaptureSession()
    private let sampleQueue = DispatchQueue(label: "com.aspectus.capture.samples", qos: .userInteractive)
    private let videoOutput = AVCaptureVideoDataOutput()
    private var nextID = FrameID(0)

    private(set) var isRunning = false
    private(set) var activeFormatDescription: String = "—"

    private let log = Logger(subsystem: "com.aspectus.app", category: "capture")

    enum CaptureError: Error, Equatable { case noDevice, cannotAddInput, cannotAddOutput }

    /// what the session did on its own, so the pipeline can decide whether to reopen it
    enum Event: Sendable, Equatable {
        case runtimeError(String)
        case interrupted
        case interruptionEnded
        case deviceDisconnected
        case deviceConnected
    }

    /// notifications arrive on whatever thread AVFoundation posts from, so both the handler and the
    /// device identity it is compared against are lock-protected
    private let eventHandler = OSAllocatedUnfairLock<(@Sendable (Event) -> Void)?>(initialState: nil)
    private let activeDeviceIDStore = OSAllocatedUnfairLock<String?>(initialState: nil)
    /// the camera the user asked for, nil meaning "whatever the system offers as default"; kept so
    /// a reopen after a disconnect makes the same choice the first configure made
    private let preferredDeviceID = OSAllocatedUnfairLock<String?>(initialState: nil)

    override init() {
        super.init()
        let center = NotificationCenter.default
        center.addObserver(self, selector: #selector(sessionRuntimeError),
                           name: AVCaptureSession.runtimeErrorNotification, object: session)
        center.addObserver(self, selector: #selector(sessionWasInterrupted),
                           name: AVCaptureSession.wasInterruptedNotification, object: session)
        center.addObserver(self, selector: #selector(sessionInterruptionEnded),
                           name: AVCaptureSession.interruptionEndedNotification, object: session)
        // device notifications carry the device as the object, so they are filtered on arrival
        // rather than re-registered every time the session is reconfigured
        center.addObserver(self, selector: #selector(deviceWasDisconnected),
                           name: AVCaptureDevice.wasDisconnectedNotification, object: nil)
        center.addObserver(self, selector: #selector(deviceWasConnected),
                           name: AVCaptureDevice.wasConnectedNotification, object: nil)
    }

    deinit { NotificationCenter.default.removeObserver(self) }

    func observeEvents(_ handler: @escaping @Sendable (Event) -> Void) {
        eventHandler.withLock { $0 = handler }
    }

    private func emit(_ event: Event) {
        eventHandler.withLock { $0 }?(event)
    }

    @objc private func sessionRuntimeError(_ note: Notification) {
        let error = note.userInfo?[AVCaptureSessionErrorKey] as? NSError
        let message = error?.localizedDescription ?? "unknown capture error"
        log.error("capture session runtime error: \(message, privacy: .public)")
        isRunning = false
        emit(.runtimeError(message))
    }

    @objc private func sessionWasInterrupted(_ note: Notification) {
        log.error("capture session interrupted")
        emit(.interrupted)
    }

    @objc private func sessionInterruptionEnded(_ note: Notification) {
        log.notice("capture session interruption ended")
        emit(.interruptionEnded)
    }

    @objc private func deviceWasDisconnected(_ note: Notification) {
        guard let device = note.object as? AVCaptureDevice,
              device.uniqueID == activeDeviceIDStore.withLock({ $0 }) else { return }
        log.error("capture device disconnected: \(device.localizedName, privacy: .public)")
        emit(.deviceDisconnected)
    }

    /// any video camera appearing is worth reporting: after a disconnect the pipeline is looking
    /// for a replacement, not specifically for the one that left
    @objc private func deviceWasConnected(_ note: Notification) {
        guard let device = note.object as? AVCaptureDevice, device.hasMediaType(.video) else { return }
        log.notice("capture device connected: \(device.localizedName, privacy: .public)")
        emit(.deviceConnected)
    }

    static func requestAccess() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .video)
        default: return false
        }
    }

    func configure(preferredDeviceID: String? = nil) throws {
        self.preferredDeviceID.withLock { $0 = preferredDeviceID }

        // a stop finished the box permanently, reopen before the delegate can fire again
        output.reopen()

        session.beginConfiguration()

        let device: AVCaptureDevice?
        if let id = preferredDeviceID, let preferred = AVCaptureDevice(uniqueID: id),
           !Self.isOwnVirtualCamera(preferred) {
            device = preferred
        } else {
            // the preferred camera may be the one that was just unplugged, so a reopen falls back
            // to whatever is attached rather than failing while a usable camera exists
            device = Self.defaultDevice()
        }
        guard let device else {
            activeDeviceIDStore.withLock { $0 = nil }
            session.commitConfiguration()
            throw CaptureError.noDevice
        }
        activeDeviceIDStore.withLock { $0 = device.uniqueID }

        // reconfigure / device-switch path
        session.inputs.forEach { session.removeInput($0) }
        session.outputs.forEach { session.removeOutput($0) }

        let input: AVCaptureDeviceInput
        do {
            input = try AVCaptureDeviceInput(device: device)
        } catch {
            session.commitConfiguration()
            throw error
        }
        guard session.canAddInput(input) else {
            session.commitConfiguration()
            throw CaptureError.cannotAddInput
        }
        session.addInput(input)

        // 60 fps may be unreachable on a given Mac, it is a hardware/format cap
        Self.selectFormat(for: device, target: Self.targetFormat)

        // width and height here are what holds the delivered size; measured on macOS 26.6 a
        // matching session preset applies cleanly and changes neither activeFormat nor the output
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: Int(Self.targetFormat.width),
            kCVPixelBufferHeightKey as String: Int(Self.targetFormat.height),
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: sampleQueue)
        guard session.canAddOutput(videoOutput) else {
            session.commitConfiguration()
            throw CaptureError.cannotAddOutput
        }
        session.addOutput(videoOutput)

        session.commitConfiguration()

        // after the commit, which resets the device's frame durations
        Self.pinFrameRate(for: device, target: Self.targetFormat)

        // the sensor may legitimately run larger than the delivered size, so both are shown
        let dims = CMVideoFormatDescriptionGetDimensions(device.activeFormat.formatDescription)
        let fps = 1.0 / CMTimeGetSeconds(device.activeVideoMinFrameDuration)
        let sensor = dims.width == Self.targetFormat.width && dims.height == Self.targetFormat.height
            ? "" : " (sensor \(dims.width)×\(dims.height))"
        activeFormatDescription = "\(device.localizedName) \(Self.targetFormat.width)×"
            + "\(Self.targetFormat.height)@\(Int(fps.rounded())) BGRA\(sensor)"
    }

    /// a camera the user can pick, identified by the id a reopen is resolved against
    struct DeviceOption: Identifiable, Hashable, Sendable {
        let id: String
        let name: String
    }

    /// every attached video camera except the one we publish into, which would otherwise feed the
    /// pipeline its own output
    static func availableDevices() -> [DeviceOption] {
        AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .external, .continuityCamera],
            mediaType: .video, position: .unspecified)
            .devices
            .filter { !isOwnVirtualCamera($0) }
            .map { DeviceOption(id: $0.uniqueID, name: $0.localizedName) }
    }

    /// the camera actually in use, which is not always the one asked for after a disconnect
    var activeDeviceID: String? { activeDeviceIDStore.withLock { $0 } }

    private static func defaultDevice() -> AVCaptureDevice? {
        if let builtIn = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) {
            return builtIn
        }
        // AVCaptureDevice.default(for:) would happily hand back our own virtual camera once the
        // physical one is gone, which would feed the pipeline its own output
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .external, .continuityCamera],
            mediaType: .video, position: .unspecified)
        return discovery.devices.first { !isOwnVirtualCamera($0) }
    }

    private static func isOwnVirtualCamera(_ device: AVCaptureDevice) -> Bool {
        device.localizedName == VirtualCameraSink.deviceName
    }

    /// the benchmark operating point, fixed so latency numbers are comparable across runs
    struct TargetFormat {
        let width: Int32
        let height: Int32
        let fps: Double
    }

    /// resolution follows the virtual camera's advertised format; the rate is our own and is
    /// clamped by hardware
    static let targetFormat = TargetFormat(width: VirtualCameraFormat.width,
                                           height: VirtualCameraFormat.height,
                                           fps: 60)

    /// re-pinned after `commitConfiguration`, which resets the device's frame durations
    private static func pinFrameRate(for device: AVCaptureDevice, target: TargetFormat) {
        do {
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }
            let cap = device.activeFormat.videoSupportedFrameRateRanges
                .map(\.maxFrameRate).max() ?? target.fps
            let duration = CMTime(value: 1, timescale: CMTimeScale(min(target.fps, cap).rounded()))
            device.activeVideoMinFrameDuration = duration
            device.activeVideoMaxFrameDuration = duration
        } catch {
            // non-fatal, the camera keeps whatever rate it defaulted to
        }
    }

    /// pins the format nearest the target, preferring an exact match then the smallest format that
    /// still reaches the target rate, so the latency budget is not spent on unnecessary pixels
    private static func selectFormat(for device: AVCaptureDevice, target: TargetFormat) {
        let candidates = device.formats.compactMap { format -> (AVCaptureDevice.Format, Double, Int32, Int32)? in
            let maxRate = format.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
            guard maxRate > 0 else { return nil }
            let d = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            return (format, maxRate, d.width, d.height)
        }
        guard !candidates.isEmpty else { return }

        let atTargetRate = candidates.filter { $0.1 >= target.fps }
        let exact = atTargetRate.first { $0.2 == target.width && $0.3 == target.height }
        // pixel distance from the target keeps the choice deterministic across cameras
        let nearest = (atTargetRate.isEmpty ? candidates : atTargetRate).min {
            abs(Int($0.2) * Int($0.3) - Int(target.width) * Int(target.height))
                < abs(Int($1.2) * Int($1.3) - Int(target.width) * Int(target.height))
        }
        guard let chosen = exact ?? nearest else { return }

        do {
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }
            device.activeFormat = chosen.0
            let rate = min(target.fps, chosen.1)
            let duration = CMTime(value: 1, timescale: CMTimeScale(rate.rounded()))
            device.activeVideoMinFrameDuration = duration
            device.activeVideoMaxFrameDuration = duration
        } catch {
            // non-fatal, keep the default active format
        }
    }

    func start() {
        guard !session.isRunning else { return }
        // startRunning blocks for tens of ms, keep it off the caller's thread
        sampleQueue.async { [self] in session.startRunning() }
        isRunning = true
    }

    func stop() {
        suspend()
        output.finish()
    }

    /// stops the session but leaves the box open, so a recovery restart does not tear down the
    /// consumer loop and start the gate and filters over from a stale blend
    func suspend() {
        guard session.isRunning else {
            isRunning = false
            return
        }
        session.stopRunning()
        isRunning = false
    }

    /// reopens against whatever camera is attached now, keeping the original device preference
    func restart() throws {
        suspend()
        try configure(preferredDeviceID: preferredDeviceID.withLock { $0 })
        start()
    }

    // MARK: - sample delivery

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        let now = HostClock.seconds
        let capture = pts.isFinite && pts > 0 ? pts : now
        let header = FrameHeader(
            id: nextID,
            timing: FrameTiming(captureHostTime: capture, ingestHostTime: now),
            width: CVPixelBufferGetWidth(pb),
            height: CVPixelBufferGetHeight(pb)
        )
        nextID = nextID.next()
        self.output.offer(CVReadyFrame(header: header, pixelBuffer: pb))
    }
}
