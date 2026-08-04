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

    enum CaptureError: Error { case noDevice, cannotAddInput, cannotAddOutput }

    static func requestAccess() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .video)
        default: return false
        }
    }

    func configure(preferredDeviceID: String? = nil) throws {
        // a stop finished the box permanently, reopen before the delegate can fire again
        output.reopen()

        session.beginConfiguration()

        let device: AVCaptureDevice?
        if let id = preferredDeviceID {
            device = AVCaptureDevice(uniqueID: id)
        } else {
            device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front)
                ?? AVCaptureDevice.default(for: .video)
        }
        guard let device else {
            session.commitConfiguration()
            throw CaptureError.noDevice
        }

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
        guard session.isRunning else { return }
        session.stopRunning()
        isRunning = false
        output.finish()
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
