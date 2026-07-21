import AVFoundation
import CoreVideo
import AspectusKit

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

    enum CaptureError: Error { case noDevice, cannotAddInput, cannotAddOutput }

    static func requestAccess() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .video)
        default: return false
        }
    }

    func configure(preferredDeviceID: String? = nil) throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        session.sessionPreset = .high

        let device: AVCaptureDevice?
        if let id = preferredDeviceID {
            device = AVCaptureDevice(uniqueID: id)
        } else {
            device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front)
                ?? AVCaptureDevice.default(for: .video)
        }
        guard let device else { throw CaptureError.noDevice }

        // reconfigure / device-switch path
        session.inputs.forEach { session.removeInput($0) }
        session.outputs.forEach { session.removeOutput($0) }

        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else { throw CaptureError.cannotAddInput }
        session.addInput(input)

        // 60 fps may be unreachable on a given Mac, it is a hardware/format cap
        Self.selectBestFormat(for: device, targetFPS: 60)

        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: sampleQueue)
        guard session.canAddOutput(videoOutput) else { throw CaptureError.cannotAddOutput }
        session.addOutput(videoOutput)

        let dims = CMVideoFormatDescriptionGetDimensions(device.activeFormat.formatDescription)
        let maxFPS = device.activeFormat.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
        activeFormatDescription = "\(device.localizedName) \(dims.width)×\(dims.height)@\(Int(maxFPS)) BGRA"
    }

    // highest supported frame rate, ties broken by pixel count, pinned so the camera runs at its ceiling
    private static func selectBestFormat(for device: AVCaptureDevice, targetFPS: Double) {
        let scored = device.formats.compactMap { format -> (AVCaptureDevice.Format, Double, Int32)? in
            let maxRate = format.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
            guard maxRate > 0 else { return nil }
            let d = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            return (format, maxRate, d.width * d.height)
        }
        // prefer formats that reach the target then the largest, else the fastest available
        let best = scored.filter { $0.1 >= targetFPS }.max { ($0.2, $0.1) < ($1.2, $1.1) }
            ?? scored.max { ($0.1, $0.2) < ($1.1, $1.2) }
        guard let best else { return }
        do {
            try device.lockForConfiguration()
            device.activeFormat = best.0
            let rate = min(targetFPS, best.1)
            let duration = CMTime(value: 1, timescale: CMTimeScale(rate))
            device.activeVideoMinFrameDuration = duration
            device.activeVideoMaxFrameDuration = duration
            device.unlockForConfiguration()
        } catch {
            // non-fatal, keep the default active format
        }
    }

    func start() {
        guard !session.isRunning else { return }
        sampleQueue.async { [session] in session.startRunning() }
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
