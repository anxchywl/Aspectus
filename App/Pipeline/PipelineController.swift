import Foundation
import Combine
import AspectusKit

/// wires capture → box → correction → Metal preview and publishes diagnostics for the HUD
/// knows nothing about specific models, so phase 2/3 stages slot in without touching this file
///
/// latency is reported as two numbers: processing (ingest → present, the < 20 ms target we own)
/// and end-to-end (camera PTS → present, includes sensor delivery we cannot remove)
@MainActor
final class PipelineController: ObservableObject {
    @Published var captureFPS: Double = 0
    @Published var processFPS: Double = 0
    @Published var outputFPS: Double = 0
    @Published var processingMeanMs: Double = 0
    @Published var processingP95Ms: Double = 0
    @Published var endToEndMeanMs: Double = 0
    @Published var endToEndP95Ms: Double = 0
    @Published var droppedFrames: Int = 0
    @Published var inFlight: Int = 0
    @Published var formatDescription: String = "—"
    @Published var memoryMB: Double = 0
    @Published var thermalState: String = "nominal"
    @Published var isRunning = false
    @Published var permissionDenied = false

    // published per frame so the overlay stays smooth
    @Published var tracking: TrackingResult?
    @Published var trackingMeanMs: Double = 0
    @Published var trackingP95Ms: Double = 0
    @Published var showOverlay = true
    @Published var imageWidth: Int = 0
    @Published var imageHeight: Int = 0

    var mirrorPreview = true { didSet { renderer?.mirror = mirrorPreview } }

    private let capture = CameraCapture()
    private let tracker = VisionFaceTracker()
    private let corrector = PassthroughCorrector()
    private let processing = StageMetrics(name: "processing", window: 240)
    private let trackingMetrics = StageMetrics(name: "tracking", window: 240)
    private let e2e = StageMetrics(name: "end-to-end", window: 240)
    private weak var renderer: MetalRenderer?

    // fps meters updated on each event, snapshotted by the stats timer
    private var processMeter = RateMeter()
    private var outputMeter = RateMeter()
    private var processFPSValue: Double = 0
    private var outputFPSValue: Double = 0
    private var lastReceivedCount = 0
    private var lastStatsTime = HostClock.seconds

    // timestamps of the frame being drawn, for present-time latency
    private var pendingIngest: Double = 0
    private var pendingCapture: Double = 0

    private var consumerTask: Task<Void, Never>?
    private var statsTimer: Timer?

    func attach(renderer: MetalRenderer) {
        self.renderer = renderer
        renderer.mirror = mirrorPreview
        renderer.onPresented = { [weak self] in
            // runs off the main actor from the Metal completion handler
            Task { @MainActor in
                guard let self else { return }
                let now = HostClock.seconds
                self.processing.record(ms: max(0, (now - self.pendingIngest) * 1000))
                self.e2e.record(ms: max(0, (now - self.pendingCapture) * 1000))
                self.outputFPSValue = self.outputMeter.tick(at: now)
            }
        }
    }

    func start() async {
        guard !isRunning else { return }
        guard await CameraCapture.requestAccess() else {
            permissionDenied = true
            return
        }
        do {
            try capture.configure()
        } catch {
            formatDescription = "capture error: \(error)"
            return
        }
        formatDescription = capture.activeFormatDescription
        capture.start()
        isRunning = true
        startConsumer()
        startStatsTimer()
    }

    func stop() {
        capture.stop()
        consumerTask?.cancel(); consumerTask = nil
        statsTimer?.invalidate(); statsTimer = nil
        isRunning = false
    }

    private func startConsumer() {
        let box = capture.output
        consumerTask = Task { @MainActor [weak self] in
            while !Task.isCancelled, let frame = await box.take() {
                guard let self else { break }
                self.processFPSValue = self.processMeter.tick(at: HostClock.seconds)

                let tStart = HostClock.seconds
                let tr = await self.tracker.track(frame, header: frame.header)
                self.trackingMetrics.record(ms: max(0, (HostClock.seconds - tStart) * 1000))
                self.tracking = tr
                if self.imageWidth != frame.header.width { self.imageWidth = frame.header.width }
                if self.imageHeight != frame.header.height { self.imageHeight = frame.header.height }

                // still identity correction, exercises the seam with real tracking or neutral fallback
                let corrected = (try? await self.corrector.correct(
                    frame,
                    tracking: tr ?? .neutral(for: frame.header),
                    request: .neutral,
                    header: frame.header)) ?? frame

                self.pendingIngest = frame.header.timing.ingestHostTime
                self.pendingCapture = frame.header.timing.captureHostTime

                if let renderer = self.renderer, let view = renderer.attachedView {
                    renderer.enqueue(corrected.pixelBuffer, view: view)
                }
            }
        }
    }

    private func startStatsTimer() {
        statsTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.refreshStats() }
        }
    }

    private func refreshStats() {
        let p = processing.snapshot()
        processingMeanMs = p.meanMs
        processingP95Ms = p.p95Ms
        let e = e2e.snapshot()
        endToEndMeanMs = e.meanMs
        endToEndP95Ms = e.p95Ms

        let t = trackingMetrics.snapshot()
        trackingMeanMs = t.meanMs
        trackingP95Ms = t.p95Ms

        processFPS = processFPSValue
        outputFPS = outputFPSValue
        droppedFrames = capture.output.dropped
        inFlight = 0 // single-slot box, drained on take

        // delivered + dropped is every frame the camera produced
        let received = capture.output.delivered + capture.output.dropped
        let now = HostClock.seconds
        let dt = now - lastStatsTime
        if dt > 0 { captureFPS = Double(received - lastReceivedCount) / dt }
        lastReceivedCount = received
        lastStatsTime = now

        memoryMB = Self.residentMemoryMB()
        thermalState = Self.thermalString()
        formatDescription = capture.activeFormatDescription
    }

    // MARK: - system telemetry

    private static func residentMemoryMB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        return Double(info.resident_size) / (1024 * 1024)
    }

    private static func thermalString() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}

private extension TrackingResult {
    static func neutral(for header: FrameHeader) -> TrackingResult {
        let eye = EyeObservation(region: NormRect(x: 0, y: 0, width: 0, height: 0),
                                 pupilCenter: NormPoint(x: 0.5, y: 0.5), openness: 1)
        return TrackingResult(faceBounds: NormRect(x: 0, y: 0, width: 1, height: 1),
                              leftEye: eye, rightEye: eye,
                              headPose: HeadPose(yaw: 0, pitch: 0, roll: 0), confidence: 0)
    }
}

private extension CorrectionRequest {
    static var neutral: CorrectionRequest { CorrectionRequest(yawOffset: 0, pitchOffset: 0, strength: 0) }
}
