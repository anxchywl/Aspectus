import Foundation
import Combine
import os
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

    @Published var correctorName: String = "—"
    @Published var correctionWeight: Double = 0
    @Published var gazeDegrees: Double = 0

    /// how far the iris actually moves on screen, the number whose being ~0 made the first
    /// prototype invisible while every other metric looked healthy
    @Published var irisTravelPixels: Double = 0

    // published per frame so the overlay stays smooth
    @Published var tracking: TrackingResult?
    @Published var trackingMeanMs: Double = 0
    @Published var trackingP95Ms: Double = 0
    @Published var correctionMeanMs: Double = 0
    @Published var correctionP95Ms: Double = 0
    @Published var showOverlay = true
    @Published var imageWidth: Int = 0
    @Published var imageHeight: Int = 0

    var mirrorPreview = true { didSet { renderer?.mirror = mirrorPreview } }

    /// A/B switch for the smoke test, read by the detached loop; it can only suppress correction,
    /// never loosen the confidence, angle, openness or head-pose gates
    var correctionEnabled: Bool {
        get { correctionSwitch.withLock { $0 } }
        set { correctionSwitch.withLock { $0 = newValue }; objectWillChange.send() }
    }

    private let correctionSwitch = OSAllocatedUnfairLock(initialState: true)

    /// screen-to-lens angle in degrees, the part of the correction that is not visible in the
    /// image; clamped to the gate's trusted angle so a setting can never widen the safety limit
    var redirectDegrees: Double {
        get { redirectStore.withLock { $0 } }
        set {
            let limit = config.gate.maxCorrectionDegrees
            redirectStore.withLock { $0 = max(-limit, min(limit, newValue)) }
            objectWillChange.send()
        }
    }

    private let redirectStore = OSAllocatedUnfairLock(initialState: 12.0)

    private let capture = CameraCapture()
    private let tracker = VisionFaceTracker()
    private let config = PipelineConfig()
    private let warpTuning = EyeWarpTuning()

    // the geometric warp when Metal is available, otherwise the original frame passes through
    private let corrector: any EyeCorrector<CVReadyFrame> = MetalEyeCorrector() ?? PassthroughCorrector()
    private let processing = StageMetrics(name: "processing", window: 240)
    private let trackingMetrics = StageMetrics(name: "tracking", window: 240)
    private let correctionMetrics = StageMetrics(name: "correction", window: 240)
    private let e2e = StageMetrics(name: "end-to-end", window: 240)
    private weak var renderer: MetalRenderer?

    // fps meters updated on each event, snapshotted by the stats timer
    private var processMeter = RateMeter()
    private var outputMeter = RateMeter()
    private var processFPSValue: Double = 0
    private var outputFPSValue: Double = 0
    private var lastReceivedCount = 0
    private var lastStatsTime = HostClock.seconds

    private var consumerTask: Task<Void, Never>?
    private var statsTimer: Timer?
    private let benchmark = BenchmarkRecorder.fromLaunchArguments()

    func attach(renderer: MetalRenderer) {
        self.renderer = renderer
        renderer.mirror = mirrorPreview
        // metrics are lock-protected and the meter is main-actor, so only the meter needs a hop
        let processing = self.processing
        let e2e = self.e2e
        renderer.onPresented = { [weak self] timing, presentedAt in
            processing.record(ms: (presentedAt - timing.ingestHostTime) * 1000)
            e2e.record(ms: (presentedAt - timing.captureHostTime) * 1000)
            Task { @MainActor in
                guard let self else { return }
                self.outputFPSValue = self.outputMeter.tick(at: presentedAt)
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
        correctorName = corrector is MetalEyeCorrector
            ? "geometric warp (metal, gpu)"
            : "passthrough (metal unavailable)"
        lastReceivedCount = capture.output.delivered + capture.output.dropped
        lastStatsTime = HostClock.seconds
        capture.start()
        isRunning = true
        startConsumer()
        startStatsTimer()
    }

    func stop() {
        capture.stop()
        consumerTask?.cancel(); consumerTask = nil
        statsTimer?.invalidate(); statsTimer = nil
        renderer?.flush()
        isRunning = false
    }

    /// the loop body runs off the main actor, hopping only to publish overlay state and to draw,
    /// since MTKView is main-actor bound; tracking and correction never touch the main actor
    private func startConsumer() {
        let box = capture.output
        let tracker = self.tracker
        let corrector = self.corrector
        let trackingMetrics = self.trackingMetrics
        let correctionMetrics = self.correctionMetrics
        let warpTuning = self.warpTuning

        // gate state is owned by the loop, so a stop/start cycle begins from a clean disengaged
        // state instead of resuming a stale blend
        var gate = CorrectionGate(config: config.gate)
        var stabilizer = TemporalStabilizer(config: .init(landmark: config.landmarkSmoothing,
                                                          gaze: config.gazeSmoothing))
        let userStrength = config.userStrength
        let correctionSwitch = self.correctionSwitch
        let redirectStore = self.redirectStore

        consumerTask = Task.detached(priority: .userInitiated) { [weak self] in
            while !Task.isCancelled, let frame = await box.take() {
                let tStart = HostClock.seconds
                let tr = await tracker.track(frame, header: frame.header)
                trackingMetrics.record(ms: (HostClock.seconds - tStart) * 1000)

                let aspect = frame.header.height > 0
                    ? Double(frame.header.width) / Double(frame.header.height) : 1
                let enabled = correctionSwitch.withLock { $0 }

                // filtering runs on the frame's own capture time, so smoothing and slew follow
                // real frame spacing rather than however long our processing happened to take
                let frameTime = frame.header.timing.captureHostTime

                // losing the face invalidates the history, otherwise reacquisition blends the new
                // position against wherever the face used to be and visibly slides into place
                guard let tr else {
                    stabilizer.reset()
                    gate.forceFallback()
                    let w = gate.update(confidence: 0, requestedCorrectionDegrees: 0, t: frameTime)
                    await MainActor.run { [weak self] in
                        guard let self else { return }
                        self.processFPSValue = self.processMeter.tick(at: HostClock.seconds)
                        self.tracking = nil
                        self.correctionWeight = w
                        self.gazeDegrees = 0
                        self.irisTravelPixels = 0
                        if let renderer = self.renderer, let view = renderer.attachedView {
                            renderer.enqueue(frame.pixelBuffer, timing: frame.header.timing, view: view)
                        }
                    }
                    continue
                }

                let stable = stabilizer.stabilize(tr, t: frameTime)
                let gaze = enabled
                    ? GazeGeometry.estimate(stable, imageAspect: aspect, tuning: warpTuning)
                        .map { stabilizer.stabilize($0, t: frameTime) }
                    : nil

                let redirect = redirectStore.withLock { $0 } * .pi / 180.0

                // the angle the gate judges is the total redirection, so the screen-to-lens term
                // is subject to the same trusted-angle limit as the observed gaze
                let request = gaze.map {
                    CorrectionRequest(yawOffset: -$0.yaw, pitchOffset: -$0.pitch + redirect,
                                      strength: 1)
                }

                // the gate is updated every frame, including fallback frames, so the blend always
                // slews instead of snapping when tracking comes and goes
                if request == nil { gate.forceFallback() }
                let weight = gate.update(confidence: gaze?.confidence ?? 0,
                                         requestedCorrectionDegrees: request?.magnitudeDegrees ?? 0,
                                         t: frameTime)

                let cStart = HostClock.seconds
                var corrected = frame
                var travelPixels = 0.0
                if let request, weight > 0 {
                    let scaled = CorrectionRequest(yawOffset: request.yawOffset,
                                                   pitchOffset: request.pitchOffset,
                                                   strength: weight * userStrength)
                    corrected = (try? await corrector.correct(frame, tracking: stable,
                                                              request: scaled,
                                                              header: frame.header)) ?? frame
                    correctionMetrics.record(ms: (HostClock.seconds - cStart) * 1000)

                    if let w = GazeGeometry.warps(stable, imageAspect: aspect, request: scaled,
                                                  tuning: warpTuning) {
                        travelPixels = GazeGeometry.displacementPixels(
                            w.left, width: frame.header.width, height: frame.header.height)
                    }
                }

                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.processFPSValue = self.processMeter.tick(at: HostClock.seconds)
                    self.tracking = stable
                    self.correctionWeight = weight
                    self.gazeDegrees = gaze?.magnitudeDegrees ?? 0
                    self.irisTravelPixels = travelPixels
                    if self.imageWidth != frame.header.width { self.imageWidth = frame.header.width }
                    if self.imageHeight != frame.header.height { self.imageHeight = frame.header.height }
                    if let renderer = self.renderer, let view = renderer.attachedView {
                        renderer.enqueue(corrected.pixelBuffer, timing: frame.header.timing, view: view)
                    }
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

        let c = correctionMetrics.snapshot()
        correctionMeanMs = c.meanMs
        correctionP95Ms = c.p95Ms

        processFPS = processFPSValue
        outputFPS = outputFPSValue
        droppedFrames = capture.output.dropped
        inFlight = capture.output.depth

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

        benchmark?.record(.init(captureFPS: captureFPS, processFPS: processFPS, outputFPS: outputFPS,
                                tracking: t, correction: c, processing: p, endToEnd: e,
                                dropped: droppedFrames, depth: inFlight,
                                memoryMB: memoryMB, thermal: thermalState))
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
