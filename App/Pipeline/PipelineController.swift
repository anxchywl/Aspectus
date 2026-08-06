import Foundation
import Combine
import os
import AspectusKit

/// wires capture → box → correction → Metal preview and publishes diagnostics for the HUD
/// knows nothing about specific models, so phase 2/3 stages slot in without touching this file
///
/// latency is reported as three numbers: pipeline (ingest → corrected frame ready, the < 20 ms
/// target we own and what the virtual camera will inherit), processing (ingest → present) and
/// end-to-end (camera PTS → present) — the last two add compositor and sensor time we do not own
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

    /// nil while frames are flowing, otherwise why they are not — a disconnect, a session error or
    /// sleep. the pipeline stays alive throughout, so recovery never restarts the correction state
    @Published private(set) var captureInterruption: String?

    @Published var correctorName: String = "—"

    /// no learned model is in the correction path, so there is no version to report; stated
    /// explicitly rather than left blank, which would read as "not wired up yet"
    let modelVersion = "n/a — geometric warp"

    /// every gaze number the HUD shows, sampled on the stats timer from the detached loop's
    /// collector; angles are signed so a systematic bias is visible instead of being hidden by a
    /// magnitude, and the distributions answer whether a gate ever actually fires
    @Published var gaze: DiagnosticsCollector.Snapshot = .empty

    /// what the virtual camera is doing, so a black picture in Zoom is diagnosable from the HUD
    @Published private(set) var virtualCameraState = "not connected"
    /// the same fact as the state string, for anything that has to branch on it rather than show it
    @Published private(set) var virtualCameraConnected = false
    @Published private(set) var virtualCameraSent = 0
    @Published private(set) var virtualCameraDropped = 0
    @Published private(set) var virtualCameraPaced = 0

    /// the cameras that can be selected, refreshed when one is attached or removed rather than
    /// polled, since discovery is not free
    @Published private(set) var cameras: [CameraCapture.DeviceOption] = []
    /// the camera the user asked for, nil meaning whatever the system offers as default
    @Published private(set) var preferredCameraID: String?
    /// the camera actually open, which after a disconnect need not be the one asked for
    @Published private(set) var activeCameraID: String?

    /// the active calibration, nil when the install has never been calibrated or was reset
    @Published private(set) var calibration: GazeCalibration?
    /// non-nil only while a calibration flow is running
    @Published private(set) var calibrationProgress: CalibrationProgress?
    /// the outcome of the last completed or failed calibration, for user-visible feedback
    @Published private(set) var calibrationResult: CalibrationOutcome?

    struct CalibrationProgress: Equatable {
        var target: CalibrationTarget
        var targetProgress: Double
        var overallProgress: Double
        var rejection: CalibrationRejection?
        /// the sweep has its own instructions and its own progress meter
        var phase: CalibrationSession.Phase = .targets
        var headMotionSpanYaw: Double = 0
        var headMotionSpanPitch: Double = 0
        /// counts down while the user is still moving their eyes to the new target
        var settleRemaining: Double?
        /// live reading for the current target, so a flat axis is visible as it happens
        var currentMeans: CalibrationSession.TargetMeans?
    }

    enum CalibrationOutcome: Equatable {
        case succeeded(GazeCalibration)
        /// carries what each target actually read, so a failure explains itself
        case failed(String, means: [CalibrationTarget: CalibrationSession.TargetMeans])
        case cancelled
    }

    /// the overlay is a visual check on the landmarks themselves, so it has to follow the pixels
    /// rather than a half-second timer; it has its own object so the per-frame publish reaches the
    /// canvas and nothing else. every numeric diagnostic goes through refreshStats, per AGENTS §7
    let overlay = TrackingOverlayModel()

    @Published var trackingMeanMs: Double = 0
    @Published var trackingP95Ms: Double = 0
    /// Vision's own cost, which the tracking figure above does not isolate
    @Published var visionMeanMs: Double = 0
    @Published var visionP95Ms: Double = 0
    @Published var correctionMeanMs: Double = 0
    @Published var correctionP95Ms: Double = 0

    /// ingest to corrected-frame-ready, the latency we own and the one the virtual camera will
    /// inherit; processing/end-to-end both include compositor time the extension will not pay
    @Published var pipelineMeanMs: Double = 0
    @Published var pipelineP95Ms: Double = 0
    @Published var showOverlay = true { didSet { preferences.showOverlay = showOverlay } }

    @Published var mirrorPreview = true {
        didSet {
            renderer?.mirror = mirrorPreview
            preferences.mirrorPreview = mirrorPreview
        }
    }

    /// A/B switch for the smoke test, read by the detached loop; it can only suppress correction,
    /// never loosen the confidence, angle, openness or head-pose gates
    var correctionEnabled: Bool {
        get { correctionSwitch.withLock { $0 } }
        set {
            objectWillChange.send()
            correctionSwitch.withLock { $0 = newValue }
            preferences.correctionEnabled = newValue
        }
    }

    private let correctionSwitch = OSAllocatedUnfairLock(initialState: true)

    /// screen-to-lens angle in degrees, the part of the correction that is not visible in the
    /// image; clamped to the gate's trusted angle so a setting can never widen the safety limit
    var redirectDegrees: Double {
        get { redirectStore.withLock { $0 } }
        set {
            objectWillChange.send()
            let clamped = max(-maxRedirectDegrees, min(maxRedirectDegrees, newValue))
            redirectStore.withLock { $0 = clamped }
            preferences.redirectDegrees = clamped
        }
    }

    /// the gate's trusted angle, exposed so the settings slider cannot offer a value the gate
    /// would refuse
    var maxRedirectDegrees: Double { config.gate.maxCorrectionDegrees }

    private let redirectStore = OSAllocatedUnfairLock(initialState: 12.0)

    /// read by the detached loop every frame; nil means run exactly as an uncalibrated install
    private let activeCalibration = OSAllocatedUnfairLock<GazeCalibration?>(initialState: nil)
    /// the running calibration flow, owned by the lock because the loop feeds it and the UI reads it
    private let sessionStore = OSAllocatedUnfairLock<CalibrationSession?>(initialState: nil)
    private let storage = CalibrationStore()

    /// supplied by the user because macOS gives no camera field of view, so it cannot be measured;
    /// the fitted gain is only as good as this number and the UI says so
    @Published var viewingDistanceMM: Double = 550 {
        didSet { preferences.viewingDistanceMM = viewingDistanceMM }
    }
    private var calibrationTimer: Timer?

    private let preferences = Preferences()

    private let capture = CameraCapture()
    private let tracker = VisionFaceTracker()
    private let config = PipelineConfig()
    private let warpTuning = EyeWarpTuning()

    // the geometric warp when Metal is available, otherwise the original frame passes through
    private let corrector: any EyeCorrector<CVReadyFrame> = MetalEyeCorrector() ?? PassthroughCorrector()
    private let processing = StageMetrics(name: "processing", window: 240)
    private let trackingMetrics = StageMetrics(name: "tracking", window: 240)
    private let correctionMetrics = StageMetrics(name: "correction", window: 240)
    private let pipelineMetrics = StageMetrics(name: "pipeline", window: 240)
    private let e2e = StageMetrics(name: "end-to-end", window: 240)
    private let diagnostics = DiagnosticsCollector()
    /// an output, never a dependency: if it is absent or fails the preview carries on untouched
    private let virtualCamera = VirtualCameraSink()
    private weak var renderer: MetalRenderer?

    // fps meters ticked on each event and read on the stats timer, so a stall reads as a stall
    private var processMeter = RateMeter()
    private var outputMeter = RateMeter()
    private var lastReceivedCount = 0
    private var lastStatsTime = HostClock.seconds

    private var consumerTask: Task<Void, Never>?
    private var statsTimer: Timer?
    private let benchmark = BenchmarkRecorder.fromLaunchArguments()

    private let log = Logger(subsystem: "com.aspectus.app", category: "pipeline")
    private var recovery = CaptureRecovery()
    private var recoveryTimer: Timer?
    private let sleepObserver = SystemSleepObserver()
    private var lastCaptureError: String?
    /// when the last reopen was issued, so a session that comes back silent is caught as a failure
    /// rather than believed
    private var restartIssuedAt: Double?
    /// the frame count at that moment, so only frames the reopened session produced can confirm it
    private var restartBaselineFrames = 0

    init() {
        // a stored calibration applies from the first frame, so a calibrated install behaves the
        // same on relaunch as it did when it was calibrated
        if let stored = storage.load() {
            calibration = stored.calibration
            activeCalibration.withLock { $0 = stored.calibration }
        }
        restorePreferences()
        cameras = CameraCapture.availableDevices()
        observeSystemEvents()
    }

    /// assigned through the same setters the UI uses, so a restored value is clamped and applied
    /// exactly as a typed one would be
    private func restorePreferences() {
        mirrorPreview = preferences.mirrorPreview
        showOverlay = preferences.showOverlay
        correctionEnabled = preferences.correctionEnabled
        redirectDegrees = preferences.redirectDegrees
        viewingDistanceMM = preferences.viewingDistanceMM
        preferredCameraID = preferences.preferredCameraID
    }

    /// switching camera reconfigures the session in place: the consumer loop, the gate and the
    /// filters stay up, so a switch costs no more state than a recovery reopen does
    func selectCamera(_ id: String?) {
        preferredCameraID = id
        preferences.preferredCameraID = id
        guard isRunning else { return }
        capture.suspend()
        do {
            try capture.configure(preferredDeviceID: id)
            capture.start()
            lastCaptureError = nil
            restartIssuedAt = nil
        } catch {
            let missing = (error as? CameraCapture.CaptureError) == .noDevice
            lastCaptureError = missing ? "no camera attached" : "\(error)"
            apply(.restartFailed(deviceMissing: missing))
        }
        formatDescription = capture.activeFormatDescription
        activeCameraID = capture.activeDeviceID
        captureInterruption = interruptionText()
    }

    // MARK: - capture lifecycle

    /// registered once for the lifetime of the controller: a camera can be unplugged or the machine
    /// slept while capture is stopped, and both must be seen when it starts again
    private func observeSystemEvents() {
        capture.observeEvents { [weak self] event in
            Task { @MainActor in self?.handle(event) }
        }
        sleepObserver.start(
            willSleep: { [weak self] in Task { @MainActor in self?.apply(.willSleep) } },
            didWake: { [weak self] in Task { @MainActor in self?.apply(.didWake) } })
    }

    private func handle(_ event: CameraCapture.Event) {
        switch event {
        case let .runtimeError(message):
            lastCaptureError = message
            apply(.runtimeError)
        case .interrupted: apply(.interrupted)
        case .interruptionEnded: apply(.interruptionEnded)
        case .deviceDisconnected:
            cameras = CameraCapture.availableDevices()
            apply(.deviceDisconnected)
        case .deviceConnected:
            cameras = CameraCapture.availableDevices()
            apply(.deviceConnected)
        }
    }

    /// the single point where the policy meets the session, so every recovery path suspends and
    /// reopens the same way
    private func apply(_ event: CaptureRecovery.Event) {
        // an explicit stop outranks recovery: the user asked for no camera, not for a broken one
        guard isRunning else { return }
        let action = recovery.handle(event)
        // the recovery paths are rare and mostly unwitnessed, so each transition is recorded where
        // it can be read after the fact rather than only shown in a HUD nobody was watching
        log.notice("""
            capture \(String(describing: event), privacy: .public) → \
            \(String(describing: self.recovery.state), privacy: .public)\
            \(action.suspendsSession ? " suspend" : "", privacy: .public)\
            \(action.restartAfter.map { " retry in \($0)s" } ?? "", privacy: .public)
            """)
        if action.suspendsSession {
            capture.suspend()
            restartIssuedAt = nil
        }
        if let delay = action.restartAfter {
            recoveryTimer?.invalidate()
            recoveryTimer = Timer.scheduledTimer(withTimeInterval: delay, repeats: false) { [weak self] _ in
                Task { @MainActor in self?.attemptRestart() }
            }
        }
        captureInterruption = interruptionText()
    }

    private func attemptRestart() {
        guard isRunning else { return }
        do {
            try capture.restart()
            formatDescription = capture.activeFormatDescription
            lastCaptureError = nil
            // startRunning is asynchronous, so success is claimed only once frames arrive; the
            // stats timer confirms it or times it out
            restartIssuedAt = HostClock.seconds
            restartBaselineFrames = capture.output.delivered + capture.output.dropped
            log.notice("capture reopened as \(self.formatDescription, privacy: .public), awaiting frames")
        } catch {
            let missing = (error as? CameraCapture.CaptureError) == .noDevice
            lastCaptureError = missing ? "no camera attached" : "\(error)"
            apply(.restartFailed(deviceMissing: missing))
        }
        captureInterruption = interruptionText()
    }

    /// a reopened session that never delivers looks identical to a healthy one from the outside,
    /// so delivery is the only accepted proof of recovery
    private func confirmRecovery(framesSoFar: Int, now: Double) {
        guard let issued = restartIssuedAt else { return }
        if framesSoFar > restartBaselineFrames {
            restartIssuedAt = nil
            apply(.restartSucceeded)
        } else if now - issued > Self.restartConfirmationSeconds {
            restartIssuedAt = nil
            lastCaptureError = "the camera reopened but delivered no frames"
            apply(.restartFailed(deviceMissing: false))
        }
    }

    /// three frame intervals of slack over a cold camera start, measured at ~0.4 s on the reference
    /// machine, so a real reopen is never mistaken for a failure
    private static let restartConfirmationSeconds = 3.0

    /// the same state in the short form the HUD has room for
    var captureStateLabel: String {
        switch recovery.state {
        case .running: return isRunning ? "running" : "stopped"
        case .interrupted: return "interrupted"
        case let .retrying(attempt): return "retrying \(attempt)"
        case .waitingForDevice: return "no camera"
        case .asleep: return "asleep"
        case .exhausted: return "failed"
        }
    }

    private func interruptionText() -> String? {
        switch recovery.state {
        case .running: return nil
        case .interrupted: return "Another app took the camera."
        case let .retrying(attempt):
            return "Reconnecting to the camera… (attempt \(attempt))"
        case .waitingForDevice: return "Camera disconnected. Waiting for one to be plugged in."
        case .asleep: return "Paused while the Mac sleeps."
        case .exhausted:
            let detail = lastCaptureError.map { "\n\($0)" } ?? ""
            return "Could not reopen the camera. Press Start to try again.\(detail)"
        }
    }

    // MARK: - calibration

    /// only ever called from an explicit user action, so calibration can never start silently
    func startCalibration() {
        guard isRunning else { return }
        calibrationResult = nil
        sessionStore.withLock { $0 = CalibrationSession() }
        calibrationProgress = CalibrationProgress(target: CalibrationTarget.allCases[0],
                                                  targetProgress: 0, overallProgress: 0,
                                                  rejection: nil, settleRemaining: nil,
                                                  currentMeans: nil)
        calibrationTimer?.invalidate()
        // faster than the stats timer because a rejection message the user cannot react to is
        // useless; bounded and only alive for the duration of the flow
        calibrationTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.refreshCalibration() }
        }
    }

    /// finishes without the sweep, keeping the five-target fit and giving up head compensation
    func skipHeadMotion() {
        sessionStore.withLock { session in
            guard session != nil else { return }
            session!.skipHeadMotion()
        }
        refreshCalibration()
    }

    func cancelCalibration() {
        endCalibrationFlow()
        calibrationResult = .cancelled
    }

    /// removes the calibration from disk as well as from the running pipeline
    func resetCalibration() {
        endCalibrationFlow()
        storage.delete()
        calibration = nil
        activeCalibration.withLock { $0 = nil }
        calibrationResult = nil
    }

    private func endCalibrationFlow() {
        calibrationTimer?.invalidate()
        calibrationTimer = nil
        sessionStore.withLock { $0 = nil }
        calibrationProgress = nil
    }

    private func refreshCalibration() {
        guard let session = sessionStore.withLock({ $0 }) else { return }
        var settling: Double?
        if case let .settling(remaining) = session.lastOutcome { settling = remaining }
        let span = session.headMotionSpan
        calibrationProgress = CalibrationProgress(
            target: session.target,
            targetProgress: session.phase == .headMotion
                ? session.headMotionProgress : session.progressForTarget,
            overallProgress: session.overallProgress,
            rejection: settling == nil ? session.lastRejection : nil,
            phase: session.phase,
            headMotionSpanYaw: span.yaw,
            headMotionSpanPitch: span.pitch,
            settleRemaining: settling,
            currentMeans: session.means(for: session.target))
        guard session.isFinished else { return }

        let means = session.allMeans
        endCalibrationFlow()
        do {
            let geometry = DisplayGeometry.geometry(viewingDistanceMM: viewingDistanceMM)
            let fitted = try session.fit(geometry: geometry)
            try storage.save(.init(calibration: fitted, samples: session.samples))
            calibration = fitted
            activeCalibration.withLock { $0 = fitted }
            calibrationResult = .succeeded(fitted)
        } catch let failure as CalibrationFit.Failure {
            calibrationResult = .failed(failure.description, means: means)
        } catch {
            calibrationResult = .failed("could not save the calibration: \(error.localizedDescription)",
                                        means: means)
        }
    }

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
                self.outputMeter.tick(at: presentedAt)
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
            try capture.configure(preferredDeviceID: preferredCameraID)
        } catch {
            formatDescription = "capture error: \(error)"
            return
        }
        formatDescription = capture.activeFormatDescription
        activeCameraID = capture.activeDeviceID
        correctorName = corrector is MetalEyeCorrector
            ? "geometric warp (metal, gpu)"
            : "passthrough (metal unavailable)"
        lastReceivedCount = capture.output.delivered + capture.output.dropped
        lastStatsTime = HostClock.seconds
        // an explicit start is a clean slate for recovery too, including one that had given up
        recovery = CaptureRecovery()
        restartIssuedAt = nil
        lastCaptureError = nil
        captureInterruption = nil
        // a restart must not report the previous session's distributions
        diagnostics.reset()
        gaze = .empty
        capture.start()
        isRunning = true
        startConsumer()
        startStatsTimer()
    }

    func stop() {
        virtualCamera.disconnect()
        capture.stop()
        consumerTask?.cancel(); consumerTask = nil
        statsTimer?.invalidate(); statsTimer = nil
        recoveryTimer?.invalidate(); recoveryTimer = nil
        restartIssuedAt = nil
        captureInterruption = nil
        renderer?.flush()
        isRunning = false
        // the stats timer is gone, so nothing would ever correct these: a stopped camera must not
        // leave the panel reporting the rate it was running at
        processMeter = RateMeter()
        outputMeter = RateMeter()
        captureFPS = 0
        processFPS = 0
        outputFPS = 0
    }

    /// the loop body runs off the main actor, hopping only to publish overlay state and to draw,
    /// since MTKView is main-actor bound; tracking and correction never touch the main actor
    ///
    /// Vision landmark detection costs ~20 ms, measured, and shrinking it would mean giving up the
    /// pupil points correction depends on. So it runs concurrently with the warp instead of ahead
    /// of it, and each frame is corrected using the previous frame's landmarks — one frame of
    /// staleness, against ~20 ms of latency removed from the display path.
    ///
    /// the loop still awaits tracking before taking the next frame, so exactly one frame is ever
    /// in flight; throughput is capped by max(tracking, frame interval), and tracking is faster
    /// than the camera's 33 ms cadence
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

        let pipelineMetrics = self.pipelineMetrics
        let diagnostics = self.diagnostics
        let activeCalibration = self.activeCalibration
        let sessionStore = self.sessionStore
        let virtualCamera = self.virtualCamera

        consumerTask = Task.detached(priority: .userInitiated) { [weak self] in
            // landmarks from the previous frame, which is what the current frame is corrected with
            var applied: TrackingResult?
            // capture time of the frame those landmarks came from, so staleness is measured
            // against real frame spacing rather than assumed to be one interval
            var appliedCapture: Double?

            while !Task.isCancelled, let frame = await box.take() {
                let tStart = HostClock.seconds
                async let tracked = tracker.track(frame, header: frame.header)

                let aspect = frame.header.height > 0
                    ? Double(frame.header.width) / Double(frame.header.height) : 1
                // correction is suppressed while calibrating: the user is being asked to judge
                // where they are looking, and warped eyes would make that impossible
                let calibrating = sessionStore.withLock { $0 != nil }
                let enabled = correctionSwitch.withLock { $0 } && !calibrating
                let calibration = activeCalibration.withLock { $0 }

                // filtering runs on the frame's own capture time, so smoothing and slew follow
                // real frame spacing rather than however long our processing happened to take
                let frameTime = frame.header.timing.captureHostTime
                let redirect = redirectStore.withLock { $0 } * .pi / 180.0

                let rawGaze = enabled ? applied.flatMap {
                    GazeGeometry.estimate($0, imageAspect: aspect, tuning: warpTuning)
                } : nil
                let gaze = rawGaze.map { raw in
                    guard let calibration, let applied else { return raw }
                    let degrees = 180.0 / Double.pi
                    return calibration.apply(raw,
                                             headYawDegrees: applied.headPose.yaw * degrees,
                                             headPitchDegrees: applied.headPose.pitch * degrees,
                                             headPoseAvailable: applied.headPoseAvailable)
                }

                // the vertical estimate carries a systematic upward bias: the upper lid covers the
                // top of the iris, so the aperture's centre sits below the neutral pupil and every
                // open eye reads as looking up. uncalibrated, subtracting that phantom angle
                // cancels real correction, so pitch falls back to the configured screen-to-lens
                // angle alone. a calibration measures the bias and removes it, which is what makes
                // the closed-loop vertical term trustworthy again
                let usePitch = calibration != nil
                let request = gaze.map {
                    CorrectionRequest(yawOffset: -$0.yaw,
                                      pitchOffset: usePitch ? -$0.pitch + redirect : redirect,
                                      strength: 1)
                }

                // the gate is updated every frame, including fallback frames, so the blend always
                // slews instead of snapping when tracking comes and goes
                if request == nil { gate.forceFallback() }
                let weight = gate.update(confidence: gaze?.confidence ?? 0,
                                         requestedCorrectionDegrees: request?.magnitudeDegrees ?? 0,
                                         t: frameTime)
                let angleFactor = gate.lastAngleFactor
                let engaged = gate.isEngaged

                let cStart = HostClock.seconds
                var corrected = frame
                var travelPixels = 0.0
                var correctionApplied = false
                if let applied, let request, weight > 0 {
                    let scaled = CorrectionRequest(yawOffset: request.yawOffset,
                                                   pitchOffset: request.pitchOffset,
                                                   strength: weight * userStrength)
                    corrected = (try? await corrector.correct(frame, tracking: applied,
                                                              request: scaled,
                                                              header: frame.header)) ?? frame
                    correctionMetrics.record(ms: (HostClock.seconds - cStart) * 1000)
                    // every corrector failure path returns its input, so buffer identity is the
                    // only honest signal that the warp actually ran
                    correctionApplied = corrected.pixelBuffer !== frame.pixelBuffer

                    if let w = GazeGeometry.warps(applied, imageAspect: aspect, request: scaled,
                                                  tuning: warpTuning) {
                        travelPixels = GazeGeometry.displacementPixels(
                            w.left, width: frame.header.width, height: frame.header.height)
                    }
                }

                // ingest to corrected-frame-ready: our own cost, and what the virtual camera will
                // see, with none of the compositor latency the preview's present time includes
                pipelineMetrics.record(ms: (HostClock.seconds - frame.header.timing.ingestHostTime) * 1000)

                let ageMs = appliedCapture.map { (frameTime - $0) * 1000 } ?? 0
                let fallback = Self.fallbackReason(enabled: enabled, applied: applied,
                                                   gaze: gaze, engaged: engaged,
                                                   angleFactor: angleFactor, weight: weight,
                                                   correctionApplied: correctionApplied,
                                                   imageAspect: aspect, tuning: warpTuning)
                let degrees = 180.0 / Double.pi
                diagnostics.record(GazeSample(
                    frameID: frame.header.id,
                    left: applied.map { EyeSample($0.leftEye) },
                    right: applied.map { EyeSample($0.rightEye) },
                    headYawDegrees: (applied?.headPose.yaw ?? 0) * degrees,
                    headPitchDegrees: (applied?.headPose.pitch ?? 0) * degrees,
                    headRollDegrees: (applied?.headPose.roll ?? 0) * degrees,
                    faceConfidence: applied?.confidence ?? 0,
                    headPoseAvailable: applied?.headPoseAvailable ?? false,
                    rawYawDegrees: (rawGaze?.yaw ?? 0) * degrees,
                    rawPitchDegrees: (rawGaze?.pitch ?? 0) * degrees,
                    gazeConfidence: gaze?.confidence,
                    calibratedYawDegrees: calibration == nil ? nil : (gaze?.yaw ?? 0) * degrees,
                    calibratedPitchDegrees: calibration == nil ? nil : (gaze?.pitch ?? 0) * degrees,
                    requestedYawDegrees: (request?.yawOffset ?? 0) * degrees,
                    requestedPitchDegrees: (request?.pitchOffset ?? 0) * degrees,
                    requestedMagnitudeDegrees: request?.magnitudeDegrees ?? 0,
                    angleFactor: angleFactor,
                    blendStrength: weight,
                    correctionAgeMs: ageMs,
                    irisTravelPixels: travelPixels,
                    fallback: fallback))

                // before mirroring, which is a display choice and must not reach a host
                virtualCamera.publish(corrected.pixelBuffer, timing: frame.header.timing)

                let landmarks = applied
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.processMeter.tick(at: HostClock.seconds)
                    self.overlay.tracking = landmarks
                    if self.overlay.imageWidth != frame.header.width {
                        self.overlay.imageWidth = frame.header.width
                    }
                    if self.overlay.imageHeight != frame.header.height {
                        self.overlay.imageHeight = frame.header.height
                    }
                    if let renderer = self.renderer, let view = renderer.attachedView {
                        renderer.enqueue(corrected.pixelBuffer, timing: frame.header.timing, view: view)
                    }
                }

                // awaiting here, after the frame is already on its way to the screen, is what keeps
                // exactly one frame in flight while still taking tracking off the critical path
                let tr = await tracked
                trackingMetrics.record(ms: (HostClock.seconds - tStart) * 1000)

                // losing the face invalidates the history, otherwise reacquisition blends the new
                // position against wherever the face used to be and visibly slides into place
                if let tr {
                    applied = stabilizer.stabilize(tr, t: frameTime)
                    appliedCapture = frameTime
                } else {
                    stabilizer.reset()
                    applied = nil
                    appliedCapture = nil
                }

                // calibration sees the same stabilized geometry the estimator will use next frame,
                // so what it measures is what the runtime path actually reads
                let forCalibration = applied
                sessionStore.withLock { session in
                    guard session != nil else { return }
                    _ = session!.offer(forCalibration, imageAspect: aspect,
                                       tuning: warpTuning, t: frameTime)
                }
            }
        }
    }

    /// names the first gate that suppressed correction, in the order the pipeline applies them, so
    /// a zero blend can be attributed instead of guessed at
    ///
    /// nonisolated and pure: it is called from the detached loop, never from the main actor
    private nonisolated static func fallbackReason(enabled: Bool,
                                                   applied: TrackingResult?,
                                                   gaze: GazeEstimate?,
                                                   engaged: Bool,
                                                   angleFactor: Double,
                                                   weight: Double,
                                                   correctionApplied: Bool,
                                                   imageAspect: Double,
                                                   tuning: EyeWarpTuning) -> FallbackReason {
        guard enabled else { return .disabled }
        guard let applied else { return .noTracking }
        guard gaze != nil else {
            return GazeGeometry.rejection(applied, imageAspect: imageAspect,
                                          tuning: tuning, requiringBothEyes: false)
        }
        guard engaged else { return .lowConfidence }
        guard angleFactor > 0 else { return .angleLimit }
        guard weight > 0 else { return .engaging }
        guard correctionApplied else {
            let geometric = GazeGeometry.rejection(applied, imageAspect: imageAspect,
                                                   tuning: tuning, requiringBothEyes: true)
            return geometric == .none ? .correctorFailed : geometric
        }
        return angleFactor < 1 ? .angleLimit : .none
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

        let v = tracker.metrics.snapshot()
        visionMeanMs = v.meanMs
        visionP95Ms = v.p95Ms

        let c = correctionMetrics.snapshot()
        correctionMeanMs = c.meanMs
        correctionP95Ms = c.p95Ms

        let pl = pipelineMetrics.snapshot()
        pipelineMeanMs = pl.meanMs
        pipelineP95Ms = pl.p95Ms

        gaze = diagnostics.snapshot()

        // picks the extension up when it is installed, and again when it is replaced, without a
        // restart either way
        virtualCamera.revalidate()
        virtualCameraConnected = virtualCamera.isConnected
        virtualCameraState = virtualCameraConnected ? "streaming" : "not connected"
        virtualCameraSent = virtualCamera.sent
        virtualCameraDropped = virtualCamera.dropped
        virtualCameraPaced = virtualCamera.paced

        droppedFrames = capture.output.dropped
        inFlight = capture.output.depth

        // delivered + dropped is every frame the camera produced
        let received = capture.output.delivered + capture.output.dropped
        let now = HostClock.seconds
        let dt = now - lastStatsTime

        // read rather than remembered: an occluded window presents nothing, and a rate that is only
        // recomputed when a frame arrives would go on reporting the last one it saw
        processFPS = processMeter.rate(asOf: now)
        outputFPS = outputMeter.rate(asOf: now)
        if dt > 0 { captureFPS = Double(received - lastReceivedCount) / dt }
        confirmRecovery(framesSoFar: received, now: now)
        lastReceivedCount = received
        lastStatsTime = now

        memoryMB = Self.residentMemoryMB()
        thermalState = Self.thermalString()
        formatDescription = capture.activeFormatDescription
        activeCameraID = capture.activeDeviceID

        benchmark?.record(.init(captureFPS: captureFPS, processFPS: processFPS, outputFPS: outputFPS,
                                tracking: t, vision: v, correction: c, pipeline: pl,
                                processing: p, endToEnd: e,
                                dropped: droppedFrames, depth: inFlight,
                                memoryMB: memoryMB, thermal: thermalState, gaze: gaze,
                                vcamState: virtualCameraState, vcamSent: virtualCameraSent,
                                vcamDropped: virtualCameraDropped, vcamPaced: virtualCameraPaced))
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
