import AppKit
import SwiftUI
import AspectusKit

/// explicit full-screen collection flow for the appearance-based gaze estimator
struct GazeDatasetView: View {
    @Environment(\.dismissWindow) private var dismissWindow
    @ObservedObject var controller: PipelineController
    @State private var split: GazeDatasetSplit = .training
    @State private var confirmingDelete = false
    @State private var datasetWindow: NSWindow?
    @State private var isFullScreen = false
    @State private var recorded: (training: Int, validation: Int) = (0, 0)

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                Color.black.ignoresSafeArea()
                switch controller.datasetProgress?.status {
                case .collecting:
                    collecting(in: geometry.size)
                case .finished:
                    outcome(title: "Session complete", systemImage: "checkmark.circle.fill",
                            tint: .green,
                            detail: "The labelled eye crops are saved locally for model development.")
                case .cancelled:
                    outcome(title: "Session stopped", systemImage: "xmark.circle.fill",
                            tint: .secondary,
                            detail: "The partial session remains local and is marked cancelled.")
                case let .failed(message):
                    outcome(title: "Recording failed",
                            systemImage: "exclamationmark.triangle.fill", tint: .orange,
                            detail: message)
                case nil:
                    setup
                }
            }
            .foregroundStyle(.white)
        }
        .background(DatasetWindowReader { window in
            window.collectionBehavior.remove(.fullScreenAuxiliary)
            window.collectionBehavior.insert(.fullScreenPrimary)
            datasetWindow = window
            isFullScreen = window.styleMask.contains(.fullScreen)
        })
        .onReceive(NotificationCenter.default.publisher(for: NSWindow.didEnterFullScreenNotification)) {
            if $0.object as? NSWindow === datasetWindow { isFullScreen = true }
        }
        .onReceive(NotificationCenter.default.publisher(for: NSWindow.didExitFullScreenNotification)) {
            if $0.object as? NSWindow === datasetWindow {
                isFullScreen = false
                if controller.datasetProgress?.status == .collecting {
                    controller.cancelGazeDataset()
                }
            }
        }
        .onAppear { refreshRecordedCounts() }
        .onChange(of: controller.datasetProgress?.status) { _, status in
            if status == .finished { refreshRecordedCounts() }
        }
        .onDisappear {
            if controller.datasetProgress?.status == .collecting {
                controller.cancelGazeDataset()
            }
        }
        .alert("Delete all model data?", isPresented: $confirmingDelete) {
            Button("Delete", role: .destructive) { controller.deleteGazeDatasets() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This permanently removes every recorded eye crop, manifest and participant ID. "
                 + "It cannot be recovered from Trash.")
        }
    }

    private var setup: some View {
        VStack(spacing: 22) {
            Image(systemName: "eye.square")
                .font(.system(size: 54)).foregroundStyle(.cyan)
            Text("Collect gaze model data").font(.largeTitle.weight(.semibold))
            Text("A dot will move across the display while you hold "
                 + "\(GazePosePrompt.allCases.count) gentle head positions. "
                 + "Look at the dot with your eyes and keep the requested head position steady.")
                .font(.title3).foregroundStyle(.secondary)
                .multilineTextAlignment(.center).frame(maxWidth: 680)

            Picker("Session", selection: $split) {
                Text("Training").tag(GazeDatasetSplit.training)
                Text("Validation").tag(GazeDatasetSplit.validation)
            }
            .pickerStyle(.segmented).frame(width: 340)

            VStack(alignment: .leading, spacing: 8) {
                Label("\(GazeDatasetPlan.totalSamples) labelled samples in six to nine minutes",
                      systemImage: "clock")
                Label("aligned eye crops, head pose, gaze labels and crop-quality metadata; no full-face images",
                      systemImage: "crop")
                Label("stored only in your private Application Support folder",
                      systemImage: "lock")
                Label("keep your usual glasses and normal call lighting",
                      systemImage: "lightbulb")
            }
            .font(.callout).foregroundStyle(.secondary)

            postureCheck

            Text(recorded.training + recorded.validation == 0
                 ? "no schema-\(GazeDatasetSchema5.version) sessions recorded yet"
                 : "schema-\(GazeDatasetSchema5.version) sessions so far: "
                   + "\(recorded.training) training · \(recorded.validation) validation")
                .font(.callout.monospaced()).foregroundStyle(.tertiary)

            if let error = controller.datasetError {
                Text(error).font(.callout).foregroundStyle(.orange)
            }

            HStack(spacing: 12) {
                Button("Close") { dismissWindow(id: "gaze-dataset") }
                if isFullScreen {
                    Button("Start \(split == .training ? "training" : "validation") session") {
                        controller.startGazeDataset(split)
                    }
                    .buttonStyle(.borderedProminent)
                } else {
                    Button("Enter full screen") { datasetWindow?.toggleFullScreen(nil) }
                        .buttonStyle(.borderedProminent)
                }
            }

            if !isFullScreen {
                Text("Collection starts after this window enters full screen.")
                    .font(.caption).foregroundStyle(.orange)
            }

            HStack(spacing: 12) {
                Button("Reveal collected data") { revealData() }
                    .disabled(!hasCollectedData)
                Button("Delete collected data", role: .destructive) {
                    confirmingDelete = true
                }
                .disabled(!hasCollectedData)
            }
            .font(.caption)
        }
        .padding(40)
    }

    /// live seating check, so an unreachable pose block is caught here rather than at block five
    @ViewBuilder
    private var postureCheck: some View {
        if let sample = controller.gaze.latest, sample.headPoseAvailable {
            let posture = GazeDatasetPosture(yawDegrees: sample.headYawDegrees,
                                             pitchDegrees: sample.headPitchDegrees,
                                             rollDegrees: sample.headRollDegrees)
            VStack(spacing: 6) {
                Label(posture.isReady
                      ? "seating leaves room for every head position"
                      : "this seating cannot complete every block",
                      systemImage: posture.isReady
                      ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                    .font(.callout.weight(.medium))
                    .foregroundStyle(posture.isReady ? .green : .orange)
                Text(String(format: "resting pose  yaw %+.0f°  pitch %+.0f°  roll %+.0f°",
                            sample.headYawDegrees, sample.headPitchDegrees,
                            sample.headRollDegrees))
                    .font(.caption.monospaced()).foregroundStyle(.secondary)
                if let advice = posture.advice {
                    Text(advice)
                        .font(.callout).foregroundStyle(.orange)
                        .multilineTextAlignment(.center).frame(maxWidth: 560)
                }
            }
            .padding(.horizontal, 18).padding(.vertical, 12)
            .background(.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 12))
            .animation(.easeInOut(duration: 0.25), value: posture.isReady)
        } else {
            // without tracking there is nothing to check, and silently omitting the panel would
            // read as a pass rather than as an unanswered question
            Label(controller.isRunning
                  ? "waiting for the tracker to find your face"
                  : "start the camera to check your seating before collecting",
                  systemImage: "video.slash")
                .font(.callout).foregroundStyle(.secondary)
                .padding(.horizontal, 18).padding(.vertical, 12)
                .background(.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 12))
        }
    }

    @ViewBuilder
    private func collecting(in size: CGSize) -> some View {
        if let snapshot = controller.datasetProgress, let target = snapshot.target {
            if target.kind == .lens {
                VStack(spacing: 8) {
                    Image(systemName: "arrow.up")
                        .font(.system(size: 34, weight: .bold))
                    Text("look at the physical camera lens")
                        .font(.title3.weight(.semibold))
                }
                .foregroundStyle(.cyan)
                .position(x: size.width / 2, y: 70)
                .transition(.opacity)
            } else {
                target_dot(accepted: snapshot.rejection == nil)
                    .position(x: target.xFraction * size.width,
                              y: target.yFraction * size.height)
                    .animation(.spring(response: 0.42, dampingFraction: 0.78), value: target.id)
            }

            blockBanner(for: target.pose)
                .position(x: size.width / 2, y: target.kind == .lens ? 150 : 62)

            VStack(spacing: 10) {
                Text(poseInstruction(target.pose))
                    .font(.headline).multilineTextAlignment(.center)
                    .frame(maxWidth: 520)

                if let guidance = snapshot.guidance {
                    GuidanceMeter(guidance: guidance)
                }

                Text(status(snapshot))
                    .font(.callout.weight(.medium))
                    .foregroundStyle(statusTint(snapshot))
                    .contentTransition(.opacity)
                    .frame(height: 18)

                sampleDots(filled: snapshot.samplesForTarget)

                ProgressView(value: snapshot.progress)
                    .progressViewStyle(.linear).frame(width: 380)
                Text("target \(min(snapshot.targetNumber + 1, snapshot.targetCount)) of "
                     + "\(snapshot.targetCount) · \(snapshot.totalSamples) samples")
                    .font(.caption.monospaced()).foregroundStyle(.secondary)
                Button("Stop session") { controller.cancelGazeDataset() }
                    .font(.caption)
            }
            .padding(.horizontal, 24).padding(.vertical, 16)
            .frame(width: 560)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
            .position(x: size.width / 2, y: size.height - 118)
            .animation(.easeInOut(duration: 0.2), value: snapshot.rejection)
        }
    }

    /// the dot answers "am I being recorded right now?" without the participant looking away to
    /// read the status line, which is the one thing they cannot do while fixating on it
    private func target_dot(accepted: Bool) -> some View {
        ZStack {
            Circle()
                .stroke(accepted ? Color.green.opacity(0.55) : Color.orange.opacity(0.5),
                        lineWidth: 2)
                .frame(width: 40, height: 40)
            Circle()
                .fill(accepted ? Color.green : Color.orange)
                .frame(width: 22, height: 22)
                .shadow(color: (accepted ? Color.green : Color.orange).opacity(0.5), radius: 9)
        }
        .animation(.easeInOut(duration: 0.22), value: accepted)
    }

    /// which of the pose blocks this is, so a nine-minute session has a visible shape
    private func blockBanner(for pose: GazePosePrompt) -> some View {
        let poses = GazePosePrompt.allCases
        let index = poses.firstIndex(of: pose) ?? 0
        return VStack(spacing: 7) {
            Text("block \(index + 1) of \(poses.count) · \(poseName(pose))")
                .font(.callout.weight(.semibold)).foregroundStyle(.secondary)
            HStack(spacing: 6) {
                ForEach(poses.indices, id: \.self) { position in
                    Capsule()
                        .fill(position < index ? Color.cyan.opacity(0.55)
                              : position == index ? Color.cyan : Color.white.opacity(0.18))
                        .frame(width: position == index ? 26 : 16, height: 4)
                }
            }
        }
        .padding(.horizontal, 16).padding(.vertical, 9)
        .background(.ultraThinMaterial, in: Capsule())
        .animation(.easeInOut(duration: 0.3), value: index)
    }

    private func sampleDots(filled: Int) -> some View {
        HStack(spacing: 5) {
            ForEach(0..<GazeDatasetPlan.samplesPerTarget, id: \.self) { index in
                Circle()
                    .fill(index < filled ? Color.green : Color.white.opacity(0.22))
                    .frame(width: 6, height: 6)
            }
        }
        .animation(.easeOut(duration: 0.15), value: filled)
    }

    private func poseName(_ pose: GazePosePrompt) -> String {
        switch pose {
        case .neutral: return "neutral"
        case .turnLeft: return "turn left"
        case .turnRight: return "turn right"
        case .lookUp: return "chin up"
        case .lookDown: return "chin down"
        case .tiltLeft: return "tilt left"
        case .tiltRight: return "tilt right"
        }
    }

    private func outcome(title: String, systemImage: String,
                         tint: Color, detail: String) -> some View {
        VStack(spacing: 18) {
            Image(systemName: systemImage).font(.system(size: 52)).foregroundStyle(tint)
            Text(title).font(.largeTitle.weight(.semibold))
            Text(detail).font(.title3).foregroundStyle(.secondary)
            if let snapshot = controller.datasetProgress {
                Text("\(snapshot.totalSamples) samples · \(snapshot.split.rawValue)")
                    .font(.callout.monospaced()).foregroundStyle(.secondary)
                Text("saved in the private model-data folder")
                    .font(.caption).foregroundStyle(.tertiary)
            }
            HStack {
                Button("Reveal data") { revealData() }
                Button("New session") { controller.clearGazeDatasetResult() }
                Button("Done") { dismissWindow(id: "gaze-dataset") }
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(40)
    }

    // "tilt" is reserved for the roll prompts, so the pitch prompts name the chin instead: a
    // participant who confuses the two axes records a physically inverted block
    private func poseInstruction(_ pose: GazePosePrompt) -> String {
        switch pose {
        case .neutral: return "face the camera straight; move only your eyes"
        case .turnLeft: return "turn your head slightly left; keep it there and move only your eyes"
        case .turnRight: return "turn your head slightly right; keep it there and move only your eyes"
        case .lookUp: return "raise your chin slightly; keep it there and move only your eyes"
        case .lookDown: return "lower your chin slightly; keep it there and move only your eyes"
        case .tiltLeft:
            return "tilt your head so your left ear moves toward your left shoulder; "
                + "keep facing the camera and move only your eyes"
        case .tiltRight:
            return "tilt your head so your right ear moves toward your right shoulder; "
                + "keep facing the camera and move only your eyes"
        }
    }

    private func status(_ snapshot: GazeDatasetRecorder.Snapshot) -> String {
        guard let rejection = snapshot.rejection else {
            return "hold steady — sample \(snapshot.samplesForTarget + 1) of "
                + "\(GazeDatasetPlan.samplesPerTarget)"
        }
        switch rejection {
        case .noTracking: return "move into view"
        case .lowConfidence: return "use steadier, clearer lighting"
        case .eyesClosed: return "open both eyes"
        case .headPoseUnavailable: return "hold your face clearly in view"
        // the limit applies to yaw, pitch and roll, so naming only turning sends a participant
        // whose chin is too low looking for a problem with how far they have turned
        case .headPose: return "your head is past the tracking limit; return toward level"
        case .posePrompt: return "move farther in the requested head direction"
        case .posePromptOvershoot: return "ease back toward centre; that is farther than needed"
        case .degenerateEyes: return "move a little closer"
        case .eyeAlignment: return "keep both eyes clearly visible"
        }
    }

    private func statusTint(_ snapshot: GazeDatasetRecorder.Snapshot) -> Color {
        snapshot.rejection == nil ? .green : .orange
    }

    private func revealData() {
        guard let url = controller.datasetProgress.map({ URL(fileURLWithPath: $0.directoryPath) })
                ?? controller.gazeDatasetRootURL else { return }
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    private var hasCollectedData: Bool {
        guard let url = controller.gazeDatasetRootURL else { return false }
        return FileManager.default.fileExists(atPath: url.path)
    }

    /// Counts finished sessions on the current schema only. A raw folder count would report the
    /// legacy and cancelled sessions too, which is worse than no number: none of them can fill a
    /// slot, so it would read as progress that does not exist. Only session metadata is opened,
    /// never a manifest or an eye crop.
    private func refreshRecordedCounts() {
        guard let root = controller.gazeDatasetRootURL,
              let names = try? FileManager.default.contentsOfDirectory(atPath: root.path) else {
            recorded = (0, 0)
            return
        }
        var training = 0
        var validation = 0
        for name in names {
            let metadata = root.appendingPathComponent(name)
                .appendingPathComponent("session.json")
            guard let data = try? Data(contentsOf: metadata),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  object["schemaVersion"] as? Int == GazeDatasetSchema5.version,
                  object["status"] as? String == "finished",
                  let split = object["split"] as? String else { continue }
            if split == GazeDatasetSplit.training.rawValue { training += 1 }
            if split == GazeDatasetSplit.validation.rawValue { validation += 1 }
        }
        recorded = (training, validation)
    }
}

/// Shows the distance to the accepting band rather than only naming the failure. The hardware
/// check that confirmed the tilt direction also produced a 23 degree roll against a 6 degree
/// request, so "not far enough" and "too far" need to be visibly different states, not one
/// sentence the participant has to re-read.
private struct GuidanceMeter: View {
    let guidance: GazePosePromptGate.Guidance

    private var span: Double { max((guidance.maximum ?? guidance.minimum * 2.5) * 1.35, 1) }
    private var overshooting: Bool {
        guard let maximum = guidance.maximum else { return false }
        return guidance.absolute > maximum
    }
    private var reached: Bool { guidance.towardGoal >= guidance.minimum && !overshooting }

    var body: some View {
        VStack(spacing: 5) {
            GeometryReader { geometry in
                let width = geometry.size.width
                let clamped = min(max(guidance.towardGoal, 0), span)
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.white.opacity(0.14))
                    // the accepting band, drawn so the participant aims at a region not a number
                    Capsule()
                        .fill(Color.green.opacity(0.22))
                        .frame(width: width * (bandEnd - bandStart))
                        .offset(x: width * bandStart)
                    Capsule()
                        .fill(overshooting ? Color.orange : reached ? Color.green : Color.cyan)
                        .frame(width: max(3, width * clamped / span))
                }
            }
            .frame(width: 380, height: 8)

            HStack {
                Text(guidance.axis.rawValue)
                Spacer()
                Text(String(format: "%+.0f°", guidance.towardGoal))
                    .foregroundStyle(overshooting ? .orange : reached ? .green : .secondary)
                Spacer()
                Text(guidance.maximum.map { String(format: "hold %.0f–%.0f°", guidance.minimum, $0) }
                     ?? String(format: "at least %.0f°", guidance.minimum))
            }
            .font(.caption2.monospaced()).foregroundStyle(.secondary)
            .frame(width: 380)
        }
        .animation(.easeOut(duration: 0.12), value: guidance.towardGoal)
    }

    private var bandStart: Double { min(guidance.minimum / span, 1) }
    private var bandEnd: Double { min((guidance.maximum ?? span) / span, 1) }
}

private struct DatasetWindowReader: NSViewRepresentable {
    var resolved: (NSWindow) -> Void

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async {
            if let window = view.window { resolved(window) }
        }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        guard let window = nsView.window else { return }
        DispatchQueue.main.async { resolved(window) }
    }
}
