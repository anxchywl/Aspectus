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
            Text("A dot will move across the display while you hold five gentle head positions. "
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
            } else {
                Circle()
                    .fill(.cyan)
                    .frame(width: 24, height: 24)
                    .shadow(color: .cyan.opacity(0.45), radius: 8)
                    .position(x: target.xFraction * size.width,
                              y: target.yFraction * size.height)
                    .animation(.easeInOut(duration: 0.18), value: target.id)
            }

            VStack(spacing: 8) {
                Text(poseInstruction(target.pose)).font(.headline)
                Text(status(snapshot))
                    .font(.callout).foregroundStyle(statusTint(snapshot))
                ProgressView(value: snapshot.progress)
                    .progressViewStyle(.linear).frame(width: 360)
                Text("target \(min(snapshot.targetNumber + 1, snapshot.targetCount)) of "
                     + "\(snapshot.targetCount) · \(snapshot.totalSamples) samples")
                    .font(.caption.monospaced()).foregroundStyle(.secondary)
                Button("Stop session") { controller.cancelGazeDataset() }
                    .font(.caption)
            }
            .padding(.horizontal, 20).padding(.vertical, 14)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
            .position(x: size.width / 2, y: size.height - 88)
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
        case .headPose: return "turn your head less"
        case .posePrompt: return "move farther in the requested head direction"
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
