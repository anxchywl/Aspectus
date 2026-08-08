import SwiftUI
import AspectusKit

/// the guided calibration flow: one target at a time, a live target dot, and a visible reason
/// whenever samples are being rejected so a stalled step never looks like a hang
struct CalibrationView: View {
    @ObservedObject var controller: PipelineController
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 20) {
            if let result = controller.calibrationResult {
                outcome(result)
            } else if let progress = controller.calibrationProgress {
                collecting(progress)
            } else {
                introduction
            }
        }
        .padding(28)
        .frame(width: 470, height: 560)
        .onDisappear {
            // closing the sheet mid-flow must not leave a session collecting in the background
            if controller.calibrationProgress != nil { controller.cancelCalibration() }
        }
    }

    private var introduction: some View {
        VStack(spacing: 16) {
            Text("Calibrate gaze").font(.title2.weight(.semibold))
            Text("""
                 Five short steps looking at the camera and display edges, then one where you turn \
                 your head. Keep your head still for the first five and move only your eyes.

                 Aspectus measures how your eyes read relative to your camera and stores the result \
                 on this Mac. Only angles are saved — never any image of you — and nothing is sent \
                 anywhere.
                 """)
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.leading)
                .fixedSize(horizontal: false, vertical: true)

            // macOS reports no camera field of view, so distance cannot be measured and the scale
            // fit is only as good as what is entered here
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Viewing distance")
                    Spacer()
                    TextField("", value: $controller.viewingDistanceMM, format: .number)
                        .frame(width: 60).multilineTextAlignment(.trailing)
                    Text("mm")
                    Stepper("", value: $controller.viewingDistanceMM, in: 250...1200, step: 25)
                        .labelsHidden()
                }
                Text("Roughly how far your eyes are from the screen. Used with your display's real "
                     + "size to work out how far you actually turned your eyes — macOS does not "
                     + "report camera optics, so this cannot be measured automatically.")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                Text(DisplayGeometry.displayDescription())
                    .font(.caption.monospaced()).foregroundStyle(.secondary)
            }

            if let calibration = controller.calibration {
                Text(String(format: "Currently calibrated %@ — neutral bias yaw %+.1f°, pitch %+.1f°",
                            calibration.createdAt.formatted(date: .abbreviated, time: .shortened),
                            calibration.yawOffsetDegrees, calibration.pitchOffsetDegrees))
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }

            Spacer()
            HStack {
                if controller.calibration != nil {
                    Button("Reset", role: .destructive) { controller.resetCalibration() }
                }
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Start") { controller.startCalibration() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(!controller.isRunning)
            }
        }
    }

    @ViewBuilder
    private func collecting(_ progress: PipelineController.CalibrationProgress) -> some View {
        if progress.phase == .headMotion { headMotion(progress) } else { fixation(progress) }
    }

    /// the sweep: gaze pinned to the lens while the head turns, so true gaze stays zero and the
    /// contamination becomes measurable
    private func headMotion(_ progress: PipelineController.CalibrationProgress) -> some View {
        VStack(spacing: 14) {
            Text("Keep looking at the lens").font(.title3.weight(.medium))
            Text("Now slowly turn your head left, right, up and down — but keep your eyes locked "
                 + "on the camera lens the whole time.")
                .font(.callout).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)

            ZStack {
                RoundedRectangle(cornerRadius: 12).stroke(.secondary.opacity(0.4), lineWidth: 1)
                lensGuide
            }
            .frame(height: 120)

            ProgressView(value: progress.targetProgress).progressViewStyle(.linear)

            // both spans have to open up before the two axes can be told apart
            VStack(alignment: .leading, spacing: 3) {
                span("turn left / right", progress.headMotionSpanYaw)
                span("nod up / down", progress.headMotionSpanPitch)
            }
            .font(.caption.monospaced())

            Text(progress.rejection?.description ?? "measuring — keep your gaze on the lens")
                .font(.callout)
                .foregroundStyle(progress.rejection == nil ? Color.green : Color.orange)
                .frame(height: 20)

            Spacer()
            HStack {
                Button("Skip this step") { controller.skipHeadMotion() }
                    .help("Finish without head-movement compensation")
                Spacer()
                Button("Cancel") { controller.cancelCalibration() }
            }
        }
    }

    private func span(_ label: String, _ degrees: Double) -> some View {
        let needed = 12.0
        return HStack {
            Text(label).foregroundStyle(.secondary)
            Spacer()
            Text(String(format: "%.0f° of %.0f°", degrees, needed))
                .foregroundStyle(degrees >= needed ? Color.green : Color.orange)
        }
        .frame(width: 320)
    }

    private func fixation(_ progress: PipelineController.CalibrationProgress) -> some View {
        VStack(spacing: 14) {
            Text(instruction(for: progress.target)).font(.title3.weight(.medium))
            Text(hint(for: progress.target))
                .font(.callout).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)

            ZStack {
                RoundedRectangle(cornerRadius: 12)
                    .stroke(.secondary.opacity(0.4), lineWidth: 1)
                targetGuide(progress.target)
            }
            .frame(height: 170)

            ProgressView(value: progress.targetProgress)
                .progressViewStyle(.linear)
                .opacity(progress.settleRemaining == nil ? 1 : 0.3)

            // a rejection is normal and transient, so it reads as guidance rather than an error
            Text(statusText(progress))
                .font(.callout)
                .foregroundStyle(statusColor(progress))
                .frame(height: 20)

            // the live reading, so a flat axis is visible while it is happening rather than only
            // as a failure message four steps later
            Text(progress.currentMeans.map {
                String(format: "reading  yaw %+.1f°  pitch %+.1f°  (%d samples)",
                       $0.yawDegrees, $0.pitchDegrees, $0.count)
            } ?? "reading  —")
                .font(.caption.monospaced()).foregroundStyle(.secondary)

            Text(String(format: "step %d of %d — %.0f%% overall",
                        stepNumber(progress.target), CalibrationTarget.allCases.count + 1,
                        progress.overallProgress * 100))
                .font(.caption.monospaced()).foregroundStyle(.secondary)

            Spacer()
            Button("Cancel") { controller.cancelCalibration() }
        }
    }

    private func outcome(_ result: PipelineController.CalibrationOutcome) -> some View {
        VStack(spacing: 16) {
            switch result {
            case let .succeeded(calibration):
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 44)).foregroundStyle(.green)
                Text("Calibration saved").font(.title2.weight(.semibold))
                VStack(alignment: .leading, spacing: 6) {
                    line("neutral bias", String(format: "yaw %+.2f°, pitch %+.2f°",
                                                calibration.yawOffsetDegrees,
                                                calibration.pitchOffsetDegrees))
                    line("up vs down", String(format: "%.1f°", calibration.verticalSeparationDegrees))
                    line("left vs right", String(format: "%.1f°",
                                                 calibration.horizontalSeparationDegrees))
                    line("samples", "\(calibration.sampleCount)")
                    line("scale yaw", String(format: "×%.2f%@", calibration.yawGain,
                                              calibration.yawGainFitted == true ? "" : "  not fitted"))
                    line("scale pitch", String(format: "×%.2f%@", calibration.pitchGain,
                                               calibration.pitchGainFitted == true ? "" : "  not fitted"))
                    if let d = calibration.viewingDistanceMM {
                        line("at distance", String(format: "%.0f mm", d))
                    }
                    line("head compensation", calibration.headCoupling.map {
                        String(format: "yaw %+.2f/%+.2f, pitch %+.2f/%+.2f",
                               $0.yawFromYaw, $0.yawFromPitch, $0.pitchFromYaw, $0.pitchFromPitch)
                    } ?? "not fitted")
                }
                .font(.callout.monospaced())
                Text(calibration.gainFitted == true
                     ? "Correction now follows your eyes, scaled to your display and distance."
                     : "Vertical gaze correction now follows your eyes instead of a fixed angle.")
                    .font(.caption).foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            case let .failed(message, means):
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 44)).foregroundStyle(.orange)
                Text("Calibration failed").font(.title2.weight(.semibold))
                Text(message)
                    .font(.callout).foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(CalibrationTarget.allCases, id: \.self) { target in
                        line(target.rawValue, means[target].map {
                            String(format: "yaw %+.1f°  pitch %+.1f°  (%d)",
                                   $0.yawDegrees, $0.pitchDegrees, $0.count)
                        } ?? "no samples")
                        if let m = means[target] {
                            line("", String(format: "offset  x%+.4f  y%+.4f", m.offsetX, m.offsetY))
                                .foregroundStyle(.tertiary)
                        }
                    }
                }
                .font(.caption.monospaced())
                Text("The previous calibration remains active. This attempt's angles were saved "
                     + "locally for diagnosis.")
                    .font(.caption).foregroundStyle(.secondary)
            case .cancelled:
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 44)).foregroundStyle(.secondary)
                Text("Calibration cancelled").font(.title2.weight(.semibold))
                Text("Nothing was changed.").font(.callout).foregroundStyle(.secondary)
            }

            Spacer()
            HStack {
                Button("Try again") { controller.startCalibration() }
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
            }
        }
    }

    private func statusText(_ progress: PipelineController.CalibrationProgress) -> String {
        if let remaining = progress.settleRemaining {
            return String(format: "get ready… %.0f", max(1, remaining.rounded(.up)))
        }
        return progress.rejection?.description ?? "measuring — hold still"
    }

    private func statusColor(_ progress: PipelineController.CalibrationProgress) -> Color {
        if progress.settleRemaining != nil { return .secondary }
        return progress.rejection == nil ? .green : .orange
    }

    private func line(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label).foregroundStyle(.secondary)
            Spacer()
            Text(value)
        }
        .frame(width: 300)
    }

    // targets are relative to the lens, not the screen, because that is what correction aims at
    private func instruction(for target: CalibrationTarget) -> String {
        switch target {
        case .center: return "Look straight at the camera lens"
        case .up: return "Look above the camera"
        case .down: return "Look below the camera"
        case .left: return "Look to the left of the camera"
        case .right: return "Look to the right of the camera"
        }
    }

    private func hint(for target: CalibrationTarget) -> String {
        switch target {
        case .center: return "Find the lens itself and hold your gaze on it."
        case .up: return "Look just above the top edge of your display."
        case .down: return "Look at the bottom edge of your display."
        case .left: return "Look at the left edge of your display."
        case .right: return "Look at the right edge of your display."
        }
    }

    @ViewBuilder
    private func targetGuide(_ target: CalibrationTarget) -> some View {
        if target == .center {
            lensGuide
        } else {
            VStack(spacing: 6) {
                Image(systemName: edgeSymbol(target))
                    .font(.title2.weight(.semibold))
                Text("look outside this window")
                    .font(.caption.weight(.medium))
                Text(edgeLabel(target))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .foregroundStyle(.tint)
        }
    }

    private var lensGuide: some View {
        VStack(spacing: 6) {
            Image(systemName: "arrow.up")
                .font(.title2.weight(.semibold))
            Text("look outside this window")
                .font(.caption.weight(.medium))
            Text("at the physical camera lens")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .foregroundStyle(.tint)
    }

    private func edgeSymbol(_ target: CalibrationTarget) -> String {
        switch target {
        case .center, .up: return "arrow.up"
        case .down: return "arrow.down"
        case .left: return "arrow.left"
        case .right: return "arrow.right"
        }
    }

    private func edgeLabel(_ target: CalibrationTarget) -> String {
        switch target {
        case .center: return "at the physical camera lens"
        case .up: return "just above the physical camera lens"
        case .down: return "at the physical bottom edge of the display"
        case .left: return "at the physical left edge of the display"
        case .right: return "at the physical right edge of the display"
        }
    }

    private func stepNumber(_ target: CalibrationTarget) -> Int {
        (CalibrationTarget.allCases.firstIndex(of: target) ?? 0) + 1
    }
}
