import SwiftUI
import AspectusKit

/// the diagnostics AGENTS §7 requires, grouped so the thirty-odd numbers can be read rather than
/// scanned; lives in the window's inspector so it never covers the picture it is describing
///
/// every value comes from the half-second stats snapshot, so this whole panel redraws twice a
/// second no matter how fast the camera runs
struct DiagnosticsHUD: View {
    @ObservedObject var controller: PipelineController

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                throughput
                latency
                tracking
                gaze
                output
                system
            }
            .padding(16)
        }
    }

    private var throughput: some View {
        section("Throughput") {
            row("capture", fps(controller.captureFPS))
            row("process", fps(controller.processFPS))
            row("output", fps(controller.outputFPS))
            row("dropped", "\(controller.droppedFrames)")
            row("in flight", "\(controller.inFlight)")
        }
    }

    private var latency: some View {
        section("Latency  mean / p95") {
            row("vision", ms(controller.visionMeanMs, controller.visionP95Ms),
                warn: controller.visionP95Ms >= 16)
            // the loop's critical path as seen from the tracking await, not Vision's own cost
            row("track loop", ms(controller.trackingMeanMs, controller.trackingP95Ms),
                warn: controller.trackingP95Ms >= 16)
            row("warp", ms(controller.correctionMeanMs, controller.correctionP95Ms),
                warn: controller.correctionP95Ms >= 8)
            row("pipeline", ms(controller.pipelineMeanMs, controller.pipelineP95Ms),
                warn: controller.pipelineP95Ms >= 20)
            row("processing", ms(controller.processingMeanMs, controller.processingP95Ms),
                warn: controller.processingP95Ms >= 20)
            row("end to end", ms(controller.endToEndMeanMs, controller.endToEndP95Ms))
        }
    }

    private var tracking: some View {
        let g = controller.gaze
        let s = g.latest
        return section("Tracking") {
            // no eye sample is how a frame with no face reaches the snapshot
            row("face", s?.left == nil ? "none" : String(format: "conf %.2f", s?.faceConfidence ?? 0),
                warn: s?.left == nil)
            row("head", String(format: "y%+.0f p%+.0f r%+.0f",
                               s?.headYawDegrees ?? 0, s?.headPitchDegrees ?? 0,
                               s?.headRollDegrees ?? 0))
            row("eyes open", String(format: "L%.2f R%.2f",
                                    s?.left?.openness ?? 0, s?.right?.openness ?? 0))
            // whether Vision hands back a real pupil landmark or the tracker falls back to the eye
            // contour centroid decides how much the estimate can be trusted at all
            row("pupil source", String(format: "%@/%@  %.0f%%", short(s?.left?.source),
                                       short(s?.right?.source), g.visionPupilShare * 100),
                warn: g.visionPupilShare < 1 && g.frames > 0)
            row("pupil points", "\(s?.left?.pupilPointCount ?? 0)/\(s?.right?.pupilPointCount ?? 0)")
            // a head-pose limit that never sees a head pose is not a safety gate at all
            row("head pose", String(format: "%.0f%% available", g.headPoseShare * 100),
                warn: g.frames > 0 && g.headPoseShare < 1)
            row("face conf", stats(g.faceConfidence))
            row("gaze conf", stats(g.gazeConfidence))
            // viewing distance is the one label input that cannot be measured from the image, and
            // recording six sessions at a declared 550 mm while actually sitting between roughly
            // 474 and 574 mm put more error into the labels than the model was being asked to beat.
            // crop side scales as 1/distance, so this makes that drift visible while it is
            // happening rather than months later in an offline audit.
            row("crop side", g.cropSidePixels.count == 0
                ? "—"
                : String(format: "%.1f px  [%.1f…%.1f]", g.cropSidePixels.mean,
                         g.cropSidePixels.minimum, g.cropSidePixels.maximum))
        }
    }

    private var gaze: some View {
        let g = controller.gaze
        let s = g.latest
        return section("Gaze") {
            row("offset L", offset(s?.left?.pupilOffset))
            row("offset R", offset(s?.right?.pupilOffset))
            // a non-zero mean over a run of steady direct gaze is the eyelid bias, in normalized units
            row("offset y mean", String(format: "%+.4f  [%+.4f…%+.4f]",
                                        g.verticalPupilOffset.mean,
                                        g.verticalPupilOffset.minimum,
                                        g.verticalPupilOffset.maximum))
            row("raw", String(format: "y%+.1f p%+.1f", s?.rawYawDegrees ?? 0, s?.rawPitchDegrees ?? 0))
            row("calibrated", calibratedText(s), warn: controller.calibration == nil)
            row("requested", String(format: "y%+.1f p%+.1f  (%.1f°)",
                                    s?.requestedYawDegrees ?? 0, s?.requestedPitchDegrees ?? 0,
                                    s?.requestedMagnitudeDegrees ?? 0))
            row("angle factor", String(format: "%.2f", s?.angleFactor ?? 0),
                warn: (s?.angleFactor ?? 1) < 1)
            row("blend", String(format: "%.0f%%", (s?.blendStrength ?? 0) * 100))
            row("iris travel", String(format: "%.1f px", s?.irisTravelPixels ?? 0),
                warn: (s?.blendStrength ?? 0) > 0.5 && (s?.irisTravelPixels ?? 0) < 2)
            row("landmark age", ms(g.correctionAgeMeanMs, g.correctionAgeP95Ms),
                warn: g.correctionAgeP95Ms >= 50)
            row("fallback", s?.fallback.rawValue ?? "none", warn: (s?.fallback ?? .none) != .none)
            row("worst gate", g.dominantFallback.rawValue, warn: g.dominantFallback != .none)
            row("corrected", String(format: "%.0f%% of %llu", correctedShare(g) * 100, g.frames))
        }
    }

    private var output: some View {
        section("Virtual camera") {
            row("stream", controller.virtualCameraState, warn: !controller.virtualCameraConnected)
            row("sent", "\(controller.virtualCameraSent)")
            // paced is the advertised-rate cap doing its job, so it is never a warning
            row("paced", "\(controller.virtualCameraPaced)")
            row("dropped", "\(controller.virtualCameraDropped)",
                warn: controller.virtualCameraDropped > 0)
        }
    }

    private var system: some View {
        section("System") {
            row("session", controller.captureStateLabel,
                warn: controller.captureInterruption != nil)
            row("memory", String(format: "%.0f MB", controller.memoryMB))
            row("thermal", controller.thermalState, warn: controller.thermalState != "nominal")
            VStack(alignment: .leading, spacing: 2) {
                Text(controller.formatDescription)
                Text(controller.correctorName)
                Text(controller.modelVersion)
            }
            .font(.caption.monospaced())
            .foregroundStyle(.secondary)
            .padding(.top, 2)
        }
    }

    @ViewBuilder
    private func section(_ title: String, @ViewBuilder rows: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
            rows()
        }
    }

    private func row(_ label: String, _ value: String, warn: Bool = false) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label).foregroundStyle(.secondary)
            Spacer(minLength: 8)
            Text(value)
                .foregroundStyle(warn ? Color.orange : Color.primary)
                .multilineTextAlignment(.trailing)
        }
        .font(.system(size: 11, design: .monospaced))
    }

    private func fps(_ value: Double) -> String { String(format: "%.0f fps", value) }

    private func ms(_ mean: Double, _ p95: Double) -> String {
        String(format: "%.1f / %.1f ms", mean, p95)
    }

    private func stats(_ v: ValueStats) -> String {
        String(format: "%.2f/%.2f/%.2f", v.minimum, v.mean, v.maximum)
    }

    private func offset(_ p: NormPoint?) -> String {
        String(format: "x%+.4f y%+.4f", p?.x ?? 0, p?.y ?? 0)
    }

    private func short(_ source: PupilSource?) -> String {
        switch source ?? PupilSource.none {
        case .visionLandmark: return "vis"
        case .contourCentroid: return "cont"
        case .none: return "—"
        }
    }

    private func calibratedText(_ s: GazeSample?) -> String {
        guard let yaw = s?.calibratedYawDegrees, let pitch = s?.calibratedPitchDegrees else {
            return "none"
        }
        return String(format: "y%+.1f p%+.1f", yaw, pitch)
    }

    private func correctedShare(_ g: DiagnosticsCollector.Snapshot) -> Double {
        g.frames == 0 ? 0 : Double(g.correctedFrames) / Double(g.frames)
    }
}
