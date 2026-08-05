import SwiftUI
import AspectusKit

/// live overlay of fps, latency, drops, queue depth, memory, thermal state, and camera format
struct DiagnosticsHUD: View {
    @ObservedObject var controller: PipelineController

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            row("capture", String(format: "%.0f fps", controller.captureFPS))
            row("process", String(format: "%.0f fps", controller.processFPS))
            row("output", String(format: "%.0f fps", controller.outputFPS))
            Divider().overlay(Color.white.opacity(0.2))
            row("track lat", String(format: "%.1f / %.1f ms", controller.trackingMeanMs, controller.trackingP95Ms),
                warn: controller.trackingP95Ms >= 16)
            row("warp lat", String(format: "%.1f / %.1f ms", controller.correctionMeanMs, controller.correctionP95Ms),
                warn: controller.correctionP95Ms >= 8)
            row("pipe lat", String(format: "%.1f / %.1f ms", controller.pipelineMeanMs, controller.pipelineP95Ms),
                warn: controller.pipelineP95Ms >= 20)
            row("proc lat", String(format: "%.1f / %.1f ms", controller.processingMeanMs, controller.processingP95Ms),
                warn: controller.processingP95Ms >= 20)
            row("e2e lat", String(format: "%.1f / %.1f ms", controller.endToEndMeanMs, controller.endToEndP95Ms))
            if let tr = controller.tracking {
                row("face", String(format: "conf %.2f", tr.confidence))
                row("head", String(format: "y%+.0f p%+.0f r%+.0f",
                                    tr.headPose.yaw * 180 / .pi,
                                    tr.headPose.pitch * 180 / .pi,
                                    tr.headPose.roll * 180 / .pi))
                row("eyes", String(format: "L%.2f R%.2f", tr.leftEye.openness, tr.rightEye.openness))
            } else {
                row("face", "none", warn: true)
            }
            gazeRows
            row("session", controller.captureStateLabel,
                warn: controller.captureInterruption != nil)
            row("vcam", controller.virtualCameraState,
                warn: controller.virtualCameraState != "streaming")
            row("vcam frames", "\(controller.virtualCameraSent) sent / \(controller.virtualCameraDropped) dropped",
                warn: controller.virtualCameraDropped > 0)
            row("dropped", "\(controller.droppedFrames)")
            row("in-flight", "\(controller.inFlight)")
            row("memory", String(format: "%.0f MB", controller.memoryMB))
            row("thermal", controller.thermalState, warn: controller.thermalState != "nominal")
            Divider().overlay(Color.white.opacity(0.2))
            Text(controller.formatDescription)
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.white.opacity(0.7))
                .lineLimit(2)
            Text(controller.correctorName)
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.white.opacity(0.7))
                .lineLimit(1)
            Text(controller.modelVersion)
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.white.opacity(0.7))
                .lineLimit(1)
        }
        .padding(10)
        .frame(width: 270, alignment: .leading)
        .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 10))
        .font(.system(size: 12, weight: .medium, design: .monospaced))
    }

    /// the phase 1 gaze diagnostics, all sampled on the stats timer rather than per frame
    @ViewBuilder
    private var gazeRows: some View {
        let g = controller.gaze
        let s = g.latest
        Divider().overlay(Color.white.opacity(0.2))
        // whether Vision hands back a real pupil landmark or the tracker falls back to the eye
        // contour centroid decides how much the estimate can be trusted at all
        row("pupil src", String(format: "%@/%@ %.0f%%",
                                short(s?.left?.source), short(s?.right?.source),
                                g.visionPupilShare * 100),
            warn: g.visionPupilShare < 1 && g.frames > 0)
        row("pupil pts", "\(s?.left?.pupilPointCount ?? 0)/\(s?.right?.pupilPointCount ?? 0)")
        // a head-pose limit that never sees a head pose is not a safety gate at all
        row("head pose", String(format: "%.0f%% available", g.headPoseShare * 100),
            warn: g.frames > 0 && g.headPoseShare < 1)
        row("pupil off L", String(format: "x%+.4f y%+.4f",
                                  s?.left?.pupilOffset.x ?? 0, s?.left?.pupilOffset.y ?? 0))
        row("pupil off R", String(format: "x%+.4f y%+.4f",
                                  s?.right?.pupilOffset.x ?? 0, s?.right?.pupilOffset.y ?? 0))
        // a non-zero mean over a run of steady direct gaze is the eyelid bias, in normalized units
        row("off y mean", String(format: "%+.4f [%+.4f…%+.4f]",
                                 g.verticalPupilOffset.mean,
                                 g.verticalPupilOffset.minimum, g.verticalPupilOffset.maximum))
        row("raw gaze", String(format: "y%+.1f p%+.1f", s?.rawYawDegrees ?? 0, s?.rawPitchDegrees ?? 0))
        row("calibrated", calibratedText(s), warn: controller.calibration == nil)
        row("asked", String(format: "y%+.1f p%+.1f (%.1f°)",
                            s?.requestedYawDegrees ?? 0, s?.requestedPitchDegrees ?? 0,
                            s?.requestedMagnitudeDegrees ?? 0))
        row("angle ×", String(format: "%.2f", s?.angleFactor ?? 0),
            warn: (s?.angleFactor ?? 1) < 1)
        row("correct", String(format: "%.0f%%", (s?.blendStrength ?? 0) * 100))
        row("iris move", String(format: "%.1f px", s?.irisTravelPixels ?? 0),
            warn: (s?.blendStrength ?? 0) > 0.5 && (s?.irisTravelPixels ?? 0) < 2)
        row("corr age", String(format: "%.1f / %.1f ms", g.correctionAgeMeanMs, g.correctionAgeP95Ms),
            warn: g.correctionAgeP95Ms >= 50)
        row("face conf", String(format: "%.2f/%.2f/%.2f", g.faceConfidence.minimum,
                                g.faceConfidence.mean, g.faceConfidence.maximum))
        row("gaze conf", String(format: "%.2f/%.2f/%.2f", g.gazeConfidence.minimum,
                                g.gazeConfidence.mean, g.gazeConfidence.maximum))
        row("fallback", s?.fallback.rawValue ?? "none", warn: (s?.fallback ?? .none) != .none)
        row("worst gate", g.dominantFallback.rawValue, warn: g.dominantFallback != .none)
        row("corrected", String(format: "%.0f%% of %llu", correctedShare(g) * 100, g.frames))
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

    private func row(_ label: String, _ value: String, warn: Bool = false) -> some View {
        HStack {
            Text(label).foregroundStyle(.white.opacity(0.6))
            Spacer()
            Text(value).foregroundStyle(warn ? .orange : .white)
        }
    }
}
