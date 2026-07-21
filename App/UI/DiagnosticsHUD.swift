import SwiftUI

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
            row("dropped", "\(controller.droppedFrames)")
            row("in-flight", "\(controller.inFlight)")
            row("memory", String(format: "%.0f MB", controller.memoryMB))
            row("thermal", controller.thermalState, warn: controller.thermalState != "nominal")
            Divider().overlay(Color.white.opacity(0.2))
            Text(controller.formatDescription)
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.white.opacity(0.7))
                .lineLimit(2)
        }
        .padding(10)
        .frame(width: 240, alignment: .leading)
        .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 10))
        .font(.system(size: 12, weight: .medium, design: .monospaced))
    }

    private func row(_ label: String, _ value: String, warn: Bool = false) -> some View {
        HStack {
            Text(label).foregroundStyle(.white.opacity(0.6))
            Spacer()
            Text(value).foregroundStyle(warn ? .orange : .white)
        }
    }
}
