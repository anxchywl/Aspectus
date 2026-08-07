import Foundation
import AspectusKit

/// appends periodic metric snapshots to a CSV so every reported number has a release-build record
/// enabled with `--benchmark <path>`, sampled on the stats timer and never on the hot path
final class BenchmarkRecorder {
    struct Sample {
        let captureFPS: Double
        let processFPS: Double
        let outputFPS: Double
        let tracking: StageMetrics.Snapshot
        let vision: StageMetrics.Snapshot
        let correction: StageMetrics.Snapshot
        let pipeline: StageMetrics.Snapshot
        let processing: StageMetrics.Snapshot
        let endToEnd: StageMetrics.Snapshot
        let dropped: Int
        let depth: Int
        let memoryMB: Double
        let thermal: String
        let gaze: DiagnosticsCollector.Snapshot
        let vcamState: String
        let vcamSent: Int
        let vcamDropped: Int
        let vcamPaced: Int
    }

    private let url: URL
    private let started = HostClock.seconds
    private var handle: FileHandle?

    private static let header = """
        elapsed_s,capture_fps,process_fps,output_fps,\
        track_mean_ms,track_p95_ms,track_max_ms,\
        vision_mean_ms,vision_p95_ms,vision_max_ms,\
        warp_mean_ms,warp_p95_ms,warp_max_ms,\
        pipe_mean_ms,pipe_p95_ms,pipe_max_ms,\
        proc_mean_ms,proc_p95_ms,proc_max_ms,\
        e2e_mean_ms,e2e_p95_ms,e2e_max_ms,\
        presented,dropped,depth,memory_mb,thermal,\
        pupil_src_l,pupil_src_r,pupil_pts_l,pupil_pts_r,vision_pupil_share,head_pose_share,\
        off_x_l,off_y_l,off_x_r,off_y_r,off_y_mean,off_y_min,off_y_max,\
        reg_cy_l,reg_h_l,pup_y_l,corner_y_l,reg_cy_r,reg_h_r,pup_y_r,corner_y_r,\
        open_l,open_r,head_yaw_deg,head_pitch_deg,head_roll_deg,\
        raw_yaw_deg,raw_pitch_deg,req_yaw_deg,req_pitch_deg,req_mag_deg,\
        angle_factor,blend,iris_px,age_mean_ms,age_p95_ms,\
        face_conf_min,face_conf_mean,face_conf_max,\
        gaze_conf_min,gaze_conf_mean,gaze_conf_max,\
        fallback,dominant_fallback,corrected_frames,gaze_frames,\
        vcam_state,vcam_sent,vcam_dropped,vcam_paced

        """

    /// nil unless --benchmark <path> was passed, so normal runs pay nothing
    static func fromLaunchArguments(_ arguments: [String] = CommandLine.arguments) -> BenchmarkRecorder? {
        guard let flag = arguments.firstIndex(of: "--benchmark"),
              arguments.index(after: flag) < arguments.endIndex else { return nil }
        return BenchmarkRecorder(path: arguments[arguments.index(after: flag)])
    }

    init?(path: String) {
        url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        guard FileManager.default.createFile(atPath: url.path,
                                             contents: Self.header.data(using: .utf8)) else { return nil }
        handle = try? FileHandle(forWritingTo: url)
        handle?.seekToEndOfFile()
    }

    /// FileHandle.write is unbuffered, so every sample is durable without an explicit close
    func record(_ s: Sample) {
        let row = [
            fmt(HostClock.seconds - started), fmt(s.captureFPS), fmt(s.processFPS), fmt(s.outputFPS),
            fmt(s.tracking.meanMs), fmt(s.tracking.p95Ms), fmt(s.tracking.maxMs),
            fmt(s.vision.meanMs), fmt(s.vision.p95Ms), fmt(s.vision.maxMs),
            fmt(s.correction.meanMs), fmt(s.correction.p95Ms), fmt(s.correction.maxMs),
            fmt(s.pipeline.meanMs), fmt(s.pipeline.p95Ms), fmt(s.pipeline.maxMs),
            fmt(s.processing.meanMs), fmt(s.processing.p95Ms), fmt(s.processing.maxMs),
            fmt(s.endToEnd.meanMs), fmt(s.endToEnd.p95Ms), fmt(s.endToEnd.maxMs),
            "\(s.endToEnd.processed)", "\(s.dropped)", "\(s.depth)",
            fmt(s.memoryMB), s.thermal,
        ].joined(separator: ",") + "," + gazeColumns(s.gaze)
            + ",\(s.vcamState),\(s.vcamSent),\(s.vcamDropped),\(s.vcamPaced)"
        handle?.write(Data((row + "\n").utf8))
    }

    /// the phase 1 diagnostic columns, written even when there is no face so a run's gaps are
    /// visible in the file rather than showing up as missing rows
    private func gazeColumns(_ g: DiagnosticsCollector.Snapshot) -> String {
        let s = g.latest
        return [
            s?.left?.source.rawValue ?? "none", s?.right?.source.rawValue ?? "none",
            "\(s?.left?.pupilPointCount ?? 0)", "\(s?.right?.pupilPointCount ?? 0)",
            fmt(g.visionPupilShare), fmt(g.headPoseShare),
            // normalized offsets are a few thousandths of the frame, so three decimals would
            // quantize the eyelid bias away entirely
            fine(s?.left?.pupilOffset.x ?? 0), fine(s?.left?.pupilOffset.y ?? 0),
            fine(s?.right?.pupilOffset.x ?? 0), fine(s?.right?.pupilOffset.y ?? 0),
            fine(g.verticalPupilOffset.mean), fine(g.verticalPupilOffset.minimum),
            fine(g.verticalPupilOffset.maximum),
            fine(s?.left?.region.center.y ?? 0), fine(s?.left?.region.height ?? 0),
            fine(s?.left?.pupil.y ?? 0), fine(s?.left?.cornerMidpointY ?? 0),
            fine(s?.right?.region.center.y ?? 0), fine(s?.right?.region.height ?? 0),
            fine(s?.right?.pupil.y ?? 0), fine(s?.right?.cornerMidpointY ?? 0),
            fmt(s?.left?.openness ?? 0), fmt(s?.right?.openness ?? 0),
            fmt(s?.headYawDegrees ?? 0), fmt(s?.headPitchDegrees ?? 0), fmt(s?.headRollDegrees ?? 0),
            fmt(s?.rawYawDegrees ?? 0), fmt(s?.rawPitchDegrees ?? 0),
            fmt(s?.requestedYawDegrees ?? 0), fmt(s?.requestedPitchDegrees ?? 0),
            fmt(s?.requestedMagnitudeDegrees ?? 0),
            fmt(s?.angleFactor ?? 0), fmt(s?.blendStrength ?? 0), fmt(s?.irisTravelPixels ?? 0),
            fmt(g.correctionAgeMeanMs), fmt(g.correctionAgeP95Ms),
            fmt(g.faceConfidence.minimum), fmt(g.faceConfidence.mean), fmt(g.faceConfidence.maximum),
            fmt(g.gazeConfidence.minimum), fmt(g.gazeConfidence.mean), fmt(g.gazeConfidence.maximum),
            s?.fallback.rawValue ?? "none", g.dominantFallback.rawValue,
            "\(g.correctedFrames)", "\(g.frames)",
        ].joined(separator: ",")
    }

    private func fmt(_ v: Double) -> String { String(format: "%.3f", v) }
    private func fine(_ v: Double) -> String { String(format: "%.6f", v) }
}
