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
        let correction: StageMetrics.Snapshot
        let processing: StageMetrics.Snapshot
        let endToEnd: StageMetrics.Snapshot
        let dropped: Int
        let depth: Int
        let memoryMB: Double
        let thermal: String
    }

    private let url: URL
    private let started = HostClock.seconds
    private var handle: FileHandle?

    private static let header = """
        elapsed_s,capture_fps,process_fps,output_fps,\
        track_mean_ms,track_p95_ms,track_max_ms,\
        warp_mean_ms,warp_p95_ms,warp_max_ms,\
        proc_mean_ms,proc_p95_ms,proc_max_ms,\
        e2e_mean_ms,e2e_p95_ms,e2e_max_ms,\
        presented,dropped,depth,memory_mb,thermal

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
            fmt(s.correction.meanMs), fmt(s.correction.p95Ms), fmt(s.correction.maxMs),
            fmt(s.processing.meanMs), fmt(s.processing.p95Ms), fmt(s.processing.maxMs),
            fmt(s.endToEnd.meanMs), fmt(s.endToEnd.p95Ms), fmt(s.endToEnd.maxMs),
            "\(s.endToEnd.processed)", "\(s.dropped)", "\(s.depth)",
            fmt(s.memoryMB), s.thermal,
        ].joined(separator: ",")
        handle?.write(Data((row + "\n").utf8))
    }

    private func fmt(_ v: Double) -> String { String(format: "%.3f", v) }
}
