import CoreImage
import CoreVideo
import Foundation
import os
import AspectusKit

/// writes a bounded set of paired eye crops for correction-quality review
///
/// enabled only with `--quality-capture <directory>`; normal runs allocate nothing, and an active
/// recorder permits one background write at a time so evidence collection cannot queue frames
final class CorrectionQualityRecorder: @unchecked Sendable {
    private struct State {
        var captured = 0
        var nextCaptureAt = 0.0
        var writeInFlight = false
    }

    private static let maximumSamples = 12
    private static let intervalSeconds = 0.75

    private let directory: URL
    private let label: String
    private let started = HostClock.seconds
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let colourSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    private let state = OSAllocatedUnfairLock(initialState: State())
    private let queue = DispatchQueue(label: "com.aspectus.quality-capture", qos: .utility)
    private let log = Logger(subsystem: "com.aspectus.app", category: "quality")

    private static let manifestHeader = """
        sample,label,frame_id,elapsed_s,head_yaw_deg,head_pitch_deg,head_roll_deg,face_conf,gaze_conf,\
        open_l,open_r,raw_yaw_deg,raw_pitch_deg,cal_yaw_deg,cal_pitch_deg,req_yaw_deg,req_pitch_deg,\
        req_mag_deg,angle_factor,blend,iris_px,age_ms,fallback,original,corrected

        """

    static func fromLaunchArguments(
        _ arguments: [String] = CommandLine.arguments
    ) -> CorrectionQualityRecorder? {
        guard let flag = arguments.firstIndex(of: "--quality-capture"),
              arguments.index(after: flag) < arguments.endIndex else { return nil }
        let labelFlag = arguments.firstIndex(of: "--quality-label")
        let label = labelFlag.flatMap { index -> String? in
            let value = arguments.index(after: index)
            return value < arguments.endIndex ? arguments[value] : nil
        } ?? "unlabelled"
        return CorrectionQualityRecorder(path: arguments[arguments.index(after: flag)], label: label)
    }

    init?(path: String, label: String) {
        directory = URL(fileURLWithPath: (path as NSString).expandingTildeInPath,
                        isDirectory: true)
        self.label = label
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let manifest = directory.appendingPathComponent("manifest.csv")
            guard !FileManager.default.fileExists(atPath: manifest.path) else { return nil }
            try Data(Self.manifestHeader.utf8).write(
                to: manifest, options: .withoutOverwriting)
        } catch {
            return nil
        }
    }

    func record(original: CVReadyFrame, corrected: CVReadyFrame,
                tracking: TrackingResult, sample: GazeSample) {
        let now = HostClock.seconds
        let index = state.withLock { state -> Int? in
            guard state.captured < Self.maximumSamples, !state.writeInFlight,
                  now >= state.nextCaptureAt else { return nil }
            state.captured += 1
            state.nextCaptureAt = now + Self.intervalSeconds
            state.writeInFlight = true
            return state.captured
        }
        guard let index else { return }

        queue.async { [self] in
            defer { state.withLock { $0.writeInFlight = false } }
            do {
                try write(index: index, elapsed: now - started, original: original,
                          corrected: corrected, tracking: tracking, sample: sample)
            } catch {
                log.error("quality capture failed: \(error.localizedDescription, privacy: .public)")
            }
        }
    }

    private func write(index: Int, elapsed: Double,
                       original: CVReadyFrame, corrected: CVReadyFrame,
                       tracking: TrackingResult, sample: GazeSample) throws {
        let stem = String(format: "sample-%02d", index)
        let originalName = "\(stem)-original.png"
        let correctedName = "\(stem)-corrected.png"
        let crop = eyeCrop(tracking, width: original.width, height: original.height)
        try writePNG(original.pixelBuffer, crop: crop,
                     to: directory.appendingPathComponent(originalName))
        try writePNG(corrected.pixelBuffer, crop: crop,
                     to: directory.appendingPathComponent(correctedName))

        let row = [
            "\(index)", csv(label), "\(sample.frameID.value)", fmt(elapsed),
            fmt(sample.headYawDegrees), fmt(sample.headPitchDegrees), fmt(sample.headRollDegrees),
            fmt(sample.faceConfidence), optional(sample.gazeConfidence),
            optional(sample.left?.openness), optional(sample.right?.openness),
            fmt(sample.rawYawDegrees), fmt(sample.rawPitchDegrees),
            optional(sample.calibratedYawDegrees), optional(sample.calibratedPitchDegrees),
            fmt(sample.requestedYawDegrees), fmt(sample.requestedPitchDegrees),
            fmt(sample.requestedMagnitudeDegrees), fmt(sample.angleFactor),
            fmt(sample.blendStrength), fmt(sample.irisTravelPixels), fmt(sample.correctionAgeMs),
            sample.fallback.rawValue, originalName, correctedName,
        ].joined(separator: ",") + "\n"
        let manifest = directory.appendingPathComponent("manifest.csv")
        let handle = try FileHandle(forWritingTo: manifest)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(row.utf8))
    }

    private func eyeCrop(_ tracking: TrackingResult, width: Int, height: Int) -> CGRect {
        let left = tracking.leftEye.region
        let right = tracking.rightEye.region
        let minX = min(left.x, right.x)
        let minY = min(left.y, right.y)
        let maxX = max(left.x + left.width, right.x + right.width)
        let maxY = max(left.y + left.height, right.y + right.height)
        let horizontalPadding = (maxX - minX) * 0.25
        let verticalPadding = max(left.height, right.height) * 1.5
        let x0 = max(0, minX - horizontalPadding)
        let x1 = min(1, maxX + horizontalPadding)
        let y0 = max(0, minY - verticalPadding)
        let y1 = min(1, maxY + verticalPadding)
        return CGRect(x: x0 * Double(width),
                      y: (1 - y1) * Double(height),
                      width: (x1 - x0) * Double(width),
                      height: (y1 - y0) * Double(height)).integral
    }

    private func writePNG(_ buffer: CVPixelBuffer, crop: CGRect, to url: URL) throws {
        let image = CIImage(cvPixelBuffer: buffer).cropped(to: crop)
        try context.writePNGRepresentation(of: image, to: url, format: .RGBA8,
                                           colorSpace: colourSpace)
    }

    private func fmt(_ value: Double) -> String { String(format: "%.3f", value) }
    private func optional(_ value: Double?) -> String { value.map(fmt) ?? "" }
    private func csv(_ value: String) -> String {
        "\"\(value.replacingOccurrences(of: "\"", with: "\"\""))\""
    }
}
