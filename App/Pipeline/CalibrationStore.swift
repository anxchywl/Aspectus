import Foundation
import AspectusKit

/// reads and writes the calibration in Application Support
///
/// the file holds derived angles only — no imagery, no landmark coordinates — and never leaves the
/// machine. there is no network code in this type or anything it calls
struct CalibrationStore {
    private struct Attempt: Codable {
        var createdAt: Date
        var failure: String
        var samples: [CalibrationSample]
        var headMotionSamples: [HeadMotionSample]
    }

    struct Stored: Codable {
        var calibration: GazeCalibration
        /// kept so the fit can be revisited without asking the user to sit through it again
        var samples: [CalibrationSample]
        /// lens-fixation sweep retained as angle-only diagnostic evidence
        var headMotionSamples: [HeadMotionSample]

        init(calibration: GazeCalibration,
             samples: [CalibrationSample],
             headMotionSamples: [HeadMotionSample] = []) {
            self.calibration = calibration
            self.samples = samples
            self.headMotionSamples = headMotionSamples
        }

        private enum CodingKeys: String, CodingKey {
            case calibration, samples, headMotionSamples
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            calibration = try container.decode(GazeCalibration.self, forKey: .calibration)
            samples = try container.decode([CalibrationSample].self, forKey: .samples)
            headMotionSamples = try container.decodeIfPresent([HeadMotionSample].self,
                                                               forKey: .headMotionSamples) ?? []
        }
    }

    private let directoryName = "Aspectus"
    private let fileName = "calibration.json"

    var fileURL: URL? {
        guard let base = try? FileManager.default.url(for: .applicationSupportDirectory,
                                                      in: .userDomainMask,
                                                      appropriateFor: nil,
                                                      create: true) else { return nil }
        return base.appendingPathComponent(directoryName, isDirectory: true)
            .appendingPathComponent(fileName, isDirectory: false)
    }

    /// nil for "no calibration", including every failure mode — a corrupt or future-version file
    /// must behave exactly like an uncalibrated install rather than steering correction
    func load() -> Stored? {
        guard let url = fileURL, let data = try? Data(contentsOf: url) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard let stored = try? decoder.decode(Stored.self, from: data),
              stored.calibration.isUsable else { return nil }
        return stored
    }

    func save(_ stored: Stored) throws {
        guard let url = fileURL else { throw CocoaError(.fileNoSuchFile) }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        if FileManager.default.fileExists(atPath: url.path) {
            let backupDirectory = url.deletingLastPathComponent()
                .appendingPathComponent("calibration-backups", isDirectory: true)
            try FileManager.default.createDirectory(at: backupDirectory,
                                                    withIntermediateDirectories: true)
            let backup = backupDirectory.appendingPathComponent(
                "calibration-\(Int(Date().timeIntervalSince1970))-\(UUID().uuidString).json")
            try FileManager.default.copyItem(at: url, to: backup)
        }
        // atomic so a crash mid-write cannot leave a half-parsed calibration behind
        try encoder.encode(stored).write(to: url, options: .atomic)
    }

    func saveAttempt(samples: [CalibrationSample],
                     headMotionSamples: [HeadMotionSample],
                     failure: String) throws {
        guard let url = fileURL else { throw CocoaError(.fileNoSuchFile) }
        let directory = url.deletingLastPathComponent()
            .appendingPathComponent("calibration-attempts", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let destination = directory.appendingPathComponent(
            "attempt-\(Int(Date().timeIntervalSince1970))-\(UUID().uuidString).json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let attempt = Attempt(createdAt: Date(), failure: failure,
                              samples: samples, headMotionSamples: headMotionSamples)
        try encoder.encode(attempt).write(to: destination, options: .atomic)
    }

    /// reset means the data is gone from disk, not just unused
    func delete() {
        guard let url = fileURL else { return }
        try? FileManager.default.removeItem(at: url)
        try? FileManager.default.removeItem(
            at: url.deletingLastPathComponent()
                .appendingPathComponent("calibration-backups", isDirectory: true))
        try? FileManager.default.removeItem(
            at: url.deletingLastPathComponent()
                .appendingPathComponent("calibration-attempts", isDirectory: true))
    }
}
