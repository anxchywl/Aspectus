import Foundation
import AspectusKit

/// reads and writes the calibration in Application Support
///
/// the file holds derived angles only — no imagery, no landmark coordinates — and never leaves the
/// machine. there is no network code in this type or anything it calls
struct CalibrationStore {
    struct Stored: Codable {
        var calibration: GazeCalibration
        /// kept so the fit can be revisited without asking the user to sit through it again
        var samples: [CalibrationSample]
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
        // atomic so a crash mid-write cannot leave a half-parsed calibration behind
        try encoder.encode(stored).write(to: url, options: .atomic)
    }

    /// reset means the data is gone from disk, not just unused
    func delete() {
        guard let url = fileURL else { return }
        try? FileManager.default.removeItem(at: url)
    }
}
