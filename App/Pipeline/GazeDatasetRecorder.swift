import CoreImage
import CoreVideo
import Foundation
import os
import AspectusKit

/// explicit biometric-data recorder for training the appearance-based gaze estimator
///
/// inactive during normal use; an active session permits one background PNG write at a time and
/// keeps every image under the user's Application Support directory with owner-only permissions
final class GazeDatasetRecorder: @unchecked Sendable {
    enum Status: Sendable, Equatable {
        case collecting
        case finished
        case cancelled
        case failed(String)
    }

    struct Snapshot: Sendable, Equatable {
        var sessionID: String
        var split: GazeDatasetSplit
        var status: Status
        var target: GazeDatasetTarget?
        var targetNumber: Int
        var targetCount: Int
        var samplesForTarget: Int
        var totalSamples: Int
        var rejection: GazeDatasetRejection?
        var directoryPath: String

        var progress: Double {
            guard targetCount > 0 else { return 0 }
            let partial = Double(samplesForTarget) / Double(GazeDatasetPlan.samplesPerTarget)
            return min(1, (Double(targetNumber) + partial) / Double(targetCount))
        }
    }

    enum StartError: Error, LocalizedError {
        case alreadyCollecting
        case applicationSupportUnavailable
        case displayGeometryUnavailable

        var errorDescription: String? {
            switch self {
            case .alreadyCollecting: return "a gaze dataset session is already running"
            case .applicationSupportUnavailable: return "Application Support is unavailable"
            case .displayGeometryUnavailable: return "the display geometry is unavailable"
            }
        }
    }

    private struct Metadata: Codable {
        var schemaVersion: Int
        var participantID: String
        var sessionID: String
        var split: GazeDatasetSplit
        var createdAt: Date
        var completedAt: Date?
        var status: String
        var displayGeometry: GazeDatasetGeometry
        var eyeImageWidth: Int
        var eyeImageHeight: Int
        var samplesPerTarget: Int
        var targetCount: Int
        var capturedSamples: Int
        var cameraFormat: String
    }

    private struct Session {
        var participantID: String
        var id: String
        var split: GazeDatasetSplit
        var geometry: GazeDatasetGeometry
        var directory: URL
        var cameraFormat: String
        var targets: [GazeDatasetTarget]
        var targetIndex = 0
        var samplesForTarget = 0
        var totalSamples = 0
        var targetStartedAt: Double
        var lastCaptureAt = -Double.infinity
        var writeInFlight = false
        var rejection: GazeDatasetRejection?
        var status: Status = .collecting
        var createdAt = Date()
        var startedAtHost: Double

        var isCollecting: Bool { status == .collecting }
    }

    private struct Reservation: Sendable {
        var participantID: String
        var sessionID: String
        var split: GazeDatasetSplit
        var directory: URL
        var target: GazeDatasetTarget
        var angles: (yaw: Double, pitch: Double)
        var sampleNumber: Int
        var elapsed: Double
    }

    private struct State {
        var session: Session?
    }

    private static let imageWidth = 60
    private static let imageHeight = 60
    private static let manifestHeader = """
        schema_version,participant_id,session_id,split,sample,frame_id,elapsed_s,target_id,target_kind,\
        target_x,target_y,target_yaw_deg,target_pitch_deg,pose_prompt,head_yaw_deg,head_pitch_deg,\
        head_roll_deg,face_conf,open_l,open_r,left_image,right_image

        """

    private let state = OSAllocatedUnfairLock(initialState: State())
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let colourSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    private let queue = DispatchQueue(label: "com.aspectus.gaze-dataset", qos: .utility)
    private let log = Logger(subsystem: "com.aspectus.app", category: "gaze-dataset")

    var isCollecting: Bool {
        state.withLock { $0.session?.isCollecting == true }
    }

    var rootURL: URL? {
        guard let base = try? FileManager.default.url(for: .applicationSupportDirectory,
                                                      in: .userDomainMask,
                                                      appropriateFor: nil,
                                                      create: true) else { return nil }
        return base.appendingPathComponent("Aspectus", isDirectory: true)
            .appendingPathComponent("gaze-datasets", isDirectory: true)
    }

    func start(split: GazeDatasetSplit, geometry: GazeDatasetGeometry?,
               cameraFormat: String, now: Double = HostClock.seconds) throws {
        guard !isCollecting else { throw StartError.alreadyCollecting }
        queue.sync {}
        guard let geometry, geometry.isUsable else { throw StartError.displayGeometryUnavailable }
        guard let rootURL else { throw StartError.applicationSupportUnavailable }

        let fileManager = FileManager.default
        try fileManager.createDirectory(at: rootURL, withIntermediateDirectories: true,
                                        attributes: [.posixPermissions: 0o700])
        try fileManager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: rootURL.path)
        let participantID = try loadOrCreateParticipantID(in: rootURL)
        let sessionID = UUID().uuidString
        let timestamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        let directory = rootURL.appendingPathComponent(
            "\(split.rawValue)-\(timestamp)-\(sessionID.prefix(8))", isDirectory: true)
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: false,
                                        attributes: [.posixPermissions: 0o700])

        let manifest = directory.appendingPathComponent("manifest.csv")
        try Data(Self.manifestHeader.utf8).write(to: manifest, options: .withoutOverwriting)
        try fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: manifest.path)

        let session = Session(participantID: participantID, id: sessionID, split: split,
                              geometry: geometry, directory: directory,
                              cameraFormat: cameraFormat,
                              targets: GazeDatasetPlan.targets(for: split),
                              targetStartedAt: now, startedAtHost: now)
        try writeMetadata(session, completedAt: nil)
        state.withLock { $0.session = session }
    }

    func cancel() {
        let cancelledID = state.withLock { state -> String? in
            guard var session = state.session, session.isCollecting else { return nil }
            session.status = .cancelled
            state.session = session
            return session.id
        }
        guard let cancelledID else { return }
        queue.async { [self] in
            let cancelled = state.withLock { state -> Session? in
                guard let session = state.session, session.id == cancelledID else { return nil }
                return session
            }
            if let cancelled { try? writeMetadata(cancelled, completedAt: Date()) }
        }
    }

    func snapshot() -> Snapshot? {
        state.withLock { state in
            guard let session = state.session else { return nil }
            return Snapshot(sessionID: session.id, split: session.split,
                            status: session.status,
                            target: session.targets.indices.contains(session.targetIndex)
                                ? session.targets[session.targetIndex] : nil,
                            targetNumber: session.targetIndex,
                            targetCount: session.targets.count,
                            samplesForTarget: session.samplesForTarget,
                            totalSamples: session.totalSamples,
                            rejection: session.rejection,
                            directoryPath: session.directory.path)
        }
    }

    func record(_ frame: CVReadyFrame, tracking: TrackingResult?,
                now: Double = HostClock.seconds) {
        let reservation = state.withLock { state -> Reservation? in
            guard var session = state.session, session.isCollecting,
                  session.targets.indices.contains(session.targetIndex) else { return nil }
            if let rejection = GazeDatasetAcceptance.rejection(tracking) {
                session.rejection = rejection
                state.session = session
                return nil
            }
            guard let tracking else { return nil }
            let target = session.targets[session.targetIndex]
            let previousPose = session.targetIndex > 0
                ? session.targets[session.targetIndex - 1].pose : nil
            let settle = previousPose == target.pose
                ? GazeDatasetPlan.settleSeconds : GazeDatasetPlan.poseTransitionSettleSeconds
            guard now - session.targetStartedAt >= settle,
                  now - session.lastCaptureAt >= GazeDatasetPlan.captureIntervalSeconds,
                  !session.writeInFlight,
                  let angles = session.geometry.angles(for: target) else {
                state.session = session
                return nil
            }
            session.writeInFlight = true
            session.lastCaptureAt = now
            session.rejection = nil
            state.session = session
            _ = tracking
            return Reservation(participantID: session.participantID,
                               sessionID: session.id, split: session.split,
                               directory: session.directory, target: target,
                               angles: angles, sampleNumber: session.totalSamples + 1,
                               elapsed: now - session.startedAtHost)
        }
        guard let reservation, let tracking else { return }

        queue.async { [self] in
            do {
                try write(reservation, frame: frame, tracking: tracking)
                completeWrite(sessionID: reservation.sessionID, now: HostClock.seconds)
            } catch {
                failWrite(sessionID: reservation.sessionID, message: error.localizedDescription)
                log.error("gaze dataset write failed: \(error.localizedDescription, privacy: .public)")
            }
        }
    }

    func deleteAll() throws {
        guard !isCollecting else { throw StartError.alreadyCollecting }
        guard let rootURL else { throw StartError.applicationSupportUnavailable }
        queue.sync {}
        if FileManager.default.fileExists(atPath: rootURL.path) {
            try FileManager.default.removeItem(at: rootURL)
        }
        state.withLock { $0.session = nil }
    }

    private func completeWrite(sessionID: String, now: Double) {
        let completed = state.withLock { state -> Session? in
            guard var session = state.session, session.id == sessionID else { return nil }
            session.writeInFlight = false
            session.totalSamples += 1
            session.samplesForTarget += 1
            if session.samplesForTarget >= GazeDatasetPlan.samplesPerTarget {
                session.targetIndex += 1
                session.samplesForTarget = 0
                session.targetStartedAt = now
            }
            if session.targetIndex >= session.targets.count {
                session.status = .finished
            }
            state.session = session
            return session.isCollecting ? nil : session
        }
        if let completed { try? writeMetadata(completed, completedAt: Date()) }
    }

    private func failWrite(sessionID: String, message: String) {
        let failed = state.withLock { state -> Session? in
            guard var session = state.session, session.id == sessionID else { return nil }
            session.writeInFlight = false
            session.status = .failed(message)
            state.session = session
            return session
        }
        if let failed { try? writeMetadata(failed, completedAt: Date()) }
    }

    private func write(_ reservation: Reservation, frame: CVReadyFrame,
                       tracking: TrackingResult) throws {
        let stem = String(format: "sample-%05d", reservation.sampleNumber)
        let leftName = "\(stem)-left.png"
        let rightName = "\(stem)-right.png"
        try writeEye(frame.pixelBuffer, eye: tracking.leftEye,
                     to: reservation.directory.appendingPathComponent(leftName))
        try writeEye(frame.pixelBuffer, eye: tracking.rightEye,
                     to: reservation.directory.appendingPathComponent(rightName))

        let degrees = 180.0 / Double.pi
        let row = [
            "1", csv(reservation.participantID), csv(reservation.sessionID),
            reservation.split.rawValue, "\(reservation.sampleNumber)",
            "\(frame.header.id.value)", fmt(reservation.elapsed),
            "\(reservation.target.id)", reservation.target.kind.rawValue,
            fmt(reservation.target.xFraction), fmt(reservation.target.yFraction),
            fmt(reservation.angles.yaw), fmt(reservation.angles.pitch),
            reservation.target.pose.rawValue,
            fmt(tracking.headPose.yaw * degrees), fmt(tracking.headPose.pitch * degrees),
            fmt(tracking.headPose.roll * degrees), fmt(tracking.confidence),
            fmt(tracking.leftEye.openness), fmt(tracking.rightEye.openness),
            leftName, rightName,
        ].joined(separator: ",") + "\n"
        let manifest = reservation.directory.appendingPathComponent("manifest.csv")
        let handle = try FileHandle(forWritingTo: manifest)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(row.utf8))
    }

    private func writeEye(_ buffer: CVPixelBuffer, eye: EyeObservation, to url: URL) throws {
        let crop = eyeCrop(eye, width: CVPixelBufferGetWidth(buffer),
                           height: CVPixelBufferGetHeight(buffer))
        let image = CIImage(cvPixelBuffer: buffer).cropped(to: crop)
            .transformed(by: CGAffineTransform(translationX: -crop.minX, y: -crop.minY))
            .transformed(by: CGAffineTransform(scaleX: Double(Self.imageWidth) / crop.width,
                                               y: Double(Self.imageHeight) / crop.height))
        try context.writePNGRepresentation(of: image, to: url, format: .RGBA8,
                                           colorSpace: colourSpace)
        try FileManager.default.setAttributes([.posixPermissions: 0o600],
                                              ofItemAtPath: url.path)
    }

    private func eyeCrop(_ eye: EyeObservation, width: Int, height: Int) -> CGRect {
        let centre = eye.region.center
        let side = max(eye.region.width * 1.8, eye.region.height * 3.6)
        let x0 = max(0, centre.x - side / 2)
        let x1 = min(1, centre.x + side / 2)
        let y0 = max(0, centre.y - side / 2)
        let y1 = min(1, centre.y + side / 2)
        return CGRect(x: x0 * Double(width), y: (1 - y1) * Double(height),
                      width: (x1 - x0) * Double(width),
                      height: (y1 - y0) * Double(height)).integral
    }

    private func loadOrCreateParticipantID(in root: URL) throws -> String {
        let url = root.appendingPathComponent("participant-id.txt")
        if let value = try? String(contentsOf: url, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines), !value.isEmpty {
            return value
        }
        let value = UUID().uuidString
        try Data((value + "\n").utf8).write(to: url, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600],
                                              ofItemAtPath: url.path)
        return value
    }

    private func writeMetadata(_ session: Session, completedAt: Date?) throws {
        let status: String
        switch session.status {
        case .collecting: status = "collecting"
        case .finished: status = "finished"
        case .cancelled: status = "cancelled"
        case .failed: status = "failed"
        }
        let metadata = Metadata(schemaVersion: 1, participantID: session.participantID,
                                sessionID: session.id, split: session.split,
                                createdAt: session.createdAt, completedAt: completedAt,
                                status: status, displayGeometry: session.geometry,
                                eyeImageWidth: Self.imageWidth, eyeImageHeight: Self.imageHeight,
                                samplesPerTarget: GazeDatasetPlan.samplesPerTarget,
                                targetCount: session.targets.count,
                                capturedSamples: session.totalSamples,
                                cameraFormat: session.cameraFormat)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let url = session.directory.appendingPathComponent("session.json")
        try encoder.encode(metadata).write(to: url, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600],
                                              ofItemAtPath: url.path)
    }

    private func fmt(_ value: Double) -> String { String(format: "%.6f", value) }
    private func csv(_ value: String) -> String {
        "\"\(value.replacingOccurrences(of: "\"", with: "\"\""))\""
    }
}
