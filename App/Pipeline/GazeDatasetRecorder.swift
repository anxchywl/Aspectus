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
        var guidance: GazePosePromptGate.Guidance?
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

    private enum WriteError: Error, LocalizedError {
        case invalidEyeEvidence
        case invalidManifestContract

        var errorDescription: String? {
            switch self {
            case .invalidEyeEvidence:
                return "the eye alignment evidence is unavailable for the sample"
            case .invalidManifestContract:
                return "the schema 5 manifest fields do not match the declared contract"
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
        var sourceImageWidth: Int? = nil
        var sourceImageHeight: Int? = nil
        var cropContract: GazeDatasetCropContract
        var labelContract: GazeDatasetLabelContract
        var headPoseContract: GazeDatasetHeadPoseContract
    }

    private struct Session {
        var participantID: String
        var id: String
        var split: GazeDatasetSplit
        var geometry: GazeDatasetGeometry
        var directory: URL
        var cameraFormat: String
        var sourceImageWidth: Int? = nil
        var sourceImageHeight: Int? = nil
        var targets: [GazeDatasetTarget]
        var targetIndex = 0
        var samplesForTarget = 0
        var totalSamples = 0
        var lastCaptureAt = -Double.infinity
        var writeInFlight = false
        var rejection: GazeDatasetRejection?
        var guidance: GazePosePromptGate.Guidance?
        var status: Status = .collecting
        var createdAt = Date()
        var startedAtHost: Double
        var posePromptGate = GazePosePromptGate()
        var settleGate = GazeDatasetSettleGate()

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
        var alignment: GazeDatasetCanonicalAlignment
    }

    private struct State {
        var session: Session?
    }

    private static let imageWidth = GazeDatasetCanonicalAlignment.outputWidth
    private static let imageHeight = GazeDatasetCanonicalAlignment.outputHeight
    private static let manifestHeader =
        GazeDatasetSchema5.manifestColumns.joined(separator: ",") + "\n"

    private let state = OSAllocatedUnfairLock(initialState: State())
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let colourSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    private let queue = DispatchQueue(label: "com.aspectus.gaze-dataset", qos: .utility)
    private let log = Logger(subsystem: "com.aspectus.app", category: "gaze-dataset")
    private let rootOverride: URL?

    // the override exists so tests can exercise the production write path against a
    // temporary directory instead of the user's real biometric store
    init(rootDirectory: URL? = nil) {
        rootOverride = rootDirectory
    }

    var isCollecting: Bool {
        state.withLock { $0.session?.isCollecting == true }
    }

    var rootURL: URL? {
        if let rootOverride { return rootOverride }
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
                              startedAtHost: now)
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
                            guidance: session.guidance,
                            directoryPath: session.directory.path)
        }
    }

    func record(_ frame: CVReadyFrame, tracking: TrackingResult?,
                now: Double = HostClock.seconds) {
        let decision = state.withLock { state -> (reservation: Reservation?, terminalID: String?) in
            guard var session = state.session, session.isCollecting,
                  session.targets.indices.contains(session.targetIndex) else { return (nil, nil) }
            let target = session.targets[session.targetIndex]
            let imageWidth = frame.header.width
            let imageHeight = frame.header.height
            if let recordedWidth = session.sourceImageWidth,
               let recordedHeight = session.sourceImageHeight,
               (recordedWidth != imageWidth || recordedHeight != imageHeight) {
                session.status = .failed("the camera frame size changed during collection")
                state.session = session
                return (nil, session.id)
            }
            session.sourceImageWidth = imageWidth
            session.sourceImageHeight = imageHeight
            if let rejection = GazeDatasetAcceptance.rejection(
                tracking, imageWidth: imageWidth, imageHeight: imageHeight) {
                _ = session.settleGate.isReady(targetID: target.id, accepted: false,
                                               now: now, settleSeconds: 0)
                session.rejection = rejection
                state.session = session
                return (nil, nil)
            }
            guard let tracking else { return (nil, nil) }
            let degrees = 180.0 / Double.pi
            let poseOutcome = session.posePromptGate.evaluate(
                target.pose,
                yawDegrees: tracking.headPose.yaw * degrees,
                pitchDegrees: tracking.headPose.pitch * degrees,
                rollDegrees: tracking.headPose.roll * degrees)
            let poseAccepted = poseOutcome == .accepted
            session.guidance = session.posePromptGate.guidance(
                for: target.pose,
                yawDegrees: tracking.headPose.yaw * degrees,
                pitchDegrees: tracking.headPose.pitch * degrees,
                rollDegrees: tracking.headPose.roll * degrees)
            let previousPose = session.targetIndex > 0
                ? session.targets[session.targetIndex - 1].pose : nil
            let settle = target.kind == .lens ? GazeDatasetPlan.lensSettleSeconds
                : previousPose == target.pose ? GazeDatasetPlan.settleSeconds
                : GazeDatasetPlan.poseTransitionSettleSeconds
            guard session.settleGate.isReady(targetID: target.id, accepted: poseAccepted,
                                              now: now, settleSeconds: settle) else {
                session.rejection = poseOutcome == .overshoot
                    ? .posePromptOvershoot
                    : poseAccepted ? nil : .posePrompt
                state.session = session
                return (nil, nil)
            }
            guard now - session.lastCaptureAt >= GazeDatasetPlan.captureIntervalSeconds,
                  !session.writeInFlight,
                  let angles = session.geometry.angles(for: target),
                  let alignment = GazeDatasetCanonicalAlignment(
                    left: tracking.leftEye, right: tracking.rightEye,
                    imageWidth: imageWidth, imageHeight: imageHeight) else {
                state.session = session
                return (nil, nil)
            }
            session.writeInFlight = true
            session.lastCaptureAt = now
            session.rejection = nil
            state.session = session
            _ = tracking
            return (Reservation(participantID: session.participantID,
                                sessionID: session.id, split: session.split,
                                directory: session.directory, target: target,
                                angles: angles, sampleNumber: session.totalSamples + 1,
                                elapsed: now - session.startedAtHost,
                                alignment: alignment), nil)
        }
        if let terminalID = decision.terminalID {
            queue.async { [self] in
                let terminal = state.withLock { state -> Session? in
                    guard let session = state.session, session.id == terminalID else { return nil }
                    return session
                }
                if let terminal { try? writeMetadata(terminal, completedAt: Date()) }
            }
            return
        }
        guard let reservation = decision.reservation, let tracking else { return }

        queue.async { [self] in
            do {
                try write(reservation, frame: frame, tracking: tracking)
                completeWrite(sessionID: reservation.sessionID)
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

    private func completeWrite(sessionID: String) {
        let completed = state.withLock { state -> Session? in
            guard var session = state.session, session.id == sessionID else { return nil }
            session.writeInFlight = false
            session.totalSamples += 1
            session.samplesForTarget += 1
            if session.samplesForTarget >= GazeDatasetPlan.samplesPerTarget {
                let completedPose = session.targets[session.targetIndex].pose
                session.targetIndex += 1
                session.samplesForTarget = 0
                // The neutral block is what defines the baseline every later block is measured
                // against, so this is the first moment the session can be checked against the pose
                // it actually recorded. Checking here turns a six-minute dead end into a
                // forty-second one.
                if completedPose == .neutral,
                   session.targets.indices.contains(session.targetIndex),
                   session.targets[session.targetIndex].pose != .neutral,
                   let unreachable = Self.unreachableBlockMessage(session: session) {
                    session.status = .failed(unreachable)
                }
            }
            if session.targetIndex >= session.targets.count {
                session.status = .finished
            }
            state.session = session
            return session.isCollecting ? nil : session
        }
        if let completed { try? writeMetadata(completed, completedAt: Date()) }
    }

    /// Reports the first upcoming block the recorded baseline cannot satisfy, naming the axis and
    /// the numbers, so stopping here is self-explanatory rather than another silent dead end.
    private static func unreachableBlockMessage(session: Session) -> String? {
        guard let baseline = session.posePromptGate.baseline else { return nil }
        let posture = GazeDatasetPosture(yawDegrees: baseline.yawDegrees,
                                         pitchDegrees: baseline.pitchDegrees,
                                         rollDegrees: baseline.rollDegrees)
        guard !posture.isReady, let advice = posture.advice else { return nil }
        return String(format: "the tilt blocks cannot be reached from this seating: your neutral "
                      + "baseline recorded %+.0f° of roll, and a tilt has to change roll by 6° "
                      + "while staying inside 15° absolute. %@",
                      baseline.rollDegrees, advice)
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
        guard let leftStart = tracking.leftEye.imageAxisStart,
              let leftEnd = tracking.leftEye.imageAxisEnd,
              let rightStart = tracking.rightEye.imageAxisStart,
              let rightEnd = tracking.rightEye.imageAxisEnd else {
            throw WriteError.invalidEyeEvidence
        }
        let stem = String(format: "sample-%05d", reservation.sampleNumber)
        let leftName = "\(stem)-left.png"
        let rightName = "\(stem)-right.png"
        try writeEye(frame.pixelBuffer, crop: reservation.alignment.left,
                     rotation: reservation.alignment.rotationRadians,
                     side: reservation.alignment.left.cropSidePixels,
                     to: reservation.directory.appendingPathComponent(leftName))
        try writeEye(frame.pixelBuffer, crop: reservation.alignment.right,
                     rotation: reservation.alignment.rotationRadians,
                     side: reservation.alignment.right.cropSidePixels,
                     to: reservation.directory.appendingPathComponent(rightName))

        let degrees = 180.0 / Double.pi
        let fields = [
            "schema_version": "\(GazeDatasetSchema5.version)",
            "participant_id": csv(reservation.participantID),
            "session_id": csv(reservation.sessionID),
            "split": reservation.split.rawValue,
            "sample": "\(reservation.sampleNumber)",
            "frame_id": "\(frame.header.id.value)",
            "elapsed_s": fmt(reservation.elapsed),
            "target_id": "\(reservation.target.id)",
            "target_kind": reservation.target.kind.rawValue,
            "target_x": fmt(reservation.target.xFraction),
            "target_y": fmt(reservation.target.yFraction),
            "target_yaw_deg": fmt(reservation.angles.yaw),
            "target_pitch_deg": fmt(reservation.angles.pitch),
            "pose_prompt": reservation.target.pose.rawValue,
            "head_yaw_deg": fmt(tracking.headPose.yaw * degrees),
            "head_pitch_deg": fmt(tracking.headPose.pitch * degrees),
            "head_roll_deg": fmt(tracking.headPose.roll * degrees),
            "face_conf": fmt(tracking.confidence),
            "open_l": fmt(tracking.leftEye.openness),
            "open_r": fmt(tracking.rightEye.openness),
            "left_image": leftName,
            "right_image": rightName,
            "contour_points_l": "\(tracking.leftEye.contourPointCount)",
            "contour_points_r": "\(tracking.rightEye.contourPointCount)",
            "pupil_source_l": tracking.leftEye.pupilSource.rawValue,
            "pupil_source_r": tracking.rightEye.pupilSource.rawValue,
            "pupil_points_l": "\(tracking.leftEye.pupilPointCount)",
            "pupil_points_r": "\(tracking.rightEye.pupilPointCount)",
            "axis_start_x_l": fmt(leftStart.x),
            "axis_start_y_l": fmt(leftStart.y),
            "axis_end_x_l": fmt(leftEnd.x),
            "axis_end_y_l": fmt(leftEnd.y),
            "axis_start_x_r": fmt(rightStart.x),
            "axis_start_y_r": fmt(rightStart.y),
            "axis_end_x_r": fmt(rightEnd.x),
            "axis_end_y_r": fmt(rightEnd.y),
            "alignment_rotation_deg": fmt(reservation.alignment.rotationRadians * degrees),
            "alignment_disagreement_deg": fmt(reservation.alignment.disagreementDegrees),
            "crop_side_px_l": fmt(reservation.alignment.left.cropSidePixels),
            "crop_side_px_r": fmt(reservation.alignment.right.cropSidePixels),
            "crop_clipped_fraction_l": fmt(reservation.alignment.left.clippedFraction),
            "crop_clipped_fraction_r": fmt(reservation.alignment.right.clippedFraction),
        ]
        let row = try Self.manifestRow(fields)
        let manifest = reservation.directory.appendingPathComponent("manifest.csv")
        let handle = try FileHandle(forWritingTo: manifest)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(row.utf8))
    }

    static func manifestRow(_ fields: [String: String]) throws -> String {
        guard fields.count == GazeDatasetSchema5.manifestColumns.count else {
            throw WriteError.invalidManifestContract
        }
        return try GazeDatasetSchema5.manifestColumns.map { column in
            guard let value = fields[column] else { throw WriteError.invalidManifestContract }
            return value
        }.joined(separator: ",") + "\n"
    }

    func writeEye(_ buffer: CVPixelBuffer,
                  crop: GazeDatasetCanonicalAlignment.EyeCrop,
                  rotation: Double, side: Double, to url: URL) throws {
        let sourceHeight = Double(CVPixelBufferGetHeight(buffer))
        let centerX = crop.centerX
        let centerY = sourceHeight - crop.centerY
        let sourceRotation = -rotation
        let scale = Double(Self.imageWidth) / side
        let cosine = cos(sourceRotation)
        let sine = sin(sourceRotation)
        let a = scale * cosine
        let b = -scale * sine
        let c = scale * sine
        let d = scale * cosine
        let outputCenter = Double(Self.imageWidth) / 2
        let transform = CGAffineTransform(
            a: a, b: b, c: c, d: d,
            tx: outputCenter - a * centerX - c * centerY,
            ty: outputCenter - b * centerX - d * centerY)
        let bounds = CGRect(x: 0, y: 0, width: Self.imageWidth, height: Self.imageHeight)
        let image = CIImage(cvPixelBuffer: buffer).clampedToExtent()
            .transformed(by: transform, highQualityDownsample: true).cropped(to: bounds)
        try context.writePNGRepresentation(of: image, to: url, format: .RGBA8,
                                           colorSpace: colourSpace)
        try FileManager.default.setAttributes([.posixPermissions: 0o600],
                                              ofItemAtPath: url.path)
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
        let metadata = Metadata(schemaVersion: GazeDatasetSchema5.version,
                                participantID: session.participantID,
                                sessionID: session.id, split: session.split,
                                createdAt: session.createdAt, completedAt: completedAt,
                                status: status, displayGeometry: session.geometry,
                                eyeImageWidth: Self.imageWidth, eyeImageHeight: Self.imageHeight,
                                samplesPerTarget: GazeDatasetPlan.samplesPerTarget,
                                targetCount: session.targets.count,
                                capturedSamples: session.totalSamples,
                                cameraFormat: session.cameraFormat,
                                sourceImageWidth: session.sourceImageWidth,
                                sourceImageHeight: session.sourceImageHeight,
                                cropContract: .canonicalPairedEyesV2,
                                labelContract: .lensAngularV1,
                                headPoseContract: .visionRevision3DegreesV1)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let url = session.directory.appendingPathComponent("session.json")
        try encoder.encode(metadata).write(to: url, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600],
                                              ofItemAtPath: url.path)
    }

    func fmt(_ value: Double) -> String {
        String(format: "%.12f", locale: Locale(identifier: "en_US_POSIX"), value)
    }
    private func csv(_ value: String) -> String {
        "\"\(value.replacingOccurrences(of: "\"", with: "\"\""))\""
    }
}
