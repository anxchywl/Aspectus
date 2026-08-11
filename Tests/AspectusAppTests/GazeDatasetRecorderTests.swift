import XCTest
import CoreImage
import CoreVideo
import ImageIO
import AspectusKit

/// exercises the production recorder's Core Image sampling and schema-4 serialization with
/// deterministic synthetic frames; no camera, no biometric data, no real storage locations
final class GazeDatasetRecorderTests: XCTestCase {
    private struct DecodeError: Error {}

    private struct Raster {
        let width: Int
        let height: Int
        let bytesPerRow: Int
        let bytes: [UInt8]
        let colorSpaceName: String?

        // decoded PNG bytes keep the file's top-to-bottom row order
        func channel(_ x: Int, _ y: Int, _ component: Int) -> Int {
            Int(bytes[y * bytesPerRow + x * 4 + component])
        }
    }

    private var temporaryRoot: URL!
    private var recorder: GazeDatasetRecorder!

    override func setUpWithError() throws {
        temporaryRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("aspectus-app-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: temporaryRoot,
                                                withIntermediateDirectories: true)
        recorder = GazeDatasetRecorder(rootDirectory: temporaryRoot)
    }

    override func tearDownWithError() throws {
        recorder = nil
        if let temporaryRoot {
            try? FileManager.default.removeItem(at: temporaryRoot)
        }
    }

    // MARK: - synthetic inputs

    private func makeBuffer(width: Int, height: Int,
                            _ fill: (Int, Int) -> (r: UInt8, g: UInt8, b: UInt8))
        -> CVPixelBuffer {
        var created: CVPixelBuffer?
        let attributes = [kCVPixelBufferCGImageCompatibilityKey: true] as CFDictionary
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                         kCVPixelFormatType_32BGRA, attributes, &created)
        precondition(status == kCVReturnSuccess)
        let buffer = created!
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        for y in 0..<height {
            for x in 0..<width {
                let (r, g, b) = fill(x, y)
                let offset = y * stride + x * 4
                base[offset] = b
                base[offset + 1] = g
                base[offset + 2] = r
                base[offset + 3] = 255
            }
        }
        return buffer
    }

    private func eye(centerX: Double, centerY: Double, lengthPixels: Double,
                     angleDegrees: Double, imageWidth: Double, imageHeight: Double,
                     pupilSource: PupilSource = .visionLandmark,
                     pupilPoints: Int = 3) -> EyeObservation {
        let angle = angleDegrees * .pi / 180
        let dx = cos(angle) * lengthPixels / 2
        let dy = sin(angle) * lengthPixels / 2
        return EyeObservation(
            region: NormRect(x: (centerX - lengthPixels / 2) / imageWidth,
                             y: (centerY - lengthPixels / 4) / imageHeight,
                             width: lengthPixels / imageWidth,
                             height: lengthPixels / 2 / imageHeight),
            pupilCenter: NormPoint(x: centerX / imageWidth, y: centerY / imageHeight),
            openness: 1,
            pupilSource: pupilSource,
            pupilPointCount: pupilPoints,
            contourPointCount: 8,
            imageAxisStart: NormPoint(x: (centerX - dx) / imageWidth,
                                      y: (centerY - dy) / imageHeight),
            imageAxisEnd: NormPoint(x: (centerX + dx) / imageWidth,
                                    y: (centerY + dy) / imageHeight))
    }

    private func alignment(left: EyeObservation, right: EyeObservation,
                           width: Int, height: Int) throws -> GazeDatasetCanonicalAlignment {
        try XCTUnwrap(GazeDatasetCanonicalAlignment(
            left: left, right: right, imageWidth: width, imageHeight: height))
    }

    private func decodePNG(_ url: URL) throws -> Raster {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil),
              let data = image.dataProvider?.data as Data? else { throw DecodeError() }
        XCTAssertEqual(image.bitsPerPixel, 32)
        XCTAssertTrue(image.alphaInfo == .premultipliedLast
                          || image.alphaInfo == .noneSkipLast
                          || image.alphaInfo == .last)
        return Raster(width: image.width, height: image.height,
                      bytesPerRow: image.bytesPerRow, bytes: [UInt8](data),
                      colorSpaceName: image.colorSpace?.name as String?)
    }

    private func renderedCrop(_ buffer: CVPixelBuffer,
                              crop: GazeDatasetCanonicalAlignment.EyeCrop,
                              rotation: Double, side: Double,
                              name: String = UUID().uuidString) throws -> Raster {
        let url = temporaryRoot.appendingPathComponent("\(name).png")
        try recorder.writeEye(buffer, crop: crop, rotation: rotation, side: side, to: url)
        return try decodePNG(url)
    }

    private func waitUntil(timeout: TimeInterval = 5,
                           _ condition: () -> Bool) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            usleep(20_000)
        }
        return condition()
    }

    // brightness-weighted row centre of one output column, ignoring the dark background
    private func lineRowCentre(_ raster: Raster, column: Int) -> Double {
        var weighted = 0.0
        var total = 0.0
        for row in 0..<raster.height {
            let weight = max(0, Double(raster.channel(column, row, 0)) - 60)
            weighted += weight * Double(row)
            total += weight
        }
        return total > 0 ? weighted / total : -1
    }

    // MARK: - core image geometry

    func testSlantedEyeAxisRendersHorizontally() throws {
        let width = 320, height = 240
        let angle = 12.0 * .pi / 180
        let (cx, cy) = (120.0, 130.0)
        let buffer = makeBuffer(width: width, height: height) { x, y in
            let distance = abs(-sin(angle) * (Double(x) - cx) + cos(angle) * (Double(y) - cy))
            return distance <= 1.5 ? (230, 230, 230) : (30, 30, 30)
        }
        let side = 60.0 / 1.8
        let value = try alignment(
            left: eye(centerX: cx, centerY: cy, lengthPixels: side, angleDegrees: 12,
                      imageWidth: 320, imageHeight: 240),
            right: eye(centerX: 220, centerY: cy, lengthPixels: side, angleDegrees: 12,
                       imageWidth: 320, imageHeight: 240),
            width: width, height: height)
        XCTAssertEqual(value.rotationRadians, angle, accuracy: 1e-12)
        XCTAssertEqual(value.cropSidePixels, 60, accuracy: 1e-9)

        let aligned = try renderedCrop(buffer, crop: value.left,
                                       rotation: value.rotationRadians,
                                       side: value.cropSidePixels)
        for column in stride(from: 6, through: 53, by: 4) {
            XCTAssertEqual(lineRowCentre(aligned, column: column), 29.5, accuracy: 1.5,
                           "column \(column) is not on the horizontal centre line")
        }

        // control: without the alignment rotation the same line must stay visibly slanted
        let unaligned = try renderedCrop(buffer, crop: value.left,
                                         rotation: 0, side: value.cropSidePixels)
        let slope = lineRowCentre(unaligned, column: 50) - lineRowCentre(unaligned, column: 10)
        XCTAssertGreaterThan(abs(slope), 5)
    }

    func testBothEyesUseTheSameRotationAndScale() throws {
        let width = 320, height = 240
        let sharedAngle = 12.0 * .pi / 180
        let centres = [(120.0, 130.0), (220.0, 130.0)]
        let buffer = makeBuffer(width: width, height: height) { x, y in
            for (cx, cy) in centres {
                let along = cos(sharedAngle) * (Double(x) - cx) + sin(sharedAngle) * (Double(y) - cy)
                let across = -sin(sharedAngle) * (Double(x) - cx) + cos(sharedAngle) * (Double(y) - cy)
                if abs(along) <= 27, abs(across) <= 1.5 { return (230, 230, 230) }
            }
            return (30, 30, 30)
        }
        // different per-eye axis angles and lengths still resolve to one rotation and one scale
        let value = try alignment(
            left: eye(centerX: 120, centerY: 130, lengthPixels: 60.0 / 1.8, angleDegrees: 10,
                      imageWidth: 320, imageHeight: 240),
            right: eye(centerX: 220, centerY: 130, lengthPixels: 50, angleDegrees: 14,
                       imageWidth: 320, imageHeight: 240),
            width: width, height: height)
        XCTAssertEqual(value.rotationRadians, sharedAngle, accuracy: 1e-9)
        XCTAssertEqual(value.cropSidePixels, 90, accuracy: 1e-9)

        for crop in [value.left, value.right] {
            let raster = try renderedCrop(buffer, crop: crop,
                                          rotation: value.rotationRadians,
                                          side: value.cropSidePixels)
            // the shared rotation keeps the 54 px segment horizontal in both crops
            for column in stride(from: 15, through: 45, by: 5) {
                XCTAssertEqual(lineRowCentre(raster, column: column), 29.5, accuracy: 1.5)
            }
            // the shared scale of 60/90 maps the segment ends to columns 30 ± 18
            XCTAssertGreaterThan(raster.channel(30, 29, 0), 150)
            XCTAssertGreaterThan(raster.channel(14, 29, 0) + raster.channel(14, 30, 0), 150)
            XCTAssertGreaterThan(raster.channel(46, 29, 0) + raster.channel(46, 30, 0), 150)
            XCTAssertLessThan(raster.channel(7, 29, 0), 80)
            XCTAssertLessThan(raster.channel(53, 29, 0), 80)
        }
    }

    func testTopLeftCoordinatesMapThroughCoreImageBottomLeftSpace() throws {
        let width = 150, height = 120
        let buffer = makeBuffer(width: width, height: height) { x, y in
            (UInt8(60 + x), UInt8(60 + y), 77)
        }
        let value = try alignment(
            left: eye(centerX: 100, centerY: 60, lengthPixels: 60.0 / 1.8, angleDegrees: 0,
                      imageWidth: 150, imageHeight: 120),
            right: eye(centerX: 40, centerY: 60, lengthPixels: 60.0 / 1.8, angleDegrees: 0,
                       imageWidth: 150, imageHeight: 120),
            width: width, height: height)
        XCTAssertEqual(value.cropSidePixels, 60, accuracy: 1e-9)

        let raster = try renderedCrop(buffer, crop: value.left, rotation: 0,
                                      side: value.cropSidePixels)
        XCTAssertEqual(raster.width, 60)
        XCTAssertEqual(raster.height, 60)
        // with unit scale, output pixel (i, j) reads source pixel (centerX - 30 + i, centerY - 30 + j)
        for (i, j) in [(30, 30), (5, 5), (55, 55), (5, 55), (55, 5)] {
            let sourceX = 100 - 30 + i
            let sourceY = 60 - 30 + j
            XCTAssertEqual(Double(raster.channel(i, j, 0)), Double(60 + sourceX), accuracy: 3,
                           "output (\(i),\(j)) does not read the expected source column")
            XCTAssertEqual(Double(raster.channel(i, j, 1)), Double(60 + sourceY), accuracy: 3,
                           "output (\(i),\(j)) does not read the expected source row")
        }
    }

    func testEdgeClampingIsDeterministicAtEveryImageBoundary() throws {
        let width = 150, height = 120
        let buffer = makeBuffer(width: width, height: height) { x, y in
            (UInt8(60 + x), UInt8(60 + y), 77)
        }
        // (crop centre, clamped output region, channel, expected clamped source value)
        let cases: [(centre: (Double, Double), region: [(Int, Int)],
                     channel: Int, expected: Int, label: String)] = [
            ((5, 75), (0...18).map { ($0, 30) }, 0, 60, "left"),
            ((145, 60), (41...59).map { ($0, 30) }, 0, 60 + 149, "right"),
            ((75, 5), (0...18).map { (30, $0) }, 1, 60, "top"),
            ((75, 115), (41...59).map { (30, $0) }, 1, 60 + 119, "bottom"),
        ]
        for test in cases {
            let value = try alignment(
                left: eye(centerX: test.centre.0, centerY: test.centre.1,
                          lengthPixels: 60.0 / 1.8, angleDegrees: 0,
                          imageWidth: 150, imageHeight: 120),
                right: eye(centerX: 75, centerY: 60, lengthPixels: 60.0 / 1.8, angleDegrees: 0,
                           imageWidth: 150, imageHeight: 120),
                width: width, height: height)
            let raster = try renderedCrop(buffer, crop: value.left, rotation: 0,
                                          side: value.cropSidePixels)
            XCTAssertEqual(raster.width, 60)
            XCTAssertEqual(raster.height, 60)
            for (x, y) in test.region {
                XCTAssertEqual(Double(raster.channel(x, y, test.channel)),
                               Double(test.expected), accuracy: 3,
                               "\(test.label) boundary is not clamped at (\(x),\(y))")
            }
        }

        // the same request twice produces identical bytes
        let value = try alignment(
            left: eye(centerX: 5, centerY: 75, lengthPixels: 60.0 / 1.8, angleDegrees: 0,
                      imageWidth: 150, imageHeight: 120),
            right: eye(centerX: 75, centerY: 60, lengthPixels: 60.0 / 1.8, angleDegrees: 0,
                       imageWidth: 150, imageHeight: 120),
            width: width, height: height)
        let first = try renderedCrop(buffer, crop: value.left, rotation: 0,
                                     side: value.cropSidePixels)
        let second = try renderedCrop(buffer, crop: value.left, rotation: 0,
                                      side: value.cropSidePixels)
        XCTAssertEqual(first.bytes, second.bytes)
    }

    func testOutputIsSixtyBySixtyFiniteSRGBForNonIntegralGeometry() throws {
        let width = 320, height = 240
        let buffer = makeBuffer(width: width, height: height) { x, y in
            (UInt8((x * 7) % 251), UInt8((y * 5) % 251), 128)
        }
        let value = try alignment(
            left: eye(centerX: 111.3, centerY: 97.6, lengthPixels: 137.3 / 1.8,
                      angleDegrees: 7.5, imageWidth: 320, imageHeight: 240),
            right: eye(centerX: 201.9, centerY: 103.4, lengthPixels: 120.0 / 1.8,
                       angleDegrees: 9.1, imageWidth: 320, imageHeight: 240),
            width: width, height: height)
        XCTAssertEqual(value.cropSidePixels, 137.3, accuracy: 1e-9)

        let raster = try renderedCrop(buffer, crop: value.left,
                                      rotation: value.rotationRadians,
                                      side: value.cropSidePixels)
        XCTAssertEqual(raster.width, 60)
        XCTAssertEqual(raster.height, 60)
        XCTAssertEqual(raster.colorSpaceName, CGColorSpace.sRGB as String?)
        for y in 0..<60 {
            for x in 0..<60 {
                XCTAssertEqual(raster.channel(x, y, 3), 255)
            }
        }
    }

    // MARK: - schema-4 serialization

    func testManifestRowRejectsSurplusMissingAndMisnamedFields() throws {
        var fields = Dictionary(uniqueKeysWithValues:
            GazeDatasetSchema4.manifestColumns.shuffled().enumerated().map {
                ($0.element, "value-\($0.offset)")
            })
        let row = try GazeDatasetRecorder.manifestRow(fields)
        // insertion order never matters: the row always follows the declared column order
        XCTAssertTrue(row.hasSuffix("\n"))
        XCTAssertEqual(row.trimmingCharacters(in: .newlines)
                           .split(separator: ",").map(String.init),
                       GazeDatasetSchema4.manifestColumns.map { fields[$0]! })

        var missing = fields
        missing.removeValue(forKey: "crop_side_px")
        XCTAssertThrowsError(try GazeDatasetRecorder.manifestRow(missing))

        var surplus = fields
        surplus["unexpected_evidence"] = "1"
        XCTAssertThrowsError(try GazeDatasetRecorder.manifestRow(surplus))

        var misnamed = fields
        misnamed.removeValue(forKey: "crop_side_px")
        misnamed["crop_side_pixels"] = "90"
        XCTAssertThrowsError(try GazeDatasetRecorder.manifestRow(misnamed))

        fields["schema_version"] = "4"
        XCTAssertNoThrow(try GazeDatasetRecorder.manifestRow(fields))
    }

    func testNumericFormattingIsPOSIXTwelveDecimal() {
        XCTAssertEqual(recorder.fmt(0.5), "0.500000000000")
        XCTAssertEqual(recorder.fmt(-0.05), "-0.050000000000")
        XCTAssertEqual(recorder.fmt(230.4), "230.400000000000")
        XCTAssertEqual(recorder.fmt(1.0 / 3.0), "0.333333333333")
        XCTAssertEqual(recorder.fmt(2.05), "2.050000000000")
        XCTAssertEqual(recorder.fmt(0), "0.000000000000")
    }

    private func syntheticTracking(width: Int, height: Int) -> TrackingResult {
        let left = eye(centerX: 100, centerY: 120, lengthPixels: 40, angleDegrees: 0,
                       imageWidth: Double(width), imageHeight: Double(height))
        let right = eye(centerX: 225, centerY: 120, lengthPixels: 50, angleDegrees: 0,
                        imageWidth: Double(width), imageHeight: Double(height),
                        pupilSource: .contourCentroid, pupilPoints: 0)
        return TrackingResult(faceBounds: NormRect(x: 0.2, y: 0.3, width: 0.5, height: 0.5),
                              leftEye: left, rightEye: right,
                              headPose: HeadPose(yaw: 0.05, pitch: -0.03, roll: 0.01),
                              confidence: 0.9)
    }

    private func frame(_ id: UInt64, width: Int, height: Int, at time: Double,
                       buffer: CVPixelBuffer) -> CVReadyFrame {
        CVReadyFrame(header: FrameHeader(id: FrameID(id),
                                         timing: FrameTiming(captureHostTime: time,
                                                             ingestHostTime: time),
                                         width: width, height: height),
                     pixelBuffer: buffer)
    }

    func testProductionSessionEmitsTheExactSchemaFourContracts() throws {
        let width = 320, height = 240
        let buffer = makeBuffer(width: width, height: height) { x, y in
            (UInt8((60 + x) % 256), UInt8((60 + y) % 256), 77)
        }
        let tracking = syntheticTracking(width: width, height: height)
        let geometry = GazeDatasetGeometry(displayWidthMM: 300, displayHeightMM: 200,
                                           viewingDistanceMM: 600)
        try recorder.start(split: .training, geometry: geometry,
                           cameraFormat: "synthetic 320x240", now: 0)
        let directory = try XCTUnwrap(recorder.snapshot()?.directoryPath)
        let sessionID = try XCTUnwrap(recorder.snapshot()?.sessionID)

        recorder.record(frame(1, width: width, height: height, at: 0, buffer: buffer),
                        tracking: tracking, now: 0)
        recorder.record(frame(2, width: width, height: height, at: 2.05, buffer: buffer),
                        tracking: tracking, now: 2.05)
        XCTAssertTrue(waitUntil { recorder.snapshot()?.totalSamples == 1 },
                      "first synthetic sample was never written")
        recorder.record(frame(3, width: width, height: height, at: 2.2, buffer: buffer),
                        tracking: tracking, now: 2.2)
        XCTAssertTrue(waitUntil { recorder.snapshot()?.totalSamples == 2 },
                      "second synthetic sample was never written")
        recorder.cancel()

        let directoryURL = URL(fileURLWithPath: directory)
        let manifestURL = directoryURL.appendingPathComponent("manifest.csv")
        let sessionURL = directoryURL.appendingPathComponent("session.json")
        XCTAssertTrue(waitUntil {
            (try? JSONSerialization.jsonObject(
                with: Data(contentsOf: sessionURL)) as? [String: Any])?["status"]
                as? String == "cancelled"
        }, "terminal session metadata was never written")

        // header: exactly the frozen 41 unique columns in order
        let lines = try String(contentsOf: manifestURL, encoding: .utf8)
            .split(separator: "\n").map(String.init)
        XCTAssertEqual(lines.count, 3)
        let header = lines[0].split(separator: ",").map(String.init)
        XCTAssertEqual(header, GazeDatasetSchema4.manifestColumns)
        XCTAssertEqual(header.count, 41)
        XCTAssertEqual(Set(header).count, 41)

        let rows = [lines[1], lines[2]].map {
            $0.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
        }
        XCTAssertEqual(rows[0].count, 41)
        XCTAssertEqual(rows[1].count, 41)
        func value(_ column: String, _ row: [String]) -> String {
            row[header.firstIndex(of: column)!]
        }

        let degrees = 180.0 / Double.pi
        let expected: [String: String] = [
            "schema_version": "4",
            "session_id": "\"\(sessionID)\"",
            "split": "training",
            "sample": "1",
            "frame_id": "2",
            "elapsed_s": "2.050000000000",
            "target_id": "0",
            "target_kind": "lens",
            "target_x": "0.500000000000",
            "target_y": "0.000000000000",
            "target_yaw_deg": "0.000000000000",
            "target_pitch_deg": "0.000000000000",
            "pose_prompt": "neutral",
            "head_yaw_deg": recorder.fmt(0.05 * degrees),
            "head_pitch_deg": recorder.fmt(-0.03 * degrees),
            "head_roll_deg": recorder.fmt(0.01 * degrees),
            "face_conf": "0.900000000000",
            "open_l": "1.000000000000",
            "open_r": "1.000000000000",
            "left_image": "sample-00001-left.png",
            "right_image": "sample-00001-right.png",
            "contour_points_l": "8",
            "contour_points_r": "8",
            "pupil_source_l": "visionLandmark",
            "pupil_source_r": "contourCentroid",
            "pupil_points_l": "3",
            "pupil_points_r": "0",
            "axis_start_x_l": "0.250000000000",
            "axis_start_y_l": "0.500000000000",
            "axis_end_x_l": "0.375000000000",
            "axis_end_y_l": "0.500000000000",
            "axis_start_x_r": "0.625000000000",
            "axis_start_y_r": "0.500000000000",
            "axis_end_x_r": "0.781250000000",
            "axis_end_y_r": "0.500000000000",
            "alignment_rotation_deg": "0.000000000000",
            "alignment_disagreement_deg": "0.000000000000",
            "crop_side_px": "90.000000000000",
            "crop_clipped_fraction_l": "0.000000000000",
            "crop_clipped_fraction_r": "0.000000000000",
        ]
        for (column, expectation) in expected {
            XCTAssertEqual(value(column, rows[0]), expectation, "column \(column)")
        }
        XCTAssertEqual(value("sample", rows[1]), "2")
        XCTAssertEqual(value("frame_id", rows[1]), "3")
        XCTAssertEqual(value("elapsed_s", rows[1]), "2.200000000000")

        // every fractional column is POSIX fixed-point with exactly 12 decimals
        let fractional = header.filter {
            $0.hasSuffix("_deg") || $0.hasSuffix("_s") || $0.hasSuffix("_px")
                || $0.hasPrefix("axis_") || $0.hasPrefix("crop_clipped")
                || $0.hasPrefix("open_") || $0 == "face_conf"
                || $0 == "target_x" || $0 == "target_y"
        }
        XCTAssertEqual(fractional.count, 24)
        for row in rows {
            for column in fractional {
                XCTAssertNotNil(value(column, row).wholeMatch(of: /-?\d+\.\d{12}/),
                                "column \(column) is not 12-decimal fixed point")
            }
        }

        // session metadata binds the exact crop, label, pose and source-dimension contracts
        let metadata = try XCTUnwrap(JSONSerialization.jsonObject(
            with: Data(contentsOf: sessionURL)) as? [String: Any])
        XCTAssertEqual(metadata["schemaVersion"] as? Int, 4)
        XCTAssertEqual(metadata["sourceImageWidth"] as? Int, width)
        XCTAssertEqual(metadata["sourceImageHeight"] as? Int, height)
        XCTAssertEqual(metadata["eyeImageWidth"] as? Int, 60)
        XCTAssertEqual(metadata["eyeImageHeight"] as? Int, 60)
        XCTAssertEqual(metadata["cameraFormat"] as? String, "synthetic 320x240")
        XCTAssertEqual(metadata["cropContract"] as? NSDictionary, [
            "version": 1,
            "coordinateSpace": "source-image-fraction-top-left",
            "axisExtractor": "farthest-contour-pair-ordered-image-x",
            "alignment": "circular-mean-paired-eye-axes",
            "center": "per-eye-axis-midpoint",
            "scale": "1.8x-maximum-eye-axis-length-pixels",
            "outputWidth": 60,
            "outputHeight": 60,
            "sampling": "core-image-affine-hq-downsample-edge-clamp",
            "colorSpace": "sRGB",
        ])
        XCTAssertEqual(metadata["labelContract"] as? NSDictionary, [
            "version": 1,
            "units": "degrees",
            "origin": "physical-lens",
            "yawPositive": "subject-right",
            "pitchPositive": "up",
        ])
        XCTAssertEqual(metadata["headPoseContract"] as? NSDictionary, [
            "version": 1,
            "source": "Vision.VNFaceObservation.face-rectangles-revision-3",
            "units": "degrees",
            "order": "yaw-pitch-roll",
            "yawPositive": "counterclockwise",
            "pitchPositive": "head-down",
            "rollPositive": "counterclockwise",
        ])

        // written files carry owner-only permissions
        for name in ["manifest.csv", "session.json", "sample-00001-left.png"] {
            let path = directoryURL.appendingPathComponent(name).path
            let permissions = try FileManager.default.attributesOfItem(
                atPath: path)[.posixPermissions] as? Int
            XCTAssertEqual(permissions, 0o600, "\(name) is not owner-only")
        }

        // the written eye crops equal a direct render with the one shared rotation and scale,
        // so the production write path used the same alignment for both eyes
        let shared = try alignment(left: tracking.leftEye, right: tracking.rightEye,
                                   width: width, height: height)
        XCTAssertEqual(shared.cropSidePixels, 90, accuracy: 1e-9)
        let expectedLeft = try renderedCrop(buffer, crop: shared.left,
                                            rotation: shared.rotationRadians,
                                            side: shared.cropSidePixels)
        let expectedRight = try renderedCrop(buffer, crop: shared.right,
                                             rotation: shared.rotationRadians,
                                             side: shared.cropSidePixels)
        let writtenLeft = try decodePNG(directoryURL.appendingPathComponent("sample-00001-left.png"))
        let writtenRight = try decodePNG(directoryURL.appendingPathComponent("sample-00001-right.png"))
        XCTAssertEqual(writtenLeft.bytes, expectedLeft.bytes)
        XCTAssertEqual(writtenRight.bytes, expectedRight.bytes)
        XCTAssertEqual(writtenLeft.width, 60)
        XCTAssertEqual(writtenLeft.height, 60)
        XCTAssertNotEqual(writtenLeft.bytes, writtenRight.bytes)
    }

    func testSourceFrameSizeChangeFailsTheSession() throws {
        let width = 320, height = 240
        let buffer = makeBuffer(width: width, height: height) { _, _ in (90, 90, 90) }
        let other = makeBuffer(width: 640, height: 480) { _, _ in (90, 90, 90) }
        let tracking = syntheticTracking(width: width, height: height)
        let geometry = GazeDatasetGeometry(displayWidthMM: 300, displayHeightMM: 200,
                                           viewingDistanceMM: 600)
        try recorder.start(split: .training, geometry: geometry,
                           cameraFormat: "synthetic 320x240", now: 0)
        let directory = try XCTUnwrap(recorder.snapshot()?.directoryPath)

        recorder.record(frame(1, width: width, height: height, at: 0, buffer: buffer),
                        tracking: tracking, now: 0)
        recorder.record(frame(2, width: 640, height: 480, at: 0.1, buffer: other),
                        tracking: tracking, now: 0.1)

        guard case .failed = try XCTUnwrap(recorder.snapshot()?.status) else {
            return XCTFail("a source-size change must fail the session")
        }
        let sessionURL = URL(fileURLWithPath: directory).appendingPathComponent("session.json")
        XCTAssertTrue(waitUntil {
            (try? JSONSerialization.jsonObject(
                with: Data(contentsOf: sessionURL)) as? [String: Any])?["status"]
                as? String == "failed"
        }, "failed session metadata was never written")
        let metadata = try XCTUnwrap(JSONSerialization.jsonObject(
            with: Data(contentsOf: sessionURL)) as? [String: Any])
        XCTAssertEqual(metadata["sourceImageWidth"] as? Int, width)
        XCTAssertEqual(metadata["sourceImageHeight"] as? Int, height)
    }
}
