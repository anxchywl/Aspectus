import Vision
import CoreVideo
import AspectusKit

/// primary-face tracker on Apple Vision rev 3 (76-point constellation with pupils and head pose)
/// perform is synchronous, so it runs on a dedicated queue bridged to async and never blocks the main actor
struct VisionFaceTracker: FaceTracker {
    typealias Pixels = CVReadyFrame

    private let queue = DispatchQueue(label: "com.aspectus.vision", qos: .userInteractive)

    func track(_ pixels: CVReadyFrame, header: FrameHeader) async -> TrackingResult? {
        await withCheckedContinuation { (cont: CheckedContinuation<TrackingResult?, Never>) in
            queue.async {
                cont.resume(returning: Self.detect(pixels.pixelBuffer))
            }
        }
    }

    private static func detect(_ pixelBuffer: CVPixelBuffer) -> TrackingResult? {
        let request = VNDetectFaceLandmarksRequest()
        request.revision = VNDetectFaceLandmarksRequestRevision3
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        do {
            try handler.perform([request])
        } catch {
            return nil
        }
        guard let faces = request.results, !faces.isEmpty else { return nil }

        // primary face is the largest bounding box
        guard let face = faces.max(by: { $0.boundingBox.area < $1.boundingBox.area }),
              let landmarks = face.landmarks else { return nil }

        let box = face.boundingBox // normalized, origin bottom-left

        func region(_ r: VNFaceLandmarkRegion2D?) -> [NormPoint] {
            guard let r else { return [] }
            return r.normalizedPoints.map { p in
                // p is relative to the face box, lift to image space and flip Y to top-left origin
                let ix = Double(box.origin.x) + Double(p.x) * Double(box.width)
                let iyBottom = Double(box.origin.y) + Double(p.y) * Double(box.height)
                return NormPoint(x: ix, y: 1.0 - iyBottom)
            }
        }

        let leftEyePts = region(landmarks.leftEye)
        let rightEyePts = region(landmarks.rightEye)
        guard !leftEyePts.isEmpty, !rightEyePts.isEmpty else { return nil }

        let leftPupil = center(of: region(landmarks.leftPupil)) ?? center(of: leftEyePts)!
        let rightPupil = center(of: region(landmarks.rightPupil)) ?? center(of: rightEyePts)!

        let left = EyeObservation(region: boundingRect(leftEyePts),
                                  pupilCenter: leftPupil,
                                  openness: openness(eyePoints: leftEyePts))
        let right = EyeObservation(region: boundingRect(rightEyePts),
                                   pupilCenter: rightPupil,
                                   openness: openness(eyePoints: rightEyePts))

        let pose = HeadPose(yaw: Double(truncating: face.yaw ?? 0),
                            pitch: Double(truncating: face.pitch ?? 0),
                            roll: Double(truncating: face.roll ?? 0))

        let confidence = Double(face.confidence)

        // top-left normalized bounds for overlay/crop
        let faceBounds = NormRect(x: Double(box.origin.x),
                                  y: 1.0 - Double(box.origin.y) - Double(box.height),
                                  width: Double(box.width),
                                  height: Double(box.height))

        return TrackingResult(faceBounds: faceBounds, leftEye: left, rightEye: right,
                              headPose: pose, confidence: confidence)
    }

    // MARK: - geometry helpers

    private static func center(of pts: [NormPoint]) -> NormPoint? {
        guard !pts.isEmpty else { return nil }
        let sx = pts.reduce(0) { $0 + $1.x }
        let sy = pts.reduce(0) { $0 + $1.y }
        return NormPoint(x: sx / Double(pts.count), y: sy / Double(pts.count))
    }

    private static func boundingRect(_ pts: [NormPoint]) -> NormRect {
        let xs = pts.map(\.x), ys = pts.map(\.y)
        let minX = xs.min()!, maxX = xs.max()!, minY = ys.min()!, maxY = ys.max()!
        return NormRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }

    // eye-aspect-ratio openness, deliberately conservative so blinks are preserved
    private static func openness(eyePoints pts: [NormPoint]) -> Double {
        let r = boundingRect(pts)
        guard r.width > 1e-6 else { return 1 }
        let ear = r.height / r.width
        return max(0, min(1, (ear - 0.10) / (0.30 - 0.10)))
    }
}

private extension CGRect {
    var area: CGFloat { width * height }
}
