import Vision
import CoreVideo
import AspectusKit

/// primary-face tracker on Apple Vision rev 3 (76-point constellation with pupils and head pose)
/// perform is synchronous, so it runs on a dedicated queue bridged to async and never blocks the main actor
struct VisionFaceTracker: FaceTracker {
    typealias Pixels = CVReadyFrame

    private let queue = DispatchQueue(label: "com.aspectus.vision", qos: .userInteractive)

    /// what the two Vision passes actually cost, measured around the request and nothing else
    ///
    /// the orchestrator's "tracking" figure starts before this is dispatched and stops after the
    /// warp, the publish and the main-actor hop, so it reports the loop's critical path instead
    let metrics = StageMetrics(name: "vision", window: 240)

    func track(_ pixels: CVReadyFrame, header: FrameHeader) async -> TrackingResult? {
        await withCheckedContinuation { (cont: CheckedContinuation<TrackingResult?, Never>) in
            queue.async {
                let result = metrics.measure { Self.detect(pixels.pixelBuffer) }
                cont.resume(returning: result)
            }
        }
    }

    /// two passes, because the SDK documents roll/yaw/pitch as populated by
    /// VNDetectFaceRectanglesRequest, and a landmarks-only request measured 0% pose availability
    ///
    /// the rectangles observation is fed to the landmarks request through VNFaceObservationAccepting,
    /// which copies it and fills in the landmarks, so the pose angles survive onto the result
    private static func detect(_ pixelBuffer: CVPixelBuffer) -> TrackingResult? {
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])

        let rectangles = VNDetectFaceRectanglesRequest()
        rectangles.revision = VNDetectFaceRectanglesRequestRevision3
        do {
            try handler.perform([rectangles])
        } catch {
            return nil
        }
        // primary face is the largest bounding box
        guard let primary = rectangles.results?.max(by: { $0.boundingBox.area < $1.boundingBox.area })
        else { return nil }

        let request = VNDetectFaceLandmarksRequest()
        request.revision = VNDetectFaceLandmarksRequestRevision3
        request.inputFaceObservations = [primary]
        do {
            try handler.perform([request])
        } catch {
            return nil
        }
        guard let face = request.results?.first, let landmarks = face.landmarks else { return nil }

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

        // Vision declares both pupil regions nullable and gives no guarantee that revision 3
        // populates them, so which source won has to be reported rather than assumed
        func pupil(_ r: VNFaceLandmarkRegion2D?,
                   contour: [NormPoint]) -> (NormPoint, PupilSource, Int) {
            let pts = region(r)
            if let c = center(of: pts) { return (c, .visionLandmark, pts.count) }
            return (center(of: contour)!, .contourCentroid, 0)
        }

        let (leftPupil, leftSource, leftCount) = pupil(landmarks.leftPupil, contour: leftEyePts)
        let (rightPupil, rightSource, rightCount) = pupil(landmarks.rightPupil, contour: rightEyePts)
        let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
        let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
        let leftAxis = EyeObservation.imageAxis(
            of: leftEyePts, imageWidth: imageWidth, imageHeight: imageHeight)
        let rightAxis = EyeObservation.imageAxis(
            of: rightEyePts, imageWidth: imageWidth, imageHeight: imageHeight)

        let left = EyeObservation(region: boundingRect(leftEyePts),
                                  pupilCenter: leftPupil,
                                  openness: openness(eyePoints: leftEyePts),
                                  pupilSource: leftSource,
                                  pupilPointCount: leftCount,
                                  cornerMidpointY: EyeObservation.cornerMidpointY(of: leftEyePts),
                                  contourPointCount: leftEyePts.count,
                                  imageAxisStart: leftAxis?.start,
                                  imageAxisEnd: leftAxis?.end)
        let right = EyeObservation(region: boundingRect(rightEyePts),
                                   pupilCenter: rightPupil,
                                   openness: openness(eyePoints: rightEyePts),
                                   pupilSource: rightSource,
                                   pupilPointCount: rightCount,
                                   cornerMidpointY: EyeObservation.cornerMidpointY(of: rightEyePts),
                                   contourPointCount: rightEyePts.count,
                                   imageAxisStart: rightAxis?.start,
                                   imageAxisEnd: rightAxis?.end)

        // read from the rectangles observation: the landmarks copy is documented to carry the
        // pose through, but the original is the object that actually computed it
        let pose = HeadPose(yaw: Double(truncating: face.yaw ?? primary.yaw ?? 0),
                            pitch: Double(truncating: face.pitch ?? primary.pitch ?? 0),
                            roll: Double(truncating: face.roll ?? primary.roll ?? 0))
        // a nil yaw or pitch becomes zero, which reads downstream as a perfectly square head and
        // would make the head-pose limit unfireable, so the absence has to travel with the result
        let poseAvailable = (face.yaw ?? primary.yaw) != nil && (face.pitch ?? primary.pitch) != nil

        let confidence = Double(face.confidence)

        // top-left normalized bounds for overlay/crop
        let faceBounds = NormRect(x: Double(box.origin.x),
                                  y: 1.0 - Double(box.origin.y) - Double(box.height),
                                  width: Double(box.width),
                                  height: Double(box.height))

        return TrackingResult(faceBounds: faceBounds, leftEye: left, rightEye: right,
                              headPose: pose, confidence: confidence,
                              headPoseAvailable: poseAvailable)
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
