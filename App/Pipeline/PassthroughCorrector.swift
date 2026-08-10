import AspectusKit

/// original-frame fallback when metal correction is unavailable
struct PassthroughCorrector: EyeCorrector {
    typealias Pixels = CVReadyFrame
    func correct(_ pixels: CVReadyFrame,
                 tracking: TrackingResult,
                 request: CorrectionRequest,
                 header: FrameHeader) async throws -> CVReadyFrame {
        pixels
    }
}
