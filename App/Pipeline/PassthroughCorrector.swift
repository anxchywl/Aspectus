import AspectusKit

/// identity corrector, proves the EyeCorrector seam while phase 3's warp model is not wired yet
struct PassthroughCorrector: EyeCorrector {
    typealias Pixels = CVReadyFrame
    func correct(_ pixels: CVReadyFrame,
                 tracking: TrackingResult,
                 request: CorrectionRequest,
                 header: FrameHeader) async throws -> CVReadyFrame {
        pixels
    }
}
