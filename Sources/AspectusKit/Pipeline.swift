import Foundation

// replaceable stage boundaries, generic over an opaque Pixels payload so this core never
// imports CoreVideo/Metal; the app layer binds Pixels = CVReadyFrame

/// nil result means no trustworthy face, which the orchestrator turns into original-frame fallback
public protocol FaceTracker: Sendable {
    associatedtype Pixels: Sendable
    func track(_ pixels: Pixels, header: FrameHeader) async -> TrackingResult?
}

public protocol GazeEstimator: Sendable {
    associatedtype Pixels: Sendable
    func estimate(_ pixels: Pixels, tracking: TrackingResult, header: FrameHeader) async -> GazeEstimate
}

/// implementations must modify only the smallest practical region and leave the rest untouched
public protocol EyeCorrector: Sendable {
    associatedtype Pixels: Sendable
    func correct(_ pixels: Pixels,
                 tracking: TrackingResult,
                 request: CorrectionRequest,
                 header: FrameHeader) async throws -> Pixels
}

/// kept separate from output so compositing and virtual camera stay independently replaceable
public protocol FrameCompositor: Sendable {
    associatedtype Pixels: Sendable
    func composite(original: Pixels, corrected: Pixels?, weight: Double, header: FrameHeader) async -> Pixels
}

public protocol FrameSink: Sendable {
    associatedtype Pixels: Sendable
    func publish(_ pixels: Pixels, header: FrameHeader) async
}

/// per-frame policy knobs, kept here so they are testable without the UI
public struct PipelineConfig: Sendable {
    public var gate: CorrectionGate.Config
    public var userStrength: Double            // 0…1 master strength
    public var landmarkSmoothing: OneEuroTuning
    public var gazeSmoothing: OneEuroTuning
    public init(gate: CorrectionGate.Config = .init(),
                userStrength: Double = 1.0,
                landmarkSmoothing: OneEuroTuning = .init(minCutoff: 1.2, beta: 0.02),
                gazeSmoothing: OneEuroTuning = .init(minCutoff: 0.8, beta: 0.01)) {
        self.gate = gate
        self.userStrength = userStrength
        self.landmarkSmoothing = landmarkSmoothing
        self.gazeSmoothing = gazeSmoothing
    }
}

public struct OneEuroTuning: Sendable {
    public var minCutoff: Double
    public var beta: Double
    public var dCutoff: Double
    public init(minCutoff: Double, beta: Double, dCutoff: Double = 1.0) {
        self.minCutoff = minCutoff; self.beta = beta; self.dCutoff = dCutoff
    }
}
