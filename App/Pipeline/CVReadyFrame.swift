import CoreVideo
import AspectusKit

/// binds AspectusKit's generic Pixels to CoreVideo in the app layer, keeping the core framework-free
/// IOSurface-backed buffers let the same memory be wrapped as a Metal texture without a copy
struct CVReadyFrame: @unchecked Sendable {
    let header: FrameHeader
    let pixelBuffer: CVPixelBuffer

    var width: Int { CVPixelBufferGetWidth(pixelBuffer) }
    var height: Int { CVPixelBufferGetHeight(pixelBuffer) }
}

/// monotonic host clock in seconds, source of all frame timestamps
enum HostClock {
    static var seconds: Double {
        Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000_000.0
    }
}
