import AppKit
import AspectusKit

/// physical layout of the calibration targets on the real display
///
/// macOS exposes no camera field of view (`videoFieldOfView` is API_UNAVAILABLE on macOS), so
/// viewing distance cannot be recovered from the image and has to be supplied by the user. the
/// display's own dimensions, by contrast, are exact: CGDisplayScreenSize reports millimetres
enum DisplayGeometry {
    /// the lens is assumed to sit at the top centre of the display, which holds for every built-in
    /// Apple camera. the few millimetres of bezel between the top edge and the lens are ignored —
    /// at a normal viewing distance that is well under a degree
    static func targetOffsets(for screen: NSScreen? = .main) -> [CalibrationTarget: TargetOffsetMM]? {
        guard let screen,
              let number = screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")]
                as? NSNumber else { return nil }
        let size = CGDisplayScreenSize(CGDirectDisplayID(number.uint32Value))
        guard size.width > 1, size.height > 1 else { return nil }

        let width = Double(size.width)
        let height = Double(size.height)
        return [
            // the lens itself, the only target whose true angle is exactly zero
            .center: TargetOffsetMM(right: 0, up: 0),
            .down: TargetOffsetMM(right: 0, up: -height),
            .left: TargetOffsetMM(right: -width / 2, up: -height / 2),
            .right: TargetOffsetMM(right: width / 2, up: -height / 2),
            // deliberately absent: "above the camera" is off the display, so it has no measurable
            // position and is a sign check only
        ]
    }

    static func geometry(viewingDistanceMM: Double,
                         screen: NSScreen? = .main) -> CalibrationGeometry? {
        guard let offsets = targetOffsets(for: screen) else { return nil }
        let geometry = CalibrationGeometry(viewingDistanceMM: viewingDistanceMM, offsets: offsets)
        return geometry.isUsable ? geometry : nil
    }

    static func datasetGeometry(viewingDistanceMM: Double,
                                screen: NSScreen? = .main) -> GazeDatasetGeometry? {
        guard let screen,
              let number = screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")]
                as? NSNumber else { return nil }
        let size = CGDisplayScreenSize(CGDirectDisplayID(number.uint32Value))
        let geometry = GazeDatasetGeometry(displayWidthMM: Double(size.width),
                                           displayHeightMM: Double(size.height),
                                           viewingDistanceMM: viewingDistanceMM)
        return geometry.isUsable ? geometry : nil
    }

    /// for the calibration sheet, so the user can sanity-check what the fit is assuming
    static func displayDescription(for screen: NSScreen? = .main) -> String {
        guard let screen,
              let number = screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")]
                as? NSNumber else { return "display size unavailable" }
        let size = CGDisplayScreenSize(CGDirectDisplayID(number.uint32Value))
        guard size.width > 1, size.height > 1 else { return "display size unavailable" }
        return String(format: "display %.0f × %.0f mm, lens assumed top centre",
                      size.width, size.height)
    }
}
