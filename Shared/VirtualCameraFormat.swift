import CoreMedia

/// the format the virtual camera advertises, compiled into both targets
///
/// a cmio extension declares its formats before any frame exists, so the app conforms to this
/// rather than the other way round; CMIO forwards a mismatch to hosts without complaint
enum VirtualCameraFormat {
    static let width: Int32 = 1280
    static let height: Int32 = 720
    static let frameRate: Int32 = 30
}
