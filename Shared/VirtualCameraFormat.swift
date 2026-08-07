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

/// the name the virtual camera vends, compiled into both targets
///
/// the extension advertises it and the app matches devices on it, so the two must not drift; capture
/// also uses it to make sure a reopen never selects the camera we publish into
enum VirtualCameraIdentity {
    static let name = "Aspectus"
}
