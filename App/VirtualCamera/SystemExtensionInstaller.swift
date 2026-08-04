import Foundation
import SystemExtensions
import os

/// installs and activates the embedded camera extension
///
/// activation is always an explicit user action. macOS shows its own approval prompt the first
/// time, and on a development signature the extension will only load once developer mode is on —
/// both are surfaced as state rather than hidden behind a spinner that never resolves
@MainActor
final class SystemExtensionInstaller: NSObject, ObservableObject {
    enum State: Equatable {
        case idle
        case requesting
        /// macOS is waiting for the user to allow it in System Settings
        case needsApproval
        case activated
        case failed(String)

        var summary: String {
            switch self {
            case .idle: return "not installed"
            case .requesting: return "installing…"
            case .needsApproval: return "allow in System Settings ▸ General ▸ Login Items & Extensions"
            case .activated: return "active"
            case let .failed(message): return "failed: \(message)"
            }
        }
    }

    @Published private(set) var state: State = .idle

    private let identifier = "com.aspectus.app.cameraextension"
    private let log = Logger(subsystem: "com.aspectus.app", category: "extension")

    func activate() {
        state = .requesting
        let request = OSSystemExtensionRequest.activationRequest(
            forExtensionWithIdentifier: identifier, queue: .main)
        request.delegate = self
        OSSystemExtensionManager.shared.submitRequest(request)
    }

    /// uninstall is offered because a camera extension that cannot be removed from the app that
    /// installed it is a support problem, not a feature
    func deactivate() {
        state = .requesting
        let request = OSSystemExtensionRequest.deactivationRequest(
            forExtensionWithIdentifier: identifier, queue: .main)
        request.delegate = self
        OSSystemExtensionManager.shared.submitRequest(request)
    }
}

extension SystemExtensionInstaller: OSSystemExtensionRequestDelegate {
    nonisolated func request(_ request: OSSystemExtensionRequest,
                             actionForReplacingExtension existing: OSSystemExtensionProperties,
                             withExtension replacement: OSSystemExtensionProperties)
        -> OSSystemExtensionRequest.ReplacementAction {
        // a rebuild during development carries the same version, so replacing unconditionally is
        // what keeps the installed copy in step with the app
        .replace
    }

    nonisolated func requestNeedsUserApproval(_ request: OSSystemExtensionRequest) {
        Task { @MainActor in self.state = .needsApproval }
    }

    nonisolated func request(_ request: OSSystemExtensionRequest,
                             didFinishWithResult result: OSSystemExtensionRequest.Result) {
        Task { @MainActor in
            switch result {
            case .completed:
                self.state = .activated
            case .willCompleteAfterReboot:
                self.state = .failed("needs a restart to finish")
            @unknown default:
                self.state = .failed("unknown result")
            }
        }
    }

    nonisolated func request(_ request: OSSystemExtensionRequest, didFailWithError error: Error) {
        let message = (error as NSError).localizedDescription
        log.error("extension request failed: \(message, privacy: .public)")
        Task { @MainActor in self.state = .failed(message) }
    }
}
