import Foundation
import CoreMediaIO

// the extension is a plain executable that vends its provider and then parks on the run loop;
// the system starts and stops the process, so there is no shutdown path to write here
let providerSource = ExtensionProviderSource(clientQueue: nil)
CMIOExtensionProvider.startService(provider: providerSource.provider)
CFRunLoopRun()
