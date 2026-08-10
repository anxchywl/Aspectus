# Infrastructure

This document covers build targets, signing, packaging, local state, CI and diagnostics. Product
and model decisions are in [DESIGN.md](./DESIGN.md); contribution rules are in
[AGENTS.md](../AGENTS.md).

## Build system

Aspectus deliberately has two build entry points:

| Source | Tool | Purpose | Current CI coverage |
|---|---|---|---|
| `Package.swift` | SwiftPM | framework-free `AspectusKit` and unit tests | debug/release build and debug tests |
| `project.yml` | XcodeGen | app and camera-extension project | generation and target-presence check |

`Aspectus.xcodeproj` is generated and ignored. Edit `project.yml`, then regenerate it; never edit or
commit the project file.

```bash
swift test -c release
swift build -c release
cp scripts/Signing.xcconfig.example Signing.xcconfig
xcodegen generate
```

An unsigned Release build can compile the app and embedded extension without activating either:

```bash
xcodebuild -project Aspectus.xcodeproj -scheme Aspectus -configuration Release \
  -destination 'platform=macOS,arch=arm64' \
  CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO ENTITLEMENTS_REQUIRED=NO build
```

Deployment target is macOS 14. Swift 6 strict concurrency is enabled. The supported architecture is
Apple Silicon.

## Targets and process boundary

| Target | Identifier | Role |
|---|---|---|
| `Aspectus` | `com.aspectus.app` | capture, inference, rendering, UI and frame publication |
| `AspectusCameraExtension` | `com.aspectus.app.cameraextension` | sandboxed CoreMediaIO camera in a separate process |

The extension is embedded under `Contents/Library/SystemExtensions/`. The app and extension share
the app group `group.com.aspectus`; the extension exposes the required prefixed mach service
`group.com.aspectus.service`.

The important entitlements are:

| Entitlement | Target | Purpose |
|---|---|---|
| `com.apple.security.device.camera` | app | physical-camera capture |
| `com.apple.developer.system-extension.install` | app | extension installation; requires a real team |
| `com.apple.security.application-groups` | both | app/extension transport |
| `com.apple.security.app-sandbox` | extension | mandatory CMIO extension sandbox |

`NSSystemExtensionUsageDescription` appears in both Info plists. Removing it prevents the camera
extension category from activating correctly.

## Signing for contributors

Local signing settings belong in ignored `Signing.xcconfig`:

```bash
cp scripts/Signing.xcconfig.example Signing.xcconfig
```

Leaving the example values unchanged is sufficient for an unsigned preview build. A working camera
extension needs an Apple Development identity and a provisioned app group. Distribution uses a
Developer ID Application identity and notarization; an Apple Developer Program membership is
required, and the Account Holder creates the Developer ID certificate.

The repository does not yet parameterize its public identifiers. A fork cannot obtain a working
signed extension by changing only `ASPECTUS_TEAM_ID`. It must coordinate the bundle ID, extension
ID, product name, app group, mach service, entitlements, installer identifier and packaged extension
path across:

- `project.yml`
- `App/Aspectus.entitlements`
- `CameraExtension/AspectusCameraExtension.entitlements`
- `CameraExtension/Info.plist`
- `App/VirtualCamera/SystemExtensionInstaller.swift`
- `scripts/package-release.sh`

Treat that as one reviewed change. A partial rename can produce a build that launches but cannot
install, connect to or locate its extension.

## Packaging and notarization

`scripts/package-release.sh` generates the project, archives and exports a Developer ID build,
submits it for notarization, staples the result, strips local filesystem metadata from the ZIP, and
writes the artifact, checksum and manifest under ignored `dist/`.

The script stops when evidence does not match the label:

| Condition | Result |
|---|---|
| no Developer ID identity | stop, unless `--allow-unsigned` produces an explicitly `UNSIGNED` artifact |
| no usable notary profile | stop; never call the build notarized |
| dirty worktree | stop, unless `--allow-dirty` stamps the diagnostic artifact `-dirty` |

A Developer ID app with a system extension must be archived and exported with the
`developer-id` method; a plain build resolves different signing behavior. Notary credentials stay
in the login keychain under the profile used by `notarytool`. No key, password, team ID or signing
configuration is tracked.

`LICENSE` and `NOTICE` are app resources, so redistributed bundles include them and the signature
seals them.

## CI

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on pushes and pull requests to
`main`:

- **Test & Quality** builds `AspectusKit` in debug and Release, runs the debug Swift suite, and
  checks the last change for whitespace errors.
- **Project Spec** installs XcodeGen, supplies the example signing configuration, generates the
  project, checks that the extension target is present, and parses the release script.

CI currently does not compile the app/extension targets, run the Python trainer suite, sign,
notarize or publish. Unsigned app compilation is possible; it is simply not part of the workflow.
Signing and publication remain local because they require developer credentials and physical
release verification.

## Local state and biometric evidence

Runtime state is separated by sensitivity:

| Data | Location | Normal reset |
|---|---|---|
| preferences | `UserDefaults` under `com.aspectus.app` | `defaults delete com.aspectus.app` |
| active calibration | `~/Library/Application Support/Aspectus/calibration.json` | **Reset** in Correction settings |
| calibration backups | `~/Library/Application Support/Aspectus/calibration-backups/` | removed with calibration reset |
| failed-fit angle evidence | `~/Library/Application Support/Aspectus/calibration-attempts/` | removed with calibration reset |
| paired-eye datasets | `~/Library/Application Support/Aspectus/gaze-datasets/` | **Delete collected data** in Correction settings |
| training reports and models | ignored `Training/runs/` | explicit retention decision only |
| final-attempt ledger | `Training/runs/.aspectus-final-evaluation-ledger.json` | no routine reset |

Calibration stores derived angles, not images. Corrupt or newer unsupported files are rejected as
uncalibrated rather than partially trusted. A successful overwrite first preserves the previous
calibration; a failed fit leaves the active calibration unchanged.

The optional paired-eye dataset is biometric. Dataset and training-evidence paths are owner-only,
remain local, and are not read by the normal app runtime. Do not commit, upload or expose their
contents. The final-attempt ledger contains hashed session and candidate identities plus claim
metadata, not imagery or raw identifiers. Deleting or replacing it destroys one-shot evaluation
evidence and invalidates the protocol until history is restored.

Immutable candidate manifests, final reports and ledger claims are written through owner-only
temporary files, full-synced, atomically published, then full-synced with their parent directory on
the reference APFS volume.

## Logs and measurements

The app and extension use unified logging:

```bash
log stream --predicate 'subsystem == "com.aspectus.app"'
log stream --predicate 'subsystem == "com.aspectus.app.cameraextension"'
```

App categories include `capture`, `pipeline`, `sink`, `extension`, `gaze-dataset` and `quality`.
Extension categories include `device`, `source` and `sink`. Recovery and reconnect transitions are
logged because they are rare and should not depend on the UI being visible.

A Release build can write timer-sampled performance metrics:

```bash
Aspectus.app/Contents/MacOS/Aspectus --benchmark run.csv
```

The quality recorder is separate and explicit:

```bash
Aspectus.app/Contents/MacOS/Aspectus \
  --quality-capture /tmp/aspectus-screen-center \
  --quality-label screen-center
```

It writes a bounded set of original/corrected eye crops and a manifest. Only one background write
may be active, so recording does not weaken the single-frame pipeline invariant. These crops are
biometric and must not be committed or shared without the subject's explicit agreement.

Evidence in [DESIGN.md](./DESIGN.md) identifies its source. Performance tables come from Release
CSV recordings; quality captures, trainer reports, HUD readings, unified logs and direct hardware
probes are labeled separately rather than presented as equivalent measurements.
