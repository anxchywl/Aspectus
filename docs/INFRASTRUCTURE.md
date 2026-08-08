# Infrastructure

How Aspectus is built, signed, shipped and observed. Product design lives in
[DESIGN.md](./DESIGN.md); coding rules live in [AGENTS.md](../AGENTS.md).

## Build system

Two build systems, deliberately:

| | Tool | Builds | Runs in CI |
|---|---|---|---|
| `Package.swift` | SwiftPM | `AspectusKit` — the framework-free core | yes, no camera needed |
| `project.yml` | XcodeGen → `Aspectus.xcodeproj` | the app and the camera extension | spec validation only |

The split is what keeps the pipeline core testable without a camera, a GUI or a signing identity.
`Aspectus.xcodeproj` is **generated and gitignored** — edit `project.yml` and re-run `xcodegen
generate`; never edit the project file.

```bash
swift test                  # 206 unit tests, no hardware
swift build -c release      # the core alone
xcodegen generate           # after any project.yml change
```

## Targets

| Target | Bundle ID | Type | Notes |
|---|---|---|---|
| Aspectus | `com.aspectus.app` | application | embeds the extension |
| AspectusCameraExtension | `com.aspectus.app.cameraextension` | system extension | own process, sandboxed |

The extension is embedded at `Contents/Library/SystemExtensions/`, which is where macOS looks for
it. Its bundle is *named* after its identifier, matching how the system stores it once installed.

The two processes meet at the app group `group.com.aspectus`, and the extension vends the mach
service `group.com.aspectus.service` — the prefix is mandatory, since a sandboxed extension cannot
vend a service outside its group.

Deployment target macOS 14, Swift 6 with `SWIFT_STRICT_CONCURRENCY: complete`, Apple Silicon only.

## Entitlements

| Entitlement | Target | Why |
|---|---|---|
| `com.apple.security.device.camera` | app | capture |
| `com.apple.developer.system-extension.install` | app | **restricted** — installing the extension needs a real team |
| `com.apple.security.application-groups` | both | the only channel between the two processes |
| `com.apple.security.app-sandbox` | extension | mandatory for a CMIO extension; the app is unsandboxed |

`NSSystemExtensionUsageDescription` is required in **both** Info.plists — without it the CMIO
category rejects the install outright rather than prompting.

## Signing

Signing identity and team come from `Signing.xcconfig`, which is gitignored so no team identifier
is ever committed:

```bash
cp scripts/Signing.xcconfig.example Signing.xcconfig
```

A plain ⌘R build gives you the app and its preview but **not** the virtual camera: the restricted
entitlements above cannot be carried by an ad-hoc signature, so macOS refuses to activate an
extension from an unsigned build. Use `Apple Development` locally and `Developer ID Application`
for anything published — the latter needs a Developer Program membership, and only the Account
Holder can create the certificate.

## Release

`scripts/package-release.sh` runs the whole path: generate → archive → export with
`method=developer-id` → notarize → staple → zip, checksum and manifest into `dist/` (gitignored).

A Developer ID build cannot come from `xcodebuild build`: automatic signing there resolves to a
development profile, which conflicts with the Developer ID identity. Archive-and-export is the only
route that provisions the System Extension capability correctly.

The script refuses to produce anything it cannot label honestly:

| Situation | Result |
|---|---|
| no Developer ID certificate | stops, unless `--allow-unsigned` — which names the artifact `-UNSIGNED` |
| no notary credentials | stops; it never calls a signed build "notarized" |
| dirty working tree | stops, unless `--allow-dirty` — which stamps the commit `-dirty` |

Notary credentials live in the login keychain under a named profile you create yourself
(`xcrun notarytool store-credentials`). No secret is ever passed to the script, and none is stored
in this repository.

`LICENSE` and `NOTICE` are bundle resources, so they are sealed by the signature and travel with
every redistributed copy, as Apache-2.0 §4(a) requires.

## CI

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml), on every push and PR to `main`:

- **Test & Quality** — builds `AspectusKit` debug and release, runs the unit tests, and checks the
  last commit for whitespace errors (needs `fetch-depth: 2`, or `HEAD~1` does not exist).
- **Project Spec** — installs XcodeGen, generates the project, asserts the camera extension target
  is wired into the app, and parses the release script. `project.yml` is hand-maintained, so a
  broken spec would otherwise only surface for whoever next ran `xcodegen` locally.

**Deliberately absent:** any job that signs, notarizes or publishes. Those need a Developer ID
private key and an app-specific password, and putting either in GitHub secrets would be a permanent
credential exposure on a public repository. Releases are cut locally.

CI does not compile the app or extension targets — that needs a signing configuration, and the
runner has none.

## State on disk

Two stores, kept apart because they answer to different rules:

| What | Where | Reset by |
|---|---|---|
| Preferences — mirror, overlay, HUD, correction, redirect angle, viewing distance, camera | `UserDefaults` under `com.aspectus.app` | `defaults delete com.aspectus.app` |
| Calibration — fitted angles, target samples and head sweep | `~/Library/Application Support/Aspectus/calibration.json` | **Reset** in Settings ▸ Correction |

The calibration file holds derived angles and no imagery. Neither store leaves the machine, and a
corrupt or future-version calibration is treated exactly as an uncalibrated install rather than
being partially trusted. Before a successful calibration overwrites the file, the previous one is
copied to `~/Library/Application Support/Aspectus/calibration-backups/`.
Failed fits leave the active calibration untouched and write their angle-only evidence under
`~/Library/Application Support/Aspectus/calibration-attempts/`. Reset removes all three locations.

## Observability

The extension runs in its own process, so its state goes to the unified log rather than anywhere
the app can print:

```bash
log stream --predicate 'subsystem == "com.aspectus.app.cameraextension"'
log stream --predicate 'subsystem == "com.aspectus.app"'
```

| Subsystem | Categories |
|---|---|
| `com.aspectus.app` | `capture`, `pipeline`, `sink`, `extension` |
| `com.aspectus.app.cameraextension` | `device`, `source`, `sink` |

Capture recovery — disconnect, session errors, sleep/wake — logs every transition under
`com.aspectus.app:pipeline`, because those paths are rare and nobody is watching the HUD when they
fire.

For numbers rather than the HUD, a release build takes a CSV path and samples every metric on the
stats timer:

```bash
.build/xcode/Build/Products/Release/Aspectus.app/Contents/MacOS/Aspectus --benchmark run.csv
```

Every measurement quoted in [DESIGN.md](./DESIGN.md) comes from a file produced this way. Benchmarks
are taken from release builds only.

Correction quality has a separate, explicitly enabled recorder because visual evidence contains
images rather than only metrics:

```bash
Aspectus.app/Contents/MacOS/Aspectus \
  --quality-capture /tmp/aspectus-screen-center \
  --quality-label screen-center
```

It writes at most 12 paired original/corrected eye crops, 0.75 seconds apart, plus `manifest.csv`
with the raw and calibrated gaze, requested correction, gate state, blend and iris travel for the
same frame, plus head pose, eye openness, confidence and landmark age. Only one background write may
be active, so the recorder cannot queue frames or weaken the single-frame pipeline invariant. Normal
launches construct no recorder and pay no cost.

These crops are biometric imagery. They stay in the directory supplied on the command line; do not
commit or share them unless the subject has explicitly agreed.
