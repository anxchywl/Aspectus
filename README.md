# Aspectus

[![CI](https://github.com/anxchywl/Aspectus/actions/workflows/ci.yml/badge.svg)](https://github.com/anxchywl/Aspectus/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)

Aspectus is an experimental Apple-Silicon macOS app for local eye-contact correction. It captures
camera frames, tracks the face and eyes with Apple Vision, resamples the original eye pixels with
Metal, shows a live preview, and can publish the result through a CoreMediaIO virtual camera.

> [!WARNING]
> Aspectus is a technical preview, not a finished gaze-correction product. The active geometric
> gaze estimator does not produce reliable lens-directed gaze, and no learned gaze model is
> integrated or shipped. Use the current build to study the pipeline and virtual camera, not for
> natural correction in important calls.

## Current status

| Area | Status |
|---|---|
| Native capture, tracking, Metal rendering and preview | Implemented and measured on the reference Mac |
| Geometric gaze estimation and eye warp | Implemented, but visual gaze quality is rejected |
| Appearance-based gaze model | Offline trainer exists; current model direction failed the Phase 3 gate |
| Original-frame fallback and temporal gate | Implemented and unit-tested; large-pose fallback measured |
| Virtual camera | Verified in Zoom, Google Meet and Microsoft Teams |
| Discord, Slack and OBS | Not tested |
| General-user model and redistributable weights | Not available |

The best consumed development result met the `2° / 5° / 3°` median, overall-p95 and lens-p95
gates, but its frozen checkpoint failed the next untouched session at `1.68° / 5.17° / 3.68°`.
All seven completed sessions have since influenced development. A three-seed follow-up rejected
further full fine-tuning of the current Open Model Zoo initializer. No candidate is frozen, another
validation recording is not justified, and native learned-model integration is blocked. The full
evidence and next research direction are in [docs/DESIGN.md](./docs/DESIGN.md).

## Install the technical preview

A signed and notarized [v0.1.0 release](https://github.com/anxchywl/Aspectus/releases/tag/v0.1.0)
is available for Apple Silicon. Download the ZIP and its SHA-256 file, verify the checksum, move
`Aspectus.app` to Applications, then launch it and grant camera access. This release is an older
technical preview with the same geometric-estimator limitation described above; current source
contains later Phase 3 protocol and documentation work.

## Build from source

Requirements:

- Apple Silicon Mac
- deployment target macOS 14; hardware evidence currently comes from macOS 26.6
- Xcode 26 (tested with 26.6)
- [XcodeGen](https://github.com/yonaskolb/XcodeGen)
- Metal toolchain component for Xcode 26

```bash
brew install xcodegen
xcodebuild -downloadComponent MetalToolchain
```

The framework-free core builds without a camera or signing identity:

```bash
swift test -c release
```

Generate and open the app project:

```bash
cp scripts/Signing.xcconfig.example Signing.xcconfig
xcodegen generate
open Aspectus.xcodeproj
```

With the example signing file unchanged, an unsigned build can run the app and preview but cannot
activate the camera extension. A contributor-signed virtual camera currently needs coordinated
team-specific bundle IDs, app-group IDs, the mach service, entitlements and package paths; setting
only a team ID is not sufficient for a fork. See
[docs/INFRASTRUCTURE.md](./docs/INFRASTRUCTURE.md) before changing signing or identifiers.

The release script is for maintainers with a correctly configured Developer ID identity and
notarization profile:

```bash
./scripts/package-release.sh
```

Explicit flags can produce clearly labeled unsigned or dirty diagnostic artifacts. Missing
notarization credentials or a failed notarization always stops the release path.

## Use Aspectus

1. Launch Aspectus and grant camera access.
2. Press **Start** to open the camera and preview the current correction pipeline.
3. Use **Settings → Correction → Calibrate** only as an experiment; calibration does not solve the
   known geometric-estimator limitation.
4. From a signed build, choose **Install camera** and approve the extension under **System Settings
   → General → Login Items & Extensions → Camera Extensions**.
5. Select **Aspectus** as the camera in Zoom, Google Meet or Microsoft Teams.

The virtual camera is an output, not a dependency. Preview continues if it is unavailable. As
tracking confidence, eye openness, head pose or requested correction leaves the trusted envelope,
Aspectus reduces correction and ultimately returns to the original frame instead of substituting a
different estimate.

## Architecture

```text
AVFoundation capture
  → Vision face and eye tracking
  → gaze estimation
  → temporal stabilization and safety gate
  → original-pixel Metal eye warp
  → preview and CoreMediaIO virtual camera
```

At most one frame is in flight. A single-slot, newest-frame-wins hand-off drops and counts stale
frames instead of building latency. The hot path keeps frames in `CVPixelBuffer` / IOSurface /
Metal textures, and inference or tracking never runs on the main actor.

The main replacement seams are `FaceTracker`, `GazeEstimator`, `EyeCorrector`, `FrameCompositor`
and `FrameSink`. `AspectusKit` contains framework-free scheduling, geometry, temporal and fallback
logic; Apple frameworks remain in the app target.

```text
Sources/AspectusKit/   framework-free pipeline core
Tests/                 deterministic core tests
App/                   capture, tracking, rendering, UI and virtual-camera sink
CameraExtension/       CoreMediaIO system extension
Shared/                app/extension format contract
Training/              offline dataset validation, training and Core ML conversion
scripts/               release packaging
docs/                  design evidence and build operations
```

## Privacy

- The runtime app has no remote telemetry, analytics or cloud-inference code. Camera processing
  stays on the Mac. A conferencing app receives virtual-camera frames only when the user selects Aspectus;
  that app's own transmission and privacy policy still apply.
- Model-data collection is off by default and must be started explicitly. It stores paired eye
  crops; participant and session UUIDs; frame and sample identifiers; timing and tracking quality;
  head pose; camera and display geometry; and target labels and coordinates locally under
  `~/Library/Application Support/Aspectus/gaze-datasets/` with owner-only permissions.
- Eye crops are biometric data. Do not commit, upload or share them. The collection UI can reveal
  or permanently delete the local dataset.
- Checkpoints, reports and conversions stay in ignored, owner-only `Training/runs/`. No dataset,
  learned weight or model artifact is tracked in this repository.
- The app itself performs no downloads. The separate offline training fetcher accesses the network
  only when invoked explicitly and verifies the downloaded archive and model checksums.

## Measured limits

On the reference Apple M3 running macOS 26.6 with a 1280×720 FaceTime HD camera, a controlled
Release run measured Vision at `6.71 ms` p95 and the geometric Metal warp at `1.32 ms` p95. The
formal processing metric, ingest to preview presentation, was `32.0 ms` p95 and therefore missed
the `<20 ms` target. Camera PTS to presentation was `60.1 ms` p95. A 97.6-minute soak kept queue
depth at one or less, published 136,479 virtual-camera frames, dropped 63 of roughly 175,000
captured frames, and showed no rising memory trend.

These measurements prove the reference pipeline's behavior, not natural gaze quality or other
hardware. Remaining limits include:

- macOS 14 and 15 have not been hardware-tested
- learned-model native latency has not been measured
- the reference camera is limited to 30 fps; the 60 fps path is not hardware-verified
- natural behavior with glasses, low light, partial occlusion and large gaze offsets is unproven
- physical camera disconnect and capture runtime-error recovery are unit-tested but not measured
- first-install virtual-camera visibility and Discord, Slack and OBS compatibility are unverified
- the current fixed `com.aspectus` identifiers make contributor signing awkward
- a personalized reference-machine pass would not establish general-user quality or redistribution
  rights

## Contributing and development

Focused issues and pull requests are welcome. Do not attach eye crops, participant/session
metadata, checkpoints or other biometric artifacts. Describe whether evidence is inspected,
unit-tested, converted, Release-tested or hardware-measured.

Read these before changing behavior:

- [AGENTS.md](./AGENTS.md) — architecture, safety, measurement and contribution rules
- [docs/DESIGN.md](./docs/DESIGN.md) — product constraints, model research and measured evidence
- [docs/INFRASTRUCTURE.md](./docs/INFRASTRUCTURE.md) — targets, signing, CI, storage and logs
- [Training/README.md](./Training/README.md) — offline Phase 3 protocol and privacy boundaries

`project.yml` is the project source. `Aspectus.xcodeproj` is generated and must not be edited or
committed. Run core tests in Release, regenerate the project after spec changes, and distinguish
unit-tested, converted, Release-tested and hardware-measured claims.

## Licensing

Repository source is licensed under [Apache-2.0](./LICENSE); attribution and research provenance
are recorded in [NOTICE](./NOTICE). A code licence does not by itself clear separately distributed
model weights, training data, derived weights or biometric collection; any patent grant depends on
the exact licence. The audited pretrained initializer is an ignored local development artifact and
is not bundled with Aspectus.
