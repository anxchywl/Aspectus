# Aspectus

**Aspectus** is a native macOS app that corrects your eye contact in real time and publishes the result as a standard virtual camera — so you look at the lens on Zoom, Google Meet, Microsoft Teams, Discord, Slack, and OBS while actually reading your screen.

Built with Swift, SwiftUI, Swift Concurrency, AVFoundation, Metal, Core ML, and a CoreMediaIO camera extension. Apple Silicon only. No Electron, no Python runtime, no conferencing-app plugins.

> **In development.** The whole pipeline runs, and the camera extension now installs and delivers
> corrected frames — Aspectus appears alongside your real cameras and has been measured feeding a
> live AVFoundation capture client at 30 FPS. It has **not** yet been tried in Zoom, Meet, Teams,
> Discord, Slack or OBS, so treat those as unverified rather than working. There are still no
> downloadable releases — the code is Apache-2.0 licensed, but no build has been cut. Building it
> yourself needs an Apple Developer ID, because macOS will not activate an unsigned camera
> extension. See
> [Status](#status) for what works and [Performance](#performance) for measured numbers.

---

## What it does

- **Capture** — takes a physical camera feed through AVFoundation with bounded, drop-stale buffering
- **Track** — detects and tracks the primary face, eyes, pupils, and head pose on the Neural Engine (Apple Vision)
- **Redirect gaze** — warps only the smallest practical eye region toward the camera, resampling the original pixels rather than synthesizing new ones
- **Preserve everything else** — blinking, eyelids, eyelashes, eyebrows, glasses, eye color, lighting, and expression are kept by construction
- **Stay stable** — temporal filtering removes flicker without adding visible input lag
- **Fall back safely** — when correction confidence is low, the original frame passes through untouched
- **Publish** — exposes the processed frames as a native virtual camera through a CoreMediaIO system extension, with no plugin to install into the conferencing app

Calibration is optional and never required. The app works out of the box on any supported Mac with a built-in or external camera.

---

## How it works

```
Camera Capture  →  Face & Eye Tracking  →  Gaze Estimation  →  Eye Correction
      →  Temporal Stabilization  →  Metal Compositing  →  Virtual Camera
```

1. The camera delivers frames into a single-slot, drop-stale hand-off — at most one frame is ever in flight.
2. Apple Vision (rev 3) locates the primary face, eye regions, pupils, and head pose off the main actor.
3. Gaze relative to the camera is estimated from pupil geometry and head pose.
4. A Metal shader rotates the iris toward the lens by resampling it, leaving every pixel outside the two eye regions bit-identical to the source. A confidence gate decides how much to apply.
5. 1€ filters smooth landmarks and gaze; openness is deliberately left unfiltered so blinks stay sharp. Hysteresis and slew limiting prevent on/off flicker.
6. The corrected frame is presented in the preview and, in the same step, enqueued into the camera extension's sink stream. The extension consumes it in its own process and forwards it to whatever host application has the camera open. The virtual camera is an output and never a dependency: if the extension is absent or fails, the preview and the correction path carry on untouched.

Step 4 currently uses a geometric eyeball model rather than the learned warp field described in
the design. It sits behind the `EyeCorrector` protocol, so replacing it touches nothing else.

Correction only ever *resamples* existing pixels, so it cannot invent a face — the worst case is falling back to the untouched frame.

---

## Design constraints

- **Correction resamples, never synthesizes.** Identity, glasses, lighting, and skin tone are preserved because the model warps original pixels — it does not hallucinate them.
- **Latency beats completeness.** Stale frames are dropped, never queued. Frame queues are bounded by construction.
- **Confidence gates correction.** Below the trusted confidence and angle limits, the original frame passes through. Engagement uses hysteresis and slew-limited blending to avoid popping.
- **Orchestration is model-agnostic.** Every meaningful stage sits behind a replaceable protocol (`FaceTracker`, `GazeEstimator`, `EyeCorrector`, `FrameCompositor`, `FrameSink`); swapping a model never touches the pipeline controller.
- **Measured, not assumed.** Every performance target is verified in a release build. Facts, estimates, and assumptions are labeled as such.

---

## Performance

Targets:

| Target | Value |
|---|---|
| Frame rate | 60 FPS where the camera supports it |
| Processing latency | < 20 ms (ingest → present) |
| Frame queue depth | ≤ 1 in flight; drops counted |
| Memory | flat over long runs — no growth |
| Under thermal pressure | graceful, measured quality reduction |

Measured in a release build on Apple M3 / macOS 26.6, FaceTime HD camera pinned to 1280×720, over
a 136 s run of 3,937 presented frames:

| Stage | mean | p95 |
|---|---|---|
| Face tracking (Vision) | 19.2 ms | 29.2 ms |
| Eye warp (Metal) | 1.8 ms | 2.4 ms |
| Ingest → present | 45.7 ms | 61.2 ms |
| Capture → present | 89.4 ms | 106.0 ms |

Frames in flight never exceeded 1, with 11 drops in 3,937 frames and thermal state nominal
throughout. Resident memory over a separate 13-minute run trended down rather than up.

The virtual camera was measured separately, with a live AVFoundation client attached to it: **309
frames delivered over 10 s**, about 31 FPS, matching the rate the extension reported forwarding.

Two caveats on that number. It was taken before the advertised format was corrected, so the rate
is current but the geometry it was measured at is not; delivery has been confirmed since from
inside the extension, not from a host. And no long-run soak has been done, so nothing is claimed
about sustained behaviour.

Two targets are **not met**:

- **Processing latency is three times over budget.** Face tracking accounts for most of it; the
  correction itself costs 2.4 ms. Queued fixes: stop blocking the display path on tracking,
  downscale the tracker input, and consume the camera's native YUV instead of converting per frame.
- **60 FPS is unreachable on this hardware.** The FaceTime HD camera reports a 30 FPS ceiling at
  every format it offers, so 30 FPS is a camera limit, not a pipeline result.

Numbers come from the app's own CSV recorder (`--benchmark`), never from estimates.

---

## Tech stack

| Concern | Technology |
|---|---|
| App & UI | Swift, SwiftUI, Swift Concurrency |
| Capture | AVFoundation, Core Media |
| Tracking | Apple Vision (Neural Engine) |
| Correction | Metal (geometric warp today; Core ML flow field planned) |
| Rendering | Metal, Core Video (zero-copy `CVPixelBuffer` ↔ IOSurface ↔ `MTLTexture`) |
| Virtual camera | CoreMediaIO Camera Extension |
| Core library | `AspectusKit` — framework-free pipeline core (builds & tests without a camera) |

PyTorch and ONNX are used only for model evaluation, training, and conversion — never at runtime.

---

## Project structure

```
aspectus/
  Package.swift            AspectusKit — framework-free pipeline core
  Sources/AspectusKit/     backpressure, temporal filters, correction gate, capture recovery, metrics, stage protocols
  Tests/AspectusKitTests/  unit tests for the real-time invariants
  project.yml              XcodeGen spec for the app and the camera extension
  App/
    Capture/               AVFoundation session → drop-stale box, disconnect and sleep/wake observers
    Render/                Metal renderer + shaders
    Pipeline/              controller wiring, frame payload, tracker, corrector
    UI/                    SwiftUI preview, tracking overlay, diagnostics HUD
    VirtualCamera/         extension installer, sink that publishes corrected frames
  CameraExtension/         the CoreMediaIO system extension, its own process
  Shared/                  the format contract, compiled into both the app and the extension
  scripts/                 signing, notarization and packaging
  docs/DESIGN.md           research, foundation choice, pipeline design, risks, test criteria
  LICENSE, NOTICE          Apache-2.0, and the attribution that travels with every build
```

Pipeline architecture and agent coding rules: [AGENTS.md](./AGENTS.md)
Research and technical design: [docs/DESIGN.md](./docs/DESIGN.md)

---

## Status

| Phase | Scope | State |
|---|---|---|
| Research & design | Candidate study, foundation choice, pipeline design | ✅ [docs/DESIGN.md](./docs/DESIGN.md) |
| 1 — Video foundation | Capture → Metal preview → passthrough → FPS/latency HUD | ✅ measured on-device |
| 2 — Tracking | Vision landmarks, pupils, head pose, openness, confidence + overlay | ✅ running; pupil-landmark accuracy unconfirmed |
| 3 — Correction | Geometric eye warp in Metal, behind the `EyeCorrector` seam | ✅ working; over the latency budget |
| 3b — Learned warp field | Core ML flow-field model | ⬜ blocked — no licence-clean weights ([why](./docs/DESIGN.md#3-rejected-alternatives)) |
| 4 — Temporal quality | 1€ filters, gate hysteresis and slew wired into the live pipeline | ✅ wired + tested |
| 5 — Virtual camera | CoreMediaIO Camera Extension | ✅ installs and delivers frames; untested in conferencing apps |
| 6 — UI & hardening | Full SwiftUI, diagnostics, settings, release packaging | ⬜ HUD, basic controls, signing/notarization/CI, camera disconnect and sleep/wake recovery, licence; no settings UI |

Known gaps are listed at the end of [docs/DESIGN.md](./docs/DESIGN.md).

---

## Build & run

**Prerequisites:** Xcode 26+, macOS 14+, Apple Silicon.

```bash
brew install xcodegen
xcodebuild -downloadComponent MetalToolchain   # Xcode 26 ships Metal separately (~690 MB)
```

Core library (no GUI — runs in CI):

```bash
swift test
swift build -c release
```

App:

```bash
xcodegen generate
open Aspectus.xcodeproj   # Run (⌘R), grant camera permission on first launch
```

A plain ⌘R build gives you the app and its preview, but **not** the virtual camera. The extension
carries system-extension and app-group entitlements, which are restricted and cannot be carried by
an ad-hoc signature, so macOS will not activate it from an unsigned build.

Virtual camera:

```bash
cp Signing.xcconfig.example Signing.xcconfig   # fill in your team; the file is gitignored
./scripts/package-release.sh                   # signs, notarizes and staples
```

Then launch the packaged app from `/Applications` and click **Install camera**. macOS asks you to
approve it once, under System Settings → General → Login Items & Extensions → Camera Extensions.
Aspectus then appears alongside your real cameras in any app that lists them.

Notarization needs a Developer ID Application certificate and a `notarytool` keychain profile; the
script explains both if either is missing, and never stores a secret in the repository. Without a
Developer Program membership, `--allow-unsigned` produces an artifact honestly labelled as
unsigned — from which the extension will not install.

The extension runs in its own process, so its state goes to the unified log rather than anywhere
the app can print:

```bash
log stream --predicate 'subsystem == "com.aspectus.app.cameraextension"'
```

To record measurements instead of reading the HUD, build Release and pass a CSV path. Every
number in this README comes from a file produced this way:

```bash
xcodebuild -project Aspectus.xcodeproj -scheme Aspectus -configuration Release \
  -derivedDataPath .build/xcode build
.build/xcode/Build/Products/Release/Aspectus.app/Contents/MacOS/Aspectus --benchmark run.csv
```

Benchmarks are taken from release builds only. Anything not yet measured on hardware is labeled as such in [docs/DESIGN.md](./docs/DESIGN.md).

---

## Licence

Apache License 2.0 — see [LICENSE](./LICENSE) and [NOTICE](./NOTICE). Copyright 2026 anxchywl.

Apache-2.0 rather than MIT for the patent grant: gaze correction is a patent-dense field, and an
explicit grant is worth more here than the two paragraphs it costs.

### Attribution

The gaze-correction stage is a native reimplementation of the warp-field method from
[chihfanhsu/gaze_correction](https://github.com/chihfanhsu/gaze_correction) ("Look at me!", ACM TOMM
2019), Copyright 2019 Chih-Fan Hsu, licensed under BSD-3-Clause. **No source code and no trained
weights from that project are used here** — the weights ship without a licence grant and were
trained on a non-commercial dataset, which is why they were rejected. The reasoning is in
[docs/DESIGN.md §2–3](./docs/DESIGN.md).

Tracking uses Apple Vision; nothing in this repository is derived from a model whose provenance is
unaccounted for.
