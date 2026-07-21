# Aspectus

**Aspectus** is a native macOS app that corrects your eye contact in real time and publishes the result as a standard virtual camera — so you look at the lens on Zoom, Google Meet, Microsoft Teams, Discord, Slack, and OBS while actually reading your screen.

Built with Swift, SwiftUI, Swift Concurrency, AVFoundation, Metal, Core ML, and a CoreMediaIO camera extension. Apple Silicon only. No Electron, no Python runtime, no conferencing-app plugins.

---

## What it does

- **Capture** — takes a physical camera feed through AVFoundation with bounded, drop-stale buffering
- **Track** — detects and tracks the primary face, eyes, pupils, and head pose on the Neural Engine (Apple Vision)
- **Redirect gaze** — warps only the smallest practical eye region toward the camera, resampling the original pixels rather than synthesizing new ones
- **Preserve everything else** — blinking, eyelids, eyelashes, eyebrows, glasses, eye color, lighting, and expression are kept by construction
- **Stay stable** — temporal filtering removes flicker without adding visible input lag
- **Fall back safely** — when correction confidence is low, the original frame passes through untouched
- **Publish** — exposes the processed frames as a native virtual camera any conferencing app can select

Calibration is optional and never required.

---

## How it works

```
Camera Capture  →  Face & Eye Tracking  →  Gaze Estimation  →  Eye Correction
      →  Temporal Stabilization  →  Metal Compositing  →  Virtual Camera
```

1. The camera delivers frames into a single-slot, drop-stale hand-off — at most one frame is ever in flight.
2. Apple Vision (rev 3) locates the primary face, eye regions, pupils, and head pose off the main actor.
3. Gaze relative to the camera is estimated from pupil geometry and head pose.
4. A Core ML warp-field model resamples the eye patch toward the lens; a confidence gate decides how much of it to blend.
5. 1€ filters smooth landmarks, gaze, and correction strength; hysteresis prevents on/off flicker.
6. Metal composites the corrected patch over the original and hands the frame to the CoreMediaIO camera extension.

Correction only ever *resamples* existing pixels, so it cannot invent a face — the worst case is falling back to the untouched frame.

---

## Design constraints

- **Correction resamples, never synthesizes.** Identity, glasses, lighting, and skin tone are preserved because the model warps original pixels — it does not hallucinate them.
- **Latency beats completeness.** Stale frames are dropped, never queued. Frame queues are bounded by construction.
- **Confidence gates correction.** Below the trusted confidence and angle limits, the original frame passes through. Engagement uses hysteresis and slew-limited blending to avoid popping.
- **Orchestration is model-agnostic.** Every meaningful stage sits behind a replaceable protocol (`FaceTracker`, `GazeEstimator`, `EyeCorrector`, `FrameCompositor`, `FrameSink`); swapping a model never touches the pipeline controller.
- **Measured, not assumed.** Every performance target is verified in a release build. Facts, estimates, and assumptions are labeled as such.

---

## Performance targets

Reference machine: Apple M3, macOS 14+. Targets, verified on release builds — not assumptions.

| Target | Value |
|---|---|
| Frame rate | 60 FPS where the camera supports it |
| Added processing latency | < 20 ms (capture → composite) |
| Frame queue depth | ≤ 1 in flight; drops counted |
| Memory | flat over long runs — no growth |
| Under thermal pressure | graceful, measured quality reduction |

---

## Tech stack

| Concern | Technology |
|---|---|
| App & UI | Swift, SwiftUI, Swift Concurrency |
| Capture | AVFoundation, Core Media |
| Tracking | Apple Vision (Neural Engine) |
| Correction | Core ML (warp-field model), Metal, Accelerate |
| Rendering | Metal, Core Video (zero-copy `CVPixelBuffer` ↔ IOSurface ↔ `MTLTexture`) |
| Virtual camera | CoreMediaIO Camera Extension |
| Core library | `AspectusKit` — framework-free pipeline core (builds & tests without a camera) |

PyTorch and ONNX are used only for model evaluation, training, and conversion — never at runtime.

---

## Project structure

```
aspectus/
  Package.swift            AspectusKit — framework-free pipeline core
  Sources/AspectusKit/     backpressure, temporal filters, correction gate, metrics, stage protocols
  Tests/AspectusKitTests/  unit tests for the real-time invariants
  project.yml              XcodeGen spec for the app (and, later, the camera extension)
  App/
    Capture/               AVFoundation session → drop-stale box
    Render/                Metal renderer + shaders
    Pipeline/              controller wiring, frame payload, tracker, corrector
    UI/                    SwiftUI preview, tracking overlay, diagnostics HUD
  docs/DESIGN.md           research, foundation choice, pipeline design, risks, test criteria
```

Pipeline architecture and agent coding rules: [AGENTS.md](./AGENTS.md)
Research and technical design: [docs/DESIGN.md](./docs/DESIGN.md)

---

## Status

| Phase | Scope | State |
|---|---|---|
| Research & design | Candidate study, foundation choice, pipeline design | ✅ [docs/DESIGN.md](./docs/DESIGN.md) |
| 1 — Video foundation | Capture → Metal preview → passthrough → FPS/latency HUD | ✅ verified on-device |
| 2 — Tracking | Vision landmarks, pupils, head pose, openness, confidence + overlay | ✅ builds; runtime accuracy pending |
| 3 — Correction | Core ML warp-field corrector | ⬜ next (model quality/speed gate) |
| 4 — Temporal quality | Filters & gate wired into the live pipeline | ⬜ primitives ready + tested |
| 5 — Virtual camera | CoreMediaIO Camera Extension | ⬜ |
| 6 — UI & hardening | Full SwiftUI, diagnostics, settings | ⬜ |

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

Benchmarks are taken from release builds only. Anything not yet measured on hardware is labeled as such in [docs/DESIGN.md](./docs/DESIGN.md).
