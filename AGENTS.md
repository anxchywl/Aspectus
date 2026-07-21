# Aspectus — AI Coding Agent Rules

Mandatory rules for AI coding agents working on Aspectus.
These rules are strict. If a required detail is missing, stop and ask before changing code.

**Sources of truth** (read before code):
- Research, foundation choice, pipeline design, risks, test criteria: `docs/DESIGN.md`
- This file: agent coding rules (mandatory)

---

## Architecture

Aspectus is a native Apple-Silicon macOS app. There is no backend, no web frontend, and no database. The product is a real-time video pipeline plus a virtual camera.

```
Sources/AspectusKit/          framework-free pipeline core — no AVFoundation / Metal / CoreML imports
  Frame.swift                 FrameID, FrameTiming, FrameHeader
  LatestValueBox.swift        single-slot drop-stale backpressure
  OneEuroFilter.swift         1€ temporal smoothing
  CorrectionGate.swift        confidence hysteresis, angle limit, slew-limited blend
  StageMetrics.swift          per-stage latency + drop counters
  GazeTypes.swift             NormPoint/NormRect, eye/pose/gaze value types
  Pipeline.swift              replaceable stage protocols + PipelineConfig

App/                          the macOS app target (binds the core to Apple frameworks)
  Capture/CameraCapture.swift AVCaptureSession → drop-stale box
  Render/                     MetalRenderer + Shaders.metal (zero-copy CVPixelBuffer → texture)
  Pipeline/                   CVReadyFrame payload, VisionFaceTracker, correctors, PipelineController
  UI/                         SwiftUI preview, TrackingOverlay, DiagnosticsHUD

Tests/AspectusKitTests/       unit tests for the real-time invariants
project.yml                   XcodeGen spec — the .xcodeproj is generated, never hand-edited
docs/DESIGN.md                design document
```

Pipeline data flow:
`Capture → FaceTracker → GazeEstimator → EyeCorrector → FrameCompositor → FrameSink`

The `PipelineController` owns orchestration only. It talks to stages through the protocols in `Pipeline.swift`; it must not know which concrete model or framework implements a stage.

---

## Domain Invariants

| Concern | Invariant |
|---|---|
| Backpressure | At most one frame in flight. Stale frames are dropped and counted, never queued. `LatestValueBox` is the only hand-off. |
| Correction | Resamples original pixels only. Never synthesizes a whole face. Modifies the smallest practical eye region. |
| Fallback | Below the confidence or angle limit, the original frame passes through. Engagement uses hysteresis; blend is slew-limited. |
| Temporal stability | Landmarks, gaze, and strength are smoothed with 1€ filters. Blinks are preserved. No smoothing that adds visible input lag. |
| Timing | Every frame carries explicit capture + ingest timestamps. Latency is measured, never estimated. |
| Latency reporting | "Processing latency" = ingest → present (ours, target < 20 ms). "End-to-end" = camera PTS → present (includes sensor delivery). Keep them distinct. |
| Zero-copy | `CVPixelBuffer` ↔ IOSurface ↔ `MTLTexture`. No per-frame pixel copies on the hot path. |
| Seams | Every meaningful stage stays behind its protocol. A protocol exists only where it is a real replacement boundary or test seam. |
| Runtime | No Python and no ONNX/PyTorch at runtime. Those are for training/conversion only. |

---

## Pre-Implementation Checklist

Before implementing a non-trivial task, state:

1. **Docs reviewed** — which parts of `docs/DESIGN.md` you read
2. **Phase** — which implementation phase this belongs to (see `docs/DESIGN.md §6`)
3. **Affected files** — exact paths
4. **Measurement** — what you will benchmark to prove it, in a release build
5. **Ordered plan**

Wait for approval on large changes before writing code.

---

## Skip These

`.build/`, `DerivedData/`, `*.xcodeproj/` (generated), model weight blobs, generated Info.plist, unrelated stages.

---

## 1. Project Rules

- Never invent APIs, model outputs, or framework behavior — verify against the SDK or measure it.
- Never modify unrelated files or rewrite large areas without explicit need.
- Never make decisions that conflict with `docs/DESIGN.md`.
- Always follow the existing architecture and inspect existing code before editing.
- Always keep changes scoped to the requested task and preserve existing behavior unless the task requires changing it.
- Clearly label measured facts, estimates, and assumptions. Do not present an estimate as a measurement.
- If information is missing, ambiguous, or conflicting, STOP and ask.

## 2. Architecture Rules

- Keep `Sources/AspectusKit/` free of AVFoundation, Metal, Core ML, Vision, and CoreVideo. It must build and unit-test without a camera.
- Bind the core to Apple frameworks only in the `App/` target.
- Add a protocol only when it provides a real replacement boundary or test seam. Avoid abstraction for its own sake.
- Keep SwiftUI separate from capture, inference, rendering, and extension logic.
- Do not move code across the core/app boundary unless the task requires it. If the correct location is unclear, STOP and ask.

## 3. Concurrency Rules

- Swift 6 strict concurrency is on. Do not silence it with unchecked escapes unless the invariant is real and documented in a comment.
- Never run capture, Vision, or Core ML inference on the main actor. Heavy work goes off-main; the main actor is for UI and light orchestration only.
- Respect cancellation. Long-running tasks must observe `Task.isCancelled` and tear down cleanly on stop.
- Avoid global mutable state and unnecessary singletons.

## 4. Capture & Rendering Rules

- Preserve the drop-stale invariant. Do not add buffering, queues, or `alwaysDiscardsLateVideoFrames = false`.
- Keep the frame path zero-copy. Do not introduce `CVPixelBuffer` copies or CPU round-trips on the hot path.
- The preview mirror is a display choice only; it must never change what the virtual camera outputs.
- Handle camera and host disconnect, device switching, and sleep/wake without crashing or leaking.

## 5. Model & Correction Rules

- Correction must resample original pixels and touch the smallest practical region.
- Respect the conservative correction-angle limit and the confidence gate. If a change would let correction run outside trusted conditions, STOP and ask.
- If the selected model cannot meet an acceptable quality/latency combination, reassess the model before building more features — do not paper over it.
- Core ML compute-unit and model-version choices must be visible in diagnostics, not hidden.

## 6. Testing & Measurement Rules

- Prioritize tests that protect the real-time pipeline: pipeline decisions, transforms, temporal stability, backpressure.
- Core logic goes in `AspectusKit` with deterministic unit tests. Do not put testable logic where it needs a camera to run.
- Benchmark each phase in a release build. Do not continue after an unexplained performance regression — diagnose it first.
- Do not claim exact ANE/GPU utilization. macOS does not expose reliable production telemetry; use Instruments for hardware-placement conclusions.

## 7. Diagnostics Rules

- The HUD shows capture/process/output FPS, per-stage and end-to-end latency, dropped frames, queue depth, memory, thermal state, camera format, model version, and configured compute units.
- Update diagnostics on a timer, not per frame — never thrash the main actor at capture rate.

## 8. Git & Commit Rules

- Do not commit until the current phase is stable and builds in release.
- Do not commit or push without explicit authorization.
- Keep changes small and reviewable. One coherent change per commit.
- The `.xcodeproj` is generated by XcodeGen and gitignored. Edit `project.yml`, never the project file.

### Commit messages

- Use Conventional Commits: `type(scope): subject`.
- Types: `feat`, `fix`, `perf`, `refactor`, `style`, `test`, `docs`, `chore`, `ci`.
- Scope is optional and lowercase (`capture`, `tracking`, `render`, `kit`, `ci`).
- Subject is lowercase, imperative, no trailing period, under ~72 chars.
- Add a body only when the *why* is non-obvious; wrap it and use `-` bullets.

Good:

```
fix(tracking): drop faces below confidence instead of clamping to zero
perf(render): reuse the metal texture cache across frames
```

Bad:

```
Fixed a bug.
update code
feat: Added New Feature.
```

---

## 9. Code Style Rules

### General

- Prefer self-documenting code over comments.
- Well-named identifiers are better than a comment explaining them.
- Keep style consistent across the whole codebase.

### Comments

Write a comment only when the **why** is non-obvious — a hidden constraint, a subtle invariant, a workaround, behavior that would surprise a reader. Never restate what the code already says.

When a comment is needed:

- Lowercase only
- No trailing punctuation
- Concise — explain intent, not implementation

Good:

```swift
// first frame only sets the time baseline, else engaging pops to full strength
guard let lt = lastTime, t > lt else { return weight }
```

Bad — states the obvious:

```swift
// return the weight if there is no last time
guard let lt = lastTime, t > lt else { return weight }
```

Bad — wrong format:

```swift
// First frame: establish the time baseline.
```

Doc comments (`///`) follow the same rules: lowercase, no trailing punctuation, one or two lines. No multi-paragraph banner blocks.

### Naming

Prefer descriptive names over comments.

Good: `activeFormatDescription`
Bad: `afd`

### Prohibited

- Obvious comments
- Commented-out code
- TODO without an issue reference
- AI-generated banners or attribution
- Large comment blocks
