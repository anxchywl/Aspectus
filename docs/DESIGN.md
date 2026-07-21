# Aspectus for macOS — Design (Phase 1)

Reference machine: Apple M-series, macOS 26.5, Xcode 26.6, Swift 6.3 (measured).
Status: design + tested framework-agnostic core (`AspectusKit`, 11 tests green, release build).

## 1. Candidate comparison

Shortlist of five, judged on reuse value — not stars, not README benchmarks.

| # | Candidate | Approach | Reuse | Apple-Silicon / Core ML fit | Maintenance | License (code / weights) | Integration difficulty | Critical limitation |
|---|-----------|----------|-------|------------------------------|-------------|--------------------------|------------------------|---------------------|
| 1 | **chihfanhsu/gaze_correction** ("Look at me!", ACM TOMM 2019) | Warping CNN: predicts a per-pixel flow field over the eye patch, bilinear-resamples the *original* pixels | The **method** and reference architecture; small net, ANE-friendly conv stack | High — conv-only + a grid-sample op; convertible via coremltools | Low (TF1.8, ~2019) | Code license present; **weights: train-your-own** | Medium — reimplement/convert; must add our own sampler | TF1.8, TCP demo, needs calibration; weights not shipped |
| 2 | **WangWilly/gaze-correction-cam** | Same warp lineage + dlib/MediaPipe tracking, wrapped as a macOS virtual cam | **Ships pretrained L/R warp weights** (`weights/warping_model/…`); proves end-to-end feasibility on Mac | Medium — weights convertible; runtime is not usable | Low, single author | **BSD-3** (code); weights inherit repo license | Medium — convert weights; discard Python runtime | 100% Python/TF — **violates "no Python in production"**; must be reimplemented natively |
| 3 | **Apple Vision** (`VNDetectFaceLandmarks`, rev 3) | 76-point landmarks **incl. pupils** + face pose, on-device | Tracking stage wholesale — face bbox, eye landmarks, pupil centers, roll/yaw/pitch | Native, runs on ANE in ms, zero conversion, no telemetry | Apple-maintained | Apple SDK | Low | 2D pupil only (no true 3D gaze vector); needs a gaze head or geometric estimate |
| 4 | **L2CS-Net / MPIIGaze-style estimator** | Appearance-based gaze *direction* regressor | Optional gaze-angle head to drive correction magnitude | Small ResNet → Core ML convertible | Moderate | Research (MIT-ish); check weights | Medium | Estimates direction, does **not** redirect; extra model in budget |
| 5 | **RTGaze (2025) / GazeNeRF / 3D-eyeball** | Full-face novel-view / 3D-aware synthesis | Reference for quality ceiling only | Poor for v1 — 61 ms/frame reported, heavy | Active research | Research | High | Too slow for 60 FPS / <20 ms; hallucinates whole face → identity/glasses/temporal risk |

## 2. Selected foundation & rationale

**Warp-field correction (candidates 1+2) + Apple Vision tracking (3), reimplemented natively in Swift/Core ML/Metal.**

Why the warp-field family wins for *this* product spec:
- It **resamples the original pixels** instead of synthesizing them, so eye color, eyelids,
  lashes, eyebrows, glasses, lighting, and skin tone are preserved by construction — directly
  satisfying the "preserve …" and "modify only the smallest region" requirements.
- Flow fields are spatially smooth and small, so they are **temporally stable** and cheap —
  the right side of the quality/latency trade for 60 FPS.
- The network is conv-only + a grid-sample; **Core ML / ANE-friendly** and small enough to
  fit the <20 ms budget with margin for two eye patches.
- We get a **BSD-3 reference implementation with shipped weights** (candidate 2) to bootstrap
  and to convert/validate against, avoiding a cold-start training dependency — while we build a
  fully native runtime (no Python/TF), as the constraints demand.

Apple Vision supplies primary-face + eye + **pupil** landmarks and head pose on the ANE for
near-zero cost, removing dlib/MediaPipe. Gaze magnitude for correction is derived geometrically
from pupil-vs-eye-center + head pose first; a small Core ML gaze head (candidate 4) is added
**only if** the prototype shows the geometric estimate is insufficient.

## 3. Rejected alternatives
- **Full-face synthesis (RTGaze/GazeNeRF/3D-eyeball, cand. 5):** too slow, and whole-face
  hallucination risks identity drift, glasses artifacts, and flicker — the opposite of the spec.
- **Running candidate 2 as-is:** Python/TF runtime is explicitly forbidden and won't hit latency.
- **dlib/MediaPipe tracking:** redundant given Vision's on-ANE landmarks+pupils.
- **NVIDIA Maxine / Apple's private FaceTime effect:** closed, not reusable.

## 4. Proposed pipeline

```
Camera Capture (AVFoundation, CVPixelBuffer/IOSurface)
    → Face & Eye Tracking (Vision rev3: bbox, eye landmarks, pupils, head pose)
    → Gaze Estimation (geometric pupil+pose → angle; optional Core ML head)
    → Eye Correction (Core ML warp-field net on eye patches, Metal grid-sample)
    → Temporal Stabilization (1€ filters on landmarks/gaze/strength; gate hysteresis)
    → Metal Compositing (blend corrected patch over original by confidence weight)
    → Virtual Camera (CoreMediaIO Camera Extension)
```

Orchestration is model-agnostic. Replaceable seams (already coded as protocols in
`AspectusKit`): `FaceTracker`, `GazeEstimator`, `EyeCorrector`, `FrameCompositor`, `FrameSink`.
Backpressure is a **single-slot drop-stale hand-off** (`LatestValueBox`) — bounded by
construction, newest-frame-wins, drops counted. Zero-copy path: `CVPixelBuffer`↔`IOSurface`↔
`MTLTexture`; correction runs on cropped eye textures only.

## 5. Main technical risks (top 5)
1. **Warp quality under glasses / large pose / low light.** Mitigation: conservative
   `maxCorrectionDegrees` (18° default), confidence gate → original-frame fallback, documented
   failure cases in Phase 3 before building further.
2. **Core ML conversion of the grid-sample / warp op & ANE residency.** `grid_sample` is not
   always ANE-native. Mitigation: validate conversion in the prototype; if it falls to GPU, do
   the resample in Metal ourselves and keep only convs in Core ML.
3. **Latency at 60 FPS (16.6 ms) for two eyes + tracking + composite.** Mitigation: measured
   stage metrics from day one (`StageMetrics`), patch-only inference, drop-stale scheduling,
   graceful FPS/quality reduction under thermal pressure.
4. **Temporal flicker vs. input lag.** Mitigation: 1€ filters (adaptive, low-lag), gate
   hysteresis + slew limiting (already unit-tested), blink preservation via openness gating.
5. **CoreMediaIO Camera Extension lifecycle / signing / host-app quirks.** Mitigation: build
   the extension early against SimpleDALPlugin/`cameraextension` references; test the full host
   matrix (Zoom/Meet/Teams/Discord/Slack/OBS); handle camera + host disconnect/reacquire.

## 6. Implementation phases
Matches the brief: (1) Video foundation, (2) Tracking, (3) Correction prototype [**hard gate**:
reassess model if quality/speed unacceptable], (4) Temporal quality, (5) Virtual camera,
(6) UI + hardening. Benchmarked in **release** builds each phase; no commit until a phase is stable.

## 7. Benchmark & test criteria (pass/fail)
- **Latency:** end-to-end capture→composite p95 **< 20 ms**; per-stage tracked. Fail if p95 ≥ 20 ms at target res.
- **Throughput:** sustained **60 FPS** where supported; graceful, measured degradation otherwise.
- **Memory:** flat RSS over a 30-min soak (no monotonic growth). Fail on upward trend.
- **Queueing:** in-flight frames ≤ 1; drops counted, never unbounded latency.
- **Temporal:** no visible flicker at rest; blinks fully preserved; smooth recovery < ~300 ms after tracking loss.
- **Quality:** natural correction within ±18°; identity/glasses/lighting preserved; clean fallback below confidence threshold.
- **Compatibility:** recognized as a standard camera + correct timing in all six host apps; survives camera/host disconnect and sleep/wake.

## Measured facts so far
- Toolchain builds `AspectusKit` in release; 11 unit tests pass (`swift test`).
- Covered by tests: drop-stale backpressure + drop counting, 1€ jitter reduction & bounded-lag
  tracking, gate hysteresis / angle-limit fallback / slew ramp (caught and fixed a first-frame
  strength-pop bug), metrics percentiles, geometry clamping.
- Not yet measured (require the app target + camera): real capture FPS/latency, Core ML warp
  inference time, Metal composite time, virtual-camera host compatibility.
