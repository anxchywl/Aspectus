# Aspectus for macOS — Design (Phase 1)

Reference machine: Apple M3, macOS 26.6 (25G72), Xcode 26.6 (17F113), Swift 6.3.3 (measured).
Status: phases 1–4 implemented; 42 unit tests green in a release build. The virtual camera
(phase 5) is not started. See "Measured facts" below for what has actually been run on hardware.

## 1. Candidate comparison

Shortlist of five, judged on reuse value — not stars, not README benchmarks.

| # | Candidate | Approach | Reuse | Apple-Silicon / Core ML fit | Maintenance | License (code / weights) | Integration difficulty | Critical limitation |
|---|-----------|----------|-------|------------------------------|-------------|--------------------------|------------------------|---------------------|
| 1 | **chihfanhsu/gaze_correction** ("Look at me!", ACM TOMM 2019) | Warping CNN: predicts a per-pixel flow field over the eye patch, bilinear-resamples the *original* pixels | The **method** and reference architecture; small net, ANE-friendly conv stack | High — conv-only + a grid-sample op; convertible via coremltools | Low (TF1.8, ~2019) | **BSD-3** © 2019 Chih-Fan Hsu (file is named `LICENSES`, so GitHub reports "no licence"); **weights: train-your-own** | Medium — reimplement/convert; must add our own sampler | TF1.8, TCP demo, needs calibration; weights not shipped |
| 2 | **WangWilly/gaze-correction-cam** | Same warp lineage + dlib/MediaPipe tracking, wrapped as a macOS virtual cam | Pretrained L/R warp weights, published as **release assets** rather than in the tree; proves end-to-end feasibility on Mac | Medium — weights convertible; runtime is not usable | Low, single author | **BSD-3** (code — same text and copyright as candidate 1); **weights carry no licence grant** | Medium — convert weights; discard Python runtime | 100% Python/TF — **violates "no Python in production"**; weights are not legally reusable (see §3) |
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
- Both reference implementations are **BSD-3**, so the *method and architecture* can be
  reimplemented natively (no Python/TF) with attribution.

The published **weights cannot be used**, which is a change from the original plan. Verified:
the BSD-3 licence covers the repository tree, but the weights ship as separate GitHub release
assets with no licence grant of any kind, and the method was trained and evaluated on the
Columbia Gaze dataset (CAVE), which states it "is made available for non-commercial use only".
Provenance of the specific shipped checkpoints is undocumented. That is the "unverified or
legally unclear model" case, so the weights are out of scope and any learned warp field must be
trained from data we can account for.

Apple Vision supplies primary-face + eye + **pupil** landmarks and head pose on the ANE for
near-zero cost, removing dlib/MediaPipe. Gaze magnitude for correction is derived geometrically
from pupil-vs-eye-center + head pose first; a small Core ML gaze head (candidate 4) is added
**only if** the prototype shows the geometric estimate is insufficient.

## 3. Rejected alternatives
- **Full-face synthesis (RTGaze/GazeNeRF/3D-eyeball, cand. 5):** too slow, and whole-face
  hallucination risks identity drift, glasses artifacts, and flicker — the opposite of the spec.
- **Running candidate 2 as-is:** Python/TF runtime is explicitly forbidden and won't hit latency.
- **Shipping candidate 2's pretrained weights:** no licence grant, undocumented provenance, and a
  training set restricted to non-commercial use. Rejected on those grounds, not technical ones.
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

Release build, Apple M3 / macOS 26.6, FaceTime HD camera pinned to 1280×720. Latency figures are
from a 136 s run with the preview window continuously visible (3,937 presented frames).

| Stage | mean | p95 |
|---|---|---|
| Face tracking (Vision rev 3) | 19.2 ms | 29.2 ms |
| Eye warp (geometric, Metal) | 1.8 ms | 2.4 ms |
| Ingest → present (processing) | 45.7 ms | 61.2 ms |
| Capture → present (end-to-end) | 89.4 ms | 106.0 ms |

- Frames in flight never exceeded 1; 11 drops in 3,937 frames; thermal state nominal throughout.
- Resident memory over a separate 801 s run trended **down** (150 → 115 MB), so no leak is visible
  at that timescale. The full 30-minute soak is still outstanding.
- 42 unit tests pass (`swift test`). Covered: drop-stale backpressure and drop counting, box depth
  and reopen, 1€ jitter reduction and bounded-lag tracking, gate hysteresis / angle-limit fallback
  / slew ramp, blink preservation, filter reset on tracking loss and timestamp discontinuity,
  recovery within the 300 ms budget, gaze geometry and warp containment, metrics percentiles.

### Phase 3 gate: not passed

The <20 ms processing target **fails at 61 ms p95**. The cause is measured and is *not* the
correction: the geometric warp costs 2.4 ms p95, while Vision landmark detection costs 29.2 ms and
the rest is display-path latency behind it. The model is therefore not the thing to reassess.
Queued: decouple tracking from the display path, downscale the tracker input, and consume the
camera's native YUV instead of converting to BGRA per frame.

### Hardware limits found

- The FaceTime HD camera reports a **30 FPS ceiling at every format** it offers (1080p, 720p,
  640×480 and the square/portrait variants). Sustained 60 FPS is unreachable on this machine, so
  30 FPS is a hardware limit rather than a pipeline result.
- The same camera offers **only `420v` (bi-planar YUV)**. Requesting BGRA makes AVFoundation
  convert every frame, which is avoidable work on the hot path.

### Correction quality

The geometric warp resamples the iris and is stable once filtered, but it cannot model eyelid
interaction or iris occlusion, so it degrades visibly at larger redirect angles. It is a working,
licence-clean baseline behind the `EyeCorrector` seam, not the final quality target.

## Known gaps

- Output FPS latches at its last value when the window is occluded instead of decaying to zero,
  because the rate meter is only ticked from the drawable-presented callback.
- Whether Vision supplies true pupil landmarks or falls back to the eye-contour centroid is
  unconfirmed; the tracker accepts either and the difference affects correction accuracy.
- No handling yet for camera disconnect, session runtime errors, or sleep/wake.
- Restart after stop is fixed and unit-tested, but not yet verified end-to-end on hardware.
