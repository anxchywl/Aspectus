# Aspectus for macOS — Design (Phase 1)

Reference machine: Apple M3, macOS 26.6 (25G72), Xcode 26.6 (17F113), Swift 6.3.3 (measured).
Status: phases 1–5 implemented; 170 unit tests green in a release build. The virtual camera
delivers frames to a capture client but has not been tried in any conferencing app. See
"Measured facts" below for what has actually been run on hardware.

## 1. Candidate comparison

Shortlist of four, judged on reuse value — not stars, not README benchmarks.

| # | Candidate | Approach | Reuse | Apple-Silicon / Core ML fit | Maintenance | License (code / weights) | Integration difficulty | Critical limitation |
|---|-----------|----------|-------|------------------------------|-------------|--------------------------|------------------------|---------------------|
| 1 | **chihfanhsu/gaze_correction** ("Look at me!", ACM TOMM 2019) | Warping CNN: predicts a per-pixel flow field over the eye patch, bilinear-resamples the *original* pixels | The **method** and reference architecture; small net, ANE-friendly conv stack | High — conv-only + a grid-sample op; convertible via coremltools | Low (TF1.8, ~2019) | **BSD-3** © 2019 Chih-Fan Hsu (file is named `LICENSES`, so GitHub reports "no licence"); **weights: train-your-own** | Medium — reimplement/convert; must add our own sampler | TF1.8, TCP demo, needs calibration; weights not shipped |
| 2 | **Apple Vision** (`VNDetectFaceLandmarks`, rev 3) | 76-point landmarks **incl. pupils** + face pose, on-device | Tracking stage wholesale — face bbox, eye landmarks, pupil centers, roll/yaw/pitch | Native, runs on ANE in ms, zero conversion, no telemetry | Apple-maintained | Apple SDK | Low | 2D pupil only (no true 3D gaze vector); needs a gaze head or geometric estimate |
| 3 | **L2CS-Net / MPIIGaze-style estimator** | Appearance-based gaze *direction* regressor | Optional gaze-angle head to drive correction magnitude | Small ResNet → Core ML convertible | Moderate | Research (MIT-ish); check weights | Medium | Estimates direction, does **not** redirect; extra model in budget |
| 4 | **RTGaze (2025) / GazeNeRF / 3D-eyeball** | Full-face novel-view / 3D-aware synthesis | Reference for quality ceiling only | Poor for v1 — 61 ms/frame reported, heavy | Active research | Research | High | Too slow for 60 FPS / <20 ms; hallucinates whole face → identity/glasses/temporal risk |

## 2. Selected foundation & rationale

**Warp-field correction (candidate 1) + Apple Vision tracking (2), reimplemented natively in Swift/Core ML/Metal.**

Why the warp-field family wins for *this* product spec:
- It **resamples the original pixels** instead of synthesizing them, so eye color, eyelids,
  lashes, eyebrows, glasses, lighting, and skin tone are preserved by construction — directly
  satisfying the "preserve …" and "modify only the smallest region" requirements.
- Flow fields are spatially smooth and small, so they are **temporally stable** and cheap —
  the right side of the quality/latency trade for 60 FPS.
- The network is conv-only + a grid-sample; **Core ML / ANE-friendly** and small enough to
  fit the <20 ms budget with margin for two eye patches.
- The reference implementation is **BSD-3**, so the *method and architecture* can be
  reimplemented natively (no Python/TF) with attribution.

**No usable pretrained weights exist**, which is a change from the original plan. Verified: the
BSD-3 licence covers the reference repository's tree, but that project ships no weights at all —
training your own is the documented path. The third-party checkpoints that circulate for this
method are published as release assets with no licence grant of any kind and undocumented
provenance, and the method was trained and evaluated on the Columbia Gaze dataset (CAVE), which
states it "is made available for non-commercial use only". That is the "unverified or legally
unclear model" case, so pretrained weights are out of scope and any learned warp field must be
trained from data we can account for.

Apple Vision supplies primary-face + eye + **pupil** landmarks and head pose on the ANE for
near-zero cost, removing dlib/MediaPipe. Gaze magnitude for correction is derived geometrically
from pupil-vs-eye-center + head pose first; a small Core ML gaze head (candidate 3) is added
**only if** the prototype shows the geometric estimate is insufficient.

## 3. Rejected alternatives
- **Full-face synthesis (RTGaze/GazeNeRF/3D-eyeball, cand. 4):** too slow, and whole-face
  hallucination risks identity drift, glasses artifacts, and flicker — the opposite of the spec.
- **Running a Python/TensorFlow reference implementation as-is:** that runtime is explicitly
  forbidden and won't hit latency.
- **Shipping third-party pretrained warp weights:** no licence grant, undocumented provenance, and
  a training set restricted to non-commercial use. Rejected on those grounds, not technical ones.
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
   **Partly retired.** Signing, notarization and activation work, and the extension being replaced
   under a running app is handled: the app rebuilds a connection whose queue has stopped draining,
   rather than believing forever that it is still connected. The host matrix remains untested, so
   host-app quirks are still an open risk — and the one host quirk found so far was in the test
   client, not in any host.

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
- 170 unit tests pass (`swift test`). Covered: drop-stale backpressure and drop counting, box depth
  and reopen, 1€ jitter reduction and bounded-lag tracking, gate hysteresis / angle-limit fallback
  / slew ramp, blink preservation, filter reset on tracking loss and timestamp discontinuity,
  recovery within the 300 ms budget, gaze geometry and warp containment, metrics percentiles,
  and the capture recovery policy for disconnect, runtime errors and sleep/wake.

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

### Phase 5: the virtual camera delivers

Signing, notarization, activation and enumeration all work: the extension installs from a
Developer ID build and macOS vends a camera that AVFoundation lists beside the physical ones. A
live capture client received **309 frames over 10 s** (≈31 FPS), matching the rate the extension
reported forwarding.

Three bugs on the way there, each real and each measured rather than reasoned about: the two
streams were wired in the wrong direction; the consume loop terminated permanently if the client
arrived after the stream started; and a trace call that resolved an app-group container inline on
a stream callback blocked it, stopping consumption entirely while leaving the process alive and
crash-free — which presented exactly like a logic bug.

The costliest error was not in the code. A host probe was trusted for several rounds without ever
being validated against a known-good control; it was an unsigned binary with no camera permission
and could not have received a frame from any camera. It reported zero frames from a transport that
was in fact working, and every conclusion drawn from it was wrong. The control — pointing the same
probe at the built-in camera — would have exposed it in one run.

## Known gaps

- Output FPS latches at its last value when the window is occluded instead of decaying to zero,
  because the rate meter is only ticked from the drawable-presented callback. Process FPS latches
  the same way for the same reason, which the sleep/wake run made visible: capture FPS correctly
  read 0 while the session was suspended, but process and output still read 30.
- ~~Whether Vision supplies true pupil landmarks or falls back to the eye-contour centroid is
  unconfirmed.~~ **Resolved by measurement** (phase 1 diagnostics, release build, M3 / macOS 26.6,
  FaceTime HD at 1280×720): `leftPupil`/`rightPupil` are populated on **100 % of tracked frames**,
  one point per eye, over two runs. The contour-centroid fallback never fired. It remains in the
  tracker as a guard and is now reported in diagnostics rather than being silent.
- `VNFaceObservation.confidence` measures **exactly 1.000** (min, mean and max) on every tracked
  frame, so the gate's 0.6/0.4 confidence hysteresis never fires on that input alone. Only the
  per-eye agreement factor (0.5 when one eye is unusable) ever moves it. The gate is correct; the
  signal feeding it carries no information.
- ~~**Vision supplies no head yaw or pitch.**~~ **Diagnosed and fixed.** A landmarks-only request
  reported head pose on **0 % of tracked frames**, making both the runtime 25° and calibration 15°
  limits inert. Cause: the SDK documents `roll`/`yaw`/`pitch` as populated by
  `VNDetectFaceRectanglesRequest`, not by the landmarks request. Fix: run rectangles at revision 3
  first, then chain the observation into the landmarks request via `inputFaceObservations`.
  Measured after: **100 % availability**, yaw spanning −40.1°…+28.9°, and the head-pose gate firing
  on 10 of 70 sampled frames — the first time it has ever engaged.
  - Cost of the second Vision pass, measured: tracking mean 11.7 → 16.5 ms, p95 ≈ 25 ms, and
    dropped frames 1 → 17 over a comparable run. Still inside the 33 ms frame interval, but the
    margin is now thin. One 593 ms tracking outlier was recorded at startup and is unexplained.
- **The estimate is contaminated by head pose.** With pose visible: when head pitch reached +22.8°,
  raw gaze pitch swung to −34.5°, driving a requested correction of 51.9° that the angle limit
  correctly cut to zero blend. The centred-pupil model has no head-pose term, so head rotation
  leaks directly into the gaze reading.
  - **Compensation implemented, coefficients not yet measured on hardware.** An uncontrolled run
    could not identify them: head rotation and eye-in-head rotation are confounded when the eyes
    move freely, and the correlations came out weak (r = 0.15…0.49) with the *strongest* term being
    a physically odd cross coupling (head pitch → gaze yaw, r = −0.49). Fitting that would encode
    incidental behaviour, not geometry.
  - The identifiable experiment is a **fixation sweep**: the user holds their gaze on the lens while
    rotating their head, so true gaze is zero on every sample and the raw reading is pure
    contamination. This is now a sixth calibration step, fitting a 2×2 linear map
    (`HeadCoupling`) by least squares on the mean-centred head angles.
  - Every sign is measured rather than assumed, which matters because Vision reports **pitch
    positive when nodding down** while this codebase treats **positive pitch as looking up**, and
    yaw's documented "counterclockwise" is ambiguous for a face. The fit refuses collinear head
    motion (a single direction of travel leaves the two axes indistinguishable), spans under 12°,
    fewer than 40 samples, and any slope outside ±2.
  - The sweep is skippable, and skipping it leaves behaviour byte-for-byte unchanged.
- ~~Per-axis scale is measured and wrong in opposite directions.~~ **Quantified and fitted.**
  From a 300-sample calibration on a 326.6 × 211.4 mm built-in display (`CGDisplayScreenSize`),
  refitted offline against real target geometry: at a 550 mm viewing distance the estimator needs
  **yawGain 0.53** and **pitchGain 2.70** — horizontal over-reads ~1.9×, vertical under-reads
  ~2.7×. Both inside the 0.2…4.0 plausibility band the fitter enforces.
  - Gain fitting is now implemented, but it depends on a **user-supplied viewing distance**:
    macOS exposes no camera field of view (`videoFieldOfView` and
    `videoFieldOfView(for:geometricDistortionCorrected:)` are both `API_UNAVAILABLE(macos)`), so
    scale cannot be recovered from the image. Sensitivity, measured: over 450–700 mm the fitted
    yawGain spans 0.64…0.42 and pitchGain 3.23…2.16, so a ±100 mm distance error moves the gain by
    roughly 10–20%.
  - Yaw is fitted from the two side targets and pitch from the lens (exactly 0°) plus the bottom
    edge. The "look above the camera" target is off the display, has no measurable position, and is
    therefore a sign/separation check only — never an input to the fit.
- An earlier reading of 0.9° vertical separation was a **procedure artifact**, not a property of
  the estimator: the calibration flow was sampling during the saccade to each target, averaging
  "up" and "down" toward each other. A 2 s settle window per target raised it to 9.7°.
- The vertical estimate carries a measured systematic bias: the pupil sits above the eye-aperture
  bounding-box centre on every frame of a run (normalized offset mean −0.0018 left, −0.0021 right),
  which reads as a constant **≈ +4° of apparent upward gaze** (raw pitch stayed within +2.2°…+5.7°
  and never went negative). This is the eyelid occlusion effect and is what phase 2 calibration
  has to remove.
- **Camera disconnect, session runtime errors and sleep/wake are handled, but only the healthy path
  is verified on hardware.** The session posts runtime errors, interruptions and device
  connect/disconnect; `NSWorkspace` supplies sleep and wake. The policy that turns those into
  decisions is `CaptureRecovery` in the kit, so it is unit-tested without a camera: sleep suspends
  and only waking restarts, a disconnect suspends and tries once more in case another camera is
  attached, a missing device stops the ladder and waits for a connect notification, and repeated
  failures back off (0.5–8 s) and then give up rather than reopening a broken camera forever.
  Recovery is only claimed once frames actually arrive — a session that reopens and stays silent is
  timed out at 3 s and counted as a failure. The pipeline itself is never torn down, so the gate and
  the filters do not restart from a stale blend.
  - **Sleep/wake is measured on hardware** (release build, M3 / macOS 26.6, software sleep at
    23:49:06, keyboard wake at 23:49:21). `willSleep` arrived 5 s *before* the machine actually
    slept and suspended the session there, so the camera was released while the system was still
    up. On wake: `didWake → retrying(attempt: 1)` at 23:49:21.80, reopen issued 0.5 s later
    (`FaceTime HD Camera 1280×720@30 BGRA`), delivery confirmed at 23:49:23.60 — **1.8 s from wake
    to frames**, then a sustained 30.0 fps for the remaining 60 s of the run. Dropped frames stayed
    at 1 across the whole cycle, so recovery costs no drop burst, and resident memory did not grow
    (240 → 149 → 157 MB).
  - Also measured: registering the observers costs nothing on the healthy path — capture holds
    30.0 fps with process and output rates unchanged.
  - The disconnect and runtime-error transitions are unit-tested but have **not** been exercised on
    hardware — the reference machine's camera is built in and cannot be unplugged.
  - An automatic reopen deliberately refuses to select Aspectus's own virtual camera, which is a
    video device like any other and would otherwise feed the pipeline its own output.
- Restart after stop is fixed and unit-tested, but not yet verified end-to-end on hardware.
- The virtual camera advertises 30 FPS while capture targets 60. Moot on the reference machine,
  whose camera caps at 30, but on a camera that reaches 60 the app would publish faster than the
  format it advertises. Resolution cannot drift the same way: it is one constant compiled into
  both targets, pinned in the capture output, and checked against real buffers before publishing.
- The virtual camera has never been soak-tested. One session recorded resident memory at 433 MB
  against a ~110 MB baseline, after the stats recorder had already stopped writing for reasons
  that were never established. Neither observation is diagnosed, and neither should be quoted as
  a result until a clean sustained run exists.
