# Aspectus for macOS — Design (Phase 1)

Reference machine: Apple M3, macOS 26.6 (25G72), Xcode 26.6 (17F113), Swift 6.3.3 (measured).
Status: phases 1–6 implemented; 193 unit tests green in a release build. The virtual camera is
verified in three of the six conferencing hosts — Zoom, Google Meet and Microsoft Teams — and
untested in the other three. See "Measured facts" below for what has actually been run on hardware.

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
   **Partly retired.** Signing, notarization and activation work. **Zoom, Google Meet and Microsoft
   Teams are verified**; Discord, Slack and OBS are untested, so host quirks remain an open risk.
   Neither host test found a fault in the host: the one bug the Zoom test surfaced was ours, a
   stopped sink stream spinning the extension's consume loop, and the Meet test found the sink's
   reconnect after an extension replacement to be broken. That second finding corrects what this
   entry used to claim. An extension replaced under a running app **cannot** be reconnected to from
   that process — macOS never shows it the new device, by any lookup — so the app detects the loss,
   says so, and offers a relaunch, which is the only thing that works. See "Known gaps".

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

Release build, Apple M3 / macOS 26.6, FaceTime HD camera pinned to 1280×720.

**The controlled run**, which supersedes the earlier table: 174 s with the subject square to the
camera (mean head yaw −0.5°), correction engaged on 90 % of samples, the preview continuously
visible, and `caffeinate -d` holding the display awake — the last of those matters, because an
earlier attempt let the display sleep and froze the present-dependent numbers.

| Stage | mean | p95 |
|---|---|---|
| **Vision rev 3, both passes** | **6.12 ms** | **6.71 ms** |
| Track loop (Vision + everything concurrent with it) | 6.44 ms | 8.80 ms |
| Eye warp (geometric, Metal) | 0.80 ms | 1.32 ms |
| **Ingest → corrected frame ready (pipeline)** | **0.75 ms** | **1.28 ms** |
| Ingest → present (processing) | 25.9 ms | 32.0 ms |
| Capture → present (end-to-end) | 54.0 ms | 60.1 ms |

A second controlled run of 224 s agrees inside noise: Vision 6.34 / 7.07, track loop 6.74 / 8.89,
warp 0.80 / 1.59, pipeline 0.84 / 1.60 ms.

The superseded table, kept because two later sections still argue from it, was a 136 s run with the
preview visible: face tracking 19.2 / 29.2, warp 1.8 / 2.4, processing 45.7 / 61.2, end-to-end
89.4 / 106.0 ms.

- Frames in flight never exceeded 1; 11 drops in 3,937 frames; thermal state nominal throughout.
- ~~Unexplained: later runs recorded face tracking at ~6 ms against the table's 19.2 ms.~~
  **Settled, and the table was wrong.** Two things were confused in it.
  - **The metric was mislabelled.** What the table called "face tracking" starts its clock before
    the Vision request is dispatched and stops it after the warp, the publish and the main-actor
    hop have all finished, so it reports `max(Vision, everything concurrent with Vision)` — the
    loop's critical path, not Vision's cost. Vision is now timed around the request itself and
    reported separately; the two are listed above as *Vision* and *track loop*.
  - **19.2 ms is not reproducible on this machine, by any build.** The obvious suspect was the
    pre-redesign per-frame `tracking` publish, which re-rendered the whole HUD on the main actor at
    capture rate and would have inflated a metric that includes the main-actor hop. That was tested
    directly: the pre-redesign commit (18be6b9) was built from a worktree and run for 90 s with the
    HUD visible, a face square to the camera and the preview presenting at 29.3 fps. It measured
    **6.26 / 6.81 ms** — the same as the current build. The hypothesis is wrong and the build is not
    the variable.
  - What remains is that 19.2 / 29.2 ms belongs to a code or machine state that no longer exists,
    and cannot be reproduced today. It is superseded rather than explained. The related note that
    the second Vision pass cost "tracking mean 11.7 → 16.5 ms" comes from the same era and inherits
    the same doubt.
  - One asymmetry in the A/B, stated because it was not controlled: the new-build leg presented at
    3.9 fps against the old leg's 29.3, its window having ended up behind another. It does not
    affect the conclusion — the figure of interest sat at ~6.3 ms in both legs, and the display path
    is not in Vision's measurement — but the two legs were not matched on that axis.
- Resident memory over a separate 801 s run trended **down** (150 → 115 MB), so no leak is visible
  at that timescale.
- **The soak criterion passes.** An unattended run of the phase 6 release build recorded 5,859 s of
  awake time — 97.6 minutes — across 10.5 hours of wall clock. Resident memory trended
  **−0.79 MB/min** (264 MB peak at startup, 81 MB at the end); the criterion is no monotonic growth
  and the trend is negative. Taking only the **34.6 minutes of continuous streaming** before the
  machine first slept, which is the criterion read literally: memory min 106 / mean 132 / max 264 MB,
  trend **−1.83 MB/min**. 63 dropped frames over ~175,000 captured (0.036 %), and the virtual camera
  published 136,479 frames with 0 paced and 0 failed. Thermal state was nominal for 11,695 of 11,713
  samples; the only excursion was a nine-second nominal → fair → serious → critical → nominal
  sequence at the exact moment the machine went to sleep, not under our load.
  - The same run exercised **sleep/wake three times, unplanned**, including one 7 h 42 m sleep.
    Every cycle recovered: `didWake → retrying(attempt: 1)` → reopen 0.5 s later → frames confirmed
    at **1.83 s, 1.95 s and 1.85 s** from wake. That matches the single 1.8 s measurement taken
    earlier and turns it from one observation into four.
  - Note that elapsed time in the CSV is *awake* time: `HostClock` does not advance while the
    machine sleeps, which is why 10.5 hours of wall clock produced 98 minutes of samples.
- 193 unit tests pass (`swift test`). Covered: drop-stale backpressure and drop counting, box depth
  and reopen, 1€ jitter reduction and bounded-lag tracking, gate hysteresis / angle-limit fallback
  / slew ramp, blink preservation, filter reset on tracking loss and timestamp discontinuity,
  recovery within the 300 ms budget, gaze geometry and warp containment, metrics percentiles,
  the capture recovery policy for disconnect, runtime errors and sleep/wake, rate-meter decay
  under a stall, and the pacer that holds publication to the advertised frame rate.

### Phase 3 gate: passed on the budget we own, missed on the one we do not

The original reading of this gate was wrong in its diagnosis, though not in its verdict. It blamed
61 ms p95 processing on Vision at 29.2 ms; the controlled run puts Vision at **6.7 ms p95**, and
that 29.2 ms figure is not reproducible by any build (see the settled note above).

Measured under control, the stage this project actually owns — **ingest to corrected frame ready —
costs 0.75 ms mean / 1.28 ms p95**, against a 20 ms budget. Vision is 6.7 ms p95 and the warp
1.3 ms, both of which fit inside that budget with room to spare. There is nothing here that
justifies reassessing the correction model on speed grounds; that question is closed.

What still misses is **ingest → present at 32.0 ms p95**, down from the 61.2 ms originally
recorded but still above 20 ms, and every millisecond of the gap is display-path latency after the
frame is finished — compositor and drawable scheduling, which the virtual camera does not pay.
Queued, and still worth doing — but as efficiency work, not as a fix for this number: downscale the
tracker input, and consume the camera's native YUV instead of converting to BGRA per frame. Neither
is on the display path, so neither can move ingest → present. Filing them under this gate was a
mistake: the stages they shrink already fit the budget.

The YUV change in particular is larger than one line of `videoSettings`. Its scope was established
and then deliberately deferred until the host matrix is finished, so that a host failure has one
variable rather than two:

- the extension advertises `kCVPixelFormatType_32BGRA`, and CMIO forwards a format mismatch to
  hosts without complaint, so what the sink publishes has to stay BGRA unless the advertised format
  changes with it — and changing that would invalidate all three host results we have.
- passthrough returns the capture buffer itself, and correction is detected downstream by buffer
  identity (`corrected.pixelBuffer !== frame.pixelBuffer`). On a YUV capture buffer both the preview
  renderer and the sink would receive pixels they cannot sample, so passthrough — 10 % of frames in
  the controlled run — would need a convert-only GPU pass, adding a pass and a pool buffer where
  there is currently none, and the identity test would have to become an explicit flag.
- the warp shader would sample two planes and apply the YCbCr matrix carried on the buffer rather
  than a hardcoded one, or colour shifts.

So the saving is one AVFoundation conversion on every frame against a new blit on a tenth of them,
and it is unmeasured in both directions.

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

The conservative large-pose path is verified on hardware. In a 112.6 s installed-release run,
head pose reached 66.8° yaw and 27.7° pitch; 38 sampled intervals named `headPose`, set correction
to passthrough and kept the virtual camera streaming. Returning inside the trusted range restored
full correction for 55 sampled intervals. Capture/output held 30/29 fps, queue depth stayed zero,
the app dropped one startup frame and the virtual camera dropped none while sending 3,329 frames.
This verifies fallback and recovery, not visual naturalness. Glasses and genuinely low illumination
remain untested because neither physical condition was confirmed during the run.

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

### Phase 6: UI and hardening

The controls that were a seven-item toolbar are now a settings window, a menu bar and three
toolbar buttons, and the choices survive a relaunch — nothing was persisted before, so every launch
reset the redirect angle, the mirror and the overlay. The camera can be picked rather than
inherited from the system default (the path existed in `CameraCapture` and had never been wired to
anything), the extension can be removed as well as installed, and the diagnostics HUD can be hidden
without giving up any of the rows AGENTS §7 requires.

The pipeline moved up to the app so the settings window and the menu commands act on the same
instance the preview shows, and **New Window** was removed with it: a second window used to bring
up a second controller, and with it a second capture session on the same camera.

The window itself was then rebuilt around the picture rather than around the telemetry. The
diagnostics moved off the video into the window's inspector, so all thirty-odd rows stay one
keystroke away without covering the thing they describe; a status pill says in words what the
pipeline is doing — correcting, passing through, looking for a face, head turned too far — instead
of leaving that to be inferred from a blend percentage; and dead states use `ContentUnavailableView`
with the action that fixes them. One behavioural correction came with it: a virtual camera that
would not install used to cover the preview with an error, which contradicted the invariant that
the virtual camera is an output and never a dependency. It is a non-blocking notice now.

Moving the diagnostics also fixed a cost that had been there since phase 1. `tracking` was
published from the controller on every frame, so *every* view observing the controller — toolbar,
overlay, HUD — re-evaluated at capture rate, which is exactly what AGENTS §7 forbids. The overlay
now has its own small observable object, and the rest of the UI redraws on the half-second stats
tick.

Two measured gaps closed with it: the FPS meters no longer latch when nothing is being presented,
and the sink no longer publishes faster than the format the extension advertised. Both policies are
pure and live in `AspectusKit` with 11 unit tests, so neither needs a camera to check.

Closed at the end of the phase: the soak (97 minutes, negative memory trend), the first host of
the conferencing matrix (Zoom, with a controlled before/after in the extension log), and the
controlled latency run the tracking discrepancy demanded — that run was taken and is reported
above, and it superseded the 19.2 ms figure rather than explaining it. Google Meet and Microsoft
Teams have since been verified the same way, taking the matrix to three of six. Still open: Discord,
Slack and OBS.

## Known gaps

- ~~Output FPS latches at its last value when the window is occluded instead of decaying to zero.~~
  **Fixed and measured.** The meters were only recomputed when an event arrived, so a source that
  stopped went on reporting the last rate it saw. `RateMeter` now measures the sample span to *now*
  rather than to the newest sample, and the stats timer reads it instead of remembering what the
  last tick returned. Measured (release build, M3 / macOS 26.6, app hidden at 13.3 s and shown
  again at 27.3 s): output FPS fell 29.8 → 17.5 → 4.1 → **0.0 within 1.0 s** of the window being
  hidden, held 0.0 for the whole 14 s it stayed hidden, and recovered to 29.3 within 1.0 s of it
  coming back. Capture held 30.0 throughout and process held 29.2, which is correct rather than a
  remaining latch: the consumer loop goes on tracking and correcting while nothing is presented.
  Four unit tests cover the decay, the recovery and the two-sample floor.
- ~~The rate meters can report a frame rate the hardware cannot produce.~~ **Found on screen and
  fixed.** During the Teams test the HUD read **output 58 fps while capture and process both read
  30**, which is impossible — output frames are a subset of processed ones. Earlier, while Teams was
  installing, capture read **78 fps** on a camera whose every format caps at 30.
  - Cause: the rate was `(samples - 1) / (now - oldest sample in the window)`. That divisor is the
    spread of the samples, not the window. A source that stalls and then catches up puts a whole
    window's frames into part of one, so the meter reported the burst instead of the rate. The two
    readings are exactly that arithmetic: 30 samples bunched into 0.5 s give 58, and into 0.37 s
    give 78.
  - Fix: divide by the window. While the meter is younger than one window there is no window to
    divide by, so its own age stands in and a young meter still reads true instead of ramping.
  - Reproduced deterministically before fixing: a test that stalls, then delivers 30 frames at
    60 fps, read **60.0 fps** for a source that put 30 events in the window. The first attempt at
    that test passed against the bug — the stall plus the burst has to outlast the window, or one
    surviving pre-stall sample keeps the divisor wide and hides it.
  - Verified on hardware (release build, extension 11): the window was left fully occluded behind
    another app for 12 s and then revealed, which is the manoeuvre that produced the 58. Capture
    read 30 and process and output 28–29 across six samples over the first six seconds, with no
    reading above 30 at any point.
  - The decay behaviour above is unchanged and still covered; 183 unit tests pass.
- ~~Whether Vision supplies true pupil landmarks or falls back to the eye-contour centroid is
  unconfirmed.~~ **Resolved by measurement** (phase 1 diagnostics, release build, M3 / macOS 26.6,
  FaceTime HD at 1280×720): `leftPupil`/`rightPupil` are populated on **100 % of tracked frames**,
  one point per eye, over two runs. The contour-centroid fallback never fired. It remains in the
  tracker as a guard and is now reported in diagnostics rather than being silent.
- ~~`VNFaceObservation.confidence` measures exactly 1.000 on every tracked frame, so the gate's
  confidence hysteresis never fires on that input alone.~~ **Superseded by measurement.** That
  reading came from the landmarks-only request, which reports a constant 1.000. Since the tracker
  was changed to run `VNDetectFaceRectanglesRequest` first and chain its observation into the
  landmarks request, the confidence surviving onto the result is the detector's own. Measured over
  a 1,051-frame release run: face confidence spans **0.584 … 0.912** (mean 0.808), and the gaze
  confidence the gate consumes — face score × the per-eye agreement factor — spans
  **0.348 … 0.912** (mean 0.855). So the face score now carries information, enough on its own to
  sit under the 0.6 engage threshold; crossing the 0.4 disengage threshold still needs the 0.5
  agreement factor, since the lowest face score alone stays above it.
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
- **Per-axis scale is fitted from physical display geometry.** The current 550 mm calibration
  measures `yawGain ×1.00` and `pitchGain ×2.69`; horizontal scale is already accurate while the
  geometric vertical estimate still under-reads by about 2.7× after the anchor fix below.
  - The fit depends on a user-supplied viewing distance. macOS exposes no camera field of view
    (`videoFieldOfView` and `videoFieldOfView(for:geometricDistortionCorrected:)` are both
    `API_UNAVAILABLE(macos)`), so scale cannot be recovered from the image.
  - Yaw is fitted from the two side targets and pitch from the lens plus the bottom edge. The target
    above the camera has no measurable physical position and remains a sign/separation check only.
- **The vertical estimator defect was isolated and fixed on hardware.** Two controlled runs using
  `pupilCenter.y − eye.region.center.y` failed calibration at **1.7°** and **1.3°** of up-versus-down
  separation while horizontal separation remained normal. The second run logged an independent
  vertical anchor: the midpoint of the two eye corners.
  - Down minus up pupil travel relative to the corner line was **+0.002477**. The eyelid-region
    centre moved **+0.001845** in the same direction, leaving only **+0.000632** in the old offset.
    The moving aperture centre therefore cancelled **74.5 %** of the usable vertical signal. Both
    eyes agreed independently: corner-relative travel was +0.002459 left and +0.002494 right.
  - Horizontal remains relative to `region.center.x`. Vertical now uses the corner midpoint, with
    the corner signal smoothed by the same 1€ filter as the other landmarks. A missing corner falls
    back to the old aperture centre rather than making the frame unusable.
  - Calibration schema version **2** prevents a version-1 fit, learned from the incompatible
    aperture-relative signal, from being applied silently.
  - Two calibrations after the change passed at **10.1° / 29.6°** and **11.8° / 33.1°** vertical /
    horizontal separation. The saved 300-sample fit has neutral bias yaw −0.79°, pitch +8.92°,
    `yawGain ×1.00`, `pitchGain ×2.69`, and 60 accepted samples at every target.
  - The 49.6 s clean Release run after the change measured Vision **5.94 / 6.45 ms**, track loop
    **5.96 / 6.47 ms**, warp **0.61 / 0.72 ms**, and owned pipeline **0.07 / 0.56 ms** mean / p95.
    One frame dropped, depth stayed zero and thermal state stayed nominal. Preview presentation was
    **30.5 / 32.3 ms**, the already-known compositor wait that the virtual camera does not pay.
- **Head compensation is procedure-sensitive, so only repeatable fits are trusted.** The first
  version-2 sweep produced large pitch cross-coupling and a false 24° live result despite a nearly
  neutral raw eye reading; that calibration was rejected and overwritten. The repeat produced
  `yawFromYaw +0.280`, `yawFromPitch −0.066`, `pitchFromYaw +0.019`, `pitchFromPitch +0.081`, close
  to the earlier independent fit (+0.192, −0.115, −0.004, +0.055). The sweep remains skippable.
- Calibration samples retain the raw per-eye offsets, and failed calibration sheets show per-target
  means, so future fit failures remain diagnosable even though unsuccessful runs are not saved.
- The apparent 15° runtime head-gate anomaly was a comparison of different limits, not a gate
  failure: fixation calibration uses 15°, while normal estimation uses the intended 25° limit.
- Pupil source remained `visionLandmark` for both eyes on 100 % of the measured frames, one point
  per eye; the contour-centroid fallback did not fire.
- **Camera disconnect and session runtime errors are handled but remain unverified on hardware;
  sleep/wake and stop/restart are verified.** The session posts runtime errors, interruptions and
  device connect/disconnect; `NSWorkspace` supplies sleep and wake. The policy that turns those into
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
    hardware. The reference machine exposes one physical device, the built-in FaceTime camera;
    Aspectus and GazeAt are virtual cameras and cannot substitute for an unplug test.
  - An automatic reopen deliberately refuses to select Aspectus's own virtual camera, which is a
    video device like any other and would otherwise feed the pipeline its own output.
- **Restart after stop is verified end to end** in the installed, notarized build. Capture and
  output returned at 30 / 29 fps, preview and virtual-camera counters resumed inside the next
  one-second sample, full output rate returned within two seconds, queue depth stayed zero and the
  dropped-frame count remained at one across the stop/restart cycle.
- ~~**The sink detects an extension replaced under it, and then fails to reconnect — silently.**~~
  **Diagnosed, and the cause turned out to be the OS rather than the lookup.** Found while preparing
  the Meet test: the extension was upgraded from version 5 to 6 through the Install button with the
  app running. `revalidate` noticed correctly and logged `the virtual camera stopped draining after
  60 consecutive frames, reconnecting`, then retried `connect` twice a second for over 80 s without
  ever succeeding, silently, with the frames-sent counter frozen.
  - **A replaced CMIO device is invisible to every process that was already running.** Measured with
    two probes held across a real replacement, identical but for one variable: both were told the
    old device had gone — a `kCMIOHardwarePropertyDevices` listener fired 140 ms before polling
    noticed — and **neither was ever told the new one arrived**. A listener is not the missing
    piece: it reports the removal and stays silent on the re-add.
  - Five in-process lookups were then tried from a blinded process, and **all five fail**: the
    device array, `kCMIOHardwarePropertyDeviceForUID` translation of the extension's stable uid,
    AVFoundation's own discovery session, AVFoundation followed by the array, and poking a system
    property to force a rescan followed by the array. A process launched at the same moment found
    the device by all five at once, at the same device id the blinded process had lost. The
    blindness held for 12 minutes before the probe was stopped; the earlier probe held it for 13.
  - So there is no in-process cure, and a uid fallback was written, measured to be useless, and
    removed rather than shipped as a fallback that never fires.
- **What the app does about it now.** Since the camera cannot come back, the fix is to stop failing
  silently and to say the one thing that works.
  - Every `connect` failure names its reason and is reported on the edge — once per run of failures
    rather than at the 2 Hz the stats timer retries. Measured over a 53 s outage, roughly 106
    attempts: **one log line**, against none at all before.
  - A camera this process held and lost for more than 3 s is reported as lost, which surfaces a
    non-blocking notice over the preview and a **Relaunch** button. The preview and correction carry
    on untouched, because the virtual camera is an output and never a dependency.
  - A camera the user removed on purpose is excluded, so **Remove** does not advise a pointless
    relaunch. That needed a smaller fix underneath it: both activation and deactivation report
    success through the same delegate callback, so the installer had been reporting `active` after a
    successful removal.
  - **Verified end to end** (release build 11, notarized, M3 / macOS 26.6): extension replaced under
    the running app at 20:26:51.341, one failure line at 20:26:51.346, Relaunch pressed, and the new
    process connected at 20:27:41 to extension version 11.
  - Not verified: a *first* install while the app is running, which may be blind for the same reason
    and would show nothing, since a camera never held cannot be reported as lost. Testing it means
    uninstalling the extension, which was judged not worth doing to a working machine. The removal
    path's wording is likewise reasoned rather than measured.
- ~~The virtual camera advertises 30 FPS while capture targets 60, so on a camera that reaches 60
  the app would publish faster than the format it advertises.~~ **Fixed, unit-tested rather than
  measured.** Publication goes through `PublishPacer`, a credit balance that holds the sink to the
  advertised frame duration and counts what it withholds apart from failures. Capping capture
  instead was rejected: 60 FPS is a stated criterion, and the cap belongs at the boundary that made
  the promise. Seven unit tests cover parity, timestamp jitter, a 60 → 30 halving, a source just
  over the cap losing only its excess rather than beating down below it, burst suppression after a
  gap, and a presentation clock that restarts in the past. On hardware the pacer is inert by
  construction — the reference camera caps at 30 — and a 39 s release run published 1,046 frames
  with **0 paced and 0 dropped**, which is the parity case and no evidence about the 60 FPS path.
  Resolution cannot drift the same way: it is one constant compiled into both targets, pinned in
  the capture output, and checked against real buffers before publishing.
- ~~The virtual camera has never been soak-tested. One session recorded resident memory at 433 MB
  against a ~110 MB baseline, after the stats recorder had already stopped writing.~~ **Superseded
  by the 97-minute soak above**, which published 136,479 frames and ended at 81 MB with a negative
  memory trend. The earlier 433 MB reading is still unexplained, but it is no longer the only
  sustained observation, and nothing like it reappeared.
- ~~The host matrix is untested — the virtual camera has never been tried in a conferencing app.~~
  **Three hosts of six are now verified: Zoom, Google Meet and Microsoft Teams.** Aspectus appears
  in Zoom's camera list beside the physical ones, and selecting it produced a live picture in Zoom's
  video preview. The control the last host test lacked is present this time, from the extension's
  own log rather than from the host's UI: `forwarding=false` before selection, `forwarding=true`
  with the source counter climbing ~30 buffers/s while selected, and `forwarding=false` again the
  moment the camera was set back to FaceTime HD. Discord, Slack and OBS remain untested.
  - **Google Meet passes**, in an instant meeting rather than a settings preview, so the in-call
    video path was the one exercised. Notarized Developer ID build of commit 33df5c4, extension
    version 6, M3 / macOS 26.6, Chrome 150. The same log control, with Meet on the physical camera
    as the baseline: `forwarding=false`, then `source startStream - a host attached` at 19:50:49 the
    moment Aspectus was selected, the source counter going `forwarded 1` → `301 buffers` in 10 s
    (≈30 fps), and `forwarding=true`. Detaching was tested twice — switching the camera back to
    FaceTime HD, and leaving the call outright — and both produced `source stopStream` followed by
    `forwarding=false`.
  - Meet's own `MediaStreamTrack` reported `label: "Aspectus"`, **1280×720 at 30 fps**, matching the
    advertised format exactly, so nothing was forwarded to the host as a mismatch. A canvas diff of
    two frames 1.2 s apart confirmed a moving picture rather than a frozen one, which a screenshot
    of the host's own preview cannot establish on its own.
  - The app's HUD read `Correcting 100%` throughout at 30 fps capture / 30 fps output, and the
    virtual camera panel ended the session at **8,433 sent · 0 paced · 0 dropped**. These are HUD
    readings taken during the host test, not a CSV benchmark run, and no latency figure is quoted
    from them.
  - **Microsoft Teams passes** (desktop app, build 26198.202.4929.7171, on extension version 11).
    Aspectus is listed in Settings ▸ Devices ▸ Camera beside FaceTime HD, and selecting it put a
    live corrected picture in Teams' own video preview. Same log control, same shape as the others:
    `forwarding=false` on the physical camera, then `source startStream - a host attached` at
    20:48:36.923 the moment Aspectus was picked, `forwarded 1` → `301 buffers` in 10 s, and
    `source stopStream` with `forwarding=false` on switching back. No spin: eight lines after the
    detach, and a peak of two log lines in any one second across the whole session.
  - **This result is weaker than the Meet one, in one specific way.** Teams is a native app, so
    there was no way to read back the `MediaStreamTrack` it negotiated; the geometry is right on our
    side, where the extension advertises it and the sink checks real buffers before publishing, but
    unlike Meet there is no reading from inside the host. Treat "Teams sees 1280×720 at 30" as
    inferred rather than measured.
  - Getting there needed the Teams desktop app updated: the installed copy was hard-blocked behind a
    mandatory update wall, and `teams.microsoft.com` failed to render at all, dying in auth
    initialisation across three loads despite a valid Microsoft session.
- **A host disconnecting used to spin the extension.** Found by that same test: when the sink
  stream stops, `consumeSampleBuffer` fails immediately, and the completion handler re-armed with
  no delay — measured at **1,189,895 log lines in 1.877 s**, roughly 634,000 failed consumes per
  second, burning a core inside a system extension and flooding the unified log. Fixed: a failed
  consume now retries on the same 0.1 s timer the late-client path already used, and a run of
  failures logs once rather than once each, with a recovery line when it clears. The loop still
  keeps exactly one request outstanding.
  - **Nothing like it reappeared under Meet**, across two detaches: five log lines each time, a peak
    of four lines in any one second over the whole five-minute session, and extension RSS flat at
    13.4 MB. That is weaker evidence than it looks, and is recorded as such — under Meet the sink
    stream never stopped, the consume counter kept climbing straight through both detaches, so the
    failing branch this fix guards was never entered. Meet shows a host detach that does not spin;
    it does not re-test the path that did.
