# Aspectus design and measured status

Aspectus is a native Apple-Silicon macOS video pipeline for subtle eye-contact correction. The
intended behavior is to redirect only the original eye pixels toward the physical lens while
preserving identity, expression, blinks, glasses and lighting. Uncertain frames must pass through
unchanged.

Current conclusion: capture, tracking, rendering, temporal safety and the virtual camera are
implemented, but the active geometric gaze estimate is not accurate enough for natural
lens-directed correction. The bounded appearance-model direction also failed its honest
cross-session gate. No learned model is integrated, no candidate is frozen, and no model weight is
shipped.

Unless stated otherwise, hardware evidence below comes from the reference Apple M3 running macOS
26.6 with Xcode 26.6 and the built-in FaceTime HD camera at 1280×720, 30 fps.

## 1. Product and pipeline invariants

| Concern | Required behavior |
|---|---|
| correction | resample original pixels and touch the smallest practical eye region |
| fallback | output the original frame when tracking, gaze or correction is unsafe |
| temporal behavior | preserve blinks; use adaptive filtering, hysteresis and slew-limited strength without visible lag |
| backpressure | keep at most one frame in flight; drop and count stale frames rather than queue them |
| hot path | keep frames in `CVPixelBuffer` / IOSurface / Metal textures without full-frame CPU copies |
| concurrency | keep capture, Vision and future Core ML inference off-main and cancellation-aware |
| timing | carry capture and ingest timestamps; measure rather than estimate latency |
| processing latency | define as ingest → present; target p95 `< 20 ms` |
| end-to-end latency | define separately as camera PTS → present because it includes sensor delivery |
| runtime | no Python, ONNX, PyTorch, cloud inference or runtime model download |
| privacy | local processing by default; biometric collection is explicit, owner-only and never committed |

Original-pixel resampling reduces whole-face synthesis and identity-drift risk. It does not prove
that a warp is natural under glasses, eyelid motion, low light or occlusion; those require controlled
visual evidence.

## 2. Architecture

```text
AVFoundation capture
  → FaceTracker
  → GazeEstimator
  → TemporalStabilizer and CorrectionGate
  → EyeCorrector
  → FrameCompositor
  → preview and FrameSink
  → CoreMediaIO camera extension
```

`PipelineController` orchestrates replaceable stages but does not own model or framework behavior.
`LatestValueBox` is the only frame hand-off: newest frame wins, stale work is counted and discarded,
and queue depth cannot grow.

The framework-free `AspectusKit` owns frame types, stage protocols, geometry, filters, confidence
gating, recovery decisions, calibration math, pacing and metrics. AVFoundation, Vision, Metal,
CoreVideo and future Core ML bindings remain in `App/`. The CoreMediaIO extension is a separate
sandboxed process and shares only the format/transport contract in `Shared/`.

The current hot path maps camera pixel buffers through IOSurface-backed Metal textures. The eye
shader reads and resamples the original frame; the preview mirror is a display transform and does
not alter virtual-camera output.

## 3. Current correction and fallback

Apple Vision supplies the primary face, eye contours, pupil points and head pose. Hardware work
established two important limits:

- a landmarks-only request did not provide usable head pose; chaining a revision-3 face-rectangle
  observation into the landmark request made roll, yaw and pitch available on tracked frames
- using the moving eyelid-aperture center as the vertical pupil reference cancelled about 74.5% of
  the measured vertical signal; the corner midpoint is now the vertical anchor

The runtime estimator combines pupil geometry, calibrated per-axis gains and a small fitted
head-coupling term. It improved target separation, but controlled quality captures still produced
implausible vertical estimates and cross-session drift. A personalized landmark-only replacement
was also tested and rejected. The runtime therefore remains a known-inadequate geometric baseline.

Safety remains useful independently of estimate quality:

- full-strength correction is limited to 18° and fades to zero by 24°
- confidence engagement and disengagement use hysteresis
- blend changes are slew-limited
- closed or uncertain eyes pass through to preserve blinks
- excessive head pose, stale tracking, invalid geometry or rendering failure returns the original
  frame
- calibration files are versioned; corrupt or incompatible files are rejected rather than partly
  applied

A future learned estimator must replace `GazeEstimator` behind the existing seam. It must not
silently fall back to the rejected geometric estimate. Failure or uncertainty must select the
original frame.

## 4. Model and data research

The bounded landscape audit was checked on 2026-08-10. Repository code licences, model-weight
licences, dataset access, commercial-training rights, derived-weight rights, raw-data redistribution
and biometric consent/retention were evaluated separately.

| Candidate | Useful part | Evidence and feasibility | Rights conclusion | Decision |
|---|---|---|---|---|
| Apple Vision | native face, eye, pupil and pose tracking | measured at 6.71 ms p95 on the reference Mac; hardware placement is not claimed | Apple SDK | keep for tracking |
| Hsu “Look at me!” | flow-field prediction and bilinear original-pixel resampling | compact learned warp reference; TensorFlow 1 and custom transform need a clean implementation | BSD-3 code; tracked checkpoints and private DIRL training data have no separate cleared terms | reject weights; retain concept |
| Open Model Zoo `gaze-estimation-adas-0002` | paired-eye plus head-pose initializer | 1.882M parameters, published 6.95° held-out mean; locally converts to a 3.8 MB FP16 ML Program with ≤0.0019 output-vector difference | model files Apache-2.0; internal 60-person data lacks public consent/retention/derived-weight evidence | local personalized audit only; full fine-tuning rejected |
| WebEyeTrack / BlazeGaze | frozen compact encoder and few-shot personalized head | about 0.16M parameters; output contract and metric do not match Aspectus | MIT code; MPIIFaceGaze-derived checkpoint is not commercially cleared | retain technique, reject checkpoint |
| TPGaze | small person-specific prompt adaptation | published means do not establish session-isolated p95 or lens tails | Apache-2.0 code, no released weights, restricted research datasets | retain concept only |
| LightGazeNet | paired eyes, explicit head geometry and calibration embedding | 3.48M parameters; no public Core ML or tail evidence | no released code or weights; training datasets are restricted | replacement reference only |
| full-face neural rendering | quality-ceiling research | too slow and materially increases identity, glasses and temporal risk | research-specific and dataset-dependent | reject for Aspectus |

Primary references are pinned where a repository artifact informed the decision:

- [Hsu gaze correction at `7cc76e4`](https://github.com/chihfanhsu/gaze_correction/tree/7cc76e4ccaf76c950730c593ffa22bed30c807a4)
- [Open Model Zoo 2021.2 model at `3386309`](https://github.com/openvinotoolkit/open_model_zoo/tree/338630987b403a6981d03ab6d04c2d5ad367793a/models/intel/gaze-estimation-adas-0002)
- [PINTO converted directory at `1daa964`](https://github.com/PINTO0309/PINTO_model_zoo/tree/1daa9648a887e0de44630cc822807ad1fc7c0bb1/091_gaze-estimation-adas-0002)
- [WebEyeTrack at `14719ad`](https://github.com/RedForestAi/WebEyeTrack/tree/14719ad861467c98890058f7c41a94638ae1db2b)
- [TPGaze at `bc8b57c`](https://github.com/hmarkamcan/TPGaze/tree/bc8b57cca9c83299bdc7f6d92a7df501bbb9476c)
- [LightGazeNet project and WACV 2026 paper](https://eyelignai.github.io/lightgazenet/)

The current runtime Metal warp is independently written geometric iris translation. It is not a
native implementation of Hsu's learned network, and no code, checkpoint, weight or data from that
project is included.

### Dataset directions

| Dataset | Commercial/derived-weight evidence | Decision |
|---|---|---|
| [MPIIGaze / MPIIFaceGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild) | noncommercial scientific use | reject |
| [ETH-XGaze](https://xgaze.ait.ethz.ch/) | commercial use and model redistribution prohibited by its terms | reject |
| [Gaze360](https://github.com/erkil1452/gaze360) | noncommercial research; derived-weight grant not established | reject |
| [Columbia CAVE](https://cave.cs.columbia.edu/repository/ColumbiaGazeDataSet) | noncommercial; commercial biometric rights absent | reject |
| [GazeGene](https://phi-ai.buaa.edu.cn/GazeGene/) ([licence](https://phi-ai.buaa.edu.cn/project/GazeGene/License.pdf)) | CC BY-NC-SA; commercial use and redistribution are not permitted | reject |
| [EVE](https://files.ait.ethz.ch/projects/EVE/EVE%20dataset%20Terms%20and%20Conditions.pdf) | noncommercial and redistribution-restricted | reject |
| [GazeCapture](https://gazecapture.csail.mit.edu/) | public site does not establish the needed commercial, withdrawal and retention terms | do not use without a separate agreement |
| first-party Aspectus collection | viable only with explicit purpose, commercial-training and derived-weight terms plus retention, deletion and withdrawal rights | recommended direction, subject to approval |

Access to data is not permission to train commercially. A code licence is not a checkpoint licence,
and checkpoint access is not permission to distribute derived weights or source data.

## 5. Phase 3 appearance-model gate

The immediate research question was deliberately narrow: can a personalized paired-eye model pass
on the same person and reference setup without session leakage? The fixed gates are:

- median angular error `≤ 2°`
- overall p95 angular error `≤ 5°`
- physical-lens p95 angular error `≤ 3°`

They are evaluated without rounding, pooling or redefining lens samples.

### Historical evidence

| Evidence | Median | Overall p95 | Lens p95 | Interpretation |
|---|---:|---:|---:|---|
| strongest consumed development candidate | 1.957047° | 4.714317° | 2.963580° | development pass only |
| same frozen candidate on the next untouched session | 1.679981° | 5.169673° | 3.677630° | final failure |
| retrained after consuming that session | 2.139990° | 5.566082° | 2.036547° | rejected |

The strongest development session used schema 2 and had already influenced preprocessing and
augmentation; schema-3 data was in its training set. The next schema-3 session was honest for the
frozen evaluation, then became consumed when its failure informed diagnosis and later training.
None of these results is final proof.

A role audit found seven completed sessions. All have influenced training, preprocessing,
augmentation, selection or diagnosis. The current protocol uses four fixed training sessions and
two physically valid schema-2/3 sessions as simultaneous development evidence; the remaining
consumed schema-1 evidence is not eligible for honest pose validation because recorded head poses
do not separate the prompts reliably.

### Schema 4 foundation and blocking pitch contradiction

The tracked schema-4 foundation defines canonical paired-eye crops directly in source-pixel
coordinates. It records source dimensions; versioned crop, physical-lens label and Apple Vision
head-pose contracts; raw eye axes; one paired rotation and scale; inter-eye disagreement; pupil and
contour evidence; and geometric crop-clipping fractions. Missing or degenerate axes reject a frame,
and a source-frame size change fails the session. These are inspectable quality signals, not a
validated occlusion score or an approved clipping, disagreement or occlusion threshold.

Historical schemas 1–3 retain only their final `60 × 60` crops. They lack the source axes and
requested crop geometry needed to reproduce the canonical affine sample, so they cannot be
converted into schema 4 or support a paired legacy-versus-canonical alignment ablation. A second
transform of those PNGs would be a different, lossy input path.

Numeric pitch direction is a pre-training gate. Target labels must reproduce the physical display
geometry. Apple Vision revision 3 defines positive head pitch as head-down, so each eligible session
must show `lookUp − neutral ≤ -5°` and `lookDown − neutral ≥ 5°` from the recorded numeric values.
Prompt names never authorize flipping historical rows.

The two eligible consumed development sessions pass the within-session magnitude and
opposite-direction checks but disagree on the cross-session `lookUp − neutral` sign. A read-only
provenance investigation on 2026-08-11 resolved the contradiction as a genuinely inverted
recording, not a producer or import bug: the installed SDK header independently confirms the
head-down-positive convention; no historical tracker, recorder, gate or loader version ever
negated pitch; and the inverted session's neutral pitch median and yaw prompt signs match every
other session, which a numeric sign flip would also have inverted. The historical collector
learned the vertical prompt sign from the first accepted `lookUp` sample and then coached the
participant to hold it, so one session recorded faithful Vision values under physically inverted
prompted motion. Its numeric values are therefore unchanged and it remains consumed, but it is not
eligible pitch-gated development evidence, and no per-session correction is permitted. With only
one eligible development session, the two-development-session comparison contract cannot be
satisfied, so no model comparison was launched. The schema-4 collector gate now enforces the
Vision convention directly and cannot re-record this inversion.

### Deterministic selection and bounded tuning

Every epoch is scored independently on each development session. Selection minimizes:

```text
max over sessions of max(median / 2, p95 / 5, lens_p95 / 3)
```

Ties prefer lower worst lens ratio, lower worst overall-p95 ratio, lower worst median ratio, then an
earlier epoch. Selected weights are restored before saving. The seed-7 baseline selected epoch 70,
not the final epoch 80, and still failed with a score of `1.043166`.

Six one-factor screens covered scheduling, weight decay, selective freezing, tail-aware loss,
augmentation and crop-quality filtering. Promising configurations were repeated with fixed seeds
7, 19 and 43:

| Procedure | Seed 7 | Seed 19 | Seed 43 | Passing seeds | Decision |
|---|---:|---:|---:|---:|---|
| baseline | 1.043166 | 1.067689 | 0.950827 | 1 / 3 | unstable |
| stronger augmentation | 0.966901 | 1.041593 | 0.959070 | 2 / 3 | reject |
| minimum face confidence 0.78 | 0.975835 | 1.075358 | 1.054084 | 1 / 3 | reject |

Every candidate seed had to pass both development sessions and improve over its paired baseline.
Stronger augmentation failed seed 19 and regressed against baseline at seed 43. The stricter
confidence filter failed seeds 19 and 43 and regressed against both paired baselines. The bounded
budget therefore rejects continued hyperparameter tuning of the current full-fine-tuned model.

### Leakage boundary and final evaluation

Training uses explicit session roles and never a row-level random split. Development sessions are
reported separately. The frozen candidate procedure requires three candidate runs, their three
paired baselines, fixed seeds, one declared factor change, immutable artifact hashes, machine and
dataset contracts, and seed 7 nominated before final evaluation.

An immutable manifest binds all six runs and a privacy-safe snapshot of completed sessions. Only
after the freeze command returns may one new session be recorded. Before scoring, a workspace ledger
atomically claims both the final-session fingerprint and the candidate's semantic identity. Each
session and candidate gets one final attempt; copied paths, changed timestamps or a replacement
baseline do not reset it. An interrupted or failed attempt is consumed.

There is no current candidate manifest and no untouched completed session. Recording another final
session now would be repeated sampling, not validation. Exact commands and durability rules are in
[Training/README.md](../Training/README.md).

## 6. Native Release measurements

### Controlled latency run

A 174-second Release run kept the preview visible and display awake, held the subject roughly
square to the camera, and engaged correction on about 90% of samples.

| Metric | Mean | p95 | Result |
|---|---:|---:|---|
| Vision face/eye/pupil/pose | 6.1 ms | 6.71 ms | stage measurement |
| geometric Metal eye warp | 0.8 ms | 1.32 ms | stage measurement |
| ingest → present | 25.9 ms | 32.0 ms | fails `<20 ms` processing target |
| camera PTS → present | 54.0 ms | 60.1 ms | end-to-end reference only |

Queue depth never exceeded one. The inexpensive geometric warp does not predict the latency of a
future learned model. The reference camera exposes only 30 fps formats, so sustained 60 fps remains
unverified.

### Soak and recovery

A separate 97.6-minute Release soak published 136,479 frames to the virtual camera and dropped 63
of roughly 175,000 captured frames (`0.036%`). Queue depth stayed at one or less and the fitted
resident-memory trend was negative. Three sleep/wake cycles recovered frame delivery in about
1.8–2.0 seconds. This supports sustained behavior on the reference setup; it is not a general
hardware guarantee.

Stop/restart and sleep/wake were exercised end to end. Physical camera disconnect and capture
runtime-error recovery are covered by deterministic policy tests but have not been exercised on
hardware because the reference Mac exposes only its built-in physical camera.

### Visual and fallback evidence

- controlled captures showed original-frame fallback at excessive head pose
- blinks and correction transitions have deterministic core coverage
- calibration target separation improved after the vertical-anchor correction, but final captures
  still rejected the geometric gaze estimate
- neutral resting flicker, natural glasses behavior, partial occlusion, practical low light and
  large off-axis correction have not completed controlled before/after validation

Unit tests and angular metrics cannot prove that corrected eyes look natural.

### Virtual camera and hosts

Signing, notarization, activation and enumeration work on the reference Mac. The virtual camera is
verified in:

- Zoom video preview
- a Google Meet call, including host readback of 1280×720 at 30 fps and moving-frame evidence
- Microsoft Teams desktop preview; format is enforced on the Aspectus side but was not readable
  from inside the native host

Discord, Slack and OBS are untested. Compatibility must be rechecked after changing the runtime
model or frame timing.

A host-detach busy loop was fixed and did not recur in later host sessions. When a running process
loses an extension that was replaced in place, macOS does not expose the replacement device back to
that process; Aspectus reports the loss and offers relaunch while leaving preview running. That
relaunch path is measured. First-install visibility from an already-running process remains
unverified.

The published v0.1.0 build is signed and notarized technical-preview evidence from an older commit.
It does not prove the current source, gaze quality or all-host compatibility.

## 7. Implementation state

| Phase | Evidence level | Current decision |
|---|---|---|
| capture, preview and diagnostics | Release hardware-measured | keep |
| Vision tracking and pose | Release hardware-measured | keep; placement is not claimed |
| geometric correction | Release hardware-exercised; naturalness rejected or unproven | keep only as rejected development baseline |
| appearance gaze estimator | schema-4 foundation, synthetic Release-tested crop/serialization path, compact comparator unit-tested offline | execution blocked by insufficient eligible development evidence; OMZ direction rejected |
| temporal gate and fallback | unit-tested; selected fallback paths hardware-measured | keep |
| CoreMediaIO virtual camera | signed/notarized and three-host measured | keep; matrix incomplete |
| UI and lifecycle hardening | Release-tested with sustained soak | keep; product quality remains blocked by gaze |

These claims remain separate:

1. personalized reference-machine model proof
2. native runtime integration
3. sustained visual and performance validation
4. general-user model and onboarding
5. conferencing-host compatibility
6. redistributable release readiness

A pass at one level does not imply the next.

## 8. Remaining blockers and next step

The head-pitch contradiction is diagnosed: one consumed development session is genuinely
physically inverted and stays excluded from pitch-gated evidence, leaving only one eligible
development session. The immediate blocker is therefore insufficient eligible development
evidence, and beyond it angular-tail stability, especially vertical and physical-lens error,
remains unresolved. The leading hypotheses are crop alignment and clipping, occlusion/tracking
quality, limited first-party session diversity and insufficient model/input structure. Core ML
conversion is not the blocker.

The next step is not training, native integration or an immediate recording. The offline label
gate must pass before model code can execute, and it cannot pass under the frozen two-session
development roles. Producing new eligible development evidence means explicitly consented
collection under the fixed schema-4 collector, which requires approval and physical
participation. The production Core Image crop sampling and schema-4 serialization are now
Release-tested against deterministic synthetic frames, but the rendered crop remains unverified
on recorded biometric evidence and canonical alignment remains unmeasured until compatible
development evidence is legitimately available.

If the pitch contract is resolved, any compact comparison must be predeclared, use only cleared
first-party data and random initialization, preserve the fixed session roles and exact gates, and
change one major factor at a time. The smallest capacity screen is exactly two 80-epoch seed-7 runs:
the `156,226`-parameter compact estimator and the fixed OMZ baseline on the same legacy crops, roles
and shared procedure fields. Run the pair at seeds 19 and 43 only if the compact seed-7 run passes
every per-session gate and strictly improves over its paired baseline, for a maximum of six runs.
This screens the declared model factor on legacy inputs; it cannot validate schema-4 alignment.

Do not integrate a learned estimator or request a new untouched recording until a redesigned
procedure passes both development sessions at all three fixed seeds and is frozen into the current
manifest/ledger protocol. If a frozen candidate later passes one new untouched session, request
approval for native integration; do not treat that personalized pass as general-user or
redistribution evidence.

Other unresolved risks remain:

- processing latency already misses the current `<20 ms` ingest-to-present target
- learned-model Release latency, compute-unit behavior and thermal impact are unmeasured
- 60 fps, other Apple-Silicon hardware, camera switching and physical disconnect need measurement
- natural quality under glasses, low light, occlusion and large head pose is unproven
- Discord, Slack and OBS remain untested
- contributor signing requires coordinated identifier changes
- first-party multi-person biometric collection needs an explicit consent and retention decision

Open-source availability of the pipeline does not make its current output production-ready and does
not clear third-party model or dataset rights.
