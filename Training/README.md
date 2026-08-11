# Gaze estimator training

`Training/` is an offline development environment for a personalized, session-isolated gaze proof.
Python, PyTorch and ONNX are never part of the Aspectus runtime. A passing run may export a Core ML
ML Program, but no learned model is currently approved for native integration or redistribution.

## Current Phase 3 result

The current Open Model Zoo full-fine-tuning direction is rejected. A strong consumed-development
checkpoint scored `1.96° / 4.71° / 2.96°` against the median, overall-p95 and physical-lens-p95
gates. Frozen evaluation on the next untouched session scored `1.68° / 5.17° / 3.68°`: median
passed, both tail gates failed. That session then became development evidence.

All seven completed sessions have influenced training, preprocessing, selection or diagnosis. None
remains untouched. With two physically valid schema-2/3 sessions fixed as simultaneous development
evidence, deterministic checkpoint selection and bounded one-factor tests produced:

| Procedure | Seed 7 | Seed 19 | Seed 43 | Passing seeds | Decision |
|---|---:|---:|---:|---:|---|
| baseline | 1.043166 | 1.067689 | 0.950827 | 1 / 3 | unstable |
| stronger augmentation | 0.966901 | 1.041593 | 0.959070 | 2 / 3 | reject |
| minimum face confidence 0.78 | 0.975835 | 1.075358 | 1.054084 | 1 / 3 | reject |

Each value is the worst normalized gate across both development sessions; results are never pooled
to hide a failing session. Every candidate seed had to pass and improve over its paired baseline.
No procedure is frozen, another final recording is not justified, and no model is ready for native
integration. The tracked input/model redesign is now adopted as a foundation, but its pre-training
label gate rejects the current evidence before any new model run. See
[docs/DESIGN.md](../docs/DESIGN.md).

Preserve the local v19–v31 artifacts as consumed-development evidence, but never freeze or evaluate
them as a final candidate: they predate checkpoint format 3. A future candidate must be retrained
under the current provenance and lens-coverage contract.

### Schema 4 input contract and current stop

Schema 4 records two canonical paired-eye crops directly from each source frame. Each eye axis is
the farthest contour-point pair measured in source pixels and ordered by image x. Both eyes use the
circular mean of their axes as one rotation and `1.8 ×` the larger axis length as one square scale;
each crop remains centred on its own axis midpoint and is sampled to `60 × 60` through the same
explicit edge-clamped affine contract.

That shared scale is the `canonical-paired-eye-v1` defect. Verifying the rendered pixels of the one
recorded schema-4 session against this contract confirmed the renderer is exact — the per-eye axis
midpoint lands at the crop centre to 2.8e-14 px and the longer axis at exactly `60 / 1.8` px — but
one shared side taken from the longer axis means head yaw foreshortens the far eye into a smaller
rendering: `33.33 px` against `24.5 px` in the turned blocks, a `0.736` ratio tracking
`cos|head yaw|` at `r = +0.980`, so a single eye spans a ~40% scale range across a session. Head
yaw is correlated with the gaze label by construction of the pose plan, so crop zoom encodes the
target. The recomputed alignment evidence cannot catch this: the arithmetic is exact and the defect
is in the contract's definition.

`canonical-paired-eye-v2` therefore changes one declared factor — the crop side becomes `1.8 ×`
that eye's own axis length, making rendered scale invariant by construction. It drops the
relative-eye-size cue, which head pose already supplies numerically. `crop_side_px` becomes the
per-eye pair `crop_side_px_l` and `crop_side_px_r`, so the manifest column set changes and the
collector now records **schema 5** with 42 columns. Schema 4 stays loadable under v1, so the
recorded session remains readable immutable evidence, and the single-input-contract guard keeps
v1 and v2 rows out of one run. Source frames are not retained, so that session cannot be
re-rendered and is superseded rather than repairable.

Roll normalization could not be validated at all under the five-pose plan, which held head roll to
`1.68°` sd and left the canonical rotation near-identity. Schema 5 therefore adds the `tiltLeft`
and `tiltRight` blocks — seven pose blocks, 189 targets and 1,134 rows, roughly a 40% longer
session. The collector requires at least `6°` of roll change from the participant's own neutral
baseline in the declared direction, and additionally refuses beyond `15°` of absolute roll so the
blocks stay inside the trainer's `20°` roll filter instead of being coached past it.

Reports from schema-5 sessions add a `crop_side_ratio` diagnostic — the per-sample ratio of the
shorter to the longer crop side. It is identically `1.0` for v1 by construction; under v2 it
measures the yaw foreshortening that v1 used to absorb into the far eye's rendered scale.

`--maximum-eye-axis-disagreement` filters on the paired-eye axis disagreement, the reliability
signal for the contour-derived alignment: the shared rotation is a circular mean of both axes, so
a large disagreement means the rotation applied to both crops came from contours that do not agree
with each other. It is a declared factor bound into the checkpoint like the confidence filter, it
binds a common source dataset at the floor when tightened, and it refuses on schemas that never
recorded the axes rather than applying to part of a run.

Its default retains everything. No threshold is defensible from the single measured session, and
tightening is steeply non-linear there: `20°` retains 99.4% of rows and `18°` retains 91.7%, but
`15°` leaves the `lookUp` block with 22 of 162 rows and `12°` or tighter empties it. Read the
per-pose retention, not the aggregate, before declaring a value.

Session metadata binds source dimensions, crop sampling, the physical-lens angular labels and the
Apple Vision revision-3 head-pose convention. Per-sample evidence includes contour and pupil counts
and source, raw eye-axis endpoints, paired rotation, axis disagreement, crop side and geometric
clipped fractions. These make alignment, clipping and occlusion-related evidence inspectable; they
do not claim a validated occlusion score or establish clipping, disagreement or occlusion filters.

Schemas 1–3 retain only their already-resampled crops. They lack source axes and requested crop
geometry, so they cannot be recropped or treated as a paired raw-versus-schema-4 alignment test.
Applying another transform to those PNGs would create a different lossy input contract.

Before training, numeric target pitch must reproduce the physical setup, screen pitch must be
negative and decrease down the display, and vertical head-pose changes must be verified without
treating prompt names as numeric evidence. Apple Vision defines positive pitch as head-down, so the
schema-4 collector requires `lookUp − neutral ≤ -5°` and `lookDown − neutral ≥ 5°`. Eligible
sessions must agree on that numeric convention. Vision likewise defines positive roll as
counterclockwise in the image, and the participant's own left shoulder is on the image's right, so
schema 5 requires `tiltLeft − neutral ≤ -6°` and `tiltRight − neutral ≥ 6°`. Both directions are
fixed declared signs rather than per-session latches.

That gate currently fails on the consumed development evidence. Both eligible sessions contain
adequate opposite vertical motion, but their numeric `lookUp − neutral` signs disagree. A
2026-08-11 read-only provenance investigation classified the disagreement: the recorded numbers
follow the Vision head-down-positive convention in every session — the SDK header, the unchanged
producer code at every historical commit, and the inverted session's convention-consistent
neutral pitch and yaw prompt signs all confirm it — but one session's prompted vertical motion
was genuinely inverted, coached by the retired per-session sign-learning gate. Historical rows
are not flipped or relabelled; that session is simply not eligible pitch-gated development
evidence. With one eligible development session the two-session comparison contract cannot be
satisfied, so no compact-model comparison was launched and no native integration is justified.
New eligible evidence requires explicitly consented collection under the fixed schema-4
collector, which needs approval and physical participation.

The repository-native comparator is a random-initialized `156,226`-parameter shared-eye encoder
with paired features, yaw/pitch/roll input, yaw/pitch-degree output and an angular-cosine objective.
It uses no public data or pretrained weights. It remains an unevaluated development comparator
until the label gate passes; legacy evidence cannot validate schema-4 canonical alignment.

If the label gate is legitimately resolved, the smallest predeclared capacity screen is exactly two
80-epoch seed-7 runs: this compact estimator and the fixed OMZ baseline, using the same legacy eye
crops, session roles and shared procedure fields. Continue the pair at seeds 19 and 43 only if the
compact run passes every gate on every development session and strictly improves the worst-session
score over its seed-7 baseline. The total budget is therefore two runs initially and six runs at
most. This isolates the declared model factor on legacy inputs; it cannot validate schema-4
canonical alignment.

## Privacy and retention

Collection must be explicit. The app records two `60 × 60` eye crops; participant and session
UUIDs; frame and sample identifiers; timing and tracking quality; head pose; camera and display
geometry; and target labels and coordinates under:

```text
~/Library/Application Support/Aspectus/gaze-datasets/
```

The directory and files are owner-only. The Settings flow can reveal or permanently delete the
collection. Eye crops are biometric data: do not commit, upload, share or use another person's data
without explicit consent and a documented purpose, retention, withdrawal and deletion policy.

Reports, checkpoints, Core ML conversions, candidate manifests and final-use history stay
owner-only under ignored `Training/runs/`. The pretrained initializer stays under ignored
`Training/data/`. Do not delete run evidence during an active protocol, and never put a participant
or raw session identifier in a run-directory name.

## Environment and tests

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.12. Install the locked environment and
run the pinned test suite:

```bash
uv sync --project Training --python 3.12
uv run --project Training --python 3.12 --with pytest==8.4.2 \
  python -m pytest Training/tests
```

Run both conversion smoke paths before training changes:

```bash
uv run --project Training python Training/train_gaze.py smoke
uv run --project Training python Training/fetch_pretrained_gaze.py
uv run --project Training python Training/train_gaze.py smoke \
  --pretrained-onnx Training/data/gaze_estimation_adas_0002.onnx
```

## Audited initializer

Intel Open Model Zoo `gaze-estimation-adas-0002` is retained only as a checksum-pinned reference
initializer. Its contract is left and right BGR eye images at `1 × 3 × 60 × 60`, values in
`0…255`, head pose in yaw/pitch/roll order, and a three-component gaze vector. Aspectus records RGB
PNGs; the trainer and exported Core ML wrapper perform the same RGB-to-BGR reorder. Paired-eye
mirror augmentation also swaps left/right inputs and changes yaw/roll signs.

- source model: [Open Model Zoo 2021.2 at `3386309`](https://github.com/openvinotoolkit/open_model_zoo/tree/338630987b403a6981d03ab6d04c2d5ad367793a/models/intel/gaze-estimation-adas-0002)
- converted directory and directory-specific Apache-2.0 licence: [PINTO model zoo at `1daa964`](https://github.com/PINTO0309/PINTO_model_zoo/tree/1daa9648a887e0de44630cc822807ad1fc7c0bb1/091_gaze-estimation-adas-0002)
- downloaded archive SHA-256: `eecddc6655ac1709ab3ce5dfcc9373e2b7f1b58712ef892be78f33e72e5831e1`
- extracted ONNX SHA-256: `f8f13707401547c7c5e146ba22da1bd6ada00f47ebf347be6d97c5acfaf4c2bd`

Open Model Zoo describes an internal 60-person training dataset but does not publish enough
consent, retention, derived-weight or redistribution evidence for Aspectus release due diligence.
The initializer and personalized outputs therefore remain local development artifacts. The code or
directory licence must not be treated as permission to use unrelated datasets or distribute a
general-user model.

## Session contract

Collect from the Release app under **Settings → Correction → Collect model data**. Training and
development sessions must be completed independently without changing participant, camera format,
display geometry, viewing distance or eye-image contract.

The trainer:

- accepts only sessions marked `finished` and requires every role explicitly on the command line
- never creates a row-level random split
- validates the schema's target plan and every eye image: 810 rows over five pose blocks for
  schemas 1-4, and 1,134 rows over seven for schema 5
- validates canonical session contracts and recomputes alignment and clipping evidence from raw axes
- checks the schema-5 roll blocks against the declared Vision tilt direction, so an inverted
  session fails loudly rather than passing as unusable evidence
- reconstructs target coordinates and angles from stored display geometry
- stops before training when numeric target pitch or head-pitch orientation is inconsistent
- keeps training and development sessions disjoint
- applies confidence, eye-openness and head-roll filtering consistently to legacy sessions
- filters on paired-eye axis disagreement only when that factor is declared, and only where the
  alignment evidence exists to honour it
- requires each development session to retain at least 100 samples per pose block, 50 lens samples
  overall, six lens samples per pose, and both repeated lens targets in every pose
- requires horizontal prompt means to separate from neutral by at least 6° in opposite directions
  and vertical prompt means by at least 5° in opposite directions

Stored `training` and `validation` labels are metadata, not authority; explicit command-line roles
win. A session that influenced preprocessing, architecture, tuning, checkpoint selection or
diagnosis is consumed development evidence and cannot be an untouched final session.

Schema 3 begins settling only after tracking and the requested pose remain valid, requires face
confidence of at least `0.70`, rejects absolute roll above `20°`, waits one second after target
changes and spreads samples over a longer interval. The loader also excludes the first lens target
from schema-1 sessions because it may overlap the end of a prompted head turn.

Schema 4 keeps schemas 1–3 loadable under their exact legacy preprocessing and digest shape, but a
single run may not mix legacy and canonical input contracts. New reports and checkpoints bind the
input preprocessing, label-contract digest and schema-specific quality evidence. Ordinary loading
still accepts checkpoint formats 1–3; only format 4 and candidate-manifest format 2 may enter a new
freeze or final evaluation.

## Train and select a checkpoint

Use a new output directory for every run; the trainer refuses to overwrite an existing report or
checkpoint. Training is currently blocked by the numeric pitch gate. The command below documents
the rejected OMZ procedure for local reproducibility; it is not a recommendation to continue tuning
or to launch a run against the current sessions.

```bash
uv run --project Training python Training/train_gaze.py train \
  --data "$HOME/Library/Application Support/Aspectus/gaze-datasets" \
  --output Training/runs/baseline-seed7 \
  --pretrained-onnx Training/data/gaze_estimation_adas_0002.onnx \
  --seed 7 \
  --learning-rate 0.0001 \
  --learning-rate-schedule constant \
  --weight-decay 0.0001 \
  --training-session-id TRAINING_SESSION_ID \
  --validation-session-id DEVELOPMENT_SESSION_ID_A \
  --validation-session-id DEVELOPMENT_SESSION_ID_B
```

Development is evaluated independently per session after every epoch. Selection minimizes:

```text
max over sessions of max(median / 2, p95 / 5, lens_p95 / 3)
```

Exact ties prefer lower worst lens ratio, lower worst overall-p95 ratio, lower worst median ratio,
then the earlier epoch. The selected weights are restored before saving or conversion. Export is
allowed only when every development session passes all three unrounded gates:

- median angular error `≤ 2°`
- overall p95 angular error `≤ 5°`
- physical-lens p95 angular error `≤ 3°`

One-factor experiments may change the declared learning-rate schedule, weight decay, trainable
scope, angular-tail loss, augmentation strength or face-confidence filter. Keep every other
procedure and data role fixed. Filtering comparisons bind the same source dataset at the recorder's
`0.70` confidence floor so a stricter retained subset cannot conceal changed source data.

A candidate must use fixed seeds `7`, `19` and `43`, pass every development session at every seed,
and improve the worst-session score over its paired baseline at every seed. Seed 7 is nominated in
advance; never select the luckiest seed after observing results. Reports persist every evaluated
epoch, per-session metrics, procedure fields, software versions, data fingerprints and artifact
checksums.

## Freeze a candidate

After three current-format baseline runs and their three paired candidate runs exist, publish one
immutable owner-only manifest:

```bash
uv run --project Training python Training/train_gaze.py freeze-candidate \
  --data "$HOME/Library/Application Support/Aspectus/gaze-datasets" \
  --candidate-report Training/runs/candidate-seed7/evaluation.json \
  --candidate-report Training/runs/candidate-seed19/evaluation.json \
  --candidate-report Training/runs/candidate-seed43/evaluation.json \
  --baseline-report Training/runs/baseline-seed7/evaluation.json \
  --baseline-report Training/runs/baseline-seed19/evaluation.json \
  --baseline-report Training/runs/baseline-seed43/evaluation.json \
  --changed-factor augmentation \
  --output Training/runs/candidate-seed7/candidate-manifest.json
```

The command validates all six reports and checkpoints, including their stored model-state hashes,
before capturing the freeze time and atomically publishing the manifest. It rejects missing or
duplicate seeds, a failing candidate, inconsistent improvement, changed data or environment, a
second changed factor, altered artifacts, and a manifest outside the seed-7 candidate directory.

The manifest binds relative artifact references, hashes, procedure and machine contracts, fixed
seed nomination, completed-session fingerprints and a semantic candidate identity. It contains no
raw participant or session ID. Replacing the paired baseline without changing candidate data,
procedure and model states does not create a new candidate.

Wait for the freeze command to return and record its printed checksum before collecting anything
else. Only then may one new untouched final session be recorded. Collection, freezing and final
evaluation must not run concurrently.

## One-shot final evaluation

Evaluate exactly one post-freeze session without retraining:

```bash
uv run --project Training python Training/train_gaze.py evaluate \
  --data "$HOME/Library/Application Support/Aspectus/gaze-datasets" \
  --checkpoint Training/runs/candidate-seed7/GazeEstimator.pt \
  --expected-checkpoint-sha256 FROZEN_CHECKPOINT_SHA256 \
  --expected-candidate-manifest-sha256 FROZEN_CANDIDATE_MANIFEST_SHA256 \
  --output Training/runs/candidate-seed7/final-evaluation.json \
  --pretrained-onnx Training/data/gaze_estimation_adas_0002.onnx \
  --validation-session-id UNTOUCHED_VALIDATION_SESSION_ID
```

The evaluator verifies the manifest, nominated checkpoint, all referenced artifact hashes,
pretrained model, frozen development replay, data fingerprints, filters, gates, hardware/backend
contract, participant/setup match, pose/lens coverage and unchanged model state. The final session
must be the only completion after the manifest freeze.

Before scoring, an owner-only workspace ledger atomically claims both the final-session fingerprint
and semantic candidate identity. A session cannot be tried against another candidate, and a
candidate cannot consume another fresh session after its first attempt, even if paths, timestamps or
baseline comparators change. A failed or interrupted attempt is consumed. The ledger stores hashes
and claim metadata, not imagery or raw IDs. Manifests, reports and ledger claims are published with
macOS full-sync durability; losing or deleting the ledger invalidates the honest final protocol.

A final pass would justify an approval request for native integration and hardware quality/latency
testing. It would not establish general-user accuracy or permission to distribute personalized
weights.
