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
integration. The recommended next step is an approval-gated input/model redesign: record clipping
and occlusion quality, add canonical paired-eye alignment, verify numeric pitch labels, and test an
independently implemented compact paired-eye/head-pose model using only explicitly cleared
first-party data. See [docs/DESIGN.md](../docs/DESIGN.md).

Preserve the local v19–v31 artifacts as consumed-development evidence, but never freeze or evaluate
them as a final candidate: they predate checkpoint format 3. A future candidate must be retrained
under the current provenance and lens-coverage contract.

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
- validates the 810-row target plan and every eye image
- reconstructs target coordinates and angles from stored display geometry
- keeps training and development sessions disjoint
- applies confidence, eye-openness and head-roll filtering consistently to legacy sessions
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

## Train and select a checkpoint

Use a new output directory for every run; the trainer refuses to overwrite an existing report or
checkpoint.

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
