# Gaze estimator training

This directory is an offline development tool. Python and PyTorch are never part of the Aspectus
app. A successful run exports an ML Program for the native Core ML runtime.

The first target is a personalized proof: two explicit full-screen recording sessions from the
same person and setup. It answers whether eye appearance contains enough vertical and horizontal
signal on the reference camera. It is not a redistributable production model and does not establish
accuracy across people, glasses, cameras or lighting.

## Collect

Start the Aspectus Release app, open Settings → Correction → Collect model data, then complete:

1. one Training session
2. one Validation session without changing the camera, display or viewing distance

The app records only two 60 × 60 eye crops, head pose and target labels. It stores them in
`~/Library/Application Support/Aspectus/gaze-datasets` with owner-only permissions. The settings
flow can reveal or permanently delete the collection.

These eye crops are biometric data. Do not commit them, upload them or use another person's data
without explicit consent and a documented retention policy.

## Train and convert

Install the pinned development environment:

```bash
uv sync --project Training --python 3.12
```

Run the conversion smoke test first:

```bash
uv run --project Training python Training/train_gaze.py smoke
```

The stronger development path starts from Intel Open Model Zoo's
`gaze-estimation-adas-0002`. Open Model Zoo and PINTO's converted artifact are Apache-2.0; the
fetcher accepts only the audited archive and model checksums. The weight stays under ignored
`Training/data/` and is never bundled automatically.

```bash
uv run --project Training python Training/fetch_pretrained_gaze.py
uv run --project Training python Training/train_gaze.py smoke \
  --pretrained-onnx Training/data/gaze_estimation_adas_0002.onnx
```

- source model: <https://github.com/openvinotoolkit/open_model_zoo/tree/2021.2/models/intel/gaze-estimation-adas-0002>
- converted artifact and licence: <https://github.com/PINTO0309/PINTO_model_zoo/tree/main/091_gaze-estimation-adas-0002>
- ONNX SHA-256: `f8f13707401547c7c5e146ba22da1bd6ada00f47ebf347be6d97c5acfaf4c2bd`

Train against explicitly selected complete sessions:

```bash
uv run --project Training python Training/train_gaze.py train \
  --data "$HOME/Library/Application Support/Aspectus/gaze-datasets" \
  --output Training/runs/personal-v1 \
  --pretrained-onnx Training/data/gaze_estimation_adas_0002.onnx \
  --training-session-id TRAINING_SESSION_ID \
  --validation-session-id UNTOUCHED_VALIDATION_SESSION_ID
```

The trainer never creates a row-level random split. It reads only sessions marked `finished`,
checks that every manifest row and eye image is present, trains on `training` sessions and evaluates
only `validation` sessions. It always saves a checkpoint and metrics. It exports
`GazeEstimator.mlpackage` only when all default held-out gates pass:

- median angular error ≤ 2°
- p95 angular error ≤ 5°
- physical-lens p95 angular error ≤ 3°

Those gates are deliberately stricter than the correction envelope. Passing them permits native
integration and hardware latency/quality testing; it does not permit distributing a personalized
weight file.

Schema-2 sessions start every settle interval only after tracking and the requested head pose are
continuously valid. The loader excludes the first lens target from older schema-1 sessions because
those samples could begin while a prompted head turn was still ending. No screen target or valid
schema-2 lens target is discarded.

After a validation session has influenced architecture or tuning, it is development data rather
than an honest final test. It can be reassigned explicitly with `--training-session-id`; a newly
recorded untouched session must then be selected with `--validation-session-id` for the final gate.
