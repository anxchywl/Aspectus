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

Then train against complete held-out sessions:

```bash
uv run --project Training python Training/train_gaze.py train \
  --data "$HOME/Library/Application Support/Aspectus/gaze-datasets" \
  --output Training/runs/personal-v1
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

