# Schema-5 screening outcome and section 8 rule 7 recommendation

Frozen baseline and screening budget completed 2026-08-12 under
`docs/schema5-experiment-protocol.md`. No candidate passed both development sessions.

## Result

| run | factor | score | s5 median/p95/lens | s6 median/p95/lens | retained |
|---|---|---|---|---|---|
| baseline | — | 1.508 | 3.02 / 7.26 / 3.80 | 1.77 / 5.12 / 2.19 | — |
| screen 1 | learning-rate schedule → cosine | 1.806 | 3.55 / 9.03 / 3.78 | 2.61 / 5.63 / 4.76 | no |
| screen 2 | augmentation strength → strong | 1.466 | 2.93 / 6.57 / 4.19 | 2.53 / 6.01 / 3.89 | no |
| screen 3 | weight decay → 1e-3 | 1.509 | 3.02 / 6.51 / 4.52 | 2.17 / 5.10 / 2.38 | no |

Gates are median ≤ 2°, p95 ≤ 5°, lens p95 ≤ 3°, each development session independently.
Retention required ≤ 1.4330 and no worsening of either session. No screen met either clause.

## Budget disposition

Three of the six declared factors were screened. The other three were not spent, because no
protocol-legal experiment exists for them on this dataset:

- **eye-axis disagreement threshold** — no viable operating point. At ≤ 24° the `lookDown` block
  falls to 17–85 rows against the 100-row coverage floor in section 5, in every session. At ≥ 28°
  the filter is inert. The only compliant tightening, 26°, removes 21 of 6,804 rows (0.3%). The
  protocol anticipated the mechanism — disagreement correlates with head pitch — but on schema-4 it
  removed `lookUp`; here the camera sits below eye level, so `lookDown` reaches the highest absolute
  pitch and is removed instead.
- **loss formulation** — `tail-angular` is rejected by the trainer for any model without pretrained
  weights (`train_gaze.py:3330`). The frozen baseline is randomly initialised, so the factor is
  incompatible with the declared architecture.
- **crop-quality rejection** — not implemented: no trainer flag and no code path.

## Diagnosis

The failure is not model capacity or regularisation strength. Three mechanically different
regularisers produced the same trade: session 5 improved on p95 in all three while session 6
degraded in all three. Lens p95 worsened under every screen, and the lens target is the only label
that is exactly correct by construction.

Measured cause: **the labels are mutually inconsistent.** Screen-target labels are computed as
`atan2(offset_mm, viewingDistanceMM)` with a hand-entered 550 mm, identical in all six sessions.
Eye-crop scale is inversely proportional to distance and shows the sessions were not at one
distance. Anchoring session 6 at its declared 550 mm:

| # | split | implied distance | label error at the corner target |
|---|---|---|---|
| 1 | training | 549 mm | 0.03° |
| 2 | training | 474 mm | **3.47°** |
| 3 | training | 574 mm | 0.90° |
| 4 | training | 551 mm | 0.04° |
| 5 | validation | 524 mm | 1.10° |
| 6 | validation | 550 mm | 0.00° |

Training session 2 carries a label error larger than the 2° median gate. Sessions 2 and 3 differ by
21% in distance, so identical screen positions carry labels differing by roughly 4.4° between them.
No hyperparameter can resolve a contradiction between training labels.

The absolute anchor is assumed; the relative spread is measured and is what creates the
contradiction. Correcting development labels alone was tested against the baseline checkpoint and
moved session 5 from 3.02 to 2.96 — as expected, since that model had been trained on uncorrected
labels.

## Rule 7 recommendation: data redesign

Recover viewing distance per session rather than declaring one constant, and recompute every screen
label. This is a data redesign under rule 7 and requires no new recording, which rule 7 forbids.

1. **Anchor the scale physically.** Eye-crop scale gives exact distance *ratios* but not absolute
   distance, because macOS reports no camera field of view. One physical measurement of
   eye-to-lens distance, taken with a simultaneous crop-scale reading, fixes the constant
   `k = crop_side_px × distance_mm` for this camera and participant. Every session then follows as
   `d_i = k / crop_side_px_i`.
2. **Recompute labels** with per-session `d_i` under the existing formula. The label contract
   changes, so this requires a protocol amendment and re-freeze before any metric is computed.
3. **Record distance per sample, not per session.** Crop scale is already recorded per row, so
   within-session drift is observable and need not be assumed constant.

A global scale error that survives step 1 would cancel in the gate — labels and predictions would be
wrong together — while leaving the deployed correction wrong. The physical anchor is therefore
required for correctness even though the gate cannot detect its absence.

## Not recommended

Further tuning of this model family, per rule 7. The three unspent factors are unavailable rather
than untried, so the budget cannot be continued by substitution.
