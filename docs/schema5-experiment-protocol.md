# Schema-5 collection and experiment protocol

Frozen before the first schema-5 recording. This document is tracked so that the commit adding it
is the freeze: any later change must be a separate commit stating what changed and why. A
predeclaration that can be edited without history is not a predeclaration.

It contains no session identifier, participant identifier or private path.

## 1. Why this replaces the phase-3 protocol

The phase-3 experiment protocol describes a configuration that no longer exists: the five-pose
collection plan, the Open Model Zoo estimator, legacy crops, and two named consumed-development
sessions. Three of those are now retired.

- the Open Model Zoo full-fine-tuning direction was rejected by its own bounded budget, which found
  no seed-robust procedure
- one of the two eligible development sessions is a proven physically inverted session and is not
  eligible pitch-gated evidence, leaving the frozen two-session development contract unsatisfiable
- the crop contract that produced every schema-4 sample couples rendered eye scale to head yaw, so
  the one recorded schema-4 session is superseded and cannot be re-rendered

The predeclared legacy compact screen in DESIGN §8 is therefore void rather than pending: it
requires preserving fixed session roles that include a session now excluded from pitch-gated
evidence, so no substitution can satisfy it without changing the thing it fixed.

## 2. Objective and evidence boundary

Determine whether a randomly initialized compact paired-eye estimator on `canonical-paired-eye-v2`
crops satisfies the personalized gate across session-isolated development sessions, under
deterministic checkpoint selection and with no development rows entering training.

This protocol can establish consumed-development evidence only. Final status additionally requires
the checksum-bound three-seed candidate manifest, the workspace-global final-use ledger, and one
new untouched session whose creation timestamp is strictly later than the manifest freeze. Those
requirements are unchanged and are not restated here.

No schema-1 to schema-4 session may enter any run under this protocol. Their input preprocessing
contract differs, the trainer refuses to mix contracts, and legacy crops cannot validate canonical
alignment. Phase-3 screening outcomes do not transfer: they were measured on a different
architecture and a different input contract, so they are neither evidence for nor against the same
factor here.

## 3. Collection plan

Fixed before the first recording.

| Property | Value |
|---|---|
| schema | 5 |
| crop contract | `canonical-paired-eye-v2`, per-eye side `1.8 ×` that eye's own axis |
| pose blocks | 7: neutral, turnLeft, turnRight, lookUp, lookDown, tiltLeft, tiltRight |
| targets per block | 27, being 25 screen positions bracketed by two physical-lens targets |
| targets per session | 189 |
| samples per target | 6 |
| rows per session | 1,134 |
| participant | one, the same person for every session |
| hardware | one machine, one camera format, one display geometry |
| valid sessions required | 6 |

Sessions must span at least two separate sittings rather than one continuous run. Recording all six
back to back would make the development sessions a near-copy of the training conditions and would
test almost nothing about transfer.

Collection is explicitly consented and requires physical participation. Eye crops are biometric
data and stay in the participant's own storage.

### Prompt direction contract

The collector enforces the Vision convention rather than learning a sign per session. Head-down is
positive pitch and counterclockwise is positive roll, both stated by the installed SDK header.

| Block | Required change from that session's neutral baseline |
|---|---|
| lookUp | `≤ -5°` pitch |
| lookDown | `≥ +5°` pitch |
| tiltLeft | `≤ -6°` roll, and `\|roll\| ≤ 15°` |
| tiltRight | `≥ +6°` roll, and `\|roll\| ≤ 15°` |

The tilt direction was confirmed on hardware before this protocol was frozen: tipping the crown
toward the participant's own left shoulder moved recorded roll from `+1°` to `-22°`.

The `15°` tilt bound is deliberate and interacts with the `20°` absolute-roll quality filter below:
it keeps every accepted tilt sample inside that filter instead of coaching the participant past it.
The same hardware check measured a `23°` roll change against a `6°` request, so overshoot is the
expected behaviour rather than a hypothetical one.

## 4. Session role assignment

Declared here, before any session exists, and by recording position rather than by outcome.

1. Valid sessions are ordered by their recorded creation timestamp.
2. The first four fill the training slots and are recorded with the collector's training split.
   The fifth and sixth fill the development slots and are recorded with the validation split.
3. No role may be reassigned after any model metric has been computed on any schema-5 session.

Rule 3 is the point of this section. A role chosen after seeing which sessions score well is
outcome-driven relabelling, and it is what made an otherwise eligible legacy session unusable as a
substitute earlier in this work.

Rule 2 is fixed at recording time rather than afterwards because the split is not only a label: the
collector presents the screen grid in reverse order for the validation split. Recording the
development sessions under that split means the development gate is not scored on the same target
presentation order the model was trained against, which removes an order confound that a
post-hoc relabelling could not.

Because the ordering rule places the development sessions last, they will generally come from a
later sitting than the training sessions. That is intended: it makes the development gate a test of
transfer to a new sitting rather than a test of memorization within one.

## 5. Session validity and exclusion

A session is valid only if it is marked finished and passes every offline check below, all of which
are computed without any model:

- 1,134 rows, complete 189-target sequence, six samples per target, strictly increasing frame ids
  and elapsed time
- schema-5 manifest columns exactly, no missing or surplus fields
- crop, label and head-pose contracts matching the frozen declarations
- per-eye alignment, crop side and clipping evidence recomputed from the raw axis endpoints
- numeric target pitch reproducing the physical setup, and screen pitch negative and decreasing
  down the display
- prompted pose separation in the declared direction for all four directional axes, including the
  two roll blocks
- pose coverage retained after filtering: at least 100 samples per block, and the declared lens
  coverage per block and per session

An invalid session is excluded and the next recorded session takes the vacant slot, recorded under
the same split as the slot it fills. Every exclusion records which check failed and is made before
any model run. Exclusion is never based on model error; a session that validates cannot be dropped
for scoring badly.

## 6. Frozen baseline configuration

| Setting | Value |
|---|---|
| architecture | `shared-eye-spatial-cnn-head-pose-v2`, 156,226 parameters |
| initialization | random; no pretrained weights and no third-party data |
| epochs | 80 |
| batch size | 64 |
| optimizer | AdamW |
| learning rate | 0.001, the native default, constant |
| weight decay | 0.0001 |
| seed | 7 for the baseline and every screen |
| augmentation | physically paired eye swap with horizontal geometry inversion, baseline strength |
| minimum face confidence | 0.70 |
| minimum eye openness | 0.40 |
| maximum absolute head roll | 20° |
| maximum eye-axis disagreement | inert, retaining every row |
| evaluation | every epoch, separately by session |
| saved model | restore the selected epoch before checksum, conversion or report generation |

The learning rate differs from the phase-3 baseline because that value was chosen for fine-tuning a
pretrained network. This is a randomly initialized model and takes the native default.

The disagreement filter starts inert deliberately. No threshold is defensible from the single
schema-4 session that has been measured, and adopting one from those numbers would be tuning on an
observed outcome. It is a declared factor under section 8, not part of the baseline.

### Development-pair supersession, 2026-08-12

The first recorded development pair was set aside before any model run and before any metric was
computed on any schema-5 session. Both sessions passed every section-5 validity check; they are not
excluded for failing one.

Training slot 4 was recorded at `08:22`, and the two development sessions at `08:34` and `08:52` —
one continuous sitting. That satisfies the two-sitting minimum in section 2, because the six
sessions spanned four sittings overall, but it defeats the purpose section 4 states for the
ordering rule: a development gate testing transfer to a new sitting rather than memorization within
one. The pair shared its conditions with the training session recorded twelve minutes earlier.

Superseded, not deleted. The two directories were moved out of the dataset root to
`gaze-datasets-superseded/` so that neither the collector's slot accounting nor the trainer's
recursive session discovery sees them, while the recorded data survives for audit:

| set aside | split | rows | recorded |
|---|---|---|---|
| `48FD7FEA` | validation | 1,134 | 2026-08-12T08:34:32Z |
| `45C405EB` | validation | 1,134 | 2026-08-12T08:52:04Z |

The vacant development slots are filled by the next two recorded validation sessions, which must
come from a sitting later than training slot 4. Because no metric had been computed, this decision
could not be informed by any session's score, which is the condition rule 3 of section 4 protects.

### Collector reconciliation, 2026-08-11

This table is the complete set of quality filters. The collector additionally enforced an
undeclared absolute bound of `25°` on yaw and pitch, carried over from the pre-protocol collection
commit and never reconciled with this table when the protocol was frozen. It has been removed.

The bound was not a spare margin. `lookDown` requires `≥ +5°` of pitch beyond the session's own
neutral baseline, so any baseline above `20°` left an empty accepting window and the block could
never complete. A recorded training session measured a `+21.2°` neutral baseline — a camera below
eye height — completed `neutral`, `turnLeft`, `turnRight` and `lookUp` at 162 rows each, and
accepted zero `lookDown` samples before being stopped. The undeclared bound made this protocol
unexecutable at that seating while silently withholding rows the protocol asks to be recorded.

Removing it changes no declared threshold and no recorded value. Absolute roll stays bounded at
`20°` here and the tilt gate at `15°`, both declared above and reasoned about in section 3. No
schema-5 session had been completed when this was found, so no recorded session is affected.

## 7. Gates and selection

The analysis grain is a retained paired-eye sample. Splits, checkpoint selection and gates remain
session-level. Rows are never randomly split.

For each development session:

- median ratio = median angular error / 2
- overall-p95 ratio = p95 angular error / 5
- lens ratio = lens p95 angular error / 3

The selection score is the maximum ratio over all gates and all development sessions. Lower is
better. Ties break on, in order: lower worst lens ratio, lower worst overall-p95 ratio, lower worst
median ratio, then earlier epoch.

Passing requires every development session independently to satisfy median at most 2°, overall p95
at most 5°, and lens p95 at most 3°. Metrics are never rounded, pooled or redefined for the
decision.

## 8. Bounded budget and stopping rule

1. Run the frozen baseline above.
2. If it does not pass both development sessions, screen at most six seed-7 single-factor changes,
   one setting per factor and no grid search. The candidate factors are: eye-axis disagreement
   threshold, augmentation strength, crop-quality rejection, learning-rate scheduling, weight decay,
   and loss formulation.
3. Stop screening immediately if a candidate passes all gates on both development sessions.
4. Retain a factor only if its worst-session score improves by at least 5% relative to the frozen
   baseline and neither development session worsens on its own score.
5. Test at most one combination, and only if two isolated factors independently meet the retention
   rule.
6. Compare the retained winner against the baseline at seeds 7, 19 and 43. Require every candidate
   seed to pass every gate and to improve on its paired baseline. Freeze the seed-7 artifact only
   after that check.
7. If the budget yields no all-session pass, stop tuning this model family and recommend an input,
   data or model redesign. Do not request a new recording.

Screening may not resume after a candidate passes all gates and then fails its seed check. Phase 3
left that case undefined and resumed once; the resulting evidence was exploratory rather than
preregistered, and this protocol closes the gap by forbidding it.

Any tightened filter binds a common source dataset at the floor, so only the retained subset may
differ as a consequence of the declared factor.

The disagreement threshold is the factor most likely to be screened first, and it needs care. On
the measured schema-4 session, `20°` retained 99.4% of rows and `18°` retained 91.7%, but `15°` left
the `lookUp` block with 22 of 162 rows and `12°` emptied it. Read per-pose retention before the
aggregate: an aggregate that looks healthy can hide an annihilated pose block, and disagreement
correlates with head pitch, so it removes the vertical blocks preferentially.

## 9. Required reports

Every run records all evaluated epochs, session-level metrics and selection ratios, selected epoch,
checkpoint and model checksums, seed, optimizer and learning-rate configuration, session roles and
content digests, filtering and augmentation configuration, trainer checksum, and software versions.

The selected model is additionally reported by session, pose block, target location, lens versus
screen target, target yaw and pitch, numeric head yaw/pitch/roll buckets, face confidence, eye
openness, crop clipped fraction, per-eye crop side, crop side ratio, and paired-eye axis
disagreement.

Named vertical and roll prompt directions may be reported as provenance. Stratified numeric reports
use head-pose buckets, never prompt names.

## 10. What this protocol cannot establish

It cannot establish a final personalized result, a general-user result, native integration
readiness, or redistribution readiness. It cannot validate any claim about a second participant,
another machine, other lighting, or glasses beyond the single participant's usual conditions.

A pass here justifies exactly one thing: proceeding to the frozen manifest and ledger path with a
candidate that has already passed every gate at all three seeds.

## 11. Amendment 1: measured viewing distance, 2026-08-12

Adopted after the section 8 budget closed without a pass, and recorded before any metric is
computed against a relabelled row. Evidence: `docs/schema5-screening-outcome.md`.

### Why the label contract has to change

Screen labels are `atan2(offset_mm, viewingDistanceMM)`. Every schema-5 session declared
`viewingDistanceMM = 550`, a value typed into Settings rather than measured, because macOS reports
no camera field of view. Canonical crop side is a fixed multiple of the eye's own axis length and
therefore scales as `1/distance`; measured per session it spans roughly `474–574 mm`. Two sessions
labelled the same screen position about `4.4°` apart, and one training session carries `3.47°` of
label error at the corner target against a `2°` median gate.

That is not a tuning problem. The training set states contradictory answers for the same input, so
no setting of any declared factor can satisfy it. The three factors that were screened could only
move error between development sessions, and every one of them degraded the physical-lens target —
the only label correct by construction, since looking at the lens is `(0,0)` at any distance.

### What changes

1. `viewingDistanceMM` stops being a declared constant and becomes a measured per-sample quantity.
2. `labelContract.version` goes to `2`, adding `distanceSource: "measured-crop-scale"`. Version 1
   rows and version 2 rows may never be mixed in one run.
3. Screen labels are recomputed as `atan2(offset_mm, d_i)` where `d_i = k / cropSidePixels_i`.
4. Lens labels are unchanged and stay exactly `(0,0)`. They are distance-invariant, which is what
   makes them the reference that exposed this.

### Fixing the scale constant `k`

Crop side yields exact distance *ratios* and no absolute distance. `k = cropSidePixels × distance_mm`
is fixed by one physical measurement: eye-to-lens distance taken with a rule while the diagnostics
HUD `crop side` row is read at the same moment. `k` is a property of this camera and this
participant's eye geometry, so a single measurement covers every recorded session.

The measurement, its date, the instrument, and the simultaneous crop-side reading are recorded here
before relabelling. A `k` inferred by fitting model error instead of measuring is forbidden: it
would select the distance that flatters the model, which is the outcome-driven fitting rule 3 of
section 4 exists to prevent.

Absolute scale cannot be checked by the gate. Labels and predictions that are wrong by one common
factor agree with each other, so a purely relative normalisation would pass while leaving the
deployed correction systematically wrong. The physical anchor is therefore required for
correctness, not for the gate.

### What is not permitted under this amendment

- No new recording. Rule 7 of section 8 forbids it, and relabelling needs none.
- No change of session roles. The four training and two development sessions stay as recorded.
- No reuse of the screening budget. It is closed; the three unspent factors were unavailable rather
  than untried, and a data redesign does not reopen them.
- No metric computed on relabelled rows until `k` is measured and recorded above.

### Re-freeze

Once `k` is recorded, the relabelled dataset is re-frozen and section 6 is rerun from the baseline.
Everything else in this protocol is unchanged: same architecture, seed, gates, selection rule, role
assignment, and validity checks. Section 5 revalidation must be repeated against the relabelled
rows, since label values feed the pose-separation and coverage checks.
