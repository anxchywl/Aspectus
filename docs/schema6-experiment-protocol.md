# Schema-6 collection and experiment protocol

It contains no session identifier, participant identifier or private path.

## 1. Why this replaces schema 5

Schema 5 closed under rule 7 of its own budget: the frozen baseline and every usable screening
factor failed, and the cause was measured rather than guessed. Its full record is
`docs/schema5-experiment-protocol.md` and `docs/schema5-screening-outcome.md`.

The binding defect was label geometry, not model capacity:

- screen labels are `atan2(offset_mm, viewingDistanceMM)`, and viewing distance was a persisted
  Settings preference typed once and reused unchanged for all six sessions
- the sessions were not recorded at one distance; one training session carries about `3.5°` of
  corner-label error against a `2°` median gate, and two sessions label the same screen position
  about `4.4°` apart
- the physical-lens target, whose label is `(0,0)` at any distance, was estimated at `0.65–0.70°` in
  both development sessions while screen targets sat at `1.9–3.2°`

The model read eyes well and mapped them to angles badly, because the angles it was trained on
disagreed with each other. Three regularisers — schedule shape, augmentation strength, weight decay
— could only move error between development sessions, and every one of them degraded the lens
target. That is the signature of contradictory labels rather than insufficient regularisation: when
labels conflict, reducing training loss requires memorising per-session idiosyncrasy, which is what
was observed.

Schema-5 recordings cannot be repaired. Viewing distance is not recoverable from the image at the
precision labels require; `docs/schema5-experiment-protocol.md` §11 records that attempt and why it
failed. No schema-1 to schema-5 session may enter any run under this protocol.

## 2. Objective and evidence boundary

Unchanged from schema 5 §2 in substance: determine whether a randomly initialised compact
paired-eye estimator on `canonical-paired-eye-v2` crops satisfies the personalized gate across
session-isolated development sessions, under deterministic checkpoint selection and with no
development rows entering training.

This protocol can establish consumed-development evidence only. Final status additionally requires
the checksum-bound three-seed candidate manifest, the workspace-global final-use ledger, and one new
untouched session recorded strictly after the manifest freeze.

## 3. Collection plan

| Property | Value |
|---|---|
| schema | 6 |
| crop contract | `canonical-paired-eye-v2`, per-eye side `1.8 ×` that eye's own axis |
| pose blocks | 7: neutral, turnLeft, turnRight, lookUp, lookDown, tiltLeft, tiltRight |
| targets per block | 27, being 25 screen positions bracketed by two physical-lens targets |
| targets per session | 189 |
| samples per target | 6 |
| rows per session | 1,134 |
| participant | one, the same person for every session |
| hardware | one machine, one camera format, one display geometry |
| valid sessions required | 6: four training, two development |

Everything in this table is carried forward unchanged. Schema 5 demonstrated the collection itself
is sound: all six sessions retained 1,134 of 1,134 rows through every declared filter, with zero
crop clipping and full lens coverage in every block.

Session count is deliberately not increased. Schema 5 showed an overfitting signature, but that is
expected when labels contradict each other, so it is not yet evidence that four training sessions
are too few. Increasing the count now would confound the label fix with a data-volume change and
waste the first clean measurement of either.

Sessions must span at least two separate sittings, and the two development sessions must come from
a sitting later than every training session. Schema 5 stated the first requirement and only implied
the second; one development pair had to be superseded for being recorded twelve minutes after a
training session. This protocol makes it explicit and checkable.

### Viewing distance is measured, not declared

This is the change that motivates schema 6.

1. Before each session the participant measures eye-to-lens distance with a rule or tape and enters
   it. The collector must prompt for it per session and must not silently inherit a stored
   preference. A value carried over from a previous session is the schema-5 defect.
2. The measurement is repeated after the session. Both values are recorded. If they differ by more
   than `15 mm` the session is invalid: the participant moved enough during recording that no single
   distance describes it.
3. Both measurements, the instrument, and the canonical crop-side reading shown in the diagnostics
   HUD at the moment of each measurement are recorded in session metadata.

The crop-side readings are recorded as a cross-check, never as a label source. Crop side tracks head
pose as strongly as distance — within-session correlation with pitch is `r = 0.28` to `0.68` at
about `0.4 px/deg`, and with `|yaw|` reaches `r = -0.69` — so it cannot be inverted to a distance.
Compared at matched pose across sessions it is still a valid *consistency* check, and §5 uses it as
one.

### Seating requirement

Resting head pitch at the setup check must be `≤ 15°` absolute. Schema 5 sessions ran from `+10.6°`
to `+18.9°` because the camera sat below eye height, and that had three measured consequences: it
pushed `lookDown` to the highest absolute pitch of any block, it drove the eye-axis disagreement
that made one declared screening factor unusable, and it is the condition under which an undeclared
`25°` pitch cap made `lookDown` physically unreachable. Raising the camera toward eye height also
moves the recording closer to the geometry the correction runs in.

### Prompt direction contract

Carried forward from schema 5 §3 unchanged, including the fixed Vision convention (head-down is
positive pitch, counterclockwise is positive roll), the per-block change requirements, and the `15°`
tilt bound that keeps accepted tilt samples inside the `20°` absolute-roll filter.

## 4. Session role assignment

Carried forward from schema 5 §4 unchanged: ordered by recorded creation timestamp, first four
training and last two development, no reassignment after any model metric has been computed on any
schema-6 session, and the split fixed at recording time because the collector presents the screen
grid in reverse order for the validation split.

Additionally, and unlike schema 5, the sitting separation in §3 is a validity condition rather than
an expectation. A development session recorded in the same sitting as any training session is
invalid and its slot passes to the next recorded session.

## 5. Session validity and exclusion

A session is valid only if it is marked finished and passes every check below, all computed without
any model:

- 1,134 rows, complete 189-target sequence, six samples per target, strictly increasing frame ids
  and elapsed time
- schema-6 manifest columns exactly, no missing or surplus fields
- crop, label and head-pose contracts matching the frozen declarations
- per-eye alignment, crop side and clipping evidence recomputed from the raw axis endpoints
- numeric target pitch reproducing the physical setup, and screen pitch negative and decreasing down
  the display
- prompted pose separation in the declared direction for all four directional axes, including the
  two roll blocks
- pose coverage retained after filtering: at least 100 samples per block, and the declared lens
  coverage per block and per session
- **pre- and post-session distance measurements both present and within `15 mm` of each other**
- **resting head pitch at setup `≤ 15°` absolute**
- **development sessions recorded in a later sitting than every training session**
- **neutral-block median crop side consistent with the measured distance across sessions**: for any
  two sessions, the ratio of neutral-block median crop sides must agree with the inverse ratio of
  their measured distances to within `10%`. Compared at matched pose this is a real check; it will
  not detect a small error, but it will catch a mistyped or stale distance, which is the failure
  this protocol exists to prevent.

An invalid session is excluded and the next recorded session takes the vacant slot under the same
split. Every exclusion records which check failed and is made before any model run. Exclusion is
never based on model error.

### How the added checks are computed

The four checks added by this schema need definitions a reader can reproduce and the trainer can
compute, so they are declared here rather than left to judgement:

- **the labels come from the measurement.** A session's recorded display geometry must carry the
  opening measurement as its viewing distance, exactly. This is the check that would have caught
  schema 5 at load time: it fails whenever screen labels were computed from any distance other
  than the one measured for that session.
- **resting pitch.** The setup reading itself is not recorded, so it is recomputed as the median
  head pitch of the neutral block. That is the same posture measured across 162 samples rather
  than at one instant, and unlike a setup reading it cannot be satisfied once and then abandoned.
- **the sitting boundary.** Sixty minutes between one session's completion and the next session's
  creation. Schema 5's superseded development pair was recorded twelve minutes apart; an hour is
  long enough that seating has to be re-established, which is the generalisation the development
  split exists to test.
- **crop side against distance.** Crop side scales as the inverse of distance, so the compared
  quantity is the product of a session's neutral-block median crop side and its measured distance.
  For any two sessions the ratio of those products must lie within `10%` of one.

One consequence is declared here because it changes an existing invariant. Every run requires a
single recording setup, and that setup hashed the viewing distance. Under schema 6 the distance is
a per-session measurement by design, so it is removed from the setup binding and replaced by the
declaration `measured-per-session`; the display geometry, camera format and every contract must
still match across sessions, and the distance itself is checked per session and between sessions
by the rules above. Without this change every schema-6 run would be rejected for exactly the
property the schema exists to introduce.

## 6. Frozen baseline configuration

Carried forward from schema 5 §6 unchanged, so that the schema-5 result remains a usable comparison
for everything except the labels:

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

There is no absolute yaw or pitch bound. Schema 5 carried an undeclared `25°` cap that made
`lookDown` unsatisfiable for any neutral baseline above `20°`; it is not reinstated, and its absence
is declared here rather than left implicit.

## 7. Gates and selection

Carried forward from schema 5 §7 unchanged. Per development session: median ratio `= median / 2`,
overall-p95 ratio `= p95 / 5`, lens ratio `= lens p95 / 3`. Selection score is the maximum ratio
over all gates and all development sessions; lower is better. Ties break on lower worst lens ratio,
then worst overall-p95, then worst median, then earlier epoch. Export requires every development
session to pass all three unrounded gates independently.

Selection stability must be reported alongside the score: the minimum, maximum and spread of the
selection score over the final twenty epochs. Schema 5's baseline selected `1.508` from a run whose
last twenty epochs spanned `0.558`, so the headline figure was a favourable fluctuation rather than
the level. A score is not comparable to another score without its spread.

## 8. Bounded budget and stopping rule

1. Run the frozen baseline above.
2. If it does not pass both development sessions, screen at most six seed-7 single-factor changes,
   one setting per factor and no grid search. The candidate factors are: **learning-rate schedule,
   weight decay, augmentation strength, minimum face confidence, epoch count, and eye-axis
   disagreement threshold**.
3. A factor is spendable only if it is implemented in the trainer and has an operating point that
   satisfies §5 on the recorded data. A factor that cannot be exercised is recorded as unavailable
   and does not consume budget.
4. Stop screening immediately if a candidate passes all gates on both development sessions.
5. Retain a factor only if its worst-session score improves by at least 5% relative to the frozen
   baseline and neither development session worsens on its own score.
6. Test at most one combination, and only if two isolated factors independently meet the retention
   rule.
7. Compare the retained winner against the baseline at seeds 7, 19 and 43. Require every candidate
   seed to pass every gate and to improve on its paired baseline. Freeze the seed-7 artifact only
   after that check.
8. If the budget yields no all-session pass, stop tuning this model family and recommend an input,
   data or model redesign. Do not request a new recording.

Screening may not resume after a candidate passes all gates and then fails its seed check.

The factor list differs from schema 5, which declared three factors that could not be spent:
`tail-angular` loss is rejected by the trainer for any model without pretrained weights, so it is
not applicable to a randomly initialised baseline; crop-quality rejection has no implementation; and
eye-axis disagreement had no operating point that both tightened anything and kept `lookDown` above
the 100-row coverage floor. The disagreement threshold is retained here because the `≤ 15°` seating
requirement is expected to reduce the pitch-correlated disagreement that made it unusable, but rule
3 applies: if no compliant operating point exists on the recorded data, it is unavailable rather
than failed.

## 9. Required collector changes

These must land, with tests, before the first schema-6 recording:

1. Prompt for the pre-session distance measurement at the start of collection. It must not default
   to a stored preference, and collection must not start without it.
2. Prompt for the post-session measurement when the session completes, and mark the session invalid
   if the two differ by more than `15 mm`.
3. Record both measurements, the instrument, and the crop-side reading at each measurement in
   session metadata, and raise the label contract to a version that declares
   `distanceSource: "measured-per-session"`.
4. Enforce the `≤ 15°` resting-pitch requirement in the existing setup seating check, refusing to
   start rather than warning.

The diagnostics HUD already shows canonical crop side live, which is what makes the cross-check in
§5 possible.

### Required trainer changes

These must also land before the first recording, because a session that cannot be read is not
evidence:

1. Accept schema 6 with schema 5's crop contract, manifest columns and pose plan, and reject a run
   that mixes label-contract versions. Schema-1-to-5 sessions are excluded by that rule alone:
   their labels declare no distance source, so they cannot share a run with measured labels.
2. Refuse a session whose measurements are absent, implausible, unattributed to an instrument, in
   disagreement past `15 mm`, or not the distance its labels were computed from.
3. Recompute the seating, sitting and crop-side checks in §5 from the recorded rows, and record
   both measurements with their crop-side readings in every report.
4. Report the selection-score stability required by §7.

## 10. Required reports

Carried forward from schema 5 §9, with the addition of both distance measurements and their
crop-side readings per session, and the selection-score spread required by §7.

## 11. What this protocol cannot establish

It cannot establish a final personalized result, a general-user result, native integration
readiness, or redistribution readiness. It cannot validate any claim about a second participant,
another machine, other lighting, or glasses beyond the single participant's usual conditions.

It cannot establish that measured viewing distance was the only defect in schema 5. It removes the
one defect that was measured and shown to be large enough to matter. If schema 6 fails with correct
labels, the overfitting signature seen in schema 5 becomes interpretable on its own for the first
time, and a data-volume or architecture change becomes the next honest question rather than a guess.

A pass here justifies exactly one thing: proceeding to the frozen manifest and ledger path with a
candidate that has already passed every gate at all three seeds.
