# Eye-closure v2: re-measured local run (2026-09-13)

Re-run of the eye-closure pipeline after the evaluation corrections (Issue
#64), the `max_lid_gap_ratio` geometry feature (Issue #65), and the
geometry-primary judgement with profile v2 (Issue #66).

The earlier run and its `no_acceptable_threshold` result remain in
[`eye-closure-run-20260913.md`](eye-closure-run-20260913.md) unchanged; this
document is the successor measurement, not a replacement of that history.

**Outcome: no threshold was adopted and no profile was written.** The runtime
feature stays dormant. The R baseline numbers are reproduced by the corrected
metrics, and the lid-gap geometry does **not** separate the existing labelled
frames better than the raw blink score.

## What changed in the measurement

- Feature: `mediapipe_max_lid_gap_blink_v2` (was
  `mediapipe_raw_eye_blink_openness_v1`); schema and algorithm version 2.
- Time classification is shared: a sequence-number jump with a normal
  capture-time step is no longer a reset; a capture-time gap of 250 ms or more
  is. Missing/non-finite observations become `Unknown` (no closure pin), never
  a carried-over `Closed`.
- Metric groups are reported separately and are never substituted:
  - reviewed-frame recall/false-close (visual labels only),
  - continuous labelled-interval time (both endpoints labelled, observed,
    sequence-adjacent and capture-time-adjacent; no nominal 33.3 ms is added),
  - closure events keyed by `(take_id, eye, event_id)`.
- `fit` writes `candidates.csv` for the R (raw blink), G (lid gap only,
  `min_blink=0`) and H (lid gap plus blink) candidate families and reports the
  frontier for each. Acceptance still requires every measured group;
  unmeasured groups fail.

## Inputs and extraction

- Same 8 trace-v2 takes, same take-level train/validation/test split as before.
- Re-extraction with schema 2 kept 2,957 frames (2,952 with a MediaPipe
  observation; 5 missing stay `null`). Two sequence-number jumps were recorded:
  `6608daa4` seq 0 -> 2 has a normal capture-time step (no reset), while
  `6158a4df` seq 264 is also a capture-time gap of 250 ms or more and does
  reset the latch.
- All stored 478-point `landmarks_xy` were converted with the shared
  `mediapipe_lid_points` / `max_lid_gap_ratio` functions; the inference image
  was 480x640 (180-degree rotation, non-mirrored) for every reported take.
- `prepare-labels` now crops the eye region from the stored `LidPoints` and
  writes sheet name, cell, take, sequence, capture time and the raw frame path
  into `review-index.csv`; future template event ids group a contiguous
  proxy-closed run into one event.

## Labels (local normalization)

The existing 768 visual labels were kept. The previous `labels.csv` used one
event id per reviewed frame (`take:frame:eye`), which would have counted one
sustained wink as many events. For this run the event ids were normalized
locally to `(take, eye, contiguous fully_closed run)`; the label values,
sources, frame set and split were not changed. The normalized label file has
sha256 `152A38A958139014C4792123D5125C5DB6BA594D31A23DA31BC979AACBA29507`.

Because the review frames are sparse, **no two visually labelled
`fully_closed` frames are sequence-adjacent**. Continuous-interval metrics are
therefore unmeasured (`None`) for every candidate on both splits, and event
metrics degenerate to one event per reviewed fully-closed frame. This is a
label-coverage limit, not a metric failure.

## Feature distributions on the visual labels

Lid-gap ratio (`max_lid_gap_ratio`) and raw MediaPipe blink over the 384
reviewed frames per eye:

| eye | label | n | gap min | gap median | gap max | blink min | blink median | blink max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| left | fully_closed | 15 | 0.0075 | 0.0571 | 0.0920 | 0.4124 | 0.6522 | 0.8771 |
| left | not_closed | 365 | 0.0209 | 0.2847 | 0.5309 | 0.0068 | 0.2599 | 0.7597 |
| right | fully_closed | 17 | 0.0178 | 0.1748 | 0.2060 | 0.3573 | 0.4496 | 0.7843 |
| right | not_closed | 362 | 0.0588 | 0.2965 | 0.5862 | 0.0019 | 0.1763 | 0.6368 |

The classes overlap on both features for both eyes. On the left eye the
maximum fully-closed gap (0.092) sits above the minimum not-closed gap (0.021);
on the right eye the closed median gap (0.175) is far above the open minimum
(0.059). The geometry is not a clean separator on this data.

## R/G/H comparison (same labels, frames and metric code)

Train split, best frame recall with frame false-close at or below 1%
(reviewed frames; interval and event groups unmeasured):

| eye | class | best candidate | frame recall | frame false-close |
| --- | --- | --- | --- | --- |
| left | R raw blink | close_at 0.250 / reopen_at 0.270 | 36.4% | 0.0% |
| left | G gap only | close_gap 0.0208 / reopen_gap 0.0209, min_blink 0 | 9.1% | 0.0% |
| left | H gap + blink | close_gap 0.0231 / reopen_gap 0.0471, min_blink 0.75 | 18.2% | 0.8% |
| right | R raw blink | close_at 0.360 / reopen_at 0.440 | 15.4% | 0.0% |
| right | G gap only | close_gap 0.0549 / reopen_gap 0.0734, min_blink 0 | 15.4% | 0.0% |
| right | H gap + blink | close_gap 0.0944 / reopen_gap 0.1288, min_blink 0.55 | 23.1% | 0.8% |

The corrected R baseline reproduces the earlier qualitative result (about
36% left / 15% right at false-close <= 1%) with small threshold shifts from
the time-classification fix.

On validation, the best candidates look better only because the validation
split contains one easy fully-closed frame per eye: left R at
close_at 0.350/reopen_at 0.360 and left G at close_gap 0.0920/reopen_gap 0.0974
both reach 100% recall with 0.7% false-close, and a right H candidate reaches
100% recall with 2.1% false-close. They still fail the train split and the
95% recall target where the difficult winks and squints are labelled.

**Conclusion:** the v2 geometry-primary hypothesis is not supported by the
current visual labels. No G/H candidate satisfies the targets; R remains a
failed baseline as well. Per the issue, no fallback to another detector, no
threshold lowering, and no MLP/SVM substitution was made.

## What is still unmeasured (not PASS)

- Continuous labelled-interval recall/false-close and onset/release timing:
  no sequence-adjacent visual labels exist. Additional contiguous review of
  existing frames is required.
- Event attainment is measured only over sparse single-frame events.
- `evaluate` was not run: `fit` wrote no `candidate_profile.json`.
- Verified profile production, `install-profile`, runtime on/off behavior:
  not reached.
- Real webcam, standard/Perfect Sync weight inspection, mirror and lifecycle
  checks, and a new independent webcam take: not performed. This document
  claims no acceptance.

## Reproduction (local data only, not in the repository)

```sh
cargo run -p xtask --release -- eye-closure extract --inputs data/eye-closure-v2/inputs.json --output data/eye-closure-v2/extracted
cargo run -p xtask --release -- eye-closure prepare-labels --data data/eye-closure-v2/extracted --output data/eye-closure-v2/review
cargo run -p xtask --release -- eye-closure fit --data data/eye-closure-v2/extracted --labels data/eye-closure-v2/labels.csv --split data/eye-closure-v2/split.json --output data/eye-closure-v2/fit
# fit prints: no acceptable geometry threshold for both eyes; candidate_profile.json was not written
```

Local-only artifacts (not committed): `data/eye-closure-v2/` with
`extracted/`, `review/` (LidPoints crops and index), `fit/report.md`,
`fit/candidates.csv`, the normalized `labels.csv`, and
`feature-comparison.csv` (`take,seq,eye,raw_blink,lid_gap,label`).
