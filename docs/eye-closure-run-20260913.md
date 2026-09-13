# Eye-closure correction: local data run (2026-09-13)

Outcome of running the `eye-closure` pipeline (Issues #51-#54) on the local
ARKit/trace-v2 recordings with visual-review labels. **No threshold profile
could be validated, so nothing was installed**; the runtime feature stays
dormant (unchanged default behavior).

## Inputs and extraction (Issue #51)

- Input format: derived trace v2 + raw `.bin` JPEG frames (480x640,
  `jpeg-rgb8-srgb-upright-nonmirrored`), 180-degree recorded pixel rotation
  applied per take, not mirrored.
- 8 takes, all `COMPLETED`, 3,057 recorded frames; 2,957 paired analysis
  frames (manifest-explained unpaired/dropped excluded by the tooling).
- ARKit teacher present on all frames. MediaPipe observation missing on 5
  frames (4 in `78fa553c`, 1 in `6158a4df`); time gaps on 2 frames
  (`6608daa4` seq 2, `6158a4df` seq 264). Missing observations stay unknown
  (null), never blink=0.
- All takes share one MediaPipe task bundle hash
  (`64184E22...BC9FF`), i.e. recorded under the current runtime.
- Total footage ~98.6 s at 30 fps.

## Subjects

The recordings contain at least four distinct subjects (home desk takes
`1fbe4b9d`, `6608daa4`, `c85aaf92`, `4df068e1`, `5e3c08c3`; three different
restaurant subjects `78fa553c`, `6158a4df`, `02374e00`). Per user decision
all subjects were mixed into one fit (personal calibration was not
restricted), with the limitation noted below.

## Visual labels (Issue #52)

`prepare-labels` produced 384 candidate frames (48 per take) with review
sheets; all 384 frames were reviewed visually (iris/sclera visibility) at
full resolution and labeled per eye. Labels were fixed before fitting; the
split was fixed before fitting.

Per-eye label counts (768 label rows, `label_source=visual_review`):

| eye | fully_closed | not_closed | uncertain |
| --- | --- | --- | --- |
| left | 15 | 365 | 4 |
| right | 17 | 362 | 5 |

Closure events (visual): 8 both-eye blinks (one per take that contains one),
1 sustained left wink (7 frames), 2 sustained right winks (4 + 5 frames).

Split (take-level, one session per take so group leakage is impossible):

- train: `4df068e1` (winks + blinks), `1fbe4b9d`, `78fa553c` (negatives)
- validation: `6608daa4`, `c85aaf92` (negatives), `02374e00`
- test: `5e3c08c3` (blink + sustained half-open), `6158a4df`

## Fit result: `no_acceptable_threshold` for both eyes

`fit` explored the fixed grid (close_at step 0.005, hysteresis widths
{0.01, 0.02, 0.04, 0.08}) with the shared hysteresis tracker and the
required constraints (closed recall >= 0.95, false-close <= 0.01). **No
candidate passed train for either eye; no candidate profile was written, so
`evaluate` had nothing to freeze and no verified profile exists.**

The measured trade-off frontier (train split):

| eye | best with false-close <= 1% | best with recall 100% |
| --- | --- | --- |
| left | close_at 0.28 -> recall 36% | close_at 0.685 -> false-close 57.6% |
| right | close_at 0.35-0.36 -> recall 15% (false-close 0%) | close_at 0.755 -> false-close 56.6% |

Root cause, from the labeled frames: the raw MediaPipe blink score
overlaps between visually fully-closed and visibly open/half-open eyes
across takes:

- left: wink-level full closure scores as low as 0.412 (mp blink), while
  half-open eyes score up to 0.760;
- right: wink-level full closure scores as low as 0.357, while squinting
  eyes score up to 0.637.

MediaPipe's raw blendshape saturates early for this data: a squeezed-shut
eye is not reliably distinguishable from a half-open eye by the one scalar,
under differing distance, pose, and lighting per take. This matches the
issue's warning that a single scalar threshold may not exist; adding
features (EAR etc.) is out of scope for this issue.

## What was NOT done (unmeasured, not to be inferred)

- `evaluate` on the test split: not run (no frozen candidate exists).
- Verified profile production and `install-profile`: not possible without a
  passing candidate.
- Real webcam / VRM appearance checks (standard and Perfect Sync, wink,
  half-open, mirror, lifecycle): not run. The runtime feature itself is
  implemented and regression-tested on synthetic profiles (Issue #53) but
  remains inactive because no validated profile is installed.
- Per-subject calibration: analyzed, also infeasible on the current takes
  (the same overlap exists within the home-subject takes alone: left wink
  closure 0.412-0.47 vs squint 0.48-0.58; right wink closure 0.357-0.46 vs
  squint 0.39-0.41 at ARKit-verified openness).

## Practical conclusions for the remaining issue

1. The current recordings cannot produce a global (or per-subject) threshold
   pair that satisfies recall >= 95% with false-close <= 1%. The limiting
   factor is the feature (raw per-eye blink scalar), not the pipeline.
2. A usable profile needs controlled recordings: sustained full blinks,
   left/right winks, half-open and down-gaze negatives, at fixed distance
   and lighting per session, reviewed visually as done here.
3. Alternatively the runtime would need a stronger closure feature; that is
   a new decision and out of scope for #54.

## Reproduction (local data only, not in the repository)

```sh
cargo run -p xtask --release -- eye-closure inspect --inputs data/eye-closure/inputs.json --output data/eye-closure/inventory.json
cargo run -p xtask --release -- eye-closure extract --inputs data/eye-closure/inputs.json --output data/eye-closure/extracted
cargo run -p xtask --release -- eye-closure prepare-labels --data data/eye-closure/extracted --output data/eye-closure/review
cargo run -p xtask --release -- eye-closure fit --data data/eye-closure/extracted --labels data/eye-closure/labels.csv --split data/eye-closure/split.json --output data/eye-closure/fit
# fit prints: no acceptable threshold for both eyes; candidate_profile.json was not written
```

- tool commit at run time: `9e8909a` (Issue #53), xtask 0.1.0
- label sha256: `3E0937FF29AFD2DC08CE562808B345206CBA48F60F659F5868EFCF1CFCDAD084`
- raw pixels, traces, review sheets, labels, and split stay in the
  gitignored local working directories; nothing personal is committed.
