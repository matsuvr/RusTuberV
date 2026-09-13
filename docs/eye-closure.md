# Eye-closure correction

RusTuberV can force an eye that is judged fully closed to an exact `1.0` blink
weight, so smoothing, loss recovery, and the avatar writer's epsilon no longer
leave a thin half-open eye. Correction is off by default and only activates
when a validated profile is installed.

Since algorithm v2 the primary condition is the MediaPipe lid geometry: for
each eye the perpendicular gaps of three upper/lower lid pairs are divided by
the eye width and the maximum is the `max_lid_gap_ratio`. The same eye's raw
MediaPipe blink score is only an auxiliary condition. The earlier raw-blink
openness feature is retained for the offline R comparison only.

The whole pipeline lives in the existing `xtask` development tool. Raw pixels,
derived traces, labels, review images, and absolute paths stay in gitignored
local working directories; only synthetic fixtures are committed.

## 1. Build the analysis inputs (#51/#65)

Create a small `inputs.json` naming each capture or derived trace explicitly.
No disk is scanned and no rotation is guessed:

```json
{
  "schema_version": 1,
  "inputs": [
    {
      "take_id": "20260827T150900Z_take_01_1fbe4b9d",
      "kind": "trace_v2",
      "path": "data/datasets/20260827T150900Z_take_01_1fbe4b9d-trace-v2",
      "origin": "20260827T150900Z_take_01_1fbe4b9d",
      "raw_frames_root": "data/raw/tmp_arkit_take/20260827T150900Z_take_01_1fbe4b9d"
    }
  ]
}
```

`kind` is `raw`, `trace_v1`, or `trace_v2`. A raw and a derived view of the
same recording share an `origin` and must not both be listed. `raw` inputs are
re-inferred with the current MediaPipe runtime; trace inputs reuse their
recorded observations.

```powershell
cargo run -p xtask --release -- eye-closure inspect --inputs data/eye-closure/inputs.json --output data/eye-closure/inventory.json
cargo run -p xtask --release -- eye-closure extract --inputs data/eye-closure/inputs.json --output data/eye-closure/extracted
```

`extract` writes `eye-frames.jsonl`, `eye-frames.csv`, and
`extraction-metadata.json` (schema 2). A trace v2 input must carry
`landmarks_xy`; a v2 observation missing it is a read error. For every
observation the extraction stores `lid_gap_left/right`,
`lid_points_left/right` (eight fixed points per eye), `inference_width/height`
of the image the normalized coordinates refer to, `mp_blink_left/right`, and
separate `seq_gap_before` / `time_gap_before` flags. Trace v1 has no
landmarks; its geometry stays `null`. A missing MediaPipe observation stays
`null` (unknown), never `0`.

## 2. Label and fit (#52/#64/#66)

```powershell
cargo run -p xtask --release -- eye-closure prepare-labels --data data/eye-closure/extracted --output data/eye-closure/review
cargo run -p xtask --release -- eye-closure fit --data data/eye-closure/extracted --labels data/eye-closure/labels.csv --split data/eye-closure/split.json --output data/eye-closure/fit
cargo run -p xtask --release -- eye-closure evaluate --data data/eye-closure/extracted --labels data/eye-closure/labels.csv --split data/eye-closure/split.json --profile data/eye-closure/fit/candidate_profile.json --output data/eye-closure/test
```

- `prepare-labels` picks ARKit-proxy candidates, crops the eye region from the
  stored lid points (falling back to the fixed band for trace v1), and writes
  review sheets, a `labels-template.csv`, and a `review-index.csv` with the
  sheet/cell, take, sequence, capture time and raw frame path. The template
  event id groups a contiguous proxy-closed run into one event.
- `labels.csv` rows are `take_id,frame_seq,eye,label,label_source,tag,event_id`.
  `label` is `fully_closed`, `not_closed`, `uncertain`, or `unobservable`.
  `label_source` is `visual_review` or `arkit_proxy`. An empty `event_id` means
  "not part of a reviewed closure event". Visual review is required before a
  profile can be verified.
- `split.json` names `train`, `validation`, and `test` takes. Takes from one
  session must stay in one split.
- `fit` reads only train/validation and replays every observation in capture
  order through the shared trackers. Re-fed samples do not advance the latch,
  sequence jumps with a normal capture-time step do not reset, capture-time
  gaps of 250 ms or more do, and missing eyes become `Unknown` (no pin).
  It writes `candidates.csv` for the R (raw blink), G (lid gap only) and H
  (lid gap plus blink) families plus `report.md`. A `candidate_profile.json`
  (status `candidate`) is written only when both eyes satisfy every measured
  target on train and validation.
- The metric groups are separate: visual reviewed-frame recall/false-close,
  continuous labelled-interval time, and `(take_id, eye, event_id)` closure
  events. Sparse review frames are never expanded into global time, no nominal
  frame duration is added, and an unmeasured group fails acceptance.
- `evaluate` reads the test split once. Only when both eyes pass every group
  with visual labels and matching inference fingerprints does it write
  `test/eye_closure_profile.json` (status `verified`).

The measured local outcome of the first v2 run (no adopted threshold because
the labelled frames overlap and continuous-interval metrics are unmeasured) is
in [`eye-closure-run-v2-20260913.md`](eye-closure-run-v2-20260913.md).

## 3. Install or remove a profile (#53)

```powershell
cargo run -p xtask --release -- eye-closure install-profile --profile data/eye-closure/test/eye_closure_profile.json
cargo run -p xtask --release -- eye-closure remove-profile
```

The profile is written to `eye_closure_profile.json` next to `settings.toml`
in the per-user config directory (`%APPDATA%\RusTuberV` on Windows). Use
`--config-dir <dir>` for a staging directory. Only a `verified` profile whose
fingerprints match the running MediaPipe task bundle is installed; a candidate
is refused.

The app loads the profile once at startup. A missing, malformed, foreign, or
incompatible profile is logged and ignored; no default threshold is
substituted. A v1 raw-blink profile is not migrated and is ignored. Removing
the file and restarting restores the previous behavior.

## Profile format (v2)

```json
{
  "schema_version": 2,
  "algorithm_version": 2,
  "feature": "mediapipe_max_lid_gap_blink_v2",
  "status": "verified",
  "left": { "close_gap": 0.09, "reopen_gap": 0.12, "min_blink": 0.0 },
  "right": { "close_gap": 0.09, "reopen_gap": 0.12, "min_blink": 0.0 },
  "fingerprints": {
    "task_bundle_sha256": "64184E22...",
    "feature": "mediapipe_max_lid_gap_blink_v2",
    "preprocess": "mediapipe face landmarker landmarks and raw blendshapes"
  },
  "applies_to": "held-out captures; real webcam conditions not yet confirmed"
}
```

Per eye, `a = max_lid_gap_ratio` and `b = raw MediaPipe blink`:

- from `Open` or `Unknown`: `a <= close_gap AND b >= min_blink` closes;
- from `Closed`: `a >= reopen_gap` reopens; the blink score is not consulted;
- otherwise the previous state is kept and `Unknown` never emits a pin.

`0 <= close_gap < reopen_gap` and `0 <= min_blink <= 1` are enforced. When
the geometry alone separates the data, `min_blink = 0` is selected and the raw
blink is effectively excluded.

## Limitations

- Thresholds fitted on one set of captures are not guaranteed on a different
  webcam, field of view, or lighting. Re-fit and re-verify for the actual
  capture conditions.
- The lid-gap ratio is an image-plane ratio, not a physical distance; it is
  invariant to translation, uniform scale, and in-plane rotation only. It does
  not correct for yaw/pitch or perspective foreshortening.
- Sparse review labels cannot measure continuous closure time or event
  boundaries: contiguous frames of the existing recordings must be reviewed
  before time/event metrics can verify a profile.
- A model with a single combined blink morph cannot show a wink; the existing
  fallback semantics are unchanged.
- Models whose closed-eye shape is driven by a competing morph are a model
  binding issue, not a threshold issue.
