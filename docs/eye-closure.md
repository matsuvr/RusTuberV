# Eye-closure correction

RusTuberV can force an eye that MediaPipe reports as fully closed to an exact
`1.0` blink weight, so smoothing, loss recovery, and the avatar writer's
epsilon no longer leave a thin half-open eye. Correction is off by default and
only activates when a validated profile is installed.

The whole pipeline lives in the existing `xtask` development tool. Raw pixels,
derived traces, labels, review images, and absolute paths stay in gitignored
local working directories; only synthetic fixtures are committed.

## 1. Build the analysis inputs (#51)

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
`extraction-metadata.json`. A missing MediaPipe observation stays `null`
(unknown), never `0`.

## 2. Label and fit (#52)

```powershell
cargo run -p xtask --release -- eye-closure prepare-labels --data data/eye-closure/extracted --output data/eye-closure/review
cargo run -p xtask --release -- eye-closure fit --data data/eye-closure/extracted --labels data/eye-closure/labels.csv --split data/eye-closure/split.json --output data/eye-closure/fit
cargo run -p xtask --release -- eye-closure evaluate --data data/eye-closure/extracted --labels data/eye-closure/labels.csv --split data/eye-closure/split.json --profile data/eye-closure/fit/candidate_profile.json --output data/eye-closure/test
```

- `prepare-labels` picks ARKit-proxy candidates (peaks, sustained closures,
  asymmetries, half-open and open negatives), writes review sheets and a
  `labels-template.csv`, and never claims the ARKit proxy is ground truth.
- `labels.csv` rows are `take_id,frame_seq,eye,label,label_source,tag,event_id`.
  `label` is `fully_closed`, `not_closed`, `uncertain`, or `unobservable`.
  `label_source` is `visual_review` or `arkit_proxy`. Visual review is required
  before a profile can be verified.
- `split.json` names `train`, `validation`, and `test` takes. Takes from one
  session must stay in one split.
- `fit` reads only train/validation and writes `candidate_profile.json` (status
  `candidate`) plus `candidates.csv` and `report.md`. If no candidate passes
  the closed-recall and false-close constraints for both eyes, no profile is
  written.
- `evaluate` reads the test split once. Only when both eyes pass with visual
  labels and matching inference fingerprints does it write
  `test/eye_closure_profile.json` (status `verified`).

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
substituted. Removing the file and restarting restores the previous behavior.

## Profile format

```json
{
  "schema_version": 1,
  "algorithm_version": 1,
  "feature": "mediapipe_raw_eye_blink_openness_v1",
  "status": "verified",
  "left": { "close_at": 0.42, "reopen_at": 0.55 },
  "right": { "close_at": 0.44, "reopen_at": 0.57 },
  "fingerprints": {
    "task_bundle_sha256": "64184E22...",
    "feature": "mediapipe_raw_eye_blink_openness_v1",
    "preprocess": "mediapipe face landmarker raw blendshapes"
  },
  "applies_to": "held-out iPhone-derived captures; real webcam conditions not yet confirmed"
}
```

The feature is the raw MediaPipe `eyeBlinkLeft/Right` score as an openness
proxy `o = 1 - raw_blink`. `o` is not a physical lid distance. The judgement is
per eye: an open eye closes at `o <= close_at`, a closed eye reopens at
`o >= reopen_at`, and values between the edges keep the previous state.
`0 <= close_at < reopen_at <= 1` is enforced.

## Limitations

- Thresholds fitted on iPhone-derived captures are not guaranteed on a
  different webcam, field of view, or lighting. Re-fit and re-verify for the
  actual capture conditions.
- A model with a single combined blink morph cannot show a wink; the existing
  fallback semantics are unchanged.
- Models whose closed-eye shape is driven by a competing morph are a model
  binding issue, not a threshold issue.
