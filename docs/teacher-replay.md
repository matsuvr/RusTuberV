# ARKit teacher offline replay and linear prior fit (GNM #68.3/#68.4)

Development-only workflow that turns a completed ARKit teacher capture
dataset (GNM #68.2 layout) into a derived `PairedTemporalSample` trace and,
optionally, a fitted causal linear prior artifact. Raw face pixels stay
outside version control at every step (GNM #68.1 privacy boundary).

## 1. Replay (`teacher-replay`)

```powershell
cargo run -p xtask --release -- teacher-replay `
  --dataset data/raw/tmp_arkit_take/<take-dir> `
  --pixel-rotation 180 `
  --fit-tolerance 0.0001
```

- `--dataset` points at a completed capture directory (must contain the
  `COMPLETED` marker). Output defaults to `data/datasets/<take-dir>/`
  (gitignored).
- `--pixel-rotation` corrects a mis-declared capture orientation without
  modifying the stored dataset bytes. take_01 was recorded with
  `stored_orientation_degrees: 0`, but the stored pixels are rotated 180
  degrees (verified visually); replaying without the correction yields
  ~67% `NoFace`. The correction is recorded in `replay-metadata.json`.
- `--fit-tolerance 0.0001` is the cold-start convergence threshold. The
  fixed model-neutral identity leaves a systematic residual floor around
  0.0065, at which the solver crawls below the production `1e-6` warm-start
  threshold; `1e-4` reaches a plateau in a few iterations. The value is
  part of the recorded config.
- Pairing is exact-identity only (`frame_seq` + identical timestamps).
  Missing, duplicate, or mismatched frames abort the run instead of being
  nearest-repaired.

Outputs:

- `derived-trace.jsonl` — schema v2, one `PairedTemporalSample` row per frame.
  `mediapipe_observation` retains 478 normalized `(x, y)` landmarks (not z),
  camera-to-face transform, direct MediaPipe → ARKit52 coefficients, and the
  existing quality values. `gnm_state` retains the pre-projection Head-v3
  expression with indices `350..382` (tongue) removed, joint rotations,
  rigid/camera state, projected ARKit52 coefficients, solver objective, and
  diagnostic per-region RMS. `baseline_output` equals the direct observation.
  Teacher and RGB-reference metadata remain present; neither pixels nor
  absolute paths are embedded.
- `replay-metadata.json` — SHA-256 of every input file and of the trace
  bytes (`trace_sha256`), plus the full config (task bundle hash, GNM model
  hash, fit config, pacing/rotation policies). Re-running with the same
  inputs and config regenerates byte-identical outputs.

Downstream `teacher-fit-prior`, `teacher-fit-residual`, and both ablation
commands require schema v2 metadata. Existing schema v1 traces must be
regenerated; there is no implicit migration or fallback.

Determinism: the replay is paced to the capture cadence so MediaPipe sees
the recorded frame intervals; the fit is a stateless per-frame cold start
with rigid-pose recovery initialization and no auxiliary terms.

Measured on take_01 (939 frames, Windows 11, CPU): ~29 minutes wall clock
with `--fit-tolerance 0.0001`, 939/939 solved.

## 2. Fit the causal linear prior (`teacher-fit-prior`)

```powershell
cargo run -p xtask --release -- teacher-fit-prior `
  --trace data/datasets/<take-dir> `
  --train-take <take-id> `
  --output data/datasets/linear-prior.json
```

- Each `--trace` contributes one take; the take id comes from
  `replay-metadata.json`. Train/validation/test splits for the #112
  ablation must select takes explicitly via `--train-take` and stay
  take-disjoint.
- The exported artifact is verified through the production load boundary
  (`LoadedLinearPrior::load`) before the bytes are accepted, and its
  SHA-256 is printed.

## 3. Raw RGB deletion workflow

The derived trace is self-sufficient for all downstream research (#111
fit, #112 ablation): every row carries the teacher coefficients, both
baseline variants, and the RGB *reference* metadata (path/dimensions/format),
never pixels.

1. Verify the derived trace:

   ```powershell
   $meta = Get-Content data/datasets/<take-dir>/replay-metadata.json | ConvertFrom-Json
   $hash = (Get-FileHash data/datasets/<take-dir>/derived-trace.jsonl -Algorithm SHA256).Hash
   $hash -eq $meta.trace_sha256   # must be True
   ```

2. Confirm `counts.solved` covers the frames you need and
   `source_dataset.input_hashes` still matches the capture dataset files
   you intend to delete.

3. Delete the raw pixels. Either the extracted `frames/` payloads, the
   extracted take directory, or the source ZIP under `data/raw/` — any or
   all. `data/raw/**` is gitignored and additionally blocked by the local
   pre-commit hook, so deletion is a storage-privacy decision, not a
   repository-safety one. Keep the ZIP only if you may need to re-run the
   replay with a corrected config (e.g. a different rotation or tolerance);
   otherwise deleting it is the privacy-preferred state.

4. Never move capture data out of `data/raw/`; never commit raw frames,
   traces with pixel payloads, or absolute paths. Derived numeric fixtures
   for tests are the only committable artifacts (GNM #68.1).

## Known capture-data caveats (take_01)

- The stored pixels are rotated 180 degrees relative to the declared
  orientation; replay with `--pixel-rotation 180`.
- The MediaPipe Tasks runtime may emit telemetry upload attempts
  (`clearcut` log lines); face frames and landmarks are processed on
  device, but zero-network behavior is not claimed (see AGENTS.md).
- Single take: fitting on it and evaluating on it is *not* a valid
  ablation. #112 requires additional held-out takes from the iPhone
  capture app.
