# Teacher residual decoder artifact (Issue #12)

This run fits only the same-frame, non-tongue target
`teacher_51(t) - MediaPipeDirect_51(t)`. It uses the existing derived traces;
no RGB replay or new capture was performed, and the decoder is not connected
to the production runtime.

## Fixed feature order

Newest-first history slots contain Direct 51, hand-projected GNM 51, then the
GNM scalar objective. The velocity slot contains Direct 51 velocity followed
by GNM 51 velocity. The final scalar is the actual `dt_seconds`. With
`history_len=4`, the feature width is 515.

Artifact feature-order value:

```text
v1:newest-first-history(direct-51+gnm-projected-51+gnm-residual)+velocity(direct-51+gnm-projected-51)+dt-seconds
```

## Training result

- Training takes: `20260827T150900Z_take_01_1fbe4b9d`,
  `20260828T053142Z_take_01_6608daa4`,
  `20260830T115106Z_take_01_5e3c08c3`,
  `20260830T115611Z_take_01_78fa553c`
- Accepted rows: 1,744
- Exclusions: `MissingDirect=4`, `MissingGnmState=3`, `SequenceBoundary=1`
- Configuration: `history_len=4`, `max_gap_micros=100000`,
  `ridge_lambda=0.001`
- Content hash: `206715820215849832`
- Artifact SHA-256:
  `7B5D4BD88B0E4B9D58E8498E7FB183166818616F801A3BB3C8E0265F0EB53BEC`
- Local artifact (gitignored):
  `data/datasets/teacher-residual-issue12-train4.json`

The command was run twice to separate output paths. Both JSON byte streams
had the SHA-256 above.

## Reproduction

```powershell
target/release/xtask.exe teacher-fit-residual `
  --trace data/datasets/20260827T150900Z_take_01_1fbe4b9d `
  --trace data/datasets/20260828T053142Z_take_01_6608daa4 `
  --trace data/datasets/20260830T115106Z_take_01_5e3c08c3 `
  --trace data/datasets/20260830T115611Z_take_01_78fa553c `
  --train-take 20260827T150900Z_take_01_1fbe4b9d `
  --train-take 20260828T053142Z_take_01_6608daa4 `
  --train-take 20260830T115106Z_take_01_5e3c08c3 `
  --train-take 20260830T115611Z_take_01_78fa553c `
  --history-len 4 --max-gap-micros 100000 --ridge-lambda 0.001 `
  --output data/datasets/teacher-residual-issue12-train4.json
```
