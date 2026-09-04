# Issue #20 reduced GNM benchmark

Measured on 2026-09-05 from the selected Issue #19 rank-16 teacher-aligned basis.
The input is the exact trace-v2 take `20260828T053203Z_take_01_c85aaf92`; model,
mapping, basis, and trace identifiers are recorded in the adjacent JSON report.

## Issue #19 gate

[The Issue #19 report](gnm-unified-ablation-20260905.md) records PASS for all seven
live-integration conditions on the primary A+B -> A+B held-out evaluation:
non-tongue macro MAE, non-tongue RMS, missed frames, blink onset/peak timing,
blink attenuation, left/right independence, and VRM morph-space RMS. The reduced
solver work therefore passed its mandatory start gate.

## Contract

The expression unknown is directly `q in R^16`. It is expanded as
`phi_non_tongue = B q`, with `J_q = J_phi B` chained from the analytic
non-tongue projection Jacobian. No full expression fit is projected into q, and
no rank column uses a finite-difference model evaluation. Rigid pose and joint
updates retain the existing block-coordinate objective, damping, bounds, and
64-iteration limit. The production worker/backend is unchanged.

A rank-351 identity-basis synthetic test compares full and reduced outcomes,
objective, and every expression coefficient. Known yaw, pitch, and roll fixtures
also pin the MediaPipe quaternion-to-`Rz * Rx * Ry` conversion. Artifact loading
fails closed on schema/hash/model/mapping/rank/orthogonality mismatch.

## Reference sequence result

Release command:

```powershell
cargo run --release -p xtask -j 1 -- teacher-benchmark-reduced-gnm --basis target/issue19/ab_to_a/gnm-basis.json --trace data/datasets/20260828T053203Z_take_01_c85aaf92-trace-v2 --take 20260828T053203Z_take_01_c85aaf92 --max-frames 120 --output target/issue20/benchmark.json
```

| Metric | Full expression | Reduced rank 16 |
|---|---:|---:|
| valid / rejected | 120 / 0 | 120 / 0 |
| wall time p50 / p95 / max (ms) | 117.820 / 173.680 / 1629.592 | 102.559 / 116.015 / 1368.319 |
| iterations p50 / p95 / max | 2 / 3 / 28 | 2 / 2 / 27 |
| weighted RMS p50 / p95 / max | 0.006971 / 0.008425 / 0.008685 | 0.008559 / 0.009582 / 0.009781 |
| final objective p50 / p95 / max | 0.006971 / 0.008425 / 0.008685 | 0.008559 / 0.009582 / 0.009781 |

Full-vs-reduced non-tongue expression RMS difference was 0.956734 p50,
1.362589 p95, and 1.411324 max. The maximum absolute coefficient in GNM indices
350..382 was exactly 0.0 across every full and reduced outcome.

This benchmark measures the sequential solver only. It does not measure worker
backlog, rendering, or live camera behavior.
