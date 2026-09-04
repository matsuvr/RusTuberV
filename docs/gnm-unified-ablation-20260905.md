# Unified GNM ablation — 2026-09-05

Issue: #19

## Result

The primary `A+B train -> A+B held-out` evaluation passes all seven promotion
conditions on 1,184 exact common frames. The selected candidate is rank 16,
history 1, ridge 0.1. Candidate selection used all 48 fixed combinations and
leave-one-take-out validation inside the four-take outer training set. The same
candidate is used by G1, H, L, and HL.

| Variant | Macro MAE | Macro RMSE | Micro MAE | Missed blinks | Median onset / peak / attenuation | VRM morph RMS |
|---|---:|---:|---:|---:|---:|---:|
| D | 0.099813 | 0.140813 | 0.099813 | 11 | 0.000 ms / 0.000 ms / 0.185033 | 0.000839039 |
| G0 | 0.202610 | 0.270024 | 0.202610 | 37 | n/a / n/a / n/a | 0.001691963 |
| G1 | 0.073478 | 0.105175 | 0.073478 | 12 | 0.000 ms / 33.328 ms / 0.246099 | 0.000704810 |
| L | 0.083203 | 0.123413 | 0.083203 | 4 | 0.000 ms / 0.000 ms / 0.240733 | 0.000715681 |
| H | 0.074235 | 0.114296 | 0.074235 | 2 | 0.000 ms / 0.000 ms / 0.155123 | 0.000744568 |
| HL | 0.084648 | 0.126188 | 0.084648 | 2 | 0.000 ms / 0.000 ms / 0.114832 | 0.000774397 |

The avatar metric uses all 51 recognized non-tongue Perfect Sync binds from
`SapphyPerfectSync.vrm`, SHA-256
`AFD51E320E97B398DB5DBA738E19C3BAE01263E0ACCEB4EFB2CCEC22D78576EE`.
No mesh or absolute local path is exported.

## Promotion conditions

- PASS: H macro MAE and RMSE are below D.
- PASS: H macro MAE and RMSE are below L.
- PASS: H missed blinks (2) are no greater than D (11).
- PASS: H median absolute blink onset and peak timing errors are no greater than D.
- PASS: H median absolute peak attenuation (0.155123) is no greater than D (0.185033).
- PASS: availability is numeric and evaluation uses an exact six-way intersection without fill.
- PASS: the A-held-out and B-held-out H-D paired macro deltas are both reported below.

## Outer splits and generalization boundary

| Outer split | Frames | H MAE | D MAE | H-D paired absolute-error delta | Split conditions |
|---|---:|---:|---:|---:|---|
| A train -> A held-out | 678 | 0.079510 | 0.099721 | -0.020211 | PASS |
| A train -> B held-out | 506 | 0.196672 | 0.099937 | +0.096735 | FAIL |
| A+B train -> A held-out | 678 | 0.066314 | 0.099721 | -0.033407 | PASS |
| A+B train -> B held-out | 506 | 0.084848 | 0.099937 | -0.015088 | value PASS; blink timing/attenuation FAIL alone |

The A-only cross-person result is materially worse than Direct. The promotion
result therefore supports the documented two-person training population only;
it is not evidence of unseen-person generalization.

## Availability

| Held-out take | Paired | Solved | No face | Fit rejected | Unpaired RGB | History/gap excluded | Common |
|---|---:|---:|---:|---:|---:|---:|---:|
| `20260828T053203Z_take_01_c85aaf92` | 331 | 331 | 0 | 0 | 0 | 1 | 330 |
| `20260828T053225Z_take_01_4df068e1` | 349 | 349 | 0 | 0 | 0 | 1 | 348 |
| `20260830T115657Z_take_01_6158a4df` | 282 | 266 | 1 | 15 | 24 | 3 | 263 |
| `20260830T115716Z_take_01_02374e00` | 244 | 244 | 0 | 0 | 186 | 1 | 243 |

`no_face`, `fit_rejected`, source pairing exclusions, and causal history/gap
exclusions remain separate. Missing values are never held, interpolated, or
replaced by Direct in the common-frame metrics.

## Reproduction

All eight trace-v2 inputs were regenerated from `data/raw/` using
`teacher-replay --pixel-rotation 180 --fit-tolerance 0.0001`. Run
`teacher-unified-gnm-ablation` with all eight `--trace` directories, the chosen
outer train/eval take IDs, `--observable-rank 48`, the local Perfect Sync VRM,
and an output directory. The A+B B-held-out run may use `--reuse-fit` pointing
to the A+B A-held-out output; reuse is rejected unless all decoder artifacts
have exactly the requested training take set.

Each output contains `summary.json`, `report.md`, `candidate-grid.json`,
`framewise-errors.csv`, and the fitted basis/decoder artifacts. The committed
machine-readable result is
[`gnm-unified-ablation-20260905.json`](gnm-unified-ablation-20260905.json).

The primary training/evaluation dataset hashes are
`6F256CDE995E5443B54DA08B45DD37B9040147DBA35516963C86F0A2F4A4F26E` and
`68A8A23F2598FE8B23BEEB00618109AAB1F653A2368058E6255DD10C357CE560`.
