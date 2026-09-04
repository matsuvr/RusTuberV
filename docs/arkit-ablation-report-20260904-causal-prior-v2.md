# ARKit teacher ablation report — 2026-09-04 (corrected causal prior, 51ch)

Successor to `docs/arkit-ablation-report-20260830.md`. This report keeps the
same take-disjoint split and evaluates the corrected ridge right-hand side
over the 51 ARKit channels excluding `TongueOut`. The historical report and
schema-v1 artifacts remain unchanged.

The prior fit previously computed each feature's right-hand side as
`Σ_r X[r,i] * (Σ_s Y[s,target])`. It now computes the row-aligned cross
covariance `Σ_r X[r,i] * Y[r,target]`. Because normalized target columns have
approximately zero sum, the old expression drove the learned weights toward
zero and was not detected by tests that checked determinism and artifact
shape without checking recovery of a known mapping.

## Data provenance and split

The eight replay traces used below were derived from the corresponding
captures under `data/raw/tmp_arkit_take/`. Their five recorded input hashes
(`capture.json`, `frames.jsonl`, `manifest.json`, `rgb.jsonl`, `session.json`)
were rechecked against the current raw files: **40/40 matched**. A matching
capture ZIP is also present in `data/raw/` for every take. Replaying the pixels
again is unnecessary for this causal-fit correction because replay output is
an input to, and is not changed by, dataset construction or ridge fitting.

| Person | Role | Takes | Rows used by fit/evaluation |
| --- | --- | --- | ---: |
| A | train | `1fbe4b9d`, `6608daa4` | 1,196 |
| B | train | `5e3c08c3`, `78fa553c` | 538 |
| A | held out | `c85aaf92`, `4df068e1` | 678 |
| B | held out | `6158a4df`, `02374e00` | 506 |

P_AB uses all four training takes (1,734 rows, two people). P_A uses only the
two person-A training takes (1,196 rows). No training take occurs in an
evaluation set. `TongueOut` is absent from history, velocity, target, metric
denominators, and channel tables; conversion back to ARKit52 fixes it to zero.

## Overall results

### P_AB on held-out person A

| Metric | Direct | GNM no-temporal | corrected prior |
| --- | ---: | ---: | ---: |
| value MAE | **0.09972** | 0.22107 | 0.22313 |
| value RMSE | **0.15726** | 0.35574 | 0.35054 |
| velocity MAE (1/s) | **20.044** | 22.900 | 22.401 |
| acceleration MAE (1/s²) | **684.38** | 719.94 | 698.96 |
| jerk MAE (1/s³) | 32,485.56 | 33,344.17 | **32,287.81** |
| variant jitter (velocity RMS) | 6.144 | 5.852 | **4.964** |

The corrected prior is 0.9% worse than GNM no-temporal and 123.8% worse than
Direct in value MAE. Its lower jitter is not accompanied by lower value error.

### P_AB on held-out person B

| Metric | Direct | GNM no-temporal | corrected prior |
| --- | ---: | ---: | ---: |
| value MAE | **0.09994** | 0.17787 | 0.20231 |
| value RMSE | **0.15505** | 0.29923 | 0.30160 |
| velocity MAE (1/s) | **22.510** | 22.931 | 23.140 |
| acceleration MAE (1/s²) | 887.59 | **765.04** | 781.49 |
| jerk MAE (1/s³) | 43,698.47 | **36,186.10** | 37,040.71 |
| variant jitter (velocity RMS) | 7.020 | 6.244 | **5.551** |

The corrected prior is 13.7% worse than GNM no-temporal and 102.4% worse
than Direct in value MAE. The lower output jitter therefore does not establish
an improvement; value and derivative errors do not jointly improve.

## Per-take results

All value and temporal metrics below use exactly the same scored frames for
the three variants within a take.

| Person/take | Variant | frames | value MAE | RMSE | velocity MAE | acceleration MAE | jitter RMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A `c85aaf92` | Direct | 330 | 0.09782 | 0.15271 | 18.939 | 638.10 | 6.443 |
| A `c85aaf92` | GNM no-temporal | 330 | 0.21574 | 0.34268 | 24.938 | 777.57 | 6.949 |
| A `c85aaf92` | corrected prior | 330 | 0.22368 | 0.35040 | 24.049 | 739.09 | 5.711 |
| A `4df068e1` | Direct | 348 | 0.10152 | 0.16145 | 21.092 | 728.25 | 5.848 |
| A `4df068e1` | GNM no-temporal | 348 | 0.22613 | 0.36770 | 20.969 | 665.32 | 4.576 |
| A `4df068e1` | corrected prior | 348 | 0.22261 | 0.35067 | 20.838 | 660.91 | 4.134 |
| B `6158a4df` | Direct | 263 | 0.10037 | 0.15791 | 25.955 | 1,045.24 | 7.552 |
| B `6158a4df` | GNM no-temporal | 263 | 0.21643 | 0.35066 | 25.308 | 823.27 | 5.611 |
| B `6158a4df` | corrected prior | 263 | 0.21933 | 0.32452 | 26.804 | 892.53 | 6.049 |
| B `02374e00` | Direct | 243 | 0.09947 | 0.15190 | 18.781 | 716.86 | 6.394 |
| B `02374e00` | GNM no-temporal | 243 | 0.13613 | 0.23101 | 20.359 | 701.99 | 6.863 |
| B `02374e00` | corrected prior | 243 | 0.18388 | 0.27464 | 19.174 | 661.23 | 4.955 |

## Teacher-detected blink pulses

| Person/take | Direct | GNM no-temporal | corrected prior |
| --- | --- | --- | --- |
| A `c85aaf92` | 14/14, 21.43 ms, attenuation 0.125 | 14/14, 33.33 ms, attenuation 1.062 | **0/14 measurable** |
| A `4df068e1` | 7/7, 14.28 ms, attenuation 0.380 | 7/7, 14.28 ms, attenuation 1.073 | **0/7 measurable** |
| B `6158a4df` | 10/10, 13.33 ms, attenuation 0.260 | 10/10, 13.33 ms, attenuation 1.184 | **0/10 measurable** |
| B `02374e00` | 6/6, 33.33 ms, attenuation 0.068 | 6/6, 11.11 ms, attenuation 0.981 | **0/6 measurable** |

The corrected prior retains **0/37** teacher-detected blink pulses across all
held-out takes. Direct and GNM no-temporal both retain **37/37**. Consequently
the corrected calculation fixes the regression math but does not fix the
prior's blink suppression.

## Cross-person check

P_A evaluated on held-out person B gives corrected-prior value MAE 0.21642,
RMSE 0.34986, velocity MAE 21.146, acceleration MAE 724.09, and jitter 4.940.
The per-take value MAE is 0.23280 (`6158a4df`) and 0.19868 (`02374e00`), with
0/16 blink pulses measurable. This is worse in value error than P_AB on the
same frames (MAE 0.20231), but two people are insufficient for a generalization
claim.

## Verdict

The implementation defect is fixed and the v2 prior is no longer the nearly
constant artifact described by the historical report. Nevertheless, it still
fails the offline adoption criteria: Direct has substantially lower value MAE
and RMSE on both people, and the prior loses every teacher-detected blink.
These measured results must remain the baseline for the next dependent issue;
they do not justify live runtime integration.

## Commands and artifacts

All commands used `--history-len 4`, `--max-gap-micros 100000`, ridge lambda
`0.001`, expected cadence `33367 µs`, gap tolerance `1.5`, and correction bound
`1.0` (the CLI defaults shown in the generated reports).

```powershell
target\release\xtask.exe teacher-fit-prior `
  --trace data/datasets/20260827T150900Z_take_01_1fbe4b9d `
  --trace data/datasets/20260828T053142Z_take_01_6608daa4 `
  --trace data/datasets/20260830T115106Z_take_01_5e3c08c3 `
  --trace data/datasets/20260830T115611Z_take_01_78fa553c `
  --train-take 20260827T150900Z_take_01_1fbe4b9d `
  --train-take 20260828T053142Z_take_01_6608daa4 `
  --train-take 20260830T115106Z_take_01_5e3c08c3 `
  --train-take 20260830T115611Z_take_01_78fa553c `
  --output data/datasets/linear-prior-issue11-train4.json

target\release\xtask.exe teacher-ablation `
  --artifact data/datasets/linear-prior-issue11-train4.json `
  --eval-trace data/datasets/20260828T053203Z_take_01_c85aaf92 `
  --eval-trace data/datasets/20260828T053225Z_take_01_4df068e1 `
  --train-take 20260827T150900Z_take_01_1fbe4b9d `
  --train-take 20260828T053142Z_take_01_6608daa4 `
  --train-take 20260830T115106Z_take_01_5e3c08c3 `
  --train-take 20260830T115611Z_take_01_78fa553c `
  --person-count 2 --output data/datasets/ablation/issue11-train4-eval-a

target\release\xtask.exe teacher-ablation `
  --artifact data/datasets/linear-prior-issue11-train4.json `
  --eval-trace data/datasets/20260830T115657Z_take_01_6158a4df `
  --eval-trace data/datasets/20260830T115716Z_take_01_02374e00 `
  --train-take 20260827T150900Z_take_01_1fbe4b9d `
  --train-take 20260828T053142Z_take_01_6608daa4 `
  --train-take 20260830T115106Z_take_01_5e3c08c3 `
  --train-take 20260830T115611Z_take_01_78fa553c `
  --person-count 2 --output data/datasets/ablation/issue11-train4-eval-b

target\release\xtask.exe teacher-fit-prior `
  --trace data/datasets/20260827T150900Z_take_01_1fbe4b9d `
  --trace data/datasets/20260828T053142Z_take_01_6608daa4 `
  --train-take 20260827T150900Z_take_01_1fbe4b9d `
  --train-take 20260828T053142Z_take_01_6608daa4 `
  --output data/datasets/linear-prior-issue11-train2.json

target\release\xtask.exe teacher-ablation `
  --artifact data/datasets/linear-prior-issue11-train2.json `
  --eval-trace data/datasets/20260830T115657Z_take_01_6158a4df `
  --eval-trace data/datasets/20260830T115716Z_take_01_02374e00 `
  --train-take 20260827T150900Z_take_01_1fbe4b9d `
  --train-take 20260828T053142Z_take_01_6608daa4 `
  --person-count 1 `
  --output data/datasets/ablation/issue11-train2-eval-b-crossperson
```

P_AB artifact SHA-256:
`AC0C7098626CF512D369B5921BC4397030E3700BD57A746BE020AAD7E97209F6`.
P_A artifact SHA-256:
`52E6D2893C536F02732C83D40B5681471711A9A5618145306E881B8AEA6BDE56`.
Generated artifacts and JSON reports are gitignored and can be regenerated
from the commands above.

## Constraints

- People: two, recorded with the same device/room/capture workflow. Results do
  not generalize beyond these people or capture conditions.
- Head-pose comparison, per-frame inference latency, memory/CPU cost, and the
  fixed/adaptive temporal baselines omitted by the trace schema remain **NOT
  VERIFIED**.
- This is an offline evaluation only. No live camera, avatar visual, or macOS
  acceptance is established by these results.
