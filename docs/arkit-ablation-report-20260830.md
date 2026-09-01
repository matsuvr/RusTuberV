# ARKit teacher ablation report — 2026-08-30 (second person, multi-person re-evaluation)

Re-evaluation of the learned causal linear prior with a **second person**,
extending `docs/arkit-ablation-report-20260828.md` (single person). The
previous report's stated constraint was that its conclusion must not be
generalized across people; this round addresses that with 4 new takes from a
different person recorded on 2026-08-30. Machine-generated numbers:
`data/datasets/ablation/*/ablation-report.json` (gitignored; regenerate with
the commands below).

## New data (person B, 2026-08-30)

All takes: iPhone17,3, capture app 0.1.0 (1), iOS 26.6.1, timestamp domain
`monotonic-micros-since-session-start`, `COMPLETED` markers present, manifest
counts match `frames.jsonl` / `rgb.jsonl` line counts exactly (no
contradictions). Replayed with `--pixel-rotation 180 --fit-tolerance 0.0001`.

| Take | paired | solved | no_face | fit_rejected | unpaired RGB (declared) |
| --- | --- | --- | --- | --- | --- |
| `20260830T115106Z_take_01_5e3c08c3` | 216 | 216 | 0 | 0 | 0 |
| `20260830T115611Z_take_01_78fa553c` | 335 | 328 | 4 | 3 | 4 |
| `20260830T115657Z_take_01_6158a4df` | 282 | 266 | 1 | 15 | 24 |
| `20260830T115716Z_take_01_02374e00` | 244 | 244 | 0 | 0 | 186 (RGB tail surplus) |

All 18 fit-rejected frames are rigid-recovery divergences (LM step overflow);
frames 39/207/210 in `78fa553c` and a contiguous block 80–94 in `6158a4df`
(~0.5 s). They enter the trace with `gnm_state: null` and teacher/direct
preserved; the ablation drops them from scoring.

## Code fixes forced by the new data

1. **Kernel bug (would abort every partial-observation frame):**
   `analytic_expression_columns` sized its skinning-derivative buffer by
   *retained* row count and indexed it by retained position, but the skinning
   derivative contract requires one slot per *mapped* point selected by
   mapping index. Any frame with even one invalid landmark failed the cold
   start with a shape error (person A's takes happened to be 100% valid every
   frame). Fixed in `crates/vtuber-gnm/src/reprojection.rs`; regression test
   `expression_jacobian_parity_with_partial_observation` proves analytic/numeric
   Jacobian parity on a strict subset observation.
2. **Replay robustness:** a divergent rigid recovery no longer aborts the
   whole take; it is recorded as a per-frame fit rejection (same semantics as
   an invalid cold-start outcome). Manifest/timestamp contradictions keep
   aborting fail-closed. All 18 rejections above are this category.
3. **`teacher-ablation --person-count <n>`:** the report's constraint block
   previously hardcoded `person_count: 1`; it now records the actual number of
   people behind the artifact and derives the generalization note.

## Split and experiments (take-disjoint)

Artifacts:

| Artifact | Training takes | Rows | People |
| --- | --- | --- | --- |
| `linear-prior-train2.json` (P_A, existing) | A: `1fbe4b9d`, `6608daa4` | 1,196 | 1 |
| `linear-prior-train4.json` (P_AB, new) | A: `1fbe4b9d`, `6608daa4` + B: `5e3c08c3`, `78fa553c` | 1,734 | 2 |

Eval sets: **E_A** = A held-out (`c85aaf92`, `4df068e1`; 678 usable), **E_B** =
B held-out (`6158a4df`, `02374e00`; 506 usable of 526 frames). Person B's
training takes are never in E_B.

| Run | Artifact | Eval | Question |
| --- | --- | --- | --- |
| baseline (08-28) | P_A | E_A | single-person reference |
| R1 | P_AB | E_A | does multi-person training help on the *same* eval frames? |
| R2 | P_AB | E_B | mirror protocol on the new person's held-out takes |
| R3 | P_A | E_B | cross-person transfer (train A → evaluate unseen person B) |

## Results

### R1 — P_AB vs P_A on E_A (678 usable, identical frames to the 08-28 report)

| Metric | direct | gnm-no-temporal | learned-prior P_AB | learned-prior P_A (08-28) |
| --- | --- | --- | --- | --- |
| value MAE | **0.0978** | 0.2168 | 0.2169 | 0.2183 |
| value RMSE | **0.1557** | 0.3523 | 0.3511 | 0.3527 |
| velocity MAE (1/s) | 20.05 | 22.90 | **15.91** | 15.91 |
| acceleration MAE (1/s²) | 684.5 | 720.1 | **474.8** | 474.8 |
| variant jitter (velocity RMS) | 6.14 | 5.85 | 0.000073 | 0.0015 |

Adding a second person (+45% rows) moved value MAE only 0.2183 → 0.2169 and
still does not beat the no-prior GNM projection (0.2168). Derivative-metric
"wins" continue to be constant-collapse artifacts.

### R2 — P_AB on E_B (506 usable)

| Metric | direct | gnm-no-temporal | learned-prior P_AB |
| --- | --- | --- | --- |
| value MAE | **0.0980** | 0.1744 | 0.1875 |
| value RMSE | **0.1536** | **0.2963** | 0.3212 |
| velocity MAE (1/s) | 22.51 | 22.93 | **13.85** |
| acceleration MAE (1/s²) | 887.6 | 765.0 | **426.8** |
| variant jitter (velocity RMS) | 7.02 | 6.24 | 0.00077 |

The learned prior **degrades** value error vs no-prior GNM on person B's
held-out takes (+7.5% MAE, +8.4% RMSE). Note the GNM no-temporal baseline
itself is noticeably better on person B (0.1744) than on person A (0.2168) —
per-person variance in the deterministic baseline is larger than anything the
prior contributes.

Event timing (teacher-detected blink pulses, per take):

- direct: `6158a4df` 10/10 measured — attenuation 0.260, timing error 13.3 ms;
  `02374e00` 6/6 measured — attenuation 0.068, timing error 33.3 ms.
- gnm-no-temporal: 10/10 and 6/6 measured, attenuation 1.184 / 0.981 (peak
  overshoot), timing error 13.3 / 11.1 ms.
- learned-prior: **0/16 measurable** — every blink pulse flattened.

### R3 — cross-person transfer (P_A evaluated on E_B, 506 usable)

learned-prior value MAE **0.1876** (P_AB on the same frames: 0.1875), RMSE
0.3224 vs 0.3212, jitter 0.00196 vs 0.00077 — statistically identical to the
two-person prior. Training on the eval person's own recorded takes (P_AB)
provides **zero** benefit over a prior fit entirely on a different person.

## Reading

1. **The prior is a mean predictor, not a dynamics model.** Its output is
   insensitive to both training volume (1,196 → 1,734 rows) and to whether the
   training data includes the eval person (R3 vs R2). A causal linear model
   with these features regresses to the per-channel mean regardless.
2. **Multi-person data does not rescue it.** R1 shows the same wash on value
   error as the single-person round, and R2 shows it actively hurts on the new
   person's held-out takes while suppressing all 16 blink pulses.
3. **Direct MediaPipe remains the best teacher-aligned baseline** on value
   error for both people (MAE ≈ 0.098 on both), with measured blink pulses and
   13–33 ms timing error. Any temporal prior must beat those reference points.
4. The GNM no-temporal baseline varies by person (0.217 on A vs 0.174 on B);
   single-person conclusions about the *baseline* should also be read with
   care.

## Verdict (successor of the 08-28 adoption implication)

The learned causal linear prior is **not adopted in its present form** — now
confirmed on two people with multi-person training data and cross-person
transfer. Re-evaluation requires a model class that actually conditions on
state (e.g. non-linear/identity-aware dynamics), not more rows of the same
features. The ablation pipeline, replay robustness fixes, and the two-person
dataset are in place for that.

## Constraints

- People: **2**, same device (iPhone 16 Pro-class), same room, same capture
  app, three recording days. Results do not generalize beyond these people or
  capture conditions.
- head pose comparison, per-frame inference latency, memory/CPU cost, and
  fixed/adaptive temporal baselines from GNM #57.x remain **NOT VERIFIED**
  (the trace schema does not store them).

## Commands

```powershell
cargo run -p xtask --release -- teacher-fit-prior `
  --trace data/datasets/20260827T150900Z_take_01_1fbe4b9d `
  --trace data/datasets/20260828T053142Z_take_01_6608daa4 `
  --trace data/datasets/20260830T115106Z_take_01_5e3c08c3 `
  --trace data/datasets/20260830T115611Z_take_01_78fa553c `
  --train-take 20260827T150900Z_take_01_1fbe4b9d `
  --train-take 20260828T053142Z_take_01_6608daa4 `
  --train-take 20260830T115106Z_take_01_5e3c08c3 `
  --train-take 20260830T115611Z_take_01_78fa553c `
  --output data/datasets/linear-prior-train4.json

cargo run -p xtask --release -- teacher-ablation `
  --artifact data/datasets/linear-prior-train4.json `
  --eval-trace data/datasets/20260828T053203Z_take_01_c85aaf92 `
  --eval-trace data/datasets/20260828T053225Z_take_01_4df068e1 `
  --train-take 20260827T150900Z_take_01_1fbe4b9d `
  --train-take 20260828T053142Z_take_01_6608daa4 `
  --train-take 20260830T115106Z_take_01_5e3c08c3 `
  --train-take 20260830T115611Z_take_01_78fa553c `
  --person-count 2 --output data/datasets/ablation/train4-eval-a

cargo run -p xtask --release -- teacher-ablation `
  --artifact data/datasets/linear-prior-train4.json `
  --eval-trace data/datasets/20260830T115657Z_take_01_6158a4df `
  --eval-trace data/datasets/20260830T115716Z_take_01_02374e00 `
  --train-take 20260827T150900Z_take_01_1fbe4b9d `
  --train-take 20260828T053142Z_take_01_6608daa4 `
  --train-take 20260830T115106Z_take_01_5e3c08c3 `
  --train-take 20260830T115611Z_take_01_78fa553c `
  --person-count 2 --output data/datasets/ablation/train4-eval-b

cargo run -p xtask --release -- teacher-ablation `
  --artifact data/datasets/linear-prior-train2.json `
  --eval-trace data/datasets/20260830T115657Z_take_01_6158a4df `
  --eval-trace data/datasets/20260830T115716Z_take_01_02374e00 `
  --train-take 20260827T150900Z_take_01_1fbe4b9d `
  --train-take 20260828T053142Z_take_01_6608daa4 `
  --person-count 1 --output data/datasets/ablation/train2-eval-b-crossperson
```

Artifact SHA-256: P_AB `93C1071DC0735F0EDFA85F863BEDD90AED8FA0B92FDD85374E0A0B8237CE4A5A`,
P_A `F83A86895918D74B1F26AC1F0EAABFA8F7731BCBD76F4AB006EFED4D0124E129`.

Derived-trace SHA-256 (person B): `5e3c08c3`
`8677457E93DC59103CB0662FDCD2D395F5129F2702970D8A3AC7E23C8DD76071`, `78fa553c`
`C5A9F21F6FB82CAC86E516877CCF06F4581B301592BD8D3EF78A27045E91A4D3`,
`6158a4df` `041BBBB8165162CD2F20CA2A4D33878A892899702E1886C07970F5AAAC16925B`,
`02374e00` `8EA30933C91B11489E709CC6EA8514EDDE4CC1C059A33453E30FD797BDE3D36B`.
