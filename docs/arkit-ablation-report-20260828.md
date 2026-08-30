# ARKit teacher ablation report — 2026-08-28 (GNM #68.5 / Issue #112)

Held-out ablation of the learned causal linear prior against the no-prior
baselines, scored against ARKit teacher coefficients on the same timeline.
Machine-generated numbers: `data/datasets/ablation/ablation-report.json`
(gitignored; regenerate with the commands below).

## Data and split (take-disjoint)

| Split | Take | Frames |
| --- | --- | --- |
| train | `20260827T150900Z_take_01_1fbe4b9d` | 939 |
| train | `20260828T053142Z_take_01_6608daa4` | 261 (1 manifest-declared unpaired RGB excluded) |
| validation | `20260828T053203Z_take_01_c85aaf92` | 331 |
| test | `20260828T053225Z_take_01_4df068e1` | 349 |

Evaluated: 680 held-out frames, 678 usable (teacher + direct + GNM all
present). Every take replayed with `--pixel-rotation 180
--fit-tolerance 0.0001`; 100% solved, 0 no-face.

## Variants (all causal, scored at frame *t* against teacher@*t*)

- `direct` — MediaPipe blendshape → ARKit52 direct observation at *t*.
- `gnm-no-temporal` — deterministic cold-start GNM projection at *t*.
- `learned-prior` — linear AR prior prediction for *t* made from causal
  history features of *t−1* only, through `PriorRuntime`
  (correction bound 1.0, output clamped to [0,1] for scoring).

## Results (test + validation, 678 frames)

| Metric | direct | gnm-no-temporal | learned-prior |
| --- | --- | --- | --- |
| value MAE | **0.0978** | 0.2168 | 0.2183 |
| value RMSE | **0.1557** | 0.3523 | 0.3527 |
| velocity MAE (1/s) | 20.05 | 22.90 | **15.91** |
| acceleration MAE (1/s²) | 684.5 | 720.1 | **474.8** |
| jerk MAE (1/s³) | 32,492 | 33,351 | **21,225** |
| variant jitter (velocity RMS) | 6.14 | 5.85 | 0.0015 |

Event timing (validation take, teacher-detected events):

- direct: 3 steps — onset delay 16.3 ms, rise (10–90%) 73.9 ms; 14 blink
  pulses — peak attenuation 0.125, peak timing error 21.4 ms.
- learned-prior: 0 measurable events out of 17 — the prediction is a
  near-constant series (variant jitter 0.0015), so pulses are entirely
  flattened.

## Reading

1. The learned prior does **not** improve value error (MAE 0.218 vs 0.217
   no-prior) — with 1,196 training rows from one person it collapses to a
   near-mean predictor.
2. It does **not** act as a usable dynamic model yet: the near-constant
   output zeroes its own jitter and flatters derivative error metrics while
   completely suppressing blink pulses (17/17 unmeasurable).
3. Direct MediaPipe remains the strongest teacher-aligned baseline on value
   error; its blink peak attenuation (0.125) and timing error (21 ms) are
   the reference points any temporal prior must beat.

## Constraints (per #112 acceptance)

- Person count: **1**. Same device (iPhone 16 Pro-class), same room, same
  capture app, two recording days. Results do **not** generalize across
  people or capture conditions.
- fixed/adaptive temporal baselines from GNM #57.x, head pose comparison,
  per-frame inference latency, and memory/CPU cost are **NOT VERIFIED** —
  the current trace schema does not store them.

## Commands

```powershell
cargo run -p xtask --release -- teacher-fit-prior `
  --trace data/datasets/20260827T150900Z_take_01_1fbe4b9d `
  --trace data/datasets/20260828T053142Z_take_01_6608daa4 `
  --train-take 20260827T150900Z_take_01_1fbe4b9d `
  --train-take 20260828T053142Z_take_01_6608daa4 `
  --output data/datasets/linear-prior-train2.json

cargo run -p xtask --release -- teacher-ablation `
  --artifact data/datasets/linear-prior-train2.json `
  --eval-trace data/datasets/20260828T053203Z_take_01_c85aaf92 `
  --eval-trace data/datasets/20260828T053225Z_take_01_4df068e1 `
  --train-take 20260827T150900Z_take_01_1fbe4b9d `
  --train-take 20260828T053142Z_take_01_6608daa4
```

Artifact SHA-256: `F83A86895918D74B1F26AC1F0EAABFA8F7731BCBD76F4AB006EFED4D0124E129`.

## Adoption implication for #113

On the current single-person data the learned prior must be reported as
**not adopted in its present form**: no value-error gain and complete pulse
suppression. The ablation pipeline is now in place; the decision can be
revisited with multi-person, higher-volume training data without re-running
the capture-side machinery.
