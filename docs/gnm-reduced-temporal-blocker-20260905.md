# Issue #21 reduced temporal tuning: no admissible candidate

Date: 2026-09-05

The Issue #20 rank-16 basis and H decoder were evaluated on all four outer-training
takes regenerated from `data/raw/`. Each valid MediaPipe source observation was
solved by `fit_single_frame_reduced_with_temporal`; no-face rows were excluded
from source history and no values were filled. The causal replay used only source
observations whose timestamps were at or before each 60 Hz render timestamp.
Future teacher samples were used only to interpolate evaluation targets.

The explicit prediction horizon was 250,000 microseconds, matching the existing
dynamic-state reuse bound. Trials at 50,000 and 100,000 microseconds failed
closed at measured gaps of 50,965 and 100,974 microseconds respectively; no hold
or horizon clamp was used.

## Fixed nine-candidate result

All candidates passed the missed-blink constraint (H 5 <= D 8) but failed onset,
peak, and attenuation constraints. Direct was identical across candidates:
onset 16.666 ms, peak 16.666 ms, attenuation 0.121874.

| eye preset | lower-face preset | H macro MAE | H missed | H onset ms | H peak ms | H attenuation | admissible |
|---|---|---:|---:|---:|---:|---:|---|
| Responsive | Responsive | 0.117317 | 5 | 16.667 | 33.3335 | 0.451678 | no |
| Responsive | Balanced | 0.117340 | 5 | 16.667 | 33.3335 | 0.451578 | no |
| Responsive | Smooth | 0.117377 | 5 | 16.667 | 33.3335 | 0.451608 | no |
| Balanced | Responsive | 0.117322 | 5 | 16.667 | 33.3335 | 0.451622 | no |
| Balanced | Balanced | 0.117350 | 5 | 16.667 | 33.3335 | 0.451606 | no |
| Balanced | Smooth | 0.117396 | 5 | 16.667 | 33.3335 | 0.451627 | no |
| Smooth | Responsive | 0.117327 | 5 | 16.667 | 33.3335 | 0.451596 | no |
| Smooth | Balanced | 0.117359 | 5 | 16.667 | 33.3335 | 0.451654 | no |
| Smooth | Smooth | 0.117415 | 5 | 16.667 | 33.3335 | 0.451635 | no |

The command therefore returned typed `NoAdmissibleTemporalGains` and emitted no
`ReducedTemporalArtifact`. This is the required no-fallback behavior. Issue #22
must not connect a temporal artifact until this gate passes.

## Input provenance

- basis content hash: `6587812130543453725`
- H decoder content hash: `17107686946285364631`
- model SHA-256: `1DFF6A319C2FA28377D7669C30AA533CC0799B45E6049AF18E709B0CB8F122DB`
- mapping revision: `1`
- training takes:
  - `20260827T150900Z_take_01_1fbe4b9d`
  - `20260828T053142Z_take_01_6608daa4`
  - `20260830T115106Z_take_01_5e3c08c3`
  - `20260830T115611Z_take_01_78fa553c`

## Data needed to clear the gate

The limiting evidence is rapid blink shape at 60 Hz: H peaks one render frame
later than Direct and attenuates substantially more. Add synchronized raw
camera + iOS ARKit teacher takes for both existing participants, with at least
10 isolated left blinks, 10 isolated right blinks, and 10 bilateral blinks per
participant. Space events by at least 0.5 seconds, include neutral lead-in and
release, and keep each capture continuous through the full close/open pulse.
Also include repeated fast jaw-open and lip-pucker/stretch transitions so the
same retraining does not improve blink by sacrificing lower-face response.

After adding those takes under `data/raw/`, regenerate trace-v2, then rerun
Issues #15-#19 training so the rank-16 basis and H decoder see the blink-rich
training evidence before rerunning this fixed nine-candidate command.
