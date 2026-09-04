# Issue #21 reduced temporal tuning: no admissible candidate

Date: 2026-09-05

The Issue #20 rank-16 basis and H decoder were evaluated in four leave-one-take-out
folds over the outer-training takes regenerated from `data/raw/`. Each fixed gain
candidate was replayed independently on each held-out take. H value MAE was
macro-averaged with equal take weight. Blink metrics were computed over the pooled
out-of-fold predictions because an individual take can contain no matched event;
the per-fold report preserves those unavailable values as `null`.

Each valid MediaPipe source observation was solved by
`fit_single_frame_reduced_with_temporal`; no-face rows were excluded from source
history and no values were filled. The causal replay used only source observations
whose timestamps were at or before each 60 Hz render timestamp. Future teacher
samples were used only to interpolate evaluation targets.

The explicit prediction horizon was 250,000 microseconds, matching the existing
dynamic-state reuse bound. Trials at 50,000 and 100,000 microseconds failed
closed at measured gaps of 50,965 and 100,974 microseconds respectively; no hold
or horizon clamp was used.

## Fixed nine-candidate result

All candidates failed the missed-blink constraint (H 13 > D 8). H produced no
matched event from which onset, peak, or attenuation could be measured, so those
three mandatory metrics are unavailable rather than filled with a default. Direct
was identical across candidates: onset 16.666 ms, peak 16.666 ms, and attenuation
0.121874.

| eye preset | lower-face preset | LOTO H macro MAE | H missed | H onset ms | H peak ms | H attenuation | admissible |
|---|---|---:|---:|---:|---:|---:|---|
| Responsive | Responsive | 0.614480992 | 13 | unavailable | unavailable | unavailable | no |
| Responsive | Balanced | 0.614481190 | 13 | unavailable | unavailable | unavailable | no |
| Responsive | Smooth | 0.614481225 | 13 | unavailable | unavailable | unavailable | no |
| Balanced | Responsive | 0.614480869 | 13 | unavailable | unavailable | unavailable | no |
| Balanced | Balanced | 0.614481040 | 13 | unavailable | unavailable | unavailable | no |
| Balanced | Smooth | 0.614481019 | 13 | unavailable | unavailable | unavailable | no |
| Smooth | Responsive | 0.614480563 | 13 | unavailable | unavailable | unavailable | no |
| Smooth | Balanced | 0.614480620 | 13 | unavailable | unavailable | unavailable | no |
| Smooth | Smooth | 0.614480477 | 13 | unavailable | unavailable | unavailable | no |

The Responsive/Responsive fold evidence illustrates the coverage gap. The other
gain pairs have the same event counts and differ only slightly in value MAE.

| held-out take | H macro MAE | H missed | D missed | H timing/attenuation |
|---|---:|---:|---:|---|
| `20260827T150900Z_take_01_1fbe4b9d` | 0.599571976 | 8 | 3 | unavailable |
| `20260828T053142Z_take_01_6608daa4` | 0.617057350 | 2 | 2 | unavailable |
| `20260830T115106Z_take_01_5e3c08c3` | 0.604125209 | 2 | 2 | unavailable |
| `20260830T115611Z_take_01_78fa553c` | 0.637169434 | 1 | 1 | unavailable |

The command wrote all nine aggregate and per-fold candidate records, then returned
typed `NoAdmissibleTemporalGains` and emitted no `ReducedTemporalArtifact`. This is
the required no-fallback behavior. Issue #22 must not connect a temporal artifact
until this gate passes.

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

The limiting evidence is blink detection coverage at 60 Hz: H missed 13 teacher
events and produced no matched event from which timing or attenuation could be
measured. Add synchronized raw camera + iOS ARKit teacher takes for both existing
participants, with at least 10 isolated left blinks, 10 isolated right blinks,
and 10 bilateral blinks per participant. Split them across at least two
independent takes per participant, with several complete events in every take.
Space events by at least 0.5 seconds, include neutral lead-in and release, and
keep each capture continuous through the full close/open pulse. Also include
repeated fast jaw-open and lip-pucker/stretch transitions so the same retraining
does not improve blink by sacrificing lower-face response.

After adding those takes under `data/raw/`, regenerate trace-v2, then rerun
Issues #15-#19 training so the rank-16 basis and H decoder see the blink-rich
training evidence before rerunning this fixed nine-candidate command.
