# Existing-trace residual ablation (Issue #13)

This offline experiment compares the same held-out frames for:

- D: MediaPipe Direct 51 channels
- G0: current hand-projected GNM 51 channels
- H0: `clamp_0_1(Direct_51 + predicted_teacher_residual_51)`

`TongueOut` is excluded from every metric and fixed to zero in H0. No
production runtime or backend selection was changed.

## Fixed configuration

- `history_len=4`
- `max_gap_micros=100000`
- `ridge_lambda=0.001`
- feature order:
  `v1:newest-first-history(direct-51+gnm-projected-51+gnm-residual)+velocity(direct-51+gnm-projected-51)+dt-seconds`
- target: same-frame `teacher_51 - MediaPipeDirect_51`

No hyperparameter was selected from an eval take.

## Held-out results

| Training / evaluation | Frames | D MAE / RMSE | G0 MAE / RMSE | H0 MAE / RMSE | H0 vs D channels | Blink detected D / G0 / H0 |
|---|---:|---:|---:|---:|---:|---:|
| A-only / A | 680 | 0.09961 / 0.15710 | 0.22096 / 0.35571 | 0.10890 / 0.19267 | 23 improved / 28 worsened | 21/21 / 0/21 / 18/21 |
| A+B / A | 680 | 0.09961 / 0.15710 | 0.22096 / 0.35571 | 0.08105 / 0.14227 | 35 improved / 16 worsened | 21/21 / 0/21 / 21/21 |
| A+B / B | 510 | 0.10001 / 0.15517 | 0.17796 / 0.29919 | 0.12723 / 0.21865 | 23 improved / 28 worsened | 15/16 / 0/16 / 16/16 |
| A-only / B cross-person | 510 | 0.10001 / 0.15517 | 0.17796 / 0.29919 | 0.27239 / 0.41482 | 4 improved / 47 worsened | 15/16 / 0/16 / 16/16 |

H0 improves aggregate value error only for A+B training evaluated on A. It
worsens D on the other three splits, especially the cross-person split. The
result therefore does not justify production adoption.

Smoothness also does not support adoption: H0 velocity RMS is 9.37 on A and
13.96 on B for the A+B model, versus D at 6.14 and 7.03. H0 acceleration MAE
is 1308.41 on A and 2184.00 on B, versus D at 683.94 and 888.12. Blink peaks
remain measurable, but H0 mean absolute peak timing error is 125.38 ms on A
and 99.98 ms on B, versus D at 19.04 ms and 20.83 ms.

## A+B model channel result

On A held-out, H0 improves:

`CheekPuff`, `CheekSquintLeft`, `CheekSquintRight`, `EyeBlinkLeft`,
`EyeBlinkRight`, `EyeLookDownLeft`, `EyeLookDownRight`, `EyeLookInLeft`,
`EyeLookInRight`, `EyeLookOutLeft`, `EyeLookOutRight`, `EyeSquintRight`,
`JawForward`, `JawOpen`, `MouthClose`, `MouthDimpleLeft`, `MouthDimpleRight`,
`MouthFrownLeft`, `MouthFrownRight`, `MouthFunnel`, `MouthLeft`,
`MouthLowerDownLeft`, `MouthLowerDownRight`, `MouthPressLeft`,
`MouthPressRight`, `MouthPucker`, `MouthRight`, `MouthRollLower`,
`MouthRollUpper`, `MouthShrugLower`, `MouthShrugUpper`, `MouthStretchLeft`,
`MouthStretchRight`, `NoseSneerLeft`, `NoseSneerRight`.

It worsens:

`BrowDownLeft`, `BrowDownRight`, `BrowInnerUp`, `BrowOuterUpLeft`,
`BrowOuterUpRight`, `EyeLookUpLeft`, `EyeLookUpRight`, `EyeSquintLeft`,
`EyeWideLeft`, `EyeWideRight`, `JawLeft`, `JawRight`, `MouthSmileLeft`,
`MouthSmileRight`, `MouthUpperUpLeft`, `MouthUpperUpRight`.

On B held-out, H0 improves:

`BrowInnerUp`, `BrowOuterUpLeft`, `BrowOuterUpRight`, `CheekSquintLeft`,
`CheekSquintRight`, `EyeLookDownLeft`, `EyeLookDownRight`, `EyeLookInLeft`,
`EyeLookOutLeft`, `EyeLookOutRight`, `MouthDimpleLeft`, `MouthDimpleRight`,
`MouthFunnel`, `MouthLowerDownLeft`, `MouthLowerDownRight`, `MouthRollLower`,
`MouthRollUpper`, `MouthStretchLeft`, `MouthStretchRight`, `MouthUpperUpLeft`,
`MouthUpperUpRight`, `NoseSneerLeft`, `NoseSneerRight`.

It worsens the other 28 channels; the generated JSON contains every
per-channel MAE, RMSE, and CCC value.

## Artifact provenance

- A-only artifact SHA-256:
  `56D6C79A5D63860495B1037C760E06FD6BD5A454F7DE09C0C6D283F32D4BCCFD`
- A-only content hash: `10286364740550850962`
- A+B artifact SHA-256:
  `7B5D4BD88B0E4B9D58E8498E7FB183166818616F801A3BB3C8E0265F0EB53BEC`
- A+B content hash: `206715820215849832`

Machine-readable reports are gitignored under
`data/datasets/ablation/issue13-*` and can be regenerated with
`teacher-residual-ablation --artifact ... --eval-trace ... --train-take ...
--person-count ... --output ...` using the take IDs listed in the table and
the Issue #12 reproduction command.
