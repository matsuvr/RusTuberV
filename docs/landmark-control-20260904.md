# Landmark controls (#18)

## Scope

This implementation keeps the landmark control offline. It uses all 478
MediaPipe x/y pairs in trace order, subtracts the per-frame x/y means, divides
by the RMS centered point distance, and flattens them as
`[x0, y0, x1, y1, ...]`. It does not use z, pixels, a landmark subset, a GNM
mapping, or rotation correction.

The training-only landmark basis is the rank-k left singular subspace of
`X^T R / N`, where `X` is centered normalized landmark x/y and `R` is the same
normalized 51-channel `teacher - Direct` residual used by #16. L uses landmark
latent and Direct histories. HL inserts the landmark latent after the reduced
GNM latent and requires exact frame-sequence and timestamp identity. Both reuse
the normalized ridge kernel and non-tongue residual boundary.

The decoder fit function takes explicit landmark and optional GNM basis
artifacts in addition to the issue's sketched arguments. Rows contain features,
not enough information to recover and verify the required basis content hashes;
the explicit inputs keep artifact provenance real instead of synthesizing it.

## Real-data run

Source raw take:
`data/raw/tmp_arkit_take/20260830T115106Z_take_01_5e3c08c3`

Derived trace:
`data/datasets/20260830T115106Z_take_01_5e3c08c3-trace-v2`

Basis command:

```powershell
cargo run -p xtask --release -- teacher-fit-landmark-basis `
  --trace data/datasets/20260830T115106Z_take_01_5e3c08c3-trace-v2 `
  --train-take 20260830T115106Z_take_01_5e3c08c3 `
  --rank 16 `
  --output data/datasets/issue18-landmark-basis-rank16.json
```

L command:

```powershell
cargo run -p xtask --release -- teacher-fit-landmark-decoder `
  --trace data/datasets/20260830T115106Z_take_01_5e3c08c3-trace-v2 `
  --landmark-basis data/datasets/issue18-landmark-basis-rank16.json `
  --kind landmark-residual `
  --train-take 20260830T115106Z_take_01_5e3c08c3 `
  --history-len 2 --max-gap-micros 100000 --ridge-lambda 0.001 `
  --output data/datasets/issue18-landmark-residual.json
```

HL adds the following arguments to the same decoder command:

```text
--kind gnm-landmark-upper-bound
--gnm-basis data/datasets/issue16-teacher-aligned-rank16.json
--output data/datasets/issue18-gnm-landmark-upper-bound.json
```

## Results

- landmark basis: 216 samples, rank 16, no inactive residual channels
- first eight singular values: `2.894835674619996`,
  `2.007713499912006`, `1.167422472982415`, `0.5042855796799318`,
  `0.35508117787961924`, `0.21159417995929933`,
  `0.14727295445332952`, `0.08261554409873206`
- maximum absolute `P^T P - I` entry: `8.35626690065538e-09`
- L/H shared controls: rank 16, history length 2, ridge lambda `0.001`,
  51-channel teacher-minus-Direct target, and the same normalized ridge kernel
- H feature dimension: 262; L feature dimension: 202; HL feature dimension: 310
- L and HL rows: 215 each
- landmark basis content hash: `7297242360588461037`
- L content hash: `4679365131437468498`
- HL content hash: `3655880082968959339`
- landmark basis: 285,103 bytes, SHA-256
  `E2CAD98B9BE5747DAB6B89589B306DDD163558983495C0F09DBC755C84492B98`
- L artifact: 224,447 bytes, SHA-256
  `B55AABE86FD234847554AE2B02CC5A76C63082DEE0D0733026EFE1445E13DCD5`
- HL artifact: 333,887 bytes, SHA-256
  `5FCF1A1EB71C4E62A53BD842C9829BDD0AF93F50EEB088E47750A1942FEFC082`
- two independent fits produced byte-identical basis, L, and HL artifacts

L is latent-capacity matched to H, but it is not parameter-count matched: H's
GNM-produced joints, pose, objective, and region diagnostics are deliberately
not padded into L. HL is an information upper-bound variant, not a
capacity-matched control.
