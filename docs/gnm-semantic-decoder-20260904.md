# Reduced-GNM semantic decoders (#17)

## Contracts

Both decoders use the same teacher-aligned rank-16 basis and the existing
normalized multi-output ridge kernel.

- G1 (`gnm-only`) target: same-frame ARKit teacher non-tongue 51 channels.
- H (`hybrid-residual`) target: same-frame
  `ARKit teacher - MediaPipe Direct` non-tongue 51 channels.
- G1 final output clamps its raw prediction to `[0, 1]` and restores
  TongueOut as zero.
- H keeps signed raw residuals and uses the existing final-boundary Direct plus
  residual clamp.

Newest-first history slots contain only reduced non-tongue expression, four
joint axis-angle rotations, rigid yaw/pitch/roll, objective, seven ordered
region `(weighted_rms, valid_points / 478)` pairs, and (for H only) Direct 51.
The tail contains reduced-expression velocity, optional Direct velocity, and
actual `dt_seconds`. G0, camera parameters, landmark coordinates, teacher head
transform, future frames, and tongue data are absent.

## Real-data fit

Training take:
`20260830T115106Z_take_01_5e3c08c3`

Common settings:

- rows: 215
- aligned basis rank: 16
- joint count: 4
- history length: 2
- maximum gap: 100,000 microseconds
- ridge lambda: `0.001`

G1 result:

- feature dimension: 109
- content hash: `9189824725422538684`
- artifact size: 115,765 bytes
- SHA-256: `ABED9FE2C6A2D13C150FF4FE47295F38C52858A2458764682EC25504C2ED8A12`

H result:

- feature dimension: 262
- content hash: `10189996436703496466`
- artifact size: 282,073 bytes
- SHA-256: `5550F16F349A542A680D3463A98710F1F26BCDC7F06B9351EE71A24878F6D404`

Both fits were run twice and produced byte-identical artifacts.

```powershell
cargo run -p xtask --release -- teacher-fit-gnm-decoder `
  --kind gnm-only `
  --basis data/datasets/issue16-teacher-aligned-rank16.json `
  --trace data/datasets/20260830T115106Z_take_01_5e3c08c3-trace-v2 `
  --train-take 20260830T115106Z_take_01_5e3c08c3 `
  --history-len 2 --max-gap-micros 100000 --ridge-lambda 0.001 `
  --output data/datasets/issue17-gnm-only.json

cargo run -p xtask --release -- teacher-fit-gnm-decoder `
  --kind hybrid-residual `
  --basis data/datasets/issue16-teacher-aligned-rank16.json `
  --trace data/datasets/20260830T115106Z_take_01_5e3c08c3-trace-v2 `
  --train-take 20260830T115106Z_take_01_5e3c08c3 `
  --history-len 2 --max-gap-micros 100000 --ridge-lambda 0.001 `
  --output data/datasets/issue17-hybrid-residual.json
```
