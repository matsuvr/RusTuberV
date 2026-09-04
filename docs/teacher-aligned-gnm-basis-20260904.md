# Teacher-aligned observable GNM basis (#16)

## Scope

The fit uses only trace-v2 rows containing an exact-frame ARKit teacher,
MediaPipe Direct output, and raw non-tongue GNM expression. Its target is the
51-channel `teacher - Direct` residual. The hand-designed GNM projection,
TongueOut, and the 32 GNM tongue coordinates are not inputs.

For the observable basis `O`, each training expression is projected as
`z = O^T phi`. Training-only means center `z`; training-only residual means and
population standard deviations center and normalize the target. Channels with
standard deviation at most `1.0e-6` are set to zero and recorded as inactive.
The implementation computes `C = Z^T R / N`, takes its left singular vectors,
and emits `B = O U_k` with unit-length, canonical-sign columns.

## Real-data run

Source raw take:
`data/raw/tmp_arkit_take/20260830T115106Z_take_01_5e3c08c3`

Derived trace:
`data/datasets/20260830T115106Z_take_01_5e3c08c3-trace-v2`

Command:

```powershell
cargo run -p xtask --release -- teacher-fit-aligned-basis `
  --observable-basis data/datasets/issue15-observable-rank32-current.json `
  --trace data/datasets/20260830T115106Z_take_01_5e3c08c3-trace-v2 `
  --train-take 20260830T115106Z_take_01_5e3c08c3 `
  --rank 16 `
  --output data/datasets/issue16-teacher-aligned-rank16.json
```

Result:

- training samples: 216
- source rank: 32
- target rank: 16
- inactive residual channels: none
- first eight singular values: `3.858451541365308`,
  `3.3110220691094256`, `1.6734816129084231`, `1.3016162617640872`,
  `0.6316879606705167`, `0.5263860832541898`, `0.4151111360329484`,
  `0.2971916454564296`
- maximum absolute `B^T B - I` entry: `3.4342776622509064e-08`
- content hash: `13702246362088035692`
- file SHA-256: `AA59272CB4F2E7027F3E7B63E841018FD0722931A5CBF03A18284D1C4570768F`
- file size: 104,292 bytes
- two independent fits produced byte-identical artifacts

The observable-basis content hash was made stable across JSON decoding by
canonicalizing signed zero and hashing reported `f64` summary values at `f32`
precision. Their serialized values and the fitted basis are not reduced.
