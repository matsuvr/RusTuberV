# Observable GNM basis fit — 2026-09-04

Issue #15 builds a geometric observability basis without using ARKit teacher
coefficients. For each trace-v2 frame it evaluates the analytic non-tongue
projection Jacobian and accumulates, in fixed take/frame order,

```text
J_t = d projected_xy / d non_tongue_expression
G   = sum_t J_t^T W_t J_t
```

`J_t` has `2 * valid_points` rows and exactly 351 columns. `W_t` duplicates
the dense mapping base weight for each point's x and y rows. Huber weights,
teacher errors, future frames, tongue indices `350..382`, and dense vertices
are not inputs to the artifact.

## Analytic parity

The analytic columns reuse the existing sparse expression basis, skinning
derivative, and perspective derivative. A unit test compares two active
non-tongue columns with central differences; the maximum relative error is
`0.0072029582`, below the fixed `0.01` tolerance. A separate test verifies the
packed Gram is finite, symmetric by construction, and positive semidefinite.

## Real trace command

```powershell
cargo run -p xtask --release -- teacher-fit-observable-basis `
  --trace data/datasets/20260830T115106Z_take_01_5e3c08c3-trace-v2 `
  --train-take 20260830T115106Z_take_01_5e3c08c3 `
  --rank 32 `
  --output data/datasets/issue15-observable-rank32.json
```

The only training take contributed all 216 solved frames. The resulting rank
32 artifact reports:

- retained energy: `0.983849456`;
- top eigenvalues: `6.5074290701`, `2.0398429554`, `1.5624879171`,
  `1.3280136926`, `0.2871593547`, `0.2666775988`, `0.2191463813`,
  `0.1600710927`;
- content hash: `5851582631488564044`;
- serialized size: 208,503 bytes;
- file SHA-256: `2EA87B3BB705AED0C6AA91242DF5FC6A2B36253F9BA32EB54E29706FF7047CE0`.

Two independent CLI runs produced byte-identical JSON and the same hashes.
The generated artifacts remain under gitignored `data/datasets/`.
