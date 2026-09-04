# Teacher trace v2 regeneration — 2026-09-04

Issue #14 was validated with the shortest completed capture under `data/raw/`:
`20260830T115106Z_take_01_5e3c08c3` (216 paired frames). No additional capture
data was needed.

## Replay command

```powershell
cargo run -p xtask --release -- teacher-replay `
  --dataset data/raw/tmp_arkit_take/20260830T115106Z_take_01_5e3c08c3 `
  --output data/datasets/20260830T115106Z_take_01_5e3c08c3-trace-v2 `
  --pixel-rotation 180 `
  --fit-tolerance 0.0001
```

Result: 216 solved, 0 no-face, 0 insufficient-coverage, 0 fit-rejected, and
no unpaired records. The output metadata reports schema 2.

## Shape and privacy checks

All 216 trace rows were decoded as JSON and checked:

- every MediaPipe observation contains exactly 478 `(x, y)` pairs;
- every solved GNM state contains 351 non-tongue expression values;
- every solved GNM state contains all seven region-fit records;
- no serialized key contains `tongue`, and there is no landmark `z` field;
- the trace SHA-256 equals `replay-metadata.json.trace_sha256`.

The v2 trace SHA-256 is
`6A6D3197FA013A7238C4F2FAA70B2C21680EEC86057307CCAA55342ACE1B91B7`.

## Size comparison

| Trace | Schema | Bytes | SHA-256 |
|---|---:|---:|---|
| Existing derived trace | 1 | 571,369 | `8677457E93DC59103CB0662FDCD2D395F5129F2702970D8A3AC7E23C8DD76071` |
| Regenerated trace | 2 | 4,086,778 | `6A6D3197FA013A7238C4F2FAA70B2C21680EEC86057307CCAA55342ACE1B91B7` |

The v2 trace is 3,515,409 bytes larger (7.153×) because it retains the
landmark geometry and compact fitted state required for later teacher-model
experiments. Both generated directories remain gitignored.

## Downstream read checks

The same v2 trace was read successfully by:

- `teacher-fit-prior`: 215 causal rows from 216 solved states;
- `teacher-fit-residual`: 216 residual rows from 216 solved states.

Both ablation commands use the same strict `load_trace` boundary. Schema v1 is
rejected with an instruction to regenerate; no migration or v1 fallback is
provided.
