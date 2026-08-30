# ADR-015: GNM Head v3 Rust model boundary

Status: Accepted for Issue #50 child #52

## Decision

Add `crates/vtuber-gnm` as the engine-neutral Rust boundary for the official
GNM Head v3 data contract. The crate validates the required NPZ arrays, rejects
unsupported versions/variants, non-finite values, malformed shapes and invalid
indices, and evaluates only the requested sparse landmarks into a reusable
output buffer. It does not render dense meshes, depend on Bevy, or decide how
GNM state is retargeted to ARKit52.

The official sparse 68-point landmark table is embedded as source data. The
model NPZ is checked in at `assets/models/gnm_head.npz` so the evaluator has a
deterministic, repository-local artifact for numeric parity tests. The model is
loaded once at startup or model selection and can then be shared immutably by
callers through `&GnmModel` or `Arc<GnmModel>`. The loaded representation retains
the validated dense model arrays required by the evaluator; sparse memory and
allocation claims apply to each evaluation frame, not to model loading.

Only `GnmVersion { major: 3, minor: 0 }` is supported. Unknown major or minor
versions, including `3.1`, return `GnmModelError::UnsupportedVersion` rather
than being treated as probably compatible.

## Provenance and licensing

- Source repository: [google/GNM](https://github.com/google/GNM)
- Source revision: `f76519f4c0340e5333146c0a8f011c56879ae5e3`
- Schema source: `gnm/shape/gnm_data_schema.py`
- Model source path: `gnm/shape/data/versions/v3_0/gnm_head.npz`
- Sparse source path: `gnm/shape/data/landmarks/head_sparse_68.txt`
- Upstream license: Apache-2.0
- Sparse table size: `2,293` bytes in the pinned upstream LF representation
- Sparse table SHA-256:
  `8B4B759042CAE8B67062794306DAE9D60FC7BA11DDAD60461BA3E2BFAAEAC222`
- Schema source SHA-256 / size:
  `47BA3A208462EDA15FF190EFBCD1C103BB9DA4C6104859AD51B5F5DBFE5DC064` /
  `1,187` bytes
- Forward semantics source (`gnm/shape/gnm_common.py`) SHA-256 / size:
  `57DDD2D98B6D7D5C718B2B7767A2D167A7EFC5482FA1A617A363C8047D6B3B82` /
  `17,800` bytes
- Model size: `53,305,389` bytes
- GNM model NPZ SHA-256:
  `1DFF6A319C2FA28377D7669C30AA533CC0799B45E6049AF18E709B0CB8F122DB`
- Model source URL and redistribution terms: fixed in
  `assets/models/manifest.toml`; redistribution is permitted under the
  upstream Apache-2.0 terms with the upstream license and this notice retained.
- Reference fixture:
  `crates/vtuber-gnm/tests/fixtures/official_gnm_head_v3_sparse.txt` was
  generated offline from the pinned official `gnm_common.py` implementation.
  It records all-zero neutral, identity plus joint pose, lower-face
  expression, and eye expression parameters and expected sparse values.

The evaluator applies identity and expression bases only to the unique vertex
subset referenced by the sparse landmarks. It follows the official
`joint_positions_bind_pose`, `joint_transforms_world`, and
`linear_blend_skinning` semantics, including identity-adjusted joint positions
for child local translations and skinning offsets. It never materializes the
dense vertex array on the per-frame path; reusable output scratch is retained
by `GnmSparseVertices`.

The caller supplies the archive path to `load_gnm_head_v3`; missing or changed
arrays fail before evaluation.

Malformed NPY headers, header-length arithmetic, dtype payloads, item-count
payload lengths, and byte ranges are checked before indexing. External archive
corruption therefore returns `GnmModelError::Npy`, `Shape`, or another typed
model error instead of panicking.

The opt-in local measurement helper is
`crates/vtuber-gnm/examples/measure_sparse.rs`:

```text
cargo run -p vtuber-gnm --example measure_sparse -- --iterations 1000
```

It loads the checked-in model before timing, warms up, reuses one model and one
`GnmSparseVertices` output buffer, and prints total and per-iteration elapsed
time without imposing a machine-dependent acceptance threshold.

## Dependency

The crate uses `zip = 6.0.0` with the `deflate-flate2-zlib-rs` feature to read
NPZ containers without a Python process or a native inference runtime. The
crate remains `forbid(unsafe_code)` and has no Bevy or MediaPipe dependency.
