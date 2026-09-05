# ADR-018: GNM identity calibration lifecycle lives in vtuber-tracking

Status: Superseded by ADR-023 (GNM removed from main; preserved on `archive/gnm`)

## Decision

`vtuber-tracking` depends on `vtuber-gnm` (path dependency) and owns the
neutral identity calibration lifecycle in
`crates/vtuber-tracking/src/calibration/gnm_identity.rs`.

The lifecycle is a small, event-driven store (`GnmIdentityCalibrationStore`)
plus a typed event enum. `Start`, `Complete`, `Cancel`, `Reset`, and
`Invalidate` are the only mutations; there is no residual-, confidence-, or
frame-driven input, so ordinary tracking can never silently re-solve identity.
Recalibration is limited to explicit events by construction.

The store keeps an `Option<GnmIdentityCalibration>` bound to its
`(model_version, mapping_version)`. Accessors gate on
`GnmIdentityCalibration::matches_runtime`: a mismatch invalidates the stored
calibration and returns `None`, so a stale identity can never reach a new
model. Tracking stop/start (`on_tracking_restarted`) deliberately preserves
the published calibration within one session; only `Reset` or an invalidation
drops it.

Handoff to per-frame fitting is read-only: callers receive
`&FixedGnmIdentity` (or the full `&GnmIdentityCalibration`). The numerical
solver itself stays in `vtuber-gnm`; this module only sequences and retains
its output.

## Consequences

- `vtuber-tracking` gains a pure-Rust dependency on `vtuber-gnm`. This does
  not violate AGENTS.md boundaries: `vtuber-tracking` still must not depend on
  Bevy or `bevy_vrm1`, and `vtuber-gnm` depends on neither.
- The existing landmark-based neutral calibration (`calibration/neutral.rs`)
  is unchanged; the GNM identity path is additive until later tasks unify the
  two neutral references.
- Disk persistence, guided capture UI, real-camera acceptance, and online
  identity adaptation remain out of scope (Issue #85).

## Alternatives considered

- Orchestrate from `vtuber-app` behind a trait port: rejected because it would
  push tracking state into the app crate and split the lifecycle across
  ownership domains for no engine-boundary benefit.
- Keep everything inside `vtuber-gnm`: rejected because lifecycle/session
  policy is tracking concern, while `vtuber-gnm` remains a schema/evaluator
  boundary.
