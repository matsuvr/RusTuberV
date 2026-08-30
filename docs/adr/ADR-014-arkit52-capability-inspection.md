# ADR-014: ARKit52 capability inspection at the avatar boundary

Status: Accepted for Issue #51

## Context

Issue #51 needs to distinguish an ARKit52/Perfect Sync expression name that is
present in VRM metadata from one that has an effective morph target binding.
`ExpressionEntityMap` alone only exposes the expression entity and therefore
cannot distinguish an empty preset from a usable expression.

## Decision

`vtuber-core` owns the engine-neutral `ArkitBlendshape` and
`Arkit52Coefficients` contract. The canonical order has exactly 52 channels,
including `TongueOut`; the MediaPipe `_neutral` channel is not part of it.
Known PascalCase and MediaPipe camelCase aliases are explicit and no fuzzy
case matching is accepted.

At the pinned `bevy_vrm1` adapter boundary, each initialized expression entity
publishes a read-only `ExpressionBindingStatus` containing the number of
resolved morph binds. `vtuber-avatar` maps known names into
`PerfectSyncCapabilities`, retaining separate `present_channels` and
`effective_channels` bitsets. Unknown custom names are diagnostics and do not
cause a load failure.

This is a narrow source-derived vendor patch to the already approved pinned
runtime revision `f9593fd78136fb9e0507bcae111e09291ec9b82a`. It does not create a
second ECS runtime, replace expression accumulation, or change expression
application semantics. A synthetic vendor regression test covers both an
empty preset and a resolved morph bind.

## Consequences

- Perfect Sync capability diagnostics are deterministic for VRM 0.x and 1.x.
- Partial and non-Perfect-Sync models remain valid normal avatars.
- Later GNM work can consume `Arkit52Coefficients` without depending on Bevy
  or VRM extension structures.
- Real-model visual/camera validation remains a separate platform gate and is
  not claimed by these unit tests.

## References

- VRM 1.0 expressions: `VRMC_vrm.expressions.preset` and morph target binds.
- `bevy_vrm1` pinned revision recorded in `Cargo.toml` and `Cargo.lock`.
