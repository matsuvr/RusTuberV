# ADR-008: Single-Active Avatar Replacement Lifecycle

## Status

Accepted (implemented in M1-04-007).

## Context

`DESIGN.md` §16.11 describes a transactional replacement: spawn the new model hidden while the old model remains visible and active, wait for the new model to initialize and bind, then atomically switch the tracking target, despawn the old root, and finally make the new model visible.

M1-04-007 explicitly requires a sequential replacement: move the old avatar to `Unloading`, despawn it, and only then activate the new avatar. It also requires coalescing rapid replacement requests to the latest one.

## Decision

Implement the sequential lifecycle in `vtuber-avatar`:

1. A `ReplaceAvatarRequest` carries the pre-spawned new root but keeps it as `pending_root` in `AvatarLifecycleState::Unloading`.
2. `apply_avatar_request_events` removes `ActiveAvatar` from the old root and does **not** mark the pending root active yet.
3. `despawn_unloading_avatar` recursively despawns the old active root, then calls `finish_unload()` and only then inserts `ActiveAvatar` on the new root.
4. `handle_load_imported_avatar_requests` coalesces additional replacement requests received while already in `Unloading` by despawning the previously spawned pending root and emitting a new `ReplaceAvatarRequest` for the latest request. Only one pending root exists at any time.
5. If the replacement load or binding fails after the old root has been despawned, the lifecycle moves to `Failed` and the slot remains empty. The old avatar is **not** revived.

## Consequences

- The single-active invariant holds at the ECS level: there is never more than one entity with `ActiveAvatar`.
- During `Unloading`, zero entities carry `ActiveAvatar`; the UI should reflect this via `AvatarLifecycleState::Unloading` rather than the marker count.
- Rapid replacement is deterministic: intermediate pending roots are discarded and the lifecycle always converges to the latest request while still in `Unloading`.
- New-load failure results in an empty slot rather than falling back to the previous avatar. This is a fixed, documented behavior.
- This differs from the transactional hidden-spawn flow in `DESIGN.md` §16.11. The simpler flow is acceptable for the current milestone; a future task may restore transactional semantics if camera framing or zero-downtime switching becomes a requirement.
