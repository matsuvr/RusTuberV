# ADR-020: Idle motion policy and breathing-writer retirement

Status: Accepted
Date: 2026-08-28
Amended: 2026-08-30 (ADR-021: loss-scoped breathing added inside the tracking-loss micro-motion layer)
Related: Issue #20, Issue #172, Issue #180, ADR-004, ADR-019

## Context

The avatar runtime previously included an always-on hips-only procedural breathing writer introduced by Issue #20. The current upper-body architecture now has explicit ownership for root translation, torso lean, dynamic arm targets, tracking-loss micro-motion, and animation/rest pose preservation. Keeping another unconditional hips translation writer would create a second low-frequency motion authority and complicate composition.

The project therefore needs a deterministic idle contract that does not invent motion when no independent idle source is available.

## Decision

1. Retire the legacy hips-only procedural breathing writer from the runtime schedule and public API.
2. Treat authored/rest/animated pose as the idle authority by default.
3. Keep `IdleMotionProfile` as a typed zero-amplitude contract. `procedural_amplitude_meters` must be finite and exactly zero.
4. Tracking-loss micro-motion remains a separate bounded layer. It is published through the body-position input path and does not write `hips.translation`.
5. No future always-on procedural idle oscillator may be added implicitly. A new idle source requires its own explicit design decision, typed ownership, deterministic tests, and composition policy.

## Writer ownership

| Channel | Owner |
| --- | --- |
| `hips.translation` | authored/animated pose; no runtime idle writer |
| avatar root translation | `apply_direct_body_position` via `BodyTrackingPositionInput` |
| torso lean | `apply_direct_body_position` |
| tracking-loss micro-motion | position-input source layer before the body-position writer |
| arm rotations | arm compositor |
| tracked head/body rotations | direct body-tracking writer |

This keeps Transform ownership unique and prevents double application.

## Validation

Machine-runnable checks cover:

- zero-amplitude profile validation;
- hips translation invariance across 30/60/120 fps-equivalent traces;
- tracking-loss/reacquire without hips drift;
- avatar replacement resetting generation-scoped motion state;
- absence of the retired breathing writer from the schedule;
- managed VRM compatibility runs preserving finite output.

## Consequences

The avatar may appear less animated when completely idle, but the runtime contract is simpler and deterministic. Any future idle animation can be added deliberately as a new source with clear ownership rather than as an unconditional transform writer.
