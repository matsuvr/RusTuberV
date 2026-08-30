# Q2-06 Head-relative Gaze Acceptance

Date: 2026-08-13
Repair base: `49de1ee9391812bb9a9aed29f27772904619bbbb`

## Automated repair scope

The review repair covers backend-specific range-map units, runtime-rate
auto-neutral aggregation, blink/low-confidence neutral gaze exclusion, VRM
zero-range behavior, quaternion sign equivalence, macOS test compilation, and
the real plugin schedule graph.

## Hardware visual gate

The following checks require a face-visible C922 run with an approved VRM 1.0
model. They were intentionally not run while this repair branch was prepared.

| Check | Status |
| --- | --- |
| Keep gaze centered while turning the head left and right | PENDING |
| Hold the head still and move only the eyes left and right | PENDING |
| Turn the head while counter-rotating the eyes toward the camera | PENDING |
| Return from side gaze to center without residual offset | PENDING |
| Blink without an eye-gaze jump | PENDING |
| Hide and reacquire the face without retained eye rotation | PENDING |

Windows result: `PENDING` (`NOT RUN` for this repair).
macOS result: `NOT RUN`.

Earlier M1 camera and gaze evidence predates this head-relative gaze repair and
is not treated as acceptance evidence for these six checks.
