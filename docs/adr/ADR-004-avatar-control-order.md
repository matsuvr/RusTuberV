# ADR-004: Avatar control order and coordinate conversion

Status: Accepted
Date: 2026-08-03
Last amended: 2026-08-29

## Context

The application combines animation, tracked head/body motion, position-aware root motion, dynamic arm solving, gaze, expressions, VRM constraints, transform propagation, and SpringBone simulation. Several stages can affect related bones in the same frame, so writer ownership and schedule order must be explicit.

The application also exposes engine-neutral tracking semantics while `bevy_vrm1` operates in the VRM/glTF model coordinate system. Coordinate conversion must occur at one adapter boundary rather than being scattered across tracking code.

## Decision

### Canonical tracking semantics

`vtuber-core` and `vtuber-tracking` publish unmirrored semantic values:

- yaw > 0: face turns toward image right;
- pitch > 0: chin rises;
- roll > 0: clockwise tilt from the observer's view;
- head translation X > 0: image right;
- Y > 0: up;
- Z > 0: away from the camera;
- gaze horizontal > 0: image right;
- gaze vertical > 0: up.

These crates do not depend on VRM generation, Bevy entities, or model-specific axes.

### Avatar-motion mirror

`AvatarMotionMirror` is adapter-local and defaults to enabled. Mirroring is applied immediately before VRM/model-space application:

```text
yaw              -> -yaw
pitch            ->  pitch
roll             -> -roll
translation X    -> -X
translation Y/Z  -> unchanged
gaze horizontal  -> -horizontal
blink left/right -> swap
```

Inference input, landmarks, calibration, and canonical tracking values are never mirrored.

### Rotation composition

The canonical model-space rotation is intrinsic Y-X-Z:

```text
R = R_y(yaw) * R_x(pitch) * R_z(roll)
```

Each humanoid bone receives a rest-relative delta derived from immutable `RestTransform` / `RestGlobalTransform`. Arbitrary authored rest rotations are preserved; tracked Euler values are never written directly into a bone's local transform.

### Position-aware body motion

Neutral-relative head translation is shaped and split before entering the avatar crate:

```text
HeadTranslationSignal
 -> scale-aware soft cap
 -> VirtualBodyTargets
 -> BodyTrackingPositionInput
 -> apply_direct_body_position
```

The default split keeps lateral X primarily in the upper body, routes most Y/Z motion to body/root compensation, and keeps all parameters in typed profiles. The avatar writer converts the semantic camera-aligned meter frame into model/root space once.

### Dynamic arm pipeline

The default arm authority is a hips-relative virtual-hand pipeline:

```text
ArmPoseSource
 -> hand target generation
 -> analytic two-bone IK
 -> elbow pole/swivel modifier
 -> forearm swing-twist relaxer
 -> shoulder elevation trim
 -> rest-relative arm deltas
 -> arm compositor Transform write
```

The legacy fixed arm pose remains fallback-only. Dynamic modifiers never create additional Transform writers. Fixed wrist/finger corrections are not automatically applied under the dynamic authority; authored/animated hand pose is preserved unless an explicit future tracking source supplies it.

Per-model dynamic parameters are stored under the versioned `DynamicArmProfileOverride` schema and keyed by stable model identity.

### Idle motion

ADR-020 defines the idle policy. There is no always-on hips breathing writer. Authored/rest/animated pose remains the default idle authority. Tracking-loss micro-motion is a bounded input-layer source that feeds the position-aware body path and does not write `hips.translation`.

## Writer ownership

| Channel | Runtime owner |
| --- | --- |
| head / neck / upperChest / chest / spine tracked rotation | direct `bevy_vrm1` body-tracking writer |
| avatar-root translation | `apply_direct_body_position` |
| torso lean | `apply_direct_body_position` |
| upper/lower arm and optional shoulder/finger dynamic deltas | `apply_default_arm_pose` compositor |
| gaze | direct head-relative gaze system |
| expressions | expression accumulator / `ModifyExpressions` path |
| node constraints | `VrmSystemSets::Constraints` |
| SpringBone | `bevy_vrm1` SpringBone systems |
| hips idle translation | no runtime writer |

No two systems may be authoritative for the same Transform channel in the same frame.

## PostUpdate order

The authoritative same-frame order is:

```text
Bevy AnimationSystems
 -> update_body_tracking_position_input
 -> update_body_tracking_pose_input
 -> update_dynamic_arm_targets
 -> direct body-tracking rotation writer
 -> apply_direct_body_position
 -> apply_default_arm_pose
 -> direct head-relative gaze
 -> expression apply
 -> VrmSystemSets::Constraints
 -> transform propagation
 -> SpringBone
```

Input producers may run before the corresponding writer, but modifier stages must not write bone transforms independently.

## Replacement and generation safety

All per-avatar runtime state is generation-scoped. Replacement or unload must invalidate stale control frames, dynamic targets, binding geometry, blend states, and loss-recovery state. A stale generation is a safe no-op and may not write an old entity.

## Validation

Machine-runnable coverage includes:

- arbitrary non-identity rest rotations;
- mirror semantics;
- deterministic 30/60/120 fps-equivalent traces;
- no frame-to-frame accumulation;
- root translation and torso-lean bounds;
- virtual-hand authority and legacy fallback selection;
- elbow/twist/shoulder finite bounds;
- tracking loss/reacquire continuity;
- avatar replacement cleanup;
- schedule ordering and unique writer ownership;
- idle zero-amplitude policy from ADR-020.

## Consequences

Tracking remains engine-neutral, model-space conversion is localized, and each Transform channel has a single runtime authority. Future hand tracking, idle animation, or full-body motion must integrate through the same source/compositor boundaries rather than adding competing writers.
