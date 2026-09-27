//! Pure adapter from engine-neutral tracking targets to the existing arm IK.
//!
//! This module never writes Transforms. Live integration must route its output
//! through the existing arm compositor, not register another bone writer.

use bevy::prelude::{Quat, Vec3};
use vtuber_core::arm_tracking::{ArmBlendWeight, ArmTrackingTarget, HandFingerPose};

use crate::arm::{
    ArmChainBinding, ArmIkError, ArmIkInput, ArmIkSolution, ArmIkTarget, ArmPoseProfile,
    ArmRestGeometry, FingerJointRestReferences, solve_two_bone_arm,
};
use crate::arm_pipeline::ArmPipelineError;
use crate::arm_pose::{
    ResolvedArmPose, ResolvedBoneDelta, ResolvedFingerJointPose, ResolvedFingerPose,
};

/// Rotation that carries the canonical tracking basis into the solver's
/// model/rest basis while removing the current shoulder-parent motion.
///
/// `B` is the parent's rest rotation, `A` its current rotation, and `V` the
/// fixed rotation from the canonical tracking basis (its +Z points toward the
/// camera) into the model basis. Removing `A` exactly once prevents a torso
/// turn from being applied to the arm twice.
#[must_use]
pub fn tracking_to_rest_rotation(
    parent_rest: Quat,
    parent_current: Quat,
    view_to_model: Quat,
) -> Quat {
    parent_rest * parent_current.inverse() * view_to_model
}

/// Composes a virtual and an observed target by explicit per-channel weights.
///
/// `weights.wrist` moves the hand between the virtual and observed wrist;
/// `weights.pole` moves the bend plane. Poles are blended by direction from the
/// blended wrist so an opposed pair cannot interpolate through the shoulder and
/// flip the elbow; an exactly opposed pair is reported as degenerate rather
/// than silently inverted. Zero weights reproduce `virtual_target` and one
/// weights reproduce `tracked_target`.
pub fn blend_arm_targets(
    virtual_target: ArmIkTarget,
    tracked_target: ArmIkTarget,
    weights: ArmBlendWeight,
) -> Result<ArmIkTarget, ArmIkError> {
    if !weights.wrist.is_finite() || !weights.pole.is_finite() {
        return Err(ArmIkError::NonFiniteInput);
    }
    let wrist_weight = weights.wrist.clamp(0.0, 1.0);
    let pole_weight = weights.pole.clamp(0.0, 1.0);
    // Endpoint wrist weights reproduce their side exactly, so a held
    // observation with a non-finite channel cannot poison the virtual return
    // through `lerp`'s `INFINITY * 0.0`.
    let wrist = if wrist_weight <= f32::EPSILON {
        virtual_target.wrist
    } else if wrist_weight >= 1.0 - f32::EPSILON {
        tracked_target.wrist
    } else {
        virtual_target
            .wrist
            .lerp(tracked_target.wrist, wrist_weight)
    };
    if !wrist.is_finite() {
        return Err(ArmIkError::NonFiniteInput);
    }
    // Endpoint poles reproduce their side exactly without inspecting the
    // other one. A frame-out holds the last observation while its weight
    // eases to zero; requiring the held pole to stay well-conditioned
    // against the virtual wrist would keep reporting degenerate and freeze
    // the compositor on the stale bend instead of returning to initial.
    if pole_weight <= f32::EPSILON {
        if !virtual_target.elbow_pole.is_finite() {
            return Err(ArmIkError::NonFiniteInput);
        }
        return Ok(ArmIkTarget {
            wrist,
            elbow_pole: virtual_target.elbow_pole,
        });
    }
    if pole_weight >= 1.0 - f32::EPSILON {
        if !tracked_target.elbow_pole.is_finite() {
            return Err(ArmIkError::NonFiniteInput);
        }
        return Ok(ArmIkTarget {
            wrist,
            elbow_pole: tracked_target.elbow_pole,
        });
    }
    let elbow_pole = blend_pole(
        virtual_target.elbow_pole,
        tracked_target.elbow_pole,
        wrist,
        pole_weight,
    )?;
    if !elbow_pole.is_finite() {
        return Err(ArmIkError::NonFiniteInput);
    }
    Ok(ArmIkTarget { wrist, elbow_pole })
}

fn blend_pole(
    virtual_pole: Vec3,
    tracked_pole: Vec3,
    origin: Vec3,
    weight: f32,
) -> Result<Vec3, ArmIkError> {
    let virtual_offset = virtual_pole - origin;
    let tracked_offset = tracked_pole - origin;
    let virtual_direction =
        crate::arm::finite_normalized(virtual_offset).ok_or(ArmIkError::DegenerateGeometry)?;
    let tracked_direction =
        crate::arm::finite_normalized(tracked_offset).ok_or(ArmIkError::DegenerateGeometry)?;
    if virtual_direction.dot(tracked_direction) < -1.0 + 1.0e-4 {
        // Interpolating an opposed pair would cross the origin and invert the
        // bend; report it as undefined instead of fabricating a plane.
        return Err(ArmIkError::DegenerateGeometry);
    }
    let blended = virtual_direction * (1.0 - weight) + tracked_direction * weight;
    let direction = crate::arm::finite_normalized(blended).ok_or(ArmIkError::DegenerateGeometry)?;
    let length =
        virtual_offset.length() + (tracked_offset.length() - virtual_offset.length()) * weight;
    Ok(origin + direction * length)
}

/// Converts an observed solve into the compositor's rest-relative pose.
///
/// This is the exact conversion the virtual path uses; it is not
/// reimplemented. `hand_delta` is the hand's share of the palm roll produced by
/// [`align_palm_twist`], when a palm observation applied one; the forearm's
/// share is already inside `solution`. `fingers` is the observed articulation
/// from [`observed_finger_deltas`], and `None` leaves the fingers at the rest
/// pose the shared conversion already produced, because no observation exists.
/// The virtual shoulder-trim/swivel/twist modifiers are not applied.
pub fn resolved_tracked_arm_pose(
    chain: &ArmChainBinding,
    solution: ArmIkSolution,
    hand_delta: Option<Quat>,
    fingers: Option<ResolvedFingerPose>,
) -> Result<ResolvedArmPose, ArmPipelineError> {
    let profile = ArmPoseProfile {
        finger_curl_radians: 0.0,
        ..ArmPoseProfile::default()
    };
    let mut pose = crate::arm_pose::resolved_from_solution(chain, &solution, profile)?
        .ok_or(ArmPipelineError::DegenerateSolvedPose)?;
    pose.hand = hand_delta.map(|delta| crate::arm_pose::ResolvedBoneDelta {
        entity: chain.hand,
        delta,
    });
    if let Some(fingers) = fingers {
        pose.fingers = fingers;
    }
    Ok(pose)
}

/// Turns the observed finger articulation into rest-relative bone deltas.
///
/// The observed values are anatomical joint angles, not rotations, so each one
/// is applied about the axis the model's own rest geometry gives that joint —
/// the same helper the relaxed curl uses. That keeps the thumb on its own axes
/// instead of being forced onto a four-finger bend, and lets each finger carry
/// its own curl instead of one shared amount.
///
/// The thumb's opening is a rotation about the palm normal, computed from the
/// observed ray and the rig's rest thumb ray, so the rest pose stays the
/// neutral: a rig whose thumb rests half open is not snapped shut by a relaxed
/// hand. `weight` scales the whole pose, so the channel's return time eases
/// every finger back to the rest pose at once.
///
/// Returns `None` when the rest geometry cannot place the rotation. Individual
/// bones are skipped rather than fabricated when the model has no such finger
/// or joint.
#[must_use]
pub fn observed_finger_deltas(
    chain: &ArmChainBinding,
    observed: HandFingerPose,
    solution: &ArmIkSolution,
    tracking_to_rest: Quat,
    weight: f32,
) -> Option<ResolvedFingerPose> {
    if !weight.is_finite() || weight <= f32::EPSILON || !tracking_to_rest.is_finite() {
        return None;
    }
    let weight = weight.clamp(0.0, 1.0);
    let rest = &chain.finger_rest;
    let four = [
        (&rest.index, observed.fingers[0]),
        (&rest.middle, observed.fingers[1]),
        (&rest.ring, observed.fingers[2]),
        (&rest.little, observed.fingers[3]),
    ];
    let thumb = ResolvedFingerJointPose {
        metacarpal: thumb_opening_delta(chain, solution, &observed, tracking_to_rest, weight),
        proximal: crate::arm_pose::resolve_finger_joint(
            rest.thumb.proximal,
            rest.thumb.intermediate,
            rest.thumb.metacarpal,
            observed.thumb[0] * weight,
        ),
        // A VRM has no thumb intermediate bone, so the observed interphalangeal
        // angle lands on the distal bone, measured from the proximal one.
        intermediate: None,
        distal: crate::arm_pose::resolve_finger_joint(
            rest.thumb.distal,
            None,
            rest.thumb.intermediate.or(rest.thumb.proximal),
            observed.thumb[1] * weight,
        ),
    };
    let [index, middle, ring, little] = four;
    Some(ResolvedFingerPose {
        thumb,
        index: three_joint_deltas(index.0, index.1, weight),
        middle: three_joint_deltas(middle.0, middle.1, weight),
        ring: three_joint_deltas(ring.0, ring.1, weight),
        little: three_joint_deltas(little.0, little.1, weight),
    })
}

/// Resolves one four-finger chain's three observed joint angles.
fn three_joint_deltas(
    finger: &FingerJointRestReferences,
    curls: [f32; 3],
    weight: f32,
) -> ResolvedFingerJointPose {
    ResolvedFingerJointPose {
        // The four fingers have no metacarpal bone in the VRM humanoid, so the
        // observed knuckle flexion lands on the proximal bone.
        metacarpal: None,
        proximal: crate::arm_pose::resolve_finger_joint(
            finger.proximal,
            finger.intermediate,
            finger.metacarpal,
            curls[0] * weight,
        ),
        intermediate: crate::arm_pose::resolve_finger_joint(
            finger.intermediate,
            finger.distal,
            finger.proximal,
            curls[1] * weight,
        ),
        distal: crate::arm_pose::resolve_finger_joint(
            finger.distal,
            None,
            finger.intermediate,
            curls[2] * weight,
        ),
    }
}

/// Rest-relative rotation that opens the thumb away from the palm.
///
/// Both the observed ray and the rig's rest thumb ray are projected
/// perpendicular to the solved hand's palm normal, so what is compared is the
/// in-plane angle between them. That angle is unaffected by the forearm
/// pronation the palm channel applies separately, so the two channels compose
/// without fighting.
///
/// The rotation is about the palm normal through the thumb's own origin, so no
/// finger tip moves because of it and the following bones' flexions are
/// unaffected. Returns `None` when the rig has no thumb bone to rotate or the
/// two rays are degenerate in the palm plane.
fn thumb_opening_delta(
    chain: &ArmChainBinding,
    solution: &ArmIkSolution,
    observed: &HandFingerPose,
    tracking_to_rest: Quat,
    weight: f32,
) -> Option<ResolvedBoneDelta> {
    let metacarpal = chain.finger_rest.thumb.metacarpal?;
    let observed_ray =
        crate::arm::finite_normalized(tracking_to_rest * Vec3::from(observed.thumb_direction))?;
    let palm_normal = current_palm_normal(chain, solution).unwrap_or(observed_ray);
    let rest_ray =
        crate::arm::finite_normalized(metacarpal.rest.position - chain.rest.wrist.position)?;
    let rest_in_plane =
        crate::arm::finite_normalized(rest_ray - palm_normal * rest_ray.dot(palm_normal))?;
    let observed_in_plane =
        crate::arm::finite_normalized(observed_ray - palm_normal * observed_ray.dot(palm_normal))?;
    let angle = palm_normal
        .dot(rest_in_plane.cross(observed_in_plane))
        .atan2(rest_in_plane.dot(observed_in_plane))
        * weight;
    if !angle.is_finite() || angle.abs() <= 1.0e-6 {
        return None;
    }
    let rest_global = metacarpal.rest.global_rotation;
    let delta = normalized_finite(
        rest_global.inverse() * Quat::from_axis_angle(palm_normal, angle) * rest_global,
    )?;
    Some(ResolvedBoneDelta {
        entity: metacarpal.entity,
        delta,
    })
}

/// The hand's model-space palm normal after the solve and the observed twist.
fn current_palm_normal(chain: &ArmChainBinding, solution: &ArmIkSolution) -> Option<Vec3> {
    let rest_normal = rest_palm_normal(chain)?;
    let rest_hand = chain.rest.wrist.global_rotation;
    let current = hand_global(chain, solution) * (rest_hand.inverse() * rest_normal);
    crate::arm::finite_normalized(current)
}

/// Normalizes a quaternion, or reports that it is not usable.
fn normalized_finite(value: Quat) -> Option<Quat> {
    (value.is_finite() && value.length_squared() > f32::EPSILON).then(|| value.normalize())
}

/// Converts subject-arm units into avatar rest-space positions.
///
/// tracking_to_rest is a UNIT rotation from the canonical tracking basis into
/// the solver's model/rest basis, including removal of the current shoulder
/// parent's motion. If B is the parent's rest rotation, A its current rotation,
/// and V maps the tracking view into the same model frame, use B * inverse(A) * V.
/// Feeding V alone when the torso turns applies torso motion twice.
///
/// Origin is the upper-arm bone, NOT the optional clavicle/shoulder bone.
/// This conversion adds neither the face's translation nor virtual-hand lag.
#[must_use]
pub fn tracked_arm_ik_target(
    rest: ArmRestGeometry,
    target: ArmTrackingTarget,
    tracking_to_rest: Quat,
) -> ArmIkTarget {
    let position = |[x, y, z]: [f32; 3]| {
        rest.upper_arm.position + (tracking_to_rest * Vec3::new(x, y, z)) * rest.total_arm_length
    };
    ArmIkTarget {
        wrist: position(target.wrist),
        elbow_pole: position(target.elbow_pole),
    }
}

/// Solves an observed arm with the existing constant-time two-bone implementation.
///
/// This raw solve deliberately omits virtual-hand swivel, torso lag, shoulder
/// trim, and twist relaxation. Those modifiers need tracked-source semantics
/// before being enabled, otherwise they can move a hand away from its target.
/// The compositor's tracked path adds the shared Stage 3b coronal descent limit
/// after this solve, so the upper arm cannot wrap past the shoulder's range.
/// Missing/occluded observations are handled by tracking, not an implicit fallback.
pub fn solve_tracked_arm(
    rest: ArmRestGeometry,
    target: ArmTrackingTarget,
    tracking_to_rest: Quat,
) -> Result<ArmIkSolution, ArmIkError> {
    let target = tracked_arm_ik_target(rest, target, tracking_to_rest);
    solve_two_bone_arm(ArmIkInput::from_geometry(rest, target))
}

/// Share of the observed palm roll left on the hand bone; the rest goes to the
/// forearm. Both shares turn about the same forearm axis, so they still compose
/// to the exact observed palm plane.
///
/// A VRM has a single forearm bone, so the whole pronation applied at one joint
/// collapses that joint's skin weights: at the hand this is the "candy wrapper"
/// wrist that reads as torn off. Halving the angle at each joint halves the
/// shear each one sees.
const PALM_TWIST_HAND_SHARE: f32 = 0.5;

/// Rolls the forearm and hand so the palm plane matches the observed normal.
///
/// The analytic solve leaves the roll on the shortest arc from rest, so the
/// palm keeps whatever orientation that arc produced. The observed normal is
/// mapped into rest space, the solved hand's own palm normal is derived from
/// the authored index/little rest positions, and the signed angle between the
/// two around the solved forearm's long axis is split between the forearm (the
/// solution is rolled in place) and the hand (the returned local delta).
///
/// The rotation axis runs through the elbow and the wrist, so no bone position
/// moves: the forearm and hand only roll, and the fingers follow the hand.
///
/// `weight` scales the roll; the channel's return time therefore eases both
/// bones back to the default roll when the palm is lost. Smoothness comes from
/// the tracking layer, which already low-passes the observed normal, so there
/// is no per-frame limiter and no carried state.
///
/// Returns `None` (leaving the solved roll) when the rest finger geometry is
/// missing, the observed normal is degenerate around the forearm axis, or a
/// direction cannot be normalized. The forearm share is still written whenever
/// the angle is usable, even if the hand share degenerates.
#[must_use]
pub fn align_palm_twist(
    chain: &ArmChainBinding,
    solution: &mut ArmIkSolution,
    palm_normal: [f32; 3],
    tracking_to_rest: Quat,
    weight: f32,
) -> Option<Quat> {
    if !weight.is_finite() || weight <= f32::EPSILON || !tracking_to_rest.is_finite() {
        return None;
    }
    let weight = weight.clamp(0.0, 1.0);
    let rest_normal = rest_palm_normal(chain)?;
    let axis = crate::arm::finite_normalized(solution.wrist - solution.elbow)?;
    let observed = crate::arm::finite_normalized(tracking_to_rest * Vec3::from(palm_normal))?;
    let rest_hand = chain.rest.wrist.global_rotation;
    let current = crate::arm::finite_normalized(
        hand_global(chain, solution) * (rest_hand.inverse() * rest_normal),
    )?;
    let observed_perp = crate::arm::finite_normalized(observed - axis * observed.dot(axis))?;
    let current_perp = crate::arm::finite_normalized(current - axis * current.dot(axis))?;
    let angle = axis
        .dot(current_perp.cross(observed_perp))
        .atan2(current_perp.dot(observed_perp))
        * weight;
    if !angle.is_finite() || angle.abs() <= 1.0e-6 {
        return None;
    }
    let hand_angle = angle * PALM_TWIST_HAND_SHARE;
    let hand = roll_hand(chain, solution, axis, hand_angle);
    roll_forearm(chain, solution, axis, angle - hand_angle);
    hand
}

/// The hand bone's model-space orientation with its authored rest-relative pose.
fn hand_global(chain: &ArmChainBinding, solution: &ArmIkSolution) -> Quat {
    solution.lower_arm_global_rotation
        * (chain.rest.elbow.global_rotation.inverse() * chain.rest.wrist.global_rotation)
}

/// Builds the hand's local rest-relative roll for a model-space rotation about
/// the forearm axis through the wrist, which leaves the solved positions
/// untouched.
fn roll_hand(
    chain: &ArmChainBinding,
    solution: &ArmIkSolution,
    axis: Vec3,
    angle: f32,
) -> Option<Quat> {
    if angle.abs() <= 1.0e-6 {
        return None;
    }
    let rest_hand = chain.rest.wrist.global_rotation;
    let rest_lower = chain.rest.elbow.global_rotation;
    let hand_relative = rest_lower.inverse() * rest_hand;
    let hand_global = solution.lower_arm_global_rotation * hand_relative;
    let rotation = Quat::from_axis_angle(axis, angle);
    let local = hand_global.inverse() * rotation * hand_global;
    (local.is_finite() && local.length_squared() > f32::EPSILON).then(|| local.normalize())
}

/// Rolls the solved forearm about its own long axis in model space.
///
/// The elbow and wrist lie on that axis, so the roll moves no bone position and
/// the hand follows the forearm rigidly. Rebuilds the lower-arm global rotation,
/// rest-relative delta, and local rotation exactly as [`solve_two_bone_arm`]
/// does, so the shared conversion still applies. Returns whether the roll was
/// written.
fn roll_forearm(
    chain: &ArmChainBinding,
    solution: &mut ArmIkSolution,
    axis: Vec3,
    angle: f32,
) -> bool {
    if !angle.is_finite() || angle.abs() <= 1.0e-6 {
        return false;
    }
    let rest_upper = chain.rest.upper_arm.global_rotation;
    let rest_lower = chain.rest.elbow.global_rotation;
    let upper_model = solution.upper_arm_global_rotation * rest_upper.inverse();
    let lower_global = Quat::from_axis_angle(axis, angle) * solution.lower_arm_global_rotation;
    let lower_local_model = upper_model.inverse() * (lower_global * rest_lower.inverse());
    let Ok(lower_delta) = crate::arm::conjugated_rest_delta(lower_local_model, rest_lower) else {
        return false;
    };
    solution.lower_arm_global_rotation = lower_global.normalize();
    solution.lower_arm_delta = lower_delta;
    solution.lower_arm_local_rotation = chain.rest.elbow.local_rotation * lower_delta;
    true
}

/// Avatar rest-space palm normal from the authored index/little rest positions.
///
/// The index/little cross product relative to the wrist uses the same
/// anatomical landmark order as the observed hand landmarks, so both sides of
/// the comparison agree without a per-side sign. Missing finger geometry yields
/// `None`.
fn rest_palm_normal(chain: &ArmChainBinding) -> Option<Vec3> {
    let index = chain.finger_rest.index.proximal?.rest.position;
    let little = chain.finger_rest.little.proximal?.rest.position;
    let wrist = chain.rest.wrist.position;
    let index = crate::arm::finite_normalized(index - wrist)?;
    let little = crate::arm::finite_normalized(little - wrist)?;
    crate::arm::finite_normalized(index.cross(little))
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )] // tests may panic (AGENTS.md)
    use super::*;
    use crate::arm::RestSpaceBonePose;

    fn bone(position: Vec3) -> RestSpaceBonePose {
        RestSpaceBonePose {
            position,
            global_rotation: Quat::IDENTITY,
            local_rotation: Quat::IDENTITY,
        }
    }

    fn geometry(scale: f32) -> ArmRestGeometry {
        let origin = Vec3::new(0.2, 1.0, 0.0);
        ArmRestGeometry {
            shoulder: None,
            upper_arm: bone(origin),
            elbow: bone(origin + Vec3::X * (0.4 * scale)),
            wrist: bone(origin + Vec3::X * (0.7 * scale)),
            upper_arm_length: 0.4 * scale,
            forearm_length: 0.3 * scale,
            total_arm_length: 0.7 * scale,
        }
    }

    fn target() -> ArmTrackingTarget {
        ArmTrackingTarget {
            wrist: [0.4, -0.3, 0.5],
            elbow_pole: [0.7, -0.5, -0.1],
            palm_normal: None,
            fingers: None,
        }
    }

    fn near(a: Vec3, b: Vec3) {
        assert!((a - b).length() < 1.0e-5, "{a:?} != {b:?}");
    }

    #[test]
    fn avatar_length_and_upper_arm_origin_determine_target() {
        let rest = geometry(2.0);
        let mapped = tracked_arm_ik_target(rest, target(), Quat::IDENTITY);
        near(
            mapped.wrist,
            rest.upper_arm.position + Vec3::new(0.4, -0.3, 0.5) * 1.4,
        );
        near(
            mapped.elbow_pole,
            rest.upper_arm.position + Vec3::new(0.7, -0.5, -0.1) * 1.4,
        );
    }

    #[test]
    fn existing_solver_preserves_bone_lengths_and_reaches_observed_target() {
        let rest = geometry(1.0);
        let solution = solve_tracked_arm(rest, target(), Quat::IDENTITY).unwrap();
        let mapped = tracked_arm_ik_target(rest, target(), Quat::IDENTITY);
        near(solution.wrist, mapped.wrist);
        assert!(((solution.elbow - rest.upper_arm.position).length() - 0.4).abs() < 1.0e-5);
        assert!(((solution.wrist - solution.elbow).length() - 0.3).abs() < 1.0e-5);
    }

    #[test]
    fn local_rotations_reconstruct_the_same_elbow_and_wrist() {
        let rest = geometry(1.0);
        let solution = solve_tracked_arm(rest, target(), Quat::IDENTITY).unwrap();
        let elbow = rest.upper_arm.position + solution.upper_arm_local_rotation * (Vec3::X * 0.4);
        let wrist = elbow
            + (solution.upper_arm_local_rotation * solution.lower_arm_local_rotation)
                * (Vec3::X * 0.3);
        near(elbow, solution.elbow);
        near(wrist, solution.wrist);
    }

    #[test]
    fn current_parent_rotation_is_removed_exactly_once() {
        let rest = geometry(1.0);
        let parent_rest = Quat::from_rotation_z(0.2);
        let parent_current = Quat::from_rotation_y(0.8) * parent_rest;
        let view_to_model = Quat::from_rotation_x(-0.1);
        let tracking_to_rest =
            tracking_to_rest_rotation(parent_rest, parent_current, view_to_model);
        let solution = solve_tracked_arm(rest, target(), tracking_to_rest).unwrap();
        let displayed_offset =
            (parent_current * parent_rest.inverse()) * (solution.wrist - rest.upper_arm.position);
        near(
            displayed_offset,
            view_to_model * Vec3::new(0.4, -0.3, 0.5) * rest.total_arm_length,
        );
    }

    #[test]
    fn unreachable_target_uses_existing_solver_reach_constraint() {
        let rest = geometry(1.0);
        let far = ArmTrackingTarget {
            wrist: [3.0, 0.0, 0.0],
            elbow_pole: [0.5, -0.5, 0.0],
            palm_normal: None,
            fingers: None,
        };
        let solution = solve_tracked_arm(rest, far, Quat::IDENTITY).unwrap();
        assert!(solution.solved_reach < rest.total_arm_length);
        assert!(solution.solved_reach > 0.69);
    }

    fn chain() -> ArmChainBinding {
        let rest = geometry(1.0);
        ArmChainBinding {
            side: crate::arm::ArmSide::Left,
            shoulder: None,
            upper_arm: bevy::prelude::Entity::from_raw_u32(0).unwrap(),
            lower_arm: bevy::prelude::Entity::from_raw_u32(1).unwrap(),
            hand: bevy::prelude::Entity::from_raw_u32(2).unwrap(),
            fingers: crate::arm::FingerReferences::default(),
            finger_rest: crate::arm::FingerRestReferences::default(),
            rest,
            capabilities: crate::arm::ArmChainCapabilities::default(),
        }
    }

    #[test]
    fn blend_arm_targets_reproduces_both_endpoints() {
        let virtual_target = ArmIkTarget {
            wrist: Vec3::new(0.0, 1.0, 0.0),
            elbow_pole: Vec3::new(0.0, 1.0, 0.4),
        };
        let tracked_target = ArmIkTarget {
            wrist: Vec3::new(0.6, 0.2, 0.1),
            elbow_pole: Vec3::new(0.6, 0.2, 0.5),
        };
        let untouched =
            blend_arm_targets(virtual_target, tracked_target, ArmBlendWeight::ZERO).unwrap();
        near(untouched.wrist, virtual_target.wrist);
        near(untouched.elbow_pole, virtual_target.elbow_pole);

        let observed =
            blend_arm_targets(virtual_target, tracked_target, ArmBlendWeight::ONE).unwrap();
        near(observed.wrist, tracked_target.wrist);
        near(observed.elbow_pole, tracked_target.elbow_pole);

        let half = blend_arm_targets(
            virtual_target,
            tracked_target,
            ArmBlendWeight {
                wrist: 0.5,
                pole: 0.5,
                palm: 0.0,
                fingers: 0.0,
            },
        )
        .unwrap();
        assert!(half.wrist.x > 0.0 && half.wrist.x < 0.6);
        assert!(half.elbow_pole.is_finite());
    }

    #[test]
    fn opposed_poles_are_degenerate_not_flipped() {
        let origin = Vec3::new(0.0, 1.0, 0.0);
        let virtual_target = ArmIkTarget {
            wrist: origin,
            elbow_pole: origin + Vec3::Z * 0.4,
        };
        let tracked_target = ArmIkTarget {
            wrist: origin,
            elbow_pole: origin - Vec3::Z * 0.4,
        };
        assert_eq!(
            blend_arm_targets(
                virtual_target,
                tracked_target,
                ArmBlendWeight {
                    wrist: 0.5,
                    pole: 0.5,
                    palm: 0.0,
                    fingers: 0.0,
                },
            ),
            Err(ArmIkError::DegenerateGeometry)
        );
    }

    #[test]
    fn lost_channels_return_to_virtual_without_inspecting_the_held_pole() {
        // Frame-out holds the last observation while its weight eases to
        // zero. The held pole may sit opposite the virtual one; the return
        // must still reproduce the initial pose instead of reporting
        // degenerate and freezing the compositor on the stale bend.
        let origin = Vec3::new(0.0, 1.0, 0.0);
        let virtual_target = ArmIkTarget {
            wrist: origin,
            elbow_pole: origin + Vec3::Z * 0.4,
        };
        let tracked_target = ArmIkTarget {
            wrist: Vec3::new(0.6, 0.2, 0.1),
            elbow_pole: origin - Vec3::Z * 0.4,
        };
        let returned =
            blend_arm_targets(virtual_target, tracked_target, ArmBlendWeight::ZERO).unwrap();
        near(returned.wrist, virtual_target.wrist);
        near(returned.elbow_pole, virtual_target.elbow_pole);

        // A pole held exactly on the blended wrist is degenerate for the
        // direction blend, but weight zero still means the initial pose.
        let degenerate_tracked = ArmIkTarget {
            wrist: Vec3::new(0.6, 0.2, 0.1),
            elbow_pole: origin,
        };
        let returned =
            blend_arm_targets(virtual_target, degenerate_tracked, ArmBlendWeight::ZERO).unwrap();
        near(returned.elbow_pole, virtual_target.elbow_pole);
    }

    #[test]
    fn full_weight_reproduces_the_observation_without_inspecting_virtual() {
        let origin = Vec3::new(0.0, 1.0, 0.0);
        let virtual_target = ArmIkTarget {
            wrist: origin,
            elbow_pole: origin + Vec3::Z * 0.4,
        };
        let tracked_target = ArmIkTarget {
            wrist: Vec3::new(0.6, 0.2, 0.1),
            elbow_pole: origin - Vec3::Z * 0.4,
        };
        let observed =
            blend_arm_targets(virtual_target, tracked_target, ArmBlendWeight::ONE).unwrap();
        near(observed.wrist, tracked_target.wrist);
        near(observed.elbow_pole, tracked_target.elbow_pole);
    }

    #[test]
    fn resolved_tracked_arm_pose_reuses_the_shared_conversion() {
        let chain = chain();
        let solution = solve_tracked_arm(chain.rest, target(), Quat::IDENTITY).unwrap();
        let tracked = resolved_tracked_arm_pose(&chain, solution, None, None).unwrap();
        let profile = ArmPoseProfile {
            finger_curl_radians: 0.0,
            ..ArmPoseProfile::default()
        };
        let shared = crate::arm_pose::resolved_from_solution(&chain, &solution, profile)
            .unwrap()
            .unwrap();
        assert_eq!(tracked, shared);
        assert!(tracked.upper_arm_delta.is_finite());
        assert!(tracked.lower_arm_delta.is_finite());
    }

    fn finger_binding(position: Vec3) -> crate::arm::FingerJointRestBinding {
        crate::arm::FingerJointRestBinding {
            entity: bevy::prelude::Entity::from_raw_u32(3).unwrap(),
            rest: bone(position),
        }
    }

    /// The default test chain plus authored index/little-finger rest positions
    /// whose cross product points along +Y, i.e. a palm facing up.
    fn palm_chain() -> ArmChainBinding {
        let mut chain = chain();
        let wrist = chain.rest.wrist.position;
        chain.finger_rest.index.proximal =
            Some(finger_binding(wrist + Vec3::new(0.05, 0.0, 0.003)));
        chain.finger_rest.little.proximal =
            Some(finger_binding(wrist + Vec3::new(0.05, 0.0, -0.003)));
        chain
    }

    /// A chain whose fingers form straight +X chains, so a flexion of zero
    /// leaves every bone at its rest orientation and any non-zero flexion is
    /// visible as a real rotation. The four roots are spread across Z so the
    /// authored palm normal is well defined, as a real hand's is.
    fn articulated_chain() -> ArmChainBinding {
        let mut chain = chain();
        let wrist = chain.rest.wrist.position;
        let row = |dx: [f32; 4], z: f32| crate::arm::FingerJointRestReferences {
            metacarpal: None,
            proximal: Some(finger_binding(wrist + Vec3::new(dx[0], 0.0, z))),
            intermediate: Some(finger_binding(wrist + Vec3::new(dx[1], 0.0, z))),
            distal: Some(finger_binding(wrist + Vec3::new(dx[2], 0.0, z))),
        };
        chain.finger_rest.index = row([0.02, 0.04, 0.055, 0.065], 0.004);
        chain.finger_rest.middle = row([0.02, 0.045, 0.062, 0.074], 0.001);
        chain.finger_rest.ring = row([0.02, 0.042, 0.057, 0.067], -0.002);
        chain.finger_rest.little = row([0.02, 0.036, 0.048, 0.056], -0.005);
        // The thumb rests half way out from the fingers, in the +Z half plane,
        // so the model's own rest spread is not the same as a flat hand's.
        chain.finger_rest.thumb = crate::arm::FingerJointRestReferences {
            metacarpal: Some(finger_binding(wrist + Vec3::new(0.012, 0.0, 0.022))),
            proximal: Some(finger_binding(wrist + Vec3::new(0.03, 0.0, 0.035))),
            intermediate: None,
            distal: Some(finger_binding(wrist + Vec3::new(0.045, 0.0, 0.045))),
        };
        chain
    }

    /// A solved pose for the articulated chain, with the observation's fingers
    /// applied at `weight`.
    fn resolved_fingers(
        chain: &ArmChainBinding,
        fingers: HandFingerPose,
        weight: f32,
    ) -> ResolvedFingerPose {
        let solution = solve_tracked_arm(chain.rest, straight_target(), Quat::IDENTITY).unwrap();
        observed_finger_deltas(chain, fingers, &solution, Quat::IDENTITY, weight)
            .expect("finger deltas")
    }

    fn straight_fingers() -> HandFingerPose {
        HandFingerPose {
            fingers: [[0.0; 3]; 4],
            thumb: [0.0; 2],
            thumb_direction: [1.0, 0.0, 0.0],
        }
    }

    #[test]
    fn a_straight_hand_leaves_every_finger_at_rest() {
        let chain = articulated_chain();
        let pose = resolved_fingers(&chain, straight_fingers(), 1.0);
        for (finger, has_intermediate) in [
            (&pose.index, true),
            (&pose.middle, true),
            (&pose.ring, true),
            (&pose.little, true),
            // A VRM has no thumb intermediate bone.
            (&pose.thumb, false),
        ] {
            for joint in [finger.proximal, finger.intermediate, finger.distal] {
                let Some(delta) = joint else {
                    assert!(!has_intermediate, "the chain has every joint");
                    continue;
                };
                assert!(
                    delta.delta.angle_between(Quat::IDENTITY) < 1.0e-5,
                    "a straight finger must stay at its rest pose: {:?}",
                    delta.delta
                );
            }
        }
    }

    #[test]
    fn each_finger_bends_by_its_own_observed_amount() {
        let chain = articulated_chain();
        let mut observed = straight_fingers();
        // A peace sign: the index and middle stay up, the ring and little fold.
        observed.fingers[0] = [0.0; 3];
        observed.fingers[1] = [0.0; 3];
        observed.fingers[2] = [1.0, 1.2, 0.6];
        observed.fingers[3] = [0.8, 0.9, 0.5];
        let pose = resolved_fingers(&chain, observed, 1.0);

        for finger in [&pose.index, &pose.middle] {
            for joint in [finger.proximal, finger.intermediate, finger.distal] {
                let delta = joint.expect("the chain has every joint");
                assert!(
                    delta.delta.angle_between(Quat::IDENTITY) < 1.0e-5,
                    "an extended finger must not be curled"
                );
            }
        }
        // The folded fingers differ from each other, so this is not one shared
        // curl amount applied to all of them.
        let ring = pose.ring.proximal.expect("ring proximal");
        let little = pose.little.proximal.expect("little proximal");
        assert!(ring.delta.angle_between(Quat::IDENTITY) > 0.5);
        assert!(little.delta.angle_between(Quat::IDENTITY) > 0.2);
        assert!(
            (ring.delta.angle_between(Quat::IDENTITY) - little.delta.angle_between(Quat::IDENTITY))
                .abs()
                > 0.05,
            "each finger must keep its own curl"
        );
    }

    #[test]
    fn the_thumb_is_not_folded_by_a_four_finger_axis() {
        // The thumb's rest chain runs diagonally and its joints do not share the
        // four fingers' geometry, so its flexion must still resolve to a real
        // rotation on its own bones rather than a missing or identity delta.
        let chain = articulated_chain();
        let mut observed = straight_fingers();
        observed.thumb = [0.5, 0.7];
        let pose = resolved_fingers(&chain, observed, 1.0);
        let proximal = pose.thumb.proximal.expect("thumb proximal");
        let distal = pose.thumb.distal.expect("thumb distal");
        assert!(proximal.delta.angle_between(Quat::IDENTITY) > 0.1);
        assert!(distal.delta.angle_between(Quat::IDENTITY) > 0.1);
        assert!(
            pose.thumb.intermediate.is_none(),
            "VRM has no thumb intermediate"
        );
    }

    #[test]
    fn the_thumb_opens_relative_to_the_models_own_rest_spread() {
        let chain = articulated_chain();
        let rest_direction = crate::arm::finite_normalized(
            chain
                .finger_rest
                .thumb
                .metacarpal
                .expect("thumb metacarpal")
                .rest
                .position
                - chain.rest.wrist.position,
        )
        .expect("rest thumb ray");

        // The rig's rest thumb points partly along +Z; a flat hand's points
        // along +X, so applying the observation must open it about the palm
        // normal by the angle between the two, not leave it where it was.
        let pose = resolved_fingers(&chain, straight_fingers(), 1.0);
        let opened = pose.thumb.metacarpal.expect("thumb metacarpal");
        assert!(
            opened.delta.angle_between(Quat::IDENTITY) > 0.1,
            "the rest spread and the observation differ, so a rotation is expected"
        );
        // Rotating the rest ray by the delta must land on the observed ray,
        // projected off the palm normal the way the rotation is defined.
        let axis = rest_palm_normal(&chain).expect("rest palm normal");
        let in_plane = |value: Vec3| value - axis * value.dot(axis);
        let rotated = in_plane(opened.delta * rest_direction);
        let expected = in_plane(Vec3::X);
        assert!(
            crate::arm::finite_normalized(rotated)
                .expect("the rotated ray stays in the plane")
                .dot(crate::arm::finite_normalized(expected).expect("expected ray"))
                > 0.99,
            "the thumb must open onto the observation"
        );
    }

    #[test]
    fn the_finger_channel_weight_scales_every_joint() {
        let chain = articulated_chain();
        let mut observed = straight_fingers();
        observed.fingers[0] = [0.6, 0.8, 0.4];
        let full = resolved_fingers(&chain, observed, 1.0);
        let half = resolved_fingers(&chain, observed, 0.5);
        for (a, b) in [
            (full.index.proximal, half.index.proximal),
            (full.index.intermediate, half.index.intermediate),
            (full.index.distal, half.index.distal),
        ] {
            let a = a.expect("index joint");
            let b = b.expect("index joint");
            let (full_angle, half_angle) = (
                a.delta.angle_between(Quat::IDENTITY),
                b.delta.angle_between(Quat::IDENTITY),
            );
            assert!(
                (half_angle - full_angle * 0.5).abs() < 1.0e-3,
                "half weight must halve the rotation: {half_angle} vs {full_angle}"
            );
        }
        // A zero weight is the virtual channel: no finger rotation at all.
        let solution = solve_tracked_arm(chain.rest, straight_target(), Quat::IDENTITY).unwrap();
        assert_eq!(
            observed_finger_deltas(&chain, straight_fingers(), &solution, Quat::IDENTITY, 0.0),
            None
        );
    }

    #[test]
    fn finger_deltas_are_skipped_where_the_model_has_no_bone() {
        // A chain with only the index bound must not invent the other fingers.
        let mut chain = articulated_chain();
        chain.finger_rest.middle = crate::arm::FingerJointRestReferences::default();
        chain.finger_rest.ring = crate::arm::FingerJointRestReferences::default();
        chain.finger_rest.little = crate::arm::FingerJointRestReferences::default();
        chain.finger_rest.thumb = crate::arm::FingerJointRestReferences::default();
        let mut observed = straight_fingers();
        observed.fingers[0] = [0.5, 0.5, 0.5];
        observed.thumb = [0.5, 0.5];
        let pose = resolved_fingers(&chain, observed, 1.0);
        assert!(pose.index.proximal.is_some());
        for missing in [&pose.middle, &pose.ring, &pose.little, &pose.thumb] {
            assert!(missing.proximal.is_none());
            assert!(missing.intermediate.is_none());
            assert!(missing.distal.is_none());
            assert!(missing.metacarpal.is_none());
        }
    }

    /// The hand's model-space palm normal after the solve and hand delta.
    fn solved_palm_normal(chain: &ArmChainBinding, solution: &ArmIkSolution, hand: Quat) -> Vec3 {
        let rest_hand = chain.rest.wrist.global_rotation;
        let hand_global = solution.lower_arm_global_rotation
            * (chain.rest.elbow.global_rotation.inverse() * rest_hand)
            * hand;
        (hand_global * (rest_hand.inverse() * rest_palm_normal(chain).unwrap())).normalize()
    }

    /// A nearly straight solve, where a humeral roll cannot move the wrist.
    fn straight_target() -> ArmTrackingTarget {
        ArmTrackingTarget {
            wrist: [0.9, -0.05, 0.05],
            elbow_pole: [0.5, 0.4, 0.0],
            palm_normal: None,
            fingers: None,
        }
    }

    #[test]
    fn palm_twist_rotates_the_wrist_to_the_observed_plane() {
        let chain = palm_chain();
        let mut solution =
            solve_tracked_arm(chain.rest, straight_target(), Quat::IDENTITY).unwrap();
        let hand = align_palm_twist(&chain, &mut solution, [0.0, 0.0, 1.0], Quat::IDENTITY, 1.0)
            .expect("hand roll");

        // The forearm and the hand each take a share; together they land the
        // palm plane on the observation.
        let axis = crate::arm::finite_normalized(solution.wrist - solution.elbow).unwrap();
        let observed = Vec3::Z;
        let expected = (observed - axis * observed.dot(axis)).normalize();
        let actual = {
            let normal = solved_palm_normal(&chain, &solution, hand);
            (normal - axis * normal.dot(axis)).normalize()
        };
        assert!(
            actual.dot(expected) > 1.0 - 1.0e-4,
            "hand plane must match the observation: {actual:?} != {expected:?}"
        );
        assert_ne!(hand, Quat::IDENTITY);
    }

    #[test]
    fn palm_twist_splits_the_roll_between_the_forearm_and_the_hand() {
        let chain = palm_chain();
        let mut solution =
            solve_tracked_arm(chain.rest, straight_target(), Quat::IDENTITY).unwrap();
        let forearm_before = solution.lower_arm_delta;
        let hand = align_palm_twist(&chain, &mut solution, [0.0, 0.0, 1.0], Quat::IDENTITY, 1.0)
            .expect("hand roll");
        assert_ne!(
            solution.lower_arm_delta, forearm_before,
            "the forearm must take a share of the roll"
        );
        assert_ne!(
            hand,
            Quat::IDENTITY,
            "the hand must take a share of the roll"
        );
    }

    #[test]
    fn palm_twist_weight_zero_is_a_no_op() {
        let chain = palm_chain();
        let mut solution = solve_tracked_arm(chain.rest, target(), Quat::IDENTITY).unwrap();
        let before = solution;
        assert_eq!(
            align_palm_twist(&chain, &mut solution, [0.0, 0.0, 1.0], Quat::IDENTITY, 0.0),
            None
        );
        assert_eq!(solution, before);
    }

    #[test]
    fn palm_twist_scales_monotonically_with_the_weight() {
        let chain = palm_chain();
        let mut full = solve_tracked_arm(chain.rest, straight_target(), Quat::IDENTITY).unwrap();
        let full_hand =
            align_palm_twist(&chain, &mut full, [0.0, 0.0, 1.0], Quat::IDENTITY, 1.0).unwrap();
        let mut half = solve_tracked_arm(chain.rest, straight_target(), Quat::IDENTITY).unwrap();
        let half_hand =
            align_palm_twist(&chain, &mut half, [0.0, 0.0, 1.0], Quat::IDENTITY, 0.5).unwrap();

        let axis = crate::arm::finite_normalized(full.wrist - full.elbow).unwrap();
        let expected = (Vec3::Z - axis * Vec3::Z.dot(axis)).normalize();
        let error = |solution: &ArmIkSolution, hand: Quat| {
            let normal = solved_palm_normal(&chain, solution, hand);
            let normal = (normal - axis * normal.dot(axis)).normalize();
            1.0 - normal.dot(expected)
        };
        let full_error = error(&full, full_hand);
        let half_error = error(&half, half_hand);
        assert!(full_error < 1.0e-4, "full weight must align: {full_error}");
        assert!(
            half_error > full_error && half_error < 1.0,
            "half weight must sit between the default roll and full alignment: \
             {half_error} vs {full_error}"
        );
    }

    #[test]
    fn palm_twist_is_a_no_op_without_usable_geometry() {
        let plain = chain();
        let mut base = solve_tracked_arm(plain.rest, target(), Quat::IDENTITY).unwrap();
        assert_eq!(
            align_palm_twist(&plain, &mut base, [0.0, 0.0, 1.0], Quat::IDENTITY, 1.0),
            None
        );

        let chain = palm_chain();
        let mut base = solve_tracked_arm(chain.rest, target(), Quat::IDENTITY).unwrap();
        let axis = crate::arm::finite_normalized(base.wrist - base.elbow).unwrap();
        assert_eq!(
            align_palm_twist(&chain, &mut base, axis.to_array(), Quat::IDENTITY, 1.0),
            None
        );
    }
}
