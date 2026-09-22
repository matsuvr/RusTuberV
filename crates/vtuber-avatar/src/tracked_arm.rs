//! Pure adapter from engine-neutral tracking targets to the existing arm IK.
//!
//! This module never writes Transforms. Live integration must route its output
//! through the existing arm compositor, not register another bone writer.

use bevy::prelude::{Quat, Vec3};
use vtuber_core::arm_tracking::{ArmBlendWeight, ArmTrackingTarget};

use crate::arm::{
    ArmChainBinding, ArmIkError, ArmIkInput, ArmIkSolution, ArmIkTarget, ArmPoseProfile,
    ArmRestGeometry, solve_two_bone_arm,
};
use crate::arm_pipeline::ArmPipelineError;
use crate::arm_pose::ResolvedArmPose;

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
/// share is already inside `solution`. Tracked fingers keep their rest pose (no
/// observation exists) and the virtual shoulder-trim/swivel/twist modifiers are
/// not applied.
pub fn resolved_tracked_arm_pose(
    chain: &ArmChainBinding,
    solution: ArmIkSolution,
    hand_delta: Option<Quat>,
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
    Ok(pose)
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
        let tracked = resolved_tracked_arm_pose(&chain, solution, None).unwrap();
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
