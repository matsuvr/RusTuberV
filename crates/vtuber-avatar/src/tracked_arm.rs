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
    let wrist = virtual_target
        .wrist
        .lerp(tracked_target.wrist, wrist_weight);
    let elbow_pole = blend_pole(
        virtual_target.elbow_pole,
        tracked_target.elbow_pole,
        wrist,
        pole_weight,
    )?;
    if !wrist.is_finite() || !elbow_pole.is_finite() {
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
/// reimplemented. Tracked fingers keep their rest pose (no observation exists)
/// and the virtual shoulder-trim/swivel/twist modifiers are not applied.
pub fn resolved_tracked_arm_pose(
    chain: &ArmChainBinding,
    solution: ArmIkSolution,
) -> Result<ResolvedArmPose, ArmPipelineError> {
    let profile = ArmPoseProfile {
        finger_curl_radians: 0.0,
        ..ArmPoseProfile::default()
    };
    crate::arm_pose::resolved_from_solution(chain, &solution, profile)?
        .ok_or(ArmPipelineError::DegenerateSolvedPose)
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
/// Missing/occluded observations are handled by tracking, not an implicit fallback.
pub fn solve_tracked_arm(
    rest: ArmRestGeometry,
    target: ArmTrackingTarget,
    tracking_to_rest: Quat,
) -> Result<ArmIkSolution, ArmIkError> {
    let target = tracked_arm_ik_target(rest, target, tracking_to_rest);
    solve_two_bone_arm(ArmIkInput::from_geometry(rest, target))
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
                },
            ),
            Err(ArmIkError::DegenerateGeometry)
        );
    }

    #[test]
    fn resolved_tracked_arm_pose_reuses_the_shared_conversion() {
        let chain = chain();
        let solution = solve_tracked_arm(chain.rest, target(), Quat::IDENTITY).unwrap();
        let tracked = resolved_tracked_arm_pose(&chain, solution).unwrap();
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
}
