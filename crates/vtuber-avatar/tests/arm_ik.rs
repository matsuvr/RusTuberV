// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Pure analytic default-arm IK tests.

use bevy::prelude::*;
use vtuber_avatar::{
    ArmChainBinding, ArmChainCapabilities, ArmIkError, ArmIkInput, ArmIkTarget, ArmPoseProfile,
    ArmRestGeometry, ArmSide, FingerReferences, FingerRestReferences, RestSpaceBonePose,
    default_arm_target, solve_two_bone_arm,
};

fn chain(side: ArmSide, upper_length: f32, forearm_length: f32) -> ArmChainBinding {
    let sign = match side {
        ArmSide::Left => 1.0,
        ArmSide::Right => -1.0,
    };
    let shoulder = Vec3::new(0.0, 1.4, 0.0);
    let upper = shoulder + Vec3::new(sign * 0.2, 0.0, 0.0);
    let elbow = upper + Vec3::new(sign * upper_length, 0.0, 0.0);
    let wrist = elbow + Vec3::new(sign * forearm_length, 0.0, 0.0);
    let rest_pose = |position: Vec3| RestSpaceBonePose {
        position,
        global_rotation: Quat::IDENTITY,
        local_rotation: Quat::IDENTITY,
    };
    ArmChainBinding {
        side,
        shoulder: None,
        upper_arm: Entity::from_raw_u32(1).unwrap(),
        lower_arm: Entity::from_raw_u32(2).unwrap(),
        hand: Entity::from_raw_u32(3).unwrap(),
        fingers: FingerReferences::default(),
        rest: ArmRestGeometry {
            shoulder: Some(rest_pose(shoulder)),
            upper_arm: rest_pose(upper),
            elbow: rest_pose(elbow),
            wrist: rest_pose(wrist),
            upper_arm_length: upper_length,
            forearm_length,
            total_arm_length: upper_length + forearm_length,
        },
        capabilities: ArmChainCapabilities::default(),
        finger_rest: FingerRestReferences::default(),
    }
}

fn input_for(chain: &ArmChainBinding, target: ArmIkTarget) -> ArmIkInput {
    ArmIkInput::from_geometry(chain.rest, target)
}

fn assert_finite_solution(solution: &vtuber_avatar::ArmIkSolution) {
    assert!(solution.solved_reach.is_finite());
    assert!(solution.elbow.is_finite());
    assert!(solution.wrist.is_finite());
    assert!(solution.upper_arm_global_rotation.is_finite());
    assert!(solution.lower_arm_global_rotation.is_finite());
    assert!(solution.upper_arm_local_rotation.is_finite());
    assert!(solution.lower_arm_local_rotation.is_finite());
    assert!(solution.upper_arm_delta.is_finite());
    assert!(solution.lower_arm_delta.is_finite());
    assert!((solution.upper_arm_delta.length() - 1.0).abs() < 1.0e-5);
    assert!((solution.lower_arm_delta.length() - 1.0).abs() < 1.0e-5);
}

#[test]
fn default_profile_is_nominally_relaxed_and_bends_the_elbow() {
    let chain = chain(ArmSide::Left, 0.7, 0.55);
    let target = default_arm_target(&chain, ArmPoseProfile::default()).unwrap();
    assert!(
        target.wrist.z > chain.rest.wrist.position.z,
        "the default hand offset must follow the VRM +Z forward convention"
    );
    assert!(target.elbow_pole.z < chain.rest.elbow.position.z);
    let solution = solve_two_bone_arm(input_for(&chain, target)).unwrap();
    assert_finite_solution(&solution);
    assert!(solution.solved_reach < chain.rest.total_arm_length);
    assert!(solution.elbow.y < chain.rest.upper_arm.position.y);
    assert!(solution.upper_arm_delta.dot(Quat::IDENTITY).abs() < 0.999_99);
    assert!(solution.lower_arm_delta.dot(Quat::IDENTITY).abs() < 0.999_99);
}

#[test]
fn mirrored_chains_produce_mirrored_model_space_solutions() {
    let left = chain(ArmSide::Left, 0.7, 0.55);
    let right = chain(ArmSide::Right, 0.7, 0.55);
    let left_solution = solve_two_bone_arm(input_for(
        &left,
        default_arm_target(&left, ArmPoseProfile::default()).unwrap(),
    ))
    .unwrap();
    let right_solution = solve_two_bone_arm(input_for(
        &right,
        default_arm_target(&right, ArmPoseProfile::default()).unwrap(),
    ))
    .unwrap();
    assert_finite_solution(&left_solution);
    assert_finite_solution(&right_solution);
    assert_eq!(left_solution.elbow.x, -right_solution.elbow.x);
    assert_eq!(left_solution.elbow.y, right_solution.elbow.y);
    assert_eq!(left_solution.elbow.z, right_solution.elbow.z);
    assert_eq!(left_solution.wrist.x, -right_solution.wrist.x);
    assert_eq!(left_solution.wrist.y, right_solution.wrist.y);
    assert_eq!(left_solution.wrist.z, right_solution.wrist.z);
}

#[test]
fn asymmetric_lengths_are_used_without_iterative_solving() {
    let chain = chain(ArmSide::Left, 0.9, 0.4);
    let target = ArmIkTarget {
        wrist: chain.rest.upper_arm.position + Vec3::new(0.95, -0.25, 0.0),
        elbow_pole: chain.rest.upper_arm.position + Vec3::new(0.0, -0.2, 0.1),
    };
    let solution = solve_two_bone_arm(input_for(&chain, target)).unwrap();
    assert_finite_solution(&solution);
    assert!((solution.elbow.distance(chain.rest.upper_arm.position) - 0.9).abs() < 1.0e-4);
    assert!((solution.wrist.distance(solution.elbow) - 0.4).abs() < 1.0e-4);
}

#[test]
fn unreachable_targets_are_clamped_at_both_annulus_boundaries() {
    let chain = chain(ArmSide::Left, 0.9, 0.4);
    let far = solve_two_bone_arm(input_for(
        &chain,
        ArmIkTarget {
            wrist: chain.rest.upper_arm.position + Vec3::new(10.0, 0.0, 0.0),
            elbow_pole: chain.rest.upper_arm.position + Vec3::new(0.0, -1.0, 0.0),
        },
    ))
    .unwrap();
    let folded = solve_two_bone_arm(input_for(
        &chain,
        ArmIkTarget {
            wrist: chain.rest.upper_arm.position,
            elbow_pole: chain.rest.upper_arm.position + Vec3::new(0.0, -1.0, 0.0),
        },
    ))
    .unwrap();
    assert_finite_solution(&far);
    assert_finite_solution(&folded);
    assert!((far.solved_reach - (1.3 - 1.0e-4)).abs() < 1.0e-5);
    assert!((folded.solved_reach - (0.5 + 1.0e-4)).abs() < 1.0e-5);
}

#[test]
fn collinear_and_near_zero_poles_fall_back_deterministically() {
    let chain = chain(ArmSide::Left, 0.7, 0.55);
    let target = chain.rest.upper_arm.position + Vec3::new(1.0, -0.1, 0.0);
    let collinear = solve_two_bone_arm(input_for(
        &chain,
        ArmIkTarget {
            wrist: target,
            elbow_pole: chain.rest.upper_arm.position + Vec3::new(10.0, -1.0, 0.0),
        },
    ))
    .unwrap();
    let near_zero = solve_two_bone_arm(input_for(
        &chain,
        ArmIkTarget {
            wrist: target,
            elbow_pole: chain.rest.upper_arm.position + Vec3::splat(1.0e-8),
        },
    ))
    .unwrap();
    assert_finite_solution(&collinear);
    assert_finite_solution(&near_zero);
    assert!(collinear.elbow.y > chain.rest.upper_arm.position.y);
    assert_eq!(collinear.elbow, near_zero.elbow);
}

#[test]
fn near_straight_target_remains_finite_and_normalized() {
    let chain = chain(ArmSide::Left, 0.8, 0.6);
    let solution = solve_two_bone_arm(input_for(
        &chain,
        ArmIkTarget {
            wrist: chain.rest.upper_arm.position + Vec3::new(1.39999, 0.0, 0.0),
            elbow_pole: chain.rest.upper_arm.position + Vec3::Y,
        },
    ))
    .unwrap();
    assert_finite_solution(&solution);
    assert!(solution.elbow.distance(chain.rest.upper_arm.position) > 0.79);
}

#[test]
fn non_identity_rest_orientations_preserve_model_solution_and_conjugate_deltas() {
    let identity_chain = chain(ArmSide::Left, 0.7, 0.55);
    let mut rotated_chain = identity_chain;
    rotated_chain.rest.upper_arm.global_rotation = Quat::from_rotation_y(0.6);
    rotated_chain.rest.upper_arm.local_rotation = Quat::from_rotation_x(-0.4);
    rotated_chain.rest.elbow.global_rotation = Quat::from_rotation_z(-0.5);
    rotated_chain.rest.elbow.local_rotation = Quat::from_rotation_y(0.3);
    let target = ArmIkTarget {
        wrist: identity_chain.rest.upper_arm.position + Vec3::new(0.9, -0.3, 0.1),
        elbow_pole: identity_chain.rest.upper_arm.position + Vec3::new(0.0, -0.2, 0.2),
    };
    let identity = solve_two_bone_arm(input_for(&identity_chain, target)).unwrap();
    let rotated = solve_two_bone_arm(input_for(&rotated_chain, target)).unwrap();
    assert_finite_solution(&rotated);
    assert_eq!(identity.elbow, rotated.elbow);
    assert_eq!(identity.wrist, rotated.wrist);
    assert!(
        rotated
            .upper_arm_global_rotation
            .dot(identity.upper_arm_global_rotation * Quat::from_rotation_y(0.6))
            .abs()
            > 0.999_9
    );
    assert!(
        rotated
            .upper_arm_global_rotation
            .dot(rotated_chain.rest.upper_arm.global_rotation * rotated.upper_arm_delta)
            .abs()
            > 0.999_9
    );
}

#[test]
fn invalid_inputs_fail_without_nan_output() {
    let chain = chain(ArmSide::Left, 0.7, 0.55);
    let mut input = input_for(
        &chain,
        ArmIkTarget {
            wrist: Vec3::new(f32::NAN, 0.0, 0.0),
            elbow_pole: Vec3::ZERO,
        },
    );
    assert_eq!(solve_two_bone_arm(input), Err(ArmIkError::NonFiniteInput));
    input = input_for(
        &chain,
        ArmIkTarget {
            wrist: Vec3::X,
            elbow_pole: Vec3::Y,
        },
    );
    input.upper_arm_length = 0.0;
    assert_eq!(
        solve_two_bone_arm(input),
        Err(ArmIkError::DegenerateGeometry)
    );
    assert_eq!(
        default_arm_target(
            &chain,
            ArmPoseProfile {
                reach_ratio: 1.1,
                ..default()
            }
        ),
        Err(ArmIkError::InvalidProfile)
    );
}
