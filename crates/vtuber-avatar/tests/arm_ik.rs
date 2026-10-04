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
    ArmIkInput::from_geometry(chain.rest, target, chain.side)
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
fn default_profile_lowers_the_arm_with_lateral_hand_clearance() {
    let chain = chain(ArmSide::Left, 0.7, 0.55);
    let target = default_arm_target(&chain, ArmPoseProfile::default()).unwrap();
    assert_eq!(target.wrist.z, chain.rest.upper_arm.position.z);
    assert!(target.elbow_pole.z < chain.rest.elbow.position.z);
    let solution = solve_two_bone_arm(input_for(&chain, target)).unwrap();
    assert_finite_solution(&solution);
    assert!(solution.solved_reach < chain.rest.total_arm_length);
    assert!(solution.elbow.y < chain.rest.upper_arm.position.y);
    let direction = Vec3::new(
        10.0_f32.to_radians().sin(),
        -10.0_f32.to_radians().cos(),
        0.0,
    );
    assert!(
        (solution.wrist - chain.rest.upper_arm.position)
            .normalize()
            .dot(direction)
            > 0.99999
    );
    assert!((solution.elbow.distance(chain.rest.upper_arm.position) - 0.7).abs() < 1.0e-5);
    assert!((solution.wrist.distance(solution.elbow) - 0.55).abs() < 1.0e-5);
    // The source carrying angle remains present even near full extension.
    assert!(
        (solution.elbow - chain.rest.upper_arm.position)
            .angle_between(solution.wrist - solution.elbow)
            > 5.0_f32.to_radians()
    );
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
fn unreachable_targets_are_clamped_to_extension_and_flexion_limits() {
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
    assert!(far.solved_reach < 1.3 - 1.0e-4);
    let input = input_for(
        &chain,
        ArmIkTarget {
            wrist: chain.rest.wrist.position,
            elbow_pole: chain.rest.elbow.position,
        },
    );
    let upper_model =
        folded.upper_arm_global_rotation * chain.rest.upper_arm.global_rotation.inverse();
    let lower = upper_model.inverse() * (folded.wrist - folded.elbow).normalize();
    let expected =
        Quat::from_axis_angle(input.elbow_axis, 130.0_f32.to_radians()) * input.neutral_forearm;
    assert!(lower.distance(expected) < 1.0e-5);
}

#[test]
fn elbow_flexion_uses_one_rest_axis_across_poles_and_non_identity_bone_axes() {
    for side in [ArmSide::Left, ArmSide::Right] {
        let mut chain = chain(side, 0.7, 0.55);
        chain.rest.upper_arm.global_rotation = Quat::from_rotation_y(0.6);
        chain.rest.elbow.global_rotation = Quat::from_rotation_x(0.4) * Quat::from_rotation_z(-0.5);
        for pole in [Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z] {
            let input = ArmIkInput::from_chain(
                &chain,
                ArmIkTarget {
                    wrist: chain.rest.upper_arm.position + Vec3::new(0.8, -0.3, 0.2),
                    elbow_pole: chain.rest.upper_arm.position + pole,
                },
            );
            let solution = solve_two_bone_arm(input).unwrap();
            let hinge = chain.rest.elbow.global_rotation
                * solution.lower_arm_delta
                * chain.rest.elbow.global_rotation.inverse();
            let rest_to_neutral = Quat::from_rotation_arc(
                (chain.rest.wrist.position - chain.rest.elbow.position).normalize(),
                input.neutral_forearm,
            );
            let flexion = hinge * rest_to_neutral.inverse();
            assert!(
                (flexion * input.elbow_axis).distance(input.elbow_axis) < 1.0e-5,
                "flexion must not introduce extra elbow axes"
            );
            let upper_model =
                solution.upper_arm_global_rotation * chain.rest.upper_arm.global_rotation.inverse();
            let elbow = chain.rest.upper_arm.position
                + upper_model * (chain.rest.elbow.position - chain.rest.upper_arm.position);
            let wrist = elbow
                + upper_model * hinge * (chain.rest.wrist.position - chain.rest.elbow.position);
            assert!(elbow.distance(solution.elbow) < 1.0e-5);
            assert!(wrist.distance(solution.wrist) < 1.0e-5);
        }
    }
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
    assert!(identity.elbow.distance(rotated.elbow) < 1.0e-6);
    assert!(identity.wrist.distance(rotated.wrist) < 1.0e-6);
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
