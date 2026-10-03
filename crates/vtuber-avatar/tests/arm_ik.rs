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
        (solution.elbow - chain.rest.upper_arm.position)
            .normalize()
            .dot(direction)
            > 0.999
    );
    assert!((solution.wrist - solution.elbow).normalize().dot(direction) > 0.999);
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
    assert!((far.solved_reach - (1.3 - 1.0e-4)).abs() < 1.0e-5);
    let upper = (folded.elbow - chain.rest.upper_arm.position).normalize();
    let lower = (folded.wrist - folded.elbow).normalize();
    assert!((upper.dot(lower).clamp(-1.0, 1.0).acos() - 130.0_f32.to_radians()).abs() < 1.0e-5);
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
            assert!(
                (hinge * input.elbow_axis).distance(input.elbow_axis) < 1.0e-5,
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
fn anterior_elbow_flexion_preserves_both_palm_surfaces_without_a_humeral_half_turn() {
    use vtuber_avatar::FingerJointRestBinding;
    use vtuber_avatar::tracked_arm::{HandOrientationFilter, align_hand_orientation};

    for side in [ArmSide::Left, ArmSide::Right] {
        let sign = if side == ArmSide::Left { 1.0 } else { -1.0 };
        let mut chain = chain(side, 0.7, 0.55);
        chain.rest.upper_arm.global_rotation = Quat::from_rotation_y(0.6);
        chain.rest.elbow.global_rotation = Quat::from_rotation_x(0.4) * Quat::from_rotation_z(-0.5);
        chain.rest.wrist.global_rotation = Quat::from_rotation_z(0.3);
        let finger = |offset| FingerJointRestBinding {
            entity: chain.hand,
            rest: RestSpaceBonePose {
                position: chain.rest.wrist.position + offset,
                global_rotation: Quat::IDENTITY,
                local_rotation: Quat::IDENTITY,
            },
        };
        // VRM T-pose: fingers point outward, index lies anterior to little.
        chain.finger_rest.index.proximal = Some(finger(Vec3::new(sign * 0.08, 0.0, 0.02)));
        chain.finger_rest.little.proximal = Some(finger(Vec3::new(sign * 0.08, 0.0, -0.02)));
        let rest_normal = Vec3::Y * sign;
        let forearm = chain.rest.wrist.position - chain.rest.elbow.position;
        let flexion = Quat::from_rotation_y(-sign * 60.0_f32.to_radians());
        let target = ArmIkTarget {
            wrist: chain.rest.elbow.position + flexion * forearm,
            elbow_pole: chain.rest.elbow.position,
        };
        let input = ArmIkInput::from_chain(&chain, target);
        let base = solve_two_bone_arm(input).unwrap();
        assert!(base.upper_arm_delta.angle_between(Quat::IDENTITY) < 1.0e-3);
        let axis = (base.wrist - base.elbow).normalize();
        let physical_palm = -Vec3::Y;
        let mut shown_palms = Vec::new();
        for roll in [-90.0_f32, 90.0] {
            let pronation = Quat::from_axis_angle(axis, roll.to_radians());
            let observed = pronation * flexion * rest_normal;
            let expected_palm = pronation * flexion * physical_palm;
            let mut filter = HandOrientationFilter::default();
            let mut solution = base;
            for _ in 0..120 {
                solution = base;
                let twist = align_hand_orientation(
                    &chain,
                    &mut solution,
                    (
                        observed.to_array(),
                        (pronation * flexion * forearm.normalize()).to_array(),
                    ),
                    Quat::IDENTITY,
                    1.0,
                    &mut filter,
                    1.0 / 60.0,
                )
                .unwrap();
                assert!(twist.hand.unwrap().angle_between(Quat::IDENTITY) < 1.0e-3);
            }
            let lower_model =
                solution.lower_arm_global_rotation * chain.rest.elbow.global_rotation.inverse();
            let shown_palm = lower_model * physical_palm;
            assert!(shown_palm.dot(expected_palm) > 0.999);
            assert!(solution.upper_arm_delta.angle_between(Quat::IDENTITY) < 1.0e-3);
            assert!(solution.elbow.distance(base.elbow) < 1.0e-5);
            assert!(solution.wrist.distance(base.wrist) < 1.0e-5);
            shown_palms.push(shown_palm);
        }
        assert!(shown_palms[0].dot(shown_palms[1]) < -0.999);
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
