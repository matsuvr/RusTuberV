//! Solver regression checks.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]
use super::*;
use crate::arm::ArmSide;
use crate::upper_limb::tests::{chain, state};

#[test]
fn near_opposed_contacts_keep_the_feasible_tangent_independent_of_row_order() {
    let mut a = [0.0; 18];
    a[0] = 1.0;
    a[1] = 0.001;
    let mut b = a;
    b[0] = -1.0;
    for rows in [[a, b], [b, a]] {
        let linear = nalgebra::DVector::from_fn(18, |i, _| match i {
            0 | 2 => -f64::from(MAX_STEP_RADIANS),
            1 => f64::from(MAX_STEP_RADIANS),
            _ => 0.0,
        });
        let step = quadratic_step(
            nalgebra::DMatrix::identity(18, 18),
            linear,
            &rows,
            &[0.0; 2],
        )
        .unwrap();
        assert!(step[0].abs() < 1.0e-5 && step[1].abs() < 1.0e-5);
        assert!((step[2] - MAX_STEP_RADIANS).abs() < 1.0e-6);
        for row in rows {
            assert!(
                row.iter()
                    .zip(step)
                    .map(|(a, b)| a * f64::from(b))
                    .sum::<f64>()
                    >= -1.0e-7
            );
        }
    }
}

#[test]
fn contradictory_linear_contacts_report_absence_of_a_step() {
    let mut a = [0.0; 18];
    a[0] = 1.0;
    let mut b = a;
    b[0] = -1.0;
    assert!(
        quadratic_step(
            nalgebra::DMatrix::identity(18, 18),
            nalgebra::DVector::zeros(18),
            &[a, b],
            &[-1.0, 0.0]
        )
        .is_none()
    );
}

#[test]
fn normal_correction_allows_a_tangent_step_at_a_curved_boundary() {
    // Unit disk at (1, 0): its tangent step violates the true boundary
    // although the linear constraint admits it. A short SOC moves back
    // inside while retaining the tangential progress.
    let step = 0.02_f64;
    let mut jacobian = nalgebra::DMatrix::zeros(1, 18);
    jacobian[(0, 0)] = -2.0;
    let correction = second_order_correction(&jacobian, &[-step * step]).unwrap();
    let x = 1.0 + f64::from(correction[0]);
    assert!(1.0 - x * x - step * step > 0.0);
    assert!(correction[0].abs() < step as f32 * step as f32);
    assert!(correction.iter().skip(1).all(|c| *c == 0.0));
}

#[test]
fn resting_pose_is_independent_of_vrm1_authored_rest_rotations() {
    for side in [ArmSide::Left, ArmSide::Right] {
        let mut chain_with_identity_arm_rotations = chain(side);
        let mut chain_with_authored_arm_rotations = chain_with_identity_arm_rotations;
        for rest in [
            &mut chain_with_identity_arm_rotations.rest.upper_arm,
            &mut chain_with_identity_arm_rotations.rest.elbow,
            &mut chain_with_identity_arm_rotations.rest.wrist,
        ] {
            rest.global_rotation = Quat::IDENTITY;
            rest.local_rotation = Quat::IDENTITY;
        }
        for (i, rest) in [
            &mut chain_with_authored_arm_rotations.rest.upper_arm,
            &mut chain_with_authored_arm_rotations.rest.elbow,
            &mut chain_with_authored_arm_rotations.rest.wrist,
        ]
        .into_iter()
        .enumerate()
        {
            rest.global_rotation = Quat::from_euler(EulerRot::XYZ, 0.34 + i as f32, -0.5, 0.7);
            rest.local_rotation = Quat::from_rotation_y(0.6 - i as f32);
        }
        let geometry = CollisionGeometry::default();
        let body = HashMap::new();
        let solve = |chain: &ArmChainBinding| {
            let target = ArmGoal::resting(chain, Default::default()).unwrap();
            let problem = Problem {
                chains: [Some(chain), None],
                goals: [Some(target), None],
                geometry: &geometry,
                body: &body,
                body_curve: None,
                tolerance: 64.0 * f32::EPSILON * chain.rest.total_arm_length,
            };
            let (solved, status) = problem.solve_neutral([Some(target.neutral), None]);
            assert!(matches!(status, SolveStatus::Feasible { .. }));
            solved[0].unwrap().forward(chain).unwrap()
        };
        let a = solve(&chain_with_identity_arm_rotations);
        let b = solve(&chain_with_authored_arm_rotations);
        for (a, b) in [
            (a.shoulder, b.shoulder),
            (a.elbow, b.elbow),
            (a.wrist, b.wrist),
        ] {
            assert!(a.distance(b) < 2.0e-5);
        }
        let (an, af) = a.palm.unwrap();
        let (bn, bf) = b.palm.unwrap();
        assert!(an.distance(bn) < 2.0e-5);
        assert!(af.distance(bf) < 2.0e-5);
        assert!(bf.cross(bn).dot(Vec3::Z) > 0.99);
    }
}

#[test]
fn resting_profile_keeps_elbow_extended_and_thumb_forward() {
    for side in [ArmSide::Left, ArmSide::Right] {
        let chain = chain(side);
        let target = ArmGoal::resting(&chain, Default::default()).unwrap();
        let geometry = CollisionGeometry::default();
        let body = HashMap::new();
        let problem = Problem {
            chains: [Some(&chain), None],
            goals: [Some(target), None],
            geometry: &geometry,
            body: &body,
            body_curve: None,
            tolerance: 64.0 * f32::EPSILON * chain.rest.total_arm_length,
        };
        let (solved, status) = problem.solve_neutral([Some(target.neutral), None]);
        let joints = solved[0].unwrap();
        let pose = joints.forward(&chain).unwrap();
        let (normal, forward) = pose.palm.unwrap();
        assert!(matches!(status, SolveStatus::Feasible { .. }));
        assert!(joints.angles[3] < 5.0_f32.to_radians());
        assert!(forward.cross(normal).dot(Vec3::Z) > 0.99);
        assert!(forward.dot((pose.wrist - pose.elbow).normalize()) > 0.99);
    }
}

#[test]
fn neutral_solve_clears_hand_torso_contact_and_preserves_arm_lengths() {
    use crate::collision::{CapsuleCollider, Region};
    for side in [ArmSide::Left, ArmSide::Right] {
        let chain = chain(side);
        let target = ArmGoal::resting(&chain, Default::default()).unwrap();
        let initial = target.neutral.forward(&chain).unwrap();
        let torso = Entity::from_raw_u32(100).unwrap();
        let geometry = CollisionGeometry::new(
            vec![
                CapsuleCollider {
                    bone: torso,
                    region: Region::Torso,
                    endpoints: [initial.wrist + Vec3::Z * 0.05; 2],
                    radius: 0.08,
                },
                CapsuleCollider {
                    bone: chain.hand,
                    region: Region::Hand(side),
                    endpoints: [chain.rest.wrist.position; 2],
                    radius: 0.03,
                },
            ],
            &[],
        );
        let body = HashMap::from([(
            torso,
            BoneMotion {
                rotation: Quat::IDENTITY,
                translation: Vec3::ZERO,
            },
        )]);
        let problem = Problem {
            chains: [Some(&chain), None],
            goals: [Some(target), None],
            geometry: &geometry,
            body: &body,
            body_curve: None,
            tolerance: 64.0 * f32::EPSILON * chain.rest.total_arm_length,
        };
        let seed = [Some(target.neutral), None];
        assert!(!problem.feasible(&problem.evaluate(seed).unwrap()));
        let (solved, status) = problem.solve_neutral(seed);
        let evaluated = problem.evaluate(solved).unwrap();
        assert!(
            matches!(status, SolveStatus::Feasible { residual, .. } if residual > 0.0),
            "{status:?}"
        );
        assert!(problem.feasible(&evaluated));
        let arm = evaluated.arms[0].as_ref().unwrap();
        assert!(arm.wrist.distance(initial.wrist + Vec3::Z * 0.05) >= 0.11);
        assert!((arm.elbow.distance(arm.wrist) - chain.rest.forearm_length).abs() < 2.0e-6);
        assert!((arm.shoulder.distance(arm.elbow) - chain.rest.upper_arm_length).abs() < 2.0e-6);
    }
}

fn goal(joints: ArmJoints, chain: &ArmChainBinding) -> ArmGoal {
    let target = joints.forward(chain).unwrap();
    ArmGoal {
        wrist: target.wrist,
        elbow: Some(target.elbow),
        palm: target.palm,
        shoulder: Some(target.shoulder),
        weight: vtuber_core::arm_tracking::ArmBlendWeight {
            wrist: 1.0,
            pole: 1.0,
            palm: 1.0,
            fingers: 0.0,
        },
        shoulder_weight: 1.0,
        neutral: joints,
        neutral_girdle: None,
    }
}

#[test]
fn elbow_contact_swivel_reaches_clear_pose_while_preserving_hand_performance() {
    use crate::collision::{CapsuleCollider, Region};
    for side in [ArmSide::Left, ArmSide::Right] {
        let chain = chain(side);
        let initial = state(1.0, 0.6, -0.4, 1.2);
        let pose = initial.forward(&chain).unwrap();
        let axis = (pose.wrist - pose.shoulder).normalize();
        let offset = pose.elbow - pose.shoulder;
        let radial = (offset - axis * offset.dot(axis)).normalize();
        let torso = Entity::from_raw_u32(100).unwrap();
        let geometry = CollisionGeometry::new(
            vec![
                CapsuleCollider {
                    bone: torso,
                    region: Region::Torso,
                    endpoints: [pose.elbow - radial * 0.04; 2],
                    radius: 0.025,
                },
                CapsuleCollider {
                    bone: chain.lower_arm,
                    region: Region::Forearm(side),
                    endpoints: [chain.rest.elbow.position, chain.rest.wrist.position],
                    radius: chain.rest.forearm_length * 0.15,
                },
            ],
            &[],
        );
        let body = HashMap::from([(
            torso,
            BoneMotion {
                rotation: Quat::IDENTITY,
                translation: Vec3::ZERO,
            },
        )]);
        let mut target = goal(initial, &chain);
        target.palm = None;
        target.weight.palm = 0.0;
        target.neutral_girdle = Some(Vec2::new(initial.angles[7], initial.angles[8]));
        target.elbow =
            Some(geometry.elbow_target(pose.shoulder, pose.elbow, pose.wrist, side, &body));
        assert_eq!(target.wrist, pose.wrist);
        assert!(target.elbow.unwrap().distance(pose.elbow) > 0.01);
        let problem = Problem {
            chains: [Some(&chain), None],
            goals: [Some(target), None],
            geometry: &geometry,
            body: &body,
            body_curve: None,
            tolerance: 64.0 * f32::EPSILON * chain.rest.total_arm_length,
        };
        assert!(!problem.feasible(&problem.evaluate([Some(initial), None]).unwrap()));
        let (solved, status) = problem.solve([Some(initial), None], 96);
        let evaluated = problem.evaluate(solved).unwrap();
        assert!(
            matches!(status, SolveStatus::Feasible { .. }),
            "{side:?}: {status:?}"
        );
        assert!(problem.feasible(&evaluated));
        let arm = evaluated.arms[0].as_ref().unwrap();
        assert!(
            arm.wrist.distance(pose.wrist) < 0.002,
            "wrist drift {}",
            arm.wrist.distance(pose.wrist)
        );
        assert!((arm.shoulder.distance(arm.elbow) - chain.rest.upper_arm_length).abs() < 2e-6);
        assert!((arm.elbow.distance(arm.wrist) - chain.rest.forearm_length).abs() < 2e-6);
    }
}

#[test]
fn bilateral_solve_reaches_a_feasible_fk_goal_and_reports_unreachable_residual() {
    let left = chain(ArmSide::Left);
    let right = chain(ArmSide::Right);
    let target = state(1.0, 1.2, -0.4, 0.8);
    let body = HashMap::new();
    let geometry = CollisionGeometry::default();
    let mut problem = Problem {
        chains: [Some(&left), Some(&right)],
        goals: [Some(goal(target, &left)), Some(goal(target, &right))],
        body: &body,
        body_curve: None,
        geometry: &geometry,
        tolerance: 64.0 * f32::EPSILON * left.rest.total_arm_length,
    };
    let start = [Some(state(0.0, 0.2, 0.0, 0.0)); 2];
    let before = problem.evaluate(start).unwrap().error;
    let clock = std::time::Instant::now();
    let (solved, status) = problem.solve(start, 96);
    let result = problem.evaluate(solved).unwrap();
    eprintln!(
        "bilateral {:?}: {:?}, before={before}, after={}",
        clock.elapsed(),
        status,
        result.error
    );
    assert!(problem.feasible(&result));
    assert!(result.error < 1.0e-4);
    let [l, r] = result.arms;
    let l = l.unwrap();
    let r = r.unwrap();
    assert!(Vec3::new(-l.wrist.x, l.wrist.y, l.wrist.z).distance(r.wrist) < 2.0e-4);
    problem.goals[0].as_mut().unwrap().wrist = Vec3::splat(100.0);
    let (_, status) = problem.solve(solved, 16);
    assert!(matches!(status,SolveStatus::Feasible{residual,..} if residual>1.0));
}

#[test]
fn lost_arm_recovers_neutral_palm_and_joint_pose() {
    let chain = chain(ArmSide::Left);
    let neutral = state(0.1, 0.3, -0.4, 0.2);
    let mut target = goal(neutral, &chain);
    target.weight = Default::default();
    target.shoulder_weight = 0.0;
    target.neutral_girdle = Some(Vec2::new(neutral.angles[7], neutral.angles[8]));
    let geometry = CollisionGeometry::default();
    let body = HashMap::new();
    let problem = Problem {
        chains: [Some(&chain), None],
        goals: [Some(target), None],
        geometry: &geometry,
        body: &body,
        body_curve: None,
        tolerance: 64.0 * f32::EPSILON * chain.rest.total_arm_length,
    };
    let mut tracked = neutral;
    tracked.angles[2] += 0.5;
    tracked.angles[4] -= 0.3;
    let (recovered, status) = problem.solve([Some(tracked), None], 96);
    assert!(matches!(status, SolveStatus::Feasible { .. }));
    let expected = neutral.forward(&chain).unwrap();
    let recovered = recovered[0].unwrap().forward(&chain).unwrap();
    assert!(recovered.wrist.distance(expected.wrist) < 0.002);
    assert!(recovered.elbow.distance(expected.elbow) < 0.002);
    let (normal, forward) = recovered.palm.unwrap();
    let (wanted_normal, wanted_forward) = expected.palm.unwrap();
    assert!(normal.dot(wanted_normal) > 0.999);
    assert!(forward.dot(wanted_forward) > 0.999);
}

#[test]
fn unreachable_palm_rotation_does_not_pull_the_arm_away_from_its_positions() {
    let chain = chain(ArmSide::Left);
    let pose = state(1.3, 1.1, -0.3, 1.5);
    let original = pose.forward(&chain).unwrap();
    let mut target = goal(pose, &chain);
    let (normal, forward) = target.palm.unwrap();
    let turn = Quat::from_axis_angle(forward, 2.5);
    target.palm = Some((turn * normal, forward));
    let body = HashMap::new();
    let geometry = CollisionGeometry::default();
    let problem = Problem {
        chains: [Some(&chain), None],
        goals: [Some(target), None],
        body: &body,
        body_curve: None,
        geometry: &geometry,
        tolerance: 64.0 * f32::EPSILON * chain.rest.total_arm_length,
    };
    let start = [Some(pose), None];
    let (solved, _) = problem.solve(start, 96);
    let result = problem.evaluate(solved).unwrap();
    let arm = result.arms[0].as_ref().unwrap();
    assert!(problem.feasible(&result));
    assert!(arm.wrist.distance(original.wrist) < chain.rest.total_arm_length * 0.05);
    assert!(arm.elbow.distance(original.elbow) < chain.rest.total_arm_length * 0.05);
    assert!(result.error < problem.evaluate(start).unwrap().error);
}

#[test]
fn narrow_palm_recovers_a_large_forearm_rotation_without_moving_the_elbow() {
    for side in [ArmSide::Left, ArmSide::Right] {
        let mut chain = chain(side);
        let wrist = chain.rest.wrist.position;
        let sign = if side == ArmSide::Left { 1.0 } else { -1.0 };
        chain
            .finger_rest
            .index
            .proximal
            .as_mut()
            .unwrap()
            .rest
            .position = wrist + Vec3::new(sign * 0.06, 0.0, 0.002);
        chain
            .finger_rest
            .little
            .proximal
            .as_mut()
            .unwrap()
            .rest
            .position = wrist + Vec3::new(sign * 0.06, 0.0, -0.002);
        let mut wanted = state(1.0, 1.2, -0.4, 1.0);
        wanted.angles[4] = -1.3;
        let mut initial = wanted;
        initial.angles[4] = 1.3;
        let expected = wanted.forward(&chain).unwrap();
        let first = initial.forward(&chain).unwrap();
        assert!(first.palm.unwrap().0.dot(expected.palm.unwrap().0) < -0.5);
        let mut target = goal(wanted, &chain);
        target.neutral = initial;
        target.neutral_girdle = Some(Vec2::new(initial.angles[7], initial.angles[8]));
        let geometry = CollisionGeometry::default();
        let body = HashMap::new();
        let problem = Problem {
            chains: [Some(&chain), None],
            goals: [Some(target), None],
            geometry: &geometry,
            body: &body,
            body_curve: None,
            tolerance: 64.0 * f32::EPSILON * chain.rest.total_arm_length,
        };
        let (solved, status) = problem.solve([Some(initial), None], 96);
        assert!(matches!(status, SolveStatus::Feasible { .. }));
        let result = solved[0].unwrap().forward(&chain).unwrap();
        assert!(result.palm.unwrap().0.dot(expected.palm.unwrap().0) > 0.999);
        assert!(result.palm.unwrap().1.dot(expected.palm.unwrap().1) > 0.999);
        assert!(result.elbow.distance(expected.elbow) < 0.002);
        assert!(result.wrist.distance(expected.wrist) < 0.002);
    }
}

#[test]
fn restoring_one_arm_contact_does_not_starve_the_other_arms_target() {
    use crate::collision::{CapsuleCollider, Region};
    let left = chain(ArmSide::Left);
    let right = chain(ArmSide::Right);
    let start = state(0.9, 1.0, -0.3, 1.2);
    let right_pose = start.forward(&right).unwrap();
    let torso = Entity::from_raw_u32(100).unwrap();
    let body = HashMap::from([(
        torso,
        BoneMotion {
            rotation: Quat::IDENTITY,
            translation: Vec3::ZERO,
        },
    )]);
    let geometry = CollisionGeometry::new(
        vec![
            CapsuleCollider {
                bone: torso,
                region: Region::Torso,
                endpoints: [right_pose.wrist + Vec3::Z * 0.139; 2],
                radius: 0.1,
            },
            CapsuleCollider {
                bone: right.hand,
                region: Region::Hand(ArmSide::Right),
                endpoints: [right.rest.wrist.position; 2],
                radius: 0.04,
            },
        ],
        &[],
    );
    let problem = Problem {
        chains: [Some(&left), Some(&right)],
        goals: [
            Some(goal(state(1.4, 1.2, -0.3, 1.2), &left)),
            Some(goal(start, &right)),
        ],
        geometry: &geometry,
        body: &body,
        body_curve: None,
        tolerance: 64.0 * f32::EPSILON * left.rest.total_arm_length,
    };
    let before = problem.evaluate([Some(start); 2]).unwrap();
    assert!(!problem.feasible(&before));
    let (solved, status) = problem.solve([Some(start); 2], 1);
    let after = problem.evaluate(solved).unwrap();
    assert!(matches!(status, SolveStatus::Feasible { .. }));
    assert!(problem.feasible(&after));
    let target = problem.goals[0].unwrap().wrist;
    assert!(
        after.arms[0].as_ref().unwrap().wrist.distance(target)
            < before.arms[0].as_ref().unwrap().wrist.distance(target) - 0.01
    );
}

fn serial_linearize(
    problem: &Problem<'_>,
    current: [Option<ArmJoints>; 2],
    evaluation: &Evaluation,
) -> Linearization {
    let mut output = Linearization {
        tasks: nalgebra::DMatrix::zeros(evaluation.residuals.len(), 18),
        constraints: nalgebra::DMatrix::zeros(evaluation.constraints.len(), 18),
    };
    for (index, (mut tasks, mut constraints)) in output
        .tasks
        .column_iter_mut()
        .zip(output.constraints.column_iter_mut())
        .enumerate()
    {
        problem
            .linearize_column(
                current,
                evaluation,
                index,
                tasks.as_mut_slice(),
                constraints.as_mut_slice(),
            )
            .unwrap();
    }
    output
}

fn with_linearization_fixture(check: impl FnOnce(Problem<'_>, [Option<ArmJoints>; 2])) {
    use crate::collision::{CapsuleCollider, Region};
    let left = chain(ArmSide::Left);
    let right = chain(ArmSide::Right);
    let current = [Some(state(0.8, 0.7, -0.3, 1.0)); 2];
    let torso = Entity::from_raw_u32(100).unwrap();
    let mut capsules = vec![CapsuleCollider {
        bone: torso,
        region: Region::Torso,
        endpoints: [current[0].unwrap().forward(&left).unwrap().wrist + Vec3::Z * 0.05; 2],
        radius: 0.08,
    }];
    for chain in [left, right] {
        capsules.extend([
            CapsuleCollider {
                bone: chain.upper_arm,
                region: Region::Upper(chain.side),
                endpoints: [chain.rest.upper_arm.position, chain.rest.elbow.position],
                radius: 0.06,
            },
            CapsuleCollider {
                bone: chain.lower_arm,
                region: Region::Forearm(chain.side),
                endpoints: [chain.rest.elbow.position, chain.rest.wrist.position],
                radius: 0.039,
            },
            CapsuleCollider {
                bone: chain.hand,
                region: Region::Hand(chain.side),
                endpoints: [
                    chain.rest.wrist.position,
                    chain.rest.wrist.position + Vec3::X * 0.06,
                ],
                radius: 0.02,
            },
        ]);
    }
    let geometry = CollisionGeometry::new(capsules, &[]);
    let mut body = HashMap::from([(
        torso,
        BoneMotion {
            rotation: Quat::IDENTITY,
            translation: Vec3::ZERO,
        },
    )]);
    body.extend(current[0].unwrap().forward(&left).unwrap().motion);
    body.extend(current[1].unwrap().forward(&right).unwrap().motion);
    let problem = Problem {
        chains: [Some(&left), Some(&right)],
        goals: [
            Some(goal(state(1.0, 0.8, -0.2, 1.2), &left)),
            Some(goal(state(1.0, 0.8, -0.2, 1.2), &right)),
        ],
        geometry: &geometry,
        body: &body,
        body_curve: None,
        tolerance: 64.0 * f32::EPSILON * left.rest.total_arm_length,
    };
    check(problem, current);
}

#[test]
fn matrix_columns_match_serial_differences_including_missing_arm_and_palm() {
    with_linearization_fixture(|problem, current| {
        let evaluation = problem.evaluate(current).unwrap();
        assert!(evaluation.contact_pairs.iter().any(|active| *active));
        let compare = |problem: Problem<'_>, current| {
            let evaluation = problem.evaluate(current).unwrap();
            let serial = serial_linearize(&problem, current, &evaluation);
            let parallel = problem.linearize(current, &evaluation).unwrap();
            assert_eq!(serial.tasks, parallel.tasks);
            assert_eq!(serial.constraints, parallel.constraints);
        };
        compare(problem, current);
        compare(
            Problem {
                chains: [problem.chains[0], None],
                ..problem
            },
            [current[0], None],
        );
        let mut left = *problem.chains[0].unwrap();
        left.finger_rest = Default::default();
        compare(
            Problem {
                chains: [Some(&left), None],
                ..problem
            },
            [current[0], None],
        );
    });
}

#[test]
#[ignore = "Release timing experiment; run alone under a three-core process affinity"]
fn compare_linearization_execution() {
    use std::{hint::black_box, time::Instant};
    let pool = bevy::tasks::AsyncComputeTaskPool::get_or_init(|| {
        bevy::tasks::TaskPoolBuilder::new().num_threads(3).build()
    });
    bevy::tasks::futures_lite::future::block_on(pool.spawn(async {
        with_linearization_fixture(|problem, current| {
            let evaluation = problem.evaluate(current).unwrap();
            let run = |parallel| {
                if parallel {
                    problem
                        .linearize(black_box(current), black_box(&evaluation))
                        .unwrap()
                } else {
                    serial_linearize(&problem, black_box(current), black_box(&evaluation))
                }
            };
            for _ in 0..20 {
                black_box(run(false));
                black_box(run(true));
            }
            for batch in 0..7 {
                // Alternate order to avoid assigning all warm-cache runs to one mode.
                for parallel in if batch % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let start = Instant::now();
                    for _ in 0..200 {
                        black_box(run(parallel));
                    }
                    println!(
                        "batch={batch} parallel={parallel} us_per_linearization={:.3}",
                        start.elapsed().as_secs_f64() * 1.0e6 / 200.0
                    );
                }
            }
        });
    }));
}
