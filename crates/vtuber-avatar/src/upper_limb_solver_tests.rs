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
    problem: &PreparedProblem<'_>,
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
            let problem = problem.prepare().unwrap();
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
            let problem = problem.prepare().unwrap();
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

// Full recomputation before the FK optimizations, retained only for comparison.
pub(super) fn uncached_kinematics_near(
    problem: &Problem<'_>,
    state: [Option<ArmJoints>; 2],
    reference: Option<&Evaluation>,
) -> Result<Evaluation, CollisionError> {
    let mut arms = [None, None];
    let mut residuals = Vec::new();
    let mut residual_counts = [0; 2];
    let mut previous_count = 0;
    for (side, (((slot, chain), state), goal)) in arms
        .iter_mut()
        .zip(problem.chains)
        .zip(state)
        .zip(problem.goals)
        .enumerate()
    {
        let Some((chain, state)) = chain.zip(state) else {
            continue;
        };
        let pose = state
            .forward(chain)
            .ok_or(CollisionError::InvalidGeometry)?;
        let reference_palm = reference
            .and_then(|r| r.arms.get(side))
            .and_then(Option::as_ref)
            .and_then(|arm| arm.palm);
        let palm_error_for_target = |wanted| {
            let current = pose.palm?;
            match reference_palm {
                Some(reference) => crate::tracked_arm::palm_orientation_error_near_reference(
                    chain, current, wanted, reference,
                ),
                None => crate::tracked_arm::palm_orientation_error(chain, current, wanted),
            }
        };
        if let Some(goal) = goal {
            let scale = chain.rest.total_arm_length;
            let mut distance = |a: Vec3, b: Vec3, weight: f32| {
                residuals.extend(((a - b) * weight.sqrt()).to_array().map(f64::from))
            };
            distance(pose.wrist / scale, goal.wrist / scale, goal.weight.wrist);
            if let Some(elbow) = goal.elbow {
                distance(pose.elbow / scale, elbow / scale, goal.weight.pole);
            }
            if let Some(error) = goal.palm.and_then(palm_error_for_target) {
                distance(error / scale, Vec3::ZERO, goal.weight.palm);
            }
            if let Some(shoulder) = goal.shoulder {
                distance(
                    pose.shoulder / scale,
                    shoulder / scale,
                    goal.shoulder_weight,
                );
            }
            // A missing channel returns to the neutral FK task. Solve it
            // before the path update; do not re-IK a damped Cartesian
            // wrist or penalize every unobserved joint as a measurement.
            let neutral = goal
                .neutral
                .forward(chain)
                .ok_or(CollisionError::InvalidGeometry)?;
            distance(
                pose.wrist / scale,
                neutral.wrist / scale,
                1.0 - goal.weight.wrist,
            );
            // Startup uses the explicit profile goals above. Only after
            // admission do missing channels return to the solved neutral
            // FK, rather than the bounded analytic starting point.
            if goal.neutral_girdle.is_some() {
                distance(
                    pose.elbow / scale,
                    neutral.elbow / scale,
                    1.0 - goal.weight.pole,
                );
            }
            // Once the entire arm is unobserved, recover its neutral palm
            // as well as elbow/wrist positions. Position-only recovery
            // leaves humeral axial rotation undetermined on a straight arm.
            let unobserved = 1.0
                - goal
                    .weight
                    .wrist
                    .max(goal.weight.pole)
                    .max(goal.weight.palm);
            if let Some(error) = neutral.palm.and_then(palm_error_for_target) {
                distance(error / scale, Vec3::ZERO, unobserved);
            }
            // A resting profile must move the arm to clear the body, not
            // bend the wrist or twist the humerus to keep pursuing its
            // unreachable Cartesian wrist. Preserve the neutral joint
            // articulation while resolving startup contacts.
            if goal.neutral_girdle.is_none() {
                residuals.extend(
                    state
                        .angles
                        .iter()
                        .zip(goal.neutral.angles)
                        .skip(2)
                        .take(2)
                        .map(|(q, neutral)| f64::from(q - neutral)),
                );
            }
            // Missing palm orientation returns only radioulnar/wrist
            // articulation to rest. A neutral world-space palm would
            // falsely oppose an observed arm lift or thorax rotation.
            residuals.extend(
                state
                    .angles
                    .iter()
                    .zip(goal.neutral.angles)
                    .skip(4)
                    .take(3)
                    .map(|(q, neutral)| {
                        let weight = if goal.neutral_girdle.is_none() {
                            1.0
                        } else {
                            1.0 - goal.weight.palm
                        };
                        f64::from((q - neutral) * weight.sqrt())
                    }),
            );
            // Unobserved SC motion is estimated from the published
            // rhythm. This is a model prior, not a measured scapula pose.
            let [p, e, _, _, _, _, _, protract, elevate] = state.angles;
            let [rp, re, ..] = crate::girdle::rhythm(p, e);
            let Vec2 { x: np, y: ne } = goal.neutral_girdle.unwrap_or(Vec2::new(rp, re));
            let observed = goal
                .weight
                .wrist
                .max(goal.weight.pole)
                .max(goal.weight.palm);
            residuals.extend(
                [
                    protract - (np + observed * (rp - np)),
                    elevate - (ne + observed * (re - ne)),
                ]
                .map(|v| f64::from(v * (1.0 - goal.shoulder_weight).sqrt())),
            );
        }
        residual_counts[side] = residuals.len() - previous_count;
        previous_count = residuals.len();
        *slot = Some(Arc::new(pose));
    }
    let joint_margin = arms
        .iter()
        .flatten()
        .map(|a| a.joint_margin)
        .fold(f32::INFINITY, f32::min);
    let error = residuals.iter().map(|v| v * v).sum();
    Ok(Evaluation {
        arms,
        clearance: f32::INFINITY,
        joint_margin,
        error,
        residuals,
        residual_counts,
        constraints: Vec::new(),
        contact_pairs: Vec::new(),
    })
}

fn assert_same_evaluation(a: &Evaluation, b: &Evaluation) {
    assert_eq!(a.residual_counts, b.residual_counts);
    assert_eq!(a.residuals, b.residuals);
    assert_eq!(a.constraints, b.constraints);
    assert_eq!(a.contact_pairs, b.contact_pairs);
    assert_eq!(a.clearance, b.clearance);
    assert_eq!(a.joint_margin, b.joint_margin);
    assert_eq!(a.error, b.error);
    for (a, b) in a.arms.iter().zip(&b.arms) {
        match (a, b) {
            (Some(a), Some(b)) => {
                assert_eq!(a.shoulder, b.shoulder);
                assert_eq!(a.elbow, b.elbow);
                assert_eq!(a.wrist, b.wrist);
                assert_eq!(a.palm, b.palm);
                assert_eq!(a.motion, b.motion);
                assert_eq!(a.joint_margins, b.joint_margins);
            }
            (None, None) => {}
            _ => panic!("arm presence differs"),
        }
    }
}

#[test]
fn cached_arm_differences_match_full_fk_residuals_contacts_jacobians_and_solutions() {
    with_linearization_fixture(|problem, initial| {
        let mut no_palm = *problem.chains[0].unwrap();
        no_palm.finger_rest = Default::default();
        for case in 0..8 {
            let mut problem = problem;
            let mut current = initial;
            match case {
                0 => {} // startup neutral tasks, both arms, body and arm contacts
                1 => {
                    for goal in problem.goals.iter_mut().flatten() {
                        goal.neutral_girdle =
                            Some(Vec2::new(goal.neutral.angles[7], goal.neutral.angles[8]));
                        goal.weight = vtuber_core::arm_tracking::ArmBlendWeight {
                            wrist: 0.7,
                            pole: 0.3,
                            palm: 0.2,
                            fingers: 0.0,
                        };
                        goal.shoulder_weight = 0.4;
                    }
                }
                2 => {
                    for goal in problem.goals.iter_mut().flatten() {
                        goal.neutral_girdle = Some(Vec2::ZERO);
                        goal.weight = Default::default();
                        goal.elbow = None;
                        goal.palm = None;
                        goal.shoulder = None;
                    }
                }
                3 | 4 => {
                    for (side, goal) in problem.goals.iter_mut().enumerate() {
                        let chain = problem.chains[side].unwrap();
                        let (normal, forward) =
                            current[side].unwrap().forward(chain).unwrap().palm.unwrap();
                        let half_turn = Quat::from_axis_angle(
                            forward,
                            std::f32::consts::PI + if case == 3 { -1.0e-4 } else { 1.0e-4 },
                        );
                        goal.as_mut().unwrap().palm = Some((half_turn * normal, forward));
                    }
                }
                5 => {
                    problem.chains[0] = None;
                    current[0] = None;
                }
                6 => {
                    problem.chains[0] = Some(&no_palm);
                }
                7 => {
                    problem.goals[1] = None;
                    current[0].as_mut().unwrap().angles[3] = 0.0;
                    current[1].as_mut().unwrap().angles[6] = crate::upper_limb::BOUNDS[6].1;
                }
                _ => unreachable!(),
            }
            let optimized = problem.prepare().unwrap();
            let mut full = problem.prepare().unwrap();
            full.full_recomputation = true;
            let evaluation = optimized.evaluate(current).unwrap();
            let reference_evaluation = full.evaluate(current).unwrap();
            assert_same_evaluation(&evaluation, &reference_evaluation);
            for side in 0..2 {
                if current[side].is_none() {
                    continue;
                }
                for index in 0..9 {
                    if matches!(index, 5 | 6)
                        && evaluation.arms[side].as_ref().unwrap().palm.is_none()
                    {
                        continue;
                    }
                    for delta in [-f32::EPSILON.cbrt(), f32::EPSILON.cbrt()] {
                        let mut sample = current;
                        let arm = sample[side].as_mut().unwrap();
                        arm.angles[index] += delta;
                        arm.bound();
                        let partial = optimized
                            .differential_kinematics(sample, side, &evaluation)
                            .unwrap();
                        let whole = full.kinematics_near(sample, Some(&evaluation)).unwrap();
                        assert_same_evaluation(&partial, &whole);
                        let partial = optimized.evaluate_geometry(partial, None).unwrap();
                        let whole = full.evaluate_geometry(whole, None).unwrap();
                        assert_same_evaluation(&partial, &whole);
                        assert_eq!(problem.feasible(&partial), problem.feasible(&whole));
                    }
                }
            }
            let a = optimized.linearize(current, &evaluation).unwrap();
            let b = full.linearize(current, &reference_evaluation).unwrap();
            assert_eq!(a.tasks, b.tasks, "task Jacobian case {case}");
            assert_eq!(
                a.constraints, b.constraints,
                "constraint Jacobian case {case}"
            );
            let (a, sa) = optimized.solve_at_resolution(current, 2, 1.0e-3);
            let (b, sb) = full.solve_at_resolution(current, 2, 1.0e-3);
            assert_eq!(a, b, "solution case {case}");
            assert_eq!(format!("{sa:?}"), format!("{sb:?}"));
            assert_same_evaluation(&optimized.evaluate(a).unwrap(), &full.evaluate(b).unwrap());
        }
    });
}

#[test]
#[ignore = "Release FK reuse experiment; run alone under a three-core process affinity"]
fn compare_fk_reuse() {
    use std::{hint::black_box, time::Instant};
    let pool = bevy::tasks::AsyncComputeTaskPool::get_or_init(|| {
        bevy::tasks::TaskPoolBuilder::new().num_threads(3).build()
    });
    bevy::tasks::futures_lite::future::block_on(pool.spawn(async {
        with_linearization_fixture(|problem, current| {
            let mut uncached = problem.prepare().unwrap();
            uncached.full_recomputation = true;
            let mut cached = problem.prepare().unwrap();
            cached.whole_arm_differences = true;
            let partial = problem.prepare().unwrap();
            let variants = [
                ("full", uncached),
                ("neutral_cache", cached),
                ("single_arm", partial),
            ];
            let evaluations = variants
                .each_ref()
                .map(|(_, p)| p.evaluate(current).unwrap());
            let mut jacobian_us = [Vec::new(), Vec::new(), Vec::new()];
            let mut solve_us = [Vec::new(), Vec::new(), Vec::new()];
            for sample in 0..80 {
                // Rotate the order, including warmup, with all variants on one fixture.
                for offset in 0..3 {
                    let i = (sample + offset) % 3;
                    let start = Instant::now();
                    black_box(
                        variants[i]
                            .1
                            .linearize(black_box(current), black_box(&evaluations[i]))
                            .unwrap(),
                    );
                    let jacobian = start.elapsed().as_secs_f64() * 1.0e6;
                    let start = Instant::now();
                    black_box(
                        variants[i]
                            .1
                            .solve_at_resolution(black_box(current), 2, 1.0e-3),
                    );
                    let solve = start.elapsed().as_secs_f64() * 1.0e6;
                    if sample >= 20 {
                        jacobian_us[i].push(jacobian);
                        solve_us[i].push(solve);
                    }
                }
            }
            for (i, (name, _)) in variants.iter().enumerate() {
                for (metric, samples) in [
                    ("jacobian", &mut jacobian_us[i]),
                    ("solve", &mut solve_us[i]),
                ] {
                    samples.sort_by(f64::total_cmp);
                    println!(
                        "variant={name} metric={metric} samples={} p50_us={:.3} p95_us={:.3}",
                        samples.len(),
                        samples[samples.len() / 2],
                        samples[samples.len() * 95 / 100]
                    );
                }
            }
        });
    }));
}
