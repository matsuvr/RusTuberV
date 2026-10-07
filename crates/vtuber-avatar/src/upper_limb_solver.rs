//! Bilateral sequential quadratic programming. Feasibility has lexical
//! priority over observation error; a residual never buys a limit violation.

use crate::{
    arm::ArmChainBinding,
    collision::{BoneMotion, CollisionError, CollisionGeometry},
    upper_limb::{ArmCandidate, ArmJoints},
};
use bevy::prelude::*;

pub(crate) const MAX_STEP_RADIANS: f32 = 0.25;
use std::collections::HashMap;

struct Linearization {
    tasks: nalgebra::DMatrix<f64>,
    constraints: nalgebra::DMatrix<f64>,
}

/// Solve the strictly convex SQP subproblem to its active-set optimum.
/// Fixed dual sweeps can leave opposed, nearly parallel contacts violated;
/// nonlinear backtracking then mistakes the failed tangent for convergence.
fn quadratic_step(
    hessian: nalgebra::DMatrix<f64>,
    linear: nalgebra::DVector<f64>,
    rows: &[[f64; 18]],
    margins: &[f64],
) -> Option<[f32; 18]> {
    use nalgebra::{DMatrix, DVector};
    // The task QP has a feasible zero step. During contact restoration only
    // the joint/trust bounds enter this QP; restoration is its objective.
    if margins.iter().any(|m| *m < 0.0) {
        return None;
    }
    let factor = hessian.cholesky()?;
    let free = factor.solve(&(-linear));
    let directions: Vec<_> = rows
        .iter()
        .map(|r| factor.solve(&DVector::from_column_slice(r)))
        .collect();
    let mut step = DVector::zeros(18);
    let mut active = Vec::<usize>::new();
    let resolution = 64.0 * f64::EPSILON.sqrt();
    for _ in 0..256 {
        let a = DMatrix::from_fn(active.len(), 18, |i, j| {
            active
                .get(i)
                .and_then(|i| rows.get(*i))
                .and_then(|r| r.get(j))
                .copied()
                .unwrap_or(0.0)
        });
        let inverse_normals = DMatrix::from_fn(18, active.len(), |i, j| {
            active
                .get(j)
                .and_then(|j| directions.get(*j))
                .and_then(|d| d.get(i))
                .copied()
                .unwrap_or(0.0)
        });
        let free_direction = &free - &step;
        let multipliers = if active.is_empty() {
            DVector::zeros(0)
        } else {
            (&a * &inverse_normals)
                .svd(true, true)
                .solve(&(-&a * &free_direction), f64::EPSILON.sqrt())
                .ok()?
        };
        let direction = free_direction + &inverse_normals * &multipliers;
        if direction.amax() < resolution {
            let remove = multipliers
                .iter()
                .enumerate()
                .filter(|(_, m)| **m < -resolution)
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i);
            if let Some(remove) = remove {
                active.remove(remove);
                continue;
            }
            return Some(std::array::from_fn(|i| {
                step.get(i).copied().unwrap_or(0.0) as f32
            }));
        }
        let mut amount = 1.0_f64;
        let mut blocker = None;
        for (i, (row, margin)) in rows.iter().zip(margins).enumerate() {
            if active.contains(&i) {
                continue;
            }
            let slope = row.iter().zip(&direction).map(|(a, b)| a * b).sum::<f64>();
            let normal = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            // An equality direction has zero slope; do not add a redundant
            // contact because its roundoff is a tiny negative number.
            if slope >= -f64::EPSILON.sqrt() * normal * direction.norm() {
                continue;
            }
            let slack = row.iter().zip(&step).map(|(a, b)| a * b).sum::<f64>() + margin;
            let candidate = slack.max(0.0) / -slope;
            if candidate < amount {
                amount = candidate;
                blocker = Some(i);
            }
        }
        step.axpy(amount, &direction, 1.0);
        if let Some(blocker) = blocker {
            active.push(blocker);
        }
    }
    // A work-limit exit is not an optimum. The caller retains its admitted
    // pose rather than treating an unfinished, violating QP as convergence.
    None
}

/// Least-norm normal correction to the violated nonlinear boundaries.
/// The numerical interior target is in the same dimensionless coordinates
/// as the existing constraint rounding margin, not an anatomical allowance.
fn second_order_correction(
    jacobian: &nalgebra::DMatrix<f64>,
    constraints: &[f64],
) -> Option<[f32; 18]> {
    let rows: Vec<_> = constraints
        .iter()
        .enumerate()
        .filter(|(_, c)| **c < 0.0)
        .collect();
    let a = nalgebra::DMatrix::from_fn(rows.len(), 18, |i, j| {
        rows.get(i)
            .and_then(|(row, _)| jacobian.get((*row, j)))
            .copied()
            .unwrap_or(0.0)
    });
    let b = nalgebra::DVector::from_iterator(
        rows.len(),
        rows.iter()
            .map(|(_, c)| 64.0 * f64::from(f32::EPSILON) - **c),
    );
    let correction = a.svd(true, true).solve(&b, f64::EPSILON.sqrt()).ok()?;
    Some(std::array::from_fn(|i| {
        correction.get(i).copied().unwrap_or(0.0) as f32
    }))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ArmGoal {
    pub wrist: Vec3,
    pub elbow: Option<Vec3>,
    pub palm: Option<(Vec3, Vec3)>,
    pub shoulder: Option<Vec3>,
    pub weight: vtuber_core::arm_tracking::ArmBlendWeight,
    pub shoulder_weight: f32,
    pub neutral: ArmJoints,
    /// SC return coordinates after the profile's neutral has been solved.
    pub neutral_girdle: Option<Vec2>,
}

impl ArmGoal {
    /// Resting targets share the moving shoulder origin. A wrist target alone
    /// lets the optimizer trade elbow flexion and wrist bend against reach.
    pub fn resting(chain: &ArmChainBinding, profile: crate::arm::ArmPoseProfile) -> Option<Self> {
        let target = crate::arm::default_arm_target(chain, profile).ok()?;
        let analytic =
            crate::arm::solve_two_bone_arm(crate::arm::ArmIkInput::from_chain(chain, target))
                .ok()?;
        let mut neutral = ArmJoints::from_solution(chain, analytic)?;
        neutral.rest_curl = profile.finger_curl_radians;
        let pose = neutral.forward(chain)?;
        let offset = pose.shoulder - chain.rest.upper_arm.position;
        let direction = (target.wrist - chain.rest.upper_arm.position).try_normalize()?;
        let palm = crate::arm::rest_palm_normal(chain)
            .map(|_| (Vec3::Z.cross(direction).normalize(), direction));
        Some(Self {
            wrist: target.wrist + offset,
            elbow: Some(analytic.elbow + offset),
            palm,
            shoulder: None,
            weight: vtuber_core::arm_tracking::ArmBlendWeight {
                wrist: 1.0,
                pole: 1.0,
                palm: 1.0,
                fingers: 0.0,
            },
            shoulder_weight: 0.0,
            neutral,
            neutral_girdle: None,
        })
    }
}

#[cfg(test)]
mod tests {
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
            let mut normalized = chain(side);
            let mut authored = normalized;
            for rest in [
                &mut normalized.rest.upper_arm,
                &mut normalized.rest.elbow,
                &mut normalized.rest.wrist,
            ] {
                rest.global_rotation = Quat::IDENTITY;
                rest.local_rotation = Quat::IDENTITY;
            }
            for (i, rest) in [
                &mut authored.rest.upper_arm,
                &mut authored.rest.elbow,
                &mut authored.rest.wrist,
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
            let a = solve(&normalized);
            let b = solve(&authored);
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
    fn initial_position_yields_to_contacts_without_shrinking_the_body() {
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
            assert!(
                (arm.shoulder.distance(arm.elbow) - chain.rest.upper_arm_length).abs() < 2.0e-6
            );
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
}

#[derive(Clone, Copy)]
pub(crate) struct Problem<'a> {
    pub chains: [Option<&'a ArmChainBinding>; 2],
    pub goals: [Option<ArmGoal>; 2],
    pub geometry: &'a CollisionGeometry,
    pub body: &'a HashMap<Entity, BoneMotion>,
    pub body_curve: Option<&'a crate::upper_limb_body::BodyCurve>,
    /// World-length rounding allowance, scaled to this avatar's arm length.
    pub tolerance: f32,
}

/// Result of the current constrained upper-limb solve and transition.
#[derive(Component, Clone, Copy, Debug)]
pub enum SolveStatus {
    /// The search is in progress. Keep any previously admitted pose visible.
    Solving,
    /// A constraint-admissible pose; residual is the dimensionless task norm.
    Feasible {
        /// Dimensionless weighted observation residual.
        residual: f32,
        /// The local search reached its numerical stopping criterion.
        converged: bool,
    },
    /// No feasible candidate was found within the local solve budget.
    NoFeasibleSolution,
    /// Render geometry could not be used for collision checks.
    InvalidGeometry(CollisionError),
    /// The current transition did not pass the continuous ROM/skin checks.
    BlockedPath,
}

#[derive(Clone, Debug)]
pub(crate) struct Evaluation {
    pub arms: [Option<ArmCandidate>; 2],
    pub clearance: f32,
    pub joint_margin: f32,
    pub error: f64,
    residuals: Vec<f64>,
    constraints: Vec<f64>,
    contact_pairs: Vec<bool>,
}

impl Problem<'_> {
    pub(crate) fn kinematics(
        &self,
        state: [Option<ArmJoints>; 2],
    ) -> Result<Evaluation, CollisionError> {
        self.kinematics_near(state, None)
    }

    fn kinematics_near(
        &self,
        state: [Option<ArmJoints>; 2],
        reference: Option<&Evaluation>,
    ) -> Result<Evaluation, CollisionError> {
        let mut arms = [None, None];
        let mut residuals = Vec::new();
        for (side, (((slot, chain), state), goal)) in arms
            .iter_mut()
            .zip(self.chains)
            .zip(state)
            .zip(self.goals)
            .enumerate()
        {
            let Some((chain, state)) = chain.zip(state) else {
                continue;
            };
            let pose = state.forward(chain).ok_or(CollisionError::InvalidMesh)?;
            let reference_palm = reference
                .and_then(|r| r.arms.get(side))
                .and_then(Option::as_ref)
                .and_then(|arm| arm.palm);
            let orientation_error = |wanted| {
                let current = pose.palm?;
                match reference_palm {
                    Some(reference) => crate::tracked_arm::palm_orientation_derivative_error(
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
                if let Some(error) = goal.palm.and_then(orientation_error) {
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
                    .ok_or(CollisionError::InvalidMesh)?;
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
                if let Some(error) = neutral.palm.and_then(orientation_error) {
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
            *slot = Some(pose);
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
            constraints: Vec::new(),
            contact_pairs: Vec::new(),
        })
    }

    pub fn evaluate(&self, state: [Option<ArmJoints>; 2]) -> Result<Evaluation, CollisionError> {
        self.evaluate_geometry(self.kinematics(state)?, None)
    }

    fn evaluate_geometry(
        &self,
        mut evaluation: Evaluation,
        differential: Option<&[bool]>,
    ) -> Result<Evaluation, CollisionError> {
        let length = self
            .chains
            .into_iter()
            .flatten()
            .map(|c| c.rest.total_arm_length)
            .fold(0.0_f32, f32::max);
        let (margins, active) = self.geometry.solver_margins(
            |bone| {
                evaluation
                    .arms
                    .iter()
                    .flatten()
                    .find_map(|arm| arm.motion.get(&bone).copied())
                    .or_else(|| self.body.get(&bone).copied())
            },
            // Include near contacts before the finite-difference stencil
            // reaches them. This selects work, not an anatomical clearance;
            // every nonlinear candidate still checks all pairs.
            self.contact_offset() + length * f32::EPSILON.cbrt(),
            differential,
        )?;
        evaluation.clearance = margins.iter().copied().fold(f32::INFINITY, f32::min);
        evaluation
            .constraints
            // Solve on the positive side of the same rounding allowance used
            // by path admission. Spending that allowance as penetration here
            // leaves no representable interval for the next contact motion.
            .extend(margins.into_iter().zip(&active).map(|(m, active)| {
                if *active { f64::from((m - self.contact_offset() - 4.0 * self.tolerance) / length) } else { 1.0 }
            }));
        evaluation.contact_pairs = active;
        evaluation.constraints.extend(
            evaluation
                .arms
                .iter()
                .flatten()
                // Keep each boundary's derivative. Differentiating only the
                // minimum margin changes the active plane inside the stencil
                // and loses the feasible tangent where two planes meet.
                .flat_map(|a| a.joint_margins.map(|m| f64::from(m - 256.0 * f32::EPSILON))),
        );
        Ok(evaluation)
    }

    fn linearize(
        &self,
        current: [Option<ArmJoints>; 2],
        evaluation: &Evaluation,
    ) -> Option<Linearization> {
        use nalgebra::DMatrix;
        let h = f32::EPSILON.cbrt();
        let mut task_columns = Vec::new();
        let mut constraint_columns = Vec::new();
        // These jobs belong to the asynchronous pose solve. Sharing the ECS
        // frame pool lets a render-critical task pick up a long collision job.
        let mut columns = bevy::tasks::AsyncComputeTaskPool::get_or_init(
            bevy::tasks::TaskPool::default,
        )
        .scope(|scope| {
            for index in 0..18 {
                scope.spawn(async move {
                    let mut task_column = Vec::new();
                    let mut constraint_column = Vec::new();
                    let mut minus = current;
                    let mut plus = current;
                    let change = |state: &mut [Option<ArmJoints>; 2], delta: f32| -> Option<f32> {
                        if matches!(index % 9, 5 | 6)
                            && evaluation.arms.get(index / 9)?.as_ref()?.palm.is_none()
                        {
                            return None;
                        }
                        let arm = state.get_mut(index / 9)?.as_mut()?;
                        *arm.angles.get_mut(index % 9)? += delta;
                        arm.bound();
                        arm.angles.get(index % 9).copied()
                    };
                    let span = change(&mut plus, h)
                        .zip(change(&mut minus, -h))
                        .map(|(a, b)| a - b);
                    if let Some(span) = span.filter(|v| *v > 0.0) {
                        let a = self.kinematics_near(minus, Some(evaluation)).ok()?;
                        let b = self.kinematics_near(plus, Some(evaluation)).ok()?;
                        let moved = [&a, &b]
                            .into_iter()
                            .flat_map(|sample| {
                                sample
                                    .arms
                                    .iter()
                                    .zip(&evaluation.arms)
                                    .filter_map(|(sample, base)| sample.as_ref().zip(base.as_ref()))
                                    .flat_map(|(sample, base)| {
                                        sample.motion.iter().filter_map(|(bone, motion)| {
                                            (base.motion.get(bone) != Some(motion)).then_some(*bone)
                                        })
                                    })
                            })
                            .collect();
                        let pairs = self
                            .geometry
                            .differential_pairs(&evaluation.contact_pairs, &moved);
                        let a = self.evaluate_geometry(a, Some(&pairs)).ok()?;
                        let b = self.evaluate_geometry(b, Some(&pairs)).ok()?;
                        task_column.extend(
                            a.residuals
                                .iter()
                                .zip(b.residuals)
                                .map(|(a, b)| (b - a) / f64::from(span)),
                        );
                        constraint_column.extend(
                            a.constraints
                                .iter()
                                .zip(b.constraints)
                                .map(|(a, b)| (b - a) / f64::from(span)),
                        );
                    } else {
                        task_column.extend(std::iter::repeat_n(0.0, evaluation.residuals.len()));
                        constraint_column
                            .extend(std::iter::repeat_n(0.0, evaluation.constraints.len()));
                    }
                    Some((index, task_column, constraint_column))
                });
            }
        });
        columns.sort_by_key(|column| column.as_ref().map(|c| c.0));
        for column in columns {
            let (_, tasks, constraints) = column?;
            task_columns.extend(tasks);
            constraint_columns.extend(constraints);
        }
        Some(Linearization {
            tasks: DMatrix::from_column_slice(evaluation.residuals.len(), 18, &task_columns),
            constraints: DMatrix::from_column_slice(
                evaluation.constraints.len(),
                18,
                &constraint_columns,
            ),
        })
    }

    // Gauss-Newton SQP: constraint restoration minimizes only negative
    // margins; once feasible, the QP improves tasks inside the linear domain.
    fn step(
        &self,
        current: [Option<ArmJoints>; 2],
        evaluation: &Evaluation,
        linearization: &Linearization,
        trust: f32,
    ) -> Option<[f32; 18]> {
        use nalgebra::{DMatrix, DVector};
        let a = &linearization.constraints;
        let restoring = evaluation.constraints.iter().any(|c| *c < 0.0);
        let (j, r) = if restoring {
            let rows: Vec<_> = evaluation
                .constraints
                .iter()
                .enumerate()
                .filter(|(_, c)| **c < 0.0)
                .collect();
            (
                DMatrix::from_fn(rows.len(), 18, |i, k| {
                    rows.get(i)
                        .and_then(|(row, _)| a.get((*row, k)))
                        .copied()
                        .unwrap_or(0.0)
                }),
                // Restore into the numerical interior. Aiming exactly at
                // zero can round to the same violating f32 pose forever.
                DVector::from_iterator(
                    rows.len(),
                    rows.iter()
                        .map(|(_, c)| **c - 64.0 * f64::from(f32::EPSILON)),
                ),
            )
        } else {
            (
                linearization.tasks.clone(),
                DVector::from_column_slice(&evaluation.residuals),
            )
        };
        let jt = j.transpose();
        let hessian = &jt * &j + DMatrix::identity(18, 18) * 1.0e-4;
        let linear = jt * r;
        let mut rows = Vec::<[f64; 18]>::new();
        let mut margins = Vec::new();
        if !restoring {
            for (i, c) in evaluation.constraints.iter().enumerate() {
                let row = std::array::from_fn(|k| a.get((i, k)).copied().unwrap_or(0.0));
                if row.iter().all(|v| *v == 0.0) {
                    continue;
                }
                rows.push(row);
                margins.push(*c);
            }
        }
        for index in 0..18 {
            let (lo, hi) = current
                .get(index / 9)
                .copied()
                .flatten()
                .map(|state| {
                    let q = state.angles.get(index % 9).copied().unwrap_or(0.0);
                    let (min, max) = crate::upper_limb::BOUNDS
                        .get(index % 9)
                        .copied()
                        .unwrap_or((q, q));
                    if matches!(index % 9, 5 | 6)
                        && evaluation
                            .arms
                            .get(index / 9)
                            .and_then(Option::as_ref)
                            .is_none_or(|a| a.palm.is_none())
                    {
                        (0.0, 0.0)
                    } else {
                        ((q - min).min(trust), (max - q).min(trust))
                    }
                })
                .unwrap_or((0.0, 0.0));
            let mut row = [0.0; 18];
            *row.get_mut(index)? = 1.0;
            rows.push(row);
            margins.push(f64::from(lo));
            *row.get_mut(index)? = -1.0;
            rows.push(row);
            margins.push(f64::from(hi));
        }
        quadratic_step(hessian, linear, &rows, &margins)
    }

    fn score(&self, evaluation: &Evaluation) -> (f64, f64) {
        (
            evaluation
                .constraints
                .iter()
                .map(|c| c.min(0.0).powi(2))
                .sum(),
            evaluation.error,
        )
    }

    /// Resolve the configured initial posture within the same anatomical and
    /// contact domain as tracking. An unreachable profile retains a residual.
    pub fn solve_neutral(
        &self,
        seed: [Option<ArmJoints>; 2],
    ) -> ([Option<ArmJoints>; 2], SolveStatus) {
        self.solve_at_resolution(seed, 96, 1.0e-4)
    }

    pub fn solve(
        &self,
        seed: [Option<ArmJoints>; 2],
        rounds: usize,
    ) -> ([Option<ArmJoints>; 2], SolveStatus) {
        self.solve_at_resolution(seed, rounds, 1.0e-3)
    }

    fn solve_at_resolution(
        &self,
        seed: [Option<ArmJoints>; 2],
        rounds: usize,
        angle_resolution: f32,
    ) -> ([Option<ArmJoints>; 2], SolveStatus) {
        let mut current = seed;
        let mut evaluation = match self.evaluate(current) {
            Ok(v) => v,
            Err(e) => return (current, SolveStatus::InvalidGeometry(e)),
        };
        let mut trust = MAX_STEP_RADIANS;
        let mut converged = false;
        let mut accepted_steps = 0;
        let mut restoration_steps = 0;
        let mut linearization = None;
        // Tracking stops at about 0.057 degrees; the cached neutral pose
        // retains its original 0.006-degree solve so its IK branch does not
        // change. Feasibility/ROM and the derivative stencil stay unchanged.
        // Failed proposals contract trust until its numerical stopping
        // threshold. Only accepted improvements spend the work budget.
        // Moving skin/body inputs can put the previous pose just outside the
        // numerical interior. Finish that restoration before spending the
        // observation-step budget; otherwise one arm's contact correction can
        // starve both arms' tracking on every live update. Restoration has the
        // same 96-iteration work limit as the existing neutral solve.
        while accepted_steps < rounds && restoration_steps < 96 {
            let within_numeric_margin = evaluation.constraints.iter().all(|c| *c >= 0.0);
            if within_numeric_margin && evaluation.error <= f64::from(f32::EPSILON).powi(2) {
                converged = true;
                break;
            }
            // A rejected proposal changes only the trust radius. The current
            // pose and its derivatives are identical until a step is accepted.
            if linearization.is_none() {
                linearization = self.linearize(current, &evaluation);
            }
            let Some(linearized) = linearization.as_ref() else {
                break;
            };
            let Some(delta) = self.step(current, &evaluation, linearized, trust) else {
                break;
            };
            let magnitude = delta.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
            if magnitude <= angle_resolution && within_numeric_margin {
                converged = true;
                break;
            }
            let mut accepted = false;
            let mut scale = 1.0;
            for _ in 0..12 {
                let mut proposal = current;
                for (side, arm) in proposal.iter_mut().enumerate() {
                    if let Some(arm) = arm {
                        for (i, q) in arm.angles.iter_mut().enumerate() {
                            *q += delta.get(side * 9 + i).copied().unwrap_or(0.0) * scale;
                        }
                        arm.bound();
                    }
                }
                // With a feasible incumbent, a proposal whose FK task error
                // is not lower cannot be accepted or receive an SOC. Avoid
                // rebuilding and clipping its collision geometry at all.
                let improves = !within_numeric_margin
                    || self
                        .kinematics(proposal)
                        .is_ok_and(|v| v.error < evaluation.error);
                if improves && let Ok(mut candidate) = self.evaluate(proposal) {
                    // A tangent QP step can leave a curved constraint by
                    // O(step²). Correct that normal error with the same
                    // Jacobian before shortening the step (Maratos SOC).
                    // Only the fully reevaluated feasible candidate is kept.
                    if within_numeric_margin
                        && candidate.error < evaluation.error
                        && candidate.constraints.iter().any(|c| *c < 0.0)
                        && let Some(correction) =
                            second_order_correction(&linearized.constraints, &candidate.constraints)
                        && correction.iter().all(|c| c.abs() <= magnitude * scale)
                    {
                        let mut corrected = proposal;
                        for (side, arm) in corrected.iter_mut().enumerate() {
                            if let Some(arm) = arm {
                                for (i, q) in arm.angles.iter_mut().enumerate() {
                                    *q += correction.get(side * 9 + i).copied().unwrap_or(0.0);
                                }
                                arm.bound();
                            }
                        }
                        if let Ok(v) = self.evaluate(corrected)
                            && self.score(&v) < self.score(&candidate)
                        {
                            proposal = corrected;
                            candidate = v;
                        }
                    }
                    if self.score(&candidate) < self.score(&evaluation) {
                        current = proposal;
                        evaluation = candidate;
                        accepted = true;
                        break;
                    }
                }
                scale *= 0.5;
            }
            if accepted {
                if within_numeric_margin {
                    accepted_steps += 1;
                } else {
                    restoration_steps += 1;
                }
                linearization = None;
            }
            // A shortened line-search step is not convergence: the full QP
            // proposal above determines that, before backtracking for contact.
            if !accepted {
                trust *= 0.5;
            }
            if trust
                <= if within_numeric_margin {
                    angle_resolution
                } else {
                    64.0 * f32::EPSILON
                }
            {
                converged = evaluation.constraints.iter().all(|c| *c >= 0.0);
                break;
            }
        }
        let status = if !self.feasible(&evaluation) {
            SolveStatus::NoFeasibleSolution
        } else {
            SolveStatus::Feasible {
                residual: evaluation.error.sqrt() as f32,
                converged,
            }
        };
        (current, status)
    }

    /// Keep an endpoint gap equal to both capsules' existing chord resolution.
    /// Near-zero contact leaves no room for the curved link motion between
    /// observations. This is a geometric contact offset, not a joint limit.
    pub fn contact_offset(&self) -> f32 {
        self.chains
            .into_iter()
            .flatten()
            .map(|c| c.rest.total_arm_length)
            .fold(0.0_f32, f32::max)
            * 0.002
    }

    pub fn feasible(&self, evaluation: &Evaluation) -> bool {
        evaluation.clearance >= self.contact_offset() - self.tolerance
            && evaluation.joint_margin >= -64.0 * f32::EPSILON
    }
}
