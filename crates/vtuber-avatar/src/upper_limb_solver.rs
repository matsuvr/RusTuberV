//! Bilateral sequential quadratic programming. Feasibility has lexical
//! priority over observation error; a residual never buys a limit violation.

use crate::{
    arm::ArmChainBinding,
    collision::{BoneMotion, CollisionError, CollisionGeometry},
    upper_limb::{ArmCandidate, ArmJoints},
};
use bevy::prelude::*;

pub(crate) const MAX_STEP_RADIANS: f32 = 0.25;
use std::{collections::HashMap, sync::Arc};

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

#[derive(Clone, Copy)]
struct NeutralPose {
    wrist: Vec3,
    elbow: Vec3,
    palm: Option<(Vec3, Vec3)>,
}

// A solve owns its neutral FK tasks; no cache survives a changed Problem.
struct PreparedProblem<'a> {
    problem: Problem<'a>,
    neutral: [Option<NeutralPose>; 2],
    #[cfg(test)]
    full_recomputation: bool,
    #[cfg(test)]
    whole_arm_differences: bool,
}

impl<'a> std::ops::Deref for PreparedProblem<'a> {
    type Target = Problem<'a>;

    fn deref(&self) -> &Self::Target {
        &self.problem
    }
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
    /// Skeleton proxies could not be used for collision checks.
    InvalidGeometry(CollisionError),
    /// The current transition did not pass the continuous ROM/skin checks.
    BlockedPath,
}

#[derive(Clone, Debug)]
pub(crate) struct Evaluation {
    // Difference samples share the unchanged arm without copying its motion map.
    pub arms: [Option<Arc<ArmCandidate>>; 2],
    pub clearance: f32,
    pub joint_margin: f32,
    pub error: f64,
    residuals: Vec<f64>,
    residual_counts: [usize; 2],
    constraints: Vec<f64>,
    contact_pairs: Vec<bool>,
}

impl<'a> Problem<'a> {
    fn prepare(&self) -> Result<PreparedProblem<'a>, CollisionError> {
        let mut neutral = [None, None];
        for ((slot, chain), goal) in neutral.iter_mut().zip(self.chains).zip(self.goals) {
            if let Some((chain, goal)) = chain.zip(goal) {
                let pose = goal
                    .neutral
                    .forward(chain)
                    .ok_or(CollisionError::InvalidGeometry)?;
                *slot = Some(NeutralPose {
                    wrist: pose.wrist,
                    elbow: pose.elbow,
                    palm: pose.palm,
                });
            }
        }
        Ok(PreparedProblem {
            problem: *self,
            neutral,
            #[cfg(test)]
            full_recomputation: false,
            #[cfg(test)]
            whole_arm_differences: false,
        })
    }

    pub(crate) fn kinematics(
        &self,
        state: [Option<ArmJoints>; 2],
    ) -> Result<Evaluation, CollisionError> {
        self.prepare()?.kinematics(state)
    }

    pub fn evaluate(&self, state: [Option<ArmJoints>; 2]) -> Result<Evaluation, CollisionError> {
        self.prepare()?.evaluate(state)
    }

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
        match self.prepare() {
            Ok(problem) => problem.solve_at_resolution(seed, rounds, angle_resolution),
            Err(error) => (seed, SolveStatus::InvalidGeometry(error)),
        }
    }
}

impl PreparedProblem<'_> {
    pub(crate) fn kinematics(
        &self,
        state: [Option<ArmJoints>; 2],
    ) -> Result<Evaluation, CollisionError> {
        self.kinematics_near(state, None)
    }

    fn arm_residuals(
        &self,
        side: usize,
        chain: &ArmChainBinding,
        state: ArmJoints,
        pose: &ArmCandidate,
        reference: Option<&ArmCandidate>,
    ) -> Result<Vec<f64>, CollisionError> {
        let goal = self.goals.get(side).copied().flatten();
        let mut residuals = Vec::new();
        let reference_palm = reference.and_then(|arm| arm.palm);
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
            let neutral = self
                .neutral
                .get(side)
                .copied()
                .flatten()
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
        Ok(residuals)
    }

    fn kinematics_near(
        &self,
        state: [Option<ArmJoints>; 2],
        reference: Option<&Evaluation>,
    ) -> Result<Evaluation, CollisionError> {
        #[cfg(test)]
        if self.full_recomputation {
            return tests::uncached_kinematics_near(&self.problem, state, reference);
        }
        let mut arms = [None, None];
        for ((slot, chain), state) in arms.iter_mut().zip(self.chains).zip(state) {
            if let Some((chain, state)) = chain.zip(state) {
                *slot = Some(Arc::new(
                    state
                        .forward(chain)
                        .ok_or(CollisionError::InvalidGeometry)?,
                ));
            }
        }
        self.residuals_near(state, arms, reference)
    }

    fn residuals_near(
        &self,
        state: [Option<ArmJoints>; 2],
        arms: [Option<Arc<ArmCandidate>>; 2],
        reference: Option<&Evaluation>,
    ) -> Result<Evaluation, CollisionError> {
        let mut residuals = Vec::new();
        let mut residual_counts = [0; 2];
        for (side, ((chain, state), pose)) in
            self.chains.into_iter().zip(state).zip(&arms).enumerate()
        {
            if let Some(((chain, state), pose)) = chain.zip(state).zip(pose.as_deref()) {
                let arm_residuals = self.arm_residuals(
                    side,
                    chain,
                    state,
                    pose,
                    reference
                        .and_then(|r| r.arms.get(side))
                        .and_then(Option::as_deref),
                )?;
                if let Some(count) = residual_counts.get_mut(side) {
                    *count = arm_residuals.len();
                }
                residuals.extend(arm_residuals);
            }
        }
        Ok(Self::kinematic_evaluation(arms, residuals, residual_counts))
    }

    fn kinematic_evaluation(
        arms: [Option<Arc<ArmCandidate>>; 2],
        residuals: Vec<f64>,
        residual_counts: [usize; 2],
    ) -> Evaluation {
        let joint_margin = arms
            .iter()
            .flatten()
            .map(|a| a.joint_margin)
            .fold(f32::INFINITY, f32::min);
        let error = residuals.iter().map(|v| v * v).sum();
        Evaluation {
            arms,
            clearance: f32::INFINITY,
            joint_margin,
            error,
            residuals,
            residual_counts,
            constraints: Vec::new(),
            contact_pairs: Vec::new(),
        }
    }

    fn differential_kinematics(
        &self,
        state: [Option<ArmJoints>; 2],
        side: usize,
        reference: &Evaluation,
    ) -> Result<Evaluation, CollisionError> {
        #[cfg(test)]
        if self.full_recomputation {
            return tests::uncached_kinematics_near(&self.problem, state, Some(reference));
        }
        #[cfg(test)]
        if self.whole_arm_differences {
            return self.kinematics_near(state, Some(reference));
        }
        let chain = self
            .chains
            .get(side)
            .copied()
            .flatten()
            .ok_or(CollisionError::InvalidGeometry)?;
        let state = state
            .get(side)
            .copied()
            .flatten()
            .ok_or(CollisionError::InvalidGeometry)?;
        let pose = state
            .forward(chain)
            .ok_or(CollisionError::InvalidGeometry)?;
        let changed = self.arm_residuals(
            side,
            chain,
            state,
            &pose,
            reference.arms.get(side).and_then(Option::as_deref),
        )?;
        let mut pose = Some(Arc::new(pose));
        let arms = std::array::from_fn(|i| {
            if i == side {
                pose.take()
            } else {
                reference.arms.get(i).cloned().flatten()
            }
        });
        // On the unchanged arm, current and reference palms coincide, so the
        // near-reference branch has the same residual as the admitted evaluation.
        let mut residuals = Vec::with_capacity(reference.residuals.len());
        let mut residual_counts = reference.residual_counts;
        let mut start = 0;
        for (i, count) in reference.residual_counts.into_iter().enumerate() {
            if i == side {
                residuals.extend(&changed);
                if let Some(count) = residual_counts.get_mut(i) {
                    *count = changed.len();
                }
            } else {
                residuals.extend(
                    reference
                        .residuals
                        .get(start..start + count)
                        .ok_or(CollisionError::InvalidGeometry)?,
                );
            }
            start += count;
        }
        Ok(Self::kinematic_evaluation(arms, residuals, residual_counts))
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
        let mut linearization = Linearization {
            tasks: DMatrix::zeros(evaluation.residuals.len(), 18),
            constraints: DMatrix::zeros(evaluation.constraints.len(), 18),
        };
        // Each task owns a disjoint final matrix column. No per-column output
        // buffers, sorting, concatenation or matrix copy is required.
        let column_results = bevy::tasks::AsyncComputeTaskPool::get_or_init(
            bevy::tasks::TaskPool::default,
        )
        .scope(|scope| {
            for (index, (mut tasks, mut constraints)) in linearization
                .tasks
                .column_iter_mut()
                .zip(linearization.constraints.column_iter_mut())
                .enumerate()
            {
                scope.spawn(async move {
                    self.linearize_column(
                        current,
                        evaluation,
                        index,
                        tasks.as_mut_slice(),
                        constraints.as_mut_slice(),
                    )
                });
            }
        });
        for column_result in column_results {
            column_result?;
        }
        Some(linearization)
    }

    fn linearize_column(
        &self,
        current: [Option<ArmJoints>; 2],
        evaluation: &Evaluation,
        index: usize,
        task_column: &mut [f64],
        constraint_column: &mut [f64],
    ) -> Option<()> {
        let h = f32::EPSILON.cbrt();
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
            let a = self
                .differential_kinematics(minus, index / 9, evaluation)
                .ok()?;
            let b = self
                .differential_kinematics(plus, index / 9, evaluation)
                .ok()?;
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
            for (slot, (a, b)) in task_column
                .iter_mut()
                .zip(a.residuals.iter().zip(&b.residuals))
            {
                *slot = (b - a) / f64::from(span);
            }
            for (slot, (a, b)) in constraint_column
                .iter_mut()
                .zip(a.constraints.iter().zip(&b.constraints))
            {
                *slot = (b - a) / f64::from(span);
            }
        }
        Some(())
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
                if let Ok(candidate) = self.kinematics(proposal)
                    && (!within_numeric_margin || candidate.error < evaluation.error)
                    && let Ok(mut candidate) = self.evaluate_geometry(candidate, None)
                {
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
}

impl Problem<'_> {
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

#[cfg(test)]
#[path = "upper_limb_solver_tests.rs"]
mod tests;
