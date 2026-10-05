//! One producer of bilateral joint paths for the existing arm compositor.

use crate::{
    arm::ArmIkInput,
    arm_pipeline::{ArmPoseSourceKind, ArmSourceSelection, DynamicArmTargets, TrackedArmControl},
    binding::AvatarBinding,
    collision::AvatarCollision,
    upper_limb::ArmJoints,
    upper_limb_body::{BodyCurve, BodyPose, BodyRig},
    upper_limb_path::JointPath,
    upper_limb_solver::{ArmGoal, Problem, SolveStatus},
};
use bevy::prelude::*;
use bevy_vrm1::prelude::RestGlobalTransform;

#[derive(Resource, Default)]
#[doc(hidden)]
pub struct UpperLimbState {
    generation: Option<crate::lifecycle::AvatarGeneration>,
    path: JointPath,
    status: Option<SolveStatus>,
    solved_goals: Option<[Option<ArmGoal>; 2]>,
    refine: bool,
    solved_body: Option<BodyPose>,
    pending: Option<PendingSolve>,
    queued: Option<PreparedStep>,
    geometry: Option<std::sync::Arc<crate::collision::CollisionGeometry>>,
    source_seq: Option<vtuber_core::FrameSeq>,
    neutral: Option<NeutralSolution>,
    secondary: Option<SecondarySolution>,
    recovery_seed: Option<[Option<ArmJoints>; 2]>,
    body_rig: Option<std::sync::Arc<BodyRig>>,
    producer_body: Option<BodyPose>,
}

impl UpperLimbState {
    pub(crate) fn owns_bone(&self, bone: Entity) -> bool {
        self.body_rig.as_ref().is_some_and(|rig| rig.contains(bone))
    }
}

#[derive(Clone, Copy)]
struct NeutralSolution {
    goals: [Option<ArmGoal>; 2],
    pose: [Option<ArmJoints>; 2],
    status: SolveStatus,
}

#[derive(Clone, Copy)]
struct SecondarySolution {
    pose: [Option<ArmJoints>; 2],
    refine: bool,
    status: SolveStatus,
}

struct PreparedStep {
    computation_seconds: f32,
    anchor: [Option<ArmJoints>; 2],
    anchor_body: Option<BodyPose>,
    path: JointPath,
    status: SolveStatus,
    refine: bool,
    source_seq: Option<vtuber_core::FrameSeq>,
    neutral: Option<NeutralSolution>,
    secondary: Option<SecondarySolution>,
    recovery_seed: Option<[Option<ArmJoints>; 2]>,
}

struct PendingSolve {
    task: bevy::tasks::Task<PreparedStep>,
    obsolete: std::sync::Arc<std::sync::atomic::AtomicBool>,
    source_seq: Option<vtuber_core::FrameSeq>,
}

impl Drop for PendingSolve {
    fn drop(&mut self) {
        self.obsolete
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Restore the upstream body result before animation/base detection. The
/// admitted path is a display result, never a new additive animation base.
pub(crate) fn restore_body_inputs(
    state: Res<UpperLimbState>,
    mut transforms: Query<&mut Transform>,
) {
    if let Some(body) = &state.producer_body {
        for (bone, local) in body.locals() {
            if let Ok(mut current) = transforms.get_mut(bone) {
                *current = local;
            }
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    clippy::type_complexity,
    reason = "Bevy system parameters join current avatar geometry, control sources and the sole arm target writer"
)]
/// Produces the only admitted bilateral arm path for the compositor.
pub fn update_upper_limb_targets(
    mut commands: Commands,
    mut profile_changes: MessageReader<crate::arm_pose::ArmPoseProfileChange>,
    lifecycle: Res<crate::lifecycle::AvatarLifecycle>,
    selection: Res<ArmSourceSelection>,
    tracked: Res<TrackedArmControl>,
    control: Res<crate::unload::ActiveControlFrame>,
    mirror: Option<Res<crate::mirror::AvatarMotionMirror>>,
    overrides: Option<Res<crate::arm_pose::ArmPoseOverrideStore>>,
    time: Res<Time>,
    mut roots: Query<(
        Entity,
        &AvatarBinding,
        &crate::load::AvatarAssetId,
        &crate::arm_motion_geometry::ArmMotionGeometry,
        &crate::body_scale::BodyScaleMeters,
        &crate::direct_position::BodyTrackingPositionInput,
        &AvatarCollision,
        &mut DynamicArmTargets,
    )>,
    mut transform_queries: ParamSet<(
        Query<(&Transform, &GlobalTransform, Option<&RestGlobalTransform>)>,
        Query<(&mut Transform, &mut GlobalTransform)>,
    )>,
    parents: Query<&ChildOf>,
    children: Query<&Children>,
    mut state: ResMut<UpperLimbState>,
) {
    let Ok((root, binding, model_id, motion, scale, position, collision, mut targets)) =
        roots.single_mut()
    else {
        return;
    };
    if lifecycle.state() != crate::lifecycle::AvatarLifecycleState::Ready {
        return;
    }
    if state.generation != Some(binding.generation) {
        *state = UpperLimbState {
            generation: Some(binding.generation),
            ..Default::default()
        };
    }
    for change in profile_changes.read().filter(|c| &c.model_id == model_id) {
        state.solved_goals = None;
        state.path.transition_seconds = Some(if change.return_to_default {
            crate::arm_pose::DEFAULT_ARM_RETURN_SECONDS
        } else {
            crate::arm_pose::DEFAULT_ARM_TRANSITION_SECONDS
        });
    }
    let transforms = transform_queries.p0();
    let frame = tracked
        .frame
        .filter(|_| tracked.generation == Some(binding.generation));
    let mirrored = mirror.as_deref().is_none_or(|m| m.is_enabled());
    let thorax = frame
        .and_then(|f| f.thorax)
        .map(|t| if mirrored { t.mirrored() } else { t });
    let pose_profile = overrides
        .as_deref()
        .and_then(|s| s.profile_for(model_id))
        .unwrap_or_default();
    let dynamic_profile = overrides
        .as_deref()
        .and_then(|s| s.dynamic_profile_for(model_id))
        .unwrap_or(selection.profile);
    let control_current = control
        .frame
        .as_ref()
        .filter(|_| control.generation == binding.generation);
    let (observations, weights) = frame
        .map(|f| vtuber_core::mirror::MotionMirror::new(mirrored).arms(f.targets, f.weights))
        .unwrap_or_default();
    let torso = binding
        .upper_chest
        .or(binding.chest)
        .or(binding.spine)
        .and_then(|e| {
            transforms
                .get(e)
                .ok()
                .and_then(|(_, g, r)| r.map(|r| (g, r)))
        });
    let (rest_rotation, current_rotation) = torso
        .map(|(g, r)| (r.0.rotation(), g.rotation()))
        .unwrap_or((Quat::IDENTITY, Quat::IDENTITY));
    let tracking_to_rest = crate::tracked_arm::tracking_to_rest_rotation(
        rest_rotation,
        current_rotation,
        tracked.view_to_model,
    );
    let mut neutral = [None, None];
    let mut neutral_goals = [None, None];
    let mut goals = [None, None];
    let chains = [binding.left_arm.as_ref(), binding.right_arm.as_ref()];
    let observed = [observations.left, observations.right];
    let weights = [weights.left, weights.right];
    let geometries = [motion.left.as_ref(), motion.right.as_ref()];
    let shoulder_width = binding.left_arm.zip(binding.right_arm).map(|(l, r)| {
        l.rest
            .upper_arm
            .position
            .distance(r.rest.upper_arm.position)
    });
    for (side, ((((slot, goal), chain), geometry), (observed, weight))) in neutral
        .iter_mut()
        .zip(goals.iter_mut())
        .zip(chains)
        .zip(geometries)
        .zip(observed.into_iter().zip(weights))
        .enumerate()
    {
        let Some(chain) = chain else { continue };
        let Some((initial, target)) = ArmJoints::resting(chain, pose_profile) else {
            continue;
        };
        *slot = Some(initial);
        let mut wanted = ArmGoal {
            wrist: target.wrist,
            elbow: None,
            palm: None,
            shoulder: None,
            weight: Default::default(),
            shoulder_weight: 0.0,
            neutral: initial,
            neutral_girdle: None,
        };
        let initial_shoulder = initial.forward(chain).map(|c| c.shoulder);
        if let Some((rest, initial_shoulder)) = chain.rest.shoulder.zip(initial_shoulder) {
            let lateral = chain.rest.elbow.position - chain.rest.upper_arm.position;
            if let Some(axis) = lateral.cross(Vec3::Y).try_normalize() {
                let offset = initial_shoulder - rest.position;
                let wanted_centre = rest.position
                    + Quat::from_axis_angle(axis, dynamic_profile.shoulder_elevation_trim_radians)
                        * offset;
                if dynamic_profile.shoulder_elevation_trim_radians != 0.0 {
                    wanted.shoulder = Some(wanted_centre);
                    wanted.shoulder_weight = 1.0;
                }
            }
        }
        if let Some(slot) = neutral_goals.get_mut(side) {
            let mut profile_goal = wanted;
            // The profile specifies the wrist location. The bounded analytic
            // seed is only a starting point and may no longer reach it.
            profile_goal.weight.wrist = 1.0;
            *slot = Some(profile_goal);
        }
        match selection.mode {
            ArmPoseSourceKind::TrackedPose => {
                if let Some(observed) = observed {
                    let target = crate::tracked_arm::tracked_arm_ik_target(
                        chain.rest,
                        observed,
                        tracking_to_rest,
                    );
                    wanted.wrist = target.wrist;
                    wanted.elbow = Some(target.elbow_pole);
                    wanted.palm = observed
                        .palm_normal
                        .zip(observed.palm_forward)
                        .map(|(n, f)| {
                            (
                                tracking_to_rest * Vec3::from_array(n),
                                tracking_to_rest * Vec3::from_array(f),
                            )
                        });
                    wanted.weight = weight;
                    if let Some((thorax, width)) = thorax.zip(shoulder_width)
                        && let Some(offset) = thorax.shoulder_offsets.get(side)
                        && let Some(neutral_centre) = initial_shoulder
                    {
                        let centre = wanted.shoulder.unwrap_or(neutral_centre)
                            + Vec3::from_array(*offset) * width;
                        wanted.shoulder = Some(centre);
                        wanted.shoulder_weight = thorax.weight;
                        wanted.wrist += centre - chain.rest.upper_arm.position;
                        wanted.elbow = wanted
                            .elbow
                            .map(|p| p + centre - chain.rest.upper_arm.position);
                    }
                    if let Some(state) = slot {
                        state.fingers = observed.fingers;
                        state.finger_weight = weight.fingers;
                    }
                }
            }
            ArmPoseSourceKind::VirtualHandAnchor => {
                if let Some(motion) = geometry.filter(|_| control_current.is_some()) {
                    let input = crate::arm_pipeline::ArmPipelineInput {
                        chain,
                        motion,
                        pose_profile,
                        dynamic_profile,
                        head_offset: position.tracked_head_target,
                        body_offset: position.tracked_body_target,
                        torso_delta: current_rotation * rest_rotation.inverse(),
                        body_scale_meters: scale.scale_meters,
                    };
                    if let Some(target) = crate::arm_pipeline::virtual_hand_target(&input) {
                        wanted.wrist = target.wrist;
                        wanted.weight.wrist = 1.0;
                        // The pole specifies a bend plane, not a measured elbow.
                        // Convert that virtual intent through fixed-length FK.
                        if let Ok(seed) =
                            crate::arm::solve_two_bone_arm(ArmIkInput::from_chain(chain, target))
                        {
                            wanted.elbow = Some(seed.elbow);
                            wanted.weight.pole = 1.0;
                        }
                    }
                }
            }
        }
        if let Some(neutral) = *slot {
            wanted.neutral = neutral;
        }
        *goal = Some(wanted);
    }
    let geometry = match &collision.0 {
        Ok(g) => g,
        Err(error) => {
            report(
                &mut commands,
                root,
                &mut state,
                SolveStatus::InvalidGeometry(*error),
            );
            return;
        }
    };
    if state
        .geometry
        .as_ref()
        .is_some_and(|previous| !std::sync::Arc::ptr_eq(previous, geometry))
    {
        *state = UpperLimbState {
            generation: Some(binding.generation),
            ..Default::default()
        };
        *targets = DynamicArmTargets::default();
        commands.entity(root).insert(Visibility::Hidden);
    }
    state.geometry = Some(std::sync::Arc::clone(geometry));
    let controlled: std::collections::HashSet<_> = chains
        .into_iter()
        .zip(neutral)
        .filter_map(|(c, s)| c.zip(s).and_then(|(c, s)| s.forward(c)))
        .flat_map(|a| a.motion.into_keys())
        .collect();
    let body_pose = (|| -> Result<Option<BodyPose>, crate::collision::CollisionError> {
        if geometry.capsules.is_empty() {
            return Ok(None);
        }
        if state.body_rig.is_none() {
            let chest = binding
                .upper_chest
                .or(binding.chest)
                .or(binding.spine)
                .ok_or(crate::collision::CollisionError::MissingBone)?;
            state.body_rig = Some(BodyRig::bind(
                root,
                chest,
                geometry.bones().filter(|b| !controlled.contains(b)),
                |bone| {
                    let (local, global, rest) = transforms.get(bone).ok()?;
                    Some((
                        *local,
                        rest.map_or(*global, |r| r.0).affine(),
                        parents.get(bone).ok().map(ChildOf::parent),
                    ))
                },
            )?);
        }
        state
            .body_rig
            .as_ref()
            .map(|rig| rig.capture(|b| transforms.get(b).ok().map(|(t, _, _)| *t)))
            .transpose()
    })();
    let body_pose = match body_pose {
        Ok(pose) => pose,
        Err(e) => {
            report(
                &mut commands,
                root,
                &mut state,
                SolveStatus::InvalidGeometry(e),
            );
            return;
        }
    };
    state.producer_body = body_pose.clone();
    let body = match body_pose.as_ref().map(BodyPose::motions).transpose() {
        Ok(body) => body.unwrap_or_default(),
        Err(e) => {
            report(
                &mut commands,
                root,
                &mut state,
                SolveStatus::InvalidGeometry(e),
            );
            return;
        }
    };
    let extent = chains
        .into_iter()
        .flatten()
        .map(|c| c.rest.total_arm_length)
        .fold(0.0_f32, f32::max);
    let problem = Problem {
        chains,
        goals,
        geometry,
        body: &body,
        body_curve: None,
        tolerance: 64.0 * f32::EPSILON * extent,
    };
    use bevy::tasks::{AsyncComputeTaskPool, futures_lite::future};
    let source_seq = frame
        .map(|f| f.source_seq)
        .or_else(|| control_current.map(|f| f.source_seq));
    let body_changed = match (&state.solved_body, &body_pose) {
        (Some(previous), Some(current)) => !previous.same(current),
        (None, None) => false,
        _ => true,
    };
    if let Some(pending) = &state.pending
        && pending.source_seq != source_seq
        && (state.solved_goals != Some(goals) || body_changed)
    {
        // Local steps can finish and be displayed, but an expensive global
        // route for an observation that has already changed is wasted work.
        // Damping/idle still evolves after a sample is held. Let that sample's
        // route finish; cancelling on every such tick would starve the search.
        pending
            .obsolete
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
    if let Some(pending) = state.pending.as_mut()
        && let Some(prepared) = future::block_on(future::poll_once(&mut pending.task))
    {
        state.queued = Some(prepared);
        state.pending = None;
    }
    let mut status = state.status.unwrap_or(SolveStatus::Solving);
    let mut frame_seconds = time.delta_secs();
    if matches!(status, SolveStatus::Feasible { .. }) && state.path.busy() {
        let current = state.path.current;
        match state.path.advance(&problem, current, frame_seconds) {
            Ok(remaining) => frame_seconds = remaining,
            Err(error) => {
                status = error;
                state.refine = false;
                frame_seconds = 0.0;
            }
        }
    }
    // A frame may cross the endpoint. Carry its remaining time into the
    // prefetched path instead of rounding every camera interval up to a tick.
    if !state.path.busy()
        && let Some(mut prepared) = state.queued.take()
    {
        if prepared.anchor == state.path.current
            && match (&prepared.anchor_body, &state.path.body) {
                (Some(a), Some(b)) => a.same(b),
                (None, None) => true,
                _ => false,
            }
        {
            if state.solved_goals.is_none() {
                prepared.path.transition_seconds = state.path.transition_seconds;
            }
            prepared.path.retime(prepared.computation_seconds);
            state.path = prepared.path;
            status = prepared.status;
            state.refine = prepared.refine;
            state.source_seq = prepared.source_seq;
            state.neutral = prepared.neutral;
            state.secondary = prepared.secondary;
            state.recovery_seed = prepared.recovery_seed;
        } else {
            // An external body/geometry change cancelled the path before the
            // worker's starting point. Never splice a different anchor in.
            state.solved_goals = None;
        }
    }
    if state.pending.is_none()
        && state.queued.is_none()
        && state.solved_goals == Some(goals)
        && !body_changed
        && !state.refine
    {
        // A new sequence with identical inputs uses the same admitted
        // solution. Retain provenance without solving that pose again.
        state.source_seq = source_seq;
    }
    if state.pending.is_none()
        && state.queued.is_none()
        && (state.solved_goals != Some(goals) || body_changed || state.refine)
    {
        let mut path = state.path.after_current_segment();
        let anchor = path.current;
        let anchor_body = path.body.clone();
        let body_curve = body_pose.clone().map(|to| BodyCurve {
            from: anchor_body.clone().unwrap_or_else(|| to.clone()),
            to,
        });
        // Constraint restoration may need more than one work batch. Resume
        // its candidate without publishing it as a display pose.
        let mut seed = state.recovery_seed.unwrap_or(anchor);
        for (seed, goal) in seed.iter_mut().zip(neutral) {
            if let Some(goal) = goal {
                if let Some(current) = seed {
                    current.fingers = goal.fingers.or(current.fingers);
                    current.finger_weight = goal.finger_weight;
                    current.rest_curl = goal.rest_curl;
                } else {
                    *seed = Some(goal);
                }
            }
        }
        let initial = path.current.iter().all(Option::is_none);
        let geometry = std::sync::Arc::clone(geometry);
        let snapshot = body.clone();
        let chains = [binding.left_arm, binding.right_arm];
        // Publish a feasible local step promptly. Extra capacity completes
        // contact restoration when that first iteration cannot yet be shown.
        let pool = AsyncComputeTaskPool::get_or_init(bevy::tasks::TaskPool::default);
        // A 30 Hz camera naturally repeats a sample on alternating render
        // ticks. That is not a stopped target: batching four task steps there
        // creates a large jump in the next two-frame display interval.
        let rounds = 1;
        let unchanged = state.solved_goals == Some(goals) && !body_changed;
        let cached_status = unchanged.then_some(status).filter(|s| {
            matches!(
                s,
                SolveStatus::Feasible {
                    converged: true,
                    ..
                }
            )
        });
        let cached_neutral = state.neutral.filter(|n| n.goals == neutral_goals);
        let secondary = state
            .secondary
            .filter(|s| s.refine || state.solved_goals == Some(goals));
        let obsolete = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let search_obsolete = std::sync::Arc::clone(&obsolete);
        let started = std::time::Instant::now();
        state.pending = Some(PendingSolve {
            obsolete,
            source_seq,
            task: pool.spawn(async move {
                let neutral_problem = Problem {
                    chains: [chains[0].as_ref(), chains[1].as_ref()],
                    goals: neutral_goals,
                    geometry: &geometry,
                    body: &snapshot,
                    body_curve: None,
                    tolerance: 64.0 * f32::EPSILON * extent,
                };
                // The return target is the same constrained pose admitted at
                // startup, not the infeasible analytic request that preceded
                // it. Recompute it only when its profile or shape changes.
                let neutral_solution = if let Some(cached) = cached_neutral {
                    cached
                } else {
                    let seed = neutral_goals.map(|g| g.map(|g| g.neutral));
                    let (pose, status) = neutral_problem.solve_neutral(seed);
                    if !matches!(status, SolveStatus::Feasible { .. }) {
                        return PreparedStep {
                            computation_seconds: started.elapsed().as_secs_f32(),
                            anchor,
                            anchor_body,
                            path,
                            status,
                            refine: false,
                            source_seq,
                            neutral: None,
                            secondary: None,
                            recovery_seed: None,
                        };
                    }
                    NeutralSolution {
                        goals: neutral_goals,
                        pose,
                        status,
                    }
                };
                let mut goals = goals;
                for (goal, neutral) in goals.iter_mut().zip(neutral_solution.pose) {
                    if let Some((goal, mut neutral)) = goal.as_mut().zip(neutral) {
                        neutral.fingers = goal.neutral.fingers;
                        neutral.finger_weight = goal.neutral.finger_weight;
                        goal.neutral = neutral;
                        let [_, _, _, _, _, _, _, protract, elevate] = neutral.angles;
                        goal.neutral_girdle = Some(Vec2::new(protract, elevate));
                    }
                }
                if initial {
                    seed = neutral_solution.pose;
                    for (seed, goal) in seed.iter_mut().zip(goals) {
                        if let Some((seed, goal)) = seed.as_mut().zip(goal) {
                            seed.fingers = goal.neutral.fingers;
                            seed.finger_weight = goal.neutral.finger_weight;
                        }
                    }
                }
                let problem = Problem {
                    chains: [chains[0].as_ref(), chains[1].as_ref()],
                    goals,
                    geometry: &geometry,
                    body: &snapshot,
                    body_curve: body_curve.as_ref(),
                    tolerance: 64.0 * f32::EPSILON * extent,
                };
                let (mut next, mut status) = if let Some(status) = cached_status { (seed, status) }
                    else { problem.solve(seed, rounds) };
                // Keep the continuation step independent of worker count.
                // Extra workers finish the same contact restoration sooner.
                if matches!(status, SolveStatus::NoFeasibleSolution) && next != seed {
                    (next, status) = problem.solve(next, 1);
                }
                let unfinished = |from, to, status| from != to && matches!(status,
                    SolveStatus::Feasible { converged: false, .. } | SolveStatus::NoFeasibleSolution);
                let progressing = next != seed && matches!(status, SolveStatus::Feasible { .. });
                let mut refine = unfinished(seed, next, status) || progressing;
                let mut secondary_next = secondary;
                // Continue a feasible local movement before trying a different
                // IK branch. A lower endpoint error alone does not make that
                // other branch reachable from the currently displayed pose.
                if !initial && !progressing {
                    let mut destination_seed = secondary.map_or_else(|| goals.map(|g| g.map(|g| g.neutral)), |s| s.pose);
                    for (seed, goal) in destination_seed.iter_mut().zip(goals) {
                        if let Some((seed, goal)) = seed.as_mut().zip(goal) {
                            seed.fingers = goal.neutral.fingers;
                            seed.finger_weight = goal.neutral.finger_weight;
                            seed.rest_curl = goal.neutral.rest_curl;
                        }
                    }
                    let (alternative, alternative_status) = if unchanged && let Some(previous) = secondary.filter(|s| !s.refine) {
                        (destination_seed, previous.status)
                    } else { problem.solve(destination_seed, rounds) };
                    let alternate_refine = unfinished(destination_seed, alternative, alternative_status);
                    if let SolveStatus::Feasible { residual: other, .. } = alternative_status
                        && !matches!(status, SolveStatus::Feasible { residual, .. } if residual <= other + 64.0*f32::EPSILON)
                    {
                        secondary_next = Some(SecondarySolution { pose: next, refine, status });
                        next = alternative;
                        status = alternative_status;
                    } else {
                        secondary_next = Some(SecondarySolution { pose: alternative, refine: alternate_refine, status: alternative_status });
                    }
                    refine |= alternate_refine;
                }
                refine |= secondary_next.is_some_and(|s| s.refine);
                let recovery_seed = (matches!(status, SolveStatus::NoFeasibleSolution)
                    && unfinished(seed, next, status)).then_some(next);
                if matches!(status, SolveStatus::Feasible { .. })
                    && (next != path.current || body_curve.as_ref().is_some_and(|c| !c.from.same(&c.to)))
                {
                    let admitted = if initial { path.advance(&problem, next, 0.0).map(|_| true) } else { path.plan(&problem, next, &search_obsolete) };
                    match admitted {
                        Err(error) => { status = error; refine = matches!(error, SolveStatus::Solving); }
                        Ok(false) => {
                            let endpoint = path.after_current_segment();
                            if let Ok(evaluation) = problem.kinematics(endpoint.current) {
                                status = SolveStatus::Feasible { residual: evaluation.error.sqrt() as f32, converged: false };
                            }
                            refine = true;
                        }
                        Ok(_) => {}
                    }
                }
                if initial
                    && let (
                        SolveStatus::Feasible {
                            residual,
                            converged,
                        },
                        SolveStatus::Feasible {
                            residual: profile_residual,
                            converged: profile_converged,
                        },
                    ) = (status, neutral_solution.status)
                {
                    status = SolveStatus::Feasible {
                        residual: residual.max(profile_residual),
                        converged: converged && profile_converged,
                    };
                }
                if refine && matches!(status, SolveStatus::NoFeasibleSolution) {
                    status = SolveStatus::Solving;
                }
                refine &= matches!(status, SolveStatus::Feasible { .. } | SolveStatus::Solving);
                PreparedStep {
                    computation_seconds: started.elapsed().as_secs_f32(),
                    anchor,
                    anchor_body,
                    path,
                    status,
                    refine,
                    source_seq,
                    neutral: Some(neutral_solution),
                    secondary: secondary_next,
                    recovery_seed,
                }
            }),
        });
        state.solved_goals = Some(goals);
        state.solved_body = body_pose;
    }
    if matches!(status, SolveStatus::Feasible { .. }) {
        let current = state.path.current;
        if state.path.busy()
            && frame_seconds > 0.0
            && let Err(error) = state.path.advance(&problem, current, frame_seconds)
        {
            status = error;
            state.refine = false;
        }
        let displayed_body = state.path.body.as_ref().map(BodyPose::motions).transpose();
        let evaluation = displayed_body.and_then(|body| {
            Problem {
                body: body.as_ref().unwrap_or(problem.body),
                ..problem
            }
            .kinematics(state.path.current)
        });
        match evaluation {
            Ok(evaluation) if problem.feasible(&evaluation) => {
                if targets.left.is_none() && targets.right.is_none() {
                    commands.entity(root).insert(Visibility::Inherited);
                }
                let [left, right] = evaluation.arms;
                *targets = DynamicArmTargets {
                    generation: Some(binding.generation),
                    source_seq: state.source_seq,
                    left: left.map(|a| a.resolved),
                    right: right.map(|a| a.resolved),
                };
            }
            Ok(_) => {
                status = SolveStatus::BlockedPath;
                state.refine = false;
            }
            Err(error) => {
                status = SolveStatus::InvalidGeometry(error);
                state.refine = false;
            }
        }
    }
    if let Some(body) = &state.path.body {
        let mut transforms = transform_queries.p1();
        for (bone, local) in body.locals() {
            if let Ok((mut current, _)) = transforms.get_mut(bone) {
                *current = local;
            }
        }
        if let Some(global) = crate::skeleton::refresh_global(root, &mut transforms, &parents, None)
        {
            crate::skeleton::refresh_subtree(root, global, &mut transforms, &children);
        }
    }
    report(&mut commands, root, &mut state, status);
}

fn report(commands: &mut Commands, root: Entity, state: &mut UpperLimbState, status: SolveStatus) {
    if state
        .status
        .is_none_or(|previous| std::mem::discriminant(&previous) != std::mem::discriminant(&status))
    {
        match status {
            SolveStatus::Solving => {}
            SolveStatus::InvalidGeometry(error) => {
                warn!("Upper-limb collision geometry unavailable: {error:?}")
            }
            SolveStatus::NoFeasibleSolution => warn!("Upper-limb solve has no feasible candidate"),
            SolveStatus::BlockedPath => warn!("Upper-limb transition has no admissible path"),
            SolveStatus::Feasible {
                residual,
                converged,
            } => debug!("Upper-limb solve residual={residual}, converged={converged}"),
        }
    }
    state.status = Some(status);
    let displayed_status = match status {
        SolveStatus::Feasible {
            residual,
            converged,
        } => SolveStatus::Feasible {
            residual,
            converged: converged
                && !state.path.busy()
                && state.pending.is_none()
                && state.queued.is_none(),
        },
        other => other,
    };
    commands.entity(root).insert(displayed_status);
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]
    use super::*;
    use crate::arm::{ArmChainBinding, ArmSide, RestSpaceBonePose};
    use bevy_vrm1::prelude::RestTransform;
    use vtuber_core::arm_tracking::{
        ArmBlendWeight, ArmBlendWeights, ArmControlFrame, ArmTrackingTarget, ArmTrackingTargets,
    };

    fn update_after_worker(app: &mut App) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while app
            .world()
            .resource::<UpperLimbState>()
            .pending
            .as_ref()
            .is_some_and(|t| !t.task.is_finished())
        {
            assert!(
                std::time::Instant::now() < deadline,
                "upper limb worker did not complete"
            );
            std::thread::yield_now();
        }
        app.update();
    }

    fn spawn_chain(app: &mut App, root: Entity, side: ArmSide) -> ArmChainBinding {
        let mut chain = crate::upper_limb::tests::chain(side);
        chain.finger_rest = Default::default();
        let mut parent = root;
        let mut parent_rest = GlobalTransform::IDENTITY;
        let mut spawn = |rest: RestSpaceBonePose| {
            let global = GlobalTransform::from(
                Transform::from_translation(rest.position).with_rotation(rest.global_rotation),
            );
            let local = global.reparented_to(&parent_rest);
            let entity = app
                .world_mut()
                .spawn((
                    local,
                    global,
                    RestTransform(local),
                    RestGlobalTransform(global),
                    ChildOf(parent),
                ))
                .id();
            parent = entity;
            parent_rest = global;
            entity
        };
        chain.shoulder = Some(spawn(chain.rest.shoulder.unwrap()));
        chain.upper_arm = spawn(chain.rest.upper_arm);
        chain.lower_arm = spawn(chain.rest.elbow);
        chain.hand = spawn(chain.rest.wrist);
        chain
    }

    fn rig() -> (App, Entity, ArmChainBinding, ArmChainBinding) {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
                std::time::Duration::from_secs_f32(1.0 / 120.0),
            ))
            .init_resource::<crate::lifecycle::AvatarLifecycle>()
            .init_resource::<ArmSourceSelection>()
            .init_resource::<TrackedArmControl>()
            .init_resource::<UpperLimbState>()
            .add_message::<crate::arm_pose::ArmPoseProfileChange>()
            .init_resource::<crate::unload::ActiveControlFrame>()
            .add_systems(
                Update,
                (
                    update_upper_limb_targets,
                    crate::arm_pose::apply_default_arm_pose,
                )
                    .chain(),
            );
        let root = app
            .world_mut()
            .spawn((
                crate::lifecycle::ActiveAvatar,
                Transform::IDENTITY,
                GlobalTransform::IDENTITY,
            ))
            .id();
        let left = spawn_chain(&mut app, root, ArmSide::Left);
        let right = spawn_chain(&mut app, root, ArmSide::Right);
        let generation = {
            let mut lifecycle = app
                .world_mut()
                .resource_mut::<crate::lifecycle::AvatarLifecycle>();
            lifecycle.request_load(root).unwrap();
            lifecycle.start_binding(root);
            lifecycle.finish_ready();
            lifecycle.current_generation()
        };
        let mut binding = AvatarBinding::head_only(root, root, generation);
        binding.left_arm = Some(left);
        binding.right_arm = Some(right);
        app.world_mut().entity_mut(root).insert((
            binding,
            crate::load::AvatarAssetId::new("joint-path-test"),
            crate::arm_motion_geometry::ArmMotionGeometry::default(),
            crate::body_scale::BodyScaleMeters {
                generation,
                scale_meters: 0.56,
            },
            crate::direct_position::BodyTrackingPositionInput::default(),
            AvatarCollision(Ok(std::sync::Arc::new(
                crate::collision::CollisionGeometry::default(),
            ))),
            DynamicArmTargets::default(),
        ));
        (app, root, left, right)
    }

    #[test]
    fn admitted_path_is_the_only_compositor_pose_across_held_frames_and_loss() {
        let (mut app, root, left, right) = rig();
        let generation = app.world().get::<AvatarBinding>(root).unwrap().generation;
        update_after_worker(&mut app);
        update_after_worker(&mut app);
        let initial_positions = [left.lower_arm, left.hand, right.lower_arm, right.hand]
            .map(|b| app.world().get::<GlobalTransform>(b).unwrap().translation());
        let initial = *app.world().get::<DynamicArmTargets>(root).unwrap();
        assert!(initial.left.is_some() && initial.right.is_some());
        let wanted = crate::upper_limb::tests::state(0.6, 0.8, -0.2, 0.7);
        let observe = |chain: &ArmChainBinding| {
            let pose = wanted.forward(chain).unwrap();
            ArmTrackingTarget {
                wrist: ((pose.wrist - chain.rest.upper_arm.position) / chain.rest.total_arm_length)
                    .to_array(),
                elbow_pole: ((pose.elbow - chain.rest.upper_arm.position)
                    / chain.rest.total_arm_length)
                    .to_array(),
                palm_normal: None,
                palm_forward: None,
                fingers: None,
            }
        };
        app.world_mut().resource_mut::<ArmSourceSelection>().mode = ArmPoseSourceKind::TrackedPose;
        *app.world_mut().resource_mut::<TrackedArmControl>() = TrackedArmControl {
            generation: Some(generation),
            view_to_model: Quat::IDENTITY,
            frame: Some(ArmControlFrame {
                thorax: None,
                source_seq: vtuber_core::FrameSeq(1),
                captured_at: vtuber_core::MonoTimeNs(1),
                produced_at: vtuber_core::MonoTimeNs(1),
                targets: ArmTrackingTargets {
                    left: Some(observe(&left)),
                    right: Some(observe(&right)),
                },
                weights: ArmBlendWeights {
                    left: ArmBlendWeight {
                        wrist: 1.0,
                        pole: 1.0,
                        palm: 0.0,
                        fingers: 0.0,
                    },
                    right: ArmBlendWeight {
                        wrist: 1.0,
                        pole: 1.0,
                        palm: 0.0,
                        fingers: 0.0,
                    },
                },
            }),
        };
        for tick in 0..400 {
            if tick == 80 {
                app.world_mut()
                    .resource_mut::<TrackedArmControl>()
                    .frame
                    .as_mut()
                    .unwrap()
                    .weights = Default::default();
            }
            update_after_worker(&mut app);
            assert!(matches!(
                app.world().get::<SolveStatus>(root),
                Some(SolveStatus::Feasible { .. })
            ));
            for chain in [left, right] {
                let p = |e| app.world().get::<GlobalTransform>(e).unwrap().translation();
                assert!(
                    (p(chain.upper_arm).distance(p(chain.lower_arm)) - chain.rest.upper_arm_length)
                        .abs()
                        < 2.0e-6
                );
                assert!(
                    (p(chain.lower_arm).distance(p(chain.hand)) - chain.rest.forearm_length).abs()
                        < 2.0e-6
                );
            }
        }
        // This fixture has no palm frame: axial coordinates are unobserved,
        // so compare the neutral FK tasks instead of an arbitrary twist split.
        let before = app.world().resource::<UpperLimbState>().path.current;
        app.world_mut()
            .resource_mut::<TrackedArmControl>()
            .frame
            .as_mut()
            .unwrap()
            .source_seq = vtuber_core::FrameSeq(2);
        app.update();
        assert_eq!(
            app.world()
                .get::<DynamicArmTargets>(root)
                .unwrap()
                .source_seq,
            Some(vtuber_core::FrameSeq(2))
        );
        let state = app.world().resource::<UpperLimbState>();
        assert!(state.pending.is_none());
        assert_eq!(state.path.current, before);
        for (bone, expected) in [left.lower_arm, left.hand, right.lower_arm, right.hand]
            .into_iter()
            .zip(initial_positions)
        {
            assert!(
                app.world()
                    .get::<GlobalTransform>(bone)
                    .unwrap()
                    .translation()
                    .distance(expected)
                    < 2.0e-6
            );
        }
    }
    #[test]
    fn forbidden_plane_branch_and_poles_do_not_jump_the_live_joint_path() {
        let (mut app, root, left, _) = rig();
        update_after_worker(&mut app);
        update_after_worker(&mut app);
        let generation = app.world().get::<AvatarBinding>(root).unwrap().generation;
        app.world_mut().resource_mut::<ArmSourceSelection>().mode = ArmPoseSourceKind::TrackedPose;
        let pi = std::f32::consts::PI;
        // Camera goals cross the old 200-degree branch, both coordinate
        // poles, then loss and reacquisition. The joint state is never
        // reconstructed from a discontinuous inverse-angle branch.
        for (plane, elevation, weight) in [
            (199.99_f32.to_radians(), pi / 2.0, 1.0),
            (200.01_f32.to_radians(), pi / 2.0, 1.0),
            (0.0, 0.001, 1.0),
            (pi, 0.001, 1.0),
            (0.0, pi - 0.001, 1.0),
            (pi, pi - 0.001, 1.0),
            (0.0, 0.001, 0.0),
            (0.5, 1.0, 1.0),
        ] {
            let direction = Vec3::new(
                elevation.sin() * plane.cos(),
                -elevation.cos(),
                elevation.sin() * plane.sin(),
            );
            let target = ArmTrackingTarget {
                wrist: (direction * 0.9).to_array(),
                elbow_pole: (direction * 0.5 + Vec3::Z * 0.1).to_array(),
                palm_normal: None,
                palm_forward: None,
                fingers: None,
            };
            *app.world_mut().resource_mut::<TrackedArmControl>() = TrackedArmControl {
                generation: Some(generation),
                view_to_model: Quat::IDENTITY,
                frame: Some(ArmControlFrame {
                    source_seq: vtuber_core::FrameSeq(1),
                    captured_at: vtuber_core::MonoTimeNs(1),
                    produced_at: vtuber_core::MonoTimeNs(1),
                    thorax: None,
                    targets: ArmTrackingTargets {
                        left: Some(target),
                        right: None,
                    },
                    weights: ArmBlendWeights {
                        left: ArmBlendWeight {
                            wrist: weight,
                            pole: weight,
                            palm: 0.0,
                            fingers: 0.0,
                        },
                        right: Default::default(),
                    },
                }),
            };
            for _ in 0..60 {
                let before = app.world().resource::<UpperLimbState>().path.current;
                update_after_worker(&mut app);
                let after = app.world().resource::<UpperLimbState>().path.current;
                for (a, b) in before.into_iter().zip(after) {
                    let (a, b) = (a.unwrap(), b.unwrap());
                    assert!(b.valid());
                    for ((a, b), (lo, hi)) in a
                        .angles
                        .into_iter()
                        .zip(b.angles)
                        .zip(crate::upper_limb::BOUNDS)
                    {
                        // A live segment traverses its checked range linearly
                        // over one camera interval, without the old rest envelope.
                        let duration = 1.0 / 30.0;
                        let bound = (hi - lo) / (120.0 * duration) + 64.0 * f32::EPSILON;
                        assert!((b - a).abs() <= bound);
                    }
                }
                let p = |e| app.world().get::<GlobalTransform>(e).unwrap().translation();
                assert!(
                    (p(left.upper_arm).distance(p(left.lower_arm)) - left.rest.upper_arm_length)
                        .abs()
                        < 2.0e-6
                );
                assert!(
                    (p(left.lower_arm).distance(p(left.hand)) - left.rest.forearm_length).abs()
                        < 2.0e-6
                );
            }
        }
    }

    #[test]
    fn mirror_matches_explicit_reflection_and_side_swap_in_the_live_producer() {
        let (mut mirrored, root_a, left_a, _) = rig();
        let (mut explicit, root_b, _, _) = rig();
        let mut off = crate::mirror::AvatarMotionMirror::default();
        off.toggle();
        explicit.insert_resource(off);
        let target = crate::upper_limb::tests::state(0.8, 0.9, -0.3, 0.8)
            .forward(&left_a)
            .unwrap();
        let frame = ArmControlFrame {
            source_seq: vtuber_core::FrameSeq(42),
            captured_at: vtuber_core::MonoTimeNs(1),
            produced_at: vtuber_core::MonoTimeNs(1),
            thorax: None,
            targets: ArmTrackingTargets {
                left: Some(ArmTrackingTarget {
                    wrist: ((target.wrist - left_a.rest.upper_arm.position)
                        / left_a.rest.total_arm_length)
                        .to_array(),
                    elbow_pole: ((target.elbow - left_a.rest.upper_arm.position)
                        / left_a.rest.total_arm_length)
                        .to_array(),
                    palm_normal: None,
                    palm_forward: None,
                    fingers: None,
                }),
                right: None,
            },
            weights: ArmBlendWeights {
                left: ArmBlendWeight {
                    wrist: 1.0,
                    pole: 1.0,
                    palm: 0.0,
                    fingers: 0.0,
                },
                right: Default::default(),
            },
        };
        let (targets, weights) =
            vtuber_core::mirror::MotionMirror::new(true).arms(frame.targets, frame.weights);
        let reflected = ArmControlFrame {
            targets,
            weights,
            ..frame
        };
        for (app, root, frame) in [
            (&mut mirrored, root_a, frame),
            (&mut explicit, root_b, reflected),
        ] {
            update_after_worker(app);
            update_after_worker(app);
            app.world_mut().resource_mut::<ArmSourceSelection>().mode =
                ArmPoseSourceKind::TrackedPose;
            let generation = app.world().get::<AvatarBinding>(root).unwrap().generation;
            *app.world_mut().resource_mut::<TrackedArmControl>() = TrackedArmControl {
                generation: Some(generation),
                frame: Some(frame),
                view_to_model: Quat::IDENTITY,
            };
        }
        for _ in 0..80 {
            update_after_worker(&mut mirrored);
            update_after_worker(&mut explicit);
            let a = mirrored.world().get::<DynamicArmTargets>(root_a).unwrap();
            let b = explicit.world().get::<DynamicArmTargets>(root_b).unwrap();
            for (a, b) in [a.left, a.right].into_iter().zip([b.left, b.right]) {
                let (a, b) = (a.unwrap(), b.unwrap());
                assert!(
                    (a.upper_arm_delta * Vec3::X).distance(b.upper_arm_delta * Vec3::X)
                        < 64.0 * f32::EPSILON
                );
                assert!(
                    (a.upper_arm_delta * Vec3::Y).distance(b.upper_arm_delta * Vec3::Y)
                        < 64.0 * f32::EPSILON
                );
                assert!(
                    (a.lower_arm_delta * Vec3::X).distance(b.lower_arm_delta * Vec3::X)
                        < 64.0 * f32::EPSILON
                );
                assert!(
                    (a.lower_arm_delta * Vec3::Y).distance(b.lower_arm_delta * Vec3::Y)
                        < 64.0 * f32::EPSILON
                );
            }
        }
        assert_eq!(
            mirrored
                .world()
                .resource::<TrackedArmControl>()
                .frame
                .unwrap()
                .targets,
            frame.targets
        );
    }
}
