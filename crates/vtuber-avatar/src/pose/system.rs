//! Direct-pose body-tracking input bridge.
//!
//! Reads the latest [`ActiveControlFrame`] and updates
//! [`BodyTrackingPoseInput`](crate::direct_pose::BodyTrackingPoseInput) on the
//! active avatar root. Bone transforms are owned exclusively by the
//! application direct-pose writer.

use bevy::prelude::*;
use vtuber_core::metrics::FixedStats;
use vtuber_core::monotonic_now;
use vtuber_core::types::AvatarControlFrame;

use crate::direct_pose::{BodyTrackingPoseInput, BodyTrackingProfile};
use crate::direct_position::BodyTrackingPositionInput;
use bevy_vrm1::prelude::{
    ChestBoneEntity, HeadBoneEntity, HipsBoneEntity, RestTransform, UpperChestBoneEntity, VrmPath,
};

use crate::binding::AvatarBinding;
use crate::lifecycle::{AvatarLifecycle, AvatarLifecycleState};
use crate::mirror::AvatarMotionMirror;
use crate::unload::{ActiveControlFrame, ControlFrameError, resolve_control_target};

/// Metrics for the pose apply system, useful for diagnostics.
#[derive(Resource, Debug, Clone)]
pub struct PoseApplyMetrics {
    /// Number of frames where the pose was successfully applied.
    pub frames_applied: u64,
    /// Number of frames skipped because lifecycle was not Ready.
    pub skipped_not_ready: u64,
    /// Number of frames skipped due to generation mismatch.
    pub skipped_generation_mismatch: u64,
    /// Number of frames skipped because no control frame was available.
    pub skipped_no_frame: u64,
    /// Number of frames skipped because the binding entity was stale.
    pub skipped_stale_entity: u64,
    /// Source sequence of the most recently observed control frame.
    pub last_applied_source_seq: Option<vtuber_core::FrameSeq>,
    /// Monotonic time when the most recent frame was applied.
    pub last_applied_at: Option<vtuber_core::MonoTimeNs>,
    /// First-apply latency of the most recently observed control frame.
    pub last_capture_to_apply_ms: Option<f64>,
    /// Fixed-size capture-to-apply latency samples.
    latency_samples: FixedStats,
}

impl Default for PoseApplyMetrics {
    fn default() -> Self {
        Self {
            frames_applied: 0,
            skipped_not_ready: 0,
            skipped_generation_mismatch: 0,
            skipped_no_frame: 0,
            skipped_stale_entity: 0,
            last_applied_source_seq: None,
            last_applied_at: None,
            last_capture_to_apply_ms: None,
            latency_samples: FixedStats::new(256),
        }
    }
}

impl PoseApplyMetrics {
    fn record_apply(
        &mut self,
        source_seq: vtuber_core::FrameSeq,
        captured_at: vtuber_core::MonoTimeNs,
        applied_at: vtuber_core::MonoTimeNs,
    ) {
        self.frames_applied += 1;
        self.last_applied_at = Some(applied_at);

        // The current control frame is intentionally re-applied after animation
        // on every render frame. Capture-to-apply measures the first application
        // of each source observation, not the age of those later re-applications.
        if self.last_applied_source_seq == Some(source_seq) {
            return;
        }

        self.last_applied_source_seq = Some(source_seq);
        let latency_ms = applied_at
            .0
            .checked_sub(captured_at.0)
            .map(|ns| ns as f64 / 1_000_000.0);
        self.last_capture_to_apply_ms = latency_ms;
        if let Some(latency_ms) = latency_ms {
            self.latency_samples.record(latency_ms);
        }
    }

    /// Number of capture-to-apply latency samples retained.
    #[must_use]
    pub fn latency_sample_count(&self) -> usize {
        self.latency_samples.count()
    }

    /// p50 capture-to-apply latency in milliseconds.
    #[must_use]
    pub fn capture_to_apply_p50_ms(&self) -> f64 {
        self.latency_samples.p50()
    }

    /// p95 capture-to-apply latency in milliseconds.
    #[must_use]
    pub fn capture_to_apply_p95_ms(&self) -> f64 {
        self.latency_samples.p95()
    }
}

/// System that updates the direct pose consumed by the application body tracking.
///
/// # Schedule
///
/// Runs in `PostUpdate`, after `AnimationSystems`. It does not write any bone
/// `Transform`; the application-owned direct body-tracking system is the sole
/// humanoid pose writer.
///
/// # Skip conditions
///
/// - Lifecycle is not `Ready`
/// - No active control frame
/// - Generation mismatch between frame and binding
/// - Direct input component is missing from the active root
pub fn update_body_tracking_pose_input(
    lifecycle: Res<AvatarLifecycle>,
    control_frame: Res<ActiveControlFrame>,
    mirror: Option<Res<AvatarMotionMirror>>,
    idle_state: Option<Res<crate::body_motion::LossIdleState>>,
    mut metrics: ResMut<PoseApplyMetrics>,
    binding_query: Query<&AvatarBinding>,
    mut inputs: Query<&mut BodyTrackingPoseInput>,
) {
    let target = resolve_control_target(
        &lifecycle,
        lifecycle
            .active_root()
            .and_then(|root| binding_query.get(root).ok()),
        control_frame
            .frame
            .as_ref()
            .map(|_| control_frame.generation),
    );
    let target = match target {
        Ok(target) => target,
        Err(error) => {
            deactivate_inputs(&mut inputs);
            match error {
                ControlFrameError::NotReady { .. } => metrics.skipped_not_ready += 1,
                ControlFrameError::StaleBinding => metrics.skipped_stale_entity += 1,
                ControlFrameError::StaleGeneration { .. } => {
                    metrics.skipped_generation_mismatch += 1
                }
            }
            return;
        }
    };
    if !target.frame_is_current {
        deactivate_inputs(&mut inputs);
        metrics.skipped_generation_mismatch += 1;
        return;
    }
    let binding = target.binding;
    let active_root = binding.root;

    let mut input = match inputs.get_mut(active_root) {
        Ok(input) => input,
        Err(_) => {
            metrics.skipped_stale_entity += 1;
            return;
        }
    };

    let mirrored = mirror.is_none_or(|mirror| mirror.is_enabled());

    let frame = match &control_frame.frame {
        Some(f) => f,
        None => {
            // No control frame (camera signal gone or capture stopped): keep
            // the loss-idle rotation flowing (ADR-021) so the avatar stays
            // alive in its default pose instead of freezing. The blend is
            // advanced by the position-input bridge earlier in the tick.
            if let Some(idle_state) = idle_state.as_ref()
                && idle_state.blend() > 0.0
            {
                let idle = idle_state.target();
                let horizontal_sign = if mirrored { -1.0 } else { 1.0 };
                *input = BodyTrackingPoseInput {
                    yaw_radians: horizontal_sign * idle.yaw_radians,
                    pitch_radians: idle.pitch_radians,
                    roll_radians: 0.0,
                    weight: idle_state.blend().clamp(0.0, 1.0),
                    active: true,
                };
            } else {
                *input = BodyTrackingPoseInput::default();
            }
            metrics.skipped_no_frame += 1;
            return;
        }
    };

    *input = body_tracking_input(frame, mirrored);

    // Tracking-loss idle (ADR-021): the bounded idle yaw/pitch cross-fades
    // with the control frame instead of replacing it. The blend continues
    // from its current value on both direction changes, so during a loss the
    // sway fades in over the still-easing pose, and after a reacquire it
    // fades out while the tracked pose resumes — the head never snaps.
    if let Some(idle_state) = idle_state.as_ref() {
        let blend = idle_state.blend();
        if blend > 0.0 {
            let idle = idle_state.target();
            let horizontal_sign = if mirrored { -1.0 } else { 1.0 };
            *input = fade_pose_input(*input, idle, horizontal_sign, blend);
        }
    }

    let applied_at = monotonic_now();
    metrics.record_apply(frame.source_seq, frame.captured_at, applied_at);
}

/// Cross-fades one pose input with the idle sway by `blend`.
///
/// The two contributions the writer multiplies by the weight are cross-faded
/// exactly, so at `blend = 0` the input is unchanged and at `blend = 1` it is
/// the idle sway alone; in between the control frame's contribution shrinks
/// with its own weight instead of being zeroed.
fn fade_pose_input(
    input: BodyTrackingPoseInput,
    idle: &vtuber_tracking::IdleTarget,
    horizontal_sign: f32,
    blend: f32,
) -> BodyTrackingPoseInput {
    let blend = blend.clamp(0.0, 1.0);
    let frame_weight = input.weight.clamp(0.0, 1.0);
    let weight = ((1.0 - blend) * frame_weight + blend).clamp(0.0, 1.0);
    let yaw_total = (1.0 - blend) * input.yaw_radians * frame_weight
        + blend * (horizontal_sign * idle.yaw_radians);
    let pitch_total =
        (1.0 - blend) * input.pitch_radians * frame_weight + blend * idle.pitch_radians;
    let roll_total = (1.0 - blend) * input.roll_radians * frame_weight;
    BodyTrackingPoseInput {
        yaw_radians: yaw_total / weight,
        pitch_radians: pitch_total / weight,
        roll_radians: roll_total / weight,
        weight,
        active: true,
    }
}

fn body_tracking_input(frame: &AvatarControlFrame, mirrored: bool) -> BodyTrackingPoseInput {
    let horizontal_sign = if mirrored { -1.0 } else { 1.0 };
    BodyTrackingPoseInput {
        // A horizontal reflection preserves pitch but reverses yaw and roll.
        yaw_radians: horizontal_sign * frame.head.yaw_rad,
        pitch_radians: frame.head.pitch_rad,
        roll_radians: horizontal_sign * frame.head.roll_rad,
        weight: frame.confidence,
        // The control frame already carries the loss glide and a confidence
        // that decays to zero. Marking the input inactive during loss would
        // zero the dependency's target instantly and snap the head and body to
        // rest the moment the face leaves the frame; the weight is the only
        // blend that may reach zero.
        active: true,
    }
}

fn deactivate_inputs(inputs: &mut Query<&mut BodyTrackingPoseInput>) {
    for mut input in inputs.iter_mut() {
        *input = BodyTrackingPoseInput::default();
    }
}

/// System that resets pose metrics when the avatar lifecycle changes.
///
/// Runs after `clear_control_cache_on_lifecycle_change` to ensure metrics
/// don't accumulate across avatar replacements.
pub fn reset_pose_metrics_on_lifecycle_change(
    lifecycle: Res<AvatarLifecycle>,
    mut metrics: ResMut<PoseApplyMetrics>,
    mut last_state: Local<Option<AvatarLifecycleState>>,
) {
    let current = lifecycle.state();
    if last_state.as_ref() != Some(&current) {
        *metrics = PoseApplyMetrics::default();
        *last_state = Some(current);
    }
}

/// Temporary propagation diagnosis (remove after the investigation).
///
/// Appends one line per second to `propagation_debug.log` next to the
/// executable's working directory (the GUI subsystem has no stderr).
#[expect(
    clippy::too_many_arguments,
    reason = "temporary propagation probe; each parameter is one stage input written to the debug log"
)]
pub fn debug_propagation_probe(
    lifecycle: Res<AvatarLifecycle>,
    mut frame_counter: Local<u64>,
    mut log_file: Local<Option<std::fs::File>>,
    control_frame: Res<ActiveControlFrame>,
    tracked: Res<crate::arm_pipeline::TrackedArmControl>,
    selection: Res<crate::arm_pipeline::ArmSourceSelection>,
    inputs: Query<(&BodyTrackingPoseInput, &BodyTrackingPositionInput)>,
    profiles: Query<&BodyTrackingProfile>,
    bindings: Query<&AvatarBinding>,
    dynamic_targets: Query<&crate::arm_pipeline::DynamicArmTargets>,
    hips: Query<&HipsBoneEntity>,
    upper_chest_markers: Query<&UpperChestBoneEntity>,
    chest_markers: Query<&ChestBoneEntity>,
    head_markers: Query<&HeadBoneEntity>,
    root_paths: Query<&VrmPath>,
    bones: Query<(&Transform, &RestTransform)>,
) {
    *frame_counter += 1;
    if log_file.is_none() {
        *log_file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open("propagation_debug.log")
            .ok();
    }
    if *frame_counter % 60 != 1 {
        return;
    }
    let Some(file) = log_file.as_mut() else {
        return;
    };
    use std::io::Write;
    let root = lifecycle.active_root();
    let tracked_text = tracked_arm_text(
        *selection,
        *tracked,
        root.and_then(|root| dynamic_targets.get(root).ok()),
    );
    let position_text = root
        .and_then(|root| inputs.get(root).ok())
        .map(|(_, input)| {
            format!(
                "|pos(head=({:+.3},{:+.3},{:+.3}),body=({:+.3},{:+.3},{:+.3}),w={:.2},active={})",
                input.head_offset.x,
                input.head_offset.y,
                input.head_offset.z,
                input.body_offset.x,
                input.body_offset.y,
                input.body_offset.z,
                input.weight,
                input.active,
            )
        })
        .unwrap_or_else(|| "|pos(<none>)".to_string());
    let root_text = root
        .and_then(|root| bones.get(root).ok())
        .map(|(transform, rest)| {
            let delta = transform.translation - rest.0.translation;
            format!("|root=({:+.3},{:+.3},{:+.3})", delta.x, delta.y, delta.z)
        })
        .unwrap_or_else(|| "|root=<none>".to_string());
    // The observation before any body-side following, so the next log can
    // separate "the face tracker wobbled" from "the body follow amplified it".
    let raw_translation_text = control_frame
        .frame
        .as_ref()
        .map(|frame| {
            let raw = frame.head_translation;
            format!(
                "|htraw=({:+.3},{:+.3},{:+.3},{:?})",
                raw.x_meters(),
                raw.y_meters(),
                raw.z_meters(),
                raw.state()
            )
        })
        .unwrap_or_else(|| "|htraw=<none>".to_string());
    let line = propagation_line(
        &lifecycle,
        *frame_counter,
        &control_frame,
        inputs.get(root.unwrap_or(Entity::PLACEHOLDER)),
        profiles.get(root.unwrap_or(Entity::PLACEHOLDER)),
        bindings.get(root.unwrap_or(Entity::PLACEHOLDER)),
        hips,
        upper_chest_markers,
        chest_markers,
        head_markers,
        root_paths,
        &bones,
    ) + &tracked_text
        + &position_text
        + &root_text
        + &raw_translation_text;
    let _ = writeln!(file, "{line}");
    let _ = file.flush();
}

/// Temporary per-frame arm diagnosis (remove after the investigation).
///
/// `propagation_debug.log` samples once per second, which cannot resolve a
/// single-frame discontinuity. This appends **every render tick** to
/// `arm_frame_debug.log` so a pop can be read off the frame before and after it.
///
/// Each side reports the composed bone rotations, because those are what the
/// viewer sees, plus the two quantities that decide whether a discontinuity is
/// reachable at all:
///
/// - `up_dot` is the dot of the solved upper-arm direction with its rest
///   direction. It reaches -1 where the shortest arc the solver takes has no
///   defined axis, which is where an upper arm can jump.
/// - `desc` is the signed coronal descent, which wraps at a half turn.
///
/// The observation is reported **after** the avatar mirror, because the mirror
/// decides which observed side drives which bone; the per-second probe reports
/// the unmirrored sides, which reads as the wrong arm.
#[expect(
    clippy::too_many_arguments,
    reason = "temporary arm probe; each parameter is one stage input written to the debug log"
)]
pub fn debug_arm_frame_probe(
    lifecycle: Res<AvatarLifecycle>,
    selection: Res<crate::arm_pipeline::ArmSourceSelection>,
    control: Res<crate::arm_pipeline::TrackedArmControl>,
    mirror: Option<Res<crate::mirror::AvatarMotionMirror>>,
    mut frame_counter: Local<u64>,
    mut log_file: Local<Option<std::fs::File>>,
    bindings: Query<&AvatarBinding>,
    dynamic_targets: Query<&crate::arm_pipeline::DynamicArmTargets>,
    bones: Query<(&Transform, &RestTransform)>,
) {
    *frame_counter += 1;
    if log_file.is_none() {
        *log_file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open("arm_frame_debug.log")
            .ok();
    }
    let Some(file) = log_file.as_mut() else {
        return;
    };
    use std::io::Write;
    let mirrored = mirror.as_deref().is_none_or(|mirror| mirror.is_enabled());
    let root = lifecycle.active_root();
    let binding = root.and_then(|root| bindings.get(root).ok());
    // A frame only counts once it belongs to the avatar that is on screen, which
    // is the same generation guard the compositor applies.
    let frame = control
        .frame
        .filter(|_| control.generation == binding.map(|binding| binding.generation));
    // Mirrored, because the mirror decides which observed side drives which bone.
    let (targets, weights) = match frame {
        Some(_) if mirrored => (
            frame.map_or_else(Default::default, |frame| frame.targets.mirrored()),
            frame.map_or_else(Default::default, |frame| frame.weights.mirrored()),
        ),
        Some(_) => (
            frame.map_or_else(Default::default, |frame| frame.targets),
            frame.map_or_else(Default::default, |frame| frame.weights),
        ),
        None => (Default::default(), Default::default()),
    };

    let mut line = format!(
        "[arm] frame={}|src={:?} mirror={mirrored}",
        *frame_counter, selection.mode
    );
    if let Some(frame) = frame {
        line.push_str(&format!("|seq={}", frame.source_seq.0));
    }
    match binding.zip(root.and_then(|root| dynamic_targets.get(root).ok())) {
        Some((binding, dynamic)) => {
            for (label, chain, pose, target, weight) in [
                (
                    "L",
                    binding.left_arm.as_ref(),
                    dynamic.left,
                    targets.left,
                    weights.left,
                ),
                (
                    "R",
                    binding.right_arm.as_ref(),
                    dynamic.right,
                    targets.right,
                    weights.right,
                ),
            ] {
                line.push_str(&format!(
                    "|{label}{}",
                    arm_side_text(chain, pose.as_ref(), target, weight, &bones)
                ));
            }
        }
        None => line.push_str("|<no-avatar>"),
    }
    let _ = writeln!(file, "{line}");
    let _ = file.flush();
}

/// One side of the per-frame arm probe: the composed rotations and the two
/// quantities that make a discontinuity reachable.
fn arm_side_text(
    chain: Option<&crate::arm::ArmChainBinding>,
    pose: Option<&crate::arm_pose::ResolvedArmPose>,
    target: Option<vtuber_core::arm_tracking::ArmTrackingTarget>,
    weight: vtuber_core::arm_tracking::ArmBlendWeight,
    bones: &Query<(&Transform, &RestTransform)>,
) -> String {
    let degrees = |entity: Option<Entity>| -> f32 {
        entity
            .and_then(|entity| bones.get(entity).ok())
            .map_or(f32::NAN, |(transform, rest)| {
                (rest.0.rotation.inverse() * transform.rotation)
                    .angle_between(Quat::IDENTITY)
                    .to_degrees()
            })
    };
    let observation = target.map_or_else(
        || "obs=none".to_string(),
        |target| {
            let [x, y, z] = target.wrist;
            let [ex, ey, ez] = target.elbow_pole;
            let palm = target.palm_normal.map_or("none".to_string(), |[x, y, z]| {
                format!("{x:+.2},{y:+.2},{z:+.2}")
            });
            format!("w={x:+.2},{y:+.2},{z:+.2} pole={ex:+.2},{ey:+.2},{ez:+.2} palm={palm}")
        },
    );
    let Some(chain) = chain else {
        return format!("(unbound {observation})");
    };
    let rest_direction =
        crate::arm::finite_normalized(chain.rest.elbow.position - chain.rest.upper_arm.position);
    let written = bones
        .get(chain.upper_arm)
        .ok()
        .map(|(transform, rest)| rest.0.rotation.inverse() * transform.rotation);
    let diagnostics = match (rest_direction, written) {
        (Some(rest_direction), Some(written)) => upper_arm_diagnostics(
            rest_direction,
            chain.rest.upper_arm.global_rotation,
            written,
        ),
        _ => (f32::NAN, f32::NAN),
    };
    let hand = pose.and_then(|pose| pose.hand).map_or(f32::NAN, |hand| {
        hand.delta.angle_between(Quat::IDENTITY).to_degrees()
    });
    format!(
        "(sh={:+.1} up={:+.1} up_dot={:+.3} desc={:+.1} lo={:+.1} hand={hand:+.1} \
         w={:.2}/{:.2}/{:.2} {observation})",
        degrees(chain.shoulder),
        degrees(Some(chain.upper_arm)),
        diagnostics.0,
        diagnostics.1,
        degrees(Some(chain.lower_arm)),
        weight.wrist,
        weight.pole,
        weight.palm,
    )
}

/// The two numbers that decide whether an upper-arm discontinuity is reachable.
///
/// `written` is the bone's rest-relative *local* rotation, so it is conjugated
/// into model space before it is applied to a model-space direction. `up_dot` is
/// how close the arm is to pointing opposite its rest direction, which is where
/// the shortest arc the solver takes has no defined axis. `desc` is the signed
/// coronal descent, which wraps at a half turn.
fn upper_arm_diagnostics(rest_direction: Vec3, rest_global: Quat, written: Quat) -> (f32, f32) {
    let model_delta = rest_global * written * rest_global.inverse();
    let upper = model_delta * rest_direction;
    let descent = crate::arm_pipeline::coronal_descent_radians(rest_direction, upper)
        .map_or(f32::NAN, |radians| radians.to_degrees());
    (upper.dot(rest_direction), descent)
}

/// Renders the observed-arm authority and per-side blend state for the probe.
///
/// This is what separates "the hand was not seen" from "it was seen and the
/// pose is still returning": the weights move only when the tracker state
/// changes, and the wrist roll is the applied palm twist.
fn tracked_arm_text(
    selection: crate::arm_pipeline::ArmSourceSelection,
    control: crate::arm_pipeline::TrackedArmControl,
    targets: Option<&crate::arm_pipeline::DynamicArmTargets>,
) -> String {
    let mode = format!("{:?}", selection.mode);
    let Some(frame) = control.frame else {
        return format!("|tracked(mode={mode} no-frame)");
    };
    let mut text = format!("|tracked(mode={mode} seq={}", frame.source_seq.0);
    for (side, target, weight, pose) in [
        (
            "L",
            frame.targets.left,
            frame.weights.left,
            targets.and_then(|targets| targets.left),
        ),
        (
            "R",
            frame.targets.right,
            frame.weights.right,
            targets.and_then(|targets| targets.right),
        ),
    ] {
        let roll_degrees = pose
            .and_then(|pose| pose.hand)
            .map(|hand| {
                let (_, angle) = hand.delta.to_axis_angle();
                angle.to_degrees()
            })
            .unwrap_or(0.0);

        let pole = target
            .map(|target| target.elbow_pole)
            .map(|pole| format!("[{:.2},{:.2},{:.2}]", pole[0], pole[1], pole[2]))
            .unwrap_or_else(|| "-".to_string());
        text.push_str(&format!(
            " {side}(w={:.2},p={:.2},pal={:.2},hand={roll_degrees:+.0}deg,pole={pole},palm={:?})",
            weight.wrist,
            weight.pole,
            weight.palm,
            target.and_then(|target| target.palm_normal),
        ));
    }
    text.push(')');
    text
}

#[expect(
    clippy::too_many_arguments,
    reason = "the probe writes one line from the lifecycle, look, arm and body state it is diagnosing"
)]
fn propagation_line(
    lifecycle: &AvatarLifecycle,
    frame_counter: u64,
    control_frame: &ActiveControlFrame,
    input: Result<
        (&BodyTrackingPoseInput, &BodyTrackingPositionInput),
        bevy::ecs::query::QueryEntityError,
    >,
    profile: Result<&BodyTrackingProfile, bevy::ecs::query::QueryEntityError>,
    binding: Result<&AvatarBinding, bevy::ecs::query::QueryEntityError>,
    hips: Query<&HipsBoneEntity>,
    upper_chest_markers: Query<&UpperChestBoneEntity>,
    chest_markers: Query<&ChestBoneEntity>,
    head_markers: Query<&HeadBoneEntity>,
    root_paths: Query<&VrmPath>,
    bones: &Query<(&Transform, &RestTransform)>,
) -> String {
    let root = lifecycle.active_root();
    let input_text = root
        .zip(input.ok())
        .map(|(_, (i, _))| {
            format!(
                "yaw={:+.1}deg pitch={:+.1}deg roll={:+.1}deg weight={:.2} active={}",
                i.yaw_radians.to_degrees(),
                i.pitch_radians.to_degrees(),
                i.roll_radians.to_degrees(),
                i.weight,
                i.active
            )
        })
        .unwrap_or_else(|| "<none>".to_string());
    let profile_text = root
        .and_then(|_| profile.ok())
        .map(|p| {
            format!(
                "small=({:.2},{:.2},{:.2},{:.2},{:.2},{:.2}) large=({:.2},{:.2},{:.2},{:.2},{:.2},{:.2}) pitch=({:.2},{:.2},{:.2},{:.2},{:.2},{:.2}) roll=({:.2},{:.2},{:.2},{:.2},{:.2},{:.2})",
                p.small_yaw_weights.head, p.small_yaw_weights.neck, p.small_yaw_weights.upper_chest, p.small_yaw_weights.chest, p.small_yaw_weights.spine, p.small_yaw_weights.hips,
                p.large_yaw_weights.head, p.large_yaw_weights.neck, p.large_yaw_weights.upper_chest, p.large_yaw_weights.chest, p.large_yaw_weights.spine, p.large_yaw_weights.hips,
                p.pitch_weights.head, p.pitch_weights.neck, p.pitch_weights.upper_chest, p.pitch_weights.chest, p.pitch_weights.spine, p.pitch_weights.hips,
                p.roll_weights.head, p.roll_weights.neck, p.roll_weights.upper_chest, p.roll_weights.chest, p.roll_weights.spine, p.roll_weights.hips,
            )
        })
        .unwrap_or_else(|| "<none>".to_string());
    let mut bone_report = String::new();
    if let Some(root) = root {
        bone_report.push_str(&format!(
            "model={:?} ",
            root_paths
                .get(root)
                .map(|p| p.0.display().to_string())
                .unwrap_or("?".to_string())
        ));
        if let Ok(binding) = binding {
            let marker = |present: bool| if present { "Y" } else { "N" };
            bone_report.push_str(&format!(
                "markers(upperChest={} chest={} head={}) ",
                marker(upper_chest_markers.get(root).is_ok()),
                marker(chest_markers.get(root).is_ok()),
                marker(head_markers.get(root).is_ok()),
            ));
            let entries = [
                ("head", Some(binding.head)),
                ("neck", binding.neck),
                ("upperChest", binding.upper_chest),
                ("chest", binding.chest),
                ("spine", binding.spine),
                ("hips", hips.get(root).ok().map(|h| h.0)),
                (
                    "lShoulder",
                    binding.left_arm.as_ref().and_then(|a| a.shoulder),
                ),
                ("lUpperArm", binding.left_arm.as_ref().map(|a| a.upper_arm)),
                ("lLowerArm", binding.left_arm.as_ref().map(|a| a.lower_arm)),
                (
                    "rShoulder",
                    binding.right_arm.as_ref().and_then(|a| a.shoulder),
                ),
                ("rUpperArm", binding.right_arm.as_ref().map(|a| a.upper_arm)),
                ("rLowerArm", binding.right_arm.as_ref().map(|a| a.lower_arm)),
            ];
            for (label, entity) in entries {
                let Some(entity) = entity else {
                    bone_report.push_str(&format!("{label}=- "));
                    continue;
                };
                match bones.get(entity) {
                    Ok((transform, rest)) => {
                        let delta = rest.0.rotation.inverse() * transform.rotation;
                        bone_report.push_str(&format!(
                            "{label}={:+.2}deg ",
                            delta.angle_between(Quat::IDENTITY).to_degrees()
                        ));
                    }
                    Err(_) => bone_report.push_str(&format!("{label}=<no-rest> ")),
                }
            }
        }
    }
    format!(
        "[propagation] frame={frame_counter}|state={:?}|input=({input_text})|profile({profile_text})|bones: {bone_report}|control_frame={}",
        lifecycle.state(),
        control_frame.frame.is_some()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vtuber_core::types::TrackingState;
    use vtuber_core::types::{
        ExpressionCoefficients, FrameSeq, HeadPose, HeadTranslationSignal, MonoTimeNs,
    };

    #[test]
    fn arm_probe_diagnostics_read_the_composed_bone_correctly() {
        // A T-pose arm, authored with a non-identity local rotation the way a real
        // VRM bone is, so the conjugation into model space is actually exercised.
        let rest_direction = Vec3::X;
        let rest_global = Quat::from_rotation_z(0.7);
        // A local rotation that must be conjugated before it means anything in
        // model space: the same local angle about a different axis is a different
        // model-space rotation.
        let written = Quat::from_rotation_y(0.5);
        let (dot, descent) = upper_arm_diagnostics(rest_direction, rest_global, written);
        let model_delta = rest_global * written * rest_global.inverse();
        let expected = model_delta * rest_direction;
        assert!((dot - expected.dot(rest_direction)).abs() < 1.0e-5);
        assert!(descent.is_finite());

        // At rest the arm points along its rest direction: dot 1, no descent.
        let (rest_dot, rest_descent) =
            upper_arm_diagnostics(rest_direction, rest_global, Quat::IDENTITY);
        assert!((rest_dot - 1.0).abs() < 1.0e-5, "rest dot was {rest_dot}");
        assert!(
            rest_descent.abs() < 1.0e-3,
            "rest descent was {rest_descent}"
        );

        // The antipode: the arm pointing opposite its rest direction is where the
        // shortest arc off rest has no defined axis. The probe must be able to
        // see the approach, so `dot` has to run down to -1 there.
        let (anti_dot, _) = upper_arm_diagnostics(
            rest_direction,
            rest_global,
            rest_global.inverse() * Quat::from_rotation_y(std::f32::consts::PI) * rest_global,
        );
        assert!(
            (anti_dot + 1.0).abs() < 1.0e-5,
            "the antipode must read as dot -1, got {anti_dot}"
        );
    }

    fn frame(state: TrackingState) -> AvatarControlFrame {
        AvatarControlFrame {
            source_seq: FrameSeq(1),
            captured_at: MonoTimeNs(2),
            produced_at: MonoTimeNs(3),
            confidence: 0.75,
            state,
            head: HeadPose {
                yaw_rad: 0.3,
                pitch_rad: 0.2,
                roll_rad: 0.1,
            },
            head_translation: HeadTranslationSignal::UNAVAILABLE,
            gaze: vtuber_core::GazeSignal::UNAVAILABLE,
            expressions: ExpressionCoefficients::default(),
            detailed_face: None,
        }
    }

    #[test]
    fn tracked_arm_text_reports_the_absence_of_a_frame() {
        let text = tracked_arm_text(
            crate::arm_pipeline::ArmSourceSelection::default(),
            crate::arm_pipeline::TrackedArmControl::default(),
            None,
        );
        assert!(text.contains("no-frame"), "{text}");
    }

    #[test]
    fn tracked_arm_text_reports_sequence_and_channel_weights() {
        let frame = vtuber_core::arm_tracking::ArmControlFrame {
            source_seq: FrameSeq(7),
            captured_at: MonoTimeNs(1),
            produced_at: MonoTimeNs(2),
            targets: Default::default(),
            weights: Default::default(),
        };
        let text = tracked_arm_text(
            crate::arm_pipeline::ArmSourceSelection::default(),
            crate::arm_pipeline::TrackedArmControl {
                generation: None,
                frame: Some(frame),
                view_to_model: Quat::IDENTITY,
            },
            None,
        );
        assert!(text.contains("seq=7"), "{text}");
        assert!(text.contains("L(w=0.00"), "{text}");
        assert!(text.contains("R(w=0.00"), "{text}");
    }

    #[test]
    fn tracked_pose_system_skips_when_not_ready() {
        let metrics = PoseApplyMetrics::default();
        assert_eq!(metrics.frames_applied, 0);
        assert_eq!(metrics.skipped_not_ready, 0);
    }

    #[test]
    fn tracked_pose_system_metrics_default() {
        let metrics = PoseApplyMetrics::default();
        assert_eq!(metrics.frames_applied, 0);
        assert_eq!(metrics.skipped_not_ready, 0);
        assert_eq!(metrics.skipped_generation_mismatch, 0);
        assert_eq!(metrics.skipped_no_frame, 0);
        assert_eq!(metrics.skipped_stale_entity, 0);
    }

    #[test]
    fn mirrored_tracking_frame_reflects_horizontal_pose_axes() {
        let input = body_tracking_input(&frame(TrackingState::Tracking), true);
        assert_eq!(input.yaw_radians, -0.3);
        assert_eq!(input.pitch_radians, 0.2);
        assert_eq!(input.roll_radians, -0.1);
        assert_eq!(input.weight, 0.75);
        assert!(input.active);
    }

    #[test]
    fn unmirrored_tracking_frame_preserves_canonical_pose_axes() {
        let input = body_tracking_input(&frame(TrackingState::Tracking), false);
        assert_eq!(input.yaw_radians, 0.3);
        assert_eq!(input.pitch_radians, 0.2);
        assert_eq!(input.roll_radians, 0.1);
    }

    #[test]
    fn pose_input_stays_active_so_loss_glides_instead_of_snapping() {
        // The control frame carries the loss glide and a confidence that
        // decays to zero, so the input must stay active: the dependency turns
        // an inactive input into an instant zero target.
        for state in [
            TrackingState::Tracking,
            TrackingState::Degraded,
            TrackingState::Starting,
            TrackingState::Searching,
            TrackingState::Acquiring,
            TrackingState::LostHold,
            TrackingState::ReturningNeutral,
        ] {
            let input = body_tracking_input(&frame(state), true);
            assert!(input.active, "state={state:?}");
            assert_eq!(input.weight, 0.75, "state={state:?}");
        }

        let lost_at_zero = AvatarControlFrame {
            confidence: 0.0,
            head: HeadPose::default(),
            ..frame(TrackingState::ReturningNeutral)
        };
        assert!(body_tracking_input(&lost_at_zero, true).active);
        assert_eq!(body_tracking_input(&lost_at_zero, true).weight, 0.0);
    }

    #[test]
    fn capture_to_apply_records_each_source_sequence_once() {
        let mut metrics = PoseApplyMetrics::default();
        metrics.record_apply(
            vtuber_core::FrameSeq(7),
            vtuber_core::MonoTimeNs(1_000_000),
            vtuber_core::MonoTimeNs(31_000_000),
        );
        metrics.record_apply(
            vtuber_core::FrameSeq(7),
            vtuber_core::MonoTimeNs(1_000_000),
            vtuber_core::MonoTimeNs(5_001_000_000),
        );

        assert_eq!(metrics.frames_applied, 2);
        assert_eq!(metrics.latency_sample_count(), 1);
        assert_eq!(metrics.capture_to_apply_p50_ms(), 30.0);
        assert_eq!(metrics.last_capture_to_apply_ms, Some(30.0));

        metrics.record_apply(
            vtuber_core::FrameSeq(8),
            vtuber_core::MonoTimeNs(6_000_000_000),
            vtuber_core::MonoTimeNs(6_040_000_000),
        );
        assert_eq!(metrics.latency_sample_count(), 2);
        assert_eq!(metrics.capture_to_apply_p95_ms(), 40.0);
    }
    #[test]
    fn idle_cross_fade_preserves_the_pose_contribution_at_zero_blend() {
        let idle = vtuber_tracking::IdleTarget::default();
        let input = body_tracking_input(&frame(TrackingState::Tracking), true);
        let faded = fade_pose_input(input, &idle, 1.0, 0.0);
        assert!(
            (faded.yaw_radians - input.yaw_radians).abs() < 1.0e-6
                && (faded.pitch_radians - input.pitch_radians).abs() < 1.0e-6
                && (faded.weight - input.weight).abs() < 1.0e-6
                && faded.active,
            "zero blend must be a no-op: {faded:?} vs {input:?}"
        );
    }

    #[test]
    fn idle_cross_fade_blends_in_the_sway_without_zeroing_the_pose() {
        // A loss while the head is turned: the pose authority has eased to
        // 0.4 and the idle sway is half faded in on top of it instead of
        // replacing it.
        let idle = vtuber_tracking::IdleTarget {
            yaw_radians: 0.05,
            pitch_radians: 0.02,
            ..vtuber_tracking::IdleTarget::default()
        };

        let eased = BodyTrackingPoseInput {
            yaw_radians: 0.5,
            pitch_radians: 0.3,
            roll_radians: 0.1,
            weight: 0.4,
            active: true,
        };
        let faded = fade_pose_input(eased, &idle, 1.0, 0.5);

        // The composed target is the exact cross-fade of the two products,
        // not a replacement that would zero the easing pose.
        let expected_yaw = 0.5 * (0.5 * 0.4) + 0.5 * idle.yaw_radians;
        assert!(
            (faded.yaw_radians * faded.weight - expected_yaw).abs() < 1.0e-5,
            "cross-fade product mismatch: {} vs {expected_yaw}",
            faded.yaw_radians * faded.weight
        );
        assert!(
            (faded.weight - (0.5 * 0.4 + 0.5)).abs() < 1.0e-6,
            "weight must cross-fade: {}",
            faded.weight
        );
    }

    #[test]
    fn idle_cross_fade_at_full_blend_is_the_idle_sway() {
        let idle = vtuber_tracking::IdleTarget {
            yaw_radians: 0.05,
            pitch_radians: 0.02,
            ..vtuber_tracking::IdleTarget::default()
        };

        let eased = BodyTrackingPoseInput {
            yaw_radians: 0.5,
            pitch_radians: 0.3,
            roll_radians: 0.1,
            weight: 0.0,
            active: true,
        };
        let faded = fade_pose_input(eased, &idle, 1.0, 1.0);
        assert!(
            (faded.yaw_radians - idle.yaw_radians).abs() < 1.0e-6
                && (faded.pitch_radians - idle.pitch_radians).abs() < 1.0e-6
                && (faded.weight - 1.0).abs() < 1.0e-6,
            "full blend must be the idle sway: {faded:?}"
        );
    }

    #[test]
    fn idle_cross_fade_fades_out_monotonically_after_reacquire() {
        let idle = vtuber_tracking::IdleTarget {
            yaw_radians: 0.05,
            ..vtuber_tracking::IdleTarget::default()
        };

        let tracked = BodyTrackingPoseInput {
            yaw_radians: 0.3,
            pitch_radians: 0.0,
            roll_radians: 0.0,
            weight: 0.9,
            active: true,
        };
        let mut previous_sway = f32::INFINITY;
        for step in 0..4u64 {
            let blend = 1.0 - step as f32 * 0.25;
            let faded = fade_pose_input(tracked, &idle, 1.0, blend);
            let contribution = faded.yaw_radians * faded.weight;
            let expected = blend * idle.yaw_radians + (1.0 - blend) * 0.3 * 0.9;
            assert!(
                (contribution - expected).abs() < 1.0e-5,
                "step {step}: {contribution} vs {expected}"
            );
            assert!(
                blend * idle.yaw_radians <= previous_sway + 1.0e-6,
                "step {step} sway grew"
            );
            previous_sway = blend * idle.yaw_radians;
        }

        // At zero blend the tracked pose is untouched.
        let zero = fade_pose_input(tracked, &idle, 1.0, 0.0);
        assert!(
            (zero.yaw_radians * zero.weight - 0.3 * 0.9).abs() < 1.0e-5,
            "zero blend must not alter the tracked pose"
        );
    }
}
