//! Position-aware upper-body motion bridge (Body Motion 5/11, Issue #167).
//!
//! Consumes the active [`AvatarControlFrame`], shapes the neutral-relative
//! head translation with the Issue #164 soft-cap profile, splits it into
//! virtual head/body targets with the Issue #165 axis-selective policy, and
//! feeds both channels to `bevy_vrm1::BodyTrackingPositionInput`.
//!
//! Writer ownership stays unique per Transform channel:
//!
//! - head/neck/upper-chest/chest/spine **rotation**: `bevy_vrm1` direct-pose
//!   writer (`update_body_tracking_pose_input` is its only input writer).
//! - hips translation: no runtime writer. The idle contract retires the #20
//!   breathing writer; the authored or animated rest value is the idle value.
//! - avatar-root translation + torso lean: `bevy_vrm1`
//!   `apply_direct_body_position`, whose only input writer is this module.
//!
//! This system never touches camera transforms/projections or VRM generation
//! normalization; mirroring is a semantic flip of the lateral axis applied in
//! one place, matching the pose bridge.

use bevy::prelude::*;
use bevy_vrm1::prelude::BodyTrackingPositionInput;

use vtuber_core::types::AvatarControlFrame;
use vtuber_core::types::MonoTimeNs;
use vtuber_tracking::micro_motion::{MicroMotionBlender, is_tracked_state};
use vtuber_tracking::{
    IdleTarget, MicroMotionProfile, TranslationShapingProfile, VirtualBodyProfile,
    blended_idle_target, build_virtual_body_targets, shape_translation,
};

use crate::binding::AvatarBinding;
use crate::body_scale::{BodyScaleMeters, DEFAULT_BODY_SCALE_METERS};
use crate::lifecycle::{AvatarGeneration, AvatarLifecycle, AvatarLifecycleState};
use crate::mirror::AvatarMotionMirror;
use crate::unload::ActiveControlFrame;

/// Typed profiles for the shaping/split pipeline.
///
/// Defaults are aggregated here so no magic constant leaks into systems.
#[derive(Resource, Debug, Clone, Default)]
pub struct BodyMotionProfiles {
    /// Soft-cap thresholds applied before splitting (Issue #164).
    pub shaping: TranslationShapingProfile,
    /// Axis-selective root compensation policy (Issue #165).
    pub split: VirtualBodyProfile,
}

/// Computes the positional channels for one control frame.
///
/// Returns `(head_offset, body_offset)` in semantic meters, or `None` when
/// the frame carries no usable translation observation. Mirroring flips only
/// the lateral axis of both channels, consistent with the rotation bridge's
/// yaw/roll reflection and [`vtuber_core::types::HeadTranslationSignal`].
#[must_use]
pub fn position_channels(
    frame: &AvatarControlFrame,
    mirrored: bool,
    profiles: &BodyMotionProfiles,
    body_scale_meters: f32,
) -> Option<(Vec3, Vec3)> {
    let shaped = shape_translation(frame.head_translation, &profiles.shaping, body_scale_meters);
    let targets =
        build_virtual_body_targets(&shaped, frame.head, &profiles.split, body_scale_meters);
    let available = matches!(
        targets.head.state,
        vtuber_core::types::HeadTranslationState::Tracked
            | vtuber_core::types::HeadTranslationState::Degraded
    );
    if !available {
        return None;
    }
    let sign = if mirrored { -1.0 } else { 1.0 };
    Some((
        Vec3::new(
            sign * targets.head.translation.x,
            targets.head.translation.y,
            targets.head.translation.z,
        ),
        Vec3::new(
            sign * targets.body_compensation.x,
            targets.body_compensation.y,
            targets.body_compensation.z,
        ),
    ))
}

fn neutral_input() -> BodyTrackingPositionInput {
    BodyTrackingPositionInput::default()
}

/// Process-agnostic monotonic timestamp from the Bevy clock.
///
/// The loss-idle envelope only consumes elapsed differences, so aligning it
/// with the render clock (instead of a second process-wide epoch) keeps the
/// behaviour identical while remaining deterministic under manual time
/// strategies in headless tests.
fn monotonic_from_time(time: &Time) -> MonoTimeNs {
    // The float-to-int cast saturates instead of panicking; `elapsed` is
    // non-negative and far below the u64 nanosecond range in practice.
    MonoTimeNs((time.elapsed_secs_f64() * 1_000_000_000.0) as u64)
}

fn deactivate_inputs(inputs: &mut Query<&mut BodyTrackingPositionInput>) {
    for mut input in inputs.iter_mut() {
        *input = neutral_input();
    }
}

/// System that updates [`BodyTrackingPositionInput`] on the active avatar root.
///
/// # Schedule
///
/// Runs in `PostUpdate` after `AnimationSystems`, before the dependency-owned
/// direct body-tracking systems. It writes only the position input component;
/// bone and root transforms are owned exclusively by `bevy_vrm1`.
///
/// # Skip conditions
///
/// - Lifecycle is not `Ready`
/// - No active control frame
/// - Generation mismatch between frame and binding
/// - Position input component is missing from the active root
#[allow(clippy::too_many_arguments)]
pub fn update_body_tracking_position_input(
    lifecycle: Res<AvatarLifecycle>,
    control_frame: Res<ActiveControlFrame>,
    mirror: Option<Res<AvatarMotionMirror>>,
    profiles: Option<Res<BodyMotionProfiles>>,
    time: Res<Time>,
    mut metrics: ResMut<PositionInputMetrics>,
    mut idle_state: ResMut<LossIdleState>,
    binding_query: Query<&AvatarBinding>,
    scale_query: Query<&BodyScaleMeters>,
    mut inputs: Query<&mut BodyTrackingPositionInput>,
) {
    if lifecycle.state() != AvatarLifecycleState::Ready {
        deactivate_inputs(&mut inputs);
        metrics.skipped_not_ready += 1;
        return;
    }
    let Some(active_root) = lifecycle.active_root() else {
        deactivate_inputs(&mut inputs);
        metrics.skipped_not_ready += 1;
        return;
    };
    let Ok(mut input) = inputs.get_mut(active_root) else {
        metrics.skipped_stale_entity += 1;
        return;
    };

    let default_profiles = BodyMotionProfiles::default();
    let profiles = match &profiles {
        Some(res) => &**res,
        None => &default_profiles,
    };

    let Ok(binding) = binding_query.get(active_root) else {
        *input = neutral_input();
        metrics.skipped_stale_entity += 1;
        return;
    };
    let body_scale = scale_query
        .get(active_root)
        .map(|scale| scale.scale_meters)
        .unwrap_or(DEFAULT_BODY_SCALE_METERS);
    let mirrored = mirror.is_none_or(|mirror| mirror.is_enabled());

    let Some(frame) = &control_frame.frame else {
        // Camera signal gone or no control frame yet: treat the silence as a
        // prolonged loss episode so the bounded idle sway and breathing
        // (ADR-021) keep the avatar alive in its default pose instead of
        // freezing the stream on a statue.
        let now = monotonic_from_time(&time);
        idle_state.prepare(binding.generation, false, body_scale, now);
        metrics.idle_blend = idle_state.blend();
        if idle_state.blend() > 0.0 {
            let idle = idle_state.target();
            *input = BodyTrackingPositionInput {
                head_offset: Vec3::new(
                    sign_for(mirrored) * idle.translation_x,
                    idle.translation_y,
                    idle.translation_z,
                ),
                body_offset: Vec3::ZERO,
                weight: idle_state.blend(),
                active: true,
            };
            metrics.frames_published += 1;
        } else {
            *input = neutral_input();
            metrics.skipped_no_frame += 1;
        }
        return;
    };
    if control_frame.generation != binding.generation {
        *input = neutral_input();
        metrics.skipped_generation_mismatch += 1;
        return;
    }

    // Advance (or start) the generation-scoped loss-idle episode. The blend
    // and target computed here are also read by the pose bridge for the
    // yaw/pitch idle component, keeping both channels coherent.
    metrics.idle_blend = 0.0;
    if control_frame.generation == binding.generation {
        let tracked = is_tracked_state(frame.state);
        idle_state.prepare(
            control_frame.generation,
            tracked,
            body_scale,
            monotonic_from_time(&time),
        );
        metrics.idle_blend = idle_state.blend();
    }

    match position_channels(frame, mirrored, profiles, body_scale) {
        Some((mut head_offset, mut body_offset)) => {
            // Tracking-loss semantics match the rotation path: only live
            // tracking states drive the solve; other states hold nothing.
            let active = is_tracked_state(frame.state);
            if active {
                *input = BodyTrackingPositionInput {
                    head_offset,
                    body_offset,
                    weight: frame.confidence.clamp(0.0, 1.0),
                    active: true,
                };
                metrics.frames_published += 1;
                metrics.last_applied_source_seq = Some(frame.source_seq);
            } else if idle_state.blend() > 0.0 {
                // Lost but blending in: publish only the idle sway and the
                // loss-scoped breathing offset.
                let idle = idle_state.target();
                head_offset.x = sign_for(mirrored) * idle.translation_x;
                head_offset.z = idle.translation_z;
                head_offset.y = idle.translation_y;
                body_offset = Vec3::ZERO;
                *input = BodyTrackingPositionInput {
                    head_offset,
                    body_offset,
                    weight: idle_state.blend(),
                    active: true,
                };
                metrics.frames_published += 1;
                metrics.last_applied_source_seq = Some(frame.source_seq);
            } else {
                *input = neutral_input();
                metrics.frames_unavailable += 1;
            }
        }
        None => {
            // Unavailable translation must not zero out a tracked rotation
            // pose; while an idle episode blends, the sway keeps flowing.
            if !is_tracked_state(frame.state) && idle_state.blend() > 0.0 {
                let idle = idle_state.target();
                *input = BodyTrackingPositionInput {
                    head_offset: Vec3::new(
                        sign_for(mirrored) * idle.translation_x,
                        idle.translation_y,
                        idle.translation_z,
                    ),
                    body_offset: Vec3::ZERO,
                    weight: idle_state.blend(),
                    active: true,
                };
                metrics.frames_published += 1;
                metrics.last_applied_source_seq = Some(frame.source_seq);
            } else {
                *input = neutral_input();
                metrics.frames_unavailable += 1;
            }
        }
    }
}

fn sign_for(mirrored: bool) -> f32 {
    if mirrored { -1.0 } else { 1.0 }
}

/// Generation-scoped tracking-loss idle episode state.
///
/// Holds the micro-motion blender and the envelope computed for the most
/// recent control-frame evaluation. The pose bridge reads the same envelope
/// so translation and rotation idle components stay coherent without either
/// system writing the other's channel.
#[derive(Resource, Debug, Clone)]
pub struct LossIdleState {
    /// Generation this episode belongs to; a mismatch resets the episode.
    generation: Option<AvatarGeneration>,
    profile: MicroMotionProfile,
    blender: Option<MicroMotionBlender>,
    blend: f32,
    target: IdleTarget,
}

impl Default for LossIdleState {
    fn default() -> Self {
        Self {
            generation: None,
            profile: MicroMotionProfile::default(),
            blender: None,
            blend: 0.0,
            target: IdleTarget::default(),
        }
    }
}

impl LossIdleState {
    /// Advances the episode for one control-frame evaluation.
    ///
    /// `now` is the current monotonic timestamp (the render clock), so the
    /// envelope is deterministic under manual time strategies in tests.
    pub fn prepare(
        &mut self,
        generation: AvatarGeneration,
        tracked: bool,
        body_scale_meters: f32,
        now: MonoTimeNs,
    ) {
        if self.generation != Some(generation) {
            self.generation = Some(generation);
            self.blender = None;
            self.blend = 0.0;
            self.target = IdleTarget::default();
        }
        let tracked_blend = match self.blender.as_mut() {
            Some(blender) => {
                let blend = blender.update(tracked, now);
                let elapsed = blender.elapsed_since_idle_start(now);
                let target = blended_idle_target(
                    &self.profile,
                    generation.0,
                    elapsed,
                    body_scale_meters,
                    blend,
                );
                self.target = target;
                blend
            }
            None => {
                // An invalid profile disables idle motion entirely instead
                // of panicking; live tracking is unaffected.
                self.blender = MicroMotionBlender::new(&self.profile).ok();
                self.target = IdleTarget::default();
                0.0
            }
        };
        self.blend = tracked_blend;
    }

    /// Clears all episode state (avatar replacement / unload).
    pub fn reset(&mut self) {
        self.generation = None;
        self.blender = None;
        self.blend = 0.0;
        self.target = IdleTarget::default();
    }

    /// Blend factor computed for the current control frame in `[0, 1]`.
    #[must_use]
    pub const fn blend(&self) -> f32 {
        self.blend
    }

    /// Blended idle target computed for the current control frame.
    #[must_use]
    pub const fn target(&self) -> &IdleTarget {
        &self.target
    }
}

/// Diagnostics counters for the position-input bridge.
#[derive(Resource, Debug, Clone, Default)]
pub struct PositionInputMetrics {
    /// Frames where a live position channel was published.
    pub frames_published: u64,
    /// Frames where the translation observation was unavailable.
    pub frames_unavailable: u64,
    /// Frames skipped because lifecycle was not ready.
    pub skipped_not_ready: u64,
    /// Frames skipped because no control frame was available.
    pub skipped_no_frame: u64,
    /// Frames skipped because the input component or binding was missing.
    pub skipped_stale_entity: u64,
    /// Frames skipped because of a generation mismatch.
    pub skipped_generation_mismatch: u64,
    /// Source sequence of the most recently published channel.
    pub last_applied_source_seq: Option<vtuber_core::FrameSeq>,
    /// Loss-idle blend factor computed for the current evaluation.
    pub idle_blend: f32,
}

/// System that resets position-input metrics when lifecycle changes.
pub fn reset_position_metrics_on_lifecycle_change(
    lifecycle: Res<AvatarLifecycle>,
    mut metrics: ResMut<PositionInputMetrics>,
    mut idle_state: ResMut<LossIdleState>,
    mut last_state: Local<Option<AvatarLifecycleState>>,
) {
    let current = lifecycle.state();
    if last_state.as_ref() != Some(&current) {
        *metrics = PositionInputMetrics::default();
        // Idle episodes never survive an avatar lifecycle transition.
        idle_state.reset();
        *last_state = Some(current);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vtuber_core::types::{
        ExpressionCoefficients, FrameSeq, GazeSignal, HeadPose, HeadTranslationSignal, MonoTimeNs,
        TrackingState,
    };

    fn frame_with_translation(x: f32, y: f32, z: f32) -> AvatarControlFrame {
        AvatarControlFrame {
            source_seq: FrameSeq(1),
            captured_at: MonoTimeNs(0),
            produced_at: MonoTimeNs(0),
            confidence: 1.0,
            state: TrackingState::Tracking,
            head: HeadPose::default(),
            head_translation: HeadTranslationSignal::tracked(x, y, z),
            gaze: GazeSignal::UNAVAILABLE,
            expressions: ExpressionCoefficients::default(),
            detailed_face: None,
        }
    }

    #[test]
    fn default_policy_routes_no_lateral_motion_to_the_root() {
        let profiles = BodyMotionProfiles::default();
        let (head, body) = position_channels(
            &frame_with_translation(0.06, 0.02, 0.04),
            false,
            &profiles,
            0.7,
        )
        .expect("available");
        assert_eq!(body.x, 0.0);
        assert!((head.x - 0.06).abs() < 1.0e-6);
        // Y/Z are routed mostly to the root per profile gains.
        assert!(body.y > head.y);
        assert!(body.z > head.z);
    }

    #[test]
    fn mirroring_flips_only_the_lateral_axis() {
        let profiles = BodyMotionProfiles::default();
        let unmirrored = position_channels(
            &frame_with_translation(0.06, 0.02, 0.04),
            false,
            &profiles,
            0.7,
        )
        .unwrap();
        let mirrored = position_channels(
            &frame_with_translation(0.06, 0.02, 0.04),
            true,
            &profiles,
            0.7,
        )
        .unwrap();
        assert!((mirrored.0.x + unmirrored.0.x).abs() < 1.0e-6);
        assert!((mirrored.1.x + unmirrored.1.x).abs() < 1.0e-6);
        assert_relative_vec3(mirrored.0.yz(), unmirrored.0.yz());
        assert_relative_vec3(mirrored.1.yz(), unmirrored.1.yz());
    }

    fn assert_relative_vec3(a: bevy::math::Vec2, b: bevy::math::Vec2) {
        assert!(a.distance(b) < 1.0e-6, "{a:?} vs {b:?}");
    }

    #[test]
    fn unavailable_translation_yields_no_channels() {
        let profiles = BodyMotionProfiles::default();
        let mut frame = frame_with_translation(0.0, 0.0, 0.0);
        frame.head_translation = HeadTranslationSignal::UNAVAILABLE;
        assert!(position_channels(&frame, false, &profiles, 0.7).is_none());
    }

    #[test]
    fn identical_inputs_reproduce_identical_channels() {
        let profiles = BodyMotionProfiles::default();
        let frame = frame_with_translation(0.05, -0.02, 0.03);
        let a = position_channels(&frame, false, &profiles, 0.85);
        let b = position_channels(&frame, false, &profiles, 0.85);
        assert_eq!(a, b);
    }
}

#[cfg(test)]
mod loss_idle_state_tests {
    use super::*;
    use crate::lifecycle::AvatarGeneration;

    #[test]
    fn idle_episode_ramps_after_loss_and_resets_on_generation_change() {
        let mut state = LossIdleState::default();
        let generation = AvatarGeneration(1);
        let t = |millis: u64| MonoTimeNs(millis * 1_000_000);

        // Live tracking: no episode, no motion.
        state.prepare(generation, true, 0.7, t(0));
        assert_eq!(state.blend(), 0.0);

        // Loss: blend starts at zero (no snap) and ramps with elapsed time.
        state.prepare(generation, false, 0.7, t(16));
        assert!(
            state.blend() <= 1e-3,
            "no snap after loss: {}",
            state.blend()
        );
        state.prepare(generation, false, 0.7, t(2_016));
        assert!(
            (state.blend() - 0.5).abs() < 0.01,
            "half-way through the transition: {}",
            state.blend()
        );

        // Generation change clears the stale episode entirely.
        state.prepare(AvatarGeneration(2), false, 0.7, t(3_000));
        assert!(state.blend() <= 1e-3);
    }

    #[test]
    fn explicit_reset_clears_everything() {
        let mut state = LossIdleState::default();
        state.prepare(AvatarGeneration(3), false, 0.7, MonoTimeNs(0));
        state.reset();
        assert_eq!(state.blend(), 0.0);
        assert_eq!(*state.target(), IdleTarget::default());
        state.prepare(AvatarGeneration(3), false, 0.7, MonoTimeNs(1_000));
        assert!(state.blend() <= 1e-3, "fresh episode restarts from zero");
    }
}
