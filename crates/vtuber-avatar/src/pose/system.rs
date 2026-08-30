//! Direct-pose body-tracking input bridge.
//!
//! Reads the latest [`ActiveControlFrame`] and updates
//! [`BodyTrackingPoseInput`](bevy_vrm1::prelude::BodyTrackingPoseInput) on the
//! active avatar root. Bone transforms are owned exclusively by
//! `bevy_vrm1::BodyTracking`.

use bevy::prelude::*;
use vtuber_core::metrics::FixedStats;
use vtuber_core::monotonic_now;
use vtuber_core::types::{AvatarControlFrame, TrackingState};

use bevy_vrm1::prelude::BodyTrackingPoseInput;

use crate::binding::AvatarBinding;
use crate::lifecycle::{AvatarLifecycle, AvatarLifecycleState};
use crate::mirror::AvatarMotionMirror;
use crate::unload::ActiveControlFrame;

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

/// System that updates the direct pose consumed by `bevy_vrm1::BodyTracking`.
///
/// # Schedule
///
/// Runs in `PostUpdate`, after `AnimationSystems`. It does not write any bone
/// `Transform`; the dependency-owned direct body-tracking system is the sole
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
    if lifecycle.state() != AvatarLifecycleState::Ready {
        deactivate_inputs(&mut inputs);
        metrics.skipped_not_ready += 1;
        return;
    }

    let active_root = match lifecycle.active_root() {
        Some(root) => root,
        None => {
            deactivate_inputs(&mut inputs);
            metrics.skipped_not_ready += 1;
            return;
        }
    };

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

    let binding = match binding_query.get(active_root) {
        Ok(b) => b,
        Err(_) => {
            *input = BodyTrackingPoseInput::default();
            metrics.skipped_stale_entity += 1;
            return;
        }
    };

    if control_frame.generation != binding.generation {
        *input = BodyTrackingPoseInput::default();
        metrics.skipped_generation_mismatch += 1;
        return;
    }

    *input = body_tracking_input(frame, mirrored);

    // Tracking-loss idle (Issue #172): while the micro-motion episode blends
    // in and the frame is not actively tracked, the pose input carries the
    // bounded idle yaw/pitch instead of a plain inactive marker. Roll stays
    // zero per the analyzed design. This keeps this system the sole pose
    // input writer; the idle contract (issue #180) adds no procedural motion.
    let tracked = matches!(
        frame.state,
        TrackingState::Tracking | TrackingState::Degraded
    );
    if !tracked && let Some(idle_state) = idle_state.as_ref() {
        let blend = idle_state.blend();
        if blend > 0.0 {
            let idle = idle_state.target();
            let horizontal_sign = if mirrored { -1.0 } else { 1.0 };
            *input = BodyTrackingPoseInput {
                yaw_radians: horizontal_sign * idle.yaw_radians,
                pitch_radians: idle.pitch_radians,
                roll_radians: 0.0,
                weight: blend.clamp(0.0, 1.0),
                active: true,
            };
        }
    }

    let applied_at = monotonic_now();
    metrics.record_apply(frame.source_seq, frame.captured_at, applied_at);
}

fn body_tracking_input(frame: &AvatarControlFrame, mirrored: bool) -> BodyTrackingPoseInput {
    let active = matches!(
        frame.state,
        TrackingState::Tracking | TrackingState::Degraded
    );
    let horizontal_sign = if mirrored { -1.0 } else { 1.0 };
    BodyTrackingPoseInput {
        // A horizontal reflection preserves pitch but reverses yaw and roll.
        yaw_radians: horizontal_sign * frame.head.yaw_rad,
        pitch_radians: frame.head.pitch_rad,
        roll_radians: horizontal_sign * frame.head.roll_rad,
        weight: frame.confidence,
        active,
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

#[cfg(test)]
mod tests {
    use super::*;
    use vtuber_core::types::{
        ExpressionCoefficients, FrameSeq, HeadPose, HeadTranslationSignal, MonoTimeNs,
    };

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
    fn degraded_tracking_remains_weighted_and_loss_targets_neutral() {
        assert!(body_tracking_input(&frame(TrackingState::Degraded), true).active);
        for state in [
            TrackingState::Starting,
            TrackingState::Searching,
            TrackingState::Acquiring,
            TrackingState::LostHold,
            TrackingState::ReturningNeutral,
        ] {
            assert!(
                !body_tracking_input(&frame(state), true).active,
                "state={state:?}"
            );
        }
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
}
