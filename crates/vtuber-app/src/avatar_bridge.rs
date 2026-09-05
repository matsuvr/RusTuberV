//! Bridge from tracking's latest control slot to the active avatar generation.

use bevy::prelude::*;
use vtuber_avatar::{
    ActiveControlFrame, AvatarLifecycle, AvatarLifecycleState, PoseApplyMetrics,
    set_active_control_frame,
};

use crate::diagnostics::DiagnosticsSnapshot;
use crate::tracking_runtime::TrackingRuntime;

/// Publishes the latest tracking frame only while the active avatar is ready.
///
/// The avatar generation is attached on the Bevy side, so an old frame cannot
/// be applied to a replacement avatar. Frames produced while loading,
/// unloading, or failed are dropped and counted by the avatar apply systems.
pub fn publish_control_frame_system(
    mut tracking: ResMut<TrackingRuntime>,
    lifecycle: Res<AvatarLifecycle>,
    mut active: ResMut<ActiveControlFrame>,
) {
    if lifecycle.state() != AvatarLifecycleState::Ready || !tracking.control_active {
        active.frame = None;
        return;
    }
    let slot = tracking.control_slot();
    let generation = slot.generation();
    if generation == 0 {
        return;
    }
    // `TrackingRuntime` retains the last value, but this generation check
    // ensures the bridge publishes at most one copy per latest-slot advance.
    if active.frame.as_ref().is_some_and(|frame| {
        tracking
            .latest_control
            .as_ref()
            .is_some_and(|value| frame.produced_at == value.produced_at)
    }) {
        return;
    }
    let Some(frame) = tracking.latest_control.take() else {
        return;
    };
    let _ = set_active_control_frame(
        &lifecycle,
        lifecycle.current_generation(),
        frame,
        &mut active,
    );
}

/// Mirrors real avatar binding/apply metrics into the application diagnostics.
pub fn sync_avatar_diagnostics(
    lifecycle: Option<Res<AvatarLifecycle>>,
    pose_metrics: Option<Res<PoseApplyMetrics>>,
    mut diagnostics: ResMut<DiagnosticsSnapshot>,
) {
    if let Some(lifecycle) = lifecycle {
        // The capability summary only changes when the lifecycle resource is
        // mutated (bind/unbind); rebuilding its joined strings every frame is
        // pure allocation churn.
        if lifecycle.is_changed() {
            diagnostics.avatar_capabilities = lifecycle.capabilities().map(|caps| caps.summary());
        }
    }
    if let Some(metrics) = pose_metrics {
        diagnostics.avatar_frames_applied = metrics.frames_applied;
        diagnostics.avatar_frames_skipped = metrics
            .skipped_not_ready
            .saturating_add(metrics.skipped_generation_mismatch)
            .saturating_add(metrics.skipped_stale_entity);
        if metrics.latency_sample_count() > 0 {
            diagnostics.capture_to_apply_p50_ms = Some(metrics.capture_to_apply_p50_ms() as f32);
            diagnostics.capture_to_apply_p95_ms = Some(metrics.capture_to_apply_p95_ms() as f32);
        } else {
            diagnostics.capture_to_apply_p50_ms = None;
            diagnostics.capture_to_apply_p95_ms = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vtuber_avatar::lifecycle::AvatarLifecycle;

    #[test]
    fn inactive_tracking_session_clears_retained_control_frame() {
        let mut app = App::new();
        app.init_resource::<TrackingRuntime>()
            .init_resource::<AvatarLifecycle>()
            .init_resource::<ActiveControlFrame>()
            .add_systems(Update, publish_control_frame_system);

        let root = app.world_mut().spawn_empty().id();
        {
            let mut lifecycle = app.world_mut().resource_mut::<AvatarLifecycle>();
            lifecycle.request_load(root).unwrap();
            lifecycle.start_binding(root);
            lifecycle.finish_ready();
        }
        let generation = app
            .world()
            .resource::<AvatarLifecycle>()
            .current_generation();
        app.world_mut()
            .resource_mut::<ActiveControlFrame>()
            .generation = generation;
        app.world_mut().resource_mut::<ActiveControlFrame>().frame =
            Some(vtuber_core::AvatarControlFrame {
                source_seq: vtuber_core::FrameSeq(1),
                captured_at: vtuber_core::MonoTimeNs(1),
                produced_at: vtuber_core::MonoTimeNs(1),
                confidence: 1.0,
                state: vtuber_core::TrackingState::Tracking,
                head: vtuber_core::HeadPose::default(),
                head_translation: vtuber_core::HeadTranslationSignal::UNAVAILABLE,
                gaze: vtuber_core::GazeSignal::UNAVAILABLE,
                expressions: vtuber_core::ExpressionCoefficients::default(),
                detailed_face: None,
            });

        app.update();

        assert!(app.world().resource::<ActiveControlFrame>().frame.is_none());
    }

    #[test]
    fn ready_avatar_receives_the_latest_real_tracking_frame() {
        let mut app = App::new();
        app.init_resource::<TrackingRuntime>()
            .init_resource::<AvatarLifecycle>()
            .init_resource::<ActiveControlFrame>()
            .add_systems(Update, publish_control_frame_system);

        let root = app.world_mut().spawn_empty().id();
        {
            let mut lifecycle = app.world_mut().resource_mut::<AvatarLifecycle>();
            lifecycle.request_load(root).unwrap();
            lifecycle.start_binding(root);
            lifecycle.finish_ready();
        }
        let expected_generation = app
            .world()
            .resource::<AvatarLifecycle>()
            .current_generation();

        let frame = vtuber_core::AvatarControlFrame {
            source_seq: vtuber_core::FrameSeq(4),
            captured_at: vtuber_core::MonoTimeNs(10),
            produced_at: vtuber_core::MonoTimeNs(12),
            confidence: 0.9,
            state: vtuber_core::TrackingState::Tracking,
            head: vtuber_core::HeadPose {
                yaw_rad: 0.2,
                ..Default::default()
            },
            head_translation: vtuber_core::HeadTranslationSignal::UNAVAILABLE,
            gaze: vtuber_core::GazeSignal::UNAVAILABLE,
            expressions: vtuber_core::ExpressionCoefficients::default(),
            detailed_face: None,
        };
        {
            let mut tracking = app.world_mut().resource_mut::<TrackingRuntime>();
            tracking.control_active = true;
            tracking.latest_control = Some(frame.clone());
            assert!(tracking.control_slot.publish(frame));
        }

        app.update();

        let active = app.world().resource::<ActiveControlFrame>();
        assert_eq!(active.generation, expected_generation);
        assert_eq!(
            active.frame.as_ref().map(|frame| frame.source_seq.0),
            Some(4)
        );
    }
}
