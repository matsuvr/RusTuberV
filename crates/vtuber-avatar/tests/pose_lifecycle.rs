// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Body-tracking input lifecycle integration tests.

use bevy::prelude::*;
use bevy_vrm1::prelude::BodyTrackingPoseInput;
use vtuber_avatar::AvatarMotionMirror;
use vtuber_avatar::binding::AvatarBinding;
use vtuber_avatar::lifecycle::{AvatarLifecycle, AvatarLifecycleState};
use vtuber_avatar::pose::{PoseApplyMetrics, update_body_tracking_pose_input};
use vtuber_avatar::unload::ActiveControlFrame;
use vtuber_core::types::{
    AvatarControlFrame, ExpressionCoefficients, FrameSeq, HeadPose, MonoTimeNs, TrackingState,
};

fn control_frame(state: TrackingState) -> AvatarControlFrame {
    AvatarControlFrame {
        source_seq: FrameSeq(7),
        captured_at: MonoTimeNs(1),
        produced_at: MonoTimeNs(2),
        confidence: 0.8,
        state,
        head: HeadPose {
            yaw_rad: 0.4,
            pitch_rad: 0.2,
            roll_rad: 0.1,
        },
        head_translation: vtuber_core::HeadTranslationSignal::UNAVAILABLE,
        gaze: vtuber_core::GazeSignal::UNAVAILABLE,
        expressions: ExpressionCoefficients::default(),
        detailed_face: None,
    }
}

fn ready_app(frame: Option<AvatarControlFrame>) -> (App, Entity) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .init_resource::<AvatarLifecycle>()
        .init_resource::<ActiveControlFrame>()
        .init_resource::<AvatarMotionMirror>()
        .init_resource::<PoseApplyMetrics>()
        .add_systems(PostUpdate, update_body_tracking_pose_input);

    let root = app.world_mut().spawn(BodyTrackingPoseInput::default()).id();
    let generation = {
        let mut lifecycle = app.world_mut().resource_mut::<AvatarLifecycle>();
        lifecycle.request_load(root).unwrap();
        lifecycle.start_binding(root);
        lifecycle.finish_ready();
        lifecycle.current_generation()
    };
    app.world_mut()
        .entity_mut(root)
        .insert(AvatarBinding::head_only(root, root, generation));
    app.world_mut()
        .insert_resource(ActiveControlFrame { generation, frame });
    (app, root)
}

#[test]
fn tracking_frame_updates_direct_input() {
    let (mut app, root) = ready_app(Some(control_frame(TrackingState::Tracking)));
    app.update();

    let input = app.world().get::<BodyTrackingPoseInput>(root).unwrap();
    assert!(input.active);
    assert_eq!(input.yaw_radians, -0.4);
    assert_eq!(input.pitch_radians, 0.2);
    assert_eq!(input.roll_radians, -0.1);
    assert_eq!(input.weight, 0.8);
}

#[test]
fn missing_frame_targets_neutral() {
    let (mut app, root) = ready_app(None);
    app.world_mut()
        .get_mut::<BodyTrackingPoseInput>(root)
        .unwrap()
        .active = true;
    app.update();

    assert_eq!(
        *app.world().get::<BodyTrackingPoseInput>(root).unwrap(),
        BodyTrackingPoseInput::default()
    );
}

#[test]
fn lost_tracking_targets_neutral_without_removing_input() {
    let (mut app, root) = ready_app(Some(control_frame(TrackingState::LostHold)));
    app.update();

    let input = app.world().get::<BodyTrackingPoseInput>(root).unwrap();
    assert!(!input.active);
    assert_eq!(input.yaw_radians, -0.4);
}

#[test]
fn generation_mismatch_deactivates_input() {
    let (mut app, root) = ready_app(Some(control_frame(TrackingState::Tracking)));
    app.world_mut()
        .resource_mut::<ActiveControlFrame>()
        .generation = Default::default();
    app.update();

    assert_eq!(
        *app.world().get::<BodyTrackingPoseInput>(root).unwrap(),
        BodyTrackingPoseInput::default()
    );
}

#[test]
fn non_ready_lifecycle_deactivates_all_inputs_without_panicking() {
    let (mut app, root) = ready_app(Some(control_frame(TrackingState::Tracking)));
    app.world_mut()
        .resource_mut::<AvatarLifecycle>()
        .request_unload()
        .unwrap();
    assert_ne!(
        app.world().resource::<AvatarLifecycle>().state(),
        AvatarLifecycleState::Ready
    );
    app.update();

    assert_eq!(
        *app.world().get::<BodyTrackingPoseInput>(root).unwrap(),
        BodyTrackingPoseInput::default()
    );
}
