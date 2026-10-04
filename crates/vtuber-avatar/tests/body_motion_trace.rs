// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Headless trace validation for the upper-body motion pipeline
//! (Body Motion 11/11, Issue #173).
//!
//! Runs the position/body bridge at 30/60/120 fps. Arm integration, mirror,
//! held observations and loss are exercised with the constrained runtime's
//! complete fixed-length rig in `upper_limb_runtime::tests`.

use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;
use bevy_vrm1::prelude::{BodyTracking, RestGlobalTransform, RestTransform};
use vtuber_avatar::{
    ActiveAvatar, AvatarAssetId, AvatarBinding, AvatarGeneration, AvatarLifecycle,
    AvatarMotionMirror, BodyTrackingPoseInput, BodyTrackingPositionInput, BodyTrackingProfile,
    apply_direct_body_tracking, update_body_tracking_pose_input,
    update_body_tracking_position_input,
};
use vtuber_core::types::AvatarControlFrame;
use vtuber_core::types::{
    ExpressionCoefficients, FrameSeq, GazeSignal, HeadPose, HeadTranslationSignal, MonoTimeNs,
    TrackingState,
};

const EPSILON: f32 = 1.0e-4;

#[derive(Clone, Copy)]
struct TraceRig {
    root: Entity,
    head: Entity,
    chest: Entity,
    spine: Entity,
}

fn instant_at(millis: u64) -> Instant {
    static BASE: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    *BASE.get_or_init(Instant::now) + Duration::from_millis(millis)
}

/// Builds a minimal headless app wired in production control order.
fn build_app() -> (App, TraceRig) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .insert_resource(TimeUpdateStrategy::ManualInstant(instant_at(0)))
        .init_resource::<vtuber_avatar::ActiveControlFrame>()
        .init_resource::<AvatarLifecycle>()
        .init_resource::<vtuber_avatar::PositionInputMetrics>()
        .init_resource::<vtuber_avatar::LossIdleState>()
        .init_resource::<vtuber_avatar::BodyFollowFilter>()
        .init_resource::<vtuber_avatar::PoseApplyMetrics>()
        .init_resource::<AvatarMotionMirror>()
        .add_systems(
            PostUpdate,
            (
                update_body_tracking_position_input,
                update_body_tracking_pose_input,
                apply_direct_body_tracking,
            )
                .chain(),
        );

    let root = app
        .world_mut()
        .spawn((Transform::IDENTITY, GlobalTransform::IDENTITY))
        .id();
    let spawn_bone = |app: &mut App, parent: Entity, offset: Vec3| {
        app.world_mut()
            .spawn((
                Transform::from_translation(offset),
                GlobalTransform::IDENTITY,
                RestTransform(Transform::from_translation(offset)),
                RestGlobalTransform(GlobalTransform::from_translation(offset)),
                ChildOf(parent),
            ))
            .id()
    };
    let spine = spawn_bone(&mut app, root, Vec3::Y * 0.12);
    let chest = spawn_bone(&mut app, spine, Vec3::Y * 0.14);
    let head = spawn_bone(&mut app, chest, Vec3::Y * 0.18);
    let mut lifecycle = AvatarLifecycle::default();
    lifecycle.request_load(root).expect("load request");
    lifecycle.start_binding(root);
    lifecycle.finish_ready();
    let generation = lifecycle.current_generation();
    app.insert_resource(lifecycle);

    let binding = AvatarBinding {
        root,
        head,
        neck: None,
        upper_chest: None,
        chest: Some(chest),
        spine: Some(spine),
        left_upper_arm: None,
        right_upper_arm: None,
        left_arm: None,
        right_arm: None,
        left_eye: None,
        right_eye: None,
        generation,
    };

    let body_scale = vtuber_avatar::body_scale::BodyScaleMeters {
        generation,
        scale_meters: 0.7,
    };
    let model_id = AvatarAssetId::new("sha256:trace-model");
    app.world_mut().entity_mut(root).insert((
        ActiveAvatar,
        binding,
        model_id,
        body_scale,
        BodyTracking::default(),
        BodyTrackingPoseInput::default(),
        BodyTrackingProfile::default(),
        BodyTrackingPositionInput::default(),
        vtuber_avatar::IdleMotionProfile::default(),
        Transform::IDENTITY,
        GlobalTransform::IDENTITY,
    ));

    let rig = TraceRig {
        root,
        head,
        chest,
        spine,
    };
    (app, rig)
}

fn tracked_frame(seq: u64, translation: Vec3) -> AvatarControlFrame {
    AvatarControlFrame {
        source_seq: FrameSeq(seq),
        captured_at: MonoTimeNs(0),
        produced_at: MonoTimeNs(0),
        confidence: 1.0,
        state: TrackingState::Tracking,
        head: HeadPose::default(),
        head_translation: HeadTranslationSignal::tracked(
            translation.x,
            translation.y,
            translation.z,
        ),
        gaze: GazeSignal::UNAVAILABLE,
        expressions: ExpressionCoefficients::default(),
        detailed_face: None,
    }
}

fn tracked_frame_with_head(seq: u64, translation: Vec3, head: HeadPose) -> AvatarControlFrame {
    AvatarControlFrame {
        source_seq: FrameSeq(seq),
        captured_at: MonoTimeNs(0),
        produced_at: MonoTimeNs(0),
        confidence: 1.0,
        state: TrackingState::Tracking,
        head,
        head_translation: HeadTranslationSignal::tracked(
            translation.x,
            translation.y,
            translation.z,
        ),
        gaze: GazeSignal::UNAVAILABLE,
        expressions: ExpressionCoefficients::default(),
        detailed_face: None,
    }
}

fn push_frame(app: &mut App, generation: AvatarGeneration, seq: u64, translation: Vec3) {
    push_frame_inner(app, generation, tracked_frame(seq, translation));
}

fn push_frame_with_head(
    app: &mut App,
    generation: AvatarGeneration,
    seq: u64,
    translation: Vec3,
    head: HeadPose,
) {
    push_frame_inner(
        app,
        generation,
        tracked_frame_with_head(seq, translation, head),
    );
}

fn push_frame_inner(app: &mut App, generation: AvatarGeneration, frame: AvatarControlFrame) {
    let mut control = app
        .world_mut()
        .resource_mut::<vtuber_avatar::ActiveControlFrame>();
    control.generation = generation;
    control.frame = Some(frame);
}

fn rotations(app: &App, rig: &TraceRig) -> [Quat; 3] {
    [rig.head, rig.chest, rig.spine]
        .map(|bone| app.world().get::<Transform>(bone).unwrap().rotation)
}

fn assert_all_finite(app: &App, rig: &TraceRig) {
    for entity in [rig.head, rig.chest, rig.spine] {
        let transform = app.world().get::<Transform>(entity).unwrap();
        assert!(transform.rotation.is_finite(), "non-finite rotation");
    }
    let root_transform = app.world().get::<Transform>(rig.root).unwrap();
    assert!(root_transform.translation.is_finite());
}

#[test]
fn trace_is_deterministic_across_30_60_and_120_fps_equivalents() {
    for frame_millis in [33_u64, 16, 8] {
        let (mut app, rig) = build_app();
        let generation = app
            .world()
            .resource::<AvatarLifecycle>()
            .current_generation();
        let mut tick_clock = 0_u64;
        let mut previous = Option::<[Quat; 3]>::None;

        for _ in 0..5 {
            tick_clock += frame_millis;
            if let Some(mut strategy) = app.world_mut().get_resource_mut::<TimeUpdateStrategy>() {
                *strategy = TimeUpdateStrategy::ManualInstant(instant_at(tick_clock));
            }
            app.update();
            assert_all_finite(&app, &rig);
        }

        let sway = Vec3::new(0.06, 0.01, 0.03);
        for step in 0..6u64 {
            push_frame(&mut app, generation, step + 1, sway);
            tick_clock += frame_millis;
            if let Some(mut strategy) = app.world_mut().get_resource_mut::<TimeUpdateStrategy>() {
                *strategy = TimeUpdateStrategy::ManualInstant(instant_at(tick_clock));
            }
            app.update();
            assert_all_finite(&app, &rig);
            let current = rotations(&app, &rig);
            if let Some(previous) = &previous {
                for (a, b) in previous.iter().zip(current.iter()) {
                    assert!(
                        a.angle_between(*b) < EPSILON,
                        "identical input must not accumulate"
                    );
                }
            }
            previous = Some(current);
        }

        let metrics = app
            .world()
            .resource::<vtuber_avatar::PositionInputMetrics>();
        assert!(metrics.frames_published > 0, "position channel published");
    }
}

#[test]
fn rotation_and_position_trace_is_deterministic_across_fps_equivalents() {
    let mut reference = Option::<(Vec3, Quat)>::None;
    for frame_millis in [33_u64, 16, 8] {
        let (mut app, rig) = build_app();
        let generation = app
            .world()
            .resource::<AvatarLifecycle>()
            .current_generation();
        let mut tick_clock = 0_u64;

        for step in 0..40u64 {
            let phase = step as f32 / 40.0;
            let head = HeadPose {
                yaw_rad: 0.5 * (phase * std::f32::consts::TAU).sin(),
                pitch_rad: 0.2 * (phase * std::f32::consts::TAU).cos(),
                roll_rad: 0.1 * phase,
            };
            let translation = Vec3::new(0.04 * phase, 0.01 * phase, -0.02 * phase);
            push_frame_with_head(&mut app, generation, step + 1, translation, head);

            tick_clock += frame_millis;
            if let Some(mut strategy) = app.world_mut().get_resource_mut::<TimeUpdateStrategy>() {
                *strategy = TimeUpdateStrategy::ManualInstant(instant_at(tick_clock));
            }
            app.update();
            assert_all_finite(&app, &rig);
        }

        let hold = HeadPose {
            yaw_rad: 0.5 * ((39.0f32 / 40.0) * std::f32::consts::TAU).sin(),
            pitch_rad: 0.2 * ((39.0f32 / 40.0) * std::f32::consts::TAU).cos(),
            roll_rad: 0.1 * (39.0f32 / 40.0),
        };
        let hold_translation = Vec3::new(0.04, 0.01, -0.02);
        push_frame_with_head(&mut app, generation, 100, hold_translation, hold);
        tick_clock += frame_millis;
        if let Some(mut strategy) = app.world_mut().get_resource_mut::<TimeUpdateStrategy>() {
            *strategy = TimeUpdateStrategy::ManualInstant(instant_at(tick_clock));
        }
        app.update();
        let held_head = app.world().get::<Transform>(rig.head).unwrap().rotation;
        let held_root = app.world().get::<Transform>(rig.root).unwrap().translation;
        push_frame_with_head(&mut app, generation, 101, hold_translation, hold);
        tick_clock += frame_millis;
        if let Some(mut strategy) = app.world_mut().get_resource_mut::<TimeUpdateStrategy>() {
            *strategy = TimeUpdateStrategy::ManualInstant(instant_at(tick_clock));
        }
        app.update();
        let head = app.world().get::<Transform>(rig.head).unwrap().rotation;
        let root_translation = app.world().get::<Transform>(rig.root).unwrap().translation;
        assert!(
            held_head.angle_between(head) < EPSILON,
            "rotation output must not accumulate at {frame_millis}ms"
        );
        assert!(
            held_root.abs_diff_eq(root_translation, EPSILON),
            "root translation output must not accumulate at {frame_millis}ms"
        );

        let current = (root_translation, head);
        if let Some((reference_root, reference_head)) = reference {
            assert!(
                reference_head.angle_between(head) < EPSILON,
                "cross-fps rotation drift: {reference_head} vs {head} at {frame_millis}ms"
            );
            assert!(
                reference_root.abs_diff_eq(root_translation, EPSILON),
                "cross-fps root translation drift at {frame_millis}ms"
            );
        }
        reference = Some(current);
    }
}

#[test]
fn idle_amplitude_in_the_trace_is_zero_by_policy() {
    let (app, rig) = build_app();

    let profile = app
        .world()
        .get::<vtuber_avatar::IdleMotionProfile>(rig.root)
        .copied()
        .expect("idle profile present on the trace rig");
    assert_eq!(profile.validate(), Ok(()));
    assert_eq!(
        profile.procedural_amplitude_meters,
        vtuber_avatar::IDLE_PROCEDURAL_AMPLITUDE_METERS
    );
}

#[test]
fn camera_silence_publishes_idle_inputs_and_clears_tracked_position_targets() {
    // ADR-021: when the control frame disappears entirely (camera signal
    // gone), the bridge publishes idle sway and breathing inputs while
    // clearing tracked position targets. This checks input publication.
    let (mut app, rig) = build_app();
    let generation = app
        .world()
        .resource::<AvatarLifecycle>()
        .current_generation();

    // Establish tracking once so the "silence" is a real transition.
    push_frame(&mut app, generation, 1, Vec3::new(0.04, 0.0, 0.01));
    app.update();

    // Clear the frame entirely: no camera signal, no pipeline output.
    let mut control = app
        .world_mut()
        .resource_mut::<vtuber_avatar::ActiveControlFrame>();
    control.frame = None;
    let _ = control;

    let mut tick_clock = 0_u64;
    let mut saw_active_rotation_idle = false;
    let mut saw_breathing_offset = false;
    // 400 ticks at 16 ms = 6.4 s: past the 4 s fade-in of the envelope.
    for _step in 0..400u64 {
        tick_clock += 16;
        if let Some(mut strategy) = app.world_mut().get_resource_mut::<TimeUpdateStrategy>() {
            *strategy = TimeUpdateStrategy::ManualInstant(instant_at(tick_clock));
        }
        app.update();
        assert_all_finite(&app, &rig);

        let pose_input = app
            .world()
            .get::<BodyTrackingPoseInput>(rig.root)
            .copied()
            .expect("pose input on the active root");
        if pose_input.active && pose_input.weight > 0.5 {
            saw_active_rotation_idle = true;
        }
        let position_input = app
            .world()
            .get::<BodyTrackingPositionInput>(rig.root)
            .copied()
            .expect("position input on the active root");
        assert_eq!(position_input.tracked_head_target, Vec3::ZERO);
        assert_eq!(position_input.tracked_body_target, Vec3::ZERO);
        if position_input.active && position_input.head_offset.y.abs() > 1.0e-4 {
            saw_breathing_offset = true;
        }
    }
    assert!(
        saw_active_rotation_idle,
        "idle rotation input never became active during camera silence"
    );
    assert!(
        saw_breathing_offset,
        "breathing offset never became visible during camera silence"
    );
}
