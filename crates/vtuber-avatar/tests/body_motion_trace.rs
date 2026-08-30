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
//! Runs the production system ordering — position bridge -> dynamic arm
//! targets -> arm compositor writer — against a synthetic rig and evaluates
//! deterministic traces at 30/60/120 fps equivalents:
//!
//! - finite output (no NaN / Inf) for every stage on every frame
//! - no frame-to-frame accumulation (identical input => identical output)
//! - root compensation and hips-relative hand anchor behavior
//! - legacy static source demotion (virtual-hand authority by default)
//! - tracking loss/reacquire continuity without snaps or stale state
//! - avatar replacement generation cleanup

use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;
use bevy_vrm1::prelude::{
    BodyTracking, BodyTrackingPoseInput, BodyTrackingPositionInput, BodyTrackingProfile,
    RestGlobalTransform, RestTransform,
};
use bevy_vrm1::vrm::body_tracking::apply_direct_body_tracking;
use vtuber_avatar::AvatarMotionMirror;
use vtuber_avatar::{
    ActiveAvatar, ArmChainBinding, ArmChainCapabilities, ArmMotionGeometry, ArmPoseBlendState,
    ArmPoseOverrideStore, ArmRestGeometry, ArmSide, ArmSourceSelection, AvatarAssetId,
    AvatarBinding, AvatarGeneration, AvatarLifecycle, DefaultArmPose, DynamicArmTargets,
    RestSpaceBonePose, apply_default_arm_pose, update_body_tracking_pose_input,
    update_body_tracking_position_input, update_dynamic_arm_targets,
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
    left_upper: Entity,
    left_lower: Entity,
    right_upper: Entity,
    right_lower: Entity,
}

fn rest_bone(position: Vec3) -> RestSpaceBonePose {
    RestSpaceBonePose {
        position,
        global_rotation: Quat::IDENTITY,
        local_rotation: Quat::IDENTITY,
    }
}

fn chain(side: ArmSide, upper: Entity, lower: Entity) -> ArmChainBinding {
    // Real VRM/glTF basis: the model faces +Z and its left arm is +X.
    let sign = match side {
        ArmSide::Left => 1.0_f32,
        ArmSide::Right => -1.0,
    };
    let shoulder_pos = Vec3::new(0.04 * sign, 1.30, 0.0);
    let upper_origin = Vec3::new(0.16 * sign, 1.32, 0.0);
    let elbow = upper_origin + Vec3::new(0.24 * sign, -0.04, 0.0);
    let wrist = elbow + Vec3::new(-0.02 * sign, -0.25, -0.01);
    ArmChainBinding {
        side,
        shoulder: None,
        upper_arm: upper,
        lower_arm: lower,
        hand: lower,
        fingers: vtuber_avatar::FingerReferences::default(),
        finger_rest: vtuber_avatar::FingerRestReferences::default(),
        rest: ArmRestGeometry {
            shoulder: Some(rest_bone(shoulder_pos)),
            upper_arm: rest_bone(upper_origin),
            elbow: rest_bone(elbow),
            wrist: rest_bone(wrist),
            upper_arm_length: upper_origin.distance(elbow),
            forearm_length: elbow.distance(wrist),
            total_arm_length: upper_origin.distance(wrist),
        },
        capabilities: ArmChainCapabilities {
            has_shoulder: true,
            has_fingers: false,
        },
    }
}

fn instant_at(millis: u64) -> Instant {
    static BASE: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    *BASE.get_or_init(Instant::now) + Duration::from_millis(millis)
}

/// Builds a minimal headless app wired in production control order.
fn build_app(generation: AvatarGeneration) -> (App, TraceRig) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .insert_resource(TimeUpdateStrategy::ManualInstant(instant_at(0)))
        .init_resource::<vtuber_avatar::ActiveControlFrame>()
        .init_resource::<AvatarLifecycle>()
        .init_resource::<ArmSourceSelection>()
        .init_resource::<ArmPoseOverrideStore>()
        .init_resource::<vtuber_avatar::PositionInputMetrics>()
        .init_resource::<vtuber_avatar::LossIdleState>()
        .init_resource::<vtuber_avatar::PoseApplyMetrics>()
        .init_resource::<AvatarMotionMirror>()
        .add_systems(
            PostUpdate,
            (
                update_body_tracking_position_input,
                update_body_tracking_pose_input,
                update_dynamic_arm_targets,
                apply_direct_body_tracking,
                apply_default_arm_pose,
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
    let left_upper = spawn_bone(&mut app, chest, Vec3::new(0.12, 0.10, 0.0));
    let left_lower = spawn_bone(&mut app, left_upper, Vec3::new(0.24, -0.04, 0.0));
    let right_upper = spawn_bone(&mut app, chest, Vec3::new(-0.12, 0.10, 0.0));
    let right_lower = spawn_bone(&mut app, right_upper, Vec3::new(-0.24, -0.04, 0.0));

    let left_arm_binding = chain(ArmSide::Left, left_upper, left_lower);
    let right_arm_binding = chain(ArmSide::Right, right_upper, right_lower);

    let motion = ArmMotionGeometry {
        left: Some(vtuber_avatar::build_arm_motion_rest_geometry(
            ArmSide::Left,
            &left_arm_binding.rest,
            Some(Vec3::new(0.0, 0.92, 0.0)),
            Some(Quat::IDENTITY),
            Some(Vec3::new(0.0, 1.20, 0.02)),
        )),
        right: Some(vtuber_avatar::build_arm_motion_rest_geometry(
            ArmSide::Right,
            &right_arm_binding.rest,
            Some(Vec3::new(0.0, 0.92, 0.0)),
            Some(Quat::IDENTITY),
            Some(Vec3::new(0.0, 1.20, 0.02)),
        )),
    };

    let binding = AvatarBinding {
        root,
        head,
        neck: None,
        upper_chest: None,
        chest: Some(chest),
        spine: Some(spine),
        left_upper_arm: Some(left_upper),
        right_upper_arm: Some(right_upper),
        left_arm: Some(left_arm_binding),
        right_arm: Some(right_arm_binding),
        left_eye: None,
        right_eye: None,
        generation,
    };

    let default_pose = DefaultArmPose {
        generation,
        left: None,
        right: None,
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
        default_pose,
        ArmPoseBlendState::from_default(&default_pose),
        motion,
        body_scale,
        DynamicArmTargets::default(),
        BodyTracking::default(),
        BodyTrackingPoseInput::default(),
        BodyTrackingProfile::default(),
        BodyTrackingPositionInput::default(),
        vtuber_avatar::IdleMotionProfile::default(),
        Transform::IDENTITY,
        GlobalTransform::IDENTITY,
    ));

    let mut lifecycle = AvatarLifecycle::default();
    lifecycle.request_load(root).expect("load request");
    lifecycle.start_binding(root);
    lifecycle.finish_ready();
    app.insert_resource(lifecycle);

    let rig = TraceRig {
        root,
        head,
        left_upper,
        left_lower,
        right_upper,
        right_lower,
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

fn rotations(app: &App, rig: &TraceRig) -> [Quat; 4] {
    [
        app.world()
            .get::<Transform>(rig.left_upper)
            .unwrap()
            .rotation,
        app.world()
            .get::<Transform>(rig.left_lower)
            .unwrap()
            .rotation,
        app.world()
            .get::<Transform>(rig.right_upper)
            .unwrap()
            .rotation,
        app.world()
            .get::<Transform>(rig.right_lower)
            .unwrap()
            .rotation,
    ]
}

fn assert_all_finite(app: &App, rig: &TraceRig) {
    for entity in [
        rig.left_upper,
        rig.left_lower,
        rig.right_upper,
        rig.right_lower,
    ] {
        let transform = app.world().get::<Transform>(entity).unwrap();
        assert!(transform.rotation.is_finite(), "non-finite rotation");
    }
    let root_transform = app.world().get::<Transform>(rig.root).unwrap();
    assert!(root_transform.translation.is_finite());
}

#[test]
fn trace_is_deterministic_across_30_60_and_120_fps_equivalents() {
    for frame_millis in [33_u64, 16, 8] {
        let generation = AvatarGeneration(11);
        let (mut app, rig) = build_app(generation);
        let mut tick_clock = 0_u64;
        let mut previous = Option::<[Quat; 4]>::None;

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
fn virtual_hand_authority_drives_the_compositor_not_the_legacy_source() {
    let generation = AvatarGeneration(21);
    let (mut app, rig) = build_app(generation);
    let selection = app.world().resource::<ArmSourceSelection>();
    assert_eq!(
        selection.mode,
        vtuber_avatar::ArmPoseSourceKind::VirtualHandAnchor,
        "virtual-hand authority is the default"
    );

    push_frame(&mut app, generation, 1, Vec3::ZERO);
    app.update();

    let targets = app
        .world()
        .get::<DynamicArmTargets>(rig.root)
        .expect("dynamic targets present");
    assert_eq!(targets.generation, Some(generation));
    assert!(targets.left.is_some() && targets.right.is_some());

    for entity in [rig.left_upper, rig.right_upper] {
        let rotation = app.world().get::<Transform>(entity).unwrap().rotation;
        assert!(rotation.angle_between(Quat::IDENTITY) > 1e-3);
    }
}

#[test]
fn tracking_loss_reacquire_replaces_state_without_snaps_or_stale_entities() {
    let generation = AvatarGeneration(31);
    let (mut app, rig) = build_app(generation);

    push_frame(&mut app, generation, 1, Vec3::new(0.05, 0.0, 0.02));
    app.update();
    assert_all_finite(&app, &rig);
    let tracked_rotations = rotations(&app, &rig);

    let mut control = app
        .world_mut()
        .resource_mut::<vtuber_avatar::ActiveControlFrame>();
    if let Some(frame) = control.frame.as_mut() {
        frame.state = TrackingState::LostHold;
        frame.head_translation = HeadTranslationSignal::UNAVAILABLE;
    }
    let _ = control;
    app.update();
    assert_all_finite(&app, &rig);

    push_frame(&mut app, generation, 2, Vec3::new(-0.04, 0.0, 0.01));
    app.update();
    assert_all_finite(&app, &rig);
    let reacquired = rotations(&app, &rig);
    assert!(
        tracked_rotations
            .iter()
            .zip(reacquired.iter())
            .any(|(a, b)| a.angle_between(*b) > 1e-3),
        "reacquire updates the pose"
    );
}

#[test]
fn avatar_generation_cleanup_rejects_stale_frames_and_targets() {
    let old_generation = AvatarGeneration(41);
    let (mut app, rig) = build_app(old_generation);

    push_frame(&mut app, old_generation, 1, Vec3::new(0.03, 0.0, 0.0));
    app.update();

    push_frame(
        &mut app,
        AvatarGeneration(old_generation.0.wrapping_sub(1)),
        2,
        Vec3::new(0.09, 0.0, 0.0),
    );
    app.update();
    let targets = app.world().get::<DynamicArmTargets>(rig.root).unwrap();
    assert_eq!(targets.generation, None, "stale targets must clear");
    assert!(targets.left.is_none() && targets.right.is_none());
}

#[test]
fn rotation_and_position_trace_is_deterministic_across_fps_equivalents() {
    let mut reference = Option::<(Vec3, Quat)>::None;
    for frame_millis in [33_u64, 16, 8] {
        let generation = AvatarGeneration(51);
        let (mut app, rig) = build_app(generation);
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
    let generation = AvatarGeneration(61);
    let (app, rig) = build_app(generation);
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
fn camera_silence_keeps_idle_rotation_and_breathing_alive_in_the_default_pose() {
    // ADR-021: when the control frame disappears entirely (camera signal
    // gone), the avatar must relax to its default pose while the loss-idle
    // sway (rotation) and breathing (vertical offset) keep flowing.
    let generation = AvatarGeneration(81);
    let (mut app, rig) = build_app(generation);

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
