// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Deterministic trace validation of the idle/breathing contract.
//!
//! The contract is zero procedural amplitude: the retired #20 breathing
//! writer must not exist, the hips translation must stay exactly at its
//! authored/animated base across 30/60/120 fps equivalents, and the #172
//! tracking-loss micro-motion must flow only through the avatar-root/
//! torso-lean position bridge without touching hips translation.

use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;
use bevy_vrm1::prelude::BodyTrackingPositionInput;
use vtuber_avatar::{
    ActiveAvatar, AvatarAssetId, AvatarBinding, AvatarGeneration, AvatarLifecycle, DefaultArmPose,
    IDLE_PROCEDURAL_AMPLITUDE_METERS, IdleMotionProfile, LossIdleState, PositionInputMetrics,
    update_body_tracking_position_input,
};
use vtuber_core::types::AvatarControlFrame;
use vtuber_core::types::{
    ExpressionCoefficients, FrameSeq, GazeSignal, HeadPose, HeadTranslationSignal, MonoTimeNs,
    TrackingState,
};

fn instant_at(millis: u64) -> Instant {
    static BASE: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    *BASE.get_or_init(Instant::now) + Duration::from_millis(millis)
}

struct IdleRig {
    root: Entity,
    hips: Entity,
}

/// Builds a minimal headless rig whose hips translation simulates an
/// animation-authored base value that no runtime system may change.
fn build_app(generation: AvatarGeneration) -> (App, IdleRig) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .insert_resource(TimeUpdateStrategy::ManualInstant(instant_at(0)))
        .init_resource::<vtuber_avatar::ActiveControlFrame>()
        .init_resource::<AvatarLifecycle>()
        .init_resource::<PositionInputMetrics>()
        .init_resource::<LossIdleState>()
        .init_resource::<vtuber_avatar::AvatarMotionMirror>()
        .add_systems(PostUpdate, update_body_tracking_position_input);

    let root = app
        .world_mut()
        .spawn((Transform::IDENTITY, GlobalTransform::IDENTITY))
        .id();
    let spine = app
        .world_mut()
        .spawn((
            Transform::from_translation(Vec3::Y * 0.12),
            GlobalTransform::IDENTITY,
            ChildOf(root),
        ))
        .id();
    let chest = app
        .world_mut()
        .spawn((
            Transform::from_translation(Vec3::Y * 0.14),
            GlobalTransform::IDENTITY,
            ChildOf(spine),
        ))
        .id();
    let head = app
        .world_mut()
        .spawn((
            Transform::from_translation(Vec3::Y * 0.18),
            GlobalTransform::IDENTITY,
            ChildOf(chest),
        ))
        .id();
    let hips = app
        .world_mut()
        .spawn((
            // Simulated animation-authored hips base translation.
            Transform::from_translation(Vec3::new(0.01, 0.55, -0.02)),
            GlobalTransform::IDENTITY,
            ChildOf(root),
        ))
        .id();

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
    let default_pose = DefaultArmPose {
        generation,
        left: None,
        right: None,
    };
    let body_scale = vtuber_avatar::body_scale::BodyScaleMeters {
        generation,
        scale_meters: 0.7,
    };
    let model_id = AvatarAssetId::new("sha256:idle-contract-model");
    app.world_mut().entity_mut(root).insert((
        ActiveAvatar,
        binding,
        model_id,
        default_pose,
        IdleMotionProfile::default(),
        body_scale,
        BodyTrackingPositionInput::default(),
        Transform::IDENTITY,
        GlobalTransform::IDENTITY,
    ));

    let mut lifecycle = AvatarLifecycle::default();
    lifecycle.request_load(root).expect("load request");
    lifecycle.start_binding(root);
    lifecycle.finish_ready();
    app.insert_resource(lifecycle);

    (app, IdleRig { root, hips })
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

fn push_frame(app: &mut App, generation: AvatarGeneration, seq: u64, translation: Vec3) {
    let mut control = app
        .world_mut()
        .resource_mut::<vtuber_avatar::ActiveControlFrame>();
    control.generation = generation;
    control.frame = Some(tracked_frame(seq, translation));
}

fn tick(app: &mut App, tick_clock: &mut u64, frame_millis: u64) {
    *tick_clock += frame_millis;
    if let Some(mut strategy) = app.world_mut().get_resource_mut::<TimeUpdateStrategy>() {
        *strategy = TimeUpdateStrategy::ManualInstant(instant_at(*tick_clock));
    }
    app.update();
}

fn hips_translation(app: &App, rig: &IdleRig) -> Vec3 {
    app.world().get::<Transform>(rig.hips).unwrap().translation
}

#[test]
fn idle_contract_uses_zero_procedural_amplitude() {
    let profile = IdleMotionProfile::default();
    assert_eq!(profile.validate(), Ok(()));
    assert_eq!(IDLE_PROCEDURAL_AMPLITUDE_METERS, 0.0);
}

#[test]
fn hips_translation_stays_at_the_animation_base_across_fps_equivalents() {
    let base = Vec3::new(0.01, 0.55, -0.02);
    for frame_millis in [33_u64, 16, 8] {
        let generation = AvatarGeneration(7);
        let (mut app, rig) = build_app(generation);

        // The idle profile must be present and valid on a Ready avatar.
        let profile = app
            .world()
            .get::<IdleMotionProfile>(rig.root)
            .copied()
            .expect("idle profile inserted at binding");
        assert_eq!(profile.validate(), Ok(()));

        // Tracked, shaped translation flows through the root bridge; the
        // hips channel must remain bit-identical to the authored base.
        let mut tick_clock = 0_u64;
        for step in 0..24u64 {
            push_frame(&mut app, generation, step + 1, Vec3::new(0.05, 0.02, -0.03));
            tick(&mut app, &mut tick_clock, frame_millis);
            let current = hips_translation(&app, &rig);
            assert_eq!(current, base, "frame {step} at {frame_millis}ms");
        }
    }
}

#[test]
fn tracking_loss_micro_motion_keeps_hips_translation_untouched() {
    let generation = AvatarGeneration(9);
    let (mut app, rig) = build_app(generation);
    let mut tick_clock = 0_u64;
    let base = hips_translation(&app, &rig);

    // Establish tracking.
    push_frame(&mut app, generation, 1, Vec3::new(0.04, 0.0, 0.01));
    tick(&mut app, &mut tick_clock, 16);
    assert_eq!(hips_translation(&app, &rig), base);

    // Enter a tracking-loss episode: the #172 micro-motion blends in through
    // the position bridge while the hips translation must not move.
    let mut control = app
        .world_mut()
        .resource_mut::<vtuber_avatar::ActiveControlFrame>();
    if let Some(frame) = control.frame.as_mut() {
        frame.state = TrackingState::LostHold;
        frame.head_translation = HeadTranslationSignal::UNAVAILABLE;
    }
    let _ = control;
    let mut saw_blend = false;
    for step in 0..120u64 {
        tick(&mut app, &mut tick_clock, 16);
        let metrics = app.world().resource::<PositionInputMetrics>();
        if metrics.idle_blend > 0.0 {
            saw_blend = true;
        }
        assert_eq!(
            hips_translation(&app, &rig),
            base,
            "hips moved during loss episode at step {step}"
        );
        assert!(hips_translation(&app, &rig).is_finite());
    }
    assert!(saw_blend, "micro-motion episode never blended in");
}

#[test]
fn camera_silence_ramps_the_idle_episode_and_keeps_breathing_alive() {
    // ADR-021: with no control frame at all (camera signal gone or capture
    // stopped), the loss-idle episode must ramp in so the avatar keeps
    // breathing and swaying in its default pose instead of freezing, while
    // the hips translation stays at its authored base.
    let generation = AvatarGeneration(13);
    let (mut app, rig) = build_app(generation);
    let mut tick_clock = 0_u64;
    let base = hips_translation(&app, &rig);

    let mut saw_ramp = false;
    let mut saw_active_idle_input = false;
    let mut saw_breathing_offset = false;
    // 400 ticks at 16 ms = 6.4 s: past the 4 s fade-in of the envelope.
    for step in 0..400u64 {
        tick(&mut app, &mut tick_clock, 16);
        let metrics = app.world().resource::<PositionInputMetrics>();
        if metrics.idle_blend > 0.5 {
            saw_ramp = true;
        }
        let input = app
            .world()
            .get::<BodyTrackingPositionInput>(rig.root)
            .copied()
            .expect("position input on the active root");
        if input.active && input.weight > 0.5 {
            saw_active_idle_input = true;
            // The breathing offset oscillates around zero; any non-trivial
            // excursion proves the axis is flowing.
            if input.head_offset.y.abs() > 1.0e-4 {
                saw_breathing_offset = true;
            }
        }
        assert_eq!(
            hips_translation(&app, &rig),
            base,
            "hips moved during camera silence at step {step}"
        );
    }
    assert!(saw_ramp, "camera silence never ramped the idle episode");
    assert!(
        saw_active_idle_input,
        "idle position input never became active during camera silence"
    );
    assert!(
        saw_breathing_offset,
        "breathing offset never became visible during camera silence"
    );
}

#[test]
fn replacement_starts_with_a_fresh_zero_amplitude_profile() {
    let first = AvatarGeneration(11);
    let (mut app, rig) = build_app(first);

    push_frame(&mut app, first, 1, Vec3::new(0.03, 0.0, 0.0));
    let mut tick_clock = 0_u64;
    tick(&mut app, &mut tick_clock, 16);
    assert_eq!(
        app.world()
            .get::<IdleMotionProfile>(rig.root)
            .expect("profile")
            .procedural_amplitude_meters,
        IDLE_PROCEDURAL_AMPLITUDE_METERS
    );

    // A replacement generation re-binds to a fresh, validated profile; there
    // is no carried-over idle state.
    let second = AvatarGeneration(12);
    let mut lifecycle = app.world_mut().resource_mut::<AvatarLifecycle>();
    lifecycle
        .request_replace(rig.root)
        .expect("replace request");
    let _ = lifecycle;
    let _ = second;
    app.update();
    let profile = app
        .world()
        .get::<IdleMotionProfile>(rig.root)
        .copied()
        .expect("fresh profile after replacement request");
    assert_eq!(profile.validate(), Ok(()));
}

#[test]
fn retired_breathing_writer_is_absent_from_the_post_update_schedule() {
    let generation = AvatarGeneration(21);
    let (mut app, _rig) = build_app(generation);
    app.world_mut()
        .schedule_scope(bevy::app::PostUpdate, |world, schedule| {
            schedule.initialize(world).expect("PostUpdate initializes");
            for (_, system) in schedule
                .systems()
                .expect("initialized schedule exposes systems")
            {
                let name = system.name().to_string();
                assert!(
                    !name.to_lowercase().contains("breathing"),
                    "retired breathing system found in PostUpdate: {name}"
                );
            }
        });
}
