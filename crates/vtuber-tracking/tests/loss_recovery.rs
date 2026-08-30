// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Integration tests for loss hold, neutral decay, and recovery blend.
//!
//! These tests verify that `LossRecovery` behaves deterministically using
//! only caller-supplied durations and monotonic timestamps, without any
//! wall-clock dependency.

use std::time::Duration;

use approx::assert_relative_eq;

use vtuber_core::types::{
    AvatarControlFrame, ExpressionCoefficients, FrameSeq, GazeSignal, HeadPose,
    HeadTranslationSignal, MonoTimeNs, TrackingState,
};
use vtuber_core::{ARKIT52_CHANNEL_COUNT, Arkit52Coefficients, ArkitBlendshape};
use vtuber_tracking::loss_recovery::{
    LossRecovery, LossRecoveryConfigError, LossRecoveryParams, MAX_DECAY_DURATION,
    MAX_GLIDE_DURATION, MAX_RECOVERY_DURATION, MIN_DECAY_DURATION, MIN_GLIDE_DURATION,
    MIN_RECOVERY_DURATION,
};
use vtuber_tracking::pose::semantic_pose_to_quaternion;

fn test_params() -> LossRecoveryParams {
    LossRecoveryParams {
        glide_duration: Duration::from_millis(100),
        decay_duration: Duration::from_millis(200),
        recovery_duration: Duration::from_millis(100),
        ..LossRecoveryParams::default()
    }
}

fn frame(seq: u64, yaw: f32, pitch: f32, roll: f32, expression_value: f32) -> AvatarControlFrame {
    AvatarControlFrame {
        source_seq: FrameSeq(seq),
        captured_at: MonoTimeNs(seq * 33_333_333),
        produced_at: MonoTimeNs(seq * 33_333_333),
        confidence: 0.9,
        state: TrackingState::Tracking,
        head: HeadPose {
            yaw_rad: yaw,
            pitch_rad: pitch,
            roll_rad: roll,
        },
        head_translation: HeadTranslationSignal::UNAVAILABLE,
        gaze: GazeSignal::UNAVAILABLE,
        expressions: ExpressionCoefficients {
            aa: expression_value,
            ..ExpressionCoefficients::default()
        },
        detailed_face: None,
    }
}

fn rotation_angle_from_identity(pose: HeadPose) -> f32 {
    semantic_pose_to_quaternion(pose).angle()
}

fn detailed_frame(seq: u64, coefficient: f32) -> AvatarControlFrame {
    let mut values = [0.0; ARKIT52_CHANNEL_COUNT];
    values[ArkitBlendshape::TongueOut.index()] = coefficient;
    let mut frame = frame(seq, 0.2, -0.1, 0.05, 0.4);
    frame.detailed_face = Some(Arkit52Coefficients::try_from_array(values).unwrap());
    frame
}

#[test]
fn loss_recovery_hold_preserves_source_sequence() {
    let mut recovery = LossRecovery::new(test_params()).unwrap();
    let tracked = frame(7, 0.3, 0.1, -0.2, 0.5);

    let _ = recovery.update(
        TrackingState::Tracking,
        Duration::from_millis(16),
        Some(tracked.clone()),
        MonoTimeNs(16_000_000),
    );
    let held = recovery
        .update(
            TrackingState::LostHold,
            Duration::from_millis(50),
            None,
            MonoTimeNs(66_000_000),
        )
        .expect("glided frame should be emitted");

    assert_eq!(held.source_seq, tracked.source_seq);
    assert_eq!(held.captured_at, tracked.captured_at);
    assert_eq!(held.state, TrackingState::LostHold);
    assert!(
        rotation_angle_from_identity(held.head) > 0.01,
        "glided pose should not already be neutral"
    );
}

#[test]
fn loss_recovery_neutral_return_uses_shortest_arc() {
    let mut recovery = LossRecovery::new(test_params()).unwrap();
    // Yaw just shy of +pi. The shortest arc to identity stays positive and
    // decreases in magnitude; the long way would wrap through negative yaw.
    let tracked = frame(1, 179.0f32.to_radians(), 0.0, 0.0, 0.0);

    let _ = recovery.update(
        TrackingState::Tracking,
        Duration::from_millis(16),
        Some(tracked.clone()),
        MonoTimeNs(16_000_000),
    );
    // Spend exactly the glide duration in LostHold (stationary frames glide in place).
    let _ = recovery.update(
        TrackingState::LostHold,
        Duration::from_millis(100),
        None,
        MonoTimeNs(116_000_000),
    );

    let mut last_yaw = tracked.head.yaw_rad;
    let mut last_angle = rotation_angle_from_identity(tracked.head);

    for step in 1..=5 {
        let out = recovery
            .update(
                TrackingState::ReturningNeutral,
                Duration::from_millis(40),
                None,
                MonoTimeNs(116_000_000 + step as u64 * 40_000_000),
            )
            .expect("returning frame should be emitted");

        let angle = rotation_angle_from_identity(out.head);
        assert!(
            angle <= last_angle + 1e-5,
            "rotation angle should not increase during return: step {step}: {angle} > {last_angle}"
        );
        assert!(
            out.head.yaw_rad >= -0.01,
            "shortest arc should stay on the positive side of yaw: step {step}: {}",
            out.head.yaw_rad
        );
        assert!(
            out.head.yaw_rad.abs() <= last_yaw.abs() + 1e-5,
            "yaw magnitude should decrease: step {step}: {} > {}",
            out.head.yaw_rad.abs(),
            last_yaw.abs()
        );

        last_yaw = out.head.yaw_rad;
        last_angle = angle;
    }

    // After the full decay duration has elapsed, the output should be neutral.
    let neutral = recovery
        .update(
            TrackingState::ReturningNeutral,
            Duration::from_millis(200),
            None,
            MonoTimeNs(500_000_000),
        )
        .expect("neutral frame should be emitted");
    assert_relative_eq!(neutral.head.yaw_rad, 0.0, epsilon = 1e-4);
    assert_relative_eq!(neutral.head.pitch_rad, 0.0, epsilon = 1e-4);
    assert_relative_eq!(neutral.head.roll_rad, 0.0, epsilon = 1e-4);
}

#[test]
fn loss_recovery_reacquire_limits_jump() {
    let mut recovery = LossRecovery::new(test_params()).unwrap();
    let first = frame(1, 0.0, 0.0, 0.0, 0.0);

    // Track a neutral pose.
    let _ = recovery.update(
        TrackingState::Tracking,
        Duration::from_millis(16),
        Some(first.clone()),
        MonoTimeNs(16_000_000),
    );

    // Lose the face and let it decay partway to neutral.
    let _ = recovery.update(
        TrackingState::LostHold,
        Duration::from_millis(100),
        None,
        MonoTimeNs(116_000_000),
    );
    let before_reacquire = recovery
        .update(
            TrackingState::ReturningNeutral,
            Duration::from_millis(100),
            None,
            MonoTimeNs(216_000_000),
        )
        .unwrap();

    // Reacquire with a pose that is far from the current recovered pose.
    let target = frame(2, -1.2, 0.6, -0.4, 0.9);
    let during_recovery = recovery
        .update(
            TrackingState::Tracking,
            Duration::from_millis(50),
            Some(target.clone()),
            MonoTimeNs(266_000_000),
        )
        .unwrap();

    // The recovery frame must not snap directly to the target.
    assert!(
        (during_recovery.head.yaw_rad - target.head.yaw_rad).abs() > 0.1,
        "recovery should not jump to target yaw immediately"
    );

    // The rotation should move toward the target, not away from it.
    let before_q = semantic_pose_to_quaternion(before_reacquire.head);
    let target_q = semantic_pose_to_quaternion(target.head);
    let during_q = semantic_pose_to_quaternion(during_recovery.head);

    let before_to_target = before_q.angle_to(&target_q);
    let during_to_target = during_q.angle_to(&target_q);
    assert!(
        during_to_target < before_to_target,
        "recovery should move closer to target: before_to_target={before_to_target}, during_to_target={during_to_target}"
    );

    // Finish the recovery.
    let after_recovery = recovery
        .update(
            TrackingState::Tracking,
            Duration::from_millis(100),
            Some(target.clone()),
            MonoTimeNs(366_000_000),
        )
        .unwrap();
    assert_relative_eq!(
        after_recovery.head.yaw_rad,
        target.head.yaw_rad,
        epsilon = 1e-4
    );
    assert!(!recovery.is_recovering());
}

#[test]
fn loss_recovery_drops_detailed_face_when_returning_to_neutral() {
    let mut recovery = LossRecovery::new(test_params()).unwrap();
    let tracked = detailed_frame(11, 0.8);

    let _ = recovery.update(
        TrackingState::Tracking,
        Duration::from_millis(16),
        Some(tracked),
        MonoTimeNs(16_000_000),
    );
    let held = recovery
        .update(
            TrackingState::LostHold,
            Duration::from_millis(50),
            None,
            MonoTimeNs(66_000_000),
        )
        .expect("glide should preserve the last detailed face state");
    assert!(held.detailed_face.is_some());

    let neutral = recovery
        .update(
            TrackingState::ReturningNeutral,
            Duration::from_millis(200),
            None,
            MonoTimeNs(266_000_000),
        )
        .expect("neutral transition should emit a frame");
    assert!(
        neutral.detailed_face.is_none(),
        "tracking loss must not retain stale ARKit52 coefficients"
    );
}

#[test]
fn loss_recovery_settings_enforce_fixed_ranges() {
    assert!(LossRecoveryParams::default().validate().is_ok());

    assert!(matches!(
        LossRecoveryParams {
            glide_duration: Duration::ZERO,
            ..LossRecoveryParams::default()
        }
        .validate()
        .unwrap_err(),
        LossRecoveryConfigError::ZeroDuration {
            field: "glide_duration"
        }
    ));

    assert!(
        LossRecoveryParams {
            glide_duration: MIN_GLIDE_DURATION - Duration::from_millis(1),
            ..LossRecoveryParams::default()
        }
        .validate()
        .is_err()
    );

    assert!(
        LossRecoveryParams {
            glide_duration: MAX_GLIDE_DURATION + Duration::from_millis(1),
            ..LossRecoveryParams::default()
        }
        .validate()
        .is_err()
    );

    assert!(
        LossRecoveryParams {
            decay_duration: MIN_DECAY_DURATION - Duration::from_millis(1),
            ..LossRecoveryParams::default()
        }
        .validate()
        .is_err()
    );

    assert!(
        LossRecoveryParams {
            decay_duration: MAX_DECAY_DURATION + Duration::from_millis(1),
            ..LossRecoveryParams::default()
        }
        .validate()
        .is_err()
    );

    assert!(
        LossRecoveryParams {
            recovery_duration: MIN_RECOVERY_DURATION - Duration::from_millis(1),
            ..LossRecoveryParams::default()
        }
        .validate()
        .is_err()
    );

    assert!(
        LossRecoveryParams {
            recovery_duration: MAX_RECOVERY_DURATION + Duration::from_millis(1),
            ..LossRecoveryParams::default()
        }
        .validate()
        .is_err()
    );
}

#[test]
fn loss_recovery_does_not_publish_stale_observation_as_new_frame() {
    let mut recovery = LossRecovery::new(test_params()).unwrap();
    let tracked = frame(5, 0.4, 0.0, 0.0, 0.7);

    let _ = recovery.update(
        TrackingState::Tracking,
        Duration::from_millis(16),
        Some(tracked.clone()),
        MonoTimeNs(16_000_000),
    );

    // Emit several synthetic frames while lost. Their source sequence must
    // remain the last valid sequence, not increment.
    let last_seq = tracked.source_seq;
    for step in 1..=10 {
        let state = if step <= 3 {
            TrackingState::LostHold
        } else {
            TrackingState::ReturningNeutral
        };
        let out = recovery
            .update(
                state,
                Duration::from_millis(50),
                None,
                MonoTimeNs(16_000_000 + step as u64 * 50_000_000),
            )
            .expect("synthetic frame should be emitted");
        assert_eq!(
            out.source_seq, last_seq,
            "stale observation should not be republished with a new sequence"
        );
    }

    // Reacquire with a new observation. Only after recovery completes should
    // the source sequence advance.
    let reacquired = frame(6, -0.4, 0.0, 0.0, 0.0);
    let mut seen_new_seq = false;
    for step in 1..=5 {
        let out = recovery
            .update(
                TrackingState::Tracking,
                Duration::from_millis(30),
                Some(reacquired.clone()),
                MonoTimeNs(600_000_000 + step as u64 * 30_000_000),
            )
            .expect("frame should be emitted during recovery");
        if out.source_seq == reacquired.source_seq {
            assert!(
                !recovery.is_recovering(),
                "source sequence must not advance until recovery is complete"
            );
            seen_new_seq = true;
            break;
        }
    }
    assert!(
        seen_new_seq,
        "recovery should eventually publish the new observation"
    );
}

fn with_translation(seq: u64, x: f32, y: f32, z: f32) -> AvatarControlFrame {
    let mut frame = frame(seq, 0.1, 0.05, -0.05, 0.0);
    frame.head_translation = HeadTranslationSignal::tracked(x, y, z);
    frame
}

#[test]
fn loss_recovery_hold_preserves_head_translation() {
    let mut recovery = LossRecovery::new(test_params()).unwrap();
    let tracked = with_translation(7, 0.03, -0.01, 0.08);

    let _ = recovery.update(
        TrackingState::Tracking,
        Duration::from_millis(16),
        Some(tracked.clone()),
        MonoTimeNs(16_000_000),
    );
    let held = recovery
        .update(
            TrackingState::LostHold,
            Duration::from_millis(50),
            None,
            MonoTimeNs(66_000_000),
        )
        .expect("glided frame should be emitted");

    assert_eq!(held.head_translation, tracked.head_translation);
}

#[test]
fn loss_recovery_decay_blends_translation_toward_zero_while_available() {
    let mut recovery = LossRecovery::new(test_params()).unwrap();
    let tracked = with_translation(1, 0.04, -0.02, 0.06);

    let _ = recovery.update(
        TrackingState::Tracking,
        Duration::from_millis(16),
        Some(tracked),
        MonoTimeNs(16_000_000),
    );
    // Spend exactly the glide duration in LostHold (stationary frames glide in place).
    let _ = recovery.update(
        TrackingState::LostHold,
        Duration::from_millis(100),
        None,
        MonoTimeNs(116_000_000),
    );

    let mid_decay = recovery
        .update(
            TrackingState::ReturningNeutral,
            Duration::from_millis(100),
            None,
            MonoTimeNs(216_000_000),
        )
        .expect("decay should emit a frame");
    assert!(
        mid_decay.head_translation.is_available(),
        "mid-decay translation must stay distinguishable from unavailable"
    );
    assert!(
        mid_decay.head_translation.x_meters.abs() < 0.04,
        "translation X should decay toward zero"
    );

    let neutral = recovery
        .update(
            TrackingState::ReturningNeutral,
            Duration::from_millis(200),
            None,
            MonoTimeNs(500_000_000),
        )
        .expect("neutral frame should be emitted");
    assert_relative_eq!(neutral.head_translation.x_meters, 0.0, epsilon = 1e-4);
    assert_relative_eq!(neutral.head_translation.y_meters, 0.0, epsilon = 1e-4);
    assert_relative_eq!(neutral.head_translation.z_meters, 0.0, epsilon = 1e-4);
}

#[test]
fn rotation_only_producer_falls_back_to_unavailable_translation() {
    let mut recovery = LossRecovery::new(test_params()).unwrap();
    // `frame()` builds a rotation-only frame whose translation is UNAVAILABLE.
    let tracked = frame(3, 0.4, 0.2, -0.1, 0.0);
    assert!(!tracked.head_translation.is_available());

    let _ = recovery.update(
        TrackingState::Tracking,
        Duration::from_millis(16),
        Some(tracked),
        MonoTimeNs(16_000_000),
    );
    let held = recovery
        .update(
            TrackingState::LostHold,
            Duration::from_millis(50),
            None,
            MonoTimeNs(66_000_000),
        )
        .expect("glided frame should be emitted");
    assert!(!held.head_translation.is_available());

    let decayed = recovery
        .update(
            TrackingState::ReturningNeutral,
            Duration::from_millis(100),
            None,
            MonoTimeNs(266_000_000),
        )
        .expect("decay should emit a frame");
    assert!(
        !decayed.head_translation.is_available(),
        "unavailable translation must not become a zero observation during decay"
    );

    // Recovery blending between two unavailable endpoints stays unavailable.
    let reacquired = frame(4, -0.2, 0.1, 0.0, 0.0);
    let recovering = recovery
        .update(
            TrackingState::Tracking,
            Duration::from_millis(50),
            Some(reacquired),
            MonoTimeNs(316_000_000),
        )
        .expect("recovery should emit a frame");
    assert!(!recovering.head_translation.is_available());
}

#[test]
fn loss_recovery_glide_continues_motion_with_inertia() {
    let mut recovery = LossRecovery::new(test_params()).unwrap();
    // Two tracked frames turning the head: yaw 0.2 -> 0.3 over one frame.
    let _ = recovery.update(
        TrackingState::Tracking,
        Duration::from_millis(16),
        Some(frame(1, 0.2, 0.0, 0.0, 0.0)),
        MonoTimeNs(33_333_333),
    );
    let _ = recovery.update(
        TrackingState::Tracking,
        Duration::from_millis(16),
        Some(frame(2, 0.3, 0.0, 0.0, 0.0)),
        MonoTimeNs(66_666_666),
    );

    let glided = recovery
        .update(
            TrackingState::LostHold,
            Duration::from_millis(33),
            None,
            MonoTimeNs(100_000_000),
        )
        .expect("glide should emit a frame");

    // The head keeps turning in the same direction instead of freezing.
    assert!(
        glided.head.yaw_rad > 0.3,
        "glide should continue the motion, got {}",
        glided.head.yaw_rad
    );
}

#[test]
fn loss_recovery_glide_excursion_is_bounded() {
    let mut recovery = LossRecovery::new(test_params()).unwrap();
    // A violent single-frame turn produces a large estimated velocity.
    let _ = recovery.update(
        TrackingState::Tracking,
        Duration::from_millis(16),
        Some(frame(1, 0.0, 0.0, 0.0, 0.0)),
        MonoTimeNs(33_333_333),
    );
    let origin = frame(2, 1.5, 0.0, 0.0, 0.0);
    let _ = recovery.update(
        TrackingState::Tracking,
        Duration::from_millis(16),
        Some(origin.clone()),
        MonoTimeNs(66_666_666),
    );

    let origin_q = semantic_pose_to_quaternion(origin.head);
    let bound = test_params().max_glide_excursion_rad;
    let mut max_glide_yaw = 0.0_f32;
    let mut previous_yaw: Option<f32> = None;
    for step in 1..=20 {
        let glided = recovery
            .update(
                TrackingState::LostHold,
                Duration::from_millis(33),
                None,
                MonoTimeNs(66_666_666 + step as u64 * 33_333_333),
            )
            .expect("glide should emit a frame");
        let yaw = glided.head.yaw_rad;
        if recovery.is_gliding() {
            let excursion = semantic_pose_to_quaternion(glided.head).angle_to(&origin_q);
            assert!(
                excursion <= bound + 1.0e-4,
                "glide excursion {excursion} exceeded the bound at step {step}"
            );
            max_glide_yaw = max_glide_yaw.max(yaw);
        } else {
            // Once the decay takes over, the pose must head back toward
            // neutral (passing through the origin yaw, which is expected).
            let previous = previous_yaw.unwrap_or(yaw);
            assert!(
                yaw.abs() <= previous.abs() + 1.0e-4,
                "decay should move toward neutral, got {yaw} after {previous} at step {step}"
            );
        }
        previous_yaw = Some(yaw);
    }
    assert!(
        max_glide_yaw > 1.5,
        "glide should keep turning in the tracked direction, got {max_glide_yaw}"
    );
}

#[test]
fn loss_recovery_reacquire_from_searching_does_not_snap() {
    let mut recovery = LossRecovery::new(test_params()).unwrap();
    // Track a neutral pose, lose the face until the episode completes in
    // Searching, then reacquire with the head turned.
    let _ = recovery.update(
        TrackingState::Tracking,
        Duration::from_millis(16),
        Some(frame(1, 0.0, 0.0, 0.0, 0.0)),
        MonoTimeNs(33_333_333),
    );
    let _ = recovery.update(
        TrackingState::LostHold,
        Duration::from_millis(100),
        None,
        MonoTimeNs(133_333_333),
    );
    let neutral = recovery
        .update(
            TrackingState::Searching,
            Duration::from_millis(200),
            None,
            MonoTimeNs(333_333_333),
        )
        .expect("searching should emit a frame");
    assert_relative_eq!(neutral.head.yaw_rad, 0.0, epsilon = 1e-4);

    let reacquired = frame(2, 0.8, 0.0, 0.0, 0.0);
    let reconnected = recovery
        .update(
            TrackingState::Acquiring,
            Duration::from_millis(16),
            Some(reacquired.clone()),
            MonoTimeNs(350_000_000),
        )
        .expect("reacquire should emit a frame");

    assert!(
        reconnected.head.yaw_rad < reacquired.head.yaw_rad,
        "reacquire from neutral must blend, got {}",
        reconnected.head.yaw_rad
    );
    assert!(recovery.is_recovering());
}

#[test]
fn loss_recovery_repeated_loss_does_not_oscillate_back_to_tracked_pose() {
    let mut recovery = LossRecovery::new(test_params()).unwrap();
    let tracked = frame(1, 0.5, 0.0, 0.0, 0.0);
    let _ = recovery.update(
        TrackingState::Tracking,
        Duration::from_millis(16),
        Some(tracked.clone()),
        MonoTimeNs(33_333_333),
    );
    // First LostHold update expires the glide window (100 ms) and starts
    // the decay with zero carry.
    let _ = recovery.update(
        TrackingState::LostHold,
        Duration::from_millis(100),
        None,
        MonoTimeNs(133_333_333),
    );
    let decaying = recovery
        .update(
            TrackingState::LostHold,
            Duration::from_millis(50),
            None,
            MonoTimeNs(183_333_333),
        )
        .expect("decay should continue while the machine stays in LostHold");

    // The pose must keep decaying toward neutral, not snap back to the
    // tracked pose because the state machine is still in LostHold.
    assert!(
        decaying.head.yaw_rad.abs() < tracked.head.yaw_rad.abs(),
        "decay must not oscillate back to the tracked pose, got {}",
        decaying.head.yaw_rad
    );
    assert!(recovery.is_returning());
}
