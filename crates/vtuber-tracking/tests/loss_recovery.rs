// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Integration tests for the shared loss hold, eased neutral return, and
//! reacquisition blend.
//!
//! These tests verify that `LossRecovery` behaves deterministically using
//! only caller-supplied durations and monotonic timestamps, without any
//! wall-clock dependency. The timeline is the shared `LossBlendProfile`
//! (also used by the arm channels): hold, then an eased return over the
//! return duration, then an acquire-time blend on reacquire.

use std::time::Duration;

use approx::assert_relative_eq;

use vtuber_core::types::{
    AvatarControlFrame, ExpressionCoefficients, FrameSeq, GazeSignal, GazeTrackingState, HeadPose,
    HeadTranslationSignal, MonoTimeNs, TrackingState,
};
use vtuber_core::{ARKIT52_CHANNEL_COUNT, Arkit52Coefficients, ArkitBlendshape};
use vtuber_tracking::loss_blend::{
    LossBlendProfile, MAX_ACQUIRE_DURATION, MAX_HOLD_DURATION, MAX_RETURN_DURATION,
    MIN_ACQUIRE_DURATION, MIN_HOLD_DURATION, MIN_RETURN_DURATION,
};
use vtuber_tracking::loss_recovery::LossRecovery;
use vtuber_tracking::pose::semantic_pose_to_quaternion;

fn test_params() -> LossBlendProfile {
    LossBlendProfile {
        hold: Duration::from_millis(100),
        return_duration: Duration::from_millis(200),
        acquire: Duration::from_millis(100),
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

/// The pose authority a consumer renders: the pose weighted by the frame
/// confidence, exactly like the arm channels' observed target weighted by the
/// channel authority.
fn pose_authority(frame: &AvatarControlFrame) -> f32 {
    semantic_pose_to_quaternion(frame.head).angle() * frame.confidence
}

/// A tracked frame whose `TongueOut` is zero, matching the coefficients the
/// detailed expression filter publishes.
fn detailed_frame(seq: u64, jaw_open: f32) -> AvatarControlFrame {
    let mut values = [0.0; ARKIT52_CHANNEL_COUNT];
    values[ArkitBlendshape::JawOpen.index()] = jaw_open;
    values[ArkitBlendshape::MouthSmileLeft.index()] = jaw_open * 0.5;
    let mut frame = frame(seq, 0.2, -0.1, 0.05, 0.4);
    frame.detailed_face = Some(Arkit52Coefficients::try_from_array(values).unwrap());
    frame
}

/// Absolute value of one channel, or `0.0` when the frame carries no
/// detailed coefficients.
fn detailed_value(frame: &AvatarControlFrame, channel: ArkitBlendshape) -> f32 {
    frame
        .detailed_face
        .as_ref()
        .map_or(0.0, |coefficients| coefficients.get(channel).abs())
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
        .expect("held frame should be emitted");

    assert_eq!(held.source_seq, tracked.source_seq);
    assert_eq!(held.captured_at, tracked.captured_at);
    assert_eq!(held.state, TrackingState::LostHold);
    assert!(
        semantic_pose_to_quaternion(held.head).angle() > 0.01,
        "held pose should not already be neutral"
    );
    assert_relative_eq!(held.confidence, tracked.confidence, epsilon = 1e-5);
}

#[test]
fn loss_recovery_eases_authority_monotonically_over_return_duration() {
    let mut recovery = LossRecovery::new(test_params()).unwrap();
    let tracked = frame(1, 179.0f32.to_radians(), 0.0, 0.0, 0.0);

    let _ = recovery.update(
        TrackingState::Tracking,
        Duration::from_millis(16),
        Some(tracked.clone()),
        MonoTimeNs(16_000_000),
    );
    // The hold keeps the pose and the authority unchanged.
    let _ = recovery.update(
        TrackingState::LostHold,
        Duration::from_millis(100),
        None,
        MonoTimeNs(116_000_000),
    );

    let mut previous = pose_authority(&tracked);
    for step in 1..=5 {
        let out = recovery
            .update(
                TrackingState::ReturningNeutral,
                Duration::from_millis(40),
                None,
                MonoTimeNs(116_000_000 + step * 40_000_000),
            )
            .expect("returning frame should be emitted");

        let authority = pose_authority(&out);
        assert!(
            authority <= previous + 1e-5,
            "the eased authority should not increase during return: step {step}: {authority} > {previous}"
        );
        previous = authority;
    }

    // After the full return duration has elapsed, the authority is exactly
    // zero: the consumers render the neutral pose.
    let neutral = recovery
        .update(
            TrackingState::ReturningNeutral,
            Duration::from_millis(200),
            None,
            MonoTimeNs(500_000_000),
        )
        .expect("neutral frame should be emitted");
    assert_relative_eq!(neutral.confidence, 0.0, epsilon = 1e-6);
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

    // Lose the face and let the return ease partway down.
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
            Duration::from_millis(200),
            Some(target.clone()),
            MonoTimeNs(466_000_000),
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
fn loss_recovery_holds_detailed_face_during_loss_hold() {
    let mut recovery = LossRecovery::new(test_params()).unwrap();
    let tracked = detailed_frame(11, 0.8);

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
        .expect("the hold should preserve the last detailed face state");

    let held = held.detailed_face.expect("detailed face is held");
    assert!((held.get(ArkitBlendshape::JawOpen) - 0.8).abs() < 1.0e-6);
    assert_eq!(held.get(ArkitBlendshape::TongueOut), 0.0);
}

#[test]
fn loss_recovery_eases_detailed_face_monotonically_toward_zero() {
    let mut recovery = LossRecovery::new(test_params()).unwrap();
    let tracked = detailed_frame(11, 0.8);

    let _ = recovery.update(
        TrackingState::Tracking,
        Duration::from_millis(16),
        Some(tracked),
        MonoTimeNs(16_000_000),
    );
    // Spend the hold so the next updates ease the coefficients.
    let _ = recovery.update(
        TrackingState::LostHold,
        Duration::from_millis(100),
        None,
        MonoTimeNs(116_000_000),
    );

    let mut previous = f32::INFINITY;
    let mut easing = 0usize;
    for step in 1..=4u64 {
        let frame = recovery
            .update(
                TrackingState::ReturningNeutral,
                Duration::from_millis(50),
                None,
                MonoTimeNs(116_000_000 + step * 50_000_000),
            )
            .expect("the eased return should emit a frame");

        // The ease must never snap `Some` straight to `None`.
        assert!(
            frame.detailed_face.is_some(),
            "step {step} dropped the detailed face mid-ease"
        );

        let jaw = detailed_value(&frame, ArkitBlendshape::JawOpen);
        let smile = detailed_value(&frame, ArkitBlendshape::MouthSmileLeft);
        assert!(jaw <= previous + 1.0e-6, "step {step} grew: {jaw}");
        assert!(smile <= previous + 1.0e-6, "step {step} grew: {smile}");
        assert_eq!(
            detailed_value(&frame, ArkitBlendshape::TongueOut),
            0.0,
            "tongue stays zero"
        );
        if recovery.is_returning() {
            easing += 1;
        }
        previous = jaw;
    }
    assert!(easing > 0, "the ease should still be in progress");
}

#[test]
fn loss_recovery_publishes_zero_detailed_face_before_dropping_it() {
    let mut recovery = LossRecovery::new(test_params()).unwrap();
    let tracked = detailed_frame(11, 0.8);

    let _ = recovery.update(
        TrackingState::Tracking,
        Duration::from_millis(16),
        Some(tracked),
        MonoTimeNs(16_000_000),
    );
    let _ = recovery.update(
        TrackingState::LostHold,
        Duration::from_millis(100),
        None,
        MonoTimeNs(116_000_000),
    );
    // One update past hold + return duration publishes the released frame
    // with exact zeros, then hands over to the neutral phase.
    let released = recovery
        .update(
            TrackingState::ReturningNeutral,
            Duration::from_millis(300),
            None,
            MonoTimeNs(416_000_000),
        )
        .expect("the final return frame should be emitted");
    let released = released
        .detailed_face
        .expect("the final return frame still carries coefficients");
    assert_eq!(released, Arkit52Coefficients::default());
    assert_relative_eq!(released.get(ArkitBlendshape::JawOpen), 0.0, epsilon = 1e-6);

    // Once neutral is reached the coefficients are dropped, and the tracker
    // in the avatar adapter has already seen the zeros.
    let after = recovery
        .update(
            TrackingState::Searching,
            Duration::from_millis(16),
            None,
            MonoTimeNs(432_000_000),
        )
        .expect("searching should keep emitting neutral frames");
    assert!(
        after.detailed_face.is_none(),
        "searching frames must not retain ARKit52 coefficients"
    );
}

#[test]
fn loss_recovery_reacquire_blends_detailed_face_continuously() {
    let mut recovery = LossRecovery::new(test_params()).unwrap();
    let tracked = detailed_frame(11, 0.8);

    let _ = recovery.update(
        TrackingState::Tracking,
        Duration::from_millis(16),
        Some(tracked.clone()),
        MonoTimeNs(16_000_000),
    );
    // Fully ease to neutral: the coefficients are dropped.
    let _ = recovery.update(
        TrackingState::LostHold,
        Duration::from_millis(100),
        None,
        MonoTimeNs(116_000_000),
    );
    let _ = recovery.update(
        TrackingState::ReturningNeutral,
        Duration::from_millis(300),
        None,
        MonoTimeNs(416_000_000),
    );
    let _ = recovery.update(
        TrackingState::Searching,
        Duration::from_millis(16),
        None,
        MonoTimeNs(432_000_000),
    );

    // Reacquire a face whose jaw is open again.
    let reacquired = detailed_frame(12, 0.6);
    let first = recovery
        .update(
            TrackingState::Tracking,
            Duration::from_millis(16),
            Some(reacquired.clone()),
            MonoTimeNs(448_000_000),
        )
        .expect("reacquire should emit a frame");

    let first_jaw = detailed_value(&first, ArkitBlendshape::JawOpen);
    assert!(
        first_jaw < 0.6,
        "reacquisition must ramp in, got {first_jaw}"
    );
    let second = recovery
        .update(
            TrackingState::Tracking,
            Duration::from_millis(16),
            Some(reacquired.clone()),
            MonoTimeNs(464_000_000),
        )
        .expect("reacquire should emit a frame");
    let second_jaw = detailed_value(&second, ArkitBlendshape::JawOpen);
    assert!(
        second_jaw > first_jaw,
        "blend must keep moving toward the target: {first_jaw} then {second_jaw}"
    );
    assert_eq!(
        detailed_value(&second, ArkitBlendshape::TongueOut),
        0.0,
        "tongue stays zero across reacquisition"
    );

    // The blend finishes within the configured acquire duration.
    let mut last = second_jaw;
    for step in 1..=6u64 {
        let frame = recovery
            .update(
                TrackingState::Tracking,
                Duration::from_millis(20),
                Some(reacquired.clone()),
                MonoTimeNs(464_000_000 + step * 20_000_000),
            )
            .expect("recovery frame");
        last = detailed_value(&frame, ArkitBlendshape::JawOpen);
    }
    assert!((last - 0.6).abs() < 1.0e-3, "recovery should reach {last}");
    assert!(!recovery.is_recovering());
}

#[test]
fn loss_recovery_settings_enforce_fixed_ranges() {
    assert!(LossBlendProfile::default().validate().is_ok());

    assert!(
        LossBlendProfile {
            hold: MIN_HOLD_DURATION - Duration::from_millis(1),
            ..test_params()
        }
        .validate()
        .is_err()
    );
    assert!(
        LossBlendProfile {
            hold: MAX_HOLD_DURATION + Duration::from_millis(1),
            ..test_params()
        }
        .validate()
        .is_err()
    );
    assert!(
        LossBlendProfile {
            return_duration: MIN_RETURN_DURATION - Duration::from_millis(1),
            ..test_params()
        }
        .validate()
        .is_err()
    );
    assert!(
        LossBlendProfile {
            return_duration: MAX_RETURN_DURATION + Duration::from_millis(1),
            ..test_params()
        }
        .validate()
        .is_err()
    );
    assert!(
        LossBlendProfile {
            acquire: MIN_ACQUIRE_DURATION - Duration::from_millis(1),
            ..test_params()
        }
        .validate()
        .is_err()
    );
    assert!(
        LossBlendProfile {
            acquire: MAX_ACQUIRE_DURATION + Duration::from_millis(1),
            ..test_params()
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
        .expect("held frame should be emitted");

    assert_eq!(held.head_translation, tracked.head_translation);
}

#[test]
fn loss_recovery_eases_translation_authority_to_zero_while_available() {
    let mut recovery = LossRecovery::new(test_params()).unwrap();
    let tracked = with_translation(1, 0.04, -0.02, 0.06);

    let _ = recovery.update(
        TrackingState::Tracking,
        Duration::from_millis(16),
        Some(tracked.clone()),
        MonoTimeNs(16_000_000),
    );
    let _ = recovery.update(
        TrackingState::LostHold,
        Duration::from_millis(100),
        None,
        MonoTimeNs(116_000_000),
    );

    // The translation observation is held; the consumers weight it by the
    // frame confidence, which eases to zero over the return duration.
    let mid = recovery
        .update(
            TrackingState::ReturningNeutral,
            Duration::from_millis(150),
            None,
            MonoTimeNs(266_000_000),
        )
        .expect("the eased return should emit a frame");
    assert!(
        mid.head_translation.is_available(),
        "mid-return translation must stay distinguishable from unavailable"
    );
    assert!(
        mid.confidence < tracked.confidence,
        "the translation authority must be easing: {}",
        mid.confidence
    );

    let neutral = recovery
        .update(
            TrackingState::ReturningNeutral,
            Duration::from_millis(500),
            None,
            MonoTimeNs(766_000_000),
        )
        .expect("fully eased frame should be emitted");
    assert_relative_eq!(neutral.confidence, 0.0, epsilon = 1e-6);
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
        .expect("held frame should be emitted");
    assert!(!held.head_translation.is_available());

    let decayed = recovery
        .update(
            TrackingState::ReturningNeutral,
            Duration::from_millis(100),
            None,
            MonoTimeNs(266_000_000),
        )
        .expect("the return should emit a frame");
    assert!(
        !decayed.head_translation.is_available(),
        "unavailable translation must not become a zero observation during return"
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
            Duration::from_millis(400),
            None,
            MonoTimeNs(533_333_333),
        )
        .expect("searching should emit a frame");
    assert_relative_eq!(neutral.confidence, 0.0, epsilon = 1e-6);

    let reacquired = frame(2, 0.8, 0.0, 0.0, 0.0);
    let reconnected = recovery
        .update(
            TrackingState::Acquiring,
            Duration::from_millis(16),
            Some(reacquired.clone()),
            MonoTimeNs(550_000_000),
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
    // The first lost update is the hold; the authority stays unchanged.
    let _ = recovery.update(
        TrackingState::LostHold,
        Duration::from_millis(100),
        None,
        MonoTimeNs(133_333_333),
    );
    let easing = recovery
        .update(
            TrackingState::LostHold,
            Duration::from_millis(150),
            None,
            MonoTimeNs(283_333_333),
        )
        .expect("the return should continue while the machine stays in LostHold");

    // The authority must keep easing down, not snap back to the tracked
    // pose because the state machine is still in LostHold.
    assert!(
        pose_authority(&easing) < pose_authority(&tracked),
        "the return must not oscillate back to the tracked pose, got {}",
        easing.head.yaw_rad
    );
    assert!(recovery.is_returning());
}

#[test]
fn loss_recovery_reacquires_with_the_same_blend_after_a_second_loss() {
    let mut recovery = LossRecovery::new(test_params()).unwrap();
    let _ = recovery.update(
        TrackingState::Tracking,
        Duration::from_millis(16),
        Some(frame(1, 0.4, 0.0, 0.0, 0.0)),
        MonoTimeNs(33_333_333),
    );
    let _ = recovery.update(
        TrackingState::LostHold,
        Duration::from_millis(100),
        None,
        MonoTimeNs(133_333_333),
    );

    // Reacquire briefly, then lose again: the return must resume from
    // wherever the recovery had reached instead of restarting.
    let target = frame(2, 0.9, 0.0, 0.0, 0.0);
    let _ = recovery.update(
        TrackingState::Tracking,
        Duration::from_millis(50),
        Some(target.clone()),
        MonoTimeNs(183_333_333),
    );
    let resumed = recovery
        .update(
            TrackingState::LostHold,
            Duration::from_millis(100),
            None,
            MonoTimeNs(283_333_333),
        )
        .expect("a second loss should resume the return");
    let authority = pose_authority(&resumed);
    assert!(
        authority <= pose_authority(&target) + 1.0e-5,
        "the resumed return must not exceed the loss authority: {authority}"
    );
    assert!(recovery.is_returning());
}

#[test]
fn loss_recovery_gaze_holds_returns_and_reacquires_without_snap() {
    let mut recovery = LossRecovery::new(test_params()).unwrap();
    let mut tracked = frame(1, 0.2, 0.0, 0.0, 0.0);
    tracked.gaze = GazeSignal::tracked(0.8, -0.4, 0.9);
    let first = recovery
        .update(
            TrackingState::Tracking,
            Duration::from_millis(16),
            Some(tracked.clone()),
            MonoTimeNs(16_000_000),
        )
        .unwrap();
    let held = recovery
        .update(
            TrackingState::LostHold,
            Duration::from_millis(50),
            None,
            MonoTimeNs(66_000_000),
        )
        .unwrap();
    assert_eq!(held.gaze.horizontal(), first.gaze.horizontal());
    assert_relative_eq!(
        held.gaze.confidence(),
        first.gaze.confidence(),
        epsilon = 1e-6
    );

    // Past the hold the gaze authority eases with the pose authority.
    let easing = recovery
        .update(
            TrackingState::LostHold,
            Duration::from_millis(150),
            None,
            MonoTimeNs(216_000_000),
        )
        .unwrap();
    assert!(
        easing.gaze.confidence() < held.gaze.confidence(),
        "the gaze authority must ease during the return"
    );

    let mut target = frame(2, -0.2, 0.0, 0.0, 0.0);
    target.gaze = GazeSignal::tracked(-0.8, 0.2, 0.9);
    let recovering = recovery
        .update(
            TrackingState::Tracking,
            Duration::from_millis(50),
            Some(target),
            MonoTimeNs(266_000_000),
        )
        .unwrap();
    assert!(recovering.gaze.horizontal() > -0.8);
    assert_eq!(recovering.gaze.state(), GazeTrackingState::Tracked);
}
