//! Dev-only synthetic tracking source.
//!
//! Generates deterministic [`AvatarControlFrame`] values from sine waves so
//! that the avatar apply path can be verified without a camera or inference
//! runtime. This module is only compiled when the `dev-synthetic-input`
//! feature is enabled and must never be active in production builds.

use bevy::prelude::*;
use vtuber_avatar::lifecycle::{AvatarLifecycle, AvatarLifecycleState};
use vtuber_avatar::unload::{ActiveControlFrame, set_active_control_frame, tag_control_frame};
use vtuber_core::types::{
    AvatarControlFrame, ExpressionCoefficients, FrameSeq, GazeSignal, HeadPose,
    HeadTranslationSignal, MonoTimeNs, TrackingState,
};

/// Configuration for the synthetic tracking source.
///
/// All fields are public for test injection. In production this resource is
/// never constructed.
#[derive(Resource, Clone, Debug)]
pub struct SyntheticTrackingSource {
    /// Monotonic sequence counter.
    seq: u64,
    /// Accumulated virtual time in seconds.
    time: f32,
    /// Seconds between generated frames (≈ 60 Hz).
    pub dt: f32,
    /// Head yaw amplitude in radians.
    pub yaw_amp: f32,
    /// Head pitch amplitude in radians.
    pub pitch_amp: f32,
    /// Head roll amplitude in radians.
    pub roll_amp: f32,
    /// Yaw oscillation period in seconds.
    pub yaw_period: f32,
    /// Pitch oscillation period in seconds.
    pub pitch_period: f32,
    /// Roll oscillation period in seconds.
    pub roll_period: f32,
    /// Blink cycle period in seconds.
    pub blink_period: f32,
    /// Mouth openness cycle period in seconds.
    pub mouth_period: f32,
    /// Normalized horizontal gaze amplitude.
    pub gaze_horizontal_amp: f32,
    /// Normalized vertical gaze amplitude.
    pub gaze_vertical_amp: f32,
}

impl Default for SyntheticTrackingSource {
    fn default() -> Self {
        Self {
            seq: 0,
            time: 0.0,
            dt: 1.0 / 60.0,
            yaw_amp: 0.35,
            pitch_amp: 0.17,
            roll_amp: 0.12,
            yaw_period: 4.0,
            pitch_period: 3.0,
            roll_period: 5.0,
            blink_period: 3.5,
            mouth_period: 2.0,
            gaze_horizontal_amp: 0.5,
            gaze_vertical_amp: 0.4,
        }
    }
}

impl SyntheticTrackingSource {
    /// Generate the next synthetic control frame.
    #[must_use]
    pub fn next_frame(&mut self) -> AvatarControlFrame {
        let t = self.time;
        self.time += self.dt;
        self.seq += 1;

        let two_pi = std::f32::consts::TAU;

        let head = HeadPose {
            yaw_rad: self.yaw_amp * (two_pi * t / self.yaw_period).sin(),
            pitch_rad: self.pitch_amp * (two_pi * t / self.pitch_period).sin(),
            roll_rad: self.roll_amp * (two_pi * t / self.roll_period).sin(),
        };

        let blink_phase = (two_pi * t / self.blink_period).sin();
        let blink = if blink_phase > 0.92 { 1.0 } else { 0.0 };

        let mouth_phase = (two_pi * t / self.mouth_period).sin();
        let mouth = (mouth_phase * 0.5 + 0.5).clamp(0.0, 1.0) * 0.6;

        let gaze = GazeSignal::tracked(
            self.gaze_horizontal_amp * (two_pi * t / (self.yaw_period * 0.7)).sin(),
            self.gaze_vertical_amp * (two_pi * t / (self.pitch_period * 1.3)).sin(),
            1.0,
        );

        let expressions = ExpressionCoefficients {
            blink_left: blink,
            blink_right: blink,
            aa: mouth,
            ..Default::default()
        };

        AvatarControlFrame {
            source_seq: FrameSeq(self.seq),
            captured_at: MonoTimeNs((t * 1e9) as u64),
            produced_at: MonoTimeNs((t * 1e9) as u64),
            confidence: 1.0,
            state: TrackingState::Tracking,
            head,
            head_translation: HeadTranslationSignal::UNAVAILABLE,
            gaze,
            expressions,
            detailed_face: None,
        }
    }
}

/// System that generates a synthetic control frame each tick and publishes it
/// to [`ActiveControlFrame`].
///
/// Drops the frame silently when no active avatar exists or the lifecycle is
/// not in a state that accepts frames.
pub fn synthetic_tracking_system(
    mut source: ResMut<SyntheticTrackingSource>,
    lifecycle: Res<AvatarLifecycle>,
    mut active: ResMut<ActiveControlFrame>,
) {
    if lifecycle.state() != AvatarLifecycleState::Ready {
        return;
    }

    let frame = source.next_frame();

    let Some((generation, frame)) = tag_control_frame(frame, &lifecycle) else {
        return;
    };

    let _ = set_active_control_frame(&lifecycle, generation, frame, &mut active);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_source_produces_finite_values() {
        let mut src = SyntheticTrackingSource::default();
        for _ in 0..300 {
            let frame = src.next_frame();
            assert!(frame.head.yaw_rad.is_finite());
            assert!(frame.head.pitch_rad.is_finite());
            assert!(frame.head.roll_rad.is_finite());
            assert!(frame.confidence.is_finite());
            assert!(frame.expressions.blink_left >= 0.0);
            assert!(frame.expressions.blink_left <= 1.0);
            assert!(frame.expressions.aa >= 0.0);
            assert!(frame.expressions.aa <= 1.0);
        }
    }

    #[test]
    fn synthetic_source_sequence_increments() {
        let mut src = SyntheticTrackingSource::default();
        let f1 = src.next_frame();
        let f2 = src.next_frame();
        assert_eq!(f2.source_seq.0, f1.source_seq.0 + 1);
    }

    #[test]
    fn synthetic_source_deterministic() {
        let mut a = SyntheticTrackingSource::default();
        let mut b = SyntheticTrackingSource::default();
        for _ in 0..100 {
            let fa = a.next_frame();
            let fb = b.next_frame();
            assert_eq!(fa, fb);
        }
    }

    #[test]
    fn synthetic_source_head_oscillates() {
        let mut src = SyntheticTrackingSource::default();
        let mut max_yaw = 0.0f32;
        let mut min_yaw = f32::MAX;
        for _ in 0..600 {
            let f = src.next_frame();
            max_yaw = max_yaw.max(f.head.yaw_rad.abs());
            min_yaw = min_yaw.min(f.head.yaw_rad.abs());
        }
        assert!(
            max_yaw > 0.1,
            "yaw should oscillate with meaningful amplitude"
        );
        assert!(min_yaw < 0.05, "yaw should pass near zero");
    }

    #[test]
    fn synthetic_source_blink_cycles() {
        let mut src = SyntheticTrackingSource::default();
        let mut saw_open = false;
        let mut saw_closed = false;
        for _ in 0..600 {
            let f = src.next_frame();
            if f.expressions.blink_left == 0.0 {
                saw_open = true;
            }
            if f.expressions.blink_left == 1.0 {
                saw_closed = true;
            }
        }
        assert!(saw_open, "should have open-eye frames");
        assert!(saw_closed, "should have closed-eye frames");
    }

    #[test]
    fn synthetic_source_gaze_present() {
        let mut src = SyntheticTrackingSource::default();
        let f = src.next_frame();
        assert!(f.gaze.is_available());
        assert!(f.gaze.horizontal.is_finite());
        assert!(f.gaze.vertical.is_finite());
    }
}
