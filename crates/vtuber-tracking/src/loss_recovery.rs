//! Loss hold, eased neutral return, and reacquisition blend for face
//! tracking.
//!
//! When a tracked face disappears, the avatar should not snap instantly to
//! neutral nor freeze in place. [`LossRecovery`] publishes the last valid
//! frame with an authority that follows the shared loss-blend timeline
//! ([`LossBlendProfile`], also used by the arm channels): the pose is held
//! for the hold time, then the confidence that every consumer weights the
//! pose by eases to zero over the return duration, so the head sinks back to
//! neutral like the arms sink back to their virtual anchor. Expressions, the
//! detailed Perfect Sync coefficients, and the gaze confidence fade along
//! with it so a briefly lost face does not collapse mid-speech but releases
//! cleanly.
//!
//! When the face reappears, a timed smoothstep blend whose duration is the
//! profile's acquire time reconnects the tracked frames from wherever the
//! return had reached instead of snapping.
//!
//! All timing uses the caller-supplied [`Duration`] delta and monotonic
//! timestamps, so behaviour is deterministic and testable without a wall
//! clock. The per-frame cost is a handful of coefficient operations with no
//! allocation, so loss handling stays negligible even when it runs on every
//! frame.

use std::time::Duration;

use nalgebra::{Quaternion, UnitQuaternion};
use thiserror::Error;

use vtuber_core::types::{
    AvatarControlFrame, ExpressionCoefficients, GazeSignal, GazeTrackingState, HeadPose,
    HeadTranslationSignal, MonoTimeNs, TrackingState,
};
use vtuber_core::{ARKIT52_CHANNEL_COUNT, Arkit52Coefficients, ArkitBlendshape};

use crate::loss_blend::{
    LossBlendConfigError, LossBlendProfile, acquire_factor, loss_return_factor,
};
use crate::pose::{quaternion_to_semantic_pose, semantic_pose_to_quaternion};

/// Minimum head-rotation gap that triggers a reacquire blend from a
/// loss-related pose. Smaller gaps are reconnected by passing the tracked
/// frame through directly.
const REACQUIRE_MIN_HEAD_GAP_RAD: f32 = 0.02;
/// Minimum expression-coefficient gap that triggers a reacquire blend.
const REACQUIRE_MIN_EXPRESSION_GAP: f32 = 0.15;
/// Minimum gaze gap that triggers a reacquire blend.
const REACQUIRE_MIN_GAZE_GAP: f32 = 0.20;

/// Errors that can occur while constructing a [`LossRecovery`] instance.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
#[error(transparent)]
pub struct LossRecoveryConfigError(#[from] LossBlendConfigError);

/// Current phase of the loss-recovery state machine.
///
/// The by-value control frame intentionally includes the fixed-size validated
/// ARKit52 payload; keeping it inline avoids a per-frame heap allocation.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
enum RecoveryState {
    /// No synthetic motion is in progress; pass tracked frames through.
    Idle,
    /// Returning from the loss pose to neutral along the shared loss
    /// timeline.
    Returning {
        /// Frame at the start of the loss episode.
        from: AvatarControlFrame,
        /// Time spent in the return phase.
        elapsed: Duration,
    },
    /// Blending from a loss-related pose to a newly tracked frame over the
    /// acquire time.
    Recovering {
        /// Pose at the start of the recovery blend.
        from: AvatarControlFrame,
        /// Time spent in the recovery phase.
        elapsed: Duration,
    },
}

/// Holds the last observed pose when a face is lost, eases the pose's
/// authority back to neutral along the shared loss timeline, and blends back
/// to tracked frames on reacquire.
#[derive(Clone, Debug, PartialEq)]
pub struct LossRecovery {
    params: LossBlendProfile,
    state: RecoveryState,
    last_valid: Option<AvatarControlFrame>,
    last_output: Option<AvatarControlFrame>,
}

impl LossRecovery {
    /// Creates a new [`LossRecovery`] with the given parameters.
    ///
    /// # Errors
    ///
    /// Returns [`LossRecoveryConfigError`] if the parameters are invalid.
    pub fn new(params: LossBlendProfile) -> Result<Self, LossRecoveryConfigError> {
        params.validate()?;
        Ok(Self {
            params,
            state: RecoveryState::Idle,
            last_valid: None,
            last_output: None,
        })
    }

    /// Returns the configured parameters.
    #[must_use]
    pub fn params(&self) -> &LossBlendProfile {
        &self.params
    }

    /// Returns `true` while returning to neutral.
    #[must_use]
    pub fn is_returning(&self) -> bool {
        matches!(self.state, RecoveryState::Returning { .. })
    }

    /// Returns `true` while blending back to a tracked frame.
    #[must_use]
    pub fn is_recovering(&self) -> bool {
        matches!(self.state, RecoveryState::Recovering { .. })
    }

    /// Updates the recovery logic and returns the synthetic or tracked frame
    /// to publish.
    ///
    /// `state` is the external tracking state (for example from
    /// [`TrackingStateMachine`](crate::TrackingStateMachine)). `dt` is the
    /// elapsed time since the last call. `tracked` is the latest valid
    /// tracked frame, if any. `produced_at` is the monotonic timestamp to
    /// stamp on any produced frame.
    ///
    /// The returned frame reuses the source sequence and capture timestamp of
    /// the last valid frame during the return and the recovery blend so that
    /// a stale observation is not published as a new frame.
    #[must_use]
    pub fn update(
        &mut self,
        state: TrackingState,
        dt: Duration,
        tracked: Option<AvatarControlFrame>,
        produced_at: MonoTimeNs,
    ) -> Option<AvatarControlFrame> {
        // Keep track of the most recent valid frame for future return phases.
        if let Some(ref t) = tracked {
            self.last_valid = Some(t.clone());
        }

        let old = std::mem::replace(&mut self.state, RecoveryState::Idle);

        let (next, output) = match (state, old, tracked) {
            // A tracked frame is available while we are actively tracking.
            // Pass it through, or start/continue a recovery blend if we were
            // previously returning or mid-recovery.
            (
                TrackingState::Tracking | TrackingState::Acquiring,
                RecoveryState::Idle,
                Some(target),
            ) => {
                if self.should_blend_on_reacquire(&target) {
                    self.start_recovery(target, dt, state, produced_at)
                } else {
                    (RecoveryState::Idle, Some(target))
                }
            }
            (
                TrackingState::Tracking | TrackingState::Acquiring,
                RecoveryState::Returning { .. },
                Some(target),
            ) => self.start_recovery(target, dt, state, produced_at),
            (
                TrackingState::Tracking | TrackingState::Acquiring,
                RecoveryState::Recovering { from, elapsed },
                Some(target),
            ) => advance_recovery(from, target, elapsed, dt, state, produced_at, &self.params),

            // Lost: a running return keeps advancing so repeated state flips
            // cannot restart the hold or oscillate the pose back and forth.
            (
                TrackingState::LostHold
                | TrackingState::ReturningNeutral
                | TrackingState::Searching,
                RecoveryState::Returning { from, elapsed },
                None,
            ) => advance_return(from, elapsed, dt, state, produced_at, &self.params),
            (
                TrackingState::LostHold
                | TrackingState::ReturningNeutral
                | TrackingState::Searching,
                _,
                None,
            ) => self.start_return(state, produced_at),

            // Any other combination: preserve the previous state and output.
            (_, old, _) => (old, self.last_output.clone()),
        };

        self.state = next;
        if let Some(ref out) = output {
            self.last_output = Some(out.clone());
        }
        output
    }

    /// Decides whether the first tracked frame after a loss episode should
    /// start a recovery blend instead of passing through directly.
    fn should_blend_on_reacquire(&self, target: &AvatarControlFrame) -> bool {
        let Some(from) = self.last_output.as_ref() else {
            return false;
        };
        if !matches!(
            from.state,
            TrackingState::LostHold | TrackingState::ReturningNeutral | TrackingState::Searching
        ) {
            return false;
        }
        let head_gap = semantic_pose_to_quaternion(from.head)
            .angle_to(&semantic_pose_to_quaternion(target.head));
        if head_gap > REACQUIRE_MIN_HEAD_GAP_RAD {
            return true;
        }
        if expression_distance(&from.expressions, &target.expressions)
            > REACQUIRE_MIN_EXPRESSION_GAP
        {
            return true;
        }
        gaze_distance(from.gaze, target.gaze) > REACQUIRE_MIN_GAZE_GAP
    }

    /// Starts a timed blend from the last output pose toward `target`.
    ///
    /// The blend runs over the profile's acquire time with the smoothstep
    /// shape the arm channels use, so a reacquired face continues from
    /// wherever the return had reached.
    fn start_recovery(
        &mut self,
        target: AvatarControlFrame,
        dt: Duration,
        state: TrackingState,
        produced_at: MonoTimeNs,
    ) -> (RecoveryState, Option<AvatarControlFrame>) {
        let from = self.last_output.clone().unwrap_or_else(|| target.clone());
        advance_recovery(
            from,
            target,
            Duration::ZERO,
            dt,
            state,
            produced_at,
            &self.params,
        )
    }

    /// Starts the return-to-neutral phase from the most recent output.
    ///
    /// The hold publishes the loss frame unchanged. When the previous episode
    /// already decayed fully (the machine can stay in a lost state past the
    /// return), static neutral frames keep being emitted so a finished
    /// episode never restarts its timeline.
    fn start_return(
        &mut self,
        state: TrackingState,
        produced_at: MonoTimeNs,
    ) -> (RecoveryState, Option<AvatarControlFrame>) {
        let Some(origin) = self.last_output.clone().or_else(|| self.last_valid.clone()) else {
            return (RecoveryState::Idle, None);
        };
        if is_fully_neutral(&origin) {
            let neutral = neutral_frame(&origin, produced_at, state);
            return (RecoveryState::Idle, Some(neutral));
        }
        let mut frame = origin;
        frame.state = state;
        frame.produced_at = produced_at;
        (
            RecoveryState::Returning {
                from: frame.clone(),
                elapsed: Duration::ZERO,
            },
            Some(frame),
        )
    }
}

/// Advances the reacquisition blend toward the latest tracked target.
///
/// The blend runs over the profile's acquire time and is reused for every
/// continuation so the progress stays monotonic. The factor is
/// smoothstep-shaped with zero slope at both ends.
fn advance_recovery(
    from: AvatarControlFrame,
    to: AvatarControlFrame,
    elapsed: Duration,
    dt: Duration,
    state: TrackingState,
    produced_at: MonoTimeNs,
    params: &LossBlendProfile,
) -> (RecoveryState, Option<AvatarControlFrame>) {
    let elapsed = elapsed.saturating_add(dt);
    let t = acquire_factor(elapsed, params);
    if t >= 1.0 {
        (RecoveryState::Idle, Some(to))
    } else {
        let blended = blend_frames(&from, &to, t, state, produced_at);
        (RecoveryState::Recovering { from, elapsed }, Some(blended))
    }
}

/// Advances the return-to-neutral one frame.
///
/// The authority of the held loss frame decays along the shared loss
/// timeline: full during the hold, then smoothstep-eased to zero. At zero the
/// last frame publishes exact zeros so the avatar releases every Perfect Sync
/// morph before the coefficients are dropped on the following frames.
fn advance_return(
    from: AvatarControlFrame,
    elapsed: Duration,
    dt: Duration,
    state: TrackingState,
    produced_at: MonoTimeNs,
    params: &LossBlendProfile,
) -> (RecoveryState, Option<AvatarControlFrame>) {
    let elapsed = elapsed.saturating_add(dt);
    let factor = loss_return_factor(elapsed, params);
    if factor <= 0.0 {
        (
            RecoveryState::Idle,
            Some(held_frame(&from, 0.0, state, produced_at)),
        )
    } else {
        let blended = held_frame(&from, factor, state, produced_at);
        (RecoveryState::Returning { from, elapsed }, Some(blended))
    }
}

/// Builds the synthetic frame for the current return state.
///
/// The head pose, translation, and gaze direction are held; the confidence
/// the consumers weight them by decays with `factor`, exactly like the arm
/// channels' weight ramps toward the virtual anchor. Expressions, the
/// detailed coefficients, and the gaze confidence fade along with it.
fn held_frame(
    origin: &AvatarControlFrame,
    factor: f32,
    state: TrackingState,
    produced_at: MonoTimeNs,
) -> AvatarControlFrame {
    let factor = factor.clamp(0.0, 1.0);
    AvatarControlFrame {
        source_seq: origin.source_seq,
        captured_at: origin.captured_at,
        produced_at,
        confidence: origin.confidence * factor,
        state,
        head: origin.head,
        head_translation: origin.head_translation,
        gaze: scale_gaze(origin.gaze, factor),
        expressions: blend_expressions(
            &origin.expressions,
            &ExpressionCoefficients::default(),
            factor,
        ),
        detailed_face: origin
            .detailed_face
            .map(|coefficients| scale_detailed_face(coefficients, factor)),
    }
}

/// Scales a gaze signal's confidence, keeping direction and availability.
fn scale_gaze(gaze: GazeSignal, factor: f32) -> GazeSignal {
    let confidence = gaze.confidence * factor;
    match gaze.state {
        GazeTrackingState::Tracked => {
            GazeSignal::tracked(gaze.horizontal, gaze.vertical, confidence)
        }
        GazeTrackingState::Degraded => {
            GazeSignal::degraded(gaze.horizontal, gaze.vertical, confidence)
        }
        GazeTrackingState::Unavailable => GazeSignal::UNAVAILABLE,
    }
}

/// Returns `true` when the frame carries no more authority.
///
/// Synthetic frames end with exactly-zero confidence, so this check
/// recognises both a completed return-to-neutral and the static neutral
/// frames emitted while searching.
fn is_fully_neutral(frame: &AvatarControlFrame) -> bool {
    frame.confidence <= 0.0
}

/// Maximum absolute coefficient difference between two expression sets.
fn expression_distance(a: &ExpressionCoefficients, b: &ExpressionCoefficients) -> f32 {
    let left = [
        a.blink_left,
        a.blink_right,
        a.aa,
        a.ih,
        a.ou,
        a.ee,
        a.oh,
        a.look_left,
        a.look_right,
        a.look_up,
        a.look_down,
        a.happy,
        a.angry,
        a.sad,
        a.relaxed,
        a.surprised,
    ];
    let right = [
        b.blink_left,
        b.blink_right,
        b.aa,
        b.ih,
        b.ou,
        b.ee,
        b.oh,
        b.look_left,
        b.look_right,
        b.look_up,
        b.look_down,
        b.happy,
        b.angry,
        b.sad,
        b.relaxed,
        b.surprised,
    ];
    left.iter()
        .zip(right.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Gaze direction distance used for the reacquire blend trigger.
///
/// Signals without an observation on either side do not contribute.
fn gaze_distance(from: GazeSignal, to: GazeSignal) -> f32 {
    if !from.is_available() || !to.is_available() {
        return 0.0;
    }
    (from.horizontal - to.horizontal)
        .abs()
        .max((from.vertical - to.vertical).abs())
}

/// Linearly interpolates two scalar values.
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// Returns `target` or `-target`, whichever is closer to `current`.
fn choose_shortest_arc(
    current: UnitQuaternion<f32>,
    target: UnitQuaternion<f32>,
) -> UnitQuaternion<f32> {
    let c = current.quaternion();
    let t = target.quaternion();
    let dot = c.w * t.w + c.i * t.i + c.j * t.j + c.k * t.k;
    if dot < 0.0 { negate(target) } else { target }
}

/// Explicitly negates a unit quaternion, preserving unit norm.
fn negate(q: UnitQuaternion<f32>) -> UnitQuaternion<f32> {
    let inner = q.quaternion();
    UnitQuaternion::from_quaternion(Quaternion::new(-inner.w, -inner.i, -inner.j, -inner.k))
}

/// Builds a frame that is fully neutral in head pose and expressions.
fn neutral_frame(
    base: &AvatarControlFrame,
    produced_at: MonoTimeNs,
    state: TrackingState,
) -> AvatarControlFrame {
    AvatarControlFrame {
        source_seq: base.source_seq,
        captured_at: base.captured_at,
        produced_at,
        confidence: 0.0,
        state,
        head: HeadPose::default(),
        head_translation: HeadTranslationSignal::UNAVAILABLE,
        gaze: GazeSignal::degraded(0.0, 0.0, 0.0),
        expressions: ExpressionCoefficients::default(),
        detailed_face: None,
    }
}

/// Blends two frames, keeping the source sequence from `from`.
fn blend_frames(
    from: &AvatarControlFrame,
    to: &AvatarControlFrame,
    t: f32,
    state: TrackingState,
    produced_at: MonoTimeNs,
) -> AvatarControlFrame {
    let q_from = semantic_pose_to_quaternion(from.head);
    let q_to = semantic_pose_to_quaternion(to.head);
    let q_to = choose_shortest_arc(q_from, q_to);
    let q = q_from.slerp(&q_to, t);

    AvatarControlFrame {
        source_seq: from.source_seq,
        captured_at: from.captured_at,
        produced_at,
        confidence: lerp(from.confidence, to.confidence, t),
        state,
        head: quaternion_to_semantic_pose(q),
        head_translation: HeadTranslationSignal::blend(
            from.head_translation,
            to.head_translation,
            t,
        ),
        gaze: blend_gaze(from.gaze, to.gaze, t),
        expressions: blend_expressions(&from.expressions, &to.expressions, t),
        detailed_face: blend_detailed_face(from.detailed_face, to.detailed_face, t),
    }
}

/// Blends two optional coefficient sets.
///
/// A missing side means "no detailed coefficients", which is zero, so both
/// acquiring and releasing Perfect Sync stay continuous across the blend.
/// `TongueOut` is zero on both sides and therefore stays zero.
fn blend_detailed_face(
    from: Option<Arkit52Coefficients>,
    to: Option<Arkit52Coefficients>,
    t: f32,
) -> Option<Arkit52Coefficients> {
    let t = t.clamp(0.0, 1.0);
    match (from, to) {
        (None, None) => None,
        (Some(from), Some(to)) => {
            let mut values = [0.0; ARKIT52_CHANNEL_COUNT];
            for (slot, channel) in values.iter_mut().zip(ArkitBlendshape::ALL) {
                *slot = from.get(channel) + (to.get(channel) - from.get(channel)) * t;
            }
            Some(finish_detailed_face(values))
        }
        (Some(value), None) => Some(scale_detailed_face(value, 1.0 - t)),
        (None, Some(value)) => Some(scale_detailed_face(value, t)),
    }
}

/// Scales every channel toward zero, keeping `TongueOut` at zero.
fn scale_detailed_face(coefficients: Arkit52Coefficients, factor: f32) -> Arkit52Coefficients {
    let mut values = [0.0; ARKIT52_CHANNEL_COUNT];
    for (slot, channel) in values.iter_mut().zip(ArkitBlendshape::ALL) {
        *slot = coefficients.get(channel) * factor;
    }
    finish_detailed_face(values)
}

// Invariant: every stored coefficient is validated to `[0, 1]` and callers
// clamp their blend factor, so the result stays within `[0, 1]`.
#[allow(clippy::expect_used)]
fn finish_detailed_face(values: [f32; ARKIT52_CHANNEL_COUNT]) -> Arkit52Coefficients {
    Arkit52Coefficients::try_from_array(values).expect("detailed blend stays within [0, 1]")
}

fn blend_gaze(from: GazeSignal, to: GazeSignal, t: f32) -> GazeSignal {
    let t = t.clamp(0.0, 1.0);
    let horizontal = lerp(from.horizontal, to.horizontal, t);
    let vertical = lerp(from.vertical, to.vertical, t);
    let confidence = lerp(from.confidence, to.confidence, t);
    let state = if t >= 1.0 {
        to.state
    } else if matches!(from.state, GazeTrackingState::Tracked)
        && matches!(to.state, GazeTrackingState::Tracked)
    {
        GazeTrackingState::Tracked
    } else {
        GazeTrackingState::Degraded
    };
    match state {
        GazeTrackingState::Tracked => GazeSignal::tracked(horizontal, vertical, confidence),
        GazeTrackingState::Degraded => GazeSignal::degraded(horizontal, vertical, confidence),
        GazeTrackingState::Unavailable => GazeSignal::UNAVAILABLE,
    }
}

/// Linearly interpolates every expression coefficient.
fn blend_expressions(
    a: &ExpressionCoefficients,
    b: &ExpressionCoefficients,
    t: f32,
) -> ExpressionCoefficients {
    ExpressionCoefficients {
        blink_left: lerp(a.blink_left, b.blink_left, t),
        blink_right: lerp(a.blink_right, b.blink_right, t),
        aa: lerp(a.aa, b.aa, t),
        ih: lerp(a.ih, b.ih, t),
        ou: lerp(a.ou, b.ou, t),
        ee: lerp(a.ee, b.ee, t),
        oh: lerp(a.oh, b.oh, t),
        look_left: lerp(a.look_left, b.look_left, t),
        look_right: lerp(a.look_right, b.look_right, t),
        look_up: lerp(a.look_up, b.look_up, t),
        look_down: lerp(a.look_down, b.look_down, t),
        happy: lerp(a.happy, b.happy, t),
        angry: lerp(a.angry, b.angry, t),
        sad: lerp(a.sad, b.sad, t),
        relaxed: lerp(a.relaxed, b.relaxed, t),
        surprised: lerp(a.surprised, b.surprised, t),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use vtuber_core::types::FrameSeq;

    fn test_params() -> LossBlendProfile {
        LossBlendProfile {
            hold: Duration::from_millis(100),
            return_duration: Duration::from_millis(200),
            acquire: Duration::from_millis(100),
        }
    }

    fn frame(
        seq: u64,
        yaw: f32,
        pitch: f32,
        roll: f32,
        expression_value: f32,
    ) -> AvatarControlFrame {
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

    /// A tracked frame whose `TongueOut` is zero, matching the coefficients
    /// the detailed expression filter publishes.
    fn detailed_frame(seq: u64, jaw_open: f32) -> AvatarControlFrame {
        let mut values = [0.0; ARKIT52_CHANNEL_COUNT];
        values[ArkitBlendshape::JawOpen.index()] = jaw_open;
        values[ArkitBlendshape::MouthSmileLeft.index()] = jaw_open * 0.5;
        let mut frame = frame(seq, 0.2, -0.1, 0.05, 0.4);
        frame.detailed_face = Some(Arkit52Coefficients::try_from_array(values).unwrap());
        frame
    }

    #[test]
    fn loss_recovery_default_params_are_valid() {
        assert!(LossBlendProfile::default().validate().is_ok());
    }

    #[test]
    fn loss_recovery_rejects_zero_duration() {
        let err = LossBlendProfile {
            hold: Duration::ZERO,
            ..LossBlendProfile::default()
        }
        .validate()
        .unwrap_err();
        assert_eq!(err, LossBlendConfigError::ZeroDuration { field: "hold" });
    }

    #[test]
    fn loss_recovery_rejects_out_of_range_duration() {
        let err = LossBlendProfile {
            return_duration: Duration::from_secs(30),
            ..LossBlendProfile::default()
        }
        .validate()
        .unwrap_err();
        assert!(matches!(
            err,
            LossBlendConfigError::DurationOutOfRange {
                field: "return_duration",
                ..
            }
        ));
    }

    #[test]
    fn loss_recovery_holds_pose_and_confidence_during_hold() {
        let mut lr = LossRecovery::new(test_params()).unwrap();
        let tracked = frame(1, 0.5, 0.2, -0.3, 0.8);

        let _ = lr.update(
            TrackingState::Tracking,
            Duration::from_millis(16),
            Some(tracked.clone()),
            MonoTimeNs(33_333_333),
        );
        let held = lr
            .update(
                TrackingState::LostHold,
                Duration::from_millis(50),
                None,
                MonoTimeNs(50_000_000),
            )
            .expect("should emit a held frame");

        assert_eq!(held.source_seq, tracked.source_seq);
        assert_relative_eq!(held.head.yaw_rad, tracked.head.yaw_rad, epsilon = 1e-5);
        assert_relative_eq!(held.expressions.aa, tracked.expressions.aa, epsilon = 1e-5);
        assert_relative_eq!(held.confidence, tracked.confidence, epsilon = 1e-5);
        assert!(lr.is_returning());
    }

    #[test]
    fn loss_recovery_eases_authority_to_neutral_over_return_duration() {
        let mut lr = LossRecovery::new(test_params()).unwrap();
        let tracked = frame(1, 0.5, 0.25, -0.4, 0.8);

        let _ = lr.update(
            TrackingState::Tracking,
            Duration::from_millis(16),
            Some(tracked.clone()),
            MonoTimeNs(33_333_333),
        );
        let _ = lr.update(
            TrackingState::LostHold,
            Duration::from_millis(100),
            None,
            MonoTimeNs(100_000_000),
        );

        // Past the hold the eased authority must decrease monotonically and
        // never exceed the authority at the moment of loss.
        let full = tracked.head.yaw_rad * tracked.confidence;
        let mut previous = full;
        for step in 1..=5 {
            let eased = lr
                .update(
                    TrackingState::LostHold,
                    Duration::from_millis(60),
                    None,
                    MonoTimeNs(100_000_000 + step as u64 * 60_000_000),
                )
                .expect("returning frame should be emitted");
            let weight = eased.head.yaw_rad * eased.confidence;
            assert!(
                weight <= previous + 1.0e-6 && weight <= full + 1.0e-5,
                "the eased authority must decrease monotonically: step {step}: {weight}"
            );
            previous = weight;
        }

        // Once the return duration is over, the authority is exactly zero.
        let final_frame = lr
            .update(
                TrackingState::ReturningNeutral,
                Duration::from_secs(2),
                None,
                MonoTimeNs(1_500_000_000),
            )
            .expect("should emit the fully eased frame");
        assert_relative_eq!(final_frame.confidence, 0.0, epsilon = 1e-6);
        assert_relative_eq!(final_frame.expressions.aa, 0.0, epsilon = 1e-6);
        assert_eq!(final_frame.state, TrackingState::ReturningNeutral);
    }

    #[test]
    fn loss_recovery_reacquire_limits_jump() {
        let mut lr = LossRecovery::new(test_params()).unwrap();
        let first = frame(1, 0.0, 0.0, 0.0, 0.0);

        // Track a neutral pose.
        let _ = lr.update(
            TrackingState::Tracking,
            Duration::from_millis(16),
            Some(first.clone()),
            MonoTimeNs(16_000_000),
        );

        // Lose the face and let the return ease partway down.
        let _ = lr.update(
            TrackingState::LostHold,
            Duration::from_millis(100),
            None,
            MonoTimeNs(116_000_000),
        );
        let before_reacquire = lr
            .update(
                TrackingState::ReturningNeutral,
                Duration::from_millis(100),
                None,
                MonoTimeNs(216_000_000),
            )
            .unwrap();

        // Reacquire with a pose that is far from the current recovered pose.
        let target = frame(2, -1.2, 0.6, -0.4, 0.9);
        let during_recovery = lr
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
        let after_recovery = lr
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
        assert!(!lr.is_recovering());
    }

    #[test]
    fn loss_recovery_holds_detailed_face_during_loss_hold() {
        let mut lr = LossRecovery::new(test_params()).unwrap();
        let tracked = detailed_frame(11, 0.8);

        let _ = lr.update(
            TrackingState::Tracking,
            Duration::from_millis(16),
            Some(tracked.clone()),
            MonoTimeNs(16_000_000),
        );
        let held = lr
            .update(
                TrackingState::LostHold,
                Duration::from_millis(50),
                None,
                MonoTimeNs(66_000_000),
            )
            .expect("the hold should preserve the last detailed face state");
        let held_coefficients = held.detailed_face.expect("held frame keeps coefficients");

        assert!((held_coefficients.get(ArkitBlendshape::JawOpen) - 0.8).abs() < 1.0e-6);
        assert_eq!(held_coefficients.get(ArkitBlendshape::TongueOut), 0.0);
    }

    #[test]
    fn loss_recovery_reacquire_from_searching_does_not_snap() {
        let mut lr = LossRecovery::new(test_params()).unwrap();
        // Track a neutral pose, lose the face until the return is over in
        // Searching, then reacquire with the head turned.
        let _ = lr.update(
            TrackingState::Tracking,
            Duration::from_millis(16),
            Some(frame(1, 0.0, 0.0, 0.0, 0.0)),
            MonoTimeNs(33_333_333),
        );
        let _ = lr.update(
            TrackingState::LostHold,
            Duration::from_millis(100),
            None,
            MonoTimeNs(133_333_333),
        );
        let neutral = lr
            .update(
                TrackingState::Searching,
                Duration::from_millis(400),
                None,
                MonoTimeNs(533_333_333),
            )
            .unwrap();
        assert_relative_eq!(neutral.confidence, 0.0, epsilon = 1e-6);

        let reacquired = frame(2, 0.8, 0.0, 0.0, 0.0);
        let reconnected = lr
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
        assert!(lr.is_recovering());
    }
}
