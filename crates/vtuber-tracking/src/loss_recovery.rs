//! Loss glide, neutral decay, and recovery blend for face tracking.
//!
//! When a tracked face disappears, the avatar should not snap instantly to
//! neutral nor freeze in place. [`LossRecovery`] keeps the motion alive with
//! an inertial glide: the last observed head velocity (and translation
//! velocity) continues with exponential damping, staying within a bounded
//! excursion of the last valid pose. Expressions, gaze, and detailed face
//! state are held during the glide so a briefly lost face does not collapse
//! the avatar's face mid-speech.
//!
//! After [`LossRecoveryParams::glide_duration`] the glide hands over to the
//! return-to-neutral decay. When the face reappears at any point, a short
//! blend whose duration scales with the pose gap reconnects the tracked
//! frames smoothly instead of snapping.
//!
//! All timing uses the caller-supplied [`Duration`] delta and monotonic
//! timestamps, so behaviour is deterministic and testable without a wall
//! clock. The per-frame cost is a handful of quaternion and vector
//! operations with no allocation, so loss handling stays negligible even
//! when it runs on every frame.

use std::time::Duration;

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use thiserror::Error;

use vtuber_core::types::{
    AvatarControlFrame, ExpressionCoefficients, GazeSignal, GazeTrackingState, HeadPose,
    HeadTranslationSignal, HeadTranslationState, MonoTimeNs, TrackingState,
};
use vtuber_core::{ARKIT52_CHANNEL_COUNT, Arkit52Coefficients, ArkitBlendshape};

use crate::pose::{quaternion_to_semantic_pose, semantic_pose_to_quaternion};

/// Minimum duration for the inertial glide phase after face loss.
pub const MIN_GLIDE_DURATION: Duration = Duration::from_millis(50);
/// Maximum duration for the inertial glide phase after face loss.
pub const MAX_GLIDE_DURATION: Duration = Duration::from_millis(2000);
/// Minimum duration for the return-to-neutral decay phase.
pub const MIN_DECAY_DURATION: Duration = Duration::from_millis(100);
/// Maximum duration for the return-to-neutral decay phase.
pub const MAX_DECAY_DURATION: Duration = Duration::from_millis(2000);
/// Minimum duration for the reacquisition recovery blend.
pub const MIN_RECOVERY_DURATION: Duration = Duration::from_millis(20);
/// Maximum duration for the reacquisition recovery blend.
pub const MAX_RECOVERY_DURATION: Duration = Duration::from_millis(500);

/// Minimum head-rotation gap that triggers a reacquire blend from a
/// loss-related pose. Smaller gaps are reconnected by passing the tracked
/// frame through directly.
const REACQUIRE_MIN_HEAD_GAP_RAD: f32 = 0.02;
/// Minimum expression-coefficient gap that triggers a reacquire blend.
const REACQUIRE_MIN_EXPRESSION_GAP: f32 = 0.15;
/// Minimum gaze gap that triggers a reacquire blend.
const REACQUIRE_MIN_GAZE_GAP: f32 = 0.20;
/// Upper sanity bound for the estimated angular velocity in rad/s.
const MAX_ESTIMATED_ANGULAR_SPEED_RAD: f32 = 10.0;
/// Upper sanity bound for the estimated linear velocity in m/s.
const MAX_ESTIMATED_LINEAR_SPEED_MPS: f32 = 2.0;
/// Lower clamp for the delta used in velocity estimation, in seconds.
const MIN_INERTIA_DT_SEC: f32 = 0.010;
/// Upper clamp for the delta used in velocity estimation, in seconds.
const MAX_INERTIA_DT_SEC: f32 = 0.250;

/// Parameters governing loss glide, neutral decay, and recovery blending.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LossRecoveryParams {
    /// How long the inertial glide keeps the motion alive after the face is
    /// lost before the return-to-neutral decay begins.
    ///
    /// Must be within [`MIN_GLIDE_DURATION`] and [`MAX_GLIDE_DURATION`].
    pub glide_duration: Duration,
    /// How long the return-to-neutral motion takes.
    ///
    /// Must be within [`MIN_DECAY_DURATION`] and [`MAX_DECAY_DURATION`].
    pub decay_duration: Duration,
    /// Upper bound for the reacquisition recovery blend duration.
    ///
    /// The actual blend duration scales with the pose gap between the
    /// recovered pose and the newly tracked frame, divided by
    /// [`LossRecoveryParams::recovery_angular_speed`], and is clamped to
    /// `[MIN_RECOVERY_DURATION, recovery_duration]`.
    ///
    /// Must be within [`MIN_RECOVERY_DURATION`] and [`MAX_RECOVERY_DURATION`].
    pub recovery_duration: Duration,
    /// Angular speed in rad/s used to scale the reacquisition blend
    /// duration. Must be within `[0.1, 20.0]`.
    pub recovery_angular_speed: f32,
    /// Exponential damping time constant in seconds for the glided head and
    /// translation velocity. Must be within `[0.05, 1.5]`.
    pub glide_velocity_tau_sec: f32,
    /// Hard bound in radians on how far the glide may drift from the last
    /// valid head orientation. Must be within `[0.01, 1.5]`.
    pub max_glide_excursion_rad: f32,
    /// Hard bound in meters on how far the glide may drift from the last
    /// valid head translation. Must be within `[0.0, 0.5]`.
    pub max_glide_translation_meters: f32,
}

impl Default for LossRecoveryParams {
    fn default() -> Self {
        Self {
            glide_duration: Duration::from_millis(800),
            decay_duration: Duration::from_millis(500),
            recovery_duration: Duration::from_millis(350),
            recovery_angular_speed: 3.5,
            glide_velocity_tau_sec: 0.30,
            max_glide_excursion_rad: 0.25,
            max_glide_translation_meters: 0.10,
        }
    }
}

/// Errors that can occur while constructing a [`LossRecovery`] instance.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum LossRecoveryConfigError {
    /// A duration is zero, so timer-driven transitions would be ambiguous.
    #[error("{field} duration must be non-zero")]
    ZeroDuration {
        /// Name of the offending field.
        field: &'static str,
    },
    /// A duration is outside its permitted fixed range.
    #[error("{field} duration {got:?} is outside [{min:?}, {max:?}]")]
    DurationOutOfRange {
        /// Name of the offending field.
        field: &'static str,
        /// Minimum permitted duration.
        min: Duration,
        /// Maximum permitted duration.
        max: Duration,
        /// Supplied duration.
        got: Duration,
    },
    /// A scalar parameter is non-finite or outside its permitted range.
    #[error("{field} value {got} is outside [{min}, {max}]")]
    ScalarOutOfRange {
        /// Name of the offending field.
        field: &'static str,
        /// Minimum permitted value.
        min: f32,
        /// Maximum permitted value.
        max: f32,
        /// Supplied value.
        got: f32,
    },
}

impl LossRecoveryParams {
    /// Validates the timing parameters.
    ///
    /// # Errors
    ///
    /// Returns [`LossRecoveryConfigError::ZeroDuration`] if any duration is
    /// zero, [`LossRecoveryConfigError::DurationOutOfRange`] if a duration
    /// is outside its fixed range, or
    /// [`LossRecoveryConfigError::ScalarOutOfRange`] if a scalar parameter
    /// is non-finite or out of range.
    pub fn validate(&self) -> Result<(), LossRecoveryConfigError> {
        let ranges: [(&'static str, Duration, Duration, Duration); 3] = [
            (
                "glide_duration",
                MIN_GLIDE_DURATION,
                MAX_GLIDE_DURATION,
                self.glide_duration,
            ),
            (
                "decay_duration",
                MIN_DECAY_DURATION,
                MAX_DECAY_DURATION,
                self.decay_duration,
            ),
            (
                "recovery_duration",
                MIN_RECOVERY_DURATION,
                MAX_RECOVERY_DURATION,
                self.recovery_duration,
            ),
        ];

        for (field, min, max, got) in ranges {
            if got.is_zero() {
                return Err(LossRecoveryConfigError::ZeroDuration { field });
            }
            if got < min || got > max {
                return Err(LossRecoveryConfigError::DurationOutOfRange {
                    field,
                    min,
                    max,
                    got,
                });
            }
        }

        let scalars: [(&'static str, f32, f32, f32); 4] = [
            (
                "recovery_angular_speed",
                0.1,
                20.0,
                self.recovery_angular_speed,
            ),
            (
                "glide_velocity_tau_sec",
                0.05,
                1.5,
                self.glide_velocity_tau_sec,
            ),
            (
                "max_glide_excursion_rad",
                0.01,
                1.5,
                self.max_glide_excursion_rad,
            ),
            (
                "max_glide_translation_meters",
                0.0,
                0.5,
                self.max_glide_translation_meters,
            ),
        ];
        for (field, min, max, got) in scalars {
            if !got.is_finite() || got < min || got > max {
                return Err(LossRecoveryConfigError::ScalarOutOfRange {
                    field,
                    min,
                    max,
                    got,
                });
            }
        }

        Ok(())
    }
}

/// Inertial glide motion carried between loss frames.
///
/// The glide extrapolates the last tracked pose with the velocity observed
/// before the loss, decaying that velocity exponentially. The pose and
/// translation stay within the configured excursion bounds of the origin
/// frame.
#[derive(Clone, Debug, PartialEq)]
struct GlideMotion {
    /// Last valid tracked frame the glide started from.
    origin: AvatarControlFrame,
    /// Current glided head orientation.
    pose: UnitQuaternion<f32>,
    /// Current damped angular velocity in rad/s (world frame).
    angular_velocity: Vector3<f32>,
    /// Current glided translation in meters.
    translation: Vector3<f32>,
    /// Current damped linear velocity in m/s.
    linear_velocity: Vector3<f32>,
    /// Time spent gliding.
    elapsed: Duration,
}

/// Current phase of the loss-recovery state machine.
#[derive(Clone, Debug, PartialEq)]
// The by-value control frame intentionally includes the fixed-size validated
// ARKit52 payload; keeping it inline avoids a per-frame heap allocation.
#[allow(clippy::large_enum_variant)]
enum RecoveryState {
    /// No synthetic motion is in progress; pass tracked frames through.
    Idle,
    /// Gliding with damped inertia from the last valid frame.
    Gliding(GlideMotion),
    /// Returning from a glided or recovered pose to neutral.
    Returning {
        /// Pose at the start of the return motion.
        from: AvatarControlFrame,
        /// Time spent in the return phase.
        elapsed: Duration,
    },
    /// Blending from a recovered pose to a newly tracked frame.
    Recovering {
        /// Pose at the start of the recovery blend.
        from: AvatarControlFrame,
        /// Blend duration computed from the initial pose gap.
        duration: Duration,
        /// Time spent in the recovery phase.
        elapsed: Duration,
    },
}

/// Holds the motion alive with damped inertia when a face is lost, decays to
/// neutral after the glide expires, and blends back to tracked frames on
/// reacquire.
#[derive(Clone, Debug, PartialEq)]
pub struct LossRecovery {
    params: LossRecoveryParams,
    state: RecoveryState,
    last_valid: Option<AvatarControlFrame>,
    last_output: Option<AvatarControlFrame>,
    previous_output: Option<AvatarControlFrame>,
}

impl LossRecovery {
    /// Creates a new [`LossRecovery`] with the given parameters.
    ///
    /// # Errors
    ///
    /// Returns [`LossRecoveryConfigError`] if the parameters are invalid.
    pub fn new(params: LossRecoveryParams) -> Result<Self, LossRecoveryConfigError> {
        params.validate()?;
        Ok(Self {
            params,
            state: RecoveryState::Idle,
            last_valid: None,
            last_output: None,
            previous_output: None,
        })
    }

    /// Returns the configured parameters.
    #[must_use]
    pub fn params(&self) -> &LossRecoveryParams {
        &self.params
    }

    /// Returns `true` while the inertial glide is in progress.
    #[must_use]
    pub fn is_gliding(&self) -> bool {
        matches!(self.state, RecoveryState::Gliding(_))
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
    /// the last valid frame during glide, decay, and recovery so that a stale
    /// observation is not published as a new frame.
    #[must_use]
    pub fn update(
        &mut self,
        state: TrackingState,
        dt: Duration,
        tracked: Option<AvatarControlFrame>,
        produced_at: MonoTimeNs,
    ) -> Option<AvatarControlFrame> {
        // Keep track of the most recent valid frame for future glide phases.
        if let Some(ref t) = tracked {
            self.last_valid = Some(t.clone());
        }

        let old = std::mem::replace(&mut self.state, RecoveryState::Idle);

        let (next, output) = match (state, old, tracked) {
            // A tracked frame is available while we are actively tracking.
            // Pass it through, or start/continue a recovery blend if we were
            // previously gliding, returning, or mid-recovery.
            (
                TrackingState::Tracking | TrackingState::Acquiring,
                RecoveryState::Idle,
                Some(target),
            ) => {
                if self.should_blend_on_reacquire(&target) {
                    let from = self.last_output.clone().unwrap_or_else(|| target.clone());
                    let duration = self.recovery_duration_for(&from, &target);
                    advance_recovery(
                        from,
                        target,
                        Duration::ZERO,
                        dt,
                        duration,
                        state,
                        produced_at,
                    )
                } else {
                    self.record_pass_through(&target);
                    (RecoveryState::Idle, Some(target))
                }
            }
            (
                TrackingState::Tracking | TrackingState::Acquiring,
                RecoveryState::Gliding(_),
                Some(target),
            ) => {
                let from = self.last_output.clone().unwrap_or_else(|| target.clone());
                let duration = self.recovery_duration_for(&from, &target);
                advance_recovery(
                    from,
                    target,
                    Duration::ZERO,
                    dt,
                    duration,
                    state,
                    produced_at,
                )
            }
            (
                TrackingState::Tracking | TrackingState::Acquiring,
                RecoveryState::Returning { from, .. },
                Some(target),
            ) => {
                let from = self.last_output.clone().unwrap_or(from);
                let duration = self.recovery_duration_for(&from, &target);
                advance_recovery(
                    from,
                    target,
                    Duration::ZERO,
                    dt,
                    duration,
                    state,
                    produced_at,
                )
            }
            (
                TrackingState::Tracking | TrackingState::Acquiring,
                RecoveryState::Recovering {
                    from,
                    duration,
                    elapsed,
                },
                Some(target),
            ) => advance_recovery(from, target, elapsed, dt, duration, state, produced_at),

            // Lost: keep gliding with damped inertia until the glide window
            // expires, then hand over to the neutral decay. A decay that is
            // already running keeps advancing so repeated state flips cannot
            // oscillate the pose back and forth.
            (
                TrackingState::LostHold | TrackingState::ReturningNeutral,
                RecoveryState::Gliding(glide),
                None,
            ) => self.advance_glide(glide, dt, state, produced_at),
            (
                TrackingState::LostHold | TrackingState::ReturningNeutral,
                RecoveryState::Returning { from, elapsed },
                None,
            ) => advance_return(from, elapsed, dt, state, produced_at, &self.params),
            (TrackingState::LostHold | TrackingState::ReturningNeutral, _, None) => {
                self.enter_glide(dt, state, produced_at)
            }

            // Searching after the machine gave up: finish the episode
            // gracefully. A running glide freezes where it is and decays
            // from there instead of snapping to neutral.
            (TrackingState::Searching, RecoveryState::Gliding(glide), None) => {
                let from = glide_frame(&glide, state, produced_at);
                let output = from.clone();
                (
                    RecoveryState::Returning {
                        from,
                        elapsed: Duration::ZERO,
                    },
                    Some(output),
                )
            }
            (TrackingState::Searching, RecoveryState::Recovering { .. }, None) => {
                match self.last_output.clone() {
                    Some(from) => (
                        RecoveryState::Returning {
                            from,
                            elapsed: Duration::ZERO,
                        },
                        self.last_output.clone(),
                    ),
                    None => (RecoveryState::Idle, None),
                }
            }
            (TrackingState::Searching, RecoveryState::Returning { from, elapsed }, None) => {
                advance_return(from, elapsed, dt, state, produced_at, &self.params)
            }

            // Searching without synthetic motion: emit neutral frames so the
            // avatar does not stay frozen in the last pose.
            (TrackingState::Searching, _, None) => {
                if let Some(last) = self.last_output.clone() {
                    let neutral = neutral_frame(&last, produced_at, state);
                    (RecoveryState::Idle, Some(neutral))
                } else {
                    (RecoveryState::Idle, None)
                }
            }

            // Any other combination: preserve the previous state and output.
            (_, old, _) => (old, self.last_output.clone()),
        };

        self.state = next;
        if let Some(ref out) = output {
            self.last_output = Some(out.clone());
        }
        output
    }

    /// Records a tracked pass-through frame for future inertia estimation.
    fn record_pass_through(&mut self, target: &AvatarControlFrame) {
        self.previous_output = self.last_output.clone();
        self.last_output = Some(target.clone());
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

    /// Computes a reacquisition blend duration from the pose gap.
    ///
    /// The duration scales linearly with the angular distance and is clamped
    /// to `[MIN_RECOVERY_DURATION, recovery_duration]`.
    fn recovery_duration_for(
        &self,
        from: &AvatarControlFrame,
        to: &AvatarControlFrame,
    ) -> Duration {
        let gap =
            semantic_pose_to_quaternion(from.head).angle_to(&semantic_pose_to_quaternion(to.head));
        let speed = self.params.recovery_angular_speed;
        if !gap.is_finite() || !speed.is_finite() || speed <= 0.0 {
            return self.params.recovery_duration;
        }
        let seconds = (gap / speed).clamp(
            MIN_RECOVERY_DURATION.as_secs_f32(),
            self.params.recovery_duration.as_secs_f32(),
        );
        if !seconds.is_finite() {
            return self.params.recovery_duration;
        }
        Duration::from_secs_f32(seconds)
    }

    /// Starts a glide from the most recent output frame.
    ///
    /// When the previous episode already decayed fully to neutral (the
    /// machine can stay in `LostHold`/`ReturningNeutral` past the decay),
    /// the glide is skipped and static neutral frames keep being emitted so
    /// a finished episode never restarts its glide window.
    fn enter_glide(
        &mut self,
        dt: Duration,
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
        let (angular_velocity, linear_velocity) = self.estimate_inertia(&origin);
        let glide = GlideMotion {
            pose: semantic_pose_to_quaternion(origin.head),
            translation: Vector3::new(
                origin.head_translation.x_meters,
                origin.head_translation.y_meters,
                origin.head_translation.z_meters,
            ),
            angular_velocity,
            linear_velocity,
            origin,
            elapsed: Duration::ZERO,
        };
        self.advance_glide(glide, dt, state, produced_at)
    }

    /// Advances the glide one frame and emits the extrapolated frame.
    ///
    /// When the glide window expires the carry-over time is applied to the
    /// neutral decay immediately so no update is wasted on a static frame.
    fn advance_glide(
        &mut self,
        mut glide: GlideMotion,
        dt: Duration,
        state: TrackingState,
        produced_at: MonoTimeNs,
    ) -> (RecoveryState, Option<AvatarControlFrame>) {
        let params = self.params;
        let dt_sec = dt.as_secs_f32();
        let tau = params.glide_velocity_tau_sec.max(f32::EPSILON);
        let decay = (-dt_sec / tau).exp();
        glide.angular_velocity *= decay;
        glide.linear_velocity *= decay;

        // Rotate within the remaining excursion budget.
        let origin_q = semantic_pose_to_quaternion(glide.origin.head);
        let remaining = params.max_glide_excursion_rad - glide.pose.angle_to(&origin_q);
        if remaining > 0.0 {
            let mut step = glide.angular_velocity * dt_sec;
            let magnitude = step.magnitude();
            if magnitude > remaining {
                step *= remaining / magnitude;
            }
            glide.pose = UnitQuaternion::from_scaled_axis(step) * glide.pose;
        }

        // Translate within the remaining translation budget.
        if glide.origin.head_translation.is_available() {
            let origin_t = Vector3::new(
                glide.origin.head_translation.x_meters,
                glide.origin.head_translation.y_meters,
                glide.origin.head_translation.z_meters,
            );
            let room =
                params.max_glide_translation_meters - (glide.translation - origin_t).magnitude();
            if room > 0.0 {
                let mut step = glide.linear_velocity * dt_sec;
                let magnitude = step.magnitude();
                if magnitude > room {
                    step *= room / magnitude;
                }
                glide.translation += step;
            }
        }

        glide.elapsed = glide.elapsed.saturating_add(dt);
        let frame = glide_frame(&glide, state, produced_at);
        if glide.elapsed >= params.glide_duration {
            let carry = glide.elapsed - params.glide_duration;
            let blended = blend_to_neutral(
                &frame,
                fraction(carry, params.decay_duration),
                state,
                produced_at,
            );
            (
                RecoveryState::Returning {
                    from: frame,
                    elapsed: carry,
                },
                Some(blended),
            )
        } else {
            (RecoveryState::Gliding(glide), Some(frame))
        }
    }

    /// Estimates the angular and linear velocity at the moment of loss.
    ///
    /// Velocities are finite-difference estimates over the last two tracked
    /// pass-through frames. Synthetic (glide/decay/recovery) frames are never
    /// used, so a repeated loss episode cannot compound fabricated motion.
    fn estimate_inertia(&self, origin: &AvatarControlFrame) -> (Vector3<f32>, Vector3<f32>) {
        let Some(previous) = self.previous_output.as_ref() else {
            return (Vector3::zeros(), Vector3::zeros());
        };
        let is_real = |frame: &AvatarControlFrame| {
            matches!(
                frame.state,
                TrackingState::Tracking | TrackingState::Degraded | TrackingState::Acquiring
            )
        };
        if !is_real(previous) || !is_real(origin) {
            return (Vector3::zeros(), Vector3::zeros());
        }

        let dt_ns = origin.produced_at.0.saturating_sub(previous.produced_at.0);
        if dt_ns == 0 {
            return (Vector3::zeros(), Vector3::zeros());
        }
        let dt_sec =
            ((dt_ns as f32) / 1_000_000_000.0).clamp(MIN_INERTIA_DT_SEC, MAX_INERTIA_DT_SEC);

        let last_q = semantic_pose_to_quaternion(origin.head);
        let previous_q = semantic_pose_to_quaternion(previous.head);
        let delta = last_q * previous_q.inverse();
        let angular = cap_magnitude(
            delta.scaled_axis() / dt_sec,
            MAX_ESTIMATED_ANGULAR_SPEED_RAD,
        );

        let linear =
            if origin.head_translation.is_available() && previous.head_translation.is_available() {
                let delta_t = Vector3::new(
                    origin.head_translation.x_meters - previous.head_translation.x_meters,
                    origin.head_translation.y_meters - previous.head_translation.y_meters,
                    origin.head_translation.z_meters - previous.head_translation.z_meters,
                );
                cap_magnitude(delta_t / dt_sec, MAX_ESTIMATED_LINEAR_SPEED_MPS)
            } else {
                Vector3::zeros()
            };

        (angular, linear)
    }
}

/// Advances a recovery blend toward the latest tracked target.
///
/// `duration` was computed when the blend started from the initial pose gap
/// and is reused for every continuation so the progress stays monotonic.
fn advance_recovery(
    from: AvatarControlFrame,
    to: AvatarControlFrame,
    elapsed: Duration,
    dt: Duration,
    duration: Duration,
    state: TrackingState,
    produced_at: MonoTimeNs,
) -> (RecoveryState, Option<AvatarControlFrame>) {
    let elapsed = elapsed.saturating_add(dt);
    if elapsed >= duration {
        (RecoveryState::Idle, Some(to))
    } else {
        let t = fraction(elapsed, duration);
        let blended = blend_frames(&from, &to, t, state, produced_at);
        (
            RecoveryState::Recovering {
                from,
                duration,
                elapsed,
            },
            Some(blended),
        )
    }
}

/// Advances the return-to-neutral decay one frame.
fn advance_return(
    from: AvatarControlFrame,
    elapsed: Duration,
    dt: Duration,
    state: TrackingState,
    produced_at: MonoTimeNs,
    params: &LossRecoveryParams,
) -> (RecoveryState, Option<AvatarControlFrame>) {
    let elapsed = elapsed.saturating_add(dt);
    if elapsed >= params.decay_duration {
        // The last decay frame publishes exact zeros so the avatar releases
        // every Perfect Sync morph before the coefficients are dropped on the
        // following frames.
        let mut neutral = neutral_frame(&from, produced_at, state);
        neutral.detailed_face = from.detailed_face.map(|_| Arkit52Coefficients::default());
        (RecoveryState::Idle, Some(neutral))
    } else {
        let blended = blend_to_neutral(
            &from,
            fraction(elapsed, params.decay_duration),
            state,
            produced_at,
        );
        (RecoveryState::Returning { from, elapsed }, Some(blended))
    }
}

/// Builds the synthetic frame for the current glide state.
fn glide_frame(
    glide: &GlideMotion,
    state: TrackingState,
    produced_at: MonoTimeNs,
) -> AvatarControlFrame {
    let origin = &glide.origin;
    let head_translation = if origin.head_translation.is_available() {
        HeadTranslationSignal {
            x_meters: glide.translation.x,
            y_meters: glide.translation.y,
            z_meters: glide.translation.z,
            state: origin.head_translation.state,
        }
    } else {
        HeadTranslationSignal::UNAVAILABLE
    };

    AvatarControlFrame {
        source_seq: origin.source_seq,
        captured_at: origin.captured_at,
        produced_at,
        confidence: origin.confidence,
        state,
        head: quaternion_to_semantic_pose(glide.pose),
        head_translation,
        gaze: origin.gaze,
        expressions: origin.expressions,
        detailed_face: origin.detailed_face,
    }
}

/// Converts a progress fraction from `0` to `total` duration.
fn fraction(elapsed: Duration, total: Duration) -> f32 {
    if total.is_zero() {
        return 1.0;
    }
    (elapsed.as_secs_f32() / total.as_secs_f32()).clamp(0.0, 1.0)
}

/// Returns `true` when the frame is a fully decayed neutral frame.
///
/// `blend_to_neutral` and `neutral_frame` produce exactly-identity head
/// poses with exactly-zero confidence at `t = 1`, so this check reliably
/// recognises a completed return-to-neutral.
fn is_fully_neutral(frame: &AvatarControlFrame) -> bool {
    frame.confidence <= 0.0 && frame.head == HeadPose::default()
}

/// Scales a vector down to `max` magnitude when it exceeds it.
fn cap_magnitude(mut vector: Vector3<f32>, max: f32) -> Vector3<f32> {
    let magnitude = vector.magnitude();
    if magnitude > max && magnitude > 0.0 {
        vector *= max / magnitude;
    }
    vector
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

/// Blends a frame toward the neutral pose and zero expressions.
fn blend_to_neutral(
    from: &AvatarControlFrame,
    t: f32,
    state: TrackingState,
    produced_at: MonoTimeNs,
) -> AvatarControlFrame {
    let q_from = semantic_pose_to_quaternion(from.head);
    let q_to = UnitQuaternion::identity();
    let q_to = choose_shortest_arc(q_from, q_to);
    let q = q_from.slerp(&q_to, t);

    AvatarControlFrame {
        source_seq: from.source_seq,
        captured_at: from.captured_at,
        produced_at,
        confidence: lerp(from.confidence, 0.0, t),
        state,
        head: quaternion_to_semantic_pose(q),
        head_translation: HeadTranslationSignal::blend(
            from.head_translation,
            zeroed_head_translation(from.head_translation),
            t,
        ),
        gaze: blend_gaze(from.gaze, GazeSignal::degraded(0.0, 0.0, 0.0), t),
        expressions: blend_expressions(&from.expressions, &ExpressionCoefficients::default(), t),
        detailed_face: from
            .detailed_face
            .map(|coefficients| scale_detailed_face(coefficients, 1.0 - t)),
    }
}

/// Builds a zero-translation copy of `signal` preserving its availability state.
///
/// Used when blending toward neutral so a tracked translation decays to zero
/// movement instead of collapsing straight to unavailable.
fn zeroed_head_translation(signal: HeadTranslationSignal) -> HeadTranslationSignal {
    match signal.state {
        HeadTranslationState::Unavailable => HeadTranslationSignal::UNAVAILABLE,
        HeadTranslationState::Degraded => HeadTranslationSignal::degraded(0.0, 0.0, 0.0),
        HeadTranslationState::Tracked => HeadTranslationSignal::tracked(0.0, 0.0, 0.0),
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

    fn test_params() -> LossRecoveryParams {
        LossRecoveryParams {
            glide_duration: Duration::from_millis(100),
            decay_duration: Duration::from_millis(200),
            recovery_duration: Duration::from_millis(100),
            ..LossRecoveryParams::default()
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

    #[test]
    fn loss_recovery_default_params_are_valid() {
        assert!(LossRecoveryParams::default().validate().is_ok());
    }

    #[test]
    fn loss_recovery_rejects_zero_duration() {
        let err = LossRecoveryParams {
            glide_duration: Duration::ZERO,
            ..LossRecoveryParams::default()
        }
        .validate()
        .unwrap_err();
        assert_eq!(
            err,
            LossRecoveryConfigError::ZeroDuration {
                field: "glide_duration"
            }
        );
    }

    #[test]
    fn loss_recovery_rejects_out_of_range_duration() {
        let err = LossRecoveryParams {
            glide_duration: Duration::from_secs(5),
            ..LossRecoveryParams::default()
        }
        .validate()
        .unwrap_err();
        assert!(matches!(
            err,
            LossRecoveryConfigError::DurationOutOfRange {
                field: "glide_duration",
                ..
            }
        ));
    }

    #[test]
    fn loss_recovery_rejects_out_of_range_scalar() {
        let err = LossRecoveryParams {
            recovery_angular_speed: 0.0,
            ..LossRecoveryParams::default()
        }
        .validate()
        .unwrap_err();
        assert!(matches!(
            err,
            LossRecoveryConfigError::ScalarOutOfRange {
                field: "recovery_angular_speed",
                ..
            }
        ));

        let err = LossRecoveryParams {
            glide_velocity_tau_sec: f32::NAN,
            ..LossRecoveryParams::default()
        }
        .validate()
        .unwrap_err();
        assert!(matches!(
            err,
            LossRecoveryConfigError::ScalarOutOfRange {
                field: "glide_velocity_tau_sec",
                ..
            }
        ));
    }

    #[test]
    fn loss_recovery_preserves_stationary_pose_during_glide() {
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
            .expect("should emit glided frame");

        assert_eq!(held.source_seq, tracked.source_seq);
        assert_relative_eq!(held.head.yaw_rad, tracked.head.yaw_rad, epsilon = 1e-5);
        assert_relative_eq!(held.expressions.aa, tracked.expressions.aa, epsilon = 1e-5);
        assert!(lr.is_gliding());
    }

    #[test]
    fn loss_recovery_glide_continues_motion_with_inertia() {
        let mut lr = LossRecovery::new(test_params()).unwrap();
        // Two tracked frames turning the head: yaw 0.2 -> 0.3 over 33 ms.
        let _ = lr.update(
            TrackingState::Tracking,
            Duration::from_millis(16),
            Some(frame(1, 0.2, 0.0, 0.0, 0.0)),
            MonoTimeNs(33_333_333),
        );
        let _ = lr.update(
            TrackingState::Tracking,
            Duration::from_millis(16),
            Some(frame(2, 0.3, 0.0, 0.0, 0.0)),
            MonoTimeNs(66_666_666),
        );

        let glided = lr
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
        let mut lr = LossRecovery::new(test_params()).unwrap();
        // A violent single-frame turn produces a large estimated velocity.
        let _ = lr.update(
            TrackingState::Tracking,
            Duration::from_millis(16),
            Some(frame(1, 0.0, 0.0, 0.0, 0.0)),
            MonoTimeNs(33_333_333),
        );
        let _ = lr.update(
            TrackingState::Tracking,
            Duration::from_millis(16),
            Some(frame(2, 1.5, 0.0, 0.0, 0.0)),
            MonoTimeNs(66_666_666),
        );

        let origin_q = semantic_pose_to_quaternion(frame(2, 1.5, 0.0, 0.0, 0.0).head);
        let mut max_glide_yaw = 0.0_f32;
        let mut previous_yaw: Option<f32> = None;
        for step in 1..=20 {
            let glided = lr
                .update(
                    TrackingState::LostHold,
                    Duration::from_millis(33),
                    None,
                    MonoTimeNs(66_666_666 + step as u64 * 33_333_333),
                )
                .expect("glide should emit a frame");
            let yaw = glided.head.yaw_rad;
            if lr.is_gliding() {
                let excursion = semantic_pose_to_quaternion(glided.head).angle_to(&origin_q);
                assert!(
                    excursion <= lr.params().max_glide_excursion_rad + 1.0e-4,
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
    fn loss_recovery_does_not_stick_after_glide_timeout() {
        let mut lr = LossRecovery::new(test_params()).unwrap();
        let tracked = frame(1, 0.5, 0.0, 0.0, 0.8);

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
        let returning = lr
            .update(
                TrackingState::LostHold,
                Duration::from_millis(50),
                None,
                MonoTimeNs(150_000_000),
            )
            .expect("should emit frame");

        // After the glide timeout (100 ms) plus a bit of decay, the pose
        // should be on its way to neutral, not still equal to the tracked
        // pose.
        assert!(
            returning.head.yaw_rad.abs() < tracked.head.yaw_rad.abs(),
            "yaw should decay toward zero, got {}",
            returning.head.yaw_rad
        );
        assert!(
            returning.expressions.aa < tracked.expressions.aa,
            "expression should decay toward zero, got {}",
            returning.expressions.aa
        );
        assert!(lr.is_returning());
    }

    #[test]
    fn loss_recovery_returns_to_neutral_over_decay_duration() {
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
        let final_frame = lr
            .update(
                TrackingState::ReturningNeutral,
                Duration::from_millis(250),
                None,
                MonoTimeNs(350_000_000),
            )
            .expect("should emit neutral frame");

        assert_relative_eq!(final_frame.head.yaw_rad, 0.0, epsilon = 1e-4);
        assert_relative_eq!(final_frame.head.pitch_rad, 0.0, epsilon = 1e-4);
        assert_relative_eq!(final_frame.head.roll_rad, 0.0, epsilon = 1e-4);
        assert_relative_eq!(final_frame.expressions.aa, 0.0, epsilon = 1e-4);
        assert_eq!(final_frame.state, TrackingState::ReturningNeutral);
    }

    #[test]
    fn loss_recovery_uses_shortest_arc_to_neutral() {
        let mut lr = LossRecovery::new(test_params()).unwrap();
        // A large positive yaw close to +pi. The shortest arc to identity is
        // to decrease the magnitude, not to wrap through -pi.
        let tracked = frame(1, 3.0, 0.0, 0.0, 0.0);

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
        let returning = lr
            .update(
                TrackingState::ReturningNeutral,
                Duration::from_millis(100),
                None,
                MonoTimeNs(200_000_000),
            )
            .expect("should be returning");

        // Halfway through the 200 ms decay, yaw should still be positive and
        // roughly half the original magnitude.
        assert!(
            returning.head.yaw_rad > 0.0 && returning.head.yaw_rad < tracked.head.yaw_rad,
            "shortest arc should stay on the positive side, got {}",
            returning.head.yaw_rad
        );
    }

    #[test]
    fn loss_recovery_reacquire_blends_smoothly() {
        let mut lr = LossRecovery::new(test_params()).unwrap();
        let first = frame(1, 0.0, 0.0, 0.0, 0.0);
        let reacquired = frame(3, -1.0, 0.0, 0.0, 0.0);

        let _ = lr.update(
            TrackingState::Tracking,
            Duration::from_millis(16),
            Some(first.clone()),
            MonoTimeNs(33_333_333),
        );
        // Lose the face and let it return partly to neutral.
        let _ = lr.update(
            TrackingState::LostHold,
            Duration::from_millis(100),
            None,
            MonoTimeNs(133_333_333),
        );
        let before_reacquire = lr
            .update(
                TrackingState::ReturningNeutral,
                Duration::from_millis(100),
                None,
                MonoTimeNs(233_333_333),
            )
            .unwrap();

        // Reacquire with a large opposing pose.
        let during_recovery = lr
            .update(
                TrackingState::Tracking,
                Duration::from_millis(50),
                Some(reacquired.clone()),
                MonoTimeNs(283_333_333),
            )
            .unwrap();

        // The recovery output should sit between the pre-reacquire pose and
        // the target, not jump all the way to the target.
        assert!(
            during_recovery.head.yaw_rad.abs() < reacquired.head.yaw_rad.abs(),
            "recovery should not jump to target immediately, got {}",
            during_recovery.head.yaw_rad
        );
        assert!(
            during_recovery.head.yaw_rad.signum() != before_reacquire.head.yaw_rad.signum()
                || during_recovery.head.yaw_rad.abs() < before_reacquire.head.yaw_rad.abs(),
            "recovery should move toward target, got {} from {}",
            during_recovery.head.yaw_rad,
            before_reacquire.head.yaw_rad
        );
        assert!(lr.is_recovering());
    }

    #[test]
    fn loss_recovery_reacquire_from_searching_does_not_snap() {
        let mut lr = LossRecovery::new(test_params()).unwrap();
        // Track a turned pose, lose the face until fully neutral in
        // Searching, then reacquire with the head still turned.
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
                Duration::from_millis(200),
                None,
                MonoTimeNs(333_333_333),
            )
            .unwrap();
        assert_relative_eq!(neutral.head.yaw_rad, 0.0, epsilon = 1e-4);

        let reacquired = frame(2, 0.8, 0.0, 0.0, 0.0);
        let reconnected = lr
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
        assert!(lr.is_recovering());
    }

    #[test]
    fn gaze_holds_returns_neutral_and_reacquires_without_snap() {
        let mut lr = LossRecovery::new(test_params()).unwrap();
        let mut tracked = frame(1, 0.2, 0.0, 0.0, 0.0);
        tracked.gaze = GazeSignal::tracked(0.8, -0.4, 0.9);
        let first = lr
            .update(
                TrackingState::Tracking,
                Duration::from_millis(16),
                Some(tracked.clone()),
                MonoTimeNs(16_000_000),
            )
            .unwrap();
        let held = lr
            .update(
                TrackingState::LostHold,
                Duration::from_millis(50),
                None,
                MonoTimeNs(66_000_000),
            )
            .unwrap();
        assert_eq!(held.gaze.horizontal, first.gaze.horizontal);

        let returning = lr
            .update(
                TrackingState::ReturningNeutral,
                Duration::from_millis(150),
                None,
                MonoTimeNs(216_000_000),
            )
            .unwrap();
        assert!(returning.gaze.horizontal.abs() < held.gaze.horizontal.abs());

        let mut target = frame(2, -0.2, 0.0, 0.0, 0.0);
        target.gaze = GazeSignal::tracked(-0.8, 0.2, 0.9);
        let recovering = lr
            .update(
                TrackingState::Tracking,
                Duration::from_millis(50),
                Some(target),
                MonoTimeNs(266_000_000),
            )
            .unwrap();
        assert!(recovering.gaze.horizontal > -0.8);
        assert!(recovering.gaze.horizontal < returning.gaze.horizontal);
        assert_eq!(recovering.gaze.state, GazeTrackingState::Degraded);
    }
}
