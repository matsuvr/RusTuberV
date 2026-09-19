//! Unified tracking-loss ramp shared by the arm channels and the face
//! pipeline.
//!
//! A tracked channel that is lost must not snap: it holds the authority it
//! had when it was lost for [`LossBlendProfile::hold`], then eases back to
//! the relaxed (virtual arm / neutral) pose over
//! [`LossBlendProfile::return_duration`]. A reacquired channel ramps back to
//! full authority over [`LossBlendProfile::acquire`], continuing from
//! wherever the return had decayed to. Both directions are smoothstep-shaped,
//! so a blend starts and lands with zero slope and never jumps.
//!
//! [`LossBlend`] tracks that timeline per channel in weight space; the face
//! pipeline applies the same timeline in pose space via
//! [`loss_return_factor`].

use std::time::Duration;

use thiserror::Error;

use vtuber_core::types::MonoTimeNs;

/// Minimum duration for the loss hold.
pub const MIN_HOLD_DURATION: Duration = Duration::from_millis(10);
/// Maximum duration for the loss hold.
pub const MAX_HOLD_DURATION: Duration = Duration::from_millis(1_000);
/// Minimum duration for the eased return to the relaxed pose.
pub const MIN_RETURN_DURATION: Duration = Duration::from_millis(100);
/// Maximum duration for the eased return to the relaxed pose.
pub const MAX_RETURN_DURATION: Duration = Duration::from_secs(10);
/// Minimum duration for the reacquire authority ramp.
pub const MIN_ACQUIRE_DURATION: Duration = Duration::from_millis(20);
/// Maximum duration for the reacquire authority ramp.
pub const MAX_ACQUIRE_DURATION: Duration = Duration::from_secs(2);

/// Errors produced while validating a [`LossBlendProfile`].
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum LossBlendConfigError {
    /// A duration was zero, so the timeline boundary would be ambiguous.
    #[error("{field} duration must be non-zero")]
    ZeroDuration {
        /// Name of the offending field.
        field: &'static str,
    },
    /// A duration was outside its permitted fixed range.
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
}

/// Timing of the unified tracking-loss ramp.
///
/// One profile drives every loss/recovery timeline in the workspace: a lost
/// channel holds its authority for [`LossBlendProfile::hold`], then eases to
/// the relaxed pose over [`LossBlendProfile::return_duration`], and a
/// reacquired channel ramps back up over [`LossBlendProfile::acquire`]. The
/// defaults match the evaluated arm-tracking values, and the face pipeline
/// shares them so head and hands lose and recover identically.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LossBlendProfile {
    /// How long a lost channel keeps its last value before returning.
    pub hold: Duration,
    /// How long the eased return to the relaxed pose takes after the hold.
    ///
    /// The default is deliberately long: when a channel is lost, it should
    /// sink back to its relaxed pose like a person lowering a hand, not snap
    /// to neutral.
    pub return_duration: Duration,
    /// How long a fully reacquired channel takes to blend back in.
    ///
    /// The ramp is smoothstep-shaped, so a reacquired channel accelerates and
    /// settles instead of starting at full speed.
    pub acquire: Duration,
}

impl Default for LossBlendProfile {
    fn default() -> Self {
        Self {
            hold: Duration::from_millis(150),
            return_duration: Duration::from_secs(5),
            acquire: Duration::from_secs(1),
        }
    }
}

impl LossBlendProfile {
    /// Validates that every duration is non-zero and in range.
    ///
    /// # Errors
    ///
    /// Returns [`LossBlendConfigError`] for a zero or out-of-range duration.
    pub fn validate(&self) -> Result<(), LossBlendConfigError> {
        let ranges = [
            ("hold", MIN_HOLD_DURATION, MAX_HOLD_DURATION, self.hold),
            (
                "return_duration",
                MIN_RETURN_DURATION,
                MAX_RETURN_DURATION,
                self.return_duration,
            ),
            (
                "acquire",
                MIN_ACQUIRE_DURATION,
                MAX_ACQUIRE_DURATION,
                self.acquire,
            ),
        ];
        for (field, min, max, got) in ranges {
            if got.is_zero() {
                return Err(LossBlendConfigError::ZeroDuration { field });
            }
            if got < min || got > max {
                return Err(LossBlendConfigError::DurationOutOfRange {
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

/// Smoothstep easing on `0.0..=1.0` with zero slope at both ends.
pub(crate) fn smoothstep(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    value * value * (3.0 - 2.0 * value)
}

/// Loss-ease factor in `[0, 1]` for `elapsed` since the channel was lost.
///
/// `1.0` during the hold, then `1 - smoothstep((elapsed - hold) / return)`,
/// then `0`. This is the timeline a lost authority follows down to the
/// relaxed pose.
#[must_use]
pub fn loss_return_factor(elapsed: Duration, profile: &LossBlendProfile) -> f32 {
    if elapsed <= profile.hold {
        1.0
    } else {
        let return_ms = profile.return_duration.as_secs_f32();
        if return_ms <= 0.0 {
            return 0.0;
        }
        let past = elapsed.saturating_sub(profile.hold).as_secs_f32();
        if past >= return_ms {
            0.0
        } else {
            1.0 - smoothstep(past / return_ms)
        }
    }
}

/// Reacquire-ramp factor in `[0, 1]` for `elapsed` since the channel was
/// reacquired. Smoothstep-shaped so the ramp accelerates and settles.
#[must_use]
pub fn acquire_factor(elapsed: Duration, profile: &LossBlendProfile) -> f32 {
    let acquire_secs = profile.acquire.as_secs_f32();
    if acquire_secs <= 0.0 {
        return 1.0;
    }
    smoothstep(elapsed.as_secs_f32() / acquire_secs)
}

/// A per-channel display blend that advances on render ticks and observations.
///
/// A present channel rises toward full authority over the acquire time; a lost
/// channel holds the authority it had when it was lost, then eases to zero over
/// the return time. Both directions are smoothstep-shaped, so a blend starts
/// and lands with no velocity discontinuity, and both continue from the current
/// weight, so losing a channel mid-acquire and reacquiring one mid-return never
/// jump.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LossBlend {
    weight: f32,
    present: bool,
    lost_at: MonoTimeNs,
    loss_weight: f32,
    /// Weight the current acquire ramp started from.
    acquire_start_weight: f32,
    /// Time the current acquire ramp started at.
    acquired_at: MonoTimeNs,
}

impl LossBlend {
    /// A blend at zero authority, waiting for its first observation.
    pub const fn new() -> Self {
        Self {
            weight: 0.0,
            present: false,
            lost_at: MonoTimeNs(0),
            loss_weight: 0.0,
            acquire_start_weight: 0.0,
            acquired_at: MonoTimeNs(0),
        }
    }

    /// Current authority weight in `[0, 1]`.
    pub const fn weight(&self) -> f32 {
        self.weight
    }

    /// Advances the blend one tick.
    ///
    /// A newly present channel starts an acquire ramp from its current
    /// weight; a newly lost channel records its weight and starts the
    /// hold/return timeline.
    pub fn advance(&mut self, now: MonoTimeNs, present: bool, profile: &LossBlendProfile) {
        if present {
            if !self.present {
                self.present = true;
                self.acquire_start_weight = self.weight;
                self.acquired_at = now;
            }
            let progress = Duration::from_nanos(now.0.saturating_sub(self.acquired_at.0));
            self.weight = self.acquire_start_weight
                + (1.0 - self.acquire_start_weight) * acquire_factor(progress, profile);
        } else {
            if self.present {
                self.present = false;
                self.lost_at = now;
                self.loss_weight = self.weight;
            }
            let elapsed = Duration::from_nanos(now.0.saturating_sub(self.lost_at.0));
            self.weight = self.loss_weight * loss_return_factor(elapsed, profile);
        }
    }
}

impl Default for LossBlend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> LossBlendProfile {
        LossBlendProfile::default()
    }

    #[test]
    fn loss_return_factor_holds_then_eases_to_zero() {
        let profile = profile();
        assert_eq!(loss_return_factor(Duration::ZERO, &profile), 1.0);
        assert_eq!(
            loss_return_factor(profile.hold, &profile),
            1.0,
            "the hold keeps full authority"
        );
        let mid = loss_return_factor(profile.hold + profile.return_duration / 2, &profile);
        assert!((mid - 0.5).abs() < 1.0e-3, "{mid}");
        assert_eq!(
            loss_return_factor(profile.hold + profile.return_duration, &profile),
            0.0
        );
        assert_eq!(
            loss_return_factor(profile.hold + profile.return_duration * 2, &profile),
            0.0
        );
    }

    #[test]
    fn acquire_factor_is_smoothstep_progress() {
        let profile = profile();
        assert_eq!(acquire_factor(Duration::ZERO, &profile), 0.0);
        let mid = acquire_factor(profile.acquire / 2, &profile);
        assert!((mid - 0.5).abs() < 1.0e-3, "{mid}");
        assert_eq!(acquire_factor(profile.acquire, &profile), 1.0);
    }

    #[test]
    fn profile_validate_rejects_zero_and_out_of_range() {
        let mut bad = profile();
        bad.hold = Duration::ZERO;
        assert_eq!(
            bad.validate(),
            Err(LossBlendConfigError::ZeroDuration { field: "hold" })
        );

        let mut bad = profile();
        bad.return_duration = Duration::from_secs(30);
        assert!(matches!(
            bad.validate(),
            Err(LossBlendConfigError::DurationOutOfRange {
                field: "return_duration",
                ..
            })
        ));

        let mut bad = profile();
        bad.acquire = Duration::from_millis(5);
        assert!(matches!(
            bad.validate(),
            Err(LossBlendConfigError::DurationOutOfRange {
                field: "acquire",
                ..
            })
        ));

        assert_eq!(profile().validate(), Ok(()));
    }

    #[test]
    fn channel_blend_holds_and_eases_like_the_loss_timeline() {
        let mut blend = LossBlend::new();
        let profile = profile();
        let start = MonoTimeNs(1_000_000);
        blend.advance(start, true, &profile);
        let settle = MonoTimeNs(start.0 + profile.acquire.as_nanos() as u64);
        blend.advance(settle, true, &profile);
        assert_eq!(blend.weight(), 1.0, "full acquire ramps to full authority");

        let lost = MonoTimeNs(settle.0 + 100_000_000);
        blend.advance(lost, false, &profile);
        assert_eq!(blend.weight(), 1.0, "hold keeps the weight at loss");

        blend.advance(
            MonoTimeNs(lost.0 + profile.hold.as_nanos() as u64),
            false,
            &profile,
        );
        assert!(
            (blend.weight() - 1.0).abs() < 1.0e-3,
            "the return has not started yet: {}",
            blend.weight()
        );

        blend.advance(
            MonoTimeNs(lost.0 + (profile.hold + profile.return_duration).as_nanos() as u64),
            false,
            &profile,
        );
        assert_eq!(blend.weight(), 0.0, "the return reaches zero");
    }

    #[test]
    fn channel_blend_ramps_up_over_acquire_from_the_current_weight() {
        let mut blend = LossBlend::new();
        let profile = profile();
        let start = MonoTimeNs(1_000_000);
        blend.advance(start, true, &profile);
        let settle = MonoTimeNs(start.0 + profile.acquire.as_nanos() as u64);
        blend.advance(settle, true, &profile);
        assert_eq!(blend.weight(), 1.0);

        let lost = MonoTimeNs(settle.0 + 100_000_000);
        blend.advance(lost, false, &profile);
        let mid_return =
            MonoTimeNs(lost.0 + (profile.hold + profile.return_duration / 2).as_nanos() as u64);
        blend.advance(mid_return, false, &profile);
        let mid = blend.weight();
        assert!(mid > 0.0 && mid < 1.0, "mid-return weight: {mid}");

        // Reacquiring continues the ramp from the retained weight.
        blend.advance(MonoTimeNs(mid_return.0 + 1_000_000), true, &profile);
        let resumed = blend.weight();
        assert!(
            (resumed - mid).abs() < 0.1,
            "the ramp starts near the retained weight: {resumed} vs {mid}"
        );

        blend.advance(
            MonoTimeNs(mid_return.0 + 1_000_000 + profile.acquire.as_nanos() as u64),
            true,
            &profile,
        );
        assert_eq!(blend.weight(), 1.0);
    }
}
