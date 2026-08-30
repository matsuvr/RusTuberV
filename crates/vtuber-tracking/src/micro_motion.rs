//! Bounded procedural micro-motion after tracking loss (`DESIGN.md` §11.10,
//! Issue #172, ADR-021).
//!
//! Instead of freezing or snapping to neutral, the upper body transitions
//! into a low-frequency bounded idle sway after the face is lost. The motion
//! is supplied purely as virtual head/body target offsets (Issues #165/#167),
//! composed with a loss-scoped vertical breathing oscillation (ADR-021) so a
//! prolonged camera or face failure keeps the avatar alive in its default
//! pose. The hips output is never touched.
//!
//! Determinism: no OS RNG. Per-bucket targets are derived by hashing the
//! avatar generation seed and a time-bucket index (splitmix-style), then
//! smoothly interpolated with a smoothstep between bucket boundaries. The
//! breathing axis is a pure sine of elapsed-since-loss time. The trajectory
//! is a pure function of elapsed-since-loss time, so 30/60/120 FPS
//! evaluations of the same loss episode produce near-identical curves.

use std::time::Duration;

use vtuber_core::types::{MonoTimeNs, TrackingState};

/// Body-scale-aware typed profile for tracking-loss micro-motion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MicroMotionProfile {
    /// Time from losing the face until full micro-motion amplitude.
    pub transition: Duration,
    /// Target update cycle of the bounded sway.
    pub period: Duration,
    /// Lateral amplitude as a ratio of body scale (seed `0.015`).
    pub x_amplitude_ratio: f32,
    /// Depth amplitude as a ratio of body scale (seed `0.030`).
    pub z_amplitude_ratio: f32,
    /// Yaw amplitude in radians (seed `4°`).
    pub yaw_amplitude_radians: f32,
    /// Pitch amplitude in radians (seed `4°`).
    pub pitch_amplitude_radians: f32,
    /// Roll offset in radians; the analyzed design generates none (seed `0`).
    pub roll_amplitude_radians: f32,
    /// Breathing cycle period during the loss episode (seed `4.0 s`,
    /// roughly 15 breaths per minute).
    pub breath_period: Duration,
    /// Vertical breathing amplitude as a ratio of body scale (seed `0.010`).
    pub breath_amplitude_ratio: f32,
}

impl Default for MicroMotionProfile {
    fn default() -> Self {
        Self {
            transition: Duration::from_millis(4_000),
            period: Duration::from_millis(2_500),
            x_amplitude_ratio: 0.015,
            z_amplitude_ratio: 0.030,
            yaw_amplitude_radians: 4.0_f32.to_radians(),
            pitch_amplitude_radians: 4.0_f32.to_radians(),
            roll_amplitude_radians: 0.0,
            breath_period: Duration::from_millis(4_000),
            breath_amplitude_ratio: 0.010,
        }
    }
}

impl MicroMotionProfile {
    /// Validates that all fields are finite and inside their contract ranges.
    ///
    /// # Errors
    ///
    /// Returns [`MicroMotionProfileError`] when durations are zero or
    /// non-finite when converted, amplitudes are negative or non-finite, or
    /// rotation amplitudes exceed half a turn.
    pub fn validate(&self) -> Result<(), MicroMotionProfileError> {
        if self.transition.is_zero() || self.period.is_zero() || self.breath_period.is_zero() {
            return Err(MicroMotionProfileError::ZeroDuration);
        }
        for (name, ratio) in [
            ("x_amplitude_ratio", self.x_amplitude_ratio),
            ("z_amplitude_ratio", self.z_amplitude_ratio),
            ("breath_amplitude_ratio", self.breath_amplitude_ratio),
        ] {
            if !ratio.is_finite() || ratio < 0.0 {
                return Err(MicroMotionProfileError::NegativeAmplitude { name, value: ratio });
            }
        }
        for (name, value) in [
            ("yaw_amplitude_radians", self.yaw_amplitude_radians),
            ("pitch_amplitude_radians", self.pitch_amplitude_radians),
            ("roll_amplitude_radians", self.roll_amplitude_radians),
        ] {
            if !value.is_finite() || !(0.0..=std::f32::consts::PI).contains(&value) {
                return Err(MicroMotionProfileError::RotationAmplitudeOutOfRange { name, value });
            }
        }
        Ok(())
    }
}

/// Errors produced while validating a [`MicroMotionProfile`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MicroMotionProfileError {
    /// Transition or period was zero.
    ZeroDuration,
    /// A translation amplitude ratio was negative or non-finite.
    NegativeAmplitude {
        /// Offending field name.
        name: &'static str,
        /// Offending value.
        value: f32,
    },
    /// A rotation amplitude was negative, non-finite, or exceeded PI.
    RotationAmplitudeOutOfRange {
        /// Offending field name.
        name: &'static str,
        /// Offending value.
        value: f32,
    },
}

/// Bounded idle target in semantic units.
///
/// `translation_x`/`translation_y`/`translation_z` are meters in the
/// camera-aligned frame; angles are radians. Roll is always zero under the
/// default profile, and `translation_y` carries the loss-scoped breathing
/// oscillation (ADR-021).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IdleTarget {
    /// Lateral sway in meters (positive toward image right).
    pub translation_x: f32,
    /// Vertical breathing offset in meters (positive up).
    pub translation_y: f32,
    /// Depth sway in meters (positive away from camera).
    pub translation_z: f32,
    /// Yaw offset in radians.
    pub yaw_radians: f32,
    /// Pitch offset in radians.
    pub pitch_radians: f32,
}

impl IdleTarget {
    const ZERO: Self = Self {
        translation_x: 0.0,
        translation_y: 0.0,
        translation_z: 0.0,
        yaw_radians: 0.0,
        pitch_radians: 0.0,
    };

    fn is_finite(self) -> bool {
        self.translation_x.is_finite()
            && self.translation_y.is_finite()
            && self.translation_z.is_finite()
            && self.yaw_radians.is_finite()
            && self.pitch_radians.is_finite()
    }

    fn scaled(self, factor: f32) -> Self {
        Self {
            translation_x: self.translation_x * factor,
            translation_y: self.translation_y * factor,
            translation_z: self.translation_z * factor,
            yaw_radians: self.yaw_radians * factor,
            pitch_radians: self.pitch_radians * factor,
        }
    }
}

/// splitmix64 finalizer: deterministic, dependency-free hash used to derive
/// per-axis bucket noise from `(seed, bucket, axis)`.
fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Deterministic per-axis noise in `[-1, 1]` for one bucket.
fn bucket_noise(seed: u64, bucket: u64, axis: u64) -> f32 {
    let hashed = mix64(seed ^ mix64(bucket.wrapping_mul(0x9e37_79b9).wrapping_add(axis)));
    // Top 53 bits scaled into [-1, 1].
    let unit = ((hashed >> 11) as f32) / ((1u64 << 53) as f32);
    unit * 2.0 - 1.0
}

fn smoothstep(t: f32) -> f32 {
    let t = if t.is_finite() {
        t.clamp(0.0, 1.0)
    } else {
        0.0
    };
    t * t * (3.0 - 2.0 * t)
}

/// Computes the bounded idle target at `elapsed` since tracking was lost.
///
/// Each axis oscillates between deterministic random bucket targets using a
/// smoothstep interpolation, so the output is continuous, bounded by the
/// profile amplitudes scaled by `body_scale_meters`, and free of drift. The
/// trajectory is a pure function of time; onset smoothing (no snap at loss)
/// is provided by [`MicroMotionBlender`] and [`blended_idle_target`]. An
/// invalid profile, non-positive scale, or invalid duration yields zero.
#[must_use]
pub fn idle_target(
    profile: &MicroMotionProfile,
    seed: u64,
    elapsed: Duration,
    body_scale_meters: f32,
) -> IdleTarget {
    if profile.validate().is_err() || !body_scale_meters.is_finite() || body_scale_meters <= 0.0 {
        return IdleTarget::ZERO;
    }
    let elapsed_secs = elapsed.as_secs_f32();
    if !elapsed_secs.is_finite() || elapsed_secs < 0.0 {
        return IdleTarget::ZERO;
    }
    let period_secs = profile.period.as_secs_f32();
    let position = elapsed_secs / period_secs;
    if !position.is_finite() || position < 0.0 {
        return IdleTarget::ZERO;
    }

    // Smoothstep-interpolated noise between consecutive buckets per axis.
    // Buckets are offset per axis so axes do not switch direction together.
    let axis_value = |axis: u64, offset_buckets: u64| -> f32 {
        let shifted = position + offset_buckets as f32 * 0.5;
        let local_bucket = shifted.floor().max(0.0) as u64;
        let local_fraction = smoothstep(shifted - local_bucket as f32);
        let from = bucket_noise(seed, local_bucket.saturating_sub(1), axis);
        let to = bucket_noise(seed, local_bucket, axis);
        from + (to - from) * local_fraction
    };

    let target = IdleTarget {
        translation_x: profile.x_amplitude_ratio * body_scale_meters * axis_value(0, 0),
        translation_y: breath_offset(profile, elapsed_secs, body_scale_meters),
        translation_z: profile.z_amplitude_ratio * body_scale_meters * axis_value(1, 1),
        yaw_radians: profile.yaw_amplitude_radians * axis_value(2, 2),
        pitch_radians: profile.pitch_amplitude_radians * axis_value(3, 3),
    };
    if target.is_finite() {
        target
    } else {
        IdleTarget::ZERO
    }
}

/// Loss-scoped breathing offset in meters at `elapsed_secs` since loss.
///
/// A pure sine of the elapsed time with the profile's breathing period: the
/// motion starts at zero (no discontinuity at loss onset), stays bounded by
/// the configured amplitude, and is deterministic. A non-finite or
/// non-positive intermediate result fails closed to zero.
fn breath_offset(profile: &MicroMotionProfile, elapsed_secs: f32, body_scale_meters: f32) -> f32 {
    let period_secs = profile.breath_period.as_secs_f32();
    if !period_secs.is_finite() || period_secs <= 0.0 {
        return 0.0;
    }
    let phase = elapsed_secs * std::f32::consts::TAU / period_secs;
    if !phase.is_finite() {
        return 0.0;
    }
    let offset = profile.breath_amplitude_ratio * body_scale_meters * phase.sin();
    if offset.is_finite() { offset } else { 0.0 }
}

/// Blend envelope state machine for one loss episode.
///
/// Tracks whether tracking is currently live. After loss the blend follows
/// `smoothstep(elapsed_since_loss / transition)`; after reacquire it follows
/// `1 - smoothstep(elapsed_since_reacquire / transition)`. Both directions
/// are pure functions of capture timestamps, so the envelope is sampling-rate
/// independent and neither direction snaps. Stale state is cleared via
/// [`MicroMotionBlender::reset`] on avatar replacement/unload.
#[derive(Clone, Copy, Debug)]
pub struct MicroMotionBlender {
    transition_secs: f32,
    blend: f32,
    idle_started_at: Option<MonoTimeNs>,
    live_started_at: Option<MonoTimeNs>,
}

impl MicroMotionBlender {
    /// Creates a blender for the given profile.
    ///
    /// # Errors
    ///
    /// Returns [`MicroMotionProfileError::ZeroDuration`] when the transition
    /// is zero; other profile fields are validated by [`idle_target`].
    pub fn new(profile: &MicroMotionProfile) -> Result<Self, MicroMotionProfileError> {
        if profile.transition.is_zero() {
            return Err(MicroMotionProfileError::ZeroDuration);
        }
        Ok(Self {
            transition_secs: profile.transition.as_secs_f32(),
            blend: 0.0,
            idle_started_at: None,
            live_started_at: None,
        })
    }

    /// Current blend factor in `[0, 1]`.
    #[must_use]
    pub const fn blend(&self) -> f32 {
        self.blend
    }

    /// Advances the envelope for one evaluation.
    ///
    /// When `tracked` is true the blend decays toward zero following the
    /// time since reacquire started; when false it rises following the time
    /// since loss started. Losing the face again mid-reacquire restarts the
    /// idle episode at blend zero, which only removes idle motion while the
    /// pose is held — never adds a jump into motion.
    pub fn update(&mut self, tracked: bool, now: MonoTimeNs) -> f32 {
        let transition = self.transition_secs.max(f32::EPSILON);
        if tracked {
            if self.live_started_at.is_none() {
                self.live_started_at = Some(now);
                self.idle_started_at = None;
            }
            let start = self.live_started_at.unwrap_or(now);
            let elapsed = Duration::from_nanos(now.0.saturating_sub(start.0)).as_secs_f32();
            self.blend = 1.0 - smoothstep(elapsed / transition);
        } else {
            if self.idle_started_at.is_none() {
                self.idle_started_at = Some(now);
                self.live_started_at = None;
            }
            let start = self.idle_started_at.unwrap_or(now);
            let elapsed = Duration::from_nanos(now.0.saturating_sub(start.0)).as_secs_f32();
            self.blend = smoothstep(elapsed / transition);
        }
        if !self.blend.is_finite() {
            self.blend = 0.0;
        }
        self.blend.clamp(0.0, 1.0)
    }

    /// Clears all episode state (avatar replacement / unload / reset).
    pub fn reset(&mut self) {
        self.blend = 0.0;
        self.idle_started_at = None;
        self.live_started_at = None;
    }

    /// Whether an idle episode is currently recorded.
    #[must_use]
    pub const fn has_episode(&self) -> bool {
        self.idle_started_at.is_some()
    }

    /// Time since the idle episode started, if one is recorded.
    #[must_use]
    pub fn elapsed_since_idle_start(&self, now: MonoTimeNs) -> Duration {
        match self.idle_started_at {
            Some(start) => Duration::from_nanos(now.0.saturating_sub(start.0)),
            None => Duration::ZERO,
        }
    }
}

/// Composes the blended idle target for one evaluation.
///
/// Returns zero while `blend` is zero; otherwise scales [`idle_target`] by
/// the blend factor so the motion fades in and out continuously. The result
/// is always finite and within the profile bounds scaled by `blend`.
#[must_use]
pub fn blended_idle_target(
    profile: &MicroMotionProfile,
    seed: u64,
    elapsed_since_loss: Duration,
    body_scale_meters: f32,
    blend: f32,
) -> IdleTarget {
    if !blend.is_finite() || blend <= 0.0 {
        return IdleTarget::ZERO;
    }
    let factor = if blend.is_finite() {
        blend.min(1.0)
    } else {
        0.0
    };
    let target = idle_target(profile, seed, elapsed_since_loss, body_scale_meters);
    let scaled = target.scaled(factor);
    if scaled.is_finite() {
        scaled
    } else {
        IdleTarget::ZERO
    }
}

/// Whether a control-frame state counts as "actively tracked".
///
/// Only [`TrackingState::Tracking`] and [`TrackingState::Degraded`] hold a
/// live observation; every other state starts or continues an idle episode.
#[must_use]
pub fn is_tracked_state(state: TrackingState) -> bool {
    matches!(state, TrackingState::Tracking | TrackingState::Degraded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn profile() -> MicroMotionProfile {
        MicroMotionProfile::default()
    }

    fn seconds(value: f32) -> Duration {
        Duration::from_secs_f32(value)
    }

    #[test]
    fn default_profile_matches_analysis_seeds_and_validates() {
        let p = profile();
        assert_eq!(p.validate(), Ok(()));
        assert_eq!(p.transition, Duration::from_millis(4_000));
        assert_eq!(p.period, Duration::from_millis(2_500));
        assert_relative_eq!(p.x_amplitude_ratio, 0.015);
        assert_relative_eq!(p.z_amplitude_ratio, 0.030);
        assert_relative_eq!(p.yaw_amplitude_radians, 4.0f32.to_radians(), epsilon = 1e-6);
        assert_relative_eq!(
            p.pitch_amplitude_radians,
            4.0f32.to_radians(),
            epsilon = 1e-6
        );
        assert_eq!(p.roll_amplitude_radians, 0.0);
        assert_eq!(p.breath_period, Duration::from_millis(4_000));
        assert_relative_eq!(p.breath_amplitude_ratio, 0.010);
    }

    #[test]
    fn profile_rejects_invalid_fields() {
        let mut bad = profile();
        bad.transition = Duration::ZERO;
        assert_eq!(bad.validate(), Err(MicroMotionProfileError::ZeroDuration));

        let mut bad = profile();
        bad.period = Duration::ZERO;
        assert_eq!(bad.validate(), Err(MicroMotionProfileError::ZeroDuration));

        let mut bad = profile();
        bad.breath_period = Duration::ZERO;
        assert_eq!(bad.validate(), Err(MicroMotionProfileError::ZeroDuration));

        let mut bad = profile();
        bad.x_amplitude_ratio = -0.01;
        assert_eq!(
            bad.validate(),
            Err(MicroMotionProfileError::NegativeAmplitude {
                name: "x_amplitude_ratio",
                value: -0.01,
            })
        );

        let mut bad = profile();
        bad.breath_amplitude_ratio = -0.01;
        assert_eq!(
            bad.validate(),
            Err(MicroMotionProfileError::NegativeAmplitude {
                name: "breath_amplitude_ratio",
                value: -0.01,
            })
        );

        let mut bad = profile();
        bad.yaw_amplitude_radians = std::f32::consts::PI + 0.1;
        assert!(matches!(
            bad.validate(),
            Err(MicroMotionProfileError::RotationAmplitudeOutOfRange { .. })
        ));
    }

    #[test]
    fn breathing_starts_at_zero_and_oscillates_with_the_period() {
        let p = profile();
        let amplitude = p.breath_amplitude_ratio * 0.7;

        // No discontinuity at loss onset.
        assert_relative_eq!(
            idle_target(&p, 42, seconds(0.0), 0.7).translation_y,
            0.0,
            epsilon = 1e-6
        );

        // A quarter period reaches the inhale peak, half a period crosses
        // back through zero, and three quarters reaches the exhale trough.
        let quarter = idle_target(&p, 42, seconds(1.0), 0.7).translation_y;
        assert_relative_eq!(quarter, amplitude, epsilon = 1e-5);
        let half = idle_target(&p, 42, seconds(2.0), 0.7).translation_y;
        assert_relative_eq!(half, 0.0, epsilon = 1e-5);
        let three_quarters = idle_target(&p, 42, seconds(3.0), 0.7).translation_y;
        assert_relative_eq!(three_quarters, -amplitude, epsilon = 1e-5);
    }

    #[test]
    fn breathing_stays_bounded_across_long_episodes() {
        let p = profile();
        let bound = p.breath_amplitude_ratio * 0.7;
        let mut t = 0.0_f32;
        while t <= 600.0 {
            let target = idle_target(&p, 7, seconds(t), 0.7);
            assert!(target.translation_y.is_finite(), "t={t}");
            assert!(
                target.translation_y.abs() <= bound + 1e-6,
                "breathing exceeded the bound at t={t}: {}",
                target.translation_y
            );
            t += 0.25;
        }
    }

    #[test]
    fn blended_composition_prevents_any_snap_after_loss() {
        let p = profile();
        // At the instant of loss, with blend zero, nothing moves.
        assert_eq!(
            blended_idle_target(&p, 42, seconds(0.0), 0.7, 0.0),
            IdleTarget::ZERO
        );
        // Early into the transition the composed motion is still tiny.
        let early = blended_idle_target(&p, 42, seconds(0.05), 0.7, 0.0125);
        assert!(early.translation_x.abs() < 0.001);
    }

    #[test]
    fn trajectory_is_bounded_immediately_and_stays_finite() {
        let p = profile();
        let scale = 0.7;
        for t in [0.0f32, 0.01, 0.1, 1.0] {
            let target = idle_target(&p, 42, seconds(t), scale);
            assert!(target.is_finite());
            assert!(target.translation_x.abs() <= p.x_amplitude_ratio * scale + 1e-6);
        }
    }

    #[test]
    fn bounds_hold_across_long_episodes() {
        let p = profile();
        let scale = 0.7;
        let max_x = p.x_amplitude_ratio * scale;
        let max_z = p.z_amplitude_ratio * scale;
        let mut t = 0.0;
        while t <= 60.0 {
            let target = idle_target(&p, 7, seconds(t), scale);
            assert!(target.translation_x.abs() <= max_x + 1e-5, "t={t}");
            assert!(target.translation_z.abs() <= max_z + 1e-5, "t={t}");
            assert!(
                target.yaw_radians.abs() <= p.yaw_amplitude_radians + 1e-6,
                "t={t}"
            );
            assert!(
                target.pitch_radians.abs() <= p.pitch_amplitude_radians + 1e-6,
                "t={t}"
            );
            t += 0.05;
        }
    }

    #[test]
    fn roll_is_always_zero_under_default_profile() {
        let p = profile();
        let mut t = 0.0;
        while t <= 12.0 {
            let _ = t;
            // IdleTarget carries no roll field at all; this test documents
            // that contract. Compile-time guarantee instead of runtime check.
            let target = idle_target(&p, 3, seconds(t), 1.0);
            assert!(target.is_finite());
            t += 0.25;
        }
    }

    #[test]
    fn same_seed_and_time_are_deterministic() {
        let p = profile();
        let a = idle_target(&p, 99, seconds(3.7), 0.8);
        let b = idle_target(&p, 99, seconds(3.7), 0.8);
        assert_eq!(a, b);

        // Different seeds diverge.
        let c = idle_target(&p, 100, seconds(3.7), 0.8);
        assert_ne!(a, c);
    }

    #[test]
    fn fps_sampling_produces_near_identical_trajectories() {
        let p = profile();
        let sample = |fps: f32| -> Vec<IdleTarget> {
            let step = 1.0 / fps;
            let mut out = Vec::new();
            let mut t = 0.0;
            while t <= 8.0 {
                out.push(idle_target(&p, 11, seconds(t), 0.7));
                t += step;
            }
            out
        };
        let at_30 = sample(30.0);
        let at_60 = sample(60.0);
        let at_120 = sample(120.0);

        for index in [10usize, 45, 90, 150] {
            let reference = &at_30[index];
            let close = |other: &IdleTarget| {
                (other.translation_x - reference.translation_x).abs() < 1e-3
                    && (other.translation_z - reference.translation_z).abs() < 1e-3
                    && (other.yaw_radians - reference.yaw_radians).abs() < 1e-3
            };
            // Time-aligned samples: index i at 30 FPS equals instant i/30 s,
            // which is index 2i at 60 FPS and 4i at 120 FPS.
            assert!(close(&at_60[index * 2]));
            assert!(close(&at_120[index * 4]));
        }
    }

    #[test]
    fn invalid_inputs_fail_closed() {
        let p = profile();
        let mut invalid = profile();
        invalid.transition = Duration::ZERO;
        assert_eq!(
            idle_target(&invalid, 1, seconds(1.0), 1.0),
            IdleTarget::ZERO
        );
        assert_eq!(idle_target(&p, 1, seconds(1.0), 0.0), IdleTarget::ZERO);
        assert_eq!(idle_target(&p, 1, seconds(1.0), f32::NAN), IdleTarget::ZERO);
        assert_eq!(idle_target(&p, 1, seconds(1.0), -1.0), IdleTarget::ZERO);
    }

    #[test]
    fn blender_ramps_up_after_loss_without_snapping() {
        let p = profile();
        let mut blender = MicroMotionBlender::new(&p).unwrap();

        let b0 = blender.update(false, MonoTimeNs(0));
        assert_eq!(b0, 0.0);
        assert!(blender.has_episode());

        // Half-way through the 4 s transition the blend is ~0.5 (smoothstep).
        let mid = blender.update(false, MonoTimeNs(2_000_000_000));
        assert!((mid - 0.5).abs() < 1e-3, "{mid}");

        // Beyond the transition it saturates at one.
        let full = blender.update(false, MonoTimeNs(6_000_000_000));
        assert!((full - 1.0).abs() < 1e-3, "{full}");
    }

    #[test]
    fn blender_decays_on_reacquire_instead_of_snapping_back() {
        let p = profile();
        let mut blender = MicroMotionBlender::new(&p).unwrap();
        let _ = blender.update(false, MonoTimeNs(0));
        let _ = blender.update(false, MonoTimeNs(6_000_000_000));
        assert!((blender.blend() - 1.0).abs() < 1e-3);

        // Reacquire: decay is gradual over the transition, not instant.
        let first = blender.update(true, MonoTimeNs(6_100_000_000));
        assert!(first <= 1.0 && first > 0.9, "{first}");
        let mut now = 6_100_000_000_u64;
        loop {
            now += 250_000_000;
            let blend = blender.update(true, MonoTimeNs(now));
            if blend == 0.0 {
                break;
            }
            assert!(now < 6_100_000_000 + 5_000_000_000, "decay took too long");
        }
        // Live state is recorded once fully reacquired.
        assert_eq!(blender.update(true, MonoTimeNs(now + 1)), 0.0);
    }

    #[test]
    fn blender_reset_clears_stale_episode_state() {
        let p = profile();
        let mut blender = MicroMotionBlender::new(&p).unwrap();
        let _ = blender.update(false, MonoTimeNs(0));
        let _ = blender.update(false, MonoTimeNs(5_000_000_000));
        assert!(blender.has_episode());

        blender.reset();
        assert!(!blender.has_episode());
        assert_eq!(blender.blend(), 0.0);

        // A brand-new loss episode restarts the ramp from zero.
        assert_eq!(blender.update(false, MonoTimeNs(9_999_999_999)), 0.0);
    }

    #[test]
    fn blender_rejects_zero_transition() {
        let mut p = profile();
        p.transition = Duration::ZERO;
        assert!(matches!(
            MicroMotionBlender::new(&p),
            Err(MicroMotionProfileError::ZeroDuration)
        ));
    }

    #[test]
    fn composition_scales_the_idle_target_by_blend() {
        let p = profile();
        let base = idle_target(&p, 5, seconds(5.0), 1.0);
        let composed = blended_idle_target(&p, 5, seconds(5.0), 1.0, 0.5);
        assert_relative_eq!(
            composed.translation_x,
            base.translation_x * 0.5,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            composed.translation_y,
            base.translation_y * 0.5,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            composed.translation_z,
            base.translation_z * 0.5,
            epsilon = 1e-6
        );
        assert_relative_eq!(composed.yaw_radians, base.yaw_radians * 0.5, epsilon = 1e-6);
        assert_relative_eq!(
            composed.pitch_radians,
            base.pitch_radians * 0.5,
            epsilon = 1e-6
        );

        assert_eq!(
            blended_idle_target(&p, 5, seconds(5.0), 1.0, 0.0),
            IdleTarget::ZERO
        );
        assert_eq!(
            blended_idle_target(&p, 5, seconds(5.0), 1.0, f32::NAN),
            IdleTarget::ZERO
        );
    }

    #[test]
    fn only_tracked_states_count_as_live() {
        assert!(is_tracked_state(TrackingState::Tracking));
        assert!(is_tracked_state(TrackingState::Degraded));
        for state in [
            TrackingState::Starting,
            TrackingState::Searching,
            TrackingState::Acquiring,
            TrackingState::LostHold,
            TrackingState::ReturningNeutral,
        ] {
            assert!(!is_tracked_state(state), "{state:?}");
        }
    }

    #[test]
    fn bucket_noise_is_bounded_and_deterministic() {
        for bucket in 0..64_u64 {
            for axis in 0..4_u64 {
                let value = bucket_noise(123, bucket, axis);
                assert!((-1.0..=1.0).contains(&value));
                assert_relative_eq!(value, bucket_noise(123, bucket, axis));
            }
        }
        assert_ne!(bucket_noise(123, 1, 0), bucket_noise(123, 2, 0));
        assert_ne!(bucket_noise(123, 1, 0), bucket_noise(123, 1, 1));
    }
}
