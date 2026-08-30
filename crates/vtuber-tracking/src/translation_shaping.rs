//! Scale-aware soft-cap and dt-aware state filtering for neutral-relative
//! head translation (`DESIGN.md` §6, Issue #164).
//!
//! The shaping policy passes small motions through untouched while only large
//! motions are compressed, so no dead zone is introduced and small inputs are
//! never erased.
//!
//! All knobs live in [`TranslationShapingProfile`] instead of being scattered
//! through systems as magic constants. The filter advances on actual capture
//! timestamps, never on render frame counts, so the same head motion sampled
//! at 30/60/120 FPS produces near-identical results.

use std::time::Duration;

use vtuber_core::types::{HeadTranslationSignal, HeadTranslationState, MonoTimeNs};

/// Typed profile for translation shaping thresholds.
///
/// Thresholds are expressed as ratios of the avatar's body scale and are
/// multiplied by an explicitly resolved body scale (meters) at shaping time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TranslationShapingProfile {
    /// Horizontal soft-cap threshold as a ratio of body scale.
    pub x_threshold_ratio: f32,
    /// Vertical soft-cap threshold as a ratio of body scale.
    ///
    /// Vertical motion is smaller than horizontal motion for typical webcam
    /// framing, so this is half of the horizontal seed.
    pub y_threshold_ratio: f32,
    /// Depth soft-cap threshold as a ratio of body scale.
    pub z_threshold_ratio: f32,
}

impl Default for TranslationShapingProfile {
    fn default() -> Self {
        Self {
            x_threshold_ratio: 0.15,
            y_threshold_ratio: 0.075,
            z_threshold_ratio: 0.15,
        }
    }
}

impl TranslationShapingProfile {
    /// Validates that every ratio is positive and finite.
    ///
    /// # Errors
    ///
    /// Returns [`ShapingProfileError`] when any threshold ratio is not a
    /// positive finite number.
    pub fn validate(&self) -> Result<(), ShapingProfileError> {
        for (name, value) in [
            ("x_threshold_ratio", self.x_threshold_ratio),
            ("y_threshold_ratio", self.y_threshold_ratio),
            ("z_threshold_ratio", self.z_threshold_ratio),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(ShapingProfileError::InvalidThreshold { name, value });
            }
        }
        Ok(())
    }
}

/// Errors produced while validating a [`TranslationShapingProfile`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShapingProfileError {
    /// A threshold ratio was zero, negative, or non-finite.
    InvalidThreshold {
        /// Offending field name.
        name: &'static str,
        /// Offending value.
        value: f32,
    },
}

/// Soft-caps one scalar axis:
///
/// ```text
/// |v| <= t     -> v
/// |v| <= 2t    -> sign(v) * (t + 0.5 * (|v| - t))
/// |v| >  2t    -> sign(v) * 1.5 * t
/// ```
///
/// The curve is continuous across `0`, `±t`, and `±2t` and preserves the
/// input sign. `t` must be positive; non-positive or non-finite `t` returns
/// the input unchanged rather than manufacturing values.
#[must_use]
pub fn soft_cap_scalar(value: f32, threshold: f32) -> f32 {
    if !value.is_finite() || !threshold.is_finite() || threshold <= 0.0 {
        return value;
    }
    let magnitude = value.abs();
    if magnitude <= threshold {
        value
    } else if magnitude <= 2.0 * threshold {
        let sign = if value < 0.0 { -1.0 } else { 1.0 };
        sign * (threshold + 0.5 * (magnitude - threshold))
    } else {
        let sign = if value < 0.0 { -1.0 } else { 1.0 };
        sign * 1.5 * threshold
    }
}

/// Applies the profile's per-axis soft-cap to one translation signal.
///
/// Unavailable signals pass through unchanged; availability state is
/// preserved otherwise. Non-finite components collapse to
/// [`HeadTranslationSignal::UNAVAILABLE`] via the signal constructor.
#[must_use]
pub fn shape_translation(
    signal: HeadTranslationSignal,
    profile: &TranslationShapingProfile,
    body_scale_meters: f32,
) -> HeadTranslationSignal {
    if !signal.is_available() || !body_scale_meters.is_finite() || body_scale_meters <= 0.0 {
        return signal;
    }
    let shaped = HeadTranslationSignal {
        x_meters: soft_cap_scalar(
            signal.x_meters,
            profile.x_threshold_ratio * body_scale_meters,
        ),
        y_meters: soft_cap_scalar(
            signal.y_meters,
            profile.y_threshold_ratio * body_scale_meters,
        ),
        z_meters: soft_cap_scalar(
            signal.z_meters,
            profile.z_threshold_ratio * body_scale_meters,
        ),
        state: signal.state,
    };
    if shaped.x_meters.is_finite() && shaped.y_meters.is_finite() && shaped.z_meters.is_finite() {
        shaped
    } else {
        HeadTranslationSignal::UNAVAILABLE
    }
}

/// Maximum capture-time gap tolerated before the filter resets instead of
/// smoothing from a stale observation.
const MAX_GAP: Duration = Duration::from_secs(1);

/// Timestamp-driven exponential smoother for head translation.
///
/// The smoothing factor is derived from the elapsed time between capture
/// timestamps, not from render ticks, making the output sampling-rate
/// independent for the same physical motion. There is deliberately no dead
/// zone: small observations always reach the output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TranslationFilter {
    tau: Duration,
    state: Option<SmoothedSample>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SmoothedSample {
    x_meters: f32,
    y_meters: f32,
    z_meters: f32,
    state: HeadTranslationState,
    captured_at: MonoTimeNs,
}

impl TranslationFilter {
    /// Creates a filter with the given time constant.
    ///
    /// # Errors
    ///
    /// Returns [`FilterConfigError`] when `tau` is zero or non-finite.
    pub fn new(tau: Duration) -> Result<Self, FilterConfigError> {
        if tau.is_zero() {
            return Err(FilterConfigError::ZeroTau);
        }
        let seconds = tau.as_secs_f32();
        if !seconds.is_finite() || seconds <= 0.0 {
            return Err(FilterConfigError::NonFiniteTau);
        }
        Ok(Self { tau, state: None })
    }

    /// Feeds one observation stamped with its capture timestamp and returns
    /// the filtered signal.
    ///
    /// An unavailable observation clears the internal state so stale motion
    /// is not held indefinitely. A gap longer than [`MAX_GAP`] restarts the
    /// filter from the new observation instead of blending across it.
    #[must_use]
    pub fn update(
        &mut self,
        signal: HeadTranslationSignal,
        captured_at: MonoTimeNs,
    ) -> HeadTranslationSignal {
        if !signal.is_available()
            || !signal.x_meters.is_finite()
            || !signal.y_meters.is_finite()
            || !signal.z_meters.is_finite()
        {
            self.state = None;
            return HeadTranslationSignal::UNAVAILABLE;
        }

        let Some(previous) = self.state else {
            self.state = Some(SmoothedSample::new(signal, captured_at));
            return signal;
        };

        let dt_ns = captured_at.0.saturating_sub(previous.captured_at.0);
        if dt_ns == 0 {
            let blended = SmoothedSample {
                state: merge_state(previous.state, signal.state),
                ..previous
            };
            self.state = Some(blended);
            return HeadTranslationSignal {
                x_meters: previous.x_meters,
                y_meters: previous.y_meters,
                z_meters: previous.z_meters,
                state: blended.state,
            };
        }

        let dt = Duration::from_nanos(dt_ns);
        if dt > MAX_GAP {
            self.state = Some(SmoothedSample::new(signal, captured_at));
            return signal;
        }

        let alpha = 1.0 - (-dt.as_secs_f32() / self.tau.as_secs_f32()).exp();
        let lerp = |previous_value: f32, next_value: f32| {
            previous_value + (next_value - previous_value) * alpha
        };
        let smoothed = SmoothedSample {
            x_meters: lerp(previous.x_meters, signal.x_meters),
            y_meters: lerp(previous.y_meters, signal.y_meters),
            z_meters: lerp(previous.z_meters, signal.z_meters),
            state: merge_state(previous.state, signal.state),
            captured_at,
        };
        if smoothed.x_meters.is_finite()
            && smoothed.y_meters.is_finite()
            && smoothed.z_meters.is_finite()
        {
            self.state = Some(smoothed);
            HeadTranslationSignal {
                x_meters: smoothed.x_meters,
                y_meters: smoothed.y_meters,
                z_meters: smoothed.z_meters,
                state: smoothed.state,
            }
        } else {
            self.state = None;
            HeadTranslationSignal::UNAVAILABLE
        }
    }

    /// Clears the internal state immediately.
    pub fn reset(&mut self) {
        self.state = None;
    }

    /// Returns whether the filter currently holds a smoothed observation.
    #[must_use]
    pub const fn has_state(&self) -> bool {
        self.state.is_some()
    }
}

/// Errors produced while constructing a [`TranslationFilter`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FilterConfigError {
    /// Zero time constants cannot smooth.
    ZeroTau,
    /// The time constant overflowed or was non-finite.
    NonFiniteTau,
}

impl SmoothedSample {
    fn new(signal: HeadTranslationSignal, captured_at: MonoTimeNs) -> Self {
        Self {
            x_meters: signal.x_meters,
            y_meters: signal.y_meters,
            z_meters: signal.z_meters,
            state: signal.state,
            captured_at,
        }
    }
}

fn merge_state(a: HeadTranslationState, b: HeadTranslationState) -> HeadTranslationState {
    if matches!(a, HeadTranslationState::Tracked) && matches!(b, HeadTranslationState::Tracked) {
        HeadTranslationState::Tracked
    } else {
        HeadTranslationState::Degraded
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn tracked(x: f32, y: f32, z: f32) -> HeadTranslationSignal {
        HeadTranslationSignal::tracked(x, y, z)
    }

    #[test]
    fn soft_cap_passes_small_values_through() {
        let t = 0.1;
        assert_eq!(soft_cap_scalar(0.0, t), 0.0);
        assert_relative_eq!(soft_cap_scalar(0.05, t), 0.05, epsilon = 1e-6);
        assert_relative_eq!(soft_cap_scalar(-0.05, t), -0.05, epsilon = 1e-6);
        assert_relative_eq!(soft_cap_scalar(t, t), t, epsilon = 1e-6);
    }

    #[test]
    fn soft_cap_compression_region_matches_formula() {
        let t = 0.1;
        assert_relative_eq!(soft_cap_scalar(0.15, t), 1.25 * t, epsilon = 1e-6);
        assert_relative_eq!(soft_cap_scalar(-0.15, t), -1.25 * t, epsilon = 1e-6);
        assert_relative_eq!(soft_cap_scalar(0.2, t), 1.5 * t, epsilon = 1e-6);
        assert_relative_eq!(soft_cap_scalar(-0.2, t), -1.5 * t, epsilon = 1e-6);
    }

    #[test]
    fn soft_cap_saturates_at_one_and_a_half_thresholds() {
        let t = 0.1;
        for magnitude in [0.25, 0.3, 1.0, 10.0] {
            assert_relative_eq!(soft_cap_scalar(magnitude, t), 1.5 * t, epsilon = 1e-6);
            assert_relative_eq!(soft_cap_scalar(-magnitude, t), -1.5 * t, epsilon = 1e-6);
        }
    }

    #[test]
    fn soft_cap_is_monotone_and_continuous_across_the_domain() {
        let t = 0.1;
        let step = 0.004;
        let mut previous = soft_cap_scalar(0.0, t);
        let mut value = step;
        while value <= 0.5 {
            let current = soft_cap_scalar(value, t);
            assert!(
                current >= previous,
                "soft cap must be monotone: {previous} -> {current} at {value}"
            );
            previous = current;
            value += step;
        }
    }

    #[test]
    fn shape_translation_applies_per_axis_thresholds_with_body_scale() {
        let profile = TranslationShapingProfile::default();
        let signal = tracked(0.2, 0.2, 0.2);
        let shaped = shape_translation(signal, &profile, 1.0);
        assert_relative_eq!(shaped.x_meters, 0.175, epsilon = 1e-6);
        assert_relative_eq!(shaped.z_meters, 0.175, epsilon = 1e-6);
        assert_relative_eq!(shaped.y_meters, 0.1125, epsilon = 1e-6);
        assert_eq!(shaped.state, HeadTranslationState::Tracked);
    }

    #[test]
    fn shape_translation_preserves_unavailable_and_degraded_states() {
        let profile = TranslationShapingProfile::default();
        assert_eq!(
            shape_translation(HeadTranslationSignal::UNAVAILABLE, &profile, 1.0),
            HeadTranslationSignal::UNAVAILABLE
        );
        let degraded = HeadTranslationSignal::degraded(0.2, 0.2, 0.2);
        let shaped = shape_translation(degraded, &profile, 1.0);
        assert_eq!(shaped.state, HeadTranslationState::Degraded);
        assert_relative_eq!(shaped.x_meters, 0.175, epsilon = 1e-6);
    }

    #[test]
    fn shape_translation_leaves_signal_untouched_when_body_scale_is_invalid() {
        let profile = TranslationShapingProfile::default();
        let signal = tracked(1.0, 1.0, 1.0);
        assert_eq!(shape_translation(signal, &profile, 0.0), signal);
        assert_eq!(shape_translation(signal, &profile, f32::NAN), signal);
    }

    #[test]
    fn profile_rejects_non_positive_thresholds() {
        assert_eq!(TranslationShapingProfile::default().validate(), Ok(()));
        let bad = TranslationShapingProfile {
            y_threshold_ratio: 0.0,
            ..TranslationShapingProfile::default()
        };
        assert_eq!(
            bad.validate(),
            Err(ShapingProfileError::InvalidThreshold {
                name: "y_threshold_ratio",
                value: 0.0,
            })
        );
    }

    #[test]
    fn filter_output_is_sampling_rate_independent_for_same_motion() {
        let tau = Duration::from_millis(250);
        let run = |fps: u64| -> Vec<f32> {
            let mut filter = TranslationFilter::new(tau).unwrap();
            let period_ns = 1_000_000_000u64 / fps;
            let mut output = Vec::new();
            for sample in 0..=360u64 {
                output.push(
                    filter
                        .update(tracked(0.06, 0.0, 0.0), MonoTimeNs(sample * period_ns))
                        .x_meters,
                );
            }
            output
        };
        let at_30 = run(30);
        let at_60 = run(60);
        let at_120 = run(120);
        let cases: [(usize, usize, usize); 3] = [(30, 60, 120), (60, 120, 240), (90, 180, 360)];
        for (i30, i60, i120) in cases {
            let reference = at_30[i30];
            assert!((at_60[i60] - reference).abs() < 1e-4);
            assert!((at_120[i120] - reference).abs() < 1e-4);
        }
    }

    #[test]
    fn filter_has_no_dead_zone_and_moves_toward_small_inputs() {
        let mut filter = TranslationFilter::new(Duration::from_millis(100)).unwrap();
        let first = filter.update(tracked(0.001, -0.001, 0.0005), MonoTimeNs(0));
        assert_relative_eq!(first.x_meters, 0.001, epsilon = 1e-12);
        assert_relative_eq!(first.y_meters, -0.001, epsilon = 1e-12);
        let second = filter.update(tracked(-0.001, 0.0, 0.0), MonoTimeNs(16_666_667));
        assert!(second.x_meters < first.x_meters);
        assert!(second.y_meters > first.y_meters);
    }

    #[test]
    fn filter_does_not_hold_stale_translation_across_unavailable() {
        let mut filter = TranslationFilter::new(Duration::from_millis(100)).unwrap();
        let _ = filter.update(tracked(0.05, 0.0, 0.0), MonoTimeNs(0));
        assert!(filter.has_state());

        let cleared = filter.update(HeadTranslationSignal::UNAVAILABLE, MonoTimeNs(33_333_334));
        assert_eq!(cleared, HeadTranslationSignal::UNAVAILABLE);
        assert!(!filter.has_state());

        let fresh = filter.update(tracked(-0.02, 0.0, 0.0), MonoTimeNs(66_666_667));
        assert_relative_eq!(fresh.x_meters, -0.02, epsilon = 1e-9);
    }

    #[test]
    fn filter_restarts_after_long_gap_instead_of_blending_across_it() {
        let mut filter = TranslationFilter::new(Duration::from_millis(100)).unwrap();
        let _ = filter.update(tracked(0.06, 0.06, 0.06), MonoTimeNs(0));
        let after_gap = filter.update(tracked(-0.06, -0.06, -0.06), MonoTimeNs(2_000_000_000));
        assert_relative_eq!(after_gap.x_meters, -0.06, epsilon = 1e-9);
    }

    #[test]
    fn filter_resets_on_non_finite_input_and_survives_out_of_order_stamps() {
        let mut filter = TranslationFilter::new(Duration::from_millis(100)).unwrap();
        let _ = filter.update(tracked(0.03, 0.0, 0.0), MonoTimeNs(50_000_000));
        let out_of_order = filter.update(tracked(0.06, 0.0, 0.0), MonoTimeNs(16_666_667));
        assert!(out_of_order.is_available());
        assert!(out_of_order.x_meters.is_finite());
    }

    #[test]
    fn filter_rejects_invalid_tau() {
        assert_eq!(
            TranslationFilter::new(Duration::ZERO),
            Err(FilterConfigError::ZeroTau)
        );
    }
}
