//! Blink and mouth expression normalization / smoothing filter.
//!
//! Converts raw per-frame expression coefficients produced by inference into
//! calibrated [`ExpressionCoefficients`] that the avatar adapter can apply.
//! The filter owns three responsibilities in one place:
//!
//! 1. **Calibration normalization**: map raw eye/mouth signals to `[0, 1]`
//!    using a neutral baseline and closed/open calibration ranges.  Blink is
//!    handled per-eye; mouth openness is mapped to the `aa` coefficient.
//! 2. **Missing-channel fallback**: mark low-confidence or non-finite raw
//!    channels as missing and fall back according to a per-channel policy.
//! 3. **Smoothing**: apply per-channel exponential smoothing with separate
//!    attack and release time constants, plus a dead zone around the neutral
//!    mouth position.
//!
//! VRM preset mapping (e.g. `blinkLeft`/`blinkRight` versus `blink`, or the
//! five vowel coarticulation) is intentionally out of scope for this module;
//! it belongs in `vtuber-avatar`.

use std::error::Error;
use std::fmt;

use vtuber_core::types::{ExpressionCoefficients, MonoTimeNs, RawExpressionObservation};

/// Identifies an expression channel for error reporting and fallback policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExpressionChannel {
    /// Left eye blink.
    BlinkLeft,
    /// Right eye blink.
    BlinkRight,
    /// Mouth openness.
    Mouth,
}

impl fmt::Display for ExpressionChannel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BlinkLeft => write!(f, "blinkLeft"),
            Self::BlinkRight => write!(f, "blinkRight"),
            Self::Mouth => write!(f, "mouth"),
        }
    }
}

/// Errors that can occur while constructing expression calibration ranges.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExpressionCalibrationError {
    /// `min` is not strictly less than `max`.
    InvertedRange {
        /// Channel that failed validation.
        channel: ExpressionChannel,
        /// Supplied minimum value.
        min: f32,
        /// Supplied maximum value.
        max: f32,
    },
    /// `min` equals `max`, giving no usable span.
    ZeroSpan {
        /// Channel that failed validation.
        channel: ExpressionChannel,
        /// The duplicated value.
        value: f32,
    },
    /// A calibration value is non-finite or outside `[0, 1]`.
    ValueOutOfRange {
        /// Channel that failed validation.
        channel: ExpressionChannel,
        /// Offending value.
        value: f32,
    },
}

impl fmt::Display for ExpressionCalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvertedRange { channel, min, max } => {
                write!(
                    f,
                    "{channel} calibration range is inverted: min={min}, max={max}"
                )
            }
            Self::ZeroSpan { channel, value } => {
                write!(f, "{channel} calibration range has zero span at {value}")
            }
            Self::ValueOutOfRange { channel, value } => {
                write!(
                    f,
                    "{channel} calibration value {value} is not finite or outside [0, 1]"
                )
            }
        }
    }
}

impl Error for ExpressionCalibrationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

impl ExpressionCalibrationError {
    /// Stable string code for logging and UI mapping.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::InvertedRange { .. } => "EXPRESSION_CALIBRATION_INVERTED_RANGE",
            Self::ZeroSpan { .. } => "EXPRESSION_CALIBRATION_ZERO_SPAN",
            Self::ValueOutOfRange { .. } => "EXPRESSION_CALIBRATION_VALUE_OUT_OF_RANGE",
        }
    }
}

/// A validated min..max calibration range for one expression channel.
///
/// The raw coefficient domain is `[0, 1]`.  `min` maps to a normalized value
/// of `0.0` and `max` maps to `1.0`; values outside the range are clamped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpressionRange {
    /// Value that maps to a normalized coefficient of `0.0`.
    min: f32,
    /// Value that maps to a normalized coefficient of `1.0`.
    max: f32,
    /// Neutral baseline observed during calibration.
    neutral: f32,
}

impl ExpressionRange {
    /// Creates a range from a validated min/max pair.
    ///
    /// # Errors
    ///
    /// Returns [`ExpressionCalibrationError`] if any argument is non-finite,
    /// outside `[0, 1]`, if `min >= max`, or if `neutral` is not between
    /// `min` and `max` inclusive.
    pub fn new(
        channel: ExpressionChannel,
        min: f32,
        max: f32,
        neutral: f32,
    ) -> Result<Self, ExpressionCalibrationError> {
        Self::validate_value(channel, min)?;
        Self::validate_value(channel, max)?;
        Self::validate_value(channel, neutral)?;
        if min == max {
            return Err(ExpressionCalibrationError::ZeroSpan {
                channel,
                value: min,
            });
        }
        if min > max {
            return Err(ExpressionCalibrationError::InvertedRange { channel, min, max });
        }
        if neutral < min || neutral > max {
            return Err(ExpressionCalibrationError::ValueOutOfRange {
                channel,
                value: neutral,
            });
        }
        Ok(Self { min, max, neutral })
    }

    /// Convenience constructor for a blink channel.
    ///
    /// * `open` is the raw value when the eye is fully open (maps to `0.0`).
    /// * `closed` is the raw value when the eye is fully closed (maps to `1.0`).
    /// * `neutral` is the typical relaxed value.
    ///
    /// # Errors
    ///
    /// Returns [`ExpressionCalibrationError`] if `open >= closed` or if any
    /// value is outside `[0, 1]`.
    pub fn for_blink(
        neutral: f32,
        open: f32,
        closed: f32,
    ) -> Result<Self, ExpressionCalibrationError> {
        Self::new(ExpressionChannel::BlinkLeft, open, closed, neutral)
    }

    /// Convenience constructor for the mouth channel.
    ///
    /// * `neutral` is the raw value when the mouth is closed/neutral.
    /// * `open` is the raw value when the mouth is fully open.
    ///
    /// # Errors
    ///
    /// Returns [`ExpressionCalibrationError`] if `neutral >= open` or if any
    /// value is outside `[0, 1]`.
    pub fn for_mouth(neutral: f32, open: f32) -> Result<Self, ExpressionCalibrationError> {
        Self::new(ExpressionChannel::Mouth, neutral, open, neutral)
    }

    /// Normalizes a raw value to `[0, 1]` against this range.
    ///
    /// The result is always finite and clamped.
    #[must_use]
    pub fn normalize(&self, raw: f32) -> f32 {
        // `max != min` is guaranteed by construction.
        ((raw - self.min) / (self.max - self.min)).clamp(0.0, 1.0)
    }

    /// Returns the neutral baseline normalized to `[0, 1]`.
    #[must_use]
    pub fn normalized_neutral(&self) -> f32 {
        self.normalize(self.neutral)
    }

    fn validate_value(
        channel: ExpressionChannel,
        value: f32,
    ) -> Result<(), ExpressionCalibrationError> {
        if value.is_finite() && (0.0..=1.0).contains(&value) {
            Ok(())
        } else {
            Err(ExpressionCalibrationError::ValueOutOfRange { channel, value })
        }
    }
}

/// What to do when a raw expression channel is missing or unusable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum MissingChannelFallback {
    /// Output `0.0` for this channel.
    #[default]
    Zero,
    /// Use the value from the opposite eye for blink channels.
    ///
    /// This has no effect on the mouth channel.
    MirrorOpposite,
}

/// Per-channel fallback policy for missing raw observations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MissingChannelPolicy {
    /// Fallback for the left blink channel.
    pub blink_left: MissingChannelFallback,
    /// Fallback for the right blink channel.
    pub blink_right: MissingChannelFallback,
    /// Fallback for the mouth channel.
    pub mouth: MissingChannelFallback,
}

impl MissingChannelPolicy {
    /// A symmetric policy: each eye mirrors the other, and the mouth falls
    /// back to zero.
    #[must_use]
    pub fn symmetric() -> Self {
        Self {
            blink_left: MissingChannelFallback::MirrorOpposite,
            blink_right: MissingChannelFallback::MirrorOpposite,
            mouth: MissingChannelFallback::Zero,
        }
    }
}

impl Default for MissingChannelPolicy {
    fn default() -> Self {
        Self::symmetric()
    }
}

/// Parameters controlling expression normalization and smoothing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpressionFilterParams {
    /// Dead zone applied around the neutral mouth position.
    ///
    /// Normalized mouth values below this threshold are forced to `0.0`.
    /// Must be in `[0, 1]`.
    pub mouth_dead_zone: f32,
    /// Attack (opening / intensifying) time constant in seconds.
    ///
    /// Smaller values make the output follow rising inputs faster.
    pub attack_time_constant_sec: f32,
    /// Release (closing / relaxing) time constant in seconds.
    ///
    /// Larger values make the output linger longer after the input relaxes.
    pub release_time_constant_sec: f32,
    /// Maximum elapsed seconds used for smoothing.
    ///
    /// Gaps larger than this are clamped so that a stale observation cannot
    /// snap the output abruptly.
    pub max_dt_sec: f32,
    /// Raw confidence values below this threshold are treated as missing.
    ///
    /// Must be in `[0, 1]`.
    pub missing_confidence_threshold: f32,
    /// Fallback policy for missing channels.
    pub missing_policy: MissingChannelPolicy,
}

impl ExpressionFilterParams {
    /// Returns parameters with the given attack/release time constants and
    /// otherwise default values.
    #[must_use]
    pub fn with_time_constants(
        attack_time_constant_sec: f32,
        release_time_constant_sec: f32,
    ) -> Self {
        Self {
            attack_time_constant_sec,
            release_time_constant_sec,
            ..Self::default()
        }
    }
}

impl Default for ExpressionFilterParams {
    fn default() -> Self {
        Self {
            mouth_dead_zone: 0.05,
            attack_time_constant_sec: 0.03,
            release_time_constant_sec: 0.10,
            max_dt_sec: 0.5,
            missing_confidence_threshold: 0.5,
            missing_policy: MissingChannelPolicy::symmetric(),
        }
    }
}

/// Calibration data for all expression channels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpressionCalibration {
    /// Calibration range for the left blink channel.
    pub blink_left: ExpressionRange,
    /// Calibration range for the right blink channel.
    pub blink_right: ExpressionRange,
    /// Calibration range for the mouth channel.
    pub mouth: ExpressionRange,
}

impl ExpressionCalibration {
    /// Creates a calibration bundle from validated per-channel ranges.
    #[must_use]
    pub fn new(
        blink_left: ExpressionRange,
        blink_right: ExpressionRange,
        mouth: ExpressionRange,
    ) -> Self {
        Self {
            blink_left,
            blink_right,
            mouth,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct ChannelState {
    value: f32,
    last_time: Option<MonoTimeNs>,
}

/// Normalizes and smooths raw expression observations into avatar expression
/// coefficients.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpressionFilter {
    calibration: ExpressionCalibration,
    params: ExpressionFilterParams,
    blink_left: ChannelState,
    blink_right: ChannelState,
    mouth: ChannelState,
}

impl ExpressionFilter {
    /// Creates a new filter with the given calibration and parameters.
    #[must_use]
    pub fn new(calibration: ExpressionCalibration, params: ExpressionFilterParams) -> Self {
        Self {
            calibration,
            params,
            blink_left: ChannelState::default(),
            blink_right: ChannelState::default(),
            mouth: ChannelState::default(),
        }
    }

    /// Returns the calibration used by this filter.
    #[must_use]
    pub fn calibration(&self) -> &ExpressionCalibration {
        &self.calibration
    }

    /// Resets all smoothed state to zero.
    pub fn reset(&mut self) {
        self.blink_left = ChannelState::default();
        self.blink_right = ChannelState::default();
        self.mouth = ChannelState::default();
    }

    /// Returns the current smoothed coefficients without consuming a new
    /// observation.
    #[must_use]
    pub fn current(&self) -> ExpressionCoefficients {
        ExpressionCoefficients {
            blink_left: self.blink_left.value,
            blink_right: self.blink_right.value,
            aa: self.mouth.value,
            ..ExpressionCoefficients::default()
        }
    }

    /// Updates the filter with a new raw observation and returns the
    /// normalized, smoothed expression coefficients.
    ///
    /// Low-confidence or non-finite raw channels are replaced according to
    /// [`ExpressionFilterParams::missing_policy`] before normalization.  The
    /// returned coefficients are always finite and clamped to `[0, 1]`.
    #[must_use]
    pub fn update(
        &mut self,
        observation: &RawExpressionObservation,
        now: MonoTimeNs,
    ) -> ExpressionCoefficients {
        let left_present = is_present(
            observation.blink_left,
            observation.blink_left_confidence,
            self.params.missing_confidence_threshold,
        );
        let right_present = is_present(
            observation.blink_right,
            observation.blink_right_confidence,
            self.params.missing_confidence_threshold,
        );
        let mouth_present = is_present(
            observation.mouth_open,
            observation.mouth_open_confidence,
            self.params.missing_confidence_threshold,
        );

        let raw_left = if left_present {
            Some(observation.blink_left)
        } else {
            None
        };
        let raw_right = if right_present {
            Some(observation.blink_right)
        } else {
            None
        };
        let raw_mouth = if mouth_present {
            Some(observation.mouth_open)
        } else {
            None
        };

        let effective_left =
            resolve_raw(raw_left, raw_right, self.params.missing_policy.blink_left);
        let effective_right =
            resolve_raw(raw_right, raw_left, self.params.missing_policy.blink_right);
        let effective_mouth = resolve_mouth(raw_mouth, self.params.missing_policy.mouth);

        let target_left = self.calibration.blink_left.normalize(effective_left);
        let target_right = self.calibration.blink_right.normalize(effective_right);
        let target_mouth_norm = self.calibration.mouth.normalize(effective_mouth);
        let target_mouth = apply_dead_zone(target_mouth_norm, self.params.mouth_dead_zone);

        let blink_left = self.blink_left.update(target_left, now, &self.params);
        let blink_right = self.blink_right.update(target_right, now, &self.params);
        let aa = self.mouth.update(target_mouth, now, &self.params);

        ExpressionCoefficients {
            blink_left,
            blink_right,
            aa,
            ..ExpressionCoefficients::default()
        }
    }
}

fn is_present(value: f32, confidence: f32, threshold: f32) -> bool {
    value.is_finite() && confidence.is_finite() && confidence >= threshold
}

fn resolve_raw(own: Option<f32>, opposite: Option<f32>, policy: MissingChannelFallback) -> f32 {
    match own {
        Some(v) => v,
        None => match policy {
            MissingChannelFallback::Zero => 0.0,
            MissingChannelFallback::MirrorOpposite => opposite.unwrap_or(0.0),
        },
    }
}

fn resolve_mouth(own: Option<f32>, policy: MissingChannelFallback) -> f32 {
    match own {
        Some(v) => v,
        None => match policy {
            MissingChannelFallback::Zero => 0.0,
            // Mirroring has no meaning for the mouth; treat it as zero.
            MissingChannelFallback::MirrorOpposite => 0.0,
        },
    }
}

fn apply_dead_zone(value: f32, dead_zone: f32) -> f32 {
    let dz = dead_zone.clamp(0.0, 1.0);
    (value - dz).clamp(0.0, 1.0)
}

impl ChannelState {
    fn update(&mut self, target: f32, now: MonoTimeNs, params: &ExpressionFilterParams) -> f32 {
        let target = target.clamp(0.0, 1.0);

        let Some(last_time) = self.last_time else {
            self.value = target;
            self.last_time = Some(now);
            return target;
        };

        // Treat backwards or equal timestamps as a state reset; this avoids
        // undefined smoothing behavior when sequence numbers wrap or clocks
        // are adjusted.
        if now.0 <= last_time.0 {
            self.value = target;
            self.last_time = Some(now);
            return target;
        }

        let dt_ns = now.0 - last_time.0;
        let dt_sec = (dt_ns as f32 / 1_000_000_000.0)
            .min(params.max_dt_sec)
            .max(0.0);
        if dt_sec <= 0.0 {
            return self.value;
        }

        let tau = if target > self.value {
            params.attack_time_constant_sec
        } else {
            params.release_time_constant_sec
        }
        .max(f32::EPSILON);
        let alpha = (1.0 - (-dt_sec / tau).exp()).clamp(0.0, 1.0);

        self.value = (self.value + alpha * (target - self.value)).clamp(0.0, 1.0);
        self.last_time = Some(now);
        self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calibration() -> ExpressionCalibration {
        // Neutral open-eye value equals the `open` endpoint so that a relaxed
        // face produces a normalized blink coefficient of zero.
        ExpressionCalibration::new(
            ExpressionRange::for_blink(0.05, 0.05, 0.9).unwrap(),
            ExpressionRange::for_blink(0.05, 0.05, 0.9).unwrap(),
            ExpressionRange::for_mouth(0.05, 0.8).unwrap(),
        )
    }

    fn params() -> ExpressionFilterParams {
        ExpressionFilterParams {
            mouth_dead_zone: 0.0,
            ..ExpressionFilterParams::with_time_constants(0.03, 0.10)
        }
    }

    fn obs(blink_left: f32, blink_right: f32, mouth_open: f32) -> RawExpressionObservation {
        RawExpressionObservation {
            blink_left,
            blink_left_confidence: 0.9,
            blink_right,
            blink_right_confidence: 0.9,
            mouth_open,
            mouth_open_confidence: 0.9,
        }
    }

    #[test]
    fn neutral_values_yield_near_zero() {
        let mut filter = ExpressionFilter::new(calibration(), params());
        let out = filter.update(&obs(0.05, 0.05, 0.05), MonoTimeNs(16_666_667));

        assert!(out.blink_left.abs() < 1e-4, "left={}", out.blink_left);
        assert!(out.blink_right.abs() < 1e-4, "right={}", out.blink_right);
        assert!(out.aa.abs() < 1e-4, "aa={}", out.aa);
    }

    #[test]
    fn max_values_yield_near_one() {
        let mut filter = ExpressionFilter::new(calibration(), params());
        let out = filter.update(&obs(0.9, 0.9, 0.8), MonoTimeNs(16_666_667));

        assert!(
            (out.blink_left - 1.0).abs() < 1e-4,
            "left={}",
            out.blink_left
        );
        assert!(
            (out.blink_right - 1.0).abs() < 1e-4,
            "right={}",
            out.blink_right
        );
        assert!((out.aa - 1.0).abs() < 1e-4, "aa={}", out.aa);
    }

    #[test]
    fn left_and_right_blink_are_separate_channels() {
        let mut filter = ExpressionFilter::new(calibration(), params());
        let out = filter.update(&obs(0.9, 0.05, 0.05), MonoTimeNs(16_666_667));

        assert!((out.blink_left - 1.0).abs() < 1e-4);
        assert!(out.blink_right.abs() < 1e-4);
        assert!(out.aa.abs() < 1e-4);
    }

    #[test]
    fn inverted_range_is_rejected() {
        let err = ExpressionRange::for_blink(0.5, 0.9, 0.05).unwrap_err();
        assert_eq!(err.code(), "EXPRESSION_CALIBRATION_INVERTED_RANGE");
    }

    #[test]
    fn zero_span_is_rejected() {
        let err = ExpressionRange::for_mouth(0.5, 0.5).unwrap_err();
        assert_eq!(err.code(), "EXPRESSION_CALIBRATION_ZERO_SPAN");
    }

    #[test]
    fn out_of_range_value_is_rejected() {
        let err = ExpressionRange::for_blink(-0.1, 0.0, 1.0).unwrap_err();
        assert_eq!(err.code(), "EXPRESSION_CALIBRATION_VALUE_OUT_OF_RANGE");
    }

    #[test]
    fn missing_left_blink_falls_back_to_right() {
        let mut observation = obs(0.05, 0.9, 0.05);
        observation.blink_left_confidence = 0.1;

        let mut filter = ExpressionFilter::new(calibration(), params());
        let out = filter.update(&observation, MonoTimeNs(16_666_667));

        assert!((out.blink_left - 1.0).abs() < 1e-4);
    }

    #[test]
    fn missing_mouth_falls_back_to_zero() {
        let mut observation = obs(0.05, 0.05, 0.8);
        observation.mouth_open_confidence = 0.1;

        let mut filter = ExpressionFilter::new(calibration(), params());
        let out = filter.update(&observation, MonoTimeNs(16_666_667));

        assert!(out.aa.abs() < 1e-4);
    }

    #[test]
    fn mouth_dead_zone_silences_small_openings() {
        let cal = ExpressionCalibration::new(
            ExpressionRange::for_blink(0.05, 0.05, 0.9).unwrap(),
            ExpressionRange::for_blink(0.05, 0.05, 0.9).unwrap(),
            ExpressionRange::for_mouth(0.05, 0.8).unwrap(),
        );
        let mut params = params();
        params.mouth_dead_zone = 0.2;

        let mut filter = ExpressionFilter::new(cal, params);
        // Normalized mouth value is exactly 0.1333...; dead zone of 0.2 -> 0.
        let out = filter.update(&obs(0.05, 0.05, 0.15), MonoTimeNs(16_666_667));
        assert!(out.aa.abs() < 1e-4);
    }

    #[test]
    fn attack_is_faster_than_release() {
        let mut params = params();
        params.attack_time_constant_sec = 0.01;
        params.release_time_constant_sec = 0.50;
        let mut filter = ExpressionFilter::new(calibration(), params);

        // Step up.
        let up = filter.update(&obs(0.9, 0.9, 0.8), MonoTimeNs(16_666_667));
        // Step down after the same elapsed time.
        let down = filter.update(&obs(0.05, 0.05, 0.05), MonoTimeNs(33_333_334));

        // After attack we should be much closer to 1.0 than after release
        // is to 0.0, because release is deliberately slow.
        assert!(up.aa > 1.0 - down.aa, "up={}, down={}", up.aa, down.aa);
    }

    #[test]
    fn output_stays_finite_and_clamped() {
        let mut filter = ExpressionFilter::new(calibration(), params());
        let bad = RawExpressionObservation {
            blink_left: f32::NAN,
            blink_left_confidence: 0.9,
            blink_right: f32::INFINITY,
            blink_right_confidence: 0.9,
            mouth_open: 2.0,
            mouth_open_confidence: 0.9,
        };

        let out = filter.update(&bad, MonoTimeNs(16_666_667));
        assert!(out.blink_left.is_finite() && (0.0..=1.0).contains(&out.blink_left));
        assert!(out.blink_right.is_finite() && (0.0..=1.0).contains(&out.blink_right));
        assert!(out.aa.is_finite() && (0.0..=1.0).contains(&out.aa));
    }
}
