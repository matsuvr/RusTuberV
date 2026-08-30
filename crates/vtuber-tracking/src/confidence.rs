//! Confidence synthesis and hysteresis gating.
//!
//! This module combines per-source confidence values into a single frame
//! confidence and decides when a sequence of good or bad frames is strong
//! enough to emit an [`Acquire`](ConfidenceSignal::Acquire) or
//! [`Degrade`](ConfidenceSignal::Degrade) signal.
//!
//! Missing confidence sources are handled field-by-field according to
//! [`MissingSourcePolicy`]:
//!
//! - `detector`: [`TreatAsZero`](MissingSourcePolicy::TreatAsZero) by
//!   default. A missing detector means no face was found, so the resulting
//!   frame confidence must be zero.
//! - `landmark`: [`TreatAsZero`](MissingSourcePolicy::TreatAsZero) by
//!   default. A missing landmark confidence means no usable face geometry
//!   was produced.
//! - `pose`: [`Ignore`](MissingSourcePolicy::Ignore) by default. The pose
//!   solver may not produce a separate confidence value in this pipeline
//!   stage, so its absence should not penalize the frame confidence.
//! - `expression`: [`Ignore`](MissingSourcePolicy::Ignore) by default.
//!   Expressions can fall back to geometry, so unavailability should not
//!   kill confidence.
//!
//! Hysteresis is implemented with separate enter and exit thresholds.
//! Once the gate is confident, confidence must fall below `exit_threshold`
//! before bad frames are counted; once the gate is not confident, confidence
//! must reach `enter_threshold` before good frames are counted. Values
//! between the two thresholds do not change the counts, which prevents
//! state oscillation when confidence hovers near a single threshold.
//!
//! Consecutive-good and consecutive-bad counters are bounded to avoid
//! overflow on a long stream of identical-quality frames.

use thiserror::Error;

/// Confidence source used for error reporting and diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConfidenceSource {
    /// Face detector confidence.
    Detector,
    /// Facial landmark confidence / visibility.
    Landmark,
    /// Head pose solver confidence.
    Pose,
    /// Expression / blendshape availability confidence.
    Expression,
}

/// How a missing confidence source participates in frame synthesis.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MissingSourcePolicy {
    /// Treat a missing source as confidence `0.0`.
    #[default]
    TreatAsZero,
    /// Ignore the source; it does not participate in the minimum.
    Ignore,
}

/// Per-source confidence inputs for one frame.
///
/// `None` means the source is not available for this frame. How `None`
/// affects synthesis is controlled by [`ConfidencePolicies`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ConfidenceInputs {
    /// Face detector confidence in `[0, 1]`.
    pub detector: Option<f32>,
    /// Landmark confidence or mean visibility in `[0, 1]`.
    pub landmark: Option<f32>,
    /// Pose solver confidence in `[0, 1]`.
    pub pose: Option<f32>,
    /// Expression/blendshape availability confidence in `[0, 1]`.
    pub expression: Option<f32>,
}

/// Per-field policy for handling missing confidence sources.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConfidencePolicies {
    /// Policy for a missing [`ConfidenceInputs::detector`].
    pub detector: MissingSourcePolicy,
    /// Policy for a missing [`ConfidenceInputs::landmark`].
    pub landmark: MissingSourcePolicy,
    /// Policy for a missing [`ConfidenceInputs::pose`].
    pub pose: MissingSourcePolicy,
    /// Policy for a missing [`ConfidenceInputs::expression`].
    pub expression: MissingSourcePolicy,
}

impl Default for ConfidencePolicies {
    fn default() -> Self {
        Self {
            detector: MissingSourcePolicy::TreatAsZero,
            landmark: MissingSourcePolicy::TreatAsZero,
            pose: MissingSourcePolicy::Ignore,
            expression: MissingSourcePolicy::Ignore,
        }
    }
}

/// Errors that can occur while synthesizing frame confidence.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum ConfidenceError {
    /// A supplied confidence value was NaN, infinite, or outside `[0, 1]`.
    #[error("confidence value for {0:?} is non-finite or out of [0, 1]")]
    InvalidValue(ConfidenceSource),
}

/// Synthesizes a single frame confidence from per-source inputs.
///
/// The result is the minimum of all participating sources. Missing sources
/// are handled according to their field policy. A value of `0.0` is returned
/// when every source is ignored and missing.
///
/// # Errors
///
/// Returns [`ConfidenceError::InvalidValue`] if any supplied value is not
/// finite or not in `[0, 1]`. Callers should treat such a frame as invalid
/// and reject its confidence.
///
/// # Examples
///
/// ```
/// use vtuber_tracking::confidence::{ConfidenceInputs, ConfidencePolicies, synthesize};
///
/// let inputs = ConfidenceInputs {
///     detector: Some(0.9),
///     landmark: Some(0.8),
///     ..ConfidenceInputs::default()
/// };
/// assert_eq!(synthesize(&inputs, &ConfidencePolicies::default()).unwrap(), 0.8);
/// ```
pub fn synthesize(
    inputs: &ConfidenceInputs,
    policies: &ConfidencePolicies,
) -> Result<f32, ConfidenceError> {
    let mut min: Option<f32> = None;

    let sources = [
        (
            ConfidenceSource::Detector,
            inputs.detector,
            policies.detector,
        ),
        (
            ConfidenceSource::Landmark,
            inputs.landmark,
            policies.landmark,
        ),
        (ConfidenceSource::Pose, inputs.pose, policies.pose),
        (
            ConfidenceSource::Expression,
            inputs.expression,
            policies.expression,
        ),
    ];

    for (source, value, policy) in sources {
        let resolved = resolve_source(source, value, policy)?;
        if let Some(v) = resolved {
            min = Some(min.map_or(v, |m| m.min(v)));
        }
    }

    Ok(min.unwrap_or(0.0))
}

fn resolve_source(
    source: ConfidenceSource,
    value: Option<f32>,
    policy: MissingSourcePolicy,
) -> Result<Option<f32>, ConfidenceError> {
    match value {
        Some(v) => {
            if v.is_finite() && (0.0..=1.0).contains(&v) {
                Ok(Some(v))
            } else {
                Err(ConfidenceError::InvalidValue(source))
            }
        }
        None => match policy {
            MissingSourcePolicy::TreatAsZero => Ok(Some(0.0)),
            MissingSourcePolicy::Ignore => Ok(None),
        },
    }
}

/// Errors that can occur when constructing a [`ConfidenceGate`].
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum ConfidenceConfigError {
    /// `enter_threshold` is not strictly greater than `exit_threshold`.
    #[error("enter_threshold ({enter}) must be greater than exit_threshold ({exit})")]
    ThresholdOrder {
        /// Enter threshold value.
        enter: f32,
        /// Exit threshold value.
        exit: f32,
    },
    /// A threshold is outside `[0, 1]` or non-finite.
    #[error("thresholds must be finite and in [0, 1], got enter={enter}, exit={exit}")]
    ThresholdRange {
        /// Enter threshold value.
        enter: f32,
        /// Exit threshold value.
        exit: f32,
    },
    /// A required consecutive count is zero.
    #[error("required consecutive counts must be positive, got good={good}, bad={bad}")]
    ZeroRequiredCount {
        /// Required good count.
        good: u32,
        /// Required bad count.
        bad: u32,
    },
    /// `max_count` is smaller than one of the required consecutive counts.
    #[error("max_count ({max}) must not be smaller than required counts (good={good}, bad={bad})")]
    MaxCountTooSmall {
        /// Maximum count.
        max: u32,
        /// Required good count.
        good: u32,
        /// Required bad count.
        bad: u32,
    },
}

/// Signal emitted by a [`ConfidenceGate`] when its confidence state changes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ConfidenceSignal {
    /// No state change this frame.
    #[default]
    None,
    /// Enough consecutive good frames were observed to become confident.
    Acquire,
    /// Enough consecutive bad frames were observed to leave the confident
    /// state.
    Degrade,
}

/// Parameters for the confidence hysteresis gate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConfidenceGateParams {
    /// Confidence threshold that must be reached to count a frame as good.
    ///
    /// Must be strictly greater than [`exit_threshold`](Self::exit_threshold).
    pub enter_threshold: f32,
    /// Confidence threshold below which a frame is counted as bad.
    pub exit_threshold: f32,
    /// Consecutive good frames required to emit
    /// [`Acquire`](ConfidenceSignal::Acquire).
    pub required_consecutive_good: u32,
    /// Consecutive bad frames required to emit
    /// [`Degrade`](ConfidenceSignal::Degrade).
    pub required_consecutive_bad: u32,
    /// Upper bound for the internal good/bad counters.
    ///
    /// Counters saturate at this value. It must be at least as large as
    /// both required counts.
    pub max_count: u32,
}

impl Default for ConfidenceGateParams {
    fn default() -> Self {
        Self {
            enter_threshold: 0.75,
            exit_threshold: 0.50,
            required_consecutive_good: 3,
            // Five misses at the normal 15 Hz tracking cadence is roughly
            // 300 ms. A separate watchdog still handles a stalled worker.
            required_consecutive_bad: 5,
            max_count: u32::MAX,
        }
    }
}

impl ConfidenceGateParams {
    /// Validates the parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ConfidenceConfigError`] if thresholds are misordered or out
    /// of range, or if the required/max counts are inconsistent.
    pub fn validate(&self) -> Result<(), ConfidenceConfigError> {
        let enter = self.enter_threshold;
        let exit = self.exit_threshold;

        if !(enter.is_finite() && exit.is_finite()) {
            return Err(ConfidenceConfigError::ThresholdRange { enter, exit });
        }
        if !(0.0..=1.0).contains(&enter) || !(0.0..=1.0).contains(&exit) {
            return Err(ConfidenceConfigError::ThresholdRange { enter, exit });
        }
        if enter <= exit {
            return Err(ConfidenceConfigError::ThresholdOrder { enter, exit });
        }
        if self.required_consecutive_good == 0 || self.required_consecutive_bad == 0 {
            return Err(ConfidenceConfigError::ZeroRequiredCount {
                good: self.required_consecutive_good,
                bad: self.required_consecutive_bad,
            });
        }
        if self.max_count < self.required_consecutive_good
            || self.max_count < self.required_consecutive_bad
        {
            return Err(ConfidenceConfigError::MaxCountTooSmall {
                max: self.max_count,
                good: self.required_consecutive_good,
                bad: self.required_consecutive_bad,
            });
        }
        Ok(())
    }
}

/// Snapshot returned by [`ConfidenceGate::update`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConfidenceAssessment {
    /// Frame confidence used for the update, clamped to `[0, 1]`.
    ///
    /// If the input was non-finite, this is `0.0` and
    /// [`valid`](Self::valid) is `false`.
    pub frame_confidence: f32,
    /// Number of consecutive good frames currently counted.
    pub consecutive_good: u32,
    /// Number of consecutive bad frames currently counted.
    pub consecutive_bad: u32,
    /// Whether the gate is currently in the confident state.
    pub is_confident: bool,
    /// Signal emitted by this update, if any.
    pub signal: ConfidenceSignal,
    /// `false` if the input confidence was non-finite.
    pub valid: bool,
}

/// Hysteresis gate for frame confidence.
///
/// Combines consecutive-frame counting with separate enter/exit thresholds
/// to avoid oscillation near a threshold. The gate starts in the
/// not-confident state.
#[derive(Clone, Debug, PartialEq)]
pub struct ConfidenceGate {
    params: ConfidenceGateParams,
    is_confident: bool,
    good_count: u32,
    bad_count: u32,
}

impl ConfidenceGate {
    /// Creates a new gate with the given parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ConfidenceConfigError`] if the parameters are invalid.
    pub fn new(params: ConfidenceGateParams) -> Result<Self, ConfidenceConfigError> {
        params.validate()?;
        Ok(Self {
            params,
            is_confident: false,
            good_count: 0,
            bad_count: 0,
        })
    }

    /// Returns whether the gate is currently confident.
    #[must_use]
    pub fn is_confident(&self) -> bool {
        self.is_confident
    }

    /// Resets internal counters and returns to the not-confident state.
    pub fn reset(&mut self) {
        self.is_confident = false;
        self.good_count = 0;
        self.bad_count = 0;
    }

    /// Updates the gate with a new frame confidence.
    ///
    /// `frame_confidence` should normally be the result of
    /// [`synthesize`]. Non-finite values are rejected: the update is treated
    /// as bad with a frame confidence of `0.0` and
    /// [`ConfidenceAssessment::valid`] is set to `false`.
    ///
    /// # Returns
    ///
    /// A [`ConfidenceAssessment`] describing the gate state after this
    /// frame. The assessment is independent of the gate's internal state;
    /// it is safe to drop without affecting later updates.
    pub fn update(&mut self, frame_confidence: f32) -> ConfidenceAssessment {
        let valid = frame_confidence.is_finite() && (0.0..=1.0).contains(&frame_confidence);
        let c = if valid { frame_confidence } else { 0.0 };

        let mut signal = ConfidenceSignal::None;

        if self.is_confident {
            if c <= self.params.exit_threshold {
                self.bad_count = increment_bounded(self.bad_count, self.params.max_count);
                self.good_count = 0;
                if self.bad_count >= self.params.required_consecutive_bad {
                    self.is_confident = false;
                    signal = ConfidenceSignal::Degrade;
                    self.bad_count = 0;
                    self.good_count = 0;
                }
            } else if c >= self.params.enter_threshold {
                self.good_count = increment_bounded(self.good_count, self.params.max_count);
                self.bad_count = 0;
            }
            // Values between `exit_threshold` and `enter_threshold` leave
            // counts unchanged, preventing oscillation.
        } else {
            if c >= self.params.enter_threshold {
                self.good_count = increment_bounded(self.good_count, self.params.max_count);
                self.bad_count = 0;
                if self.good_count >= self.params.required_consecutive_good {
                    self.is_confident = true;
                    signal = ConfidenceSignal::Acquire;
                    self.good_count = 0;
                    self.bad_count = 0;
                }
            } else if c <= self.params.exit_threshold {
                self.bad_count = increment_bounded(self.bad_count, self.params.max_count);
                self.good_count = 0;
            }
            // Values between `exit_threshold` and `enter_threshold` leave
            // counts unchanged, preventing oscillation.
        }

        ConfidenceAssessment {
            frame_confidence: c,
            consecutive_good: self.good_count,
            consecutive_bad: self.bad_count,
            is_confident: self.is_confident,
            signal,
            valid,
        }
    }
}

#[must_use]
fn increment_bounded(count: u32, max: u32) -> u32 {
    count.saturating_add(1).min(max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_params() -> ConfidenceGateParams {
        ConfidenceGateParams {
            enter_threshold: 0.8,
            exit_threshold: 0.4,
            required_consecutive_good: 2,
            required_consecutive_bad: 3,
            max_count: 100,
        }
    }

    #[test]
    fn synthesize_takes_minimum() {
        let inputs = ConfidenceInputs {
            detector: Some(0.9),
            landmark: Some(0.7),
            pose: Some(0.85),
            expression: Some(0.99),
        };
        assert_eq!(
            synthesize(&inputs, &ConfidencePolicies::default()).unwrap(),
            0.7
        );
    }

    #[test]
    fn missing_detector_treated_as_zero() {
        let inputs = ConfidenceInputs {
            detector: None,
            landmark: Some(0.9),
            ..ConfidenceInputs::default()
        };
        assert_eq!(
            synthesize(&inputs, &ConfidencePolicies::default()).unwrap(),
            0.0
        );
    }

    #[test]
    fn missing_expression_ignored_by_default() {
        let inputs = ConfidenceInputs {
            detector: Some(0.9),
            landmark: Some(0.8),
            expression: None,
            ..ConfidenceInputs::default()
        };
        assert_eq!(
            synthesize(&inputs, &ConfidencePolicies::default()).unwrap(),
            0.8
        );
    }

    #[test]
    fn all_ignored_missing_yields_zero() {
        let inputs = ConfidenceInputs::default();
        let policies = ConfidencePolicies {
            detector: MissingSourcePolicy::Ignore,
            landmark: MissingSourcePolicy::Ignore,
            pose: MissingSourcePolicy::Ignore,
            expression: MissingSourcePolicy::Ignore,
        };
        assert_eq!(synthesize(&inputs, &policies).unwrap(), 0.0);
    }

    #[test]
    fn nan_confidence_rejected() {
        let inputs = ConfidenceInputs {
            detector: Some(f32::NAN),
            landmark: Some(0.9),
            ..ConfidenceInputs::default()
        };
        let err = synthesize(&inputs, &ConfidencePolicies::default()).unwrap_err();
        assert_eq!(
            err,
            ConfidenceError::InvalidValue(ConfidenceSource::Detector)
        );
    }

    #[test]
    fn out_of_range_confidence_rejected() {
        let inputs = ConfidenceInputs {
            detector: Some(1.1),
            ..ConfidenceInputs::default()
        };
        let err = synthesize(&inputs, &ConfidencePolicies::default()).unwrap_err();
        assert_eq!(
            err,
            ConfidenceError::InvalidValue(ConfidenceSource::Detector)
        );
    }

    #[test]
    fn invalid_params_detected() {
        assert!(
            ConfidenceGateParams {
                enter_threshold: 0.4,
                exit_threshold: 0.8,
                ..ConfidenceGateParams::default()
            }
            .validate()
            .is_err()
        );

        assert!(
            ConfidenceGateParams {
                enter_threshold: 1.2,
                exit_threshold: 0.5,
                ..ConfidenceGateParams::default()
            }
            .validate()
            .is_err()
        );

        assert!(
            ConfidenceGateParams {
                enter_threshold: 0.8,
                exit_threshold: 0.4,
                required_consecutive_good: 0,
                ..ConfidenceGateParams::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn gate_acquires_after_consecutive_good() {
        let params = test_params();
        let mut gate = ConfidenceGate::new(params).unwrap();

        let a1 = gate.update(0.9);
        assert!(!a1.is_confident);
        assert_eq!(a1.signal, ConfidenceSignal::None);
        assert_eq!(a1.consecutive_good, 1);

        let a2 = gate.update(0.85);
        assert!(a2.is_confident);
        assert_eq!(a2.signal, ConfidenceSignal::Acquire);
        assert_eq!(a2.consecutive_good, 0);
    }

    #[test]
    fn gate_degrades_after_consecutive_bad() {
        let params = test_params();
        let mut gate = ConfidenceGate::new(params).unwrap();
        gate.update(0.9);
        gate.update(0.85); // Acquire

        gate.update(0.35);
        gate.update(0.30);
        let a = gate.update(0.25);
        assert!(!a.is_confident);
        assert_eq!(a.signal, ConfidenceSignal::Degrade);
    }

    #[test]
    fn gate_does_not_oscillate_in_hysteresis_band() {
        let params = test_params();
        let mut gate = ConfidenceGate::new(params).unwrap();
        gate.update(0.9);
        gate.update(0.85); // Acquire

        for _ in 0..20 {
            let a = gate.update(0.6);
            assert!(a.is_confident);
            assert_eq!(a.signal, ConfidenceSignal::None);
            assert_eq!(a.consecutive_good, 0);
            assert_eq!(a.consecutive_bad, 0);
        }
    }

    #[test]
    fn non_finite_input_is_rejected_as_bad() {
        let params = test_params();
        let mut gate = ConfidenceGate::new(params).unwrap();
        let a = gate.update(f32::NAN);
        assert!(!a.is_confident);
        assert!(!a.valid);
        assert_eq!(a.frame_confidence, 0.0);
        assert_eq!(a.consecutive_bad, 1);
    }

    #[test]
    fn counters_saturate_at_max() {
        let params = ConfidenceGateParams {
            max_count: 5,
            ..test_params()
        };
        let mut gate = ConfidenceGate::new(params).unwrap();

        // Five bad frames should saturate; later frames must not overflow.
        for i in 0..10 {
            let a = gate.update(0.0);
            assert!(
                a.consecutive_bad <= 5,
                "counter overflow at frame {i}: {a:?}"
            );
        }
        let a = gate.update(0.0);
        assert_eq!(a.consecutive_bad, 5);
    }
}
