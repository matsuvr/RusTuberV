//! Control settings used by the app to configure the tracking pipeline.
//!
//! This module is intentionally engine-independent: it does not depend on
//! Bevy, `bevy_vrm1`, camera backends, or inference runtimes.

use std::error::Error;
use std::fmt;

/// Minimum number of samples any calibration must collect.
///
/// This is a hard floor independent of the landmark schema: fewer than three
/// points cannot produce a stable pose via Kabsch, and even three samples are
/// too noisy for a trustworthy neutral reference.
pub const MIN_CALIBRATION_SAMPLES: usize = 5;

/// Versioned calibration settings.
///
/// Settings are persisted by the app and consumed by the tracking pipeline.
/// The version field lets future code migrate older persisted values instead
/// of silently ignoring new thresholds.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationSettings {
    version: u16,
    inner: CalibrationSettingsV1,
}

impl CalibrationSettings {
    /// Default calibration settings.
    ///
    /// Defaults are chosen for a 30 FPS webcam stream with MediaPipe-style
    /// face landmarks:
    ///
    /// * `required_sample_count = 30` — one second of stable frames at 30 FPS.
    ///   Fewer samples are too sensitive to a single bad frame; more samples
    ///   lengthen the user experience without improving robustness at MVP.
    /// * `max_duration_seconds = 5.0` — hard upper bound so a wandering face
    ///   does not leave the UI stuck in `Collecting` indefinitely.
    /// * `min_confidence = 0.5` — rejects frames where the face detector or
    ///   landmark model is uncertain. This is below typical detection
    ///   confidence but above pure noise.
    /// * `max_head_motion_rad = 0.0873` (5 degrees) — the user's head must
    ///   stay still during calibration. Expressed in radians.
    /// * `max_expression_motion = 0.15` — combined eye/mouth coefficient
    ///   movement between consecutive frames must stay below this value.
    #[must_use]
    pub fn new() -> Self {
        Self {
            version: 1,
            inner: CalibrationSettingsV1 {
                required_sample_count: 30,
                max_duration_seconds: 5.0,
                min_confidence: 0.5,
                max_head_motion_rad: 5.0f32.to_radians(),
                max_expression_motion: 0.15,
            },
        }
    }

    /// Construct settings from raw values, rejecting invalid combinations.
    ///
    /// # Errors
    ///
    /// Returns [`CalibrationError`] when any argument is outside its valid
    /// domain or would make calibration impossible.
    pub fn try_new(
        required_sample_count: usize,
        max_duration_seconds: f32,
        min_confidence: f32,
        max_head_motion_rad: f32,
        max_expression_motion: f32,
    ) -> Result<Self, CalibrationError> {
        if required_sample_count < MIN_CALIBRATION_SAMPLES {
            return Err(CalibrationError::InsufficientSamples(required_sample_count));
        }
        if !max_duration_seconds.is_finite() || max_duration_seconds <= 0.0 {
            return Err(CalibrationError::InvalidDuration(max_duration_seconds));
        }
        if !(0.0..=1.0).contains(&min_confidence) {
            return Err(CalibrationError::ConfidenceOutOfRange(min_confidence));
        }
        if !max_head_motion_rad.is_finite() || max_head_motion_rad <= 0.0 {
            return Err(CalibrationError::InvalidMotionThreshold(
                max_head_motion_rad,
            ));
        }
        if !(0.0..=1.0).contains(&max_expression_motion) {
            return Err(CalibrationError::ExpressionMotionOutOfRange(
                max_expression_motion,
            ));
        }
        Ok(Self {
            version: 1,
            inner: CalibrationSettingsV1 {
                required_sample_count,
                max_duration_seconds,
                min_confidence,
                max_head_motion_rad,
                max_expression_motion,
            },
        })
    }

    /// Format version of these settings.
    #[must_use]
    pub fn version(&self) -> u16 {
        self.version
    }

    /// Minimum number of valid neutral frames required to finish calibration.
    #[must_use]
    pub fn required_sample_count(&self) -> usize {
        self.inner.required_sample_count
    }

    /// Maximum wall-clock duration, in seconds, allowed for a single
    /// calibration session.
    #[must_use]
    pub fn max_duration_seconds(&self) -> f32 {
        self.inner.max_duration_seconds
    }

    /// Minimum per-frame face confidence for a frame to count toward
    /// calibration.
    #[must_use]
    pub fn min_confidence(&self) -> f32 {
        self.inner.min_confidence
    }

    /// Maximum head rotation, in radians, allowed between consecutive
    /// calibration samples.
    #[must_use]
    pub fn max_head_motion_rad(&self) -> f32 {
        self.inner.max_head_motion_rad
    }

    /// Maximum expression coefficient movement, in the `[0, 1]` range,
    /// allowed between consecutive calibration samples.
    #[must_use]
    pub fn max_expression_motion(&self) -> f32 {
        self.inner.max_expression_motion
    }
}

impl Default for CalibrationSettings {
    fn default() -> Self {
        Self::new()
    }
}

/// Concrete fields for calibration settings version 1.
#[derive(Clone, Debug, PartialEq)]
struct CalibrationSettingsV1 {
    required_sample_count: usize,
    max_duration_seconds: f32,
    min_confidence: f32,
    max_head_motion_rad: f32,
    max_expression_motion: f32,
}

/// Errors that can occur when constructing or applying calibration settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CalibrationError {
    /// Required sample count is below the minimum.
    InsufficientSamples(usize),
    /// Maximum duration is not a positive finite number.
    InvalidDuration(f32),
    /// Confidence threshold is outside `[0, 1]`.
    ConfidenceOutOfRange(f32),
    /// Head motion threshold is not a positive finite value.
    InvalidMotionThreshold(f32),
    /// Expression motion threshold is outside `[0, 1]`.
    ExpressionMotionOutOfRange(f32),
    /// Requested session state transition is illegal in the current state.
    InvalidStateTransition {
        /// Current state name.
        from: &'static str,
        /// Requested state name.
        to: &'static str,
    },
    /// Too few landmarks were supplied to build a neutral reference.
    InsufficientLandmarks(usize),
    /// The neutral landmark point cloud is degenerate (collinear or all identical).
    DegeneratePointCloud,
    /// Head pose spread across calibration samples is too large.
    PoseSpreadTooLarge {
        /// Observed yaw spread in radians.
        yaw: f32,
        /// Observed pitch spread in radians.
        pitch: f32,
        /// Observed roll spread in radians.
        roll: f32,
        /// Maximum allowed spread in radians.
        max: f32,
    },
    /// Landmark position spread across calibration samples is too large.
    LandmarkSpreadTooLarge {
        /// Observed normalized spread.
        spread: f32,
        /// Maximum allowed normalized spread.
        max: f32,
    },
    /// The profile's model hash does not match the active model.
    ModelHashMismatch,
}

impl fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientSamples(n) => {
                write!(
                    f,
                    "required sample count {n} is below minimum {MIN_CALIBRATION_SAMPLES}"
                )
            }
            Self::InvalidDuration(d) => {
                write!(f, "max duration {d} seconds is not positive and finite")
            }
            Self::ConfidenceOutOfRange(c) => {
                write!(f, "confidence threshold {c} must be in [0, 1]")
            }
            Self::InvalidMotionThreshold(m) => {
                write!(
                    f,
                    "head motion threshold {m} rad is not positive and finite"
                )
            }
            Self::ExpressionMotionOutOfRange(m) => {
                write!(f, "expression motion threshold {m} must be in [0, 1]")
            }
            Self::InvalidStateTransition { from, to } => {
                write!(
                    f,
                    "cannot transition calibration session from {from} to {to}"
                )
            }
            Self::InsufficientLandmarks(n) => {
                write!(f, "insufficient landmarks: need at least 3, got {n}")
            }
            Self::DegeneratePointCloud => {
                write!(f, "neutral landmark point cloud is degenerate")
            }
            Self::PoseSpreadTooLarge {
                yaw,
                pitch,
                roll,
                max,
            } => {
                write!(
                    f,
                    "head pose spread too large: yaw={yaw}, pitch={pitch}, roll={roll}, max={max}"
                )
            }
            Self::LandmarkSpreadTooLarge { spread, max } => {
                write!(f, "landmark spread too large: spread={spread}, max={max}")
            }
            Self::ModelHashMismatch => {
                write!(f, "profile model hash does not match active model")
            }
        }
    }
}

impl Error for CalibrationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

impl CalibrationError {
    /// Stable string code for logging and UI mapping.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::InsufficientSamples(_) => "CALIBRATION_INSUFFICIENT_SAMPLES",
            Self::InvalidDuration(_) => "CALIBRATION_INVALID_DURATION",
            Self::ConfidenceOutOfRange(_) => "CALIBRATION_CONFIDENCE_OUT_OF_RANGE",
            Self::InvalidMotionThreshold(_) => "CALIBRATION_INVALID_MOTION_THRESHOLD",
            Self::ExpressionMotionOutOfRange(_) => "CALIBRATION_EXPRESSION_MOTION_OUT_OF_RANGE",
            Self::InvalidStateTransition { .. } => "CALIBRATION_INVALID_STATE_TRANSITION",
            Self::InsufficientLandmarks(_) => "CALIBRATION_INSUFFICIENT_LANDMARKS",
            Self::DegeneratePointCloud => "CALIBRATION_DEGENERATE_POINT_CLOUD",
            Self::PoseSpreadTooLarge { .. } => "CALIBRATION_POSE_SPREAD_TOO_LARGE",
            Self::LandmarkSpreadTooLarge { .. } => "CALIBRATION_LANDMARK_SPREAD_TOO_LARGE",
            Self::ModelHashMismatch => "CALIBRATION_MODEL_HASH_MISMATCH",
        }
    }
}

/// Versioned pipeline settings persisted by the app.
///
/// This struct intentionally contains only the knobs that the user or app
/// should persist across sessions. Runtime tuning parameters that change
/// every frame or are derived from calibration live in
/// `vtuber-tracking::pipeline::PipelineConfig`.
#[derive(Clone, Debug, PartialEq)]
pub struct TrackingPipelineSettings {
    version: u16,
    inner: TrackingPipelineSettingsV1,
}

impl TrackingPipelineSettings {
    /// Creates default pipeline settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            version: 1,
            inner: TrackingPipelineSettingsV1 {
                calibration: CalibrationSettings::new(),
            },
        }
    }

    /// Creates settings with an explicit calibration configuration.
    ///
    /// `calibration` is expected to have been constructed through
    /// [`CalibrationSettings::try_new`]; this constructor does not re-validate
    /// it because the calibration type's fields are private.
    #[must_use]
    pub fn with_calibration(calibration: CalibrationSettings) -> Self {
        Self {
            version: 1,
            inner: TrackingPipelineSettingsV1 { calibration },
        }
    }

    /// Format version of these settings.
    #[must_use]
    pub fn version(&self) -> u16 {
        self.version
    }

    /// Calibration settings for the tracking pipeline.
    #[must_use]
    pub fn calibration(&self) -> &CalibrationSettings {
        &self.inner.calibration
    }
}

impl Default for TrackingPipelineSettings {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, PartialEq)]
struct TrackingPipelineSettingsV1 {
    calibration: CalibrationSettings,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings_are_valid() {
        let s = CalibrationSettings::new();
        assert_eq!(s.version(), 1);
        assert_eq!(s.required_sample_count(), 30);
        assert_eq!(s.max_duration_seconds(), 5.0);
        assert_eq!(s.min_confidence(), 0.5);
        assert!((s.max_head_motion_rad() - 5.0f32.to_radians()).abs() < 1e-4);
        assert_eq!(s.max_expression_motion(), 0.15);
    }

    #[test]
    fn try_new_accepts_valid_values() {
        let s = CalibrationSettings::try_new(10, 3.0, 0.75, 0.1, 0.2).unwrap();
        assert_eq!(s.required_sample_count(), 10);
        assert_eq!(s.max_duration_seconds(), 3.0);
    }

    #[test]
    fn try_new_rejects_insufficient_samples() {
        let err = CalibrationSettings::try_new(2, 5.0, 0.5, 0.1, 0.1).unwrap_err();
        assert_eq!(err, CalibrationError::InsufficientSamples(2));
        assert_eq!(err.code(), "CALIBRATION_INSUFFICIENT_SAMPLES");
    }

    #[test]
    fn try_new_rejects_invalid_duration() {
        assert!(matches!(
            CalibrationSettings::try_new(10, 0.0, 0.5, 0.1, 0.1).unwrap_err(),
            CalibrationError::InvalidDuration(_)
        ));
        assert!(matches!(
            CalibrationSettings::try_new(10, f32::NAN, 0.5, 0.1, 0.1).unwrap_err(),
            CalibrationError::InvalidDuration(_)
        ));
    }

    #[test]
    fn try_new_rejects_confidence_out_of_range() {
        assert!(matches!(
            CalibrationSettings::try_new(10, 5.0, -0.1, 0.1, 0.1).unwrap_err(),
            CalibrationError::ConfidenceOutOfRange(_)
        ));
        assert!(matches!(
            CalibrationSettings::try_new(10, 5.0, 1.1, 0.1, 0.1).unwrap_err(),
            CalibrationError::ConfidenceOutOfRange(_)
        ));
    }

    #[test]
    fn try_new_rejects_invalid_motion_threshold() {
        assert!(matches!(
            CalibrationSettings::try_new(10, 5.0, 0.5, -0.1, 0.1).unwrap_err(),
            CalibrationError::InvalidMotionThreshold(_)
        ));
    }

    #[test]
    fn try_new_rejects_expression_motion_out_of_range() {
        assert!(matches!(
            CalibrationSettings::try_new(10, 5.0, 0.5, 0.1, 1.1).unwrap_err(),
            CalibrationError::ExpressionMotionOutOfRange(_)
        ));
    }

    #[test]
    fn default_pipeline_settings_are_valid() {
        let settings = TrackingPipelineSettings::new();
        assert_eq!(settings.version(), 1);
        assert_eq!(settings.calibration().version(), 1);
        assert_eq!(settings.calibration().required_sample_count(), 30);
    }

    #[test]
    fn pipeline_settings_default_matches_explicit() {
        let default = TrackingPipelineSettings::default();
        let explicit = TrackingPipelineSettings::new();
        assert_eq!(default, explicit);
    }

    #[test]
    fn pipeline_settings_with_calibration_persists_values() {
        let calibration = CalibrationSettings::try_new(10, 5.0, 0.5, 0.1, 0.1).unwrap();
        let settings = TrackingPipelineSettings::with_calibration(calibration.clone());
        assert_eq!(settings.calibration(), &calibration);
        assert_eq!(settings.version(), 1);
    }
}
