//! Calibration domain types for `vtuber-tracking`.
//!
//! This module contains the per-frame input to calibration, the saved neutral
//! profile, and the calibration session state machine. It does not depend on
//! Bevy, camera APIs, or `bevy_vrm1`.

use vtuber_core::control::CalibrationError;
use vtuber_core::types::{
    FrameSeq, HeadPose, Landmark3, LandmarkSchemaId, MonoTimeNs, RawExpressionObservation,
};

use crate::calibration::GazeNeutralBaseline;

/// A single neutral candidate frame supplied to the calibration collector.
///
/// `CalibrationInput` is deliberately separate from [`NeutralProfile`]: the
/// input is raw per-frame data, while the profile is an aggregated and
/// validated neutral reference that may outlive the session.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationInput {
    /// Sequence number of the source video frame.
    pub source_seq: FrameSeq,
    /// When the source frame was captured.
    pub captured_at: MonoTimeNs,
    /// Overall face confidence in `[0, 1]`.
    pub face_confidence: f32,
    /// Facial landmarks captured in this frame.
    pub landmarks: Vec<Landmark3>,
    /// Raw expression coefficients captured in this frame.
    pub expressions: RawExpressionObservation,
    /// Landmark schema used by `landmarks`.
    pub schema: LandmarkSchemaId,
}

/// A validated neutral reference produced by calibration.
///
/// This profile can be persisted, but the fields are intentionally
/// model-agnostic: it stores landmark positions, a baseline head pose, and
/// expression baselines rather than avatar-specific coefficients.
#[derive(Clone, Debug, PartialEq)]
pub struct NeutralProfile {
    /// Profile format version.
    pub version: u32,
    /// Landmark schema that was used to build the profile.
    pub schema: LandmarkSchemaId,
    /// Stable neutral landmarks (typically a median or robust mean).
    pub landmarks: Vec<Landmark3>,
    /// Baseline head pose at calibration time.
    pub head_pose: HeadPose,
    /// Per-eye forward-looking gaze baseline. Version-1 profiles migrate as zero.
    pub gaze_baseline: GazeNeutralBaseline,
    /// Left eye blink baseline in `[0, 1]`.
    pub blink_left_baseline: f32,
    /// Right eye blink baseline in `[0, 1]`.
    pub blink_right_baseline: f32,
    /// Mouth openness baseline in `[0, 1]`.
    pub mouth_open_baseline: f32,
    /// Face scale estimate used to normalize pose solving.
    pub face_scale: f32,
    /// Aggregated face confidence baseline in `[0, 1]`.
    pub confidence_baseline: f32,
    /// When the profile was finalized.
    pub collected_at: MonoTimeNs,
    /// Hash of the VRM model the profile was collected for, if known.
    pub model_hash: Option<String>,
    /// Fingerprint of the camera used during collection, if known.
    pub camera_fingerprint: Option<String>,
}

impl NeutralProfile {
    /// Returns `true` if this profile may be reused for the given model hash.
    ///
    /// When both sides provide a hash, they must match exactly.  When either
    /// side lacks a hash, compatibility cannot be verified and the profile is
    /// allowed; this preserves legacy profiles collected before model-hash
    /// tracking was added while still blocking definitely-different hashes.
    #[must_use]
    pub fn is_compatible_with(&self, model_hash: Option<&str>) -> bool {
        match (&self.model_hash, model_hash) {
            (Some(a), Some(b)) => a == b,
            _ => true,
        }
    }
}

/// Lifecycle of a single calibration session.
///
/// The session state is independent of any UI or Bevy state. Transitions are
/// explicit and return [`CalibrationError`] for illegal moves.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum CalibrationSession {
    /// No calibration has been started for the current avatar.
    #[default]
    NotStarted,
    /// Samples are being collected.
    Collecting {
        /// When the session started.
        started_at: MonoTimeNs,
        /// How many valid samples have been accepted so far.
        samples_collected: usize,
    },
    /// Enough samples were collected and a profile is available to commit.
    Ready {
        /// Validated neutral profile.
        profile: NeutralProfile,
    },
    /// The session failed validation and no profile was produced.
    Rejected {
        /// Why the session was rejected.
        reason: CalibrationError,
    },
    /// The profile has been committed as the active neutral reference.
    Completed {
        /// Committed neutral profile.
        profile: NeutralProfile,
    },
}

impl CalibrationSession {
    /// Human-readable state name for logging and error reporting.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::NotStarted => "NotStarted",
            Self::Collecting { .. } => "Collecting",
            Self::Ready { .. } => "Ready",
            Self::Rejected { .. } => "Rejected",
            Self::Completed { .. } => "Completed",
        }
    }

    /// Start a new collection phase.
    ///
    /// # Errors
    ///
    /// Returns [`CalibrationError::InvalidStateTransition`] if the session is
    /// not currently [`NotStarted`](Self::NotStarted).
    pub fn start(&self, now: MonoTimeNs) -> Result<Self, CalibrationError> {
        match self {
            Self::NotStarted => Ok(Self::Collecting {
                started_at: now,
                samples_collected: 0,
            }),
            _ => Err(CalibrationError::InvalidStateTransition {
                from: self.kind(),
                to: "Collecting",
            }),
        }
    }

    /// Mark the current collecting session as rejected.
    ///
    /// # Errors
    ///
    /// Returns [`CalibrationError::InvalidStateTransition`] if the session is
    /// not currently [`Collecting`](Self::Collecting).
    pub fn reject(&self, reason: CalibrationError) -> Result<Self, CalibrationError> {
        match self {
            Self::Collecting { .. } => Ok(Self::Rejected { reason }),
            _ => Err(CalibrationError::InvalidStateTransition {
                from: self.kind(),
                to: "Rejected",
            }),
        }
    }

    /// Move from collecting to ready once a profile has been validated.
    ///
    /// # Errors
    ///
    /// Returns [`CalibrationError::InvalidStateTransition`] if the session is
    /// not currently [`Collecting`](Self::Collecting).
    pub fn ready(&self, profile: NeutralProfile) -> Result<Self, CalibrationError> {
        match self {
            Self::Collecting { .. } => Ok(Self::Ready { profile }),
            _ => Err(CalibrationError::InvalidStateTransition {
                from: self.kind(),
                to: "Ready",
            }),
        }
    }

    /// Commit the ready profile.
    ///
    /// # Errors
    ///
    /// Returns [`CalibrationError::InvalidStateTransition`] if the session is
    /// not currently [`Ready`](Self::Ready).
    pub fn complete(&self) -> Result<Self, CalibrationError> {
        match self {
            Self::Ready { profile } => Ok(Self::Completed {
                profile: profile.clone(),
            }),
            _ => Err(CalibrationError::InvalidStateTransition {
                from: self.kind(),
                to: "Completed",
            }),
        }
    }

    /// Reset the session back to [`NotStarted`](Self::NotStarted).
    ///
    /// This is allowed from any state.
    pub fn reset(&self) -> Result<Self, CalibrationError> {
        Ok(Self::NotStarted)
    }

    /// Return the committed or ready profile, if any.
    #[must_use]
    pub fn profile(&self) -> Option<&NeutralProfile> {
        match self {
            Self::Ready { profile } | Self::Completed { profile } => Some(profile),
            _ => None,
        }
    }

    /// Return the rejection reason, if the session was rejected.
    #[must_use]
    pub fn rejection_reason(&self) -> Option<&CalibrationError> {
        match self {
            Self::Rejected { reason } => Some(reason),
            _ => None,
        }
    }
}

#[cfg(test)]
mod calibration_types {
    use super::*;
    use vtuber_core::control::CalibrationSettings;

    fn dummy_profile() -> NeutralProfile {
        NeutralProfile {
            version: 2,
            schema: LandmarkSchemaId("test"),
            landmarks: vec![
                Landmark3 {
                    x: 0.5,
                    y: 0.5,
                    z: 0.0,
                    visibility: 1.0,
                },
                Landmark3 {
                    x: 0.6,
                    y: 0.5,
                    z: 0.0,
                    visibility: 1.0,
                },
                Landmark3 {
                    x: 0.55,
                    y: 0.6,
                    z: 0.0,
                    visibility: 1.0,
                },
            ],
            head_pose: HeadPose::default(),
            gaze_baseline: GazeNeutralBaseline::default(),
            blink_left_baseline: 0.1,
            blink_right_baseline: 0.1,
            mouth_open_baseline: 0.05,
            face_scale: 1.0,
            confidence_baseline: 0.9,
            collected_at: MonoTimeNs(1_000_000_000),
            model_hash: None,
            camera_fingerprint: None,
        }
    }

    #[test]
    fn default_settings_are_valid_and_documented() {
        let settings = CalibrationSettings::default();
        assert_eq!(settings.version(), 1);
        assert_eq!(settings.required_sample_count(), 30);
        assert_eq!(settings.max_duration_seconds(), 5.0);
        assert_eq!(settings.min_confidence(), 0.5);
        assert!((settings.max_head_motion_rad() - 5.0f32.to_radians()).abs() < 1e-4);
        assert_eq!(settings.max_expression_motion(), 0.15);
    }

    #[test]
    fn invalid_settings_are_rejected_by_constructor() {
        assert!(CalibrationSettings::try_new(2, 5.0, 0.5, 0.1, 0.1).is_err());
        assert!(CalibrationSettings::try_new(10, 0.0, 0.5, 0.1, 0.1).is_err());
        assert!(CalibrationSettings::try_new(10, 5.0, -0.1, 0.1, 0.1).is_err());
        assert!(CalibrationSettings::try_new(10, 5.0, 0.5, -0.1, 0.1).is_err());
        assert!(CalibrationSettings::try_new(10, 5.0, 0.5, 0.1, 1.1).is_err());
    }

    #[test]
    fn calibration_input_and_profile_are_distinct() {
        let input = CalibrationInput {
            source_seq: FrameSeq(1),
            captured_at: MonoTimeNs(0),
            face_confidence: 0.9,
            landmarks: vec![Landmark3::default()],
            expressions: RawExpressionObservation::default(),
            schema: LandmarkSchemaId("input"),
        };
        let profile = dummy_profile();

        assert_ne!(input.schema, profile.schema);
        assert_eq!(input.landmarks.len(), 1);
        assert_eq!(profile.landmarks.len(), 3);
    }

    #[test]
    fn session_starts_from_not_started() {
        let session = CalibrationSession::default();
        let next = session.start(MonoTimeNs(0)).unwrap();
        assert_eq!(next.kind(), "Collecting");
        assert!(matches!(
            next,
            CalibrationSession::Collecting {
                started_at: MonoTimeNs(0),
                samples_collected: 0,
            }
        ));
    }

    #[test]
    fn session_cannot_start_when_already_started() {
        let session = CalibrationSession::default().start(MonoTimeNs(0)).unwrap();
        let err = session.start(MonoTimeNs(1)).unwrap_err();
        assert_eq!(err.code(), "CALIBRATION_INVALID_STATE_TRANSITION");
    }

    #[test]
    fn session_can_be_rejected_while_collecting() {
        let session = CalibrationSession::default().start(MonoTimeNs(0)).unwrap();
        let reason = CalibrationError::InsufficientSamples(2);
        let next = session.reject(reason).unwrap();
        assert_eq!(next.kind(), "Rejected");
        assert_eq!(
            next.rejection_reason().unwrap().code(),
            "CALIBRATION_INSUFFICIENT_SAMPLES"
        );
    }

    #[test]
    fn session_becomes_ready_from_collecting() {
        let session = CalibrationSession::default().start(MonoTimeNs(0)).unwrap();
        let profile = dummy_profile();
        let next = session.ready(profile.clone()).unwrap();
        assert_eq!(next.kind(), "Ready");
        assert_eq!(next.profile().unwrap().schema, profile.schema);
    }

    #[test]
    fn session_completes_from_ready() {
        let profile = dummy_profile();
        let session = CalibrationSession::default()
            .start(MonoTimeNs(0))
            .unwrap()
            .ready(profile)
            .unwrap()
            .complete()
            .unwrap();
        assert_eq!(session.kind(), "Completed");
        assert!(session.profile().is_some());
    }

    #[test]
    fn session_cannot_complete_directly_from_collecting() {
        let session = CalibrationSession::default().start(MonoTimeNs(0)).unwrap();
        let err = session.complete().unwrap_err();
        assert_eq!(err.code(), "CALIBRATION_INVALID_STATE_TRANSITION");
    }

    #[test]
    fn session_reset_returns_to_not_started() {
        let profile = dummy_profile();
        let states = [
            CalibrationSession::default(),
            CalibrationSession::default().start(MonoTimeNs(0)).unwrap(),
            CalibrationSession::default()
                .start(MonoTimeNs(0))
                .unwrap()
                .ready(profile.clone())
                .unwrap(),
            CalibrationSession::default()
                .start(MonoTimeNs(0))
                .unwrap()
                .ready(profile.clone())
                .unwrap()
                .complete()
                .unwrap(),
            CalibrationSession::default()
                .start(MonoTimeNs(0))
                .unwrap()
                .reject(CalibrationError::InsufficientSamples(2))
                .unwrap(),
        ];
        for state in states {
            let reset = state.reset().unwrap();
            assert_eq!(reset.kind(), "NotStarted");
        }
    }
}
