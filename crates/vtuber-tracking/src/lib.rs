// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]
//! `vtuber-tracking`: calibration, pose solving, filtering, and tracking state.
//!
//! This crate must not depend on Bevy or `bevy_vrm1`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// Pure observed-arm retargeting and render-clock filtering.
pub mod arm_tracking;

/// Pure quality evaluation of a recorded arm sequence (Issue #50).
pub mod arm_evaluation;

/// Axis-selective root compensation and virtual head/body targets (#165).
pub mod body_targets;
/// Calibration: neutral reference collection and session state.
pub mod calibration;
/// Confidence synthesis and hysteresis gating.
pub mod confidence;
/// MediaPipe blendshape and gaze mapping.
pub mod expressions;
/// Per-eye closure judgement and profile validation (Issues #52/#53/#64/#66).
pub mod eye_closure;
/// Fixed MediaPipe lid geometry features (Issue #65).
pub mod eye_geometry;
/// Tracking filters: rotation smoothing and expression filtering.
pub mod filter;
/// Unified tracking-loss hold / return / reacquire ramp.
pub mod loss_blend;
/// Loss hold, neutral decay, and recovery blend.
pub mod loss_recovery;
/// Bounded procedural micro-motion after tracking loss (Issue #172).
pub mod micro_motion;
/// Neutral-relative head pose generation and tracking pipeline stages.
pub mod pipeline;
/// Placeholder for tracking subsystem.
pub mod placeholder;
/// Head pose estimation from landmark sets.
pub mod pose;
/// Explicit tracking state machine and transition table.
pub mod state_machine;
/// Neutral-relative head translation from webcam face geometry (Issue #166).
pub mod translation;
/// Scale-aware soft-cap and dt-aware filtering for translation (Issue #164).
pub mod translation_shaping;

pub use arm_evaluation::{ArmEvaluationSample, ArmEvaluationSummary, evaluate_arm_sequence};
pub use body_targets::{
    BodyTranslationCompensation, TranslationMeters, VirtualBodyProfile, VirtualBodyProfileError,
    VirtualBodyTargets, VirtualHeadTarget, build_virtual_body_targets,
};
pub use calibration::{
    AUTO_NEUTRAL_MIN_SAMPLES, AUTO_NEUTRAL_WINDOW, AutoNeutralCollector, AutoNeutralError,
    AutoNeutralState, AutoNeutralUpdate, CalibrationCollector, CalibrationInput,
    CalibrationSession, CollectorMetrics, GazeNeutralBaseline, NeutralContext, NeutralProfile,
    NeutralReference, NeutralValidationSettings, RejectionReason, SampleDecision,
};
pub use confidence::{
    ConfidenceAssessment, ConfidenceConfigError, ConfidenceError, ConfidenceGate,
    ConfidenceGateParams, ConfidenceInputs, ConfidencePolicies, ConfidenceSignal, ConfidenceSource,
    MissingSourcePolicy, synthesize,
};
pub use expressions::{
    BinocularGazeObservation, PerEyeGazeObservation, fuse_binocular_gaze,
    map_mediapipe_expressions, map_mediapipe_gaze, map_mediapipe_perfect_sync,
    map_mediapipe_raw_expressions, observe_mediapipe_gaze, parse_mediapipe_blendshapes,
};
pub use eye_closure::{
    EYE_CLOSURE_ALGORITHM_VERSION, EYE_CLOSURE_FEATURE, EYE_CLOSURE_PROFILE_SCHEMA_VERSION,
    EYE_CLOSURE_SAMPLE_GAP_NS, EyeClosureFingerprints, EyeClosureObservation,
    EyeClosureProfileDocument, EyeClosureProfileError, EyeClosureState, EyeClosureThresholds,
    EyeClosureTracker, EyeClosureVerificationStatus, EyeGeometryThreshold,
    EyeGeometryThresholdValues, EyeGeometryThresholds, EyeOpenness, EyeSampleStep, EyeSide,
    EyeThreshold, EyeThresholdError, GeometryEyeClosureTracker, classify_eye_sample,
    decide_geometry_eye, decide_geometry_pair,
};
pub use eye_geometry::{
    EyeClosureFeatures, EyeGeometryError, LidPoints, eye_closure_features, max_lid_gap_ratio,
    mediapipe_lid_points,
};
pub use filter::{
    DetailedExpressionFilter, ExpressionCalibration, ExpressionCalibrationError, ExpressionChannel,
    ExpressionFilter, ExpressionFilterParams, ExpressionRange, GazeFilter, GazeFilterParams,
    HeadFilterParams, HeadRotationFilter, MissingChannelFallback, MissingChannelPolicy,
};
pub use loss_blend::{
    LossBlend, LossBlendConfigError, LossBlendProfile, MAX_ACQUIRE_DURATION, MAX_HOLD_DURATION,
    MAX_RETURN_DURATION, MIN_ACQUIRE_DURATION, MIN_HOLD_DURATION, MIN_RETURN_DURATION,
    acquire_factor, loss_return_factor,
};
pub use loss_recovery::{LossRecovery, LossRecoveryConfigError};
pub use micro_motion::{
    IdleTarget, MicroMotionBlender, MicroMotionProfile, MicroMotionProfileError,
    blended_idle_target, idle_target, is_tracked_state,
};
pub use pipeline::{
    HeadPoseFailure, HeadPoseFrame, PipelineConfig, PipelineConfigError, PipelineUpdate,
    PoseFailureReason, TrackingPipeline, compute_neutral_relative_pose,
};
pub use pose::mediapipe::{
    MediaPipePoseError, RelativeFaceTransform, mediapipe_to_application_basis, relative_pose,
    relative_transform,
};
pub use pose::planar::{
    CANONICAL_FACE_TEMPLATE, CanonicalFacePoint, PlanarCorrespondence, PlanarLandmark,
    PlanarPoseAlignment, PlanarPoseError, solve_planar_pose,
};
pub use pose::{LandmarkSet, PoseAlignment, PoseError, WeightedPoint, solve_relative_pose};
pub use state_machine::{
    StateMachineConfigError, StateMachineParams, StateTransitionResult, TrackingAction,
    TrackingStateMachine, TransitionInput,
};
pub use translation::{
    MEDIAPIPE_TRANSFORM_UNITS_TO_METERS, REFERENCE_DISTANCE_METERS, REFERENCE_HEAD_RADIUS_METERS,
    signal_from_alignment, signal_from_face_transform,
};
pub use translation_shaping::{
    FilterConfigError, ShapingProfileError, TranslationFilter, TranslationShapingProfile,
    shape_translation, soft_cap_scalar,
};
