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

/// Source-aligned A/B backend contract and fallback arbitration.
pub mod ab_backend;
/// Robustness, cross-talk, and runtime-cost A/B report kernels (GNM #57.5).
pub mod ab_report;
/// Pure motion/confidence-adaptive temporal-weight policy.
pub mod adaptive_temporal;
/// ARKit teacher dataset schema and privacy boundary (GNM #68.1).
pub mod arkit_teacher;
/// Avatar-output authority gate applying backend arbitration to published
/// ARKit52 coefficients (GNM #57.3).
pub mod authority_gate;
/// Optional MediaPipe semantic observations used only as an auxiliary fitting term.
pub mod auxiliary_expression;
/// Geometry-to-auxiliary-residual adapter for one exact frame.
pub mod auxiliary_geometry;
/// Axis-selective root compensation and virtual head/body targets (#165).
pub mod body_targets;
/// Calibration: neutral reference collection and session state.
pub mod calibration;
/// Causal history feature/target dataset generation (GNM #68.4a).
pub mod causal_dataset;
/// Linear AR prior fitting and versioned artifact export (GNM #68.4b).
pub mod causal_prior;
/// Load/inference boundary for exported linear priors (GNM #68.4c).
pub mod causal_prior_inference;
/// Bounded prior corrections, resets, and causality regression (GNM #68.4d).
pub mod causal_prior_runtime;
/// Confidence synthesis and hysteresis gating.
pub mod confidence;
/// MediaPipe blendshape and gaze mapping.
pub mod expressions;
/// Tracking filters: rotation smoothing and expression filtering.
pub mod filter;
/// Region projectors from the GNM facial feature snapshot onto canonical
/// ARKit52 blendshape channels.
pub mod gnm_arkit_projector;
/// Per-frame fitter input, dynamic state, config, and result contracts.
pub mod gnm_fitter_contract;
/// Latest-frame worker connection publishing lifecycle-gated GNM state.
pub mod gnm_latest_frame_worker;
/// Persistent fitter warm-start glue between lifecycle directives and the
/// bounded single-frame solver.
pub mod gnm_persistent_fitter;
/// Reduced-GNM semantic decoder datasets and artifacts (Issue #17).
pub mod gnm_semantic_decoder;
/// Deterministic synthetic-sequence regression harness for the persistent
/// GNM fitter (test infrastructure; see issue #94).
pub mod gnm_sequence_regression;
pub mod gnm_shadow;
/// Loss hold, neutral decay, and recovery blend.
pub mod loss_recovery;
/// Bounded procedural micro-motion after tracking loss (Issue #172).
pub mod micro_motion;
/// Observable non-tongue GNM expression basis artifacts (Issue #15).
pub mod observable_basis;
/// Neutral-relative head pose generation and tracking pipeline stages.
pub mod pipeline;
/// Placeholder for tracking subsystem.
pub mod placeholder;
/// Head pose estimation from landmark sets.
pub mod pose;
pub mod same_frame_fanout;
/// Explicit tracking state machine and transition table.
pub mod state_machine;
/// Teacher-residual-aligned observable GNM basis artifacts (Issue #16).
pub mod teacher_aligned_basis;
/// Same-frame teacher-minus-Direct residual dataset and decoder.
pub mod teacher_residual;
/// Timestamp-aware pure metrics for temporal tracking quality.
pub mod temporal_metrics;
/// Direct/GNM temporal quality report composition over the metric kernels.
pub mod temporal_report;
/// Neutral-relative head translation from webcam face geometry (Issue #166).
pub mod translation;
/// Scale-aware soft-cap and dt-aware filtering for translation (Issue #164).
pub mod translation_shaping;

pub use ab_backend::{
    AbBackendError, AlignedBackendOutputs, AlignedLatencyComparison, BackendLatencyMetrics,
    BackendOutputTiming, BackendSelectionConfig, BackendSelectionDecision, BackendSelectionState,
    FaceTrackingBackend, FaceTrackingMode, GnmFallbackReason, GnmRuntimeHealth, GnmTransientIssue,
    GnmUnavailableReason, SourceFrameStamp, StampedBackendOutput, advance_backend_selection,
    backend_latency_metrics,
};
pub use ab_report::{
    AbMeasuredInputs, CrossTalkMetrics, PromotionBlocker, PromotionCriteria, PromotionDecision,
    PromotionVerdict, RobustnessMetrics, crosstalk_metrics, promotion_verdict, robustness_metrics,
};
pub use adaptive_temporal::{
    AdaptiveTemporalConfig, AdaptiveTemporalError, AdaptiveTemporalInput, AdaptiveTemporalRegime,
    AdaptiveTemporalState, TemporalGroupWeights, TemporalObservationHealth,
    advance_adaptive_temporal_policy,
};
pub use arkit_teacher::{
    ARKIT_TEACHER_DATASET_SCHEMA_VERSION, ArkitTeacherFrame, DeterministicGnmState,
    GnmTeacherStateRecord, HeadTransform, MediaPipeTeacherObservation, PairedTemporalSample,
    RgbFrameReference, TeacherDatasetError, validate_paired_samples,
};
pub use authority_gate::{AuthorityGate, AuthorityOutcome};
pub use auxiliary_expression::{
    AuxChannelReliability, AuxiliaryChannelConfig, AuxiliaryExpressionChannel,
    AuxiliaryExpressionError, AuxiliaryExpressionGroup, AuxiliaryExpressionObservation,
    AuxiliaryExpressionSemantic, AuxiliaryExpressionStatus, AuxiliaryGroupResiduals,
    AuxiliaryLossConfig, AuxiliaryLossDiagnostics, AuxiliaryNeutralCalibration,
    PredictedAuxiliaryFeature, evaluate_auxiliary_expression_loss,
    validate_auxiliary_source_alignment,
};
pub use auxiliary_geometry::{AuxiliaryGeometryFeatures, GeometryAuxiliaryObjective};
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
pub use causal_dataset::{
    CausalDataset, CausalFeatureConfig, CausalRow, ExclusionReason, build_causal_dataset,
};
pub use causal_prior::{
    LinearPriorArtifact, LinearPriorFitError, LinearPriorTrainingConfig, fit_linear_prior,
    hash_artifact,
};
pub use causal_prior_inference::{LinearPriorLoadError, LoadedLinearPrior, PriorInference};
pub use causal_prior_runtime::{
    CorrectionGroup, PriorRuntime, PriorRuntimeConfig, PriorRuntimeError, PriorStepOutcome,
    ResetReason,
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
pub use filter::{
    DetailedExpressionFilter, ExpressionCalibration, ExpressionCalibrationError, ExpressionChannel,
    ExpressionFilter, ExpressionFilterParams, ExpressionRange, GazeFilter, GazeFilterParams,
    HeadFilterParams, HeadRotationFilter, MissingChannelFallback, MissingChannelPolicy,
};
pub use gnm_arkit_projector::{
    Arkit52DecodeResult, BrowCheekNoseProjection, EyeGazeProjection, JawCoreMouthProjection,
    LipCornerProjection, LipLowerUpperProjection, LipMouthCornerProjectorResult,
    LipRollShrugPressProjection, ProjectedChannel, ProjectedSupport, decode_gnm_arkit52,
    project_brow_cheek_nose_channels, project_eye_gaze_channels, project_jaw_core_mouth_channels,
    project_lip_corner_channels, project_lip_lower_upper_channels,
    project_lip_mouth_corner_channels, project_lip_roll_shrug_press_channels,
};
pub use gnm_fitter_contract::{
    GnmCameraBlock, GnmDynamicState, GnmFitResult, GnmFitStatus, GnmFitterContractError,
    GnmRigidPoseBlock, GnmSolverConfig, GnmSolverFrameInput, MAX_SOLVER_ITERATIONS_BOUND,
};
pub use gnm_latest_frame_worker::{
    GnmFaceState, GnmFitterResources, GnmFitterWorkerMetrics, GnmLatestFrameWorker,
    GnmWorkerFrameInput, GnmWorkerInput, GnmWorkerStep, spawn_gnm_latest_frame_worker,
};
pub use gnm_persistent_fitter::{
    GnmSolvedFrameReport, GnmValidatedDynamicFrame, PersistentGnmFitter, PersistentGnmFitterError,
    PersistentGnmFrameOutcome,
};
pub use gnm_semantic_decoder::{
    GNM_DIAGNOSTIC_REGION_ORDER, GNM_SEMANTIC_DECODER_SCHEMA_VERSION, GnmSemanticDatasetError,
    GnmSemanticDecoderArtifact, GnmSemanticDecoderKind, GnmSemanticFeatureConfig,
    GnmSemanticFitError, GnmSemanticFrame, GnmSemanticRow, build_gnm_semantic_features,
    build_gnm_semantic_rows, fit_gnm_semantic_decoder, gnm_only_prediction_to_arkit52,
    gnm_semantic_feature_order, gnm_semantic_frame_from_sample, predict_gnm_semantic_raw,
};
pub use gnm_sequence_regression::{synthetic_head_model, synthetic_mapping};
pub use gnm_shadow::{
    GnmShadowCandidate, GnmShadowOutcome, GnmShadowSkip, align_shadow_pair, decode_shadow_features,
    shadow_worker_input,
};
pub use loss_recovery::{
    LossRecovery, LossRecoveryConfigError, LossRecoveryParams, MAX_DECAY_DURATION,
    MAX_GLIDE_DURATION, MAX_RECOVERY_DURATION, MIN_DECAY_DURATION, MIN_GLIDE_DURATION,
    MIN_RECOVERY_DURATION,
};
pub use micro_motion::{
    IdleTarget, MicroMotionBlender, MicroMotionProfile, MicroMotionProfileError,
    blended_idle_target, idle_target, is_tracked_state,
};
pub use observable_basis::{
    OBSERVABLE_GNM_BASIS_SCHEMA_VERSION, ObservableBasisError, ObservableBasisProvenance,
    ObservableGnmBasisArtifact, fit_observable_gnm_basis, project_non_tongue_expression,
    reconstruct_non_tongue_expression,
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
pub use same_frame_fanout::{SameFrameFanOut, SameFrameFanOutError, fan_out_same_frame};
pub use state_machine::{
    StateMachineConfigError, StateMachineParams, StateTransitionResult, TrackingAction,
    TrackingStateMachine, TransitionInput,
};
pub use teacher_aligned_basis::{
    TEACHER_ALIGNED_GNM_BASIS_SCHEMA_VERSION, TeacherAlignedBasisError,
    TeacherAlignedGnmBasisArtifact, TeacherAlignmentSample, build_teacher_alignment_samples,
    fit_teacher_aligned_gnm_basis, project_teacher_aligned_expression,
    reconstruct_teacher_aligned_expression,
};
pub use teacher_residual::{
    ExistingTraceResidualVariants, LoadedTeacherResidualDecoder, NormalizedLinearMapArtifact,
    ResidualAblationError, TEACHER_RESIDUAL_DECODER_SCHEMA_VERSION, TEACHER_RESIDUAL_FEATURE_ORDER,
    TeacherResidualDataset, TeacherResidualDecoderArtifact, TeacherResidualExclusion,
    TeacherResidualFeatureConfig, TeacherResidualHistory, TeacherResidualRow,
    apply_non_tongue_residual, build_teacher_residual_rows, existing_trace_residual_variants,
    fit_teacher_residual_decoder, predict_teacher_residual,
};
pub use temporal_metrics::{
    CheekTakeEvaluation, PulseResponseMetrics, PulseResponseSpec, StepResponseMetrics,
    StepResponseSpec, TemporalMetricError, TemporalNoiseMetrics, TemporalSample, TemporalTrace,
    cheek_hold_detected, cheek_takes_are_disjoint, pulse_response_metrics, step_response_metrics,
    temporal_noise_metrics,
};
pub use temporal_report::{
    BackendTemporalQuality, ChannelTemporalQuality, TemporalChannelSpecs, backend_temporal_quality,
    channel_temporal_quality,
};
pub use translation::{
    MEDIAPIPE_TRANSFORM_UNITS_TO_METERS, REFERENCE_DISTANCE_METERS, REFERENCE_HEAD_RADIUS_METERS,
    signal_from_alignment, signal_from_face_transform,
};
pub use translation_shaping::{
    FilterConfigError, ShapingProfileError, TranslationFilter, TranslationShapingProfile,
    shape_translation, soft_cap_scalar,
};
