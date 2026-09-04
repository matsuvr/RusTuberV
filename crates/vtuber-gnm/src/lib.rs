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
//! Rust-side GNM Head v3 model boundary for sparse and selected-surface face-state evaluation.
//!
//! This crate deliberately stops at validated, engine-neutral GNM geometry,
//! observation, calibration, dynamic-state lifecycle, and temporal-energy
//! contracts. It does not contain a renderer, a Bevy system, or an avatar
//! retargeting policy; those belong to later Issue #50 leaves.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod aux_geometry;
mod dense;
mod dense_regions;
mod error;
mod identity_calibration;
mod landmarks;
mod lifecycle;
mod model;
mod npz;
mod reprojection;
mod single_frame_temporal;
mod temporal_regularization;

pub use aux_geometry::{
    BrowAuxFeatures, BrowSideAuxFeatures, CheekAuxFeatures, EyeApertureFeature, EyeAuxFeatures,
    GnmAuxGeometryError, GnmFacialFeatures, IrisAuxFeatures, IrisSideAuxFeature, MouthAuxFeatures,
    compute_brow_aux_features, compute_cheek_aux_features, compute_eye_aperture_features,
    compute_gnm_facial_features, compute_mouth_aux_features,
};
pub use dense::{
    AnatomicalSide, CorrespondenceProvenance, CorrespondenceReliability, DenseCorrespondenceSet,
    DenseCoveragePolicy, DenseCoverageSummary, DenseMappingVersion, DenseObservationStatus,
    FaceRegion, GnmDenseError, GnmDenseObservation, GnmDenseObservationPoint, GnmSurfacePointRef,
    MEDIAPIPE_FACE_LANDMARK_COUNT, MediaPipeGnmDenseCorrespondence, RepositoryDenseMapping,
    SPARSE_BOOTSTRAP_POINT_COUNT, canonicalize_mediapipe_xy, repository_dense_mapping,
    sparse_bootstrap_baseline,
};
pub use dense_regions::{
    BrowRows, CentralFaceRows, ContourRows, DenseRegionGroups, EXCLUDED_MEDIAPIPE_GROUPS,
    ExcludedMediaPipeGroup, EyeRegionRows, EyelidRing, IndexedRow, IrisRows, MouthRows,
    OtherValidatedRows, RegionSummary, topology,
};
pub use error::GnmModelError;
pub use identity_calibration::{
    FixedGnmIdentity, GnmIdentityCalibration, GnmIdentityCalibrationError, IdentityFitDiagnostics,
    NeutralCalibrationCandidate, NeutralCalibrationReadiness, NeutralCalibrationRejection,
    NeutralCalibrationRejectionReason, NeutralCalibrationSelection,
    NeutralCalibrationSelectionConfig, NeutralCalibrationWindowDiagnostics,
    NeutralNormalizationScales, NeutralPoseDiversity, NeutralSampleResidual,
    NeutralSampleSolveInput, SampleNuisance, SampleNuisanceBlock, SharedIdentityLinearSystem,
    SharedIdentitySolveConfig, SharedIdentitySolveInput, SharedIdentitySolveOutcome,
    assemble_shared_identity_linear_system, evaluate_neutral_sample_residual,
    finalize_identity_calibration, normalization_scales_from_mapping,
    select_neutral_calibration_candidates, solve_shared_identity, validate_shared_identity_solve,
};
pub use landmarks::{SparseLandmark, SparseLandmarkSet, head_sparse_68};
pub use lifecycle::{
    GnmFitInitialization, GnmFitOutcome, GnmFrameStamp, PersistentGnmAction, PersistentGnmEvent,
    PersistentGnmLifecycleConfig, PersistentGnmLifecycleDecision, PersistentGnmLifecycleError,
    PersistentGnmLifecycleState, PersistentGnmPhase, advance_persistent_gnm_lifecycle,
};
pub use model::{
    DenseArray, GNM_HEAD_V3_EXPRESSION_DIM, GNM_HEAD_V3_IDENTITY_DIM, GNM_HEAD_V3_VERSION,
    GnmExpressionState, GnmIdentityState, GnmJointState, GnmModel, GnmModelData, GnmSparseVertices,
    GnmVariant, GnmVersion, SparsePreparedVertices, SparseSkinningDerivatives,
};
pub use npz::{GNM_DATA_SCHEMA_KEYS, load_gnm_head_v3};
pub use reprojection::{
    AuxiliaryObjectiveTerm, AuxiliaryTermEvaluation, BlockJacobian, CasePlan, ConditioningBaseline,
    ConditioningStats, DenseExpressionJointStepConfig, DenseExpressionJointStepOutcome,
    DenseLinearization, DenseProjection, DenseReprojectionConfig, DenseReprojectionReport,
    DenseReprojectionResidual, DenseRigidStepConfig, DenseRigidStepOutcome, GnmReprojectionError,
    LinearizationStepSizes, MAX_SINGLE_FRAME_FIT_ITERATIONS, ReprojectionBlock,
    RigidRecoveryConfig, RigidRecoveryOutcome, SingleFrameFitConfig, SingleFrameFitOutcome,
    SingleFrameFitStatus, SynthesisOptions, SyntheticCase, compare_conditioning,
    evaluate_dense_reprojection, fit_single_frame_cold_start, fit_single_frame_with_temporal,
    fitting_projection, linearize_dense_reprojection, recover_rigid_projection,
    synthesize_observation_from_projection, take_dense_expression_joint_step,
    take_dense_rigid_step,
};
pub use single_frame_temporal::{
    CandidateTemporalScratch, SingleFrameTemporalPenalty, TemporalGroupLinearization,
    TemporalLinearization, candidate_state_view,
};
pub use temporal_regularization::{
    GnmTemporalNormalization, GnmTemporalStateView, TemporalGroupPenaltyMetrics,
    TemporalGroupPenaltyWeights, TemporalHistoryTiming, TemporalRegularizationConfig,
    TemporalRegularizationError, TemporalRegularizationInput, TemporalRegularizationMetrics,
    evaluate_temporal_regularization,
};
