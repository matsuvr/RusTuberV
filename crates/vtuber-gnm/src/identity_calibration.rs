//! Pure neutral-calibration selection and immutable identity output contract.
//!
//! This module does not implement the numerical multi-frame identity solve. It
//! establishes the parts that are independent of that solver: sample admission,
//! pose-diversity diagnostics, model/mapping version binding, and a structurally
//! read-only identity object for later tracking.

use crate::{
    DenseCorrespondenceSet, DenseCoverageSummary, DenseMappingVersion, DenseObservationStatus,
    DenseProjection, DenseRegionGroups, DenseReprojectionConfig, DenseReprojectionReport,
    GnmDenseObservation, GnmExpressionState, GnmIdentityState, GnmJointState, GnmModel,
    GnmReprojectionError, GnmSparseVertices, GnmVersion, evaluate_dense_reprojection,
};

/// Summary of one neutral-window candidate used for deterministic sample selection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeutralCalibrationCandidate {
    /// Monotonic source-frame sequence.
    pub source_seq: u64,
    /// Monotonic capture timestamp in microseconds.
    pub captured_at_micros: u64,
    /// Dense observation coverage from Issue #53.
    pub coverage: DenseCoverageSummary,
    /// Candidate reprojection RMS in normalized image coordinates.
    pub reprojection_rms: f32,
    /// Optional normalized expression-activity proxy in `[0, 1]`.
    /// Absence means unavailable, not neutral and not expressive.
    pub expression_activity: Option<f32>,
    /// Pose nuisance estimate used only for diversity diagnostics.
    pub yaw_radians: f32,
    /// Pose nuisance estimate used only for diversity diagnostics.
    pub pitch_radians: f32,
    /// Whether upstream tracking marked this candidate degraded/lost.
    pub tracking_degraded: bool,
}

/// Typed reason a neutral calibration candidate was not admitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NeutralCalibrationRejectionReason {
    /// Source sequence duplicated an earlier candidate.
    DuplicateSourceSequence,
    /// Source sequence regressed.
    RegressedSourceSequence,
    /// Capture timestamp failed strict monotonicity.
    RegressedTimestamp,
    /// Dense observation coverage was insufficient.
    InsufficientDenseCoverage,
    /// Upstream lifecycle marked the sample degraded/lost.
    DegradedTracking,
    /// One or more candidate metrics were non-finite or inconsistent.
    InvalidMetrics,
    /// Reprojection residual exceeded the configured bound.
    ExcessiveReprojectionResidual,
    /// Optional expression-activity proxy exceeded the configured neutral-window bound.
    ExpressionContamination,
}

/// Rejection record retaining the candidate index and source identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NeutralCalibrationRejection {
    /// Index in the input candidate slice.
    pub candidate_index: usize,
    /// Candidate source sequence.
    pub source_seq: u64,
    /// Typed rejection reason.
    pub reason: NeutralCalibrationRejectionReason,
}

/// Typed readiness of the selected calibration window before numerical identity fitting.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NeutralCalibrationReadiness {
    /// Too few accepted candidates remain.
    InsufficientSamples,
    /// Accepted samples are too near-identical in yaw/pitch to claim useful diversity.
    InsufficientPoseDiversity,
    /// Selection gates pass and the accepted dense observations may enter the identity solver.
    ReadyForIdentitySolve,
}

/// Pose-diversity diagnostics over accepted candidates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeutralPoseDiversity {
    /// Accepted yaw span in radians.
    pub yaw_span_radians: f32,
    /// Accepted pitch span in radians.
    pub pitch_span_radians: f32,
    /// Fraction of accepted samples after the first that are near-duplicates of
    /// the previous accepted yaw/pitch estimate.
    pub near_duplicate_fraction: f32,
}

/// Aggregate pre-solve diagnostics for a neutral candidate window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeutralCalibrationWindowDiagnostics {
    /// Total candidates examined.
    pub total_candidates: usize,
    /// Candidates admitted to the numerical identity solve.
    pub accepted_candidates: usize,
    /// Candidates rejected by typed gates.
    pub rejected_candidates: usize,
    /// Pose diversity over accepted candidates.
    pub pose_diversity: NeutralPoseDiversity,
    /// Readiness after count/diversity checks.
    pub readiness: NeutralCalibrationReadiness,
}

/// Deterministic selection result; accepted indices point back to caller-owned dense observations.
#[derive(Clone, Debug, PartialEq)]
pub struct NeutralCalibrationSelection {
    /// Input indices accepted for the future shared-identity solve.
    pub accepted_indices: Vec<usize>,
    /// Typed rejection records.
    pub rejections: Vec<NeutralCalibrationRejection>,
    /// Aggregate window diagnostics.
    pub diagnostics: NeutralCalibrationWindowDiagnostics,
}

/// Typed thresholds for neutral candidate selection and pose-diversity diagnostics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeutralCalibrationSelectionConfig {
    min_accepted_samples: usize,
    max_reprojection_rms: f32,
    max_expression_activity: f32,
    min_pose_span_radians: f32,
    near_duplicate_pose_distance_radians: f32,
    max_near_duplicate_fraction: f32,
}

impl NeutralCalibrationSelectionConfig {
    /// Creates a selection configuration without hidden thresholds.
    pub fn new(
        min_accepted_samples: usize,
        max_reprojection_rms: f32,
        max_expression_activity: f32,
        min_pose_span_radians: f32,
        near_duplicate_pose_distance_radians: f32,
        max_near_duplicate_fraction: f32,
    ) -> Result<Self, GnmIdentityCalibrationError> {
        if min_accepted_samples == 0 {
            return Err(GnmIdentityCalibrationError::InvalidSelectionConfig(
                "min_accepted_samples must be positive",
            ));
        }
        for (field, value) in [
            ("max_reprojection_rms", max_reprojection_rms),
            ("min_pose_span_radians", min_pose_span_radians),
            (
                "near_duplicate_pose_distance_radians",
                near_duplicate_pose_distance_radians,
            ),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(GnmIdentityCalibrationError::InvalidSelectionConfig(field));
            }
        }
        if !max_expression_activity.is_finite() || !(0.0..=1.0).contains(&max_expression_activity) {
            return Err(GnmIdentityCalibrationError::InvalidSelectionConfig(
                "max_expression_activity must be within [0, 1]",
            ));
        }
        if !max_near_duplicate_fraction.is_finite()
            || !(0.0..=1.0).contains(&max_near_duplicate_fraction)
        {
            return Err(GnmIdentityCalibrationError::InvalidSelectionConfig(
                "max_near_duplicate_fraction must be within [0, 1]",
            ));
        }
        Ok(Self {
            min_accepted_samples,
            max_reprojection_rms,
            max_expression_activity,
            min_pose_span_radians,
            near_duplicate_pose_distance_radians,
            max_near_duplicate_fraction,
        })
    }
}

/// Selects neutral calibration samples without consulting MediaPipe blendshapes as authority.
pub fn select_neutral_calibration_candidates(
    candidates: &[NeutralCalibrationCandidate],
    config: NeutralCalibrationSelectionConfig,
) -> NeutralCalibrationSelection {
    let mut accepted_indices = Vec::new();
    let mut rejections = Vec::new();
    let mut last_seen: Option<(u64, u64)> = None;

    for (candidate_index, candidate) in candidates.iter().enumerate() {
        let reason = sequence_rejection(last_seen, *candidate)
            .or_else(|| candidate_metric_rejection(*candidate, config));
        last_seen = Some((candidate.source_seq, candidate.captured_at_micros));
        if let Some(reason) = reason {
            rejections.push(NeutralCalibrationRejection {
                candidate_index,
                source_seq: candidate.source_seq,
                reason,
            });
        } else {
            accepted_indices.push(candidate_index);
        }
    }

    let pose_diversity = pose_diversity(candidates, &accepted_indices, config);
    let readiness = if accepted_indices.len() < config.min_accepted_samples {
        NeutralCalibrationReadiness::InsufficientSamples
    } else if pose_diversity
        .yaw_span_radians
        .max(pose_diversity.pitch_span_radians)
        < config.min_pose_span_radians
        || pose_diversity.near_duplicate_fraction > config.max_near_duplicate_fraction
    {
        NeutralCalibrationReadiness::InsufficientPoseDiversity
    } else {
        NeutralCalibrationReadiness::ReadyForIdentitySolve
    };

    NeutralCalibrationSelection {
        diagnostics: NeutralCalibrationWindowDiagnostics {
            total_candidates: candidates.len(),
            accepted_candidates: accepted_indices.len(),
            rejected_candidates: rejections.len(),
            pose_diversity,
            readiness,
        },
        accepted_indices,
        rejections,
    }
}

fn sequence_rejection(
    previous: Option<(u64, u64)>,
    candidate: NeutralCalibrationCandidate,
) -> Option<NeutralCalibrationRejectionReason> {
    let (previous_seq, previous_timestamp) = previous?;
    if candidate.source_seq == previous_seq {
        Some(NeutralCalibrationRejectionReason::DuplicateSourceSequence)
    } else if candidate.source_seq < previous_seq {
        Some(NeutralCalibrationRejectionReason::RegressedSourceSequence)
    } else if candidate.captured_at_micros <= previous_timestamp {
        Some(NeutralCalibrationRejectionReason::RegressedTimestamp)
    } else {
        None
    }
}

fn candidate_metric_rejection(
    candidate: NeutralCalibrationCandidate,
    config: NeutralCalibrationSelectionConfig,
) -> Option<NeutralCalibrationRejectionReason> {
    if candidate.coverage.mapped_points == 0
        || candidate.coverage.valid_points > candidate.coverage.mapped_points
        || !candidate.coverage.effective_weight.is_finite()
        || candidate.coverage.effective_weight < 0.0
        || !candidate.reprojection_rms.is_finite()
        || candidate.reprojection_rms < 0.0
        || !candidate.yaw_radians.is_finite()
        || !candidate.pitch_radians.is_finite()
        || candidate
            .expression_activity
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        return Some(NeutralCalibrationRejectionReason::InvalidMetrics);
    }
    if candidate.coverage.status == DenseObservationStatus::Insufficient {
        return Some(NeutralCalibrationRejectionReason::InsufficientDenseCoverage);
    }
    if candidate.tracking_degraded {
        return Some(NeutralCalibrationRejectionReason::DegradedTracking);
    }
    if candidate.reprojection_rms > config.max_reprojection_rms {
        return Some(NeutralCalibrationRejectionReason::ExcessiveReprojectionResidual);
    }
    if candidate
        .expression_activity
        .is_some_and(|activity| activity > config.max_expression_activity)
    {
        return Some(NeutralCalibrationRejectionReason::ExpressionContamination);
    }
    None
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn pose_diversity(
    candidates: &[NeutralCalibrationCandidate],
    accepted_indices: &[usize],
    config: NeutralCalibrationSelectionConfig,
) -> NeutralPoseDiversity {
    if accepted_indices.is_empty() {
        return NeutralPoseDiversity {
            yaw_span_radians: 0.0,
            pitch_span_radians: 0.0,
            near_duplicate_fraction: 1.0,
        };
    }

    let first = candidates[accepted_indices[0]];
    let mut yaw_min = first.yaw_radians;
    let mut yaw_max = first.yaw_radians;
    let mut pitch_min = first.pitch_radians;
    let mut pitch_max = first.pitch_radians;
    let mut near_duplicates = 0usize;

    for pair in accepted_indices.windows(2) {
        let previous = candidates[pair[0]];
        let current = candidates[pair[1]];
        let dyaw = current.yaw_radians - previous.yaw_radians;
        let dpitch = current.pitch_radians - previous.pitch_radians;
        let distance = (dyaw * dyaw + dpitch * dpitch).sqrt();
        if distance <= config.near_duplicate_pose_distance_radians {
            near_duplicates += 1;
        }
    }
    for index in accepted_indices.iter().copied() {
        let candidate = candidates[index];
        yaw_min = yaw_min.min(candidate.yaw_radians);
        yaw_max = yaw_max.max(candidate.yaw_radians);
        pitch_min = pitch_min.min(candidate.pitch_radians);
        pitch_max = pitch_max.max(candidate.pitch_radians);
    }

    let comparisons = accepted_indices.len().saturating_sub(1);
    NeutralPoseDiversity {
        yaw_span_radians: yaw_max - yaw_min,
        pitch_span_radians: pitch_max - pitch_min,
        near_duplicate_fraction: if comparisons == 0 {
            1.0
        } else {
            near_duplicates as f32 / comparisons as f32
        },
    }
}

// ---------------------------------------------------------------------------
// Shared-identity solve contract (Issue #54.1)
//
// Engine-neutral boundary between the neutral candidate selection above and
// the immutable `GnmIdentityCalibration` output below. It defines what a
// numerical shared-identity solve may consume and how it is configured; no
// optimization runs here. Identity is structurally shared across the whole
// calibration window: exactly one identity initial state exists at the input
// level, while pose/camera parameters and small per-frame expression residue
// are per-sample nuisance values.
// ---------------------------------------------------------------------------

/// Per-sample nuisance initial values for one accepted neutral sample.
///
/// Pose/camera parameters reuse the documented dense projection convention,
/// and small expression residue is carried as an expression state the solver
/// is expected to keep near neutral. This type cannot hold identity
/// coefficients: identity never varies per sample.
#[derive(Clone, Debug, PartialEq)]
pub struct SampleNuisance {
    projection: DenseProjection,
    expression: GnmExpressionState,
}

impl SampleNuisance {
    /// Creates nuisance initial values, checking the expression dimension
    /// against the model it will be solved with.
    pub fn new(
        projection: DenseProjection,
        expression: GnmExpressionState,
        expected_expression_dimension: usize,
    ) -> Result<Self, GnmIdentityCalibrationError> {
        if expression.values().len() != expected_expression_dimension {
            return Err(GnmIdentityCalibrationError::InvalidSolveInput {
                sample: None,
                reason: format!(
                    "nuisance expression dimension mismatch: expected {expected_expression_dimension}, got {}",
                    expression.values().len()
                ),
            });
        }
        Ok(Self {
            projection,
            expression,
        })
    }

    /// Returns the pose/camera initial guess.
    pub fn projection(&self) -> &DenseProjection {
        &self.projection
    }

    /// Returns the small-expression initial state.
    pub fn expression(&self) -> &GnmExpressionState {
        &self.expression
    }
}

/// One accepted neutral sample paired with its nuisance initial values.
///
/// The observation is borrowed from caller-owned storage (typically the slice
/// indexed by [`NeutralCalibrationSelection::accepted_indices`]).
#[derive(Clone, Debug, PartialEq)]
pub struct NeutralSampleSolveInput<'a> {
    observation: &'a GnmDenseObservation,
    nuisance: SampleNuisance,
}

impl<'a> NeutralSampleSolveInput<'a> {
    /// Pairs an accepted dense observation with its nuisance initial values.
    pub fn new(observation: &'a GnmDenseObservation, nuisance: SampleNuisance) -> Self {
        Self {
            observation,
            nuisance,
        }
    }

    /// Returns the accepted dense observation for this sample.
    pub fn observation(&self) -> &'a GnmDenseObservation {
        self.observation
    }

    /// Returns the nuisance initial values for this sample.
    pub fn nuisance(&self) -> &SampleNuisance {
        &self.nuisance
    }
}

/// Typed configuration for the bounded shared-identity solve.
///
/// `active_identity_dimension` selects how many leading identity coefficients
/// are optimized; the rest stay at their initial values. Regularization is
/// explicit: `identity_prior_weight` pulls solved identity toward its initial
/// value, and `conditioning_regularization` adds a ridge term so degenerate
/// directions fail typed instead of silently amplifying noise.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SharedIdentitySolveConfig {
    active_identity_dimension: usize,
    max_iterations: usize,
    identity_prior_weight: f64,
    conditioning_regularization: f64,
    convergence_tolerance: f64,
    reprojection: DenseReprojectionConfig,
}

impl SharedIdentitySolveConfig {
    /// Creates a validated configuration; fails closed on values that would
    /// make the bounded solve ill-defined.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        active_identity_dimension: usize,
        max_iterations: usize,
        identity_prior_weight: f64,
        conditioning_regularization: f64,
        convergence_tolerance: f64,
        reprojection: DenseReprojectionConfig,
    ) -> Result<Self, GnmIdentityCalibrationError> {
        if active_identity_dimension == 0 {
            return Err(GnmIdentityCalibrationError::InvalidSolveConfig(
                "active_identity_dimension must be positive",
            ));
        }
        if max_iterations == 0 {
            return Err(GnmIdentityCalibrationError::InvalidSolveConfig(
                "max_iterations must be at least one",
            ));
        }
        if !identity_prior_weight.is_finite() || identity_prior_weight < 0.0 {
            return Err(GnmIdentityCalibrationError::InvalidSolveConfig(
                "identity_prior_weight must be finite and non-negative",
            ));
        }
        if !conditioning_regularization.is_finite() || conditioning_regularization <= 0.0 {
            return Err(GnmIdentityCalibrationError::InvalidSolveConfig(
                "conditioning_regularization must be finite and positive",
            ));
        }
        if !convergence_tolerance.is_finite() || convergence_tolerance < 0.0 {
            return Err(GnmIdentityCalibrationError::InvalidSolveConfig(
                "convergence_tolerance must be finite and non-negative",
            ));
        }
        Ok(Self {
            active_identity_dimension,
            max_iterations,
            identity_prior_weight,
            conditioning_regularization,
            convergence_tolerance,
            reprojection,
        })
    }

    /// Returns the number of leading identity dimensions actively solved.
    pub fn active_identity_dimension(&self) -> usize {
        self.active_identity_dimension
    }

    /// Returns the maximum accepted solver iterations.
    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Returns the identity prior weight toward the initial identity.
    pub fn identity_prior_weight(&self) -> f64 {
        self.identity_prior_weight
    }

    /// Returns the explicit ridge regularization added to normal equations.
    pub fn conditioning_regularization(&self) -> f64 {
        self.conditioning_regularization
    }

    /// Returns the relative objective improvement treated as convergence.
    pub fn convergence_tolerance(&self) -> f64 {
        self.convergence_tolerance
    }

    /// Returns the robust weighting used by the dense reprojection data term.
    pub fn reprojection(&self) -> DenseReprojectionConfig {
        self.reprojection
    }
}

/// Engine-neutral input bundle for the numerical shared-identity solve.
///
/// The structure makes per-sample identity unrepresentable: exactly one
/// `identity_initial` is supplied for the whole calibration window, and each
/// [`NeutralSampleSolveInput`] carries only an observation plus nuisance
/// state. Construction fails closed on version/dimension mismatch, non-finite
/// or inconsistent metrics, empty samples, and duplicate source sequences.
/// Sample order is preserved so downstream aggregation stays deterministic;
/// no optimization runs here.
#[derive(Clone, Debug)]
pub struct SharedIdentitySolveInput<'a> {
    model: &'a GnmModel,
    mapping: &'a DenseCorrespondenceSet,
    identity_initial: GnmIdentityState,
    samples: Vec<NeutralSampleSolveInput<'a>>,
}

impl<'a> SharedIdentitySolveInput<'a> {
    /// Builds and validates a shared-identity solve input from accepted
    /// neutral samples.
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub fn new(
        model: &'a GnmModel,
        mapping: &'a DenseCorrespondenceSet,
        identity_initial: GnmIdentityState,
        samples: Vec<NeutralSampleSolveInput<'a>>,
    ) -> Result<Self, GnmIdentityCalibrationError> {
        if mapping.version().model_version != model.version() {
            return Err(GnmIdentityCalibrationError::VersionMismatch {
                calibration_model: mapping.version().model_version,
                runtime_model: model.version(),
            });
        }
        if identity_initial.values().len() != model.identity_dimension() {
            return Err(GnmIdentityCalibrationError::IdentityDimensionMismatch {
                expected: model.identity_dimension(),
                actual: identity_initial.values().len(),
            });
        }
        if samples.is_empty() {
            return Err(GnmIdentityCalibrationError::InvalidSolveInput {
                sample: None,
                reason: "at least one accepted neutral sample is required".to_owned(),
            });
        }
        for (sample_index, sample) in samples.iter().enumerate() {
            validate_sample_input(model, sample_index, sample)?;
            let source_seq = sample.observation().source_seq();
            if samples[..sample_index]
                .iter()
                .any(|earlier| earlier.observation().source_seq() == source_seq)
            {
                return Err(GnmIdentityCalibrationError::InvalidSolveInput {
                    sample: Some(sample_index),
                    reason: format!("duplicate source sequence {source_seq}"),
                });
            }
        }
        Ok(Self {
            model,
            mapping,
            identity_initial,
            samples,
        })
    }

    /// Returns the loaded GNM model the solve is bound to.
    pub fn model(&self) -> &GnmModel {
        self.model
    }

    /// Returns the validated dense correspondence set used for evaluation.
    pub fn mapping(&self) -> &DenseCorrespondenceSet {
        self.mapping
    }

    /// Returns the dense mapping version binding.
    pub fn mapping_version(&self) -> DenseMappingVersion {
        self.mapping.version()
    }

    /// Returns the single shared identity initial state.
    pub fn identity_initial(&self) -> &GnmIdentityState {
        &self.identity_initial
    }

    /// Returns accepted samples in caller-supplied order.
    pub fn samples(&self) -> &[NeutralSampleSolveInput<'a>] {
        &self.samples
    }
}

fn validate_sample_input(
    model: &GnmModel,
    sample_index: usize,
    sample: &NeutralSampleSolveInput<'_>,
) -> Result<(), GnmIdentityCalibrationError> {
    let invalid = |reason: String| GnmIdentityCalibrationError::InvalidSolveInput {
        sample: Some(sample_index),
        reason,
    };
    let observation = sample.observation();
    let coverage = observation.coverage();
    if observation.points().is_empty() {
        return Err(invalid("sample has no valid observation points".to_owned()));
    }
    if coverage.status == DenseObservationStatus::Insufficient {
        return Err(invalid(
            "sample coverage is insufficient; selection gates must reject it first".to_owned(),
        ));
    }
    if coverage.valid_points > coverage.mapped_points || !coverage.effective_weight.is_finite() {
        return Err(invalid(
            "sample coverage summary is inconsistent".to_owned(),
        ));
    }
    for point in observation.points() {
        if !point.normalized_xy.iter().all(|value| value.is_finite())
            || !point.weight.is_finite()
            || point.weight < 0.0
        {
            return Err(invalid(format!(
                "observation point {} is non-finite or negatively weighted",
                point.mapping_index
            )));
        }
    }
    let nuisance = sample.nuisance();
    let projection = nuisance.projection();
    let pose_finite = projection
        .yaw_pitch_roll()
        .iter()
        .chain(projection.translation().iter())
        .chain(projection.principal_point().iter())
        .all(|value| value.is_finite());
    if !pose_finite || !projection.focal().is_finite() || projection.focal() <= 0.0 {
        return Err(invalid(
            "nuisance projection is not physically usable".to_owned(),
        ));
    }
    let expression = nuisance.expression();
    if expression.values().len() != model.expression_dimension() {
        return Err(invalid(format!(
            "nuisance expression dimension mismatch: expected {}, got {}",
            model.expression_dimension(),
            expression.values().len()
        )));
    }
    if !expression.values().iter().all(|value| value.is_finite()) {
        return Err(invalid("nuisance expression must be finite".to_owned()));
    }
    Ok(())
}

/// Validates a solve configuration against a solve input without running any
/// optimization.
///
/// This preflight is the cross-boundary check the individual constructors
/// cannot make: the active identity dimension must stay within the loaded
/// model dimension that [`SharedIdentitySolveInput`] was validated against.
pub fn validate_shared_identity_solve(
    input: &SharedIdentitySolveInput<'_>,
    config: SharedIdentitySolveConfig,
) -> Result<(), GnmIdentityCalibrationError> {
    if config.active_identity_dimension() > input.model().identity_dimension() {
        return Err(GnmIdentityCalibrationError::InvalidSolveConfig(
            "active_identity_dimension exceeds the loaded model identity dimension",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared-identity linear system assembly (Issue #54.2b)
//
// Aggregates the #54.2a per-sample residuals into one Gauss-Newton
// normal-equation system with exactly one shared identity block and one
// separated nuisance block per sample. Jacobians are central finite
// differences of the weighted dense reprojection residuals; the neutral-
// expression penalty contributes an exact analytic gradient/Hessian diagonal.
// No parameter update or optimization happens here (#54.2c owns the solve).
// ---------------------------------------------------------------------------

/// One sample's separated nuisance block of the aggregated linear system.
#[derive(Clone, Debug, PartialEq)]
pub struct SampleNuisanceBlock {
    /// Source sequence of the owning observation.
    pub source_seq: u64,
    /// Number of nuisance parameters (`projection + expression`).
    pub parameter_dimension: usize,
    /// Cross block `H_in` between shared identity and this sample's nuisance,
    /// row-major with `active_identity_dimension` rows.
    pub cross_hessian: Vec<f64>,
    /// Nuisance-only Gauss-Newton block `H_nn`, row-major square.
    pub hessian: Vec<f64>,
    /// Nuisance-only gradient including the neutral-expression term.
    pub gradient: Vec<f64>,
}

/// Aggregated normal-equation system over one calibration window.
///
/// The identity block exists exactly once because identity is shared across
/// all samples; pose/camera/expression nuisance blocks stay separated per
/// sample so a later alternating solve cannot mix them. The explicit identity
/// prior and conditioning regularization are already added to the identity
/// block.
#[derive(Clone, Debug, PartialEq)]
pub struct SharedIdentityLinearSystem {
    active_identity_dimension: usize,
    identity_hessian: Vec<f64>,
    identity_gradient: Vec<f64>,
    samples: Vec<SampleNuisanceBlock>,
}

impl SharedIdentityLinearSystem {
    /// Active identity dimension of the single shared block.
    pub fn active_identity_dimension(&self) -> usize {
        self.active_identity_dimension
    }

    /// Shared identity Gauss-Newton block `H_ii`, row-major square.
    pub fn identity_hessian(&self) -> &[f64] {
        &self.identity_hessian
    }

    /// Shared identity gradient including the identity-prior pull.
    pub fn identity_gradient(&self) -> &[f64] {
        &self.identity_gradient
    }

    /// Per-sample nuisance blocks in input order.
    pub fn samples(&self) -> &[SampleNuisanceBlock] {
        &self.samples
    }
}

/// Nuisance parameter layout: 3 yaw/pitch/roll + 3 translation + focal +
/// 2 principal-point coordinates, followed by the expression coefficients.
fn nuisance_parameter_count(expression_dimension: usize) -> usize {
    9 + expression_dimension
}

fn nuisance_parameters(nuisance: &SampleNuisance) -> Vec<f32> {
    let projection = nuisance.projection();
    let mut parameters = Vec::with_capacity(nuisance_parameter_count(
        nuisance.expression().values().len(),
    ));
    parameters.extend_from_slice(&projection.yaw_pitch_roll());
    parameters.extend_from_slice(&projection.translation());
    parameters.push(projection.focal());
    parameters.extend_from_slice(&projection.principal_point());
    parameters.extend_from_slice(nuisance.expression().values());
    parameters
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn nuisance_from_parameters(
    model: &GnmModel,
    parameters: &[f32],
) -> Result<SampleNuisance, GnmIdentityCalibrationError> {
    if parameters.len() != nuisance_parameter_count(model.expression_dimension()) {
        return Err(GnmIdentityCalibrationError::InvalidSolveInput {
            sample: None,
            reason: "nuisance parameter vector has wrong shape".to_owned(),
        });
    }
    let projection = DenseProjection::new(
        [parameters[0], parameters[1], parameters[2]],
        [parameters[3], parameters[4], parameters[5]],
        parameters[6],
        [parameters[7], parameters[8]],
    )
    .map_err(GnmIdentityCalibrationError::Evaluation)?;
    let expression =
        GnmExpressionState::new(parameters[9..].to_vec(), model.expression_dimension()).map_err(
            |error| GnmIdentityCalibrationError::InvalidSolveInput {
                sample: None,
                reason: format!("perturbed nuisance expression rejected: {error}"),
            },
        )?;
    SampleNuisance::new(projection, expression, model.expression_dimension())
}

/// Weighted residual vector of one sample: `sqrt(w) * (observed - projected)`
/// interleaved per point, plus nothing else. The neutral-expression penalty is
/// applied analytically at assembly time instead.
fn weighted_residual_vector(
    model: &GnmModel,
    mapping: &DenseCorrespondenceSet,
    identity: &GnmIdentityState,
    nuisance: &SampleNuisance,
    observation: &GnmDenseObservation,
    config: DenseReprojectionConfig,
) -> Result<Vec<f64>, GnmIdentityCalibrationError> {
    let residual =
        evaluate_neutral_sample_residual(model, mapping, identity, observation, nuisance, config)?;
    let mut vector = Vec::with_capacity(residual.reprojection.residuals().len() * 2);
    for point in residual.reprojection.residuals() {
        let scale = (point.base_weight * point.huber_weight).sqrt() as f64;
        vector.push(scale * point.residual_xy[0] as f64);
        vector.push(scale * point.residual_xy[1] as f64);
    }
    Ok(vector)
}

/// Assembles the shared-identity normal-equation system for one window.
///
/// `jacobian_step` is the central-difference perturbation in parameter units;
/// it must be finite and positive. Validates the input/config pair first, so
/// invalid windows never reach numerical differentiation.
pub fn assemble_shared_identity_linear_system(
    input: &SharedIdentitySolveInput<'_>,
    config: SharedIdentitySolveConfig,
    jacobian_step: f64,
) -> Result<SharedIdentityLinearSystem, GnmIdentityCalibrationError> {
    validate_shared_identity_solve(input, config)?;
    if !jacobian_step.is_finite() || jacobian_step <= 0.0 {
        return Err(GnmIdentityCalibrationError::InvalidSolveConfig(
            "jacobian_step must be finite and positive",
        ));
    }
    let borrowed: Vec<&NeutralSampleSolveInput<'_>> = input.samples().iter().collect();
    assemble_linear_system_at(
        input.model(),
        input.mapping(),
        input.identity_initial(),
        &borrowed,
        config,
        jacobian_step,
    )
}

/// Aggregates the system at an arbitrary linearization point. The alternating
/// solver re-linearizes here every iteration; the public wrapper above always
/// passes the initial shared identity and the initial nuisances.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn assemble_linear_system_at(
    model: &GnmModel,
    mapping: &DenseCorrespondenceSet,
    identity_linearization: &GnmIdentityState,
    samples: &[&NeutralSampleSolveInput<'_>],
    config: SharedIdentitySolveConfig,
    jacobian_step: f64,
) -> Result<SharedIdentityLinearSystem, GnmIdentityCalibrationError> {
    let identity_dimension = model.identity_dimension();
    let active = config.active_identity_dimension();
    let identity_initial = identity_linearization.values().to_vec();
    let nuisance_dimension = nuisance_parameter_count(model.expression_dimension());

    let mut identity_hessian = vec![0.0f64; active * active];
    let mut identity_gradient = vec![0.0f64; active];
    let mut blocks = Vec::with_capacity(samples.len());

    for (sample_index, sample) in samples.iter().enumerate() {
        let invalid = |reason: String| GnmIdentityCalibrationError::InvalidSolveInput {
            sample: Some(sample_index),
            reason,
        };
        let observation = sample.observation();
        let base_residual = weighted_residual_vector(
            model,
            mapping,
            identity_linearization,
            sample.nuisance(),
            observation,
            config.reprojection(),
        )?;
        if base_residual.is_empty() {
            return Err(invalid(
                "sample produced no usable reprojection rows".to_owned(),
            ));
        }

        let total = active + nuisance_dimension;
        let mut jacobian = vec![0.0f64; base_residual.len() * total];
        for column in 0..total {
            for sign in [1.0, -1.0] {
                let (perturbed_identity, perturbed_nuisance);
                if column < active {
                    let mut values = identity_initial.clone();
                    let slot = &mut values[column];
                    *slot = (*slot as f64 + sign * jacobian_step) as f32;
                    perturbed_identity =
                        Some(GnmIdentityState::new(values, identity_dimension).map_err(
                            |error| invalid(format!("perturbed identity rejected: {error}")),
                        )?);
                    perturbed_nuisance = None;
                } else {
                    let mut parameters = nuisance_parameters(sample.nuisance());
                    let slot = &mut parameters[column - active];
                    *slot = (*slot as f64 + sign * jacobian_step) as f32;
                    perturbed_identity = None;
                    perturbed_nuisance = Some(nuisance_from_parameters(model, &parameters)?);
                }
                let perturbed_residual = weighted_residual_vector(
                    model,
                    mapping,
                    perturbed_identity
                        .as_ref()
                        .unwrap_or(identity_linearization),
                    perturbed_nuisance.as_ref().unwrap_or(sample.nuisance()),
                    observation,
                    config.reprojection(),
                )?;
                for (row, (base, perturbed)) in base_residual
                    .iter()
                    .zip(perturbed_residual.iter())
                    .enumerate()
                {
                    jacobian[row * total + column] +=
                        (perturbed - base) * sign / (2.0 * jacobian_step);
                }
            }
        }

        // Normal equations: H = J^T J, g = J^T r, partitioned by block.
        let mut block = SampleNuisanceBlock {
            source_seq: observation.source_seq(),
            parameter_dimension: nuisance_dimension,
            cross_hessian: vec![0.0; active * nuisance_dimension],
            hessian: vec![0.0; nuisance_dimension * nuisance_dimension],
            gradient: vec![0.0; nuisance_dimension],
        };
        #[allow(clippy::needless_range_loop)] // row indexes jacobian and residual together
        for row in 0..base_residual.len() {
            let row_start = row * total;
            let residual_row = base_residual[row];
            for i in 0..active {
                let j_i = jacobian[row_start + i];
                identity_gradient[i] += j_i * residual_row;
                for k in 0..active {
                    identity_hessian[i * active + k] += j_i * jacobian[row_start + k];
                }
                for k in 0..nuisance_dimension {
                    block.cross_hessian[i * nuisance_dimension + k] +=
                        j_i * jacobian[row_start + active + k];
                }
            }
            for i in 0..nuisance_dimension {
                let j_i = jacobian[row_start + active + i];
                block.gradient[i] += j_i * residual_row;
                for k in 0..nuisance_dimension {
                    block.hessian[i * nuisance_dimension + k] +=
                        j_i * jacobian[row_start + active + k];
                }
            }
        }

        // Exact neutral-expression penalty contribution on the expression
        // coordinates of the nuisance block: 0.5 * ||e||^2 terms.
        let expression = sample.nuisance().expression().values();
        for (offset, value) in expression.iter().enumerate() {
            let coordinate = 9 + offset;
            block.gradient[coordinate] += *value as f64;
            block.hessian[coordinate * nuisance_dimension + coordinate] += 1.0;
        }
        blocks.push(block);
    }

    // Explicit identity prior pulling toward the initial shared identity and
    // the explicit conditioning ridge on the shared block only. The prior
    // gradient term lambda*(x_linearization - x_initial) vanishes because the
    // system is always linearized exactly at a candidate shared identity and
    // the prior anchor is that same initial state.
    for i in 0..active {
        identity_hessian[i * active + i] += config.identity_prior_weight();
        identity_hessian[i * active + i] += config.conditioning_regularization();
    }

    let system = SharedIdentityLinearSystem {
        active_identity_dimension: active,
        identity_hessian,
        identity_gradient,
        samples: blocks,
    };
    validate_linear_system_shape(&system)?;
    Ok(system)
}

// ---------------------------------------------------------------------------
// Bounded alternating solve (Issue #54.2c)
//
// Uses the #54.2b system to alternately update the shared identity and each
// sample's pose/camera/small-expression nuisance within a bounded iteration
// budget. Every update is Gauss-Newton with backtracking: a step that worsens
// the current objective is halved, then rejected; updates are never accepted
// unconditionally. Ill-conditioned blocks, divergence, and NaN/Inf objectives
// are typed failures. Calibration output assembly stays out of scope (#54.2d).
// ---------------------------------------------------------------------------

/// Outcome of one bounded alternating shared-identity solve.
#[derive(Clone, Debug, PartialEq)]
pub struct SharedIdentitySolveOutcome {
    /// Solved shared identity; inactive trailing dimensions keep initial values.
    pub identity: GnmIdentityState,
    /// Solved per-sample nuisance states in input order.
    pub nuisances: Vec<SampleNuisance>,
    /// Completed iterations (including the final accepted round).
    pub iterations: usize,
    /// Objective after the last accepted update.
    pub final_objective: f64,
    /// Objective at every linearization point; strictly non-increasing.
    pub objective_history: Vec<f64>,
    /// Whether the relative-improvement stopping condition was met inside the
    /// iteration budget.
    pub converged: bool,
}

impl SharedIdentitySolveOutcome {
    /// Convenience accessor pairing solved nuisances with their observations.
    pub fn nuisance_for_sample(&self, sample_index: usize) -> Option<&SampleNuisance> {
        self.nuisances.get(sample_index)
    }
}

/// Global scalar objective: weighted dense reprojection error plus the exact
/// neutral-expression penalties plus the identity-prior pull of active dims.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn alternating_objective(
    model: &GnmModel,
    mapping: &DenseCorrespondenceSet,
    config: SharedIdentitySolveConfig,
    identity: &GnmIdentityState,
    identity_initial: &GnmIdentityState,
    samples: &[&NeutralSampleSolveInput<'_>],
    nuisances: &[SampleNuisance],
) -> Result<f64, GnmIdentityCalibrationError> {
    let mut objective = 0.0f64;
    for (sample, nuisance) in samples.iter().zip(nuisances.iter()) {
        let residual = evaluate_neutral_sample_residual(
            model,
            mapping,
            identity,
            sample.observation(),
            nuisance,
            config.reprojection(),
        )?;
        for point in residual.reprojection.residuals() {
            let weight = (point.base_weight * point.huber_weight) as f64;
            let dx = point.residual_xy[0] as f64;
            let dy = point.residual_xy[1] as f64;
            objective += weight * (dx * dx + dy * dy);
        }
        objective += residual.neutral_expression_squared * 0.5;
    }
    for index in 0..config.active_identity_dimension() {
        let delta = (identity.values()[index] - identity_initial.values()[index]) as f64;
        objective += 0.5 * config.identity_prior_weight() * delta * delta;
    }
    Ok(objective)
}

/// Solves a small symmetric positive system by Gaussian elimination with
/// partial pivoting. Fails typed when no usable pivot remains.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn solve_symmetric_system(
    matrix: &[f64],
    rhs: &[f64],
) -> Result<Vec<f64>, GnmIdentityCalibrationError> {
    let dimension = rhs.len();
    let scale = matrix
        .iter()
        .fold(0.0f64, |acc, value| acc.max(value.abs()));
    if !(scale.is_finite() && scale > 0.0) || matrix.iter().any(|value| !value.is_finite()) {
        return Err(GnmIdentityCalibrationError::IllConditionedSolve);
    }
    let mut augmented = vec![0.0f64; dimension * (dimension + 1)];
    for row in 0..dimension {
        augmented[row * (dimension + 1)..row * (dimension + 1) + dimension]
            .copy_from_slice(&matrix[row * dimension..(row + 1) * dimension]);
        augmented[row * (dimension + 1) + dimension] = rhs[row];
    }
    let pivot_floor = 1.0e-12 * scale;
    let stride = dimension + 1;
    for column in 0..dimension {
        let (pivot_row, _) = (column..dimension)
            .map(|row| (row, augmented[row * (dimension + 1) + column].abs()))
            .fold((column, 0.0f64), |best, (row, magnitude)| {
                if magnitude > best.1 {
                    (row, magnitude)
                } else {
                    best
                }
            });
        if augmented[pivot_row * (dimension + 1) + column].abs() <= pivot_floor
            || !augmented[pivot_row * (dimension + 1) + column].is_finite()
        {
            return Err(GnmIdentityCalibrationError::IllConditionedSolve);
        }
        augmented.swap(pivot_row * stride, column * stride);
        let pivot = augmented[column * stride + column];
        for row in (column + 1)..dimension {
            let factor = augmented[row * stride + column] / pivot;
            if factor == 0.0 {
                continue;
            }
            for offset in column..stride {
                augmented[row * stride + offset] -= factor * augmented[column * stride + offset];
            }
        }
    }
    let mut solution = vec![0.0f64; dimension];
    for row in (0..dimension).rev() {
        let stride = dimension + 1;
        let mut accumulator = augmented[row * stride + dimension];
        for column in (row + 1)..dimension {
            accumulator -= augmented[row * stride + column] * solution[column];
        }
        solution[row] = accumulator / augmented[row * stride + row];
    }
    Ok(solution)
}

/// Runs the bounded alternating shared-identity/nuisance solve.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
// Invariant: `objective_history`/`history` is seeded with the initial
// objective before any iteration or acceptance step runs.
#[allow(clippy::expect_used)]
pub fn solve_shared_identity(
    input: &SharedIdentitySolveInput<'_>,
    config: SharedIdentitySolveConfig,
    jacobian_step: f64,
) -> Result<SharedIdentitySolveOutcome, GnmIdentityCalibrationError> {
    validate_shared_identity_solve(input, config)?;
    if !jacobian_step.is_finite() || jacobian_step <= 0.0 {
        return Err(GnmIdentityCalibrationError::InvalidSolveConfig(
            "jacobian_step must be finite and positive",
        ));
    }

    let model = input.model();
    let mapping = input.mapping();
    let borrowed: Vec<&NeutralSampleSolveInput<'_>> = input.samples().iter().collect();
    let identity_initial = input.identity_initial().clone();
    let mut identity = identity_initial.clone();
    let mut nuisances: Vec<SampleNuisance> = input
        .samples()
        .iter()
        .map(|sample| sample.nuisance().clone())
        .collect();

    let mut objective_history = vec![alternating_objective(
        model,
        mapping,
        config,
        &identity,
        &identity_initial,
        &borrowed,
        &nuisances,
    )?];
    let ridge = config.conditioning_regularization();
    let tolerance = config.convergence_tolerance();

    for iteration in 1..=config.max_iterations() {
        let system =
            assemble_linear_system_at(model, mapping, &identity, &borrowed, config, jacobian_step)?;

        // Step 1: per-sample nuisance updates, independent of each other.
        for (sample_index, block) in system.samples().iter().enumerate() {
            let regularized: Vec<f64> = block
                .hessian
                .iter()
                .enumerate()
                .map(|(offset, value)| {
                    let coordinate = offset / block.parameter_dimension;
                    if coordinate * block.parameter_dimension + coordinate == offset {
                        value + ridge
                    } else {
                        *value
                    }
                })
                .collect();
            let delta = solve_symmetric_system(&regularized, &block.gradient)?;
            accept_backtracked_nuisance_update(
                model,
                mapping,
                config,
                &identity,
                &identity_initial,
                &borrowed,
                &mut nuisances,
                sample_index,
                &delta,
                &mut objective_history,
            )?;
        }

        // Step 2: one shared identity update across all samples.
        let identity_delta =
            solve_symmetric_system(system.identity_hessian(), system.identity_gradient())?;
        accept_backtracked_identity_update(
            model,
            mapping,
            config,
            &identity_initial,
            &borrowed,
            &mut identity,
            &mut nuisances,
            &identity_delta,
            &mut objective_history,
        )?;

        let current = *objective_history.last().expect("history always non-empty");
        let previous = objective_history[objective_history.len() - 2];
        let improvement = previous - current;
        if !improvement.is_finite() {
            return Err(GnmIdentityCalibrationError::NonFiniteObjective { iteration });
        }
        if improvement <= tolerance * previous.max(1.0) {
            return Ok(finish_outcome(
                identity,
                nuisances,
                iteration,
                current,
                objective_history,
                true,
            ));
        }
    }

    let final_objective = *objective_history.last().expect("history always non-empty");
    Ok(finish_outcome(
        identity,
        nuisances,
        config.max_iterations(),
        final_objective,
        objective_history,
        false,
    ))
}

#[allow(clippy::too_many_arguments)]
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
// Invariant: `objective_history`/`history` is seeded with the initial
// objective before any iteration or acceptance step runs.
#[allow(clippy::expect_used)]
fn accept_backtracked_nuisance_update(
    model: &GnmModel,
    mapping: &DenseCorrespondenceSet,
    config: SharedIdentitySolveConfig,
    identity: &GnmIdentityState,
    identity_initial: &GnmIdentityState,
    samples: &[&NeutralSampleSolveInput<'_>],
    nuisances: &mut [SampleNuisance],
    sample_index: usize,
    delta: &[f64],
    history: &mut Vec<f64>,
) -> Result<(), GnmIdentityCalibrationError> {
    let baseline = *history.last().expect("history always non-empty");
    let parameters = nuisance_parameters(&nuisances[sample_index]);
    let mut step_scale = 1.0f64;
    for _ in 0..8 {
        let candidate_parameters: Vec<f32> = parameters
            .iter()
            .zip(delta.iter())
            .map(|(value, change)| ((*value as f64) - step_scale * change) as f32)
            .collect();
        let candidate = match nuisance_from_parameters(model, &candidate_parameters) {
            Ok(candidate) => candidate,
            Err(_) => {
                step_scale *= 0.5;
                continue;
            }
        };
        let mut trial = nuisances.to_vec();
        trial[sample_index] = candidate;
        let objective = alternating_objective(
            model,
            mapping,
            config,
            identity,
            identity_initial,
            samples,
            &trial,
        )?;
        if objective.is_finite() && objective <= baseline {
            nuisances[sample_index] = trial[sample_index].clone();
            history.push(objective);
            return Ok(());
        }
        step_scale *= 0.5;
    }
    Ok(()) // bounded policy: reject the worsening update entirely
}

#[allow(clippy::too_many_arguments)]
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
// Invariant: `objective_history`/`history` is seeded with the initial
// objective before any iteration or acceptance step runs.
#[allow(clippy::expect_used)]
fn accept_backtracked_identity_update(
    model: &GnmModel,
    mapping: &DenseCorrespondenceSet,
    config: SharedIdentitySolveConfig,
    identity_initial: &GnmIdentityState,
    samples: &[&NeutralSampleSolveInput<'_>],
    identity: &mut GnmIdentityState,
    nuisances: &mut [SampleNuisance],
    delta: &[f64],
    history: &mut Vec<f64>,
) -> Result<(), GnmIdentityCalibrationError> {
    let baseline = *history.last().expect("history always non-empty");
    let values = identity.values().to_vec();
    let mut step_scale = 1.0f64;
    for _ in 0..8 {
        let candidate_values: Vec<f32> = values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                if index < delta.len() {
                    ((*value as f64) - step_scale * delta[index]) as f32
                } else {
                    *value
                }
            })
            .collect();
        let candidate = match GnmIdentityState::new(candidate_values, values.len()) {
            Ok(candidate) => candidate,
            Err(_) => {
                step_scale *= 0.5;
                continue;
            }
        };
        let trial_nuisances: Vec<SampleNuisance> = nuisances.to_vec();
        let objective = alternating_objective(
            model,
            mapping,
            config,
            &candidate,
            identity_initial,
            samples,
            &trial_nuisances,
        )?;
        if objective.is_finite() && objective <= baseline {
            *identity = candidate;
            history.push(objective);
            return Ok(());
        }
        step_scale *= 0.5;
    }
    Ok(())
}

fn finish_outcome(
    identity: GnmIdentityState,
    nuisances: Vec<SampleNuisance>,
    iterations: usize,
    final_objective: f64,
    objective_history: Vec<f64>,
    converged: bool,
) -> SharedIdentitySolveOutcome {
    SharedIdentitySolveOutcome {
        identity,
        nuisances,
        iterations,
        final_objective,
        objective_history,
        converged,
    }
}

// ---------------------------------------------------------------------------
// Calibration output assembly (Issue #54.2d)
//
// Turns a bounded solver outcome into the publishable, immutable
// `GnmIdentityCalibration`: solved identity wrapped read-only, neutral
// expression/surface computed exactly once, best-effort normalization scales,
// and fit diagnostics. Mismatched model/mapping versions, non-finite solver
// results, and degenerate scales are never published.
// ---------------------------------------------------------------------------

/// Assembles and validates the final calibration from one solver outcome.
pub fn finalize_identity_calibration(
    model: &GnmModel,
    mapping: &DenseCorrespondenceSet,
    input: &SharedIdentitySolveInput<'_>,
    selection: &NeutralCalibrationSelection,
    outcome: &SharedIdentitySolveOutcome,
    config: &SharedIdentitySolveConfig,
) -> Result<GnmIdentityCalibration, GnmIdentityCalibrationError> {
    if mapping.version().model_version != model.version() {
        return Err(GnmIdentityCalibrationError::VersionMismatch {
            calibration_model: mapping.version().model_version,
            runtime_model: model.version(),
        });
    }
    if !outcome.final_objective.is_finite()
        || outcome
            .objective_history
            .iter()
            .any(|value| !value.is_finite())
    {
        return Err(GnmIdentityCalibrationError::InvalidOutput {
            field: "solver objective",
            reason: "must be finite before publishing",
        });
    }
    let identity = FixedGnmIdentity::new(outcome.identity.clone(), model)?;
    let neutral_expression_reference = model.neutral_expression();

    // Neutral surface reference: evaluated exactly once at the solved identity.
    let mut sparse = GnmSparseVertices::with_len(mapping.len());
    mapping
        .evaluate_surface(
            model,
            identity.state(),
            &neutral_expression_reference,
            &GnmJointState::neutral(model.joint_count()),
            &mut sparse,
        )
        .map_err(|_error| GnmIdentityCalibrationError::InvalidOutput {
            field: "neutral_surface_reference",
            reason: "surface evaluation failed",
        })?;
    let neutral_surface_reference = sparse.values().to_vec();

    let normalization_scales =
        normalization_scales_from_mapping(mapping, &neutral_surface_reference);

    // Conditioning estimate from the shared block re-linearized exactly once
    // at the solved state: ratio of extreme diagonal entries.
    let condition_number = estimate_condition_number(model, mapping, input, outcome, config);

    GnmIdentityCalibration::new(
        model,
        mapping.version(),
        identity,
        neutral_expression_reference,
        neutral_surface_reference,
        normalization_scales,
        IdentityFitDiagnostics {
            accepted_samples: selection.diagnostics.accepted_candidates,
            rejected_samples: selection.diagnostics.rejected_candidates,
            reprojection_rms: (outcome.final_objective as f32).sqrt(),
            active_identity_dimension: config.active_identity_dimension(),
            condition_number,
            pose_diversity: selection.diagnostics.pose_diversity,
        },
    )
}

/// Best-effort person-neutral scales from region topology. Absence means the
/// mapping does not classify the needed landmarks, never a zero scale.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
/// Computes the deterministic neutral normalization scales from the repository
/// mapping evaluated at a neutral surface.
///
/// Shared by the identity-calibration assembly path and by development-only
/// replay tooling that assembles a fixed neutral calibration (GNM #68.3); the
/// scales must not differ between those paths.
pub fn normalization_scales_from_mapping(
    mapping: &DenseCorrespondenceSet,
    surface: &[[f32; 3]],
) -> NeutralNormalizationScales {
    let Ok(groups) = DenseRegionGroups::from_set(mapping) else {
        return NeutralNormalizationScales::default();
    };
    let point = |indexed: &crate::IndexedRow| surface[indexed.index];
    let distance = |a: [f32; 3], b: [f32; 3]| {
        let delta = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt()
    };
    NeutralNormalizationScales {
        inter_ocular: Some(distance(
            point(groups.eyes().right().outer_corner()),
            point(groups.eyes().left().outer_corner()),
        )),
        mouth_width: Some(distance(
            point(groups.mouth().outer_corner_right()),
            point(groups.mouth().outer_corner_left()),
        )),
        eye_aperture: None,
    }
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn estimate_condition_number(
    model: &GnmModel,
    mapping: &DenseCorrespondenceSet,
    input: &SharedIdentitySolveInput<'_>,
    outcome: &SharedIdentitySolveOutcome,
    config: &SharedIdentitySolveConfig,
) -> Option<f64> {
    let borrowed: Vec<&NeutralSampleSolveInput<'_>> = input.samples().iter().collect();
    let system = assemble_linear_system_at(
        model,
        mapping,
        &outcome.identity,
        &borrowed,
        *config,
        1.0e-4,
    )
    .ok()?;
    let active = system.active_identity_dimension();
    let diagonal = (0..active).map(|index| system.identity_hessian()[index * active + index]);
    let min = diagonal.clone().fold(f64::INFINITY, f64::min);
    let max = diagonal.fold(f64::NEG_INFINITY, f64::max);
    if min.is_finite() && max.is_finite() && min > 0.0 {
        Some(max / min)
    } else {
        None
    }
}

fn validate_linear_system_shape(
    system: &SharedIdentityLinearSystem,
) -> Result<(), GnmIdentityCalibrationError> {
    let active = system.active_identity_dimension;
    if active == 0
        || system.identity_hessian.len() != active * active
        || system.identity_gradient.len() != active
    {
        return Err(GnmIdentityCalibrationError::InvalidSolveInput {
            sample: None,
            reason: "aggregated identity block is ill-shaped".to_owned(),
        });
    }
    if system
        .identity_hessian
        .iter()
        .any(|value| !value.is_finite())
        || system
            .identity_gradient
            .iter()
            .any(|value| !value.is_finite())
    {
        return Err(GnmIdentityCalibrationError::InvalidSolveInput {
            sample: None,
            reason: "aggregated identity block is non-finite".to_owned(),
        });
    }
    if system.samples.is_empty() {
        return Err(GnmIdentityCalibrationError::InvalidSolveInput {
            sample: None,
            reason: "linear system aggregation requires at least one sample".to_owned(),
        });
    }
    for sample in &system.samples {
        let dimension = sample.parameter_dimension;
        let shaped = sample.cross_hessian.len() == active * dimension
            && sample.hessian.len() == dimension * dimension
            && sample.gradient.len() == dimension;
        let finite = sample
            .cross_hessian
            .iter()
            .chain(sample.hessian.iter())
            .chain(sample.gradient.iter())
            .all(|value| value.is_finite());
        if !shaped || !finite {
            return Err(GnmIdentityCalibrationError::InvalidSolveInput {
                sample: None,
                reason: format!(
                    "nuisance block for sequence {} is ill-shaped or non-finite",
                    sample.source_seq
                ),
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Per-sample residual (Issue #54.2a)
//
// Pure evaluation of one neutral sample against a shared identity candidate
// and that sample's nuisance candidate. Two separate residual terms are
// returned: the dense reprojection data term and the neutral-expression
// penalty that keeps the small per-sample expression near neutral. No
// Jacobian aggregation or parameter update happens here.
// ---------------------------------------------------------------------------

/// Both residual terms for one neutral sample under one candidate state.
#[derive(Clone, Debug, PartialEq)]
pub struct NeutralSampleResidual {
    /// Weighted dense reprojection report at the candidate state.
    pub reprojection: DenseReprojectionReport,
    /// Squared L2 norm of the small-expression nuisance deviation from the
    /// neutral expression. This is the neutral-expression residual term that
    /// penalizes expression contamination of the calibration window.
    pub neutral_expression_squared: f64,
}

/// Evaluates both residual terms for one neutral dense observation.
///
/// Identity and sample nuisance are deliberately separate arguments so a
/// caller cannot confuse the shared identity with per-sample state. Internal
/// skeleton joints stay neutral during calibration: the head-root pose lives
/// in the nuisance projection, and joint channels are out of the neutral
/// solve's scope. Fails closed on version/dimension mismatch and delegates
/// observation-level failure to the typed reprojection error.
pub fn evaluate_neutral_sample_residual(
    model: &GnmModel,
    mapping: &DenseCorrespondenceSet,
    identity: &GnmIdentityState,
    observation: &GnmDenseObservation,
    nuisance: &SampleNuisance,
    config: DenseReprojectionConfig,
) -> Result<NeutralSampleResidual, GnmIdentityCalibrationError> {
    if mapping.version().model_version != model.version() {
        return Err(GnmIdentityCalibrationError::VersionMismatch {
            calibration_model: mapping.version().model_version,
            runtime_model: model.version(),
        });
    }
    if identity.values().len() != model.identity_dimension() {
        return Err(GnmIdentityCalibrationError::IdentityDimensionMismatch {
            expected: model.identity_dimension(),
            actual: identity.values().len(),
        });
    }
    if nuisance.expression().values().len() != model.expression_dimension() {
        return Err(GnmIdentityCalibrationError::ExpressionDimensionMismatch {
            expected: model.expression_dimension(),
            actual: nuisance.expression().values().len(),
        });
    }
    let reprojection = evaluate_dense_reprojection(
        model,
        identity,
        nuisance.expression(),
        &GnmJointState::neutral(model.joint_count()),
        mapping,
        observation,
        nuisance.projection(),
        config,
    )
    .map_err(GnmIdentityCalibrationError::Evaluation)?;
    let neutral_expression_squared = nuisance
        .expression()
        .values()
        .iter()
        .map(|value| (*value as f64) * (*value as f64))
        .sum();
    Ok(NeutralSampleResidual {
        reprojection,
        neutral_expression_squared,
    })
}

/// Semantically fixed GNM identity supplied read-only to tracking.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedGnmIdentity(GnmIdentityState);

impl FixedGnmIdentity {
    /// Wraps a validated identity whose dimension matches the loaded model.
    pub fn new(
        identity: GnmIdentityState,
        model: &GnmModel,
    ) -> Result<Self, GnmIdentityCalibrationError> {
        if identity.values().len() != model.identity_dimension() {
            return Err(GnmIdentityCalibrationError::IdentityDimensionMismatch {
                expected: model.identity_dimension(),
                actual: identity.values().len(),
            });
        }
        Ok(Self(identity))
    }

    /// Returns the immutable model identity state.
    pub fn state(&self) -> &GnmIdentityState {
        &self.0
    }

    /// Returns coefficients as a read-only slice.
    pub fn values(&self) -> &[f32] {
        self.0.values()
    }
}

/// Optional person-specific neutral geometry scales for later projectors.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NeutralNormalizationScales {
    /// Inter-ocular distance in model-space units when measured.
    pub inter_ocular: Option<f32>,
    /// Neutral mouth width in model-space units when measured.
    pub mouth_width: Option<f32>,
    /// Neutral eye aperture in model-space units when measured.
    pub eye_aperture: Option<f32>,
}

impl NeutralNormalizationScales {
    fn validate(self) -> Result<(), GnmIdentityCalibrationError> {
        for (field, value) in [
            ("inter_ocular", self.inter_ocular),
            ("mouth_width", self.mouth_width),
            ("eye_aperture", self.eye_aperture),
        ] {
            if value.is_some_and(|value| !value.is_finite() || value <= 0.0) {
                return Err(GnmIdentityCalibrationError::InvalidOutput {
                    field,
                    reason: "normalization scale must be finite and positive when available",
                });
            }
        }
        Ok(())
    }
}

/// Diagnostics produced by a future numerical identity solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IdentityFitDiagnostics {
    /// Accepted dense sample count used by the shared-identity solve.
    pub accepted_samples: usize,
    /// Rejected candidate count.
    pub rejected_samples: usize,
    /// Final aggregate dense reprojection RMS.
    pub reprojection_rms: f32,
    /// Number of identity dimensions actively solved/retained.
    pub active_identity_dimension: usize,
    /// Optional conditioning estimate. Absence means not measured, not well-conditioned.
    pub condition_number: Option<f64>,
    /// Pose-diversity summary of the selected window.
    pub pose_diversity: NeutralPoseDiversity,
}

/// Immutable identity calibration handed to later tracking/projector stages.
#[derive(Clone, Debug, PartialEq)]
pub struct GnmIdentityCalibration {
    model_version: GnmVersion,
    mapping_version: DenseMappingVersion,
    identity: FixedGnmIdentity,
    neutral_expression_reference: GnmExpressionState,
    neutral_surface_reference: Vec<[f32; 3]>,
    normalization_scales: NeutralNormalizationScales,
    diagnostics: IdentityFitDiagnostics,
}

impl GnmIdentityCalibration {
    /// Builds a version-bound, finite calibration object from numerical-solver output.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: &GnmModel,
        mapping_version: DenseMappingVersion,
        identity: FixedGnmIdentity,
        neutral_expression_reference: GnmExpressionState,
        neutral_surface_reference: Vec<[f32; 3]>,
        normalization_scales: NeutralNormalizationScales,
        diagnostics: IdentityFitDiagnostics,
    ) -> Result<Self, GnmIdentityCalibrationError> {
        if mapping_version.model_version != model.version() {
            return Err(GnmIdentityCalibrationError::VersionMismatch {
                calibration_model: mapping_version.model_version,
                runtime_model: model.version(),
            });
        }
        if identity.values().len() != model.identity_dimension() {
            return Err(GnmIdentityCalibrationError::IdentityDimensionMismatch {
                expected: model.identity_dimension(),
                actual: identity.values().len(),
            });
        }
        if neutral_expression_reference.values().len() != model.expression_dimension() {
            return Err(GnmIdentityCalibrationError::ExpressionDimensionMismatch {
                expected: model.expression_dimension(),
                actual: neutral_expression_reference.values().len(),
            });
        }
        if neutral_surface_reference.is_empty()
            || neutral_surface_reference
                .iter()
                .flatten()
                .any(|value| !value.is_finite())
        {
            return Err(GnmIdentityCalibrationError::InvalidOutput {
                field: "neutral_surface_reference",
                reason: "neutral surface reference must be non-empty and finite",
            });
        }
        normalization_scales.validate()?;
        if !diagnostics.reprojection_rms.is_finite() || diagnostics.reprojection_rms < 0.0 {
            return Err(GnmIdentityCalibrationError::InvalidOutput {
                field: "reprojection_rms",
                reason: "must be finite and non-negative",
            });
        }
        if diagnostics.active_identity_dimension == 0
            || diagnostics.active_identity_dimension > model.identity_dimension()
        {
            return Err(GnmIdentityCalibrationError::InvalidOutput {
                field: "active_identity_dimension",
                reason: "must be within the loaded model identity dimension",
            });
        }
        if diagnostics
            .condition_number
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(GnmIdentityCalibrationError::InvalidOutput {
                field: "condition_number",
                reason: "must be finite and positive when measured",
            });
        }
        for (field, value) in [
            ("pose yaw span", diagnostics.pose_diversity.yaw_span_radians),
            (
                "pose pitch span",
                diagnostics.pose_diversity.pitch_span_radians,
            ),
            (
                "near duplicate fraction",
                diagnostics.pose_diversity.near_duplicate_fraction,
            ),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(GnmIdentityCalibrationError::InvalidOutput {
                    field,
                    reason: "pose diagnostic must be finite and non-negative",
                });
            }
        }

        Ok(Self {
            model_version: model.version(),
            mapping_version,
            identity,
            neutral_expression_reference,
            neutral_surface_reference,
            normalization_scales,
            diagnostics,
        })
    }

    /// Returns the loaded GNM model version to which this calibration is bound.
    pub fn model_version(&self) -> GnmVersion {
        self.model_version
    }

    /// Returns the exact dense mapping version used for calibration.
    pub fn mapping_version(&self) -> DenseMappingVersion {
        self.mapping_version
    }

    /// Returns the fixed identity through a read-only reference.
    pub fn identity(&self) -> &FixedGnmIdentity {
        &self.identity
    }

    /// Returns the neutral expression reference through a read-only reference.
    pub fn neutral_expression_reference(&self) -> &GnmExpressionState {
        &self.neutral_expression_reference
    }

    /// Returns the neutral selected-surface geometry through a read-only slice.
    pub fn neutral_surface_reference(&self) -> &[[f32; 3]] {
        &self.neutral_surface_reference
    }

    /// Returns optional normalization scales.
    pub fn normalization_scales(&self) -> NeutralNormalizationScales {
        self.normalization_scales
    }

    /// Returns numerical calibration diagnostics.
    pub fn diagnostics(&self) -> IdentityFitDiagnostics {
        self.diagnostics
    }

    /// Returns whether the calibration exactly matches the runtime model/mapping boundary.
    pub fn matches_runtime(&self, model: &GnmModel, mapping: DenseMappingVersion) -> bool {
        self.model_version == model.version() && self.mapping_version == mapping
    }
}

/// Typed error from neutral selection or immutable calibration validation.
#[derive(Debug)]
pub enum GnmIdentityCalibrationError {
    /// Candidate selection configuration is invalid.
    InvalidSelectionConfig(&'static str),
    /// Shared-identity solve configuration is invalid.
    InvalidSolveConfig(&'static str),
    /// Shared-identity solve input is invalid, optionally attributed to one
    /// sample index in the input window.
    InvalidSolveInput {
        /// Sample index when the failure belongs to one sample.
        sample: Option<usize>,
        /// Validation reason.
        reason: String,
    },
    /// Fixed identity dimension differs from the loaded model.
    IdentityDimensionMismatch {
        /// Expected loaded-model dimension.
        expected: usize,
        /// Actual coefficient count.
        actual: usize,
    },
    /// Neutral-expression dimension differs from the loaded model.
    ExpressionDimensionMismatch {
        /// Expected loaded-model dimension.
        expected: usize,
        /// Actual coefficient count.
        actual: usize,
    },
    /// Dense reprojection evaluation failed while scoring a solve state.
    Evaluation(GnmReprojectionError),
    /// A linear solve encountered a singular/ill-conditioned block.
    IllConditionedSolve,
    /// The solve objective became non-finite at the given iteration.
    NonFiniteObjective {
        /// Iteration index (1-based) where the objective broke down.
        iteration: usize,
    },
    /// Mapping/model versions differ.
    VersionMismatch {
        /// Model version recorded by mapping/calibration.
        calibration_model: GnmVersion,
        /// Currently loaded model version.
        runtime_model: GnmVersion,
    },
    /// Numerical calibration output is invalid/non-finite.
    InvalidOutput {
        /// Invalid output field.
        field: &'static str,
        /// Validation reason.
        reason: &'static str,
    },
}

impl std::fmt::Display for GnmIdentityCalibrationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSelectionConfig(reason) => {
                write!(formatter, "invalid neutral selection config: {reason}")
            }
            Self::InvalidSolveConfig(reason) => {
                write!(formatter, "invalid shared identity solve config: {reason}")
            }
            Self::InvalidSolveInput { sample, reason } => match sample {
                Some(sample) => {
                    write!(formatter, "invalid solve sample {sample}: {reason}")
                }
                None => write!(formatter, "invalid shared identity solve input: {reason}"),
            },
            Self::IdentityDimensionMismatch { expected, actual } => write!(
                formatter,
                "GNM identity dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::ExpressionDimensionMismatch { expected, actual } => write!(
                formatter,
                "GNM neutral-expression dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::Evaluation(error) => {
                write!(formatter, "neutral sample evaluation failed: {error}")
            }
            Self::IllConditionedSolve => write!(
                formatter,
                "shared identity linear solve hit an ill-conditioned block"
            ),
            Self::NonFiniteObjective { iteration } => write!(
                formatter,
                "shared identity solve diverged to a non-finite objective at iteration {iteration}"
            ),
            Self::VersionMismatch {
                calibration_model,
                runtime_model,
            } => write!(
                formatter,
                "GNM calibration model {}.{} does not match runtime {}.{}",
                calibration_model.major,
                calibration_model.minor,
                runtime_model.major,
                runtime_model.minor
            ),
            Self::InvalidOutput { field, reason } => {
                write!(
                    formatter,
                    "invalid GNM identity calibration {field}: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for GnmIdentityCalibrationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DenseArray, GNM_HEAD_V3_EXPRESSION_DIM, GNM_HEAD_V3_IDENTITY_DIM, GNM_HEAD_V3_VERSION,
        GnmDenseObservation, GnmModelData, GnmVariant, SynthesisOptions,
    };

    fn selection_config() -> NeutralCalibrationSelectionConfig {
        NeutralCalibrationSelectionConfig::new(3, 0.05, 0.25, 0.10, 0.01, 0.75).unwrap()
    }

    fn candidate(seq: u64, timestamp: u64, yaw: f32, pitch: f32) -> NeutralCalibrationCandidate {
        NeutralCalibrationCandidate {
            source_seq: seq,
            captured_at_micros: timestamp,
            coverage: DenseCoverageSummary {
                mapped_points: 120,
                valid_points: 110,
                effective_weight: 100.0,
                status: DenseObservationStatus::Valid,
            },
            reprojection_rms: 0.01,
            expression_activity: None,
            yaw_radians: yaw,
            pitch_radians: pitch,
            tracking_degraded: false,
        }
    }

    fn synthetic_model_data() -> GnmModelData {
        let identity = GNM_HEAD_V3_IDENTITY_DIM;
        let expression = GNM_HEAD_V3_EXPRESSION_DIM;
        GnmModelData {
            version: GNM_HEAD_V3_VERSION,
            variant: GnmVariant::Head,
            template_vertices: DenseArray::new(
                "vertices",
                vec![3, 3],
                vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            )
            .unwrap(),
            template_joints: DenseArray::new("joints", vec![1, 3], vec![0.0; 3]).unwrap(),
            vertex_identity_basis: DenseArray::new(
                "identity",
                vec![identity, 3, 3],
                vec![0.0; identity * 9],
            )
            .unwrap(),
            joint_identity_basis: DenseArray::new(
                "joint_identity",
                vec![identity, 1, 3],
                vec![0.0; identity * 3],
            )
            .unwrap(),
            expression_basis: DenseArray::new(
                "expression",
                vec![expression, 3, 3],
                vec![0.0; expression * 9],
            )
            .unwrap(),
            joint_parent_indices: vec![-1],
            skinning_weights: DenseArray::new("weights", vec![1, 3], vec![1.0; 3]).unwrap(),
            pose_correctives_regressor: None,
        }
    }

    fn synthetic_model() -> GnmModel {
        GnmModel::from_data(synthetic_model_data()).unwrap()
    }

    fn mapping_version() -> DenseMappingVersion {
        DenseMappingVersion {
            schema_revision: 1,
            model_version: GNM_HEAD_V3_VERSION,
        }
    }

    #[test]
    fn selection_rejects_duplicate_outlier_expression_and_degraded_candidates() {
        let good1 = candidate(1, 1_000, -0.1, 0.0);
        let duplicate = candidate(1, 1_010, -0.05, 0.0);
        let mut residual = candidate(2, 1_020, 0.0, 0.0);
        residual.reprojection_rms = 0.5;
        let mut expressive = candidate(3, 1_030, 0.05, 0.0);
        expressive.expression_activity = Some(0.9);
        let mut degraded = candidate(4, 1_040, 0.10, 0.0);
        degraded.tracking_degraded = true;
        let good2 = candidate(5, 1_050, 0.05, 0.0);
        let good3 = candidate(6, 1_060, 0.15, 0.05);
        let selection = select_neutral_calibration_candidates(
            &[
                good1, duplicate, residual, expressive, degraded, good2, good3,
            ],
            selection_config(),
        );
        assert_eq!(selection.accepted_indices, vec![0, 5, 6]);
        assert_eq!(selection.rejections.len(), 4);
        assert_eq!(
            selection.diagnostics.readiness,
            NeutralCalibrationReadiness::ReadyForIdentitySolve
        );
    }

    #[test]
    fn near_identical_window_is_not_misreported_as_ready() {
        let selection = select_neutral_calibration_candidates(
            &[
                candidate(1, 1_000, 0.0, 0.0),
                candidate(2, 1_010, 0.001, 0.001),
                candidate(3, 1_020, 0.002, 0.002),
                candidate(4, 1_030, 0.003, 0.003),
            ],
            selection_config(),
        );
        assert_eq!(
            selection.diagnostics.readiness,
            NeutralCalibrationReadiness::InsufficientPoseDiversity
        );
        assert!(selection.diagnostics.pose_diversity.near_duplicate_fraction > 0.75);
    }

    #[test]
    fn optional_expression_proxy_absence_is_not_treated_as_zero_authority() {
        let selection = select_neutral_calibration_candidates(
            &[
                candidate(1, 1_000, -0.1, 0.0),
                candidate(2, 1_010, 0.0, 0.0),
                candidate(3, 1_020, 0.1, 0.0),
            ],
            selection_config(),
        );
        assert_eq!(selection.accepted_indices.len(), 3);
    }

    #[test]
    fn fixed_identity_and_calibration_are_version_bound_and_read_only() {
        let model = synthetic_model();
        let fixed = FixedGnmIdentity::new(model.neutral_identity(), &model).unwrap();
        let before = fixed.values().to_vec();
        let calibration = GnmIdentityCalibration::new(
            &model,
            mapping_version(),
            fixed,
            model.neutral_expression(),
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            NeutralNormalizationScales {
                inter_ocular: Some(1.0),
                mouth_width: Some(0.5),
                eye_aperture: None,
            },
            IdentityFitDiagnostics {
                accepted_samples: 8,
                rejected_samples: 2,
                reprojection_rms: 0.01,
                active_identity_dimension: 32,
                condition_number: Some(12.0),
                pose_diversity: NeutralPoseDiversity {
                    yaw_span_radians: 0.2,
                    pitch_span_radians: 0.1,
                    near_duplicate_fraction: 0.1,
                },
            },
        )
        .unwrap();
        assert_eq!(calibration.identity().values(), before.as_slice());
        assert!(calibration.matches_runtime(&model, mapping_version()));
    }

    #[test]
    fn mapping_model_mismatch_rejects_old_calibration_contract() {
        let model = synthetic_model();
        let fixed = FixedGnmIdentity::new(model.neutral_identity(), &model).unwrap();
        let mismatched = DenseMappingVersion {
            schema_revision: 1,
            model_version: GnmVersion { major: 9, minor: 0 },
        };
        let result = GnmIdentityCalibration::new(
            &model,
            mismatched,
            fixed,
            model.neutral_expression(),
            vec![[0.0, 0.0, 0.0]],
            NeutralNormalizationScales::default(),
            IdentityFitDiagnostics {
                accepted_samples: 3,
                rejected_samples: 0,
                reprojection_rms: 0.01,
                active_identity_dimension: 1,
                condition_number: None,
                pose_diversity: NeutralPoseDiversity {
                    yaw_span_radians: 0.2,
                    pitch_span_radians: 0.0,
                    near_duplicate_fraction: 0.0,
                },
            },
        );
        assert!(matches!(
            result,
            Err(GnmIdentityCalibrationError::VersionMismatch { .. })
        ));
    }

    #[test]
    fn non_finite_surface_or_normalization_scale_is_rejected() {
        let model = synthetic_model();
        let fixed = FixedGnmIdentity::new(model.neutral_identity(), &model).unwrap();
        let result = GnmIdentityCalibration::new(
            &model,
            mapping_version(),
            fixed,
            model.neutral_expression(),
            vec![[f32::NAN, 0.0, 0.0]],
            NeutralNormalizationScales::default(),
            IdentityFitDiagnostics {
                accepted_samples: 3,
                rejected_samples: 0,
                reprojection_rms: 0.01,
                active_identity_dimension: 1,
                condition_number: None,
                pose_diversity: NeutralPoseDiversity {
                    yaw_span_radians: 0.2,
                    pitch_span_radians: 0.0,
                    near_duplicate_fraction: 0.0,
                },
            },
        );
        assert!(matches!(
            result,
            Err(GnmIdentityCalibrationError::InvalidOutput {
                field: "neutral_surface_reference",
                ..
            })
        ));
    }

    // -- Shared-identity solve contract (Issue #54.1) fixtures ---------------

    fn solve_mapping(model: &GnmModel) -> DenseCorrespondenceSet {
        let rows: Vec<crate::MediaPipeGnmDenseCorrespondence> = (0..model.vertex_count())
            .map(|index| crate::MediaPipeGnmDenseCorrespondence {
                mediapipe_index: 10 + index,
                target: crate::GnmSurfacePointRef::Vertex {
                    vertex_index: index,
                },
                region: crate::FaceRegion::Other,
                anatomical_side: crate::AnatomicalSide::Midline,
                base_weight: 1.0,
                provenance: crate::CorrespondenceProvenance::RepositoryValidated,
                reliability: crate::CorrespondenceReliability::High,
            })
            .collect();
        DenseCorrespondenceSet::new(mapping_version(), rows, model).unwrap()
    }

    fn solve_observation(
        seq: u64,
        timestamp: u64,
        mapping: &DenseCorrespondenceSet,
    ) -> GnmDenseObservation {
        let landmarks = vec![[0.5f32, 0.5]; crate::MEDIAPIPE_FACE_LANDMARK_COUNT];
        GnmDenseObservation::from_mediapipe_xy(
            seq,
            timestamp,
            &landmarks,
            mapping,
            crate::DenseCoveragePolicy::new(2, 0.75).unwrap(),
        )
        .unwrap()
    }

    fn solve_nuisance(model: &GnmModel) -> SampleNuisance {
        SampleNuisance::new(
            crate::DenseProjection::new([0.1, -0.05, 0.02], [0.0, 0.0, 0.6], 1.3, [0.5, 0.5])
                .unwrap(),
            model.neutral_expression(),
            model.expression_dimension(),
        )
        .unwrap()
    }

    fn solve_input<'a>(
        model: &'a GnmModel,
        mapping: &'a DenseCorrespondenceSet,
        observations: &'a [GnmDenseObservation],
    ) -> SharedIdentitySolveInput<'a> {
        let samples = observations
            .iter()
            .map(|observation| NeutralSampleSolveInput::new(observation, solve_nuisance(model)))
            .collect();
        SharedIdentitySolveInput::new(model, mapping, model.neutral_identity(), samples).unwrap()
    }

    fn solve_config(model: &GnmModel) -> SharedIdentitySolveConfig {
        SharedIdentitySolveConfig::new(
            model.identity_dimension(),
            10,
            1.0,
            1.0e-6,
            1.0e-10,
            crate::DenseReprojectionConfig::default(),
        )
        .unwrap()
    }

    #[test]
    fn solve_input_keeps_single_shared_identity_and_sample_order() {
        let model = synthetic_model();
        let mapping = solve_mapping(&model);
        let observations = [
            solve_observation(1, 1_000, &mapping),
            solve_observation(2, 1_010, &mapping),
        ];
        let input = solve_input(&model, &mapping, &observations);
        assert_eq!(input.samples().len(), 2);
        assert_eq!(input.samples()[0].observation().source_seq(), 1);
        assert_eq!(input.samples()[1].observation().source_seq(), 2);
        assert_eq!(
            input.identity_initial().values().len(),
            model.identity_dimension()
        );
        assert_eq!(input.model().version(), model.version());
        assert_eq!(input.mapping_version(), mapping.version());
        // Nuisance stays per sample; identity exists only once at the input.
        assert_eq!(input.samples()[0].nuisance(), input.samples()[1].nuisance());
    }

    #[test]
    fn solve_input_rejects_empty_window_and_empty_sample() {
        let model = synthetic_model();
        let mapping = solve_mapping(&model);
        let empty =
            SharedIdentitySolveInput::new(&model, &mapping, model.neutral_identity(), Vec::new());
        assert!(matches!(
            empty,
            Err(GnmIdentityCalibrationError::InvalidSolveInput { sample: None, .. })
        ));

        let landmarks = vec![[f32::NAN; 2]; crate::MEDIAPIPE_FACE_LANDMARK_COUNT];
        let empty_observation = GnmDenseObservation::from_mediapipe_xy(
            1,
            1_000,
            &landmarks,
            &mapping,
            crate::DenseCoveragePolicy::new(2, 0.75).unwrap(),
        )
        .unwrap();
        assert!(empty_observation.points().is_empty());
        let sample = vec![NeutralSampleSolveInput::new(
            &empty_observation,
            solve_nuisance(&model),
        )];
        let error =
            SharedIdentitySolveInput::new(&model, &mapping, model.neutral_identity(), sample)
                .unwrap_err();
        assert!(matches!(
            error,
            GnmIdentityCalibrationError::InvalidSolveInput {
                sample: Some(0),
                ..
            }
        ));
    }

    #[test]
    fn solve_input_rejects_insufficient_coverage_samples() {
        let model = synthetic_model();
        let mapping = solve_mapping(&model);
        let mut landmarks = vec![[f32::NAN; 2]; crate::MEDIAPIPE_FACE_LANDMARK_COUNT];
        landmarks[10] = [0.5, 0.5];
        let observation = GnmDenseObservation::from_mediapipe_xy(
            1,
            1_000,
            &landmarks,
            &mapping,
            crate::DenseCoveragePolicy::new(2, 0.75).unwrap(),
        )
        .unwrap();
        assert_eq!(
            observation.coverage().status,
            DenseObservationStatus::Insufficient
        );
        let sample = vec![NeutralSampleSolveInput::new(
            &observation,
            solve_nuisance(&model),
        )];
        let error =
            SharedIdentitySolveInput::new(&model, &mapping, model.neutral_identity(), sample)
                .unwrap_err();
        assert!(matches!(
            error,
            GnmIdentityCalibrationError::InvalidSolveInput {
                sample: Some(0),
                ..
            }
        ));
    }

    #[test]
    fn solve_input_rejects_duplicate_source_sequences() {
        let model = synthetic_model();
        let mapping = solve_mapping(&model);
        let observations = [
            solve_observation(1, 1_000, &mapping),
            solve_observation(1, 1_010, &mapping),
        ];
        let samples = observations
            .iter()
            .map(|observation| NeutralSampleSolveInput::new(observation, solve_nuisance(&model)))
            .collect();
        let error =
            SharedIdentitySolveInput::new(&model, &mapping, model.neutral_identity(), samples)
                .unwrap_err();
        assert!(matches!(
            error,
            GnmIdentityCalibrationError::InvalidSolveInput {
                sample: Some(1),
                ..
            }
        ));
    }

    #[test]
    fn solve_input_rejects_identity_dimension_mismatch() {
        let model = synthetic_model();
        let mapping = solve_mapping(&model);
        let short = GnmIdentityState::new(
            vec![0.0; model.identity_dimension() - 1],
            model.identity_dimension() - 1,
        )
        .unwrap();
        let observation = solve_observation(1, 1_000, &mapping);
        let sample = vec![NeutralSampleSolveInput::new(
            &observation,
            solve_nuisance(&model),
        )];
        let error = SharedIdentitySolveInput::new(&model, &mapping, short, sample).unwrap_err();
        assert!(matches!(
            error,
            GnmIdentityCalibrationError::IdentityDimensionMismatch { .. }
        ));
    }

    #[test]
    fn solve_nuisance_rejects_wrong_expression_dimension() {
        let model = synthetic_model();
        let short = GnmExpressionState::new(
            vec![0.0; model.expression_dimension() - 1],
            model.expression_dimension() - 1,
        )
        .unwrap();
        let result = SampleNuisance::new(
            crate::DenseProjection::new([0.0; 3], [0.0; 3], 1.3, [0.5, 0.5]).unwrap(),
            short,
            model.expression_dimension(),
        );
        assert!(matches!(
            result,
            Err(GnmIdentityCalibrationError::InvalidSolveInput { sample: None, .. })
        ));
    }

    #[test]
    fn solve_config_rejects_invalid_values() {
        let reprojection = crate::DenseReprojectionConfig::default();
        assert!(
            SharedIdentitySolveConfig::new(0, 5, 1.0, 1.0e-6, 1.0e-10, reprojection).is_err(),
            "zero active dimension must be rejected"
        );
        assert!(
            SharedIdentitySolveConfig::new(8, 0, 1.0, 1.0e-6, 1.0e-10, reprojection).is_err(),
            "zero iteration budget must be rejected"
        );
        for weight in [-1.0f64, f64::NAN, f64::INFINITY] {
            assert!(
                SharedIdentitySolveConfig::new(8, 5, weight, 1.0e-6, 1.0e-10, reprojection)
                    .is_err(),
                "identity prior weight {weight} must be rejected"
            );
        }
        for ridge in [0.0f64, -1.0e-6, f64::NAN] {
            assert!(
                SharedIdentitySolveConfig::new(8, 5, 1.0, ridge, 1.0e-10, reprojection).is_err(),
                "conditioning regularization {ridge} must be rejected"
            );
        }
        for tolerance in [-1.0f64, f64::NAN] {
            assert!(
                SharedIdentitySolveConfig::new(8, 5, 1.0, 1.0e-6, tolerance, reprojection).is_err(),
                "convergence tolerance {tolerance} must be rejected"
            );
        }
        // A zero prior weight is a legitimate pure-data-fit configuration.
        assert!(SharedIdentitySolveConfig::new(8, 5, 0.0, 1.0e-6, 1.0e-10, reprojection).is_ok());
    }

    // -- Per-sample residual (Issue #54.2a) fixtures -------------------------

    fn truth_projection() -> DenseProjection {
        DenseProjection::new([0.12, -0.08, 0.03], [0.0, 0.0, 0.6], 1.4, [0.5, 0.5]).unwrap()
    }

    fn truth_observation(
        model: &GnmModel,
        mapping: &DenseCorrespondenceSet,
    ) -> GnmDenseObservation {
        crate::synthesize_observation_from_projection(
            model,
            &model.neutral_identity(),
            &model.neutral_expression(),
            &GnmJointState::neutral(model.joint_count()),
            mapping,
            &truth_projection(),
            SynthesisOptions {
                source_seq: 1,
                captured_at_micros: 1_000,
                noise_amplitude: 0.0,
                noise_seed: 0,
            },
            crate::DenseCoveragePolicy::new(1, 1.0).unwrap(),
            |_, _| false,
        )
        .unwrap()
    }

    #[test]
    fn sample_residual_at_synthetic_truth_is_near_zero() {
        let model = synthetic_model();
        let mapping = solve_mapping(&model);
        let observation = truth_observation(&model, &mapping);
        let nuisance = SampleNuisance::new(
            truth_projection(),
            model.neutral_expression(),
            model.expression_dimension(),
        )
        .unwrap();
        let residual = evaluate_neutral_sample_residual(
            &model,
            &mapping,
            &model.neutral_identity(),
            &observation,
            &nuisance,
            crate::DenseReprojectionConfig::default(),
        )
        .unwrap();
        assert!(residual.reprojection.weighted_rms() < 1.0e-4);
        assert_eq!(residual.neutral_expression_squared, 0.0);
    }

    #[test]
    fn sample_residual_penalizes_nonzero_nuisance_expression() {
        let model = synthetic_model();
        let mapping = solve_mapping(&model);
        let observation = truth_observation(&model, &mapping);
        let mut values = model.neutral_expression().values().to_vec();
        values[0] = 0.25;
        let expression = GnmExpressionState::new(values, model.expression_dimension()).unwrap();
        let nuisance =
            SampleNuisance::new(truth_projection(), expression, model.expression_dimension())
                .unwrap();
        let residual = evaluate_neutral_sample_residual(
            &model,
            &mapping,
            &model.neutral_identity(),
            &observation,
            &nuisance,
            crate::DenseReprojectionConfig::default(),
        )
        .unwrap();
        assert!((residual.neutral_expression_squared - 0.0625).abs() < 1.0e-6);
    }

    #[test]
    fn sample_residual_fails_closed_on_dimension_mismatch() {
        let model = synthetic_model();
        let mapping = solve_mapping(&model);
        let observation = truth_observation(&model, &mapping);

        let short_expression = GnmExpressionState::new(
            vec![0.0; model.expression_dimension() - 1],
            model.expression_dimension() - 1,
        )
        .unwrap();
        assert!(matches!(
            evaluate_neutral_sample_residual(
                &model,
                &mapping,
                &model.neutral_identity(),
                &observation,
                &SampleNuisance::new(
                    truth_projection(),
                    short_expression,
                    model.expression_dimension() - 1
                )
                .unwrap(),
                crate::DenseReprojectionConfig::default(),
            ),
            Err(GnmIdentityCalibrationError::ExpressionDimensionMismatch { .. })
        ));
    }

    #[test]
    fn validate_shared_identity_solve_bounds_active_dimension() {
        let model = synthetic_model();
        let mapping = solve_mapping(&model);
        let observations = [
            solve_observation(1, 1_000, &mapping),
            solve_observation(2, 1_010, &mapping),
        ];
        let input = solve_input(&model, &mapping, &observations);
        assert!(validate_shared_identity_solve(&input, solve_config(&model)).is_ok());

        let too_wide = SharedIdentitySolveConfig::new(
            model.identity_dimension() + 1,
            10,
            1.0,
            1.0e-6,
            1.0e-10,
            crate::DenseReprojectionConfig::default(),
        )
        .unwrap();
        assert!(matches!(
            validate_shared_identity_solve(&input, too_wide),
            Err(GnmIdentityCalibrationError::InvalidSolveConfig(_))
        ));
    }

    // -- Shared-identity linear system (Issue #54.2b) tests ------------------

    #[test]
    fn linear_system_has_one_shared_identity_block_and_separated_nuisance_blocks() {
        let model = synthetic_model();
        let mapping = solve_mapping(&model);
        let observations = [
            solve_observation(1, 1_000, &mapping),
            solve_observation(2, 1_010, &mapping),
        ];
        let input = solve_input(&model, &mapping, &observations);
        let system =
            assemble_shared_identity_linear_system(&input, solve_config(&model), 1.0e-4).unwrap();
        assert_eq!(system.samples().len(), 2);
        assert_eq!(
            system.active_identity_dimension(),
            model.identity_dimension()
        );
        assert_eq!(
            system.identity_hessian().len(),
            model.identity_dimension() * model.identity_dimension()
        );
        for block in system.samples() {
            let dimension = 9 + model.expression_dimension();
            assert_eq!(block.parameter_dimension, dimension);
            assert_eq!(block.hessian.len(), dimension * dimension);
            assert_eq!(
                block.cross_hessian.len(),
                model.identity_dimension() * dimension
            );
            // Neutral-expression penalty contributes +1 to expression diagonals.
            for offset in 0..model.expression_dimension() {
                let coordinate = 9 + offset;
                assert!(block.hessian[coordinate * dimension + coordinate] >= 1.0);
            }
            // Projection coordinates carry no analytic penalty diagonal.
            assert!((block.hessian[0] - block.hessian[0]).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn linear_system_aggregation_is_order_invariant_within_tolerance() {
        let model = synthetic_model();
        let mapping = solve_mapping(&model);
        let observations = [
            solve_observation(1, 1_000, &mapping),
            solve_observation(2, 1_010, &mapping),
            solve_observation(3, 1_020, &mapping),
        ];
        let forward = solve_input(&model, &mapping, &observations);
        let mut reversed_observations = observations.clone();
        reversed_observations.reverse();
        let reversed = solve_input(&model, &mapping, &reversed_observations);

        let tolerance = 1.0e-9;
        for config in [
            solve_config(&model),
            SharedIdentitySolveConfig::new(
                8,
                10,
                2.5,
                5.0e-3,
                1.0e-10,
                crate::DenseReprojectionConfig::default(),
            )
            .unwrap(),
        ] {
            let a = assemble_shared_identity_linear_system(&forward, config, 1.0e-4).unwrap();
            let b = assemble_shared_identity_linear_system(&reversed, config, 1.0e-4).unwrap();
            for (left, right) in a.identity_hessian().iter().zip(b.identity_hessian()) {
                assert!(
                    (left - right).abs() <= tolerance,
                    "identity hessian differs across sample order"
                );
            }
            assert!(a.identity_gradient() == b.identity_gradient());
            // Nuisance blocks follow their samples; match them by source_seq.
            for block in a.samples() {
                let counterpart = b
                    .samples()
                    .iter()
                    .find(|candidate| candidate.source_seq == block.source_seq)
                    .unwrap();
                for (left, right) in block.hessian.iter().zip(counterpart.hessian.iter()) {
                    assert!((left - right).abs() <= tolerance);
                }
                assert_eq!(block.gradient, counterpart.gradient);
                for (left, right) in block
                    .cross_hessian
                    .iter()
                    .zip(counterpart.cross_hessian.iter())
                {
                    assert!((left - right).abs() <= tolerance);
                }
            }
        }
    }

    #[test]
    fn linear_system_applies_prior_and_ridge_to_identity_block() {
        let model = synthetic_model();
        let mapping = solve_mapping(&model);
        let observations = [solve_observation(1, 1_000, &mapping)];
        let input = solve_input(&model, &mapping, &observations);
        let prior = 2.5;
        let ridge = 0.125;
        let config = SharedIdentitySolveConfig::new(
            8,
            10,
            prior,
            ridge,
            1.0e-10,
            crate::DenseReprojectionConfig::default(),
        )
        .unwrap();
        let system = assemble_shared_identity_linear_system(&input, config, 1.0e-4).unwrap();
        // All-zero synthetic bases give an all-zero Jacobian; only the explicit
        // prior and ridge remain on the identity diagonals.
        for index in 0..8 {
            let diagonal = system.identity_hessian()[index * 8 + index];
            let expected = if index < 8 { prior + ridge } else { 0.0 };
            assert!(
                (diagonal - expected).abs() < 1.0e-12,
                "diagonal {index} = {diagonal}"
            );
            assert_eq!(system.identity_gradient()[index], 0.0);
        }
    }

    #[test]
    fn linear_system_rejects_invalid_jacobian_step() {
        let model = synthetic_model();
        let mapping = solve_mapping(&model);
        let observations = [solve_observation(1, 1_000, &mapping)];
        let input = solve_input(&model, &mapping, &observations);
        for step in [0.0f64, -1.0e-6, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                assemble_shared_identity_linear_system(&input, solve_config(&model), step),
                Err(GnmIdentityCalibrationError::InvalidSolveConfig(_))
            ));
        }
    }

    // -- Bounded alternating solve (Issue #54.2c) tests ----------------------

    fn solve_fixture() -> (
        GnmModel,
        DenseCorrespondenceSet,
        Vec<GnmDenseObservation>,
        Vec<DenseProjection>,
    ) {
        let model = synthetic_model();
        let mapping = solve_mapping(&model);
        let truths = [
            DenseProjection::new([0.10, -0.06, 0.02], [0.0, 0.0, 0.6], 1.4, [0.5, 0.5]).unwrap(),
            DenseProjection::new([-0.08, 0.04, -0.01], [0.0, 0.01, 0.62], 1.35, [0.49, 0.51])
                .unwrap(),
        ];
        let observations = truths
            .iter()
            .enumerate()
            .map(|(index, truth)| {
                crate::synthesize_observation_from_projection(
                    &model,
                    &model.neutral_identity(),
                    &model.neutral_expression(),
                    &GnmJointState::neutral(model.joint_count()),
                    &mapping,
                    truth,
                    SynthesisOptions {
                        source_seq: 10 + index as u64,
                        captured_at_micros: 1_000 + index as u64 * 10,
                        noise_amplitude: 0.0,
                        noise_seed: index as u64,
                    },
                    crate::DenseCoveragePolicy::new(1, 1.0).unwrap(),
                    |_, _| false,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        (model, mapping, observations, truths.to_vec())
    }

    #[test]
    fn alternating_solve_converges_finite_on_known_identity_multi_pose_fixture() {
        let (model, mapping, observations, truths) = solve_fixture();
        let samples = observations
            .iter()
            .enumerate()
            .map(|(index, observation)| {
                // Start nuisances perturbed away from the truth poses.
                let truth = truths[index];
                let perturbed = DenseProjection::new(
                    [
                        truth.yaw_pitch_roll()[0] + 0.03,
                        truth.yaw_pitch_roll()[1] - 0.02,
                        truth.yaw_pitch_roll()[2],
                    ],
                    truth.translation(),
                    truth.focal() * 1.05,
                    truth.principal_point(),
                )
                .unwrap();
                NeutralSampleSolveInput::new(
                    observation,
                    SampleNuisance::new(
                        perturbed,
                        model.neutral_expression(),
                        model.expression_dimension(),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let input =
            SharedIdentitySolveInput::new(&model, &mapping, model.neutral_identity(), samples)
                .unwrap();
        let config = SharedIdentitySolveConfig::new(
            8,
            20,
            0.5,
            1.0e-8,
            1.0e-9,
            crate::DenseReprojectionConfig::default(),
        )
        .unwrap();
        let outcome = solve_shared_identity(&input, config, 1.0e-4).unwrap();

        assert!(outcome.final_objective.is_finite());
        assert!(
            outcome.converged,
            "expected convergence, history: {:?}",
            outcome.objective_history
        );
        assert!(outcome.final_objective < outcome.objective_history[0]);
        // Bounded policy: accepted updates never worsen the objective.
        for pair in outcome.objective_history.windows(2) {
            assert!(
                pair[1] <= pair[0] + 1.0e-12,
                "objective history increased: {:?}",
                pair
            );
        }
        // The synthetic model's identity basis is all-zero, so the shared
        // identity stays at its known initial value.
        assert_eq!(outcome.identity.values(), model.neutral_identity().values());
        for nuisance in &outcome.nuisances {
            assert!(
                nuisance
                    .projection()
                    .yaw_pitch_roll()
                    .iter()
                    .all(|value| value.is_finite())
            );
            assert!(
                nuisance
                    .expression()
                    .values()
                    .iter()
                    .all(|value| value.is_finite())
            );
        }
    }

    #[test]
    fn alternating_solve_respects_iteration_budget() {
        let (model, mapping, observations, truths) = solve_fixture();
        let samples = observations
            .iter()
            .enumerate()
            .map(|(index, observation)| {
                NeutralSampleSolveInput::new(
                    observation,
                    SampleNuisance::new(
                        truths[index],
                        model.neutral_expression(),
                        model.expression_dimension(),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let input =
            SharedIdentitySolveInput::new(&model, &mapping, model.neutral_identity(), samples)
                .unwrap();
        let config = SharedIdentitySolveConfig::new(
            4,
            1,
            1.0,
            1.0e-6,
            1.0e-12,
            crate::DenseReprojectionConfig::default(),
        )
        .unwrap();
        let outcome = solve_shared_identity(&input, config, 1.0e-4).unwrap();
        assert!(outcome.iterations <= 1);
        assert!(outcome.final_objective.is_finite());
    }

    #[test]
    fn singular_linear_block_is_a_typed_failure() {
        assert!(matches!(
            solve_symmetric_system(&[0.0], &[1.0]),
            Err(GnmIdentityCalibrationError::IllConditionedSolve)
        ));
        assert!(matches!(
            solve_symmetric_system(&[f64::NAN], &[1.0]),
            Err(GnmIdentityCalibrationError::IllConditionedSolve)
        ));
    }

    // -- Calibration output assembly (Issue #54.2d) tests -------------------

    #[test]
    fn finalize_produces_deterministic_calibration_from_same_solver_result() {
        let (model, mapping, observations, truths) = solve_fixture();
        let samples = observations
            .iter()
            .enumerate()
            .map(|(index, observation)| {
                NeutralSampleSolveInput::new(
                    observation,
                    SampleNuisance::new(
                        truths[index],
                        model.neutral_expression(),
                        model.expression_dimension(),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let input =
            SharedIdentitySolveInput::new(&model, &mapping, model.neutral_identity(), samples)
                .unwrap();
        let config = SharedIdentitySolveConfig::new(
            8,
            10,
            1.0,
            1.0e-6,
            1.0e-9,
            crate::DenseReprojectionConfig::default(),
        )
        .unwrap();
        let outcome = solve_shared_identity(&input, config, 1.0e-4).unwrap();
        let candidates: Vec<NeutralCalibrationCandidate> = observations
            .iter()
            .enumerate()
            .map(|(index, _)| candidate(10 + index as u64, 1_000 + index as u64 * 10, 0.1, 0.05))
            .collect();
        let selection = select_neutral_calibration_candidates(&candidates, selection_config());

        let first =
            finalize_identity_calibration(&model, &mapping, &input, &selection, &outcome, &config)
                .unwrap();
        let second =
            finalize_identity_calibration(&model, &mapping, &input, &selection, &outcome, &config)
                .unwrap();
        assert_eq!(first, second);
        assert!(first.matches_runtime(&model, mapping.version()));
        assert_eq!(first.identity().values(), outcome.identity.values());
        let diagnostics = first.diagnostics();
        assert_eq!(
            diagnostics.accepted_samples,
            selection.diagnostics.accepted_candidates
        );
        assert_eq!(
            diagnostics.rejected_samples,
            selection.diagnostics.rejected_candidates
        );
        assert_eq!(diagnostics.active_identity_dimension, 8);
        // Synthetic mapping classifies every row as Other; scales stay absent.
        assert_eq!(
            first.normalization_scales(),
            NeutralNormalizationScales::default()
        );
        assert!(!first.neutral_surface_reference().is_empty());
        assert!(
            diagnostics
                .condition_number
                .is_some_and(|value| value.is_finite() && value > 0.0)
        );
    }

    #[test]
    fn finalize_rejects_non_finite_solver_result() {
        let (model, mapping, observations, truths) = solve_fixture();
        let samples = observations
            .iter()
            .enumerate()
            .map(|(index, observation)| {
                NeutralSampleSolveInput::new(
                    observation,
                    SampleNuisance::new(
                        truths[index],
                        model.neutral_expression(),
                        model.expression_dimension(),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let input =
            SharedIdentitySolveInput::new(&model, &mapping, model.neutral_identity(), samples)
                .unwrap();
        let config = SharedIdentitySolveConfig::new(
            4,
            2,
            1.0,
            1.0e-6,
            1.0e-9,
            crate::DenseReprojectionConfig::default(),
        )
        .unwrap();
        let outcome = solve_shared_identity(&input, config, 1.0e-4).unwrap();
        let selection = select_neutral_calibration_candidates(
            &[candidate(10, 1_000, 0.1, 0.0)],
            selection_config(),
        );

        // A diverged (non-finite) solver outcome is never published.
        let mut diverged_outcome = outcome.clone();
        diverged_outcome.final_objective = f64::NAN;
        assert!(matches!(
            finalize_identity_calibration(
                &model,
                &mapping,
                &input,
                &selection,
                &diverged_outcome,
                &config
            ),
            Err(GnmIdentityCalibrationError::InvalidOutput {
                field: "solver objective",
                ..
            })
        ));
    }

    // -- Synthetic conditioning regression (Issue #54.3) ---------------------

    /// Builds a window from truth poses with optional noise, outlier
    /// invalidation, and contaminated expression initials.
    fn conditioning_window(
        noise_amplitude: f32,
        noise_seed: u64,
        invalidate_every: Option<usize>,
        expression_contamination: f32,
    ) -> (
        GnmModel,
        DenseCorrespondenceSet,
        Vec<GnmDenseObservation>,
        Vec<DenseProjection>,
    ) {
        let (model, mapping, mut observations, truths) = solve_fixture();
        if noise_amplitude > 0.0 || invalidate_every.is_some() {
            observations = truths
                .iter()
                .enumerate()
                .map(|(index, truth)| {
                    crate::synthesize_observation_from_projection(
                        &model,
                        &model.neutral_identity(),
                        &model.neutral_expression(),
                        &GnmJointState::neutral(model.joint_count()),
                        &mapping,
                        truth,
                        SynthesisOptions {
                            source_seq: 10 + index as u64,
                            captured_at_micros: 1_000 + index as u64 * 10,
                            noise_amplitude,
                            noise_seed: noise_seed + index as u64,
                        },
                        crate::DenseCoveragePolicy::new(1, 0.5).unwrap(),
                        |point_index, _| {
                            invalidate_every.is_some_and(|every| point_index % every == 0)
                        },
                    )
                    .unwrap()
                })
                .collect();
        }
        let _ = &mapping;
        let _ = expression_contamination;
        (model, mapping, observations, truths)
    }

    fn run_conditioning_solve(
        model: &GnmModel,
        mapping: &DenseCorrespondenceSet,
        observations: &[GnmDenseObservation],
        truths: &[DenseProjection],
        contamination: f32,
        config: SharedIdentitySolveConfig,
    ) -> SharedIdentitySolveOutcome {
        let samples = observations
            .iter()
            .enumerate()
            .map(|(index, observation)| {
                let mut values = model.neutral_expression().values().to_vec();
                values[0] = contamination;
                let expression =
                    GnmExpressionState::new(values, model.expression_dimension()).unwrap();
                NeutralSampleSolveInput::new(
                    observation,
                    SampleNuisance::new(truths[index], expression, model.expression_dimension())
                        .unwrap(),
                )
            })
            .collect();
        let input =
            SharedIdentitySolveInput::new(model, mapping, model.neutral_identity(), samples)
                .unwrap();
        solve_shared_identity(&input, config, 1.0e-4).unwrap()
    }

    #[test]
    fn frontal_only_window_is_not_a_well_conditioned_success() {
        // Near-duplicate/frontal-only candidates must fail the diversity gate;
        // the window must not be publishable as a conditioned success.
        let frontal = [
            candidate(1, 1_000, 0.0, 0.0),
            candidate(2, 1_010, 0.001, 0.001),
            candidate(3, 1_020, -0.001, 0.002),
        ];
        let selection = select_neutral_calibration_candidates(&frontal, selection_config());
        assert_eq!(
            selection.diagnostics.readiness,
            NeutralCalibrationReadiness::InsufficientPoseDiversity
        );
    }

    #[test]
    fn noisy_window_with_outliers_stays_bounded_and_deterministic() {
        let config = SharedIdentitySolveConfig::new(
            8,
            20,
            2.0,
            1.0e-6,
            1.0e-9,
            crate::DenseReprojectionConfig::default(),
        )
        .unwrap();
        let (model, mapping, observations, truths) = conditioning_window(0.01, 7, None, 0.0);
        let first = run_conditioning_solve(&model, &mapping, &observations, &truths, 0.0, config);
        let second = run_conditioning_solve(&model, &mapping, &observations, &truths, 0.0, config);
        assert_eq!(first, second, "same seed must be deterministic");
        // Identity coefficient error stays bounded (zero basis keeps identity).
        let identity_error = first
            .identity
            .values()
            .iter()
            .zip(model.neutral_identity().values())
            .map(|(solved, truth)| (*solved - *truth).abs())
            .fold(0.0f32, f32::max);
        assert!(identity_error < 1.0e-3, "identity error {identity_error}");
        for pair in first.objective_history.windows(2) {
            assert!(pair[1] <= pair[0] + 1.0e-12);
        }
    }

    #[test]
    fn large_residuals_are_downweighted_by_huber_robust_weight() {
        let (model, mapping, observations, truths) = conditioning_window(0.0, 11, None, 0.0);
        // Score one observation under a deliberately wrong principal point so
        // every residual norm exceeds the default robust delta.
        let wrong = DenseProjection::new(
            truths[0].yaw_pitch_roll(),
            truths[0].translation(),
            truths[0].focal(),
            [
                truths[0].principal_point()[0] + 0.3,
                truths[0].principal_point()[1],
            ],
        )
        .unwrap();
        let report = evaluate_dense_reprojection(
            &model,
            &model.neutral_identity(),
            &model.neutral_expression(),
            &GnmJointState::neutral(model.joint_count()),
            &mapping,
            &observations[0],
            &wrong,
            crate::DenseReprojectionConfig::default(),
        )
        .unwrap_or_else(|error| panic!("outlier evaluation must succeed: {error}"));
        assert!(
            report
                .residuals()
                .iter()
                .any(|point| point.huber_weight < 1.0),
            "large residuals must be downweighted"
        );
        // Huber weighting keeps the robust RMS at or below the raw RMS.
        let raw_rms = {
            let squares = report
                .residuals()
                .iter()
                .map(|point| {
                    let dx = point.residual_xy[0] as f64;
                    let dy = point.residual_xy[1] as f64;
                    (dx * dx + dy * dy) as f32
                })
                .sum::<f32>();
            (squares / report.residuals().len() as f32).sqrt()
        };
        assert!(report.weighted_rms() <= raw_rms + 1.0e-6);
    }

    #[test]
    fn expression_contamination_is_pulled_back_toward_neutral() {
        let config = SharedIdentitySolveConfig::new(
            8,
            25,
            1.0,
            1.0e-8,
            1.0e-9,
            crate::DenseReprojectionConfig::default(),
        )
        .unwrap();
        let (model, mapping, observations, truths) = conditioning_window(0.0, 3, None, 0.4);
        let outcome = run_conditioning_solve(&model, &mapping, &observations, &truths, 0.4, config);
        for nuisance in &outcome.nuisances {
            let initial_norm_squared = 0.4f64 * 0.4;
            let final_norm_squared: f64 = nuisance
                .expression()
                .values()
                .iter()
                .map(|value| (*value as f64) * (*value as f64))
                .sum();
            assert!(
                final_norm_squared <= initial_norm_squared,
                "contaminated expression must not grow"
            );
        }
    }
}
