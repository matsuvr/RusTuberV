//! Engine-neutral dense 2D reprojection objective and rigid-pose evidence
//! helpers (Issue #53).
//!
//! This module owns the primary data-term contract for dense observations: a
//! documented head-space-to-normalized-image projection, a weighted robust
//! residual evaluation, and a deterministic Levenberg-Marquardt rigid recovery
//! used for conditioning studies. It deliberately contains no camera driver,
//! renderer, or solver library dependency.
//!
//! # Conventions
//!
//! - Camera space is reached from GNM head space by
//!   `p_cam = Rz(roll) · Rx(pitch) · Ry(yaw) · p_head + translation`
//!   (right-handed; yaw about +Y first, then pitch about +X, then roll about
//!   +Z).
//! - Canonical normalized image space is `x` right, `y` down, both in `[0, 1]`:
//!   `u = cx + focal · X/Z`, `v = cy - focal · Y/Z`.
//! - The principal point is held fixed during rigid recovery; only root
//!   translation, rotation, and focal length are estimated.

use crate::single_frame_temporal::{
    CandidateTemporalScratch, SingleFrameTemporalPenalty, candidate_state_view,
};
use crate::{
    AnatomicalSide, DenseCorrespondenceSet, DenseCoveragePolicy, FaceRegion,
    GNM_HEAD_V3_IRIS_EXPRESSION_INDEX, GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM, GnmDenseError,
    GnmDenseObservation, GnmExpressionState, GnmIdentityState, GnmJointState, GnmModel,
    GnmModelError, GnmNonTongueExpression, GnmSparseVertices, MEDIAPIPE_FACE_LANDMARK_COUNT,
    MediaPipeGnmDenseCorrespondence, SparsePreparedVertices, SparseSkinningDerivatives,
    TemporalRegularizationError,
};

/// Typed failure from reprojection evaluation or rigid recovery.
#[derive(Debug)]
pub enum GnmReprojectionError {
    /// Projection parameters are not physically usable.
    InvalidProjection(&'static str),
    /// Evaluator or solver configuration is not physically usable.
    InvalidConfig(&'static str),
    /// An evaluation or recovery found no usable residual.
    InsufficientObservation,
    /// Underlying validated GNM model failure.
    Model(GnmModelError),
    /// Underlying dense correspondence or observation failure.
    Dense(GnmDenseError),
    /// A linearized Jacobian entry became non-finite.
    NonFiniteLinearization {
        /// Name of the offending parameter block.
        block: &'static str,
    },
    /// Temporal-energy evaluation failed, including a required history reset
    /// when the source gap crossed the configured bound.
    Temporal(TemporalRegularizationError),
}

impl std::fmt::Display for GnmReprojectionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidProjection(reason) => {
                write!(formatter, "invalid dense projection: {reason}")
            }
            Self::InvalidConfig(reason) => {
                write!(formatter, "invalid reprojection configuration: {reason}")
            }
            Self::InsufficientObservation => {
                write!(formatter, "no usable dense reprojection residual")
            }
            Self::Model(error) => write!(formatter, "{error}"),
            Self::Dense(error) => write!(formatter, "{error}"),
            Self::NonFiniteLinearization { block } => write!(
                formatter,
                "linearization of the `{block}` block produced a non-finite entry"
            ),
            Self::Temporal(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for GnmReprojectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Model(error) => Some(error),
            Self::Dense(error) => Some(error),
            _ => None,
        }
    }
}

impl From<GnmModelError> for GnmReprojectionError {
    fn from(error: GnmModelError) -> Self {
        Self::Model(error)
    }
}

impl From<GnmDenseError> for GnmReprojectionError {
    fn from(error: GnmDenseError) -> Self {
        Self::Dense(error)
    }
}

impl From<TemporalRegularizationError> for GnmReprojectionError {
    fn from(error: TemporalRegularizationError) -> Self {
        Self::Temporal(error)
    }
}

/// Perspective projection from GNM head space into canonical normalized image
/// space. See the module documentation for the exact conventions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseProjection {
    yaw_pitch_roll: [f32; 3],
    translation: [f32; 3],
    focal: f32,
    principal_point: [f32; 2],
}

impl DenseProjection {
    /// Creates a validated projection.
    pub fn new(
        yaw_pitch_roll: [f32; 3],
        translation: [f32; 3],
        focal: f32,
        principal_point: [f32; 2],
    ) -> Result<Self, GnmReprojectionError> {
        if yaw_pitch_roll
            .iter()
            .chain(translation.iter())
            .chain(principal_point.iter())
            .any(|value| !value.is_finite())
        {
            return Err(GnmReprojectionError::InvalidProjection(
                "pose and principal point must be finite",
            ));
        }
        if !focal.is_finite() || focal <= 0.0 {
            return Err(GnmReprojectionError::InvalidProjection(
                "focal length must be finite and positive",
            ));
        }
        Ok(Self {
            yaw_pitch_roll,
            translation,
            focal,
            principal_point,
        })
    }

    /// Returns the yaw/pitch/roll Euler angles in radians.
    pub fn yaw_pitch_roll(&self) -> [f32; 3] {
        self.yaw_pitch_roll
    }

    /// Returns the head-space to camera-space translation.
    pub fn translation(&self) -> [f32; 3] {
        self.translation
    }

    /// Returns the focal length in normalized-image units.
    pub fn focal(&self) -> f32 {
        self.focal
    }

    /// Returns the principal point in normalized image coordinates.
    pub fn principal_point(&self) -> [f32; 2] {
        self.principal_point
    }

    /// Projects one head-space point; `None` when the point is at or behind
    /// the principal plane or the result is non-finite.
    pub fn project(&self, point: [f32; 3]) -> Option<[f32; 2]> {
        self.project_f64(
            [
                self.yaw_pitch_roll[0] as f64,
                self.yaw_pitch_roll[1] as f64,
                self.yaw_pitch_roll[2] as f64,
            ],
            [
                self.translation[0] as f64,
                self.translation[1] as f64,
                self.translation[2] as f64,
            ],
            self.focal as f64,
            point,
        )
        .map(|projected| [projected[0] as f32, projected[1] as f32])
    }

    fn project_f64(
        &self,
        yaw_pitch_roll: [f64; 3],
        translation: [f64; 3],
        focal: f64,
        point: [f32; 3],
    ) -> Option<[f64; 2]> {
        let rotated = rotate_f64(
            yaw_pitch_roll,
            [point[0] as f64, point[1] as f64, point[2] as f64],
        );
        let z = rotated[2] + translation[2];
        if !z.is_finite() || z <= 1.0e-6 {
            return None;
        }
        let u = self.principal_point[0] as f64 + focal * (rotated[0] + translation[0]) / z;
        let v = self.principal_point[1] as f64 - focal * (rotated[1] + translation[1]) / z;
        if !(u.is_finite() && v.is_finite()) {
            return None;
        }
        Some([u, v])
    }
}

fn rotate_f64(yaw_pitch_roll: [f64; 3], point: [f64; 3]) -> [f64; 3] {
    let [yaw, pitch, roll] = yaw_pitch_roll;
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let (sr, cr) = roll.sin_cos();
    // Ry
    let x1 = cy * point[0] + sy * point[2];
    let y1 = point[1];
    let z1 = -sy * point[0] + cy * point[2];
    // Rx
    let x2 = x1;
    let y2 = cp * y1 - sp * z1;
    let z2 = sp * y1 + cp * z1;
    // Rz
    [cr * x2 - sr * y2, sr * x2 + cr * y2, z2]
}

/// Robustness and weighting configuration for residual evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseReprojectionConfig {
    /// Huber transition point in normalized image units. Residuals below it
    /// are quadratic; above it they are linearly downweighted.
    pub robust_delta: f32,
}

impl DenseReprojectionConfig {
    /// Creates a validated configuration; fails closed on non-finite or
    /// non-positive `robust_delta` instead of silently disabling robust
    /// weighting (or producing NaN weights).
    pub fn new(robust_delta: f32) -> Result<Self, GnmReprojectionError> {
        let config = Self { robust_delta };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), GnmReprojectionError> {
        if !self.robust_delta.is_finite() || self.robust_delta <= 0.0 {
            return Err(GnmReprojectionError::InvalidConfig(
                "robust_delta must be finite and positive",
            ));
        }
        Ok(())
    }
}

impl Default for DenseReprojectionConfig {
    fn default() -> Self {
        Self { robust_delta: 0.02 }
    }
}

/// One weighted dense reprojection residual.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseReprojectionResidual {
    /// Stable row index in the dense correspondence set.
    pub mapping_index: usize,
    /// Facial region of the point.
    pub region: FaceRegion,
    /// Subject-relative anatomical side.
    pub anatomical_side: AnatomicalSide,
    /// Observed canonical normalized image coordinate.
    pub observed_xy: [f32; 2],
    /// Projected canonical normalized image coordinate.
    pub projected_xy: [f32; 2],
    /// `observed - projected`.
    pub residual_xy: [f32; 2],
    /// Static weight from the observation.
    pub base_weight: f32,
    /// Huber factor in `(0, 1]` applied on top of the base weight.
    pub huber_weight: f32,
}

/// Weighted residual bundle for one observation frame under one projection.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseReprojectionReport {
    residuals: Vec<DenseReprojectionResidual>,
    excluded_points: usize,
    weighted_rms: f32,
    effective_weight_sum: f32,
}

/// Per-region fit quality recorded in teacher replay traces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GnmRegionFitRecord {
    /// Fixed facial region.
    pub region: FaceRegion,
    /// Number of observation points with a finite projected counterpart.
    pub valid_points: usize,
    /// Static mapping-weighted 2D RMS for the region.
    pub weighted_rms: f32,
}

/// Analytic projection Jacobian for the fixed 351-dimensional non-tongue state.
#[derive(Clone, Debug, PartialEq)]
pub struct NonTongueProjectionJacobian {
    /// Number of x/y rows retained from valid mapped observations.
    pub row_count: usize,
    /// Always [`GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM`].
    pub column_count: usize,
    /// Projection derivatives in row-major order.
    pub values_row_major: Vec<f64>,
    /// Mapping base weight for each x/y row.
    pub row_weights: Vec<f64>,
}

/// Builds the analytic 2D projection Jacobian for non-tongue Head-v3 expression.
///
/// # Errors
///
/// Returns a typed model/reprojection error for incompatible state, mapping,
/// or projection inputs, or when no observation row can be projected.
#[allow(clippy::too_many_arguments, clippy::indexing_slicing)]
pub fn non_tongue_projection_jacobian(
    model: &GnmModel,
    identity: &GnmIdentityState,
    expression: &GnmNonTongueExpression,
    joints: &GnmJointState,
    mapping: &DenseCorrespondenceSet,
    observation: &GnmDenseObservation,
    projection: &DenseProjection,
) -> Result<NonTongueProjectionJacobian, GnmReprojectionError> {
    let full_expression = expression.expand_with_zero_tongue()?;
    let prepared = model.prepare_sparse_vertices(
        identity,
        &full_expression,
        joints,
        mapping.surface_landmarks(),
    )?;
    let surface = prepared.skin(model, identity, joints, mapping.surface_landmarks())?;
    let report = evaluate_report_from_surface(
        observation,
        projection,
        DenseReprojectionConfig::default(),
        &surface,
    )?;
    let retained: Vec<usize> = report
        .residuals()
        .iter()
        .map(|residual| residual.mapping_index)
        .collect();
    let row_count = 2 * retained.len();
    let column_count = GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM;
    let mut values_row_major = vec![0.0; row_count * column_count];
    let row_weights = report
        .residuals()
        .iter()
        .flat_map(|residual| [f64::from(residual.base_weight); 2])
        .collect();
    let skinning =
        model.sparse_skinning_derivatives(identity, joints, mapping.surface_landmarks())?;
    let active_columns = skinning.active_expression_columns(model);
    let camera = CachedCameraRotation::new(projection);
    let mut offsets = vec![[0.0; 3]; skinning.len()];
    for compact_column in 0..column_count {
        let model_column = if compact_column + 1 == column_count {
            GNM_HEAD_V3_IRIS_EXPRESSION_INDEX
        } else {
            compact_column
        };
        if !active_columns[model_column] {
            continue;
        }
        skinning.expression_point_offsets(model, model_column, &mut offsets)?;
        for (point_row, &mapping_index) in retained.iter().enumerate() {
            let offset = offsets[mapping_index];
            if offset == [0.0; 3] {
                continue;
            }
            let Some(projected_derivative) = camera.projection_jacobian(surface[mapping_index])
            else {
                return Err(GnmReprojectionError::InsufficientObservation);
            };
            values_row_major[(2 * point_row) * column_count + compact_column] =
                projected_derivative[0][0] * f64::from(offset[0])
                    + projected_derivative[0][1] * f64::from(offset[1])
                    + projected_derivative[0][2] * f64::from(offset[2]);
            values_row_major[(2 * point_row + 1) * column_count + compact_column] =
                projected_derivative[1][0] * f64::from(offset[0])
                    + projected_derivative[1][1] * f64::from(offset[1])
                    + projected_derivative[1][2] * f64::from(offset[2]);
        }
    }
    Ok(NonTongueProjectionJacobian {
        row_count,
        column_count,
        values_row_major,
        row_weights,
    })
}

/// Adds `J^T W J` to a caller-owned packed lower-triangle buffer.
///
/// # Errors
///
/// Returns a typed configuration error for inconsistent dimensions or a
/// non-finite Jacobian, weight, or accumulated Gram entry.
#[allow(clippy::indexing_slicing)]
pub fn accumulate_observability_gram(
    jacobian: &NonTongueProjectionJacobian,
    gram_lower_triangle: &mut [f64],
) -> Result<(), GnmReprojectionError> {
    let expected_values = jacobian.row_count * jacobian.column_count;
    let expected_gram = jacobian.column_count * (jacobian.column_count + 1) / 2;
    if jacobian.values_row_major.len() != expected_values
        || jacobian.row_weights.len() != jacobian.row_count
        || gram_lower_triangle.len() != expected_gram
    {
        return Err(GnmReprojectionError::InvalidConfig(
            "observability Jacobian or Gram dimensions disagree",
        ));
    }
    for row in 0..jacobian.row_count {
        let weight = jacobian.row_weights[row];
        if !weight.is_finite() || weight <= 0.0 {
            return Err(GnmReprojectionError::InvalidConfig(
                "observability row weights must be finite and positive",
            ));
        }
        let row_start = row * jacobian.column_count;
        for lower_row in 0..jacobian.column_count {
            let left = jacobian.values_row_major[row_start + lower_row];
            if !left.is_finite() {
                return Err(GnmReprojectionError::NonFiniteLinearization {
                    block: "non_tongue_expression",
                });
            }
            let triangle_start = lower_row * (lower_row + 1) / 2;
            for column in 0..=lower_row {
                let right = jacobian.values_row_major[row_start + column];
                let slot = &mut gram_lower_triangle[triangle_start + column];
                *slot += weight * left * right;
                if !slot.is_finite() {
                    return Err(GnmReprojectionError::NonFiniteLinearization {
                        block: "observability_gram",
                    });
                }
            }
        }
    }
    Ok(())
}

/// Computes diagnostic region RMS values without changing fit acceptance.
///
/// # Errors
///
/// Returns a typed configuration error when projected points are not aligned
/// to the dense mapping rows.
#[allow(clippy::indexing_slicing)]
pub fn region_fit_records(
    mapping: &DenseCorrespondenceSet,
    observation: &GnmDenseObservation,
    projected_points: &[[f32; 2]],
) -> Result<Vec<GnmRegionFitRecord>, GnmReprojectionError> {
    if projected_points.len() != mapping.len() {
        return Err(GnmReprojectionError::InvalidConfig(
            "projected points must match dense mapping rows",
        ));
    }
    let regions = [
        FaceRegion::Contour,
        FaceRegion::Brow,
        FaceRegion::Eye,
        FaceRegion::Nose,
        FaceRegion::Mouth,
        FaceRegion::Iris,
        FaceRegion::Other,
    ];
    let mut records = Vec::with_capacity(regions.len());
    for region in regions {
        let mut valid_points = 0;
        let mut weighted_squared_sum = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        for point in observation
            .points()
            .iter()
            .filter(|point| point.region == region)
        {
            let projected = projected_points[point.mapping_index];
            if !projected.iter().all(|value| value.is_finite()) {
                continue;
            }
            let dx = point.normalized_xy[0] - projected[0];
            let dy = point.normalized_xy[1] - projected[1];
            weighted_squared_sum += f64::from(point.weight * (dx * dx + dy * dy));
            weight_sum += f64::from(point.weight);
            valid_points += 1;
        }
        records.push(GnmRegionFitRecord {
            region,
            valid_points,
            weighted_rms: if weight_sum > 0.0 {
                (weighted_squared_sum / weight_sum).sqrt() as f32
            } else {
                0.0
            },
        });
    }
    Ok(records)
}

impl DenseReprojectionReport {
    /// Returns valid residuals in observation order.
    pub fn residuals(&self) -> &[DenseReprojectionResidual] {
        &self.residuals
    }

    /// Returns how many observation points were excluded (invalid observed
    /// coordinate or projection failure).
    pub fn excluded_points(&self) -> usize {
        self.excluded_points
    }

    /// Returns `sqrt(Σ w·|r|² / Σ w)` with `w = base_weight · huber_weight`.
    pub fn weighted_rms(&self) -> f32 {
        self.weighted_rms
    }

    /// Returns the total effective weight of retained residuals.
    pub fn effective_weight_sum(&self) -> f32 {
        self.effective_weight_sum
    }
}

/// Evaluates the dense 2D reprojection objective for one observation.
///
/// The GNM surface is evaluated once through the selected-surface path; no
/// render mesh or material dependency is introduced.
#[allow(clippy::too_many_arguments)]
// explicit state/mapping/observation contract
pub fn evaluate_dense_reprojection(
    model: &GnmModel,
    identity: &GnmIdentityState,
    expression: &GnmExpressionState,
    joints: &GnmJointState,
    mapping: &DenseCorrespondenceSet,
    observation: &GnmDenseObservation,
    projection: &DenseProjection,
    config: DenseReprojectionConfig,
) -> Result<DenseReprojectionReport, GnmReprojectionError> {
    let mut surface_sink = Vec::new();
    evaluate_dense_reprojection_with_surface(
        model,
        identity,
        expression,
        joints,
        mapping,
        observation,
        projection,
        config,
        &mut surface_sink,
    )
}

/// Same contract as [`evaluate_dense_reprojection`], additionally returning
/// the evaluated baseline surface points (in mapping-row order) through
/// `surface_values` so callers can reuse them instead of re-evaluating.
#[allow(clippy::too_many_arguments)]
// explicit state/mapping/observation contract
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
pub fn evaluate_dense_reprojection_with_surface(
    model: &GnmModel,
    identity: &GnmIdentityState,
    expression: &GnmExpressionState,
    joints: &GnmJointState,
    mapping: &DenseCorrespondenceSet,
    observation: &GnmDenseObservation,
    projection: &DenseProjection,
    config: DenseReprojectionConfig,
    surface_values: &mut Vec<[f32; 3]>,
) -> Result<DenseReprojectionReport, GnmReprojectionError> {
    config.validate()?;
    let mut surface = GnmSparseVertices::with_len(mapping.len());
    mapping.evaluate_surface(model, identity, expression, joints, &mut surface)?;
    *surface_values = surface.values().to_vec();
    evaluate_report_from_surface(observation, projection, config, surface.values())
}

/// Weights and robust-clips residuals for an already-evaluated surface.
///
/// Shared tail of [`evaluate_dense_reprojection`] so callers holding a
/// pre-computed surface (for example the linearizer's staged evaluation) avoid
/// any duplicate geometry work.
///
/// # Errors
///
/// Fails when every point was excluded or carries no effective weight.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn evaluate_report_from_surface(
    observation: &GnmDenseObservation,
    projection: &DenseProjection,
    config: DenseReprojectionConfig,
    surface_values: &[[f32; 3]],
) -> Result<DenseReprojectionReport, GnmReprojectionError> {
    let mut residuals = Vec::with_capacity(observation.points().len());
    let mut excluded_points = 0usize;
    let mut weighted_squared_sum = 0.0f64;
    let mut effective_weight_sum = 0.0f64;
    for point in observation.points() {
        let Some(projected) = projection.project(surface_values[point.mapping_index]) else {
            excluded_points += 1;
            continue;
        };
        let residual = [
            point.normalized_xy[0] - projected[0],
            point.normalized_xy[1] - projected[1],
        ];
        let norm = (residual[0] * residual[0] + residual[1] * residual[1]).sqrt();
        if !norm.is_finite() {
            excluded_points += 1;
            continue;
        }
        let huber_weight = if norm <= config.robust_delta {
            1.0
        } else {
            config.robust_delta / norm
        };
        let weight = point.weight * huber_weight;
        weighted_squared_sum += (weight * norm * norm) as f64;
        effective_weight_sum += weight as f64;
        residuals.push(DenseReprojectionResidual {
            mapping_index: point.mapping_index,
            region: point.region,
            anatomical_side: point.anatomical_side,
            observed_xy: point.normalized_xy,
            projected_xy: projected,
            residual_xy: residual,
            base_weight: point.weight,
            huber_weight,
        });
    }

    if residuals.is_empty() || effective_weight_sum <= 0.0 {
        return Err(GnmReprojectionError::InsufficientObservation);
    }
    let weighted_rms = (weighted_squared_sum / effective_weight_sum).sqrt() as f32;
    Ok(DenseReprojectionReport {
        residuals,
        excluded_points,
        weighted_rms,
        effective_weight_sum: effective_weight_sum as f32,
    })
}

/// Deterministic solver settings for rigid pose/camera recovery.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidRecoveryConfig {
    /// Maximum accepted Levenberg-Marquardt updates.
    pub max_iterations: usize,
    /// Huber transition point used for iteratively reweighted least squares.
    pub robust_delta: f32,
    /// Relative cost improvement below which the solver declares convergence.
    pub convergence_tolerance: f64,
    /// Initial damping factor for the normal equations.
    pub initial_damping: f64,
}

impl RigidRecoveryConfig {
    /// Creates a validated solver configuration; fails closed on values that
    /// would make the deterministic loop ill-defined (non-finite scales, a
    /// zero iteration budget, or non-positive damping).
    pub fn new(
        max_iterations: usize,
        robust_delta: f32,
        convergence_tolerance: f64,
        initial_damping: f64,
    ) -> Result<Self, GnmReprojectionError> {
        let config = Self {
            max_iterations,
            robust_delta,
            convergence_tolerance,
            initial_damping,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), GnmReprojectionError> {
        if self.max_iterations == 0 {
            return Err(GnmReprojectionError::InvalidConfig(
                "max_iterations must be at least one",
            ));
        }
        if !self.robust_delta.is_finite() || self.robust_delta <= 0.0 {
            return Err(GnmReprojectionError::InvalidConfig(
                "robust_delta must be finite and positive",
            ));
        }
        if !self.convergence_tolerance.is_finite() || self.convergence_tolerance < 0.0 {
            return Err(GnmReprojectionError::InvalidConfig(
                "convergence_tolerance must be finite and non-negative",
            ));
        }
        if !self.initial_damping.is_finite() || self.initial_damping <= 0.0 {
            return Err(GnmReprojectionError::InvalidConfig(
                "initial_damping must be finite and positive",
            ));
        }
        Ok(())
    }
}

impl Default for RigidRecoveryConfig {
    fn default() -> Self {
        Self {
            max_iterations: 40,
            robust_delta: 0.02,
            convergence_tolerance: 1.0e-10,
            initial_damping: 1.0e-4,
        }
    }
}

/// Deterministic rigid recovery outcome with conditioning evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct RigidRecoveryOutcome {
    /// Recovered projection.
    pub projection: DenseProjection,
    /// Accepted solver updates.
    pub iterations: usize,
    /// False only when the iteration budget was exhausted while the objective
    /// was still improving; reaching a numerical minimum counts as converged.
    pub converged: bool,
    /// Residual report at the recovered projection.
    pub final_report: DenseReprojectionReport,
    /// `sqrt(λmax/λmin)` of the final weighted normal matrix; `INFINITY` when
    /// the smallest eigenvalue is numerically degenerate.
    pub condition_proxy: f32,
}

/// Recovers root translation, yaw/pitch/roll, and focal length from a dense
/// observation against a fixed GNM state.
///
/// The surface is evaluated once for the supplied state (identity/expression
/// and joint pose stay fixed), then a deterministic Levenberg-Marquardt solver
/// with Huber iteratively reweighted least squares minimizes the weighted dense
/// reprojection cost. The principal point is never estimated.
#[allow(clippy::too_many_arguments)]
// explicit state/mapping/observation contract
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
pub fn recover_rigid_projection(
    model: &GnmModel,
    identity: &GnmIdentityState,
    expression: &GnmExpressionState,
    joints: &GnmJointState,
    mapping: &DenseCorrespondenceSet,
    observation: &GnmDenseObservation,
    initial: DenseProjection,
    config: RigidRecoveryConfig,
) -> Result<RigidRecoveryOutcome, GnmReprojectionError> {
    config.validate()?;
    let mut surface = GnmSparseVertices::with_len(mapping.len());
    mapping.evaluate_surface(model, identity, expression, joints, &mut surface)?;

    // Align observation points with evaluated surface positions.
    struct ProblemPoint {
        surface: [f32; 3],
        observed: [f32; 2],
        base_weight: f32,
    }
    let points: Vec<ProblemPoint> = observation
        .points()
        .iter()
        .map(|point| ProblemPoint {
            surface: surface.values()[point.mapping_index],
            observed: point.normalized_xy,
            base_weight: point.weight,
        })
        .collect();
    if points.len() < 3 {
        return Err(GnmReprojectionError::InsufficientObservation);
    }
    let principal = initial.principal_point();

    let residuals_at = |theta: &[f64; 7],
                        weights: &[f64],
                        flat: &mut Vec<f64>,
                        weighted_cost: &mut f64|
     -> Result<(), GnmReprojectionError> {
        flat.clear();
        *weighted_cost = 0.0;
        let focal = theta[6].exp();
        let projection = DenseProjection::new(
            [theta[3] as f32, theta[4] as f32, theta[5] as f32],
            [theta[0] as f32, theta[1] as f32, theta[2] as f32],
            focal as f32,
            principal,
        )?;
        for (point, weight) in points.iter().zip(weights.iter()) {
            let Some(projected) = projection.project_f64(
                [theta[3], theta[4], theta[5]],
                [theta[0], theta[1], theta[2]],
                focal,
                point.surface,
            ) else {
                flat.push(f64::NAN);
                flat.push(f64::NAN);
                continue;
            };
            let rx = point.observed[0] as f64 - projected[0];
            let ry = point.observed[1] as f64 - projected[1];
            flat.push(rx);
            flat.push(ry);
            *weighted_cost += weight * (rx * rx + ry * ry);
        }
        Ok(())
    };

    let huber_weights = |theta: &[f64; 7]| -> Vec<f64> {
        let focal = theta[6].exp();
        let projection = DenseProjection::new(
            [theta[3] as f32, theta[4] as f32, theta[5] as f32],
            [theta[0] as f32, theta[1] as f32, theta[2] as f32],
            focal as f32,
            principal,
        );
        let Ok(projection) = projection else {
            return vec![0.0; points.len()];
        };
        let delta = config.robust_delta as f64;
        points
            .iter()
            .map(|point| {
                let Some(projected) = projection.project_f64(
                    [theta[3], theta[4], theta[5]],
                    [theta[0], theta[1], theta[2]],
                    focal,
                    point.surface,
                ) else {
                    return 0.0;
                };
                let rx = point.observed[0] as f64 - projected[0];
                let ry = point.observed[1] as f64 - projected[1];
                let norm = (rx * rx + ry * ry).sqrt();
                if norm <= delta {
                    point.base_weight as f64
                } else {
                    point.base_weight as f64 * delta / norm
                }
            })
            .collect()
    };

    // Parameter vector: [tx, ty, tz, yaw, pitch, roll, ln(focal)].
    let mut theta: [f64; 7] = [
        initial.translation()[0] as f64,
        initial.translation()[1] as f64,
        initial.translation()[2] as f64,
        initial.yaw_pitch_roll()[0] as f64,
        initial.yaw_pitch_roll()[1] as f64,
        initial.yaw_pitch_roll()[2] as f64,
        (initial.focal() as f64).ln(),
    ];
    const STEPS: [f64; 7] = [1.0e-4, 1.0e-4, 1.0e-4, 1.0e-5, 1.0e-5, 1.0e-5, 1.0e-5];

    let mut weights = huber_weights(&theta);
    let mut flat = Vec::with_capacity(points.len() * 2);
    let mut cost = 0.0f64;
    residuals_at(&theta, &weights, &mut flat, &mut cost)?;
    if !flat.iter().all(|value| value.is_finite()) {
        return Err(GnmReprojectionError::InsufficientObservation);
    }

    let mut lambda = config.initial_damping;
    let mut iterations = 0usize;
    let mut converged = false;
    let mut residual_plus = Vec::with_capacity(flat.len());
    let mut residual_minus = Vec::with_capacity(flat.len());
    let mut cost_probe = 0.0f64;

    for _ in 0..config.max_iterations {
        // Numeric Jacobian (central differences) at the current theta.
        let mut jacobian = vec![0.0f64; flat.len() * 7];
        for (parameter, step) in STEPS.iter().enumerate() {
            let mut probe = theta;
            probe[parameter] += step;
            residuals_at(&probe, &weights, &mut residual_plus, &mut cost_probe)?;
            probe[parameter] -= 2.0 * step;
            residuals_at(&probe, &weights, &mut residual_minus, &mut cost_probe)?;
            for row in 0..flat.len() {
                jacobian[row * 7 + parameter] =
                    (residual_plus[row] - residual_minus[row]) / (2.0 * step);
            }
        }

        // Weighted normal equations A·δ = -g with A = JᵀWJ.
        let mut normal = [[0.0f64; 7]; 7];
        let mut gradient = [0.0f64; 7];
        for row in 0..points.len() {
            let weight = weights[row];
            for k in 0..7 {
                let jx = jacobian[(row * 2) * 7 + k];
                let jy = jacobian[(row * 2 + 1) * 7 + k];
                gradient[k] -= weight * (jx * flat[row * 2] + jy * flat[row * 2 + 1]);
                for l in 0..7 {
                    let jxl = jacobian[(row * 2) * 7 + l];
                    let jyl = jacobian[(row * 2 + 1) * 7 + l];
                    normal[k][l] += weight * (jx * jxl + jy * jyl);
                }
            }
        }

        let mut accepted = false;
        for _ in 0..8 {
            let mut damped = normal;
            for (diagonal, damped_row) in damped.iter_mut().enumerate() {
                damped_row[diagonal] *= 1.0 + lambda;
                if damped_row[diagonal] <= 0.0 {
                    damped_row[diagonal] = lambda;
                }
            }
            let Some(delta) = solve_linear(&mut damped, gradient) else {
                lambda *= 10.0;
                continue;
            };
            let mut trial = theta;
            for (value, step) in trial.iter_mut().zip(delta.iter()) {
                *value += step;
            }
            let mut trial_cost = 0.0f64;
            residuals_at(&trial, &weights, &mut residual_plus, &mut trial_cost)?;
            if trial_cost.is_finite() && trial_cost < cost {
                let improvement = cost - trial_cost;
                theta = trial;
                cost = trial_cost;
                iterations += 1;
                weights = huber_weights(&theta);
                residuals_at(&theta, &weights, &mut flat, &mut cost)?;
                lambda = (lambda / 3.0).max(1.0e-12);
                accepted = true;
                if improvement <= config.convergence_tolerance * cost.max(1.0e-30) {
                    converged = true;
                }
                break;
            }
            lambda *= 10.0;
        }
        if !accepted {
            // No damping factor produced an improving step: the solver sits
            // at a numerical minimum of the objective.
            converged = true;
        }
        if converged {
            break;
        }
    }

    // Final conditioning evidence from the weighted normal matrix.
    let mut jacobian = vec![0.0f64; flat.len() * 7];
    for (parameter, step) in STEPS.iter().enumerate() {
        let mut probe = theta;
        probe[parameter] += step;
        residuals_at(&probe, &weights, &mut residual_plus, &mut cost_probe)?;
        probe[parameter] -= 2.0 * step;
        residuals_at(&probe, &weights, &mut residual_minus, &mut cost_probe)?;
        for row in 0..flat.len() {
            jacobian[row * 7 + parameter] =
                (residual_plus[row] - residual_minus[row]) / (2.0 * step);
        }
    }
    let mut normal = [[0.0f64; 7]; 7];
    for row in 0..points.len() {
        let weight = weights[row];
        for k in 0..7 {
            let jx = jacobian[(row * 2) * 7 + k];
            let jy = jacobian[(row * 2 + 1) * 7 + k];
            for l in 0..7 {
                let jxl = jacobian[(row * 2) * 7 + l];
                let jyl = jacobian[(row * 2 + 1) * 7 + l];
                normal[k][l] += weight * (jx * jxl + jy * jyl);
            }
        }
    }
    let (smallest, largest) = eigen_range(&normal);
    let condition_proxy = if smallest <= 1.0e-12 {
        f32::INFINITY
    } else {
        (largest / smallest).sqrt() as f32
    };

    let projection = DenseProjection::new(
        [theta[3] as f32, theta[4] as f32, theta[5] as f32],
        [theta[0] as f32, theta[1] as f32, theta[2] as f32],
        (theta[6].exp()) as f32,
        principal,
    )?;
    let final_report = evaluate_dense_reprojection(
        model,
        identity,
        expression,
        joints,
        mapping,
        observation,
        &projection,
        DenseReprojectionConfig {
            robust_delta: config.robust_delta,
        },
    )?;

    Ok(RigidRecoveryOutcome {
        projection,
        iterations,
        converged,
        final_report,
        condition_proxy,
    })
}

/// Deterministic fixture: projects mapped GNM surface points through a known
/// projection and writes them into the canonical MediaPipe slot array.
///
/// Slots not referenced by the mapping stay `NaN`, which proves the observation
/// path excludes unmapped and invalid points. `invalidate` receives each row
/// index and correspondence and marks additional slots invalid.
#[allow(clippy::too_many_arguments)]
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
pub fn synthesize_observation_from_projection(
    model: &GnmModel,
    identity: &GnmIdentityState,
    expression: &GnmExpressionState,
    joints: &GnmJointState,
    mapping: &DenseCorrespondenceSet,
    truth: &DenseProjection,
    options: SynthesisOptions,
    policy: DenseCoveragePolicy,
    invalidate: impl Fn(usize, &MediaPipeGnmDenseCorrespondence) -> bool,
) -> Result<GnmDenseObservation, GnmReprojectionError> {
    let mut surface = GnmSparseVertices::with_len(mapping.len());
    mapping.evaluate_surface(model, identity, expression, joints, &mut surface)?;

    let mut landmarks = vec![[f32::NAN; 2]; MEDIAPIPE_FACE_LANDMARK_COUNT];
    let mut noise = Lcg::new(options.noise_seed);
    for (row_index, row) in mapping.rows().iter().enumerate() {
        if invalidate(row_index, row) {
            continue;
        }
        let Some(projected) = truth.project(surface.values()[row_index]) else {
            continue;
        };
        let noisy = [
            projected[0] + noise.quasi_gaussian(options.noise_amplitude),
            projected[1] + noise.quasi_gaussian(options.noise_amplitude),
        ];
        landmarks[row.mediapipe_index] = noisy;
    }

    Ok(GnmDenseObservation::from_mediapipe_xy(
        options.source_seq,
        options.captured_at_micros,
        &landmarks,
        mapping,
        policy,
    )?)
}

/// Options for [`synthesize_observation_from_projection`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SynthesisOptions {
    /// Source frame sequence number recorded on the observation.
    pub source_seq: u64,
    /// Capture timestamp in microseconds recorded on the observation.
    pub captured_at_micros: u64,
    /// Amplitude of the additive quasi-Gaussian coordinate noise.
    pub noise_amplitude: f32,
    /// Deterministic seed for the noise stream.
    pub noise_seed: u64,
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            source_seq: 0,
            captured_at_micros: 0,
            noise_amplitude: 0.0,
            noise_seed: 0x5DEECE66D,
        }
    }
}

/// Deterministic xorshift64* pseudo-random stream (dependency-free).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E3779B97F4A7C15)
    }

    fn next_u64(&mut self) -> u64 {
        let mut state = self.0;
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        self.0 = state;
        state.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform value in `[-1, 1]`.
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)) * 2.0 - 1.0
    }

    /// Sum of three uniforms scaled by `amplitude`; a documented
    /// quasi-Gaussian approximation, not a true Gaussian sample.
    fn quasi_gaussian(&mut self, amplitude: f32) -> f32 {
        if amplitude <= 0.0 {
            return 0.0;
        }
        let sum = self.uniform() + self.uniform() + self.uniform();
        (sum / 3.0 * amplitude as f64) as f32
    }
}

/// One labeled recovery baseline for a conditioning comparison.
#[derive(Clone, Copy, Debug)]
pub struct ConditioningBaseline<'a> {
    /// Report label (for example `"sparse-37"` or `"dense-470"`).
    pub label: &'static str,
    /// Correspondence set for this baseline.
    pub mapping: &'a DenseCorrespondenceSet,
    /// Observation aligned with this baseline's MediaPipe indices.
    pub observation: &'a GnmDenseObservation,
    /// Deterministic starting guess for the recovery.
    pub initial_guess: DenseProjection,
}

/// Typed per-baseline conditioning evidence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConditioningStats {
    /// Baseline label.
    pub label: &'static str,
    /// Retained residual count.
    pub valid_points: usize,
    /// Excluded observation points.
    pub excluded_points: usize,
    /// Weighted RMS at the initial guess.
    pub initial_rms: f32,
    /// Weighted RMS at the recovered projection.
    pub final_rms: f32,
    /// Wrapped rotation error versus ground truth: `[yaw, pitch, roll]`.
    pub rotation_error: [f32; 3],
    /// Translation error versus ground truth per axis.
    pub translation_error: [f32; 3],
    /// Relative focal error `(f - f*) / f*`.
    pub relative_focal_error: f32,
    /// Solver conditioning proxy of the weighted normal matrix.
    pub condition_proxy: f32,
}

/// Runs rigid recovery for every baseline against the same ground truth and
/// returns typed conditioning statistics in baseline order.
pub fn compare_conditioning(
    model: &GnmModel,
    identity: &GnmIdentityState,
    expression: &GnmExpressionState,
    joints: &GnmJointState,
    baselines: &[ConditioningBaseline<'_>],
    ground_truth: &DenseProjection,
    config: RigidRecoveryConfig,
) -> Result<Vec<ConditioningStats>, GnmReprojectionError> {
    let mut stats = Vec::with_capacity(baselines.len());
    for baseline in baselines {
        let initial_report = evaluate_dense_reprojection(
            model,
            identity,
            expression,
            joints,
            baseline.mapping,
            baseline.observation,
            &baseline.initial_guess,
            DenseReprojectionConfig {
                robust_delta: config.robust_delta,
            },
        )?;
        let outcome = recover_rigid_projection(
            model,
            identity,
            expression,
            joints,
            baseline.mapping,
            baseline.observation,
            baseline.initial_guess,
            config,
        )?;
        let recovered = outcome.projection;
        let rotation_error = [
            wrapped_angle(recovered.yaw_pitch_roll()[0] - ground_truth.yaw_pitch_roll()[0]),
            wrapped_angle(recovered.yaw_pitch_roll()[1] - ground_truth.yaw_pitch_roll()[1]),
            wrapped_angle(recovered.yaw_pitch_roll()[2] - ground_truth.yaw_pitch_roll()[2]),
        ];
        let translation_error = [
            recovered.translation()[0] - ground_truth.translation()[0],
            recovered.translation()[1] - ground_truth.translation()[1],
            recovered.translation()[2] - ground_truth.translation()[2],
        ];
        let relative_focal_error =
            (recovered.focal() - ground_truth.focal()) / ground_truth.focal();
        stats.push(ConditioningStats {
            label: baseline.label,
            valid_points: outcome.final_report.residuals().len(),
            excluded_points: outcome.final_report.excluded_points(),
            initial_rms: initial_report.weighted_rms(),
            final_rms: outcome.final_report.weighted_rms(),
            rotation_error,
            translation_error,
            relative_focal_error,
            condition_proxy: outcome.condition_proxy,
        });
    }
    Ok(stats)
}

/// Wraps an angle difference into `[-π, π]`.
fn wrapped_angle(delta: f32) -> f32 {
    let tau = std::f32::consts::TAU;
    let wrapped = (delta + std::f32::consts::PI).rem_euclid(tau) - std::f32::consts::PI;
    if wrapped <= -std::f32::consts::PI {
        std::f32::consts::PI
    } else {
        wrapped
    }
}

/// Named synthetic observation cases required by the Issue #81 acceptance:
/// neutral, yaw/pitch, mouth, and eyelid, combinable with coordinate noise
/// and partial invalidation at synthesis time.
///
/// Expression semantics are not decoded until Issue #67, so the mouth and
/// eyelid cases select their expression probe *geometrically*: among the
/// strongest movers of the target region's mapped targets, the single
/// coefficient with the best region-vs-rest specificity (measured on the
/// pinned model through the public evaluator). This keeps the cases
/// region-targeted, deterministic, and honest about what they assert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyntheticCase {
    /// Neutral pose and neutral expression.
    Neutral,
    /// Pose-only yaw/pitch displacement; expression stays neutral.
    YawPitch,
    /// Near-neutral pose plus a mouth-region expression probe.
    Mouth,
    /// Near-neutral pose plus an eyelid-region expression probe.
    Eyelid,
}

impl SyntheticCase {
    /// Peak magnitude for expression-coefficient probes.
    pub const EXPRESSION_PROBE_AMPLITUDE: f32 = 0.30;
    /// Peak axis-angle magnitude (radians) for single-joint probes.
    pub const JOINT_PROBE_AMPLITUDE: f32 = 0.15;

    /// Stable label used in reports and documentation tables.
    pub fn label(self) -> &'static str {
        match self {
            Self::Neutral => "neutral",
            Self::YawPitch => "yaw-pitch",
            Self::Mouth => "mouth",
            Self::Eyelid => "eyelid",
        }
    }

    /// Ground-truth yaw/pitch/roll for this case.
    fn yaw_pitch_roll(self) -> [f32; 3] {
        match self {
            Self::Neutral => [0.05, 0.03, -0.02],
            Self::YawPitch => [0.24, -0.18, 0.04],
            Self::Mouth => [0.04, -0.03, 0.01],
            Self::Eyelid => [-0.03, 0.05, -0.01],
        }
    }

    /// Region whose displacement drives the mouth/eyelid probes.
    fn probe_region(self) -> Option<FaceRegion> {
        match self {
            Self::Mouth => Some(FaceRegion::Mouth),
            Self::Eyelid => Some(FaceRegion::Eye),
            _ => None,
        }
    }

    /// Deterministically plans this case against a fixed model state.
    ///
    /// Mouth/eyelid cases carry exactly one probe after scanning **both**
    /// deformation channels geometrically: every expression coefficient
    /// (±[`SyntheticCase::EXPRESSION_PROBE_AMPLITUDE`]) and every single-joint
    /// axis rotation (±[`SyntheticCase::JOINT_PROBE_AMPLITUDE`]). Whichever
    /// candidate gives the mapped target region the highest mean displacement
    /// relative to the rest of the face wins; on the pinned GNM Head v3 asset
    /// that is an expression coefficient for the mouth and an eyeball-joint
    /// rotation for the eyelids (the expression basis contains no strong
    /// isolated eyelid component). Ties resolve to the lowest index and then
    /// to the positive sign. All other state passes through unchanged.
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    // Invariant: probe vectors are unit probes within validated bounds and
    // the evaluator is deterministic over a validated mapping, so state
    // construction below cannot fail.
    #[allow(clippy::expect_used)]
    pub fn plan(
        self,
        model: &GnmModel,
        mapping: &DenseCorrespondenceSet,
        identity: &GnmIdentityState,
        joints: &GnmJointState,
    ) -> CasePlan {
        let mut coefficients = vec![0.0f32; model.expression_dimension()];
        let mut rotations = joints.rotations().to_vec();
        if let Some(region) = self.probe_region() {
            let (coefficient_index, sign, expression_ratio) = select_region_probe(
                model,
                mapping,
                identity,
                joints,
                region,
                Self::EXPRESSION_PROBE_AMPLITUDE,
            );
            let winning_probe = match select_joint_probe(
                model,
                mapping,
                identity,
                joints,
                region,
                Self::JOINT_PROBE_AMPLITUDE,
            ) {
                Some((joint_index, axis, joint_sign, joint_ratio))
                    if joint_ratio > expression_ratio =>
                {
                    rotations[joint_index][axis] += joint_sign * Self::JOINT_PROBE_AMPLITUDE;
                    None
                }
                _ => Some((coefficient_index, sign)),
            };
            if let Some((coefficient_index, sign)) = winning_probe {
                coefficients[coefficient_index] = sign * Self::EXPRESSION_PROBE_AMPLITUDE;
            }
        }
        let expression = GnmExpressionState::new(coefficients, model.expression_dimension())
            .expect("case expression must be valid");
        let joints = GnmJointState::new(rotations, joints.translation(), model.joint_count())
            .expect("case joint state must be valid");
        CasePlan {
            label: self.label(),
            yaw_pitch_roll: self.yaw_pitch_roll(),
            expression,
            joints,
        }
    }
}

/// Mean per-point displacement over the target region and over the rest of
/// the mapped surface.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn region_displacement_means(
    positions: &[[f32; 3]],
    baseline: &[[f32; 3]],
    region_flags: &[bool],
    region_count: usize,
) -> (f64, f64) {
    let mut region_sum = 0.0f64;
    let mut other_sum = 0.0f64;
    for (index, position) in positions.iter().enumerate() {
        let delta = [
            position[0] - baseline[index][0],
            position[1] - baseline[index][1],
            position[2] - baseline[index][2],
        ];
        let norm = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        if region_flags[index] {
            region_sum += norm as f64;
        } else {
            other_sum += norm as f64;
        }
    }
    let other_count = positions.len() - region_count;
    (
        region_sum / region_count as f64,
        other_sum / other_count.max(1) as f64,
    )
}

// Invariant: every `FaceRegion` occurs in the committed dense mapping tables,
// which repository tests verify; a violated invariant here is a build-time bug.
#[allow(clippy::panic)]
fn region_flags_for(mapping: &DenseCorrespondenceSet, region: FaceRegion) -> (Vec<bool>, usize) {
    let flags: Vec<bool> = mapping
        .rows()
        .iter()
        .map(|row| row.region == region)
        .collect();
    let count = flags.iter().filter(|flag| **flag).count();
    assert!(count > 0, "region must be present in the mapping");
    (flags, count)
}

/// Finds the most specific expression-coefficient probe for a region.
///
/// Scans every coefficient with both signs, evaluating the mapped surface
/// through the public evaluator. Selection is two-stage to stay numerically
/// meaningful: pure ratio scoring would let a coefficient with near-zero
/// motion everywhere win by moving the region an epsilon more than the rest,
/// so candidates first have to reach half of the strongest absolute regional
/// motion before specificity breaks the tie. Returns
/// `(coefficient index, sign, specificity ratio)`; ties resolve to the lowest
/// index and then to the positive sign.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
// Invariant: unit probes stay within validated bounds and the evaluator is
// deterministic; only reachable with a non-empty expression dimension.
#[allow(clippy::indexing_slicing)]
#[allow(clippy::expect_used)]
fn select_region_probe(
    model: &GnmModel,
    mapping: &DenseCorrespondenceSet,
    identity: &GnmIdentityState,
    joints: &GnmJointState,
    region: FaceRegion,
    amplitude: f32,
) -> (usize, f32, f32) {
    let mut scratch = GnmSparseVertices::with_len(mapping.len());
    let mut evaluate = move |expression: &GnmExpressionState| -> Vec<[f32; 3]> {
        mapping
            .evaluate_surface(model, identity, expression, joints, &mut scratch)
            .expect("case probe evaluation must succeed");
        scratch.values().to_vec()
    };

    let baseline = evaluate(&model.neutral_expression());
    let (region_flags, region_count) = region_flags_for(mapping, region);

    const MIN_ABSOLUTE_FRACTION: f64 = 0.5;
    let mut candidates: Vec<(usize, f32, f64, f64)> = Vec::new(); // (index, sign, region mean, other mean)
    for coefficient_index in 0..model.expression_dimension() {
        for sign in [1.0f32, -1.0] {
            let mut coefficients = vec![0.0f32; model.expression_dimension()];
            coefficients[coefficient_index] = sign * amplitude;
            let expression = GnmExpressionState::new(coefficients, model.expression_dimension())
                .expect("unit probe vector must be valid");
            let means = region_displacement_means(
                &evaluate(&expression),
                &baseline,
                &region_flags,
                region_count,
            );
            candidates.push((coefficient_index, sign, means.0, means.1));
        }
    }
    let strongest_region_motion = candidates
        .iter()
        .map(|&(_, _, region_mean, _)| region_mean)
        .fold(0.0f64, f64::max);
    let mut best: Option<(usize, f32, f32)> = None; // (index, sign, ratio)
    for &(index, sign, region_mean, other_mean) in &candidates {
        if region_mean < MIN_ABSOLUTE_FRACTION * strongest_region_motion {
            continue;
        }
        let ratio = (region_mean / (other_mean + 1.0e-9)) as f32;
        if best.is_none_or(|(_, _, best_ratio)| ratio > best_ratio) {
            best = Some((index, sign, ratio));
        }
    }
    best.expect("expression dimension must be non-empty")
}

/// Finds the most specific single-joint axis-rotation probe for a region.
///
/// Scans every joint and axis with both signs on top of the base state.
/// Unlike the expression scan this uses pure specificity scoring: skeleton
/// channels are few and semantically coarse, so there is no large pool of
/// degenerate epsilon-movers to filter, and an absolute-motion stage would
/// wrongly promote global pose joints that move the whole face. Returns
/// `None` when no candidate beats a unit specificity ratio.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
// Invariant: small axis-aligned perturbations of a validated joint state;
// the evaluator is deterministic over a validated mapping.
#[allow(clippy::indexing_slicing)]
#[allow(clippy::expect_used)]
fn select_joint_probe(
    model: &GnmModel,
    mapping: &DenseCorrespondenceSet,
    identity: &GnmIdentityState,
    joints: &GnmJointState,
    region: FaceRegion,
    amplitude: f32,
) -> Option<(usize, usize, f32, f32)> {
    let mut scratch = GnmSparseVertices::with_len(mapping.len());
    let neutral_expression = model.neutral_expression();
    let mut evaluate = move |joint_state: &GnmJointState| -> Vec<[f32; 3]> {
        mapping
            .evaluate_surface(
                model,
                identity,
                &neutral_expression,
                joint_state,
                &mut scratch,
            )
            .expect("case probe evaluation must succeed");
        scratch.values().to_vec()
    };

    let baseline = evaluate(joints);
    let (region_flags, region_count) = region_flags_for(mapping, region);

    let mut best: Option<(usize, usize, f32, f64)> = None; // (joint, axis, sign, ratio)
    for joint_index in 0..model.joint_count() {
        for axis in 0..3usize {
            for sign in [1.0f32, -1.0] {
                let mut rotations = joints.rotations().to_vec();
                rotations[joint_index][axis] += sign * amplitude;
                let joint_state =
                    GnmJointState::new(rotations, joints.translation(), model.joint_count())
                        .expect("probe joint state must be valid");
                let (region_mean, other_mean) = region_displacement_means(
                    &evaluate(&joint_state),
                    &baseline,
                    &region_flags,
                    region_count,
                );
                let ratio = region_mean / (other_mean + 1.0e-9);
                if best.is_none_or(|(_, _, _, best_ratio)| ratio > best_ratio) {
                    best = Some((joint_index, axis, sign, ratio));
                }
            }
        }
    }
    best.filter(|&(_, _, _, ratio)| ratio > 1.0)
        .map(|(joint, axis, sign, ratio)| (joint, axis, sign, ratio as f32))
}

/// Planned state and pose for one [`SyntheticCase`].
#[derive(Clone, Debug)]
pub struct CasePlan {
    /// Case label.
    pub label: &'static str,
    /// Ground-truth yaw/pitch/roll for this case.
    pub yaw_pitch_roll: [f32; 3],
    /// Deterministic expression state for this case (neutral unless the case
    /// carries a region probe).
    pub expression: GnmExpressionState,
    /// Deterministic joint state for this case (the planner input, plus one
    /// axis perturbation when a joint probe wins the region scan).
    pub joints: GnmJointState,
}

/// Builds a ground-truth projection that automatically fits a mapped surface
/// cloud: centered on its principal axis and pushed back far enough that
/// every projected point lands well inside `[0, 1]²`.
///
/// Depth is `8·radius`, so camera-space z stays within roughly `[7r, 9r]`;
/// with focal length 1.2 the image spread stays bounded around the principal
/// point regardless of which correspondence subset produced the cloud.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
pub fn fitting_projection(
    surface: &[[f32; 3]],
    yaw_pitch_roll: [f32; 3],
) -> Result<DenseProjection, GnmReprojectionError> {
    let count = surface.len();
    if count == 0 {
        return Err(GnmReprojectionError::InvalidProjection(
            "surface cloud must be non-empty",
        ));
    }
    let count = count as f32;
    let mut centroid = [0.0f32; 3];
    for point in surface {
        for axis in 0..3 {
            centroid[axis] += point[axis];
        }
    }
    for value in &mut centroid {
        *value /= count;
    }
    let mut radius = 0.0f32;
    for point in surface {
        for axis in 0..3 {
            radius = radius.max((point[axis] - centroid[axis]).abs());
        }
    }
    if !radius.is_finite() || radius <= 0.0 {
        return Err(GnmReprojectionError::InvalidProjection(
            "surface cloud is degenerate",
        ));
    }
    let translation = [-centroid[0], -centroid[1], 8.0 * radius - centroid[2]];
    DenseProjection::new(yaw_pitch_roll, translation, 1.2, [0.5, 0.5])
}

/// Gaussian elimination with partial pivoting for a 7×7 system.
/// Returns `None` when the (damped) matrix is numerically singular.
#[allow(clippy::needless_range_loop)] // fixed-size dense linear algebra reads best indexed
#[allow(clippy::unwrap_used)]
// see invariant at the `max_by` pivot search below
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn solve_linear(matrix: &mut [[f64; 7]; 7], rhs: [f64; 7]) -> Option<[f64; 7]> {
    let mut augmented = [[0.0f64; 8]; 7];
    for row in 0..7 {
        augmented[row][..7].copy_from_slice(&matrix[row]);
        augmented[row][7] = rhs[row];
    }
    for col in 0..7 {
        let pivot_row = (col..7)
            .max_by(|a, b| {
                augmented[*a][col]
                    .abs()
                    .total_cmp(&augmented[*b][col].abs())
            })
            .unwrap();
        if augmented[pivot_row][col].abs() < 1.0e-14 {
            return None;
        }
        augmented.swap(col, pivot_row);
        let pivot = augmented[col][col];
        for row in 0..7 {
            if row == col {
                continue;
            }
            let factor = augmented[row][col] / pivot;
            if factor == 0.0 {
                continue;
            }
            for k in col..8 {
                augmented[row][k] -= factor * augmented[col][k];
            }
        }
    }
    let mut solution = [0.0f64; 7];
    for row in 0..7 {
        solution[row] = augmented[row][7] / augmented[row][row];
    }
    Some(solution)
}

/// Smallest and largest eigenvalues of a symmetric 7×7 matrix via cyclic
/// Jacobi rotations (deterministic).
#[allow(clippy::needless_range_loop)]
// fixed-size dense linear algebra reads best indexed
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn eigen_range(matrix: &[[f64; 7]; 7]) -> (f64, f64) {
    let mut a = *matrix;
    for _ in 0..100 {
        let mut off = 0.0f64;
        for row in 0..7 {
            for col in row + 1..7 {
                off += a[row][col] * a[row][col];
            }
        }
        if off < 1.0e-22 {
            break;
        }
        for p in 0..7 {
            for q in p + 1..7 {
                if a[p][q].abs() < 1.0e-16 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Two-sided rotation: columns then rows keep the matrix
                // symmetric without accumulating eigenvectors.
                for k in 0..7 {
                    let akp = a[k][p];
                    let akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..7 {
                    let apk = a[p][k];
                    let aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
            }
        }
    }
    let mut smallest = f64::INFINITY;
    let mut largest = f64::NEG_INFINITY;
    for k in 0..7 {
        smallest = smallest.min(a[k][k]);
        largest = largest.max(a[k][k]);
    }
    (smallest, largest)
}

/// Bounded one-step configuration for the rigid pose + camera-translation
/// update (Issue #64.2b / #119).
///
/// Expression, joint, and identity parameters are intentionally absent: a
/// rigid step never touches them. Focal length and principal point are also
/// fixed; only the camera-space head translation moves with the pose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseRigidStepConfig {
    /// Maximum accepted pose step magnitude in radians per solve.
    pub max_pose_step: f32,
    /// Maximum accepted translation step magnitude in head-space units.
    pub max_translation_step: f32,
    /// Levenberg-style damping added to the diagonal of the normal equations.
    pub damping: f32,
}

impl DenseRigidStepConfig {
    /// Creates a validated configuration.
    ///
    /// # Errors
    ///
    /// Returns [`GnmReprojectionError::InvalidConfig`] for non-finite or
    /// non-positive bounds/damping.
    pub fn new(
        max_pose_step: f32,
        max_translation_step: f32,
        damping: f32,
    ) -> Result<Self, GnmReprojectionError> {
        for value in [max_pose_step, max_translation_step, damping] {
            if !value.is_finite() || value <= 0.0 {
                return Err(GnmReprojectionError::InvalidConfig(
                    "rigid step bounds and damping must be finite and positive",
                ));
            }
        }
        Ok(Self {
            max_pose_step,
            max_translation_step,
            damping,
        })
    }
}

impl Default for DenseRigidStepConfig {
    fn default() -> Self {
        Self {
            max_pose_step: 0.2,
            max_translation_step: 0.2,
            damping: 1.0e-3,
        }
    }
}

/// Result of one bounded rigid pose + camera-translation step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseRigidStepOutcome {
    /// Whether the step decreased the weighted residual and was accepted.
    pub accepted: bool,
    /// Pose to continue from: the updated pose when accepted, the input pose
    /// when rejected.
    pub yaw_pitch_roll: [f32; 3],
    /// Camera-space head translation to continue from (same acceptance rule).
    pub translation: [f32; 3],
    /// Weighted RMS residual at the input state.
    pub residual_before: f32,
    /// Weighted RMS residual at the candidate state when it was evaluable.
    pub residual_after: Option<f32>,
    /// Magnitude of the (possibly clamped) pose step.
    pub pose_step_norm: f32,
    /// Magnitude of the (possibly clamped) translation step.
    pub translation_step_norm: f32,
}

/// Solves the square system `a * x = b` with Gaussian elimination and
/// partial pivoting. Returns `None` for singular systems.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
#[allow(clippy::needless_range_loop)] // fixed-size dense linear algebra reads best indexed
fn solve_linear_6(a: &mut [[f64; 6]; 6], b: &mut [f64; 6]) -> Option<[f64; 6]> {
    for column in 0..6 {
        let mut pivot_row = column;
        for candidate in (column + 1)..6 {
            if a[candidate][column].abs() > a[pivot_row][column].abs() {
                pivot_row = candidate;
            }
        }
        if a[pivot_row][column].abs() < 1.0e-12 {
            return None;
        }
        a.swap(column, pivot_row);
        b.swap(column, pivot_row);
        let pivot = a[column][column];
        for row in (column + 1)..6 {
            let factor = a[row][column] / pivot;
            if factor == 0.0 {
                continue;
            }
            for k in column..6 {
                a[row][k] -= factor * a[column][k];
            }
            b[row] -= factor * b[column];
        }
    }
    let mut x = [0.0; 6];
    for row in (0..6).rev() {
        let mut sum = b[row];
        for k in (row + 1)..6 {
            sum -= a[row][k] * x[k];
        }
        x[row] = sum / a[row][row];
    }
    Some(x)
}

/// Takes exactly one bounded Gauss-Newton/Levenberg step over the rigid head
/// rotation and camera-space head translation, holding expression, joints,
/// identity, focal length, and principal point fixed.
///
/// The rotation parameterization and update order are fixed here and only
/// here: the update vector is ordered `[yaw, pitch, roll, tx, ty, tz]`, the
/// step is clamped per group to the configured bounds, and the candidate is
/// accepted only when the weighted residual decreases. Invalid candidates
/// (projection failure, non-evaluable residual, non-decreasing residual)
/// leave the input state unchanged with `accepted == false`.
///
/// # Errors
///
/// Propagates typed failures from linearization or from an observation with
/// no usable residual, and fails closed on invalid configuration.
#[allow(clippy::too_many_arguments)] // explicit state/mapping/observation contract
pub fn take_dense_rigid_step(
    model: &GnmModel,
    identity: &GnmIdentityState,
    expression: &GnmExpressionState,
    joints: &GnmJointState,
    mapping: &DenseCorrespondenceSet,
    observation: &GnmDenseObservation,
    projection: &DenseProjection,
    config: DenseRigidStepConfig,
) -> Result<DenseRigidStepOutcome, GnmReprojectionError> {
    take_dense_rigid_step_impl(
        model,
        identity,
        expression,
        joints,
        mapping,
        observation,
        projection,
        config,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
// explicit state/mapping/observation contract
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn take_dense_rigid_step_impl(
    model: &GnmModel,
    identity: &GnmIdentityState,
    expression: &GnmExpressionState,
    joints: &GnmJointState,
    mapping: &DenseCorrespondenceSet,
    observation: &GnmDenseObservation,
    projection: &DenseProjection,
    config: DenseRigidStepConfig,
    temporal: Option<&SingleFrameTemporalPenalty<'_>>,
) -> Result<DenseRigidStepOutcome, GnmReprojectionError> {
    let linearization = linearize_dense_reprojection(
        model,
        identity,
        expression,
        joints,
        mapping,
        observation,
        projection,
        DenseReprojectionConfig::default(),
        LinearizationStepSizes::default(),
    )?;
    let report = linearization.report();
    let residual_before = report.weighted_rms();

    let pose_jacobian = linearization.block(ReprojectionBlock::RigidPose).ok_or(
        GnmReprojectionError::InvalidConfig("missing rigid pose block"),
    )?;
    let translation_jacobian = linearization
        .block(ReprojectionBlock::CameraTranslation)
        .ok_or(GnmReprojectionError::InvalidConfig(
            "missing camera translation block",
        ))?;

    // Optional temporal term (Issue #64.4): analytic diagonal gradient and
    // curvature at the input state, plus its energy for the combined
    // acceptance objective. With no term this is an exact no-op.
    let mut temporal_scratch = CandidateTemporalScratch::default();
    let (temporal_linearization, temporal_energy_before) = match temporal {
        Some(penalty) => {
            let view = candidate_state_view(expression, joints, projection, &mut temporal_scratch);
            let linearization = penalty.linearize_at(view)?;
            let energy = penalty.energy_at(view)?.total_weighted_energy;
            (Some(linearization), energy)
        }
        None => (None, 0.0),
    };

    // Weighted normal equations over the 6 rigid parameters, ordered
    // [yaw, pitch, roll, tx, ty, tz]. Each retained point contributes its two
    // residual components with equal share of the point weight.
    let mut normal = [[0.0f64; 6]; 6];
    let mut rhs = [0.0f64; 6];
    for (point_row, residual) in report.residuals().iter().enumerate() {
        let weight = f64::from(residual.base_weight * residual.huber_weight);
        if weight <= 0.0 {
            continue;
        }
        let rx = f64::from(residual.residual_xy[0]);
        let ry = f64::from(residual.residual_xy[1]);
        let mut jacobian_rows = [[0.0f64; 6]; 2];
        for parameter in 0..3 {
            jacobian_rows[0][parameter] = f64::from(
                pose_jacobian
                    .get(2 * point_row, parameter)
                    .unwrap_or_default(),
            );
            jacobian_rows[0][3 + parameter] = f64::from(
                translation_jacobian
                    .get(2 * point_row, parameter)
                    .unwrap_or_default(),
            );
            jacobian_rows[1][parameter] = f64::from(
                pose_jacobian
                    .get(2 * point_row + 1, parameter)
                    .unwrap_or_default(),
            );
            jacobian_rows[1][3 + parameter] = f64::from(
                translation_jacobian
                    .get(2 * point_row + 1, parameter)
                    .unwrap_or_default(),
            );
        }
        let component_residuals = [rx, ry];
        for component in 0..2 {
            let jacobian_row = &jacobian_rows[component];
            for i in 0..6 {
                for k in 0..6 {
                    normal[i][k] += 0.5 * weight * jacobian_row[i] * jacobian_row[k];
                }
                rhs[i] -= 0.5 * weight * jacobian_row[i] * component_residuals[component];
            }
        }
    }
    for (index, diagonal) in normal.iter_mut().enumerate() {
        diagonal[index] += f64::from(config.damping);
    }
    if let Some(linearization) = &temporal_linearization {
        // Parameter order is [yaw, pitch, roll, tx, ty, tz]; head_pose owns
        // the first three coordinates and camera translation the last three.
        #[allow(clippy::indexing_slicing)] // fixed six-parameter update layout
        for index in 0..3 {
            normal[index][index] += 0.5 * linearization.head_pose.curvature[index];
            rhs[index] -= 0.5 * linearization.head_pose.gradient[index];
            normal[3 + index][3 + index] += 0.5 * linearization.translation.curvature[index];
            rhs[3 + index] -= 0.5 * linearization.translation.gradient[index];
        }
    }
    let Some(step) = solve_linear_6(&mut normal, &mut rhs) else {
        return Ok(DenseRigidStepOutcome {
            accepted: false,
            yaw_pitch_roll: projection.yaw_pitch_roll(),
            translation: projection.translation(),
            residual_before,
            residual_after: None,
            pose_step_norm: 0.0,
            translation_step_norm: 0.0,
        });
    };
    if step.iter().any(|value| !value.is_finite()) {
        return Ok(DenseRigidStepOutcome {
            accepted: false,
            yaw_pitch_roll: projection.yaw_pitch_roll(),
            translation: projection.translation(),
            residual_before,
            residual_after: None,
            pose_step_norm: 0.0,
            translation_step_norm: 0.0,
        });
    }

    // Clamp per group to the configured bounds.
    let pose_norm = (step[0] * step[0] + step[1] * step[1] + step[2] * step[2]).sqrt();
    let translation_norm = (step[3] * step[3] + step[4] * step[4] + step[5] * step[5]).sqrt();
    let mut scale = 1.0f64;
    if pose_norm > f64::from(config.max_pose_step) {
        scale = scale.min(f64::from(config.max_pose_step) / pose_norm);
    }
    if translation_norm > f64::from(config.max_translation_step) {
        scale = scale.min(f64::from(config.max_translation_step) / translation_norm);
    }

    #[allow(clippy::indexing_slicing)] // fixed six-parameter update layout
    let yaw_pitch_roll = [
        projection.yaw_pitch_roll()[0] + (step[0] * scale) as f32,
        projection.yaw_pitch_roll()[1] + (step[1] * scale) as f32,
        projection.yaw_pitch_roll()[2] + (step[2] * scale) as f32,
    ];
    let base_translation = projection.translation();
    #[allow(clippy::indexing_slicing)] // fixed six-parameter update layout
    let translation = [
        base_translation[0] + (step[3] * scale) as f32,
        base_translation[1] + (step[4] * scale) as f32,
        base_translation[2] + (step[5] * scale) as f32,
    ];

    let rejected = |residual_after: Option<f32>| DenseRigidStepOutcome {
        accepted: false,
        yaw_pitch_roll: projection.yaw_pitch_roll(),
        translation: base_translation,
        residual_before,
        residual_after,
        pose_step_norm: (pose_norm * scale) as f32,
        translation_step_norm: (translation_norm * scale) as f32,
    };

    let Ok(candidate_projection) = DenseProjection::new(
        yaw_pitch_roll,
        translation,
        projection.focal(),
        projection.principal_point(),
    ) else {
        return Ok(rejected(None));
    };
    let Ok(candidate_report) = evaluate_dense_reprojection(
        model,
        identity,
        expression,
        joints,
        mapping,
        observation,
        &candidate_projection,
        DenseReprojectionConfig::default(),
    ) else {
        return Ok(rejected(None));
    };
    let residual_after = candidate_report.weighted_rms();
    if !residual_after.is_finite() {
        return Ok(rejected(Some(residual_after)));
    }
    // Acceptance uses the combined dense + temporal objective so a candidate
    // that lowers reprojection only by injecting motion energy is rejected.
    let temporal_energy_after = match temporal {
        Some(penalty) => {
            let view = candidate_state_view(
                expression,
                joints,
                &candidate_projection,
                &mut temporal_scratch,
            );
            match penalty.energy_at(view) {
                Ok(metrics) => metrics.total_weighted_energy,
                Err(_) => return Ok(rejected(Some(residual_after))),
            }
        }
        None => 0.0,
    };
    if f64::from(residual_after) + temporal_energy_after
        >= f64::from(residual_before) + temporal_energy_before
    {
        return Ok(rejected(Some(residual_after)));
    }

    Ok(DenseRigidStepOutcome {
        accepted: true,
        yaw_pitch_roll,
        translation,
        residual_before,
        residual_after: Some(residual_after),
        pose_step_norm: (pose_norm * scale) as f32,
        translation_step_norm: (translation_norm * scale) as f32,
    })
}

/// Bounded one-step configuration for the expression + joint update
/// (Issue #64.2c / #120).
///
/// Rigid pose, camera translation, and identity are intentionally absent: an
/// expression/joint step never touches them. The GNM prior is applied
/// explicitly as a configurable-strength pull of every coefficient toward its
/// neutral (zero) value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseExpressionJointStepConfig {
    /// Maximum accepted expression update magnitude per solve.
    pub max_expression_step: f32,
    /// Maximum accepted joint rotation update magnitude in radians.
    pub max_joint_rotation_step: f32,
    /// Maximum accepted joint translation update magnitude.
    pub max_joint_translation_step: f32,
    /// Strength of the neutral-pull prior on expression and joint parameters.
    pub prior_weight: f32,
    /// Levenberg-style damping added to the diagonal of the normal equations.
    pub damping: f32,
}

impl DenseExpressionJointStepConfig {
    /// Creates a validated configuration.
    ///
    /// # Errors
    ///
    /// Returns [`GnmReprojectionError::InvalidConfig`] for non-finite or
    /// non-positive bounds/weights.
    pub fn new(
        max_expression_step: f32,
        max_joint_rotation_step: f32,
        max_joint_translation_step: f32,
        prior_weight: f32,
        damping: f32,
    ) -> Result<Self, GnmReprojectionError> {
        for value in [
            max_expression_step,
            max_joint_rotation_step,
            max_joint_translation_step,
            prior_weight,
            damping,
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(GnmReprojectionError::InvalidConfig(
                    "expression/joint step bounds, prior weight, and damping \
                     must be finite and positive",
                ));
            }
        }
        Ok(Self {
            max_expression_step,
            max_joint_rotation_step,
            max_joint_translation_step,
            prior_weight,
            damping,
        })
    }
}

impl Default for DenseExpressionJointStepConfig {
    fn default() -> Self {
        Self {
            max_expression_step: 0.5,
            max_joint_rotation_step: 0.1,
            max_joint_translation_step: 0.1,
            prior_weight: 1.0e-4,
            damping: 1.0e-3,
        }
    }
}

/// Result of one bounded expression + joint step.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseExpressionJointStepOutcome {
    /// Whether the step decreased the weighted residual and was accepted.
    pub accepted: bool,
    /// Expression state to continue from (input state when rejected).
    pub expression: GnmExpressionState,
    /// Joint state to continue from (input state when rejected).
    pub joints: GnmJointState,
    /// Weighted RMS residual at the input state.
    pub residual_before: f32,
    /// Weighted RMS residual at the candidate state when it was evaluable.
    pub residual_after: Option<f32>,
    /// Magnitude of the (possibly clamped) expression update.
    pub expression_step_norm: f32,
    /// Magnitude of the (possibly clamped) joint rotation update.
    pub joint_rotation_step_norm: f32,
    /// Magnitude of the (possibly clamped) joint translation update.
    pub joint_translation_step_norm: f32,
}

/// Solves the SPD normal equations given in packed lower-triangular form via
/// Cholesky–Banachiewicz (Issue #148).
///
/// The system is symmetric positive-definite by construction: weighted
/// normal equations plus a strictly positive diagonal prior/damping. `a`
/// holds only the lower triangle in row-major packed order — entry
/// `(row, column)` for `column <= row` lives at
/// `row * (row + 1) / 2 + column`, so `a.len()` must equal
/// `n * (n + 1) / 2` — halving both the memory footprint and the factorization
/// traffic versus a full symmetric matrix with the same O(n³) class.
/// The factorization overwrites `a` with `L`; `b` holds the right-hand side.
/// Returns `None` when the storage size is inconsistent or a pivot is not
/// numerically positive (singular or indefinite input), matching the old
/// solver's failure contract.
// Bounds are guaranteed by construction: `a` is the packed triangle of the
// `b.len() × b.len()` matrix, so `row_start + column` and the prefix slices
// stay within `a`; see AGENTS.md panic policy. The substitution loops stay
// range-based because the back substitution walks a strided packed column
// that no iterator zip can express.
#[allow(clippy::indexing_slicing, clippy::needless_range_loop)]
fn solve_spd_packed_lower(a: &mut [f64], b: &mut [f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let packed_len = n.checked_mul(n.checked_add(1)?)? / 2;
    if a.len() != packed_len {
        return None;
    }
    // In-place LLᵀ factorization over the packed lower triangle.
    for row in 0..n {
        let row_start = row * (row + 1) / 2;
        for column in 0..=row {
            let column_start = column * (column + 1) / 2;
            let mut sum = a[row_start + column];
            let row_prefix = &a[row_start..row_start + column];
            let column_prefix = &a[column_start..column_start + column];
            for (left, right) in row_prefix.iter().zip(column_prefix.iter()) {
                sum -= left * right;
            }
            if row == column {
                // Positive pivots certify positive-definiteness.
                if !(sum > 1.0e-12 && sum.is_finite()) {
                    return None;
                }
                a[row_start + column] = sum.sqrt();
            } else {
                let diagonal = a[column_start + column];
                if diagonal == 0.0 {
                    return None;
                }
                a[row_start + column] = sum / diagonal;
            }
        }
    }
    // Forward substitution (L y = b) then back substitution (Lᵀ x = y).
    let mut x = vec![0.0; n];
    for row in 0..n {
        let row_start = row * (row + 1) / 2;
        let mut sum = b[row];
        for k in 0..row {
            sum -= a[row_start + k] * x[k];
        }
        x[row] = sum / a[row_start + row];
    }
    for row in (0..n).rev() {
        let row_start = row * (row + 1) / 2;
        let mut sum = x[row];
        for k in (row + 1)..n {
            let k_start = k * (k + 1) / 2;
            sum -= a[k_start + row] * x[k];
        }
        x[row] = sum / a[row_start + row];
    }
    Some(x)
}

/// Weighted auxiliary objective evaluation supplied by the caller.
#[derive(Clone, Debug)]
pub struct AuxiliaryTermEvaluation {
    /// Weighted auxiliary loss at the evaluated state.
    pub loss: f32,
    /// Gradient of the loss with respect to the expression coefficients.
    pub expression_gradient: Vec<f32>,
    /// Gradient of the loss with respect to the joint parameters
    /// (rotations flattened `[yaw, pitch, roll]` per joint, then the single
    /// joint translation).
    pub joint_gradient: Vec<f32>,
}

/// Caller-supplied optional auxiliary objective for the expression/joint
/// step (Issue #64.2d / #121).
///
/// Implementations live outside `vtuber-gnm` because the semantic pairing
/// between geometry-derived predictions and MediaPipe observations is a
/// tracking-layer concern. The dense reprojection remains the primary
/// objective; this term only contributes a weighted gradient to the normal
/// equations and a weighted loss to the acceptance test. No ARKit52 output
/// API may be built on this trait.
pub trait AuxiliaryObjectiveTerm {
    /// Evaluates the weighted auxiliary loss and its gradient at the given
    /// dynamic state slices. Parameter order matches the step's fixed layout:
    /// expression coefficients first, then joint rotations, then the single
    /// joint translation.
    ///
    /// # Errors
    ///
    /// Implementations must fail closed on non-finite values or invalid state
    /// dimensions instead of returning partial evidence.
    fn evaluate(
        &self,
        expression_values: &[f32],
        joint_rotations: &[[f32; 3]],
        joint_translation: [f32; 3],
    ) -> Result<AuxiliaryTermEvaluation, GnmReprojectionError>;
}

/// Takes exactly one bounded Gauss-Newton/Levenberg step over the GNM
/// expression coefficients and the articulated joint state (rotations and
/// global joint translation), holding rigid pose, camera translation,
/// identity, focal length, and principal point fixed.
///
/// The rotation parameterization and update order are fixed here and only
/// here: all expression coefficients first, then joint rotations (`[yaw,
/// pitch, roll]` per joint), then the single joint translation. A neutral-pull
/// prior with the configured strength is applied explicitly to every
/// parameter, updates are clamped per group, and the candidate is accepted
/// only when the combined weighted objective (dense reprojection plus the
/// optional auxiliary term) decreases. With no auxiliary term or a zero
/// weight, the update is identical to the dense-only step. Non-finite or
/// singular updates leave the input state unchanged with `accepted == false`.
///
/// # Errors
///
/// Propagates typed failures from linearization or from an observation with
/// no usable residual, and fails closed on invalid configuration.
#[allow(clippy::too_many_arguments)]
// explicit state/mapping/observation contract
// Jacobian dimensions, the parameter layout, and the update-vector length are
// all derived from the same validated model dimensions, so every index here is
// in bounds by construction.
#[allow(clippy::indexing_slicing, clippy::needless_range_loop)]
pub fn take_dense_expression_joint_step(
    model: &GnmModel,
    identity: &GnmIdentityState,
    expression: &GnmExpressionState,
    joints: &GnmJointState,
    mapping: &DenseCorrespondenceSet,
    observation: &GnmDenseObservation,
    projection: &DenseProjection,
    config: DenseExpressionJointStepConfig,
    auxiliary: Option<(&dyn AuxiliaryObjectiveTerm, f32)>,
) -> Result<DenseExpressionJointStepOutcome, GnmReprojectionError> {
    take_dense_expression_joint_step_impl(
        model,
        identity,
        expression,
        joints,
        mapping,
        observation,
        projection,
        config,
        auxiliary,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
// explicit state/mapping/observation contract
// Jacobian dimensions, the parameter layout, and the update-vector length are
// all derived from the same validated model dimensions, so every index here is
// in bounds by construction.
#[allow(clippy::indexing_slicing, clippy::needless_range_loop)]
fn take_dense_expression_joint_step_impl(
    model: &GnmModel,
    identity: &GnmIdentityState,
    expression: &GnmExpressionState,
    joints: &GnmJointState,
    mapping: &DenseCorrespondenceSet,
    observation: &GnmDenseObservation,
    projection: &DenseProjection,
    config: DenseExpressionJointStepConfig,
    auxiliary: Option<(&dyn AuxiliaryObjectiveTerm, f32)>,
    temporal: Option<&SingleFrameTemporalPenalty<'_>>,
) -> Result<DenseExpressionJointStepOutcome, GnmReprojectionError> {
    let linearization = linearize_dense_reprojection(
        model,
        identity,
        expression,
        joints,
        mapping,
        observation,
        projection,
        DenseReprojectionConfig::default(),
        LinearizationStepSizes::default(),
    )?;
    let report = linearization.report();
    let residual_before = report.weighted_rms();

    let expression_jacobian = linearization.block(ReprojectionBlock::Expression).ok_or(
        GnmReprojectionError::InvalidConfig("missing expression block"),
    )?;
    let joint_jacobian = linearization
        .block(ReprojectionBlock::Joints)
        .ok_or(GnmReprojectionError::InvalidConfig("missing joints block"))?;

    let expression_count = model.expression_dimension();
    let joint_count = model.joint_count();
    let rotation_parameters = 3 * joint_count;
    let joint_parameters = 3 * (joint_count + 1);
    let total = expression_count + joint_parameters;

    // Weighted normal equations with an explicit neutral-pull prior,
    // accumulated directly into the packed lower triangle (Issue #148):
    // cell `(k, parameter)` with `k >= parameter` lives at
    // `k * (k + 1) / 2 + parameter` and receives exactly the contributions
    // the previous full symmetric assembly wrote into both triangles.
    let mut normal = vec![0.0f64; total * (total + 1) / 2];
    let mut rhs = vec![0.0f64; total];
    // Reused per-row gather of nonzero Jacobian entries `(parameter, value)`,
    // ascending by parameter so the suffix from any index covers exactly the
    // `k >= parameter` pairs.
    let mut row_entries: Vec<(usize, f64)> = Vec::with_capacity(total);
    for (point_row, residual) in report.residuals().iter().enumerate() {
        let weight = f64::from(residual.base_weight * residual.huber_weight);
        if weight <= 0.0 {
            continue;
        }
        let component_residuals = [
            f64::from(residual.residual_xy[0]),
            f64::from(residual.residual_xy[1]),
        ];
        for (component, component_residual) in component_residuals.into_iter().enumerate() {
            let row_offset = 2 * point_row + component;
            row_entries.clear();
            for parameter in 0..expression_count {
                let entry = f64::from(
                    expression_jacobian
                        .get(row_offset, parameter)
                        .unwrap_or_default(),
                );
                if entry != 0.0 {
                    row_entries.push((parameter, entry));
                }
            }
            for parameter in 0..joint_parameters {
                let entry = f64::from(
                    joint_jacobian
                        .get(row_offset, parameter)
                        .unwrap_or_default(),
                );
                if entry != 0.0 {
                    row_entries.push((expression_count + parameter, entry));
                }
            }
            // Rank-1 update over the row's nonzero pairs only. Every cell
            // receives the same additions in the same order as the previous
            // full-matrix assembly, so results stay bit-identical.
            for (index, &(parameter, entry)) in row_entries.iter().enumerate() {
                for &(k, other) in row_entries[index..].iter() {
                    let cell = k * (k + 1) / 2 + parameter;
                    normal[cell] += 0.5 * weight * entry * other;
                }
                rhs[parameter] -= 0.5 * weight * entry * component_residual;
            }
        }
    }
    for index in 0..total {
        normal[index * (index + 1) / 2 + index] +=
            f64::from(config.prior_weight) + f64::from(config.damping);
    }

    // Optional temporal term (Issue #64.4): analytic diagonal gradient and
    // curvature at the input state plus its energy for the combined acceptance
    // objective. With no term this is an exact no-op.
    let mut temporal_scratch = CandidateTemporalScratch::default();
    let (temporal_linearization, temporal_energy_before) = match temporal {
        Some(penalty) => {
            let view = candidate_state_view(expression, joints, projection, &mut temporal_scratch);
            let linearization = penalty.linearize_at(view)?;
            let energy = penalty.energy_at(view)?.total_weighted_energy;
            (Some(linearization), energy)
        }
        None => (None, 0.0),
    };
    if let Some(linearization) = &temporal_linearization {
        // Layout is [expression][joint rotations][joint translation]; the
        // temporal joint-group coordinate order matches exactly.
        for index in 0..expression_count {
            normal[index * (index + 1) / 2 + index] +=
                0.5 * linearization.expression.curvature[index];
            rhs[index] -= 0.5 * linearization.expression.gradient[index];
        }
        for index in 0..joint_parameters {
            let destination = expression_count + index;
            normal[destination * (destination + 1) / 2 + destination] +=
                0.5 * linearization.joints.curvature[index];
            rhs[destination] -= 0.5 * linearization.joints.gradient[index];
        }
    }

    // Optional weighted auxiliary term: inject its gradient into the normal
    // equations and remember its contribution to the acceptance objective.
    // With no term or a zero weight this is an exact no-op, so the update is
    // identical to the dense-only step.
    let (auxiliary_term, auxiliary_weight) = match auxiliary {
        Some((term, weight)) if weight.is_finite() && weight > 0.0 => {
            (Some(term), f64::from(weight))
        }
        Some((_, weight)) if !weight.is_finite() || weight < 0.0 => {
            return Err(GnmReprojectionError::InvalidConfig(
                "auxiliary weight must be finite and non-negative",
            ));
        }
        // Zero or subnormal positive weights disable the term exactly, so the
        // step is bit-identical to the dense-only update.
        _ => (None, 0.0),
    };
    let mut auxiliary_loss_before = 0.0f64;
    if let Some(term) = auxiliary_term {
        let evaluation = term.evaluate(
            expression.values(),
            joints.rotations(),
            joints.translation(),
        )?;
        if evaluation.loss.is_finite() && evaluation.loss < 0.0 {
            return Err(GnmReprojectionError::InvalidConfig(
                "auxiliary loss must be non-negative",
            ));
        }
        auxiliary_loss_before = auxiliary_weight * f64::from(evaluation.loss);
        let gradient = [
            evaluation.expression_gradient.as_slice(),
            evaluation.joint_gradient.as_slice(),
        ];
        for (offset, values) in gradient.iter().enumerate() {
            let expected = if offset == 0 {
                expression_count
            } else {
                joint_parameters
            };
            if values.len() != expected || values.iter().any(|value| !value.is_finite()) {
                return Err(GnmReprojectionError::InvalidConfig(
                    "auxiliary gradient dimension mismatch or non-finite entry",
                ));
            }
            #[allow(clippy::indexing_slicing)] // length verified immediately above
            for (index, value) in values.iter().enumerate() {
                // Joint parameters follow the expression block in `rhs`.
                let destination = if offset == 0 {
                    index
                } else {
                    expression_count + index
                };
                rhs[destination] -= auxiliary_weight * f64::from(*value);
            }
        }
    }

    // Combined-objective value at the input state.
    let objective_before =
        f64::from(residual_before) + auxiliary_loss_before + temporal_energy_before;

    let Some(step) = solve_spd_packed_lower(&mut normal, &mut rhs) else {
        return Ok(DenseExpressionJointStepOutcome {
            accepted: false,
            expression: expression.clone(),
            joints: joints.clone(),
            residual_before,
            residual_after: None,
            expression_step_norm: 0.0,
            joint_rotation_step_norm: 0.0,
            joint_translation_step_norm: 0.0,
        });
    };
    if step.iter().any(|value| !value.is_finite()) {
        return Ok(DenseExpressionJointStepOutcome {
            accepted: false,
            expression: expression.clone(),
            joints: joints.clone(),
            residual_before,
            residual_after: None,
            expression_step_norm: 0.0,
            joint_rotation_step_norm: 0.0,
            joint_translation_step_norm: 0.0,
        });
    }

    // Split and clamp the update per group.
    #[allow(clippy::indexing_slicing)] // layout fixed above: expression then joints
    {
        let expression_update = &step[..expression_count];
        let rotation_update = &step[expression_count..expression_count + rotation_parameters];
        let translation_update = &step[expression_count + rotation_parameters..];

        let norm = |values: &[f64]| values.iter().map(|value| value * value).sum::<f64>().sqrt();
        let mut scale = 1.0f64;
        let expression_norm = norm(expression_update);
        let rotation_norm = norm(rotation_update);
        let translation_norm = norm(translation_update);
        if expression_norm > f64::from(config.max_expression_step) {
            scale = scale.min(f64::from(config.max_expression_step) / expression_norm);
        }
        if rotation_norm > f64::from(config.max_joint_rotation_step) {
            scale = scale.min(f64::from(config.max_joint_rotation_step) / rotation_norm);
        }
        if translation_norm > f64::from(config.max_joint_translation_step) {
            scale = scale.min(f64::from(config.max_joint_translation_step) / translation_norm);
        }

        let mut candidate_values = expression.values().to_vec();
        for (value, delta) in candidate_values.iter_mut().zip(expression_update) {
            *value += (*delta * scale) as f32;
        }
        let Ok(candidate_expression) = GnmExpressionState::new(candidate_values, expression_count)
        else {
            return Ok(DenseExpressionJointStepOutcome {
                accepted: false,
                expression: expression.clone(),
                joints: joints.clone(),
                residual_before,
                residual_after: None,
                expression_step_norm: (expression_norm * scale) as f32,
                joint_rotation_step_norm: (rotation_norm * scale) as f32,
                joint_translation_step_norm: (translation_norm * scale) as f32,
            });
        };

        let mut rotations = joints.rotations().to_vec();
        let mut translation = joints.translation();
        for (value, delta) in rotations.iter_mut().flatten().zip(rotation_update) {
            *value += (*delta * scale) as f32;
        }
        for (value, delta) in translation.iter_mut().zip(translation_update) {
            *value += (*delta * scale) as f32;
        }
        let Ok(candidate_joints) = GnmJointState::new(rotations, translation, joint_count) else {
            return Ok(DenseExpressionJointStepOutcome {
                accepted: false,
                expression: expression.clone(),
                joints: joints.clone(),
                residual_before,
                residual_after: None,
                expression_step_norm: (expression_norm * scale) as f32,
                joint_rotation_step_norm: (rotation_norm * scale) as f32,
                joint_translation_step_norm: (translation_norm * scale) as f32,
            });
        };

        let rejected = |residual_after: Option<f32>| {
            Ok::<_, GnmReprojectionError>(DenseExpressionJointStepOutcome {
                accepted: false,
                expression: expression.clone(),
                joints: joints.clone(),
                residual_before,
                residual_after,
                expression_step_norm: (expression_norm * scale) as f32,
                joint_rotation_step_norm: (rotation_norm * scale) as f32,
                joint_translation_step_norm: (translation_norm * scale) as f32,
            })
        };

        let Ok(candidate_report) = evaluate_dense_reprojection(
            model,
            identity,
            &candidate_expression,
            &candidate_joints,
            mapping,
            observation,
            projection,
            DenseReprojectionConfig::default(),
        ) else {
            return rejected(None);
        };
        let residual_after = candidate_report.weighted_rms();
        let mut objective_after = f64::from(residual_after);
        if let Some(term) = auxiliary_term {
            let candidate_evaluation = term.evaluate(
                candidate_expression.values(),
                candidate_joints.rotations(),
                candidate_joints.translation(),
            )?;
            objective_after += auxiliary_weight * f64::from(candidate_evaluation.loss);
        }
        if let Some(penalty) = temporal {
            let view = candidate_state_view(
                &candidate_expression,
                &candidate_joints,
                projection,
                &mut temporal_scratch,
            );
            match penalty.energy_at(view) {
                Ok(metrics) => objective_after += metrics.total_weighted_energy,
                Err(_) => return rejected(Some(residual_after)),
            }
        }
        if !residual_after.is_finite()
            || !objective_after.is_finite()
            || objective_after >= objective_before
        {
            return rejected(Some(residual_after));
        }

        Ok(DenseExpressionJointStepOutcome {
            accepted: true,
            expression: candidate_expression,
            joints: candidate_joints,
            residual_before,
            residual_after: Some(residual_after),
            expression_step_norm: (expression_norm * scale) as f32,
            joint_rotation_step_norm: (rotation_norm * scale) as f32,
            joint_translation_step_norm: (translation_norm * scale) as f32,
        })
    }
}

/// Upper bound on block-coordinate iterations for one cold-start single-frame
/// fit. Matches the bounded per-frame contract in the tracking layer.
pub const MAX_SINGLE_FRAME_FIT_ITERATIONS: usize = 64;

/// Fixed cold-start schedule for one single-frame dense fit (Issue #64.2e).
///
/// The update order is fixed here and only here: every iteration applies one
/// bounded rigid pose + camera translation step ([`take_dense_rigid_step`]) and
/// then one bounded expression + joint step
/// ([`take_dense_expression_joint_step`]), both against the same observation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SingleFrameFitConfig {
    /// Bounded rigid pose + camera translation step configuration.
    pub rigid: DenseRigidStepConfig,
    /// Bounded expression + joint step configuration.
    pub expression_joint: DenseExpressionJointStepConfig,
    /// Maximum block-coordinate iterations. `1..=MAX_SINGLE_FRAME_FIT_ITERATIONS`.
    pub max_iterations: usize,
    /// Convergence threshold on the absolute combined-objective decrease per
    /// iteration. Must be finite and non-negative.
    pub tolerance: f32,
}

impl SingleFrameFitConfig {
    /// Creates a validated cold-start fit configuration.
    ///
    /// # Errors
    ///
    /// Returns [`GnmReprojectionError::InvalidConfig`] when the iteration bound
    /// is outside `1..=MAX_SINGLE_FRAME_FIT_ITERATIONS`, the tolerance is not
    /// finite and non-negative, or either nested step config is invalid.
    pub fn new(
        rigid: DenseRigidStepConfig,
        expression_joint: DenseExpressionJointStepConfig,
        max_iterations: usize,
        tolerance: f32,
    ) -> Result<Self, GnmReprojectionError> {
        if max_iterations == 0 || max_iterations > MAX_SINGLE_FRAME_FIT_ITERATIONS {
            return Err(GnmReprojectionError::InvalidConfig(
                "max_iterations must be within 1..=MAX_SINGLE_FRAME_FIT_ITERATIONS",
            ));
        }
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(GnmReprojectionError::InvalidConfig(
                "tolerance must be finite and non-negative",
            ));
        }
        Ok(Self {
            rigid,
            expression_joint,
            max_iterations,
            tolerance,
        })
    }
}

impl Default for SingleFrameFitConfig {
    fn default() -> Self {
        Self {
            rigid: DenseRigidStepConfig::default(),
            expression_joint: DenseExpressionJointStepConfig::default(),
            max_iterations: 40,
            tolerance: 1.0e-6,
        }
    }
}

/// Completion classification of one cold-start single-frame fit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SingleFrameFitStatus {
    /// The objective stopped decreasing beyond the configured tolerance (or
    /// neither block could accept a step), with finite state throughout.
    Converged,
    /// The iteration budget was exhausted while the objective was still
    /// improving beyond the tolerance.
    MaxIterationsReached,
    /// The objective or the dynamic state became non-finite during the fit.
    /// The carried state must not be published.
    NonFiniteState,
}

/// Result of one cold-start single-frame dense fit.
///
/// Only a [`SingleFrameFitStatus::Converged`] outcome is valid for publication;
/// [`SingleFrameFitOutcome::valid`] encodes that contract.
#[derive(Clone, Debug, PartialEq)]
pub struct SingleFrameFitOutcome {
    status: SingleFrameFitStatus,
    projection: DenseProjection,
    expression: GnmExpressionState,
    joints: GnmJointState,
    objective: f32,
    iterations: usize,
}

impl SingleFrameFitOutcome {
    /// Returns whether this result may be published as the frame's fit state.
    pub fn valid(&self) -> bool {
        self.status == SingleFrameFitStatus::Converged
    }

    /// Returns the completion classification.
    pub const fn status(&self) -> SingleFrameFitStatus {
        self.status
    }

    /// Returns the final projection (pose + camera translation blocks).
    pub const fn projection(&self) -> &DenseProjection {
        &self.projection
    }

    /// Returns the final expression state.
    pub const fn expression(&self) -> &GnmExpressionState {
        &self.expression
    }

    /// Returns the final joint state.
    pub const fn joints(&self) -> &GnmJointState {
        &self.joints
    }

    /// Returns the combined weighted objective at the final state when finite.
    pub const fn objective(&self) -> f32 {
        self.objective
    }

    /// Returns the number of completed block-coordinate iterations.
    pub const fn iterations(&self) -> usize {
        self.iterations
    }
}

/// Fits one frame from a cold start by alternating the bounded rigid/camera
/// step and the bounded expression/joint step until convergence.
///
/// No warm start, temporal energy, or worker semantics are applied here; those
/// belong to later leaves. The optional auxiliary term is forwarded to every
/// expression/joint step unchanged.
///
/// # Errors
///
/// Propagates typed failures from the underlying steps and from an observation
/// that cannot be evaluated at all; non-finite objectives instead terminate the
/// loop with [`SingleFrameFitStatus::NonFiniteState`].
#[allow(clippy::too_many_arguments)]
// explicit state/mapping/observation contract
pub fn fit_single_frame_cold_start(
    model: &GnmModel,
    identity: &GnmIdentityState,
    expression: &GnmExpressionState,
    joints: &GnmJointState,
    mapping: &DenseCorrespondenceSet,
    observation: &GnmDenseObservation,
    projection: &DenseProjection,
    config: SingleFrameFitConfig,
    auxiliary: Option<(&dyn AuxiliaryObjectiveTerm, f32)>,
) -> Result<SingleFrameFitOutcome, GnmReprojectionError> {
    fit_single_frame_cold_start_impl(
        model,
        identity,
        expression,
        joints,
        mapping,
        observation,
        projection,
        config,
        auxiliary,
        None,
    )
}

/// Fits one frame from an optional warm start with an optional temporal
/// energy term connected into the solver objective (Issue #64.4).
///
/// When `temporal` is `Some`, every iteration evaluates the fixed first/second
/// order temporal energy with the actual source-frame `dt`, injects its
/// analytic diagonal gradient into both bounded steps, and includes it in the
/// combined convergence objective. A source gap beyond the configured bound
/// propagates [`TemporalRegularizationError::HistoryResetRequired`] instead of
/// applying stale history.
#[allow(clippy::too_many_arguments)]
// explicit state/mapping/observation contract
pub fn fit_single_frame_with_temporal(
    model: &GnmModel,
    identity: &GnmIdentityState,
    expression: &GnmExpressionState,
    joints: &GnmJointState,
    mapping: &DenseCorrespondenceSet,
    observation: &GnmDenseObservation,
    projection: &DenseProjection,
    config: SingleFrameFitConfig,
    auxiliary: Option<(&dyn AuxiliaryObjectiveTerm, f32)>,
    temporal: Option<&SingleFrameTemporalPenalty<'_>>,
) -> Result<SingleFrameFitOutcome, GnmReprojectionError> {
    fit_single_frame_cold_start_impl(
        model,
        identity,
        expression,
        joints,
        mapping,
        observation,
        projection,
        config,
        auxiliary,
        temporal,
    )
}

#[allow(clippy::too_many_arguments)]
// explicit state/mapping/observation contract
fn fit_single_frame_cold_start_impl(
    model: &GnmModel,
    identity: &GnmIdentityState,
    expression: &GnmExpressionState,
    joints: &GnmJointState,
    mapping: &DenseCorrespondenceSet,
    observation: &GnmDenseObservation,
    projection: &DenseProjection,
    config: SingleFrameFitConfig,
    auxiliary: Option<(&dyn AuxiliaryObjectiveTerm, f32)>,
    temporal: Option<&SingleFrameTemporalPenalty<'_>>,
) -> Result<SingleFrameFitOutcome, GnmReprojectionError> {
    // Fail closed on invalid solver-level auxiliary weight exactly like the
    // expression/joint step does.
    if let Some((_, weight)) = auxiliary
        && (!weight.is_finite() || weight < 0.0)
    {
        return Err(GnmReprojectionError::InvalidConfig(
            "auxiliary weight must be finite and non-negative",
        ));
    }

    let mut projection = *projection;
    let mut expression = expression.clone();
    let mut joints = joints.clone();
    let mut temporal_scratch = CandidateTemporalScratch::default();

    let mut objective_at = |expression: &GnmExpressionState,
                            joints: &GnmJointState,
                            projection: &DenseProjection|
     -> Result<f64, GnmReprojectionError> {
        let report = evaluate_dense_reprojection(
            model,
            identity,
            expression,
            joints,
            mapping,
            observation,
            projection,
            DenseReprojectionConfig::default(),
        )?;
        let dense = f64::from(report.weighted_rms());
        let auxiliary_loss = match auxiliary {
            Some((term, weight)) => {
                let evaluation = term.evaluate(
                    expression.values(),
                    joints.rotations(),
                    joints.translation(),
                )?;
                if evaluation.loss < 0.0 {
                    return Err(GnmReprojectionError::InvalidConfig(
                        "auxiliary loss must be non-negative",
                    ));
                }
                f64::from(weight) * f64::from(evaluation.loss)
            }
            None => 0.0,
        };
        let temporal_energy = match temporal {
            Some(penalty) => {
                penalty
                    .energy_at(candidate_state_view(
                        expression,
                        joints,
                        projection,
                        &mut temporal_scratch,
                    ))?
                    .total_weighted_energy
            }
            None => 0.0,
        };
        Ok(dense + auxiliary_loss + temporal_energy)
    };

    let mut objective = objective_at(&expression, &joints, &projection)?;
    if !objective.is_finite() {
        return Ok(SingleFrameFitOutcome {
            status: SingleFrameFitStatus::NonFiniteState,
            projection,
            expression,
            joints,
            objective: f32::NAN,
            iterations: 0,
        });
    }

    for iteration in 1..=config.max_iterations {
        let rigid = take_dense_rigid_step_impl(
            model,
            identity,
            &expression,
            &joints,
            mapping,
            observation,
            &projection,
            config.rigid,
            temporal,
        )?;
        if rigid.accepted {
            projection = DenseProjection::new(
                rigid.yaw_pitch_roll,
                rigid.translation,
                projection.focal(),
                projection.principal_point(),
            )?;
        }

        let expression_joint = take_dense_expression_joint_step_impl(
            model,
            identity,
            &expression,
            &joints,
            mapping,
            observation,
            &projection,
            config.expression_joint,
            auxiliary,
            temporal,
        )?;
        if expression_joint.accepted {
            expression = expression_joint.expression;
            joints = expression_joint.joints;
        }

        let updated_objective = objective_at(&expression, &joints, &projection)?;
        if !updated_objective.is_finite()
            || !expression
                .values()
                .iter()
                .chain(joints.rotations().iter().flatten())
                .chain(joints.translation().iter())
                .all(|value| value.is_finite())
        {
            return Ok(SingleFrameFitOutcome {
                status: SingleFrameFitStatus::NonFiniteState,
                projection,
                expression,
                joints,
                objective: f32::NAN,
                iterations: iteration,
            });
        }

        let improvement = objective - updated_objective;
        objective = updated_objective;
        let stalled = !rigid.accepted && !expression_joint.accepted;
        if stalled || improvement <= f64::from(config.tolerance) {
            return Ok(SingleFrameFitOutcome {
                status: SingleFrameFitStatus::Converged,
                projection,
                expression,
                joints,
                objective: objective as f32,
                iterations: iteration,
            });
        }
    }

    Ok(SingleFrameFitOutcome {
        status: SingleFrameFitStatus::MaxIterationsReached,
        projection,
        expression,
        joints,
        objective: objective as f32,
        iterations: config.max_iterations,
    })
}

/// Parameter block of the dense reprojection linearization.
///
/// Identity is intentionally absent: fixed identity is calibration evidence,
/// never a linearized update block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReprojectionBlock {
    /// Expression coefficients.
    Expression,
    /// Joint rotations followed by the joint translation.
    Joints,
    /// Rigid head pose (yaw/pitch/roll).
    RigidPose,
    /// Camera-space head translation held by the projection.
    CameraTranslation,
}

impl ReprojectionBlock {
    /// Stable block name used in diagnostics and errors.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Expression => "expression",
            Self::Joints => "joints",
            Self::RigidPose => "rigid_pose",
            Self::CameraTranslation => "camera_translation",
        }
    }

    /// Parameter count for the given model dimensions.
    pub fn parameter_count(self, model: &GnmModel) -> usize {
        match self {
            Self::Expression => model.expression_dimension(),
            Self::Joints => 3 * (model.joint_count() + 1),
            Self::RigidPose | Self::CameraTranslation => 3,
        }
    }

    /// All blocks in canonical update order.
    pub const ALL: [Self; 4] = [
        Self::Expression,
        Self::Joints,
        Self::RigidPose,
        Self::CameraTranslation,
    ];
}

/// Finite-difference step sizes for one linearization pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearizationStepSizes {
    /// Step per expression coefficient.
    pub expression: f32,
    /// Step per joint rotation Euler component.
    pub joint_rotation: f32,
    /// Step for each joint-translation component.
    pub joint_translation: f32,
    /// Step per rigid pose Euler component.
    pub rigid_pose: f32,
    /// Step for each camera-translation component.
    pub camera_translation: f32,
}

impl LinearizationStepSizes {
    /// Validates all steps as finite and positive.
    ///
    /// # Errors
    ///
    /// Returns [`GnmReprojectionError::InvalidConfig`] for any non-finite or
    /// non-positive step.
    pub fn validate(&self) -> Result<(), GnmReprojectionError> {
        for (_name, step) in [
            ("expression", self.expression),
            ("joint_rotation", self.joint_rotation),
            ("joint_translation", self.joint_translation),
            ("rigid_pose", self.rigid_pose),
            ("camera_translation", self.camera_translation),
        ] {
            if !step.is_finite() || step <= 0.0 {
                return Err(GnmReprojectionError::InvalidConfig(
                    "linearization steps must be finite and positive",
                ));
            }
        }
        Ok(())
    }
}

impl Default for LinearizationStepSizes {
    fn default() -> Self {
        Self {
            // Steps are sized above the f32 ulp of projected coordinates so a
            // single forward difference stays meaningful.
            expression: 1.0e-3,
            joint_rotation: 1.0e-3,
            joint_translation: 1.0e-3,
            rigid_pose: 1.0e-4,
            camera_translation: 1.0e-3,
        }
    }
}

/// Row-major Jacobian of the retained residual vector with respect to one
/// parameter block. Rows are `2 * residuals.len()` (`x` then `y` per point).
#[derive(Clone, Debug, PartialEq)]
pub struct BlockJacobian {
    /// Block these columns belong to.
    pub block: ReprojectionBlock,
    /// Number of parameters (columns).
    pub parameter_count: usize,
    /// Number of rows (`2 * residual count`).
    pub row_count: usize,
    /// Row-major entries.
    pub entries: Vec<f32>,
}

impl BlockJacobian {
    /// Returns the entry at `(row, column)`, or `None` when out of bounds.
    pub fn get(&self, row: usize, column: usize) -> Option<f32> {
        if row >= self.row_count || column >= self.parameter_count {
            return None;
        }
        self.entries
            .get(row * self.parameter_count + column)
            .copied()
    }
}

/// Residual bundle plus per-block linearization for one observation frame.
#[derive(Clone, Debug)]
pub struct DenseLinearization {
    report: DenseReprojectionReport,
    blocks: Vec<BlockJacobian>,
}

impl DenseLinearization {
    /// Returns the baseline weighted residual report at the current state.
    pub fn report(&self) -> &DenseReprojectionReport {
        &self.report
    }

    /// Returns the per-block Jacobians in [`ReprojectionBlock::ALL`] order.
    pub fn blocks(&self) -> &[BlockJacobian] {
        &self.blocks
    }

    /// Returns the Jacobian of one block.
    pub fn block(&self, block: ReprojectionBlock) -> Option<&BlockJacobian> {
        self.blocks
            .iter()
            .find(|candidate| candidate.block == block)
    }
}

/// Computes the dense reprojection objective and its first-order
/// linearization with respect to expression, joint, rigid-pose, and
/// camera-translation blocks.
///
/// The expression block is assembled **analytically**: the surface is linear
/// in the expression coefficients (`∂surface_v / ∂e_k = basis_k[v]`, skinned
/// by the fixed pose), so no perturbed surface re-evaluation happens for it.
/// The remaining nonlinear blocks (joints, rigid pose, camera translation)
/// still use forward finite differences. This is a pure numerical component:
/// it computes residuals and Jacobian entries only and performs no parameter
/// update. Point weights (static base weight and Huber robust weight) live in
/// the baseline report; invalid points excluded there also receive no Jacobian
/// rows. Fixed identity is read-only surface input and is never included as an
/// update block.
///
/// # Errors
///
/// Fails closed on invalid configuration, state dimension mismatch against the
/// model, invalid projection parameters, an observation with no usable
/// residual, or any non-finite Jacobian entry.
#[allow(clippy::too_many_arguments)] // explicit state/mapping/observation contract
pub fn linearize_dense_reprojection(
    model: &GnmModel,
    identity: &GnmIdentityState,
    expression: &GnmExpressionState,
    joints: &GnmJointState,
    mapping: &DenseCorrespondenceSet,
    observation: &GnmDenseObservation,
    projection: &DenseProjection,
    config: DenseReprojectionConfig,
    steps: LinearizationStepSizes,
) -> Result<DenseLinearization, GnmReprojectionError> {
    config.validate()?;
    steps.validate()?;
    if expression.values().len() != model.expression_dimension() {
        return Err(GnmReprojectionError::InvalidConfig(
            "expression dimension does not match the model",
        ));
    }
    if joints.rotations().len() != model.joint_count() {
        return Err(GnmReprojectionError::InvalidConfig(
            "joint count does not match the model",
        ));
    }

    // Staged evaluation: one pre-skinning pass shared by the baseline report,
    // the analytic expression block, and the joint-perturbation columns.
    let prepared =
        model.prepare_sparse_vertices(identity, expression, joints, mapping.surface_landmarks())?;
    let baseline_surface = prepared.skin(model, identity, joints, mapping.surface_landmarks())?;
    let report = evaluate_report_from_surface(observation, projection, config, &baseline_surface)?;
    let retained: Vec<usize> = report
        .residuals()
        .iter()
        .map(|residual| residual.mapping_index)
        .collect();
    // Baseline projections in retained order, reused by every block so the
    // shared surface is never re-projected per column.
    let baseline_projected: Vec<[f32; 2]> = report
        .residuals()
        .iter()
        .map(|residual| residual.projected_xy)
        .collect();
    let row_count = 2 * retained.len();

    // Per-point skinning derivative maps for the analytic expression block;
    // independent of which coefficient is being differentiated.
    let skinning =
        model.sparse_skinning_derivatives(identity, joints, mapping.surface_landmarks())?;

    let mut blocks = Vec::with_capacity(ReprojectionBlock::ALL.len());
    for block in ReprojectionBlock::ALL {
        let parameter_count = block.parameter_count(model);
        let mut entries = vec![0.0f32; row_count * parameter_count];
        match block {
            ReprojectionBlock::Expression => {
                analytic_expression_columns(
                    model,
                    projection,
                    &baseline_surface,
                    &skinning,
                    &retained,
                    parameter_count,
                    &mut entries,
                )?;
            }
            _ => {
                for parameter in 0..parameter_count {
                    let column = perturbed_residuals(
                        model,
                        identity,
                        expression,
                        joints,
                        mapping,
                        projection,
                        &baseline_surface,
                        &baseline_projected,
                        &prepared,
                        block,
                        parameter,
                        steps,
                        &retained,
                    )?;
                    for (point_row, delta) in column.iter().enumerate() {
                        // Row count is exactly 2 * retained.len(), so these indices
                        // are in bounds by construction.
                        #[allow(clippy::indexing_slicing)]
                        {
                            entries[(2 * point_row) * parameter_count + parameter] = delta[0];
                            entries[(2 * point_row + 1) * parameter_count + parameter] = delta[1];
                        }
                    }
                }
            }
        }
        if entries.iter().any(|entry| !entry.is_finite()) {
            return Err(GnmReprojectionError::NonFiniteLinearization {
                block: block.name(),
            });
        }
        blocks.push(BlockJacobian {
            block,
            parameter_count,
            row_count,
            entries,
        });
    }

    Ok(DenseLinearization { report, blocks })
}

/// Evaluates `(perturbed - baseline) / step` for every retained point under a
/// single-parameter perturbation of `block[parameter]`.
///
/// `baseline_projected` holds the baseline projections in retained order
/// (taken from the residual report) so the baseline is never re-projected per
/// column. `prepared` carries the pre-skinning vertex values; when the model
/// has no pose correctives, joint-pose perturbations reuse it instead of
/// re-running the identity/expression basis loops.
///
/// Returns one `[dx/step, dy/step]` pair per retained point in report order.
#[allow(clippy::too_many_arguments)] // explicit state/mapping contract
fn perturbed_residuals(
    model: &GnmModel,
    identity: &GnmIdentityState,
    expression: &GnmExpressionState,
    joints: &GnmJointState,
    mapping: &DenseCorrespondenceSet,
    projection: &DenseProjection,
    baseline_surface: &[[f32; 3]],
    baseline_projected: &[[f32; 2]],
    prepared: &SparsePreparedVertices,
    block: ReprojectionBlock,
    parameter: usize,
    steps: LinearizationStepSizes,
    retained: &[usize],
) -> Result<Vec<[f32; 2]>, GnmReprojectionError> {
    // Surface-affecting blocks re-evaluate the surface for this single
    // perturbed parameter; pose/camera blocks reuse the baseline surface.
    let owned_surface;
    let surface_points: &[[f32; 3]] = match block {
        ReprojectionBlock::Expression => {
            let mut values = expression.values().to_vec();
            // parameter < expression_dimension was checked by the caller.
            #[allow(clippy::indexing_slicing)]
            {
                values[parameter] += steps.expression;
            }
            let perturbed = GnmExpressionState::new(values, model.expression_dimension())
                .map_err(GnmReprojectionError::Model)?;
            let mut surface = GnmSparseVertices::with_len(mapping.len());
            mapping.evaluate_surface(model, identity, &perturbed, joints, &mut surface)?;
            owned_surface = surface.values().to_vec();
            &owned_surface
        }
        ReprojectionBlock::Joints => {
            let rotation_count = 3 * model.joint_count();
            let mut rotations = joints.rotations().to_vec();
            let mut translation = joints.translation();
            if parameter < rotation_count {
                // joint/axis are within range because parameter < 3*joint_count.
                #[allow(clippy::indexing_slicing)]
                {
                    let joint = parameter / 3;
                    let axis = parameter % 3;
                    rotations[joint][axis] += steps.joint_rotation;
                }
            } else {
                // Axis is within 0..3 because the joint block has exactly
                // three translation parameters.
                #[allow(clippy::indexing_slicing)]
                {
                    translation[parameter - rotation_count] += steps.joint_translation;
                }
            }
            let perturbed = GnmJointState::new(rotations, translation, model.joint_count())
                .map_err(GnmReprojectionError::Model)?;
            if prepared.depends_on_joint_state() {
                // Pose correctives embed the joint rotations in the pre-skin
                // values; only a full re-evaluation is valid here.
                let mut surface = GnmSparseVertices::with_len(mapping.len());
                mapping.evaluate_surface(model, identity, expression, &perturbed, &mut surface)?;
                owned_surface = surface.values().to_vec();
            } else {
                owned_surface =
                    prepared.skin(model, identity, &perturbed, mapping.surface_landmarks())?;
            }
            &owned_surface
        }
        _ => baseline_surface,
    };

    // Per-block perturbation step and effective camera after perturbation.
    let rotation_parameters = 3 * model.joint_count();
    let (step, active_projection) = match block {
        ReprojectionBlock::Expression => (steps.expression, *projection),
        ReprojectionBlock::Joints => {
            let step = if parameter < rotation_parameters {
                steps.joint_rotation
            } else {
                steps.joint_translation
            };
            (step, *projection)
        }
        ReprojectionBlock::RigidPose => {
            let mut yaw_pitch_roll = projection.yaw_pitch_roll();
            // RigidPose has exactly three parameters.
            #[allow(clippy::indexing_slicing)]
            {
                yaw_pitch_roll[parameter] += steps.rigid_pose;
            }
            let perturbed = DenseProjection::new(
                yaw_pitch_roll,
                projection.translation(),
                projection.focal(),
                projection.principal_point(),
            )?;
            (steps.rigid_pose, perturbed)
        }
        ReprojectionBlock::CameraTranslation => {
            let mut translation = projection.translation();
            // CameraTranslation has exactly three parameters.
            #[allow(clippy::indexing_slicing)]
            {
                translation[parameter] += steps.camera_translation;
            }
            let perturbed = DenseProjection::new(
                projection.yaw_pitch_roll(),
                translation,
                projection.focal(),
                projection.principal_point(),
            )?;
            (steps.camera_translation, perturbed)
        }
    };

    let mut deltas = Vec::with_capacity(retained.len());
    for (point_row, &mapping_index) in retained.iter().enumerate() {
        // mapping_index is validated by the dense observation contract.
        #[allow(clippy::indexing_slicing)]
        let Some(projected) = active_projection.project(surface_points[mapping_index]) else {
            return Err(GnmReprojectionError::InsufficientObservation);
        };
        #[allow(clippy::indexing_slicing)]
        let baseline = baseline_projected[point_row];
        // Forward difference of the residual `observed - projected`:
        // ((observed - p') - (observed - p0)) / step = (p0 - p') / step.
        deltas.push([
            (baseline[0] - projected[0]) / step,
            (baseline[1] - projected[1]) / step,
        ]);
    }
    Ok(deltas)
}

/// Hoisted camera state shared by every analytic expression column.
///
/// [`DenseProjection::project`] recomputes its trigonometry for every call;
/// because yaw/pitch/roll and intrinsics are fixed across all expression
/// columns, the composed camera rotation and its projection Jacobian are
/// computed once here.
struct CachedCameraRotation {
    /// Camera rotation `R = Rz(roll)·Rx(pitch)·Ry(yaw)` in f64, rows first.
    rotation: [[f64; 3]; 3],
    translation: [f64; 3],
    focal: f64,
}

impl CachedCameraRotation {
    fn new(projection: &DenseProjection) -> Self {
        let [yaw, pitch, roll] = [
            projection.yaw_pitch_roll()[0] as f64,
            projection.yaw_pitch_roll()[1] as f64,
            projection.yaw_pitch_roll()[2] as f64,
        ];
        let (sy, cy) = yaw.sin_cos();
        let (sp, cp) = pitch.sin_cos();
        let (sr, cr) = roll.sin_cos();
        let ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]];
        let rx = [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]];
        let rz = [[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]];
        Self {
            rotation: multiply3(&rz, &multiply3(&rx, &ry)),
            translation: [
                projection.translation()[0] as f64,
                projection.translation()[1] as f64,
                projection.translation()[2] as f64,
            ],
            focal: projection.focal() as f64,
        }
    }

    /// Returns `(∂u/∂p, ∂v/∂p)` of the projected coordinate at `point`.
    ///
    /// With `u = cx + f·(r0·p + tx)/z`, `v = cy − f·(r1·p + ty)/z` and
    /// `z = r2·p + tz`, the quotient rule gives `∂u/∂p = f/z·(r0 − wx·r2)`
    /// and `∂v/∂p = −f/z·(r1 − wy·r2)` with `wx = (r0·p + tx)/z`,
    /// `wy = (r1·p + ty)/z`.
    fn projection_jacobian(&self, point: [f32; 3]) -> Option<[[f64; 3]; 2]> {
        let rotated = rotate3(&self.rotation, point);
        let z = rotated[2] + self.translation[2];
        if !z.is_finite() || z <= 1.0e-6 {
            return None;
        }
        let wx = (rotated[0] + self.translation[0]) / z;
        let wy = (rotated[1] + self.translation[1]) / z;
        let scale = self.focal / z;
        let [r00, r01, r02] = self.rotation[0];
        let [r10, r11, r12] = self.rotation[1];
        let [r20, r21, r22] = self.rotation[2];
        Some([
            [
                scale * (r00 - wx * r20),
                scale * (r01 - wx * r21),
                scale * (r02 - wx * r22),
            ],
            [
                -scale * (r10 - wy * r20),
                -scale * (r11 - wy * r21),
                -scale * (r12 - wy * r22),
            ],
        ])
    }
}

fn rotate3(rotation: &[[f64; 3]; 3], point: [f32; 3]) -> [f64; 3] {
    let x = point[0] as f64;
    let y = point[1] as f64;
    let z = point[2] as f64;
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    [
        rotation[0][0] * x + rotation[0][1] * y + rotation[0][2] * z,
        rotation[1][0] * x + rotation[1][1] * y + rotation[1][2] * z,
        rotation[2][0] * x + rotation[2][1] * y + rotation[2][2] * z,
    ]
}

fn multiply3(left: &[[f64; 3]; 3], right: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    fn compose(left: &[[f64; 3]; 3], right: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
        let mut result = [[0.0; 3]; 3];
        for row in 0..3 {
            for column in 0..3 {
                for index in 0..3 {
                    result[row][column] += left[row][index] * right[index][column];
                }
            }
        }
        result
    }
    compose(left, right)
}

/// Fills the analytic expression-block Jacobian columns.
///
/// The surface is linear in the expression coefficients, so the exact surface
/// derivative for coefficient `e_k` is the skinned basis offset from
/// `skinning`. Chaining it with the closed-form projection Jacobian yields the
/// residual derivative `∂(observed − projected)/∂e_k = −J_proj · offset`
/// without any surface re-evaluation or re-projection. Zero basis entries keep
/// their zero rows, mirroring the sparse structure of the basis.
///
/// # Errors
///
/// Fails when a retained baseline projection is unusable, or when the skinning
/// derivatives reject a column index.
#[allow(clippy::too_many_arguments)] // explicit state/mapping contract
fn analytic_expression_columns(
    model: &GnmModel,
    projection: &DenseProjection,
    baseline_surface: &[[f32; 3]],
    skinning: &SparseSkinningDerivatives,
    retained: &[usize],
    parameter_count: usize,
    entries: &mut [f32],
) -> Result<(), GnmReprojectionError> {
    let camera = CachedCameraRotation::new(projection);
    // One skinning derivative slot per mapped point (not per retained row):
    // `expression_point_offsets` writes every mapped point and the loop below
    // selects each retained row by its mapping index.
    let mut offsets = vec![[0.0f32; 3]; skinning.len()];
    let active_columns = skinning.active_expression_columns(model);
    for parameter in 0..parameter_count {
        if !active_columns.get(parameter).copied().unwrap_or(false) {
            // Sparse basis channel: no mapped point moves; both Jacobian rows
            // stay exactly zero.
            continue;
        }
        skinning.expression_point_offsets(model, parameter, &mut offsets)?;
        for (point_row, &mapping_index) in retained.iter().enumerate() {
            // Mapping indices are constructed against the same mapped-point
            // table the skinning derivatives were built from, so every
            // retained index is inside `offsets`.
            #[allow(clippy::indexing_slicing)]
            let offset = offsets[mapping_index];
            if offset == [0.0; 3] {
                continue;
            }
            #[allow(clippy::indexing_slicing)]
            let base_point = baseline_surface[mapping_index];
            let Some(jacobian) = camera.projection_jacobian(base_point) else {
                return Err(GnmReprojectionError::InsufficientObservation);
            };
            let dx = -(jacobian[0][0] * offset[0] as f64
                + jacobian[0][1] * offset[1] as f64
                + jacobian[0][2] * offset[2] as f64) as f32;
            let dy = -(jacobian[1][0] * offset[0] as f64
                + jacobian[1][1] * offset[1] as f64
                + jacobian[1][2] * offset[2] as f64) as f32;
            // Row count is exactly 2 * retained.len(), so these indices are
            // in bounds by construction.
            #[allow(clippy::indexing_slicing)]
            {
                entries[(2 * point_row) * parameter_count + parameter] = dx;
                entries[(2 * point_row + 1) * parameter_count + parameter] = dy;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::test_support::version;
    use crate::{
        CorrespondenceProvenance, CorrespondenceReliability, DenseArray, DenseCorrespondenceSet,
        GNM_HEAD_V3_EXPRESSION_DIM, GNM_HEAD_V3_IDENTITY_DIM, GNM_HEAD_V3_VERSION, GnmModelData,
        GnmSurfacePointRef, GnmVariant, MediaPipeGnmDenseCorrespondence,
        SPARSE_BOOTSTRAP_POINT_COUNT,
    };

    /// A pseudo-3D deterministic point cloud with depth variation, unlike the
    /// planar strip used by the dense-module validation fixtures.
    fn spread_model(vertex_count: usize) -> GnmModel {
        // Reuse the version-matched builder, then override template vertices by
        // building a dedicated model here (the dense helper is planar).
        let identity = crate::GNM_HEAD_V3_IDENTITY_DIM;
        let expression = crate::GNM_HEAD_V3_EXPRESSION_DIM;
        let mut vertices = Vec::with_capacity(vertex_count * 3);
        for index in 0..vertex_count {
            let angle = (index as f32) / (vertex_count as f32) * std::f32::consts::TAU;
            vertices.extend_from_slice(&[
                0.10 * angle.cos(),
                0.12 * angle.sin(),
                0.05 * (3.0 * angle).sin(),
            ]);
        }
        crate::GnmModel::from_data(crate::GnmModelData {
            version: crate::GNM_HEAD_V3_VERSION,
            variant: crate::GnmVariant::Head,
            template_vertices: crate::DenseArray::new("vertices", vec![vertex_count, 3], vertices)
                .unwrap(),
            template_joints: crate::DenseArray::new("joints", vec![1, 3], vec![0.0; 3]).unwrap(),
            vertex_identity_basis: crate::DenseArray::new(
                "identity",
                vec![identity, vertex_count, 3],
                vec![0.0; identity * vertex_count * 3],
            )
            .unwrap(),
            joint_identity_basis: crate::DenseArray::new(
                "joint_identity",
                vec![identity, 1, 3],
                vec![0.0; identity * 3],
            )
            .unwrap(),
            expression_basis: crate::DenseArray::new(
                "expression",
                vec![expression, vertex_count, 3],
                vec![0.0; expression * vertex_count * 3],
            )
            .unwrap(),
            joint_parent_indices: vec![-1],
            skinning_weights: crate::DenseArray::new(
                "weights",
                vec![1, vertex_count],
                vec![1.0; vertex_count],
            )
            .unwrap(),
            pose_correctives_regressor: None,
        })
        .unwrap()
    }

    fn mapping_for(model: &GnmModel, count: usize) -> DenseCorrespondenceSet {
        let rows: Vec<MediaPipeGnmDenseCorrespondence> = (0..count)
            .map(|index| MediaPipeGnmDenseCorrespondence {
                mediapipe_index: 10 + index,
                target: GnmSurfacePointRef::Vertex {
                    vertex_index: index,
                },
                region: if index % 3 == 0 {
                    FaceRegion::Nose
                } else if index % 3 == 1 {
                    FaceRegion::Contour
                } else {
                    FaceRegion::Other
                },
                anatomical_side: if index % 3 == 0 {
                    AnatomicalSide::Midline
                } else if index % 3 == 1 {
                    AnatomicalSide::Right
                } else {
                    AnatomicalSide::Left
                },
                base_weight: 1.0,
                provenance: CorrespondenceProvenance::RepositoryValidated,
                reliability: CorrespondenceReliability::High,
            })
            .collect();
        DenseCorrespondenceSet::new(version(), rows, model).unwrap()
    }

    fn truth_projection() -> DenseProjection {
        DenseProjection::new([0.15, -0.10, 0.05], [0.02, -0.03, 0.60], 1.3, [0.5, 0.5]).unwrap()
    }

    fn perturbed_guess() -> DenseProjection {
        DenseProjection::new([0.20, -0.14, 0.09], [0.06, 0.01, 0.66], 1.45, [0.5, 0.5]).unwrap()
    }

    #[test]
    fn projection_follows_documented_conventions() {
        let projection = DenseProjection::new([0.0; 3], [0.0; 3], 1.3, [0.5, 0.5]).unwrap();
        let projected = projection.project([0.1, 0.2, 1.0]).unwrap();
        assert!((projected[0] - 0.63).abs() < 1.0e-6);
        assert!((projected[1] - 0.24).abs() < 1.0e-6);

        // +90° yaw sends +x to -z, which is behind the principal plane.
        let yawed = DenseProjection::new(
            [std::f32::consts::FRAC_PI_2, 0.0, 0.0],
            [0.0; 3],
            1.3,
            [0.5, 0.5],
        )
        .unwrap();
        assert_eq!(yawed.project([1.0, 0.0, 0.05]), None);

        // Roll-only rotation keeps the +x point on the image x axis; because y
        // grows downward, positive v corresponds to -Y in camera space.
        let rolled = DenseProjection::new(
            [0.0, 0.0, std::f32::consts::FRAC_PI_2],
            [0.0; 3],
            1.3,
            [0.5, 0.5],
        )
        .unwrap();
        let projected = rolled.project([1.0, 0.0, 2.0]).unwrap();
        assert!((projected[0] - 0.5).abs() < 1.0e-6);
        assert!(projected[1] < 0.5);
    }

    #[test]
    fn region_fit_records_cover_every_fixed_region_without_affecting_acceptance() {
        let model = spread_model(3);
        let mapping = mapping_for(&model, 3);
        let mut landmarks = vec![[f32::NAN; 2]; MEDIAPIPE_FACE_LANDMARK_COUNT];
        let projected = [[0.10, 0.20], [0.30, 0.40], [0.50, 0.60]];
        for (row, point) in mapping.rows().iter().zip(projected) {
            landmarks[row.mediapipe_index] = point;
        }
        let observation = GnmDenseObservation::from_mediapipe_xy(
            1,
            1_000,
            &landmarks,
            &mapping,
            DenseCoveragePolicy::new(2, 0.5).unwrap(),
        )
        .unwrap();

        let records = region_fit_records(&mapping, &observation, &projected).unwrap();
        assert_eq!(records.len(), 7);
        for region in [FaceRegion::Nose, FaceRegion::Contour, FaceRegion::Other] {
            let record = records
                .iter()
                .find(|record| record.region == region)
                .unwrap();
            assert_eq!(record.valid_points, 1);
            assert_eq!(record.weighted_rms, 0.0);
        }
        assert_eq!(
            records
                .iter()
                .find(|record| record.region == FaceRegion::Mouth)
                .unwrap()
                .valid_points,
            0
        );
    }

    #[test]
    fn jacobi_eigen_range_matches_known_values() {
        // Eigenvalues of [[4, 1], [1, 3]] are (7 ± sqrt(5))/2.
        // Filler diagonal values stay strictly inside (2.382, 21) so the
        // 2×2 block owns both extremes of the spectrum.
        let matrix = [
            [4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 11.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 21.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.0],
        ];
        let (smallest, largest) = eigen_range(&matrix);
        let expected_small = (7.0 - 5.0f64.sqrt()) / 2.0;
        assert!(
            (smallest - expected_small).abs() < 1.0e-9,
            "smallest {smallest} != {expected_small} (largest {largest})"
        );
        assert!(
            (largest - 21.0).abs() < 1.0e-9,
            "largest {largest} != 21 (smallest {smallest})"
        );
    }

    #[test]
    fn invalid_projection_parameters_are_rejected() {
        assert!(DenseProjection::new([0.0; 3], [0.0; 3], 0.0, [0.5, 0.5]).is_err());
        assert!(DenseProjection::new([f32::NAN, 0.0, 0.0], [0.0; 3], 1.0, [0.5, 0.5]).is_err());
    }

    #[test]
    fn non_finite_or_non_positive_configs_are_rejected_by_constructors() {
        for delta in [0.0f32, -0.01, f32::NAN, f32::INFINITY] {
            assert!(
                DenseReprojectionConfig::new(delta).is_err(),
                "robust_delta {delta} must be rejected"
            );
        }
        assert!(DenseReprojectionConfig::new(0.02).is_ok());

        assert!(RigidRecoveryConfig::new(0, 0.02, 1.0e-10, 1.0e-4).is_err());
        assert!(
            RigidRecoveryConfig::new(40, f32::NAN, 1.0e-10, 1.0e-4).is_err(),
            "NaN robust_delta must be rejected"
        );
        assert!(
            RigidRecoveryConfig::new(40, 0.02, -1.0, 1.0e-4).is_err(),
            "negative convergence tolerance must be rejected"
        );
        assert!(
            RigidRecoveryConfig::new(40, 0.02, f64::NAN, 1.0e-4).is_err(),
            "NaN convergence tolerance must be rejected"
        );
        assert!(RigidRecoveryConfig::new(40, 0.02, 1.0e-10, 0.0).is_err());
        assert!(RigidRecoveryConfig::new(40, 0.02, 1.0e-10, f64::INFINITY).is_err());
        assert!(RigidRecoveryConfig::new(40, 0.02, 1.0e-10, 1.0e-4).is_ok());
    }

    #[test]
    fn evaluators_fail_closed_on_invalid_config_values_at_the_boundary() {
        let model = spread_model(12);
        let mapping = mapping_for(&model, 12);
        let identity = model.neutral_identity();
        let expression = model.neutral_expression();
        let joints = crate::GnmJointState::neutral(model.joint_count());
        let truth = truth_projection();
        let observation = synthesize_observation_from_projection(
            &model,
            &identity,
            &expression,
            &joints,
            &mapping,
            &truth,
            SynthesisOptions::default(),
            DenseCoveragePolicy::new(1, 0.5).unwrap(),
            |_, _| false,
        )
        .unwrap();

        // A struct literal can carry values the constructors would reject;
        // the public evaluator and solver boundaries must still fail closed
        // instead of returning NaN weights or RMS as success.
        let evaluation_error = evaluate_dense_reprojection(
            &model,
            &identity,
            &expression,
            &joints,
            &mapping,
            &observation,
            &truth,
            DenseReprojectionConfig {
                robust_delta: f32::NAN,
            },
        )
        .unwrap_err();
        assert!(matches!(
            evaluation_error,
            GnmReprojectionError::InvalidConfig(_)
        ));

        let recovery_error = recover_rigid_projection(
            &model,
            &identity,
            &expression,
            &joints,
            &mapping,
            &observation,
            truth,
            RigidRecoveryConfig {
                max_iterations: 0,
                ..RigidRecoveryConfig::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            recovery_error,
            GnmReprojectionError::InvalidConfig(_)
        ));

        let negative_delta_recovery = recover_rigid_projection(
            &model,
            &identity,
            &expression,
            &joints,
            &mapping,
            &observation,
            truth,
            RigidRecoveryConfig {
                robust_delta: -0.02,
                ..RigidRecoveryConfig::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            negative_delta_recovery,
            GnmReprojectionError::InvalidConfig(_)
        ));
    }

    #[test]
    fn neutral_exact_observation_recovers_known_pose() {
        let model = spread_model(120);
        let mapping = mapping_for(&model, 120);
        let truth = truth_projection();
        let identity = model.neutral_identity();
        let expression = model.neutral_expression();
        let joints = crate::GnmJointState::neutral(model.joint_count());

        let observation = synthesize_observation_from_projection(
            &model,
            &identity,
            &expression,
            &joints,
            &mapping,
            &truth,
            SynthesisOptions::default(),
            DenseCoveragePolicy::new(10, 0.5).unwrap(),
            |_, _| false,
        )
        .unwrap();
        assert_eq!(observation.points().len(), 120);

        let outcome = recover_rigid_projection(
            &model,
            &identity,
            &expression,
            &joints,
            &mapping,
            &observation,
            perturbed_guess(),
            RigidRecoveryConfig::default(),
        )
        .unwrap();
        assert!(outcome.final_report.weighted_rms() < 1.0e-4);
        let recovered = outcome.projection;
        for axis in 0..3 {
            assert!(
                wrapped_angle(recovered.yaw_pitch_roll()[axis] - truth.yaw_pitch_roll()[axis])
                    .abs()
                    < 1.0e-3
            );
            assert!((recovered.translation()[axis] - truth.translation()[axis]).abs() < 1.0e-3);
        }
        assert!((recovered.focal() - truth.focal()).abs() < 1.0e-3);
    }

    #[test]
    fn outliers_are_downweighted_not_embraced() {
        let model = spread_model(80);
        let mapping = mapping_for(&model, 80);
        let truth = truth_projection();
        let identity = model.neutral_identity();
        let expression = model.neutral_expression();
        let joints = crate::GnmJointState::neutral(model.joint_count());
        // Inject one gross outlier: every point is projected exactly, then the
        // slot for row 7 is moved far away from its projection.
        let mut shifted = vec![[f32::NAN; 2]; MEDIAPIPE_FACE_LANDMARK_COUNT];
        let mut surface = GnmSparseVertices::with_len(mapping.len());
        mapping
            .evaluate_surface(&model, &identity, &expression, &joints, &mut surface)
            .unwrap();
        for (row_index, row) in mapping.rows().iter().enumerate() {
            let Some(projected) = truth.project(surface.values()[row_index]) else {
                continue;
            };
            shifted[row.mediapipe_index] = projected;
        }
        shifted[mapping.rows()[7].mediapipe_index] = [0.99, 0.01];
        let observation = GnmDenseObservation::from_mediapipe_xy(
            1,
            1_000,
            &shifted,
            &mapping,
            DenseCoveragePolicy::new(10, 0.5).unwrap(),
        )
        .unwrap();

        let report = evaluate_dense_reprojection(
            &model,
            &identity,
            &expression,
            &joints,
            &mapping,
            &observation,
            &truth,
            DenseReprojectionConfig::default(),
        )
        .unwrap();
        let outlier = report
            .residuals()
            .iter()
            .find(|residual| residual.mapping_index == 7)
            .unwrap();
        assert!(outlier.huber_weight < 0.2);

        // Recovery must ignore the outlier and still find the truth.
        let config = RigidRecoveryConfig::default();
        let outcome = recover_rigid_projection(
            &model,
            &identity,
            &expression,
            &joints,
            &mapping,
            &observation,
            perturbed_guess(),
            config,
        )
        .unwrap();
        for axis in 0..3 {
            assert!(
                wrapped_angle(
                    outcome.projection.yaw_pitch_roll()[axis] - truth.yaw_pitch_roll()[axis]
                )
                .abs()
                    < 5.0e-3
            );
        }
        // The rejected outlier still contributes its Huber-capped linear
        // share (≈ δ·|r|/N ≈ 0.011 here), so the aggregate RMS floor is the
        // robust scale, not zero. Pose accuracy above is the real criterion.
        assert!(outcome.final_report.weighted_rms() < config.robust_delta);
    }

    #[test]
    fn projection_failures_are_counted_as_excluded_points() {
        let model = spread_model(40);
        let mapping = mapping_for(&model, 40);
        let identity = model.neutral_identity();
        let expression = model.neutral_expression();
        let joints = crate::GnmJointState::neutral(model.joint_count());

        // Observation captured with every point safely in front of the camera.
        let truth = DenseProjection::new([0.0; 3], [0.0, 0.0, 0.60], 1.3, [0.5, 0.5]).unwrap();
        let observation = synthesize_observation_from_projection(
            &model,
            &identity,
            &expression,
            &joints,
            &mapping,
            &truth,
            SynthesisOptions::default(),
            DenseCoveragePolicy::new(10, 0.5).unwrap(),
            |_, _| false,
        )
        .unwrap();
        assert_eq!(observation.points().len(), 40);

        // Evaluating with a camera pushed into the cloud places the deepest
        // points at or behind the principal plane; they must be excluded by
        // count rather than poisoning the retained residuals.
        let near = DenseProjection::new([0.0; 3], [0.0, 0.0, 0.045], 1.3, [0.5, 0.5]).unwrap();
        let report = evaluate_dense_reprojection(
            &model,
            &identity,
            &expression,
            &joints,
            &mapping,
            &observation,
            &near,
            DenseReprojectionConfig::default(),
        )
        .unwrap();
        assert!(report.excluded_points() > 0);
        assert!(report.residuals().len() + report.excluded_points() == observation.points().len());
    }

    #[test]
    fn fully_behind_camera_evaluation_is_a_typed_error() {
        let model = spread_model(24);
        let mapping = mapping_for(&model, 24);
        let identity = model.neutral_identity();
        let expression = model.neutral_expression();
        let joints = crate::GnmJointState::neutral(model.joint_count());

        let truth = DenseProjection::new([0.0; 3], [0.0, 0.0, 0.60], 1.3, [0.5, 0.5]).unwrap();
        let observation = synthesize_observation_from_projection(
            &model,
            &identity,
            &expression,
            &joints,
            &mapping,
            &truth,
            SynthesisOptions::default(),
            DenseCoveragePolicy::new(10, 0.5).unwrap(),
            |_, _| false,
        )
        .unwrap();

        // A yaw of π flips every cloud depth across the principal plane, and
        // the pulled-back translation keeps every flipped depth negative.
        let flipped = DenseProjection::new(
            [std::f32::consts::PI, 0.0, 0.0],
            [0.0, 0.0, -0.06],
            1.3,
            [0.5, 0.5],
        )
        .unwrap();
        let error = evaluate_dense_reprojection(
            &model,
            &identity,
            &expression,
            &joints,
            &mapping,
            &observation,
            &flipped,
            DenseReprojectionConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            GnmReprojectionError::InsufficientObservation
        ));
    }

    #[test]
    fn fully_invalid_observation_is_a_typed_error() {
        let model = spread_model(8);
        let mapping = mapping_for(&model, 8);
        let identity = model.neutral_identity();
        let expression = model.neutral_expression();
        let joints = crate::GnmJointState::neutral(model.joint_count());
        let observation = synthesize_observation_from_projection(
            &model,
            &identity,
            &expression,
            &joints,
            &mapping,
            &truth_projection(),
            SynthesisOptions::default(),
            DenseCoveragePolicy::new(1, 0.5).unwrap(),
            |_, _| true,
        )
        .unwrap();
        assert_eq!(observation.points().len(), 0);
        let error = evaluate_dense_reprojection(
            &model,
            &identity,
            &expression,
            &joints,
            &mapping,
            &observation,
            &truth_projection(),
            DenseReprojectionConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            GnmReprojectionError::InsufficientObservation
        ));
    }

    #[test]
    fn conditioning_comparison_ranks_dense_over_sparse() {
        let model = spread_model(200);
        let dense_mapping = mapping_for(&model, 200);
        let sparse_mapping = dense_mapping
            .filter_rows(&model, |row| row.mediapipe_index % 10 == 0)
            .unwrap();
        assert!(sparse_mapping.len() >= SPARSE_BOOTSTRAP_POINT_COUNT / 4);

        let identity = model.neutral_identity();
        let expression = model.neutral_expression();
        let joints = crate::GnmJointState::neutral(model.joint_count());
        let truth = truth_projection();
        let guess = perturbed_guess();

        let dense_observation = synthesize_observation_from_projection(
            &model,
            &identity,
            &expression,
            &joints,
            &dense_mapping,
            &truth,
            SynthesisOptions {
                noise_amplitude: 0.004,
                noise_seed: 11,
                ..SynthesisOptions::default()
            },
            DenseCoveragePolicy::new(10, 0.5).unwrap(),
            |_, _| false,
        )
        .unwrap();
        let sparse_observation = synthesize_observation_from_projection(
            &model,
            &identity,
            &expression,
            &joints,
            &sparse_mapping,
            &truth,
            SynthesisOptions {
                noise_amplitude: 0.004,
                noise_seed: 11,
                ..SynthesisOptions::default()
            },
            DenseCoveragePolicy::new(5, 0.5).unwrap(),
            |_, _| false,
        )
        .unwrap();

        let stats = compare_conditioning(
            &model,
            &identity,
            &expression,
            &joints,
            &[
                ConditioningBaseline {
                    label: "sparse",
                    mapping: &sparse_mapping,
                    observation: &sparse_observation,
                    initial_guess: guess,
                },
                ConditioningBaseline {
                    label: "dense",
                    mapping: &dense_mapping,
                    observation: &dense_observation,
                    initial_guess: guess,
                },
            ],
            &truth,
            RigidRecoveryConfig::default(),
        )
        .unwrap();

        assert_eq!(stats[0].label, "sparse");
        assert_eq!(stats[1].label, "dense");
        // Dense must retain more evidence and recover the pose at least as
        // precisely as the sparse baseline under identical noise.
        assert!(stats[1].valid_points > stats[0].valid_points * 5);
        let dense_rotation: f32 = stats[1]
            .rotation_error
            .iter()
            .map(|error| error.abs())
            .sum();
        let sparse_rotation: f32 = stats[0]
            .rotation_error
            .iter()
            .map(|error| error.abs())
            .sum();
        assert!(
            dense_rotation <= sparse_rotation + 1.0e-4,
            "dense rotation error {dense_rotation} exceeded sparse {sparse_rotation}"
        );
    }

    // -- Dense reprojection linearization (Issue #64.2a / #118) fixtures ------

    /// Model with a real expression channel: channel 0 translates every
    /// vertex by `[0.05, -0.03, 0.01]`, so the expression block has
    /// non-trivial surface derivatives.
    fn lin_model() -> GnmModel {
        let vertex_count = 64;
        let identity = GNM_HEAD_V3_IDENTITY_DIM;
        let expression = GNM_HEAD_V3_EXPRESSION_DIM;
        let mut vertices = Vec::with_capacity(vertex_count * 3);
        for index in 0..vertex_count {
            let angle = (index as f32) / (vertex_count as f32) * std::f32::consts::TAU;
            vertices.extend_from_slice(&[
                0.10 * angle.cos(),
                0.12 * angle.sin(),
                0.05 * (3.0 * angle).sin(),
            ]);
        }
        let mut expression_basis = vec![0.0f32; expression * vertex_count * 3];
        for vertex in 0..vertex_count {
            let base = vertex * 3;
            expression_basis[base] = 0.05;
            expression_basis[base + 1] = -0.03;
            expression_basis[base + 2] = 0.01;
        }
        GnmModel::from_data(GnmModelData {
            version: GNM_HEAD_V3_VERSION,
            variant: GnmVariant::Head,
            template_vertices: DenseArray::new("vertices", vec![vertex_count, 3], vertices)
                .unwrap(),
            template_joints: DenseArray::new("joints", vec![1, 3], vec![0.0; 3]).unwrap(),
            vertex_identity_basis: DenseArray::new(
                "identity",
                vec![identity, vertex_count, 3],
                vec![0.0; identity * vertex_count * 3],
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
                vec![expression, vertex_count, 3],
                expression_basis,
            )
            .unwrap(),
            joint_parent_indices: vec![-1],
            skinning_weights: DenseArray::new(
                "weights",
                vec![1, vertex_count],
                vec![1.0; vertex_count],
            )
            .unwrap(),
            pose_correctives_regressor: None,
        })
        .unwrap()
    }

    fn lin_expression() -> GnmExpressionState {
        let mut values = vec![0.0; GNM_HEAD_V3_EXPRESSION_DIM];
        values[0] = 1.2;
        GnmExpressionState::new(values, GNM_HEAD_V3_EXPRESSION_DIM).unwrap()
    }

    fn lin_observation(model: &GnmModel, mapping: &DenseCorrespondenceSet) -> GnmDenseObservation {
        synthesize_observation_from_projection(
            model,
            &model.neutral_identity(),
            &lin_expression(),
            &GnmJointState::neutral(model.joint_count()),
            mapping,
            &truth_projection(),
            SynthesisOptions {
                source_seq: 7,
                captured_at_micros: 1_000,
                noise_amplitude: 0.0,
                ..SynthesisOptions::default()
            },
            DenseCoveragePolicy::new(2, 0.75).unwrap(),
            |_, _| false,
        )
        .unwrap()
    }

    #[test]
    fn residual_at_the_known_projection_is_near_zero() {
        let model = lin_model();
        let mapping = mapping_for(&model, 64);
        let observation = lin_observation(&model, &mapping);
        let linearization = linearize_dense_reprojection(
            &model,
            &model.neutral_identity(),
            &lin_expression(),
            &GnmJointState::neutral(model.joint_count()),
            &mapping,
            &observation,
            &truth_projection(),
            DenseReprojectionConfig::default(),
            LinearizationStepSizes::default(),
        )
        .unwrap();
        assert!(
            linearization.report().weighted_rms() < 1.0e-5,
            "rms {}",
            linearization.report().weighted_rms()
        );
    }

    #[test]
    fn blocks_preserve_boundaries_and_exclude_identity() {
        let model = lin_model();
        let mapping = mapping_for(&model, 64);
        let observation = lin_observation(&model, &mapping);
        let linearization = linearize_dense_reprojection(
            &model,
            &model.neutral_identity(),
            &lin_expression(),
            &GnmJointState::neutral(model.joint_count()),
            &mapping,
            &observation,
            &perturbed_guess(),
            DenseReprojectionConfig::default(),
            LinearizationStepSizes::default(),
        )
        .unwrap();

        let rows = 2 * linearization.report().residuals().len();
        let expected = [
            (ReprojectionBlock::Expression, GNM_HEAD_V3_EXPRESSION_DIM),
            (ReprojectionBlock::Joints, 3 * (model.joint_count() + 1)),
            (ReprojectionBlock::RigidPose, 3),
            (ReprojectionBlock::CameraTranslation, 3),
        ];
        assert_eq!(linearization.blocks().len(), expected.len());
        for (block, (expected_block, expected_parameters)) in
            linearization.blocks().iter().zip(expected)
        {
            assert_eq!(block.block, expected_block);
            assert_eq!(block.parameter_count, expected_parameters);
            assert_eq!(block.row_count, rows);
            assert_eq!(block.entries.len(), rows * expected_parameters);
        }
        // Identity is calibration evidence and must never appear as a block.
        assert!(
            !format!(
                "{:?}",
                linearization
                    .blocks()
                    .iter()
                    .map(|block| block.block.name())
                    .collect::<Vec<_>>()
            )
            .contains("identity")
        );
    }

    #[test]
    fn rigid_pose_jacobian_matches_central_differences() {
        let model = lin_model();
        let mapping = mapping_for(&model, 64);
        let observation = lin_observation(&model, &mapping);
        let guess = perturbed_guess();
        let linearization = linearize_dense_reprojection(
            &model,
            &model.neutral_identity(),
            &lin_expression(),
            &GnmJointState::neutral(model.joint_count()),
            &mapping,
            &observation,
            &guess,
            DenseReprojectionConfig::default(),
            LinearizationStepSizes::default(),
        )
        .unwrap();

        // Central difference of the projected u of the first retained point
        // with respect to yaw, evaluated independently of the adapter.
        let mut surface = GnmSparseVertices::with_len(mapping.len());
        mapping
            .evaluate_surface(
                &model,
                &model.neutral_identity(),
                &lin_expression(),
                &GnmJointState::neutral(model.joint_count()),
                &mut surface,
            )
            .unwrap();
        let point = surface.values()[linearization.report().residuals()[0].mapping_index];
        // f32 coordinates limit cancellation: keep h well above the ulp of
        // the projected coordinate while staying in the linear regime.
        let h = 1.0e-4;
        let plus = DenseProjection::new(
            [
                guess.yaw_pitch_roll()[0] + h,
                guess.yaw_pitch_roll()[1],
                guess.yaw_pitch_roll()[2],
            ],
            guess.translation(),
            guess.focal(),
            guess.principal_point(),
        )
        .unwrap()
        .project(point)
        .unwrap();
        let minus = DenseProjection::new(
            [
                guess.yaw_pitch_roll()[0] - h,
                guess.yaw_pitch_roll()[1],
                guess.yaw_pitch_roll()[2],
            ],
            guess.translation(),
            guess.focal(),
            guess.principal_point(),
        )
        .unwrap()
        .project(point)
        .unwrap();
        let central = (plus[0] - minus[0]) / (2.0 * h);
        // Residual is observed - projected, so the residual derivative is the
        // negated projection derivative.
        let forward = linearization
            .block(ReprojectionBlock::RigidPose)
            .unwrap()
            .get(0, 0)
            .unwrap();
        let scale = central.abs().max(1.0e-6);
        assert!(
            ((forward + central).abs()) / scale < 1.0e-2,
            "forward {forward} vs central {central}"
        );
    }

    // -- Bounded rigid pose + camera translation step (Issue #64.2b / #119) ----

    #[test]
    fn rigid_steps_recover_known_yaw_pitch_and_translation() {
        let model = lin_model();
        let mapping = mapping_for(&model, 64);
        let observation = lin_observation(&model, &mapping);
        let config = DenseRigidStepConfig::default();

        // Iterate bounded single steps from a perturbed guess toward truth.
        let mut projection = perturbed_guess();
        for _ in 0..40 {
            let outcome = take_dense_rigid_step(
                &model,
                &model.neutral_identity(),
                &lin_expression(),
                &GnmJointState::neutral(model.joint_count()),
                &mapping,
                &observation,
                &projection,
                config,
            )
            .unwrap();
            if !outcome.accepted {
                break;
            }
            projection =
                DenseProjection::new(outcome.yaw_pitch_roll, outcome.translation, 1.3, [0.5, 0.5])
                    .unwrap();
        }

        let truth = truth_projection();
        for (actual, expected) in projection
            .yaw_pitch_roll()
            .into_iter()
            .zip(truth.yaw_pitch_roll())
        {
            assert!((actual - expected).abs() < 5.0e-3, "{actual} vs {expected}");
        }
        for (actual, expected) in projection
            .translation()
            .into_iter()
            .zip(truth.translation())
        {
            assert!((actual - expected).abs() < 5.0e-3, "{actual} vs {expected}");
        }
    }

    #[test]
    fn expression_only_deformation_does_not_move_the_rigid_block() {
        let model = lin_model();
        let mapping = mapping_for(&model, 64);
        // Observation synthesized at the SAME projection as the guess but with
        // an active expression channel: all residual is expression evidence.
        let observation = synthesize_observation_from_projection(
            &model,
            &model.neutral_identity(),
            &lin_expression(),
            &GnmJointState::neutral(model.joint_count()),
            &mapping,
            &truth_projection(),
            SynthesisOptions::default(),
            DenseCoveragePolicy::new(2, 0.75).unwrap(),
            |_, _| false,
        )
        .unwrap();

        let outcome = take_dense_rigid_step(
            &model,
            &model.neutral_identity(),
            // The step is told the true expression state, so no residual
            // should be attributable to pose.
            &lin_expression(),
            &GnmJointState::neutral(model.joint_count()),
            &mapping,
            &observation,
            &truth_projection(),
            DenseRigidStepConfig::default(),
        )
        .unwrap();
        // With zero residual there is nothing to fit; any candidate cannot
        // decrease it, so the step must be rejected and state unchanged.
        assert!(!outcome.accepted);
        assert_eq!(outcome.yaw_pitch_roll, truth_projection().yaw_pitch_roll());
        assert_eq!(outcome.translation, truth_projection().translation());
        assert!(outcome.residual_before.abs() < 1.0e-5);
    }

    #[test]
    fn clamped_step_and_invalid_config_are_handled() {
        assert!(DenseRigidStepConfig::new(0.0, 0.1, 0.1).is_err());
        assert!(DenseRigidStepConfig::new(f32::NAN, 0.1, 0.1).is_err());

        let model = lin_model();
        let mapping = mapping_for(&model, 64);
        let observation = lin_observation(&model, &mapping);
        // A tiny cap forces clamping; the accepted step must respect it.
        let config = DenseRigidStepConfig::new(1.0e-3, 1.0e-3, 1.0e-3).unwrap();
        let outcome = take_dense_rigid_step(
            &model,
            &model.neutral_identity(),
            &lin_expression(),
            &GnmJointState::neutral(model.joint_count()),
            &mapping,
            &observation,
            &perturbed_guess(),
            config,
        )
        .unwrap();
        assert!(outcome.pose_step_norm <= config.max_pose_step + 1.0e-9);
        assert!(outcome.translation_step_norm <= config.max_translation_step + 1.0e-9);
    }

    // -- Bounded expression + joint step (Issue #64.2c / #120) fixtures --------

    /// Model with two localized expression channels: channel 0 ("mouth")
    /// lowers the even-indexed vertices, channel 1 ("eyelid") moves the
    /// odd-indexed vertices.
    fn ej_model() -> GnmModel {
        let vertex_count = 64;
        let identity = GNM_HEAD_V3_IDENTITY_DIM;
        let expression = GNM_HEAD_V3_EXPRESSION_DIM;
        let mut vertices = Vec::with_capacity(vertex_count * 3);
        for index in 0..vertex_count {
            let angle = (index as f32) / (vertex_count as f32) * std::f32::consts::TAU;
            vertices.extend_from_slice(&[
                0.10 * angle.cos(),
                0.12 * angle.sin(),
                0.05 * (3.0 * angle).sin(),
            ]);
        }
        let mut expression_basis = vec![0.0f32; expression * vertex_count * 3];
        let eyelid_offset = vertex_count * 3;
        for vertex in 0..vertex_count {
            let base = vertex * 3;
            if vertex % 2 == 0 {
                // Channel 0 ("mouth"): vertical motion.
                #[allow(clippy::indexing_slicing)]
                {
                    expression_basis[base + 1] = -0.04;
                }
            } else {
                // Channel 1 ("eyelid"): lateral motion, image-distinguishable
                // from the vertical channel.
                #[allow(clippy::indexing_slicing)]
                {
                    expression_basis[eyelid_offset + base] = 0.03;
                }
            }
        }
        GnmModel::from_data(GnmModelData {
            version: GNM_HEAD_V3_VERSION,
            variant: GnmVariant::Head,
            template_vertices: DenseArray::new("vertices", vec![vertex_count, 3], vertices)
                .unwrap(),
            template_joints: DenseArray::new("joints", vec![1, 3], vec![0.0; 3]).unwrap(),
            vertex_identity_basis: DenseArray::new(
                "identity",
                vec![identity, vertex_count, 3],
                vec![0.0; identity * vertex_count * 3],
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
                vec![expression, vertex_count, 3],
                expression_basis,
            )
            .unwrap(),
            joint_parent_indices: vec![-1],
            skinning_weights: DenseArray::new(
                "weights",
                vec![1, vertex_count],
                vec![1.0; vertex_count],
            )
            .unwrap(),
            pose_correctives_regressor: None,
        })
        .unwrap()
    }

    fn ej_expression(mouth: f32, eyelid: f32) -> GnmExpressionState {
        let mut values = vec![0.0; GNM_HEAD_V3_EXPRESSION_DIM];
        values[0] = mouth;
        values[1] = eyelid;
        GnmExpressionState::new(values, GNM_HEAD_V3_EXPRESSION_DIM).unwrap()
    }

    /// Phase-timing harness for the #148 follow-up (see issue: finite-
    /// difference Jacobian construction). Ignored by default because it is a
    /// measurement, not an assertion; run explicitly with
    /// `cargo test -p vtuber-gnm --lib -- --ignored --nocapture
    /// phase_timing`.
    /// Issue #161 acceptance: the analytic expression-block Jacobian must
    /// match the legacy forward-difference construction column-for-column.
    /// The fixture exercises rotated skinning, mixed vertex/barycentric
    /// mapping targets, and all-zero sparse basis channels.
    #[test]
    fn expression_jacobian_matches_finite_difference_parity() {
        let model = ej_model();
        let vertex_count = model.vertex_count();
        let rows: Vec<MediaPipeGnmDenseCorrespondence> = (0..24)
            .map(|index| {
                let target = if index % 2 == 0 {
                    GnmSurfacePointRef::Vertex {
                        vertex_index: index,
                    }
                } else {
                    GnmSurfacePointRef::Barycentric {
                        vertex_indices: [
                            index % vertex_count,
                            (index + 7) % vertex_count,
                            (index + 13) % vertex_count,
                        ],
                        weights: [0.2, 0.3, 0.5],
                    }
                };
                MediaPipeGnmDenseCorrespondence {
                    mediapipe_index: index,
                    target,
                    region: if index % 3 == 0 {
                        FaceRegion::Nose
                    } else if index % 3 == 1 {
                        FaceRegion::Contour
                    } else {
                        FaceRegion::Other
                    },
                    anatomical_side: AnatomicalSide::Midline,
                    base_weight: 1.0,
                    provenance: CorrespondenceProvenance::RepositoryValidated,
                    reliability: CorrespondenceReliability::High,
                }
            })
            .collect();
        let mapping = DenseCorrespondenceSet::new(version(), rows, &model).unwrap();
        let identity = model.neutral_identity();
        let expression = ej_expression(0.35, 0.25);
        // Rotated root joint so the skinning rotation participates in the
        // analytic derivative chain instead of degenerating to identity.
        let joints = GnmJointState::new(vec![[0.21, -0.13, 0.09]], [0.01, -0.02, 0.03], 1).unwrap();
        let observation = synthesize_observation_from_projection(
            &model,
            &identity,
            &expression,
            &joints,
            &mapping,
            &truth_projection(),
            SynthesisOptions::default(),
            DenseCoveragePolicy::new(2, 0.75).unwrap(),
            |_, _| false,
        )
        .unwrap();
        let config = DenseReprojectionConfig::default();
        let steps = LinearizationStepSizes::default();

        let linearization = linearize_dense_reprojection(
            &model,
            &identity,
            &expression,
            &joints,
            &mapping,
            &observation,
            &truth_projection(),
            config,
            steps,
        )
        .unwrap();
        let analytic = linearization.block(ReprojectionBlock::Expression).unwrap();

        // Rebuild the same retained-point contract the linearizer uses so the
        // finite-difference reference path sees identical rows.
        let report = evaluate_dense_reprojection(
            &model,
            &identity,
            &expression,
            &joints,
            &mapping,
            &observation,
            &truth_projection(),
            config,
        )
        .unwrap();
        let retained: Vec<usize> = report
            .residuals()
            .iter()
            .map(|residual| residual.mapping_index)
            .collect();
        assert_eq!(analytic.row_count, 2 * retained.len());
        let baseline_projected: Vec<[f32; 2]> = report
            .residuals()
            .iter()
            .map(|residual| residual.projected_xy)
            .collect();
        let mut baseline_surface = GnmSparseVertices::with_len(mapping.len());
        mapping
            .evaluate_surface(
                &model,
                &identity,
                &expression,
                &joints,
                &mut baseline_surface,
            )
            .unwrap();
        let baseline_surface = baseline_surface.values().to_vec();
        let prepared = model
            .prepare_sparse_vertices(&identity, &expression, &joints, mapping.surface_landmarks())
            .unwrap();

        let mut max_abs_diff = 0.0f32;
        let mut max_abs_entry = 0.0f32;
        for parameter in 0..analytic.parameter_count {
            let column = perturbed_residuals(
                &model,
                &identity,
                &expression,
                &joints,
                &mapping,
                &truth_projection(),
                &baseline_surface,
                &baseline_projected,
                &prepared,
                ReprojectionBlock::Expression,
                parameter,
                steps,
                &retained,
            )
            .unwrap();
            for (point_row, delta) in column.iter().enumerate() {
                for (component, expected) in delta.iter().copied().enumerate() {
                    let actual = analytic.get(2 * point_row + component, parameter).unwrap();
                    max_abs_diff = max_abs_diff.max((expected - actual).abs());
                    max_abs_entry = max_abs_entry.max(expected.abs().max(actual.abs()));
                }
            }
        }
        assert!(
            max_abs_diff < 1.0e-3,
            "expression Jacobian parity failed: max |finite_difference - analytic| \
             = {max_abs_diff} (max entry magnitude {max_abs_entry})"
        );
    }

    #[test]
    fn non_tongue_projection_jacobian_matches_central_difference() {
        let model = ej_model();
        let mapping = mapping_for(&model, 24);
        let identity = model.neutral_identity();
        let full = ej_expression(0.35, 0.25);
        let compact = GnmNonTongueExpression::try_from_full(&full).unwrap();
        let joints = GnmJointState::neutral(model.joint_count());
        let projection = truth_projection();
        let observation = synthesize_observation_from_projection(
            &model,
            &identity,
            &full,
            &joints,
            &mapping,
            &projection,
            SynthesisOptions::default(),
            DenseCoveragePolicy::new(2, 0.75).unwrap(),
            |_, _| false,
        )
        .unwrap();
        let analytic = non_tongue_projection_jacobian(
            &model,
            &identity,
            &compact,
            &joints,
            &mapping,
            &observation,
            &projection,
        )
        .unwrap();
        assert_eq!(analytic.column_count, GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM);
        assert_eq!(analytic.row_count, 2 * mapping.len());

        let step = 1.0e-3_f32;
        let mut maximum_relative_error = 0.0_f64;
        for column in [0_usize, 1] {
            let mut plus = compact.values().to_vec();
            let mut minus = compact.values().to_vec();
            plus[column] += step;
            minus[column] -= step;
            let plus = GnmNonTongueExpression::try_from_values(plus)
                .unwrap()
                .expand_with_zero_tongue()
                .unwrap();
            let minus = GnmNonTongueExpression::try_from_values(minus)
                .unwrap()
                .expand_with_zero_tongue()
                .unwrap();
            let mut plus_surface = GnmSparseVertices::with_len(mapping.len());
            let mut minus_surface = GnmSparseVertices::with_len(mapping.len());
            mapping
                .evaluate_surface(&model, &identity, &plus, &joints, &mut plus_surface)
                .unwrap();
            mapping
                .evaluate_surface(&model, &identity, &minus, &joints, &mut minus_surface)
                .unwrap();
            for point in 0..mapping.len() {
                let plus_xy = projection.project(plus_surface.values()[point]).unwrap();
                let minus_xy = projection.project(minus_surface.values()[point]).unwrap();
                for axis in 0..2 {
                    let finite_difference =
                        f64::from((plus_xy[axis] - minus_xy[axis]) / (2.0 * step));
                    let actual = analytic.values_row_major
                        [(2 * point + axis) * analytic.column_count + column];
                    let relative =
                        (actual - finite_difference).abs() / finite_difference.abs().max(1.0e-6);
                    maximum_relative_error = maximum_relative_error.max(relative);
                }
            }
        }
        assert!(maximum_relative_error < 1.0e-2, "{maximum_relative_error}");
    }

    #[test]
    fn observability_gram_accumulation_is_finite_symmetric_and_psd() {
        let jacobian = NonTongueProjectionJacobian {
            row_count: 3,
            column_count: 2,
            values_row_major: vec![1.0, 2.0, -1.0, 0.5, 0.25, -3.0],
            row_weights: vec![1.0, 2.0, 0.5],
        };
        let mut packed = vec![0.0; 3];
        accumulate_observability_gram(&jacobian, &mut packed).unwrap();
        assert!(packed.iter().all(|value| value.is_finite()));
        let gram = [[packed[0], packed[1]], [packed[1], packed[2]]];
        for vector in [[1.0, 0.0], [0.0, 1.0], [2.0, -3.0]] {
            let quadratic = vector[0] * (gram[0][0] * vector[0] + gram[0][1] * vector[1])
                + vector[1] * (gram[1][0] * vector[0] + gram[1][1] * vector[1]);
            assert!(quadratic >= 0.0);
        }
    }

    #[test]
    fn expression_jacobian_parity_with_partial_observation() {
        // Regression: with dropped mapping rows the retained point list is a
        // strict subset of the mapped points, so the analytic expression block
        // must size its skinning-derivative buffer by mapped-point count and
        // select each retained row by mapping index (not by retained position).
        let model = ej_model();
        let vertex_count = model.vertex_count();
        let rows: Vec<MediaPipeGnmDenseCorrespondence> = (0..24)
            .map(|index| MediaPipeGnmDenseCorrespondence {
                mediapipe_index: index,
                target: GnmSurfacePointRef::Vertex {
                    vertex_index: index % vertex_count,
                },
                region: if index % 3 == 0 {
                    FaceRegion::Nose
                } else if index % 3 == 1 {
                    FaceRegion::Contour
                } else {
                    FaceRegion::Other
                },
                anatomical_side: AnatomicalSide::Midline,
                base_weight: 1.0,
                provenance: CorrespondenceProvenance::RepositoryValidated,
                reliability: CorrespondenceReliability::High,
            })
            .collect();
        let mapping = DenseCorrespondenceSet::new(version(), rows, &model).unwrap();
        let identity = model.neutral_identity();
        let expression = ej_expression(0.35, 0.25);
        let joints = GnmJointState::new(vec![[0.21, -0.13, 0.09]], [0.01, -0.02, 0.03], 1).unwrap();
        let observation = synthesize_observation_from_projection(
            &model,
            &identity,
            &expression,
            &joints,
            &mapping,
            &truth_projection(),
            SynthesisOptions::default(),
            DenseCoveragePolicy::new(2, 0.75).unwrap(),
            |row_index, _| row_index % 5 == 0,
        )
        .unwrap();
        let config = DenseReprojectionConfig::default();
        let steps = LinearizationStepSizes::default();

        let linearization = linearize_dense_reprojection(
            &model,
            &identity,
            &expression,
            &joints,
            &mapping,
            &observation,
            &truth_projection(),
            config,
            steps,
        )
        .unwrap();
        let analytic = linearization.block(ReprojectionBlock::Expression).unwrap();

        let report = evaluate_dense_reprojection(
            &model,
            &identity,
            &expression,
            &joints,
            &mapping,
            &observation,
            &truth_projection(),
            config,
        )
        .unwrap();
        let retained: Vec<usize> = report
            .residuals()
            .iter()
            .map(|residual| residual.mapping_index)
            .collect();
        assert!(
            retained.len() < mapping.len(),
            "observation must retain a strict subset of the mapped points"
        );
        assert_eq!(analytic.row_count, 2 * retained.len());
        let baseline_projected: Vec<[f32; 2]> = report
            .residuals()
            .iter()
            .map(|residual| residual.projected_xy)
            .collect();
        let mut baseline_surface = GnmSparseVertices::with_len(mapping.len());
        mapping
            .evaluate_surface(
                &model,
                &identity,
                &expression,
                &joints,
                &mut baseline_surface,
            )
            .unwrap();
        let baseline_surface = baseline_surface.values().to_vec();
        let prepared = model
            .prepare_sparse_vertices(&identity, &expression, &joints, mapping.surface_landmarks())
            .unwrap();

        let mut max_abs_diff = 0.0f32;
        let mut max_abs_entry = 0.0f32;
        for parameter in 0..analytic.parameter_count {
            let column = perturbed_residuals(
                &model,
                &identity,
                &expression,
                &joints,
                &mapping,
                &truth_projection(),
                &baseline_surface,
                &baseline_projected,
                &prepared,
                ReprojectionBlock::Expression,
                parameter,
                steps,
                &retained,
            )
            .unwrap();
            for (point_row, delta) in column.iter().enumerate() {
                for (component, expected) in delta.iter().copied().enumerate() {
                    let actual = analytic.get(2 * point_row + component, parameter).unwrap();
                    max_abs_diff = max_abs_diff.max((expected - actual).abs());
                    max_abs_entry = max_abs_entry.max(expected.abs().max(actual.abs()));
                }
            }
        }
        assert!(
            max_abs_diff < 1.0e-3,
            "partial-observation expression Jacobian parity failed: \
             max |finite_difference - analytic| = {max_abs_diff} \
             (max entry magnitude {max_abs_entry})"
        );
    }

    #[test]
    #[ignore]
    fn phase_timing_report_issue148() {
        let model = ej_model();
        let mapping = mapping_for(&model, 64);
        let identity = model.neutral_identity();
        let expression = ej_expression(0.7, 0.4);
        let joints = GnmJointState::neutral(model.joint_count());
        let observation = synthesize_observation_from_projection(
            &model,
            &identity,
            &expression,
            &joints,
            &mapping,
            &truth_projection(),
            SynthesisOptions::default(),
            DenseCoveragePolicy::new(2, 0.75).unwrap(),
            |_, _| false,
        )
        .unwrap();
        let projection = perturbed_guess();
        let config = DenseExpressionJointStepConfig::default();
        let iterations = 20;

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let linearization = linearize_dense_reprojection(
                &model,
                &identity,
                &expression,
                &joints,
                &mapping,
                &observation,
                &projection,
                DenseReprojectionConfig::default(),
                LinearizationStepSizes::default(),
            )
            .unwrap();
            std::hint::black_box(&linearization);
        }
        let linearize_per_call = start.elapsed() / iterations;

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let outcome = take_dense_expression_joint_step(
                &model,
                &identity,
                &expression,
                &joints,
                &mapping,
                &observation,
                &projection,
                config,
                None,
            )
            .unwrap();
            std::hint::black_box(&outcome);
        }
        let step_per_call = start.elapsed() / iterations;

        let joint_parameters = 3 * (model.joint_count() + 1);
        let total = GNM_HEAD_V3_EXPRESSION_DIM + joint_parameters;
        // Packed lower triangle with the same entries the symmetric fixture
        // matrix used, so the factorization cost matches the step's solver.
        let matrix: Vec<f64> = (0..total)
            .flat_map(|i| {
                (0..=i).map(move |j| {
                    if i == j {
                        1_000.0 + ((i * 7) % 13) as f64
                    } else {
                        (((i * j) % 7) as f64) * 1.0e-3
                    }
                })
            })
            .collect();
        let rhs: Vec<f64> = (0..total).map(|i| 1.0 + i as f64).collect();
        let solver_iterations = 50;
        let start = std::time::Instant::now();
        for _ in 0..solver_iterations {
            let mut a = matrix.clone();
            let mut b = rhs.clone();
            let x = solve_spd_packed_lower(&mut a, &mut b).unwrap();
            std::hint::black_box(&x);
        }
        let solve_per_call = start.elapsed() / solver_iterations;

        println!(
            "== phase timing (mapping rows: {}, K+J: {}) ==",
            mapping.len(),
            total
        );
        println!("linearize_dense_reprojection : {:?}", linearize_per_call);
        println!("take_dense_expression_joint  : {:?}", step_per_call);
        println!("solve_spd_packed_lower only  : {:?}", solve_per_call);
        println!(
            "assembly+temporal+aux share  : {:?}",
            step_per_call.saturating_sub(linearize_per_call)
        );
        // Baseline surface evaluation alone, to separate the K perturbed
        // re-evaluations from the shared baseline inside linearize.
        let start = std::time::Instant::now();
        for _ in 0..(iterations * 4) {
            let mut surface = GnmSparseVertices::with_len(mapping.len());
            mapping
                .evaluate_surface(&model, &identity, &expression, &joints, &mut surface)
                .unwrap();
            std::hint::black_box(&surface);
        }
        let single_surface_eval = start.elapsed() / (iterations * 4);

        let expression_params = GNM_HEAD_V3_EXPRESSION_DIM;
        println!("single evaluate_surface      : {:?}", single_surface_eval);
        println!(
            "per-parameter column         : {:?} (x{})",
            linearize_per_call / u32::try_from(expression_params).unwrap(),
            expression_params
        );
        println!(
            "K re-evaluations / one eval  : {:.1}",
            linearize_per_call.as_secs_f64() / single_surface_eval.as_secs_f64()
        );
    }

    #[test]
    fn mouth_eyelid_and_mixed_states_recover_from_zero() {
        let model = ej_model();
        let mapping = mapping_for(&model, 64);
        let joints = GnmJointState::neutral(model.joint_count());
        let config = DenseExpressionJointStepConfig::default();

        for (mouth, eyelid) in [(0.8, 0.0), (0.0, 0.6), (0.7, 0.5)] {
            let truth = ej_expression(mouth, eyelid);
            let observation = synthesize_observation_from_projection(
                &model,
                &model.neutral_identity(),
                &truth,
                &joints,
                &mapping,
                &truth_projection(),
                SynthesisOptions::default(),
                DenseCoveragePolicy::new(2, 0.75).unwrap(),
                |_, _| false,
            )
            .unwrap();

            let mut expression = model.neutral_expression();
            for _iteration in 0..60 {
                let outcome = take_dense_expression_joint_step(
                    &model,
                    &model.neutral_identity(),
                    &expression,
                    &joints,
                    &mapping,
                    &observation,
                    &truth_projection(),
                    config,
                    None,
                )
                .unwrap();
                if !outcome.accepted {
                    break;
                }
                expression = outcome.expression;
            }
            assert!(
                (expression.values()[0] - mouth).abs() < 5.0e-2,
                "mouth {mouth}"
            );
            assert!(
                (expression.values()[1] - eyelid).abs() < 5.0e-2,
                "eyelid {eyelid}"
            );
        }
    }

    #[test]
    fn rigid_only_motion_does_not_move_the_expression_block() {
        let model = ej_model();
        let mapping = mapping_for(&model, 64);
        let neutral = model.neutral_expression();
        let joints = GnmJointState::neutral(model.joint_count());
        // Observation synthesized at a DIFFERENT projection with the same
        // neutral expression: all residual is rigid evidence, and the step is
        // told the true (neutral) expression state.
        let observation = synthesize_observation_from_projection(
            &model,
            &model.neutral_identity(),
            &neutral,
            &joints,
            &mapping,
            &perturbed_guess(),
            SynthesisOptions::default(),
            DenseCoveragePolicy::new(2, 0.75).unwrap(),
            |_, _| false,
        )
        .unwrap();

        let outcome = take_dense_expression_joint_step(
            &model,
            &model.neutral_identity(),
            &neutral,
            &joints,
            &mapping,
            &observation,
            &perturbed_guess(),
            DenseExpressionJointStepConfig::default(),
            None,
        )
        .unwrap();
        assert_eq!(outcome.expression, neutral, "expression must not move");
    }

    #[test]
    fn clamped_update_bounds_are_respected_and_config_fails_closed() {
        assert!(DenseExpressionJointStepConfig::new(-1.0, 0.1, 0.1, 0.1, 0.1).is_err());
        assert!(DenseExpressionJointStepConfig::new(0.1, 0.1, 0.1, 0.0, 0.1).is_err());

        let model = ej_model();
        let mapping = mapping_for(&model, 64);
        let observation = synthesize_observation_from_projection(
            &model,
            &model.neutral_identity(),
            &ej_expression(0.9, 0.9),
            &GnmJointState::neutral(model.joint_count()),
            &mapping,
            &truth_projection(),
            SynthesisOptions::default(),
            DenseCoveragePolicy::new(2, 0.75).unwrap(),
            |_, _| false,
        )
        .unwrap();
        let config =
            DenseExpressionJointStepConfig::new(1.0e-2, 1.0e-3, 1.0e-3, 1.0e-4, 1.0e-3).unwrap();
        let outcome = take_dense_expression_joint_step(
            &model,
            &model.neutral_identity(),
            &model.neutral_expression(),
            &GnmJointState::neutral(model.joint_count()),
            &mapping,
            &observation,
            &truth_projection(),
            config,
            None,
        )
        .unwrap();
        if outcome.accepted {
            assert!(outcome.expression_step_norm <= config.max_expression_step + 1.0e-9);
            assert!(outcome.joint_rotation_step_norm <= config.max_joint_rotation_step + 1.0e-9);
        }
    }

    #[test]
    fn dimension_mismatch_and_invalid_steps_fail_closed() {
        let model = lin_model();
        let mapping = mapping_for(&model, 64);
        let observation = lin_observation(&model, &mapping);
        let short_expression = GnmExpressionState::new(
            vec![0.0; GNM_HEAD_V3_EXPRESSION_DIM - 1],
            GNM_HEAD_V3_EXPRESSION_DIM - 1,
        )
        .unwrap();
        assert!(
            linearize_dense_reprojection(
                &model,
                &model.neutral_identity(),
                &short_expression,
                &GnmJointState::neutral(model.joint_count()),
                &mapping,
                &observation,
                &truth_projection(),
                DenseReprojectionConfig::default(),
                LinearizationStepSizes::default(),
            )
            .is_err()
        );

        let steps = LinearizationStepSizes {
            rigid_pose: 0.0,
            ..LinearizationStepSizes::default()
        };
        assert!(
            linearize_dense_reprojection(
                &model,
                &model.neutral_identity(),
                &lin_expression(),
                &GnmJointState::neutral(model.joint_count()),
                &mapping,
                &observation,
                &truth_projection(),
                DenseReprojectionConfig::default(),
                steps,
            )
            .is_err()
        );
    }

    // -- Auxiliary objective connection (Issue #64.2d / #121) --------------------

    /// Test objective with a constant loss and a constant gradient on
    /// expression channel 0 only.
    struct ConstantGradientTerm {
        loss: f32,
        expression_gradient0: f32,
    }

    impl AuxiliaryObjectiveTerm for ConstantGradientTerm {
        fn evaluate(
            &self,
            expression_values: &[f32],
            joint_rotations: &[[f32; 3]],
            _joint_translation: [f32; 3],
        ) -> Result<AuxiliaryTermEvaluation, GnmReprojectionError> {
            let mut expression_gradient = vec![0.0; expression_values.len()];
            #[allow(clippy::indexing_slicing)] // channel 0 exists by construction
            {
                expression_gradient[0] = self.expression_gradient0;
            }
            Ok(AuxiliaryTermEvaluation {
                loss: self.loss,
                expression_gradient,
                joint_gradient: vec![0.0; 3 * (joint_rotations.len() + 1)],
            })
        }
    }

    fn ej_observation(
        mouth: f32,
        eyelid: f32,
    ) -> (GnmModel, DenseCorrespondenceSet, GnmDenseObservation) {
        let model = ej_model();
        let mapping = mapping_for(&model, 64);
        let observation = synthesize_observation_from_projection(
            &model,
            &model.neutral_identity(),
            &ej_expression(mouth, eyelid),
            &GnmJointState::neutral(model.joint_count()),
            &mapping,
            &truth_projection(),
            SynthesisOptions::default(),
            DenseCoveragePolicy::new(2, 0.75).unwrap(),
            |_, _| false,
        )
        .unwrap();
        (model, mapping, observation)
    }

    #[test]
    fn zero_auxiliary_weight_is_identical_to_the_dense_only_step() {
        let (model, mapping, observation) = ej_observation(0.8, 0.0);
        let joints = GnmJointState::neutral(model.joint_count());
        let config = DenseExpressionJointStepConfig::default();
        let term = ConstantGradientTerm {
            loss: 3.0,
            expression_gradient0: 5.0,
        };

        let dense_only = take_dense_expression_joint_step(
            &model,
            &model.neutral_identity(),
            &model.neutral_expression(),
            &joints,
            &mapping,
            &observation,
            &truth_projection(),
            config,
            None,
        )
        .unwrap();
        let zero_weight = take_dense_expression_joint_step(
            &model,
            &model.neutral_identity(),
            &model.neutral_expression(),
            &joints,
            &mapping,
            &observation,
            &truth_projection(),
            config,
            Some((&term, 0.0)),
        )
        .unwrap();

        assert_eq!(dense_only, zero_weight);
    }

    #[test]
    fn auxiliary_gradient_shifts_the_update_against_its_sign() {
        let (model, mapping, observation) = ej_observation(0.8, 0.0);
        let joints = GnmJointState::neutral(model.joint_count());
        let config = DenseExpressionJointStepConfig::default();
        let term = ConstantGradientTerm {
            loss: 1.0,
            expression_gradient0: 1.0e-3,
        };

        let dense_only = take_dense_expression_joint_step(
            &model,
            &model.neutral_identity(),
            &model.neutral_expression(),
            &joints,
            &mapping,
            &observation,
            &truth_projection(),
            config,
            None,
        )
        .unwrap();
        let with_aux = take_dense_expression_joint_step(
            &model,
            &model.neutral_identity(),
            &model.neutral_expression(),
            &joints,
            &mapping,
            &observation,
            &truth_projection(),
            config,
            Some((&term, 1.0)),
        )
        .unwrap();

        assert!(dense_only.accepted && with_aux.accepted);
        // The rhs receives `-w * gradient`, so a positive channel-0 gradient
        // must pull the accepted update downward relative to dense-only.
        #[allow(clippy::indexing_slicing)] // channel 0 exists by construction
        {
            assert!(with_aux.expression.values()[0] < dense_only.expression.values()[0]);
        }
    }

    #[test]
    fn non_finite_or_negative_auxiliary_weight_fails_closed() {
        let (model, mapping, observation) = ej_observation(0.8, 0.0);
        let term = ConstantGradientTerm {
            loss: 1.0,
            expression_gradient0: 1.0,
        };
        for weight in [f32::NAN, f32::INFINITY, -0.5] {
            assert!(matches!(
                take_dense_expression_joint_step(
                    &model,
                    &model.neutral_identity(),
                    &model.neutral_expression(),
                    &GnmJointState::neutral(model.joint_count()),
                    &mapping,
                    &observation,
                    &truth_projection(),
                    DenseExpressionJointStepConfig::default(),
                    Some((&term, weight)),
                ),
                Err(GnmReprojectionError::InvalidConfig(_))
            ));
        }
    }

    // -- Bounded block-coordinate single-frame fit (Issue #64.2e / #122) ---------

    fn fit_config() -> SingleFrameFitConfig {
        SingleFrameFitConfig::default()
    }

    #[test]
    fn cold_start_fit_recovers_neutral_pose_without_moving_expression() {
        let model = lin_model();
        let mapping = mapping_for(&model, 64);
        let observation = synthesize_observation_from_projection(
            &model,
            &model.neutral_identity(),
            &model.neutral_expression(),
            &GnmJointState::neutral(model.joint_count()),
            &mapping,
            &truth_projection(),
            SynthesisOptions::default(),
            DenseCoveragePolicy::new(2, 0.75).unwrap(),
            |_, _| false,
        )
        .unwrap();

        let outcome = fit_single_frame_cold_start(
            &model,
            &model.neutral_identity(),
            &model.neutral_expression(),
            &GnmJointState::neutral(model.joint_count()),
            &mapping,
            &observation,
            &perturbed_guess(),
            fit_config(),
            None,
        )
        .unwrap();

        assert_eq!(outcome.status(), SingleFrameFitStatus::Converged);
        assert!(outcome.valid());
        let truth = truth_projection();
        for (actual, expected) in outcome
            .projection()
            .yaw_pitch_roll()
            .into_iter()
            .zip(truth.yaw_pitch_roll())
        {
            assert!((actual - expected).abs() < 2.0e-2, "{actual} vs {expected}");
        }
        for value in outcome.expression().values() {
            assert!(
                value.abs() < 5.0e-2,
                "expression moved on neutral data: {value}"
            );
        }
        assert!(outcome.objective() < 5.0e-3);
    }

    #[test]
    fn cold_start_fit_recovers_mouth_and_pose_together() {
        let (model, mapping, observation) = ej_observation(0.8, 0.0);

        let outcome = fit_single_frame_cold_start(
            &model,
            &model.neutral_identity(),
            &model.neutral_expression(),
            &GnmJointState::neutral(model.joint_count()),
            &mapping,
            &observation,
            &perturbed_guess(),
            fit_config(),
            None,
        )
        .unwrap();

        assert_eq!(outcome.status(), SingleFrameFitStatus::Converged);
        assert!(outcome.valid());
        #[allow(clippy::indexing_slicing)] // channel 0 exists by construction
        {
            assert!((outcome.expression().values()[0] - 0.8).abs() < 5.0e-2);
        }
        let truth = truth_projection();
        for (actual, expected) in outcome
            .projection()
            .yaw_pitch_roll()
            .into_iter()
            .zip(truth.yaw_pitch_roll())
        {
            assert!((actual - expected).abs() < 2.0e-2, "{actual} vs {expected}");
        }
    }

    #[test]
    fn cold_start_fit_recovers_blink_channel() {
        let (model, mapping, observation) = ej_observation(0.0, 1.0);

        let outcome = fit_single_frame_cold_start(
            &model,
            &model.neutral_identity(),
            &model.neutral_expression(),
            &GnmJointState::neutral(model.joint_count()),
            &mapping,
            &observation,
            &perturbed_guess(),
            fit_config(),
            None,
        )
        .unwrap();

        assert_eq!(outcome.status(), SingleFrameFitStatus::Converged);
        assert!(outcome.valid());
        #[allow(clippy::indexing_slicing)] // channel 1 exists by construction
        {
            assert!((outcome.expression().values()[1] - 1.0).abs() < 5.0e-2);
        }
        #[allow(clippy::indexing_slicing)] // channel 0 exists by construction
        {
            assert!(outcome.expression().values()[0].abs() < 5.0e-2);
        }
    }

    #[test]
    fn cold_start_fit_recovers_head_yaw() {
        let model = lin_model();
        let mapping = mapping_for(&model, 64);
        let yawed_truth =
            DenseProjection::new([0.35, -0.10, 0.05], [0.02, -0.03, 0.60], 1.3, [0.5, 0.5])
                .unwrap();
        let yawed_guess =
            DenseProjection::new([0.10, -0.06, 0.02], [0.05, 0.00, 0.66], 1.3, [0.5, 0.5]).unwrap();
        // Pin the joint rotation/translation blocks: the dynamic joint
        // translation shares a gauge direction with the camera translation
        // block, so an unconstrained fit can explain the observation at a
        // different pose decomposition. A cold-start head-yaw recovery pins
        // the joint blocks and lets the rigid/camera block own the motion.
        let config = SingleFrameFitConfig::new(
            DenseRigidStepConfig::default(),
            DenseExpressionJointStepConfig::new(0.5, 1.0e-6, 1.0e-9, 1.0e-4, 1.0e-3).unwrap(),
            40,
            1.0e-6,
        )
        .unwrap();
        let observation = synthesize_observation_from_projection(
            &model,
            &model.neutral_identity(),
            &model.neutral_expression(),
            &GnmJointState::neutral(model.joint_count()),
            &mapping,
            &yawed_truth,
            SynthesisOptions::default(),
            DenseCoveragePolicy::new(2, 0.75).unwrap(),
            |_, _| false,
        )
        .unwrap();

        let outcome = fit_single_frame_cold_start(
            &model,
            &model.neutral_identity(),
            &model.neutral_expression(),
            &GnmJointState::neutral(model.joint_count()),
            &mapping,
            &observation,
            &yawed_guess,
            config,
            None,
        )
        .unwrap();

        assert_eq!(outcome.status(), SingleFrameFitStatus::Converged);
        assert!(outcome.valid());
        assert!((outcome.projection().yaw_pitch_roll()[0] - 0.35).abs() < 5.0e-3);
    }

    #[test]
    fn one_iteration_budget_with_tiny_steps_reports_max_iterations() {
        let (model, mapping, observation) = ej_observation(0.8, 0.0);
        let config = SingleFrameFitConfig::new(
            DenseRigidStepConfig::new(1.0e-4, 1.0e-4, 1.0e-6).unwrap(),
            DenseExpressionJointStepConfig::new(1.0e-4, 1.0e-5, 1.0e-5, 1.0e-8, 1.0e-7).unwrap(),
            1,
            0.0,
        )
        .unwrap();

        let outcome = fit_single_frame_cold_start(
            &model,
            &model.neutral_identity(),
            &model.neutral_expression(),
            &GnmJointState::neutral(model.joint_count()),
            &mapping,
            &observation,
            &truth_projection(),
            config,
            None,
        )
        .unwrap();

        assert_eq!(outcome.status(), SingleFrameFitStatus::MaxIterationsReached);
        assert!(!outcome.valid());
        assert_eq!(outcome.iterations(), 1);
    }

    #[test]
    fn fit_config_fails_closed() {
        assert!(
            SingleFrameFitConfig::new(
                DenseRigidStepConfig::default(),
                DenseExpressionJointStepConfig::default(),
                0,
                1.0e-6,
            )
            .is_err()
        );
        assert!(
            SingleFrameFitConfig::new(
                DenseRigidStepConfig::default(),
                DenseExpressionJointStepConfig::default(),
                MAX_SINGLE_FRAME_FIT_ITERATIONS + 1,
                1.0e-6,
            )
            .is_err()
        );
        assert!(
            SingleFrameFitConfig::new(
                DenseRigidStepConfig::default(),
                DenseExpressionJointStepConfig::default(),
                8,
                f32::NAN,
            )
            .is_err()
        );
        assert!(
            SingleFrameFitConfig::new(
                DenseRigidStepConfig::default(),
                DenseExpressionJointStepConfig::default(),
                8,
                -1.0,
            )
            .is_err()
        );
    }
}
