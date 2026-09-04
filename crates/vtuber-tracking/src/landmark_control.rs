//! MediaPipe landmark latent controls and upper-bound decoder (Issue #18).

use std::collections::BTreeSet;

use nalgebra::{DMatrix, linalg::SVD};
use serde::{Deserialize, Serialize};
use vtuber_core::{ARKIT_NON_TONGUE_CHANNEL_COUNT, Arkit52Coefficients, arkit_non_tongue_values};

use crate::arkit_teacher::{PairedTemporalSample, TeacherDatasetError, validate_paired_samples};
use crate::causal_prior::{
    LinearPriorFitError, LinearPriorTrainingConfig, fit_normalized_multi_output_ridge,
};
use crate::causal_prior_inference::LinearPriorLoadError;
use crate::gnm_semantic_decoder::{
    GnmSemanticDecoderKind, GnmSemanticFeatureConfig, GnmSemanticFrame, GnmSemanticRow,
    build_gnm_semantic_features, gnm_semantic_frame_from_sample,
};
use crate::teacher_aligned_basis::{
    TeacherAlignedBasisError, TeacherAlignedGnmBasisArtifact, normalize_teacher_residuals,
    reconstruct_teacher_aligned_expression,
};
use crate::teacher_residual::NormalizedLinearMapArtifact;

/// Flattened 478-point MediaPipe `(x, y)` dimension.
pub const NORMALIZED_LANDMARK_XY_DIM: usize = 478 * 2;
/// Landmark teacher-aligned basis schema version.
pub const LANDMARK_ALIGNED_BASIS_SCHEMA_VERSION: u32 = 1;
/// L/HL decoder schema version.
pub const LANDMARK_CONTROL_DECODER_SCHEMA_VERSION: u32 = 1;

/// Typed landmark-control failure.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum LandmarkControlError {
    /// Exactly 478 `(x, y)` points are required.
    #[error("expected 478 landmark points, found {0}")]
    LandmarkCount(usize),
    /// Landmark coordinates or a computed number are non-finite.
    #[error("invalid landmark-control numeric value: {0}")]
    InvalidNumeric(&'static str),
    /// Centered landmark RMS scale is at most `1.0e-6`.
    #[error("landmark RMS scale is degenerate")]
    DegenerateScale,
    /// Requested rank is outside `1..=51`.
    #[error("landmark basis rank {0} is outside 1..=51")]
    InvalidRank(usize),
    /// No selected training samples/rows exist.
    #[error("no landmark-control training data belongs to the selected takes")]
    EmptyTrainingSet,
    /// Artifact or feature dimensions disagree with the fixed contract.
    #[error("invalid landmark-control shape: {0}")]
    InvalidShape(&'static str),
    /// Artifact hash does not match its fields.
    #[error("landmark-control content hash mismatch")]
    HashMismatch,
    /// Paired trace validation failed.
    #[error("invalid paired trace: {0:?}")]
    TeacherDataset(TeacherDatasetError),
    /// Shared teacher-residual normalization failed.
    #[error(transparent)]
    ResidualNormalization(TeacherAlignedBasisError),
    /// Reduced GNM history/basis failed.
    #[error("invalid reduced GNM input: {0}")]
    Gnm(String),
    /// GNM and landmark frame identities differ.
    #[error("GNM and landmark histories are not exact-frame aligned")]
    IdentityMismatch,
    /// Shared ridge fit failed.
    #[error("normalized ridge fit failed: {0:?}")]
    Ridge(LinearPriorFitError),
    /// Decoder kind and supplied basis provenance disagree.
    #[error("decoder kind and GNM basis presence disagree")]
    KindMismatch,
}

/// Removes per-frame translation and RMS scale from all 478 `(x, y)` points.
///
/// # Errors
///
/// Rejects the wrong point count, non-finite coordinates, or degenerate scale.
pub fn normalize_landmarks_xy(
    landmarks: &[[f32; 2]],
) -> Result<[f32; NORMALIZED_LANDMARK_XY_DIM], LandmarkControlError> {
    if landmarks.len() != 478 {
        return Err(LandmarkControlError::LandmarkCount(landmarks.len()));
    }
    if landmarks.iter().flatten().any(|value| !value.is_finite()) {
        return Err(LandmarkControlError::InvalidNumeric("landmark coordinate"));
    }
    let count = landmarks.len() as f32;
    let mean_x = landmarks.iter().map(|[x, _]| x).sum::<f32>() / count;
    let mean_y = landmarks.iter().map(|[_, y]| y).sum::<f32>() / count;
    let scale = (landmarks
        .iter()
        .map(|[x, y]| (x - mean_x).powi(2) + (y - mean_y).powi(2))
        .sum::<f32>()
        / count)
        .sqrt();
    if !scale.is_finite() {
        return Err(LandmarkControlError::InvalidNumeric("landmark scale"));
    }
    if scale <= 1.0e-6 {
        return Err(LandmarkControlError::DegenerateScale);
    }
    let mut normalized = [0.0_f32; NORMALIZED_LANDMARK_XY_DIM];
    for (pair, &[x, y]) in normalized.chunks_exact_mut(2).zip(landmarks) {
        pair.copy_from_slice(&[(x - mean_x) / scale, (y - mean_y) / scale]);
    }
    Ok(normalized)
}

/// One normalized landmark row paired with its exact-frame teacher residual.
#[derive(Clone, Debug, PartialEq)]
pub struct LandmarkAlignmentSample {
    /// Source take id.
    pub take_id: String,
    /// Source frame sequence.
    pub frame_seq: u64,
    /// Translation/scale-normalized 478-point `(x, y)` vector.
    pub normalized_xy: [f32; NORMALIZED_LANDMARK_XY_DIM],
    /// Same-frame teacher-minus-Direct non-tongue residual.
    pub teacher_residual: [f32; ARKIT_NON_TONGUE_CHANNEL_COUNT],
}

/// Builds exact-frame landmark/residual alignment samples without GNM input.
///
/// # Errors
///
/// Returns paired-trace or landmark-normalization failures.
pub fn build_landmark_alignment_samples(
    take_id: &str,
    samples: &[PairedTemporalSample],
) -> Result<Vec<LandmarkAlignmentSample>, LandmarkControlError> {
    validate_paired_samples(samples).map_err(LandmarkControlError::TeacherDataset)?;
    let mut output = Vec::new();
    for sample in samples {
        let (Some(observation), Some(teacher)) = (
            sample.mediapipe_observation.as_ref(),
            sample.teacher.as_ref(),
        ) else {
            continue;
        };
        let teacher_values = arkit_non_tongue_values(&teacher.coefficients);
        let direct_values = arkit_non_tongue_values(&observation.direct_coefficients);
        let mut teacher_residual = teacher_values;
        for (value, direct) in teacher_residual.iter_mut().zip(direct_values) {
            *value -= direct;
        }
        output.push(LandmarkAlignmentSample {
            take_id: take_id.to_owned(),
            frame_seq: sample.frame_seq,
            normalized_xy: normalize_landmarks_xy(&observation.landmarks_xy)?,
            teacher_residual,
        });
    }
    Ok(output)
}

/// Versioned teacher-residual-aligned landmark basis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LandmarkAlignedBasisArtifact {
    /// Artifact schema version.
    pub schema_version: u32,
    /// Retained latent rank.
    pub rank: usize,
    /// Fixed source dimension (956).
    pub source_dimension: usize,
    /// Ordered training take ids.
    pub training_takes: Vec<String>,
    /// Training-only mean normalized landmark vector.
    pub source_mean: Vec<f32>,
    /// Training-only residual mean.
    #[serde(with = "residual_array")]
    pub residual_mean: [f32; ARKIT_NON_TONGUE_CHANNEL_COUNT],
    /// Training-only residual standard deviation.
    #[serde(with = "residual_array")]
    pub residual_std: [f32; ARKIT_NON_TONGUE_CHANNEL_COUNT],
    /// Residual channels inactive under the shared `1.0e-6` threshold.
    pub inactive_residual_channels: Vec<usize>,
    /// Cross-covariance singular values in descending order.
    pub singular_values_descending: Vec<f64>,
    /// Row-major `[956, rank]` orthonormal basis.
    pub basis_row_major: Vec<f32>,
    /// Deterministic content hash over every preceding field.
    pub content_hash: u64,
}

mod residual_array {
    use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error};
    use vtuber_core::ARKIT_NON_TONGUE_CHANNEL_COUNT;

    pub fn serialize<S>(
        values: &[f32; ARKIT_NON_TONGUE_CHANNEL_COUNT],
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        values.as_slice().serialize(serializer)
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<[f32; ARKIT_NON_TONGUE_CHANNEL_COUNT], D::Error>
    where
        D: Deserializer<'de>,
    {
        Vec::<f32>::deserialize(deserializer)?
            .try_into()
            .map_err(|values: Vec<f32>| {
                D::Error::invalid_length(values.len(), &"exactly 51 residual channels")
            })
    }
}

/// Fits the direct 956-by-51 teacher-residual cross-covariance SVD.
///
/// # Errors
///
/// Rejects rank, training selection, numeric, and SVD shape failures.
#[allow(clippy::indexing_slicing)]
pub fn fit_landmark_aligned_basis(
    samples: &[LandmarkAlignmentSample],
    training_takes: &BTreeSet<String>,
    rank: usize,
) -> Result<LandmarkAlignedBasisArtifact, LandmarkControlError> {
    if !(1..=ARKIT_NON_TONGUE_CHANNEL_COUNT).contains(&rank) {
        return Err(LandmarkControlError::InvalidRank(rank));
    }
    let training: Vec<&LandmarkAlignmentSample> = samples
        .iter()
        .filter(|sample| training_takes.contains(&sample.take_id))
        .collect();
    if training.is_empty() {
        return Err(LandmarkControlError::EmptyTrainingSet);
    }
    let count = training.len() as f32;
    let mut source_mean = vec![0.0_f32; NORMALIZED_LANDMARK_XY_DIM];
    for sample in &training {
        for (mean, value) in source_mean.iter_mut().zip(sample.normalized_xy) {
            *mean += value / count;
        }
    }
    let residuals: Vec<[f32; ARKIT_NON_TONGUE_CHANNEL_COUNT]> = training
        .iter()
        .map(|sample| sample.teacher_residual)
        .collect();
    let normalization = normalize_teacher_residuals(&residuals)
        .map_err(LandmarkControlError::ResidualNormalization)?;
    let mut cross =
        DMatrix::<f64>::zeros(NORMALIZED_LANDMARK_XY_DIM, ARKIT_NON_TONGUE_CHANNEL_COUNT);
    for (sample, residual) in training.iter().zip(&normalization.normalized) {
        for source in 0..NORMALIZED_LANDMARK_XY_DIM {
            let centered = sample.normalized_xy[source] - source_mean[source];
            for channel in 0..ARKIT_NON_TONGUE_CHANNEL_COUNT {
                cross[(source, channel)] +=
                    f64::from(centered * residual[channel]) / f64::from(count);
            }
        }
    }
    let decomposition = SVD::new(cross, true, false);
    let left = decomposition
        .u
        .ok_or(LandmarkControlError::InvalidShape("left singular vectors"))?;
    let mut order: Vec<usize> = (0..decomposition.singular_values.len()).collect();
    order.sort_by(|left_index, right_index| {
        decomposition.singular_values[*right_index]
            .total_cmp(&decomposition.singular_values[*left_index])
    });
    let singular_values_descending = order
        .iter()
        .take(rank)
        .map(|index| decomposition.singular_values[*index])
        .collect();
    let mut basis_row_major = vec![0.0_f32; NORMALIZED_LANDMARK_XY_DIM * rank];
    for (target_column, source_column) in order.iter().take(rank).enumerate() {
        let mut norm = 0.0_f64;
        for row in 0..NORMALIZED_LANDMARK_XY_DIM {
            let value = left[(row, *source_column)];
            norm += value * value;
            basis_row_major[row * rank + target_column] = value as f32;
        }
        norm = norm.sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            return Err(LandmarkControlError::InvalidNumeric("basis norm"));
        }
        let sign = (0..NORMALIZED_LANDMARK_XY_DIM)
            .map(|row| basis_row_major[row * rank + target_column])
            .max_by(|left_value, right_value| left_value.abs().total_cmp(&right_value.abs()))
            .map_or(1.0, |value| if value < 0.0 { -1.0 } else { 1.0 });
        for row in 0..NORMALIZED_LANDMARK_XY_DIM {
            basis_row_major[row * rank + target_column] *= sign / norm as f32;
        }
    }
    let mut artifact = LandmarkAlignedBasisArtifact {
        schema_version: LANDMARK_ALIGNED_BASIS_SCHEMA_VERSION,
        rank,
        source_dimension: NORMALIZED_LANDMARK_XY_DIM,
        training_takes: training_takes.iter().cloned().collect(),
        source_mean,
        residual_mean: normalization.mean,
        residual_std: normalization.std,
        inactive_residual_channels: normalization.inactive_channels,
        singular_values_descending,
        basis_row_major,
        content_hash: 0,
    };
    artifact.content_hash = hash_landmark_basis(&artifact);
    Ok(artifact)
}

/// Projects `P^T (x - mean_train)`.
///
/// # Errors
///
/// Rejects malformed/hash-mismatched artifacts or non-finite results.
pub fn project_landmark_latent(
    normalized_xy: &[f32; NORMALIZED_LANDMARK_XY_DIM],
    basis: &LandmarkAlignedBasisArtifact,
) -> Result<Vec<f32>, LandmarkControlError> {
    validate_landmark_basis(basis)?;
    let mut latent = vec![0.0_f32; basis.rank];
    for ((value, mean), basis_row) in normalized_xy
        .iter()
        .zip(&basis.source_mean)
        .zip(basis.basis_row_major.chunks_exact(basis.rank))
    {
        let centered = *value - *mean;
        for (output, coefficient) in latent.iter_mut().zip(basis_row) {
            *output += centered * coefficient;
        }
    }
    if latent.iter().any(|value| !value.is_finite()) {
        return Err(LandmarkControlError::InvalidNumeric("landmark latent"));
    }
    Ok(latent)
}

fn validate_landmark_basis(
    basis: &LandmarkAlignedBasisArtifact,
) -> Result<(), LandmarkControlError> {
    if basis.schema_version != LANDMARK_ALIGNED_BASIS_SCHEMA_VERSION
        || basis.source_dimension != NORMALIZED_LANDMARK_XY_DIM
        || !(1..=ARKIT_NON_TONGUE_CHANNEL_COUNT).contains(&basis.rank)
        || basis.source_mean.len() != NORMALIZED_LANDMARK_XY_DIM
        || basis.singular_values_descending.len() != basis.rank
        || basis.basis_row_major.len() != NORMALIZED_LANDMARK_XY_DIM * basis.rank
    {
        return Err(LandmarkControlError::InvalidShape("landmark basis"));
    }
    if hash_landmark_basis(basis) != basis.content_hash {
        return Err(LandmarkControlError::HashMismatch);
    }
    Ok(())
}

/// L/HL decoder variants.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum LandmarkControlDecoderKind {
    /// Direct plus a landmark-latent residual.
    LandmarkResidual,
    /// Direct plus reduced GNM and landmark-latent residual (information upper bound).
    GnmLandmarkUpperBound,
}

/// Exact-frame landmark latent and Direct record.
#[derive(Clone, Debug, PartialEq)]
pub struct LandmarkControlFrame {
    /// Source frame sequence.
    pub frame_seq: u64,
    /// Source monotonic timestamp.
    pub timestamp_micros: u64,
    /// Teacher-aligned landmark latent.
    pub landmark_latent: Vec<f32>,
    /// Exact-frame Direct coefficients.
    pub direct: Arkit52Coefficients,
}

/// Builds L features without GNM state or diagnostics.
///
/// # Errors
///
/// Rejects invalid configuration, rank changes, frame gaps, or non-finite values.
#[allow(clippy::indexing_slicing)]
pub fn build_landmark_residual_features(
    newest_first_history: &[LandmarkControlFrame],
    config: GnmSemanticFeatureConfig,
) -> Result<Option<Vec<f32>>, LandmarkControlError> {
    if config.history_len == 0 || config.max_gap_micros == 0 {
        return Err(LandmarkControlError::InvalidShape("feature configuration"));
    }
    let required = config.history_len.max(2);
    if newest_first_history.len() < required {
        return Ok(None);
    }
    let rank = newest_first_history[0].landmark_latent.len();
    if rank == 0 {
        return Err(LandmarkControlError::InvalidShape("zero landmark rank"));
    }
    for (index, frame) in newest_first_history.iter().take(required).enumerate() {
        if frame.landmark_latent.len() != rank
            || frame.landmark_latent.iter().any(|value| !value.is_finite())
        {
            return Err(LandmarkControlError::InvalidShape("landmark history rank"));
        }
        if index > 0 {
            let newer = &newest_first_history[index - 1];
            if newer.frame_seq != frame.frame_seq + 1
                || newer.timestamp_micros <= frame.timestamp_micros
                || newer.timestamp_micros - frame.timestamp_micros > config.max_gap_micros
            {
                return Err(LandmarkControlError::IdentityMismatch);
            }
        }
    }
    let slot_width = rank + ARKIT_NON_TONGUE_CHANNEL_COUNT;
    let mut features =
        vec![0.0_f32; config.history_len * slot_width + rank + ARKIT_NON_TONGUE_CHANNEL_COUNT + 1];
    for (slot, frame) in newest_first_history
        .iter()
        .take(config.history_len)
        .enumerate()
    {
        let base = slot * slot_width;
        features[base..base + rank].copy_from_slice(&frame.landmark_latent);
        features[base + rank..base + slot_width]
            .copy_from_slice(&arkit_non_tongue_values(&frame.direct));
    }
    let current = &newest_first_history[0];
    let previous = &newest_first_history[1];
    let dt = (current.timestamp_micros - previous.timestamp_micros) as f32 / 1_000_000.0;
    let velocity_base = config.history_len * slot_width;
    for index in 0..rank {
        features[velocity_base + index] =
            (current.landmark_latent[index] - previous.landmark_latent[index]) / dt;
    }
    let current_direct = arkit_non_tongue_values(&current.direct);
    let previous_direct = arkit_non_tongue_values(&previous.direct);
    for index in 0..ARKIT_NON_TONGUE_CHANNEL_COUNT {
        features[velocity_base + rank + index] =
            (current_direct[index] - previous_direct[index]) / dt;
    }
    features[velocity_base + rank + ARKIT_NON_TONGUE_CHANNEL_COUNT] = dt;
    Ok(Some(features))
}

/// Builds HL features by exact-frame joining reduced GNM and landmark histories.
///
/// # Errors
///
/// Rejects any identity, rank, history, or numeric mismatch.
#[allow(clippy::indexing_slicing)]
pub fn build_gnm_landmark_upper_bound_features(
    newest_first_gnm: &[GnmSemanticFrame],
    newest_first_landmark: &[LandmarkControlFrame],
    config: GnmSemanticFeatureConfig,
) -> Result<Option<Vec<f32>>, LandmarkControlError> {
    let Some(gnm_features) = build_gnm_semantic_features(
        newest_first_gnm,
        GnmSemanticDecoderKind::HybridResidual,
        config,
    )
    .map_err(|error| LandmarkControlError::Gnm(error.to_string()))?
    else {
        return Ok(None);
    };
    let Some(_) = build_landmark_residual_features(newest_first_landmark, config)? else {
        return Ok(None);
    };
    let required = config.history_len.max(2);
    if newest_first_gnm.len() < required || newest_first_landmark.len() < required {
        return Ok(None);
    }
    for (gnm, landmark) in newest_first_gnm
        .iter()
        .zip(newest_first_landmark)
        .take(required)
    {
        if gnm.frame_seq != landmark.frame_seq || gnm.timestamp_micros != landmark.timestamp_micros
        {
            return Err(LandmarkControlError::IdentityMismatch);
        }
    }
    let gnm_rank = newest_first_gnm[0].reduced_expression.len();
    let landmark_rank = newest_first_landmark[0].landmark_latent.len();
    if gnm_rank != landmark_rank {
        return Err(LandmarkControlError::InvalidShape("HL latent ranks"));
    }
    let joint_count = newest_first_gnm[0].joint_rotations.len();
    let gnm_slot = gnm_rank + joint_count * 3 + 3 + 1 + 14 + 51;
    let hl_slot = gnm_slot + landmark_rank;
    let mut features =
        vec![0.0_f32; config.history_len * hl_slot + gnm_rank + landmark_rank + 51 + 1];
    for (slot, landmark) in newest_first_landmark
        .iter()
        .enumerate()
        .take(config.history_len)
    {
        let gnm_base = slot * gnm_slot;
        let hl_base = slot * hl_slot;
        features[hl_base..hl_base + gnm_rank]
            .copy_from_slice(&gnm_features[gnm_base..gnm_base + gnm_rank]);
        features[hl_base + gnm_rank..hl_base + gnm_rank + landmark_rank]
            .copy_from_slice(&landmark.landmark_latent);
        features[hl_base + gnm_rank + landmark_rank..hl_base + hl_slot]
            .copy_from_slice(&gnm_features[gnm_base + gnm_rank..gnm_base + gnm_slot]);
    }
    let gnm_velocity_base = config.history_len * gnm_slot;
    let hl_velocity_base = config.history_len * hl_slot;
    features[hl_velocity_base..hl_velocity_base + gnm_rank]
        .copy_from_slice(&gnm_features[gnm_velocity_base..gnm_velocity_base + gnm_rank]);
    let dt = (newest_first_landmark[0].timestamp_micros - newest_first_landmark[1].timestamp_micros)
        as f32
        / 1_000_000.0;
    for index in 0..landmark_rank {
        features[hl_velocity_base + gnm_rank + index] = (newest_first_landmark[0].landmark_latent
            [index]
            - newest_first_landmark[1].landmark_latent[index])
            / dt;
    }
    let gnm_direct_base = gnm_velocity_base + gnm_rank;
    let hl_direct_base = hl_velocity_base + gnm_rank + landmark_rank;
    features[hl_direct_base..hl_direct_base + 51]
        .copy_from_slice(&gnm_features[gnm_direct_base..gnm_direct_base + 51]);
    features[hl_direct_base + 51] = dt;
    Ok(Some(features))
}

/// Builds L or HL supervised rows with exact-frame residual targets.
///
/// # Errors
///
/// Returns typed trace, basis, normalization, join, or feature failures.
pub fn build_landmark_control_rows(
    take_id: &str,
    samples: &[PairedTemporalSample],
    landmark_basis: &LandmarkAlignedBasisArtifact,
    gnm_basis: Option<&TeacherAlignedGnmBasisArtifact>,
    kind: LandmarkControlDecoderKind,
    config: GnmSemanticFeatureConfig,
) -> Result<Vec<GnmSemanticRow>, LandmarkControlError> {
    match (kind, gnm_basis) {
        (LandmarkControlDecoderKind::LandmarkResidual, None)
        | (LandmarkControlDecoderKind::GnmLandmarkUpperBound, Some(_)) => {}
        _ => return Err(LandmarkControlError::KindMismatch),
    }
    validate_paired_samples(samples).map_err(LandmarkControlError::TeacherDataset)?;
    let mut rows = Vec::new();
    let mut landmark_history = Vec::<LandmarkControlFrame>::new();
    let mut gnm_history = Vec::<GnmSemanticFrame>::new();
    for sample in samples {
        let (Some(observation), Some(teacher)) = (
            sample.mediapipe_observation.as_ref(),
            sample.teacher.as_ref(),
        ) else {
            landmark_history.clear();
            gnm_history.clear();
            continue;
        };
        let normalized = normalize_landmarks_xy(&observation.landmarks_xy)?;
        let landmark = LandmarkControlFrame {
            frame_seq: sample.frame_seq,
            timestamp_micros: sample.timestamp_micros,
            landmark_latent: project_landmark_latent(&normalized, landmark_basis)?,
            direct: observation.direct_coefficients,
        };
        let discontinuous = landmark_history.first().is_some_and(|previous| {
            landmark.frame_seq != previous.frame_seq + 1
                || landmark.timestamp_micros <= previous.timestamp_micros
                || landmark.timestamp_micros - previous.timestamp_micros > config.max_gap_micros
        });
        if discontinuous {
            landmark_history.clear();
            gnm_history.clear();
        }
        landmark_history.insert(0, landmark);
        let features = match kind {
            LandmarkControlDecoderKind::LandmarkResidual => {
                build_landmark_residual_features(&landmark_history, config)?
            }
            LandmarkControlDecoderKind::GnmLandmarkUpperBound => {
                let basis = gnm_basis.ok_or(LandmarkControlError::KindMismatch)?;
                let Some(gnm) = gnm_semantic_frame_from_sample(sample, basis)
                    .map_err(|error| LandmarkControlError::Gnm(error.to_string()))?
                else {
                    landmark_history.clear();
                    gnm_history.clear();
                    continue;
                };
                gnm_history.insert(0, gnm);
                build_gnm_landmark_upper_bound_features(&gnm_history, &landmark_history, config)?
            }
        };
        if let Some(features) = features {
            let mut target = arkit_non_tongue_values(&teacher.coefficients);
            for (value, direct) in target
                .iter_mut()
                .zip(arkit_non_tongue_values(&observation.direct_coefficients))
            {
                *value -= direct;
            }
            rows.push(GnmSemanticRow {
                take_id: take_id.to_owned(),
                frame_seq: sample.frame_seq,
                feature_config: config,
                features,
                target,
            });
        }
        landmark_history.truncate(config.history_len.max(2));
        gnm_history.truncate(config.history_len.max(2));
    }
    Ok(rows)
}

/// Versioned L/HL residual decoder.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LandmarkControlDecoderArtifact {
    /// Artifact schema version.
    pub schema_version: u32,
    /// L or HL semantics.
    pub kind: LandmarkControlDecoderKind,
    /// Landmark basis content hash.
    pub landmark_basis_content_hash: u64,
    /// HL reduced-GNM basis hash; absent for L.
    pub gnm_basis_content_hash: Option<u64>,
    /// Shared landmark/GNM latent rank.
    pub rank: usize,
    /// Exact feature order.
    pub feature_order: String,
    /// Causal feature configuration.
    pub feature_config: GnmSemanticFeatureConfig,
    /// Ordered training take ids.
    pub training_takes: Vec<String>,
    /// Shared normalized ridge map.
    pub linear_map: NormalizedLinearMapArtifact,
    /// Deterministic content hash.
    pub content_hash: u64,
}

/// Stable L/HL feature-order string.
#[must_use]
pub fn landmark_control_feature_order(kind: LandmarkControlDecoderKind) -> String {
    match kind {
        LandmarkControlDecoderKind::LandmarkResidual => "v1:newest-first(landmark-latent+direct-51)+landmark-velocity+direct-velocity-51+dt-seconds".to_owned(),
        LandmarkControlDecoderKind::GnmLandmarkUpperBound => "v1:newest-first(reduced-gnm+landmark-latent+joint-axis-angle+rigid-ypr+objective+regions-14+direct-51)+reduced-gnm-velocity+landmark-velocity+direct-velocity-51+dt-seconds".to_owned(),
    }
}

/// Fits L or HL with the existing normalized ridge kernel.
///
/// The explicit basis arguments are required to place verifiable content hashes
/// in the artifact; row features alone cannot recover artifact provenance.
///
/// # Errors
///
/// Rejects kind/basis mismatch, malformed rows, or shared ridge failures.
pub fn fit_landmark_control_decoder(
    rows: &[GnmSemanticRow],
    training_takes: &BTreeSet<String>,
    kind: LandmarkControlDecoderKind,
    landmark_basis: &LandmarkAlignedBasisArtifact,
    gnm_basis: Option<&TeacherAlignedGnmBasisArtifact>,
    config: LinearPriorTrainingConfig,
) -> Result<LandmarkControlDecoderArtifact, LandmarkControlError> {
    validate_landmark_basis(landmark_basis)?;
    let gnm_basis_content_hash = match (kind, gnm_basis) {
        (LandmarkControlDecoderKind::LandmarkResidual, None) => None,
        (LandmarkControlDecoderKind::GnmLandmarkUpperBound, Some(basis)) => {
            if basis.rank != landmark_basis.rank {
                return Err(LandmarkControlError::InvalidShape("L/HL rank mismatch"));
            }
            reconstruct_teacher_aligned_expression(&vec![0.0; basis.rank], basis)
                .map_err(LandmarkControlError::ResidualNormalization)?;
            Some(basis.content_hash)
        }
        _ => return Err(LandmarkControlError::KindMismatch),
    };
    let selected: Vec<&GnmSemanticRow> = rows
        .iter()
        .filter(|row| training_takes.contains(&row.take_id))
        .collect();
    let first = selected
        .first()
        .ok_or(LandmarkControlError::EmptyTrainingSet)?;
    if selected.iter().any(|row| {
        row.feature_config != first.feature_config
            || row.features.len() != first.features.len()
            || row.target.iter().any(|value| !value.is_finite())
    }) {
        return Err(LandmarkControlError::InvalidShape("training rows"));
    }
    let features: Vec<Vec<f32>> = selected.iter().map(|row| row.features.clone()).collect();
    let targets: Vec<Vec<f32>> = selected.iter().map(|row| row.target.to_vec()).collect();
    let map = fit_normalized_multi_output_ridge(&features, &targets, config)
        .map_err(LandmarkControlError::Ridge)?;
    let mut artifact = LandmarkControlDecoderArtifact {
        schema_version: LANDMARK_CONTROL_DECODER_SCHEMA_VERSION,
        kind,
        landmark_basis_content_hash: landmark_basis.content_hash,
        gnm_basis_content_hash,
        rank: landmark_basis.rank,
        feature_order: landmark_control_feature_order(kind),
        feature_config: first.feature_config,
        training_takes: training_takes.iter().cloned().collect(),
        linear_map: NormalizedLinearMapArtifact {
            feature_mean: map.feature_mean,
            feature_std: map.feature_std,
            target_mean: map.target_mean,
            target_std: map.target_std,
            weights: map.weights,
        },
        content_hash: 0,
    };
    artifact.content_hash = hash_landmark_decoder(&artifact);
    Ok(artifact)
}

/// Predicts an unclamped signed L/HL residual.
///
/// # Errors
///
/// Rejects schema, hash, kind/provenance, dimension, or numeric failures.
#[allow(clippy::indexing_slicing)]
pub fn predict_landmark_control_raw(
    artifact: &LandmarkControlDecoderArtifact,
    features: &[f32],
) -> Result<[f32; ARKIT_NON_TONGUE_CHANNEL_COUNT], LinearPriorLoadError> {
    if artifact.schema_version != LANDMARK_CONTROL_DECODER_SCHEMA_VERSION {
        return Err(LinearPriorLoadError::UnsupportedSchemaVersion {
            found: artifact.schema_version,
        });
    }
    let computed = hash_landmark_decoder(artifact);
    if computed != artifact.content_hash {
        return Err(LinearPriorLoadError::ContentHashMismatch {
            recorded: artifact.content_hash,
            computed,
        });
    }
    let provenance_matches = matches!(
        (artifact.kind, artifact.gnm_basis_content_hash),
        (LandmarkControlDecoderKind::LandmarkResidual, None)
            | (LandmarkControlDecoderKind::GnmLandmarkUpperBound, Some(_))
    );
    if !provenance_matches
        || artifact.feature_order != landmark_control_feature_order(artifact.kind)
    {
        return Err(LinearPriorLoadError::FeatureOrderMismatch {
            expected: landmark_control_feature_order(artifact.kind),
            found: artifact.feature_order.clone(),
        });
    }
    let map = &artifact.linear_map;
    if features.len() != map.feature_mean.len()
        || map.feature_std.len() != features.len()
        || map.target_mean.len() != ARKIT_NON_TONGUE_CHANNEL_COUNT
        || map.target_std.len() != ARKIT_NON_TONGUE_CHANNEL_COUNT
        || map.weights.len() != ARKIT_NON_TONGUE_CHANNEL_COUNT
        || map.weights.iter().any(|row| row.len() != features.len())
    {
        return Err(LinearPriorLoadError::DimensionMismatch {
            detail: "landmark-control decoder dimensions disagree".to_owned(),
        });
    }
    if features.iter().any(|value| !value.is_finite())
        || map.feature_mean.iter().any(|value| !value.is_finite())
        || map
            .feature_std
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || map.target_mean.iter().any(|value| !value.is_finite())
        || map
            .target_std
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || map.weights.iter().flatten().any(|value| !value.is_finite())
    {
        return Err(LinearPriorLoadError::InvalidNormalization {
            field: "landmark-control feature/normalization".to_owned(),
        });
    }
    let mut prediction = [0.0_f32; ARKIT_NON_TONGUE_CHANNEL_COUNT];
    for (target, output) in prediction.iter_mut().enumerate() {
        let normalized: f32 = features
            .iter()
            .enumerate()
            .map(|(index, value)| {
                (*value - map.feature_mean[index]) / map.feature_std[index]
                    * map.weights[target][index]
            })
            .sum();
        *output = map.target_mean[target] + normalized * map.target_std[target];
    }
    if prediction.iter().any(|value| !value.is_finite()) {
        return Err(LinearPriorLoadError::InvalidNormalization {
            field: "landmark-control prediction".to_owned(),
        });
    }
    Ok(prediction)
}

fn hash_landmark_basis(artifact: &LandmarkAlignedBasisArtifact) -> u64 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&artifact.schema_version.to_le_bytes());
    bytes.extend_from_slice(&(artifact.rank as u64).to_le_bytes());
    bytes.extend_from_slice(&(artifact.source_dimension as u64).to_le_bytes());
    for take in &artifact.training_takes {
        bytes.extend_from_slice(take.as_bytes());
        bytes.push(0xff);
    }
    append_f32(&mut bytes, artifact.source_mean.iter().copied());
    append_f32(&mut bytes, artifact.residual_mean);
    append_f32(&mut bytes, artifact.residual_std);
    for index in &artifact.inactive_residual_channels {
        bytes.extend_from_slice(&(*index as u64).to_le_bytes());
    }
    append_f32(
        &mut bytes,
        artifact
            .singular_values_descending
            .iter()
            .map(|value| *value as f32),
    );
    append_f32(&mut bytes, artifact.basis_row_major.iter().copied());
    fnv1a(&bytes)
}

fn hash_landmark_decoder(artifact: &LandmarkControlDecoderArtifact) -> u64 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&artifact.schema_version.to_le_bytes());
    bytes.push(artifact.kind as u8);
    bytes.extend_from_slice(&artifact.landmark_basis_content_hash.to_le_bytes());
    match artifact.gnm_basis_content_hash {
        Some(hash) => {
            bytes.push(1);
            bytes.extend_from_slice(&hash.to_le_bytes());
        }
        None => bytes.push(0),
    }
    bytes.extend_from_slice(&(artifact.rank as u64).to_le_bytes());
    bytes.extend_from_slice(artifact.feature_order.as_bytes());
    bytes.extend_from_slice(&(artifact.feature_config.history_len as u64).to_le_bytes());
    bytes.extend_from_slice(&artifact.feature_config.max_gap_micros.to_le_bytes());
    for take in &artifact.training_takes {
        bytes.extend_from_slice(take.as_bytes());
        bytes.push(0xff);
    }
    append_f32(
        &mut bytes,
        artifact
            .linear_map
            .feature_mean
            .iter()
            .chain(&artifact.linear_map.feature_std)
            .chain(&artifact.linear_map.target_mean)
            .chain(&artifact.linear_map.target_std)
            .chain(artifact.linear_map.weights.iter().flatten())
            .copied(),
    );
    fnv1a(&bytes)
}

fn append_f32(bytes: &mut Vec<u8>, values: impl IntoIterator<Item = f32>) {
    for value in values {
        let canonical = if value == 0.0 { 0.0 } else { value };
        bytes.extend_from_slice(&canonical.to_bits().to_le_bytes());
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::teacher_residual::apply_non_tongue_residual;
    use vtuber_core::ArkitBlendshape;
    use vtuber_gnm::{FaceRegion, GnmRegionFitRecord};

    fn raw_landmarks(scale: f32, offset: [f32; 2]) -> Vec<[f32; 2]> {
        (0..478)
            .map(|index| {
                let x = (index % 23) as f32 / 23.0;
                let y = (index / 23) as f32 / 21.0;
                [x * scale + offset[0], y * scale + offset[1]]
            })
            .collect()
    }

    #[test]
    fn normalization_removes_translation_and_positive_scale_in_fixed_xy_order() {
        let first = normalize_landmarks_xy(&raw_landmarks(1.0, [0.0, 0.0])).unwrap();
        let transformed = normalize_landmarks_xy(&raw_landmarks(3.5, [8.0, -4.0])).unwrap();
        for (left, right) in first.iter().zip(transformed) {
            assert!((left - right).abs() < 1.0e-5);
        }
        assert_eq!(first.len(), 956);
        assert!(first[0] < 0.0);
        assert!(first[1] < 0.0);
        assert!(matches!(
            normalize_landmarks_xy(&vec![[0.5, 0.5]; 478]),
            Err(LandmarkControlError::DegenerateScale)
        ));
    }

    fn alignment_sample(take: &str, frame: u64, x: f32, y: f32) -> LandmarkAlignmentSample {
        let mut normalized_xy = [0.0; NORMALIZED_LANDMARK_XY_DIM];
        normalized_xy[0] = x;
        normalized_xy[1] = y;
        let mut teacher_residual = [0.0; ARKIT_NON_TONGUE_CHANNEL_COUNT];
        teacher_residual[0] = x;
        teacher_residual[1] = y;
        LandmarkAlignmentSample {
            take_id: take.to_owned(),
            frame_seq: frame,
            normalized_xy,
            teacher_residual,
        }
    }

    fn landmark_basis() -> LandmarkAlignedBasisArtifact {
        fit_landmark_aligned_basis(
            &[
                alignment_sample("train", 1, -2.0, 1.0),
                alignment_sample("train", 2, -1.0, -1.0),
                alignment_sample("train", 3, 1.0, -1.0),
                alignment_sample("train", 4, 2.0, 1.0),
            ],
            &BTreeSet::from(["train".to_owned()]),
            2,
        )
        .unwrap()
    }

    #[test]
    fn basis_is_training_only_orthonormal_deterministic_and_json_stable() {
        let training = vec![
            alignment_sample("train", 1, -2.0, 1.0),
            alignment_sample("train", 2, -1.0, -1.0),
            alignment_sample("train", 3, 1.0, -1.0),
            alignment_sample("train", 4, 2.0, 1.0),
        ];
        let mut with_eval = training.clone();
        with_eval.push(alignment_sample("eval", 5, 100.0, 100.0));
        let takes = BTreeSet::from(["train".to_owned()]);
        let first = fit_landmark_aligned_basis(&training, &takes, 2).unwrap();
        let second = fit_landmark_aligned_basis(&with_eval, &takes, 2).unwrap();
        assert_eq!(first, second);
        for left in 0..2 {
            for right in 0..2 {
                let dot: f32 = (0..NORMALIZED_LANDMARK_XY_DIM)
                    .map(|row| {
                        first.basis_row_major[row * 2 + left]
                            * first.basis_row_major[row * 2 + right]
                    })
                    .sum();
                assert!((dot - if left == right { 1.0 } else { 0.0 }).abs() < 1.0e-5);
            }
        }
        let json = serde_json::to_vec(&first).unwrap();
        let loaded: LandmarkAlignedBasisArtifact = serde_json::from_slice(&json).unwrap();
        project_landmark_latent(&training[0].normalized_xy, &loaded).unwrap();
    }

    fn coefficients(value: f32) -> Arkit52Coefficients {
        let mut values = [0.0; 52];
        values[ArkitBlendshape::JawOpen.index()] = value;
        Arkit52Coefficients::try_from_array(values).unwrap()
    }

    fn landmark_frame(seq: u64, latent: [f32; 2], direct: f32) -> LandmarkControlFrame {
        LandmarkControlFrame {
            frame_seq: seq,
            timestamp_micros: seq * 10_000,
            landmark_latent: latent.to_vec(),
            direct: coefficients(direct),
        }
    }

    fn gnm_frame(seq: u64, reduced: [f32; 2], direct: f32) -> GnmSemanticFrame {
        GnmSemanticFrame {
            frame_seq: seq,
            timestamp_micros: seq * 10_000,
            reduced_expression: reduced.to_vec(),
            joint_rotations: vec![[0.1, 0.2, 0.3]],
            rigid_yaw_pitch_roll: [0.4, 0.5, 0.6],
            objective: 0.7,
            region_fits: [
                FaceRegion::Contour,
                FaceRegion::Brow,
                FaceRegion::Eye,
                FaceRegion::Nose,
                FaceRegion::Mouth,
                FaceRegion::Iris,
                FaceRegion::Other,
            ]
            .into_iter()
            .map(|region| GnmRegionFitRecord {
                region,
                valid_points: 478,
                weighted_rms: 0.1,
            })
            .collect(),
            direct: coefficients(direct),
        }
    }

    fn decoder_row(take: &str, frame_seq: u64, value: f32) -> GnmSemanticRow {
        let mut target = [0.0; ARKIT_NON_TONGUE_CHANNEL_COUNT];
        target[0] = value;
        target[1] = value * value;
        GnmSemanticRow {
            take_id: take.to_owned(),
            frame_seq,
            feature_config: GnmSemanticFeatureConfig {
                history_len: 2,
                max_gap_micros: 20_000,
            },
            features: vec![value, value * value],
            target,
        }
    }

    #[test]
    fn decoder_is_training_only_deterministic_and_json_stable() {
        let training = vec![
            decoder_row("train", 1, -2.0),
            decoder_row("train", 2, -1.0),
            decoder_row("train", 3, 1.0),
            decoder_row("train", 4, 2.0),
        ];
        let mut with_eval = training.clone();
        with_eval.push(decoder_row("eval", 5, 100.0));
        let takes = BTreeSet::from(["train".to_owned()]);
        let basis = landmark_basis();
        let first = fit_landmark_control_decoder(
            &training,
            &takes,
            LandmarkControlDecoderKind::LandmarkResidual,
            &basis,
            None,
            LinearPriorTrainingConfig::default(),
        )
        .unwrap();
        let second = fit_landmark_control_decoder(
            &with_eval,
            &takes,
            LandmarkControlDecoderKind::LandmarkResidual,
            &basis,
            None,
            LinearPriorTrainingConfig::default(),
        )
        .unwrap();
        assert_eq!(first, second);
        let json = serde_json::to_vec(&first).unwrap();
        let loaded: LandmarkControlDecoderArtifact = serde_json::from_slice(&json).unwrap();
        predict_landmark_control_raw(&loaded, &[0.5, 0.25]).unwrap();
    }

    #[test]
    fn l_has_no_gnm_padding_and_hl_requires_exact_identity() {
        let config = GnmSemanticFeatureConfig {
            history_len: 2,
            max_gap_micros: 20_000,
        };
        let landmarks = [
            landmark_frame(2, [2.0, 4.0], 0.6),
            landmark_frame(1, [1.0, 2.0], 0.2),
        ];
        let l = build_landmark_residual_features(&landmarks, config)
            .unwrap()
            .unwrap();
        assert_eq!(l.len(), 160);
        let gnm = [gnm_frame(2, [2.0, 4.0], 0.6), gnm_frame(1, [1.0, 2.0], 0.2)];
        let hl = build_gnm_landmark_upper_bound_features(&gnm, &landmarks, config)
            .unwrap()
            .unwrap();
        assert_eq!(hl.len(), 208);
        let mut mismatched = landmarks.clone();
        mismatched[0].timestamp_micros += 1;
        assert_eq!(
            build_gnm_landmark_upper_bound_features(&gnm, &mismatched, config),
            Err(LandmarkControlError::IdentityMismatch)
        );
    }

    #[test]
    fn final_l_residual_boundary_zeroes_tongue() {
        let mut residual = [0.0; ARKIT_NON_TONGUE_CHANNEL_COUNT];
        residual[ArkitBlendshape::JawOpen.index()] = -0.4;
        let output = apply_non_tongue_residual(&coefficients(0.2), residual).unwrap();
        assert_eq!(output.get(ArkitBlendshape::JawOpen), 0.0);
        assert_eq!(output.get(ArkitBlendshape::TongueOut), 0.0);
        let basis = landmark_basis();
        assert_eq!(basis.rank, 2);
    }
}
