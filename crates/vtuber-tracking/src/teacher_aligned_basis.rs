//! Teacher-residual-aligned observable GNM basis artifacts (Issue #16).

use std::collections::BTreeSet;

use nalgebra::{DMatrix, linalg::SVD};
use serde::{Deserialize, Serialize};
use vtuber_core::{ARKIT_NON_TONGUE_CHANNEL_COUNT, arkit_non_tongue_values};
use vtuber_gnm::{GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM, GnmNonTongueExpression};

use crate::arkit_teacher::{PairedTemporalSample, TeacherDatasetError, validate_paired_samples};
use crate::observable_basis::{
    ObservableBasisError, ObservableGnmBasisArtifact, project_non_tongue_expression,
};

/// Schema version of [`TeacherAlignedGnmBasisArtifact`].
pub const TEACHER_ALIGNED_GNM_BASIS_SCHEMA_VERSION: u32 = 1;

const INACTIVE_RESIDUAL_STD_THRESHOLD: f32 = 1.0e-6;

pub(crate) struct ResidualNormalization {
    pub(crate) mean: [f32; ARKIT_NON_TONGUE_CHANNEL_COUNT],
    pub(crate) std: [f32; ARKIT_NON_TONGUE_CHANNEL_COUNT],
    pub(crate) inactive_channels: Vec<usize>,
    pub(crate) normalized: Vec<[f32; ARKIT_NON_TONGUE_CHANNEL_COUNT]>,
}

#[allow(clippy::indexing_slicing)]
pub(crate) fn normalize_teacher_residuals(
    residuals: &[[f32; ARKIT_NON_TONGUE_CHANNEL_COUNT]],
) -> Result<ResidualNormalization, TeacherAlignedBasisError> {
    if residuals.is_empty() {
        return Err(TeacherAlignedBasisError::NoTrainingSamples);
    }
    if residuals.iter().flatten().any(|value| !value.is_finite()) {
        return Err(TeacherAlignedBasisError::InvalidNumeric("teacher residual"));
    }
    let count = residuals.len() as f32;
    let mut mean = [0.0_f32; ARKIT_NON_TONGUE_CHANNEL_COUNT];
    for residual in residuals {
        for (mean, value) in mean.iter_mut().zip(residual) {
            *mean += *value / count;
        }
    }
    let mut std = [0.0_f32; ARKIT_NON_TONGUE_CHANNEL_COUNT];
    for residual in residuals {
        for ((variance, value), mean) in std.iter_mut().zip(residual).zip(mean) {
            *variance += (*value - mean).powi(2) / count;
        }
    }
    for value in &mut std {
        *value = value.sqrt();
    }
    let inactive_channels: Vec<usize> = std
        .iter()
        .enumerate()
        .filter_map(|(index, std)| (*std <= INACTIVE_RESIDUAL_STD_THRESHOLD).then_some(index))
        .collect();
    let normalized = residuals
        .iter()
        .map(|residual| {
            let mut row = [0.0_f32; ARKIT_NON_TONGUE_CHANNEL_COUNT];
            for index in 0..ARKIT_NON_TONGUE_CHANNEL_COUNT {
                if std[index] > INACTIVE_RESIDUAL_STD_THRESHOLD {
                    row[index] = (residual[index] - mean[index]) / std[index];
                }
            }
            row
        })
        .collect();
    Ok(ResidualNormalization {
        mean,
        std,
        inactive_channels,
        normalized,
    })
}

/// One exact-frame expression and ARKit-teacher residual pair.
#[derive(Clone, Debug, PartialEq)]
pub struct TeacherAlignmentSample {
    /// Source take id.
    pub take_id: String,
    /// Exact source frame sequence.
    pub frame_seq: u64,
    /// Raw fitted non-tongue Head-v3 expression.
    pub expression: GnmNonTongueExpression,
    /// Same-frame `ARKit teacher - MediaPipe Direct`, excluding TongueOut.
    pub teacher_residual: [f32; ARKIT_NON_TONGUE_CHANNEL_COUNT],
}

/// Builds exact-frame alignment samples without using the hand-projected GNM output.
///
/// # Errors
///
/// Returns the paired trace's first typed validation failure.
pub fn build_teacher_alignment_samples(
    take_id: &str,
    samples: &[PairedTemporalSample],
) -> Result<Vec<TeacherAlignmentSample>, TeacherDatasetError> {
    validate_paired_samples(samples)?;
    let mut output = Vec::new();
    for sample in samples {
        let (Some(teacher), Some(direct), Some(gnm)) = (
            sample.teacher.as_ref(),
            sample.mediapipe_observation.as_ref(),
            sample.gnm_state.as_ref(),
        ) else {
            continue;
        };
        let teacher_values = arkit_non_tongue_values(&teacher.coefficients);
        let direct_values = arkit_non_tongue_values(&direct.direct_coefficients);
        let mut teacher_residual = [0.0_f32; ARKIT_NON_TONGUE_CHANNEL_COUNT];
        for ((residual, teacher), direct) in teacher_residual
            .iter_mut()
            .zip(teacher_values)
            .zip(direct_values)
        {
            *residual = teacher - direct;
        }
        output.push(TeacherAlignmentSample {
            take_id: take_id.to_owned(),
            frame_seq: sample.frame_seq,
            expression: gnm.expression.clone(),
            teacher_residual,
        });
    }
    Ok(output)
}

/// Versioned teacher-aligned non-tongue GNM basis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TeacherAlignedGnmBasisArtifact {
    /// Artifact schema version.
    pub schema_version: u32,
    /// Content hash of the source observable basis.
    pub observable_basis_content_hash: u64,
    /// SHA-256 of the pinned GNM model bytes.
    pub model_sha256: String,
    /// Dense mapping schema revision.
    pub mapping_schema_revision: u32,
    /// Rank of the source observable basis.
    pub source_rank: usize,
    /// Retained teacher-aligned rank.
    pub rank: usize,
    /// Ordered training take ids.
    pub training_takes: Vec<String>,
    /// Training-only residual mean in canonical non-tongue ARKit order.
    #[serde(with = "residual_array")]
    pub residual_mean: [f32; ARKIT_NON_TONGUE_CHANNEL_COUNT],
    /// Training-only residual population standard deviation.
    #[serde(with = "residual_array")]
    pub residual_std: [f32; ARKIT_NON_TONGUE_CHANNEL_COUNT],
    /// Residual channels whose standard deviation is at most `1.0e-6`.
    pub inactive_residual_channels: Vec<usize>,
    /// Cross-covariance singular values in descending order.
    pub singular_values_descending: Vec<f64>,
    /// Row-major `[351, rank]` orthonormal basis.
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

/// Typed teacher-aligned-basis failure.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum TeacherAlignedBasisError {
    /// The source observable artifact is invalid.
    #[error(transparent)]
    Observable(#[from] ObservableBasisError),
    /// Rank is outside the fixed cross-covariance limit.
    #[error("teacher-aligned rank {rank} is outside 1..={maximum}")]
    InvalidRank {
        /// Requested rank.
        rank: usize,
        /// Largest permitted rank.
        maximum: usize,
    },
    /// No selected training samples were supplied.
    #[error("no teacher-alignment samples belong to the selected training takes")]
    NoTrainingSamples,
    /// An input or computed number was non-finite or had no usable norm.
    #[error("invalid teacher-aligned numeric value: {0}")]
    InvalidNumeric(&'static str),
    /// Artifact dimensions or schema do not match the fixed contract.
    #[error("invalid teacher-aligned basis shape: {0}")]
    InvalidShape(&'static str),
    /// Artifact content hash does not match its fields.
    #[error("teacher-aligned basis content hash mismatch")]
    HashMismatch,
    /// Compact expression validation failed.
    #[error(transparent)]
    Expression(#[from] vtuber_gnm::GnmModelError),
}

/// Fits `B = O U_k` from training-only observable coordinates and normalized
/// same-frame teacher residuals.
///
/// # Errors
///
/// Rejects an invalid source artifact, rank, empty training selection, or
/// non-finite numeric input/result.
#[allow(clippy::indexing_slicing)]
pub fn fit_teacher_aligned_gnm_basis(
    observable: &ObservableGnmBasisArtifact,
    samples: &[TeacherAlignmentSample],
    training_takes: &BTreeSet<String>,
    rank: usize,
) -> Result<TeacherAlignedGnmBasisArtifact, TeacherAlignedBasisError> {
    let neutral =
        GnmNonTongueExpression::try_from_values(vec![0.0; GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM])?;
    project_non_tongue_expression(&neutral, observable)?;
    let maximum_rank = observable.rank.min(ARKIT_NON_TONGUE_CHANNEL_COUNT);
    if !(1..=maximum_rank).contains(&rank) {
        return Err(TeacherAlignedBasisError::InvalidRank {
            rank,
            maximum: maximum_rank,
        });
    }
    let training: Vec<&TeacherAlignmentSample> = samples
        .iter()
        .filter(|sample| training_takes.contains(&sample.take_id))
        .collect();
    if training.is_empty() {
        return Err(TeacherAlignedBasisError::NoTrainingSamples);
    }
    let count = training.len() as f64;
    let mut coordinates = Vec::with_capacity(training.len());
    let mut coordinate_mean = vec![0.0_f64; observable.rank];
    for sample in &training {
        let projected = project_non_tongue_expression(&sample.expression, observable)?;
        for (mean, value) in coordinate_mean.iter_mut().zip(&projected) {
            *mean += f64::from(*value) / count;
        }
        coordinates.push(projected);
    }
    let residuals: Vec<[f32; ARKIT_NON_TONGUE_CHANNEL_COUNT]> = training
        .iter()
        .map(|sample| sample.teacher_residual)
        .collect();
    let normalization = normalize_teacher_residuals(&residuals)?;

    let mut cross_covariance =
        DMatrix::<f64>::zeros(observable.rank, ARKIT_NON_TONGUE_CHANNEL_COUNT);
    for (projected, normalized_residual) in coordinates.iter().zip(&normalization.normalized) {
        for source in 0..observable.rank {
            let centered_coordinate = f64::from(projected[source]) - coordinate_mean[source];
            for channel in 0..ARKIT_NON_TONGUE_CHANNEL_COUNT {
                cross_covariance[(source, channel)] +=
                    centered_coordinate * f64::from(normalized_residual[channel]) / count;
            }
        }
    }
    let decomposition = SVD::new(cross_covariance, true, false);
    let left = decomposition
        .u
        .ok_or(TeacherAlignedBasisError::InvalidShape(
            "left singular vectors",
        ))?;
    if decomposition
        .singular_values
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(TeacherAlignedBasisError::InvalidNumeric("singular value"));
    }
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
    let mut basis_row_major = vec![0.0_f32; GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM * rank];
    for (target_column, source_column) in order.iter().take(rank).enumerate() {
        for row in 0..GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM {
            let value: f64 = (0..observable.rank)
                .map(|observable_column| {
                    f64::from(observable.basis_row_major[row * observable.rank + observable_column])
                        * left[(observable_column, *source_column)]
                })
                .sum();
            basis_row_major[row * rank + target_column] = value as f32;
        }
        let norm = (0..GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM)
            .map(|row| f64::from(basis_row_major[row * rank + target_column]).powi(2))
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            return Err(TeacherAlignedBasisError::InvalidNumeric("basis norm"));
        }
        for row in 0..GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM {
            basis_row_major[row * rank + target_column] /= norm as f32;
        }
        let sign = (0..GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM)
            .map(|row| basis_row_major[row * rank + target_column])
            .max_by(|left_value, right_value| left_value.abs().total_cmp(&right_value.abs()))
            .map_or(1.0, |value| if value < 0.0 { -1.0 } else { 1.0 });
        for row in 0..GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM {
            basis_row_major[row * rank + target_column] *= sign;
        }
    }
    let mut artifact = TeacherAlignedGnmBasisArtifact {
        schema_version: TEACHER_ALIGNED_GNM_BASIS_SCHEMA_VERSION,
        observable_basis_content_hash: observable.content_hash,
        model_sha256: observable.model_sha256.clone(),
        mapping_schema_revision: observable.mapping_schema_revision,
        source_rank: observable.rank,
        rank,
        training_takes: training_takes.iter().cloned().collect(),
        residual_mean: normalization.mean,
        residual_std: normalization.std,
        inactive_residual_channels: normalization.inactive_channels,
        singular_values_descending,
        basis_row_major,
        content_hash: 0,
    };
    artifact.content_hash = teacher_aligned_basis_hash(&artifact);
    Ok(artifact)
}

/// Projects `q = B^T φ`.
///
/// # Errors
///
/// Rejects an invalid artifact or non-finite output.
#[allow(clippy::indexing_slicing)]
pub fn project_teacher_aligned_expression(
    expression: &GnmNonTongueExpression,
    basis: &TeacherAlignedGnmBasisArtifact,
) -> Result<Vec<f32>, TeacherAlignedBasisError> {
    validate_artifact(basis)?;
    let mut reduced = vec![0.0_f32; basis.rank];
    for (row, expression_value) in expression.values().iter().enumerate() {
        for (column, reduced_value) in reduced.iter_mut().enumerate() {
            *reduced_value += basis.basis_row_major[row * basis.rank + column] * expression_value;
        }
    }
    if reduced.iter().any(|value| !value.is_finite()) {
        return Err(TeacherAlignedBasisError::InvalidNumeric(
            "projected expression",
        ));
    }
    Ok(reduced)
}

/// Reconstructs `φ_hat = B q`; excluded tongue values remain absent/zero.
///
/// # Errors
///
/// Rejects an invalid artifact, reduced dimension, or expression result.
#[allow(clippy::indexing_slicing)]
pub fn reconstruct_teacher_aligned_expression(
    reduced: &[f32],
    basis: &TeacherAlignedGnmBasisArtifact,
) -> Result<GnmNonTongueExpression, TeacherAlignedBasisError> {
    validate_artifact(basis)?;
    if reduced.len() != basis.rank {
        return Err(TeacherAlignedBasisError::InvalidShape("reduced expression"));
    }
    let mut expression = vec![0.0_f32; GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM];
    for (row, expression_value) in expression.iter_mut().enumerate() {
        for (column, reduced_value) in reduced.iter().enumerate() {
            *expression_value += basis.basis_row_major[row * basis.rank + column] * reduced_value;
        }
    }
    Ok(GnmNonTongueExpression::try_from_values(expression)?)
}

fn validate_artifact(
    basis: &TeacherAlignedGnmBasisArtifact,
) -> Result<(), TeacherAlignedBasisError> {
    if basis.schema_version != TEACHER_ALIGNED_GNM_BASIS_SCHEMA_VERSION
        || !(1..=basis.source_rank.min(ARKIT_NON_TONGUE_CHANNEL_COUNT)).contains(&basis.rank)
        || basis.model_sha256.is_empty()
        || basis.training_takes.is_empty()
        || basis.residual_mean.iter().any(|value| !value.is_finite())
        || basis.residual_std.iter().any(|value| !value.is_finite())
        || basis.singular_values_descending.len() != basis.rank
        || basis
            .singular_values_descending
            .iter()
            .any(|value| !value.is_finite())
        || basis.basis_row_major.len() != GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM * basis.rank
        || basis.basis_row_major.iter().any(|value| !value.is_finite())
    {
        return Err(TeacherAlignedBasisError::InvalidShape("artifact"));
    }
    if teacher_aligned_basis_hash(basis) != basis.content_hash {
        return Err(TeacherAlignedBasisError::HashMismatch);
    }
    Ok(())
}

fn teacher_aligned_basis_hash(artifact: &TeacherAlignedGnmBasisArtifact) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut update = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    update(&artifact.schema_version.to_le_bytes());
    update(&artifact.observable_basis_content_hash.to_le_bytes());
    update(artifact.model_sha256.as_bytes());
    update(&artifact.mapping_schema_revision.to_le_bytes());
    update(&(artifact.source_rank as u64).to_le_bytes());
    update(&(artifact.rank as u64).to_le_bytes());
    for take in &artifact.training_takes {
        update(take.as_bytes());
        update(&[0xff]);
    }
    for value in artifact.residual_mean.iter().chain(&artifact.residual_std) {
        let canonical = if *value == 0.0 { 0.0 } else { *value };
        update(&canonical.to_bits().to_le_bytes());
    }
    for index in &artifact.inactive_residual_channels {
        update(&(*index as u64).to_le_bytes());
    }
    for value in &artifact.singular_values_descending {
        update(&(*value as f32).to_bits().to_le_bytes());
    }
    for value in &artifact.basis_row_major {
        let canonical = if *value == 0.0 { 0.0 } else { *value };
        update(&canonical.to_bits().to_le_bytes());
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arkit_teacher::{
        ArkitTeacherFrame, HeadTransform, PairedTemporalSample, test_gnm_state,
        test_mediapipe_observation,
    };
    use vtuber_core::{Arkit52Coefficients, ArkitBlendshape};

    fn coefficients(channel: ArkitBlendshape, value: f32) -> Arkit52Coefficients {
        let mut values = [0.0; 52];
        values[channel.index()] = value;
        Arkit52Coefficients::try_from_array(values).unwrap()
    }

    fn paired_sample(seq: u64) -> PairedTemporalSample {
        let direct = coefficients(ArkitBlendshape::JawOpen, 0.2);
        let teacher = coefficients(ArkitBlendshape::JawOpen, 0.7);
        PairedTemporalSample {
            frame_seq: seq,
            timestamp_micros: seq * 1_000,
            mediapipe_observation: Some(test_mediapipe_observation(direct)),
            gnm_state: Some(test_gnm_state(
                coefficients(ArkitBlendshape::JawOpen, 0.9),
                4.0,
            )),
            baseline_output: coefficients(ArkitBlendshape::JawOpen, 0.1),
            teacher: Some(ArkitTeacherFrame {
                frame_seq: seq,
                timestamp_micros: seq * 1_000,
                coefficients: teacher,
                head_transform: HeadTransform {
                    rotation_unit_quaternion_wxyz: [1.0, 0.0, 0.0, 0.0],
                    translation_meters: [0.0; 3],
                },
            }),
            rgb_reference: None,
        }
    }

    fn observable(rank: usize) -> ObservableGnmBasisArtifact {
        use crate::observable_basis::{ObservableBasisProvenance, fit_observable_gnm_basis};
        let dimension = GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM;
        let mut gram = vec![0.0; dimension * (dimension + 1) / 2];
        for index in 0..rank {
            gram[index * (index + 1) / 2 + index] = (rank - index) as f64;
        }
        fit_observable_gnm_basis(
            &gram,
            rank,
            rank,
            ObservableBasisProvenance {
                model_sha256: "MODEL".to_owned(),
                mapping_schema_revision: 7,
                training_takes: vec!["geometry".to_owned()],
            },
        )
        .unwrap()
    }

    fn alignment_sample(take: &str, frame: u64, x: f32, y: f32) -> TeacherAlignmentSample {
        let mut expression = vec![0.0; GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM];
        expression[0] = x;
        expression[1] = y;
        let mut teacher_residual = [0.0; ARKIT_NON_TONGUE_CHANNEL_COUNT];
        teacher_residual[0] = 2.0 * x;
        teacher_residual[1] = 0.25 * y;
        TeacherAlignmentSample {
            take_id: take.to_owned(),
            frame_seq: frame,
            expression: GnmNonTongueExpression::try_from_values(expression).unwrap(),
            teacher_residual,
        }
    }

    #[test]
    fn samples_use_same_frame_teacher_minus_direct_and_raw_expression() {
        let mut sample = paired_sample(1);
        let mut expression = vec![0.0; GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM];
        expression[3] = 0.4;
        sample.gnm_state.as_mut().unwrap().expression =
            GnmNonTongueExpression::try_from_values(expression.clone()).unwrap();
        let rows = build_teacher_alignment_samples("take", &[sample]).unwrap();
        assert_eq!(rows.len(), 1);
        assert!((rows[0].teacher_residual[ArkitBlendshape::JawOpen.index()] - 0.5).abs() < 1.0e-6);
        assert_eq!(rows[0].expression.values(), expression);
    }

    #[test]
    fn synthetic_fit_recovers_known_teacher_direction_and_is_orthonormal() {
        let samples = vec![
            alignment_sample("train", 1, -2.0, 1.0),
            alignment_sample("train", 2, -1.0, -1.0),
            alignment_sample("train", 3, 1.0, -1.0),
            alignment_sample("train", 4, 2.0, 1.0),
        ];
        let artifact = fit_teacher_aligned_gnm_basis(
            &observable(3),
            &samples,
            &BTreeSet::from(["train".to_owned()]),
            2,
        )
        .unwrap();
        assert_eq!(
            artifact.inactive_residual_channels,
            (2..51).collect::<Vec<_>>()
        );
        assert!(artifact.singular_values_descending[0] >= artifact.singular_values_descending[1]);
        for left in 0..artifact.rank {
            for right in 0..artifact.rank {
                let dot: f32 = (0..GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM)
                    .map(|row| {
                        artifact.basis_row_major[row * artifact.rank + left]
                            * artifact.basis_row_major[row * artifact.rank + right]
                    })
                    .sum();
                let expected = if left == right { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1.0e-5);
            }
        }
        assert!(artifact.basis_row_major[0].abs() > 0.99);
    }

    #[test]
    fn eval_rows_do_not_change_training_artifact_or_hash() {
        let training = vec![
            alignment_sample("train", 1, -1.0, 0.0),
            alignment_sample("train", 2, 1.0, 0.0),
        ];
        let mut with_eval = training.clone();
        with_eval.push(alignment_sample("eval", 3, 100.0, 100.0));
        let takes = BTreeSet::from(["train".to_owned()]);
        let first = fit_teacher_aligned_gnm_basis(&observable(2), &training, &takes, 1).unwrap();
        let second = fit_teacher_aligned_gnm_basis(&observable(2), &with_eval, &takes, 1).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
    }

    #[test]
    fn projection_and_reconstruction_stay_inside_aligned_basis() {
        let samples = vec![
            alignment_sample("train", 1, -1.0, 0.0),
            alignment_sample("train", 2, 1.0, 0.0),
        ];
        let artifact = fit_teacher_aligned_gnm_basis(
            &observable(2),
            &samples,
            &BTreeSet::from(["train".to_owned()]),
            1,
        )
        .unwrap();
        let reduced =
            project_teacher_aligned_expression(&samples[1].expression, &artifact).unwrap();
        let reconstructed = reconstruct_teacher_aligned_expression(&reduced, &artifact).unwrap();
        assert!((reconstructed.values()[0] - 1.0).abs() < 1.0e-5);
        assert!(reconstructed.values()[1].abs() < 1.0e-5);
    }

    #[test]
    fn runtime_basis_loader_requires_exact_model_mapping_and_hash() {
        let samples = vec![
            alignment_sample("train", 1, -1.0, 0.0),
            alignment_sample("train", 2, 1.0, 0.0),
        ];
        let artifact = fit_teacher_aligned_gnm_basis(
            &observable(2),
            &samples,
            &BTreeSet::from(["train".to_owned()]),
            1,
        )
        .unwrap();
        let loaded = crate::load_reduced_gnm_basis(&artifact, "MODEL", 7).unwrap();
        assert_eq!(loaded.rank(), 1);
        assert_eq!(loaded.values_row_major(), artifact.basis_row_major);
        assert!(crate::load_reduced_gnm_basis(&artifact, "OTHER", 7).is_err());
        assert!(crate::load_reduced_gnm_basis(&artifact, "MODEL", 8).is_err());
        let mut tampered = artifact;
        tampered.basis_row_major[0] += 0.1;
        assert!(crate::load_reduced_gnm_basis(&tampered, "MODEL", 7).is_err());
    }

    #[test]
    fn rank_above_cross_covariance_limit_is_typed_error() {
        let error = fit_teacher_aligned_gnm_basis(
            &observable(52),
            &[alignment_sample("train", 1, 0.0, 0.0)],
            &BTreeSet::from(["train".to_owned()]),
            52,
        )
        .unwrap_err();
        assert_eq!(
            error,
            TeacherAlignedBasisError::InvalidRank {
                rank: 52,
                maximum: 51
            }
        );
    }

    #[test]
    fn artifact_hash_survives_json_float_round_trip() {
        let samples = vec![
            alignment_sample("train", 1, -1.0, 0.0),
            alignment_sample("train", 2, 1.0, 0.0),
        ];
        let mut artifact = fit_teacher_aligned_gnm_basis(
            &observable(2),
            &samples,
            &BTreeSet::from(["train".to_owned()]),
            1,
        )
        .unwrap();
        artifact.singular_values_descending[0] = 1.562_487_917_078_665_9;
        artifact.residual_mean[2] = -0.0;
        artifact.basis_row_major[2] = -0.0;
        artifact.content_hash = teacher_aligned_basis_hash(&artifact);
        let json = serde_json::to_vec(&artifact).unwrap();
        let loaded: TeacherAlignedGnmBasisArtifact = serde_json::from_slice(&json).unwrap();
        validate_artifact(&loaded).unwrap();
    }
}
