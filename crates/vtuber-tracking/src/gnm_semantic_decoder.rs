//! Reduced-GNM semantic decoder datasets and artifacts (Issue #17).

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use vtuber_core::{
    ARKIT_NON_TONGUE_CHANNEL_COUNT, Arkit52Coefficients, Arkit52ValueError,
    arkit_non_tongue_values, arkit52_with_zero_tongue,
};
use vtuber_gnm::{FaceRegion, GnmRegionFitRecord};

use crate::arkit_teacher::{PairedTemporalSample, TeacherDatasetError, validate_paired_samples};
use crate::causal_prior::{
    LinearPriorFitError, LinearPriorTrainingConfig, fit_normalized_multi_output_ridge,
};
use crate::causal_prior_inference::LinearPriorLoadError;
use crate::teacher_aligned_basis::{
    TeacherAlignedBasisError, TeacherAlignedGnmBasisArtifact, project_teacher_aligned_expression,
    reconstruct_teacher_aligned_expression,
};
use crate::teacher_residual::NormalizedLinearMapArtifact;

/// Stable diagnostic-region order in every semantic feature slot.
pub const GNM_DIAGNOSTIC_REGION_ORDER: [FaceRegion; 7] = [
    FaceRegion::Contour,
    FaceRegion::Brow,
    FaceRegion::Eye,
    FaceRegion::Nose,
    FaceRegion::Mouth,
    FaceRegion::Iris,
    FaceRegion::Other,
];

/// Semantic decoder artifact schema version.
pub const GNM_SEMANTIC_DECODER_SCHEMA_VERSION: u32 = 1;

/// Fixed semantic decoder variants.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum GnmSemanticDecoderKind {
    /// Predict teacher non-tongue ARKit channels from reduced GNM features.
    GnmOnly,
    /// Predict the signed teacher-minus-Direct residual with Direct as a skip connection.
    HybridResidual,
}

/// Causal semantic feature settings.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GnmSemanticFeatureConfig {
    /// Number of newest-first current/past frame slots.
    pub history_len: usize,
    /// Largest permitted timestamp gap within one history.
    pub max_gap_micros: u64,
}

/// Engine-neutral reduced GNM record for one trace-v2 frame.
#[derive(Clone, Debug, PartialEq)]
pub struct GnmSemanticFrame {
    /// Source frame sequence.
    pub frame_seq: u64,
    /// Source monotonic timestamp in microseconds.
    pub timestamp_micros: u64,
    /// Teacher-aligned reduced non-tongue expression.
    pub reduced_expression: Vec<f32>,
    /// Raw fitted joint rotations in axis-angle order.
    pub joint_rotations: Vec<[f32; 3]>,
    /// Raw rigid yaw, pitch, and roll.
    pub rigid_yaw_pitch_roll: [f32; 3],
    /// Raw fit objective.
    pub objective: f32,
    /// Seven stable-order regional fit records.
    pub region_fits: Vec<GnmRegionFitRecord>,
    /// Exact-frame MediaPipe Direct coefficients.
    pub direct: Arkit52Coefficients,
}

/// One supervised G1/H decoder row.
#[derive(Clone, Debug, PartialEq)]
pub struct GnmSemanticRow {
    /// Capture take identity.
    pub take_id: String,
    /// Source frame sequence.
    pub frame_seq: u64,
    /// Feature configuration used to construct this row.
    pub feature_config: GnmSemanticFeatureConfig,
    /// Current/past-only semantic features.
    pub features: Vec<f32>,
    /// G1 teacher target or H teacher-minus-Direct target.
    pub target: [f32; ARKIT_NON_TONGUE_CHANNEL_COUNT],
}

/// Typed semantic dataset/feature failure.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum GnmSemanticDatasetError {
    /// Paired trace validation failed.
    #[error("invalid paired trace: {0:?}")]
    TeacherDataset(TeacherDatasetError),
    /// Aligned basis validation or projection failed.
    #[error(transparent)]
    Basis(#[from] TeacherAlignedBasisError),
    /// History length or maximum gap is invalid.
    #[error("invalid GNM semantic feature configuration")]
    InvalidConfig,
    /// History ordering, shape, or continuity is invalid.
    #[error("invalid GNM semantic history: {0}")]
    InvalidHistory(&'static str),
    /// Regional diagnostics do not contain the fixed seven regions exactly once.
    #[error("GNM region diagnostics do not match the fixed order")]
    InvalidRegions,
}

/// Converts one trace-v2 row into an engine-neutral semantic frame.
///
/// # Errors
///
/// Returns an aligned-basis error for an incompatible expression/artifact.
pub fn gnm_semantic_frame_from_sample(
    sample: &PairedTemporalSample,
    basis: &TeacherAlignedGnmBasisArtifact,
) -> Result<Option<GnmSemanticFrame>, GnmSemanticDatasetError> {
    let (Some(gnm), Some(direct)) = (
        sample.gnm_state.as_ref(),
        sample.mediapipe_observation.as_ref(),
    ) else {
        return Ok(None);
    };
    Ok(Some(GnmSemanticFrame {
        frame_seq: sample.frame_seq,
        timestamp_micros: sample.timestamp_micros,
        reduced_expression: project_teacher_aligned_expression(&gnm.expression, basis)?,
        joint_rotations: gnm.joint_rotations.clone(),
        rigid_yaw_pitch_roll: gnm.rigid_yaw_pitch_roll,
        objective: gnm.objective,
        region_fits: gnm.region_fits.clone(),
        direct: direct.direct_coefficients,
    }))
}

/// Stable human-readable feature order for one decoder kind.
#[must_use]
pub fn gnm_semantic_feature_order(kind: GnmSemanticDecoderKind) -> String {
    let direct_tail = match kind {
        GnmSemanticDecoderKind::GnmOnly => "",
        GnmSemanticDecoderKind::HybridResidual => "+direct-non-tongue-51",
    };
    let direct_velocity = match kind {
        GnmSemanticDecoderKind::GnmOnly => "",
        GnmSemanticDecoderKind::HybridResidual => "+direct-velocity-51",
    };
    format!(
        "v1:newest-first(reduced-expression+joint-axis-angle+rigid-ypr+objective+regions(contour,brow,eye,nose,mouth,iris,other)[weighted-rms,valid/478]{direct_tail})+reduced-expression-velocity{direct_velocity}+dt-seconds"
    )
}

fn common_slot_width(rank: usize, joint_count: usize) -> usize {
    rank + joint_count * 3 + 3 + 1 + GNM_DIAGNOSTIC_REGION_ORDER.len() * 2
}

fn feature_width(
    rank: usize,
    joint_count: usize,
    kind: GnmSemanticDecoderKind,
    config: GnmSemanticFeatureConfig,
) -> usize {
    let direct_slot = usize::from(kind == GnmSemanticDecoderKind::HybridResidual)
        * ARKIT_NON_TONGUE_CHANNEL_COUNT;
    let direct_velocity = direct_slot;
    config.history_len * (common_slot_width(rank, joint_count) + direct_slot)
        + rank
        + direct_velocity
        + 1
}

/// Builds one fixed-order current/past feature vector.
///
/// `newest_first_history[0]` is the current frame. No future-frame input exists.
///
/// # Errors
///
/// Rejects malformed ranks, joints, regions, or non-descending/over-gap history.
#[allow(clippy::indexing_slicing)]
pub fn build_gnm_semantic_features(
    newest_first_history: &[GnmSemanticFrame],
    kind: GnmSemanticDecoderKind,
    config: GnmSemanticFeatureConfig,
) -> Result<Option<Vec<f32>>, GnmSemanticDatasetError> {
    if config.history_len == 0 || config.max_gap_micros == 0 {
        return Err(GnmSemanticDatasetError::InvalidConfig);
    }
    let required = config.history_len.max(2);
    if newest_first_history.len() < required {
        return Ok(None);
    }
    let current = &newest_first_history[0];
    let rank = current.reduced_expression.len();
    let joint_count = current.joint_rotations.len();
    if rank == 0 {
        return Err(GnmSemanticDatasetError::InvalidHistory("zero rank"));
    }
    for (index, frame) in newest_first_history.iter().take(required).enumerate() {
        if frame.reduced_expression.len() != rank || frame.joint_rotations.len() != joint_count {
            return Err(GnmSemanticDatasetError::InvalidHistory(
                "rank or joint count changed",
            ));
        }
        validate_regions(&frame.region_fits)?;
        if frame
            .reduced_expression
            .iter()
            .chain(frame.joint_rotations.iter().flatten())
            .chain(&frame.rigid_yaw_pitch_roll)
            .chain(std::iter::once(&frame.objective))
            .any(|value| !value.is_finite())
        {
            return Err(GnmSemanticDatasetError::InvalidHistory("non-finite value"));
        }
        if index > 0 {
            let newer = &newest_first_history[index - 1];
            if newer.frame_seq != frame.frame_seq + 1
                || newer.timestamp_micros <= frame.timestamp_micros
                || newer.timestamp_micros - frame.timestamp_micros > config.max_gap_micros
            {
                return Err(GnmSemanticDatasetError::InvalidHistory(
                    "sequence or timestamp gap",
                ));
            }
        }
    }

    let direct_tail = usize::from(kind == GnmSemanticDecoderKind::HybridResidual)
        * ARKIT_NON_TONGUE_CHANNEL_COUNT;
    let slot_width = common_slot_width(rank, joint_count) + direct_tail;
    let mut features = vec![0.0_f32; feature_width(rank, joint_count, kind, config)];
    for (slot, frame) in newest_first_history
        .iter()
        .take(config.history_len)
        .enumerate()
    {
        let mut offset = slot * slot_width;
        features[offset..offset + rank].copy_from_slice(&frame.reduced_expression);
        offset += rank;
        for rotation in &frame.joint_rotations {
            features[offset..offset + 3].copy_from_slice(rotation);
            offset += 3;
        }
        features[offset..offset + 3].copy_from_slice(&frame.rigid_yaw_pitch_roll);
        offset += 3;
        features[offset] = frame.objective;
        offset += 1;
        for region in GNM_DIAGNOSTIC_REGION_ORDER {
            let record = frame
                .region_fits
                .iter()
                .find(|record| record.region == region)
                .ok_or(GnmSemanticDatasetError::InvalidRegions)?;
            features[offset] = record.weighted_rms;
            features[offset + 1] = record.valid_points as f32 / 478.0;
            offset += 2;
        }
        if kind == GnmSemanticDecoderKind::HybridResidual {
            features[offset..offset + ARKIT_NON_TONGUE_CHANNEL_COUNT]
                .copy_from_slice(&arkit_non_tongue_values(&frame.direct));
        }
    }
    let previous = &newest_first_history[1];
    let dt_seconds = (current.timestamp_micros - previous.timestamp_micros) as f32 / 1_000_000.0;
    let velocity_base = config.history_len * slot_width;
    for index in 0..rank {
        features[velocity_base + index] =
            (current.reduced_expression[index] - previous.reduced_expression[index]) / dt_seconds;
    }
    let mut dt_index = velocity_base + rank;
    if kind == GnmSemanticDecoderKind::HybridResidual {
        let current_direct = arkit_non_tongue_values(&current.direct);
        let previous_direct = arkit_non_tongue_values(&previous.direct);
        for index in 0..ARKIT_NON_TONGUE_CHANNEL_COUNT {
            features[dt_index + index] =
                (current_direct[index] - previous_direct[index]) / dt_seconds;
        }
        dt_index += ARKIT_NON_TONGUE_CHANNEL_COUNT;
    }
    features[dt_index] = dt_seconds;
    if features.iter().any(|value| !value.is_finite()) {
        return Err(GnmSemanticDatasetError::InvalidHistory(
            "non-finite feature",
        ));
    }
    Ok(Some(features))
}

fn validate_regions(region_fits: &[GnmRegionFitRecord]) -> Result<(), GnmSemanticDatasetError> {
    if region_fits.len() != GNM_DIAGNOSTIC_REGION_ORDER.len()
        || GNM_DIAGNOSTIC_REGION_ORDER.iter().any(|region| {
            region_fits
                .iter()
                .filter(|record| record.region == *region)
                .count()
                != 1
        })
        || region_fits
            .iter()
            .any(|record| !record.weighted_rms.is_finite())
    {
        return Err(GnmSemanticDatasetError::InvalidRegions);
    }
    Ok(())
}

/// Builds causal G1 or H rows, clearing history at missing or discontinuous frames.
///
/// # Errors
///
/// Returns typed source, basis, configuration, and feature validation failures.
pub fn build_gnm_semantic_rows(
    take_id: &str,
    samples: &[PairedTemporalSample],
    basis: &TeacherAlignedGnmBasisArtifact,
    kind: GnmSemanticDecoderKind,
    config: GnmSemanticFeatureConfig,
) -> Result<Vec<GnmSemanticRow>, GnmSemanticDatasetError> {
    if config.history_len == 0 || config.max_gap_micros == 0 {
        return Err(GnmSemanticDatasetError::InvalidConfig);
    }
    validate_paired_samples(samples).map_err(GnmSemanticDatasetError::TeacherDataset)?;
    let mut rows = Vec::new();
    let mut history = Vec::<GnmSemanticFrame>::new();
    for sample in samples {
        let Some(teacher) = sample.teacher.as_ref() else {
            history.clear();
            continue;
        };
        let Some(frame) = gnm_semantic_frame_from_sample(sample, basis)? else {
            history.clear();
            continue;
        };
        if history.first().is_some_and(|previous| {
            sample.frame_seq != previous.frame_seq + 1
                || sample.timestamp_micros <= previous.timestamp_micros
                || sample.timestamp_micros - previous.timestamp_micros > config.max_gap_micros
        }) {
            history.clear();
        }
        history.insert(0, frame);
        if let Some(features) = build_gnm_semantic_features(&history, kind, config)? {
            let teacher_values = arkit_non_tongue_values(&teacher.coefficients);
            let direct_values = history
                .first()
                .map(|current| arkit_non_tongue_values(&current.direct))
                .ok_or(GnmSemanticDatasetError::InvalidHistory(
                    "current frame is absent",
                ))?;
            let mut target = teacher_values;
            if kind == GnmSemanticDecoderKind::HybridResidual {
                for (value, direct) in target.iter_mut().zip(direct_values) {
                    *value -= direct;
                }
            }
            rows.push(GnmSemanticRow {
                take_id: take_id.to_owned(),
                frame_seq: sample.frame_seq,
                feature_config: config,
                features,
                target,
            });
        }
        history.truncate(config.history_len.max(2));
    }
    Ok(rows)
}

/// Versioned normalized ridge decoder for G1 or H.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GnmSemanticDecoderArtifact {
    /// Artifact schema version.
    pub schema_version: u32,
    /// G1 or H target/output semantics.
    pub kind: GnmSemanticDecoderKind,
    /// Content hash of the teacher-aligned basis.
    pub aligned_basis_content_hash: u64,
    /// SHA-256 of the GNM model.
    pub model_sha256: String,
    /// Teacher-aligned expression rank.
    pub rank: usize,
    /// GNM joint count encoded in each slot.
    pub joint_count: usize,
    /// Exact stable feature order.
    pub feature_order: String,
    /// Causal history configuration.
    pub feature_config: GnmSemanticFeatureConfig,
    /// Ordered training take ids.
    pub training_takes: Vec<String>,
    /// Shared normalized ridge map.
    pub linear_map: NormalizedLinearMapArtifact,
    /// Deterministic hash over every preceding field.
    pub content_hash: u64,
}

/// Typed G1/H fit failure.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum GnmSemanticFitError {
    /// Teacher-aligned basis validation failed.
    #[error("teacher-aligned basis is incompatible: {0}")]
    Basis(TeacherAlignedBasisError),
    /// No selected training rows exist.
    #[error("no GNM semantic rows belong to the selected training takes")]
    EmptyTrainingSet,
    /// Row feature dimensions cannot represent the requested contract.
    #[error("GNM semantic row dimensions do not match kind, basis, and history")]
    DimensionMismatch,
    /// Feature-order string does not match the fixed kind contract.
    #[error("GNM semantic feature order does not match decoder kind")]
    FeatureOrderMismatch,
    /// Shared ridge fitting failed.
    #[error("normalized ridge fit failed: {0:?}")]
    Ridge(LinearPriorFitError),
}

/// Fits G1 or H with the existing normalized multi-output ridge kernel.
///
/// # Errors
///
/// Rejects empty selections, contract/dimension mismatch, or shared ridge failures.
pub fn fit_gnm_semantic_decoder(
    rows: &[GnmSemanticRow],
    training_takes: &BTreeSet<String>,
    kind: GnmSemanticDecoderKind,
    basis: &TeacherAlignedGnmBasisArtifact,
    config: LinearPriorTrainingConfig,
    feature_order: &str,
) -> Result<GnmSemanticDecoderArtifact, GnmSemanticFitError> {
    reconstruct_teacher_aligned_expression(&vec![0.0; basis.rank], basis)
        .map_err(GnmSemanticFitError::Basis)?;
    if feature_order != gnm_semantic_feature_order(kind) {
        return Err(GnmSemanticFitError::FeatureOrderMismatch);
    }
    let selected: Vec<&GnmSemanticRow> = rows
        .iter()
        .filter(|row| training_takes.contains(&row.take_id))
        .collect();
    if selected.is_empty() {
        return Err(GnmSemanticFitError::EmptyTrainingSet);
    }
    let first = selected
        .first()
        .ok_or(GnmSemanticFitError::EmptyTrainingSet)?;
    let feature_dimension = first.features.len();
    let feature_config = first.feature_config;
    if selected.iter().any(|row| {
        row.feature_config != feature_config
            || row.features.len() != feature_dimension
            || row.target.iter().any(|value| !value.is_finite())
    }) {
        return Err(GnmSemanticFitError::DimensionMismatch);
    }
    let direct_width = usize::from(kind == GnmSemanticDecoderKind::HybridResidual)
        * ARKIT_NON_TONGUE_CHANNEL_COUNT;
    let fixed = basis.rank + direct_width + 1;
    let slot_numerator = feature_dimension
        .checked_sub(fixed)
        .ok_or(GnmSemanticFitError::DimensionMismatch)?;
    if feature_config.history_len == 0 || slot_numerator % feature_config.history_len != 0 {
        return Err(GnmSemanticFitError::DimensionMismatch);
    }
    let slot_width = slot_numerator / feature_config.history_len;
    let common_width = slot_width
        .checked_sub(direct_width)
        .ok_or(GnmSemanticFitError::DimensionMismatch)?;
    let joint_terms = common_width
        .checked_sub(basis.rank + 3 + 1 + 14)
        .ok_or(GnmSemanticFitError::DimensionMismatch)?;
    if joint_terms % 3 != 0 {
        return Err(GnmSemanticFitError::DimensionMismatch);
    }
    let joint_count = joint_terms / 3;
    if feature_width(basis.rank, joint_count, kind, feature_config) != feature_dimension {
        return Err(GnmSemanticFitError::DimensionMismatch);
    }
    let features: Vec<Vec<f32>> = selected.iter().map(|row| row.features.clone()).collect();
    let targets: Vec<Vec<f32>> = selected.iter().map(|row| row.target.to_vec()).collect();
    let map = fit_normalized_multi_output_ridge(&features, &targets, config)
        .map_err(GnmSemanticFitError::Ridge)?;
    let mut artifact = GnmSemanticDecoderArtifact {
        schema_version: GNM_SEMANTIC_DECODER_SCHEMA_VERSION,
        kind,
        aligned_basis_content_hash: basis.content_hash,
        model_sha256: basis.model_sha256.clone(),
        rank: basis.rank,
        joint_count,
        feature_order: feature_order.to_owned(),
        feature_config,
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
    artifact.content_hash = hash_decoder(&artifact);
    Ok(artifact)
}

/// Predicts unclamped G1 channels or signed H residuals.
///
/// # Errors
///
/// Rejects schema/hash/order/dimension and non-finite normalization failures.
#[allow(clippy::indexing_slicing)]
pub fn predict_gnm_semantic_raw(
    artifact: &GnmSemanticDecoderArtifact,
    features: &[f32],
) -> Result<[f32; ARKIT_NON_TONGUE_CHANNEL_COUNT], LinearPriorLoadError> {
    if artifact.schema_version != GNM_SEMANTIC_DECODER_SCHEMA_VERSION {
        return Err(LinearPriorLoadError::UnsupportedSchemaVersion {
            found: artifact.schema_version,
        });
    }
    let computed = hash_decoder(artifact);
    if computed != artifact.content_hash {
        return Err(LinearPriorLoadError::ContentHashMismatch {
            recorded: artifact.content_hash,
            computed,
        });
    }
    if artifact.feature_order != gnm_semantic_feature_order(artifact.kind) {
        return Err(LinearPriorLoadError::FeatureOrderMismatch {
            expected: gnm_semantic_feature_order(artifact.kind),
            found: artifact.feature_order.clone(),
        });
    }
    let map = &artifact.linear_map;
    let expected_feature_width = feature_width(
        artifact.rank,
        artifact.joint_count,
        artifact.kind,
        artifact.feature_config,
    );
    if features.len() != expected_feature_width
        || features.len() != map.feature_mean.len()
        || map.feature_std.len() != features.len()
        || map.target_mean.len() != ARKIT_NON_TONGUE_CHANNEL_COUNT
        || map.target_std.len() != ARKIT_NON_TONGUE_CHANNEL_COUNT
        || map.weights.len() != ARKIT_NON_TONGUE_CHANNEL_COUNT
        || map.weights.iter().any(|row| row.len() != features.len())
    {
        return Err(LinearPriorLoadError::DimensionMismatch {
            detail: "GNM semantic decoder dimensions disagree".to_owned(),
        });
    }
    if features.iter().any(|value| !value.is_finite())
        || map
            .feature_std
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || map
            .target_std
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || map.target_mean.iter().any(|value| !value.is_finite())
        || map.feature_mean.iter().any(|value| !value.is_finite())
        || map.weights.iter().flatten().any(|value| !value.is_finite())
    {
        return Err(LinearPriorLoadError::InvalidNormalization {
            field: "GNM semantic decoder feature/normalization".to_owned(),
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
            field: "GNM semantic decoder prediction".to_owned(),
        });
    }
    Ok(prediction)
}

/// Clamps a raw G1 prediction only at the final ARKit52 boundary.
///
/// # Errors
///
/// Returns an ARKit value error when an input is non-finite.
pub fn gnm_only_prediction_to_arkit52(
    mut raw: [f32; ARKIT_NON_TONGUE_CHANNEL_COUNT],
) -> Result<Arkit52Coefficients, Arkit52ValueError> {
    for value in &mut raw {
        *value = value.clamp(0.0, 1.0);
    }
    arkit52_with_zero_tongue(raw)
}

fn hash_decoder(artifact: &GnmSemanticDecoderArtifact) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut update = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    update(&artifact.schema_version.to_le_bytes());
    update(&[artifact.kind as u8]);
    update(&artifact.aligned_basis_content_hash.to_le_bytes());
    update(artifact.model_sha256.as_bytes());
    update(&(artifact.rank as u64).to_le_bytes());
    update(&(artifact.joint_count as u64).to_le_bytes());
    update(artifact.feature_order.as_bytes());
    update(&(artifact.feature_config.history_len as u64).to_le_bytes());
    update(&artifact.feature_config.max_gap_micros.to_le_bytes());
    for take in &artifact.training_takes {
        update(take.as_bytes());
        update(&[0xff]);
    }
    for value in artifact
        .linear_map
        .feature_mean
        .iter()
        .chain(&artifact.linear_map.feature_std)
        .chain(&artifact.linear_map.target_mean)
        .chain(&artifact.linear_map.target_std)
        .chain(artifact.linear_map.weights.iter().flatten())
    {
        let canonical = if *value == 0.0 { 0.0 } else { *value };
        update(&canonical.to_bits().to_le_bytes());
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observable_basis::{ObservableBasisProvenance, fit_observable_gnm_basis};
    use crate::teacher_aligned_basis::{TeacherAlignmentSample, fit_teacher_aligned_gnm_basis};
    use crate::teacher_residual::apply_non_tongue_residual;
    use vtuber_core::ArkitBlendshape;
    use vtuber_gnm::{GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM, GnmNonTongueExpression};

    fn aligned_basis() -> TeacherAlignedGnmBasisArtifact {
        let dimension = GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM;
        let mut gram = vec![0.0; dimension * (dimension + 1) / 2];
        gram[0] = 2.0;
        gram[2] = 1.0;
        let observable = fit_observable_gnm_basis(
            &gram,
            2,
            2,
            ObservableBasisProvenance {
                model_sha256: "MODEL".to_owned(),
                mapping_schema_revision: 1,
                training_takes: vec!["geometry".to_owned()],
            },
        )
        .unwrap();
        let samples: Vec<TeacherAlignmentSample> = [-1.0_f32, 1.0]
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                let mut expression = vec![0.0; dimension];
                expression[index] = value;
                let mut teacher_residual = [0.0; ARKIT_NON_TONGUE_CHANNEL_COUNT];
                teacher_residual[index] = value;
                TeacherAlignmentSample {
                    take_id: "align".to_owned(),
                    frame_seq: index as u64,
                    expression: GnmNonTongueExpression::try_from_values(expression).unwrap(),
                    teacher_residual,
                }
            })
            .collect();
        fit_teacher_aligned_gnm_basis(
            &observable,
            &samples,
            &BTreeSet::from(["align".to_owned()]),
            2,
        )
        .unwrap()
    }

    fn coefficients(value: f32) -> Arkit52Coefficients {
        let mut values = [0.0; 52];
        values[ArkitBlendshape::JawOpen.index()] = value;
        Arkit52Coefficients::try_from_array(values).unwrap()
    }

    fn regions(value: f32) -> Vec<GnmRegionFitRecord> {
        GNM_DIAGNOSTIC_REGION_ORDER
            .into_iter()
            .map(|region| GnmRegionFitRecord {
                region,
                valid_points: 239,
                weighted_rms: value,
            })
            .collect()
    }

    fn frame(seq: u64, reduced: [f32; 2], direct: f32) -> GnmSemanticFrame {
        GnmSemanticFrame {
            frame_seq: seq,
            timestamp_micros: seq * 10_000,
            reduced_expression: reduced.to_vec(),
            joint_rotations: vec![[0.1, 0.2, 0.3]],
            rigid_yaw_pitch_roll: [0.4, 0.5, 0.6],
            objective: 0.7,
            region_fits: regions(0.8),
            direct: coefficients(direct),
        }
    }

    #[test]
    fn feature_layout_is_fixed_and_hybrid_adds_only_direct_slot_and_velocity() {
        let config = GnmSemanticFeatureConfig {
            history_len: 2,
            max_gap_micros: 20_000,
        };
        let history = [frame(2, [2.0, 4.0], 0.6), frame(1, [1.0, 2.0], 0.2)];
        let gnm = build_gnm_semantic_features(&history, GnmSemanticDecoderKind::GnmOnly, config)
            .unwrap()
            .unwrap();
        let hybrid =
            build_gnm_semantic_features(&history, GnmSemanticDecoderKind::HybridResidual, config)
                .unwrap()
                .unwrap();
        assert_eq!(gnm.len(), 49);
        assert_eq!(hybrid.len(), 202);
        assert_eq!(&gnm[..9], &[2.0, 4.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]);
        assert_eq!(gnm[9], 0.8);
        assert_eq!(gnm[10], 0.5);
        assert_eq!(&gnm[46..], &[100.0, 200.0, 0.01]);
    }

    fn decoder_rows(kind: GnmSemanticDecoderKind) -> Vec<GnmSemanticRow> {
        let config = GnmSemanticFeatureConfig {
            history_len: 1,
            max_gap_micros: 20_000,
        };
        (-10..=10)
            .map(|index| {
                let x = index as f32 / 10.0;
                let mut features = vec![
                    0.0;
                    match kind {
                        GnmSemanticDecoderKind::GnmOnly => 26,
                        GnmSemanticDecoderKind::HybridResidual => 128,
                    }
                ];
                features[0] = x;
                let mut target = [0.0; ARKIT_NON_TONGUE_CHANNEL_COUNT];
                target[0] = match kind {
                    GnmSemanticDecoderKind::GnmOnly => 0.5 + 0.25 * x,
                    GnmSemanticDecoderKind::HybridResidual => -0.2 * x,
                };
                GnmSemanticRow {
                    take_id: "train".to_owned(),
                    frame_seq: (index + 10) as u64,
                    feature_config: config,
                    features,
                    target,
                }
            })
            .collect()
    }

    #[test]
    fn shared_ridge_recovers_gnm_only_and_signed_hybrid_targets() {
        let basis = aligned_basis();
        for kind in [
            GnmSemanticDecoderKind::GnmOnly,
            GnmSemanticDecoderKind::HybridResidual,
        ] {
            let rows = decoder_rows(kind);
            let artifact = fit_gnm_semantic_decoder(
                &rows,
                &BTreeSet::from(["train".to_owned()]),
                kind,
                &basis,
                LinearPriorTrainingConfig {
                    ridge_lambda: 1.0e-4,
                    ..LinearPriorTrainingConfig::default()
                },
                &gnm_semantic_feature_order(kind),
            )
            .unwrap();
            let predicted = predict_gnm_semantic_raw(&artifact, &rows[0].features).unwrap();
            assert!((predicted[0] - rows[0].target[0]).abs() < 1.0e-3);
            let json = serde_json::to_vec(&artifact).unwrap();
            let loaded: GnmSemanticDecoderArtifact = serde_json::from_slice(&json).unwrap();
            assert_eq!(
                predict_gnm_semantic_raw(&loaded, &rows[0].features).unwrap(),
                predicted
            );
            if kind == GnmSemanticDecoderKind::HybridResidual {
                assert!(predicted[0] > 0.0);
                let negative =
                    predict_gnm_semantic_raw(&artifact, &rows.last().unwrap().features).unwrap();
                assert!(negative[0] < 0.0);
                let output = apply_non_tongue_residual(&coefficients(0.9), predicted).unwrap();
                assert_eq!(output.get(ArkitBlendshape::TongueOut), 0.0);
            } else {
                let output = gnm_only_prediction_to_arkit52(predicted).unwrap();
                assert_eq!(output.get(ArkitBlendshape::TongueOut), 0.0);
            }
        }
    }

    #[test]
    fn decoder_artifact_rejects_kind_basis_rank_and_joint_tampering() {
        let kind = GnmSemanticDecoderKind::GnmOnly;
        let basis = aligned_basis();
        let rows = decoder_rows(kind);
        let artifact = fit_gnm_semantic_decoder(
            &rows,
            &BTreeSet::from(["train".to_owned()]),
            kind,
            &basis,
            LinearPriorTrainingConfig::default(),
            &gnm_semantic_feature_order(kind),
        )
        .unwrap();
        let mut mutations = Vec::new();
        let mut changed = artifact.clone();
        changed.kind = GnmSemanticDecoderKind::HybridResidual;
        mutations.push(changed);
        let mut changed = artifact.clone();
        changed.aligned_basis_content_hash ^= 1;
        mutations.push(changed);
        let mut changed = artifact.clone();
        changed.rank += 1;
        mutations.push(changed);
        let mut changed = artifact.clone();
        changed.joint_count += 1;
        mutations.push(changed);
        for changed in mutations {
            assert!(matches!(
                predict_gnm_semantic_raw(&changed, &rows[0].features),
                Err(LinearPriorLoadError::ContentHashMismatch { .. })
            ));
        }
    }

    #[test]
    fn gap_and_incomplete_history_do_not_emit_features() {
        let config = GnmSemanticFeatureConfig {
            history_len: 2,
            max_gap_micros: 5_000,
        };
        assert_eq!(
            build_gnm_semantic_features(
                &[frame(1, [0.0; 2], 0.0)],
                GnmSemanticDecoderKind::GnmOnly,
                config
            )
            .unwrap(),
            None
        );
        assert!(matches!(
            build_gnm_semantic_features(
                &[frame(3, [0.0; 2], 0.0), frame(1, [0.0; 2], 0.0)],
                GnmSemanticDecoderKind::GnmOnly,
                config
            ),
            Err(GnmSemanticDatasetError::InvalidHistory(_))
        ));
    }
}
