//! Same-frame teacher-minus-Direct residual dataset and linear decoder.

use std::collections::{BTreeMap, BTreeSet};

use vtuber_core::{
    ARKIT_NON_TONGUE_CHANNEL_COUNT, Arkit52Coefficients, Arkit52ValueError,
    arkit_non_tongue_values, arkit52_with_zero_tongue,
};

use crate::arkit_teacher::{PairedTemporalSample, TeacherDatasetError};
use crate::causal_prior::{
    LinearPriorFitError, LinearPriorTrainingConfig, fit_normalized_multi_output_ridge,
};
use crate::causal_prior_inference::LinearPriorLoadError;

/// Residual-decoder artifact schema.
pub const TEACHER_RESIDUAL_DECODER_SCHEMA_VERSION: u32 = 1;

/// Feature layout used by the residual decoder.
pub const TEACHER_RESIDUAL_FEATURE_ORDER: &str = "v1:newest-first-history(direct-51+gnm-projected-51+gnm-residual)+velocity(direct-51+gnm-projected-51)+dt-seconds";

const HISTORY_SLOT_WIDTH: usize = ARKIT_NON_TONGUE_CHANNEL_COUNT * 2 + 1;
const VELOCITY_WIDTH: usize = ARKIT_NON_TONGUE_CHANNEL_COUNT * 2;

/// Causal feature construction settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TeacherResidualFeatureConfig {
    /// Number of current/past snapshots in each feature row.
    pub history_len: usize,
    /// Largest timestamp gap retained in one causal run.
    pub max_gap_micros: u64,
}

impl TeacherResidualFeatureConfig {
    /// Exact feature width for this configuration.
    #[must_use]
    pub const fn feature_width(self) -> usize {
        self.history_len * HISTORY_SLOT_WIDTH + VELOCITY_WIDTH + 1
    }
}

/// One same-frame supervised residual row.
#[derive(Clone, Debug, PartialEq)]
pub struct TeacherResidualRow {
    /// Capture take identity.
    pub take_id: String,
    /// Source frame sequence.
    pub frame_seq: u64,
    /// Source timestamp in microseconds.
    pub timestamp_micros: u64,
    /// Current/past features only.
    pub features: Vec<f32>,
    /// `teacher_51 - Direct_51` for this exact frame.
    pub target_residual: [f32; ARKIT_NON_TONGUE_CHANNEL_COUNT],
}

/// Typed reason that a candidate frame was not emitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TeacherResidualExclusion {
    /// MediaPipe Direct coefficients were absent.
    MissingDirect,
    /// Projected GNM state was absent.
    MissingGnmState,
    /// ARKit teacher coefficients were absent.
    MissingTeacher,
    /// Frame identity or timestamp continuity broke.
    SequenceBoundary,
    /// Residual or generated feature was non-finite.
    NonFinite,
}

/// Generated rows and exclusion counts.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TeacherResidualDataset {
    /// Accepted rows in capture order.
    pub rows: Vec<TeacherResidualRow>,
    /// Aggregated typed exclusions.
    pub exclusions: Vec<(TeacherResidualExclusion, usize)>,
}

/// Reusable per-frame feature lookup built by the canonical dataset builder.
#[derive(Clone, Debug, PartialEq)]
pub struct TeacherResidualHistory {
    rows: BTreeMap<(u64, u64), Vec<f32>>,
}

impl TeacherResidualHistory {
    /// Builds causal features once for offline evaluation.
    ///
    /// # Errors
    ///
    /// Returns the canonical dataset error when source pairing is invalid.
    pub fn build(
        take_id: &str,
        samples: &[PairedTemporalSample],
        config: TeacherResidualFeatureConfig,
    ) -> Result<Self, TeacherDatasetError> {
        let dataset = build_teacher_residual_rows(take_id, samples, config)?;
        Ok(Self {
            rows: dataset
                .rows
                .into_iter()
                .map(|row| ((row.frame_seq, row.timestamp_micros), row.features))
                .collect(),
        })
    }

    fn features(&self, sample: &PairedTemporalSample) -> Option<&[f32]> {
        self.rows
            .get(&(sample.frame_seq, sample.timestamp_micros))
            .map(Vec::as_slice)
    }
}

impl TeacherResidualDataset {
    fn exclude(&mut self, reason: TeacherResidualExclusion) {
        if let Some((_, count)) = self.exclusions.iter_mut().find(|(item, _)| *item == reason) {
            *count += 1;
        } else {
            self.exclusions.push((reason, 1));
        }
    }
}

#[derive(Clone, Copy)]
struct Snapshot {
    direct: [f32; ARKIT_NON_TONGUE_CHANNEL_COUNT],
    gnm: [f32; ARKIT_NON_TONGUE_CHANNEL_COUNT],
    residual: f32,
}

/// Builds same-frame teacher residual rows without future-frame features.
///
/// # Errors
///
/// Returns the existing typed teacher-dataset errors for invalid configuration
/// or source pairing.
#[allow(clippy::indexing_slicing)]
pub fn build_teacher_residual_rows(
    take_id: &str,
    samples: &[PairedTemporalSample],
    config: TeacherResidualFeatureConfig,
) -> Result<TeacherResidualDataset, TeacherDatasetError> {
    if config.history_len == 0 {
        return Err(TeacherDatasetError::NonFinite {
            field: "history_len must be positive",
        });
    }
    crate::arkit_teacher::validate_paired_samples(samples)?;

    let mut dataset = TeacherResidualDataset::default();
    let mut history = Vec::<Snapshot>::with_capacity(config.history_len);
    let mut last_identity = None;
    for sample in samples {
        let Some(direct) = sample.mediapipe_observation.as_ref() else {
            dataset.exclude(TeacherResidualExclusion::MissingDirect);
            history.clear();
            last_identity = None;
            continue;
        };
        let Some(gnm_state) = sample.gnm_state.as_ref() else {
            dataset.exclude(TeacherResidualExclusion::MissingGnmState);
            history.clear();
            last_identity = None;
            continue;
        };
        let Some(teacher) = sample.teacher.as_ref() else {
            dataset.exclude(TeacherResidualExclusion::MissingTeacher);
            history.clear();
            last_identity = None;
            continue;
        };

        let continuous = matches!(last_identity, Some((seq, time))
            if sample.frame_seq == seq + 1
                && sample.timestamp_micros.saturating_sub(time) <= config.max_gap_micros);
        if last_identity.is_some() && !continuous {
            dataset.exclude(TeacherResidualExclusion::SequenceBoundary);
            history.clear();
        }

        let snapshot = Snapshot {
            direct: arkit_non_tongue_values(direct),
            gnm: arkit_non_tongue_values(&gnm_state.projected_coefficients),
            residual: gnm_state.residual,
        };
        let mut run = history.clone();
        run.push(snapshot);
        let mut features = vec![0.0; config.feature_width()];
        for (slot, item) in run.iter().rev().take(config.history_len).enumerate() {
            let base = slot * HISTORY_SLOT_WIDTH;
            features[base..base + ARKIT_NON_TONGUE_CHANNEL_COUNT].copy_from_slice(&item.direct);
            let gnm_base = base + ARKIT_NON_TONGUE_CHANNEL_COUNT;
            features[gnm_base..gnm_base + ARKIT_NON_TONGUE_CHANNEL_COUNT]
                .copy_from_slice(&item.gnm);
            features[base + HISTORY_SLOT_WIDTH - 1] = item.residual;
        }

        let velocity_base = config.history_len * HISTORY_SLOT_WIDTH;
        let dt_micros = if continuous {
            last_identity.map_or(0, |(_, time)| sample.timestamp_micros.saturating_sub(time))
        } else {
            0
        };
        let dt_seconds = dt_micros as f32 / 1_000_000.0;
        if continuous && let Some(previous) = history.last() {
            for index in 0..ARKIT_NON_TONGUE_CHANNEL_COUNT {
                features[velocity_base + index] =
                    (snapshot.direct[index] - previous.direct[index]) / dt_seconds;
                features[velocity_base + ARKIT_NON_TONGUE_CHANNEL_COUNT + index] =
                    (snapshot.gnm[index] - previous.gnm[index]) / dt_seconds;
            }
        }
        features[velocity_base + VELOCITY_WIDTH] = dt_seconds;

        let teacher_values = arkit_non_tongue_values(&teacher.coefficients);
        let mut target_residual = [0.0; ARKIT_NON_TONGUE_CHANNEL_COUNT];
        for index in 0..ARKIT_NON_TONGUE_CHANNEL_COUNT {
            target_residual[index] = teacher_values[index] - snapshot.direct[index];
        }
        if !snapshot.residual.is_finite()
            || features.iter().any(|value| !value.is_finite())
            || target_residual.iter().any(|value| !value.is_finite())
        {
            dataset.exclude(TeacherResidualExclusion::NonFinite);
            history.clear();
            last_identity = None;
            continue;
        }
        dataset.rows.push(TeacherResidualRow {
            take_id: take_id.to_owned(),
            frame_seq: sample.frame_seq,
            timestamp_micros: sample.timestamp_micros,
            features,
            target_residual,
        });
        history.push(snapshot);
        if history.len() > config.history_len {
            history.remove(0);
        }
        last_identity = Some((sample.frame_seq, sample.timestamp_micros));
    }
    Ok(dataset)
}

/// Serializable normalized map embedded in the decoder artifact.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NormalizedLinearMapArtifact {
    /// Per-feature training mean.
    pub feature_mean: Vec<f32>,
    /// Per-feature training standard deviation.
    pub feature_std: Vec<f32>,
    /// Per-target training mean.
    pub target_mean: Vec<f32>,
    /// Per-target training standard deviation.
    pub target_std: Vec<f32>,
    /// Normalized target-by-feature weight matrix.
    pub weights: Vec<Vec<f32>>,
}

/// Versioned teacher-residual decoder artifact.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TeacherResidualDecoderArtifact {
    /// Artifact schema.
    pub schema_version: u32,
    /// Exact feature order.
    pub feature_order: String,
    /// History slots used during fitting.
    pub history_len: usize,
    /// Maximum continuous gap used during fitting.
    pub max_gap_micros: u64,
    /// Sorted training-take identities.
    pub training_takes: Vec<String>,
    /// Ridge strength.
    pub ridge_lambda: f32,
    /// Normalized linear map.
    pub linear_map: NormalizedLinearMapArtifact,
    /// Stable hash of every preceding semantic field.
    pub content_hash: u64,
}

/// Validated residual decoder ready for offline prediction.
#[derive(Clone, Debug, PartialEq)]
pub struct LoadedTeacherResidualDecoder {
    artifact: TeacherResidualDecoderArtifact,
}

impl LoadedTeacherResidualDecoder {
    /// Validates schema, hash, order, and dimensions.
    ///
    /// # Errors
    ///
    /// Returns a typed load error when the artifact is incompatible.
    pub fn load(
        artifact: TeacherResidualDecoderArtifact,
        expected_feature_order: &str,
    ) -> Result<Self, LinearPriorLoadError> {
        if artifact.feature_order != expected_feature_order {
            return Err(LinearPriorLoadError::FeatureOrderMismatch {
                expected: expected_feature_order.to_owned(),
                found: artifact.feature_order.clone(),
            });
        }
        let features = vec![0.0; artifact.history_len * HISTORY_SLOT_WIDTH + VELOCITY_WIDTH + 1];
        let _ = predict_teacher_residual(&artifact, &features)?;
        Ok(Self { artifact })
    }

    fn predict(
        &self,
        features: &[f32],
    ) -> Result<[f32; ARKIT_NON_TONGUE_CHANNEL_COUNT], LinearPriorLoadError> {
        predict_teacher_residual(&self.artifact, features)
    }
}

/// D/G0/H0 values aligned to one existing trace frame.
#[derive(Clone, Debug, PartialEq)]
pub struct ExistingTraceResidualVariants {
    /// MediaPipe Direct coefficients.
    pub direct: Arkit52Coefficients,
    /// Existing hand-projected GNM coefficients.
    pub gnm_projected: Arkit52Coefficients,
    /// Direct plus the learned residual, clamped only at this final boundary.
    pub hybrid_projected_residual: Arkit52Coefficients,
}

/// Typed failure while producing aligned offline ablation variants.
#[derive(Clone, Debug, PartialEq)]
pub enum ResidualAblationError {
    /// The frame has no Direct coefficients.
    MissingDirect,
    /// The frame has no projected GNM state.
    MissingGnmState,
    /// The canonical history builder emitted no row for this frame.
    MissingHistory,
    /// Decoder validation or prediction failed.
    Decoder(LinearPriorLoadError),
    /// Final ARKit52 reconstruction failed.
    InvalidOutput(Arkit52ValueError),
}

/// Adds a signed residual to the non-tongue Direct values and clamps the final output.
///
/// # Errors
///
/// Returns a typed ARKit value error for non-finite inputs.
pub fn apply_non_tongue_residual(
    direct: &Arkit52Coefficients,
    residual: [f32; ARKIT_NON_TONGUE_CHANNEL_COUNT],
) -> Result<Arkit52Coefficients, Arkit52ValueError> {
    let mut values = arkit_non_tongue_values(direct);
    for (value, correction) in values.iter_mut().zip(residual) {
        *value = (*value + correction).clamp(0.0, 1.0);
    }
    arkit52_with_zero_tongue(values)
}

/// Produces aligned D/G0/H0 variants using the canonical #12 feature builder.
///
/// # Errors
///
/// Returns a typed error for missing aligned inputs, features, or invalid prediction.
pub fn existing_trace_residual_variants(
    sample: &PairedTemporalSample,
    history: &TeacherResidualHistory,
    decoder: &LoadedTeacherResidualDecoder,
) -> Result<ExistingTraceResidualVariants, ResidualAblationError> {
    let direct = sample
        .mediapipe_observation
        .as_ref()
        .ok_or(ResidualAblationError::MissingDirect)?;
    let gnm = sample
        .gnm_state
        .as_ref()
        .ok_or(ResidualAblationError::MissingGnmState)?;
    let features = history
        .features(sample)
        .ok_or(ResidualAblationError::MissingHistory)?;
    let residual = decoder
        .predict(features)
        .map_err(ResidualAblationError::Decoder)?;
    let hybrid_projected_residual = apply_non_tongue_residual(direct, residual)
        .map_err(ResidualAblationError::InvalidOutput)?;
    Ok(ExistingTraceResidualVariants {
        direct: *direct,
        gnm_projected: gnm.projected_coefficients,
        hybrid_projected_residual,
    })
}

/// Fits a residual decoder using only explicitly selected takes.
///
/// # Errors
///
/// Returns the shared typed ridge errors for empty, invalid, or singular data.
pub fn fit_teacher_residual_decoder(
    rows: &[TeacherResidualRow],
    training_takes: &BTreeSet<String>,
    feature_config: TeacherResidualFeatureConfig,
    config: LinearPriorTrainingConfig,
    feature_order: &str,
) -> Result<TeacherResidualDecoderArtifact, LinearPriorFitError> {
    let selected: Vec<&TeacherResidualRow> = rows
        .iter()
        .filter(|row| training_takes.contains(&row.take_id))
        .collect();
    if selected.is_empty() {
        return Err(LinearPriorFitError::EmptyTrainingSet);
    }
    let features: Vec<Vec<f32>> = selected.iter().map(|row| row.features.clone()).collect();
    let targets: Vec<Vec<f32>> = selected
        .iter()
        .map(|row| row.target_residual.to_vec())
        .collect();
    let map = fit_normalized_multi_output_ridge(&features, &targets, config)?;
    let mut artifact = TeacherResidualDecoderArtifact {
        schema_version: TEACHER_RESIDUAL_DECODER_SCHEMA_VERSION,
        feature_order: feature_order.to_owned(),
        history_len: feature_config.history_len,
        max_gap_micros: feature_config.max_gap_micros,
        training_takes: training_takes.iter().cloned().collect(),
        ridge_lambda: config.ridge_lambda,
        linear_map: NormalizedLinearMapArtifact {
            feature_mean: map.feature_mean,
            feature_std: map.feature_std,
            target_mean: map.target_mean,
            target_std: map.target_std,
            weights: map.weights,
        },
        content_hash: 0,
    };
    artifact.content_hash = hash_teacher_residual_artifact(&artifact);
    Ok(artifact)
}

/// Predicts an unclamped signed 51-channel teacher residual.
///
/// # Errors
///
/// Rejects invalid schema, hash, dimensions, normalization, or input values.
#[allow(clippy::indexing_slicing)]
pub fn predict_teacher_residual(
    artifact: &TeacherResidualDecoderArtifact,
    features: &[f32],
) -> Result<[f32; ARKIT_NON_TONGUE_CHANNEL_COUNT], LinearPriorLoadError> {
    if artifact.schema_version != TEACHER_RESIDUAL_DECODER_SCHEMA_VERSION {
        return Err(LinearPriorLoadError::UnsupportedSchemaVersion {
            found: artifact.schema_version,
        });
    }
    let computed = hash_teacher_residual_artifact(artifact);
    if computed != artifact.content_hash {
        return Err(LinearPriorLoadError::ContentHashMismatch {
            recorded: artifact.content_hash,
            computed,
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
            detail: "teacher residual map dimensions disagree".to_owned(),
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
        || map.weights.iter().flatten().any(|value| !value.is_finite())
    {
        return Err(LinearPriorLoadError::InvalidNormalization {
            field: "teacher residual feature/normalization".to_owned(),
        });
    }
    let mut prediction = [0.0; ARKIT_NON_TONGUE_CHANNEL_COUNT];
    for (target, output) in prediction.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (index, value) in features.iter().enumerate() {
            sum += (*value - map.feature_mean[index]) / map.feature_std[index]
                * map.weights[target][index];
        }
        *output = map.target_mean[target] + sum * map.target_std[target];
    }
    if prediction.iter().any(|value| !value.is_finite()) {
        return Err(LinearPriorLoadError::InvalidNormalization {
            field: "teacher residual prediction".to_owned(),
        });
    }
    Ok(prediction)
}

fn hash_teacher_residual_artifact(artifact: &TeacherResidualDecoderArtifact) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&artifact.schema_version.to_le_bytes());
    bytes.extend_from_slice(artifact.feature_order.as_bytes());
    bytes.push(0xff);
    bytes.extend_from_slice(&(artifact.history_len as u64).to_le_bytes());
    bytes.extend_from_slice(&artifact.max_gap_micros.to_le_bytes());
    for take in &artifact.training_takes {
        bytes.extend_from_slice(take.as_bytes());
        bytes.push(0xff);
    }
    bytes.extend_from_slice(&artifact.ridge_lambda.to_bits().to_le_bytes());
    for value in artifact
        .linear_map
        .feature_mean
        .iter()
        .chain(&artifact.linear_map.feature_std)
        .chain(&artifact.linear_map.target_mean)
        .chain(&artifact.linear_map.target_std)
        .chain(artifact.linear_map.weights.iter().flatten())
    {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arkit_teacher::{ArkitTeacherFrame, DeterministicGnmState, HeadTransform};
    use vtuber_core::{ARKIT52_CHANNEL_COUNT, Arkit52Coefficients, ArkitBlendshape};

    fn coefficients(jaw: f32) -> Arkit52Coefficients {
        let mut values = [0.0; ARKIT52_CHANNEL_COUNT];
        values[ArkitBlendshape::JawOpen.index()] = jaw;
        Arkit52Coefficients::try_from_array(values).unwrap()
    }

    fn sample(seq: u64, direct: f32, gnm: f32, teacher: f32) -> PairedTemporalSample {
        PairedTemporalSample {
            frame_seq: seq,
            timestamp_micros: seq * 20_000,
            mediapipe_observation: Some(coefficients(direct)),
            gnm_state: Some(DeterministicGnmState {
                projected_coefficients: coefficients(gnm),
                residual: 0.01 * seq as f32,
            }),
            baseline_output: coefficients(direct),
            teacher: Some(ArkitTeacherFrame {
                frame_seq: seq,
                timestamp_micros: seq * 20_000,
                coefficients: coefficients(teacher),
                head_transform: HeadTransform {
                    rotation_unit_quaternion_wxyz: [1.0, 0.0, 0.0, 0.0],
                    translation_meters: [0.0; 3],
                },
            }),
            rgb_reference: None,
        }
    }

    fn config() -> TeacherResidualFeatureConfig {
        TeacherResidualFeatureConfig {
            history_len: 2,
            max_gap_micros: 40_000,
        }
    }

    #[test]
    fn target_is_same_frame_teacher_minus_direct_and_keeps_negative_values() {
        let rows = build_teacher_residual_rows("a", &[sample(1, 0.8, 0.4, 0.2)], config())
            .unwrap()
            .rows;
        assert_eq!(rows.len(), 1);
        assert!((rows[0].target_residual[ArkitBlendshape::JawOpen.index()] + 0.6).abs() < 1e-6);
    }

    #[test]
    fn future_changes_do_not_change_current_features() {
        let first = build_teacher_residual_rows(
            "a",
            &[sample(1, 0.1, 0.2, 0.3), sample(2, 0.4, 0.5, 0.6)],
            config(),
        )
        .unwrap();
        let changed = build_teacher_residual_rows(
            "a",
            &[sample(1, 0.1, 0.2, 0.3), sample(2, 0.9, 0.8, 0.7)],
            config(),
        )
        .unwrap();
        assert_eq!(first.rows[0], changed.rows[0]);
    }

    #[test]
    fn gap_and_missing_state_reset_history() {
        let mut samples = vec![sample(1, 0.1, 0.2, 0.3), sample(2, 0.2, 0.3, 0.4)];
        samples.push(sample(10, 0.3, 0.4, 0.5));
        samples[1].gnm_state = None;
        let dataset = build_teacher_residual_rows("a", &samples, config()).unwrap();
        assert!(
            dataset
                .exclusions
                .iter()
                .any(|(reason, _)| *reason == TeacherResidualExclusion::MissingGnmState)
        );
        let newest = &dataset.rows[1].features;
        assert!(
            newest[HISTORY_SLOT_WIDTH..HISTORY_SLOT_WIDTH * 2]
                .iter()
                .all(|value| *value == 0.0)
        );
        assert_eq!(newest.last(), Some(&0.0));
    }

    #[test]
    fn fit_uses_only_selected_takes_and_is_deterministic() {
        let mut rows = build_teacher_residual_rows(
            "train",
            &[
                sample(1, 0.1, 0.2, 0.3),
                sample(2, 0.2, 0.4, 0.1),
                sample(3, 0.3, 0.1, 0.8),
                sample(4, 0.4, 0.5, 0.2),
            ],
            config(),
        )
        .unwrap()
        .rows;
        rows.extend(
            build_teacher_residual_rows("eval", &[sample(1, 0.9, 0.9, 0.0)], config())
                .unwrap()
                .rows,
        );
        let takes = BTreeSet::from(["train".to_owned()]);
        let training = LinearPriorTrainingConfig::default();
        let a = fit_teacher_residual_decoder(
            &rows,
            &takes,
            config(),
            training,
            TEACHER_RESIDUAL_FEATURE_ORDER,
        )
        .unwrap();
        let mut changed = rows.clone();
        changed.last_mut().unwrap().features.fill(42.0);
        let b = fit_teacher_residual_decoder(
            &changed,
            &takes,
            config(),
            training,
            TEACHER_RESIDUAL_FEATURE_ORDER,
        )
        .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn synthetic_known_map_predicts_signed_residual() {
        let rows: Vec<TeacherResidualRow> = [-2.0_f32, -1.0, 0.0, 1.0, 2.0]
            .into_iter()
            .enumerate()
            .map(|(index, x)| {
                let mut target = [0.0; ARKIT_NON_TONGUE_CHANNEL_COUNT];
                target[0] = 2.0 * x - 0.5;
                TeacherResidualRow {
                    take_id: "train".to_owned(),
                    frame_seq: index as u64,
                    timestamp_micros: index as u64,
                    features: vec![x],
                    target_residual: target,
                }
            })
            .collect();
        let artifact = fit_teacher_residual_decoder(
            &rows,
            &BTreeSet::from(["train".to_owned()]),
            TeacherResidualFeatureConfig {
                history_len: 1,
                max_gap_micros: 1,
            },
            LinearPriorTrainingConfig {
                ridge_lambda: 1e-6,
                ..LinearPriorTrainingConfig::default()
            },
            "known",
        )
        .unwrap();
        let prediction = predict_teacher_residual(&artifact, &[0.25]).unwrap();
        assert!((prediction[0] - 0.0).abs() < 1e-3);
        assert!(prediction.iter().skip(1).all(|value| value.abs() < 1e-6));
    }

    #[test]
    fn residual_application_clamps_and_zeroes_tongue() {
        let mut residual = [0.0; ARKIT_NON_TONGUE_CHANNEL_COUNT];
        residual[ArkitBlendshape::JawOpen.index()] = 0.4;
        residual[ArkitBlendshape::EyeBlinkLeft.index()] = -0.8;
        let output = apply_non_tongue_residual(&coefficients(0.8), residual).unwrap();
        assert_eq!(output.get(ArkitBlendshape::JawOpen), 1.0);
        assert_eq!(output.get(ArkitBlendshape::EyeBlinkLeft), 0.0);
        assert_eq!(output.get(ArkitBlendshape::TongueOut), 0.0);
    }

    #[test]
    fn aligned_variants_use_the_canonical_history_features() {
        let samples = vec![
            sample(1, 0.1, 0.2, 0.3),
            sample(2, 0.2, 0.4, 0.1),
            sample(3, 0.3, 0.1, 0.8),
            sample(4, 0.4, 0.5, 0.2),
        ];
        let rows = build_teacher_residual_rows("train", &samples, config())
            .unwrap()
            .rows;
        let artifact = fit_teacher_residual_decoder(
            &rows,
            &BTreeSet::from(["train".to_owned()]),
            config(),
            LinearPriorTrainingConfig::default(),
            TEACHER_RESIDUAL_FEATURE_ORDER,
        )
        .unwrap();
        let decoder =
            LoadedTeacherResidualDecoder::load(artifact, TEACHER_RESIDUAL_FEATURE_ORDER).unwrap();
        let history = TeacherResidualHistory::build("train", &samples, config()).unwrap();
        let variants = existing_trace_residual_variants(&samples[0], &history, &decoder).unwrap();
        assert_eq!(variants.direct, coefficients(0.1));
        assert_eq!(variants.gnm_projected, coefficients(0.2));
        assert_eq!(
            variants
                .hybrid_projected_residual
                .get(ArkitBlendshape::TongueOut),
            0.0
        );
    }
}
