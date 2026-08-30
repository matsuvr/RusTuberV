//! Linear autoregressive prior fitting and versioned artifact export
//! (GNM #68.4b).
//!
//! Fits one ridge-regularized linear map from [`CausalRow`] features onto the
//! next-step GNM coefficients using only the takes selected for training.
//! The closed-form normal-equations solution is fully deterministic: the same
//! input rows and configuration always produce the same artifact bytes.
//!
//! The exported artifact carries everything inference needs later (#132):
//! model/schema versions, the explicit training configuration (including the
//! recorded seed), feature/target normalization statistics, weights, the
//! feature order description, and a stable content hash for load-time
//! validation. Non-finite data or an ill-conditioned system is a typed error;
//! nothing unusable is ever exported.

use std::collections::BTreeSet;

use crate::causal_dataset::CausalRow;

/// Schema version of exported artifacts.
pub const LINEAR_PRIOR_SCHEMA_VERSION: u32 = 1;

/// Explicit, re-runnable training configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearPriorTrainingConfig {
    /// Ridge regularization strength added to the diagonal. Must be positive.
    pub ridge_lambda: f32,
    /// Recorded random seed. The closed-form solver is deterministic, so the
    /// seed is carried in the artifact purely for provenance/re-runs.
    pub seed: u64,
    /// Numerical pivot threshold below which the system counts singular.
    pub pivot_epsilon: f32,
}

impl Default for LinearPriorTrainingConfig {
    fn default() -> Self {
        Self {
            ridge_lambda: 1e-3,
            seed: 0,
            pivot_epsilon: 1e-8,
        }
    }
}

/// Typed fit failures; an errored fit never exports an artifact.
#[derive(Clone, Debug, PartialEq)]
pub enum LinearPriorFitError {
    /// Training input contained NaN or infinite values.
    NonFiniteInput,
    /// No rows survived the training-take filter.
    EmptyTrainingSet,
    /// The regularized normal equations were numerically singular.
    IllConditioned {
        /// Offending pivot magnitude that triggered rejection.
        pivot: f32,
    },
    /// Feature/target dimensions were inconsistent across rows.
    InconsistentDimensions,
}

/// Versioned, self-describing linear prior artifact.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LinearPriorArtifact {
    /// Artifact schema version.
    pub schema_version: u32,
    /// Model version label (bumped when semantics change).
    pub model_version: u32,
    /// Recorded training seed (provenance).
    pub seed: u64,
    /// Ridge lambda used during the fit.
    pub ridge_lambda: f32,
    /// Human-readable feature order description pinned at fit time.
    pub feature_order: String,
    /// Per-feature mean subtracted before scaling.
    pub feature_mean: Vec<f32>,
    /// Per-feature standard deviation divisor (never zero).
    pub feature_std: Vec<f32>,
    /// Per-target mean subtracted from predictions.
    pub target_mean: Vec<f32>,
    /// Per-target standard deviation divisor (never zero).
    pub target_std: Vec<f32>,
    /// Row-major weight matrix `[target_dim][feature_dim]` in normalized
    /// space.
    pub weights: Vec<Vec<f32>>,
    /// Stable FNV-1a hash over every field above except itself.
    pub content_hash: u64,
}

/// Fits the linear prior over the training takes only.
///
/// # Errors
///
/// Returns a typed error for empty/non-finite training data, inconsistent
/// row dimensions, or an ill-conditioned normal-equations system.
// Bounds are guaranteed by construction: dimensions are validated against the
// first training row before every indexed access; see AGENTS.md panic policy.
#[allow(clippy::indexing_slicing)]
pub fn fit_linear_prior(
    rows: &[CausalRow],
    training_takes: &BTreeSet<String>,
    config: LinearPriorTrainingConfig,
    feature_order: &str,
) -> Result<LinearPriorArtifact, LinearPriorFitError> {
    if !(config.ridge_lambda.is_finite() && config.ridge_lambda > 0.0) {
        return Err(LinearPriorFitError::NonFiniteInput);
    }
    let training: Vec<&CausalRow> = rows
        .iter()
        .filter(|row| training_takes.contains(&row.take_id))
        .collect();
    if training.is_empty() {
        return Err(LinearPriorFitError::EmptyTrainingSet);
    }

    let feature_dim = training[0].features.len();
    let target_dim = training[0].target.len();
    if feature_dim == 0 || target_dim == 0 {
        return Err(LinearPriorFitError::InconsistentDimensions);
    }
    if training
        .iter()
        .any(|row| row.features.len() != feature_dim || row.target.len() != target_dim)
    {
        return Err(LinearPriorFitError::InconsistentDimensions);
    }

    let mut feature_sum = vec![0.0_f64; feature_dim];
    let mut target_sum = vec![0.0_f64; target_dim];
    for row in &training {
        for (index, value) in row.features.iter().enumerate() {
            if !value.is_finite() {
                return Err(LinearPriorFitError::NonFiniteInput);
            }
            feature_sum[index] += f64::from(*value);
        }
        for (index, value) in row.target.iter().enumerate() {
            if !value.is_finite() {
                return Err(LinearPriorFitError::NonFiniteInput);
            }
            target_sum[index] += f64::from(*value);
        }
    }
    let count = training.len() as f64;
    let feature_mean: Vec<f32> = feature_sum
        .iter()
        .map(|sum| (*sum / count) as f32)
        .collect();
    let target_mean: Vec<f32> = target_sum.iter().map(|sum| (*sum / count) as f32).collect();

    let mut feature_m2 = vec![0.0_f64; feature_dim];
    let mut target_m2 = vec![0.0_f64; target_dim];
    for row in &training {
        for (index, value) in row.features.iter().enumerate() {
            let centered = f64::from(*value) - f64::from(feature_mean[index]);
            feature_m2[index] += centered * centered;
        }
        for (index, value) in row.target.iter().enumerate() {
            let centered = f64::from(*value) - f64::from(target_mean[index]);
            target_m2[index] += centered * centered;
        }
    }
    let feature_std: Vec<f32> = feature_m2
        .iter()
        .map(|m2| ((m2 / count).sqrt() as f32).max(1e-6))
        .collect();
    let target_std: Vec<f32> = target_m2
        .iter()
        .map(|m2| ((m2 / count).sqrt() as f32).max(1e-6))
        .collect();

    // Normalized design matrix X (rows x features), targets Y (rows x targets).
    let mut x = vec![vec![0.0_f64; feature_dim]; training.len()];
    let mut y = vec![vec![0.0_f64; target_dim]; training.len()];
    for (row_index, row) in training.iter().enumerate() {
        for (index, value) in row.features.iter().enumerate() {
            x[row_index][index] = f64::from((*value - feature_mean[index]) / feature_std[index]);
        }
        for (index, value) in row.target.iter().enumerate() {
            y[row_index][index] = f64::from((*value - target_mean[index]) / target_std[index]);
        }
    }

    // Normal equations A = XᵀX + λI, b = XᵀY solved per-target column via
    // Gaussian elimination with partial pivoting.
    let dim = feature_dim;
    let mut a = vec![vec![0.0_f64; dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            a[i][j] = x.iter().map(|row| row[i] * row[j]).sum();
        }
        a[i][i] += f64::from(config.ridge_lambda);
    }

    let mut weights_normalized = vec![vec![0.0_f64; dim]; target_dim];
    for target_index in 0..target_dim {
        let mut b: Vec<f64> = (0..dim)
            .map(|i| {
                x.iter()
                    .map(|row| row[i] * y.iter().map(|yr| yr[target_index]).sum::<f64>())
                    .sum()
            })
            .collect();
        let solution = gauss_solve(&a, &mut b, config.pivot_epsilon)?;
        weights_normalized[target_index] = solution;
    }

    let mut artifact = LinearPriorArtifact {
        schema_version: LINEAR_PRIOR_SCHEMA_VERSION,
        model_version: 1,
        seed: config.seed,
        ridge_lambda: config.ridge_lambda,
        feature_order: feature_order.to_owned(),
        feature_mean,
        feature_std,
        target_mean,
        target_std,
        weights: weights_normalized
            .iter()
            .map(|row| row.iter().map(|value| *value as f32).collect())
            .collect(),
        content_hash: 0,
    };
    artifact.content_hash = hash_artifact(&artifact);
    Ok(artifact)
}

// Bounds are guaranteed by construction: square matrices sized `dim`;
// see AGENTS.md panic policy.
#[allow(clippy::indexing_slicing)]
fn gauss_solve(
    matrix: &[Vec<f64>],
    rhs: &mut [f64],
    pivot_epsilon: f32,
) -> Result<Vec<f64>, LinearPriorFitError> {
    let dim = rhs.len();
    let mut work = matrix.to_vec();
    for column in 0..dim {
        let mut pivot_row = column;
        for candidate in (column + 1)..dim {
            if work[candidate][column].abs() > work[pivot_row][column].abs() {
                pivot_row = candidate;
            }
        }
        if work[pivot_row][column].abs() < f64::from(pivot_epsilon) {
            return Err(LinearPriorFitError::IllConditioned {
                pivot: work[pivot_row][column] as f32,
            });
        }
        work.swap(column, pivot_row);
        rhs.swap(column, pivot_row);
        let pivot = work[column][column];
        for row in (column + 1)..dim {
            let factor = work[row][column] / pivot;
            let pivot_row_values = work[column][column..].to_vec();
            for (offset, value) in pivot_row_values.into_iter().enumerate() {
                work[row][column + offset] -= factor * value;
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    let mut solution = vec![0.0_f64; dim];
    for row in (0..dim).rev() {
        let tail: f64 = (row + 1..dim)
            .map(|inner| work[row][inner] * solution[inner])
            .sum();
        solution[row] = (rhs[row] - tail) / work[row][row];
    }
    Ok(solution)
}

/// Stable FNV-1a hash over the artifact's semantic fields.
#[must_use]
pub fn hash_artifact(artifact: &LinearPriorArtifact) -> u64 {
    let mut hasher = Fnv1a::new();
    hasher.update_u64(u64::from(artifact.schema_version));
    hasher.update_u64(u64::from(artifact.model_version));
    hasher.update_u64(artifact.seed);
    hasher.update_f32(artifact.ridge_lambda);
    hasher.update_str(&artifact.feature_order);
    for slice in [
        &artifact.feature_mean,
        &artifact.feature_std,
        &artifact.target_mean,
        &artifact.target_std,
    ] {
        for value in slice.iter() {
            hasher.update_f32(*value);
        }
    }
    for row in &artifact.weights {
        for value in row {
            hasher.update_f32(*value);
        }
    }
    hasher.finish()
}

struct Fnv1a(u64);

impl Fnv1a {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    const fn new() -> Self {
        Self(Self::OFFSET)
    }

    fn update_byte(&mut self, byte: u8) {
        self.0 ^= u64::from(byte);
        self.0 = self.0.wrapping_mul(Self::PRIME);
    }

    fn update_u64(&mut self, value: u64) {
        for byte in value.to_le_bytes() {
            self.update_byte(byte);
        }
    }

    fn update_f32(&mut self, value: f32) {
        self.update_u64(u64::from(value.to_bits()));
    }

    fn update_str(&mut self, value: &str) {
        for byte in value.as_bytes() {
            self.update_byte(*byte);
        }
        self.update_byte(0xff); // terminator so prefixes cannot collide
    }

    const fn finish(self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arkit_teacher::PairedTemporalSample;
    use crate::causal_dataset::{CausalFeatureConfig, build_causal_dataset};
    use vtuber_core::{ARKIT52_CHANNEL_COUNT, Arkit52Coefficients, ArkitBlendshape};

    fn sample(seq: u64, jaw_open: f32) -> PairedTemporalSample {
        let mut values = [0.0_f32; ARKIT52_CHANNEL_COUNT];
        values[ArkitBlendshape::JawOpen.index()] = jaw_open;
        PairedTemporalSample {
            frame_seq: seq,
            timestamp_micros: seq * 16_667,
            mediapipe_observation: None,
            gnm_state: Some(crate::arkit_teacher::DeterministicGnmState {
                projected_coefficients: Arkit52Coefficients::try_from_array(values).unwrap(),
                residual: 0.01,
            }),
            baseline_output: Arkit52Coefficients::default(),
            teacher: None,
            rgb_reference: None,
        }
    }

    fn dataset(take_a_jaw: &[f32], take_b_jaw: &[f32]) -> Vec<CausalRow> {
        let build = |take: &str, values: &[f32]| {
            let samples: Vec<PairedTemporalSample> = values
                .iter()
                .enumerate()
                .map(|(index, value)| sample(index as u64 + 1, *value))
                .collect();
            let config = CausalFeatureConfig {
                history_len: 2,
                max_gap_micros: 40_000,
            };
            build_causal_dataset(take, &samples, config).unwrap().rows
        };
        let mut rows = build("take-a", take_a_jaw);
        rows.extend(build("take-b", take_b_jaw));
        rows
    }

    #[test]
    fn deterministic_fit_produces_identical_artifacts() {
        let rows = dataset(&[0.1, 0.5, 0.9, 0.7, 0.2], &[0.3, 0.6]);
        let takes = BTreeSet::from(["take-a".to_owned()]);
        let config = LinearPriorTrainingConfig::default();
        let first = fit_linear_prior(&rows, &takes, config, "history+velocity").unwrap();
        let second = fit_linear_prior(&rows, &takes, config, "history+velocity").unwrap();
        assert_eq!(first, second);
        assert_eq!(first.content_hash, second.content_hash);
    }

    #[test]
    fn only_training_takes_influence_the_weights() {
        let base_rows = dataset(&[0.1, 0.5, 0.9, 0.7, 0.2], &[0.3, 0.6]);
        let takes_a = BTreeSet::from(["take-a".to_owned()]);
        let config = LinearPriorTrainingConfig::default();
        let without_b = fit_linear_prior(&base_rows, &takes_a, config, "order").unwrap();

        let mut changed_rows = dataset(&[0.1, 0.5, 0.9, 0.7, 0.2], &[0.95, 0.05]);
        // Corrupt every take-b row; the take-a-only fit must not notice.
        for row in changed_rows
            .iter_mut()
            .filter(|row| row.take_id == "take-b")
        {
            for value in &mut row.features {
                *value = 42.0;
            }
        }
        let with_corrupted_b = fit_linear_prior(&changed_rows, &takes_a, config, "order").unwrap();
        assert_eq!(without_b.weights, with_corrupted_b.weights);
        assert_eq!(without_b.content_hash, with_corrupted_b.content_hash);

        // The holdout take does participate when included.
        let both = BTreeSet::from(["take-a".to_owned(), "take-b".to_owned()]);
        let with_clean_b = fit_linear_prior(&base_rows, &both, config, "order").unwrap();
        assert_ne!(without_b.weights, with_clean_b.weights);
    }

    #[test]
    fn normalization_is_recorded_and_non_degenerate() {
        let rows = dataset(&[0.1, 0.5, 0.9], &[]);
        let takes = BTreeSet::from(["take-a".to_owned()]);
        let artifact =
            fit_linear_prior(&rows, &takes, LinearPriorTrainingConfig::default(), "order").unwrap();
        assert!(artifact.feature_std.iter().all(|std| *std >= 1e-6));
        assert!(artifact.target_std.iter().all(|std| *std >= 1e-6));
        assert_eq!(artifact.feature_order, "order");
        assert_eq!(artifact.schema_version, LINEAR_PRIOR_SCHEMA_VERSION);
    }

    #[test]
    fn bad_inputs_are_typed_errors_without_export() {
        let takes = BTreeSet::from(["take-a".to_owned()]);
        assert_eq!(
            fit_linear_prior(&[], &takes, LinearPriorTrainingConfig::default(), "o"),
            Err(LinearPriorFitError::EmptyTrainingSet)
        );

        let rows = dataset(&[0.1, 0.5, 0.9], &[]);
        let zero_lambda = LinearPriorTrainingConfig {
            ridge_lambda: 0.0,
            ..LinearPriorTrainingConfig::default()
        };
        assert_eq!(
            fit_linear_prior(&rows, &takes, zero_lambda, "o"),
            Err(LinearPriorFitError::NonFiniteInput)
        );
    }
}
