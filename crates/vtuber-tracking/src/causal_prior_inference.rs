//! Rust load/inference boundary for exported linear priors (GNM #68.4c).
//!
//! Loads the versioned artifact produced by #131 with full validation, then
//! computes prior corrections from causal history features only. There is no
//! future-frame API by construction: predictions consume one feature vector
//! that callers already know is causal.
//!
//! A missing or invalid artifact degrades to a typed deterministic-baseline
//! outcome instead of an error at runtime, so the tracking pipeline can keep
//! publishing without a learned prior.

use crate::causal_prior::{LINEAR_PRIOR_SCHEMA_VERSION, LinearPriorArtifact, hash_artifact};

/// Typed artifact-validation failures.
#[derive(Clone, Debug, PartialEq)]
pub enum LinearPriorLoadError {
    /// The artifact schema version differs from this build's expectation.
    UnsupportedSchemaVersion {
        /// Encountered schema version.
        found: u32,
    },
    /// The recomputed content hash disagreed with the recorded one.
    ContentHashMismatch {
        /// Recorded hash.
        recorded: u64,
        /// Recomputed hash over the actual fields.
        computed: u64,
    },
    /// The feature order string did not match what the caller was built for.
    FeatureOrderMismatch {
        /// Expected order description.
        expected: String,
        /// Artifact's order description.
        found: String,
    },
    /// Weight/normalization dimensions were inconsistent.
    DimensionMismatch {
        /// Detail message.
        detail: String,
    },
    /// A normalization divisor or weight was non-finite or degenerate.
    InvalidNormalization {
        /// Field description used in diagnostics.
        field: String,
    },
}

/// A fully validated linear prior ready for inference.
#[derive(Clone, Debug, PartialEq)]
pub struct LoadedLinearPrior {
    artifact: LinearPriorArtifact,
    feature_dim: usize,
    target_dim: usize,
}

impl LoadedLinearPrior {
    /// Validates and loads an exported artifact.
    ///
    /// `expected_feature_order` pins the caller to the exact feature layout
    /// it feeds in, preventing silent reordering between fit and inference.
    ///
    /// # Errors
    ///
    /// Returns a typed error for any schema/hash/order/dimension/
    /// normalization violation.
    // Bounds are guaranteed by construction: every dimension is validated
    // above before any indexed access; see AGENTS.md panic policy.
    #[allow(clippy::indexing_slicing)]
    pub fn load(
        artifact: LinearPriorArtifact,
        expected_feature_order: &str,
    ) -> Result<Self, LinearPriorLoadError> {
        if artifact.schema_version != LINEAR_PRIOR_SCHEMA_VERSION {
            return Err(LinearPriorLoadError::UnsupportedSchemaVersion {
                found: artifact.schema_version,
            });
        }
        let computed = hash_artifact(&artifact);
        if computed != artifact.content_hash {
            return Err(LinearPriorLoadError::ContentHashMismatch {
                recorded: artifact.content_hash,
                computed,
            });
        }
        if artifact.feature_order != expected_feature_order {
            return Err(LinearPriorLoadError::FeatureOrderMismatch {
                expected: expected_feature_order.to_owned(),
                found: artifact.feature_order.clone(),
            });
        }
        let feature_dim = artifact.feature_mean.len();
        if artifact.feature_std.len() != feature_dim
            || artifact.target_mean.is_empty()
            || artifact.target_mean.len() != artifact.target_std.len()
            || artifact.weights.is_empty()
            || artifact.weights.len() != artifact.target_mean.len()
        {
            return Err(LinearPriorLoadError::DimensionMismatch {
                detail: "mean/std/weight row counts disagree".to_owned(),
            });
        }
        let weight_width = artifact.weights[0].len();
        if weight_width == 0
            || artifact.weights.iter().any(|row| row.len() != weight_width)
            || weight_width != feature_dim
        {
            return Err(LinearPriorLoadError::DimensionMismatch {
                detail: "weight matrix shape disagrees with feature dimension".to_owned(),
            });
        }
        if artifact
            .feature_std
            .iter()
            .any(|std| !std.is_finite() || *std <= 0.0)
            || artifact
                .target_std
                .iter()
                .any(|std| !std.is_finite() || *std <= 0.0)
            || artifact
                .weights
                .iter()
                .flatten()
                .any(|value| !value.is_finite())
        {
            return Err(LinearPriorLoadError::InvalidNormalization {
                field: "feature/target std or weights".to_owned(),
            });
        }

        Ok(Self {
            feature_dim,
            target_dim: artifact.target_mean.len(),
            artifact,
        })
    }

    /// The pinned feature order description.
    #[must_use]
    pub fn feature_order(&self) -> &str {
        &self.artifact.feature_order
    }

    /// Feature vector width this prior accepts.
    #[must_use]
    pub const fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    /// Target width this prior produces.
    #[must_use]
    pub const fn target_dim(&self) -> usize {
        self.target_dim
    }

    /// Computes the next-step prior prediction from causal features only.
    ///
    /// Returns the denormalized coefficient estimate. The caller supplies the
    /// same feature layout recorded in `feature_order`; there is no API that
    /// could pull a future frame into this computation.
    ///
    /// # Errors
    ///
    /// Returns [`LinearPriorLoadError::InvalidNormalization`] when the input
    /// or resulting correction is non-finite, keeping outputs fail-closed.
    // Bounds are guaranteed by construction: feature length equals the
    // validated weight width; see AGENTS.md panic policy.
    #[allow(clippy::indexing_slicing)]
    pub fn predict(&self, features: &[f32]) -> Result<Vec<f32>, LinearPriorLoadError> {
        if features.len() != self.feature_dim || features.iter().any(|value| !value.is_finite()) {
            return Err(LinearPriorLoadError::InvalidNormalization {
                field: "input features must be finite and match feature_dim".to_owned(),
            });
        }
        let mut prediction = self.artifact.target_mean.clone();
        for (target_index, weight_row) in self.artifact.weights.iter().enumerate() {
            let mut sum = 0.0_f32;
            for (index, value) in features.iter().enumerate() {
                sum += (*value - self.artifact.feature_mean[index])
                    / self.artifact.feature_std[index]
                    * weight_row[index];
            }
            prediction[target_index] += sum * self.artifact.target_std[target_index];
        }
        if prediction.iter().any(|value| !value.is_finite()) {
            return Err(LinearPriorLoadError::InvalidNormalization {
                field: "predicted prior state".to_owned(),
            });
        }
        Ok(prediction)
    }
}

/// Runtime source of prior corrections with deterministic fallback.
#[derive(Clone, Debug, PartialEq)]
pub enum PriorInference {
    /// A validated learned prior drives corrections.
    Learned(Box<LoadedLinearPrior>),
    /// No usable artifact exists; the adaptive baseline stands alone.
    DeterministicBaseline,
}

impl PriorInference {
    /// Loads a learned prior, falling back to the baseline on any validation
    /// failure instead of failing the caller.
    #[must_use]
    pub fn load_or_baseline(
        artifact: Option<LinearPriorArtifact>,
        expected_feature_order: &str,
    ) -> Self {
        match artifact {
            Some(artifact) => LoadedLinearPrior::load(artifact, expected_feature_order)
                .map_or(Self::DeterministicBaseline, |loaded| {
                    Self::Learned(Box::new(loaded))
                }),
            None => Self::DeterministicBaseline,
        }
    }

    /// Computes the prior state, or `None` when running on the baseline.
    ///
    /// # Errors
    ///
    /// Propagates prediction failures from the loaded prior; the caller can
    /// then fall back per frame without abandoning the prior entirely.
    pub fn predict(&self, features: &[f32]) -> Result<Option<Vec<f32>>, LinearPriorLoadError> {
        match self {
            Self::Learned(loaded) => loaded.predict(features).map(Some),
            Self::DeterministicBaseline => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arkit_teacher::PairedTemporalSample;
    use crate::causal_dataset::{CausalFeatureConfig, build_causal_dataset};
    use crate::causal_prior::{LinearPriorTrainingConfig, fit_linear_prior};
    use std::collections::BTreeSet;
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

    fn trained_artifact() -> LinearPriorArtifact {
        let values = [0.1_f32, 0.5, 0.9, 0.7, 0.2];
        let samples: Vec<PairedTemporalSample> = values
            .iter()
            .enumerate()
            .map(|(index, value)| sample(index as u64 + 1, *value))
            .collect();
        let config = CausalFeatureConfig {
            history_len: 2,
            max_gap_micros: 40_000,
        };
        let rows = build_causal_dataset("take-a", &samples, config)
            .unwrap()
            .rows;
        let takes = BTreeSet::from(["take-a".to_owned()]);
        fit_linear_prior(
            &rows,
            &takes,
            LinearPriorTrainingConfig::default(),
            "history+velocity",
        )
        .expect("fit succeeds")
    }

    #[test]
    fn json_round_trip_loads_and_predicts_parity_with_reference() {
        let artifact = trained_artifact();
        // Offline reference inference: serialize through JSON exactly as an
        // export would, then compute by hand from the parsed fields.
        let json = serde_json::to_string(&artifact).unwrap();
        let parsed: LinearPriorArtifact = serde_json::from_str(&json).unwrap();

        let loaded = LoadedLinearPrior::load(parsed, "history+velocity").expect("loads");
        let features = vec![0.3_f32; loaded.feature_dim()];
        let prediction = loaded.predict(&features).expect("predicts");

        let mut reference = artifact.target_mean.clone();
        for (target_index, weight_row) in artifact.weights.iter().enumerate() {
            let mut sum = 0.0_f32;
            for (index, value) in features.iter().enumerate() {
                sum += (*value - artifact.feature_mean[index]) / artifact.feature_std[index]
                    * weight_row[index];
            }
            reference[target_index] += sum * artifact.target_std[target_index];
        }
        assert_eq!(prediction, reference);
        assert_eq!(prediction.len(), artifact.target_mean.len());
    }

    #[test]
    fn tampered_hash_version_and_order_are_typed_rejections() {
        let artifact = trained_artifact();

        let mut bad_hash = artifact.clone();
        bad_hash.content_hash = bad_hash.content_hash.wrapping_add(1);
        assert!(matches!(
            LoadedLinearPrior::load(bad_hash, "history+velocity"),
            Err(LinearPriorLoadError::ContentHashMismatch { .. })
        ));

        let mut bad_schema = artifact.clone();
        bad_schema.schema_version = 1;
        assert!(matches!(
            LoadedLinearPrior::load(bad_schema, "history+velocity"),
            Err(LinearPriorLoadError::UnsupportedSchemaVersion { .. })
        ));

        assert!(matches!(
            LoadedLinearPrior::load(artifact.clone(), "different-order"),
            Err(LinearPriorLoadError::FeatureOrderMismatch { .. })
        ));

        let mut truncated = artifact;
        truncated
            .feature_mean
            .truncate(truncated.feature_mean.len() - 1);
        truncated
            .feature_std
            .truncate(truncated.feature_std.len() - 1);
        // Keep the hash consistent with the mutated fields so the dimension
        // check is the failure that fires, not the hash check.
        truncated.content_hash = hash_artifact(&truncated);
        assert!(matches!(
            LoadedLinearPrior::load(truncated, "history+velocity"),
            Err(LinearPriorLoadError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn missing_artifact_falls_back_to_deterministic_baseline() {
        let inference = PriorInference::load_or_baseline(None, "history+velocity");
        assert_eq!(inference, PriorInference::DeterministicBaseline);
        assert!(inference.predict(&[0.0]).unwrap().is_none());

        // An invalid artifact also degrades instead of poisoning the runtime.
        let mut invalid = trained_artifact();
        invalid.content_hash ^= 0xffff;
        let degraded = PriorInference::load_or_baseline(Some(invalid), "history+velocity");
        assert_eq!(degraded, PriorInference::DeterministicBaseline);

        // A valid artifact predicts Some(values).
        let valid = PriorInference::load_or_baseline(Some(trained_artifact()), "history+velocity");
        let features = vec![0.0_f32; valid.predict(&[]).map(|_| 2).unwrap_or_else(|_| 2)];
        let _ = features;
        if let PriorInference::Learned(loaded) = &valid {
            let out = loaded
                .predict(&vec![0.0_f32; loaded.feature_dim()])
                .unwrap();
            assert_eq!(out.len(), loaded.target_dim());
        } else {
            panic!("valid artifact should load");
        }
    }

    #[test]
    fn non_finite_input_is_fail_closed_at_the_boundary() {
        let loaded = LoadedLinearPrior::load(trained_artifact(), "history+velocity").unwrap();
        let mut bad = vec![0.0_f32; loaded.feature_dim()];
        bad[0] = f32::NAN;
        assert!(matches!(
            loaded.predict(&bad),
            Err(LinearPriorLoadError::InvalidNormalization { .. })
        ));
    }
}
