//! Pure reduced-state temporal prediction and fixed gain selection (Issue #21).

use serde::{Deserialize, Serialize};
use vtuber_core::{
    ARKIT_NON_TONGUE_CHANNEL_COUNT, Arkit52Coefficients, arkit_non_tongue_values,
    arkit52_with_zero_tongue,
};
use vtuber_gnm::{GnmReducedExpressionBasis, GnmReducedExpressionState};

/// Reduced temporal artifact schema.
pub const REDUCED_TEMPORAL_ARTIFACT_SCHEMA_VERSION: u32 = 1;

/// One alpha-beta filter gain pair.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AlphaBetaGain {
    /// Position correction fraction.
    pub alpha: f32,
    /// Velocity correction fraction.
    pub beta: f32,
}

/// Responsive fixed preset.
pub const RESPONSIVE_GAIN: AlphaBetaGain = AlphaBetaGain {
    alpha: 0.90,
    beta: 0.20,
};
/// Balanced fixed preset.
pub const BALANCED_GAIN: AlphaBetaGain = AlphaBetaGain {
    alpha: 0.70,
    beta: 0.10,
};
/// Smooth fixed preset.
pub const SMOOTH_GAIN: AlphaBetaGain = AlphaBetaGain {
    alpha: 0.50,
    beta: 0.05,
};

/// Per-coordinate reduced alpha-beta gains.
#[derive(Clone, Debug, PartialEq)]
pub struct ReducedAlphaBetaGains {
    /// Position gains in basis-column order.
    pub alpha: Vec<f32>,
    /// Velocity gains in basis-column order.
    pub beta: Vec<f32>,
}

/// Source-space gains projected through basis-column energy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SourceGroupGains {
    /// Shared left/right-eye gain.
    pub eye: AlphaBetaGain,
    /// Lower-face gain.
    pub lower_face: AlphaBetaGain,
    /// Iris gain.
    pub iris: AlphaBetaGain,
}

/// Accepted reduced state and its causal velocity estimate.
#[derive(Clone, Debug, PartialEq)]
pub struct ReducedTemporalState {
    /// Source timestamp at which the state was corrected.
    pub corrected_at_micros: u64,
    /// Corrected reduced position.
    pub position: GnmReducedExpressionState,
    /// Reduced velocity per second.
    pub velocity_per_second: Vec<f32>,
}

/// Two direct observations used for causal secant prediction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimestampedDirectCoefficients {
    /// Source timestamp.
    pub timestamp_micros: u64,
    /// Direct MediaPipe coefficients.
    pub coefficients: Arkit52Coefficients,
}

/// One fixed gain candidate's training-side validation metrics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TemporalCandidateMetrics {
    /// Eye and iris preset.
    pub eye_preset: AlphaBetaGain,
    /// Lower-face preset.
    pub lower_face_preset: AlphaBetaGain,
    /// Hybrid macro 51-channel value MAE.
    pub h_macro_mae: f64,
    /// Hybrid missed blink count.
    pub h_missed_blinks: usize,
    /// Direct missed blink count on the same validation frames.
    pub d_missed_blinks: usize,
    /// Hybrid median absolute onset timing error in milliseconds.
    pub h_onset_error_ms: Option<f64>,
    /// Direct counterpart.
    pub d_onset_error_ms: Option<f64>,
    /// Hybrid median absolute peak timing error in milliseconds.
    pub h_peak_error_ms: Option<f64>,
    /// Direct counterpart.
    pub d_peak_error_ms: Option<f64>,
    /// Hybrid median absolute peak attenuation.
    pub h_peak_attenuation: Option<f64>,
    /// Direct counterpart.
    pub d_peak_attenuation: Option<f64>,
}

impl TemporalCandidateMetrics {
    /// Returns whether all four mandatory blink constraints hold.
    pub fn admissible(&self) -> bool {
        let (
            Some(h_onset),
            Some(d_onset),
            Some(h_peak),
            Some(d_peak),
            Some(h_attenuation),
            Some(d_attenuation),
        ) = (
            self.h_onset_error_ms,
            self.d_onset_error_ms,
            self.h_peak_error_ms,
            self.d_peak_error_ms,
            self.h_peak_attenuation,
            self.d_peak_attenuation,
        )
        else {
            return false;
        };
        self.h_missed_blinks <= self.d_missed_blinks
            && h_onset <= d_onset
            && h_peak <= d_peak
            && h_attenuation <= d_attenuation
    }
}

/// Provenance shared by all nine gain candidates.
#[derive(Clone, Debug, PartialEq)]
pub struct ReducedTemporalProvenance {
    /// Selected teacher-aligned basis hash.
    pub basis_content_hash: u64,
    /// Selected H decoder hash.
    pub decoder_content_hash: u64,
    /// Maximum allowed render prediction horizon.
    pub max_prediction_horizon_micros: u64,
    /// Outer-training takes only.
    pub training_takes: Vec<String>,
}

/// Selected reduced temporal gain artifact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReducedTemporalArtifact {
    /// Artifact schema.
    pub schema_version: u32,
    /// Teacher-aligned basis hash.
    pub basis_content_hash: u64,
    /// H decoder hash.
    pub decoder_content_hash: u64,
    /// Selected eye/iris preset.
    pub eye_preset: AlphaBetaGain,
    /// Selected lower-face preset.
    pub lower_face_preset: AlphaBetaGain,
    /// Maximum render prediction horizon.
    pub max_prediction_horizon_micros: u64,
    /// Training-only take ids.
    pub training_takes: Vec<String>,
    /// All nine validation results.
    pub validation_metrics: Vec<TemporalCandidateMetrics>,
    /// Stable FNV-1a content hash over the preceding fields.
    pub content_hash: u64,
}

/// Typed reduced temporal failure.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum ReducedTemporalError {
    /// State/gain dimensions differ.
    #[error("reduced temporal {field} dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Field name.
        field: &'static str,
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },
    /// Gain/state number is invalid.
    #[error("invalid reduced temporal numeric value: {0}")]
    InvalidNumeric(&'static str),
    /// Source timestamps are not strictly increasing.
    #[error("reduced temporal source timestamps are not strictly increasing")]
    TimestampOrder,
    /// Requested timestamp precedes the corrected/current state.
    #[error("target timestamp precedes the current temporal state")]
    PastTarget,
    /// Requested prediction crosses the explicit horizon.
    #[error("prediction horizon {actual_micros}us exceeds {maximum_micros}us")]
    PredictionHorizon {
        /// Requested horizon.
        actual_micros: u64,
        /// Configured maximum.
        maximum_micros: u64,
    },
    /// None of the fixed nine candidates meets the blink constraints.
    #[error("no reduced temporal gain candidate satisfies all blink constraints")]
    NoAdmissibleTemporalGains,
    /// Candidate set differs from the fixed three-by-three grid.
    #[error("reduced temporal candidate set must contain the fixed nine unique gain pairs")]
    InvalidCandidateGrid,
    /// ARKit output validation failed.
    #[error(transparent)]
    Arkit(#[from] vtuber_core::Arkit52ValueError),
}

fn valid_gain(gain: AlphaBetaGain) -> bool {
    gain.alpha.is_finite()
        && gain.beta.is_finite()
        && (0.0..=1.0).contains(&gain.alpha)
        && (0.0..=1.0).contains(&gain.beta)
}

/// Returns the fixed eye-by-lower-face three-by-three gain grid.
pub fn reduced_temporal_gain_grid() -> [(AlphaBetaGain, AlphaBetaGain); 9] {
    [
        (RESPONSIVE_GAIN, RESPONSIVE_GAIN),
        (RESPONSIVE_GAIN, BALANCED_GAIN),
        (RESPONSIVE_GAIN, SMOOTH_GAIN),
        (BALANCED_GAIN, RESPONSIVE_GAIN),
        (BALANCED_GAIN, BALANCED_GAIN),
        (BALANCED_GAIN, SMOOTH_GAIN),
        (SMOOTH_GAIN, RESPONSIVE_GAIN),
        (SMOOTH_GAIN, BALANCED_GAIN),
        (SMOOTH_GAIN, SMOOTH_GAIN),
    ]
}

/// Projects source-group gains into each basis column by squared energy.
pub fn project_source_group_gains(
    basis: &GnmReducedExpressionBasis,
    source: SourceGroupGains,
) -> Result<ReducedAlphaBetaGains, ReducedTemporalError> {
    if !valid_gain(source.eye) || !valid_gain(source.lower_face) || !valid_gain(source.iris) {
        return Err(ReducedTemporalError::InvalidNumeric("source gain"));
    }
    let mut alpha = vec![0.0; basis.rank()];
    let mut beta = vec![0.0; basis.rank()];
    for column in 0..basis.rank() {
        let mut eye = 0.0_f64;
        let mut lower = 0.0_f64;
        let mut iris = 0.0_f64;
        for (row, values) in basis
            .values_row_major()
            .chunks_exact(basis.rank())
            .enumerate()
        {
            #[allow(clippy::indexing_slicing)]
            let energy = f64::from(values[column]) * f64::from(values[column]);
            match row {
                0..200 => eye += energy,
                200..350 => lower += energy,
                350 => iris += energy,
                _ => {}
            }
        }
        let total = eye + lower + iris;
        if !total.is_finite() || total <= 0.0 {
            return Err(ReducedTemporalError::InvalidNumeric("basis column energy"));
        }
        #[allow(clippy::indexing_slicing)]
        {
            alpha[column] = ((eye * f64::from(source.eye.alpha)
                + lower * f64::from(source.lower_face.alpha)
                + iris * f64::from(source.iris.alpha))
                / total) as f32;
            beta[column] = ((eye * f64::from(source.eye.beta)
                + lower * f64::from(source.lower_face.beta)
                + iris * f64::from(source.iris.beta))
                / total) as f32;
        }
    }
    Ok(ReducedAlphaBetaGains { alpha, beta })
}

/// Initializes a zero-velocity temporal state from one accepted observation.
pub fn initialize_reduced_temporal_state(
    observed_at_micros: u64,
    observed: GnmReducedExpressionState,
) -> ReducedTemporalState {
    ReducedTemporalState {
        corrected_at_micros: observed_at_micros,
        velocity_per_second: vec![0.0; observed.values().len()],
        position: observed,
    }
}

/// Applies one causal per-coordinate alpha-beta correction.
pub fn correct_reduced_temporal_state(
    previous: &ReducedTemporalState,
    observed_at_micros: u64,
    observed: &GnmReducedExpressionState,
    gains: &ReducedAlphaBetaGains,
) -> Result<ReducedTemporalState, ReducedTemporalError> {
    let rank = previous.position.values().len();
    for (field, actual) in [
        ("observed", observed.values().len()),
        ("velocity", previous.velocity_per_second.len()),
        ("alpha", gains.alpha.len()),
        ("beta", gains.beta.len()),
    ] {
        if actual != rank {
            return Err(ReducedTemporalError::DimensionMismatch {
                field,
                expected: rank,
                actual,
            });
        }
    }
    if observed_at_micros <= previous.corrected_at_micros {
        return Err(ReducedTemporalError::TimestampOrder);
    }
    if gains
        .alpha
        .iter()
        .copied()
        .zip(gains.beta.iter().copied())
        .any(|(alpha, beta)| !valid_gain(AlphaBetaGain { alpha, beta }))
    {
        return Err(ReducedTemporalError::InvalidNumeric("reduced gain"));
    }
    let dt = (observed_at_micros - previous.corrected_at_micros) as f32 / 1_000_000.0;
    let mut position = Vec::with_capacity(rank);
    let mut velocity = Vec::with_capacity(rank);
    for (((previous_position, previous_velocity), observed), (alpha, beta)) in previous
        .position
        .values()
        .iter()
        .zip(&previous.velocity_per_second)
        .zip(observed.values())
        .zip(gains.alpha.iter().zip(&gains.beta))
    {
        let predicted = previous_position + dt * previous_velocity;
        let innovation = observed - predicted;
        position.push(predicted + alpha * innovation);
        velocity.push(previous_velocity + beta / dt * innovation);
    }
    Ok(ReducedTemporalState {
        corrected_at_micros: observed_at_micros,
        position: GnmReducedExpressionState::new(position, rank)
            .map_err(|_| ReducedTemporalError::InvalidNumeric("corrected state"))?,
        velocity_per_second: velocity,
    })
}

/// Predicts the reduced state at a render timestamp without hold or clamping.
pub fn sample_reduced_state_at(
    state: &ReducedTemporalState,
    target_micros: u64,
    max_prediction_horizon_micros: u64,
) -> Result<GnmReducedExpressionState, ReducedTemporalError> {
    let horizon = target_micros
        .checked_sub(state.corrected_at_micros)
        .ok_or(ReducedTemporalError::PastTarget)?;
    if horizon > max_prediction_horizon_micros {
        return Err(ReducedTemporalError::PredictionHorizon {
            actual_micros: horizon,
            maximum_micros: max_prediction_horizon_micros,
        });
    }
    if state.velocity_per_second.len() != state.position.values().len() {
        return Err(ReducedTemporalError::DimensionMismatch {
            field: "velocity",
            expected: state.position.values().len(),
            actual: state.velocity_per_second.len(),
        });
    }
    let dt = horizon as f32 / 1_000_000.0;
    let values = state
        .position
        .values()
        .iter()
        .zip(&state.velocity_per_second)
        .map(|(position, velocity)| position + dt * velocity)
        .collect::<Vec<_>>();
    GnmReducedExpressionState::new(values, state.position.values().len())
        .map_err(|_| ReducedTemporalError::InvalidNumeric("sampled state"))
}

/// Secant-predicts direct non-tongue coefficients at a render timestamp.
pub fn sample_direct_coefficients_at(
    previous: &TimestampedDirectCoefficients,
    current: &TimestampedDirectCoefficients,
    target_micros: u64,
    max_prediction_horizon_micros: u64,
) -> Result<Arkit52Coefficients, ReducedTemporalError> {
    if previous.timestamp_micros >= current.timestamp_micros {
        return Err(ReducedTemporalError::TimestampOrder);
    }
    let horizon = target_micros
        .checked_sub(current.timestamp_micros)
        .ok_or(ReducedTemporalError::PastTarget)?;
    if horizon > max_prediction_horizon_micros {
        return Err(ReducedTemporalError::PredictionHorizon {
            actual_micros: horizon,
            maximum_micros: max_prediction_horizon_micros,
        });
    }
    let source_dt = (current.timestamp_micros - previous.timestamp_micros) as f32 / 1_000_000.0;
    let target_dt = horizon as f32 / 1_000_000.0;
    let before = arkit_non_tongue_values(&previous.coefficients);
    let now = arkit_non_tongue_values(&current.coefficients);
    let mut predicted = [0.0; ARKIT_NON_TONGUE_CHANNEL_COUNT];
    for ((output, before), now) in predicted.iter_mut().zip(before).zip(now) {
        *output = (now + (now - before) / source_dt * target_dt).clamp(0.0, 1.0);
    }
    Ok(arkit52_with_zero_tongue(predicted)?)
}

fn same_gain(left: AlphaBetaGain, right: AlphaBetaGain) -> bool {
    left.alpha.to_bits() == right.alpha.to_bits() && left.beta.to_bits() == right.beta.to_bits()
}

fn fixed_grid(candidates: &[TemporalCandidateMetrics]) -> bool {
    let grid = reduced_temporal_gain_grid();
    candidates.len() == grid.len()
        && grid.iter().all(|(eye, lower)| {
            candidates
                .iter()
                .filter(|candidate| {
                    same_gain(candidate.eye_preset, *eye)
                        && same_gain(candidate.lower_face_preset, *lower)
                })
                .count()
                == 1
        })
}

fn finite_metrics(candidate: &TemporalCandidateMetrics) -> bool {
    candidate.h_macro_mae.is_finite()
        && [
            candidate.h_onset_error_ms,
            candidate.d_onset_error_ms,
            candidate.h_peak_error_ms,
            candidate.d_peak_error_ms,
            candidate.h_peak_attenuation,
            candidate.d_peak_attenuation,
        ]
        .into_iter()
        .flatten()
        .all(f64::is_finite)
}

fn hash_artifact(artifact: &ReducedTemporalArtifact) -> u64 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&artifact.schema_version.to_le_bytes());
    bytes.extend_from_slice(&artifact.basis_content_hash.to_le_bytes());
    bytes.extend_from_slice(&artifact.decoder_content_hash.to_le_bytes());
    for gain in [artifact.eye_preset, artifact.lower_face_preset] {
        bytes.extend_from_slice(&gain.alpha.to_bits().to_le_bytes());
        bytes.extend_from_slice(&gain.beta.to_bits().to_le_bytes());
    }
    bytes.extend_from_slice(&artifact.max_prediction_horizon_micros.to_le_bytes());
    for take in &artifact.training_takes {
        bytes.extend_from_slice(take.as_bytes());
        bytes.push(0xff);
    }
    for candidate in &artifact.validation_metrics {
        for gain in [candidate.eye_preset, candidate.lower_face_preset] {
            bytes.extend_from_slice(&gain.alpha.to_bits().to_le_bytes());
            bytes.extend_from_slice(&gain.beta.to_bits().to_le_bytes());
        }
        bytes.extend_from_slice(&candidate.h_macro_mae.to_bits().to_le_bytes());
        bytes.extend_from_slice(&(candidate.h_missed_blinks as u64).to_le_bytes());
        bytes.extend_from_slice(&(candidate.d_missed_blinks as u64).to_le_bytes());
        for value in [
            candidate.h_onset_error_ms,
            candidate.d_onset_error_ms,
            candidate.h_peak_error_ms,
            candidate.d_peak_error_ms,
            candidate.h_peak_attenuation,
            candidate.d_peak_attenuation,
        ] {
            match value {
                Some(value) => {
                    bytes.push(1);
                    bytes.extend_from_slice(&value.to_bits().to_le_bytes());
                }
                None => bytes.push(0),
            }
        }
    }
    bytes.into_iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Selects the best admissible candidate from the fixed nine-pair grid.
pub fn select_reduced_temporal_artifact(
    candidates: &[TemporalCandidateMetrics],
    mut provenance: ReducedTemporalProvenance,
) -> Result<ReducedTemporalArtifact, ReducedTemporalError> {
    if !fixed_grid(candidates) {
        return Err(ReducedTemporalError::InvalidCandidateGrid);
    }
    if provenance.max_prediction_horizon_micros == 0
        || candidates
            .iter()
            .any(|candidate| !finite_metrics(candidate))
    {
        return Err(ReducedTemporalError::InvalidNumeric("artifact input"));
    }
    let selected = candidates
        .iter()
        .filter(|candidate| candidate.admissible())
        .min_by(|left, right| {
            left.h_macro_mae
                .total_cmp(&right.h_macro_mae)
                .then_with(|| right.eye_preset.alpha.total_cmp(&left.eye_preset.alpha))
                .then_with(|| right.eye_preset.beta.total_cmp(&left.eye_preset.beta))
                .then_with(|| {
                    right
                        .lower_face_preset
                        .alpha
                        .total_cmp(&left.lower_face_preset.alpha)
                })
                .then_with(|| {
                    right
                        .lower_face_preset
                        .beta
                        .total_cmp(&left.lower_face_preset.beta)
                })
        })
        .ok_or(ReducedTemporalError::NoAdmissibleTemporalGains)?;
    provenance.training_takes.sort();
    provenance.training_takes.dedup();
    let mut artifact = ReducedTemporalArtifact {
        schema_version: REDUCED_TEMPORAL_ARTIFACT_SCHEMA_VERSION,
        basis_content_hash: provenance.basis_content_hash,
        decoder_content_hash: provenance.decoder_content_hash,
        eye_preset: selected.eye_preset,
        lower_face_preset: selected.lower_face_preset,
        max_prediction_horizon_micros: provenance.max_prediction_horizon_micros,
        training_takes: provenance.training_takes,
        validation_metrics: candidates.to_vec(),
        content_hash: 0,
    };
    artifact.content_hash = hash_artifact(&artifact);
    Ok(artifact)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vtuber_gnm::GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM;

    fn identity_prefix_basis(rank: usize) -> GnmReducedExpressionBasis {
        let mut values = vec![0.0; GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM * rank];
        for index in 0..rank {
            values[index * rank + index] = 1.0;
        }
        GnmReducedExpressionBasis::new(rank, values).unwrap()
    }

    #[test]
    fn group_projection_keeps_left_and_right_eye_identical() {
        let gains = project_source_group_gains(
            &identity_prefix_basis(201),
            SourceGroupGains {
                eye: RESPONSIVE_GAIN,
                lower_face: SMOOTH_GAIN,
                iris: BALANCED_GAIN,
            },
        )
        .unwrap();
        assert_eq!(gains.alpha[0], gains.alpha[100]);
        assert_eq!(gains.beta[0], gains.beta[100]);
        assert_eq!(gains.alpha[200], SMOOTH_GAIN.alpha);
    }

    #[test]
    fn alpha_beta_transition_and_sampling_use_timestamps() {
        let initial = initialize_reduced_temporal_state(
            1_000_000,
            GnmReducedExpressionState::new(vec![0.0], 1).unwrap(),
        );
        let corrected = correct_reduced_temporal_state(
            &initial,
            1_100_000,
            &GnmReducedExpressionState::new(vec![1.0], 1).unwrap(),
            &ReducedAlphaBetaGains {
                alpha: vec![0.5],
                beta: vec![0.1],
            },
        )
        .unwrap();
        assert!((corrected.position.values()[0] - 0.5).abs() < 1.0e-6);
        assert!((corrected.velocity_per_second[0] - 1.0).abs() < 1.0e-6);
        let sampled = sample_reduced_state_at(&corrected, 1_150_000, 50_000).unwrap();
        assert!((sampled.values()[0] - 0.55).abs() < 1.0e-6);
        assert!(sample_reduced_state_at(&corrected, 1_150_001, 50_000).is_err());
    }

    #[test]
    fn direct_prediction_is_secant_only_and_zeroes_tongue() {
        let mut first = [0.0; 52];
        let mut second = [0.0; 52];
        first[0] = 0.2;
        second[0] = 0.4;
        second[51] = 0.0;
        let predicted = sample_direct_coefficients_at(
            &TimestampedDirectCoefficients {
                timestamp_micros: 1_000_000,
                coefficients: Arkit52Coefficients::try_from_array(first).unwrap(),
            },
            &TimestampedDirectCoefficients {
                timestamp_micros: 1_100_000,
                coefficients: Arkit52Coefficients::try_from_array(second).unwrap(),
            },
            1_150_000,
            50_000,
        )
        .unwrap();
        assert!((predicted.as_array()[0] - 0.5).abs() < 1.0e-6);
        assert_eq!(predicted.as_array()[51], 0.0);
    }

    fn candidate(eye: AlphaBetaGain, lower: AlphaBetaGain, mae: f64) -> TemporalCandidateMetrics {
        TemporalCandidateMetrics {
            eye_preset: eye,
            lower_face_preset: lower,
            h_macro_mae: mae,
            h_missed_blinks: 0,
            d_missed_blinks: 0,
            h_onset_error_ms: Some(0.0),
            d_onset_error_ms: Some(0.0),
            h_peak_error_ms: Some(0.0),
            d_peak_error_ms: Some(0.0),
            h_peak_attenuation: Some(0.0),
            d_peak_attenuation: Some(0.0),
        }
    }

    #[test]
    fn selection_rejects_blink_failure_and_has_no_empty_fallback() {
        let mut candidates = reduced_temporal_gain_grid()
            .into_iter()
            .map(|(eye, lower)| candidate(eye, lower, 1.0))
            .collect::<Vec<_>>();
        candidates[8].h_macro_mae = 0.0;
        candidates[8].h_missed_blinks = 1;
        let artifact = select_reduced_temporal_artifact(
            &candidates,
            ReducedTemporalProvenance {
                basis_content_hash: 1,
                decoder_content_hash: 2,
                max_prediction_horizon_micros: 50_000,
                training_takes: vec!["b".to_owned(), "a".to_owned()],
            },
        )
        .unwrap();
        assert_ne!(artifact.eye_preset, SMOOTH_GAIN);
        for candidate in &mut candidates {
            candidate.h_missed_blinks = 1;
        }
        assert!(matches!(
            select_reduced_temporal_artifact(
                &candidates,
                ReducedTemporalProvenance {
                    basis_content_hash: 1,
                    decoder_content_hash: 2,
                    max_prediction_horizon_micros: 50_000,
                    training_takes: Vec::new(),
                }
            ),
            Err(ReducedTemporalError::NoAdmissibleTemporalGains)
        ));
    }

    #[test]
    fn candidate_with_missing_event_metric_is_not_admissible() {
        let mut candidate = candidate(RESPONSIVE_GAIN, RESPONSIVE_GAIN, 0.1);
        candidate.h_onset_error_ms = None;

        assert!(!candidate.admissible());
    }
}
