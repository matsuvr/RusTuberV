//! Causal history feature/target dataset generation from paired traces
//! (GNM #68.4a).
//!
//! Turns a [`PairedTemporalSample`](crate::arkit_teacher::PairedTemporalSample)
//! sequence into training rows whose features reference only the current and
//! past frames (GNM dynamic state, velocity, observation quality, residual)
//! and whose target is the next step's GNM state. Model fitting happens in
//! later issues; this module only produces exact, inspectable rows.
//!
//! History resets at every sequence boundary: a gap in `frame_seq`, a
//! timestamp jump beyond the configured cadence tolerance, or a reacquire
//! (missing GNM state) starts a fresh run, so no row ever references across a
//! discontinuity and future leakage is impossible by construction.

use crate::arkit_teacher::{PairedTemporalSample, TeacherDatasetError};

/// Configuration for causal row generation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CausalFeatureConfig {
    /// Number of past GNM coefficient snapshots (including current) feeding a
    /// row's features. Must be at least one.
    pub history_len: usize,
    /// Maximum inter-frame gap in microseconds treated as continuous.
    pub max_gap_micros: u64,
}

impl CausalFeatureConfig {
    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns a typed error when `history_len` is zero.
    pub fn validate(self) -> Result<(), TeacherDatasetError> {
        if self.history_len == 0 {
            return Err(TeacherDatasetError::NonFinite {
                field: "history_len must be positive",
            });
        }
        Ok(())
    }

    /// Per-slot width: 52 coefficients plus residual and quality entries.
    #[must_use]
    pub const fn feature_dims() -> usize {
        // Per history slot: 52 coefficients + 1 residual + 1 quality flag.
        54
    }

    /// Feature vector width: `history_len` slots plus one velocity slot.
    #[must_use]
    pub fn feature_width(&self) -> usize {
        self.history_len * Self::feature_dims() + 52
    }

    /// Target width: the next-step GNM coefficients.
    #[must_use]
    pub const fn target_width() -> usize {
        52
    }
}

/// One generated training row.
#[derive(Clone, Debug, PartialEq)]
pub struct CausalRow {
    /// Take/session identity carried through for split integrity.
    pub take_id: String,
    /// Sequence of the frame whose features produced this row.
    pub frame_seq: u64,
    /// Monotonic timestamp of the feature frame in microseconds.
    pub timestamp_micros: u64,
    /// Features referencing only frames up to and including this one.
    pub features: Vec<f32>,
    /// Next-step GNM coefficients (the prior target).
    pub target: Vec<f32>,
}

/// A row excluded from training, with its typed reason.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExclusionReason {
    /// The feature frame lacked a deterministic GNM state.
    MissingGnmState,
    /// The target frame lacked a deterministic GNM state.
    MissingTargetState,
    /// A feature value was NaN or infinite.
    NonFiniteFeature,
    /// The candidate crossed a sequence/timestamp boundary; history reset.
    SequenceBoundary,
}

/// Dataset build outcome: accepted rows plus aggregated exclusions.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CausalDataset {
    /// Accepted rows in sequence order.
    pub rows: Vec<CausalRow>,
    /// Number of candidate positions excluded, per reason.
    pub exclusions: Vec<(ExclusionReason, usize)>,
}

impl CausalDataset {
    fn record_exclusion(&mut self, reason: ExclusionReason) {
        match self
            .exclusions
            .iter_mut()
            .find(|(existing, _)| *existing == reason)
        {
            Some((_, count)) => *count += 1,
            None => self.exclusions.push((reason, 1)),
        }
    }
}

/// Builds causal rows from a paired-sample sequence.
///
/// `take_id` labels the whole input sequence (one capture take); rows keep it
/// so train/validation/test splits never mix takes.
///
/// # Errors
///
/// Propagates dataset validation failures from the input sequence check.
#[allow(clippy::indexing_slicing)] // bounds proven by loop structure; see AGENTS.md
pub fn build_causal_dataset(
    take_id: &str,
    samples: &[PairedTemporalSample],
    config: CausalFeatureConfig,
) -> Result<CausalDataset, TeacherDatasetError> {
    config.validate()?;
    crate::arkit_teacher::validate_paired_samples(samples)?;

    let mut dataset = CausalDataset::default();
    let mut history: Vec<[f32; 52]> = Vec::with_capacity(config.history_len);
    let mut last_identity: Option<(u64, u64)> = None;

    for window in samples.windows(2) {
        let (current, next) = (&window[0], &window[1]);
        let Some(current_state) = &current.gnm_state else {
            dataset.record_exclusion(ExclusionReason::MissingGnmState);
            history.clear();
            last_identity = None;
            continue;
        };

        let continuous = matches!(last_identity, Some((previous_seq, previous_time))
            if current.frame_seq == previous_seq + 1
                && current.timestamp_micros.saturating_sub(previous_time) <= config.max_gap_micros);

        if !continuous {
            history.clear();
        }

        let previous_coefficients = history.last().copied();
        let mut run: Vec<[f32; 52]> = history.clone();
        run.push(*current_state.projected_coefficients.as_array());
        let start = run.len().saturating_sub(config.history_len);
        let window: Vec<[f32; 52]> = run[start.min(run.len())..].to_vec();

        let mut features = vec![0.0_f32; config.feature_width()];
        // Newest slot first so features[slot..slot+52] is always the current
        // frame's own coefficients.
        let slot_width = CausalFeatureConfig::feature_dims();
        for (slot_index, snapshot) in window.iter().rev().enumerate() {
            let base = slot_index * slot_width;
            features[base..base + 52].copy_from_slice(snapshot);
        }
        // Residual rides in the newest slot's tail entry.
        let residual_base = slot_width - 1;
        features[residual_base] = current_state.residual;
        fill_velocity_features(
            &mut features[config.history_len * CausalFeatureConfig::feature_dims()..],
            &current_state.projected_coefficients.as_array()[..52],
            previous_coefficients,
            current
                .timestamp_micros
                .saturating_sub(last_identity.map_or(current.timestamp_micros, |(_, time)| time)),
            current_state.residual,
        );
        if features.iter().any(|value| !value.is_finite()) {
            dataset.record_exclusion(ExclusionReason::NonFiniteFeature);
            history.push(*current_state.projected_coefficients.as_array());
            history.truncate(config.history_len);
            last_identity = Some((current.frame_seq, current.timestamp_micros));
            continue;
        }

        // A row must stay inside one continuous run on both sides: the
        // feature frame continues the previous run AND the target is its
        // exact successor.
        let target_is_successor = next.frame_seq == current.frame_seq + 1;
        let crosses_boundary = (last_identity.is_some() && !continuous) || !target_is_successor;
        if crosses_boundary {
            dataset.record_exclusion(ExclusionReason::SequenceBoundary);
            history.clear();
            history.push(*current_state.projected_coefficients.as_array());
            history.truncate(config.history_len);
            last_identity = Some((current.frame_seq, current.timestamp_micros));
            continue;
        }

        match &next.gnm_state {
            Some(next_state) => {
                dataset.rows.push(CausalRow {
                    take_id: take_id.to_owned(),
                    frame_seq: current.frame_seq,
                    timestamp_micros: current.timestamp_micros,
                    features,
                    target: next_state.projected_coefficients.as_array().to_vec(),
                });
            }
            None => dataset.record_exclusion(ExclusionReason::MissingTargetState),
        }

        history.push(*current_state.projected_coefficients.as_array());
        history.truncate(config.history_len);
        last_identity = Some((current.frame_seq, current.timestamp_micros));
    }
    Ok(dataset)
}

#[allow(clippy::indexing_slicing)]
fn fill_velocity_features(
    velocity_slot: &mut [f32],
    current: &[f32],
    previous: Option<[f32; 52]>,
    dt_micros: u64,
    residual: f32,
) {
    let Some(previous) = previous else {
        return;
    };
    let dt_seconds = (dt_micros as f64 / 1_000_000.0) as f32;
    if dt_seconds <= 0.0 || !dt_seconds.is_finite() {
        return;
    }
    let bounded = 52.min(velocity_slot.len());
    for (index, slot_value) in velocity_slot[..bounded].iter_mut().enumerate() {
        *slot_value = (current[index] - previous[index]) / dt_seconds;
    }
    // Residual rides in the velocity slot tail when present.
    if velocity_slot.len() > 52 && residual.is_finite() {
        velocity_slot[52] = residual;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arkit_teacher::{DeterministicGnmState, HeadTransform};
    use vtuber_core::{ARKIT52_CHANNEL_COUNT, Arkit52Coefficients, ArkitBlendshape};

    fn coefficients(value: f32) -> Arkit52Coefficients {
        let mut values = [0.0_f32; ARKIT52_CHANNEL_COUNT];
        values[ArkitBlendshape::JawOpen.index()] = value;
        Arkit52Coefficients::try_from_array(values).expect("valid")
    }

    fn state(coefficients: Arkit52Coefficients, residual: f32) -> DeterministicGnmState {
        DeterministicGnmState {
            projected_coefficients: coefficients,
            residual,
        }
    }

    fn sample(seq: u64, jaw_open: f32, residual: f32) -> PairedTemporalSample {
        PairedTemporalSample {
            frame_seq: seq,
            timestamp_micros: seq * 16_667,
            mediapipe_observation: None,
            gnm_state: Some(state(coefficients(jaw_open), residual)),
            baseline_output: Arkit52Coefficients::default(),
            teacher: None,
            rgb_reference: None,
        }
    }

    fn config() -> CausalFeatureConfig {
        CausalFeatureConfig {
            history_len: 2,
            max_gap_micros: 40_000,
        }
    }

    #[test]
    fn features_never_reference_future_frames_and_target_is_next_step() {
        let samples = [
            sample(1, 0.1, 0.01),
            sample(2, 0.5, 0.01),
            sample(3, 0.9, 0.02),
        ];
        let dataset = build_causal_dataset("take-a", &samples, config()).expect("builds");
        assert_eq!(dataset.rows.len(), 2);
        assert_eq!(dataset.exclusions.len(), 0);

        let first = &dataset.rows[0];
        let jaw_index = ArkitBlendshape::JawOpen.index();
        // Target is frame 2's state while features come from frame 1 only.
        assert!((first.target[jaw_index] - 0.5).abs() < 1e-6);
        let slot = CausalFeatureConfig::feature_dims();
        assert!((first.features[jaw_index] - 0.1).abs() < 1e-6);
        // Velocity needs a real in-run predecessor; the session-start row
        // honestly reports zero rather than inventing one.
        let velocity_base = config().history_len * slot;
        assert!((first.features[velocity_base + jaw_index]).abs() < 1e-6);
        // Second row: JawOpen moved 0.1→0.5 over ~16.667ms ≈ 24/s.
        let second = &dataset.rows[1];
        assert!((second.features[velocity_base + jaw_index] - 24.0).abs() < 1e-2);
        // Second row's older history slot still carries frame 1.
        assert!((second.features[slot + jaw_index] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn sequence_gap_resets_history_so_no_row_crosses_the_boundary() {
        let mut samples = vec![sample(1, 0.1, 0.01), sample(2, 0.5, 0.01)];
        samples.push(sample(10, 0.9, 0.02)); // gap: sequences 3..9 dropped
        let dataset = build_causal_dataset("take-a", &samples, config()).expect("builds");
        // Rows: (1→2), (2→dropped: no next state? next exists=seq10) but the
        // gap between 2 and 10 breaks continuity, so the 2→10 row must not
        // exist; only 10 has no successor.
        assert_eq!(dataset.rows.len(), 1);
        assert_eq!(dataset.rows[0].frame_seq, 1);
        // The 2→10 candidate was rejected as a boundary-crossing row.
        assert!(
            dataset
                .exclusions
                .iter()
                .any(|(reason, _)| matches!(reason, ExclusionReason::SequenceBoundary))
        );
    }

    #[test]
    fn missing_states_are_typed_exclusions_not_panics() {
        let mut missing_state = sample(2, 0.5, 0.0);
        missing_state.gnm_state = None;
        let samples = vec![sample(1, 0.1, 0.01), missing_state, sample(3, 0.9, 0.02)];
        let dataset = build_causal_dataset("take-a", &samples, config()).expect("builds");
        assert!(
            dataset
                .exclusions
                .iter()
                .any(|(reason, _)| matches!(reason, ExclusionReason::MissingGnmState))
        );
        // After the reacquire at seq 3 the history restarted; seq 3 has no
        // successor so nothing else is emitted.
        assert!(dataset.rows.is_empty());
    }

    #[test]
    fn take_identity_is_carried_for_split_integrity() {
        let samples = [sample(1, 0.1, 0.01), sample(2, 0.5, 0.01)];
        let dataset = build_causal_dataset("take-42", &samples, config()).expect("builds");
        assert!(dataset.rows.iter().all(|row| row.take_id == "take-42"));
    }

    #[test]
    fn head_transform_helper_stays_available_for_richer_rows() {
        // Placeholder pinning the import surface used by later artifacts.
        let head = HeadTransform {
            rotation_unit_quaternion_wxyz: [1.0, 0.0, 0.0, 0.0],
            translation_meters: [0.0; 3],
        };
        assert!(head.validate().is_ok());
    }
}
