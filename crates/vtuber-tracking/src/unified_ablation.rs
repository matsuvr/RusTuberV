//! Pure common-frame and avatar-space metrics for the unified GNM ablation.

use std::collections::{BTreeMap, BTreeSet};

use vtuber_core::{
    ARKIT_NON_TONGUE_CHANNEL_COUNT, ARKIT_NON_TONGUE_LEFT_RIGHT_PAIRS, Arkit52Coefficients,
    ArkitBlendshape, arkit_non_tongue_values,
};

/// Fixed variants in the unified offline comparison.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum GnmResearchVariant {
    /// MediaPipe Direct.
    Direct,
    /// Existing hand-designed GNM projection.
    GnmProjected,
    /// Learned reduced-GNM decoder.
    GnmLearned,
    /// Direct plus landmark residual.
    LandmarkResidual,
    /// Direct plus reduced-GNM residual.
    HybridGnmResidual,
    /// Direct plus reduced-GNM and landmark residual.
    HybridGnmLandmarkResidual,
}

impl GnmResearchVariant {
    /// All required variants in report order.
    pub const ALL: [Self; 6] = [
        Self::Direct,
        Self::GnmProjected,
        Self::GnmLearned,
        Self::LandmarkResidual,
        Self::HybridGnmResidual,
        Self::HybridGnmLandmarkResidual,
    ];
}

/// One exact-frame teacher/output pair for one variant.
#[derive(Clone, Debug, PartialEq)]
pub struct VariantFrame {
    /// Capture take identity.
    pub take_id: String,
    /// Source frame sequence.
    pub frame_seq: u64,
    /// Source timestamp.
    pub timestamp_micros: u64,
    /// Same-frame ARKit teacher.
    pub teacher: Arkit52Coefficients,
    /// Variant output.
    pub output: Arkit52Coefficients,
}

/// Unified offline evaluation failure.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum UnifiedAblationError {
    /// A required fixed variant has no input series.
    #[error("missing required variant {0:?}")]
    MissingVariant(GnmResearchVariant),
    /// A variant contains the same exact identity more than once.
    #[error("duplicate frame identity in {0:?}")]
    DuplicateIdentity(GnmResearchVariant),
    /// Teachers disagree at an otherwise exact common identity.
    #[error("teacher mismatch at an exact common frame")]
    TeacherMismatch,
    /// No frames remain in the exact six-way intersection.
    #[error("no exact common frames")]
    NoCommonFrames,
    /// A metric received no frames.
    #[error("empty variant frame series")]
    EmptyFrames,
    /// Frame ordering, identity, coefficient, or response content is invalid.
    #[error("invalid unified ablation data: {0}")]
    InvalidData(&'static str),
    /// Avatar response has no non-tongue bound vertices.
    #[error("avatar morph response has no non-tongue vertices")]
    EmptyMorphResponse,
}

type FrameIdentity = (String, u64, u64);

/// Restricts all six variants to their exact `(take, sequence, timestamp)` intersection.
///
/// # Errors
///
/// Rejects missing variants, duplicate identities, teacher disagreement, or
/// an empty intersection. It never performs nearest joins, holds, or filling.
// Every result series is constructed from the same `common` identity list, so
// the shared `0..common.len()` index is in bounds for all six variants.
#[allow(clippy::indexing_slicing)]
pub fn exact_common_variant_frames(
    frames: &BTreeMap<GnmResearchVariant, Vec<VariantFrame>>,
) -> Result<BTreeMap<GnmResearchVariant, Vec<VariantFrame>>, UnifiedAblationError> {
    let mut indexed = BTreeMap::new();
    for variant in GnmResearchVariant::ALL {
        let source = frames
            .get(&variant)
            .ok_or(UnifiedAblationError::MissingVariant(variant))?;
        let mut by_identity = BTreeMap::new();
        for frame in source {
            let identity = (
                frame.take_id.clone(),
                frame.frame_seq,
                frame.timestamp_micros,
            );
            if by_identity.insert(identity, frame).is_some() {
                return Err(UnifiedAblationError::DuplicateIdentity(variant));
            }
        }
        indexed.insert(variant, by_identity);
    }

    let direct =
        indexed
            .get(&GnmResearchVariant::Direct)
            .ok_or(UnifiedAblationError::MissingVariant(
                GnmResearchVariant::Direct,
            ))?;
    let common: Vec<FrameIdentity> = direct
        .keys()
        .filter(|identity| {
            GnmResearchVariant::ALL.iter().all(|variant| {
                indexed
                    .get(variant)
                    .is_some_and(|series| series.contains_key(*identity))
            })
        })
        .cloned()
        .collect();
    if common.is_empty() {
        return Err(UnifiedAblationError::NoCommonFrames);
    }

    let mut result = BTreeMap::new();
    for variant in GnmResearchVariant::ALL {
        let series = indexed
            .get(&variant)
            .ok_or(UnifiedAblationError::MissingVariant(variant))?;
        let selected: Vec<VariantFrame> = common
            .iter()
            .filter_map(|identity| series.get(identity).map(|frame| (*frame).clone()))
            .collect();
        result.insert(variant, selected);
    }
    for index in 0..common.len() {
        let teacher = &result.get(&GnmResearchVariant::Direct).ok_or(
            UnifiedAblationError::MissingVariant(GnmResearchVariant::Direct),
        )?[index]
            .teacher;
        if GnmResearchVariant::ALL.iter().any(|variant| {
            result
                .get(variant)
                .is_some_and(|series| series[index].teacher != *teacher)
        }) {
            return Err(UnifiedAblationError::TeacherMismatch);
        }
    }
    Ok(result)
}

/// Accuracy for one non-tongue channel.
#[derive(Clone, Debug, PartialEq)]
pub struct NonTongueChannelMetrics {
    /// ARKit semantic.
    pub channel: ArkitBlendshape,
    /// Mean absolute error.
    pub mae: f64,
    /// Root mean square error.
    pub rmse: f64,
    /// Pearson correlation when the teacher has variance.
    pub pearson: Option<f64>,
    /// Signed prediction bias on frames where this teacher channel is <= 0.05.
    pub neutral_bias: Option<f64>,
}

/// Error in one real bilateral `(left - right)` semantic pair.
#[derive(Clone, Debug, PartialEq)]
pub struct LeftRightDifferenceMetrics {
    /// Left semantic.
    pub left: ArkitBlendshape,
    /// Right semantic.
    pub right: ArkitBlendshape,
    /// MAE of predicted versus teacher `(left - right)`.
    pub difference_mae: f64,
}

/// Timestamp-aware aggregate errors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NonTongueTemporalMetrics {
    /// Mean absolute first-derivative error.
    pub velocity_mae: f64,
    /// Mean absolute second-derivative error.
    pub acceleration_mae: f64,
    /// RMS predicted velocity while the corresponding teacher channel is neutral.
    pub neutral_jitter: f64,
    /// Mean absolute third-derivative error in intervals touching a teacher peak >= 0.5.
    pub peak_jerk_mae: f64,
}

/// Complete coefficient and temporal metrics for one variant.
#[derive(Clone, Debug, PartialEq)]
pub struct NonTongueVariantMetrics {
    /// Evaluated exact-common frame count.
    pub frame_count: usize,
    /// Mean of channel MAEs.
    pub macro_mae: f64,
    /// Mean of channel RMSE values.
    pub macro_rmse: f64,
    /// MAE over every frame/channel cell.
    pub micro_mae: f64,
    /// RMSE over every frame/channel cell.
    pub micro_rmse: f64,
    /// Per-channel accuracy in stable non-tongue order.
    pub channels: Vec<NonTongueChannelMetrics>,
    /// Bilateral semantic-difference errors.
    pub left_right: Vec<LeftRightDifferenceMetrics>,
    /// Timestamp-aware errors.
    pub temporal: NonTongueTemporalMetrics,
}

/// Teacher-driven blink event metrics for one exact-common variant.
#[derive(Clone, Debug, PartialEq)]
pub struct BlinkEventMetrics {
    /// Teacher pulse count across both eyes.
    pub teacher_events: usize,
    /// Teacher pulses with a matching output pulse.
    pub matched_events: usize,
    /// Teacher pulses without a matching output pulse.
    pub missed_events: usize,
    /// Output pulses left unmatched.
    pub extra_events: usize,
    /// Median absolute onset timing error in milliseconds.
    pub median_absolute_onset_error_ms: Option<f64>,
    /// Median absolute peak timing error in milliseconds.
    pub median_absolute_peak_error_ms: Option<f64>,
    /// Median absolute release timing error in milliseconds.
    pub median_absolute_release_error_ms: Option<f64>,
    /// Median absolute peak attenuation relative to teacher amplitude.
    pub median_absolute_peak_attenuation: Option<f64>,
}

#[derive(Clone, Copy)]
struct BlinkPulse {
    onset_micros: u64,
    peak_micros: u64,
    release_micros: u64,
    baseline: f64,
    peak: f64,
}

// Loop indices are generated from the same `times` and `values` series and
// remain within both arrays. The scan mirrors the established teacher-event
// detector's 0.30 onset, 0.3 s separation, and 350 ms release window.
#[allow(clippy::indexing_slicing)]
fn detect_blink_pulses(times: &[u64], values: &[f64]) -> Vec<BlinkPulse> {
    let mut pulses = Vec::new();
    let mut last_onset = 0;
    for index in 1..values.len() {
        let previous = values[index - 1];
        let current = values[index];
        let onset = times[index];
        if onset.saturating_sub(last_onset) < 300_000 || current < 0.30 || previous >= 0.6 * current
        {
            continue;
        }
        let mut peak = current;
        let mut peak_index = index;
        let mut release_index = None;
        for at in index..values.len() {
            if times[at].saturating_sub(onset) > 350_000 {
                break;
            }
            if values[at] > peak {
                peak = values[at];
                peak_index = at;
            }
            if at > peak_index && values[at] <= 0.5 * peak {
                release_index = Some(at);
                break;
            }
        }
        let Some(release_index) = release_index else {
            continue;
        };
        let baseline = if index >= 3 {
            values[index - 3..index].iter().sum::<f64>() / 3.0
        } else {
            previous
        };
        pulses.push(BlinkPulse {
            onset_micros: onset,
            peak_micros: times[peak_index],
            release_micros: times[release_index],
            baseline,
            peak,
        });
        last_onset = onset;
    }
    pulses
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        Some((values.get(middle - 1)? + values.get(middle)?) / 2.0)
    } else {
        values.get(middle).copied()
    }
}

/// Scores left and right blink pulses on exact common frames.
///
/// A match is an output pulse whose onset is in `[-150, +600]` ms around the
/// teacher onset and whose peak attenuation is at most 0.7, matching the
/// established teacher-ablation detector and detection criterion.
///
/// # Errors
///
/// Rejects empty, misordered, or invalid frame series.
pub fn evaluate_blink_events(
    frames: &[VariantFrame],
) -> Result<BlinkEventMetrics, UnifiedAblationError> {
    if frames.is_empty() {
        return Err(UnifiedAblationError::EmptyFrames);
    }
    validate_frame_order(frames)?;
    let mut teacher_events = 0;
    let mut matched_events = 0;
    let mut output_events = 0;
    let mut onset_errors = Vec::new();
    let mut peak_errors = Vec::new();
    let mut release_errors = Vec::new();
    let mut attenuations = Vec::new();
    for take in take_slices(frames) {
        let times = take
            .iter()
            .map(|frame| frame.timestamp_micros)
            .collect::<Vec<_>>();
        for channel in [
            ArkitBlendshape::EyeBlinkLeft,
            ArkitBlendshape::EyeBlinkRight,
        ] {
            let teacher = take
                .iter()
                .map(|frame| f64::from(frame.teacher.get(channel)))
                .collect::<Vec<_>>();
            let output = take
                .iter()
                .map(|frame| f64::from(frame.output.get(channel)))
                .collect::<Vec<_>>();
            let teacher_pulses = detect_blink_pulses(&times, &teacher);
            let output_pulses = detect_blink_pulses(&times, &output);
            teacher_events += teacher_pulses.len();
            output_events += output_pulses.len();
            let mut used = BTreeSet::new();
            for teacher_pulse in teacher_pulses {
                let selected = output_pulses
                    .iter()
                    .enumerate()
                    .filter(|(index, pulse)| {
                        !used.contains(index)
                            && pulse.onset_micros
                                >= teacher_pulse.onset_micros.saturating_sub(150_000)
                            && pulse.onset_micros
                                <= teacher_pulse.onset_micros.saturating_add(600_000)
                    })
                    .min_by_key(|(_, pulse)| pulse.peak_micros.abs_diff(teacher_pulse.peak_micros));
                let Some((index, output_pulse)) = selected else {
                    continue;
                };
                let amplitude = teacher_pulse.peak - teacher_pulse.baseline;
                if amplitude <= 0.0 {
                    return Err(UnifiedAblationError::InvalidData("blink amplitude"));
                }
                let attenuation = (teacher_pulse.peak - output_pulse.peak) / amplitude;
                if attenuation > 0.7 {
                    continue;
                }
                used.insert(index);
                matched_events += 1;
                onset_errors.push(
                    output_pulse
                        .onset_micros
                        .abs_diff(teacher_pulse.onset_micros) as f64
                        / 1_000.0,
                );
                peak_errors.push(
                    output_pulse.peak_micros.abs_diff(teacher_pulse.peak_micros) as f64 / 1_000.0,
                );
                release_errors.push(
                    output_pulse
                        .release_micros
                        .abs_diff(teacher_pulse.release_micros) as f64
                        / 1_000.0,
                );
                attenuations.push(attenuation.abs());
            }
        }
    }
    Ok(BlinkEventMetrics {
        teacher_events,
        matched_events,
        missed_events: teacher_events.saturating_sub(matched_events),
        extra_events: output_events.saturating_sub(matched_events),
        median_absolute_onset_error_ms: median(&mut onset_errors),
        median_absolute_peak_error_ms: median(&mut peak_errors),
        median_absolute_release_error_ms: median(&mut release_errors),
        median_absolute_peak_attenuation: median(&mut attenuations),
    })
}

/// Evaluates one already aligned variant over non-tongue channels only.
///
/// # Errors
///
/// Rejects empty, misordered, or cross-take-regressing input.
pub fn evaluate_non_tongue_variant(
    frames: &[VariantFrame],
) -> Result<NonTongueVariantMetrics, UnifiedAblationError> {
    if frames.is_empty() {
        return Err(UnifiedAblationError::EmptyFrames);
    }
    validate_frame_order(frames)?;
    let mut channels = Vec::with_capacity(ARKIT_NON_TONGUE_CHANNEL_COUNT);
    let mut micro_abs = 0.0;
    let mut micro_square = 0.0;
    for channel in ArkitBlendshape::ALL
        .into_iter()
        .take(ARKIT_NON_TONGUE_CHANNEL_COUNT)
    {
        let teacher: Vec<f64> = frames
            .iter()
            .map(|frame| f64::from(frame.teacher.get(channel)))
            .collect();
        let predicted: Vec<f64> = frames
            .iter()
            .map(|frame| f64::from(frame.output.get(channel)))
            .collect();
        let errors: Vec<f64> = predicted
            .iter()
            .zip(&teacher)
            .map(|(predicted, teacher)| predicted - teacher)
            .collect();
        let absolute = errors.iter().map(|error| error.abs()).sum::<f64>();
        let square = errors.iter().map(|error| error * error).sum::<f64>();
        micro_abs += absolute;
        micro_square += square;
        let neutral: Vec<f64> = errors
            .iter()
            .zip(&teacher)
            .filter_map(|(error, teacher)| (*teacher <= 0.05).then_some(*error))
            .collect();
        channels.push(NonTongueChannelMetrics {
            channel,
            mae: absolute / frames.len() as f64,
            rmse: (square / frames.len() as f64).sqrt(),
            pearson: pearson_with_teacher_variance(&teacher, &predicted),
            neutral_bias: (!neutral.is_empty())
                .then(|| neutral.iter().sum::<f64>() / neutral.len() as f64),
        });
    }
    let cell_count = (frames.len() * ARKIT_NON_TONGUE_CHANNEL_COUNT) as f64;
    let macro_mae = channels.iter().map(|channel| channel.mae).sum::<f64>()
        / ARKIT_NON_TONGUE_CHANNEL_COUNT as f64;
    let macro_rmse = channels.iter().map(|channel| channel.rmse).sum::<f64>()
        / ARKIT_NON_TONGUE_CHANNEL_COUNT as f64;
    Ok(NonTongueVariantMetrics {
        frame_count: frames.len(),
        macro_mae,
        macro_rmse,
        micro_mae: micro_abs / cell_count,
        micro_rmse: (micro_square / cell_count).sqrt(),
        channels,
        left_right: left_right_metrics(frames),
        temporal: temporal_metrics(frames),
    })
}

// `windows(2)` yields exactly two elements.
#[allow(clippy::indexing_slicing)]
fn validate_frame_order(frames: &[VariantFrame]) -> Result<(), UnifiedAblationError> {
    let mut seen = BTreeSet::new();
    for frame in frames {
        let identity = (&frame.take_id, frame.frame_seq, frame.timestamp_micros);
        if !seen.insert(identity) {
            return Err(UnifiedAblationError::InvalidData("duplicate frame"));
        }
    }
    for pair in frames.windows(2) {
        if pair[0].take_id == pair[1].take_id
            && (pair[1].frame_seq <= pair[0].frame_seq
                || pair[1].timestamp_micros <= pair[0].timestamp_micros)
        {
            return Err(UnifiedAblationError::InvalidData("frame order"));
        }
    }
    Ok(())
}

fn pearson_with_teacher_variance(teacher: &[f64], predicted: &[f64]) -> Option<f64> {
    let count = teacher.len() as f64;
    let teacher_mean = teacher.iter().sum::<f64>() / count;
    let predicted_mean = predicted.iter().sum::<f64>() / count;
    let teacher_square = teacher
        .iter()
        .map(|value| (value - teacher_mean).powi(2))
        .sum::<f64>();
    let predicted_square = predicted
        .iter()
        .map(|value| (value - predicted_mean).powi(2))
        .sum::<f64>();
    if teacher_square <= f64::EPSILON || predicted_square <= f64::EPSILON {
        return None;
    }
    let covariance = teacher
        .iter()
        .zip(predicted)
        .map(|(teacher, predicted)| (teacher - teacher_mean) * (predicted - predicted_mean))
        .sum::<f64>();
    Some(covariance / (teacher_square * predicted_square).sqrt())
}

fn left_right_metrics(frames: &[VariantFrame]) -> Vec<LeftRightDifferenceMetrics> {
    ARKIT_NON_TONGUE_LEFT_RIGHT_PAIRS
        .iter()
        .map(|&(left, right)| {
            let difference_mae = frames
                .iter()
                .map(|frame| {
                    let teacher = frame.teacher.get(left) - frame.teacher.get(right);
                    let output = frame.output.get(left) - frame.output.get(right);
                    f64::from((output - teacher).abs())
                })
                .sum::<f64>()
                / frames.len() as f64;
            LeftRightDifferenceMetrics {
                left,
                right,
                difference_mae,
            }
        })
        .collect()
}

// All indexes below are bounded by windows or derivative-vector lengths built
// from the same take. The four-frame jerk slice exists for every second-order
// derivative pair (`index < take.len() - 2`).
#[allow(clippy::indexing_slicing)]
fn temporal_metrics(frames: &[VariantFrame]) -> NonTongueTemporalMetrics {
    let mut velocity_abs = 0.0;
    let mut velocity_count = 0_usize;
    let mut acceleration_abs = 0.0;
    let mut acceleration_count = 0_usize;
    let mut jitter_square = 0.0;
    let mut jitter_count = 0_usize;
    let mut jerk_abs = 0.0;
    let mut jerk_count = 0_usize;
    for take in take_slices(frames) {
        for channel in ArkitBlendshape::ALL
            .into_iter()
            .take(ARKIT_NON_TONGUE_CHANNEL_COUNT)
        {
            let mut teacher_velocity = Vec::new();
            let mut output_velocity = Vec::new();
            let mut velocity_times = Vec::new();
            for pair in take.windows(2) {
                let dt = (pair[1].timestamp_micros - pair[0].timestamp_micros) as f64 / 1_000_000.0;
                let teacher =
                    f64::from(pair[1].teacher.get(channel) - pair[0].teacher.get(channel)) / dt;
                let output =
                    f64::from(pair[1].output.get(channel) - pair[0].output.get(channel)) / dt;
                velocity_abs += (output - teacher).abs();
                velocity_count += 1;
                if pair[0].teacher.get(channel) <= 0.05 && pair[1].teacher.get(channel) <= 0.05 {
                    jitter_square += output * output;
                    jitter_count += 1;
                }
                teacher_velocity.push(teacher);
                output_velocity.push(output);
                velocity_times.push((pair[0].timestamp_micros + pair[1].timestamp_micros) / 2);
            }
            let mut teacher_acceleration = Vec::new();
            let mut output_acceleration = Vec::new();
            let mut acceleration_times = Vec::new();
            for index in 1..teacher_velocity.len() {
                let dt = (velocity_times[index] - velocity_times[index - 1]) as f64 / 1_000_000.0;
                let teacher = (teacher_velocity[index] - teacher_velocity[index - 1]) / dt;
                let output = (output_velocity[index] - output_velocity[index - 1]) / dt;
                acceleration_abs += (output - teacher).abs();
                acceleration_count += 1;
                teacher_acceleration.push(teacher);
                output_acceleration.push(output);
                acceleration_times.push((velocity_times[index - 1] + velocity_times[index]) / 2);
            }
            for index in 1..teacher_acceleration.len() {
                let peak_window = take[index - 1..=index + 2]
                    .iter()
                    .any(|frame| frame.teacher.get(channel) >= 0.5);
                if peak_window {
                    let dt = (acceleration_times[index] - acceleration_times[index - 1]) as f64
                        / 1_000_000.0;
                    let teacher =
                        (teacher_acceleration[index] - teacher_acceleration[index - 1]) / dt;
                    let output = (output_acceleration[index] - output_acceleration[index - 1]) / dt;
                    jerk_abs += (output - teacher).abs();
                    jerk_count += 1;
                }
            }
        }
    }
    NonTongueTemporalMetrics {
        velocity_mae: velocity_abs / velocity_count.max(1) as f64,
        acceleration_mae: acceleration_abs / acceleration_count.max(1) as f64,
        neutral_jitter: (jitter_square / jitter_count.max(1) as f64).sqrt(),
        peak_jerk_mae: jerk_abs / jerk_count.max(1) as f64,
    }
}

// Caller validation guarantees non-empty frames; `start` and `index` are
// advanced only within `1..frames.len()`.
#[allow(clippy::indexing_slicing)]
fn take_slices(frames: &[VariantFrame]) -> Vec<&[VariantFrame]> {
    let mut slices = Vec::new();
    let mut start = 0;
    for index in 1..frames.len() {
        if frames[index].take_id != frames[start].take_id {
            slices.push(&frames[start..index]);
            start = index;
        }
    }
    slices.push(&frames[start..]);
    slices
}

/// Sparse avatar vertex identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct AvatarVertexKey {
    /// glTF mesh index.
    pub mesh_index: usize,
    /// Vertex index inside the mesh primitive response.
    pub vertex_index: usize,
}

/// One expression's displacement of one avatar vertex.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AvatarMorphDelta {
    /// Affected vertex.
    pub vertex: AvatarVertexKey,
    /// Position displacement at coefficient one.
    pub delta_xyz: [f32; 3],
}

/// Engine-neutral non-tongue Perfect Sync morph response.
#[derive(Clone, Debug, PartialEq)]
pub struct PerfectSyncMorphResponse {
    /// SHA-256 of the local model bytes; no path is retained.
    pub model_sha256: String,
    /// Sparse response by non-tongue semantic.
    pub channels: Vec<(ArkitBlendshape, Vec<AvatarMorphDelta>)>,
}

/// Composes coefficient errors through sparse morph deltas and returns the
/// per-coordinate RMS over the union of bound vertices.
///
/// # Errors
///
/// Rejects TongueOut, duplicate channels/vertices, non-finite deltas, or an
/// empty response.
pub fn avatar_space_rms_error(
    predicted: &Arkit52Coefficients,
    teacher: &Arkit52Coefficients,
    response: &PerfectSyncMorphResponse,
) -> Result<f64, UnifiedAblationError> {
    let composed = prepare_avatar_response(response)?;
    Ok(score_avatar_response(predicted, teacher, &composed))
}

type PreparedAvatarResponse = BTreeMap<AvatarVertexKey, Vec<(ArkitBlendshape, [f64; 3])>>;

fn prepare_avatar_response(
    response: &PerfectSyncMorphResponse,
) -> Result<PreparedAvatarResponse, UnifiedAblationError> {
    let mut channels = BTreeSet::new();
    let mut composed: PreparedAvatarResponse = BTreeMap::new();
    for (channel, deltas) in &response.channels {
        if *channel == ArkitBlendshape::TongueOut || !channels.insert(channel.index()) {
            return Err(UnifiedAblationError::InvalidData("avatar morph channel"));
        }
        let mut vertices = BTreeSet::new();
        for delta in deltas {
            if !vertices.insert(delta.vertex)
                || delta.delta_xyz.iter().any(|value| !value.is_finite())
            {
                return Err(UnifiedAblationError::InvalidData("avatar morph delta"));
            }
            composed
                .entry(delta.vertex)
                .or_default()
                .push((*channel, delta.delta_xyz.map(f64::from)));
        }
    }
    if composed.is_empty() {
        return Err(UnifiedAblationError::EmptyMorphResponse);
    }
    Ok(composed)
}

fn score_avatar_response(
    predicted: &Arkit52Coefficients,
    teacher: &Arkit52Coefficients,
    response: &PreparedAvatarResponse,
) -> f64 {
    let square = response
        .values()
        .map(|terms| {
            let mut displacement = [0.0; 3];
            for (channel, delta) in terms {
                let error = f64::from(predicted.get(*channel) - teacher.get(*channel));
                for (output, value) in displacement.iter_mut().zip(delta) {
                    *output += error * value;
                }
            }
            displacement.iter().map(|value| value * value).sum::<f64>()
        })
        .sum::<f64>();
    (square / (response.len() * 3) as f64).sqrt()
}

/// Computes per-frame avatar morph-space RMS while preparing the sparse VRM
/// response only once.
///
/// # Errors
///
/// Rejects malformed or empty morph responses.
pub fn avatar_space_rms_errors(
    frames: &[VariantFrame],
    response: &PerfectSyncMorphResponse,
) -> Result<Vec<f64>, UnifiedAblationError> {
    let prepared = prepare_avatar_response(response)?;
    Ok(frames
        .iter()
        .map(|frame| score_avatar_response(&frame.output, &frame.teacher, &prepared))
        .collect())
}

/// Mean paired absolute-error delta (`candidate - reference`) on exact frames.
///
/// # Errors
///
/// Rejects mismatched identities or frame counts.
pub fn paired_absolute_error_delta(
    reference: &[VariantFrame],
    candidate: &[VariantFrame],
) -> Result<f64, UnifiedAblationError> {
    if reference.len() != candidate.len() || reference.is_empty() {
        return Err(UnifiedAblationError::InvalidData("paired delta length"));
    }
    let mut delta = 0.0;
    for (reference, candidate) in reference.iter().zip(candidate) {
        if reference.take_id != candidate.take_id
            || reference.frame_seq != candidate.frame_seq
            || reference.timestamp_micros != candidate.timestamp_micros
            || reference.teacher != candidate.teacher
        {
            return Err(UnifiedAblationError::InvalidData("paired delta identity"));
        }
        for ((reference, candidate), teacher) in arkit_non_tongue_values(&reference.output)
            .iter()
            .zip(arkit_non_tongue_values(&candidate.output))
            .zip(arkit_non_tongue_values(&reference.teacher))
        {
            delta += f64::from((candidate - teacher).abs() - (reference - teacher).abs());
        }
    }
    Ok(delta / (reference.len() * ARKIT_NON_TONGUE_CHANNEL_COUNT) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coefficients(jaw: f32, left: f32, right: f32) -> Arkit52Coefficients {
        let mut values = [0.0; 52];
        values[ArkitBlendshape::JawOpen.index()] = jaw;
        values[ArkitBlendshape::EyeBlinkLeft.index()] = left;
        values[ArkitBlendshape::EyeBlinkRight.index()] = right;
        Arkit52Coefficients::try_from_array(values).unwrap()
    }

    fn frame(
        sequence: u64,
        teacher: Arkit52Coefficients,
        output: Arkit52Coefficients,
    ) -> VariantFrame {
        VariantFrame {
            take_id: "take".to_owned(),
            frame_seq: sequence,
            timestamp_micros: sequence * 10_000,
            teacher,
            output,
        }
    }

    #[test]
    fn common_frames_use_exact_six_way_intersection() {
        let teacher = coefficients(0.5, 0.8, 0.2);
        let mut variants = BTreeMap::new();
        for variant in GnmResearchVariant::ALL {
            variants.insert(
                variant,
                vec![frame(1, teacher, teacher), frame(2, teacher, teacher)],
            );
        }
        variants
            .get_mut(&GnmResearchVariant::GnmLearned)
            .unwrap()
            .remove(0);
        let common = exact_common_variant_frames(&variants).unwrap();
        assert!(common.values().all(|frames| frames.len() == 1));
        assert!(common.values().all(|frames| frames[0].frame_seq == 2));
    }

    #[test]
    fn coefficient_metrics_exclude_tongue_and_measure_asymmetry() {
        let teacher = coefficients(0.5, 0.8, 0.2);
        let output = coefficients(0.4, 0.6, 0.4);
        let frames = vec![frame(1, teacher, output), frame(2, teacher, output)];
        let metrics = evaluate_non_tongue_variant(&frames).unwrap();
        assert_eq!(metrics.frame_count, 2);
        assert!((metrics.micro_mae - 0.5 / 51.0).abs() < 1.0e-7);
        let blink = metrics
            .left_right
            .iter()
            .find(|pair| pair.left == ArkitBlendshape::EyeBlinkLeft)
            .unwrap();
        assert!((blink.difference_mae - 0.4).abs() < 1.0e-6);
    }

    #[test]
    fn avatar_space_composes_shared_vertices_and_rejects_tongue() {
        let teacher = coefficients(0.0, 0.0, 0.0);
        let predicted = coefficients(1.0, 0.0, 0.0);
        let response = PerfectSyncMorphResponse {
            model_sha256: "abc".to_owned(),
            channels: vec![(
                ArkitBlendshape::JawOpen,
                vec![AvatarMorphDelta {
                    vertex: AvatarVertexKey {
                        mesh_index: 1,
                        vertex_index: 2,
                    },
                    delta_xyz: [3.0, 0.0, 0.0],
                }],
            )],
        };
        assert!(
            (avatar_space_rms_error(&predicted, &teacher, &response).unwrap() - 3.0_f64.sqrt())
                .abs()
                < 1.0e-7
        );
        let tongue = PerfectSyncMorphResponse {
            model_sha256: "abc".to_owned(),
            channels: vec![(ArkitBlendshape::TongueOut, Vec::new())],
        };
        assert!(avatar_space_rms_error(&predicted, &teacher, &tongue).is_err());
    }

    #[test]
    fn blink_events_match_identical_exact_frame_pulses() {
        let values = [0.0, 0.0, 0.0, 0.0, 0.4, 0.9, 0.2, 0.0];
        let frames = values
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                let coefficients = coefficients(0.0, value, value);
                VariantFrame {
                    take_id: "take".to_owned(),
                    frame_seq: index as u64,
                    timestamp_micros: index as u64 * 100_000,
                    teacher: coefficients,
                    output: coefficients,
                }
            })
            .collect::<Vec<_>>();
        let metrics = evaluate_blink_events(&frames).unwrap();
        assert_eq!(metrics.teacher_events, 2);
        assert_eq!(metrics.matched_events, 2);
        assert_eq!(metrics.missed_events, 0);
        assert_eq!(metrics.extra_events, 0);
        assert_eq!(metrics.median_absolute_onset_error_ms, Some(0.0));
        assert_eq!(metrics.median_absolute_peak_error_ms, Some(0.0));
        assert_eq!(metrics.median_absolute_release_error_ms, Some(0.0));
        assert_eq!(metrics.median_absolute_peak_attenuation, Some(0.0));
    }
}
