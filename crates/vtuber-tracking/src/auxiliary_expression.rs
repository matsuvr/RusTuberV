//! Optional MediaPipe blendshape observations for GNM fitting.
//!
//! This module deliberately does **not** define an ARKit output mapping. MediaPipe
//! blendshape scores are adapted into a small engine-neutral semantic observation
//! that may contribute a weak current-frame fitting loss. Dense geometry remains
//! the primary observation and callers may disable the auxiliary term entirely.

use vtuber_core::{FaceBlendshapeSet, MediaPipeBlendshape};

/// Engine-neutral semantic supported by the first auxiliary-expression boundary.
///
/// The variants describe geometry-observable facial actions rather than GNM latent
/// indices or ARKit actuator channels. Eye-look is intentionally absent until GNM
/// eye-joint ownership is settled, and there is no tongue semantic because the
/// approved MediaPipe Face Landmarker set does not provide one.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AuxiliaryExpressionSemantic {
    /// Anatomical left eyelid closure.
    EyeClosureLeft,
    /// Anatomical right eyelid closure.
    EyeClosureRight,
    /// Anatomical left eye widening.
    EyeWideLeft,
    /// Anatomical right eye widening.
    EyeWideRight,
    /// Jaw opening.
    JawOpen,
    /// Jaw forward motion.
    JawForward,
    /// Jaw left motion.
    JawLeft,
    /// Jaw right motion.
    JawRight,
    /// Anatomical left mouth-corner smile motion.
    MouthSmileLeft,
    /// Anatomical right mouth-corner smile motion.
    MouthSmileRight,
    /// Anatomical left mouth-corner frown motion.
    MouthFrownLeft,
    /// Anatomical right mouth-corner frown motion.
    MouthFrownRight,
    /// Lip pucker.
    MouthPucker,
    /// Lip funnel.
    MouthFunnel,
    /// Anatomical left mouth stretch.
    MouthStretchLeft,
    /// Anatomical right mouth stretch.
    MouthStretchRight,
    /// Inner brow raise.
    BrowInnerUp,
    /// Anatomical left brow lowering.
    BrowDownLeft,
    /// Anatomical right brow lowering.
    BrowDownRight,
    /// Anatomical left outer brow raise.
    BrowOuterUpLeft,
    /// Anatomical right outer brow raise.
    BrowOuterUpRight,
}

impl AuxiliaryExpressionSemantic {
    /// All semantics supported by this first auxiliary boundary.
    pub const ALL: [Self; 21] = [
        Self::EyeClosureLeft,
        Self::EyeClosureRight,
        Self::EyeWideLeft,
        Self::EyeWideRight,
        Self::JawOpen,
        Self::JawForward,
        Self::JawLeft,
        Self::JawRight,
        Self::MouthSmileLeft,
        Self::MouthSmileRight,
        Self::MouthFrownLeft,
        Self::MouthFrownRight,
        Self::MouthPucker,
        Self::MouthFunnel,
        Self::MouthStretchLeft,
        Self::MouthStretchRight,
        Self::BrowInnerUp,
        Self::BrowDownLeft,
        Self::BrowDownRight,
        Self::BrowOuterUpLeft,
        Self::BrowOuterUpRight,
    ];

    /// Returns the canonical MediaPipe source category for this semantic.
    ///
    /// `_neutral` cannot be returned from this function, so it cannot silently
    /// become an expression/ARKit channel through this adapter.
    pub const fn mediapipe_source(self) -> MediaPipeBlendshape {
        match self {
            Self::EyeClosureLeft => MediaPipeBlendshape::EyeBlinkLeft,
            Self::EyeClosureRight => MediaPipeBlendshape::EyeBlinkRight,
            Self::EyeWideLeft => MediaPipeBlendshape::EyeWideLeft,
            Self::EyeWideRight => MediaPipeBlendshape::EyeWideRight,
            Self::JawOpen => MediaPipeBlendshape::JawOpen,
            Self::JawForward => MediaPipeBlendshape::JawForward,
            Self::JawLeft => MediaPipeBlendshape::JawLeft,
            Self::JawRight => MediaPipeBlendshape::JawRight,
            Self::MouthSmileLeft => MediaPipeBlendshape::MouthSmileLeft,
            Self::MouthSmileRight => MediaPipeBlendshape::MouthSmileRight,
            Self::MouthFrownLeft => MediaPipeBlendshape::MouthFrownLeft,
            Self::MouthFrownRight => MediaPipeBlendshape::MouthFrownRight,
            Self::MouthPucker => MediaPipeBlendshape::MouthPucker,
            Self::MouthFunnel => MediaPipeBlendshape::MouthFunnel,
            Self::MouthStretchLeft => MediaPipeBlendshape::MouthStretchLeft,
            Self::MouthStretchRight => MediaPipeBlendshape::MouthStretchRight,
            Self::BrowInnerUp => MediaPipeBlendshape::BrowInnerUp,
            Self::BrowDownLeft => MediaPipeBlendshape::BrowDownLeft,
            Self::BrowDownRight => MediaPipeBlendshape::BrowDownRight,
            Self::BrowOuterUpLeft => MediaPipeBlendshape::BrowOuterUpLeft,
            Self::BrowOuterUpRight => MediaPipeBlendshape::BrowOuterUpRight,
        }
    }

    /// Returns the coarse group used for bounded diagnostics.
    pub const fn group(self) -> AuxiliaryExpressionGroup {
        match self {
            Self::EyeClosureLeft
            | Self::EyeClosureRight
            | Self::EyeWideLeft
            | Self::EyeWideRight => AuxiliaryExpressionGroup::Eye,
            Self::JawOpen | Self::JawForward | Self::JawLeft | Self::JawRight => {
                AuxiliaryExpressionGroup::Jaw
            }
            Self::MouthSmileLeft
            | Self::MouthSmileRight
            | Self::MouthFrownLeft
            | Self::MouthFrownRight
            | Self::MouthPucker
            | Self::MouthFunnel
            | Self::MouthStretchLeft
            | Self::MouthStretchRight => AuxiliaryExpressionGroup::Mouth,
            Self::BrowInnerUp
            | Self::BrowDownLeft
            | Self::BrowDownRight
            | Self::BrowOuterUpLeft
            | Self::BrowOuterUpRight => AuxiliaryExpressionGroup::Brow,
        }
    }
}

/// Coarse diagnostic group for auxiliary residuals.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(usize)]
pub enum AuxiliaryExpressionGroup {
    /// Eyelid / eye-wide semantics.
    Eye = 0,
    /// Jaw semantics.
    Jaw = 1,
    /// Mouth/lip semantics.
    Mouth = 2,
    /// Brow semantics.
    Brow = 3,
}

impl AuxiliaryExpressionGroup {
    /// All diagnostic groups in stable order.
    pub const ALL: [Self; 4] = [Self::Eye, Self::Jaw, Self::Mouth, Self::Brow];

    const fn index(self) -> usize {
        self as usize
    }
}

/// Repository-owned reliability class for one optional auxiliary channel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuxChannelReliability {
    /// Intended for the auxiliary loss when repository fixtures support it.
    TrustedForAux,
    /// Retained as weaker evidence.
    Weak,
    /// Observed only for diagnostics and excluded from loss.
    Disabled,
}

/// Optional person-specific neutral normalization for one MediaPipe score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AuxiliaryNeutralCalibration {
    /// Neutral-window baseline in MediaPipe score space `[0, 1]`.
    pub baseline: f32,
    /// Positive scale used to express a neutral-relative value.
    pub scale: f32,
}

impl AuxiliaryNeutralCalibration {
    /// Creates a finite neutral calibration without inventing missing scale.
    pub fn new(baseline: f32, scale: f32) -> Result<Self, AuxiliaryExpressionError> {
        if !baseline.is_finite() || !(0.0..=1.0).contains(&baseline) {
            return Err(AuxiliaryExpressionError::InvalidConfig(
                "neutral baseline must be finite and within [0, 1]",
            ));
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(AuxiliaryExpressionError::InvalidConfig(
                "neutral scale must be finite and positive",
            ));
        }
        Ok(Self { baseline, scale })
    }

    fn normalize(self, value: f32) -> f32 {
        (value - self.baseline) / self.scale
    }
}

/// Configuration for one semantic channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AuxiliaryChannelConfig {
    /// Engine-neutral semantic.
    pub semantic: AuxiliaryExpressionSemantic,
    /// Repository-owned reliability class.
    pub reliability: AuxChannelReliability,
    /// Relative per-channel loss weight. Absolute `w_aux` remains solver-owned.
    pub relative_weight: f32,
    /// Optional neutral-window baseline/scale.
    pub neutral_calibration: Option<AuxiliaryNeutralCalibration>,
}

impl AuxiliaryChannelConfig {
    /// Validates one channel configuration.
    pub fn new(
        semantic: AuxiliaryExpressionSemantic,
        reliability: AuxChannelReliability,
        relative_weight: f32,
        neutral_calibration: Option<AuxiliaryNeutralCalibration>,
    ) -> Result<Self, AuxiliaryExpressionError> {
        if !relative_weight.is_finite() || relative_weight < 0.0 {
            return Err(AuxiliaryExpressionError::InvalidConfig(
                "relative channel weight must be finite and non-negative",
            ));
        }
        if reliability != AuxChannelReliability::Disabled && relative_weight <= 0.0 {
            return Err(AuxiliaryExpressionError::InvalidConfig(
                "enabled auxiliary channel must have positive relative weight",
            ));
        }
        Ok(Self {
            semantic,
            reliability,
            relative_weight,
            neutral_calibration,
        })
    }
}

/// One canonical auxiliary channel for a single source frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AuxiliaryExpressionChannel {
    /// Engine-neutral semantic.
    pub semantic: AuxiliaryExpressionSemantic,
    /// Raw validated MediaPipe score retained for diagnostics only.
    pub raw_source_score: f32,
    /// Value consumed by the geometry sensor model residual. When neutral
    /// calibration is available this is neutral-relative; otherwise it equals
    /// the raw source score.
    pub observed_value: f32,
    /// Reliability class.
    pub reliability: AuxChannelReliability,
    /// Relative per-channel loss weight.
    pub relative_weight: f32,
}

/// Availability state of the auxiliary observation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuxiliaryExpressionStatus {
    /// At least one channel is enabled for loss evaluation.
    Available,
    /// All configured channels are disabled; the observation is diagnostic-only.
    DiagnosticOnly,
}

/// Engine-neutral optional auxiliary observation for one exact camera source frame.
#[derive(Clone, Debug, PartialEq)]
pub struct AuxiliaryExpressionObservation {
    source_seq: u64,
    captured_at_micros: u64,
    channels: Vec<AuxiliaryExpressionChannel>,
    status: AuxiliaryExpressionStatus,
}

impl AuxiliaryExpressionObservation {
    /// Adapts validated MediaPipe scores using one central semantic mapping.
    ///
    /// The MediaPipe `_neutral` category and all unsupported categories are not
    /// copied into the observation. No ARKit coefficient type is involved.
    pub fn from_mediapipe(
        source_seq: u64,
        captured_at_micros: u64,
        scores: &FaceBlendshapeSet,
        config: &[AuxiliaryChannelConfig],
    ) -> Result<Self, AuxiliaryExpressionError> {
        validate_channel_config(config)?;
        let mut channels = Vec::with_capacity(config.len());
        let mut enabled = 0usize;
        for channel_config in config {
            let raw_source_score = scores.get(channel_config.semantic.mediapipe_source());
            let observed_value = channel_config
                .neutral_calibration
                .map_or(raw_source_score, |neutral| {
                    neutral.normalize(raw_source_score)
                });
            if !observed_value.is_finite() {
                return Err(AuxiliaryExpressionError::NonFiniteObservedValue {
                    semantic: channel_config.semantic,
                });
            }
            if channel_config.reliability != AuxChannelReliability::Disabled {
                enabled += 1;
            }
            channels.push(AuxiliaryExpressionChannel {
                semantic: channel_config.semantic,
                raw_source_score,
                observed_value,
                reliability: channel_config.reliability,
                relative_weight: channel_config.relative_weight,
            });
        }
        Ok(Self {
            source_seq,
            captured_at_micros,
            channels,
            status: if enabled == 0 {
                AuxiliaryExpressionStatus::DiagnosticOnly
            } else {
                AuxiliaryExpressionStatus::Available
            },
        })
    }

    /// Returns the exact source sequence.
    pub fn source_seq(&self) -> u64 {
        self.source_seq
    }

    /// Returns the exact source capture timestamp.
    pub fn captured_at_micros(&self) -> u64 {
        self.captured_at_micros
    }

    /// Returns canonical channels in configured order.
    pub fn channels(&self) -> &[AuxiliaryExpressionChannel] {
        &self.channels
    }

    /// Returns whether at least one channel can contribute to the auxiliary loss.
    pub fn status(&self) -> AuxiliaryExpressionStatus {
        self.status
    }
}

/// Requires exact source-frame identity between primary dense and auxiliary data.
pub fn validate_auxiliary_source_alignment(
    dense_source_seq: u64,
    dense_captured_at_micros: u64,
    auxiliary: &AuxiliaryExpressionObservation,
) -> Result<(), AuxiliaryExpressionError> {
    if dense_source_seq != auxiliary.source_seq {
        return Err(AuxiliaryExpressionError::SourceSequenceMismatch {
            dense: dense_source_seq,
            auxiliary: auxiliary.source_seq,
        });
    }
    if dense_captured_at_micros != auxiliary.captured_at_micros {
        return Err(AuxiliaryExpressionError::CaptureTimestampMismatch {
            dense: dense_captured_at_micros,
            auxiliary: auxiliary.captured_at_micros,
        });
    }
    Ok(())
}

/// One semantic feature predicted from reconstructed GNM geometry/state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PredictedAuxiliaryFeature {
    /// Geometry-derived semantic.
    pub semantic: AuxiliaryExpressionSemantic,
    /// Prediction in the same calibrated sensor coordinate as `observed_value`.
    pub value: f32,
}

/// Robust-loss configuration. Absolute solver-level `w_aux` is explicit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AuxiliaryLossConfig {
    /// Absolute weight of the complete auxiliary term. `0` is a supported no-op.
    pub auxiliary_weight: f32,
    /// Positive Huber transition magnitude.
    pub huber_delta: f32,
    /// Absolute residual above which disagreement diagnostics increment.
    pub disagreement_threshold: f32,
}

impl AuxiliaryLossConfig {
    /// Creates a finite robust-loss configuration.
    pub fn new(
        auxiliary_weight: f32,
        huber_delta: f32,
        disagreement_threshold: f32,
    ) -> Result<Self, AuxiliaryExpressionError> {
        if !auxiliary_weight.is_finite() || auxiliary_weight < 0.0 {
            return Err(AuxiliaryExpressionError::InvalidConfig(
                "auxiliary weight must be finite and non-negative",
            ));
        }
        if !huber_delta.is_finite() || huber_delta <= 0.0 {
            return Err(AuxiliaryExpressionError::InvalidConfig(
                "Huber delta must be finite and positive",
            ));
        }
        if !disagreement_threshold.is_finite() || disagreement_threshold < 0.0 {
            return Err(AuxiliaryExpressionError::InvalidConfig(
                "disagreement threshold must be finite and non-negative",
            ));
        }
        Ok(Self {
            auxiliary_weight,
            huber_delta,
            disagreement_threshold,
        })
    }
}

/// Mean absolute residual for one coarse semantic group.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AuxiliaryGroupResiduals {
    values: [Option<f32>; 4],
}

impl AuxiliaryGroupResiduals {
    /// Returns the mean absolute residual for one group when at least one channel
    /// from the group contributed to the loss.
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub fn get(self, group: AuxiliaryExpressionGroup) -> Option<f32> {
        self.values[group.index()]
    }
}

/// Bounded diagnostics returned with one auxiliary loss evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AuxiliaryLossDiagnostics {
    /// Final weighted robust auxiliary energy.
    pub weighted_loss: f32,
    /// Channels actually paired with a geometry prediction.
    pub used_channels: usize,
    /// Configured channels excluded by `Disabled` reliability.
    pub disabled_channels: usize,
    /// Enabled observations for which the geometry sensor model supplied no prediction.
    pub missing_prediction_channels: usize,
    /// Used channels whose absolute residual exceeded the configured diagnostic threshold.
    pub disagreement_count: usize,
    /// Largest absolute residual among used channels.
    pub max_abs_residual: f32,
    /// Mean absolute residual per coarse region.
    pub group_residuals: AuxiliaryGroupResiduals,
}

/// Evaluates a robust optional auxiliary term against geometry-derived predictions.
///
/// If `auxiliary_weight == 0`, this is an unconditional finite no-op even when no
/// predictions are supplied. This makes the GNM fitter structurally capable of
/// operating on dense geometry alone.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
pub fn evaluate_auxiliary_expression_loss(
    observation: Option<&AuxiliaryExpressionObservation>,
    predictions: &[PredictedAuxiliaryFeature],
    config: AuxiliaryLossConfig,
) -> Result<AuxiliaryLossDiagnostics, AuxiliaryExpressionError> {
    validate_predictions(predictions)?;
    let Some(observation) = observation else {
        return Ok(empty_diagnostics());
    };
    if config.auxiliary_weight == 0.0 {
        return Ok(AuxiliaryLossDiagnostics {
            disabled_channels: observation
                .channels
                .iter()
                .filter(|channel| channel.reliability == AuxChannelReliability::Disabled)
                .count(),
            ..empty_diagnostics()
        });
    }

    let mut weighted_loss = 0.0;
    let mut used_channels = 0usize;
    let mut disabled_channels = 0usize;
    let mut missing_prediction_channels = 0usize;
    let mut disagreement_count = 0usize;
    let mut max_abs_residual = 0.0_f32;
    let mut group_abs_sums = [0.0_f32; 4];
    let mut group_counts = [0usize; 4];

    for channel in &observation.channels {
        if channel.reliability == AuxChannelReliability::Disabled {
            disabled_channels += 1;
            continue;
        }
        let Some(prediction) = predictions
            .iter()
            .find(|prediction| prediction.semantic == channel.semantic)
        else {
            missing_prediction_channels += 1;
            continue;
        };
        let residual = prediction.value - channel.observed_value;
        let abs_residual = residual.abs();
        let robust = huber_loss(residual, config.huber_delta);
        weighted_loss += config.auxiliary_weight * channel.relative_weight * robust;
        used_channels += 1;
        max_abs_residual = max_abs_residual.max(abs_residual);
        if abs_residual > config.disagreement_threshold {
            disagreement_count += 1;
        }
        let group_index = channel.semantic.group().index();
        group_abs_sums[group_index] += abs_residual;
        group_counts[group_index] += 1;
    }

    if !weighted_loss.is_finite() {
        return Err(AuxiliaryExpressionError::NonFiniteLoss);
    }
    let mut group_residuals = AuxiliaryGroupResiduals::default();
    for group in AuxiliaryExpressionGroup::ALL {
        let index = group.index();
        if group_counts[index] > 0 {
            group_residuals.values[index] =
                Some(group_abs_sums[index] / group_counts[index] as f32);
        }
    }

    Ok(AuxiliaryLossDiagnostics {
        weighted_loss,
        used_channels,
        disabled_channels,
        missing_prediction_channels,
        disagreement_count,
        max_abs_residual,
        group_residuals,
    })
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
// Invariant: `AuxiliaryExpressionSemantic::ALL` enumerates every variant of
// the same enum, so the position lookup cannot fail.
#[allow(clippy::indexing_slicing)]
#[allow(clippy::expect_used)]
fn validate_channel_config(
    config: &[AuxiliaryChannelConfig],
) -> Result<(), AuxiliaryExpressionError> {
    let mut seen = [false; AuxiliaryExpressionSemantic::ALL.len()];
    for channel in config {
        let index = AuxiliaryExpressionSemantic::ALL
            .iter()
            .position(|candidate| *candidate == channel.semantic)
            .expect("ALL contains every semantic variant");
        if seen[index] {
            return Err(AuxiliaryExpressionError::DuplicateSemantic(
                channel.semantic,
            ));
        }
        seen[index] = true;
        AuxiliaryChannelConfig::new(
            channel.semantic,
            channel.reliability,
            channel.relative_weight,
            channel.neutral_calibration,
        )?;
    }
    Ok(())
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn validate_predictions(
    predictions: &[PredictedAuxiliaryFeature],
) -> Result<(), AuxiliaryExpressionError> {
    for (index, prediction) in predictions.iter().enumerate() {
        if !prediction.value.is_finite() {
            return Err(AuxiliaryExpressionError::NonFinitePrediction {
                index,
                semantic: prediction.semantic,
            });
        }
        if predictions[..index]
            .iter()
            .any(|previous| previous.semantic == prediction.semantic)
        {
            return Err(AuxiliaryExpressionError::DuplicatePrediction(
                prediction.semantic,
            ));
        }
    }
    Ok(())
}

fn huber_loss(residual: f32, delta: f32) -> f32 {
    let magnitude = residual.abs();
    if magnitude <= delta {
        0.5 * residual * residual
    } else {
        delta * (magnitude - 0.5 * delta)
    }
}

const fn empty_diagnostics() -> AuxiliaryLossDiagnostics {
    AuxiliaryLossDiagnostics {
        weighted_loss: 0.0,
        used_channels: 0,
        disabled_channels: 0,
        missing_prediction_channels: 0,
        disagreement_count: 0,
        max_abs_residual: 0.0,
        group_residuals: AuxiliaryGroupResiduals {
            values: [None, None, None, None],
        },
    }
}

/// Typed auxiliary-observation validation error.
#[derive(Clone, Debug, PartialEq)]
pub enum AuxiliaryExpressionError {
    /// A configuration value is invalid.
    InvalidConfig(&'static str),
    /// One semantic was configured more than once.
    DuplicateSemantic(AuxiliaryExpressionSemantic),
    /// Neutral-relative observed value became non-finite.
    NonFiniteObservedValue {
        /// Affected semantic.
        semantic: AuxiliaryExpressionSemantic,
    },
    /// Primary dense and auxiliary observations have different source sequence.
    SourceSequenceMismatch {
        /// Dense observation sequence.
        dense: u64,
        /// Auxiliary observation sequence.
        auxiliary: u64,
    },
    /// Primary dense and auxiliary observations have different capture timestamp.
    CaptureTimestampMismatch {
        /// Dense observation capture timestamp.
        dense: u64,
        /// Auxiliary observation capture timestamp.
        auxiliary: u64,
    },
    /// Geometry sensor model supplied a non-finite feature.
    NonFinitePrediction {
        /// Prediction array index.
        index: usize,
        /// Affected semantic.
        semantic: AuxiliaryExpressionSemantic,
    },
    /// Geometry sensor model supplied one semantic more than once.
    DuplicatePrediction(AuxiliaryExpressionSemantic),
    /// Robust accumulation became non-finite.
    NonFiniteLoss,
}

impl std::fmt::Display for AuxiliaryExpressionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(reason) => {
                write!(formatter, "invalid auxiliary expression config: {reason}")
            }
            Self::DuplicateSemantic(semantic) => {
                write!(formatter, "duplicate auxiliary semantic {semantic:?}")
            }
            Self::NonFiniteObservedValue { semantic } => {
                write!(
                    formatter,
                    "non-finite auxiliary observed value for {semantic:?}"
                )
            }
            Self::SourceSequenceMismatch { dense, auxiliary } => write!(
                formatter,
                "dense source sequence {dense} does not match auxiliary {auxiliary}"
            ),
            Self::CaptureTimestampMismatch { dense, auxiliary } => write!(
                formatter,
                "dense capture timestamp {dense} does not match auxiliary {auxiliary}"
            ),
            Self::NonFinitePrediction { index, semantic } => write!(
                formatter,
                "non-finite auxiliary prediction {index} for {semantic:?}"
            ),
            Self::DuplicatePrediction(semantic) => {
                write!(formatter, "duplicate auxiliary prediction {semantic:?}")
            }
            Self::NonFiniteLoss => write!(formatter, "auxiliary expression loss became non-finite"),
        }
    }
}

impl std::error::Error for AuxiliaryExpressionError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn media_pipe_set(overrides: &[(MediaPipeBlendshape, f32)]) -> FaceBlendshapeSet {
        let pairs: Vec<(&str, f32)> = MediaPipeBlendshape::ALL
            .into_iter()
            .map(|category| {
                let value = overrides
                    .iter()
                    .find(|(candidate, _)| *candidate == category)
                    .map_or(0.0, |(_, value)| *value);
                (category.as_str(), value)
            })
            .collect();
        FaceBlendshapeSet::from_pairs(&pairs).unwrap()
    }

    fn enabled(
        semantic: AuxiliaryExpressionSemantic,
        relative_weight: f32,
    ) -> AuxiliaryChannelConfig {
        AuxiliaryChannelConfig::new(
            semantic,
            AuxChannelReliability::TrustedForAux,
            relative_weight,
            None,
        )
        .unwrap()
    }

    #[test]
    fn neutral_category_and_tongue_cannot_enter_the_auxiliary_semantic_set() {
        assert!(
            AuxiliaryExpressionSemantic::ALL
                .iter()
                .all(|semantic| semantic.mediapipe_source() != MediaPipeBlendshape::Neutral)
        );
        assert_eq!(AuxiliaryExpressionSemantic::ALL.len(), 21);
    }

    #[test]
    fn adapter_keeps_media_pipe_scores_separate_from_arkit_output_contract() {
        let scores = media_pipe_set(&[(MediaPipeBlendshape::JawOpen, 0.7)]);
        let observation = AuxiliaryExpressionObservation::from_mediapipe(
            7,
            123_000,
            &scores,
            &[enabled(AuxiliaryExpressionSemantic::JawOpen, 1.0)],
        )
        .unwrap();
        assert_eq!(observation.source_seq(), 7);
        assert_eq!(observation.channels()[0].raw_source_score, 0.7);
        assert_eq!(observation.channels()[0].observed_value, 0.7);
    }

    #[test]
    fn exact_source_sequence_and_timestamp_are_required() {
        let scores = media_pipe_set(&[]);
        let observation = AuxiliaryExpressionObservation::from_mediapipe(
            7,
            123_000,
            &scores,
            &[enabled(AuxiliaryExpressionSemantic::JawOpen, 1.0)],
        )
        .unwrap();
        assert!(validate_auxiliary_source_alignment(7, 123_000, &observation).is_ok());
        assert!(matches!(
            validate_auxiliary_source_alignment(8, 123_000, &observation),
            Err(AuxiliaryExpressionError::SourceSequenceMismatch { .. })
        ));
        assert!(matches!(
            validate_auxiliary_source_alignment(7, 123_001, &observation),
            Err(AuxiliaryExpressionError::CaptureTimestampMismatch { .. })
        ));
    }

    #[test]
    fn neutral_relative_calibration_removes_person_specific_baseline() {
        let scores = media_pipe_set(&[(MediaPipeBlendshape::JawOpen, 0.30)]);
        let config = AuxiliaryChannelConfig::new(
            AuxiliaryExpressionSemantic::JawOpen,
            AuxChannelReliability::TrustedForAux,
            1.0,
            Some(AuxiliaryNeutralCalibration::new(0.20, 0.50).unwrap()),
        )
        .unwrap();
        let observation =
            AuxiliaryExpressionObservation::from_mediapipe(1, 10, &scores, &[config]).unwrap();
        assert!((observation.channels()[0].observed_value - 0.20).abs() < 1.0e-6);
    }

    #[test]
    fn auxiliary_weight_zero_is_a_safe_no_op_without_predictions() {
        let scores = media_pipe_set(&[(MediaPipeBlendshape::JawOpen, 0.8)]);
        let observation = AuxiliaryExpressionObservation::from_mediapipe(
            1,
            10,
            &scores,
            &[enabled(AuxiliaryExpressionSemantic::JawOpen, 1.0)],
        )
        .unwrap();
        let diagnostics = evaluate_auxiliary_expression_loss(
            Some(&observation),
            &[],
            AuxiliaryLossConfig::new(0.0, 0.2, 0.5).unwrap(),
        )
        .unwrap();
        assert_eq!(diagnostics.weighted_loss, 0.0);
        assert_eq!(diagnostics.used_channels, 0);
    }

    #[test]
    fn missing_observation_is_a_safe_no_op() {
        let diagnostics = evaluate_auxiliary_expression_loss(
            None,
            &[],
            AuxiliaryLossConfig::new(1.0, 0.2, 0.5).unwrap(),
        )
        .unwrap();
        assert_eq!(diagnostics, empty_diagnostics());
    }

    #[test]
    fn robust_residual_has_correct_sign_independent_energy_for_selected_channel() {
        let scores = media_pipe_set(&[(MediaPipeBlendshape::JawOpen, 0.75)]);
        let observation = AuxiliaryExpressionObservation::from_mediapipe(
            1,
            10,
            &scores,
            &[enabled(AuxiliaryExpressionSemantic::JawOpen, 2.0)],
        )
        .unwrap();
        let prediction = [PredictedAuxiliaryFeature {
            semantic: AuxiliaryExpressionSemantic::JawOpen,
            value: 0.25,
        }];
        let diagnostics = evaluate_auxiliary_expression_loss(
            Some(&observation),
            &prediction,
            AuxiliaryLossConfig::new(0.5, 0.2, 0.3).unwrap(),
        )
        .unwrap();
        assert!(diagnostics.weighted_loss > 0.0);
        assert_eq!(diagnostics.used_channels, 1);
        assert_eq!(diagnostics.disagreement_count, 1);
        assert!((diagnostics.max_abs_residual - 0.5).abs() < 1.0e-6);
        assert_eq!(
            diagnostics
                .group_residuals
                .get(AuxiliaryExpressionGroup::Jaw),
            Some(0.5)
        );
    }

    #[test]
    fn disabled_and_missing_prediction_channels_are_diagnostic_not_fatal() {
        let scores = media_pipe_set(&[
            (MediaPipeBlendshape::JawOpen, 0.7),
            (MediaPipeBlendshape::EyeBlinkLeft, 0.9),
        ]);
        let configs = [
            enabled(AuxiliaryExpressionSemantic::JawOpen, 1.0),
            AuxiliaryChannelConfig::new(
                AuxiliaryExpressionSemantic::EyeClosureLeft,
                AuxChannelReliability::Disabled,
                0.0,
                None,
            )
            .unwrap(),
            enabled(AuxiliaryExpressionSemantic::BrowInnerUp, 0.5),
        ];
        let observation =
            AuxiliaryExpressionObservation::from_mediapipe(1, 10, &scores, &configs).unwrap();
        let diagnostics = evaluate_auxiliary_expression_loss(
            Some(&observation),
            &[PredictedAuxiliaryFeature {
                semantic: AuxiliaryExpressionSemantic::JawOpen,
                value: 0.7,
            }],
            AuxiliaryLossConfig::new(1.0, 0.2, 0.3).unwrap(),
        )
        .unwrap();
        assert_eq!(diagnostics.disabled_channels, 1);
        assert_eq!(diagnostics.missing_prediction_channels, 1);
        assert_eq!(diagnostics.used_channels, 1);
    }

    #[test]
    fn one_large_wrong_channel_is_huber_bounded() {
        let scores = media_pipe_set(&[(MediaPipeBlendshape::JawOpen, 1.0)]);
        let observation = AuxiliaryExpressionObservation::from_mediapipe(
            1,
            10,
            &scores,
            &[enabled(AuxiliaryExpressionSemantic::JawOpen, 1.0)],
        )
        .unwrap();
        let diagnostics = evaluate_auxiliary_expression_loss(
            Some(&observation),
            &[PredictedAuxiliaryFeature {
                semantic: AuxiliaryExpressionSemantic::JawOpen,
                value: -10.0,
            }],
            AuxiliaryLossConfig::new(1.0, 0.1, 0.5).unwrap(),
        )
        .unwrap();
        assert!(diagnostics.weighted_loss < 2.0);
        assert_eq!(diagnostics.disagreement_count, 1);
    }

    #[test]
    fn duplicate_semantics_and_predictions_fail_closed() {
        let scores = media_pipe_set(&[]);
        let duplicate = enabled(AuxiliaryExpressionSemantic::JawOpen, 1.0);
        assert!(matches!(
            AuxiliaryExpressionObservation::from_mediapipe(1, 10, &scores, &[duplicate, duplicate],),
            Err(AuxiliaryExpressionError::DuplicateSemantic(
                AuxiliaryExpressionSemantic::JawOpen
            ))
        ));
        assert!(matches!(
            evaluate_auxiliary_expression_loss(
                None,
                &[
                    PredictedAuxiliaryFeature {
                        semantic: AuxiliaryExpressionSemantic::JawOpen,
                        value: 0.0,
                    },
                    PredictedAuxiliaryFeature {
                        semantic: AuxiliaryExpressionSemantic::JawOpen,
                        value: 0.1,
                    },
                ],
                AuxiliaryLossConfig::new(1.0, 0.2, 0.5).unwrap(),
            ),
            Err(AuxiliaryExpressionError::DuplicatePrediction(
                AuxiliaryExpressionSemantic::JawOpen
            ))
        ));
    }
}
