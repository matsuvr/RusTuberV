//! Scale-aware, timestamp-aware temporal regularization primitives for GNM state.
//!
//! This module implements only the deterministic energy described by Issue #65.
//! It contains no adaptive policy, no ARKit/output filtering, and no fixed tuning
//! constants. Identity is structurally absent: only dynamic expression, articulated
//! joint, rigid head-pose, and translation blocks can be temporally penalized.

/// Dynamic GNM state view participating in temporal regularization.
///
/// Values may be in their native model units. [`GnmTemporalNormalization`] is
/// mandatory when evaluating the energy, so raw coefficient deltas are never
/// implicitly treated as directly comparable across components.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GnmTemporalStateView<'a> {
    /// Expression-latent coefficients.
    pub expression: &'a [f32],
    /// Eye/jaw or other articulated joint coordinates chosen by the fitter.
    pub joints: &'a [f32],
    /// Rigid head-pose coordinates chosen by the fitter.
    pub head_pose: &'a [f32],
    /// Root/camera-related translation coordinates chosen by the fitter.
    pub translation: &'a [f32],
}

/// Per-component positive scales used before temporal differences are squared.
///
/// A later solver may derive these from model-provided scales, induced vertex
/// displacement, or a validated standard scale. This module deliberately does
/// not guess which normalization is correct for GNM expression components.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GnmTemporalNormalization<'a> {
    /// Positive expression scales, one per expression component.
    pub expression: &'a [f32],
    /// Positive articulated-joint scales.
    pub joints: &'a [f32],
    /// Positive rigid-pose scales.
    pub head_pose: &'a [f32],
    /// Positive translation scales.
    pub translation: &'a [f32],
}

/// Fixed temporal weights for one parameter group.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalGroupPenaltyWeights {
    /// Weight on squared normalized velocity.
    pub velocity_lambda: f64,
    /// Weight on squared change in normalized velocity.
    pub velocity_change_lambda: f64,
}

impl TemporalGroupPenaltyWeights {
    /// Creates finite non-negative fixed weights.
    pub fn new(
        velocity_lambda: f64,
        velocity_change_lambda: f64,
    ) -> Result<Self, TemporalRegularizationError> {
        for (field, value) in [
            ("velocity_lambda", velocity_lambda),
            ("velocity_change_lambda", velocity_change_lambda),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(TemporalRegularizationError::InvalidConfig {
                    field,
                    reason: "must be finite and non-negative",
                });
            }
        }
        Ok(Self {
            velocity_lambda,
            velocity_change_lambda,
        })
    }
}

/// Group-separated fixed temporal regularization configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalRegularizationConfig {
    /// Expression-latent weights.
    pub expression: TemporalGroupPenaltyWeights,
    /// Articulated-joint weights.
    pub joints: TemporalGroupPenaltyWeights,
    /// Rigid head-pose weights.
    pub head_pose: TemporalGroupPenaltyWeights,
    /// Root/camera-related translation weights.
    pub translation: TemporalGroupPenaltyWeights,
    /// Maximum source-frame `dt` for which history remains valid.
    pub max_dt_seconds: f64,
}

impl TemporalRegularizationConfig {
    /// Creates a fixed temporal configuration without supplying any project-wide
    /// default lambda values.
    pub fn new(
        expression: TemporalGroupPenaltyWeights,
        joints: TemporalGroupPenaltyWeights,
        head_pose: TemporalGroupPenaltyWeights,
        translation: TemporalGroupPenaltyWeights,
        max_dt_seconds: f64,
    ) -> Result<Self, TemporalRegularizationError> {
        if !max_dt_seconds.is_finite() || max_dt_seconds <= 0.0 {
            return Err(TemporalRegularizationError::InvalidConfig {
                field: "max_dt_seconds",
                reason: "must be finite and positive",
            });
        }
        Ok(Self {
            expression,
            joints,
            head_pose,
            translation,
            max_dt_seconds,
        })
    }
}

/// Timing/history supplied by the source capture timeline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalHistoryTiming {
    /// Seconds between `current` and `previous` capture timestamps.
    pub dt_seconds: f64,
    /// Seconds between `previous` and `previous_previous`, when second-order
    /// velocity-continuity history is available.
    pub previous_dt_seconds: Option<f64>,
}

/// Complete non-owning input to the fixed temporal energy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalRegularizationInput<'a> {
    /// Candidate current dynamic state being optimized.
    pub current: GnmTemporalStateView<'a>,
    /// Previous valid dynamic state.
    pub previous: GnmTemporalStateView<'a>,
    /// Previous-previous valid state when continuity history is available.
    pub previous_previous: Option<GnmTemporalStateView<'a>>,
    /// Mandatory scale normalization.
    pub normalization: GnmTemporalNormalization<'a>,
    /// Source-timestamp timing.
    pub timing: TemporalHistoryTiming,
}

/// Unweighted and weighted energy for one dynamic parameter group.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TemporalGroupPenaltyMetrics {
    /// Number of coordinates in this parameter block.
    pub dimension: usize,
    /// Sum of squared normalized velocity components.
    pub normalized_velocity_squared: f64,
    /// Sum of squared differences between current and previous normalized
    /// velocity, or zero when second-order history is not available.
    pub normalized_velocity_change_squared: f64,
    /// Final group contribution after fixed lambda weights.
    pub weighted_energy: f64,
}

/// Deterministic fixed temporal energy breakdown.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TemporalRegularizationMetrics {
    /// Expression-latent contribution.
    pub expression: TemporalGroupPenaltyMetrics,
    /// Articulated-joint contribution.
    pub joints: TemporalGroupPenaltyMetrics,
    /// Rigid head-pose contribution.
    pub head_pose: TemporalGroupPenaltyMetrics,
    /// Translation contribution.
    pub translation: TemporalGroupPenaltyMetrics,
    /// Sum of all four weighted group energies.
    pub total_weighted_energy: f64,
    /// Whether a second-order velocity-continuity term was actually evaluated.
    pub used_velocity_change_history: bool,
}

/// Evaluates fixed first-order and optional second-order temporal terms.
///
/// For each component `i`, first-order velocity is:
///
/// `v_i = ((x_t - x_{t-1}) / scale_i) / dt`.
///
/// When previous-previous history is available, the second term follows the
/// Issue #65 formulation and penalizes change in normalized velocity:
///
/// `dv_i = v_i - ((x_{t-1} - x_{t-2}) / scale_i) / dt_prev`.
///
/// No output allocation is performed. A `dt` larger than the configured bound
/// returns [`TemporalRegularizationError::HistoryResetRequired`] rather than
/// stretching stale history across a tracking gap.
pub fn evaluate_temporal_regularization(
    input: TemporalRegularizationInput<'_>,
    config: TemporalRegularizationConfig,
) -> Result<TemporalRegularizationMetrics, TemporalRegularizationError> {
    validate_timing(
        input.timing,
        input.previous_previous.is_some(),
        config.max_dt_seconds,
    )?;

    let expression = evaluate_group(
        "expression",
        input.current.expression,
        input.previous.expression,
        input.previous_previous.map(|state| state.expression),
        input.normalization.expression,
        input.timing,
        config.expression,
    )?;
    let joints = evaluate_group(
        "joints",
        input.current.joints,
        input.previous.joints,
        input.previous_previous.map(|state| state.joints),
        input.normalization.joints,
        input.timing,
        config.joints,
    )?;
    let head_pose = evaluate_group(
        "head_pose",
        input.current.head_pose,
        input.previous.head_pose,
        input.previous_previous.map(|state| state.head_pose),
        input.normalization.head_pose,
        input.timing,
        config.head_pose,
    )?;
    let translation = evaluate_group(
        "translation",
        input.current.translation,
        input.previous.translation,
        input.previous_previous.map(|state| state.translation),
        input.normalization.translation,
        input.timing,
        config.translation,
    )?;

    let total_weighted_energy = expression.weighted_energy
        + joints.weighted_energy
        + head_pose.weighted_energy
        + translation.weighted_energy;
    if !total_weighted_energy.is_finite() {
        return Err(TemporalRegularizationError::NonFiniteEnergy);
    }

    Ok(TemporalRegularizationMetrics {
        expression,
        joints,
        head_pose,
        translation,
        total_weighted_energy,
        used_velocity_change_history: input.previous_previous.is_some(),
    })
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn evaluate_group(
    group: &'static str,
    current: &[f32],
    previous: &[f32],
    previous_previous: Option<&[f32]>,
    scales: &[f32],
    timing: TemporalHistoryTiming,
    weights: TemporalGroupPenaltyWeights,
) -> Result<TemporalGroupPenaltyMetrics, TemporalRegularizationError> {
    validate_group_shapes(group, current, previous, previous_previous, scales)?;
    let previous_dt = timing.previous_dt_seconds;
    let mut normalized_velocity_squared = 0.0_f64;
    let mut normalized_velocity_change_squared = 0.0_f64;

    for index in 0..current.len() {
        let scale = f64::from(scales[index]);
        let current_value = finite_value(group, "current", index, current[index])?;
        let previous_value = finite_value(group, "previous", index, previous[index])?;
        let normalized_velocity = ((current_value - previous_value) / scale) / timing.dt_seconds;
        normalized_velocity_squared += normalized_velocity * normalized_velocity;

        if let (Some(previous_previous), Some(previous_dt)) = (previous_previous, previous_dt) {
            let previous_previous_value =
                finite_value(group, "previous_previous", index, previous_previous[index])?;
            let previous_velocity =
                ((previous_value - previous_previous_value) / scale) / previous_dt;
            let velocity_change = normalized_velocity - previous_velocity;
            normalized_velocity_change_squared += velocity_change * velocity_change;
        }
    }

    let weighted_energy = weights.velocity_lambda * normalized_velocity_squared
        + weights.velocity_change_lambda * normalized_velocity_change_squared;
    if !normalized_velocity_squared.is_finite()
        || !normalized_velocity_change_squared.is_finite()
        || !weighted_energy.is_finite()
    {
        return Err(TemporalRegularizationError::NonFiniteEnergy);
    }

    Ok(TemporalGroupPenaltyMetrics {
        dimension: current.len(),
        normalized_velocity_squared,
        normalized_velocity_change_squared,
        weighted_energy,
    })
}

fn validate_group_shapes(
    group: &'static str,
    current: &[f32],
    previous: &[f32],
    previous_previous: Option<&[f32]>,
    scales: &[f32],
) -> Result<(), TemporalRegularizationError> {
    if current.len() != previous.len() {
        return Err(TemporalRegularizationError::DimensionMismatch {
            group,
            field: "previous",
            expected: current.len(),
            actual: previous.len(),
        });
    }
    if scales.len() != current.len() {
        return Err(TemporalRegularizationError::DimensionMismatch {
            group,
            field: "normalization",
            expected: current.len(),
            actual: scales.len(),
        });
    }
    if let Some(previous_previous) = previous_previous
        && previous_previous.len() != current.len()
    {
        return Err(TemporalRegularizationError::DimensionMismatch {
            group,
            field: "previous_previous",
            expected: current.len(),
            actual: previous_previous.len(),
        });
    }
    for (index, scale) in scales.iter().copied().enumerate() {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(TemporalRegularizationError::InvalidScale {
                group,
                index,
                value: scale,
            });
        }
    }
    Ok(())
}

fn finite_value(
    group: &'static str,
    field: &'static str,
    index: usize,
    value: f32,
) -> Result<f64, TemporalRegularizationError> {
    if !value.is_finite() {
        return Err(TemporalRegularizationError::NonFiniteState {
            group,
            field,
            index,
        });
    }
    Ok(f64::from(value))
}

fn validate_timing(
    timing: TemporalHistoryTiming,
    has_previous_previous: bool,
    max_dt_seconds: f64,
) -> Result<(), TemporalRegularizationError> {
    if !timing.dt_seconds.is_finite() || timing.dt_seconds <= 0.0 {
        return Err(TemporalRegularizationError::InvalidTiming(
            "dt_seconds must be finite and positive",
        ));
    }
    if timing.dt_seconds > max_dt_seconds {
        return Err(TemporalRegularizationError::HistoryResetRequired {
            dt_seconds: timing.dt_seconds,
            max_dt_seconds,
        });
    }

    match (has_previous_previous, timing.previous_dt_seconds) {
        (true, Some(previous_dt)) => {
            if !previous_dt.is_finite() || previous_dt <= 0.0 {
                return Err(TemporalRegularizationError::InvalidTiming(
                    "previous_dt_seconds must be finite and positive when second-order history exists",
                ));
            }
            if previous_dt > max_dt_seconds {
                return Err(TemporalRegularizationError::HistoryResetRequired {
                    dt_seconds: previous_dt,
                    max_dt_seconds,
                });
            }
        }
        (true, None) => {
            return Err(TemporalRegularizationError::InvalidTiming(
                "previous_dt_seconds is required when previous_previous state exists",
            ));
        }
        (false, Some(_)) => {
            return Err(TemporalRegularizationError::InvalidTiming(
                "previous_dt_seconds must be absent when previous_previous state is absent",
            ));
        }
        (false, None) => {}
    }
    Ok(())
}

/// Typed validation failure for fixed GNM temporal regularization.
#[derive(Clone, Debug, PartialEq)]
pub enum TemporalRegularizationError {
    /// A fixed temporal configuration value is invalid.
    InvalidConfig {
        /// Invalid configuration field.
        field: &'static str,
        /// Validation reason.
        reason: &'static str,
    },
    /// State/history timing is invalid.
    InvalidTiming(&'static str),
    /// Source gap is too large to reuse temporal history safely.
    HistoryResetRequired {
        /// Observed `dt` that crossed the bound.
        dt_seconds: f64,
        /// Configured maximum history `dt`.
        max_dt_seconds: f64,
    },
    /// Current/history/normalization dimensions differ.
    DimensionMismatch {
        /// Dynamic parameter group.
        group: &'static str,
        /// Mismatched field.
        field: &'static str,
        /// Required dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },
    /// A normalization scale is non-finite or non-positive.
    InvalidScale {
        /// Dynamic parameter group.
        group: &'static str,
        /// Component index.
        index: usize,
        /// Invalid scale.
        value: f32,
    },
    /// Dynamic GNM state contains NaN or infinity.
    NonFiniteState {
        /// Dynamic parameter group.
        group: &'static str,
        /// Current/history field.
        field: &'static str,
        /// Component index.
        index: usize,
    },
    /// Accumulated temporal energy overflowed/became non-finite.
    NonFiniteEnergy,
}

impl std::fmt::Display for TemporalRegularizationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig { field, reason } => {
                write!(formatter, "invalid temporal config {field}: {reason}")
            }
            Self::InvalidTiming(reason) => write!(formatter, "invalid temporal timing: {reason}"),
            Self::HistoryResetRequired {
                dt_seconds,
                max_dt_seconds,
            } => write!(
                formatter,
                "temporal history reset required: dt {dt_seconds}s exceeds {max_dt_seconds}s"
            ),
            Self::DimensionMismatch {
                group,
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "temporal {group} {field} dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidScale {
                group,
                index,
                value,
            } => write!(
                formatter,
                "invalid temporal normalization scale for {group}[{index}]: {value}"
            ),
            Self::NonFiniteState {
                group,
                field,
                index,
            } => write!(
                formatter,
                "non-finite temporal state in {group} {field}[{index}]"
            ),
            Self::NonFiniteEnergy => write!(
                formatter,
                "temporal regularization energy became non-finite"
            ),
        }
    }
}

impl std::error::Error for TemporalRegularizationError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights(velocity: f64, velocity_change: f64) -> TemporalGroupPenaltyWeights {
        TemporalGroupPenaltyWeights::new(velocity, velocity_change).unwrap()
    }

    fn config() -> TemporalRegularizationConfig {
        TemporalRegularizationConfig::new(
            weights(1.0, 1.0),
            weights(2.0, 0.0),
            weights(3.0, 0.0),
            weights(4.0, 0.0),
            0.25,
        )
        .unwrap()
    }

    fn state<'a>(
        expression: &'a [f32],
        joints: &'a [f32],
        head_pose: &'a [f32],
        translation: &'a [f32],
    ) -> GnmTemporalStateView<'a> {
        GnmTemporalStateView {
            expression,
            joints,
            head_pose,
            translation,
        }
    }

    fn normalization<'a>(
        expression: &'a [f32],
        joints: &'a [f32],
        head_pose: &'a [f32],
        translation: &'a [f32],
    ) -> GnmTemporalNormalization<'a> {
        GnmTemporalNormalization {
            expression,
            joints,
            head_pose,
            translation,
        }
    }

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-6,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn same_physical_velocity_has_near_identical_energy_at_30_60_120_fps() {
        fn energy(fps: f64) -> f64 {
            let dt = 1.0 / fps;
            let previous = [0.0_f32];
            let current = [dt as f32];
            let empty: [f32; 0] = [];
            let scale = [1.0_f32];
            evaluate_temporal_regularization(
                TemporalRegularizationInput {
                    current: state(&current, &empty, &empty, &empty),
                    previous: state(&previous, &empty, &empty, &empty),
                    previous_previous: None,
                    normalization: normalization(&scale, &empty, &empty, &empty),
                    timing: TemporalHistoryTiming {
                        dt_seconds: dt,
                        previous_dt_seconds: None,
                    },
                },
                config(),
            )
            .unwrap()
            .expression
            .normalized_velocity_squared
        }

        close(energy(30.0), 1.0);
        close(energy(60.0), 1.0);
        close(energy(120.0), 1.0);
    }

    #[test]
    fn constant_velocity_has_zero_velocity_change_with_irregular_dt() {
        let previous_previous = [0.0_f32];
        let previous = [0.1_f32];
        let current = [0.35_f32];
        let scale = [1.0_f32];
        let empty: [f32; 0] = [];
        let metrics = evaluate_temporal_regularization(
            TemporalRegularizationInput {
                current: state(&current, &empty, &empty, &empty),
                previous: state(&previous, &empty, &empty, &empty),
                previous_previous: Some(state(&previous_previous, &empty, &empty, &empty)),
                normalization: normalization(&scale, &empty, &empty, &empty),
                timing: TemporalHistoryTiming {
                    dt_seconds: 0.25,
                    previous_dt_seconds: Some(0.10),
                },
            },
            config(),
        )
        .unwrap();
        close(metrics.expression.normalized_velocity_squared, 1.0);
        close(metrics.expression.normalized_velocity_change_squared, 0.0);
        assert!(metrics.used_velocity_change_history);
    }

    #[test]
    fn component_scales_prevent_raw_magnitude_from_becoming_implicit_authority() {
        let previous = [0.0_f32, 0.0];
        let current = [0.025_f32, 2.5];
        let scales = [0.1_f32, 10.0];
        let empty: [f32; 0] = [];
        let metrics = evaluate_temporal_regularization(
            TemporalRegularizationInput {
                current: state(&current, &empty, &empty, &empty),
                previous: state(&previous, &empty, &empty, &empty),
                previous_previous: None,
                normalization: normalization(&scales, &empty, &empty, &empty),
                timing: TemporalHistoryTiming {
                    dt_seconds: 0.25,
                    previous_dt_seconds: None,
                },
            },
            config(),
        )
        .unwrap();
        close(metrics.expression.normalized_velocity_squared, 2.0);
    }

    #[test]
    fn expression_pose_joint_and_translation_use_separate_weights() {
        let one = [0.25_f32];
        let zero = [0.0_f32];
        let scale = [1.0_f32];
        let metrics = evaluate_temporal_regularization(
            TemporalRegularizationInput {
                current: state(&one, &one, &one, &one),
                previous: state(&zero, &zero, &zero, &zero),
                previous_previous: None,
                normalization: normalization(&scale, &scale, &scale, &scale),
                timing: TemporalHistoryTiming {
                    dt_seconds: 0.25,
                    previous_dt_seconds: None,
                },
            },
            config(),
        )
        .unwrap();
        close(metrics.expression.weighted_energy, 1.0);
        close(metrics.joints.weighted_energy, 2.0);
        close(metrics.head_pose.weighted_energy, 3.0);
        close(metrics.translation.weighted_energy, 4.0);
        close(metrics.total_weighted_energy, 10.0);
    }

    #[test]
    fn absent_second_order_history_does_not_invent_acceleration() {
        let previous = [0.0_f32];
        let current = [1.0_f32];
        let scale = [1.0_f32];
        let empty: [f32; 0] = [];
        let metrics = evaluate_temporal_regularization(
            TemporalRegularizationInput {
                current: state(&current, &empty, &empty, &empty),
                previous: state(&previous, &empty, &empty, &empty),
                previous_previous: None,
                normalization: normalization(&scale, &empty, &empty, &empty),
                timing: TemporalHistoryTiming {
                    dt_seconds: 0.1,
                    previous_dt_seconds: None,
                },
            },
            config(),
        )
        .unwrap();
        assert!(!metrics.used_velocity_change_history);
        close(metrics.expression.normalized_velocity_change_squared, 0.0);
    }

    #[test]
    fn huge_gap_requires_history_reset_instead_of_stale_penalty() {
        let value = [0.0_f32];
        let scale = [1.0_f32];
        let empty: [f32; 0] = [];
        let result = evaluate_temporal_regularization(
            TemporalRegularizationInput {
                current: state(&value, &empty, &empty, &empty),
                previous: state(&value, &empty, &empty, &empty),
                previous_previous: None,
                normalization: normalization(&scale, &empty, &empty, &empty),
                timing: TemporalHistoryTiming {
                    dt_seconds: 1.0,
                    previous_dt_seconds: None,
                },
            },
            config(),
        );
        assert!(matches!(
            result,
            Err(TemporalRegularizationError::HistoryResetRequired { .. })
        ));
    }

    #[test]
    fn invalid_scale_dimension_and_non_finite_state_fail_closed() {
        let current = [f32::NAN];
        let previous = [0.0_f32];
        let scale = [1.0_f32];
        let empty: [f32; 0] = [];
        let result = evaluate_temporal_regularization(
            TemporalRegularizationInput {
                current: state(&current, &empty, &empty, &empty),
                previous: state(&previous, &empty, &empty, &empty),
                previous_previous: None,
                normalization: normalization(&scale, &empty, &empty, &empty),
                timing: TemporalHistoryTiming {
                    dt_seconds: 0.1,
                    previous_dt_seconds: None,
                },
            },
            config(),
        );
        assert!(matches!(
            result,
            Err(TemporalRegularizationError::NonFiniteState { .. })
        ));

        let bad_scale = [0.0_f32];
        let value = [0.0_f32];
        let result = evaluate_temporal_regularization(
            TemporalRegularizationInput {
                current: state(&value, &empty, &empty, &empty),
                previous: state(&value, &empty, &empty, &empty),
                previous_previous: None,
                normalization: normalization(&bad_scale, &empty, &empty, &empty),
                timing: TemporalHistoryTiming {
                    dt_seconds: 0.1,
                    previous_dt_seconds: None,
                },
            },
            config(),
        );
        assert!(matches!(
            result,
            Err(TemporalRegularizationError::InvalidScale { .. })
        ));

        let two = [0.0_f32, 0.0];
        let result = evaluate_temporal_regularization(
            TemporalRegularizationInput {
                current: state(&two, &empty, &empty, &empty),
                previous: state(&value, &empty, &empty, &empty),
                previous_previous: None,
                normalization: normalization(&[1.0, 1.0], &empty, &empty, &empty),
                timing: TemporalHistoryTiming {
                    dt_seconds: 0.1,
                    previous_dt_seconds: None,
                },
            },
            config(),
        );
        assert!(matches!(
            result,
            Err(TemporalRegularizationError::DimensionMismatch { .. })
        ));
    }
}
