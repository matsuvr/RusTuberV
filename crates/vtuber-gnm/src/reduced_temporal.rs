//! Projection of the existing expression temporal term into reduced q-space.

use crate::{
    GNM_HEAD_V3_EXPRESSION_DIM, GNM_HEAD_V3_TONGUE_EXPRESSION_RANGE, GnmReducedExpressionBasis,
    GnmReducedExpressionState, GnmTemporalNormalization, GnmTemporalStateView,
    SingleFrameTemporalPenalty, TemporalHistoryTiming, TemporalRegularizationConfig,
    TemporalRegularizationError, expand_reduced_expression,
};

/// Full quadratic expression penalty projected into one reduced basis.
#[derive(Clone, Debug, PartialEq)]
pub struct ReducedTemporalRegularization {
    /// Temporal energy at the current q.
    pub energy: f64,
    /// `B^T g_phi` in reduced-coordinate order.
    pub gradient: Vec<f64>,
    /// `B^T H_phi B` in row-major `[rank, rank]` order.
    pub hessian_row_major: Vec<f64>,
}

/// Immutable source-history input for a reduced single-frame solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReducedTemporalHistory<'a> {
    /// Previous accepted q.
    pub previous: &'a GnmReducedExpressionState,
    /// Previous-previous accepted q, when available.
    pub previous_previous: Option<&'a GnmReducedExpressionState>,
    /// Exact source-timestamp intervals.
    pub timing: TemporalHistoryTiming,
    /// Existing full-expression normalization; tongue entries are not read.
    pub normalization: GnmTemporalNormalization<'a>,
}

fn compact_non_tongue(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, value)| {
            (!GNM_HEAD_V3_TONGUE_EXPRESSION_RANGE.contains(&index)).then_some(value)
        })
        .collect()
}

fn expression_view(expression: &[f32]) -> GnmTemporalStateView<'_> {
    GnmTemporalStateView {
        expression,
        joints: &[],
        head_pose: &[],
        translation: &[],
    }
}

fn validate_rank(
    basis: &GnmReducedExpressionBasis,
    state: &GnmReducedExpressionState,
    field: &'static str,
) -> Result<(), TemporalRegularizationError> {
    if state.values().len() != basis.rank() {
        return Err(TemporalRegularizationError::DimensionMismatch {
            group: "reduced_expression",
            field,
            expected: basis.rank(),
            actual: state.values().len(),
        });
    }
    Ok(())
}

/// Evaluates the existing full-expression temporal term and chains it as
/// `g_q = B^T g_phi`, `H_q = B^T H_phi B`.
///
/// The 32 tongue indices are removed before the existing group evaluator is
/// called, so tongue contributes neither energy, gradient, nor Hessian.
pub fn evaluate_reduced_temporal_regularization(
    basis: &GnmReducedExpressionBasis,
    current: &GnmReducedExpressionState,
    previous: &GnmReducedExpressionState,
    previous_previous: Option<&GnmReducedExpressionState>,
    timing: TemporalHistoryTiming,
    normalization: &GnmTemporalNormalization<'_>,
    config: TemporalRegularizationConfig,
) -> Result<ReducedTemporalRegularization, TemporalRegularizationError> {
    validate_rank(basis, current, "current")?;
    validate_rank(basis, previous, "previous")?;
    if let Some(previous_previous) = previous_previous {
        validate_rank(basis, previous_previous, "previous_previous")?;
    }
    if normalization.expression.len() != GNM_HEAD_V3_EXPRESSION_DIM {
        return Err(TemporalRegularizationError::DimensionMismatch {
            group: "expression",
            field: "normalization",
            expected: GNM_HEAD_V3_EXPRESSION_DIM,
            actual: normalization.expression.len(),
        });
    }

    let expand = |state: &GnmReducedExpressionState| {
        expand_reduced_expression(basis, state)
            .map(|full| compact_non_tongue(full.values()))
            .map_err(|_| TemporalRegularizationError::NonFiniteEnergy)
    };
    let current_full = expand(current)?;
    let previous_full = expand(previous)?;
    let previous_previous_full = previous_previous.map(expand).transpose()?;
    let scales = compact_non_tongue(normalization.expression);
    let penalty = SingleFrameTemporalPenalty::new(
        expression_view(&previous_full),
        previous_previous_full.as_deref().map(expression_view),
        GnmTemporalNormalization {
            expression: &scales,
            joints: &[],
            head_pose: &[],
            translation: &[],
        },
        timing,
        config,
    )?;
    let current_view = expression_view(&current_full);
    let energy = penalty.energy_at(current_view)?.total_weighted_energy;
    let full = penalty.linearize_at(current_view)?.expression;
    let rank = basis.rank();
    let mut gradient = vec![0.0; rank];
    let mut hessian = vec![0.0; rank * rank];
    for (row, basis_row) in basis.values_row_major().chunks_exact(rank).enumerate() {
        #[allow(clippy::indexing_slicing)]
        for left in 0..rank {
            gradient[left] += f64::from(basis_row[left]) * full.gradient[row];
            for right in 0..rank {
                hessian[left * rank + right] +=
                    f64::from(basis_row[left]) * full.curvature[row] * f64::from(basis_row[right]);
            }
        }
    }
    Ok(ReducedTemporalRegularization {
        energy,
        gradient,
        hessian_row_major: hessian,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM, TemporalGroupPenaltyWeights};

    fn basis() -> GnmReducedExpressionBasis {
        let mut values = vec![0.0; GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM * 2];
        values[0] = 1.0;
        values[3] = 1.0;
        GnmReducedExpressionBasis::new(2, values).unwrap()
    }

    fn config() -> TemporalRegularizationConfig {
        let expression = TemporalGroupPenaltyWeights::new(2.0, 3.0).unwrap();
        let zero = TemporalGroupPenaltyWeights::new(0.0, 0.0).unwrap();
        TemporalRegularizationConfig::new(expression, zero, zero, zero, 0.2).unwrap()
    }

    #[test]
    fn identity_columns_match_full_gradient_and_hessian_without_tongue() {
        let current = GnmReducedExpressionState::new(vec![0.4, -0.2], 2).unwrap();
        let previous = GnmReducedExpressionState::new(vec![0.1, 0.0], 2).unwrap();
        let mut scales = vec![1.0; GNM_HEAD_V3_EXPRESSION_DIM];
        for index in GNM_HEAD_V3_TONGUE_EXPRESSION_RANGE {
            scales[index] = f32::NAN;
        }
        let empty = [];
        let result = evaluate_reduced_temporal_regularization(
            &basis(),
            &current,
            &previous,
            None,
            TemporalHistoryTiming {
                dt_seconds: 0.1,
                previous_dt_seconds: None,
            },
            &GnmTemporalNormalization {
                expression: &scales,
                joints: &empty,
                head_pose: &empty,
                translation: &empty,
            },
            config(),
        )
        .unwrap();
        assert_eq!(result.gradient.len(), 2);
        assert_eq!(result.hessian_row_major.len(), 4);
        assert!(result.energy > 0.0);
        assert_eq!(result.hessian_row_major[1], 0.0);
        assert_eq!(result.hessian_row_major[2], 0.0);
    }

    #[test]
    fn stale_history_is_not_stretched() {
        let state = GnmReducedExpressionState::neutral(2);
        let scales = vec![1.0; GNM_HEAD_V3_EXPRESSION_DIM];
        let empty = [];
        assert!(matches!(
            evaluate_reduced_temporal_regularization(
                &basis(),
                &state,
                &state,
                None,
                TemporalHistoryTiming {
                    dt_seconds: 0.3,
                    previous_dt_seconds: None,
                },
                &GnmTemporalNormalization {
                    expression: &scales,
                    joints: &empty,
                    head_pose: &empty,
                    translation: &empty,
                },
                config(),
            ),
            Err(TemporalRegularizationError::HistoryResetRequired { .. })
        ));
    }
}
