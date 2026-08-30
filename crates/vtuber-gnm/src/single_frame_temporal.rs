//! Temporal-energy connection for the bounded single-frame fitter
//! (Issue #64.4 / #93, parent #64).
//!
//! This module wires the scale-aware fixed temporal energy from
//! [`crate::temporal_regularization`] into the per-frame solver objective:
//!
//! - First-order and optional second-order terms are always evaluated with the
//!   actual source-frame `dt`; nothing assumes a fixed frame rate.
//! - The quadratic temporal term contributes an analytic diagonal gradient and
//!   curvature to both bounded update steps, so temporal smoothing trades off
//!   against the dense residual instead of acting as a post-decoder filter.
//! - Expression, articulated-joint, rigid-head-pose, and camera-translation
//!   motion stay separate groups with independent weights.
//! - A source gap larger than the configured bound surfaces as
//!   [`TemporalRegularizationError::HistoryResetRequired`], so stale history is
//!   never stretched across a tracking gap.
//!
//! Group coordinate conventions (must match between the stored history and the
//! solved candidate):
//! - `expression`: expression coefficients.
//! - `joints`: flattened joint rotations followed by the global joint
//!   translation (the same order as the solver's joint parameter block).
//! - `head_pose`: rigid yaw/pitch/roll radians.
//! - `translation`: camera-space head translation.

use crate::model::{GnmExpressionState, GnmJointState};
use crate::reprojection::DenseProjection;
use crate::temporal_regularization::{
    GnmTemporalNormalization, GnmTemporalStateView, TemporalGroupPenaltyWeights,
    TemporalHistoryTiming, TemporalRegularizationConfig, TemporalRegularizationError,
    TemporalRegularizationInput, TemporalRegularizationMetrics, evaluate_temporal_regularization,
};

/// Per-coordinate analytic derivative blocks of the quadratic temporal term.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TemporalGroupLinearization {
    /// `dE/dx_i` for every coordinate of the group.
    pub gradient: Vec<f64>,
    /// Diagonal curvature `d²E/dx_i²` for every coordinate of the group.
    pub curvature: Vec<f64>,
}

/// Full analytic diagonal linearization of the temporal term at one candidate.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TemporalLinearization {
    /// Expression group derivatives.
    pub expression: TemporalGroupLinearization,
    /// Articulated-joint group derivatives.
    pub joints: TemporalGroupLinearization,
    /// Rigid-head-pose group derivatives.
    pub head_pose: TemporalGroupLinearization,
    /// Camera-translation group derivatives.
    pub translation: TemporalGroupLinearization,
}

/// Owned scratch buffers used to expose one solver candidate as a temporal
/// state view without changing [`DenseProjection`]'s copying accessors.
#[derive(Clone, Debug, Default)]
pub struct CandidateTemporalScratch {
    /// Flattened joint rotations followed by the global joint translation.
    pub joints: Vec<f32>,
    /// Copy of the candidate rigid yaw/pitch/roll.
    pub head_pose: [f32; 3],
    /// Copy of the candidate camera-space head translation.
    pub translation: [f32; 3],
}

/// Caller-supplied temporal history context for exactly one per-frame solve.
///
/// The context is immutable during the solve. History slices must use the
/// documented group coordinate conventions so candidate and history entries
/// line up one-to-one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SingleFrameTemporalPenalty<'a> {
    previous: GnmTemporalStateView<'a>,
    previous_previous: Option<GnmTemporalStateView<'a>>,
    normalization: GnmTemporalNormalization<'a>,
    timing: TemporalHistoryTiming,
    config: TemporalRegularizationConfig,
}

impl<'a> SingleFrameTemporalPenalty<'a> {
    /// Assembles a temporal context from lifecycle-owned history.
    ///
    /// # Errors
    ///
    /// Returns [`TemporalRegularizationError::HistoryResetRequired`] when the
    /// supplied timing crosses the configured gap bound, and other typed
    /// failures for invalid timing. State shapes are validated lazily on the
    /// first evaluation.
    pub fn new(
        previous: GnmTemporalStateView<'a>,
        previous_previous: Option<GnmTemporalStateView<'a>>,
        normalization: GnmTemporalNormalization<'a>,
        timing: TemporalHistoryTiming,
        config: TemporalRegularizationConfig,
    ) -> Result<Self, TemporalRegularizationError> {
        // Fail fast on invalid/reset-required timing so callers can reset
        // history before wasting a solve on stale energy.
        if !timing.dt_seconds.is_finite() || timing.dt_seconds <= 0.0 {
            return Err(TemporalRegularizationError::InvalidTiming(
                "dt_seconds must be finite and positive",
            ));
        }
        if timing.dt_seconds > config.max_dt_seconds {
            return Err(TemporalRegularizationError::HistoryResetRequired {
                dt_seconds: timing.dt_seconds,
                max_dt_seconds: config.max_dt_seconds,
            });
        }
        Ok(Self {
            previous,
            previous_previous,
            normalization,
            timing,
            config,
        })
    }

    /// Returns the configured maximum history age in seconds.
    pub fn max_dt_seconds(&self) -> f64 {
        self.config.max_dt_seconds
    }

    /// Evaluates the fixed first/second-order temporal energy of one candidate
    /// state with the actual source-frame `dt`.
    ///
    /// # Errors
    ///
    /// Propagates shape, finiteness, and reset-required failures verbatim.
    pub fn energy_at(
        &self,
        current: GnmTemporalStateView<'_>,
    ) -> Result<TemporalRegularizationMetrics, TemporalRegularizationError> {
        evaluate_temporal_regularization(
            TemporalRegularizationInput {
                current,
                previous: self.previous,
                previous_previous: self.previous_previous,
                normalization: self.normalization,
                timing: self.timing,
            },
            self.config,
        )
    }

    /// Computes the analytic gradient and diagonal curvature of the quadratic
    /// temporal term at one candidate state.
    ///
    /// Shapes and finiteness are validated by evaluating the energy first, so
    /// the returned blocks are guaranteed consistent with the history.
    ///
    /// # Errors
    ///
    /// Propagates the typed failure of the underlying energy evaluation.
    pub fn linearize_at(
        &self,
        current: GnmTemporalStateView<'_>,
    ) -> Result<TemporalLinearization, TemporalRegularizationError> {
        // Validate shapes/finiteness through the canonical evaluator first.
        self.energy_at(current)?;

        let dt = self.timing.dt_seconds;

        #[allow(clippy::indexing_slicing)] // lengths validated by energy_at above
        fn group_linearization(
            values: &[f32],
            previous: &[f32],
            previous_previous: Option<&[f32]>,
            scales: &[f32],
            weights: TemporalGroupPenaltyWeights,
            dt: f64,
            previous_dt: Option<f64>,
        ) -> Result<TemporalGroupLinearization, TemporalRegularizationError> {
            let mut gradient = vec![0.0f64; values.len()];
            let mut curvature = vec![0.0f64; values.len()];
            for index in 0..values.len() {
                let scale = f64::from(scales[index]);
                let value = f64::from(values[index]);
                let previous_value = f64::from(previous[index]);
                let velocity = ((value - previous_value) / scale) / dt;
                let mut g = 2.0 * weights.velocity_lambda * velocity / (scale * dt);
                let mut h = 2.0 * weights.velocity_lambda / (scale * scale * dt * dt);
                if let (Some(previous_previous), Some(previous_dt)) =
                    (previous_previous, previous_dt)
                {
                    let previous_previous_value = f64::from(previous_previous[index]);
                    let previous_velocity =
                        ((previous_value - previous_previous_value) / scale) / previous_dt;
                    let velocity_change = velocity - previous_velocity;
                    g += 2.0 * weights.velocity_change_lambda * velocity_change / (scale * dt);
                    h += 2.0 * weights.velocity_change_lambda / (scale * scale * dt * dt);
                }
                gradient[index] = g;
                curvature[index] = h;
            }
            Ok(TemporalGroupLinearization {
                gradient,
                curvature,
            })
        }

        Ok(TemporalLinearization {
            expression: group_linearization(
                current.expression,
                self.previous.expression,
                self.previous_previous.map(|state| state.expression),
                self.normalization.expression,
                self.config.expression,
                dt,
                self.timing.previous_dt_seconds,
            )?,
            joints: group_linearization(
                current.joints,
                self.previous.joints,
                self.previous_previous.map(|state| state.joints),
                self.normalization.joints,
                self.config.joints,
                dt,
                self.timing.previous_dt_seconds,
            )?,
            head_pose: group_linearization(
                current.head_pose,
                self.previous.head_pose,
                self.previous_previous.map(|state| state.head_pose),
                self.normalization.head_pose,
                self.config.head_pose,
                dt,
                self.timing.previous_dt_seconds,
            )?,
            translation: group_linearization(
                current.translation,
                self.previous.translation,
                self.previous_previous.map(|state| state.translation),
                self.normalization.translation,
                self.config.translation,
                dt,
                self.timing.previous_dt_seconds,
            )?,
        })
    }
}

/// Builds the temporal state view of one solver candidate, filling `scratch`.
///
/// The joint group uses flattened rotations followed by the global joint
/// translation, matching the solver's joint parameter ordering.
pub fn candidate_state_view<'a>(
    expression: &'a GnmExpressionState,
    joints: &GnmJointState,
    projection: &DenseProjection,
    scratch: &'a mut CandidateTemporalScratch,
) -> GnmTemporalStateView<'a> {
    scratch.joints.clear();
    scratch
        .joints
        .extend(joints.rotations().iter().flatten().copied());
    scratch.joints.extend_from_slice(&joints.translation());
    scratch.head_pose = projection.yaw_pitch_roll();
    scratch.translation = projection.translation();
    GnmTemporalStateView {
        expression: expression.values(),
        joints: &scratch.joints,
        head_pose: &scratch.head_pose,
        translation: &scratch.translation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal_regularization::TemporalGroupPenaltyWeights;

    fn uniform_weights(lambda: f64, lambda_change: f64) -> TemporalRegularizationConfig {
        TemporalRegularizationConfig::new(
            TemporalGroupPenaltyWeights::new(lambda, lambda_change).unwrap(),
            TemporalGroupPenaltyWeights::new(lambda, lambda_change).unwrap(),
            TemporalGroupPenaltyWeights::new(lambda, lambda_change).unwrap(),
            TemporalGroupPenaltyWeights::new(lambda, lambda_change).unwrap(),
            0.25,
        )
        .unwrap()
    }

    #[test]
    fn constant_velocity_motion_has_fps_independent_energy() {
        // The same continuous motion (3 units/second) sampled at 30, 60, and
        // 120 fps must produce identical first-order energy because the actual
        // dt divides the displacement.
        let config = uniform_weights(2.0, 0.0);
        for (fps, displacement) in [(30.0, 0.1), (60.0, 0.05), (120.0, 0.025)] {
            let dt = 1.0 / fps;
            let previous = GnmTemporalStateView {
                expression: &[0.0, 0.0],
                joints: &[0.0],
                head_pose: &[0.0; 3],
                translation: &[0.0; 3],
            };
            let current = GnmTemporalStateView {
                expression: &[displacement, 0.0],
                joints: &[0.0],
                head_pose: &[0.0; 3],
                translation: &[0.0; 3],
            };
            let penalty = SingleFrameTemporalPenalty::new(
                previous,
                None,
                GnmTemporalNormalization {
                    expression: &[1.0, 1.0],
                    joints: &[1.0],
                    head_pose: &[1.0; 3],
                    translation: &[1.0; 3],
                },
                TemporalHistoryTiming {
                    dt_seconds: dt,
                    previous_dt_seconds: None,
                },
                config,
            )
            .unwrap();
            let metrics = penalty.energy_at(current).unwrap();
            assert!(
                (metrics.total_weighted_energy - 2.0 * 9.0).abs() < 1.0e-3,
                "fps {fps}: {}",
                metrics.total_weighted_energy
            );
        }
    }

    #[test]
    fn still_candidates_have_zero_energy_and_zero_gradient() {
        let config = uniform_weights(5.0, 5.0);
        let state = GnmTemporalStateView {
            expression: &[0.4, -0.2],
            joints: &[0.1, 0.2, 0.3],
            head_pose: &[0.05, -0.02, 0.01],
            translation: &[0.0, 0.0, 0.6],
        };
        let scales = GnmTemporalNormalization {
            expression: &[0.5, 0.5],
            joints: &[0.1, 0.1, 0.1],
            head_pose: &[0.01; 3],
            translation: &[0.1; 3],
        };
        let penalty = SingleFrameTemporalPenalty::new(
            state,
            Some(state),
            scales,
            TemporalHistoryTiming {
                dt_seconds: 1.0 / 60.0,
                previous_dt_seconds: Some(1.0 / 60.0),
            },
            config,
        )
        .unwrap();

        let metrics = penalty.energy_at(state).unwrap();
        assert_eq!(metrics.total_weighted_energy, 0.0);
        assert!(metrics.used_velocity_change_history);

        let linearization = penalty.linearize_at(state).unwrap();
        for group in [
            &linearization.expression,
            &linearization.joints,
            &linearization.head_pose,
            &linearization.translation,
        ] {
            assert!(group.gradient.iter().all(|value| *value == 0.0));
            assert!(group.curvature.iter().all(|value| *value > 0.0));
        }
    }

    #[test]
    fn second_order_term_penalizes_velocity_change() {
        let config = uniform_weights(0.0, 3.0);
        let previous = GnmTemporalStateView {
            expression: &[0.1],
            joints: &[],
            head_pose: &[],
            translation: &[],
        };
        let previous_previous = GnmTemporalStateView {
            expression: &[0.0],
            joints: &[],
            head_pose: &[],
            translation: &[],
        };
        // Constant velocity of 1 unit/second from pp to current: the
        // second-order change term must stay zero.
        let accelerating = GnmTemporalStateView {
            expression: &[0.2],
            joints: &[],
            head_pose: &[],
            translation: &[],
        };
        let scales = GnmTemporalNormalization {
            expression: &[1.0],
            joints: &[],
            head_pose: &[],
            translation: &[],
        };
        let timing = TemporalHistoryTiming {
            dt_seconds: 0.1,
            previous_dt_seconds: Some(0.1),
        };
        let penalty = SingleFrameTemporalPenalty::new(
            previous,
            Some(previous_previous),
            scales,
            timing,
            config,
        )
        .unwrap();
        let metrics = penalty.energy_at(accelerating).unwrap();
        assert_eq!(metrics.total_weighted_energy, 0.0);
        assert!(metrics.used_velocity_change_history);

        // Velocity changed from 1 unit/s to 2 units/s between intervals.
        let faster = GnmTemporalStateView {
            expression: &[0.3],
            joints: &[],
            head_pose: &[],
            translation: &[],
        };
        let metrics = penalty.energy_at(faster).unwrap();
        let expected = 3.0 * (2.0 - 1.0) * (2.0 - 1.0);
        assert!((metrics.total_weighted_energy - expected).abs() < 1.0e-5);

        // The gradient points against further acceleration.
        let linearization = penalty.linearize_at(faster).unwrap();
        assert!(linearization.expression.gradient[0] > 0.0);
    }

    #[test]
    fn long_source_gap_requires_history_reset_before_any_energy() {
        let config = uniform_weights(1.0, 0.0);
        let previous = GnmTemporalStateView {
            expression: &[0.0],
            joints: &[],
            head_pose: &[],
            translation: &[],
        };
        let scales = GnmTemporalNormalization {
            expression: &[1.0],
            joints: &[],
            head_pose: &[],
            translation: &[],
        };

        // Construction fails closed when the gap crosses the bound.
        let error = SingleFrameTemporalPenalty::new(
            previous,
            None,
            scales,
            TemporalHistoryTiming {
                dt_seconds: 0.5,
                previous_dt_seconds: None,
            },
            config,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            TemporalRegularizationError::HistoryResetRequired { .. }
        ));

        // Within the bound the energy is usable.
        let penalty = SingleFrameTemporalPenalty::new(
            previous,
            None,
            scales,
            TemporalHistoryTiming {
                dt_seconds: 1.0 / 30.0,
                previous_dt_seconds: None,
            },
            config,
        )
        .unwrap();
        assert_eq!(penalty.max_dt_seconds(), 0.25);
        let current = GnmTemporalStateView {
            expression: &[0.1],
            joints: &[],
            head_pose: &[],
            translation: &[],
        };
        assert!(penalty.energy_at(current).is_ok());
    }
}

#[cfg(test)]
mod solver_integration {
    use super::*;
    use crate::DenseArray;
    use crate::dense::{
        AnatomicalSide, CorrespondenceProvenance, CorrespondenceReliability,
        DenseCorrespondenceSet, DenseCoveragePolicy, DenseMappingVersion, FaceRegion,
        GnmDenseObservation, GnmSurfacePointRef, MediaPipeGnmDenseCorrespondence,
    };
    use crate::model::{
        GNM_HEAD_V3_EXPRESSION_DIM, GNM_HEAD_V3_IDENTITY_DIM, GNM_HEAD_V3_VERSION, GnmModelData,
        GnmVariant,
    };
    use crate::reprojection::{
        DenseProjection, SingleFrameFitConfig, SynthesisOptions,
        synthesize_observation_from_projection,
    };

    fn lin_model() -> crate::GnmModel {
        let vertex_count = 64;
        let identity = GNM_HEAD_V3_IDENTITY_DIM;
        let expression = GNM_HEAD_V3_EXPRESSION_DIM;
        let mut vertices = Vec::with_capacity(vertex_count * 3);
        for index in 0..vertex_count {
            let angle = (index as f32) / (vertex_count as f32) * std::f32::consts::TAU;
            vertices.extend_from_slice(&[
                0.10 * angle.cos(),
                0.12 * angle.sin(),
                0.05 * (3.0 * angle).sin(),
            ]);
        }
        let mut expression_basis = vec![0.0f32; expression * vertex_count * 3];
        for vertex in 0..vertex_count {
            let base = vertex * 3;
            expression_basis[base] = 0.05;
            expression_basis[base + 1] = -0.03;
            expression_basis[base + 2] = 0.01;
        }
        crate::GnmModel::from_data(GnmModelData {
            version: GNM_HEAD_V3_VERSION,
            variant: GnmVariant::Head,
            template_vertices: DenseArray::new("vertices", vec![vertex_count, 3], vertices)
                .unwrap(),
            template_joints: DenseArray::new("joints", vec![1, 3], vec![0.0; 3]).unwrap(),
            vertex_identity_basis: DenseArray::new(
                "identity",
                vec![identity, vertex_count, 3],
                vec![0.0; identity * vertex_count * 3],
            )
            .unwrap(),
            joint_identity_basis: DenseArray::new(
                "joint_identity",
                vec![identity, 1, 3],
                vec![0.0; identity * 3],
            )
            .unwrap(),
            expression_basis: DenseArray::new(
                "expression",
                vec![expression, vertex_count, 3],
                expression_basis,
            )
            .unwrap(),
            joint_parent_indices: vec![-1],
            skinning_weights: DenseArray::new(
                "weights",
                vec![1, vertex_count],
                vec![1.0; vertex_count],
            )
            .unwrap(),
            pose_correctives_regressor: None,
        })
        .unwrap()
    }

    fn mapping_for(model: &crate::GnmModel) -> DenseCorrespondenceSet {
        let rows: Vec<MediaPipeGnmDenseCorrespondence> = (0..64)
            .map(|index| MediaPipeGnmDenseCorrespondence {
                mediapipe_index: 10 + index,
                target: GnmSurfacePointRef::Vertex {
                    vertex_index: index,
                },
                region: FaceRegion::Other,
                anatomical_side: AnatomicalSide::Midline,
                base_weight: 1.0,
                provenance: CorrespondenceProvenance::RepositoryValidated,
                reliability: CorrespondenceReliability::High,
            })
            .collect();
        DenseCorrespondenceSet::new(
            DenseMappingVersion {
                schema_revision: 1,
                model_version: GNM_HEAD_V3_VERSION,
            },
            rows,
            model,
        )
        .unwrap()
    }

    fn truth_projection() -> DenseProjection {
        DenseProjection::new([0.15, -0.10, 0.05], [0.02, -0.03, 0.60], 1.3, [0.5, 0.5]).unwrap()
    }

    fn perturbed_guess() -> DenseProjection {
        DenseProjection::new([0.20, -0.14, 0.09], [0.06, 0.01, 0.66], 1.45, [0.5, 0.5]).unwrap()
    }

    fn observation_at(
        model: &crate::GnmModel,
        mapping: &DenseCorrespondenceSet,
    ) -> GnmDenseObservation {
        synthesize_observation_from_projection(
            model,
            &model.neutral_identity(),
            &model.neutral_expression(),
            &crate::GnmJointState::neutral(model.joint_count()),
            mapping,
            &truth_projection(),
            SynthesisOptions::default(),
            DenseCoveragePolicy::new(2, 0.75).unwrap(),
            |_, _| false,
        )
        .unwrap()
    }

    #[test]
    fn strong_pose_prior_keeps_the_solution_near_previous_state() {
        let model = lin_model();
        let mapping = mapping_for(&model);
        let observation = observation_at(&model, &mapping);
        let guess = perturbed_guess();

        // Previous history sits exactly at the perturbed guess with a strong
        // rigid-pose prior: the solution must stay closer to the guess than
        // the truth is. Neutral expression/joint history keeps those groups'
        // contributions at their neutral-pull baseline.
        let expression_dim = GNM_HEAD_V3_EXPRESSION_DIM;
        let joint_coordinates = 3 * (model.joint_count() + 1);
        let previous_expression = vec![0.0; expression_dim];
        let previous_joints = vec![0.0; joint_coordinates];
        let scale_expression = vec![1.0; expression_dim];
        let scale_joints = vec![1.0; joint_coordinates];
        let previous_view = GnmTemporalStateView {
            expression: &previous_expression,
            joints: &previous_joints,
            head_pose: &[0.20, -0.14, 0.09],
            translation: &[0.06, 0.01, 0.66],
        };
        let scales = GnmTemporalNormalization {
            expression: &scale_expression,
            joints: &scale_joints,
            head_pose: &[0.01, 0.01, 0.01],
            translation: &[0.01, 0.01, 0.01],
        };
        let config = TemporalRegularizationConfig::new(
            TemporalGroupPenaltyWeights::new(0.0, 0.0).unwrap(),
            TemporalGroupPenaltyWeights::new(0.0, 0.0).unwrap(),
            // Very strong head-pose velocity prior.
            TemporalGroupPenaltyWeights::new(5.0e4, 0.0).unwrap(),
            TemporalGroupPenaltyWeights::new(5.0e4, 0.0).unwrap(),
            0.25,
        )
        .unwrap();
        let penalty = SingleFrameTemporalPenalty::new(
            previous_view,
            None,
            scales,
            TemporalHistoryTiming {
                dt_seconds: 1.0 / 60.0,
                previous_dt_seconds: None,
            },
            config,
        )
        .unwrap();

        let outcome = crate::fit_single_frame_with_temporal(
            &model,
            &model.neutral_identity(),
            &model.neutral_expression(),
            &crate::GnmJointState::neutral(model.joint_count()),
            &mapping,
            &observation,
            &guess,
            SingleFrameFitConfig::default(),
            None,
            Some(&penalty),
        )
        .unwrap();
        assert!(outcome.valid());

        let unconstrained = crate::fit_single_frame_cold_start(
            &model,
            &model.neutral_identity(),
            &model.neutral_expression(),
            &crate::GnmJointState::neutral(model.joint_count()),
            &mapping,
            &observation,
            &guess,
            SingleFrameFitConfig::default(),
            None,
        )
        .unwrap();

        // Distance from the guess must shrink less under the strong prior.
        let distance_from_guess = |yaw: f32| (yaw - 0.20).abs();
        assert!(
            distance_from_guess(outcome.projection().yaw_pitch_roll()[0])
                <= distance_from_guess(unconstrained.projection().yaw_pitch_roll()[0])
        );
    }

    #[test]
    fn long_gap_history_reset_propagates_through_the_fit() {
        let previous_view = GnmTemporalStateView {
            expression: &[],
            joints: &[],
            head_pose: &[0.0; 3],
            translation: &[0.0; 3],
        };
        let scales = GnmTemporalNormalization {
            expression: &[],
            joints: &[],
            head_pose: &[0.01; 3],
            translation: &[0.01; 3],
        };
        // dt far beyond the configured bound: the fit must demand a history
        // reset instead of applying stale energy.
        let error = SingleFrameTemporalPenalty::new(
            previous_view,
            None,
            scales,
            TemporalHistoryTiming {
                dt_seconds: 2.0,
                previous_dt_seconds: None,
            },
            uniform_weights_for_solver(0.25),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            TemporalRegularizationError::HistoryResetRequired { .. }
        ));
    }

    fn uniform_weights_for_solver(max_dt: f64) -> TemporalRegularizationConfig {
        TemporalRegularizationConfig::new(
            TemporalGroupPenaltyWeights::new(1.0, 0.0).unwrap(),
            TemporalGroupPenaltyWeights::new(1.0, 0.0).unwrap(),
            TemporalGroupPenaltyWeights::new(1.0, 0.0).unwrap(),
            TemporalGroupPenaltyWeights::new(1.0, 0.0).unwrap(),
            max_dt,
        )
        .unwrap()
    }
}
