//! Warm-start connection between the persistent GNM lifecycle directives and
//! the bounded single-frame solver (Issue #64.3 / #92, parent #64).
//!
//! This module owns the glue only: it maps a lifecycle initialization
//! directive to concrete solver initial values, runs exactly one bounded
//! solve, and routes the outcome back through the lifecycle so publication of
//! a new valid dynamic state stays lifecycle-gated.
//!
//! Warm-start semantics are intentionally narrow:
//! - Only [`GnmFitInitialization::PreviousValid`] reuses the stored validated
//!   dynamic state, and only as optimizer initial values.
//! - No temporal penalty, prediction/hold, or smoothing is added here; the
//!   solver objective is untouched.
//! - An invalid solve never replaces the stored previous valid state, and
//!   long-gap reacquisition never reuses old expression/joint values.

use crate::gnm_fitter_contract::{
    GnmCameraBlock, GnmDynamicState, GnmFitterContractError, GnmRigidPoseBlock, GnmSolverFrameInput,
};
use vtuber_gnm::{
    AuxiliaryObjectiveTerm, DenseCorrespondenceSet, DenseProjection, FixedGnmIdentity,
    GnmDenseObservation, GnmExpressionState, GnmFitInitialization, GnmFitOutcome, GnmFrameStamp,
    GnmJointState, GnmModel, GnmReprojectionError, GnmSparseVertices, PersistentGnmAction,
    PersistentGnmEvent, PersistentGnmLifecycleConfig, PersistentGnmLifecycleError,
    PersistentGnmLifecycleState, RigidRecoveryConfig, SingleFrameFitConfig, SingleFrameFitStatus,
    advance_persistent_gnm_lifecycle, fit_single_frame_cold_start, fitting_projection,
    recover_rigid_projection,
};

/// Fail-closed error for one persistent fitter step.
#[derive(Debug)]
pub enum PersistentGnmFitterError {
    /// The lifecycle rejected the event (duplicate/regressed source, pending
    /// fit ownership violation).
    Lifecycle(PersistentGnmLifecycleError),
    /// A dynamic-state or contract validation failed.
    Contract(GnmFitterContractError),
    /// The numerical solver rejected its inputs or failed deterministically.
    Solve(GnmReprojectionError),
    /// The lifecycle directed a warm start from a previous valid state but no
    /// matching validated dynamic state is stored. This is an internal
    /// ownership invariant violation and is treated as fail-closed; the
    /// pending fit is resolved before the error is returned.
    MissingWarmStartState {
        /// Source frame whose validated state was expected.
        source: GnmFrameStamp,
    },
    /// The lifecycle emitted an action that cannot originate from the event
    /// this fitter sent. Fail-closed by construction.
    UnexpectedLifecycleAction {
        /// Impossible action reported by the lifecycle.
        action: PersistentGnmAction,
        /// Source frame being processed when it happened.
        stamp: GnmFrameStamp,
    },
}

impl std::fmt::Display for PersistentGnmFitterError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Lifecycle(error) => write!(formatter, "GNM lifecycle: {error}"),
            Self::Contract(error) => write!(formatter, "GNM fitter contract: {error}"),
            Self::Solve(error) => write!(formatter, "GNM single-frame solve: {error}"),
            Self::MissingWarmStartState { source } => write!(
                formatter,
                "warm start directed from source {} but no matching validated dynamic state is stored",
                source.source_seq
            ),
            Self::UnexpectedLifecycleAction { action, stamp } => write!(
                formatter,
                "lifecycle emitted impossible action {action:?} for source {}",
                stamp.source_seq
            ),
        }
    }
}

impl std::error::Error for PersistentGnmFitterError {}

/// Stored last-valid dynamic state together with its exact source frame.
#[derive(Clone, Debug, PartialEq)]
pub struct GnmValidatedDynamicFrame {
    stamp: GnmFrameStamp,
    dynamic: GnmDynamicState,
}

impl GnmValidatedDynamicFrame {
    /// Returns the source frame this state was fitted on.
    pub fn stamp(&self) -> GnmFrameStamp {
        self.stamp
    }

    /// Returns the validated dynamic state (read-only).
    pub fn dynamic(&self) -> &GnmDynamicState {
        &self.dynamic
    }
}

/// Report of one completed bounded solve driven by a lifecycle directive.
#[derive(Clone, Debug, PartialEq)]
pub struct GnmSolvedFrameReport {
    /// Initialization directive the solve started from.
    pub initialization: GnmFitInitialization,
    /// Solver completion classification.
    pub status: SingleFrameFitStatus,
    /// Iterations actually spent by the block-coordinate loop.
    pub iterations: usize,
    /// Combined objective at the final state when finite.
    pub objective: f32,
    /// Whether the solved state was published as the new validated state.
    pub published: bool,
}

/// Outcome of one [`PersistentGnmFitter::fit_frame`] call.
#[derive(Clone, Debug, PartialEq)]
pub enum PersistentGnmFrameOutcome {
    /// Frame skipped because no fixed identity/calibration exists yet.
    SkippedUncalibrated,
    /// No usable observation was available; no solve was attempted.
    NoObservation {
        /// Whether stale dynamic state crossed the reuse-age limit so both
        /// the lifecycle authority and the stored warm-start frame cleared.
        dynamic_state_cleared: bool,
    },
    /// One bounded solve ran to completion under the lifecycle directive.
    Solved(GnmSolvedFrameReport),
}

/// Initial values handed to the bounded solver for one frame.
struct InitialDynamicValues {
    expression: GnmExpressionState,
    joints: GnmJointState,
    projection: DenseProjection,
}

/// Builds neutral expression/joints plus an observation-independent fitting
/// projection derived from the neutral model surface.
///
/// Used by [`GnmFitInitialization::NeutralFirstFit`].
fn neutral_initial_values(
    model: &GnmModel,
    identity: &FixedGnmIdentity,
    mapping: &DenseCorrespondenceSet,
) -> Result<InitialDynamicValues, GnmReprojectionError> {
    let expression = model.neutral_expression();
    let joints = GnmJointState::neutral(model.joint_count());
    let mut surface = GnmSparseVertices::with_len(mapping.len());
    mapping.evaluate_surface(model, identity.state(), &expression, &joints, &mut surface)?;
    let projection = fitting_projection(surface.values(), [0.0; 3])?;
    Ok(InitialDynamicValues {
        expression,
        joints,
        projection,
    })
}

/// Builds neutral expression/joints and recovers the rigid pose/camera
/// projection from the current observation against that neutral state.
///
/// Used by [`GnmFitInitialization::ReinitializeDynamicState`]: expression and
/// joint history is discarded while pose evidence comes from the fresh
/// observation only.
fn reinitialized_initial_values(
    model: &GnmModel,
    identity: &FixedGnmIdentity,
    mapping: &DenseCorrespondenceSet,
    observation: &GnmDenseObservation,
) -> Result<InitialDynamicValues, GnmReprojectionError> {
    let neutral = neutral_initial_values(model, identity, mapping)?;
    let recovered = recover_rigid_projection(
        model,
        identity.state(),
        &neutral.expression,
        &neutral.joints,
        mapping,
        observation,
        neutral.projection,
        RigidRecoveryConfig::default(),
    )?;
    Ok(InitialDynamicValues {
        expression: neutral.expression,
        joints: neutral.joints,
        projection: recovered.projection,
    })
}

/// Converts a validated dynamic state back into the solver's projection form.
///
/// The stored blocks were validated on construction (finite pose, positive
/// focal), so failure here can only be a numerical invariant break; it stays
/// typed instead of panicking.
fn projection_of_dynamic(
    dynamic: &GnmDynamicState,
) -> Result<DenseProjection, GnmReprojectionError> {
    DenseProjection::new(
        dynamic.rigid_pose.yaw_pitch_roll(),
        dynamic.camera.translation(),
        dynamic.camera.focal(),
        dynamic.camera.principal_point(),
    )
}

/// Converts solver output into the validated dynamic-state contract form.
fn dynamic_from_solver_output(
    projection: &DenseProjection,
    expression: GnmExpressionState,
    joints: GnmJointState,
) -> Result<GnmDynamicState, GnmFitterContractError> {
    Ok(GnmDynamicState {
        expression,
        joints,
        rigid_pose: GnmRigidPoseBlock::new(projection.yaw_pitch_roll())?,
        camera: GnmCameraBlock::new(
            projection.translation(),
            projection.focal(),
            projection.principal_point(),
        )?,
    })
}

/// Persistent GNM fitter: pure lifecycle ownership plus warm-start selection
/// for one bounded per-frame solve.
///
/// The fitter never stores fixed identity; identity is read-only input. The
/// only mutable cross-frame state is the lifecycle state and the last
/// lifecycle-validated dynamic frame, used exclusively as warm-start initial
/// values.
pub struct PersistentGnmFitter {
    config: PersistentGnmLifecycleConfig,
    lifecycle: PersistentGnmLifecycleState,
    validated: Option<GnmValidatedDynamicFrame>,
}

impl PersistentGnmFitter {
    /// Creates a fitter with the given lifecycle configuration.
    pub fn new(config: PersistentGnmLifecycleConfig) -> Self {
        Self {
            config,
            lifecycle: PersistentGnmLifecycleState::default(),
            validated: None,
        }
    }

    /// Returns the current pure lifecycle state.
    pub fn lifecycle_state(&self) -> &PersistentGnmLifecycleState {
        &self.lifecycle
    }

    /// Returns the stored last-valid dynamic frame, if any.
    pub fn validated(&self) -> Option<&GnmValidatedDynamicFrame> {
        self.validated.as_ref()
    }

    /// Applies the explicit calibration-ready event and clears all dynamic
    /// tracker state.
    ///
    /// # Errors
    ///
    /// Propagates lifecycle failures.
    pub fn calibration_ready(&mut self) -> Result<(), PersistentGnmFitterError> {
        let decision = advance_persistent_gnm_lifecycle(
            self.lifecycle,
            PersistentGnmEvent::CalibrationReady,
            self.config,
        )
        .map_err(PersistentGnmFitterError::Lifecycle)?;
        self.lifecycle = decision.state;
        self.validated = None;
        Ok(())
    }

    /// Applies the explicit calibration-invalidated event and clears all
    /// dynamic tracker state.
    ///
    /// # Errors
    ///
    /// Propagates lifecycle failures.
    pub fn calibration_invalidated(&mut self) -> Result<(), PersistentGnmFitterError> {
        let decision = advance_persistent_gnm_lifecycle(
            self.lifecycle,
            PersistentGnmEvent::CalibrationInvalidated,
            self.config,
        )
        .map_err(PersistentGnmFitterError::Lifecycle)?;
        self.lifecycle = decision.state;
        self.validated = None;
        Ok(())
    }

    /// Admits one source frame, optionally running exactly one bounded solve
    /// selected by the lifecycle initialization directive.
    ///
    /// Duplicate/regressed sources and pending-fit ownership violations are
    /// rejected by the existing lifecycle before any solve runs. If anything
    /// fails after a solve was started, the pending fit is deterministically
    /// resolved as invalid first so lifecycle ownership stays consistent,
    /// then the typed failure is returned.
    ///
    /// # Errors
    ///
    /// Returns the first typed failure from the lifecycle, contract
    /// validation, or the solver.
    #[allow(clippy::too_many_arguments)]
    pub fn fit_frame(
        &mut self,
        input: &GnmSolverFrameInput<'_>,
        model: &GnmModel,
        mapping: &DenseCorrespondenceSet,
        solver_config: SingleFrameFitConfig,
        auxiliary: Option<(&dyn AuxiliaryObjectiveTerm, f32)>,
    ) -> Result<PersistentGnmFrameOutcome, PersistentGnmFitterError> {
        let stamp = input.stamp();
        let decision = advance_persistent_gnm_lifecycle(
            self.lifecycle,
            PersistentGnmEvent::SourceFrame {
                stamp,
                observation_available: input.usable_observation(),
            },
            self.config,
        )
        .map_err(PersistentGnmFitterError::Lifecycle)?;
        self.lifecycle = decision.state;

        match decision.action {
            PersistentGnmAction::SkipUncalibratedFrame => {
                Ok(PersistentGnmFrameOutcome::SkippedUncalibrated)
            }
            PersistentGnmAction::NoObservation {
                dynamic_state_cleared,
            } => {
                if dynamic_state_cleared {
                    self.validated = None;
                }
                Ok(PersistentGnmFrameOutcome::NoObservation {
                    dynamic_state_cleared,
                })
            }
            PersistentGnmAction::StartFit { initialization } => self.solve_started_frame(
                stamp,
                initialization,
                input,
                model,
                mapping,
                solver_config,
                auxiliary,
            ),
            impossible @ (PersistentGnmAction::ResetDynamicState
            | PersistentGnmAction::PublishCurrentFit
            | PersistentGnmAction::RejectInvalidFit
            | PersistentGnmAction::RejectInvalidFitAndLose) => {
                // Unreachable from a SourceFrame event by lifecycle
                // construction; fail closed without panicking.
                Err(PersistentGnmFitterError::UnexpectedLifecycleAction {
                    action: impossible,
                    stamp,
                })
            }
        }
    }

    /// Runs the bounded solve for an already-admitted StartFit frame and
    /// routes the result back through the lifecycle.
    #[allow(clippy::too_many_arguments)]
    fn solve_started_frame(
        &mut self,
        stamp: GnmFrameStamp,
        initialization: GnmFitInitialization,
        input: &GnmSolverFrameInput<'_>,
        model: &GnmModel,
        mapping: &DenseCorrespondenceSet,
        solver_config: SingleFrameFitConfig,
        auxiliary: Option<(&dyn AuxiliaryObjectiveTerm, f32)>,
    ) -> Result<PersistentGnmFrameOutcome, PersistentGnmFitterError> {
        let initial = match initialization {
            GnmFitInitialization::NeutralFirstFit => {
                neutral_initial_values(model, input.identity(), mapping)
                    .map_err(PersistentGnmFitterError::Solve)
            }
            GnmFitInitialization::PreviousValid { source } => {
                self.previous_valid_initial_values(source)
            }
            GnmFitInitialization::ReinitializeDynamicState => {
                reinitialized_initial_values(model, input.identity(), mapping, input.observation())
                    .map_err(PersistentGnmFitterError::Solve)
            }
        };
        let initial = match initial {
            Ok(initial) => initial,
            Err(error) => {
                // Keep lifecycle ownership consistent before failing closed.
                self.reject_pending(stamp)?;
                return Err(error);
            }
        };

        let outcome = match fit_single_frame_cold_start(
            model,
            input.identity().state(),
            &initial.expression,
            &initial.joints,
            mapping,
            input.observation(),
            &initial.projection,
            solver_config,
            auxiliary,
        ) {
            Ok(outcome) => outcome,
            Err(error) => {
                self.reject_pending(stamp)?;
                return Err(PersistentGnmFitterError::Solve(error));
            }
        };

        if !outcome.valid() {
            self.reject_pending(stamp)?;
            return Ok(PersistentGnmFrameOutcome::Solved(GnmSolvedFrameReport {
                initialization,
                status: outcome.status(),
                iterations: outcome.iterations(),
                objective: outcome.objective(),
                published: false,
            }));
        }

        // Converged outcomes carry finite state by contract; convert into the
        // validated dynamic-state shape and re-validate against the model.
        let dynamic = match dynamic_from_solver_output(
            outcome.projection(),
            outcome.expression().clone(),
            outcome.joints().clone(),
        ) {
            Ok(dynamic) => match dynamic.validate(model) {
                Ok(()) => dynamic,
                Err(error) => {
                    self.reject_pending(stamp)?;
                    return Err(PersistentGnmFitterError::Contract(error));
                }
            },
            Err(error) => {
                self.reject_pending(stamp)?;
                return Err(PersistentGnmFitterError::Contract(error));
            }
        };

        let result_decision = advance_persistent_gnm_lifecycle(
            self.lifecycle,
            PersistentGnmEvent::FitResult {
                stamp,
                outcome: GnmFitOutcome::Valid,
            },
            self.config,
        )
        .map_err(PersistentGnmFitterError::Lifecycle)?;
        self.lifecycle = result_decision.state;

        let published = matches!(
            result_decision.action,
            PersistentGnmAction::PublishCurrentFit
        );
        if published {
            self.validated = Some(GnmValidatedDynamicFrame { stamp, dynamic });
        } else {
            // RejectInvalidFitAndLose cleared the lifecycle's previous-valid
            // authority; the stored warm-start frame must go with it.
            self.validated = None;
        }

        Ok(PersistentGnmFrameOutcome::Solved(GnmSolvedFrameReport {
            initialization,
            status: outcome.status(),
            iterations: outcome.iterations(),
            objective: outcome.objective(),
            published,
        }))
    }

    /// Clones the stored validated frame named by a `PreviousValid` directive
    /// as solver initial values.
    fn previous_valid_initial_values(
        &mut self,
        source: GnmFrameStamp,
    ) -> Result<InitialDynamicValues, PersistentGnmFitterError> {
        let Some(validated) = self.validated.as_ref() else {
            return Err(PersistentGnmFitterError::MissingWarmStartState { source });
        };
        if validated.stamp != source {
            return Err(PersistentGnmFitterError::MissingWarmStartState { source });
        }
        // The conversion below cannot fail for a validated block set (the
        // camera focal was checked positive and the pose finite on
        // construction); keep the typed failure path anyway instead of
        // assuming it.
        projection_of_dynamic(&validated.dynamic)
            .map(|projection| InitialDynamicValues {
                expression: validated.dynamic.expression.clone(),
                joints: validated.dynamic.joints.clone(),
                projection,
            })
            .map_err(PersistentGnmFitterError::Solve)
    }

    /// Resolves the currently pending solve as invalid in the lifecycle.
    fn reject_pending(&mut self, stamp: GnmFrameStamp) -> Result<(), PersistentGnmFitterError> {
        let decision = advance_persistent_gnm_lifecycle(
            self.lifecycle,
            PersistentGnmEvent::FitResult {
                stamp,
                outcome: GnmFitOutcome::Invalid,
            },
            self.config,
        )
        .map_err(PersistentGnmFitterError::Lifecycle)?;
        self.lifecycle = decision.state;
        if matches!(
            decision.action,
            PersistentGnmAction::RejectInvalidFitAndLose
        ) {
            self.validated = None;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vtuber_gnm::{
        AnatomicalSide, CorrespondenceProvenance, CorrespondenceReliability, DenseCoveragePolicy,
        DenseExpressionJointStepConfig, DenseMappingVersion, DenseRigidStepConfig, FaceRegion,
        GNM_HEAD_V3_EXPRESSION_DIM, GNM_HEAD_V3_IDENTITY_DIM, GNM_HEAD_V3_VERSION, GnmModelData,
        GnmSurfacePointRef, GnmVariant, MediaPipeGnmDenseCorrespondence, SynthesisOptions,
        synthesize_observation_from_projection,
    };

    fn lifecycle_config() -> PersistentGnmLifecycleConfig {
        PersistentGnmLifecycleConfig::new(50_000, 250_000, 3).unwrap()
    }

    fn solver_config() -> SingleFrameFitConfig {
        SingleFrameFitConfig::default()
    }

    /// Small deterministic head model with an expression channel pair
    /// (mouth/eyelid), matching the vtuber-gnm solver fixtures.
    fn lin_model() -> GnmModel {
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
        GnmModel::from_data(GnmModelData {
            version: GNM_HEAD_V3_VERSION,
            variant: GnmVariant::Head,
            template_vertices: vtuber_gnm::DenseArray::new(
                "vertices",
                vec![vertex_count, 3],
                vertices,
            )
            .unwrap(),
            template_joints: vtuber_gnm::DenseArray::new("joints", vec![1, 3], vec![0.0; 3])
                .unwrap(),
            vertex_identity_basis: vtuber_gnm::DenseArray::new(
                "identity",
                vec![identity, vertex_count, 3],
                vec![0.0; identity * vertex_count * 3],
            )
            .unwrap(),
            joint_identity_basis: vtuber_gnm::DenseArray::new(
                "joint_identity",
                vec![identity, 1, 3],
                vec![0.0; identity * 3],
            )
            .unwrap(),
            expression_basis: vtuber_gnm::DenseArray::new(
                "expression",
                vec![expression, vertex_count, 3],
                expression_basis,
            )
            .unwrap(),
            joint_parent_indices: vec![-1],
            skinning_weights: vtuber_gnm::DenseArray::new(
                "weights",
                vec![1, vertex_count],
                vec![1.0; vertex_count],
            )
            .unwrap(),
            pose_correctives_regressor: None,
        })
        .unwrap()
    }

    fn mapping_for(model: &GnmModel) -> DenseCorrespondenceSet {
        let rows: Vec<MediaPipeGnmDenseCorrespondence> = (0..64)
            .map(|index| MediaPipeGnmDenseCorrespondence {
                mediapipe_index: 10 + index,
                target: GnmSurfacePointRef::Vertex {
                    vertex_index: index,
                },
                region: if index % 3 == 0 {
                    FaceRegion::Nose
                } else if index % 3 == 1 {
                    FaceRegion::Contour
                } else {
                    FaceRegion::Other
                },
                anatomical_side: if index % 3 == 0 {
                    AnatomicalSide::Midline
                } else if index % 3 == 1 {
                    AnatomicalSide::Right
                } else {
                    AnatomicalSide::Left
                },
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

    fn mouth_expression(mouth: f32) -> GnmExpressionState {
        let mut values = vec![0.0; GNM_HEAD_V3_EXPRESSION_DIM];
        values[0] = mouth;
        GnmExpressionState::new(values, GNM_HEAD_V3_EXPRESSION_DIM).unwrap()
    }

    fn stamped_observation(
        model: &GnmModel,
        mapping: &DenseCorrespondenceSet,
        seq: u64,
        micros: u64,
    ) -> GnmDenseObservation {
        synthesize_observation_from_projection(
            model,
            &model.neutral_identity(),
            &mouth_expression(0.8),
            &GnmJointState::neutral(model.joint_count()),
            mapping,
            &truth_projection(),
            SynthesisOptions {
                source_seq: seq,
                captured_at_micros: micros,
                ..SynthesisOptions::default()
            },
            DenseCoveragePolicy::new(2, 0.75).unwrap(),
            |_, _| false,
        )
        .unwrap()
    }

    fn insufficient_observation(
        mapping: &DenseCorrespondenceSet,
        seq: u64,
        micros: u64,
    ) -> GnmDenseObservation {
        let landmarks = vec![[f32::NAN; 2]; vtuber_gnm::MEDIAPIPE_FACE_LANDMARK_COUNT];
        GnmDenseObservation::from_mediapipe_xy(
            seq,
            micros,
            &landmarks,
            mapping,
            DenseCoveragePolicy::new(2, 0.75).unwrap(),
        )
        .unwrap()
    }

    fn frame_input<'a>(
        observation: &'a GnmDenseObservation,
        identity: &'a FixedGnmIdentity,
        model: &GnmModel,
        mapping: &DenseCorrespondenceSet,
    ) -> GnmSolverFrameInput<'a> {
        GnmSolverFrameInput::new(
            GnmFrameStamp {
                source_seq: observation.source_seq(),
                captured_at_micros: observation.captured_at_micros(),
            },
            observation,
            None,
            identity,
            model,
            mapping,
        )
        .unwrap()
    }

    fn calibrated_fitter() -> (
        GnmModel,
        DenseCorrespondenceSet,
        FixedGnmIdentity,
        PersistentGnmFitter,
    ) {
        let model = lin_model();
        let mapping = mapping_for(&model);
        let identity = FixedGnmIdentity::new(model.neutral_identity(), &model).unwrap();
        let mut fitter = PersistentGnmFitter::new(lifecycle_config());
        fitter.calibration_ready().unwrap();
        (model, mapping, identity, fitter)
    }

    #[test]
    fn cold_then_warm_start_reuses_only_the_validated_dynamic_state() {
        let (model, mapping, identity, mut fitter) = calibrated_fitter();

        // Cold first fit starts from neutral dynamic state.
        let obs1 = stamped_observation(&model, &mapping, 1, 1_000_000);
        let report1 = fitter
            .fit_frame(
                &frame_input(&obs1, &identity, &model, &mapping),
                &model,
                &mapping,
                solver_config(),
                None,
            )
            .unwrap();
        let PersistentGnmFrameOutcome::Solved(cold) = report1 else {
            panic!("expected a solved cold-start frame");
        };
        assert_eq!(cold.initialization, GnmFitInitialization::NeutralFirstFit);
        assert!(cold.published);
        assert_eq!(cold.status, SingleFrameFitStatus::Converged);
        let stored = fitter.validated().cloned().unwrap();
        assert_eq!(stored.stamp().source_seq, 1);

        // Warm second fit must declare PreviousValid and reuse the stored
        // validated state as initial values only.
        let obs2 = stamped_observation(&model, &mapping, 2, 1_016_000);
        let report2 = fitter
            .fit_frame(
                &frame_input(&obs2, &identity, &model, &mapping),
                &model,
                &mapping,
                solver_config(),
                None,
            )
            .unwrap();
        let PersistentGnmFrameOutcome::Solved(warm) = report2 else {
            panic!("expected a solved warm-start frame");
        };
        assert_eq!(
            warm.initialization,
            GnmFitInitialization::PreviousValid {
                source: GnmFrameStamp {
                    source_seq: 1,
                    captured_at_micros: 1_000_000
                }
            }
        );
        assert!(warm.published);
        // Both iteration counts are retrievable, and a warm start from an
        // already-converged validated state cannot need more block-coordinate
        // iterations than the original cold start on identical data.
        assert!(cold.iterations >= 1);
        assert!(warm.iterations >= 1);
        assert!(
            warm.iterations < cold.iterations,
            "warm {} vs cold {}",
            warm.iterations,
            cold.iterations
        );
    }

    #[test]
    fn duplicate_and_regressed_sources_are_rejected_by_the_lifecycle() {
        let (model, mapping, identity, mut fitter) = calibrated_fitter();

        let obs = stamped_observation(&model, &mapping, 7, 1_000_000);
        let input = frame_input(&obs, &identity, &model, &mapping);
        assert!(
            fitter
                .fit_frame(&input, &model, &mapping, solver_config(), None)
                .is_ok()
        );

        // Exact duplicate sequence is rejected before any solve runs.
        let duplicate = stamped_observation(&model, &mapping, 7, 1_100_000);
        assert!(matches!(
            fitter.fit_frame(
                &frame_input(&duplicate, &identity, &model, &mapping),
                &model,
                &mapping,
                solver_config(),
                None,
            ),
            Err(PersistentGnmFitterError::Lifecycle(
                PersistentGnmLifecycleError::DuplicateSourceSequence { .. }
            ))
        ));

        // Timestamp regression is fail-closed as well.
        let regressed = stamped_observation(&model, &mapping, 8, 900_000);
        assert!(matches!(
            fitter.fit_frame(
                &frame_input(&regressed, &identity, &model, &mapping),
                &model,
                &mapping,
                solver_config(),
                None,
            ),
            Err(PersistentGnmFitterError::Lifecycle(
                PersistentGnmLifecycleError::RegressedTimestamp { .. }
            ))
        ));
    }

    #[test]
    fn invalid_solve_never_replaces_the_previous_valid_state() {
        let (model, mapping, identity, mut fitter) = calibrated_fitter();

        let obs1 = stamped_observation(&model, &mapping, 1, 1_000_000);
        assert!(
            fitter
                .fit_frame(
                    &frame_input(&obs1, &identity, &model, &mapping),
                    &model,
                    &mapping,
                    solver_config(),
                    None,
                )
                .is_ok()
        );
        let stored = fitter.validated().cloned().unwrap();

        // Force an invalid solve: a one-iteration, tiny-step budget starting
        // far from freshly changed data cannot converge.
        let starved = SingleFrameFitConfig::new(
            DenseRigidStepConfig::new(1.0e-4, 1.0e-4, 1.0e-6).unwrap(),
            DenseExpressionJointStepConfig::new(1.0e-4, 1.0e-5, 1.0e-5, 1.0e-8, 1.0e-7).unwrap(),
            1,
            0.0,
        )
        .unwrap();
        // Neutral-mouth truth so the stored mouth-open state is not already
        // optimal for this frame and one starved iteration cannot finish.
        let obs2 = synthesize_observation_from_projection(
            &model,
            &model.neutral_identity(),
            &mouth_expression(0.0),
            &GnmJointState::neutral(model.joint_count()),
            &mapping,
            &truth_projection(),
            SynthesisOptions {
                source_seq: 2,
                captured_at_micros: 1_016_000,
                ..SynthesisOptions::default()
            },
            DenseCoveragePolicy::new(2, 0.75).unwrap(),
            |_, _| false,
        )
        .unwrap();
        let report = fitter
            .fit_frame(
                &frame_input(&obs2, &identity, &model, &mapping),
                &model,
                &mapping,
                starved,
                None,
            )
            .unwrap();
        let PersistentGnmFrameOutcome::Solved(rejected) = report else {
            panic!("expected a solved-but-invalid frame");
        };
        assert!(!rejected.published);
        assert_eq!(rejected.status, SingleFrameFitStatus::MaxIterationsReached);
        assert_eq!(fitter.validated(), Some(&stored));
        assert_eq!(
            fitter.lifecycle_state().previous_valid,
            Some(GnmFrameStamp {
                source_seq: 1,
                captured_at_micros: 1_000_000
            })
        );

        // The next normal frame still warm-starts from the retained state.
        let obs3 = stamped_observation(&model, &mapping, 3, 1_032_000);
        let report3 = fitter
            .fit_frame(
                &frame_input(&obs3, &identity, &model, &mapping),
                &model,
                &mapping,
                solver_config(),
                None,
            )
            .unwrap();
        let PersistentGnmFrameOutcome::Solved(retry) = report3 else {
            panic!("expected the retry to solve");
        };
        assert_eq!(
            retry.initialization,
            GnmFitInitialization::PreviousValid {
                source: GnmFrameStamp {
                    source_seq: 1,
                    captured_at_micros: 1_000_000
                }
            }
        );
        assert!(retry.published);
    }

    #[test]
    fn long_gap_reacquire_discards_old_expression_and_joint_state() {
        let (model, mapping, identity, mut fitter) = calibrated_fitter();

        let obs1 = stamped_observation(&model, &mapping, 1, 1_000_000);
        assert!(
            fitter
                .fit_frame(
                    &frame_input(&obs1, &identity, &model, &mapping),
                    &model,
                    &mapping,
                    solver_config(),
                    None,
                )
                .is_ok()
        );
        assert!(fitter.validated().is_some());

        // Reacquire far beyond the dynamic reuse gap: the directive must be
        // ReinitializeDynamicState.
        let obs2 = stamped_observation(&model, &mapping, 2, 1_300_001);
        let report = fitter
            .fit_frame(
                &frame_input(&obs2, &identity, &model, &mapping),
                &model,
                &mapping,
                solver_config(),
                None,
            )
            .unwrap();
        let PersistentGnmFrameOutcome::Solved(reacquired) = report else {
            panic!("expected the reacquire frame to solve");
        };
        assert_eq!(
            reacquired.initialization,
            GnmFitInitialization::ReinitializeDynamicState
        );
        assert!(reacquired.published);

        // The reinitialize helper itself never carries expression or joint
        // history: both blocks are exactly neutral and only pose evidence is
        // recovered from the fresh observation.
        let initial = reinitialized_initial_values(&model, &identity, &mapping, &obs2).unwrap();
        assert_eq!(initial.expression, model.neutral_expression());
        assert_eq!(initial.joints, GnmJointState::neutral(model.joint_count()));
        assert_eq!(
            fitter.validated().map(|frame| frame.stamp().source_seq),
            Some(2)
        );
    }

    #[test]
    fn long_no_face_gap_clears_the_stored_warm_start_state() {
        let (model, mapping, identity, mut fitter) = calibrated_fitter();

        let obs1 = stamped_observation(&model, &mapping, 1, 1_000_000);
        assert!(
            fitter
                .fit_frame(
                    &frame_input(&obs1, &identity, &model, &mapping),
                    &model,
                    &mapping,
                    solver_config(),
                    None,
                )
                .is_ok()
        );

        // Short no-face gap keeps the validated frame available.
        let blind_short = insufficient_observation(&mapping, 2, 1_048_000);
        assert!(matches!(
            fitter.fit_frame(
                &frame_input(&blind_short, &identity, &model, &mapping),
                &model,
                &mapping,
                solver_config(),
                None,
            ),
            Ok(PersistentGnmFrameOutcome::NoObservation {
                dynamic_state_cleared: false
            })
        ));
        assert!(fitter.validated().is_some());

        // Long no-face gap clears it everywhere.
        let blind_long = insufficient_observation(&mapping, 3, 1_300_001);
        assert!(matches!(
            fitter.fit_frame(
                &frame_input(&blind_long, &identity, &model, &mapping),
                &model,
                &mapping,
                solver_config(),
                None,
            ),
            Ok(PersistentGnmFrameOutcome::NoObservation {
                dynamic_state_cleared: true
            })
        ));
        assert!(fitter.validated().is_none());
        assert_eq!(fitter.lifecycle_state().previous_valid, None);

        // Reacquisition afterwards never warm-starts the cleared state.
        let obs4 = stamped_observation(&model, &mapping, 4, 1_400_000);
        let report = fitter
            .fit_frame(
                &frame_input(&obs4, &identity, &model, &mapping),
                &model,
                &mapping,
                solver_config(),
                None,
            )
            .unwrap();
        let PersistentGnmFrameOutcome::Solved(reacquired) = report else {
            panic!("expected the reacquire frame to solve");
        };
        assert_eq!(
            reacquired.initialization,
            GnmFitInitialization::ReinitializeDynamicState
        );
    }

    #[test]
    fn missing_warm_start_state_is_fail_closed_and_resolves_the_pending_fit() {
        let (model, mapping, identity, mut fitter) = calibrated_fitter();

        let obs1 = stamped_observation(&model, &mapping, 1, 1_000_000);
        assert!(
            fitter
                .fit_frame(
                    &frame_input(&obs1, &identity, &model, &mapping),
                    &model,
                    &mapping,
                    solver_config(),
                    None,
                )
                .is_ok()
        );

        // Simulate the internal invariant violation directly: the lifecycle
        // says a previous valid state exists but the store is gone.
        drop(fitter.validated.take());
        let obs2 = stamped_observation(&model, &mapping, 2, 1_016_000);
        assert!(matches!(
            fitter.fit_frame(
                &frame_input(&obs2, &identity, &model, &mapping),
                &model,
                &mapping,
                solver_config(),
                None,
            ),
            Err(PersistentGnmFitterError::MissingWarmStartState { .. })
        ));
        // The pending fit was resolved as invalid each time, so subsequent
        // frames are still admitted (no FitStillPending deadlock). Each new
        // frame repeats the PreviousValid directive while previous_valid
        // exists, keeps failing closed, and counts toward the configured
        // consecutive-invalid bound of three.
        let obs3 = stamped_observation(&model, &mapping, 3, 1_032_000);
        assert!(matches!(
            fitter.fit_frame(
                &frame_input(&obs3, &identity, &model, &mapping),
                &model,
                &mapping,
                solver_config(),
                None,
            ),
            Err(PersistentGnmFitterError::MissingWarmStartState { .. })
        ));

        // The final allowed failure clears the previous-valid authority
        // entirely (RejectInvalidFitAndLose path).
        let obs4 = stamped_observation(&model, &mapping, 4, 1_048_000);
        assert!(matches!(
            fitter.fit_frame(
                &frame_input(&obs4, &identity, &model, &mapping),
                &model,
                &mapping,
                solver_config(),
                None,
            ),
            Err(PersistentGnmFitterError::MissingWarmStartState { .. })
        ));
        assert_eq!(fitter.lifecycle_state().previous_valid, None);
        assert_eq!(
            fitter.lifecycle_state().phase,
            vtuber_gnm::PersistentGnmPhase::Lost
        );

        // With the stale authority gone, the next observation reacquires
        // through ReinitializeDynamicState and succeeds without a store.
        let recovered = stamped_observation(&model, &mapping, 5, 1_064_000);
        assert!(matches!(
            fitter.fit_frame(
                &frame_input(&recovered, &identity, &model, &mapping),
                &model,
                &mapping,
                solver_config(),
                None,
            ),
            Ok(PersistentGnmFrameOutcome::Solved(solved))
                if solved.initialization == GnmFitInitialization::ReinitializeDynamicState
                    && solved.published
        ));
    }

    #[test]
    fn calibration_events_clear_dynamic_state() {
        let (model, mapping, identity, mut fitter) = calibrated_fitter();

        let obs1 = stamped_observation(&model, &mapping, 1, 1_000_000);
        assert!(
            fitter
                .fit_frame(
                    &frame_input(&obs1, &identity, &model, &mapping),
                    &model,
                    &mapping,
                    solver_config(),
                    None,
                )
                .is_ok()
        );
        assert!(fitter.validated().is_some());

        fitter.calibration_invalidated().unwrap();
        assert!(fitter.validated().is_none());
        assert_eq!(
            fitter.lifecycle_state().phase,
            vtuber_gnm::PersistentGnmPhase::Uncalibrated
        );

        // Uncalibrated frames are skipped, not solved.
        let obs2 = stamped_observation(&model, &mapping, 2, 1_100_000);
        assert!(matches!(
            fitter.fit_frame(
                &frame_input(&obs2, &identity, &model, &mapping),
                &model,
                &mapping,
                solver_config(),
                None,
            ),
            Ok(PersistentGnmFrameOutcome::SkippedUncalibrated)
        ));

        fitter.calibration_ready().unwrap();
        assert_eq!(
            fitter.lifecycle_state().phase,
            vtuber_gnm::PersistentGnmPhase::ReadyForFirstFit
        );
        assert!(fitter.validated().is_none());
    }

    #[test]
    fn uncalibrated_frames_are_skipped_without_a_solve() {
        let model = lin_model();
        let mapping = mapping_for(&model);
        let identity = FixedGnmIdentity::new(model.neutral_identity(), &model).unwrap();
        let mut fitter = PersistentGnmFitter::new(lifecycle_config());

        let obs = stamped_observation(&model, &mapping, 1, 1_000_000);
        assert!(matches!(
            fitter.fit_frame(
                &frame_input(&obs, &identity, &model, &mapping),
                &model,
                &mapping,
                solver_config(),
                None,
            ),
            Ok(PersistentGnmFrameOutcome::SkippedUncalibrated)
        ));
        assert!(fitter.validated().is_none());
    }

    #[test]
    fn neutral_initial_values_use_neutral_blocks_and_a_fitting_projection() {
        let model = lin_model();
        let mapping = mapping_for(&model);
        let identity = FixedGnmIdentity::new(model.neutral_identity(), &model).unwrap();

        let initial = neutral_initial_values(&model, &identity, &mapping).unwrap();
        assert_eq!(initial.expression, model.neutral_expression());
        assert_eq!(initial.joints, GnmJointState::neutral(model.joint_count()));
        assert!(initial.projection.focal() > 0.0);
        assert!(initial.projection.translation()[2] > 0.0);
    }

    #[test]
    fn dynamic_state_round_trips_through_the_projection_form() {
        let dynamic = GnmDynamicState {
            expression: mouth_expression(0.8),
            joints: GnmJointState::neutral(1),
            rigid_pose: GnmRigidPoseBlock::new([0.15, -0.10, 0.05]).unwrap(),
            camera: GnmCameraBlock::new([0.02, -0.03, 0.60], 1.3, [0.5, 0.5]).unwrap(),
        };
        let projection = projection_of_dynamic(&dynamic).unwrap();
        assert_eq!(projection.yaw_pitch_roll(), [0.15, -0.10, 0.05]);
        assert_eq!(projection.translation(), [0.02, -0.03, 0.60]);
        assert_eq!(projection.focal(), 1.3);
        assert_eq!(projection.principal_point(), [0.5, 0.5]);
    }
}
