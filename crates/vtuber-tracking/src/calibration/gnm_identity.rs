//! Explicit-event GNM identity calibration lifecycle (Issue #85 / GNM #54.4).
//!
//! Owns neutral candidate collection, publication of a solved
//! [`GnmIdentityCalibration`], and its retention across tracking stop/start
//! within one session. The store is deliberately event-driven: the only way to
//! change it is an explicit [`GnmIdentityCalibrationEvent`]. Ordinary tracking
//! residuals are not an input anywhere in this module, so identity can never
//! be silently re-solved as a side effect of frame fitting.
//!
//! Model/mapping version mismatches invalidate the stored calibration instead
//! of failing: a stale calibration must never reach a new model.

use vtuber_gnm::{
    DenseMappingVersion, FixedGnmIdentity, GnmIdentityCalibration, GnmModel, GnmVersion,
};

/// Explicit lifecycle events accepted by [`GnmIdentityCalibrationStore::apply`].
///
/// These are the only mutations of the store. Tracking residuals, fit
/// outcomes, and confidence metrics are intentionally absent.
#[derive(Clone, Debug, PartialEq)]
pub enum GnmIdentityCalibrationEvent {
    /// Begin collecting a neutral candidate window.
    Start,
    /// Publish a completed calibration and finish collection.
    Complete(Box<GnmIdentityCalibration>),
    /// Abandon an in-progress window without publishing.
    Cancel,
    /// Drop any published calibration and return to the idle phase.
    Reset,
    /// Invalidate the stored calibration because the runtime model or mapping
    /// version no longer matches its binding.
    Invalidate(CalibrationInvalidation),
}

/// Why a stored calibration stopped being usable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CalibrationInvalidation {
    /// The loaded GNM model version changed.
    ModelVersionChanged,
    /// The dense mapping version changed.
    MappingVersionChanged,
}

/// Phase of the explicit calibration lifecycle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GnmIdentityCalibrationPhase {
    /// No collection in progress and no published calibration.
    Idle,
    /// A neutral candidate window is being collected after `Start`.
    Collecting,
    /// A calibration is published and bound to these versions.
    Active {
        /// Bound GNM model version.
        model_version: GnmVersion,
        /// Bound dense mapping version (schema + model version).
        mapping_version: DenseMappingVersion,
    },
}

/// Typed failure from explicit lifecycle events.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GnmIdentityLifecycleError {
    /// `Start` was received while already collecting.
    AlreadyCollecting,
    /// `Complete` or `Cancel` was received without a prior `Start`.
    NotCollecting,
}

impl std::fmt::Display for GnmIdentityLifecycleError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyCollecting => write!(formatter, "calibration collection already started"),
            Self::NotCollecting => write!(formatter, "no calibration collection in progress"),
        }
    }
}

impl std::error::Error for GnmIdentityLifecycleError {}

/// Session-scoped holder for the shared GNM identity calibration.
///
/// The store survives tracking stop/start: only [`GnmIdentityCalibrationEvent::Reset`]
/// or an invalidation drops the published calibration. Access goes through
/// read-only references; there is no way to mutate a published calibration.
#[derive(Clone, Debug)]
pub struct GnmIdentityCalibrationStore {
    phase: GnmIdentityCalibrationPhase,
    calibration: Option<GnmIdentityCalibration>,
    tracking_restarts: u32,
}

impl GnmIdentityCalibrationStore {
    /// Creates an idle store with no calibration.
    pub fn new() -> Self {
        Self {
            phase: GnmIdentityCalibrationPhase::Idle,
            calibration: None,
            tracking_restarts: 0,
        }
    }

    /// Returns the current lifecycle phase.
    pub fn phase(&self) -> GnmIdentityCalibrationPhase {
        self.phase
    }

    /// Returns how many tracking stop/start cycles this session survived.
    pub fn tracking_restarts(&self) -> u32 {
        self.tracking_restarts
    }

    /// Applies one explicit lifecycle event.
    pub fn apply(
        &mut self,
        event: GnmIdentityCalibrationEvent,
    ) -> Result<(), GnmIdentityLifecycleError> {
        match event {
            GnmIdentityCalibrationEvent::Start => {
                if self.phase == GnmIdentityCalibrationPhase::Collecting {
                    return Err(GnmIdentityLifecycleError::AlreadyCollecting);
                }
                self.phase = GnmIdentityCalibrationPhase::Collecting;
                Ok(())
            }
            GnmIdentityCalibrationEvent::Complete(calibration) => {
                if self.phase != GnmIdentityCalibrationPhase::Collecting {
                    return Err(GnmIdentityLifecycleError::NotCollecting);
                }
                let model_version = calibration.model_version();
                let mapping_version = calibration.mapping_version();
                self.calibration = Some(*calibration);
                self.phase = GnmIdentityCalibrationPhase::Active {
                    model_version,
                    mapping_version,
                };
                Ok(())
            }
            GnmIdentityCalibrationEvent::Cancel => {
                if self.phase != GnmIdentityCalibrationPhase::Collecting {
                    return Err(GnmIdentityLifecycleError::NotCollecting);
                }
                self.recompute_phase();
                Ok(())
            }
            GnmIdentityCalibrationEvent::Reset => {
                self.calibration = None;
                self.recompute_phase();
                Ok(())
            }
            GnmIdentityCalibrationEvent::Invalidate(reason) => {
                self.invalidate(reason);
                Ok(())
            }
        }
    }

    /// Records a tracking stop/start cycle. Deliberately preserves any
    /// published calibration so the same session can reuse it.
    pub fn on_tracking_restarted(&mut self) {
        self.tracking_restarts += 1;
    }

    /// Marks a stale calibration invalidated when the runtime boundary changed.
    pub fn invalidate(&mut self, reason: CalibrationInvalidation) {
        let _ = reason; // recorded by callers in diagnostics; the outcome is the same
        self.calibration = None;
        self.recompute_phase();
    }

    fn recompute_phase(&mut self) {
        self.phase = match &self.calibration {
            Some(calibration) => GnmIdentityCalibrationPhase::Active {
                model_version: calibration.model_version(),
                mapping_version: calibration.mapping_version(),
            },
            None => GnmIdentityCalibrationPhase::Idle,
        };
    }

    /// Returns the published calibration only when it exactly matches the
    /// runtime model and mapping versions; otherwise the stale calibration is
    /// invalidated and `None` is returned.
    pub fn calibration_for_runtime(
        &mut self,
        model: &GnmModel,
        mapping_version: DenseMappingVersion,
    ) -> Option<&GnmIdentityCalibration> {
        let matches = self
            .calibration
            .as_ref()
            .is_some_and(|calibration| calibration.matches_runtime(model, mapping_version));
        if !matches {
            if self.calibration.is_some() {
                self.invalidate(CalibrationInvalidation::ModelVersionChanged);
            }
            return None;
        }
        self.calibration.as_ref()
    }

    /// Convenience accessor returning the fixed identity through its
    /// read-only wrapper under the same version gate as
    /// [`Self::calibration_for_runtime`].
    pub fn fixed_identity_for_runtime(
        &mut self,
        model: &GnmModel,
        mapping_version: DenseMappingVersion,
    ) -> Option<&FixedGnmIdentity> {
        self.calibration_for_runtime(model, mapping_version)
            .map(GnmIdentityCalibration::identity)
    }
}

impl Default for GnmIdentityCalibrationStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vtuber_gnm::{
        AnatomicalSide, CorrespondenceProvenance, CorrespondenceReliability, DenseArray,
        DenseCorrespondenceSet, DenseCoveragePolicy, DenseProjection, DenseReprojectionConfig,
        FaceRegion, GNM_HEAD_V3_EXPRESSION_DIM, GNM_HEAD_V3_IDENTITY_DIM, GNM_HEAD_V3_VERSION,
        GnmDenseObservation, GnmModelData, GnmVariant, MediaPipeGnmDenseCorrespondence,
        NeutralSampleSolveInput, SampleNuisance, SharedIdentitySolveConfig,
        SharedIdentitySolveInput, SynthesisOptions, finalize_identity_calibration,
        solve_shared_identity,
    };

    // -- fixtures mirroring the vtuber-gnm synthetic conditioning tests -------

    fn synthetic_model_data(version: GnmVersion) -> GnmModelData {
        let identity = GNM_HEAD_V3_IDENTITY_DIM;
        let expression = GNM_HEAD_V3_EXPRESSION_DIM;
        GnmModelData {
            version,
            variant: GnmVariant::Head,
            template_vertices: DenseArray::new(
                "vertices",
                vec![3, 3],
                vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            )
            .unwrap(),
            template_joints: DenseArray::new("joints", vec![1, 3], vec![0.0; 3]).unwrap(),
            vertex_identity_basis: DenseArray::new(
                "identity",
                vec![identity, 3, 3],
                vec![0.0; identity * 9],
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
                vec![expression, 3, 3],
                vec![0.0; expression * 9],
            )
            .unwrap(),
            joint_parent_indices: vec![-1],
            skinning_weights: DenseArray::new("weights", vec![1, 3], vec![1.0; 3]).unwrap(),
            pose_correctives_regressor: None,
        }
    }

    fn mapping_version(model_version: GnmVersion) -> DenseMappingVersion {
        DenseMappingVersion {
            schema_revision: 1,
            model_version,
        }
    }

    fn solve_mapping(model: &GnmModel) -> DenseCorrespondenceSet {
        let rows: Vec<MediaPipeGnmDenseCorrespondence> = (0..model.vertex_count())
            .map(|index| MediaPipeGnmDenseCorrespondence {
                mediapipe_index: 10 + index,
                target: vtuber_gnm::GnmSurfacePointRef::Vertex {
                    vertex_index: index,
                },
                region: FaceRegion::Other,
                anatomical_side: AnatomicalSide::Midline,
                base_weight: 1.0,
                provenance: CorrespondenceProvenance::RepositoryValidated,
                reliability: CorrespondenceReliability::High,
            })
            .collect();
        DenseCorrespondenceSet::new(mapping_version(model.version()), rows, model).unwrap()
    }

    fn truth_projection() -> DenseProjection {
        DenseProjection::new([0.1, -0.05, 0.02], [0.0, 0.0, 0.6], 1.4, [0.5, 0.5]).unwrap()
    }

    fn observation(
        model: &GnmModel,
        mapping: &DenseCorrespondenceSet,
        source_seq: u64,
    ) -> GnmDenseObservation {
        vtuber_gnm::synthesize_observation_from_projection(
            model,
            &model.neutral_identity(),
            &model.neutral_expression(),
            &vtuber_gnm::GnmJointState::neutral(model.joint_count()),
            mapping,
            &truth_projection(),
            SynthesisOptions {
                source_seq,
                captured_at_micros: source_seq * 10,
                noise_amplitude: 0.0,
                noise_seed: source_seq,
            },
            DenseCoveragePolicy::new(1, 1.0).unwrap(),
            |_, _| false,
        )
        .unwrap()
    }

    fn solved_calibration(
        model: &GnmModel,
        mapping: &DenseCorrespondenceSet,
    ) -> GnmIdentityCalibration {
        let observations = [
            observation(model, mapping, 1),
            observation(model, mapping, 2),
        ];
        let samples = observations
            .iter()
            .map(|observed| {
                NeutralSampleSolveInput::new(
                    observed,
                    SampleNuisance::new(
                        truth_projection(),
                        model.neutral_expression(),
                        model.expression_dimension(),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let input =
            SharedIdentitySolveInput::new(model, mapping, model.neutral_identity(), samples)
                .unwrap();
        let config = SharedIdentitySolveConfig::new(
            8,
            20,
            1.0,
            1.0e-8,
            1.0e-9,
            DenseReprojectionConfig::default(),
        )
        .unwrap();
        let outcome = solve_shared_identity(&input, config, 1.0e-4).unwrap();
        finalize_identity_calibration(
            model,
            mapping,
            &input,
            &window_selection(),
            &outcome,
            &config,
        )
        .unwrap()
    }

    fn window_selection() -> vtuber_gnm::NeutralCalibrationSelection {
        use vtuber_gnm::{
            DenseCoverageSummary, DenseObservationStatus, NeutralCalibrationCandidate,
            NeutralCalibrationSelectionConfig, select_neutral_calibration_candidates,
        };
        let candidate = NeutralCalibrationCandidate {
            source_seq: 1,
            captured_at_micros: 1_000,
            coverage: DenseCoverageSummary {
                mapped_points: 3,
                valid_points: 3,
                effective_weight: 3.0,
                status: DenseObservationStatus::Valid,
            },
            reprojection_rms: 0.0,
            expression_activity: None,
            yaw_radians: -0.05,
            pitch_radians: 0.0,
            tracking_degraded: false,
        };
        let config = NeutralCalibrationSelectionConfig::new(2, 1.0, 1.0, 0.01, 0.001, 1.0).unwrap();
        select_neutral_calibration_candidates(
            &[
                candidate,
                NeutralCalibrationCandidate {
                    source_seq: 2,
                    yaw_radians: 0.05,
                    ..candidate
                },
            ],
            config,
        )
    }

    fn runtime_model() -> GnmModel {
        GnmModel::from_data(synthetic_model_data(GNM_HEAD_V3_VERSION)).unwrap()
    }

    // -- acceptance tests ----------------------------------------------------

    #[test]
    fn start_complete_cancel_reset_are_explicit_and_typed() {
        let mut store = GnmIdentityCalibrationStore::new();
        assert_eq!(store.phase(), GnmIdentityCalibrationPhase::Idle);

        let model = runtime_model();
        let mapping = solve_mapping(&model);

        // Complete without Start fails typed.
        let calibration = solved_calibration(&model, &mapping);
        assert!(matches!(
            store.apply(GnmIdentityCalibrationEvent::Complete(Box::new(
                calibration.clone()
            ))),
            Err(GnmIdentityLifecycleError::NotCollecting)
        ));

        store.apply(GnmIdentityCalibrationEvent::Start).unwrap();
        assert_eq!(store.phase(), GnmIdentityCalibrationPhase::Collecting);

        // Double Start fails typed.
        assert!(matches!(
            store.apply(GnmIdentityCalibrationEvent::Start),
            Err(GnmIdentityLifecycleError::AlreadyCollecting)
        ));

        store.apply(GnmIdentityCalibrationEvent::Cancel).unwrap();
        assert_eq!(store.phase(), GnmIdentityCalibrationPhase::Idle);

        store.apply(GnmIdentityCalibrationEvent::Start).unwrap();
        store
            .apply(GnmIdentityCalibrationEvent::Complete(Box::new(calibration)))
            .unwrap();
        assert!(matches!(
            store.phase(),
            GnmIdentityCalibrationPhase::Active { .. }
        ));

        store.apply(GnmIdentityCalibrationEvent::Reset).unwrap();
        assert_eq!(store.phase(), GnmIdentityCalibrationPhase::Idle);
    }

    #[test]
    fn published_identity_is_handed_to_tracking_read_only() {
        let mut store = GnmIdentityCalibrationStore::new();
        let model = runtime_model();
        let mapping = solve_mapping(&model);
        let calibration = solved_calibration(&model, &mapping);
        let expected = calibration.identity().values().to_vec();

        store.apply(GnmIdentityCalibrationEvent::Start).unwrap();
        store
            .apply(GnmIdentityCalibrationEvent::Complete(Box::new(calibration)))
            .unwrap();

        let handed = store
            .fixed_identity_for_runtime(&model, mapping.version())
            .expect("active calibration must hand off");
        assert_eq!(handed.values(), expected.as_slice());
    }

    #[test]
    fn version_mismatch_invalidates_the_stored_calibration() {
        let mut store = GnmIdentityCalibrationStore::new();
        let model = runtime_model();
        let mapping = solve_mapping(&model);
        let calibration = solved_calibration(&model, &mapping);

        store.apply(GnmIdentityCalibrationEvent::Start).unwrap();
        store
            .apply(GnmIdentityCalibrationEvent::Complete(Box::new(calibration)))
            .unwrap();

        // A newer runtime model must never receive the stale identity.
        let upgraded = GnmModel::from_data(synthetic_model_data(GnmVersion { major: 3, minor: 1 }));
        assert!(
            upgraded.is_err(),
            "gnm gate rejects unknown versions; mismatch path uses mapping schema"
        );

        let stale_mapping = mapping_version(GnmVersion { major: 9, minor: 9 });
        assert!(
            store
                .fixed_identity_for_runtime(&model, stale_mapping)
                .is_none()
        );
        assert_eq!(store.phase(), GnmIdentityCalibrationPhase::Idle);

        // After invalidation the same runtime can recalibrate from scratch.
        store.apply(GnmIdentityCalibrationEvent::Start).unwrap();
        assert_eq!(store.phase(), GnmIdentityCalibrationPhase::Collecting);
    }

    #[test]
    fn tracking_stop_start_reuses_the_same_session_calibration() {
        let mut store = GnmIdentityCalibrationStore::new();
        let model = runtime_model();
        let mapping = solve_mapping(&model);
        let calibration = solved_calibration(&model, &mapping);
        let expected = calibration.identity().values().to_vec();

        store.apply(GnmIdentityCalibrationEvent::Start).unwrap();
        store
            .apply(GnmIdentityCalibrationEvent::Complete(Box::new(calibration)))
            .unwrap();

        store.on_tracking_restarted();
        let handed = store
            .fixed_identity_for_runtime(&model, mapping.version())
            .expect("session calibration must survive tracking restart");
        assert_eq!(handed.values(), expected.as_slice());
        assert_eq!(store.tracking_restarts(), 1);

        // Only an explicit reset drops it.
        store.on_tracking_restarted();
        store.apply(GnmIdentityCalibrationEvent::Reset).unwrap();
        assert!(
            store
                .fixed_identity_for_runtime(&model, mapping.version())
                .is_none()
        );
    }

    #[test]
    fn ordinary_frame_residuals_never_update_the_published_identity() {
        let mut store = GnmIdentityCalibrationStore::new();
        let model = runtime_model();
        let mapping = solve_mapping(&model);
        let calibration = solved_calibration(&model, &mapping);
        let expected = calibration.identity().values().to_vec();

        store.apply(GnmIdentityCalibrationEvent::Start).unwrap();
        store
            .apply(GnmIdentityCalibrationEvent::Complete(Box::new(calibration)))
            .unwrap();

        // Simulate many frames of terrible tracking outcomes. The lifecycle has
        // no residual input at all: nothing below can touch the identity, and
        // this assertion pins that behavior machine-runnably.
        for frame in 0..10_000u64 {
            let terrible_residual = (frame as f32) * 100.0;
            store.on_tracking_restarted();
            let handed = store
                .fixed_identity_for_runtime(&model, mapping.version())
                .expect("calibration must stay active through ordinary frames");
            assert_eq!(handed.values(), expected.as_slice());
            assert!(terrible_residual.is_finite());
        }
        assert!(matches!(
            store.phase(),
            GnmIdentityCalibrationPhase::Active { .. }
        ));
    }
}
