//! Per-frame input, dynamic state, config, and result contracts for the
//! persistent GNM numerical fitter (Issue #55.5 / #90, parent #64).
//!
//! This module defines only the engine-neutral data contracts consumed by a
//! future bounded per-frame solver. It deliberately contains no optimization
//! loop, warm-start policy, temporal energy, worker, or ARKit decode.
//!
//! Identity is a read-only [`FixedGnmIdentity`] reference carried by the
//! per-frame input; it is never part of the mutable dynamic state, so no
//! solver path can silently recalibrate.

use crate::auxiliary_expression::{
    AuxiliaryExpressionError, AuxiliaryExpressionObservation, validate_auxiliary_source_alignment,
};
use vtuber_gnm::{
    DenseCorrespondenceSet, DenseObservationStatus, FixedGnmIdentity, GNM_HEAD_V3_VERSION,
    GnmDenseObservation, GnmExpressionState, GnmFrameStamp, GnmJointState, GnmModel,
};

/// Maximum solver iterations accepted by the bounded per-frame contract.
pub const MAX_SOLVER_ITERATIONS_BOUND: usize = 64;

/// Fail-closed validation failure for one per-frame fitter input.
#[derive(Clone, Debug, PartialEq)]
pub enum GnmFitterContractError {
    /// The frame stamp does not match the dense observation's source sequence.
    SourceSequenceMismatch {
        /// Stamp sequence.
        stamp: u64,
        /// Observation sequence.
        observation: u64,
    },
    /// The frame stamp does not match the dense observation capture timestamp.
    CaptureTimestampMismatch {
        /// Stamp timestamp.
        stamp: u64,
        /// Observation timestamp.
        observation: u64,
    },
    /// The loaded model generation is not the pinned Head v3 version.
    UnsupportedModelVersion {
        /// Model major/minor version.
        actual: (u16, u16),
    },
    /// The correspondence set was built for a different model or schema.
    MappingVersionMismatch {
        /// Version recorded by the mapping.
        mapping: (u16, u16),
        /// Loaded model version.
        model: (u16, u16),
    },
    /// The fixed identity dimension differs from the model identity dimension.
    IdentityDimensionMismatch {
        /// Model identity dimension.
        expected: usize,
        /// Fixed identity dimension.
        actual: usize,
    },
    /// The dynamic expression state has the wrong dimension.
    ExpressionDimensionMismatch {
        /// Model expression dimension.
        expected: usize,
        /// Dynamic state dimension.
        actual: usize,
    },
    /// The dynamic joint count differs from the model joint count.
    JointCountMismatch {
        /// Model joint count.
        expected: usize,
        /// Dynamic state joint count.
        actual: usize,
    },
    /// A dynamic value became non-finite.
    NonFiniteDynamicValue(&'static str),
    /// A configuration value is invalid.
    InvalidConfig(&'static str),
    /// The auxiliary observation rejected its own validation.
    Auxiliary(AuxiliaryExpressionError),
}

impl std::fmt::Display for GnmFitterContractError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SourceSequenceMismatch { stamp, observation } => write!(
                formatter,
                "frame stamp source_seq {stamp} does not match observation {observation}"
            ),
            Self::CaptureTimestampMismatch { stamp, observation } => write!(
                formatter,
                "frame stamp timestamp {stamp} does not match observation {observation}"
            ),
            Self::UnsupportedModelVersion { actual } => write!(
                formatter,
                "model version {:?}.{:?} is not the supported Head v3",
                actual.0, actual.1
            ),
            Self::MappingVersionMismatch { mapping, model } => write!(
                formatter,
                "mapping version {:?}.{:?} does not match model version {:?}.{:?}",
                mapping.0, mapping.1, model.0, model.1
            ),
            Self::IdentityDimensionMismatch { expected, actual } => write!(
                formatter,
                "fixed identity has {actual} coefficients but the model expects {expected}"
            ),
            Self::ExpressionDimensionMismatch { expected, actual } => write!(
                formatter,
                "dynamic expression has {actual} channels but the model expects {expected}"
            ),
            Self::JointCountMismatch { expected, actual } => write!(
                formatter,
                "dynamic state has {actual} joints but the model expects {expected}"
            ),
            Self::NonFiniteDynamicValue(field) => {
                write!(
                    formatter,
                    "dynamic state `{field}` contains a non-finite value"
                )
            }
            Self::InvalidConfig(reason) => {
                write!(formatter, "invalid solver config: {reason}")
            }
            Self::Auxiliary(error) => write!(formatter, "auxiliary observation: {error}"),
        }
    }
}

impl std::error::Error for GnmFitterContractError {}

/// Rigid head pose block of the dynamic state (yaw/pitch/roll radians).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GnmRigidPoseBlock {
    yaw_pitch_roll: [f32; 3],
}

impl GnmRigidPoseBlock {
    /// Creates a validated rigid pose block.
    ///
    /// # Errors
    ///
    /// Returns [`GnmFitterContractError::NonFiniteDynamicValue`] for non-finite angles.
    pub fn new(yaw_pitch_roll: [f32; 3]) -> Result<Self, GnmFitterContractError> {
        if yaw_pitch_roll.iter().any(|value| !value.is_finite()) {
            return Err(GnmFitterContractError::NonFiniteDynamicValue(
                "rigid_yaw_pitch_roll",
            ));
        }
        Ok(Self { yaw_pitch_roll })
    }

    /// Returns the yaw/pitch/roll Euler angles in radians.
    pub fn yaw_pitch_roll(&self) -> [f32; 3] {
        self.yaw_pitch_roll
    }
}

/// Translation/camera block of the dynamic state.
///
/// Translation is head-space to camera-space; focal and principal point use
/// the canonical normalized image space conventions of the reprojection
/// objective. The principal point is held fixed during solving and is carried
/// here only as validated evidence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GnmCameraBlock {
    translation: [f32; 3],
    focal: f32,
    principal_point: [f32; 2],
}

impl GnmCameraBlock {
    /// Creates a validated camera block.
    ///
    /// # Errors
    ///
    /// Returns [`GnmFitterContractError::NonFiniteDynamicValue`] for any
    /// non-finite component and [`GnmFitterContractError::InvalidConfig`]
    /// for a non-positive focal length.
    pub fn new(
        translation: [f32; 3],
        focal: f32,
        principal_point: [f32; 2],
    ) -> Result<Self, GnmFitterContractError> {
        if translation
            .iter()
            .chain(principal_point.iter())
            .any(|value| !value.is_finite())
        {
            return Err(GnmFitterContractError::NonFiniteDynamicValue(
                "camera_block",
            ));
        }
        if !focal.is_finite() || focal <= 0.0 {
            return Err(GnmFitterContractError::InvalidConfig(
                "camera focal length must be finite and positive",
            ));
        }
        Ok(Self {
            translation,
            focal,
            principal_point,
        })
    }

    /// Returns the head-space to camera-space translation.
    pub fn translation(&self) -> [f32; 3] {
        self.translation
    }

    /// Returns the focal length.
    pub fn focal(&self) -> f32 {
        self.focal
    }

    /// Returns the fixed principal point.
    pub fn principal_point(&self) -> [f32; 2] {
        self.principal_point
    }
}

/// Mutable dynamic state optimized across frames.
///
/// Identity is intentionally absent: it lives in the read-only
/// [`FixedGnmIdentity`] on the per-frame input instead.
#[derive(Clone, Debug, PartialEq)]
pub struct GnmDynamicState {
    /// Expression coefficients.
    pub expression: GnmExpressionState,
    /// Joint rotations and global translation.
    pub joints: GnmJointState,
    /// Rigid head pose block.
    pub rigid_pose: GnmRigidPoseBlock,
    /// Translation/camera block.
    pub camera: GnmCameraBlock,
}

impl GnmDynamicState {
    /// Validates every dynamic block against the loaded model dimensions and
    /// finiteness requirements.
    ///
    /// # Errors
    ///
    /// Returns the typed dimension/finiteness failure of the first invalid block.
    pub fn validate(&self, model: &GnmModel) -> Result<(), GnmFitterContractError> {
        if self.expression.values().len() != model.expression_dimension() {
            return Err(GnmFitterContractError::ExpressionDimensionMismatch {
                expected: model.expression_dimension(),
                actual: self.expression.values().len(),
            });
        }
        if self.joints.rotations().len() != model.joint_count() {
            return Err(GnmFitterContractError::JointCountMismatch {
                expected: model.joint_count(),
                actual: self.joints.rotations().len(),
            });
        }
        if self
            .expression
            .values()
            .iter()
            .chain(self.joints.rotations().iter().flatten())
            .chain(self.joints.translation().iter())
            .any(|value| !value.is_finite())
        {
            return Err(GnmFitterContractError::NonFiniteDynamicValue(
                "expression_or_joints",
            ));
        }
        Ok(())
    }
}

/// Bounded per-frame solver configuration with explicit objective weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GnmSolverConfig {
    /// Maximum iterations for one bounded solve. Strictly positive and capped.
    pub max_iterations: usize,
    /// Weight of the dense reprojection objective. Must be finite and positive.
    pub dense_weight: f32,
    /// Absolute weight of the optional auxiliary term. `0` disables it.
    pub auxiliary_weight: f32,
    /// Non-negative expression smoothness/regularization weight.
    pub expression_regularization: f32,
    /// Non-negative joint regularization weight.
    pub joint_regularization: f32,
    /// Non-negative rigid-pose regularization weight.
    pub pose_regularization: f32,
    /// Non-negative camera-block regularization weight.
    pub camera_regularization: f32,
}

impl GnmSolverConfig {
    /// Creates a validated bounded solver configuration.
    ///
    /// # Errors
    ///
    /// Returns [`GnmFitterContractError::InvalidConfig`] when the iteration
    /// bound or any weight violates the contract.
    pub fn new(
        max_iterations: usize,
        dense_weight: f32,
        auxiliary_weight: f32,
        expression_regularization: f32,
        joint_regularization: f32,
        pose_regularization: f32,
        camera_regularization: f32,
    ) -> Result<Self, GnmFitterContractError> {
        if max_iterations == 0 || max_iterations > MAX_SOLVER_ITERATIONS_BOUND {
            return Err(GnmFitterContractError::InvalidConfig(
                "max_iterations must be within 1..=MAX_SOLVER_ITERATIONS_BOUND",
            ));
        }
        let check = |value: f32| -> Result<(), GnmFitterContractError> {
            if !value.is_finite() || value < 0.0 {
                return Err(GnmFitterContractError::InvalidConfig(
                    "weights must be finite and non-negative",
                ));
            }
            Ok(())
        };
        if !dense_weight.is_finite() || dense_weight <= 0.0 {
            return Err(GnmFitterContractError::InvalidConfig(
                "dense weight must be finite and positive",
            ));
        }
        check(auxiliary_weight)?;
        check(expression_regularization)?;
        check(joint_regularization)?;
        check(pose_regularization)?;
        check(camera_regularization)?;
        Ok(Self {
            max_iterations,
            dense_weight,
            auxiliary_weight,
            expression_regularization,
            joint_regularization,
            pose_regularization,
            camera_regularization,
        })
    }

    /// Returns whether the auxiliary term contributes to this configuration.
    pub fn auxiliary_enabled(&self) -> bool {
        self.auxiliary_weight > 0.0
    }
}

/// Engine-neutral input for exactly one bounded per-frame solve.
#[derive(Clone, Debug)]
pub struct GnmSolverFrameInput<'a> {
    stamp: GnmFrameStamp,
    observation: &'a GnmDenseObservation,
    auxiliary: Option<&'a AuxiliaryExpressionObservation>,
    identity: &'a FixedGnmIdentity,
}

impl<'a> GnmSolverFrameInput<'a> {
    /// Assembles and validates one per-frame input against the loaded model
    /// and correspondence set.
    ///
    /// Validation is fail-closed: exact stamp/observation alignment, supported
    /// model generation, mapping/model version agreement, and fixed-identity
    /// dimension are all checked here so a solver cannot receive an
    /// inconsistent frame.
    ///
    /// # Errors
    ///
    /// Returns the first typed mismatch as [`GnmFitterContractError`].
    pub fn new(
        stamp: GnmFrameStamp,
        observation: &'a GnmDenseObservation,
        auxiliary: Option<&'a AuxiliaryExpressionObservation>,
        identity: &'a FixedGnmIdentity,
        model: &GnmModel,
        mapping: &DenseCorrespondenceSet,
    ) -> Result<Self, GnmFitterContractError> {
        if observation.source_seq() != stamp.source_seq {
            return Err(GnmFitterContractError::SourceSequenceMismatch {
                stamp: stamp.source_seq,
                observation: observation.source_seq(),
            });
        }
        if observation.captured_at_micros() != stamp.captured_at_micros {
            return Err(GnmFitterContractError::CaptureTimestampMismatch {
                stamp: stamp.captured_at_micros,
                observation: observation.captured_at_micros(),
            });
        }
        if model.version() != GNM_HEAD_V3_VERSION {
            return Err(GnmFitterContractError::UnsupportedModelVersion {
                actual: (model.version().major, model.version().minor),
            });
        }
        let mapping_version = mapping.version();
        if mapping_version.model_version != model.version() {
            return Err(GnmFitterContractError::MappingVersionMismatch {
                mapping: (
                    mapping_version.model_version.major,
                    mapping_version.model_version.minor,
                ),
                model: (model.version().major, model.version().minor),
            });
        }
        if identity.values().len() != model.identity_dimension() {
            return Err(GnmFitterContractError::IdentityDimensionMismatch {
                expected: model.identity_dimension(),
                actual: identity.values().len(),
            });
        }
        if let Some(auxiliary) = auxiliary {
            validate_auxiliary_source_alignment(
                stamp.source_seq,
                stamp.captured_at_micros,
                auxiliary,
            )
            .map_err(GnmFitterContractError::Auxiliary)?;
        }
        Ok(Self {
            stamp,
            observation,
            auxiliary,
            identity,
        })
    }

    /// Returns the exact frame stamp.
    pub fn stamp(&self) -> GnmFrameStamp {
        self.stamp
    }

    /// Returns the primary dense observation.
    pub fn observation(&self) -> &GnmDenseObservation {
        self.observation
    }

    /// Returns the optional auxiliary observation.
    pub fn auxiliary(&self) -> Option<&AuxiliaryExpressionObservation> {
        self.auxiliary
    }

    /// Returns the read-only calibrated identity. Never part of dynamic state.
    pub fn identity(&self) -> &FixedGnmIdentity {
        self.identity
    }

    /// Returns whether the dense observation carries enough valid points to
    /// attempt a solve at all.
    pub fn usable_observation(&self) -> bool {
        matches!(
            self.observation.coverage().status,
            DenseObservationStatus::Valid | DenseObservationStatus::Degraded
        )
    }
}

/// Classification of one completed bounded solve.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GnmFitStatus {
    /// Residual met the validity checks before the iteration bound.
    Converged,
    /// Iteration budget exhausted without meeting convergence evidence.
    MaxIterationsReached,
    /// Final residual exceeded the configured validity bound.
    ResidualAboveBound,
    /// Residual accumulation became non-finite during the solve.
    NonFiniteResidual,
}

/// Result of one bounded per-frame solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GnmFitResult {
    residual: f32,
    iterations: usize,
    status: GnmFitStatus,
}

impl GnmFitResult {
    /// Creates a result after checking internal consistency: the residual must
    /// be finite exactly when the status says so.
    ///
    /// # Errors
    ///
    /// Returns [`GnmFitterContractError::NonFiniteDynamicValue`] when a finite
    /// status carries a non-finite residual or vice versa.
    pub fn new(
        residual: f32,
        iterations: usize,
        status: GnmFitStatus,
    ) -> Result<Self, GnmFitterContractError> {
        let requires_finite = matches!(
            status,
            GnmFitStatus::Converged | GnmFitStatus::ResidualAboveBound
        );
        if residual.is_finite() != requires_finite || iterations == 0 {
            if iterations == 0 {
                return Err(GnmFitterContractError::InvalidConfig(
                    "a completed solve must report at least one iteration",
                ));
            }
            return Err(GnmFitterContractError::NonFiniteDynamicValue(
                "fit_residual",
            ));
        }
        Ok(Self {
            residual,
            iterations,
            status,
        })
    }

    /// Returns the final scalar objective residual when finite.
    pub fn residual(&self) -> f32 {
        self.residual
    }

    /// Returns the iteration count actually spent.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Returns the completion classification.
    pub fn status(&self) -> GnmFitStatus {
        self.status
    }

    /// Returns whether this result may become the next valid authority state
    /// (lifecycle `GnmFitOutcome::Valid` equivalent).
    pub fn valid(&self) -> bool {
        self.status == GnmFitStatus::Converged
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auxiliary_expression::{
        AuxChannelReliability::TrustedForAux, AuxiliaryChannelConfig, AuxiliaryExpressionSemantic,
    };
    use vtuber_core::{FaceBlendshapeSet, MediaPipeBlendshape};
    use vtuber_gnm::{
        AnatomicalSide, CorrespondenceProvenance, CorrespondenceReliability, DenseCoveragePolicy,
        DenseMappingVersion, FaceRegion, GNM_HEAD_V3_EXPRESSION_DIM, GNM_HEAD_V3_IDENTITY_DIM,
        GnmModelData, GnmSurfacePointRef, GnmVariant, MEDIAPIPE_FACE_LANDMARK_COUNT,
        MediaPipeGnmDenseCorrespondence,
    };

    const STAMP_SEQ: u64 = 42;
    const STAMP_MICROS: u64 = 100_000;

    fn stamp() -> GnmFrameStamp {
        GnmFrameStamp {
            source_seq: STAMP_SEQ,
            captured_at_micros: STAMP_MICROS,
        }
    }

    fn simple_model() -> GnmModel {
        let vertex_count = MEDIAPIPE_FACE_LANDMARK_COUNT;
        GnmModel::from_data(GnmModelData {
            version: GNM_HEAD_V3_VERSION,
            variant: GnmVariant::Head,
            template_vertices: vtuber_gnm::DenseArray::new(
                "vertices",
                vec![vertex_count, 3],
                vec![0.0; vertex_count * 3],
            )
            .unwrap(),
            template_joints: vtuber_gnm::DenseArray::new("joints", vec![1, 3], vec![0.0; 3])
                .unwrap(),
            vertex_identity_basis: vtuber_gnm::DenseArray::new(
                "identity",
                vec![GNM_HEAD_V3_IDENTITY_DIM, vertex_count, 3],
                vec![0.0; GNM_HEAD_V3_IDENTITY_DIM * vertex_count * 3],
            )
            .unwrap(),
            joint_identity_basis: vtuber_gnm::DenseArray::new(
                "joint_identity",
                vec![GNM_HEAD_V3_IDENTITY_DIM, 1, 3],
                vec![0.0; GNM_HEAD_V3_IDENTITY_DIM * 3],
            )
            .unwrap(),
            expression_basis: vtuber_gnm::DenseArray::new(
                "expression",
                vec![GNM_HEAD_V3_EXPRESSION_DIM, vertex_count, 3],
                vec![0.0; GNM_HEAD_V3_EXPRESSION_DIM * vertex_count * 3],
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

    fn full_mapping(model: &GnmModel) -> DenseCorrespondenceSet {
        let rows: Vec<MediaPipeGnmDenseCorrespondence> = (0..MEDIAPIPE_FACE_LANDMARK_COUNT)
            .map(|mp| MediaPipeGnmDenseCorrespondence {
                mediapipe_index: mp,
                target: GnmSurfacePointRef::Vertex { vertex_index: mp },
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

    fn landmarks() -> [[f32; 2]; MEDIAPIPE_FACE_LANDMARK_COUNT] {
        [[0.5; 2]; MEDIAPIPE_FACE_LANDMARK_COUNT]
    }

    fn observation(mapping: &DenseCorrespondenceSet) -> GnmDenseObservation {
        GnmDenseObservation::from_mediapipe_xy(
            STAMP_SEQ,
            STAMP_MICROS,
            &landmarks(),
            mapping,
            DenseCoveragePolicy::new(2, 0.75).unwrap(),
        )
        .unwrap()
    }

    fn identity(model: &GnmModel) -> FixedGnmIdentity {
        FixedGnmIdentity::new(model.neutral_identity(), model).unwrap()
    }

    fn neutral_dynamic_state(model: &GnmModel) -> GnmDynamicState {
        GnmDynamicState {
            expression: model.neutral_expression(),
            joints: GnmJointState::neutral(model.joint_count()),
            rigid_pose: GnmRigidPoseBlock::new([0.0; 3]).unwrap(),
            camera: GnmCameraBlock::new([0.0, 0.0, 5.0], 1.0, [0.5, 0.5]).unwrap(),
        }
    }

    fn default_config() -> GnmSolverConfig {
        GnmSolverConfig::new(8, 1.0, 0.0, 0.01, 0.01, 0.01, 0.01).unwrap()
    }

    fn auxiliary_observation(seq: u64, micros: u64) -> AuxiliaryExpressionObservation {
        let pairs: Vec<(&str, f32)> = MediaPipeBlendshape::ALL
            .into_iter()
            .map(|category| (category.as_str(), 0.0))
            .collect();
        let scores = FaceBlendshapeSet::from_pairs(&pairs).unwrap();
        let config = AuxiliaryChannelConfig::new(
            AuxiliaryExpressionSemantic::JawOpen,
            TrustedForAux,
            1.0,
            None,
        )
        .unwrap();
        AuxiliaryExpressionObservation::from_mediapipe(seq, micros, &scores, &[config]).unwrap()
    }

    #[test]
    fn valid_input_is_accepted_and_exposes_read_only_identity() {
        let model = simple_model();
        let mapping = full_mapping(&model);
        let obs = observation(&mapping);
        let fixed = identity(&model);
        let aux = auxiliary_observation(STAMP_SEQ, STAMP_MICROS);

        let input =
            GnmSolverFrameInput::new(stamp(), &obs, Some(&aux), &fixed, &model, &mapping).unwrap();

        assert_eq!(input.stamp(), stamp());
        assert_eq!(
            input.auxiliary().map(|aux| aux.source_seq()),
            Some(STAMP_SEQ)
        );
        assert_eq!(input.identity().values().len(), model.identity_dimension());
        assert!(input.usable_observation());
    }

    #[test]
    fn stamp_observation_alignment_is_fail_closed() {
        let model = simple_model();
        let mapping = full_mapping(&model);
        let obs = observation(&mapping);
        let fixed = identity(&model);

        let wrong_seq = GnmFrameStamp {
            source_seq: STAMP_SEQ + 1,
            ..stamp()
        };
        assert!(matches!(
            GnmSolverFrameInput::new(wrong_seq, &obs, None, &fixed, &model, &mapping),
            Err(GnmFitterContractError::SourceSequenceMismatch { .. })
        ));

        let wrong_micros = GnmFrameStamp {
            captured_at_micros: STAMP_MICROS + 1,
            ..stamp()
        };
        assert!(matches!(
            GnmSolverFrameInput::new(wrong_micros, &obs, None, &fixed, &model, &mapping),
            Err(GnmFitterContractError::CaptureTimestampMismatch { .. })
        ));
    }

    #[test]
    fn auxiliary_alignment_is_enforced_against_the_dense_frame() {
        let model = simple_model();
        let mapping = full_mapping(&model);
        let obs = observation(&mapping);
        let fixed = identity(&model);
        let aux = auxiliary_observation(STAMP_SEQ + 5, STAMP_MICROS);

        assert!(matches!(
            GnmSolverFrameInput::new(stamp(), &obs, Some(&aux), &fixed, &model, &mapping),
            Err(GnmFitterContractError::Auxiliary(
                AuxiliaryExpressionError::SourceSequenceMismatch { .. }
            ))
        ));
    }

    #[test]
    fn dynamic_state_dimensions_are_validated_against_the_model() {
        let model = simple_model();

        let mut good = neutral_dynamic_state(&model);
        assert!(good.validate(&model).is_ok());

        good.expression = GnmExpressionState::new(
            vec![0.0; GNM_HEAD_V3_EXPRESSION_DIM - 1],
            GNM_HEAD_V3_EXPRESSION_DIM - 1,
        )
        .unwrap();
        assert!(matches!(
            good.validate(&model),
            Err(GnmFitterContractError::ExpressionDimensionMismatch { .. })
        ));

        let mut bad_joints = neutral_dynamic_state(&model);
        bad_joints.joints = GnmJointState::neutral(model.joint_count() + 1);
        assert!(matches!(
            bad_joints.validate(&model),
            Err(GnmFitterContractError::JointCountMismatch { .. })
        ));

        assert!(matches!(
            GnmRigidPoseBlock::new([f32::NAN, 0.0, 0.0]),
            Err(GnmFitterContractError::NonFiniteDynamicValue(
                "rigid_yaw_pitch_roll"
            ))
        ));
    }

    #[test]
    fn config_rejects_bad_bounds_and_weights() {
        assert!(matches!(
            GnmSolverConfig::new(0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Err(GnmFitterContractError::InvalidConfig(_))
        ));
        assert!(matches!(
            GnmSolverConfig::new(
                MAX_SOLVER_ITERATIONS_BOUND + 1,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0
            ),
            Err(GnmFitterContractError::InvalidConfig(_))
        ));
        assert!(matches!(
            GnmSolverConfig::new(4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Err(GnmFitterContractError::InvalidConfig(_))
        ));
        assert!(matches!(
            GnmSolverConfig::new(4, 1.0, -0.5, 0.0, 0.0, 0.0, 0.0),
            Err(GnmFitterContractError::InvalidConfig(_))
        ));
        let ok = GnmSolverConfig::new(4, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0).unwrap();
        assert!(ok.auxiliary_enabled());
        assert!(!default_config().auxiliary_enabled());
    }

    #[test]
    fn fit_result_validity_follows_status_and_finiteness() {
        let converged = GnmFitResult::new(0.25, 3, GnmFitStatus::Converged).unwrap();
        assert!(converged.valid());
        assert_eq!(converged.iterations(), 3);

        let above_bound = GnmFitResult::new(
            10.0,
            MAX_SOLVER_ITERATIONS_BOUND,
            GnmFitStatus::ResidualAboveBound,
        )
        .unwrap();
        assert!(!above_bound.valid());

        assert!(GnmFitResult::new(f32::INFINITY, 1, GnmFitStatus::NonFiniteResidual).is_ok());
        assert!(GnmFitResult::new(f32::NAN, 1, GnmFitStatus::Converged).is_err());
        assert!(GnmFitResult::new(1.0, 0, GnmFitStatus::Converged).is_err());
    }
}
