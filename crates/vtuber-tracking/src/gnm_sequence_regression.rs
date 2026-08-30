//! Deterministic synthetic-sequence regression harness for the persistent
//! GNM fitter (Issue #64.5 / #94, parent #64).
//!
//! The harness synthesizes dense observations along known GNM trajectories
//! and drives the same lifecycle/bounded-solve building blocks used by the
//! persistent tracker under four configurations on the identical sequence:
//!
//! - [`SequenceFitMode::Cold`]: every solve starts from neutral dynamics plus
//!   rigid recovery against the current observation; no cross-frame reuse.
//! - [`SequenceFitMode::WarmStart`]: the lifecycle directive decides, so a
//!   valid previous state seeds the optimizer.
//! - [`SequenceFitMode::FixedTemporal`]: warm start plus a fixed bounded
//!   temporal penalty connected through `fit_single_frame_with_temporal`.
//! - [`SequenceFitMode::AdaptiveTemporal`]: warm start plus temporal weights
//!   produced per frame by the adaptive strength policy mapped onto explicit
//!   lambda ranges.
//!
//! Everything is pure synchronous f32/f64 math with no randomness and no
//! clock access, so the same scenario and mode always produce an equal
//! [`SequenceRunReport`]. Solver cost is intentionally bounded through a
//! reduced iteration budget; see issue #148 for the underlying solver cost
//! reduction work.
//!
//! Wall-clock reference (issue #148): the ten in-crate regression tests
//! target a combined runtime under ~3 seconds in the default dev profile on
//! the reference Windows workstation. Measured after the packed
//! lower-triangular solver and the dev-profile optimization of `vtuber-gnm`:
//! ~1.0 second (was ~10.8 seconds before issue #148). This is a measurement
//! guideline, not an asserted bound; wall-clock assertions would be flaky
//! across machines.

use std::fmt;

use vtuber_gnm::{
    AnatomicalSide, CorrespondenceProvenance, CorrespondenceReliability, DenseCorrespondenceSet,
    DenseCoveragePolicy, DenseMappingVersion, DenseObservationStatus, DenseProjection, FaceRegion,
    FixedGnmIdentity, GnmDenseObservation, GnmExpressionState, GnmFitInitialization, GnmFitOutcome,
    GnmFrameStamp, GnmJointState, GnmModel, GnmModelError, GnmReprojectionError, GnmSparseVertices,
    GnmSurfacePointRef, GnmTemporalNormalization, GnmTemporalStateView,
    MediaPipeGnmDenseCorrespondence, PersistentGnmAction, PersistentGnmEvent,
    PersistentGnmLifecycleConfig, PersistentGnmLifecycleError, PersistentGnmLifecycleState,
    RigidRecoveryConfig, SingleFrameFitConfig, SingleFrameTemporalPenalty, SynthesisOptions,
    TemporalHistoryTiming, TemporalRegularizationConfig, TemporalRegularizationError,
    advance_persistent_gnm_lifecycle, fit_single_frame_with_temporal, fitting_projection,
    recover_rigid_projection,
};

use crate::adaptive_temporal::{
    AdaptiveTemporalConfig, AdaptiveTemporalError, AdaptiveTemporalState, GroupLambdaRange,
    TemporalGroupWeights, TemporalLambdaRanges, TemporalObservationHealth,
    advance_adaptive_temporal_policy, map_strengths_to_temporal_config,
};
use crate::gnm_fitter_contract::{GnmCameraBlock, GnmDynamicState, GnmRigidPoseBlock};

/// Known synthetic trajectory driven by the harness.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SequenceTrajectory {
    /// Constant expression and constant pose.
    Static,
    /// Expression ramps slowly from neutral to a strong target.
    SlowRamp,
    /// Expression holds a strong target and releases quickly.
    FastRelease,
    /// Low baseline expression with one short strong pulse (blink-like).
    BlinkPulse,
    /// Slow yaw ramp combined with an expression ramp.
    HeadPoseAndExpression,
}

/// Solver configuration under which one scenario run executes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SequenceFitMode {
    /// Neutral initialization every frame; no cross-frame reuse.
    Cold,
    /// Lifecycle-directed warm start from the previous valid state.
    WarmStart,
    /// Warm start plus a fixed bounded temporal penalty.
    FixedTemporal,
    /// Warm start plus adaptively weighted temporal penalties.
    AdaptiveTemporal,
}

/// Anomalies injected into a scenario timeline.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SequenceAnomalies {
    /// Frame index receiving a geometrically displaced (outlier) observation.
    pub outlier_frame: Option<usize>,
    /// First frame of an observation dropout, if any.
    pub dropout_start_frame: Option<usize>,
    /// Number of consecutive dropped frames starting at the dropout start.
    pub dropout_length: usize,
    /// Extra timeline gap in microseconds inserted once after the last
    /// dropout frame. Values above the lifecycle reuse bound force a
    /// reacquisition and a temporal history reset.
    pub extra_gap_after_dropout_micros: u64,
}

/// Full specification of one synthetic sequence run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequenceScenario {
    /// Trajectory driving the ground-truth states.
    pub trajectory: SequenceTrajectory,
    /// Number of source frames in the sequence.
    pub frame_count: usize,
    /// Nominal source frame rate in hertz; inter-frame time is `1 / fps`.
    pub fps: f64,
    /// Injected anomalies.
    pub anomalies: SequenceAnomalies,
}

impl SequenceScenario {
    /// Creates a clean (anomaly-free) scenario.
    ///
    /// # Errors
    ///
    /// Returns [`SequenceRegressionError::InvalidScenario`] for a frame count
    /// below the minimum required by the latency metrics or a non-positive or
    /// non-finite frame rate. The frame rate must also stay above
    /// `1 / MAX_TEMPORAL_DT_SECONDS` so the adaptive policy accepts the
    /// nominal inter-frame delta.
    pub fn new(
        trajectory: SequenceTrajectory,
        frame_count: usize,
        fps: f64,
    ) -> Result<Self, SequenceRegressionError> {
        if frame_count < 12 {
            return Err(SequenceRegressionError::InvalidScenario(format!(
                "frame_count {frame_count} must be at least 12"
            )));
        }
        if !fps.is_finite() || fps < 4.0 {
            return Err(SequenceRegressionError::InvalidScenario(format!(
                "fps {fps} must be finite and at least 4"
            )));
        }
        Ok(Self {
            trajectory,
            frame_count,
            fps,
            anomalies: SequenceAnomalies::default(),
        })
    }
}

/// One tracked (published) frame retained for report computation.
#[derive(Clone, Copy, Debug, PartialEq)]
struct TrackedSample {
    source_seq: u64,
    iterations: usize,
    expression_channel0: f32,
    yaw_pitch_roll: [f32; 3],
    truth_expression_channel0: f32,
    truth_yaw_pitch_roll: [f32; 3],
}

/// Aggregated result of one deterministic scenario run.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceRunReport {
    /// Trajectory that was driven.
    pub trajectory: SequenceTrajectory,
    /// Configuration the run executed under.
    pub mode: SequenceFitMode,
    /// Frames admitted with a usable observation and solved.
    pub solved_frames: usize,
    /// Frames whose observation was unusable (dropout path).
    pub no_observation_frames: usize,
    /// Sum of block-coordinate iterations over all solved frames.
    pub total_iterations: u64,
    /// Mean iterations per solved frame.
    pub mean_iterations: f64,
    /// Mean absolute channel-0 expression error versus truth.
    pub mean_expression_error: f64,
    /// Absolute channel-0 expression error on the final solved frame.
    pub final_expression_error: f64,
    /// Mean Euler-angle L2 error (radians) versus truth.
    pub mean_pose_error_rad: f64,
    /// RMS of the fitted channel-0 time derivative in units per second;
    /// lower means smoother tracking.
    pub expression_jitter_per_second: f64,
    /// Frames between the truth and the fitted signal crossing the ramp
    /// threshold (`SlowRamp` only).
    pub onset_latency_frames: Option<f64>,
    /// Frames between the truth and the fitted signal crossing the release
    /// threshold (`FastRelease` only).
    pub release_latency_frames: Option<f64>,
    /// Maximum fitted channel-0 value (`BlinkPulse` only).
    pub pulse_peak_expression: Option<f64>,
    /// Temporal history resets forced by long gaps.
    pub temporal_history_resets: usize,
}

/// Typed failure of one scenario run.
#[derive(Debug)]
pub enum SequenceRegressionError {
    /// Scenario parameters violated the harness contract.
    InvalidScenario(String),
    /// The pinned GNM model schema rejected the synthetic fixture.
    Model(GnmModelError),
    /// Observation synthesis or contract validation failed.
    Synthesis(GnmReprojectionError),
    /// The lifecycle rejected an event; indicates harness/lifecycle drift.
    Lifecycle(PersistentGnmLifecycleError),
    /// The bounded solve failed.
    Solve(GnmReprojectionError),
    /// The adaptive temporal policy rejected its configuration or input.
    AdaptivePolicy(AdaptiveTemporalError),
    /// A lifecycle directive referenced state the harness does not hold,
    /// which means harness bookkeeping drifted from the lifecycle.
    Bookkeeping {
        /// Source frame being processed when the drift appeared.
        stamp: GnmFrameStamp,
    },
}

impl fmt::Display for SequenceRegressionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidScenario(reason) => write!(formatter, "invalid scenario: {reason}"),
            Self::Model(error) => write!(formatter, "synthetic model: {error}"),
            Self::Synthesis(error) => write!(formatter, "observation synthesis: {error}"),
            Self::Lifecycle(error) => write!(formatter, "lifecycle: {error}"),
            Self::Solve(error) => write!(formatter, "bounded solve: {error}"),
            Self::AdaptivePolicy(error) => write!(formatter, "adaptive policy: {error}"),
            Self::Bookkeeping { stamp } => write!(
                formatter,
                "lifecycle directed warm start from source {} but the harness holds no matching state",
                stamp.source_seq
            ),
        }
    }
}

impl std::error::Error for SequenceRegressionError {}

/// Bounded solver budget shared by every regression run.
///
/// Iterations are themselves a reported comparison metric, so the budget is a
/// fixed constant well below the production bound.
// Invariant: the nested defaults and the constants below are validated
// compile-time values; the constructor can only reject changed schema bounds.
#[allow(clippy::expect_used)]
fn regression_solver_config() -> SingleFrameFitConfig {
    SingleFrameFitConfig::new(
        vtuber_gnm::DenseRigidStepConfig::default(),
        vtuber_gnm::DenseExpressionJointStepConfig::default(),
        16,
        1.0e-6,
    )
    .expect("regression solver config is within validated bounds")
}

/// Lifecycle configuration shared by all runs: three-frame warm-start window
/// at 60 fps and a 250 ms dynamic reuse age.
// Invariant: compile-time constants within the lifecycle validator bounds.
#[allow(clippy::expect_used)]
fn regression_lifecycle_config() -> PersistentGnmLifecycleConfig {
    PersistentGnmLifecycleConfig::new(50_000, 250_000, 3)
        .expect("regression lifecycle config is within validated bounds")
}

/// Explicit bounded lambda ranges for the regression temporal penalties.
// Invariant: compile-time constants within the range validator bounds.
#[allow(clippy::expect_used)]
fn temporal_lambda_ranges() -> TemporalLambdaRanges {
    // The quadratic temporal curvature added to the normal equations is
    // `2 * lambda / (scale^2 * dt^2)`. The fixture's dense curvature is
    // O(0.1) for expression channels and O(3) for rigid yaw, so each group
    // gets a band whose top stays below its own dense curvature.
    // Invariant: compile-time constants within the validator bounds.
    #[allow(clippy::expect_used)]
    let soft_group = GroupLambdaRange::new(1.0e-6, 1.0e-4, 1.0e-8, 1.0e-6)
        .expect("expression lambda range is within validated bounds");
    // Invariant: compile-time constants within the validator bounds.
    #[allow(clippy::expect_used)]
    let rigid_group = GroupLambdaRange::new(1.0e-8, 1.0e-6, 1.0e-10, 1.0e-8)
        .expect("rigid lambda range is within validated bounds");
    TemporalLambdaRanges {
        expression: soft_group,
        joints: soft_group,
        head_pose: rigid_group,
        translation: rigid_group,
    }
}

/// Constant mid-range strengths used by [`SequenceFitMode::FixedTemporal`].
// Invariant: compile-time constants within `[0, 1]`.
#[allow(clippy::expect_used)]
fn fixed_strengths() -> TemporalGroupWeights {
    TemporalGroupWeights::new(0.5, 0.5, 0.5, 0.5)
        .expect("fixed regression strengths are valid constants")
}

/// Adaptive policy configuration for [`SequenceFitMode::AdaptiveTemporal`].
// Invariant: compile-time constants within the policy validator bounds.
#[allow(clippy::expect_used)]
fn adaptive_policy_config() -> AdaptiveTemporalConfig {
    let strong =
        TemporalGroupWeights::new(0.9, 0.9, 0.9, 0.9).expect("policy weight constants are valid");
    let active =
        TemporalGroupWeights::new(0.15, 0.3, 0.3, 0.3).expect("policy weight constants are valid");
    AdaptiveTemporalConfig::new(
        strong, active, strong, 0.05, // expression motion start (normalized units per second)
        0.60, // expression motion full
        0.02, // rigid motion start (normalized units per second)
        0.50, // rigid motion full
        0.7,  // quality degraded start
        0.3,  // quality degraded full (below start)
        3.0,  // strengthen per second
        6.0,  // weaken per second
        0.25, // max dt seconds
    )
    .expect("adaptive policy constants are valid")
}

/// Builds the small deterministic head model used by the harness (same
/// construction as the solver/persistent-fitter fixtures).
///
/// # Errors
///
/// Propagates [`GnmModelError`] when the pinned GNM schema changes shape.
pub fn synthetic_head_model() -> Result<GnmModel, GnmModelError> {
    let vertex_count = 64;
    let identity_dim = vtuber_gnm::GNM_HEAD_V3_IDENTITY_DIM;
    let expression_dim = vtuber_gnm::GNM_HEAD_V3_EXPRESSION_DIM;
    let mut vertices = Vec::with_capacity(vertex_count * 3);
    for index in 0..vertex_count {
        let angle = (index as f32) / (vertex_count as f32) * std::f32::consts::TAU;
        vertices.extend_from_slice(&[
            0.10 * angle.cos(),
            0.12 * angle.sin(),
            0.05 * (3.0 * angle).sin(),
        ]);
    }
    let mut expression_basis = vec![0.0f32; expression_dim * vertex_count * 3];
    let eyelid_offset = vertex_count * 3;
    for vertex in 0..vertex_count {
        let base = vertex * 3;
        // Bounds are guaranteed by construction: `base` and
        // `eyelid_offset + base` index a buffer sized
        // `expression_dim * vertex_count * 3` with vertex < vertex_count.
        #[allow(clippy::indexing_slicing)]
        if vertex % 2 == 0 {
            // Channel 0 ("mouth"): vertical motion.
            expression_basis[base + 1] = -0.04;
        } else {
            // Channel 1 ("eyelid"): lateral motion, image-distinguishable
            // from the vertical channel so expression cannot be absorbed by
            // a rigid camera translation.
            expression_basis[eyelid_offset + base] = 0.03;
        }
    }
    GnmModel::from_data(vtuber_gnm::GnmModelData {
        version: vtuber_gnm::GNM_HEAD_V3_VERSION,
        variant: vtuber_gnm::GnmVariant::Head,
        template_vertices: vtuber_gnm::DenseArray::new(
            "vertices",
            vec![vertex_count, 3],
            vertices,
        )?,
        template_joints: vtuber_gnm::DenseArray::new("joints", vec![1, 3], vec![0.0; 3])?,
        vertex_identity_basis: vtuber_gnm::DenseArray::new(
            "identity",
            vec![identity_dim, vertex_count, 3],
            vec![0.0; identity_dim * vertex_count * 3],
        )?,
        joint_identity_basis: vtuber_gnm::DenseArray::new(
            "joint_identity",
            vec![identity_dim, 1, 3],
            vec![0.0; identity_dim * 3],
        )?,
        expression_basis: vtuber_gnm::DenseArray::new(
            "expression",
            vec![expression_dim, vertex_count, 3],
            expression_basis,
        )?,
        joint_parent_indices: vec![-1],
        skinning_weights: vtuber_gnm::DenseArray::new(
            "weights",
            vec![1, vertex_count],
            vec![1.0; vertex_count],
        )?,
        pose_correctives_regressor: None,
    })
}

/// Vertex-to-landmark correspondence table matching `synthetic_head_model`.
///
/// # Errors
///
/// Propagates correspondence validation failures verbatim.
pub fn synthetic_mapping(
    model: &GnmModel,
) -> Result<DenseCorrespondenceSet, vtuber_gnm::GnmDenseError> {
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
            model_version: vtuber_gnm::GNM_HEAD_V3_VERSION,
        },
        rows,
        model,
    )
}

/// Ground truth for one frame: rigid pose and the driven expression channel.
fn truth_at(trajectory: SequenceTrajectory, index: usize, frame_count: usize) -> ([f32; 3], f32) {
    let progress = index as f32 / (frame_count.saturating_sub(1)).max(1) as f32;
    match trajectory {
        SequenceTrajectory::Static => ([0.15, -0.10, 0.05], 0.5),
        SequenceTrajectory::SlowRamp => ([0.15, -0.10, 0.05], progress * 0.8),
        SequenceTrajectory::FastRelease => {
            let hold_end = (frame_count as f32 * 0.4).round() as usize;
            let mouth = if index < hold_end {
                0.8
            } else {
                let release_progress = ((index - hold_end) as f32 / 6.0).min(1.0);
                0.8 - 0.7 * release_progress
            };
            ([0.15, -0.10, 0.05], mouth)
        }
        SequenceTrajectory::BlinkPulse => {
            let center = frame_count as f32 * 0.4;
            let width = 3.0_f32;
            let pulse = (-((index as f32 - center) / width).powi(2)).exp();
            ([0.15, -0.10, 0.05], 0.15 + 0.75 * pulse)
        }
        SequenceTrajectory::HeadPoseAndExpression => {
            ([-0.15 + 0.30 * progress, -0.08, 0.04], 0.2 + 0.5 * progress)
        }
    }
}

/// Owned temporal-history entry for one published valid state.
#[derive(Clone)]
struct HistoryEntry {
    stamp: GnmFrameStamp,
    expression: Vec<f32>,
    joints: Vec<f32>,
    head_pose: [f32; 3],
    translation: [f32; 3],
}

impl HistoryEntry {
    fn from_dynamic(stamp: GnmFrameStamp, dynamic: &GnmDynamicState) -> Self {
        let mut joints = Vec::new();
        for rotation in dynamic.joints.rotations() {
            joints.extend_from_slice(rotation);
        }
        joints.extend_from_slice(&dynamic.joints.translation());
        Self {
            stamp,
            expression: dynamic.expression.values().to_vec(),
            joints,
            head_pose: dynamic.rigid_pose.yaw_pitch_roll(),
            translation: dynamic.camera.translation(),
        }
    }

    fn view(&self) -> GnmTemporalStateView<'_> {
        GnmTemporalStateView {
            expression: &self.expression,
            joints: &self.joints,
            head_pose: &self.head_pose,
            translation: &self.translation,
        }
    }
}

/// Per-run owned resources handed down through the frame loop.
struct RunContext<'a> {
    model: &'a GnmModel,
    mapping: &'a DenseCorrespondenceSet,
    identity: &'a FixedGnmIdentity,
    coverage_policy: DenseCoveragePolicy,
    solver_config: SingleFrameFitConfig,
    lifecycle_config: PersistentGnmLifecycleConfig,
    normalization: GnmTemporalNormalization<'a>,
}

/// Internal mutable state threaded through the run loop.
struct RunState {
    lifecycle: PersistentGnmLifecycleState,
    published: Option<(GnmFrameStamp, GnmDynamicState)>,
    history_previous: Option<HistoryEntry>,
    history_previous_previous: Option<HistoryEntry>,
    adaptive_state: Option<AdaptiveTemporalState>,
    samples: Vec<TrackedSample>,
    no_observation_frames: usize,
    temporal_history_resets: usize,
}

/// Runs one scenario under one configuration and returns the aggregated
/// deterministic report.
///
/// # Errors
///
/// Returns the first typed failure from scenario validation, fixture
/// construction, synthesis, the lifecycle, the adaptive policy, or the
/// bounded solver.
pub fn run_sequence_regression(
    scenario: SequenceScenario,
    mode: SequenceFitMode,
) -> Result<SequenceRunReport, SequenceRegressionError> {
    if scenario.frame_count < 12 || !scenario.fps.is_finite() || scenario.fps < 4.0 {
        return Err(SequenceRegressionError::InvalidScenario(
            "frame_count must be >= 12 and fps finite >= 4".to_owned(),
        ));
    }
    let model = synthetic_head_model().map_err(SequenceRegressionError::Model)?;
    let mapping = synthetic_mapping(&model)
        .map_err(|error| SequenceRegressionError::Synthesis(GnmReprojectionError::from(error)))?;
    let identity = FixedGnmIdentity::new(model.neutral_identity(), &model)
        .map_err(|error| SequenceRegressionError::InvalidScenario(error.to_string()))?;
    let coverage_policy = DenseCoveragePolicy::new(2, 0.75).map_err(|_| {
        SequenceRegressionError::InvalidScenario("coverage policy constants are invalid".to_owned())
    })?;

    // Owned normalization scales; borrowed immutably by every temporal
    // penalty built during the run.
    let expression_scale = vec![1.0_f32; model.expression_dimension()];
    let joints_scale = vec![1.0_f32; 3 * (model.joint_count() + 1)];
    // Radian groups use a 0.2 rad unit so the shared lambda band produces
    // temporal curvature of the same order as this fixture's dense
    // curvature (O(0.1)) at the top of the range.
    let normalization = GnmTemporalNormalization {
        expression: &expression_scale,
        joints: &joints_scale,
        head_pose: &[0.2_f32; 3],
        translation: &[0.2_f32; 3],
    };

    let context = RunContext {
        model: &model,
        mapping: &mapping,
        identity: &identity,
        coverage_policy,
        solver_config: regression_solver_config(),
        lifecycle_config: regression_lifecycle_config(),
        normalization,
    };

    let frame_delta_micros = (1.0e6 / scenario.fps).round() as u64;
    let mut state = RunState {
        lifecycle: PersistentGnmLifecycleState::default(),
        published: None,
        history_previous: None,
        history_previous_previous: None,
        adaptive_state: None,
        samples: Vec::with_capacity(scenario.frame_count),
        no_observation_frames: 0,
        temporal_history_resets: 0,
    };

    // Calibrate up front so every frame exercises the tracking path.
    let decision = advance_persistent_gnm_lifecycle(
        state.lifecycle,
        PersistentGnmEvent::CalibrationReady,
        context.lifecycle_config,
    )
    .map_err(SequenceRegressionError::Lifecycle)?;
    state.lifecycle = decision.state;

    let mut extra_gap_micros: u64 = 0;
    let dropout_end = scenario
        .anomalies
        .dropout_start_frame
        .map(|start| start.saturating_add(scenario.anomalies.dropout_length));

    for index in 0..scenario.frame_count {
        let source_seq = index as u64 + 1;
        let captured_at_micros = (index as u64)
            .saturating_mul(frame_delta_micros)
            .saturating_add(extra_gap_micros);

        // Insert the long-gap extension once, after the last dropout frame,
        // so the reacquire frame observes a timeline jump.
        if let (Some(start), Some(end)) = (scenario.anomalies.dropout_start_frame, dropout_end)
            && index + 1 == end
            && index >= start
        {
            extra_gap_micros =
                extra_gap_micros.saturating_add(scenario.anomalies.extra_gap_after_dropout_micros);
        }

        let in_dropout = matches!(
            (scenario.anomalies.dropout_start_frame, dropout_end),
            (Some(start), Some(end)) if index >= start && index < end
        );
        let (truth_pose, truth_mouth) = truth_at(scenario.trajectory, index, scenario.frame_count);

        let observation = if in_dropout {
            insufficient_observation(
                source_seq,
                captured_at_micros,
                &mapping,
                context.coverage_policy,
            )?
        } else if scenario.anomalies.outlier_frame == Some(index) {
            let outlier_projection = DenseProjection::new(
                [
                    truth_pose[0] + 0.25,
                    truth_pose[1] + 0.05,
                    truth_pose[2] - 0.03,
                ],
                [0.07, -0.03, 0.60],
                1.3,
                [0.5, 0.5],
            )
            .map_err(SequenceRegressionError::Synthesis)?;
            synthesize_observation(
                &context,
                truth_mouth,
                &outlier_projection,
                source_seq,
                captured_at_micros,
            )?
        } else {
            let projection = DenseProjection::new(truth_pose, [0.02, -0.03, 0.60], 1.3, [0.5, 0.5])
                .map_err(SequenceRegressionError::Synthesis)?;
            synthesize_observation(
                &context,
                truth_mouth,
                &projection,
                source_seq,
                captured_at_micros,
            )?
        };

        let stamp = GnmFrameStamp {
            source_seq,
            captured_at_micros,
        };
        step_frame(
            &observation,
            stamp,
            (truth_pose, truth_mouth),
            scenario,
            mode,
            &context,
            &mut state,
        )?;
    }

    finalize_report(scenario, mode, state)
}

/// Synthesizes one stamped dense observation for the given truth channel.
fn synthesize_observation(
    context: &RunContext<'_>,
    truth_mouth: f32,
    projection: &DenseProjection,
    source_seq: u64,
    captured_at_micros: u64,
) -> Result<GnmDenseObservation, SequenceRegressionError> {
    let mut expression_values = vec![0.0_f32; context.model.expression_dimension()];
    if let Some(slot) = expression_values.first_mut() {
        *slot = truth_mouth;
    }
    let expression =
        GnmExpressionState::new(expression_values, context.model.expression_dimension())
            .map_err(|error| SequenceRegressionError::InvalidScenario(error.to_string()))?;
    let joints = GnmJointState::neutral(context.model.joint_count());
    synthesize_observation_from_projection_helper(
        context.model,
        context.identity.state(),
        &expression,
        &joints,
        context.mapping,
        projection,
        SynthesisOptions {
            source_seq,
            captured_at_micros,
            ..SynthesisOptions::default()
        },
        context.coverage_policy,
    )
    .map_err(SequenceRegressionError::Synthesis)
}

/// Thin wrapper so the harness compiles against the synthesis API without
/// importing the full name twice.
#[allow(clippy::too_many_arguments)]
fn synthesize_observation_from_projection_helper(
    model: &GnmModel,
    identity: &vtuber_gnm::GnmIdentityState,
    expression: &GnmExpressionState,
    joints: &GnmJointState,
    mapping: &DenseCorrespondenceSet,
    projection: &DenseProjection,
    options: SynthesisOptions,
    coverage_policy: DenseCoveragePolicy,
) -> Result<GnmDenseObservation, GnmReprojectionError> {
    vtuber_gnm::synthesize_observation_from_projection(
        model,
        identity,
        expression,
        joints,
        mapping,
        projection,
        options,
        coverage_policy,
        |_, _| false,
    )
}

/// Builds the unusable (dropout) observation carrying NaN landmarks.
///
/// The mapping is consulted only for region metadata; every NaN point slot
/// becomes invalid regardless.
fn insufficient_observation(
    source_seq: u64,
    captured_at_micros: u64,
    mapping: &DenseCorrespondenceSet,
    coverage_policy: DenseCoveragePolicy,
) -> Result<GnmDenseObservation, SequenceRegressionError> {
    let landmarks = vec![[f32::NAN; 2]; vtuber_gnm::MEDIAPIPE_FACE_LANDMARK_COUNT];
    GnmDenseObservation::from_mediapipe_xy(
        source_seq,
        captured_at_micros,
        &landmarks,
        mapping,
        coverage_policy,
    )
    .map_err(|error| SequenceRegressionError::Synthesis(GnmReprojectionError::from(error)))
}

/// Admits one frame through the lifecycle and, when directed, runs exactly
/// one bounded solve under the requested configuration.
#[allow(clippy::too_many_arguments)]
fn step_frame(
    observation: &GnmDenseObservation,
    stamp: GnmFrameStamp,
    truth: ([f32; 3], f32),
    scenario: SequenceScenario,
    mode: SequenceFitMode,
    context: &RunContext<'_>,
    state: &mut RunState,
) -> Result<(), SequenceRegressionError> {
    let observation_available = matches!(
        observation.coverage().status,
        DenseObservationStatus::Valid | DenseObservationStatus::Degraded
    );
    let decision = advance_persistent_gnm_lifecycle(
        state.lifecycle,
        PersistentGnmEvent::SourceFrame {
            stamp,
            observation_available,
        },
        context.lifecycle_config,
    )
    .map_err(SequenceRegressionError::Lifecycle)?;
    state.lifecycle = decision.state;

    match decision.action {
        PersistentGnmAction::SkipUncalibratedFrame => {}
        PersistentGnmAction::NoObservation {
            dynamic_state_cleared,
        } => {
            state.no_observation_frames += 1;
            if dynamic_state_cleared {
                state.published = None;
                state.history_previous = None;
                state.history_previous_previous = None;
                state.adaptive_state = None;
            }
        }
        PersistentGnmAction::StartFit { initialization } => {
            let effective_initialization = if mode == SequenceFitMode::Cold {
                // Force neutral dynamics plus rigid recovery: no cross-frame
                // reuse even when the lifecycle offers a warm start.
                GnmFitInitialization::ReinitializeDynamicState
            } else {
                initialization
            };
            solve_directed_frame(
                observation,
                stamp,
                truth,
                scenario,
                mode,
                effective_initialization,
                context,
                state,
            )?;
        }
        PersistentGnmAction::ResetDynamicState
        | PersistentGnmAction::PublishCurrentFit
        | PersistentGnmAction::RejectInvalidFit
        | PersistentGnmAction::RejectInvalidFitAndLose => {
            // Unreachable from a SourceFrame event; fail closed.
            return Err(SequenceRegressionError::Bookkeeping { stamp });
        }
    }
    Ok(())
}

/// Runs the bounded solve for an admitted frame, routes the result back
/// through the lifecycle, and records the sample when published.
#[allow(clippy::too_many_arguments)]
fn solve_directed_frame(
    observation: &GnmDenseObservation,
    stamp: GnmFrameStamp,
    truth: ([f32; 3], f32),
    scenario: SequenceScenario,
    mode: SequenceFitMode,
    initialization: GnmFitInitialization,
    context: &RunContext<'_>,
    state: &mut RunState,
) -> Result<(), SequenceRegressionError> {
    // --- Initial values -------------------------------------------------
    let (mut expression, mut joints, mut projection) = match initialization {
        GnmFitInitialization::NeutralFirstFit | GnmFitInitialization::ReinitializeDynamicState => {
            let neutral_expression = context.model.neutral_expression();
            let neutral_joints = GnmJointState::neutral(context.model.joint_count());
            let mut surface = GnmSparseVertices::with_len(context.mapping.len());
            context
                .mapping
                .evaluate_surface(
                    context.model,
                    context.identity.state(),
                    &neutral_expression,
                    &neutral_joints,
                    &mut surface,
                )
                .map_err(|error| {
                    SequenceRegressionError::Synthesis(GnmReprojectionError::from(error))
                })?;
            let neutral_projection = fitting_projection(surface.values(), [0.0; 3])
                .map_err(SequenceRegressionError::Synthesis)?;
            let recovered = recover_rigid_projection(
                context.model,
                context.identity.state(),
                &neutral_expression,
                &neutral_joints,
                context.mapping,
                observation,
                neutral_projection,
                RigidRecoveryConfig::default(),
            )
            .map_err(SequenceRegressionError::Solve)?;
            (neutral_expression, neutral_joints, recovered.projection)
        }
        GnmFitInitialization::PreviousValid { source } => {
            let Some((stored_stamp, dynamic)) = state.published.as_ref() else {
                return Err(SequenceRegressionError::Bookkeeping { stamp });
            };
            if *stored_stamp != source {
                return Err(SequenceRegressionError::Bookkeeping { stamp });
            }
            let projection = DenseProjection::new(
                dynamic.rigid_pose.yaw_pitch_roll(),
                dynamic.camera.translation(),
                dynamic.camera.focal(),
                dynamic.camera.principal_point(),
            )
            .map_err(SequenceRegressionError::Solve)?;
            (
                dynamic.expression.clone(),
                dynamic.joints.clone(),
                projection,
            )
        }
    };

    // --- Optional temporal penalty --------------------------------------
    let temporal_config: Option<TemporalRegularizationConfig> = match mode {
        SequenceFitMode::Cold | SequenceFitMode::WarmStart => None,
        SequenceFitMode::FixedTemporal => Some(
            map_strengths_to_temporal_config(fixed_strengths(), &temporal_lambda_ranges(), 0.25)
                .map_err(SequenceRegressionError::AdaptivePolicy)?,
        ),
        SequenceFitMode::AdaptiveTemporal => {
            let strengths = advance_adaptive_strengths(state, scenario.fps)?;
            Some(
                map_strengths_to_temporal_config(strengths, &temporal_lambda_ranges(), 0.25)
                    .map_err(SequenceRegressionError::AdaptivePolicy)?,
            )
        }
    };

    // Build the temporal penalty from lifecycle-owned history (cloned into
    // local buffers so the borrow ends before the history ring mutates). A
    // long gap surfaces as HistoryResetRequired: drop stale history and
    // continue the reacquire frame without temporal energy.
    let mut penalty: Option<SingleFrameTemporalPenalty<'_>> = None;
    let mut history_reset_required = false;
    // Cloned history entries live until after the fit call because the
    // penalty borrows their buffers.
    let previous_owned = state.history_previous.clone();
    let previous_previous_owned = state.history_previous_previous.clone();
    if let Some(config) = temporal_config.as_ref()
        && let Some(previous) = previous_owned.as_ref()
    {
        let dt_seconds = (stamp
            .captured_at_micros
            .saturating_sub(previous.stamp.captured_at_micros)) as f64
            / 1.0e6;
        let previous_dt_seconds = previous_previous_owned.as_ref().map(|entry| {
            (previous
                .stamp
                .captured_at_micros
                .saturating_sub(entry.stamp.captured_at_micros)) as f64
                / 1.0e6
        });
        match SingleFrameTemporalPenalty::new(
            previous.view(),
            previous_previous_owned.as_ref().map(HistoryEntry::view),
            context.normalization,
            TemporalHistoryTiming {
                dt_seconds,
                previous_dt_seconds,
            },
            *config,
        ) {
            Ok(built) => penalty = Some(built),
            Err(TemporalRegularizationError::HistoryResetRequired { .. }) => {
                history_reset_required = true;
            }
            Err(error) => return Err(SequenceRegressionError::Solve(error.into())),
        }
    }

    // --- Bounded solve ---------------------------------------------------
    let outcome = fit_single_frame_with_temporal(
        context.model,
        context.identity.state(),
        &expression,
        &joints,
        context.mapping,
        observation,
        &projection,
        context.solver_config,
        None,
        penalty.as_ref(),
    )
    .map_err(SequenceRegressionError::Solve)?;

    // Resolve the pending fit in the lifecycle before touching any stored
    // authority mirrors.
    let fit_outcome = if outcome.valid() {
        GnmFitOutcome::Valid
    } else {
        GnmFitOutcome::Invalid
    };
    let result_decision = advance_persistent_gnm_lifecycle(
        state.lifecycle,
        PersistentGnmEvent::FitResult {
            stamp,
            outcome: fit_outcome,
        },
        context.lifecycle_config,
    )
    .map_err(SequenceRegressionError::Lifecycle)?;
    state.lifecycle = result_decision.state;

    match result_decision.action {
        PersistentGnmAction::PublishCurrentFit => {}
        PersistentGnmAction::RejectInvalidFitAndLose => {
            state.published = None;
            state.history_previous = None;
            state.history_previous_previous = None;
            return Ok(());
        }
        PersistentGnmAction::RejectInvalidFit => return Ok(()),
        impossible @ (PersistentGnmAction::ResetDynamicState
        | PersistentGnmAction::SkipUncalibratedFrame
        | PersistentGnmAction::NoObservation { .. }
        | PersistentGnmAction::StartFit { .. }) => {
            // Unreachable from a FitResult event; fail closed.
            let _ = impossible;
            return Err(SequenceRegressionError::Bookkeeping { stamp });
        }
    }

    if history_reset_required {
        state.temporal_history_resets += 1;
        state.history_previous = None;
        state.history_previous_previous = None;
    }

    expression = outcome.expression().clone();
    joints = outcome.joints().clone();
    projection = *outcome.projection();

    let dynamic = GnmDynamicState {
        expression: expression.clone(),
        joints: joints.clone(),
        rigid_pose: GnmRigidPoseBlock::new(projection.yaw_pitch_roll()).map_err(|error| {
            SequenceRegressionError::InvalidScenario(format!("published pose rejected: {error}"))
        })?,
        camera: GnmCameraBlock::new(
            projection.translation(),
            projection.focal(),
            projection.principal_point(),
        )
        .map_err(|error| {
            SequenceRegressionError::InvalidScenario(format!("published camera rejected: {error}"))
        })?,
    };

    // Record the sample against truth.
    let fitted_channel0 = expression.values().first().copied().unwrap_or(f32::NAN);
    state.samples.push(TrackedSample {
        source_seq: stamp.source_seq,
        iterations: outcome.iterations(),
        expression_channel0: fitted_channel0,
        yaw_pitch_roll: projection.yaw_pitch_roll(),
        truth_expression_channel0: truth.1,
        truth_yaw_pitch_roll: truth.0,
    });

    // Advance the temporal history ring with the freshly published state.
    let entry = HistoryEntry::from_dynamic(stamp, &dynamic);
    state.history_previous_previous = state.history_previous.take();
    state.history_previous = Some(entry);
    state.published = Some((stamp, dynamic));

    Ok(())
}

/// Advances the adaptive strength policy from the last two published states.
fn advance_adaptive_strengths(
    state: &mut RunState,
    fps: f64,
) -> Result<TemporalGroupWeights, SequenceRegressionError> {
    let config = adaptive_policy_config();
    let (expression_motion, rigid_motion) =
        match (&state.history_previous, &state.history_previous_previous) {
            (Some(current), Some(previous)) => {
                let dt = (current
                    .stamp
                    .captured_at_micros
                    .saturating_sub(previous.stamp.captured_at_micros))
                    as f64
                    / 1.0e6;
                if dt <= 0.0 || !dt.is_finite() {
                    (0.0, 0.0)
                } else {
                    let expression_delta = current
                        .expression
                        .iter()
                        .zip(previous.expression.iter())
                        .map(|(now, before)| (*now - *before).abs() as f64)
                        .sum::<f64>()
                        / dt;
                    let rigid_delta = current
                        .head_pose
                        .iter()
                        .zip(previous.head_pose.iter())
                        .map(|(now, before)| (*now - *before).abs() as f64)
                        .fold(0.0_f64, f64::max)
                        / dt;
                    (expression_delta, rigid_delta)
                }
            }
            _ => (0.0, 0.0),
        };
    let next = advance_adaptive_temporal_policy(
        state.adaptive_state,
        crate::adaptive_temporal::AdaptiveTemporalInput {
            dt_seconds: 1.0 / fps,
            expression_motion,
            rigid_motion,
            observation_quality: None,
            observation_health: TemporalObservationHealth::Nominal,
        },
        config,
    )
    .map_err(SequenceRegressionError::AdaptivePolicy)?;
    state.adaptive_state = Some(next);
    Ok(next.weights)
}

/// Aggregates the collected samples into the deterministic report.
fn finalize_report(
    scenario: SequenceScenario,
    mode: SequenceFitMode,
    state: RunState,
) -> Result<SequenceRunReport, SequenceRegressionError> {
    let samples = &state.samples;
    if samples.is_empty() {
        return Err(SequenceRegressionError::InvalidScenario(
            "no frames were published; cannot build a report".to_owned(),
        ));
    }
    let solved_frames = samples.len();
    let total_iterations: u64 = samples.iter().map(|sample| sample.iterations as u64).sum();
    let mean_iterations = total_iterations as f64 / solved_frames as f64;

    let expression_errors: Vec<f64> = samples
        .iter()
        .map(|sample| (sample.expression_channel0 - sample.truth_expression_channel0).abs() as f64)
        .collect();
    let mean_expression_error =
        expression_errors.iter().sum::<f64>() / expression_errors.len() as f64;
    let final_expression_error = expression_errors.last().copied().unwrap_or(f64::NAN);

    let pose_error_sum: f64 = samples
        .iter()
        .map(|sample| {
            let yaw = (sample.yaw_pitch_roll[0] - sample.truth_yaw_pitch_roll[0]) as f64;
            let pitch = (sample.yaw_pitch_roll[1] - sample.truth_yaw_pitch_roll[1]) as f64;
            let roll = (sample.yaw_pitch_roll[2] - sample.truth_yaw_pitch_roll[2]) as f64;
            (yaw * yaw + pitch * pitch + roll * roll).sqrt()
        })
        .sum();
    let mean_pose_error_rad = pose_error_sum / samples.len() as f64;

    // Jitter: RMS of the fitted channel-0 derivative between consecutive
    // tracked samples using their nominal inter-frame time.
    let mut jitter_sum = 0.0_f64;
    let mut jitter_terms = 0_usize;
    for pair in samples.windows(2) {
        // `windows(2)` yields exactly two elements; destructuring keeps the
        // invariant local instead of relying on slice indexing.
        let (Some(first), Some(second)) = (pair.first(), pair.get(1)) else {
            continue;
        };
        let dt = (second.source_seq.saturating_sub(first.source_seq)) as f64 / scenario.fps;
        if dt <= 0.0 {
            continue;
        }
        let derivative = (second.expression_channel0 - first.expression_channel0) as f64 / dt;
        jitter_sum += derivative * derivative;
        jitter_terms += 1;
    }
    let expression_jitter_per_second = if jitter_terms > 0 {
        (jitter_sum / jitter_terms as f64).sqrt()
    } else {
        0.0
    };

    // Latency metrics compare the truth crossing with the fitted crossing of
    // the same threshold, expressed in frames.
    let upward_crossing_latency = |threshold: f32| -> Option<f64> {
        let truth_index = samples
            .iter()
            .position(|sample| sample.truth_expression_channel0 >= threshold);
        let fitted_index = samples
            .iter()
            .position(|sample| sample.expression_channel0 >= threshold);
        match (truth_index, fitted_index) {
            (Some(truth), Some(fitted)) => Some(fitted as f64 - truth as f64),
            _ => None,
        }
    };
    let downward_crossing_latency = |threshold: f32| -> Option<f64> {
        let truth_index = samples
            .iter()
            .position(|sample| sample.truth_expression_channel0 <= threshold);
        let fitted_index = samples
            .iter()
            .position(|sample| sample.expression_channel0 <= threshold);
        match (truth_index, fitted_index) {
            (Some(truth), Some(fitted)) => Some(fitted as f64 - truth as f64),
            _ => None,
        }
    };

    let onset_latency_frames = match scenario.trajectory {
        SequenceTrajectory::SlowRamp => upward_crossing_latency(0.4),
        _ => None,
    };
    let release_latency_frames = match scenario.trajectory {
        SequenceTrajectory::FastRelease => downward_crossing_latency(0.45),
        _ => None,
    };
    let pulse_peak_expression = match scenario.trajectory {
        SequenceTrajectory::BlinkPulse => samples
            .iter()
            .map(|sample| sample.expression_channel0 as f64)
            .max_by(f64::total_cmp),
        _ => None,
    };

    Ok(SequenceRunReport {
        trajectory: scenario.trajectory,
        mode,
        solved_frames,
        no_observation_frames: state.no_observation_frames,
        total_iterations,
        mean_iterations,
        mean_expression_error,
        final_expression_error,
        mean_pose_error_rad,
        expression_jitter_per_second,
        onset_latency_frames,
        release_latency_frames,
        pulse_peak_expression,
        temporal_history_resets: state.temporal_history_resets,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // tests may panic (AGENTS.md)

    use super::*;

    const FRAMES: usize = 24;

    fn scenario(trajectory: SequenceTrajectory) -> SequenceScenario {
        SequenceScenario::new(trajectory, FRAMES, 60.0).unwrap()
    }

    #[test]
    fn static_sequence_is_deterministic_across_runs() {
        let first = run_sequence_regression(
            scenario(SequenceTrajectory::Static),
            SequenceFitMode::WarmStart,
        )
        .unwrap();
        let second = run_sequence_regression(
            scenario(SequenceTrajectory::Static),
            SequenceFitMode::WarmStart,
        )
        .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn all_modes_complete_on_the_static_sequence() {
        for mode in [
            SequenceFitMode::Cold,
            SequenceFitMode::WarmStart,
            SequenceFitMode::FixedTemporal,
            SequenceFitMode::AdaptiveTemporal,
        ] {
            let report =
                run_sequence_regression(scenario(SequenceTrajectory::Static), mode).unwrap();
            assert!(
                report.solved_frames >= FRAMES - 2,
                "{mode:?}: solved {} of {FRAMES}",
                report.solved_frames
            );
            assert_eq!(report.no_observation_frames, 0, "{mode:?}");
            assert!(
                report.mean_expression_error < 0.15,
                "{mode:?}: mean expression error {}",
                report.mean_expression_error
            );
            assert!(
                report.final_expression_error < 0.15,
                "{mode:?}: final expression error {}",
                report.final_expression_error
            );
        }
    }

    #[test]
    fn warm_start_does_not_worsen_mean_iterations_versus_cold() {
        let cold = run_sequence_regression(
            scenario(SequenceTrajectory::HeadPoseAndExpression),
            SequenceFitMode::Cold,
        )
        .unwrap();
        let warm = run_sequence_regression(
            scenario(SequenceTrajectory::HeadPoseAndExpression),
            SequenceFitMode::WarmStart,
        )
        .unwrap();
        assert!(
            warm.mean_iterations <= cold.mean_iterations + 0.5,
            "warm mean {} must not exceed cold mean {} (+0.5)",
            warm.mean_iterations,
            cold.mean_iterations
        );
    }

    #[test]
    fn slow_ramp_reports_a_finite_onset_latency() {
        let report = run_sequence_regression(
            scenario(SequenceTrajectory::SlowRamp),
            SequenceFitMode::WarmStart,
        )
        .unwrap();
        let latency = report
            .onset_latency_frames
            .expect("slow ramp must produce an onset latency");
        assert!(
            latency.abs() <= 6.0,
            "onset latency {latency} frames exceeded the bound"
        );
    }

    #[test]
    fn fast_release_reports_a_finite_release_latency() {
        let report = run_sequence_regression(
            scenario(SequenceTrajectory::FastRelease),
            SequenceFitMode::WarmStart,
        )
        .unwrap();
        let latency = report
            .release_latency_frames
            .expect("fast release must produce a release latency");
        assert!(
            latency.abs() <= 8.0,
            "release latency {latency} frames exceeded the bound"
        );
    }

    #[test]
    fn blink_pulse_captures_the_peak_expression() {
        let report = run_sequence_regression(
            scenario(SequenceTrajectory::BlinkPulse),
            SequenceFitMode::WarmStart,
        )
        .unwrap();
        let peak = report
            .pulse_peak_expression
            .expect("blink pulse must produce a peak metric");
        assert!(
            (0.6..=1.0).contains(&peak),
            "pulse peak {peak} outside the expected band"
        );
    }

    #[test]
    fn head_pose_and_expression_tracks_both_groups() {
        let report = run_sequence_regression(
            scenario(SequenceTrajectory::HeadPoseAndExpression),
            SequenceFitMode::AdaptiveTemporal,
        )
        .unwrap();
        assert!(
            report.mean_pose_error_rad < 0.05,
            "mean pose error {} rad",
            report.mean_pose_error_rad
        );
        assert!(
            report.mean_expression_error < 0.15,
            "mean expression error {}",
            report.mean_expression_error
        );
    }

    #[test]
    fn one_frame_outlier_does_not_corrupt_the_final_state() {
        let scenario = SequenceScenario {
            anomalies: SequenceAnomalies {
                outlier_frame: Some(FRAMES / 2),
                ..SequenceAnomalies::default()
            },
            ..scenario(SequenceTrajectory::Static)
        };
        let report = run_sequence_regression(scenario, SequenceFitMode::FixedTemporal).unwrap();
        assert!(
            report.final_expression_error < 0.15,
            "final error after outlier recovery {}",
            report.final_expression_error
        );
        assert!(report.solved_frames >= FRAMES - 2);
    }

    #[test]
    fn short_dropout_survives_and_resumes_tracking() {
        let scenario = SequenceScenario {
            anomalies: SequenceAnomalies {
                dropout_start_frame: Some(10),
                dropout_length: 2,
                ..SequenceAnomalies::default()
            },
            ..scenario(SequenceTrajectory::SlowRamp)
        };
        let report = run_sequence_regression(scenario, SequenceFitMode::WarmStart).unwrap();
        assert_eq!(report.no_observation_frames, 2);
        assert!(report.solved_frames >= FRAMES - 3);
    }

    #[test]
    fn long_gap_forces_temporal_history_reset_and_reacquires() {
        let scenario = SequenceScenario {
            // 8 dropped frames at 60 fps plus an explicit extra gap far
            // beyond the 250 ms lifecycle reuse age.
            anomalies: SequenceAnomalies {
                dropout_start_frame: Some(8),
                dropout_length: 8,
                extra_gap_after_dropout_micros: 400_000,
                ..SequenceAnomalies::default()
            },
            ..scenario(SequenceTrajectory::Static)
        };
        let temporal_report =
            run_sequence_regression(scenario, SequenceFitMode::FixedTemporal).unwrap();
        assert!(
            temporal_report.temporal_history_resets >= 1,
            "long gap must force at least one temporal history reset"
        );
        assert!(
            temporal_report.solved_frames >= FRAMES - 22,
            "reacquire after long gap: solved {}",
            temporal_report.solved_frames
        );
        // The plain warm-start path must also recover through the lifecycle's
        // reacquire handling without temporal state.
        let warm_report = run_sequence_regression(scenario, SequenceFitMode::WarmStart).unwrap();
        assert!(warm_report.solved_frames >= FRAMES - 10);
    }
}
