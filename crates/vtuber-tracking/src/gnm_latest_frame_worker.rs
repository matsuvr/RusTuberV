//! Latest-frame worker connection for the persistent GNM fitter
//! (Issue #64.6 / #95, parent #64).
//!
//! The worker owns the persistent fitter and all heavy read-only resources
//! ([`GnmModel`], [`DenseCorrespondenceSet`], fixed identity/calibration)
//! for its entire lifetime. Per-frame work receives only an [`Arc`] view of
//! those resources; nothing is reloaded or cloned per frame.
//!
//! Input uses capacity-one latest-value semantics
//! ([`LatestSlot`]): a slow fitter never accumulates a backlog. Frames that
//! arrive while a fit is running overwrite the pending input and are counted
//! as bounded replacements in [`GnmFitterWorkerMetrics`].
//!
//! Publication is lifecycle-gated: a state reaches the output slot only
//! when the lifecycle emits its publish action. Invalid fits,
//! observation loss, and reacquisitions never bypass that authority; the
//! previous published state simply stays visible downstream until a newly
//! validated frame replaces it.
//!
//! The worker never touches Bevy types, OS handles, or clocks other than a
//! monotonic latency measurement around the fit call.
//!
//! Wall-clock reference (issue #148): the six in-crate worker tests target a
//! combined runtime under ~1 second in the default dev profile on the
//! reference Windows workstation. Measured after the packed lower-triangular
//! solver and the dev-profile optimization of `vtuber-gnm`: ~0.04 seconds
//! (was ~0.63 seconds before issue #148). This is a measurement guideline,
//! not an asserted bound; wall-clock assertions would be flaky across
//! machines.

use std::sync::Arc;
use std::time::{Duration, Instant};

use vtuber_core::{LatestSlot, ReadResult, StopToken, WorkerHandle};
use vtuber_gnm::{
    DenseCorrespondenceSet, FixedGnmIdentity, GnmFrameStamp, GnmModel,
    PersistentGnmLifecycleConfig, SingleFrameFitConfig,
};

use crate::gnm_fitter_contract::{GnmDynamicState, GnmFitterContractError, GnmSolverFrameInput};
use crate::gnm_persistent_fitter::{
    PersistentGnmFitter, PersistentGnmFitterError, PersistentGnmFrameOutcome,
};

/// One source frame offered to the worker.
///
/// The dense observation is shared through an [`Arc`]; producing stages pay
/// one allocation per frame and the worker never copies point data.
#[derive(Clone, Debug)]
pub struct GnmWorkerFrameInput {
    stamp: GnmFrameStamp,
    observation: Arc<vtuber_gnm::GnmDenseObservation>,
}

impl GnmWorkerFrameInput {
    /// Pairs a source stamp with its dense observation.
    ///
    /// Stamp/observation alignment is re-validated by the fitter contract
    /// when the frame is admitted, so a mismatching pair is rejected as a
    /// typed error instead of being trusted here.
    #[must_use]
    pub fn new(stamp: GnmFrameStamp, observation: Arc<vtuber_gnm::GnmDenseObservation>) -> Self {
        Self { stamp, observation }
    }

    /// Returns the exact source stamp.
    #[must_use]
    pub fn stamp(&self) -> GnmFrameStamp {
        self.stamp
    }

    /// Returns the shared dense observation.
    #[must_use]
    pub fn observation(&self) -> &Arc<vtuber_gnm::GnmDenseObservation> {
        &self.observation
    }
}

/// Command or frame offered through the worker's single input slot.
#[derive(Clone, Debug)]
pub enum GnmWorkerInput {
    /// Apply the explicit calibration-ready lifecycle event.
    CalibrationReady,
    /// Apply the explicit calibration-invalidated lifecycle event.
    CalibrationInvalidated,
    /// Admit one source frame.
    Frame(GnmWorkerFrameInput),
}

/// Lifecycle-validated tracking state published by the worker.
///
/// Every instance passed through the output slot satisfied the persistent
/// lifecycle's validity checks on its exact source frame.
#[derive(Clone, Debug, PartialEq)]
pub struct GnmFaceState {
    stamp: GnmFrameStamp,
    dynamic: GnmDynamicState,
}

impl GnmFaceState {
    /// Returns the source frame this state was fitted on.
    #[must_use]
    pub fn stamp(&self) -> GnmFrameStamp {
        self.stamp
    }

    /// Returns the validated dynamic state.
    #[must_use]
    pub const fn dynamic(&self) -> &GnmDynamicState {
        &self.dynamic
    }
}

/// Counters and timings produced by one worker.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GnmFitterWorkerMetrics {
    /// Frames admitted with a usable observation and solved.
    pub solved_frames: u64,
    /// Solves published as the new lifecycle-valid state.
    pub published_frames: u64,
    /// Solves that completed but failed lifecycle validation.
    pub invalid_fits: u64,
    /// Frames without a usable observation (dropout path).
    pub no_observation_frames: u64,
    /// Frames skipped because no calibration exists.
    pub skipped_uncalibrated_frames: u64,
    /// New frames that overwrote the pending input while a fit was running.
    pub replaced_during_fit: u64,
    /// Input-slot contract violations (stamp/observation mismatch).
    pub contract_errors: u64,
    /// Deterministic solver failures.
    pub solve_errors: u64,
    /// Lifecycle/invariant failures; always zero in correct operation.
    pub internal_errors: u64,
    /// Wall-clock duration of the most recent fit, in microseconds.
    pub last_fit_latency_micros: u64,
    /// Largest observed fit latency, in microseconds.
    pub max_fit_latency_micros: u64,
    /// Block-coordinate iterations spent by the most recent fit.
    pub last_fit_iterations: usize,
}

/// Outcome of one non-blocking worker step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GnmWorkerStep {
    /// No fresh input arrived before the poll timeout elapsed.
    Idle,
    /// One input was processed.
    Processed,
    /// The input slot was closed; the worker should exit.
    Closed,
}

/// Heavy read-only resources owned by the worker for its whole lifetime.
#[derive(Clone, Debug)]
pub struct GnmFitterResources {
    /// Loaded Head v3 model. Never mutated or cloned per frame.
    pub model: Arc<GnmModel>,
    /// Dense correspondence basis. Never mutated or cloned per frame.
    pub mapping: Arc<DenseCorrespondenceSet>,
    /// Fixed identity/calibration. Set once; replaced only by constructing
    /// a new worker after an explicit calibration change.
    pub identity: Arc<FixedGnmIdentity>,
}

/// Poll timeout while waiting for fresh input; bounds shutdown latency.
const INPUT_POLL_TIMEOUT: Duration = Duration::from_millis(20);

/// Persistent-GNM latest-frame worker state machine over shared slots.
///
/// Construct it on the owning thread (or spawn it with
/// [`spawn_gnm_latest_frame_worker`]) and either drive [`Self::step_once`]
/// directly (tests, embedded loops) or call [`Self::run`] with a stop token.
pub struct GnmLatestFrameWorker {
    resources: GnmFitterResources,
    fitter: PersistentGnmFitter,
    solver_config: SingleFrameFitConfig,
    input: Arc<LatestSlot<GnmWorkerInput>>,
    output: Arc<LatestSlot<GnmFaceState>>,
    metrics_slot: Arc<LatestSlot<GnmFitterWorkerMetrics>>,
    metrics: GnmFitterWorkerMetrics,
    last_generation: u64,
}

impl GnmLatestFrameWorker {
    /// Creates a worker over the supplied shared slots.
    ///
    /// The worker starts uncalibrated; send
    /// [`GnmWorkerInput::CalibrationReady`] before expecting solves.
    #[must_use]
    pub fn new(
        resources: GnmFitterResources,
        solver_config: SingleFrameFitConfig,
        lifecycle_config: PersistentGnmLifecycleConfig,
        input: Arc<LatestSlot<GnmWorkerInput>>,
        output: Arc<LatestSlot<GnmFaceState>>,
        metrics_slot: Arc<LatestSlot<GnmFitterWorkerMetrics>>,
    ) -> Self {
        Self {
            fitter: PersistentGnmFitter::new(lifecycle_config),
            resources,
            solver_config,
            input,
            output,
            metrics_slot,
            metrics: GnmFitterWorkerMetrics::default(),
            last_generation: 0,
        }
    }

    /// Returns a snapshot of the current metrics.
    #[must_use]
    pub fn metrics(&self) -> GnmFitterWorkerMetrics {
        self.metrics.clone()
    }

    /// Runs the blocking loop until `stop` is requested or the input slot is
    /// closed. Metrics are published after every processed input.
    pub fn run(&mut self, stop: &StopToken) {
        while !stop.is_stopped() {
            if self.step_once() == GnmWorkerStep::Closed {
                break;
            }
        }
        self.publish_metrics();
    }

    /// Waits briefly for one fresh input and processes exactly it.
    ///
    /// This is the whole worker body minus the loop: latest-value semantics
    /// guarantee that whatever is oldest in the slot has already been
    /// dropped by the slot itself, so stale backlog can never accumulate.
    pub fn step_once(&mut self) -> GnmWorkerStep {
        let Some(read) = self
            .input
            .wait_read_after(self.last_generation, INPUT_POLL_TIMEOUT)
        else {
            return GnmWorkerStep::Idle;
        };
        match read {
            ReadResult::Closed => GnmWorkerStep::Closed,
            ReadResult::New(input) => {
                self.last_generation = self.input.generation();
                self.handle_input(&input);
                self.publish_metrics();
                GnmWorkerStep::Processed
            }
        }
    }

    /// Dispatches one input through the lifecycle and records the outcome.
    fn handle_input(&mut self, input: &GnmWorkerInput) {
        match input {
            GnmWorkerInput::CalibrationReady => {
                // A calibration reset invalidates every previously published
                // state; clear the output so downstream cannot act on stale
                // authority from the old calibration.
                if self.fitter.calibration_ready().is_err() {
                    self.metrics.internal_errors += 1;
                }
                self.output.clear();
            }
            GnmWorkerInput::CalibrationInvalidated => {
                if self.fitter.calibration_invalidated().is_err() {
                    self.metrics.internal_errors += 1;
                }
                self.output.clear();
            }
            GnmWorkerInput::Frame(frame) => self.handle_frame(frame),
        }
    }

    /// Admits one frame: measures latency and replacement, routes the
    /// outcome through the lifecycle, and publishes only gated states.
    fn handle_frame(&mut self, frame: &GnmWorkerFrameInput) {
        let solver_input = GnmSolverFrameInput::new(
            frame.stamp(),
            frame.observation(),
            None,
            &self.resources.identity,
            &self.resources.model,
            &self.resources.mapping,
        );
        let solver_input = match solver_input {
            Ok(solver_input) => solver_input,
            Err(error @ GnmFitterContractError::SourceSequenceMismatch { .. })
            | Err(error @ GnmFitterContractError::CaptureTimestampMismatch { .. }) => {
                // Upstream fan-out bug: typed, counted, never fatal.
                let _ = error;
                self.metrics.contract_errors += 1;
                return;
            }
            Err(error) => {
                let _ = error;
                self.metrics.contract_errors += 1;
                return;
            }
        };

        let overwritten_before = self.input.overwritten_count();
        let started = Instant::now();
        let outcome = self.fitter.fit_frame(
            &solver_input,
            &self.resources.model,
            &self.resources.mapping,
            self.solver_config,
            None,
        );
        let elapsed = started.elapsed();
        self.metrics.last_fit_latency_micros = elapsed.as_micros().min(u128::from(u64::MAX)) as u64;
        self.metrics.max_fit_latency_micros = self
            .metrics
            .max_fit_latency_micros
            .max(self.metrics.last_fit_latency_micros);
        self.metrics.replaced_during_fit += self
            .input
            .overwritten_count()
            .saturating_sub(overwritten_before);

        match outcome {
            Ok(PersistentGnmFrameOutcome::SkippedUncalibrated) => {
                self.metrics.skipped_uncalibrated_frames += 1;
            }
            Ok(PersistentGnmFrameOutcome::NoObservation { .. }) => {
                // Lost/dropped observation: a normal tracking state. The
                // previously published state stays visible downstream; the
                // lifecycle decides internally whether dynamic state cleared.
                self.metrics.no_observation_frames += 1;
            }
            Ok(PersistentGnmFrameOutcome::Solved(report)) => {
                self.metrics.solved_frames += 1;
                self.metrics.last_fit_iterations = report.iterations;
                if report.published
                    && let Some(validated) = self.fitter.validated()
                {
                    let state = GnmFaceState {
                        stamp: validated.stamp(),
                        dynamic: validated.dynamic().clone(),
                    };
                    if !self.output.publish(state) {
                        self.metrics.internal_errors += 1;
                    }
                    self.metrics.published_frames += 1;
                } else {
                    self.metrics.invalid_fits += 1;
                }
            }
            Err(PersistentGnmFitterError::Contract(_))
            | Err(PersistentGnmFitterError::MissingWarmStartState { .. }) => {
                self.metrics.contract_errors += 1;
            }
            Err(PersistentGnmFitterError::Solve(_)) => {
                self.metrics.solve_errors += 1;
            }
            Err(PersistentGnmFitterError::Lifecycle(_))
            | Err(PersistentGnmFitterError::UnexpectedLifecycleAction { .. }) => {
                self.metrics.internal_errors += 1;
            }
        }
    }

    /// Publishes the current metrics snapshot.
    fn publish_metrics(&self) {
        let _ = self.metrics_slot.publish(self.metrics.clone());
    }
}

/// Spawns the worker on a named thread with cooperative stop handling.
///
/// The returned handle owns the join handle; dropping it without joining
/// does not detach the thread (see [`WorkerHandle`]). Spawn failure surfaces
/// through [`WorkerHandle::join`] as [`vtuber_core::WorkerResult::SpawnFailed`].
#[must_use]
pub fn spawn_gnm_latest_frame_worker(
    name: impl Into<String>,
    mut worker: GnmLatestFrameWorker,
) -> WorkerHandle<()> {
    WorkerHandle::spawn(name, move |stop| {
        worker.run(&stop);
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // tests may panic (AGENTS.md)

    use std::sync::Arc;

    use vtuber_core::{LatestSlot, ReadResult};
    use vtuber_gnm::{
        DenseCoveragePolicy, DenseProjection, FixedGnmIdentity, GnmDenseObservation,
        GnmExpressionState, GnmFrameStamp, GnmJointState, SynthesisOptions,
        synthesize_observation_from_projection,
    };

    use super::*;
    use crate::gnm_sequence_regression::{synthetic_head_model, synthetic_mapping};

    const FRAME_MICROS: u64 = 16_667;

    type TestWorkerParts = (
        GnmLatestFrameWorker,
        GnmFitterResources,
        Arc<LatestSlot<GnmWorkerInput>>,
        Arc<LatestSlot<GnmFaceState>>,
        Arc<LatestSlot<GnmFitterWorkerMetrics>>,
    );

    fn resources() -> (GnmFitterResources, PersistentGnmLifecycleConfig) {
        let model = Arc::new(synthetic_head_model().unwrap());
        let mapping = Arc::new(synthetic_mapping(&model).unwrap());
        let identity = Arc::new(FixedGnmIdentity::new(model.neutral_identity(), &model).unwrap());
        let lifecycle_config = PersistentGnmLifecycleConfig::new(50_000, 250_000, 3).unwrap();
        (
            GnmFitterResources {
                model,
                mapping,
                identity,
            },
            lifecycle_config,
        )
    }

    fn worker() -> TestWorkerParts {
        let (res, lifecycle) = resources();
        let input = Arc::new(LatestSlot::new());
        let output = Arc::new(LatestSlot::new());
        let metrics_slot = Arc::new(LatestSlot::new());
        let solver_config = SingleFrameFitConfig::new(
            vtuber_gnm::DenseRigidStepConfig::default(),
            vtuber_gnm::DenseExpressionJointStepConfig::default(),
            16,
            1.0e-6,
        )
        .unwrap();
        let worker = GnmLatestFrameWorker::new(
            res.clone(),
            solver_config,
            lifecycle,
            Arc::clone(&input),
            Arc::clone(&output),
            Arc::clone(&metrics_slot),
        );
        (worker, res, input, output, metrics_slot)
    }

    fn stamped_observation(
        resources: &GnmFitterResources,
        seq: u64,
        micros: u64,
        mouth: f32,
    ) -> Arc<GnmDenseObservation> {
        let mut values = vec![0.0_f32; resources.model.expression_dimension()];
        values[0] = mouth;
        let len = values.len();
        let expression = GnmExpressionState::new(values, len).unwrap();
        let projection =
            DenseProjection::new([0.15, -0.10, 0.05], [0.02, -0.03, 0.60], 1.3, [0.5, 0.5])
                .unwrap();
        Arc::new(
            synthesize_observation_from_projection(
                &resources.model,
                resources.identity.state(),
                &expression,
                &GnmJointState::neutral(resources.model.joint_count()),
                &resources.mapping,
                &projection,
                SynthesisOptions {
                    source_seq: seq,
                    captured_at_micros: micros,
                    ..SynthesisOptions::default()
                },
                DenseCoveragePolicy::new(2, 0.75).unwrap(),
                |_, _| false,
            )
            .unwrap(),
        )
    }

    fn dropout_observation(
        resources: &GnmFitterResources,
        seq: u64,
        micros: u64,
    ) -> Arc<GnmDenseObservation> {
        let landmarks = vec![[f32::NAN; 2]; vtuber_gnm::MEDIAPIPE_FACE_LANDMARK_COUNT];
        Arc::new(
            GnmDenseObservation::from_mediapipe_xy(
                seq,
                micros,
                &landmarks,
                &resources.mapping,
                DenseCoveragePolicy::new(2, 0.75).unwrap(),
            )
            .unwrap(),
        )
    }

    fn frame(resources: &GnmFitterResources, seq: u64) -> GnmWorkerInput {
        GnmWorkerInput::Frame(GnmWorkerFrameInput::new(
            GnmFrameStamp {
                source_seq: seq,
                captured_at_micros: seq * FRAME_MICROS,
            },
            stamped_observation(resources, seq, seq * FRAME_MICROS, 0.5),
        ))
    }

    fn latest_output(output: &LatestSlot<GnmFaceState>) -> Option<GnmFaceState> {
        match output.try_read_after(0) {
            Some(ReadResult::New(state)) => Some(state),
            _ => None,
        }
    }

    #[test]
    fn publishes_only_lifecycle_gated_states_with_matching_stamps() {
        let (mut worker, res, input, output, _metrics_slot) = worker();

        // Uncalibrated frames are skipped, never published.
        input.publish(frame(&res, 1));
        assert_eq!(worker.step_once(), GnmWorkerStep::Processed);
        assert!(latest_output(&output).is_none());

        // Calibration unlocks the tracking path.
        input.publish(GnmWorkerInput::CalibrationReady);
        assert_eq!(worker.step_once(), GnmWorkerStep::Processed);

        for seq in 1_u64..=3 {
            input.publish(frame(&res, seq));
            assert_eq!(worker.step_once(), GnmWorkerStep::Processed);
        }
        let state = latest_output(&output).expect("tracking states published");
        assert_eq!(state.stamp().source_seq, 3);
        let metrics = worker.metrics();
        assert_eq!(metrics.published_frames, 3);
        assert_eq!(metrics.solved_frames, 3);
        assert_eq!(metrics.skipped_uncalibrated_frames, 1);
        assert!(metrics.last_fit_iterations >= 1);
    }

    #[test]
    fn stale_backlog_is_never_processed_latest_value_wins() {
        let (mut worker, res, input, output, _metrics_slot) = worker();
        input.publish(GnmWorkerInput::CalibrationReady);
        worker.step_once();

        // Two frames published before the worker runs: the first is already
        // overwritten inside the slot and must never reach the fitter.
        // The slot retains the last value after it is read (generation
        // gating provides freshness), so both publishes count as
        // overwrites: the second replaces the still-unread first.
        let baseline_overwritten = input.overwritten_count();
        input.publish(frame(&res, 1));
        input.publish(frame(&res, 2));
        assert_eq!(input.overwritten_count() - baseline_overwritten, 2);
        assert_eq!(worker.step_once(), GnmWorkerStep::Processed);

        let metrics = worker.metrics();
        assert_eq!(metrics.solved_frames, 1);
        assert_eq!(latest_output(&output).unwrap().stamp().source_seq, 2);
        // The latest frame was consumed; nothing else may run.
        assert_eq!(worker.step_once(), GnmWorkerStep::Idle);
        assert_eq!(worker.metrics().solved_frames, 1);
    }

    #[test]
    fn no_observation_never_bypasses_published_authority() {
        let (mut worker, res, input, output, _metrics_slot) = worker();
        input.publish(GnmWorkerInput::CalibrationReady);
        worker.step_once();

        input.publish(frame(&res, 1));
        worker.step_once();
        let published = latest_output(&output).unwrap();
        assert_eq!(published.stamp().source_seq, 1);

        // Dropout frame: normal tracking state; previous publication stays.
        input.publish(GnmWorkerInput::Frame(GnmWorkerFrameInput::new(
            GnmFrameStamp {
                source_seq: 2,
                captured_at_micros: 2 * FRAME_MICROS,
            },
            dropout_observation(&res, 2, 2 * FRAME_MICROS),
        )));
        worker.step_once();
        assert_eq!(latest_output(&output), Some(published));
        assert_eq!(worker.metrics().no_observation_frames, 1);
        assert_eq!(worker.metrics().published_frames, 1);
    }

    #[test]
    fn calibration_invalidation_clears_published_state() {
        let (mut worker, res, input, output, _metrics_slot) = worker();
        input.publish(GnmWorkerInput::CalibrationReady);
        worker.step_once();
        input.publish(frame(&res, 1));
        worker.step_once();
        assert!(latest_output(&output).is_some());

        input.publish(GnmWorkerInput::CalibrationInvalidated);
        worker.step_once();
        assert!(latest_output(&output).is_none());
        // Frames are skipped again until recalibration.
        input.publish(frame(&res, 2));
        worker.step_once();
        assert!(latest_output(&output).is_none());
        assert_eq!(worker.metrics().skipped_uncalibrated_frames, 1);
    }

    #[test]
    fn contract_mismatch_is_counted_and_non_fatal() {
        let (mut worker, res, input, output, _metrics_slot) = worker();
        input.publish(GnmWorkerInput::CalibrationReady);
        worker.step_once();

        // Stamp claims sequence 9 while the observation carries sequence 1.
        input.publish(GnmWorkerInput::Frame(GnmWorkerFrameInput::new(
            GnmFrameStamp {
                source_seq: 9,
                captured_at_micros: FRAME_MICROS,
            },
            stamped_observation(&res, 1, FRAME_MICROS, 0.5),
        )));
        worker.step_once();
        assert_eq!(worker.metrics().contract_errors, 1);
        assert!(latest_output(&output).is_none());

        // The next well-formed frame still works.
        input.publish(frame(&res, 1));
        worker.step_once();
        assert_eq!(worker.metrics().published_frames, 1);
    }

    #[test]
    fn long_gap_reacquires_and_republishes_fresh_authority() {
        let (mut worker, res, input, output, _metrics_slot) = worker();
        input.publish(GnmWorkerInput::CalibrationReady);
        worker.step_once();

        input.publish(frame(&res, 1));
        worker.step_once();

        // A frame far beyond the reuse age forces a reacquire path.
        let gap_stamp = GnmFrameStamp {
            source_seq: 2,
            captured_at_micros: FRAME_MICROS + 500_000,
        };
        input.publish(GnmWorkerInput::Frame(GnmWorkerFrameInput::new(
            gap_stamp,
            stamped_observation(&res, 2, gap_stamp.captured_at_micros, 0.5),
        )));
        worker.step_once();

        let state = latest_output(&output).expect("reacquired state published");
        assert_eq!(state.stamp().source_seq, 2);
        assert_eq!(worker.metrics().published_frames, 2);
    }
}
