//! Public state and status types for the inference worker.
//!
//! These types are designed to be read from the Bevy main thread or other
//! observers. They contain no runtime objects and cross thread boundaries only
//! as plain data.

use std::sync::Arc;
use std::time::Duration;

use vtuber_core::types::{FrameSeq, MonoTimeNs, NormalizedRect};

use crate::error::InferenceError;
use crate::metrics::{InferenceMetrics, InferenceMetricsState, InferenceStage};

/// Lifecycle state of the inference worker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum InferenceWorkerState {
    /// Worker has been created but not yet started.
    #[default]
    Idle,
    /// Worker is loading and optimizing the model.
    LoadingModel,
    /// Worker is running inference on incoming frames.
    Running,
    /// Worker is stopping after a shutdown request.
    Stopping,
    /// Worker failed and will not process frames until restarted.
    Failed,
}

/// Snapshot of the inference worker status, safe to read from any thread.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InferenceWorkerStatus {
    /// Current lifecycle state.
    pub state: InferenceWorkerState,
    /// Sequence of the last source frame consumed, if any.
    pub last_source_seq: Option<FrameSeq>,
    /// Timestamp when the last inference finished, if any.
    pub last_finished_at: Option<MonoTimeNs>,
    /// Last inference duration, if measured.
    pub last_inference_duration: Option<Duration>,
    /// Total frames processed since worker start.
    pub frames_processed: u64,
    /// Total frames completed with the ordinary no-face result.
    pub no_face_frames: u64,
    /// Total frames dropped because the output slot could not accept them.
    pub frames_dropped: u64,
    /// Frames overwritten in the input slot before being read.
    pub frames_overwritten: u64,
    /// Frames suppressed because their source sequence was already processed.
    pub duplicate_frames_suppressed: u64,
    /// Last failure, separated by lifecycle stage.
    pub last_failure: Option<WorkerFailure>,
    /// Consecutive recoverable per-frame errors since the last success.
    pub consecutive_errors: u32,
    /// ROI from the most recent face outcome, if one was available.
    pub last_roi: Option<NormalizedRect>,
    /// Stable production pipeline identifier.
    pub pipeline_id: Option<String>,
    /// Detector model hash for diagnostics.
    pub detector_model_hash: Option<String>,
    /// Landmark model hash for diagnostics.
    pub landmark_model_hash: Option<String>,
    /// Detector confidence associated with the current ROI.
    pub detector_confidence: Option<f32>,
    /// Human-readable ROI lifecycle state.
    pub roi_state: Option<String>,
    metrics_state: InferenceMetricsState,
}

/// A failure that occurred at a specific worker lifecycle stage.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkerFailure {
    /// When the failure was observed.
    pub observed_at: MonoTimeNs,
    /// Lifecycle stage that produced the failure.
    pub stage: FailureStage,
    /// Typed error that occurred.
    pub error: InferenceError,
}

/// Lifecycle stage where a failure occurred.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FailureStage {
    /// Failure during model load or optimization.
    ModelLoad,
    /// Failure while running the full-frame detector.
    Detector,
    /// Failure while preparing the detector-selected crop.
    Crop,
    /// Failure while running the crop landmark model.
    Landmark,
    /// Failure during a single-frame inference (legacy aggregate stage).
    FrameInference,
    /// Failure while preprocessing a video frame.
    Preprocess,
    /// Failure during model runtime execution.
    Runtime,
    /// Failure while decoding model outputs into an observation.
    Decode,
    /// Failure because the input channel or frame slot closed unexpectedly.
    InputClosed,
    /// Failure because the worker thread panicked.
    WorkerPanic,
    /// Failure while shutting down.
    Shutdown,
}

/// Thread-safe shared worker status.
pub type SharedStatus = Arc<std::sync::Mutex<InferenceWorkerStatus>>;

impl InferenceWorkerStatus {
    /// Creates a new status snapshot in [`InferenceWorkerState::Idle`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a state transition.
    pub fn transition_to(&mut self, state: InferenceWorkerState) {
        self.state = state;
    }

    /// Returns a snapshot of the current inference metrics.
    #[must_use]
    pub fn metrics(&self) -> InferenceMetrics {
        self.metrics_state.snapshot()
    }

    /// Records a duration sample for an inference stage.
    pub fn record_stage_duration(&mut self, stage: InferenceStage, duration: Duration) {
        self.metrics_state.record_stage_duration(stage, duration);
    }

    /// Records a successful inference frame.
    pub fn record_processed(&mut self, seq: FrameSeq, finished_at: MonoTimeNs, duration: Duration) {
        self.last_source_seq = Some(seq);
        self.last_finished_at = Some(finished_at);
        self.last_inference_duration = Some(duration);
        self.frames_processed += 1;
        self.metrics_state.record_processed();
    }

    /// Records an ordinary no-face frame without treating it as a failure.
    pub fn record_no_face(&mut self, seq: FrameSeq, finished_at: MonoTimeNs, duration: Duration) {
        self.last_source_seq = Some(seq);
        self.last_finished_at = Some(finished_at);
        self.last_inference_duration = Some(duration);
        self.no_face_frames += 1;
        self.metrics_state.record_no_face();
    }

    /// Records a dropped frame.
    pub fn record_dropped(&mut self) {
        self.frames_dropped += 1;
    }

    /// Records an input slot overwrite.
    pub fn record_overwritten(&mut self, count: u64) {
        self.frames_overwritten += count;
        self.metrics_state.record_input_overwritten(count);
    }

    /// Records a frame that was skipped before inference.
    pub fn record_skipped_sequence(&mut self) {
        self.metrics_state.record_skipped_sequence();
    }

    /// Records a frame that was suppressed because its sequence was a duplicate.
    pub fn record_duplicate_suppressed(&mut self) {
        self.duplicate_frames_suppressed += 1;
        self.metrics_state.record_skipped_sequence();
    }

    /// Records frames overwritten in the output slot.
    pub fn record_output_overwritten(&mut self, count: u64) {
        self.metrics_state.record_output_overwritten(count);
    }

    /// Records a failure and transitions to [`InferenceWorkerState::Failed`].
    pub fn record_failure(&mut self, stage: FailureStage, error: InferenceError) {
        self.last_failure = Some(WorkerFailure {
            observed_at: MonoTimeNs(now_nanos()),
            stage,
            error,
        });
        self.state = InferenceWorkerState::Failed;
    }

    /// Records a recoverable per-frame error.
    ///
    /// The error is stored in [`Self::last_failure`] and the consecutive-error
    /// counter is incremented. Returns `true` when the counter exceeds the
    /// configured tolerance and the worker should transition to
    /// [`InferenceWorkerState::Failed`].
    pub fn record_frame_error(
        &mut self,
        stage: FailureStage,
        error: InferenceError,
        max_consecutive: u32,
    ) -> bool {
        self.consecutive_errors += 1;
        self.last_failure = Some(WorkerFailure {
            observed_at: MonoTimeNs(now_nanos()),
            stage,
            error,
        });
        if self.consecutive_errors > max_consecutive {
            self.state = InferenceWorkerState::Failed;
            true
        } else {
            false
        }
    }

    /// Resets the consecutive error counter, for example after a successful
    /// frame or a worker reset.
    pub fn clear_consecutive_errors(&mut self) {
        self.consecutive_errors = 0;
    }

    /// Records the latest source-image ROI, or clears it for a no-face result.
    pub fn set_last_roi(&mut self, roi: Option<NormalizedRect>) {
        self.last_roi = roi;
    }

    /// Records the manifest identity used by the active worker.
    pub fn set_pipeline_info(
        &mut self,
        pipeline_id: Option<String>,
        detector_model_hash: Option<String>,
        landmark_model_hash: Option<String>,
    ) {
        self.pipeline_id = pipeline_id;
        self.detector_model_hash = detector_model_hash;
        self.landmark_model_hash = landmark_model_hash;
    }

    /// Records composite ROI diagnostics.
    pub fn set_roi_diagnostics(&mut self, confidence: Option<f32>, state: Option<String>) {
        self.detector_confidence = confidence;
        self.roi_state = state;
    }
}

fn now_nanos() -> u64 {
    vtuber_core::monotonic_now().0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_starts_idle() {
        let status = InferenceWorkerStatus::new();
        assert_eq!(status.state, InferenceWorkerState::Idle);
        assert_eq!(status.frames_processed, 0);
    }

    #[test]
    fn transition_to_running() {
        let mut status = InferenceWorkerStatus::new();
        status.transition_to(InferenceWorkerState::LoadingModel);
        status.transition_to(InferenceWorkerState::Running);
        assert_eq!(status.state, InferenceWorkerState::Running);
    }

    #[test]
    fn record_processed_updates_counters() {
        let mut status = InferenceWorkerStatus::new();
        status.record_processed(FrameSeq(7), MonoTimeNs(1000), Duration::from_millis(16));
        assert_eq!(status.frames_processed, 1);
        assert_eq!(status.last_source_seq, Some(FrameSeq(7)));
        assert_eq!(status.last_finished_at, Some(MonoTimeNs(1000)));
    }

    #[test]
    fn record_no_face_is_not_a_failure() {
        let mut status = InferenceWorkerStatus::new();
        status.transition_to(InferenceWorkerState::Running);
        status.record_no_face(FrameSeq(8), MonoTimeNs(2000), Duration::from_millis(4));
        assert_eq!(status.no_face_frames, 1);
        assert_eq!(status.state, InferenceWorkerState::Running);
        assert_eq!(status.metrics().drops.no_face, 1);
        assert_eq!(status.last_source_seq, Some(FrameSeq(8)));
    }

    #[test]
    fn record_failure_sets_state_and_last_failure() {
        let mut status = InferenceWorkerStatus::new();
        status.transition_to(InferenceWorkerState::LoadingModel);
        let err = InferenceError::LoadFailed("missing file".into());
        status.record_failure(FailureStage::ModelLoad, err.clone());
        assert_eq!(status.state, InferenceWorkerState::Failed);
        assert!(matches!(
            status.last_failure,
            Some(WorkerFailure {
                stage: FailureStage::ModelLoad,
                error: InferenceError::LoadFailed(_),
                ..
            })
        ));
    }

    #[test]
    fn record_failure_distinguishes_stages() {
        let mut status = InferenceWorkerStatus::new();
        status.record_failure(
            FailureStage::FrameInference,
            InferenceError::ExecutionFailed("oops".into()),
        );
        assert_eq!(
            status.last_failure.as_ref().map(|f| f.stage),
            Some(FailureStage::FrameInference)
        );
    }
}
