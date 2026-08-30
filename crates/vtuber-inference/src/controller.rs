//! Controller for the inference worker.
//!
//! The controller lives on the application main thread. It spawns a single
//! inference worker thread that owns the model runtime. Frames are consumed
//! from a [`LatestSlot<VideoFrame>`] and results are published to a
//! [`LatestSlot<RawFaceObservation>`].

use std::sync::Arc;

use vtuber_core::types::{RawFaceObservation, VideoFrame};
use vtuber_core::{FaceTrackingOutcome, LatestSlot, WorkerHandle, WorkerResult};

use crate::descriptor::{FacePipelineDescriptor, ModelDescriptor, RuntimeSettings};
use crate::error::InferenceError;
use crate::metrics::InferenceMetrics;
use crate::runtime::InferenceOutcome;
use crate::state::{FailureStage, InferenceWorkerState, SharedStatus};
use crate::worker::run_inference_worker;

/// Capacity of the control channel between controller and worker.
const CONTROL_CHANNEL_CAPACITY: usize = 8;

/// Where the MediaPipe Face Landmarker task bundle comes from.
#[derive(Clone, Debug, PartialEq)]
pub enum MediaPipeTaskSource {
    /// Load and SHA-256-verify the task bundle at this filesystem path.
    Path(std::path::PathBuf),
    /// Use the approved task bundle embedded in the binary at compile time.
    Embedded,
}

/// Control commands sent from [`InferenceController`] to the inference worker.
#[derive(Clone, Debug, PartialEq)]
pub enum ControlCommand {
    /// Load the model described by `descriptor` using `settings`.
    LoadModel {
        /// Model descriptor. Boxed to keep the enum small.
        descriptor: Box<ModelDescriptor>,
        /// Runtime settings.
        settings: RuntimeSettings,
    },
    /// Load the manifest-resolved production detector-to-landmark pipeline.
    LoadPipeline {
        /// Runtime-free composite pipeline descriptor.
        descriptor: Box<FacePipelineDescriptor>,
        /// Directory containing the descriptor's relative artifact paths.
        artifact_root: std::path::PathBuf,
        /// Runtime settings, including detector cadence.
        settings: RuntimeSettings,
    },
    /// Load the approved MediaPipe Face Landmarker task in VIDEO mode.
    LoadMediaPipe {
        /// SHA-256-verified task bundle source.
        task: MediaPipeTaskSource,
    },
    /// Stop inference but keep the worker alive.
    Pause,
    /// Resume inference after a pause.
    Resume,
    /// Reset the worker to an idle state.
    Reset,
    /// Test-only: intentionally panic the worker.
    #[cfg(test)]
    Panic,
}

/// Controller for the inference worker.
pub struct InferenceController {
    pub(crate) status: SharedStatus,
    pub(crate) command_tx: Option<std::sync::mpsc::SyncSender<ControlCommand>>,
    pub(crate) frame_slot: Arc<LatestSlot<VideoFrame>>,
    pub(crate) output_slot: Arc<LatestSlot<RawFaceObservation>>,
    pub(crate) outcome_slot: Arc<LatestSlot<InferenceOutcome>>,
    pub(crate) canonical_outcome_slot: Arc<LatestSlot<FaceTrackingOutcome>>,
    pub(crate) worker: Option<WorkerHandle<InferenceWorkerResult>>,
}

impl Drop for InferenceController {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            worker.stop();
            self.frame_slot.close();
            let _ = worker.join();
        }
        self.output_slot.close();
        self.outcome_slot.close();
        self.canonical_outcome_slot.close();
    }
}

/// Result returned by the inference worker when it finishes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InferenceWorkerResult {
    /// Final metrics captured by the worker.
    pub final_metrics: InferenceMetrics,
}

impl InferenceController {
    /// Creates a new controller in the idle state.
    #[must_use]
    pub fn new(
        frame_slot: Arc<LatestSlot<VideoFrame>>,
        output_slot: Arc<LatestSlot<RawFaceObservation>>,
    ) -> Self {
        let outcome_slot = Arc::new(LatestSlot::new());
        let canonical_outcome_slot = Arc::new(LatestSlot::new());
        Self {
            status: Arc::new(std::sync::Mutex::new(
                crate::state::InferenceWorkerStatus::new(),
            )),
            command_tx: None,
            frame_slot,
            output_slot,
            outcome_slot,
            canonical_outcome_slot,
            worker: None,
        }
    }

    /// Returns the capacity-one slot used for frame transport.
    #[must_use]
    pub fn frame_slot(&self) -> Arc<LatestSlot<VideoFrame>> {
        Arc::clone(&self.frame_slot)
    }

    /// Returns the capacity-one slot used for inference output transport.
    #[must_use]
    pub fn output_slot(&self) -> Arc<LatestSlot<RawFaceObservation>> {
        Arc::clone(&self.output_slot)
    }

    /// Returns the latest-only face/no-face outcome slot.
    #[must_use]
    pub fn outcome_slot(&self) -> Arc<LatestSlot<InferenceOutcome>> {
        Arc::clone(&self.outcome_slot)
    }

    /// Returns the latest-only canonical MediaPipe face/no-face outcome slot.
    #[must_use]
    pub fn canonical_outcome_slot(&self) -> Arc<LatestSlot<FaceTrackingOutcome>> {
        Arc::clone(&self.canonical_outcome_slot)
    }

    /// Returns a snapshot of the current worker status.
    #[must_use]
    pub fn status(&self) -> crate::state::InferenceWorkerStatus {
        if self.worker.as_ref().is_some_and(WorkerHandle::is_finished) {
            let mut status = self
                .status
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if matches!(
                status.state,
                InferenceWorkerState::LoadingModel | InferenceWorkerState::Running
            ) {
                status.record_failure(FailureStage::WorkerPanic, InferenceError::WorkerPanicked);
            }
        }
        self.status
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Starts the inference worker.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::AlreadyRunning`] if the worker is already started.
    pub fn start_worker(&mut self) -> Result<(), InferenceError> {
        if self.worker.is_some() {
            return Err(InferenceError::AlreadyRunning);
        }

        let (tx, rx) = std::sync::mpsc::sync_channel::<ControlCommand>(CONTROL_CHANNEL_CAPACITY);
        self.command_tx = Some(tx.clone());

        let status = Arc::clone(&self.status);
        let frame_slot = Arc::clone(&self.frame_slot);
        let output_slot = Arc::clone(&self.output_slot);
        let outcome_slot = Arc::clone(&self.outcome_slot);
        let canonical_outcome_slot = Arc::clone(&self.canonical_outcome_slot);

        let worker = WorkerHandle::spawn("inference-worker", move |stop| {
            run_inference_worker(
                rx,
                stop,
                status,
                frame_slot,
                output_slot,
                outcome_slot,
                canonical_outcome_slot,
            )
        });

        self.worker = Some(worker);
        Ok(())
    }

    /// Loads a model described by `descriptor` with the given runtime settings.
    ///
    /// The model runtime is constructed inside the worker; the descriptor is
    /// plain data and can safely cross thread boundaries.
    ///
    /// # Errors
    ///
    /// Returns an error if the worker has not been started or the command
    /// channel has been closed.
    pub fn load_model(
        &mut self,
        descriptor: ModelDescriptor,
        settings: RuntimeSettings,
    ) -> Result<(), InferenceError> {
        let tx = self
            .command_tx
            .as_ref()
            .ok_or_else(|| InferenceError::Internal("inference worker not started".into()))?;

        {
            let mut status = self
                .status
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            status.transition_to(InferenceWorkerState::LoadingModel);
        }

        tx.send(ControlCommand::LoadModel {
            descriptor: Box::new(descriptor),
            settings,
        })
        .map_err(|_| InferenceError::Internal("inference worker command channel closed".into()))?;
        Ok(())
    }

    /// Loads the manifest-resolved production pipeline.
    ///
    /// Both detector and landmark runtimes are constructed in the worker
    /// thread from this runtime-free descriptor.
    pub fn load_pipeline(
        &mut self,
        descriptor: FacePipelineDescriptor,
        artifact_root: std::path::PathBuf,
        settings: RuntimeSettings,
    ) -> Result<(), InferenceError> {
        let tx = self
            .command_tx
            .as_ref()
            .ok_or_else(|| InferenceError::Internal("inference worker not started".into()))?;

        {
            let mut status = self
                .status
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            status.transition_to(InferenceWorkerState::LoadingModel);
        }

        tx.send(ControlCommand::LoadPipeline {
            descriptor: Box::new(descriptor),
            artifact_root,
            settings,
        })
        .map_err(|_| InferenceError::Internal("inference worker command channel closed".into()))?;
        Ok(())
    }

    /// Loads the approved MediaPipe Face Landmarker task from `task_path`.
    ///
    /// The task and native runtime are both constructed inside the worker;
    /// only the plain path crosses the controller boundary.
    pub fn load_mediapipe(&mut self, task_path: std::path::PathBuf) -> Result<(), InferenceError> {
        self.load_mediapipe_task(MediaPipeTaskSource::Path(task_path))
    }

    /// Loads the approved MediaPipe Face Landmarker task embedded in the
    /// binary (`backend::mediapipe::embedded_task_bundle`).
    ///
    /// The task and native runtime are both constructed inside the worker.
    pub fn load_mediapipe_embedded(&mut self) -> Result<(), InferenceError> {
        self.load_mediapipe_task(MediaPipeTaskSource::Embedded)
    }

    fn load_mediapipe_task(&mut self, task: MediaPipeTaskSource) -> Result<(), InferenceError> {
        let tx = self
            .command_tx
            .as_ref()
            .ok_or_else(|| InferenceError::Internal("inference worker not started".into()))?;

        {
            let mut status = self
                .status
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            status.transition_to(InferenceWorkerState::LoadingModel);
        }

        tx.send(ControlCommand::LoadMediaPipe { task })
            .map_err(|_| {
                InferenceError::Internal("inference worker command channel closed".into())
            })?;
        Ok(())
    }

    /// Pauses inference without stopping the worker.
    pub fn pause(&mut self) {
        if let Some(tx) = self.command_tx.as_ref() {
            let _ = tx.send(ControlCommand::Pause);
        }
    }

    /// Resumes inference after a pause.
    pub fn resume(&mut self) {
        if let Some(tx) = self.command_tx.as_ref() {
            let _ = tx.send(ControlCommand::Resume);
        }
    }

    /// Resets the worker to idle, releasing any loaded model.
    pub fn reset(&mut self) {
        if let Some(tx) = self.command_tx.as_ref() {
            let _ = tx.send(ControlCommand::Reset);
        }
        {
            let mut status = self
                .status
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            status.transition_to(InferenceWorkerState::Idle);
        }
    }

    /// Requests graceful shutdown and joins the worker.
    ///
    /// This consumes the controller. After this call returns, no worker thread
    /// is running and the output slot is closed.
    pub fn shutdown(self) -> InferenceMetrics {
        self.shutdown_inner(true)
    }

    /// Stops and joins the worker without closing the externally-owned input
    /// frame slot. This is used when capture and inference share one slot and
    /// the application must support Stop/Start without rebuilding capture.
    pub fn shutdown_preserving_input(self) -> InferenceMetrics {
        self.shutdown_inner(false)
    }

    fn shutdown_inner(mut self, close_input: bool) -> InferenceMetrics {
        let result = if let Some(worker) = self.worker.take() {
            worker.stop();
            // Closing the frame slot wakes the worker if it is blocked waiting
            // for a frame, so shutdown completes promptly.
            if close_input {
                self.frame_slot.close();
            }
            worker.join()
        } else {
            WorkerResult::Completed(InferenceWorkerResult::default())
        };

        let final_metrics = match result {
            WorkerResult::Completed(r) => r.final_metrics,
            WorkerResult::Panicked => {
                let mut status = self.status.lock().unwrap_or_else(|e| e.into_inner());
                status.record_failure(FailureStage::WorkerPanic, InferenceError::WorkerPanicked);
                status.metrics()
            }
            WorkerResult::SpawnFailed => self
                .status
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .metrics(),
        };

        self.output_slot.close();
        self.outcome_slot.close();
        self.canonical_outcome_slot.close();
        final_metrics
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )] // tests may panic (AGENTS.md)

    use super::*;
    use std::time::Duration;

    use vtuber_core::types::{FrameSeq, MonoTimeNs, PixelFormat};

    use crate::descriptor::{
        ChannelOrder, ModelDescriptor, ModelFormat, Normalization, RuntimeSettings,
    };
    use crate::state::{FailureStage, InferenceWorkerState};

    fn dummy_descriptor() -> ModelDescriptor {
        ModelDescriptor {
            id: "dummy".into(),
            format: ModelFormat::Tflite,
            path: std::path::PathBuf::from("/dev/null"),
            sha256: "0000".into(),
            input_name: "input".into(),
            input_shape: vec![1, 256, 256, 3],
            input_dtype: "f32".into(),
            channel_order: ChannelOrder::Rgb,
            normalization: Normalization::ZeroToOne,
            schema: vtuber_core::types::LandmarkSchemaId("dummy"),
            expression_mapping: None,
        }
    }

    fn dummy_frame(seq: u64) -> VideoFrame {
        VideoFrame {
            seq: FrameSeq(seq),
            captured_at: MonoTimeNs(seq * 1_000_000),
            width: 64,
            height: 64,
            stride_bytes: 64 * 3,
            format: PixelFormat::Rgb8,
            data: vec![0u8; 64 * 64 * 3].into(),
        }
    }

    #[test]
    fn worker_state_transitions_idle_loading_failed() {
        let frame_slot: Arc<LatestSlot<VideoFrame>> = Arc::new(LatestSlot::new());
        let output_slot: Arc<LatestSlot<RawFaceObservation>> = Arc::new(LatestSlot::new());
        let mut controller =
            InferenceController::new(Arc::clone(&frame_slot), Arc::clone(&output_slot));

        assert_eq!(controller.status().state, InferenceWorkerState::Idle);

        controller.start_worker().expect("start worker");
        assert!(controller.worker.is_some());

        controller
            .load_model(dummy_descriptor(), RuntimeSettings::default())
            .expect("send load model");

        // Wait for the worker to process the load command and fail.
        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(controller.status().state, InferenceWorkerState::Failed);
        assert_eq!(
            controller.status().last_failure.as_ref().map(|f| f.stage),
            Some(FailureStage::ModelLoad)
        );

        controller.shutdown();
    }

    #[test]
    fn controller_does_not_start_twice() {
        let frame_slot: Arc<LatestSlot<VideoFrame>> = Arc::new(LatestSlot::new());
        let output_slot: Arc<LatestSlot<RawFaceObservation>> = Arc::new(LatestSlot::new());
        let mut controller =
            InferenceController::new(Arc::clone(&frame_slot), Arc::clone(&output_slot));

        controller.start_worker().expect("first start");
        assert_eq!(
            controller.start_worker().unwrap_err(),
            InferenceError::AlreadyRunning
        );
        controller.shutdown();
    }

    #[test]
    fn worker_drops_output_when_slot_full() {
        let frame_slot: Arc<LatestSlot<VideoFrame>> = Arc::new(LatestSlot::new());
        let output_slot: Arc<LatestSlot<RawFaceObservation>> = Arc::new(LatestSlot::new());
        let mut controller =
            InferenceController::new(Arc::clone(&frame_slot), Arc::clone(&output_slot));

        controller.start_worker().expect("start worker");
        controller
            .load_model(dummy_descriptor(), RuntimeSettings::default())
            .expect("send load model");
        std::thread::sleep(Duration::from_millis(50));

        // Even though runtime load fails, the output slot remains closed on shutdown.
        controller.shutdown();
        assert!(output_slot.is_closed());
    }

    #[test]
    fn pause_prevents_frame_processing() {
        let frame_slot: Arc<LatestSlot<VideoFrame>> = Arc::new(LatestSlot::new());
        let output_slot: Arc<LatestSlot<RawFaceObservation>> = Arc::new(LatestSlot::new());
        let mut controller =
            InferenceController::new(Arc::clone(&frame_slot), Arc::clone(&output_slot));

        controller.start_worker().expect("start worker");
        controller.pause();

        // Publish a frame while paused; it should remain unread because no
        // runtime is loaded and the worker is paused.
        frame_slot.publish(dummy_frame(1));
        std::thread::sleep(Duration::from_millis(120));
        assert_eq!(controller.status().frames_processed, 0);

        controller.shutdown();
    }

    #[test]
    fn reset_returns_to_idle() {
        let frame_slot: Arc<LatestSlot<VideoFrame>> = Arc::new(LatestSlot::new());
        let output_slot: Arc<LatestSlot<RawFaceObservation>> = Arc::new(LatestSlot::new());
        let mut controller =
            InferenceController::new(Arc::clone(&frame_slot), Arc::clone(&output_slot));

        controller.start_worker().expect("start worker");
        controller.reset();
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(controller.status().state, InferenceWorkerState::Idle);
        controller.shutdown();
    }

    #[test]
    fn inference_shutdown() {
        let frame_slot: Arc<LatestSlot<VideoFrame>> = Arc::new(LatestSlot::new());
        let output_slot: Arc<LatestSlot<RawFaceObservation>> = Arc::new(LatestSlot::new());
        let mut controller =
            InferenceController::new(Arc::clone(&frame_slot), Arc::clone(&output_slot));
        let canonical_outcome_slot = controller.canonical_outcome_slot();

        controller.start_worker().expect("start worker");

        // Leave the worker waiting on an empty frame slot and shut down. The
        // worker must unblock, join, and leave both slots closed.
        let metrics = controller.shutdown();
        assert!(frame_slot.is_closed());
        assert!(output_slot.is_closed());
        assert!(canonical_outcome_slot.is_closed());
        assert_eq!(metrics.drops.processed, 0);
    }

    #[test]
    fn mediapipe_load_failure_is_reported_as_a_typed_model_error() {
        let frame_slot: Arc<LatestSlot<VideoFrame>> = Arc::new(LatestSlot::new());
        let output_slot: Arc<LatestSlot<RawFaceObservation>> = Arc::new(LatestSlot::new());
        let mut controller =
            InferenceController::new(Arc::clone(&frame_slot), Arc::clone(&output_slot));

        controller.start_worker().expect("start worker");
        controller
            .load_mediapipe(std::path::PathBuf::from("missing-face-landmarker.task"))
            .expect("send MediaPipe load command");
        std::thread::sleep(Duration::from_millis(150));

        let status = controller.status();
        assert_eq!(status.state, InferenceWorkerState::Failed);
        assert!(matches!(
            status.last_failure.as_ref().map(|failure| &failure.error),
            Some(InferenceError::MediaPipeLoadFailed(_))
        ));
        controller.shutdown();
    }

    #[test]
    fn worker_failure() {
        let frame_slot: Arc<LatestSlot<VideoFrame>> = Arc::new(LatestSlot::new());
        let output_slot: Arc<LatestSlot<RawFaceObservation>> = Arc::new(LatestSlot::new());
        let mut controller =
            InferenceController::new(Arc::clone(&frame_slot), Arc::clone(&output_slot));

        controller.start_worker().expect("start worker");

        let tx = controller
            .command_tx
            .as_ref()
            .expect("command channel exists after start");
        tx.send(ControlCommand::Panic)
            .expect("send panic command to worker");

        // Give the worker a moment to process the command and panic.
        std::thread::sleep(Duration::from_millis(100));

        let status = Arc::clone(&controller.status);
        controller.shutdown();

        let status = status.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(status.state, InferenceWorkerState::Failed);
        assert_eq!(
            status.last_failure.as_ref().map(|f| f.stage),
            Some(FailureStage::WorkerPanic)
        );
        assert!(
            matches!(
                status.last_failure.as_ref().map(|f| &f.error),
                Some(InferenceError::WorkerPanicked)
            ),
            "worker panic should be recorded as WorkerPanicked error, got {:?}",
            status.last_failure
        );
    }
}
