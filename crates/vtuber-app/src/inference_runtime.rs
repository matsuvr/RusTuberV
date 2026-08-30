//! Application bridge for the inference worker.

use std::path::PathBuf;
use std::sync::Arc;

use bevy::prelude::*;
use vtuber_core::metrics::RateCounter;
use vtuber_core::{
    FaceTrackingOutcome, FaceTrackingSample, FrameSeq, LatestSlot, RawFaceObservation, VideoFrame,
    monotonic_now,
};
use vtuber_inference::InferenceStage;
use vtuber_inference::{InferenceController, InferenceWorkerState};

use crate::capture_runtime::CaptureRuntime;
use crate::diagnostics::DiagnosticsSnapshot;
use crate::orchestrator::{Orchestrator, OrchestratorError, PipelineState};

const MEDIAPIPE_TASK_FILE: &str = "face_landmarker.task";

/// Filesystem root containing the packaged `assets/models/manifest.toml`.
///
/// Desktop applications provide this from their resource locator. Tests and
/// development runs default to the workspace current directory.
#[derive(Resource, Clone, Debug, Default)]
pub struct InferenceProjectRoot(pub PathBuf);

/// Resource containing the inference controller and its latest output slot.
#[derive(Resource)]
pub struct InferenceRuntime {
    controller: InferenceController,
    frame_slot: Arc<LatestSlot<VideoFrame>>,
    project_root: PathBuf,
    worker_started: bool,
    model_requested: bool,
    canonical_outcome_generation: u64,
    /// Latest canonical MediaPipe sample made available to tracking.
    pub latest_face_sample: Option<FaceTrackingSample>,
    /// Deprecated legacy observation retained for compatibility diagnostics.
    pub latest_observation: Option<RawFaceObservation>,
}

impl InferenceRuntime {
    /// Creates an inference runtime sharing the capture frame slot.
    #[must_use]
    pub fn new(frame_slot: Arc<LatestSlot<VideoFrame>>, project_root: PathBuf) -> Self {
        let output_slot = Arc::new(LatestSlot::new());
        Self {
            controller: InferenceController::new(Arc::clone(&frame_slot), output_slot),
            frame_slot,
            project_root,
            worker_started: false,
            model_requested: false,
            canonical_outcome_generation: 0,
            latest_face_sample: None,
            latest_observation: None,
        }
    }

    /// Returns a status snapshot safe for diagnostics.
    #[must_use]
    pub fn status(&self) -> vtuber_inference::InferenceWorkerStatus {
        self.controller.status()
    }

    /// Returns the output slot used by tracking.
    #[must_use]
    pub fn output_slot(&self) -> Arc<LatestSlot<RawFaceObservation>> {
        self.controller.output_slot()
    }

    /// Starts the worker and queues the approved MediaPipe task bundle.
    pub fn start_model(&mut self) -> Result<(), String> {
        if !self.worker_started {
            self.controller
                .start_worker()
                .map_err(|error| error.to_string())?;
            self.worker_started = true;
        }
        if !self.model_requested {
            let task_path = self
                .project_root
                .join("assets")
                .join("models")
                .join(MEDIAPIPE_TASK_FILE);
            // Prefer the packaged resource bundle (dev runs and resource
            // packages); release builds started without the resource tree
            // fall back to the SHA-256-verified bundle embedded in the binary.
            let load_result = if task_path.is_file() {
                self.controller.load_mediapipe(task_path)
            } else {
                self.controller.load_mediapipe_embedded()
            };
            load_result.map_err(|error| error.to_string())?;
            self.model_requested = true;
        }
        Ok(())
    }

    /// Resets inference to idle for the next capture session.
    pub fn stop_model(&mut self) {
        if self.worker_started {
            let replacement =
                InferenceController::new(Arc::clone(&self.frame_slot), Arc::new(LatestSlot::new()));
            let controller = std::mem::replace(&mut self.controller, replacement);
            let _ = controller.shutdown_preserving_input();
            self.worker_started = false;
        }
        self.model_requested = false;
        self.canonical_outcome_generation = 0;
        self.latest_face_sample = None;
        self.latest_observation = None;
    }

    /// Reads one latest-only canonical MediaPipe result, suppressing duplicate output.
    pub fn read_latest(&mut self) -> Option<FaceTrackingSample> {
        let outcome_slot = self.controller.canonical_outcome_slot();
        if let Some(vtuber_core::ReadResult::New(outcome)) =
            outcome_slot.try_read_after(self.canonical_outcome_generation)
        {
            self.canonical_outcome_generation = outcome_slot.generation();
            match outcome {
                FaceTrackingOutcome::Face(sample) => {
                    self.latest_face_sample = Some(sample.clone());
                    return Some(sample);
                }
                FaceTrackingOutcome::NoFace { .. } => {
                    self.latest_face_sample = None;
                    self.latest_observation = None;
                }
            }
        }
        None
    }
}

/// Starts/stops inference in response to the capture-backed application state.
pub fn inference_bridge_system(
    capture: Res<CaptureRuntime>,
    mut inference: ResMut<InferenceRuntime>,
    mut orchestrator: ResMut<Orchestrator>,
) {
    if orchestrator.take_inference_retry_request() {
        // Retain the capture session and replace only the inference worker.
        // `stop_model` joins the old worker before the new one is started.
        inference.stop_model();
        if orchestrator.capture_desired() {
            orchestrator.set_pipeline_state(PipelineState::Starting);
            orchestrator.set_last_error(None);
        }
    }

    if orchestrator.capture_desired()
        && matches!(
            capture.state(),
            vtuber_camera::CaptureServiceState::Starting
                | vtuber_camera::CaptureServiceState::Running
        )
        && matches!(
            orchestrator.pipeline_state(),
            PipelineState::Starting | PipelineState::Running
        )
        && !matches!(inference.status().state, InferenceWorkerState::Running)
        && let Err(error) = inference.start_model()
    {
        orchestrator.set_pipeline_state(PipelineState::Failed);
        orchestrator.set_last_error(Some(OrchestratorError::InferenceFailed(error)));
    }

    if !orchestrator.capture_desired()
        && matches!(
            orchestrator.pipeline_state(),
            PipelineState::Stopping | PipelineState::Failed
        )
    {
        inference.stop_model();
    }

    if orchestrator.capture_desired() {
        let status = inference.status();
        match status.state {
            InferenceWorkerState::Failed => {
                let message = status
                    .last_failure
                    .map(|failure| failure.error.to_string())
                    .unwrap_or_else(|| "inference worker failed".into());
                // A failed worker cannot be reused. Request the normal
                // inference-first, capture-second shutdown so a partial
                // runtime is never left behind.
                orchestrator.fail_inference(message);
            }
            InferenceWorkerState::Running
                if capture.state() == vtuber_camera::CaptureServiceState::Running
                    && orchestrator.pipeline_state() == PipelineState::Starting =>
            {
                orchestrator.set_pipeline_state(PipelineState::Running);
            }
            _ => {}
        }
    }
}

/// Local state used to derive the inference, detector, and landmark rates.
#[derive(Default)]
pub(crate) struct InferenceRateState {
    last_seq: Option<FrameSeq>,
    rate: Option<RateCounter>,
    detector_last_count: u64,
    detector_rate: Option<RateCounter>,
    landmark_last_count: u64,
    landmark_rate: Option<RateCounter>,
}

/// Reads the latest observation and updates worker diagnostics.
pub(crate) fn read_inference_output_system(
    mut inference: ResMut<InferenceRuntime>,
    mut diagnostics: ResMut<DiagnosticsSnapshot>,
    mut rates: Local<InferenceRateState>,
) {
    let _ = inference.read_latest();
    let status = inference.status();
    let new_observation = match (status.last_source_seq, status.last_finished_at) {
        (Some(seq), Some(finished_at)) if rates.last_seq != Some(seq) => Some((seq, finished_at.0)),
        _ => None,
    };
    if let Some((seq, finished_at)) = new_observation {
        rates
            .rate
            .get_or_insert_with(|| RateCounter::new(1_000_000_000))
            .record(finished_at);
        rates.last_seq = Some(seq);
    }
    let now = monotonic_now().0;
    diagnostics.inference_rate = rates
        .rate
        .get_or_insert_with(|| RateCounter::new(1_000_000_000))
        .rate_hz(now) as f32;
    diagnostics.inference_state = format!("{:?}", status.state);
    diagnostics.inference_last_source_seq = status.last_source_seq.map(|seq| seq.0);
    diagnostics.inference_frames_processed = status.frames_processed;
    diagnostics.inference_no_face_frames = status.no_face_frames;
    diagnostics.inference_duplicates_suppressed = status.duplicate_frames_suppressed;
    diagnostics.inference_input_overwrites = status.frames_overwritten;
    diagnostics.last_inference_ms = status
        .last_inference_duration
        .map(|duration| duration.as_secs_f32() * 1_000.0);
    diagnostics.inference_last_roi = status
        .last_roi
        .map(|roi| (roi.x, roi.y, roi.width, roi.height));
    diagnostics.detector_confidence = status.detector_confidence;
    diagnostics.roi_state = status.roi_state.clone();
    diagnostics.pipeline_id = status.pipeline_id.clone().or_else(|| {
        inference
            .latest_face_sample
            .as_ref()
            .map(|_| "mediapipe-face-landmarker".to_string())
    });
    diagnostics.model_hash = model_hash_summary(
        status.detector_model_hash.as_deref(),
        status.landmark_model_hash.as_deref(),
    )
    .or_else(|| {
        (status.pipeline_id.as_deref() == Some("mediapipe-face-landmarker")).then(|| {
            format!(
                "task:{}",
                short_hash(vtuber_inference::backend::mediapipe::TASK_BUNDLE_SHA256)
            )
        })
    });
    diagnostics.inference_failure_stage = status
        .last_failure
        .as_ref()
        .map(|failure| format!("{:?}", failure.stage));
    if let Some(failure) = status.last_failure.as_ref() {
        diagnostics.last_error_code = Some(inference_error_code(failure).to_string());
    }

    let metrics = status.metrics();
    let stage_time = status.last_finished_at.map_or(now, |time| time.0);
    let detector_count = metrics.stage(InferenceStage::Detector).count;
    let landmark_count = metrics.stage(InferenceStage::Landmark).count;
    if detector_count < rates.detector_last_count {
        rates
            .detector_rate
            .get_or_insert_with(|| RateCounter::new(1_000_000_000))
            .reset();
        rates.detector_last_count = 0;
    }
    if landmark_count < rates.landmark_last_count {
        rates
            .landmark_rate
            .get_or_insert_with(|| RateCounter::new(1_000_000_000))
            .reset();
        rates.landmark_last_count = 0;
    }
    let detector_delta = detector_count.saturating_sub(rates.detector_last_count);
    {
        let detector_counter = rates
            .detector_rate
            .get_or_insert_with(|| RateCounter::new(1_000_000_000));
        for _ in 0..detector_delta {
            detector_counter.record(stage_time);
        }
    }
    rates.detector_last_count = detector_count;
    let landmark_delta = landmark_count.saturating_sub(rates.landmark_last_count);
    {
        let landmark_counter = rates
            .landmark_rate
            .get_or_insert_with(|| RateCounter::new(1_000_000_000));
        for _ in 0..landmark_delta {
            landmark_counter.record(stage_time);
        }
    }
    rates.landmark_last_count = landmark_count;
    diagnostics.detector_rate = rates
        .detector_rate
        .as_mut()
        .map_or(0.0, |counter| counter.rate_hz(now) as f32);
    diagnostics.landmark_rate = rates
        .landmark_rate
        .as_mut()
        .map_or(0.0, |counter| counter.rate_hz(now) as f32);
    diagnostics.stage_timings = InferenceStage::ALL
        .into_iter()
        .filter_map(|stage| {
            let timing = metrics.stage(stage);
            (timing.count > 0).then(|| {
                (
                    format!("inference_{stage:?}_mean"),
                    timing.mean_ns as f32 / 1_000_000.0,
                )
            })
        })
        .collect();
    diagnostics.stage_percentiles = InferenceStage::ALL
        .into_iter()
        .filter_map(|stage| {
            let timing = metrics.stage(stage);
            (timing.count > 0).then(|| {
                (
                    format!("inference_{stage:?}"),
                    timing.p50_ns as f32 / 1_000_000.0,
                    timing.p95_ns as f32 / 1_000_000.0,
                )
            })
        })
        .collect();
    if let Some(failure) = status.last_failure.as_ref() {
        diagnostics.last_error = Some(failure.error.to_string());
    }
}

fn model_hash_summary(detector: Option<&str>, landmark: Option<&str>) -> Option<String> {
    match (detector, landmark) {
        (Some(detector), Some(landmark)) => Some(format!(
            "det:{} lm:{}",
            short_hash(detector),
            short_hash(landmark)
        )),
        (Some(detector), None) => Some(format!("det:{}", short_hash(detector))),
        (None, Some(landmark)) => Some(format!("lm:{}", short_hash(landmark))),
        (None, None) => None,
    }
}

fn short_hash(hash: &str) -> &str {
    hash.get(..12).unwrap_or(hash)
}

fn inference_error_code(failure: &vtuber_inference::WorkerFailure) -> &'static str {
    use vtuber_inference::InferenceError;

    if matches!(&failure.error, InferenceError::WorkerPanicked) {
        return "WORKER_PANICKED";
    }

    match failure.stage {
        vtuber_inference::FailureStage::ModelLoad => match &failure.error {
            InferenceError::HashMismatch { .. } => "INFERENCE_MODEL_HASH_MISMATCH",
            InferenceError::LoadFailed(message)
                if message.contains("read model file")
                    || message.contains("No such file")
                    || message.contains("cannot find") =>
            {
                "INFERENCE_MODEL_MISSING"
            }
            InferenceError::OptimizationFailed(_) => "INFERENCE_UNSUPPORTED_OPERATOR",
            _ => "INFERENCE_MODEL_LOAD_FAILED",
        },
        vtuber_inference::FailureStage::Detector => "INFERENCE_DETECTOR_FAILED",
        vtuber_inference::FailureStage::Crop => "INFERENCE_CROP_FAILED",
        vtuber_inference::FailureStage::Landmark => "INFERENCE_LANDMARK_FAILED",
        vtuber_inference::FailureStage::Decode => "INFERENCE_MALFORMED_OUTPUT",
        vtuber_inference::FailureStage::Preprocess => "INFERENCE_PREPROCESS_FAILED",
        _ => "INFERENCE_RUN_FAILED",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vtuber_core::MonoTimeNs;
    use vtuber_inference::{FailureStage, InferenceError, WorkerFailure};

    fn failure(stage: FailureStage, error: InferenceError) -> WorkerFailure {
        WorkerFailure {
            observed_at: MonoTimeNs(1),
            stage,
            error,
        }
    }

    #[test]
    fn inference_error_codes_distinguish_composite_stages() {
        assert_eq!(
            inference_error_code(&failure(
                FailureStage::Detector,
                InferenceError::ExecutionFailed("detector: failed".into()),
            )),
            "INFERENCE_DETECTOR_FAILED"
        );
        assert_eq!(
            inference_error_code(&failure(
                FailureStage::Landmark,
                InferenceError::ExecutionFailed("landmark: failed".into()),
            )),
            "INFERENCE_LANDMARK_FAILED"
        );
        assert_eq!(
            inference_error_code(&failure(
                FailureStage::Decode,
                InferenceError::InvalidOutputValue {
                    index: 0,
                    value: f32::NAN,
                },
            )),
            "INFERENCE_MALFORMED_OUTPUT"
        );
    }
}
