//! Inference worker for the face tracking pipeline.

use std::sync::Arc;
use std::time::{Duration, Instant};

use vtuber_core::types::{FrameSeq, RawFaceObservation, VideoFrame};
use vtuber_core::{FaceTrackingOutcome, LatestSlot, ReadResult, StopToken};

use crate::controller::{ControlCommand, InferenceWorkerResult, MediaPipeTaskSource};
use crate::descriptor::{FacePipelineDescriptor, ModelDescriptor, RuntimeSettings};
use crate::error::{InferenceError, Result};
use crate::metrics::InferenceStage;
use crate::pipeline::Pipeline;
use crate::preprocess::{PreprocessBuffers, PreprocessParams, preprocess_frame};
use crate::runtime::{
    FaceInference, FaceTrackingInference, FrameFaceInference, FrameInferenceOutcome,
    InferenceOutcome,
};
use crate::state::{FailureStage, InferenceWorkerState, SharedStatus};

/// Maximum consecutive recoverable per-frame errors before the worker halts.
const MAX_CONSECUTIVE_RECOVERABLE_ERRORS: u32 = 10;

/// Owned inference context held by the worker thread.
///
/// The runtime, preprocess parameters, and reusable preprocess buffers are
/// constructed together during model load and dropped in the worker thread.
struct InferenceContext {
    runtime: Box<dyn FaceInference>,
    params: PreprocessParams,
    buffers: PreprocessBuffers,
    pipeline: Pipeline,
}

/// Runs the inference worker loop.
///
/// The worker owns the model runtime and processes frames from the input slot.
/// It is spawned by [`crate::controller::InferenceController::start_worker`].
pub fn run_inference_worker(
    command_rx: std::sync::mpsc::Receiver<ControlCommand>,
    stop: StopToken,
    status: SharedStatus,
    frame_slot: Arc<LatestSlot<VideoFrame>>,
    output_slot: Arc<LatestSlot<RawFaceObservation>>,
    outcome_slot: Arc<LatestSlot<InferenceOutcome>>,
    canonical_outcome_slot: Arc<LatestSlot<FaceTrackingOutcome>>,
) -> InferenceWorkerResult {
    let mut context: Option<InferenceContext> = None;
    let mut composite_runtime: Option<Box<dyn FrameFaceInference>> = None;
    let mut mediapipe_runtime: Option<Box<dyn FaceTrackingInference>> = None;
    let mut last_gen = 0u64;
    let mut last_overwritten = 0u64;
    let mut last_processed_seq: Option<FrameSeq> = None;
    let mut paused = false;
    let mut failed = false;

    update_status(&status, |s| {
        s.transition_to(InferenceWorkerState::Idle);
    });

    'worker: while !stop.is_stopped() {
        // Drain control commands first so state changes take effect immediately.
        loop {
            match command_rx.try_recv() {
                Ok(ControlCommand::LoadModel {
                    descriptor,
                    settings,
                }) => {
                    update_status(&status, |s| {
                        s.set_pipeline_info(
                            Some(descriptor.id.clone()),
                            Some(descriptor.sha256.clone()),
                            None,
                        );
                        s.transition_to(InferenceWorkerState::LoadingModel);
                    });

                    match load_inference_context(&descriptor, &settings) {
                        Ok(ctx) => {
                            context = Some(ctx);
                            composite_runtime = None;
                            mediapipe_runtime = None;
                            failed = false;
                            update_status(&status, |s| {
                                s.transition_to(InferenceWorkerState::Running);
                                s.clear_consecutive_errors();
                            });
                        }
                        Err(err) => {
                            failed = true;
                            update_status(&status, |s| {
                                s.record_failure(FailureStage::ModelLoad, err);
                            });
                        }
                    }
                }
                Ok(ControlCommand::LoadPipeline {
                    descriptor,
                    artifact_root,
                    settings,
                }) => {
                    update_status(&status, |s| {
                        s.set_pipeline_info(
                            Some(descriptor.id.clone()),
                            Some(descriptor.detector.sha256.clone()),
                            Some(descriptor.landmarks.sha256.clone()),
                        );
                        s.transition_to(InferenceWorkerState::LoadingModel);
                    });

                    match load_composite_runtime(&descriptor, &artifact_root, &settings) {
                        Ok(runtime) => {
                            context = None;
                            composite_runtime = Some(runtime);
                            mediapipe_runtime = None;
                            failed = false;
                            update_status(&status, |s| {
                                s.transition_to(InferenceWorkerState::Running);
                                s.clear_consecutive_errors();
                            });
                        }
                        Err(err) => {
                            failed = true;
                            update_status(&status, |s| {
                                s.record_failure(FailureStage::ModelLoad, err);
                            });
                        }
                    }
                }
                Ok(ControlCommand::LoadMediaPipe { task }) => {
                    update_status(&status, |s| {
                        s.set_pipeline_info(
                            Some("mediapipe-face-landmarker".into()),
                            Some(crate::backend::mediapipe::TASK_BUNDLE_SHA256.into()),
                            None,
                        );
                        s.transition_to(InferenceWorkerState::LoadingModel);
                    });

                    match load_mediapipe_runtime(&task) {
                        Ok(runtime) => {
                            context = None;
                            composite_runtime = None;
                            mediapipe_runtime = Some(runtime);
                            failed = false;
                            update_status(&status, |s| {
                                s.transition_to(InferenceWorkerState::Running);
                                s.clear_consecutive_errors();
                            });
                        }
                        Err(err) => {
                            failed = true;
                            update_status(&status, |s| {
                                s.record_failure(FailureStage::ModelLoad, err);
                            });
                        }
                    }
                }
                Ok(ControlCommand::Pause) => {
                    paused = true;
                }
                Ok(ControlCommand::Resume) => {
                    paused = false;
                }
                Ok(ControlCommand::Reset) => {
                    context = None;
                    composite_runtime = None;
                    mediapipe_runtime = None;
                    failed = false;
                    last_gen = 0;
                    last_processed_seq = None;
                    update_status(&status, |s| {
                        s.transition_to(InferenceWorkerState::Idle);
                        s.clear_consecutive_errors();
                        s.set_pipeline_info(None, None, None);
                        s.set_roi_diagnostics(None, None);
                    });
                }
                #[cfg(test)]
                Ok(ControlCommand::Panic) => {
                    panic!("inference worker panic requested by test");
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    stop.stop();
                    update_status(&status, |s| {
                        s.transition_to(InferenceWorkerState::Stopping);
                    });
                    break 'worker;
                }
            }
        }

        if paused
            || (context.is_none() && composite_runtime.is_none() && mediapipe_runtime.is_none())
            || failed
        {
            // Wait briefly before polling again so the loop remains responsive.
            std::thread::sleep(Duration::from_millis(10));
            continue;
        }

        let wait_start = Instant::now();
        match frame_slot.wait_read_after(last_gen, Duration::from_millis(50)) {
            Some(ReadResult::New(frame)) => {
                let wait_duration = wait_start.elapsed();
                last_gen = frame_slot.generation();
                let overwritten = frame_slot.overwritten_count();
                let overwrite_delta = overwritten.saturating_sub(last_overwritten);
                update_status(&status, |s| s.record_overwritten(overwrite_delta));
                last_overwritten = overwritten;

                if let Some(last_seq) = last_processed_seq
                    && frame.seq <= last_seq
                {
                    update_status(&status, |s| s.record_duplicate_suppressed());
                    continue;
                }
                last_processed_seq = Some(frame.seq);

                if let Some(runtime) = composite_runtime.as_mut() {
                    if process_composite_frame(
                        runtime,
                        &frame,
                        wait_duration,
                        &status,
                        &output_slot,
                        &outcome_slot,
                    ) {
                        failed = true;
                    }
                    continue;
                }

                if let Some(runtime) = mediapipe_runtime.as_mut() {
                    if process_mediapipe_frame(
                        runtime,
                        &frame,
                        wait_duration,
                        &status,
                        &canonical_outcome_slot,
                    ) {
                        failed = true;
                    }
                    continue;
                }

                // Invariant: the composite and MediaPipe branches above
                // `continue`d, and the pause guard ensured `context` is
                // `Some` whenever neither runtime is active.
                #[allow(clippy::expect_used)]
                let InferenceContext {
                    runtime,
                    params,
                    buffers,
                    pipeline,
                } = context.as_mut().expect("context present when not paused");

                // The legacy runtime is a combined model and has no separate
                // landmark stage. Do not use detector cadence to skip the
                // entire frame: the composite runtime owns detector cadence
                // while keeping landmark inference on every active frame.
                pipeline.record_detector_run(frame.seq);

                let start = Instant::now();
                let started_at = vtuber_core::monotonic_now();

                match preprocess_frame(buffers, &frame, params) {
                    Ok(tensor) => {
                        let preprocess_duration = start.elapsed();
                        let infer_start = Instant::now();
                        match runtime.infer(tensor, &params.input_shape) {
                            Ok(observation) => {
                                let infer_duration = infer_start.elapsed();
                                let decode_start = Instant::now();

                                if let Err(err) = validate_observation(&observation) {
                                    pipeline.mark_lost();
                                    let halt = update_status(&status, |s| {
                                        s.record_frame_error(
                                            FailureStage::Decode,
                                            err,
                                            MAX_CONSECUTIVE_RECOVERABLE_ERRORS,
                                        )
                                    });
                                    if halt {
                                        failed = true;
                                    }
                                } else {
                                    // A low-confidence result or out-of-bounds ROI moves the pipeline
                                    // back to the lost state so the detector runs again.
                                    let _ = pipeline.update_from_observation(
                                        &observation,
                                        frame.width,
                                        frame.height,
                                    );
                                    let decode_duration = decode_start.elapsed();

                                    let elapsed = start.elapsed();
                                    let finished_at = vtuber_core::monotonic_now();
                                    let observation = RawFaceObservation {
                                        source_seq: frame.seq,
                                        captured_at: frame.captured_at,
                                        inference_started_at: started_at,
                                        inference_finished_at: finished_at,
                                        ..observation
                                    };

                                    let output_overwritten_before = output_slot.overwritten_count();
                                    let observation_roi = observation.roi;
                                    let outcome = InferenceOutcome::Face(observation.clone());
                                    if !output_slot.publish(observation) {
                                        update_status(&status, |s| s.record_dropped());
                                    }
                                    let _ = outcome_slot.publish(outcome);
                                    let output_overwritten_delta = output_slot
                                        .overwritten_count()
                                        .saturating_sub(output_overwritten_before);

                                    update_status(&status, |s| {
                                        s.record_stage_duration(
                                            InferenceStage::Wait,
                                            wait_duration,
                                        );
                                        s.record_stage_duration(
                                            InferenceStage::Preprocess,
                                            preprocess_duration,
                                        );
                                        // Detector and landmark are combined in the current single-
                                        // stage runtime. Landmark timing will be split out when the
                                        // pipeline supports a separate landmark-only path.
                                        s.record_stage_duration(
                                            InferenceStage::Detector,
                                            infer_duration,
                                        );
                                        s.record_stage_duration(
                                            InferenceStage::Decode,
                                            decode_duration,
                                        );
                                        s.record_stage_duration(InferenceStage::Total, elapsed);
                                        s.record_output_overwritten(output_overwritten_delta);
                                        s.record_processed(frame.seq, finished_at, elapsed);
                                        s.set_last_roi(Some(observation_roi));
                                    });
                                    update_status(&status, |s| s.clear_consecutive_errors());
                                }
                            }
                            Err(err) => {
                                pipeline.mark_lost();
                                let halt = update_status(&status, |s| {
                                    s.record_frame_error(
                                        FailureStage::Runtime,
                                        err,
                                        MAX_CONSECUTIVE_RECOVERABLE_ERRORS,
                                    )
                                });
                                if halt {
                                    failed = true;
                                }
                            }
                        }
                    }
                    Err(err) => {
                        pipeline.mark_lost();
                        let halt = update_status(&status, |s| {
                            s.record_frame_error(
                                FailureStage::Preprocess,
                                err,
                                MAX_CONSECUTIVE_RECOVERABLE_ERRORS,
                            )
                        });
                        if halt {
                            failed = true;
                        }
                    }
                }
            }
            Some(ReadResult::Closed) => {
                update_status(&status, |s| {
                    s.transition_to(InferenceWorkerState::Stopping);
                });
                break;
            }
            None => {
                // The wait timed out; check the stop token before looping so
                // shutdown completes promptly even when no frames are arriving.
                if stop.is_stopped() {
                    break;
                }
            }
        }
    }

    update_status(&status, |s| {
        if s.state != InferenceWorkerState::Failed {
            s.transition_to(InferenceWorkerState::Stopping);
        }
    });

    let final_metrics = status
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .metrics();

    InferenceWorkerResult { final_metrics }
}

/// Processes one frame through the worker-owned composite runtime.
///
/// Returns `true` when the caller should transition the worker to failed after
/// the bounded consecutive-error policy is exceeded.
fn process_composite_frame(
    runtime: &mut Box<dyn FrameFaceInference>,
    frame: &VideoFrame,
    wait_duration: Duration,
    status: &SharedStatus,
    output_slot: &LatestSlot<RawFaceObservation>,
    outcome_slot: &LatestSlot<InferenceOutcome>,
) -> bool {
    let total_started = Instant::now();
    let outcome = runtime.infer_frame(frame);
    let timing = runtime.take_timing();
    let elapsed = total_started.elapsed();
    let finished_at = vtuber_core::monotonic_now();
    record_composite_timing(status, wait_duration, timing, elapsed);

    match outcome {
        Ok(FrameInferenceOutcome::Face(observation)) => {
            if let Err(error) = validate_observation(&observation) {
                return update_status(status, |s| {
                    s.record_frame_error(
                        FailureStage::Decode,
                        error,
                        MAX_CONSECUTIVE_RECOVERABLE_ERRORS,
                    )
                });
            }

            let output_overwritten_before = output_slot.overwritten_count();
            let observation_roi = observation.roi;
            let _ = outcome_slot.publish(InferenceOutcome::Face(observation.clone()));
            if !output_slot.publish(observation) {
                update_status(status, |s| s.record_dropped());
            }
            let output_overwritten_delta = output_slot
                .overwritten_count()
                .saturating_sub(output_overwritten_before);
            update_status(status, |s| {
                s.record_output_overwritten(output_overwritten_delta);
                s.record_processed(frame.seq, finished_at, elapsed);
                s.set_last_roi(Some(observation_roi));
                s.clear_consecutive_errors();
            });
            false
        }
        Ok(FrameInferenceOutcome::NoFace) => {
            let _ = outcome_slot.publish(InferenceOutcome::NoFace {
                source_seq: frame.seq,
                captured_at: frame.captured_at,
                inference_finished_at: finished_at,
            });
            update_status(status, |s| {
                s.record_no_face(frame.seq, finished_at, elapsed);
                s.set_last_roi(None);
                s.clear_consecutive_errors();
            });
            false
        }
        Err(error) => update_status(status, |s| {
            s.record_frame_error(
                composite_failure_stage(&error),
                error,
                MAX_CONSECUTIVE_RECOVERABLE_ERRORS,
            )
        }),
    }
}

/// Processes one frame through the worker-owned canonical MediaPipe runtime.
fn process_mediapipe_frame(
    runtime: &mut Box<dyn FaceTrackingInference>,
    frame: &VideoFrame,
    wait_duration: Duration,
    status: &SharedStatus,
    canonical_outcome_slot: &LatestSlot<FaceTrackingOutcome>,
) -> bool {
    let total_started = Instant::now();
    let outcome = runtime.infer_face_tracking(frame);
    let elapsed = total_started.elapsed();
    record_mediapipe_timing(status, wait_duration, elapsed);

    match outcome {
        Ok(outcome) => {
            let source_seq = outcome.source_seq();
            let finished_at = match &outcome {
                FaceTrackingOutcome::Face(sample) => sample.inference_finished_at,
                FaceTrackingOutcome::NoFace {
                    inference_finished_at,
                    ..
                } => *inference_finished_at,
            };
            let overwritten_before = canonical_outcome_slot.overwritten_count();
            if !canonical_outcome_slot.publish(outcome.clone()) {
                update_status(status, |s| s.record_dropped());
            }
            let overwritten_delta = canonical_outcome_slot
                .overwritten_count()
                .saturating_sub(overwritten_before);
            update_status(status, |s| {
                s.record_output_overwritten(overwritten_delta);
                s.set_last_roi(None);
                s.clear_consecutive_errors();
                match outcome {
                    FaceTrackingOutcome::Face(_) => {
                        s.record_processed(source_seq, finished_at, elapsed);
                    }
                    FaceTrackingOutcome::NoFace { .. } => {
                        s.record_no_face(source_seq, finished_at, elapsed);
                    }
                }
            });
            false
        }
        Err(error) => update_status(status, |s| {
            s.record_frame_error(
                mediapipe_failure_stage(&error),
                error,
                MAX_CONSECUTIVE_RECOVERABLE_ERRORS,
            )
        }),
    }
}

fn mediapipe_failure_stage(error: &InferenceError) -> FailureStage {
    match error {
        InferenceError::MediaPipeFrameConversion(_) => FailureStage::Preprocess,
        InferenceError::MediaPipeOutputContract(_) => FailureStage::Decode,
        InferenceError::MediaPipeFrameInference(_) => FailureStage::Runtime,
        _ => FailureStage::FrameInference,
    }
}

fn record_mediapipe_timing(status: &SharedStatus, wait_duration: Duration, elapsed: Duration) {
    update_status(status, |s| {
        s.record_stage_duration(InferenceStage::Wait, wait_duration);
        s.record_stage_duration(InferenceStage::Total, elapsed);
    });
}

fn composite_failure_stage(error: &InferenceError) -> FailureStage {
    match error {
        InferenceError::ExecutionFailed(message) if message.starts_with("detector:") => {
            FailureStage::Detector
        }
        InferenceError::ExecutionFailed(message) if message.starts_with("landmark:") => {
            FailureStage::Landmark
        }
        InferenceError::InvalidInput(message) if message.starts_with("crop:") => FailureStage::Crop,
        InferenceError::InvalidRoi(_) => FailureStage::Crop,
        _ => FailureStage::FrameInference,
    }
}

/// Runs a frame-level composite runtime on a capacity-one frame slot.
///
/// This loop intentionally has no detector cadence gate. Cadence and ROI
/// recovery belong to [`FrameFaceInference`], so landmark inference remains
/// active on every frame while an ROI is tracked. `NoFace` is recorded as a
/// normal result and does not increment the recoverable-error counter.
pub fn run_composite_inference_worker(
    mut runtime: Box<dyn FrameFaceInference>,
    stop: StopToken,
    status: SharedStatus,
    frame_slot: Arc<LatestSlot<VideoFrame>>,
    output_slot: Arc<LatestSlot<RawFaceObservation>>,
) -> InferenceWorkerResult {
    let mut last_gen = 0u64;
    let mut last_overwritten = 0u64;
    let mut last_processed_seq: Option<FrameSeq> = None;

    update_status(&status, |s| s.transition_to(InferenceWorkerState::Running));

    while !stop.is_stopped() {
        let wait_started = Instant::now();
        match frame_slot.wait_read_after(last_gen, Duration::from_millis(50)) {
            Some(ReadResult::New(frame)) => {
                let wait_duration = wait_started.elapsed();
                last_gen = frame_slot.generation();
                let overwritten = frame_slot.overwritten_count();
                let overwrite_delta = overwritten.saturating_sub(last_overwritten);
                update_status(&status, |s| s.record_overwritten(overwrite_delta));
                last_overwritten = overwritten;

                if let Some(last_seq) = last_processed_seq
                    && frame.seq <= last_seq
                {
                    update_status(&status, |s| s.record_duplicate_suppressed());
                    continue;
                }
                last_processed_seq = Some(frame.seq);

                let total_started = Instant::now();
                let outcome = runtime.infer_frame(&frame);
                let timing = runtime.take_timing();
                let elapsed = total_started.elapsed();
                let finished_at = vtuber_core::monotonic_now();
                record_composite_timing(&status, wait_duration, timing, elapsed);

                match outcome {
                    Ok(FrameInferenceOutcome::Face(observation)) => {
                        if let Err(error) = validate_observation(&observation) {
                            let halt = update_status(&status, |s| {
                                s.record_frame_error(
                                    FailureStage::Decode,
                                    error,
                                    MAX_CONSECUTIVE_RECOVERABLE_ERRORS,
                                )
                            });
                            if halt {
                                break;
                            }
                            continue;
                        }

                        let output_overwritten_before = output_slot.overwritten_count();
                        if !output_slot.publish(observation.clone()) {
                            update_status(&status, |s| s.record_dropped());
                        }
                        let output_overwritten_delta = output_slot
                            .overwritten_count()
                            .saturating_sub(output_overwritten_before);
                        update_status(&status, |s| {
                            s.record_output_overwritten(output_overwritten_delta);
                            s.record_processed(frame.seq, finished_at, elapsed);
                            s.set_last_roi(Some(observation.roi));
                            s.clear_consecutive_errors();
                        });
                    }
                    Ok(FrameInferenceOutcome::NoFace) => {
                        update_status(&status, |s| {
                            s.record_no_face(frame.seq, finished_at, elapsed);
                            s.set_last_roi(None);
                            s.clear_consecutive_errors();
                        });
                    }
                    Err(error) => {
                        let halt = update_status(&status, |s| {
                            s.record_frame_error(
                                composite_failure_stage(&error),
                                error,
                                MAX_CONSECUTIVE_RECOVERABLE_ERRORS,
                            )
                        });
                        if halt {
                            break;
                        }
                    }
                }
            }
            Some(ReadResult::Closed) | None if frame_slot.is_closed() => break,
            Some(ReadResult::Closed) => break,
            None => {}
        }
    }

    update_status(&status, |s| {
        if s.state != InferenceWorkerState::Failed {
            s.transition_to(InferenceWorkerState::Stopping);
        }
    });

    let final_metrics = status
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .metrics();
    InferenceWorkerResult { final_metrics }
}

fn record_composite_timing(
    status: &SharedStatus,
    wait_duration: Duration,
    timing: crate::runtime::FrameInferenceTiming,
    elapsed: Duration,
) {
    update_status(status, |s| {
        s.set_roi_diagnostics(timing.detector_confidence, timing.roi_state);
        s.record_stage_duration(InferenceStage::Wait, wait_duration);
        if let Some(duration) = timing.detector {
            s.record_stage_duration(InferenceStage::Detector, duration);
        }
        if let Some(duration) = timing.crop {
            s.record_stage_duration(InferenceStage::Crop, duration);
        }
        if let Some(duration) = timing.landmark {
            s.record_stage_duration(InferenceStage::Landmark, duration);
        }
        if let Some(duration) = timing.decode {
            s.record_stage_duration(InferenceStage::Decode, duration);
        }
        s.record_stage_duration(
            InferenceStage::Total,
            if timing.total.is_zero() {
                elapsed
            } else {
                timing.total
            },
        );
    });
}

fn load_inference_context(
    descriptor: &ModelDescriptor,
    settings: &RuntimeSettings,
) -> Result<InferenceContext> {
    let runtime = crate::backend::tract::load_model_runtime(descriptor)?;
    let params = PreprocessParams::from_descriptor(descriptor)?;
    let buffers = PreprocessBuffers::for_shape(&params.input_shape)?;
    Ok(InferenceContext {
        runtime,
        params,
        buffers,
        pipeline: Pipeline::new(settings),
    })
}

fn load_composite_runtime(
    descriptor: &FacePipelineDescriptor,
    artifact_root: &std::path::Path,
    settings: &RuntimeSettings,
) -> Result<Box<dyn FrameFaceInference>> {
    #[cfg(feature = "onnx")]
    {
        let runtime = crate::composite::CompositeFrameInference::from_pipeline_descriptor(
            descriptor,
            artifact_root,
            settings,
        )?;
        Ok(Box::new(runtime))
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = (descriptor, artifact_root, settings);
        Err(crate::error::InferenceError::LoadFailed(
            "the production composite pipeline requires the onnx feature".into(),
        ))
    }
}

fn load_mediapipe_runtime(task: &MediaPipeTaskSource) -> Result<Box<dyn FaceTrackingInference>> {
    let runtime = match task {
        MediaPipeTaskSource::Path(path) => {
            crate::backend::mediapipe::MediaPipeRuntime::from_task_path(path)?
        }
        MediaPipeTaskSource::Embedded => {
            crate::backend::mediapipe::MediaPipeRuntime::from_task_bytes(
                crate::backend::mediapipe::embedded_task_bundle(),
            )?
        }
    };
    Ok(Box::new(runtime))
}

fn update_status<F, R>(status: &SharedStatus, f: F) -> R
where
    F: FnOnce(&mut crate::state::InferenceWorkerStatus) -> R,
{
    let mut s = status
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    f(&mut s)
}

/// Validates decoded model outputs before they are used for tracking.
///
/// Non-finite values are treated as decode failures because they cannot be
/// safely consumed downstream.
fn validate_observation(observation: &RawFaceObservation) -> crate::error::Result<()> {
    if !observation.face_confidence.is_finite() {
        return Err(crate::error::InferenceError::InvalidOutputValue {
            index: 0,
            value: observation.face_confidence,
        });
    }

    for (index, lm) in observation.landmarks.iter().enumerate() {
        if !lm.x.is_finite() || !lm.y.is_finite() || !lm.z.is_finite() {
            return Err(crate::error::InferenceError::InvalidOutputValue {
                index,
                value: if !lm.x.is_finite() {
                    lm.x
                } else if !lm.y.is_finite() {
                    lm.y
                } else {
                    lm.z
                },
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )] // tests may panic (AGENTS.md)

    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use vtuber_core::types::{
        FrameSeq, LandmarkSchemaId, MonoTimeNs, PixelFormat, RawFaceObservation, VideoFrame,
    };
    use vtuber_core::{LatestSlot, ReadResult};

    use super::{
        InferenceContext, MAX_CONSECUTIVE_RECOVERABLE_ERRORS, update_status, validate_observation,
    };
    use crate::controller::{InferenceController, InferenceWorkerResult};
    use crate::descriptor::{
        ChannelOrder, ModelDescriptor, ModelFormat, Normalization, RuntimeSettings,
    };
    use crate::error::InferenceError;
    use crate::metrics::InferenceStage;
    use crate::pipeline::Pipeline;
    use crate::preprocess::{PreprocessBuffers, PreprocessParams, preprocess_frame};
    use crate::runtime::FaceInference;
    use crate::state::{FailureStage, InferenceWorkerState, InferenceWorkerStatus, SharedStatus};
    use vtuber_core::types::NormalizedRect;
    use vtuber_core::{StopToken, WorkerHandle, WorkerResult};

    fn sha256_hex(data: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data);
        hex::encode(hasher.finalize())
    }

    fn dummy_video_frame(seq: u64) -> VideoFrame {
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

    fn test_descriptor(
        path: PathBuf,
        sha256: &str,
        format: ModelFormat,
        normalization: Normalization,
    ) -> ModelDescriptor {
        ModelDescriptor {
            id: "test-model".into(),
            format,
            path,
            sha256: sha256.into(),
            input_name: "input".into(),
            input_shape: vec![1, 256, 256, 3],
            input_dtype: "f32".into(),
            channel_order: ChannelOrder::Rgb,
            normalization,
            schema: LandmarkSchemaId("test-schema"),
            expression_mapping: None,
        }
    }

    fn new_controller() -> (
        InferenceController,
        Arc<LatestSlot<VideoFrame>>,
        Arc<LatestSlot<RawFaceObservation>>,
    ) {
        let frame_slot: Arc<LatestSlot<VideoFrame>> = Arc::new(LatestSlot::new());
        let output_slot: Arc<LatestSlot<RawFaceObservation>> = Arc::new(LatestSlot::new());
        let controller =
            InferenceController::new(Arc::clone(&frame_slot), Arc::clone(&output_slot));
        (controller, frame_slot, output_slot)
    }

    #[test]
    fn worker_model_startup_hash_mismatch() {
        let path = std::env::temp_dir().join("vtuber_inference_hash_mismatch_test.bin");
        let contents = b"not a real model";
        std::fs::write(&path, contents).expect("write temp model file");
        let actual_sha256 = sha256_hex(contents);

        let descriptor = test_descriptor(
            path.clone(),
            "0000000000000000000000000000000000000000000000000000000000000000",
            ModelFormat::Tflite,
            Normalization::ZeroToOne,
        );

        let (mut controller, frame_slot, _output_slot) = new_controller();
        controller.start_worker().expect("start worker");

        // Publish a frame before asking the worker to load the model. A startup
        // failure must not consume it.
        frame_slot.publish(dummy_video_frame(1));

        controller
            .load_model(descriptor, RuntimeSettings::default())
            .expect("send load model");

        std::thread::sleep(Duration::from_millis(200));
        let status = controller.status();
        assert_eq!(status.state, InferenceWorkerState::Failed);
        assert_eq!(
            status.last_failure.as_ref().map(|f| f.stage),
            Some(FailureStage::ModelLoad)
        );

        let failure = status.last_failure.expect("failure recorded");
        match failure.error {
            InferenceError::HashMismatch { expected, actual } => {
                assert_eq!(
                    expected,
                    "0000000000000000000000000000000000000000000000000000000000000000"
                );
                assert_eq!(actual, actual_sha256);
            }
            other => panic!("expected HashMismatch, got {other:?}"),
        }

        // The frame must still be unread because the worker failed before
        // entering the inference loop.
        assert_eq!(status.frames_processed, 0);
        assert_eq!(frame_slot.generation(), 1);
        assert!(
            matches!(frame_slot.try_read_after(0), Some(ReadResult::New(_))),
            "frame slot should still contain the published frame"
        );

        controller.shutdown();
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn worker_model_startup_load_error() {
        let path = std::env::temp_dir().join("vtuber_inference_load_error_test.bin");
        std::fs::write(&path, b"").expect("write empty temp model file");
        let actual_sha256 = sha256_hex(b"");

        let descriptor = test_descriptor(
            path.clone(),
            &actual_sha256,
            ModelFormat::Tflite,
            Normalization::ZeroToOne,
        );

        let (mut controller, frame_slot, _output_slot) = new_controller();
        controller.start_worker().expect("start worker");

        frame_slot.publish(dummy_video_frame(1));

        controller
            .load_model(descriptor, RuntimeSettings::default())
            .expect("send load model");

        std::thread::sleep(Duration::from_millis(200));
        let status = controller.status();
        assert_eq!(status.state, InferenceWorkerState::Failed);
        assert_eq!(
            status.last_failure.as_ref().map(|f| f.stage),
            Some(FailureStage::ModelLoad)
        );

        let failure = status.last_failure.expect("failure recorded");
        assert!(
            matches!(
                failure.error,
                InferenceError::LoadFailed(_) | InferenceError::OptimizationFailed(_)
            ),
            "expected load/optimization error, got {:?}",
            failure.error
        );

        assert_eq!(status.frames_processed, 0);
        assert_eq!(frame_slot.generation(), 1);
        assert!(
            matches!(frame_slot.try_read_after(0), Some(ReadResult::New(_))),
            "frame slot should still contain the published frame"
        );

        controller.shutdown();
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[cfg(feature = "onnx")]
    #[ignore = "requires the downloaded legacy Peppa Pig ONNX research artifact"]
    /// Runs the real legacy model-loading integration after the ignored
    /// artifact has been provisioned.
    ///
    /// Explicit opt-in command:
    /// `cargo test -p vtuber-inference --features onnx worker::tests::worker_model_startup_success_onnx_with_downloaded_artifact -- --ignored --exact`
    fn worker_model_startup_success_onnx_with_downloaded_artifact() {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("assets");
        path.push("models");
        path.push("peppapig_student_1x3x256x256.onnx");

        assert!(
            path.is_file(),
            "expected repository-relative artifact assets/models/peppapig_student_1x3x256x256.onnx; it is intentionally not committed. Provision it before running this ignored test; its download URL and SHA-256 are recorded in assets/models/manifest.toml."
        );

        let descriptor = ModelDescriptor {
            id: "peppapig-onnx".into(),
            format: ModelFormat::Onnx,
            path,
            sha256: "73EDF90954F05EBEF4639E7FA8620C5F83CCA09D2476DE66AB100F26C2B25E7A".into(),
            input_name: "input".into(),
            input_shape: vec![1, 3, 256, 256],
            input_dtype: "f32".into(),
            channel_order: ChannelOrder::Rgb,
            normalization: Normalization::MeanStd {
                mean: [0.485, 0.456, 0.406],
                std: [0.229, 0.224, 0.225],
            },
            schema: LandmarkSchemaId("peppapig-98"),
            expression_mapping: None,
        };

        let (mut controller, _frame_slot, _output_slot) = new_controller();
        controller.start_worker().expect("start worker");
        controller
            .load_model(descriptor, RuntimeSettings::default())
            .expect("send load model");

        let deadline = std::time::Instant::now() + Duration::from_secs(60);
        let mut last_state = InferenceWorkerState::LoadingModel;
        while std::time::Instant::now() < deadline {
            let status = controller.status();
            last_state = status.state;
            if last_state == InferenceWorkerState::Running
                || last_state == InferenceWorkerState::Failed
            {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }

        assert_eq!(
            last_state,
            InferenceWorkerState::Running,
            "worker should reach Running after loading a valid ONNX model; last_failure={:?}",
            controller.status().last_failure
        );

        controller.shutdown();
    }

    #[test]
    fn injected_runtime_worker_reaches_running_and_shuts_down_cleanly() {
        let frame_slot: Arc<LatestSlot<VideoFrame>> = Arc::new(LatestSlot::new());
        let output_slot: Arc<LatestSlot<RawFaceObservation>> = Arc::new(LatestSlot::new());
        let status: SharedStatus = Arc::new(std::sync::Mutex::new(InferenceWorkerStatus::new()));
        let params = PreprocessParams {
            input_shape: [1, 64, 64, 3],
            channel_order: ChannelOrder::Rgb,
            normalization: Normalization::ZeroToOne,
        };
        let buffers = PreprocessBuffers::for_shape(&params.input_shape).unwrap();

        struct SuccessfulRuntime {
            schema: LandmarkSchemaId,
        }

        impl FaceInference for SuccessfulRuntime {
            fn infer(
                &self,
                _tensor: &[f32],
                _input_shape: &[usize; 4],
            ) -> crate::error::Result<RawFaceObservation> {
                Ok(default_observation(self.schema))
            }

            fn schema_id(&self) -> LandmarkSchemaId {
                self.schema
            }
        }

        let handle = WorkerHandle::spawn("inference-hermetic-startup", {
            let frame_slot = Arc::clone(&frame_slot);
            let output_slot = Arc::clone(&output_slot);
            let status = Arc::clone(&status);
            move |stop| {
                run_inference_worker_with_runtime(
                    stop,
                    status,
                    frame_slot,
                    output_slot,
                    Box::new(SuccessfulRuntime {
                        schema: LandmarkSchemaId("hermetic"),
                    }),
                    params,
                    buffers,
                )
            }
        });

        let deadline = Instant::now() + Duration::from_secs(1);
        while Instant::now() < deadline {
            if status.lock().unwrap().state == InferenceWorkerState::Running {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            status.lock().unwrap().state,
            InferenceWorkerState::Running,
            "injected runtime worker should reach Running without a model artifact"
        );

        frame_slot.publish(dummy_video_frame(1));
        let deadline = Instant::now() + Duration::from_secs(1);
        while Instant::now() < deadline {
            if status.lock().unwrap().frames_processed > 0 {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let status_running = status.lock().unwrap();
        assert_eq!(status_running.state, InferenceWorkerState::Running);
        assert_eq!(status_running.frames_processed, 1);
        drop(status_running);

        handle.stop();
        let result = handle.join();
        assert!(
            matches!(result, WorkerResult::Completed(_)),
            "injected runtime worker should shut down cleanly, got {result:?}"
        );
        assert_eq!(
            status.lock().unwrap().state,
            InferenceWorkerState::Stopping,
            "worker should publish its clean stopping state after shutdown"
        );
    }

    #[cfg(test)]
    fn run_inference_worker_with_runtime(
        stop: StopToken,
        status: SharedStatus,
        frame_slot: Arc<LatestSlot<VideoFrame>>,
        output_slot: Arc<LatestSlot<RawFaceObservation>>,
        runtime: Box<dyn FaceInference>,
        params: PreprocessParams,
        buffers: PreprocessBuffers,
    ) -> InferenceWorkerResult {
        use std::time::Instant;

        let mut context = InferenceContext {
            runtime,
            params,
            buffers,
            pipeline: Pipeline::new(&RuntimeSettings {
                frame_wait_timeout_ms: 50,
                detector_interval_frames: 1,
            }),
        };
        let mut last_gen = 0u64;
        let mut last_overwritten = 0u64;
        let mut last_processed_seq: Option<FrameSeq> = None;

        update_status(&status, |s| {
            s.transition_to(InferenceWorkerState::Running);
        });

        while !stop.is_stopped() {
            let wait_start = Instant::now();
            match frame_slot.wait_read_after(last_gen, Duration::from_millis(50)) {
                Some(ReadResult::New(frame)) => {
                    let wait_duration = wait_start.elapsed();
                    last_gen = frame_slot.generation();
                    let overwritten = frame_slot.overwritten_count();
                    let overwrite_delta = overwritten.saturating_sub(last_overwritten);
                    update_status(&status, |s| s.record_overwritten(overwrite_delta));
                    last_overwritten = overwritten;

                    if let Some(last_seq) = last_processed_seq
                        && frame.seq <= last_seq
                    {
                        update_status(&status, |s| s.record_duplicate_suppressed());
                        continue;
                    }
                    last_processed_seq = Some(frame.seq);

                    let start = Instant::now();
                    let started_at = vtuber_core::monotonic_now();

                    let InferenceContext {
                        runtime,
                        params,
                        buffers,
                        pipeline: _,
                    } = &mut context;

                    match preprocess_frame(buffers, &frame, params) {
                        Ok(tensor) => {
                            let preprocess_duration = start.elapsed();
                            let infer_start = Instant::now();
                            match runtime.infer(tensor, &params.input_shape) {
                                Ok(observation) => {
                                    if let Err(err) = validate_observation(&observation) {
                                        let halt = update_status(&status, |s| {
                                            s.record_frame_error(
                                                FailureStage::Decode,
                                                err,
                                                MAX_CONSECUTIVE_RECOVERABLE_ERRORS,
                                            )
                                        });
                                        if halt {
                                            break;
                                        }
                                        continue;
                                    }
                                    let infer_duration = infer_start.elapsed();
                                    let elapsed = start.elapsed();
                                    let finished_at = vtuber_core::monotonic_now();
                                    let observation = RawFaceObservation {
                                        source_seq: frame.seq,
                                        captured_at: frame.captured_at,
                                        inference_started_at: started_at,
                                        inference_finished_at: finished_at,
                                        ..observation
                                    };

                                    let output_overwritten_before = output_slot.overwritten_count();
                                    if !output_slot.publish(observation) {
                                        update_status(&status, |s| s.record_dropped());
                                    }
                                    let output_overwritten_delta = output_slot
                                        .overwritten_count()
                                        .saturating_sub(output_overwritten_before);

                                    update_status(&status, |s| {
                                        s.record_stage_duration(
                                            InferenceStage::Wait,
                                            wait_duration,
                                        );
                                        s.record_stage_duration(
                                            InferenceStage::Preprocess,
                                            preprocess_duration,
                                        );
                                        s.record_stage_duration(
                                            InferenceStage::Detector,
                                            infer_duration,
                                        );
                                        s.record_stage_duration(
                                            InferenceStage::Decode,
                                            Duration::ZERO,
                                        );
                                        s.record_stage_duration(InferenceStage::Total, elapsed);
                                        s.record_output_overwritten(output_overwritten_delta);
                                        s.record_processed(frame.seq, finished_at, elapsed);
                                    });
                                    update_status(&status, |s| s.clear_consecutive_errors());
                                }
                                Err(err) => {
                                    let halt = update_status(&status, |s| {
                                        s.record_frame_error(
                                            FailureStage::Runtime,
                                            err,
                                            MAX_CONSECUTIVE_RECOVERABLE_ERRORS,
                                        )
                                    });
                                    if halt {
                                        break;
                                    }
                                }
                            }
                        }
                        Err(err) => {
                            let halt = update_status(&status, |s| {
                                s.record_frame_error(
                                    FailureStage::Preprocess,
                                    err,
                                    MAX_CONSECUTIVE_RECOVERABLE_ERRORS,
                                )
                            });
                            if halt {
                                break;
                            }
                        }
                    }
                }
                Some(ReadResult::Closed) => break,
                None => {
                    if stop.is_stopped() {
                        break;
                    }
                }
            }
        }

        update_status(&status, |s| {
            if s.state != InferenceWorkerState::Failed {
                s.transition_to(InferenceWorkerState::Stopping);
            }
        });

        let final_metrics = status
            .lock()
            .expect("InferenceController status mutex poisoned")
            .metrics();

        InferenceWorkerResult { final_metrics }
    }

    #[test]
    fn latest_frame_consumption() {
        let frame_slot: Arc<LatestSlot<VideoFrame>> = Arc::new(LatestSlot::new());
        let output_slot: Arc<LatestSlot<RawFaceObservation>> = Arc::new(LatestSlot::new());
        let status: SharedStatus = Arc::new(std::sync::Mutex::new(InferenceWorkerStatus::new()));
        let params = PreprocessParams {
            input_shape: [1, 64, 64, 3],
            channel_order: ChannelOrder::Rgb,
            normalization: Normalization::ZeroToOne,
        };
        let buffers = PreprocessBuffers::for_shape(&params.input_shape).unwrap();

        let last_seen_seq = Arc::new(std::sync::Mutex::new(None::<u64>));

        struct FakeRuntime {
            delay: Duration,
            schema: LandmarkSchemaId,
            last_seen_seq: Arc<std::sync::Mutex<Option<u64>>>,
        }

        impl FaceInference for FakeRuntime {
            fn infer(
                &self,
                tensor: &[f32],
                _input_shape: &[usize; 4],
            ) -> crate::error::Result<RawFaceObservation> {
                std::thread::sleep(self.delay);

                // Test frames are filled with the sequence number as the pixel
                // value, so the first normalized tensor value is seq/255.
                let seq = (tensor[0] * 255.0).round() as u64;
                let mut last = self.last_seen_seq.lock().unwrap();
                if let Some(prev) = *last {
                    assert!(
                        seq > prev,
                        "processed seq {seq} after {prev}; sequences must be strictly increasing"
                    );
                }
                *last = Some(seq);

                Ok(RawFaceObservation {
                    source_seq: FrameSeq(0),
                    captured_at: MonoTimeNs(0),
                    inference_started_at: MonoTimeNs(0),
                    inference_finished_at: MonoTimeNs(0),
                    face_confidence: 1.0,
                    landmarks: Vec::new(),
                    blendshapes: None,
                    expressions: vtuber_core::types::RawExpressionObservation::default(),
                    roi: NormalizedRect {
                        x: 0.0,
                        y: 0.0,
                        width: 1.0,
                        height: 1.0,
                        rotation_rad: 0.0,
                    },
                    schema: self.schema,
                })
            }

            fn schema_id(&self) -> LandmarkSchemaId {
                self.schema
            }
        }

        let handle = WorkerHandle::spawn("inference-latest-frame", {
            let frame_slot = Arc::clone(&frame_slot);
            let output_slot = Arc::clone(&output_slot);
            let status = Arc::clone(&status);
            let last_seen_seq = Arc::clone(&last_seen_seq);
            move |stop| {
                run_inference_worker_with_runtime(
                    stop,
                    status,
                    frame_slot,
                    output_slot,
                    Box::new(FakeRuntime {
                        delay: Duration::from_millis(66),
                        schema: LandmarkSchemaId("fake"),
                        last_seen_seq,
                    }),
                    params,
                    buffers,
                )
            }
        });

        const FRAME_COUNT: u64 = 20;
        let producer_slot = Arc::clone(&frame_slot);
        let producer = std::thread::spawn(move || {
            for seq in 1..=FRAME_COUNT {
                producer_slot.publish(VideoFrame {
                    seq: FrameSeq(seq),
                    captured_at: MonoTimeNs(seq * 1_000_000),
                    width: 64,
                    height: 64,
                    stride_bytes: 64 * 3,
                    format: PixelFormat::Rgb8,
                    data: vec![seq as u8; 64 * 64 * 3].into(),
                });
                std::thread::sleep(Duration::from_millis(33));
            }
        });

        producer.join().expect("producer panicked");

        // Give the worker time to drain the backlog, then request shutdown.
        std::thread::sleep(Duration::from_millis(200));
        handle.stop();
        let result = handle.join();
        assert!(
            matches!(result, WorkerResult::Completed(_)),
            "worker should complete cleanly, got {result:?}"
        );

        let metrics = match result {
            WorkerResult::Completed(m) => m.final_metrics,
            _ => unreachable!(),
        };
        let status_final = status.lock().unwrap();

        println!(
            "latest_frame_consumption: produced={FRAME_COUNT}, processed={}, overwritten={}, suppressed={}",
            status_final.frames_processed,
            status_final.frames_overwritten,
            status_final.duplicate_frames_suppressed
        );

        assert!(
            status_final.frames_processed > 0,
            "worker should process at least one frame"
        );
        assert!(
            status_final.frames_processed < FRAME_COUNT,
            "worker should skip frames when inference is slower than the camera"
        );
        assert!(
            status_final.frames_overwritten > 0,
            "input slot should overwrite unread frames"
        );
        assert_eq!(
            status_final.duplicate_frames_suppressed, 0,
            "no duplicate sequence should be inferred"
        );
        assert_eq!(
            metrics.drops.skipped_sequence, 0,
            "metrics should report zero skipped sequences"
        );
        assert_eq!(
            output_slot.generation(),
            status_final.frames_processed,
            "output slot generation should match processed count"
        );
    }

    #[test]
    fn duplicate_frame_sequence_is_suppressed() {
        let frame_slot: Arc<LatestSlot<VideoFrame>> = Arc::new(LatestSlot::new());
        let output_slot: Arc<LatestSlot<RawFaceObservation>> = Arc::new(LatestSlot::new());
        let status: SharedStatus = Arc::new(std::sync::Mutex::new(InferenceWorkerStatus::new()));
        let params = PreprocessParams {
            input_shape: [1, 64, 64, 3],
            channel_order: ChannelOrder::Rgb,
            normalization: Normalization::ZeroToOne,
        };
        let buffers = PreprocessBuffers::for_shape(&params.input_shape).unwrap();

        let infer_count = Arc::new(std::sync::atomic::AtomicU64::new(0));

        struct CountingRuntime {
            schema: LandmarkSchemaId,
            count: Arc<std::sync::atomic::AtomicU64>,
        }

        impl FaceInference for CountingRuntime {
            fn infer(
                &self,
                _tensor: &[f32],
                _input_shape: &[usize; 4],
            ) -> crate::error::Result<RawFaceObservation> {
                self.count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(20));
                Ok(RawFaceObservation {
                    source_seq: FrameSeq(0),
                    captured_at: MonoTimeNs(0),
                    inference_started_at: MonoTimeNs(0),
                    inference_finished_at: MonoTimeNs(0),
                    face_confidence: 1.0,
                    landmarks: Vec::new(),
                    blendshapes: None,
                    expressions: vtuber_core::types::RawExpressionObservation::default(),
                    roi: NormalizedRect {
                        x: 0.0,
                        y: 0.0,
                        width: 1.0,
                        height: 1.0,
                        rotation_rad: 0.0,
                    },
                    schema: self.schema,
                })
            }

            fn schema_id(&self) -> LandmarkSchemaId {
                self.schema
            }
        }

        let handle = WorkerHandle::spawn("inference-duplicate-suppression", {
            let frame_slot = Arc::clone(&frame_slot);
            let output_slot = Arc::clone(&output_slot);
            let status = Arc::clone(&status);
            let count = Arc::clone(&infer_count);
            move |stop| {
                run_inference_worker_with_runtime(
                    stop,
                    status,
                    frame_slot,
                    output_slot,
                    Box::new(CountingRuntime {
                        schema: LandmarkSchemaId("fake"),
                        count,
                    }),
                    params,
                    buffers,
                )
            }
        });

        // Publish seq 1, wait for it to be processed, then publish seq 1 again.
        frame_slot.publish(VideoFrame {
            seq: FrameSeq(1),
            captured_at: MonoTimeNs(1_000_000),
            width: 64,
            height: 64,
            stride_bytes: 64 * 3,
            format: PixelFormat::Rgb8,
            data: vec![1u8; 64 * 64 * 3].into(),
        });

        std::thread::sleep(Duration::from_millis(60));

        frame_slot.publish(VideoFrame {
            seq: FrameSeq(1),
            captured_at: MonoTimeNs(2_000_000),
            width: 64,
            height: 64,
            stride_bytes: 64 * 3,
            format: PixelFormat::Rgb8,
            data: vec![1u8; 64 * 64 * 3].into(),
        });

        std::thread::sleep(Duration::from_millis(60));
        handle.stop();
        let result = handle.join();
        assert!(matches!(result, WorkerResult::Completed(_)));

        let status_final = status.lock().unwrap();
        assert_eq!(
            status_final.frames_processed, 1,
            "only one frame should be inferred"
        );
        assert_eq!(
            status_final.duplicate_frames_suppressed, 1,
            "duplicate sequence should be counted once"
        );
        assert_eq!(
            infer_count.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "runtime should only be invoked once for seq 1"
        );
    }

    #[test]
    fn inference_metrics_records_stage_timings_and_drops() {
        let frame_slot: Arc<LatestSlot<VideoFrame>> = Arc::new(LatestSlot::new());
        let output_slot: Arc<LatestSlot<RawFaceObservation>> = Arc::new(LatestSlot::new());
        let status: SharedStatus = Arc::new(std::sync::Mutex::new(InferenceWorkerStatus::new()));
        let params = PreprocessParams {
            input_shape: [1, 64, 64, 3],
            channel_order: ChannelOrder::Rgb,
            normalization: Normalization::ZeroToOne,
        };
        let buffers = PreprocessBuffers::for_shape(&params.input_shape).unwrap();

        const FRAME_COUNT: u64 = 15;
        const INFER_DELAY_MS: u64 = 25;
        const PRODUCER_INTERVAL_MS: u64 = 8;

        struct SlowRuntime {
            schema: LandmarkSchemaId,
        }

        impl FaceInference for SlowRuntime {
            fn infer(
                &self,
                _tensor: &[f32],
                _input_shape: &[usize; 4],
            ) -> crate::error::Result<RawFaceObservation> {
                std::thread::sleep(Duration::from_millis(INFER_DELAY_MS));
                Ok(RawFaceObservation {
                    source_seq: FrameSeq(0),
                    captured_at: MonoTimeNs(0),
                    inference_started_at: MonoTimeNs(0),
                    inference_finished_at: MonoTimeNs(0),
                    face_confidence: 1.0,
                    landmarks: Vec::new(),
                    blendshapes: None,
                    expressions: vtuber_core::types::RawExpressionObservation::default(),
                    roi: NormalizedRect {
                        x: 0.0,
                        y: 0.0,
                        width: 1.0,
                        height: 1.0,
                        rotation_rad: 0.0,
                    },
                    schema: self.schema,
                })
            }

            fn schema_id(&self) -> LandmarkSchemaId {
                self.schema
            }
        }

        let handle = WorkerHandle::spawn("inference-metrics", {
            let frame_slot = Arc::clone(&frame_slot);
            let output_slot = Arc::clone(&output_slot);
            let status = Arc::clone(&status);
            move |stop| {
                run_inference_worker_with_runtime(
                    stop,
                    status,
                    frame_slot,
                    output_slot,
                    Box::new(SlowRuntime {
                        schema: LandmarkSchemaId("fake"),
                    }),
                    params,
                    buffers,
                )
            }
        });

        let producer_slot = Arc::clone(&frame_slot);
        let producer = std::thread::spawn(move || {
            for seq in 1..=FRAME_COUNT {
                producer_slot.publish(VideoFrame {
                    seq: FrameSeq(seq),
                    captured_at: MonoTimeNs(seq * 1_000_000),
                    width: 64,
                    height: 64,
                    stride_bytes: 64 * 3,
                    format: PixelFormat::Rgb8,
                    data: vec![seq as u8; 64 * 64 * 3].into(),
                });
                std::thread::sleep(Duration::from_millis(PRODUCER_INTERVAL_MS));
            }
        });

        producer.join().expect("producer panicked");

        // Allow the worker to drain the backlog before shutdown.
        std::thread::sleep(Duration::from_millis(300));
        handle.stop();
        let result = handle.join();
        assert!(
            matches!(result, WorkerResult::Completed(_)),
            "worker should complete cleanly, got {result:?}"
        );

        let metrics = match result {
            WorkerResult::Completed(m) => m.final_metrics,
            _ => unreachable!(),
        };
        let status_final = status.lock().unwrap();

        println!(
            "inference_metrics: produced={FRAME_COUNT}, processed={}, input_overwritten={}, output_overwritten={}, skipped={}",
            status_final.frames_processed,
            metrics.drops.input_overwritten,
            metrics.drops.output_overwritten,
            metrics.drops.skipped_sequence,
        );

        assert!(
            status_final.frames_processed > 0,
            "worker should process at least one frame"
        );
        assert!(
            status_final.frames_processed < FRAME_COUNT,
            "worker should skip frames when inference is slower than the camera"
        );
        assert!(
            status_final
                .last_source_seq
                .is_some_and(|s| s.0 >= FRAME_COUNT - 1),
            "worker should catch up to the latest frames, got {:?}",
            status_final.last_source_seq
        );

        // The capture timestamp must be carried through to the observation.
        if let Some(ReadResult::New(observation)) = output_slot.try_read_after(0) {
            let expected_captured_at = status_final
                .last_source_seq
                .map(|s| MonoTimeNs(s.0 * 1_000_000))
                .unwrap_or(MonoTimeNs(0));
            assert_eq!(
                observation.captured_at, expected_captured_at,
                "captured_at should be carried from the source frame"
            );
            assert_eq!(
                observation.source_seq,
                status_final.last_source_seq.unwrap_or(FrameSeq(0)),
                "source_seq should match the last processed frame"
            );
        } else {
            panic!("output slot should contain the latest observation");
        }

        // Stage timing counts must match processed count.
        assert_eq!(
            metrics.stage(InferenceStage::Wait).count,
            status_final.frames_processed,
            "wait samples should match processed frames"
        );
        assert_eq!(
            metrics.stage(InferenceStage::Preprocess).count,
            status_final.frames_processed,
            "preprocess samples should match processed frames"
        );
        assert_eq!(
            metrics.stage(InferenceStage::Detector).count,
            status_final.frames_processed,
            "detector samples should match processed frames"
        );
        assert_eq!(
            metrics.stage(InferenceStage::Total).count,
            status_final.frames_processed,
            "total samples should match processed frames"
        );

        // Detector timing should reflect the fake runtime delay.
        let detector_mean_ms = metrics.stage(InferenceStage::Detector).mean_ns as f64 / 1_000_000.0;
        assert!(
            detector_mean_ms >= INFER_DELAY_MS as f64 - 5.0,
            "detector mean {detector_mean_ms}ms should be close to {INFER_DELAY_MS}ms"
        );

        // Drop accounting.
        assert!(
            metrics.drops.input_overwritten > 0,
            "input slot should overwrite unread frames"
        );
        assert!(
            metrics.drops.output_overwritten > 0 || status_final.frames_processed <= 1,
            "output slot should overwrite unread observations"
        );
        assert_eq!(
            metrics.drops.processed, status_final.frames_processed,
            "processed counter should match status"
        );
        assert_eq!(
            metrics.drops.skipped_sequence, 0,
            "no sequence should be skipped in this scenario"
        );
    }

    #[test]
    fn consecutive_frame_errors_transition_to_failed() {
        let frame_slot: Arc<LatestSlot<VideoFrame>> = Arc::new(LatestSlot::new());
        let output_slot: Arc<LatestSlot<RawFaceObservation>> = Arc::new(LatestSlot::new());
        let status: SharedStatus = Arc::new(std::sync::Mutex::new(InferenceWorkerStatus::new()));
        let params = PreprocessParams {
            input_shape: [1, 64, 64, 3],
            channel_order: ChannelOrder::Rgb,
            normalization: Normalization::ZeroToOne,
        };
        let buffers = PreprocessBuffers::for_shape(&params.input_shape).unwrap();

        struct FailingRuntime {
            schema: LandmarkSchemaId,
        }

        impl FaceInference for FailingRuntime {
            fn infer(
                &self,
                _tensor: &[f32],
                _input_shape: &[usize; 4],
            ) -> crate::error::Result<RawFaceObservation> {
                Err(crate::error::InferenceError::ExecutionFailed(
                    "always fails".into(),
                ))
            }

            fn schema_id(&self) -> LandmarkSchemaId {
                self.schema
            }
        }

        let handle = WorkerHandle::spawn("inference-consecutive-errors", {
            let frame_slot = Arc::clone(&frame_slot);
            let output_slot = Arc::clone(&output_slot);
            let status = Arc::clone(&status);
            move |stop| {
                run_inference_worker_with_runtime(
                    stop,
                    status,
                    frame_slot,
                    output_slot,
                    Box::new(FailingRuntime {
                        schema: LandmarkSchemaId("fake"),
                    }),
                    params,
                    buffers,
                )
            }
        });

        // Publish frames one at a time so the worker processes each one.
        // Enough frames must be sent to exceed the recoverable error threshold.
        const ERROR_THRESHOLD: u32 = MAX_CONSECUTIVE_RECOVERABLE_ERRORS;
        for seq in 1..=ERROR_THRESHOLD + 5 {
            frame_slot.publish(VideoFrame {
                seq: FrameSeq(seq as u64),
                captured_at: MonoTimeNs(seq as u64 * 1_000_000),
                width: 64,
                height: 64,
                stride_bytes: 64 * 3,
                format: PixelFormat::Rgb8,
                data: vec![seq as u8; 64 * 64 * 3].into(),
            });
            std::thread::sleep(Duration::from_millis(20));
        }

        // Wait for the worker to process the frames and halt.
        std::thread::sleep(Duration::from_millis(50));
        handle.stop();
        let result = handle.join();
        assert!(
            matches!(result, WorkerResult::Completed(_)),
            "worker should complete after halting, got {result:?}"
        );

        let status_final = status.lock().unwrap();
        assert_eq!(
            status_final.state,
            InferenceWorkerState::Failed,
            "worker should transition to Failed after consecutive errors"
        );
        assert_eq!(
            status_final.last_failure.as_ref().map(|f| f.stage),
            Some(FailureStage::Runtime),
            "last failure should be a runtime failure"
        );
        assert!(
            status_final.consecutive_errors > ERROR_THRESHOLD,
            "consecutive error counter should exceed the threshold"
        );
        assert_eq!(
            status_final.frames_processed, 0,
            "no frame should be published"
        );
    }

    fn default_observation(schema: LandmarkSchemaId) -> RawFaceObservation {
        RawFaceObservation {
            source_seq: FrameSeq(0),
            captured_at: MonoTimeNs(0),
            inference_started_at: MonoTimeNs(0),
            inference_finished_at: MonoTimeNs(0),
            face_confidence: 1.0,
            landmarks: Vec::new(),
            blendshapes: None,
            expressions: vtuber_core::types::RawExpressionObservation::default(),
            roi: NormalizedRect {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
                rotation_rad: 0.0,
            },
            schema,
        }
    }

    #[test]
    fn decode_failure_transitions_to_failed() {
        let frame_slot: Arc<LatestSlot<VideoFrame>> = Arc::new(LatestSlot::new());
        let output_slot: Arc<LatestSlot<RawFaceObservation>> = Arc::new(LatestSlot::new());
        let status: SharedStatus = Arc::new(std::sync::Mutex::new(InferenceWorkerStatus::new()));
        let params = PreprocessParams {
            input_shape: [1, 64, 64, 3],
            channel_order: ChannelOrder::Rgb,
            normalization: Normalization::ZeroToOne,
        };
        let buffers = PreprocessBuffers::for_shape(&params.input_shape).unwrap();

        struct NanConfidenceRuntime {
            schema: LandmarkSchemaId,
        }

        impl FaceInference for NanConfidenceRuntime {
            fn infer(
                &self,
                _tensor: &[f32],
                _input_shape: &[usize; 4],
            ) -> crate::error::Result<RawFaceObservation> {
                Ok(RawFaceObservation {
                    face_confidence: f32::NAN,
                    ..default_observation(self.schema)
                })
            }

            fn schema_id(&self) -> LandmarkSchemaId {
                self.schema
            }
        }

        let handle = WorkerHandle::spawn("inference-decode-failure", {
            let frame_slot = Arc::clone(&frame_slot);
            let output_slot = Arc::clone(&output_slot);
            let status = Arc::clone(&status);
            move |stop| {
                run_inference_worker_with_runtime(
                    stop,
                    status,
                    frame_slot,
                    output_slot,
                    Box::new(NanConfidenceRuntime {
                        schema: LandmarkSchemaId("fake"),
                    }),
                    params,
                    buffers,
                )
            }
        });

        const ERROR_THRESHOLD: u32 = MAX_CONSECUTIVE_RECOVERABLE_ERRORS;
        for seq in 1..=ERROR_THRESHOLD + 5 {
            frame_slot.publish(VideoFrame {
                seq: FrameSeq(u64::from(seq)),
                captured_at: MonoTimeNs(u64::from(seq) * 1_000_000),
                width: 64,
                height: 64,
                stride_bytes: 64 * 3,
                format: PixelFormat::Rgb8,
                data: vec![(seq % 256) as u8; 64 * 64 * 3].into(),
            });
            std::thread::sleep(Duration::from_millis(20));
        }

        std::thread::sleep(Duration::from_millis(50));
        handle.stop();
        let result = handle.join();
        assert!(
            matches!(result, WorkerResult::Completed(_)),
            "worker should complete cleanly after halting, got {result:?}"
        );

        let status_final = status.lock().unwrap();
        assert_eq!(
            status_final.state,
            InferenceWorkerState::Failed,
            "worker should transition to Failed after consecutive decode errors"
        );
        assert_eq!(
            status_final.last_failure.as_ref().map(|f| f.stage),
            Some(FailureStage::Decode),
            "last failure should be a decode failure"
        );
        assert!(
            status_final.consecutive_errors > ERROR_THRESHOLD,
            "consecutive error counter should exceed the threshold"
        );
        assert_eq!(
            output_slot.generation(),
            0,
            "no observation should be published when decode fails"
        );
        assert_eq!(
            status_final.frames_processed, 0,
            "no frame should be recorded as processed"
        );
    }

    #[test]
    fn inference_context_reuses_buffers_and_single_runtime() {
        let params = PreprocessParams {
            input_shape: [1, 64, 64, 3],
            channel_order: ChannelOrder::Rgb,
            normalization: Normalization::ZeroToOne,
        };
        let buffers = PreprocessBuffers::for_shape(&params.input_shape).unwrap();
        let initial_capacity = buffers.capacity();
        let infer_count = Arc::new(std::sync::atomic::AtomicU64::new(0));

        struct CountingRuntime {
            schema: LandmarkSchemaId,
            count: Arc<std::sync::atomic::AtomicU64>,
        }

        impl FaceInference for CountingRuntime {
            fn infer(
                &self,
                _tensor: &[f32],
                _input_shape: &[usize; 4],
            ) -> crate::error::Result<RawFaceObservation> {
                self.count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(default_observation(self.schema))
            }

            fn schema_id(&self) -> LandmarkSchemaId {
                self.schema
            }
        }

        let mut context = InferenceContext {
            runtime: Box::new(CountingRuntime {
                schema: LandmarkSchemaId("fake"),
                count: Arc::clone(&infer_count),
            }),
            params,
            buffers,
            pipeline: Pipeline::new(&RuntimeSettings {
                frame_wait_timeout_ms: 50,
                detector_interval_frames: 1,
            }),
        };

        for seq in 1..=5 {
            let frame = VideoFrame {
                seq: FrameSeq(seq),
                captured_at: MonoTimeNs(seq * 1_000_000),
                width: 64,
                height: 64,
                stride_bytes: 64 * 3,
                format: PixelFormat::Rgb8,
                data: vec![(seq % 256) as u8; 64 * 64 * 3].into(),
            };
            let tensor = preprocess_frame(&mut context.buffers, &frame, &context.params).unwrap();
            let _ = context
                .runtime
                .infer(tensor, &context.params.input_shape)
                .unwrap();
        }

        assert_eq!(
            context.buffers.capacity(),
            initial_capacity,
            "buffer capacity must not grow for same-size frames"
        );
        assert_eq!(
            infer_count.load(std::sync::atomic::Ordering::SeqCst),
            5,
            "runtime must be invoked exactly once per frame"
        );
    }
}
