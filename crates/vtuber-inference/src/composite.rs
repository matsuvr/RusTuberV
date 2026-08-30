//! Composite detector-to-landmark inference runtime.
//!
//! The runtime owns the active ROI and reusable crop buffers. Detector cadence
//! only controls the detector stage; an active landmark crop is evaluated for
//! every source frame.

use std::time::Instant;

use vtuber_core::types::{
    Landmark3, LandmarkSchemaId, NormalizedRect, RawFaceObservation, VideoFrame,
};

use crate::crop::{FaceCropPreprocessBuffers, FaceCropTransform, LandmarkCoordinateEncoding};
use crate::descriptor::{FacePipelineDescriptor, ModelRole, RuntimeSettings};
use crate::detector::{
    DetectorDecodeOutcome, FaceDetection, UltraFaceDetector, UltraFacePreprocessBuffers,
    decode_detections, select_primary_face,
};
use crate::error::{InferenceError, Result};
use crate::pipeline::Pipeline;
use crate::roi::FaceRoi;
use crate::runtime::{
    FrameFaceInference, FrameInferenceOutcome, FrameInferenceTiming, OnnxRuntime,
};
use crate::schema::BasicExpressionFallback;

const LANDMARK_COUNT: usize = 98;
const MIN_FACE_CONFIDENCE: f32 = 0.5;
const MIN_LANDMARK_VISIBILITY: f32 = 0.5;
const LANDMARK_BOUNDARY_MARGIN: f32 = 0.25;

/// Detector stage boundary used by [`CompositeRuntime`].
pub trait DetectorStage: Send {
    /// Runs full-frame detection and returns the validated single-frame result.
    fn detect(&mut self, frame: &VideoFrame) -> Result<DetectorDecodeOutcome>;
}

/// Landmark stage boundary used by [`CompositeRuntime`].
pub trait LandmarkStage: Send {
    /// Runs landmark inference on a prepared crop tensor.
    fn infer_landmarks(
        &mut self,
        tensor: &[f32],
        input_shape: [usize; 4],
    ) -> Result<Vec<Landmark3>>;
}

/// Generic, mockable composite detector and landmark runtime.
pub struct CompositeRuntime<D, L> {
    descriptor: FacePipelineDescriptor,
    detector: D,
    landmark: L,
    crop_buffers: FaceCropPreprocessBuffers,
    pipeline: Pipeline,
    active_transform: Option<FaceCropTransform>,
    active_detection_confidence: f32,
    previous_roi: Option<NormalizedRect>,
    coordinate_encoding: LandmarkCoordinateEncoding,
    landmark_input_shape: [usize; 4],
    last_timing: FrameInferenceTiming,
}

impl<D, L> CompositeRuntime<D, L>
where
    D: DetectorStage,
    L: LandmarkStage,
{
    /// Creates a composite runtime with worker-owned crop buffers and ROI state.
    pub fn new(
        descriptor: FacePipelineDescriptor,
        detector: D,
        landmark: L,
        settings: &RuntimeSettings,
    ) -> Result<Self> {
        let (coordinate_encoding, landmark_input_shape) = validate_descriptor(&descriptor)?;
        let crop_buffers = FaceCropPreprocessBuffers::new(descriptor.crop.output_size)
            .map_err(|error| InferenceError::InvalidInput(error.to_string()))?;

        Ok(Self {
            descriptor,
            detector,
            landmark,
            crop_buffers,
            pipeline: Pipeline::new(settings),
            active_transform: None,
            active_detection_confidence: 0.0,
            previous_roi: None,
            coordinate_encoding,
            landmark_input_shape,
            last_timing: FrameInferenceTiming::default(),
        })
    }

    /// Returns the runtime-free descriptor used by this pipeline.
    #[must_use]
    pub fn descriptor(&self) -> &FacePipelineDescriptor {
        &self.descriptor
    }

    /// Returns the current ROI lifecycle state.
    #[must_use]
    pub fn roi_state(&self) -> &crate::roi::RoiState {
        self.pipeline.roi_state()
    }

    /// Returns the reusable crop-buffer capacities for replay allocation checks.
    #[must_use]
    pub fn crop_buffer_capacities(&self) -> (usize, usize) {
        self.crop_buffers.capacities()
    }

    fn infer_frame_inner(&mut self, frame: &VideoFrame) -> Result<FrameInferenceOutcome> {
        let detector_due = self.pipeline.should_run_detector(frame.seq);
        if detector_due {
            let detector_started = Instant::now();
            let detection_result = self.detector.detect(frame);
            self.last_timing.detector = Some(detector_started.elapsed());
            self.pipeline.record_detector_run(frame.seq);

            let detection = match detection_result {
                Ok(detection) => detection,
                Err(error) => {
                    self.invalidate_roi(frame);
                    return Err(InferenceError::ExecutionFailed(format!(
                        "detector: {error}"
                    )));
                }
            };
            match detection {
                DetectorDecodeOutcome::NoFace => {
                    self.invalidate_roi(frame);
                    return Ok(FrameInferenceOutcome::NoFace);
                }
                DetectorDecodeOutcome::Detections(detections) => {
                    let Some(primary) =
                        select_primary_face(&detections, self.previous_roi.as_ref())
                    else {
                        self.invalidate_roi(frame);
                        return Ok(FrameInferenceOutcome::NoFace);
                    };
                    match self.activate_detection(frame, primary) {
                        Ok(true) => {}
                        Ok(false) => return Ok(FrameInferenceOutcome::NoFace),
                        Err(error) => {
                            self.invalidate_roi(frame);
                            return Err(error);
                        }
                    }
                }
            }
        }

        let Some(transform) = self.active_transform else {
            self.pipeline.mark_lost();
            return Ok(FrameInferenceOutcome::NoFace);
        };

        let crop_started = Instant::now();
        let tensor = match self.crop_buffers.preprocess(
            frame,
            &transform,
            &self.descriptor.landmarks.input,
            self.descriptor.crop,
        ) {
            Ok(tensor) => tensor,
            Err(error) => {
                self.invalidate_roi(frame);
                return Err(InferenceError::InvalidInput(format!("crop: {error}")));
            }
        };
        self.last_timing.crop = Some(crop_started.elapsed());

        let landmark_started = Instant::now();
        let mut landmarks = match self
            .landmark
            .infer_landmarks(tensor, self.landmark_input_shape)
        {
            Ok(landmarks) => landmarks,
            Err(error) => {
                self.invalidate_roi(frame);
                return Err(InferenceError::ExecutionFailed(format!(
                    "landmark: {error}"
                )));
            }
        };
        self.last_timing.landmark = Some(landmark_started.elapsed());

        let decode_started = Instant::now();
        let valid = landmarks.len() == LANDMARK_COUNT
            && transform
                .map_landmarks_to_source_normalized(&mut landmarks, self.coordinate_encoding)
                .is_ok()
            && landmarks_valid(&landmarks);
        if !valid {
            self.last_timing.decode = Some(decode_started.elapsed());
            self.invalidate_roi(frame);
            return Ok(FrameInferenceOutcome::NoFace);
        }

        let visibility = average_visibility(&landmarks);
        if visibility < MIN_LANDMARK_VISIBILITY {
            self.last_timing.decode = Some(decode_started.elapsed());
            self.invalidate_roi(frame);
            return Ok(FrameInferenceOutcome::NoFace);
        }
        let face_confidence = self
            .active_detection_confidence
            .min(visibility)
            .clamp(0.0, 1.0);
        if face_confidence < MIN_FACE_CONFIDENCE {
            self.last_timing.decode = Some(decode_started.elapsed());
            self.invalidate_roi(frame);
            return Ok(FrameInferenceOutcome::NoFace);
        }

        let schema = landmark_schema(&self.descriptor)?;
        let roi = transform.source_roi();
        let mut observation = RawFaceObservation {
            source_seq: frame.seq,
            captured_at: frame.captured_at,
            inference_started_at: vtuber_core::monotonic_now(),
            inference_finished_at: vtuber_core::monotonic_now(),
            face_confidence,
            expressions: RawExpressionObservation::default(),
            landmarks,
            blendshapes: None,
            roi,
            schema,
        };
        observation.expressions = BasicExpressionFallback::from_landmarks(
            &observation.landmarks,
            schema,
            face_confidence,
        )
        .unwrap_or_default();
        self.last_timing.decode = Some(decode_started.elapsed());

        if self
            .pipeline
            .update_from_observation(&observation, frame.width, frame.height)
            .is_err()
        {
            self.invalidate_roi(frame);
            return Ok(FrameInferenceOutcome::NoFace);
        }

        Ok(FrameInferenceOutcome::Face(observation))
    }

    fn activate_detection(&mut self, frame: &VideoFrame, detection: FaceDetection) -> Result<bool> {
        let transform = FaceCropTransform::from_detector_box(
            frame.width,
            frame.height,
            &detection.rect,
            self.descriptor.crop,
        )
        .map_err(|error| InferenceError::InvalidRoi(format!("detector crop: {error}")))?;
        let roi = FaceRoi::from_normalized_rect(&detection.rect, frame.width, frame.height);
        let mut roi = roi;
        roi.confidence = detection.confidence;
        self.pipeline
            .record_detector_result(frame.seq, frame.width, frame.height, Some(roi));
        if !self.pipeline.is_tracking() {
            self.invalidate_roi(frame);
            return Ok(false);
        }
        self.previous_roi = Some(detection.rect);
        self.active_detection_confidence = detection.confidence;
        self.active_transform = Some(transform);
        Ok(true)
    }

    fn invalidate_roi(&mut self, _frame: &VideoFrame) {
        self.active_transform = None;
        self.active_detection_confidence = 0.0;
        self.previous_roi = None;
        self.pipeline.mark_lost();
    }
}

impl<D, L> FrameFaceInference for CompositeRuntime<D, L>
where
    D: DetectorStage,
    L: LandmarkStage,
{
    fn infer_frame(&mut self, frame: &VideoFrame) -> Result<FrameInferenceOutcome> {
        self.last_timing = FrameInferenceTiming::default();
        let inference_started_at = vtuber_core::monotonic_now();
        let started = Instant::now();
        let mut result = self.infer_frame_inner(frame);
        let inference_finished_at = vtuber_core::monotonic_now();
        self.last_timing.total = started.elapsed();
        if let Ok(FrameInferenceOutcome::Face(observation)) = &mut result {
            observation.inference_started_at = inference_started_at;
            observation.inference_finished_at = inference_finished_at;
        }
        result
    }

    fn take_timing(&mut self) -> FrameInferenceTiming {
        let mut timing = std::mem::take(&mut self.last_timing);
        timing.detector_confidence = Some(self.active_detection_confidence);
        timing.roi_state = Some(format!("{:?}", self.pipeline.roi_state()));
        timing
    }
}

/// Production detector adapter for the pinned UltraFace ONNX artifact.
pub struct ProductionDetectorStage {
    detector: UltraFaceDetector,
    buffers: UltraFacePreprocessBuffers,
    config: crate::descriptor::DetectorPostprocessConfig,
}

impl ProductionDetectorStage {
    /// Loads a production detector and creates its reusable preprocessing buffers.
    pub fn from_path(
        path: impl AsRef<std::path::Path>,
        config: crate::descriptor::DetectorPostprocessConfig,
    ) -> Result<Self> {
        let detector = UltraFaceDetector::from_path(path)
            .map_err(|error| InferenceError::LoadFailed(error.to_string()))?;
        Ok(Self {
            detector,
            buffers: UltraFacePreprocessBuffers::new(),
            config,
        })
    }
}

impl DetectorStage for ProductionDetectorStage {
    fn detect(&mut self, frame: &VideoFrame) -> Result<DetectorDecodeOutcome> {
        let raw = self
            .detector
            .infer(&mut self.buffers, frame)
            .map_err(|error| InferenceError::ExecutionFailed(error.to_string()))?;
        decode_detections(&raw, self.config)
            .map_err(|error| InferenceError::ExecutionFailed(format!("decode: {error}")))
    }
}

/// Production landmark adapter for the manifest-described ONNX artifact.
pub struct ProductionLandmarkStage {
    runtime: OnnxRuntime,
}

impl ProductionLandmarkStage {
    /// Creates an adapter around a worker-owned landmark runtime.
    #[must_use]
    pub fn new(runtime: OnnxRuntime) -> Self {
        Self { runtime }
    }
}

impl LandmarkStage for ProductionLandmarkStage {
    fn infer_landmarks(
        &mut self,
        tensor: &[f32],
        input_shape: [usize; 4],
    ) -> Result<Vec<Landmark3>> {
        self.runtime.infer_landmarks(tensor, &input_shape)
    }
}

/// Production composite runtime that owns detector, landmark, and buffers.
pub struct CompositeFrameInference {
    runtime: CompositeRuntime<ProductionDetectorStage, ProductionLandmarkStage>,
}

impl CompositeFrameInference {
    /// Constructs both live runtimes from a worker-safe pipeline descriptor.
    pub fn from_pipeline_descriptor(
        descriptor: &FacePipelineDescriptor,
        artifact_root: &std::path::Path,
        settings: &RuntimeSettings,
    ) -> Result<Self> {
        let schema = landmark_schema(descriptor)?;
        let detector_path = artifact_root.join(&descriptor.detector.file);
        let landmark_path = artifact_root.join(&descriptor.landmarks.file);
        crate::backend::tract::verify_model_file(&detector_path, &descriptor.detector.sha256)?;
        crate::backend::tract::verify_model_file(&landmark_path, &descriptor.landmarks.sha256)?;
        let detector =
            ProductionDetectorStage::from_path(detector_path, descriptor.detector_postprocess)?;
        let landmark = ProductionLandmarkStage::new(OnnxRuntime::new(landmark_path, schema)?);
        let runtime = CompositeRuntime::new(descriptor.clone(), detector, landmark, settings)?;
        Ok(Self { runtime })
    }

    /// Returns the plain descriptor used to construct this runtime.
    #[must_use]
    pub fn descriptor(&self) -> &FacePipelineDescriptor {
        self.runtime.descriptor()
    }
}

impl FrameFaceInference for CompositeFrameInference {
    fn infer_frame(&mut self, frame: &VideoFrame) -> Result<FrameInferenceOutcome> {
        self.runtime.infer_frame(frame)
    }

    fn take_timing(&mut self) -> FrameInferenceTiming {
        self.runtime.take_timing()
    }
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn validate_descriptor(
    descriptor: &FacePipelineDescriptor,
) -> Result<(LandmarkCoordinateEncoding, [usize; 4])> {
    if descriptor.detector.role != ModelRole::FaceDetector {
        return Err(InferenceError::InvalidInput(
            "pipeline detector descriptor has the wrong role".into(),
        ));
    }
    if descriptor.landmarks.role != ModelRole::FaceLandmarks {
        return Err(InferenceError::InvalidInput(
            "pipeline landmark descriptor has the wrong role".into(),
        ));
    }
    if !descriptor.landmarks.requires_crop {
        return Err(InferenceError::InvalidInput(
            "landmark descriptor must require a face crop".into(),
        ));
    }
    let _ = landmark_schema(descriptor)?;
    let encoding = descriptor
        .landmarks
        .landmark_coordinate_encoding
        .as_deref()
        .and_then(LandmarkCoordinateEncoding::parse)
        .ok_or_else(|| {
            InferenceError::InvalidInput(
                "unsupported or missing landmark coordinate encoding".into(),
            )
        })?;
    if descriptor.landmarks.input.shape.len() != 4 {
        return Err(InferenceError::InvalidInput(
            "landmark input shape must have four dimensions".into(),
        ));
    }
    let input_shape = descriptor
        .landmarks
        .input
        .shape
        .clone()
        .try_into()
        .map_err(|_| {
            InferenceError::InvalidInput("landmark input shape is not four-dimensional".into())
        })?;
    let expected_output = [1, LANDMARK_COUNT, 3];
    if descriptor.landmarks.outputs.len() != 1
        || descriptor.landmarks.outputs[0].shape != expected_output
        || descriptor.landmarks.outputs[0].dtype != "float32"
    {
        return Err(InferenceError::InvalidInput(
            "landmark output contract must be one float32 [1,98,3] tensor".into(),
        ));
    }
    Ok((encoding, input_shape))
}

fn landmark_schema(descriptor: &FacePipelineDescriptor) -> Result<LandmarkSchemaId> {
    match descriptor.landmarks.schema.as_deref() {
        Some("peppapig-98") => Ok(LandmarkSchemaId("peppapig-98")),
        Some(other) => Err(InferenceError::InvalidInput(format!(
            "unsupported landmark schema `{other}`"
        ))),
        None => Err(InferenceError::InvalidInput(
            "landmark descriptor has no schema".into(),
        )),
    }
}

fn landmarks_valid(landmarks: &[Landmark3]) -> bool {
    landmarks.iter().all(|landmark| {
        landmark.x.is_finite()
            && landmark.y.is_finite()
            && landmark.z.is_finite()
            && landmark.visibility.is_finite()
            && (0.0..=1.0).contains(&landmark.visibility)
            && landmark.x >= -LANDMARK_BOUNDARY_MARGIN
            && landmark.x <= 1.0 + LANDMARK_BOUNDARY_MARGIN
            && landmark.y >= -LANDMARK_BOUNDARY_MARGIN
            && landmark.y <= 1.0 + LANDMARK_BOUNDARY_MARGIN
    })
}

fn average_visibility(landmarks: &[Landmark3]) -> f32 {
    if landmarks.is_empty() {
        return 0.0;
    }
    landmarks
        .iter()
        .map(|landmark| landmark.visibility)
        .sum::<f32>()
        / landmarks.len() as f32
}

use vtuber_core::types::RawExpressionObservation;
