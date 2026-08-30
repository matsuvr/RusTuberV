//! Deterministic, repository-local fixtures for composite golden/replay tests.

use std::collections::VecDeque;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use vtuber_core::types::{
    FrameSeq, Landmark3, MonoTimeNs, NormalizedRect, PixelFormat, VideoFrame,
};
use vtuber_inference::composite::{CompositeRuntime, DetectorStage, LandmarkStage};
use vtuber_inference::detector::{DetectorDecodeOutcome, FaceDetection};
use vtuber_inference::{
    ChannelOrder, CropInterpolation, CropOutsideFill, DetectorPostprocessConfig, FaceCropConfig,
    FacePipelineDescriptor, InputValueDomain, ModelArtifactDescriptor, ModelRole,
    NormalizationContract, OutputTensorContract, RuntimeSettings, TensorContract, TensorLayout,
};

/// Detector mock with a bounded scripted result prefix and deterministic default.
pub struct MockDetector {
    /// Number of detector calls.
    pub calls: Arc<AtomicUsize>,
    results: VecDeque<DetectorDecodeOutcome>,
}

impl MockDetector {
    /// Creates a detector mock with scripted results.
    pub fn new(results: impl IntoIterator<Item = DetectorDecodeOutcome>) -> Self {
        Self {
            calls: Arc::new(AtomicUsize::new(0)),
            results: results.into_iter().collect(),
        }
    }
}

impl DetectorStage for MockDetector {
    fn detect(&mut self, _frame: &VideoFrame) -> vtuber_inference::Result<DetectorDecodeOutcome> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(self
            .results
            .pop_front()
            .unwrap_or_else(|| DetectorDecodeOutcome::Detections(vec![detection()])))
    }
}

/// Landmark mock with a bounded scripted result prefix.
pub struct MockLandmark {
    /// Number of landmark calls.
    pub calls: Arc<AtomicUsize>,
    results: VecDeque<Vec<Landmark3>>,
}

impl MockLandmark {
    /// Creates a landmark mock with scripted results.
    pub fn new(results: impl IntoIterator<Item = Vec<Landmark3>>) -> Self {
        Self {
            calls: Arc::new(AtomicUsize::new(0)),
            results: results.into_iter().collect(),
        }
    }
}

impl LandmarkStage for MockLandmark {
    fn infer_landmarks(
        &mut self,
        _tensor: &[f32],
        _input_shape: [usize; 4],
    ) -> vtuber_inference::Result<Vec<Landmark3>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(self.results.pop_front().unwrap_or_else(valid_landmarks))
    }
}

/// Creates a mock runtime with the production crop and tensor contract.
pub fn runtime(
    detector: MockDetector,
    landmark: MockLandmark,
) -> CompositeRuntime<MockDetector, MockLandmark> {
    CompositeRuntime::new(
        production_descriptor(),
        detector,
        landmark,
        &RuntimeSettings {
            frame_wait_timeout_ms: 50,
            detector_interval_frames: 5,
        },
    )
    .expect("production crop contract is valid")
}

/// Production pipeline descriptor copied from `assets/models/manifest.toml`.
pub fn production_descriptor() -> FacePipelineDescriptor {
    let detector_input = TensorContract {
        shape: vec![1, 3, 240, 320],
        dtype: "float32".into(),
        layout: TensorLayout::Nchw,
        channel_order: ChannelOrder::Rgb,
        value_domain: InputValueDomain::RawU8,
        normalization: NormalizationContract {
            mean: [127.0; 3],
            scale: [128.0; 3],
        },
    };
    let landmark_input = TensorContract {
        shape: vec![1, 3, 256, 256],
        dtype: "float32".into(),
        layout: TensorLayout::Nchw,
        channel_order: ChannelOrder::Rgb,
        value_domain: InputValueDomain::UnitFloat,
        normalization: NormalizationContract {
            mean: [0.485, 0.456, 0.406],
            scale: [0.229, 0.224, 0.225],
        },
    };
    FacePipelineDescriptor {
        id: "ultraface-rfb-320-peppapig-98".into(),
        detector: ModelArtifactDescriptor {
            id: "ultraface-rfb-320".into(),
            role: ModelRole::FaceDetector,
            file: "version-RFB-320.onnx".into(),
            byte_size: 1_270_727,
            sha256: "34cd7e60aeff28744c657de7a3dc64e872d506741de66987f3426f2b79f88017".into(),
            input_name: "input".into(),
            source: "https://huggingface.co/onnxmodelzoo/version-RFB-320/resolve/main/version-RFB-320.onnx".into(),
            upstream: "https://github.com/onnx/models/tree/main/vision/body_analysis/ultraface".into(),
            license: "MIT".into(),
            license_url: Some("https://opensource.org/license/mit/".into()),
            input: detector_input,
            outputs: vec![
                OutputTensorContract {
                    name: "scores".into(),
                    shape: vec![1, 4420, 2],
                    dtype: "float32".into(),
                    description: "Per-anchor background and face scores".into(),
                },
                OutputTensorContract {
                    name: "boxes".into(),
                    shape: vec![1, 4420, 4],
                    dtype: "float32".into(),
                    description: "Per-anchor encoded face boxes".into(),
                },
            ],
            requires_crop: false,
            schema: None,
            landmark_coordinate_encoding: None,
            pose_method: None,
            representative_indices: Vec::new(),
        },
        landmarks: ModelArtifactDescriptor {
            id: "peppapig-98".into(),
            role: ModelRole::FaceLandmarks,
            file: "peppapig_student_1x3x256x256.onnx".into(),
            byte_size: 13_728_231,
            sha256: "73EDF90954F05EBEF4639E7FA8620C5F83CCA09D2476DE66AB100F26C2B25E7A".into(),
            input_name: "input".into(),
            source: "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/436_Peppa_Pig_Face_Landmark/resources.tar.gz".into(),
            upstream: "https://github.com/610265158/Peppa_Pig_Face_Landmark".into(),
            license: "Apache-2.0".into(),
            license_url: Some("https://github.com/610265158/Peppa_Pig_Face_Landmark/blob/master/LICENSE".into()),
            input: landmark_input,
            outputs: vec![OutputTensorContract {
                name: "/Concat_1".into(),
                shape: vec![1, 98, 3],
                dtype: "float32".into(),
                description: "98 facial landmarks with visibility/confidence in third channel".into(),
            }],
            requires_crop: true,
            schema: Some("peppapig-98".into()),
            landmark_coordinate_encoding: Some("normalized_0_1".into()),
            pose_method: Some("canonical_orthographic_2d".into()),
            representative_indices: vec![16, 37, 46, 52, 63, 71, 76, 82],
        },
        detector_postprocess: DetectorPostprocessConfig {
            score_threshold: 0.7,
            nms_iou: 0.3,
            max_pre_nms_candidates: 256,
            max_post_nms_detections: 16,
        },
        crop: FaceCropConfig {
            square_scale: 1.35,
            center_y_offset_fraction: -0.05,
            output_size: [256, 256],
            interpolation: CropInterpolation::Bilinear,
            outside_fill: CropOutsideFill::NormalizationMean,
        },
    }
}

/// Synthetic detector result used as the positive golden fixture.
pub fn detection() -> FaceDetection {
    FaceDetection {
        rect: NormalizedRect {
            x: 0.25,
            y: 0.25,
            width: 0.25,
            height: 0.5,
            rotation_rad: 0.0,
        },
        confidence: 0.9,
        anchor_index: 7,
    }
}

/// Generates a deterministic 64x48 mean image; it has no external asset provenance.
pub fn mean_frame(seq: u64) -> VideoFrame {
    VideoFrame {
        seq: FrameSeq(seq),
        captured_at: MonoTimeNs(seq * 1_000),
        width: 64,
        height: 48,
        stride_bytes: 64 * 3,
        format: PixelFormat::Rgb8,
        data: vec![127u8; 64 * 48 * 3].into(),
    }
}

/// Generates finite 98-point crop-normalized landmark output.
pub fn valid_landmarks() -> Vec<Landmark3> {
    vec![
        Landmark3 {
            x: 0.5,
            y: 0.5,
            z: 0.0,
            visibility: 1.0,
        };
        98
    ]
}
