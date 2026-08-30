// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Camera-free tests for detector cadence and composite ROI recovery.

#![cfg(feature = "legacy-face-stack")]

use std::collections::VecDeque;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

use vtuber_core::types::{
    FrameSeq, Landmark3, MonoTimeNs, NormalizedRect, PixelFormat, VideoFrame,
};
use vtuber_core::{LatestSlot, WorkerHandle, WorkerResult};
use vtuber_inference::composite::{CompositeRuntime, DetectorStage, LandmarkStage};
use vtuber_inference::detector::{DetectorDecodeOutcome, FaceDetection};
use vtuber_inference::state::{InferenceWorkerState, SharedStatus};
use vtuber_inference::worker::run_composite_inference_worker;
use vtuber_inference::{
    ChannelOrder, CropInterpolation, CropOutsideFill, DetectorPostprocessConfig, FaceCropConfig,
    FacePipelineDescriptor, FrameFaceInference, FrameInferenceOutcome, InputValueDomain,
    ModelArtifactDescriptor, ModelRole, NormalizationContract, OutputTensorContract,
    RuntimeSettings, TensorContract, TensorLayout,
};

#[derive(Clone)]
struct MockDetector {
    calls: Arc<AtomicUsize>,
    results: VecDeque<DetectorDecodeOutcome>,
}

impl DetectorStage for MockDetector {
    fn detect(&mut self, _frame: &VideoFrame) -> vtuber_inference::Result<DetectorDecodeOutcome> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(self
            .results
            .pop_front()
            .unwrap_or(DetectorDecodeOutcome::Detections(vec![detection()])))
    }
}

struct MockLandmark {
    calls: Arc<AtomicUsize>,
    results: VecDeque<Vec<Landmark3>>,
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

#[test]
fn composite_runtime_detector_cadence_keeps_landmark_inference_on_every_input_frame() {
    let detector_calls = Arc::new(AtomicUsize::new(0));
    let landmark_calls = Arc::new(AtomicUsize::new(0));
    let detector = MockDetector {
        calls: Arc::clone(&detector_calls),
        results: VecDeque::from([DetectorDecodeOutcome::Detections(vec![detection()])]),
    };
    let landmark = MockLandmark {
        calls: Arc::clone(&landmark_calls),
        results: VecDeque::new(),
    };
    let mut runtime = CompositeRuntime::new(
        descriptor(),
        detector,
        landmark,
        &RuntimeSettings {
            frame_wait_timeout_ms: 50,
            detector_interval_frames: 5,
        },
    )
    .expect("mock descriptor is valid");

    for sequence in 1..=7 {
        assert!(matches!(
            runtime.infer_frame(&frame(sequence)),
            Ok(FrameInferenceOutcome::Face(_))
        ));
    }

    assert_eq!(detector_calls.load(Ordering::Relaxed), 2);
    assert_eq!(landmark_calls.load(Ordering::Relaxed), 7);
}

#[test]
fn composite_runtime_roi_recovery_searches_after_no_face_and_reacquires() {
    let detector_calls = Arc::new(AtomicUsize::new(0));
    let landmark_calls = Arc::new(AtomicUsize::new(0));
    let detector = MockDetector {
        calls: Arc::clone(&detector_calls),
        results: VecDeque::from([
            DetectorDecodeOutcome::NoFace,
            DetectorDecodeOutcome::Detections(vec![detection()]),
        ]),
    };
    let landmark = MockLandmark {
        calls: Arc::clone(&landmark_calls),
        results: VecDeque::new(),
    };
    let mut runtime = runtime_with(detector, landmark);

    assert!(matches!(
        runtime.infer_frame(&frame(1)),
        Ok(FrameInferenceOutcome::NoFace)
    ));
    assert!(matches!(
        runtime.infer_frame(&frame(2)),
        Ok(FrameInferenceOutcome::Face(_))
    ));
    assert!(matches!(
        runtime.infer_frame(&frame(3)),
        Ok(FrameInferenceOutcome::Face(_))
    ));

    assert_eq!(detector_calls.load(Ordering::Relaxed), 2);
    assert_eq!(landmark_calls.load(Ordering::Relaxed), 2);
}

#[test]
fn composite_runtime_roi_recovery_forces_detector_after_low_confidence_landmarks() {
    let detector_calls = Arc::new(AtomicUsize::new(0));
    let landmark_calls = Arc::new(AtomicUsize::new(0));
    let detector = MockDetector {
        calls: Arc::clone(&detector_calls),
        results: VecDeque::from([
            DetectorDecodeOutcome::Detections(vec![detection()]),
            DetectorDecodeOutcome::Detections(vec![detection()]),
        ]),
    };
    let landmark = MockLandmark {
        calls: Arc::clone(&landmark_calls),
        results: VecDeque::from([low_confidence_landmarks(), valid_landmarks()]),
    };
    let mut runtime = runtime_with(detector, landmark);

    assert!(matches!(
        runtime.infer_frame(&frame(1)),
        Ok(FrameInferenceOutcome::NoFace)
    ));
    assert!(matches!(
        runtime.infer_frame(&frame(2)),
        Ok(FrameInferenceOutcome::Face(_))
    ));

    assert_eq!(detector_calls.load(Ordering::Relaxed), 2);
    assert_eq!(landmark_calls.load(Ordering::Relaxed), 2);
}

#[test]
fn composite_runtime_worker_reports_no_face_without_runtime_failure() {
    let detector = MockDetector {
        calls: Arc::new(AtomicUsize::new(0)),
        results: VecDeque::from([DetectorDecodeOutcome::NoFace]),
    };
    let landmark = MockLandmark {
        calls: Arc::new(AtomicUsize::new(0)),
        results: VecDeque::new(),
    };
    let runtime = runtime_with(detector, landmark);
    let frame_slot = Arc::new(LatestSlot::new());
    let output_slot = Arc::new(LatestSlot::new());
    let status: SharedStatus = Arc::new(std::sync::Mutex::new(
        vtuber_inference::InferenceWorkerStatus::new(),
    ));
    let worker = WorkerHandle::spawn("composite-runtime-test", {
        let frame_slot = Arc::clone(&frame_slot);
        let output_slot = Arc::clone(&output_slot);
        let status = Arc::clone(&status);
        move |stop| {
            run_composite_inference_worker(Box::new(runtime), stop, status, frame_slot, output_slot)
        }
    });

    frame_slot.publish(frame(1));
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        if status.lock().expect("test status mutex").no_face_frames == 1 {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }

    worker.stop();
    frame_slot.close();
    assert!(matches!(worker.join(), WorkerResult::Completed(_)));
    let final_status = status.lock().expect("test status mutex");
    assert_eq!(final_status.no_face_frames, 1);
    assert_eq!(final_status.consecutive_errors, 0);
    assert_eq!(final_status.state, InferenceWorkerState::Stopping);
}

fn runtime_with(
    detector: MockDetector,
    landmark: MockLandmark,
) -> CompositeRuntime<MockDetector, MockLandmark> {
    CompositeRuntime::new(
        descriptor(),
        detector,
        landmark,
        &RuntimeSettings {
            frame_wait_timeout_ms: 50,
            detector_interval_frames: 5,
        },
    )
    .expect("mock descriptor is valid")
}

fn detection() -> FaceDetection {
    FaceDetection {
        rect: NormalizedRect {
            x: 0.25,
            y: 0.25,
            width: 0.25,
            height: 0.5,
            rotation_rad: 0.0,
        },
        confidence: 0.9,
        anchor_index: 0,
    }
}

fn valid_landmarks() -> Vec<Landmark3> {
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

fn low_confidence_landmarks() -> Vec<Landmark3> {
    vec![
        Landmark3 {
            x: 0.5,
            y: 0.5,
            z: 0.0,
            visibility: 0.1,
        };
        98
    ]
}

fn frame(seq: u64) -> VideoFrame {
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

fn descriptor() -> FacePipelineDescriptor {
    let input = TensorContract {
        shape: vec![1, 3, 4, 4],
        dtype: "float32".into(),
        layout: TensorLayout::Nchw,
        channel_order: ChannelOrder::Rgb,
        value_domain: InputValueDomain::UnitFloat,
        normalization: NormalizationContract {
            mean: [0.5; 3],
            scale: [0.5; 3],
        },
    };
    FacePipelineDescriptor {
        id: "mock-composite".into(),
        detector: artifact(
            "mock-detector",
            ModelRole::FaceDetector,
            TensorContract {
                shape: vec![1, 3, 240, 320],
                dtype: "float32".into(),
                layout: TensorLayout::Nchw,
                channel_order: ChannelOrder::Rgb,
                value_domain: InputValueDomain::RawU8,
                normalization: NormalizationContract {
                    mean: [127.0; 3],
                    scale: [128.0; 3],
                },
            },
            vec![
                OutputTensorContract {
                    name: "scores".into(),
                    shape: vec![1, 4420, 2],
                    dtype: "float32".into(),
                    description: "scores".into(),
                },
                OutputTensorContract {
                    name: "boxes".into(),
                    shape: vec![1, 4420, 4],
                    dtype: "float32".into(),
                    description: "boxes".into(),
                },
            ],
        ),
        landmarks: artifact(
            "mock-landmarks",
            ModelRole::FaceLandmarks,
            input,
            vec![OutputTensorContract {
                name: "landmarks".into(),
                shape: vec![1, 98, 3],
                dtype: "float32".into(),
                description: "landmarks".into(),
            }],
        ),
        detector_postprocess: DetectorPostprocessConfig {
            score_threshold: 0.7,
            nms_iou: 0.3,
            max_pre_nms_candidates: 16,
            max_post_nms_detections: 4,
        },
        crop: FaceCropConfig {
            square_scale: 1.0,
            center_y_offset_fraction: 0.0,
            output_size: [4, 4],
            interpolation: CropInterpolation::Bilinear,
            outside_fill: CropOutsideFill::NormalizationMean,
        },
    }
}

fn artifact(
    id: &str,
    role: ModelRole,
    input: TensorContract,
    outputs: Vec<OutputTensorContract>,
) -> ModelArtifactDescriptor {
    ModelArtifactDescriptor {
        id: id.into(),
        role,
        file: "mock.onnx".into(),
        byte_size: 1,
        sha256: "0".repeat(64),
        input_name: "input".into(),
        source: "test".into(),
        upstream: "test".into(),
        license: "test".into(),
        license_url: None,
        input,
        outputs,
        requires_crop: matches!(role, ModelRole::FaceLandmarks),
        schema: matches!(role, ModelRole::FaceLandmarks).then(|| "peppapig-98".into()),
        landmark_coordinate_encoding: matches!(role, ModelRole::FaceLandmarks)
            .then(|| "normalized_0_1".into()),
        pose_method: None,
        representative_indices: Vec::new(),
    }
}
