// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Integration tests for the production inference worker.
//!
//! These tests exercise the public [`InferenceController`] API end-to-end:
//! startup, frame consumption, output publication, and clean shutdown. They
//! require the `onnx` feature because the golden production model is an ONNX
//! file.

#![cfg(feature = "legacy-face-stack")]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use vtuber_core::types::{
    FrameSeq, LandmarkSchemaId, MonoTimeNs, PixelFormat, RawFaceObservation, VideoFrame,
};
use vtuber_core::{LatestSlot, ReadResult};
use vtuber_inference::descriptor::{ChannelOrder, ModelDescriptor, ModelFormat, Normalization};
use vtuber_inference::state::InferenceWorkerState;
use vtuber_inference::{InferenceController, RuntimeSettings};

fn peppa_descriptor() -> ModelDescriptor {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop();
    path.pop();
    path.push("assets");
    path.push("models");
    path.push("peppapig_student_1x3x256x256.onnx");

    ModelDescriptor {
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
    }
}

fn make_frame(seq: u64) -> VideoFrame {
    VideoFrame {
        seq: FrameSeq(seq),
        captured_at: MonoTimeNs(seq * 1_000_000),
        width: 256,
        height: 256,
        stride_bytes: 256 * 3,
        format: PixelFormat::Rgb8,
        data: vec![(seq % 256) as u8; 256 * 256 * 3].into(),
    }
}

#[test]
#[ignore = "requires the downloaded legacy Peppa Pig ONNX research artifact"]
fn golden_model_startup_frames_output_stop() {
    let frame_slot: Arc<LatestSlot<VideoFrame>> = Arc::new(LatestSlot::new());
    let output_slot: Arc<LatestSlot<RawFaceObservation>> = Arc::new(LatestSlot::new());

    let mut controller =
        InferenceController::new(Arc::clone(&frame_slot), Arc::clone(&output_slot));

    controller.start_worker().expect("start worker");
    controller
        .load_model(
            peppa_descriptor(),
            RuntimeSettings {
                frame_wait_timeout_ms: 100,
                detector_interval_frames: 1,
            },
        )
        .expect("send load model");

    // Wait for the model to load and reach Running.
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let status = controller.status();
        if status.state == InferenceWorkerState::Running {
            break;
        }
        if status.state == InferenceWorkerState::Failed {
            panic!("worker failed during model load: {:?}", status.last_failure);
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for worker to reach Running"
        );
        std::thread::sleep(Duration::from_millis(50));
    }

    // Publish frames at roughly 30 fps while the model runs.
    const FRAME_COUNT: u64 = 8;
    for seq in 1..=FRAME_COUNT {
        frame_slot.publish(make_frame(seq));
        std::thread::sleep(Duration::from_millis(33));
    }

    // Wait for the worker to drain the backlog.
    let deadline = Instant::now() + Duration::from_secs(15);
    while controller.status().frames_processed < FRAME_COUNT && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }

    let status = controller.status();
    assert!(
        status.frames_processed >= 1,
        "worker should process at least one frame, got {}",
        status.frames_processed
    );
    assert_eq!(
        status.state,
        InferenceWorkerState::Running,
        "worker should remain Running after frames"
    );

    // Read the latest published observation before shutdown clears the slot.
    let observation = output_slot
        .try_read_after(0)
        .expect("output slot should contain a value")
        .expect_new("output slot should contain a new value");

    assert_eq!(observation.schema.0, "peppapig-98");
    assert!(
        !observation.landmarks.is_empty(),
        "observation should contain landmarks"
    );
    assert_eq!(
        observation.source_seq,
        status
            .last_source_seq
            .expect("last source seq should be set"),
        "observation source_seq should match the last processed frame"
    );
    assert_eq!(
        observation.captured_at,
        MonoTimeNs(observation.source_seq.0 * 1_000_000),
        "captured_at should be carried from the source frame"
    );

    // Clean shutdown must join the worker and close both slots.
    let metrics = controller.shutdown();
    assert!(frame_slot.is_closed());
    assert!(output_slot.is_closed());

    assert_eq!(
        metrics.drops.processed, status.frames_processed,
        "metrics processed counter should match status"
    );
}

/// Helper to unwrap a [`ReadResult`] as [`ReadResult::New`].
trait ExpectNew<T> {
    fn expect_new(self, msg: &str) -> T;
}

impl<T> ExpectNew<T> for ReadResult<T> {
    fn expect_new(self, msg: &str) -> T {
        match self {
            ReadResult::New(value) => value,
            ReadResult::Closed => panic!("{msg}: slot was closed"),
        }
    }
}
