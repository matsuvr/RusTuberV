// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Application-level checks for the production MediaPipe inference handoff.

use std::sync::Arc;
use std::time::Duration;

use vtuber_app::inference_runtime::InferenceRuntime;
use vtuber_core::{LatestSlot, VideoFrame};
use vtuber_inference::{FailureStage, InferenceWorkerState};

#[test]
fn production_start_queues_mediapipe_pipeline_and_classifies_load_failure() {
    let project = tempfile::tempdir().expect("temporary project root");
    let model_root = project.path().join("assets").join("models");
    std::fs::create_dir_all(&model_root).expect("model directory");

    // A task file that is present but fails the pinned SHA-256 check must be
    // rejected. A missing file legitimately falls back to the embedded bundle.
    let task_path = model_root.join("face_landmarker.task");
    std::fs::write(&task_path, b"corrupt task bundle").expect("write corrupt task bundle");

    let frame_slot: Arc<LatestSlot<VideoFrame>> = Arc::new(LatestSlot::new());
    let mut runtime = InferenceRuntime::new(frame_slot, project.path().to_path_buf());
    runtime
        .start_model()
        .expect("pipeline load command should be queued");

    for _ in 0..100 {
        if runtime.status().state == InferenceWorkerState::Failed {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    let status = runtime.status();
    assert_eq!(status.state, InferenceWorkerState::Failed);
    assert_eq!(
        status.last_failure.as_ref().map(|failure| failure.stage),
        Some(FailureStage::ModelLoad)
    );

    runtime.stop_model();
}

#[test]
fn production_status_exposes_mediapipe_identity_before_task_load() {
    let project = tempfile::tempdir().expect("temporary project root");
    let frame_slot: Arc<LatestSlot<VideoFrame>> = Arc::new(LatestSlot::new());
    let mut runtime = InferenceRuntime::new(frame_slot, project.path().to_path_buf());
    runtime
        .start_model()
        .expect("pipeline load command should be queued");

    for _ in 0..100 {
        if runtime.status().state == InferenceWorkerState::Failed {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    let status = runtime.status();
    assert_eq!(
        status.pipeline_id.as_deref(),
        Some("mediapipe-face-landmarker")
    );
    assert_eq!(
        status.detector_model_hash.as_deref(),
        Some(vtuber_inference::backend::mediapipe::TASK_BUNDLE_SHA256)
    );
    assert_eq!(status.landmark_model_hash, None);

    runtime.stop_model();
    assert_eq!(runtime.status().state, InferenceWorkerState::Idle);
}
