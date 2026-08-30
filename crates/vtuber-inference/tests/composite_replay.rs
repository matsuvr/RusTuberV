// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Fixed-sequence replay and allocation-boundedness checks.

#![cfg(feature = "legacy-face-stack")]

#[path = "support/composite_fixture.rs"]
mod fixture;

use std::sync::atomic::Ordering;

use vtuber_core::types::FrameSeq;
use vtuber_inference::detector::DetectorDecodeOutcome;

type ReplayRow = (FrameSeq, f32, f32, f32, usize);
type ReplaySummary = (Vec<ReplayRow>, usize, usize, (usize, usize));
use vtuber_inference::{FrameFaceInference, FrameInferenceOutcome};

#[test]
fn composite_replay_repeats_the_same_outcomes_and_keeps_buffers_bounded() {
    let (first, first_detector_calls, first_landmark_calls, first_capacity) = replay_once();
    let (second, second_detector_calls, second_landmark_calls, second_capacity) = replay_once();

    assert_eq!(first, second);
    assert_eq!(first_detector_calls, 13);
    assert_eq!(second_detector_calls, 13);
    assert_eq!(first_landmark_calls, 64);
    assert_eq!(second_landmark_calls, 64);
    assert_eq!(first_capacity, second_capacity);
}

fn replay_once() -> ReplaySummary {
    let detector =
        fixture::MockDetector::new([DetectorDecodeOutcome::Detections(
            vec![fixture::detection()],
        )]);
    let detector_calls = detector.calls.clone();
    let landmark = fixture::MockLandmark::new([]);
    let landmark_calls = landmark.calls.clone();
    let mut runtime = fixture::runtime(detector, landmark);
    let initial_capacity = runtime.crop_buffer_capacities();
    let mut replay = Vec::with_capacity(64);

    for sequence in 1..=64 {
        let outcome = runtime
            .infer_frame(&fixture::mean_frame(sequence))
            .expect("fixed synthetic replay should not fail");
        let FrameInferenceOutcome::Face(observation) = outcome else {
            panic!("fixed synthetic replay unexpectedly lost face at {sequence}");
        };
        replay.push((
            observation.source_seq,
            observation.landmarks[0].x,
            observation.landmarks[0].y,
            observation.face_confidence,
            observation.landmarks.len(),
        ));
        let _ = runtime.take_timing();
    }

    assert_eq!(runtime.crop_buffer_capacities(), initial_capacity);
    (
        replay,
        detector_calls.load(Ordering::Relaxed),
        landmark_calls.load(Ordering::Relaxed),
        initial_capacity,
    )
}
