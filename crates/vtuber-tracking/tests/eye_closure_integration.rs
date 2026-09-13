// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! End-to-end eye-closure correction through the real tracking pipeline.
//!
//! These tests drive `TrackingPipeline::update_mediapipe` the same way the app
//! does and assert on the final `AvatarControlFrame`, so a correction that is
//! lost in smoothing, loss/recovery, or the writer is caught here.

use std::time::Duration;

use vtuber_core::{
    ArkitBlendshape, CameraFaceTransform, FaceBlendshapeSet, FaceLandmark, FaceTrackingQuality,
    FaceTrackingSample, FrameSeq, MEDIAPIPE_FACE_LANDMARK_COUNT, MediaPipeBlendshape, MonoTimeNs,
};
use vtuber_tracking::{EyeClosureThresholds, EyeThreshold, PipelineConfig, TrackingPipeline};

const DT: Duration = Duration::from_millis(33);

fn sample(seq: u64, left_blink: f32, right_blink: f32) -> FaceTrackingSample {
    let pairs: Vec<(&str, f32)> = MediaPipeBlendshape::ALL
        .iter()
        .map(|channel| {
            let value = match channel {
                MediaPipeBlendshape::EyeBlinkLeft => left_blink,
                MediaPipeBlendshape::EyeBlinkRight => right_blink,
                _ => 0.0,
            };
            (channel.as_str(), value)
        })
        .collect();
    FaceTrackingSample {
        source_seq: FrameSeq(seq),
        captured_at: MonoTimeNs(seq * 33_000_000),
        inference_started_at: MonoTimeNs(seq * 33_000_000 + 1_000_000),
        inference_finished_at: MonoTimeNs(seq * 33_000_000 + 20_000_000),
        camera_to_face: CameraFaceTransform::identity(),
        face_center: [0.5, 0.5],
        landmarks: vec![FaceLandmark::default(); MEDIAPIPE_FACE_LANDMARK_COUNT].into(),
        blendshapes: FaceBlendshapeSet::from_pairs(&pairs).expect("synthetic blendshapes"),
        quality: FaceTrackingQuality {
            landmark_presence_median: Some(1.0),
            matrix_orthogonality_error: 0.0,
            matrix_determinant: 1.0,
        },
    }
}

fn pipeline() -> TrackingPipeline {
    let thresholds = EyeClosureThresholds::new(
        EyeThreshold::new(0.4, 0.6).expect("left"),
        EyeThreshold::new(0.4, 0.6).expect("right"),
    );
    let mut pipeline = TrackingPipeline::new(PipelineConfig::default()).expect("pipeline");
    pipeline.set_eye_closure_thresholds(thresholds);
    pipeline
}

#[test]
fn a_closed_eye_reaches_exactly_one_on_both_routes() {
    let mut pipeline = pipeline();
    let mut last = None;
    for seq in 0..10 {
        let sample = sample(seq, 0.9, 0.9);
        let update =
            pipeline.update_mediapipe(Some(&sample), None, None, MonoTimeNs(seq * 33_000_000), DT);
        last = update.frame;
    }
    let frame = last.expect("a tracked frame is emitted");
    assert_eq!(frame.expressions.blink_left, 1.0);
    assert_eq!(frame.expressions.blink_right, 1.0);
    let detailed = frame.detailed_face.as_ref().expect("detailed route");
    assert_eq!(detailed.get(ArkitBlendshape::EyeBlinkLeft), 1.0);
    assert_eq!(detailed.get(ArkitBlendshape::EyeBlinkRight), 1.0);
}

#[test]
fn releasing_a_closed_eye_leaves_one_smoothly() {
    let mut pipeline = pipeline();
    let mut last = None;
    for seq in 0..10 {
        let sample = sample(seq, 0.9, 0.9);
        let update =
            pipeline.update_mediapipe(Some(&sample), None, None, MonoTimeNs(seq * 33_000_000), DT);
        last = update.frame;
    }
    let closed = last.expect("closed frame");
    assert_eq!(closed.expressions.blink_left, 1.0);

    let open = sample(10, 0.0, 0.0);
    let update =
        pipeline.update_mediapipe(Some(&open), None, None, MonoTimeNs(11 * 33_000_000), DT);
    let frame = update.frame.expect("open frame");
    assert!(
        frame.expressions.blink_left < 1.0,
        "release must leave the endpoint instead of sticking"
    );
    assert!(frame.expressions.blink_left > 0.0);
    assert!(
        frame
            .detailed_face
            .unwrap()
            .get(ArkitBlendshape::EyeBlinkLeft)
            < 1.0
    );
}

#[test]
fn a_wink_closes_only_the_intended_eye() {
    let mut pipeline = pipeline();
    let mut last = None;
    for seq in 0..10 {
        let sample = sample(seq, 0.9, 0.0);
        let update =
            pipeline.update_mediapipe(Some(&sample), None, None, MonoTimeNs(seq * 33_000_000), DT);
        last = update.frame;
    }
    let frame = last.expect("wink frame");
    assert_eq!(frame.expressions.blink_left, 1.0);
    assert!(
        frame.expressions.blink_right < 0.1,
        "the opposite eye must not close: {}",
        frame.expressions.blink_right
    );
    let detailed = frame.detailed_face.unwrap();
    assert_eq!(detailed.get(ArkitBlendshape::EyeBlinkLeft), 1.0);
    assert!(detailed.get(ArkitBlendshape::EyeBlinkRight) < 0.1);
}

#[test]
fn without_a_profile_the_existing_output_is_unchanged() {
    let mut pipeline = TrackingPipeline::new(PipelineConfig::default()).expect("pipeline");
    assert!(!pipeline.eye_closure_active());
    let mut last = None;
    for seq in 0..10 {
        let sample = sample(seq, 0.9, 0.9);
        let update =
            pipeline.update_mediapipe(Some(&sample), None, None, MonoTimeNs(seq * 33_000_000), DT);
        last = update.frame;
    }
    let frame = last.expect("frame");
    assert!(
        frame.expressions.blink_left < 1.0,
        "without a profile the existing smoothing must still be visible"
    );
}
