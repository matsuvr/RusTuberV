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
//! lost in smoothing, loss/recovery, or the writer is caught here. The
//! synthetic landmarks place both eyes in the canonical 478-point layout.

use std::sync::Arc;
use std::time::Duration;

use vtuber_core::{
    ArkitBlendshape, CameraFaceTransform, FaceBlendshapeSet, FaceLandmark, FaceTrackingQuality,
    FaceTrackingSample, FrameSeq, MEDIAPIPE_FACE_LANDMARK_COUNT, MediaPipeBlendshape, MonoTimeNs,
};
use vtuber_tracking::{
    EyeGeometryThreshold, EyeGeometryThresholds, EyeSide, GeometryEyeClosureTracker,
    PipelineConfig, TrackingPipeline, eye_closure_features,
};

const DT: Duration = Duration::from_millis(33);
const IMAGE_SIZE: [u32; 2] = [640, 480];

const LEFT_INDICES: [usize; 8] = [362, 263, 385, 386, 387, 380, 374, 373];
const RIGHT_INDICES: [usize; 8] = [33, 133, 160, 159, 158, 144, 145, 153];

fn set_eye(landmarks: &mut [FaceLandmark], indices: &[usize; 8], openness: f32) {
    let pixels = [
        [0.0_f32, 0.0_f32],
        [0.4, 0.0],
        [0.1, -openness],
        [0.2, -openness],
        [0.3, -openness],
        [0.1, openness],
        [0.2, openness],
        [0.3, openness],
    ];
    for (pixel, index) in pixels.iter().zip(indices.iter()) {
        let landmark = &mut landmarks[*index];
        landmark.x = pixel[0] / IMAGE_SIZE[0] as f32;
        landmark.y = pixel[1] / IMAGE_SIZE[1] as f32;
    }
}

fn sample(
    seq: u64,
    captured_at: u64,
    left_blink: f32,
    right_blink: f32,
    left_gap: f32,
    right_gap: f32,
) -> FaceTrackingSample {
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
    let mut landmarks = vec![FaceLandmark::default(); MEDIAPIPE_FACE_LANDMARK_COUNT];
    set_eye(&mut landmarks, &LEFT_INDICES, left_gap);
    set_eye(&mut landmarks, &RIGHT_INDICES, right_gap);
    FaceTrackingSample {
        source_seq: FrameSeq(seq),
        captured_at: MonoTimeNs(captured_at),
        inference_started_at: MonoTimeNs(captured_at + 1_000_000),
        inference_finished_at: MonoTimeNs(captured_at + 20_000_000),
        camera_to_face: CameraFaceTransform::identity(),
        face_center: [0.5, 0.5],
        image_size: IMAGE_SIZE,
        landmarks: Arc::from(landmarks),
        blendshapes: FaceBlendshapeSet::from_pairs(&pairs).expect("synthetic blendshapes"),
        quality: FaceTrackingQuality {
            landmark_presence_median: Some(1.0),
            matrix_orthogonality_error: 0.0,
            matrix_determinant: 1.0,
        },
    }
}

fn pipeline() -> TrackingPipeline {
    let threshold = EyeGeometryThreshold::new(0.2, 0.5, 0.0).expect("geometry threshold");
    let thresholds = EyeGeometryThresholds::new(threshold, threshold);
    let mut pipeline = TrackingPipeline::new(PipelineConfig::default()).expect("pipeline");
    pipeline.set_eye_closure_thresholds(thresholds);
    pipeline
}

#[test]
fn a_closed_eye_reaches_exactly_one_on_both_routes() {
    let mut pipeline = pipeline();
    let mut last = None;
    for seq in 0..10 {
        let captured_at = seq * 33_000_000;
        let sample = sample(seq, captured_at, 0.9, 0.9, 0.0, 0.0);
        last = pipeline
            .update_mediapipe(Some(&sample), None, None, MonoTimeNs(captured_at), DT)
            .frame;
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
        let captured_at = seq * 33_000_000;
        let sample = sample(seq, captured_at, 0.9, 0.9, 0.0, 0.0);
        last = pipeline
            .update_mediapipe(Some(&sample), None, None, MonoTimeNs(captured_at), DT)
            .frame;
    }
    let closed = last.expect("closed frame");
    assert_eq!(closed.expressions.blink_left, 1.0);

    let open = sample(10, 330_000_000, 0.0, 0.0, 0.9, 0.9);
    let update = pipeline.update_mediapipe(Some(&open), None, None, MonoTimeNs(363_000_000), DT);
    let frame = update.frame.expect("open frame");
    assert!(
        frame.expressions.blink_left < 1.0,
        "release must leave the endpoint instead of sticking"
    );
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
        let captured_at = seq * 33_000_000;
        let sample = sample(seq, captured_at, 0.9, 0.0, 0.0, 0.9);
        last = pipeline
            .update_mediapipe(Some(&sample), None, None, MonoTimeNs(captured_at), DT)
            .frame;
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
fn high_blink_with_an_open_gap_does_not_pin() {
    let threshold = EyeGeometryThreshold::new(0.2, 0.5, 0.5).expect("geometry threshold");
    let thresholds = EyeGeometryThresholds::new(threshold, threshold);
    let mut pipeline = TrackingPipeline::new(PipelineConfig::default()).expect("pipeline");
    pipeline.set_eye_closure_thresholds(thresholds);
    let mut last = None;
    for seq in 0..10 {
        let captured_at = seq * 33_000_000;
        let sample = sample(seq, captured_at, 0.95, 0.95, 0.4, 0.4);
        last = pipeline
            .update_mediapipe(Some(&sample), None, None, MonoTimeNs(captured_at), DT)
            .frame;
    }
    let frame = last.expect("frame");
    assert!(
        frame.expressions.blink_left < 1.0,
        "raw blink alone must not reach the closure pin: {}",
        frame.expressions.blink_left
    );
}

#[test]
fn a_long_capture_gap_drops_the_carried_closure() {
    let mut pipeline = pipeline();
    let mut last = None;
    for seq in 0..5 {
        let captured_at = seq * 33_000_000;
        let sample = sample(seq, captured_at, 0.9, 0.9, 0.0, 0.0);
        last = pipeline
            .update_mediapipe(Some(&sample), None, None, MonoTimeNs(captured_at), DT)
            .frame;
    }
    assert!(last.expect("frame").expressions.blink_left > 0.99);
    // A 1 s capture gap with an open observation must not carry the latch.
    let open = sample(5, 1_033_000_000, 0.0, 0.0, 0.9, 0.9);
    let frame = pipeline
        .update_mediapipe(Some(&open), None, None, MonoTimeNs(1_066_000_000), DT)
        .frame
        .expect("frame");
    assert!(frame.expressions.blink_left < 1.0);
}

#[test]
fn live_pins_agree_with_the_offline_replay() {
    let threshold = EyeGeometryThreshold::new(0.2, 0.5, 0.0).expect("geometry threshold");
    let thresholds = EyeGeometryThresholds::new(threshold, threshold);
    let mut pipeline = TrackingPipeline::new(PipelineConfig::default()).expect("pipeline");
    pipeline.set_eye_closure_thresholds(thresholds);
    let mut offline = GeometryEyeClosureTracker::new(thresholds);
    for seq in 0..12 {
        let closed = seq < 6;
        let captured_at = seq * 33_000_000;
        let left_gap = if closed { 0.0 } else { 0.9 };
        let sample = sample(
            seq,
            captured_at,
            if closed { 0.9 } else { 0.0 },
            if closed { 0.9 } else { 0.0 },
            left_gap,
            left_gap,
        );
        let left = eye_closure_features(&sample, EyeSide::Left)
            .expect("features")
            .expect("lid geometry");
        let right = eye_closure_features(&sample, EyeSide::Right)
            .expect("features")
            .expect("lid geometry");
        let offline_state = offline.observe(
            sample.source_seq,
            sample.captured_at,
            [Some(left), Some(right)],
        );
        assert_eq!(offline_state.left.is_closed(), closed);
        // The consumer re-feeds the same observation at twice the rate.
        for tick in 0..2 {
            let now = captured_at + tick * 16_000_000;
            let frame = pipeline
                .update_mediapipe(Some(&sample), None, None, MonoTimeNs(now), DT)
                .frame
                .expect("frame");
            if closed {
                assert_eq!(frame.expressions.blink_left, 1.0);
            } else {
                assert!(frame.expressions.blink_left < 1.0);
            }
        }
    }
}

#[test]
fn without_a_profile_the_existing_output_is_unchanged() {
    let mut pipeline = TrackingPipeline::new(PipelineConfig::default()).expect("pipeline");
    assert!(!pipeline.eye_closure_active());
    let mut last = None;
    for seq in 0..10 {
        let captured_at = seq * 33_000_000;
        let sample = sample(seq, captured_at, 0.9, 0.9, 0.0, 0.0);
        last = pipeline
            .update_mediapipe(Some(&sample), None, None, MonoTimeNs(captured_at), DT)
            .frame;
    }
    let frame = last.expect("frame");
    assert!(
        frame.expressions.blink_left < 1.0,
        "without a profile the existing smoothing must still be visible"
    );
}
