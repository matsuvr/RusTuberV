//! Native Pose video ABI test.
//!
//! This drives `MpPoseLandmarkerCreate`, `MpPoseLandmarkerDetectForVideo`,
//! `MpPoseLandmarkerCloseResult`, and `MpPoseLandmarkerClose` through the same
//! published `libmediapipe` the application loads. It does not assert that a
//! person is found — a blank frame is allowed to produce no pose — only that the
//! real symbols execute without an ABI or schema error. Without the model
//! fixture the test skips rather than fails.

use std::path::Path;

use mediapipe::{Image, ModelSource, PoseLandmarker, Size, Timestamp};

const MODEL: &str = "../../assets/models/pose_landmarker_full.task";

fn model_available() -> bool {
    if Path::new(MODEL).exists() {
        true
    } else {
        eprintln!("skipping: {MODEL} is not present; run the manifest download first");
        false
    }
}

#[test]
fn pose_video_symbols_create_detect_and_close() {
    if !model_available() {
        return;
    }
    let mut landmarker = PoseLandmarker::builder(ModelSource::path(MODEL))
        .build_for_video()
        .expect("native pose landmarker should be created");

    let size = Size {
        width: 64,
        height: 64,
    };
    let pixels = vec![128_u8; 64 * 64 * 3];
    let image = Image::from_rgb(size, &pixels).expect("blank RGB image");

    let first = landmarker.detect_for_video(&image, Timestamp::from_millis(0));
    assert!(first.is_ok(), "first video frame failed: {first:?}");

    let second = landmarker.detect_for_video(&image, Timestamp::from_millis(33));
    let result = second.expect("second video frame failed");
    assert!(result.pose_world_landmarks.len() <= 1);
    assert_eq!(
        result.pose_landmarks.len(),
        result.pose_world_landmarks.len()
    );
}
