//! Native Hand Landmarker video ABI test.
//!
//! This drives `MpHandLandmarkerCreate`, `MpHandLandmarkerDetectForVideo`,
//! `MpHandLandmarkerCloseResult`, and `MpHandLandmarkerClose` through the same
//! published `libmediapipe` the application loads. It does not assert that a
//! hand is found — a blank frame is allowed to produce an empty result — only
//! that the real symbols execute without an ABI or schema error. Without the
//! model fixture the test skips rather than fails.

use std::path::Path;

use mediapipe::{HandLandmarker, Image, ModelSource, Size, Timestamp};

const MODEL: &str = "../../assets/models/hand_landmarker.task";

fn model_available() -> bool {
    if Path::new(MODEL).exists() {
        true
    } else {
        eprintln!("skipping: {MODEL} is not present; run the manifest download first");
        false
    }
}

#[test]
fn hand_video_symbols_create_detect_and_close() {
    if !model_available() {
        return;
    }
    let mut landmarker = HandLandmarker::builder(ModelSource::path(MODEL))
        .build_for_video()
        .expect("native hand landmarker should be created");

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
    assert!(result.hand_landmarks.len() <= 2);
    assert_eq!(result.hand_landmarks.len(), result.hand_world_landmarks.len());
    assert_eq!(result.hand_landmarks.len(), result.handedness.len());
}
