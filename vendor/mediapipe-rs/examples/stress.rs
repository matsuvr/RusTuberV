//! Exercises every FFI path in a loop, for leak checking.
//!
//! MediaPipe leaks a fixed amount at startup (Mesa's EGL driver and MediaPipe's
//! own `GlContext` never free their one-time state), so an absolute leak total
//! means nothing. What matters is whether the total *grows with the iteration
//! count* — that is what a missed `Mp*CloseResult` or `MpImageFree` looks like.
//!
//! ```sh
//! RUSTFLAGS=-Zsanitizer=leak cargo +nightly run --target x86_64-unknown-linux-gnu \
//!     --example stress -- 1
//! RUSTFLAGS=-Zsanitizer=leak cargo +nightly run --target x86_64-unknown-linux-gnu \
//!     --example stress -- 25
//! ```
//!
//! Compare the two `SUMMARY: LeakSanitizer:` lines. Equal totals mean per-call
//! ownership is correct.

use std::sync::mpsc;

use mediapipe::{FaceDetector, FaceLandmarker, Image, ModelSource, Rotation, Timestamp};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let iterations: i64 = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "10".into())
        .parse()?;

    let mut detector =
        FaceDetector::builder(ModelSource::path("models/blaze_face_short_range.tflite")).build()?;
    let mut landmarker = FaceLandmarker::builder(ModelSource::path("models/face_landmarker.task"))
        .output_blendshapes(true)
        .output_transformation_matrixes(true)
        .build()?;
    let (tx, rx) = mpsc::channel();
    let mut stream = FaceLandmarker::builder(ModelSource::path("models/face_landmarker.task"))
        .build_stream(move |result, _, ts| {
            let _ = tx.send((ts, result.map(|r| r.landmarks.len()).unwrap_or(0)));
        })?;

    let mut faces = 0usize;
    for i in 0..iterations {
        // Decode afresh each time so image ownership is exercised too.
        let image = Image::from_file("models/portrait.jpg")?;
        faces += detector.detect(&image)?.len();
        faces += detector.detect_rotated(&image, Rotation::Cw90)?.len();
        faces += landmarker.detect(&image)?.landmarks.len();
        stream.send(&image, Timestamp::from_millis(i * 33))?;
        // Error paths allocate and free a C error string; exercise one.
        let _ = Image::from_file("does_not_exist.png");
    }

    drop(stream);
    let delivered = rx.try_iter().count();
    println!("{iterations} iterations, {faces} faces, {delivered} stream results");
    Ok(())
}
