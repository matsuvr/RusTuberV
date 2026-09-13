//! Integration tests. Run `./scripts/fetch-fixtures.sh` first; without the
//! fixtures every test skips rather than fails.
//!
//! The expected numbers below were measured by driving the MediaPipe C API
//! directly, before any of this crate existed. That makes them an ABI check as
//! much as a behaviour check: if the options-struct layout picked for the loaded
//! library were wrong, these would not merely drift, they would fail outright.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::mpsc;

use mediapipe::{
    Confidence, Delegate, Error, FaceDetector, FaceLandmarker, Image, IouThreshold, ModelSource,
    Rotation, Size, Timestamp,
};

const DETECTOR: &str = "models/blaze_face_short_range.tflite";
const LANDMARKER: &str = "models/face_landmarker.task";
const PORTRAIT: &str = "models/portrait.jpg";

/// True when fixtures are present. Tests return early otherwise so a fresh
/// checkout without network does not report failures.
fn fixtures() -> bool {
    let ok = Path::new(DETECTOR).exists()
        && Path::new(LANDMARKER).exists()
        && Path::new(PORTRAIT).exists();
    if !ok {
        eprintln!("skipping: run ./scripts/fetch-fixtures.sh first");
    }
    ok
}

fn detector() -> FaceDetector {
    FaceDetector::builder(ModelSource::path(DETECTOR))
        .delegate(Delegate::Cpu)
        .min_suppression_threshold(
            IouThreshold::new(0.3).expect("0.3 should be a valid overlap ratio"),
        )
        .build()
        .expect("the fetched BlazeFace model should build a detector")
}

fn portrait() -> Image {
    Image::from_file(PORTRAIT).expect("the fetched portrait should decode")
}

#[test]
fn detects_one_face_at_the_known_position() {
    if !fixtures() {
        return;
    }
    let image = portrait();
    assert_eq!(
        image.size(),
        Size {
            width: 820,
            height: 1024
        }
    );

    let faces = detector()
        .detect(&image)
        .expect("an image-mode detector should accept an image");
    assert_eq!(faces.len(), 1, "portrait.jpg has exactly one face");

    let face = &faces[0];
    let score = face
        .score()
        .expect("BlazeFace should report one scored category per face")
        .get();
    assert!(score > 0.9, "score {score} should be > 0.9");

    let b = face.bounding_box;
    for (name, got, want) in [
        ("left", b.left(), 283),
        ("top", b.top(), 115),
        ("width", b.width(), 234),
        ("height", b.height(), 234),
    ] {
        assert!(
            (got - want).abs() <= 10,
            "{name}: got {got}, expected ~{want}"
        );
    }
    assert_eq!(face.keypoints.len(), 6);
}

/// The bounding box is in pixels and the keypoints are normalized. If those two
/// spaces were ever wired to the wrong C fields, the eyes and nose would not
/// land inside the face.
#[test]
fn keypoints_convert_into_the_pixel_bounding_box() {
    if !fixtures() {
        return;
    }
    let image = portrait();
    let faces = detector()
        .detect(&image)
        .expect("an image-mode detector should accept an image");
    let face = &faces[0];

    for (i, kp) in face.keypoints.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&kp.point.x()) && (0.0..=1.0).contains(&kp.point.y()),
            "keypoint {i} is not normalized: {:?}",
            kp.point
        );
        let p = kp.point.to_pixels(image.size());
        assert!(
            face.bounding_box.contains(p),
            "keypoint {i} at {p:?} falls outside {:?}",
            face.bounding_box
        );
    }
}

#[test]
fn landmarks_blendshapes_and_transform() {
    if !fixtures() {
        return;
    }
    let mut landmarker = FaceLandmarker::builder(ModelSource::path(LANDMARKER))
        .num_faces(NonZeroU32::new(1).expect("1 should be non-zero"))
        .output_blendshapes(true)
        .output_transformation_matrixes(true)
        .build()
        .expect("the fetched face_landmarker.task should build a landmarker");

    let result = landmarker
        .detect(&portrait())
        .expect("an image-mode landmarker should accept an image");

    assert_eq!(result.landmarks.len(), 1);
    assert_eq!(result.landmarks[0].len(), 478, "standard face mesh");
    for lm in &result.landmarks[0] {
        assert!((0.0..=1.0).contains(&lm.point.x()) && (0.0..=1.0).contains(&lm.point.y()));
    }

    assert_eq!(result.blendshapes.len(), 1);
    assert_eq!(result.blendshapes[0].len(), 52, "ARKit-style coefficients");
    assert!(
        result.blendshapes[0]
            .iter()
            .all(|c| c.category_name.is_some()),
        "blendshapes should be named"
    );

    assert_eq!(result.transformation_matrixes.len(), 1);
    let m = result.transformation_matrixes[0];
    // An affine transform's bottom row is (0, 0, 0, 1). This also pins down the
    // column-major indexing in Transform4x4::get.
    assert_eq!(
        [m.get(3, 0), m.get(3, 1), m.get(3, 2), m.get(3, 3)],
        [0.0, 0.0, 0.0, 1.0],
        "bottom row of an affine transform, read column-major"
    );
    assert_eq!(m.to_row_major()[12..16], [0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn video_mode_accepts_increasing_timestamps() {
    if !fixtures() {
        return;
    }
    let mut detector = FaceDetector::builder(ModelSource::path(DETECTOR))
        .min_suppression_threshold(
            IouThreshold::new(0.3).expect("0.3 should be a valid overlap ratio"),
        )
        .build_for_video()
        .expect("the fetched BlazeFace model should build a video detector");

    let image = portrait();
    for frame in 0..10 {
        let ts = Timestamp::from_millis(frame * 33);
        let faces = detector
            .detect_for_video(&image, ts)
            .unwrap_or_else(|e| panic!("frame {frame}: {e}"));
        assert_eq!(faces.len(), 1, "frame {frame}");
    }
}

/// A video-mode handle must reject an image-mode call rather than misbehave.
#[test]
fn wrong_running_mode_is_an_error() {
    if !fixtures() {
        return;
    }
    let mut detector = FaceDetector::builder(ModelSource::path(DETECTOR))
        .build_for_video()
        .expect("the fetched BlazeFace model should build a video detector");
    let err = detector
        .detect(&portrait())
        .expect_err("a video-mode task should reject an image-mode call");
    assert!(matches!(err, Error::Mp { .. }), "got {err:?}");
}

/// Live-stream mode is lossy by design: MediaPipe flow-limits the graph and
/// drops frames that arrive while it is busy, which is the whole difference
/// between it and video mode. So this asserts the invariants that must hold —
/// results are delivered, in order, from the frames we sent, and delivery stops
/// cleanly at drop — rather than a frame count, which would be flaky.
#[test]
fn live_stream_delivers_results_in_order_and_stops_at_drop() {
    if !fixtures() {
        return;
    }
    let (tx, rx) = mpsc::channel();
    let mut stream = FaceLandmarker::builder(ModelSource::path(LANDMARKER))
        .build_stream(move |result, image, timestamp| {
            let faces = result.map(|r| r.landmarks.len()).unwrap_or(usize::MAX);
            // The callback's image borrows a stack MpImage owned by MediaPipe;
            // reading it here is fine, keeping it would not be.
            tx.send((timestamp, faces, image.size()))
                .expect("the receiver should outlive the stream");
        })
        .expect("a free callback slot should be available");

    let image = portrait();
    const FRAMES: i64 = 10;
    let sent: Vec<Timestamp> = (0..FRAMES)
        .map(|f| Timestamp::from_millis(f * 33))
        .collect();
    for (frame, ts) in sent.iter().enumerate() {
        stream
            .send(&image, *ts)
            .unwrap_or_else(|e| panic!("frame {frame}: {e}"));
    }

    // Dropping closes the task, which flushes and joins MediaPipe's worker, so
    // every callback that will ever run has run by the time this returns.
    drop(stream);

    let got: Vec<_> = rx.try_iter().collect();
    assert!(!got.is_empty(), "no results at all");
    assert!(got.len() <= sent.len(), "more results than frames sent");

    let mut last: Option<Timestamp> = None;
    for (ts, faces, size) in &got {
        assert!(sent.contains(ts), "unexpected timestamp {ts:?}");
        if let Some(prev) = last {
            assert!(*ts > prev, "timestamps out of order: {prev:?} then {ts:?}");
        }
        last = Some(*ts);
        assert_eq!(*faces, 1, "at {ts:?}");
        assert_eq!(
            *size,
            Size {
                width: 820,
                height: 1024
            }
        );
    }

    // The sender lived in the dropped callback, so the channel being closed is
    // proof that the slot was released and no callback can fire again.
    assert!(
        matches!(rx.recv(), Err(mpsc::RecvError)),
        "callback outlived the stream"
    );

    // And the freed slot must be reusable.
    let again =
        FaceLandmarker::builder(ModelSource::path(LANDMARKER)).build_stream(move |_, _, _| {});
    assert!(again.is_ok(), "callback slot was not released on drop");
}

/// `send` hands MediaPipe a frame and returns immediately, so the natural thing
/// for a caller to do is drop the frame right away. That is only sound if the
/// underlying `mediapipe::Image` is internally shared — which it is, but it is
/// worth proving rather than assuming, because the failure mode would be a
/// use-after-free on a worker thread rather than a clean error.
#[test]
fn input_image_may_be_dropped_immediately_after_send() {
    if !fixtures() {
        return;
    }
    let (tx, rx) = mpsc::channel();
    let mut stream = FaceLandmarker::builder(ModelSource::path(LANDMARKER))
        .build_stream(move |result, _, ts| {
            let n = result.map(|r| r.landmarks.len()).unwrap_or(usize::MAX);
            let _ = tx.send((ts, n));
        })
        .expect("a free callback slot should be available");

    for frame in 0..30 {
        // Fresh decode each iteration, dropped before the next send.
        let image = portrait();
        stream
            .send(&image, Timestamp::from_millis(frame * 33))
            .unwrap_or_else(|e| panic!("frame {frame}: {e}"));
        drop(image);
    }
    drop(stream);

    let got: Vec<_> = rx.try_iter().collect();
    assert!(!got.is_empty(), "no results survived dropping the inputs");
    for (ts, n) in got {
        assert_eq!(n, 1, "corrupt result at {ts:?}");
    }
}

/// The pool is finite; exhausting it must be a clean error, not a panic or a
/// silently mis-routed callback.
#[test]
fn live_stream_slot_pool_is_bounded() {
    if !fixtures() {
        return;
    }
    let mut streams = Vec::new();
    let mut exhausted = None;
    for i in 0..16 {
        match FaceLandmarker::builder(ModelSource::path(LANDMARKER)).build_stream(move |_, _, _| {})
        {
            Ok(s) => streams.push(s),
            Err(e) => {
                exhausted = Some((i, e));
                break;
            }
        }
    }
    let (i, err) = exhausted.expect("the 8-slot pool should be exhausted within 16 attempts");
    assert!(matches!(err, Error::NoFreeSlot(8)), "got {err:?}");
    // Not `== 8`: cargo runs tests in parallel and the other live-stream test
    // may be holding a slot. The invariant is that the pool is bounded at 8 and
    // refuses cleanly, not that this test gets all of them.
    assert!(
        (1..=8).contains(&i),
        "claimed {i} streams before the pool of 8 ran out"
    );
    assert_eq!(
        streams.len(),
        i,
        "every successful build should hold a slot"
    );

    drop(streams);
    assert!(
        FaceLandmarker::builder(ModelSource::path(LANDMARKER))
            .build_stream(move |_, _, _| {})
            .is_ok(),
        "slots not reusable after dropping every stream"
    );
}

/// Rotation reaches MediaPipe. An upright portrait rotated 90 degrees no longer
/// looks like an upright face to BlazeFace, so the detection should change —
/// that is what proves the option was applied rather than silently dropped.
#[test]
fn rotation_is_applied() {
    if !fixtures() {
        return;
    }
    let image = portrait();
    let upright = detector()
        .detect(&image)
        .expect("an image-mode detector should accept an image");
    assert_eq!(upright.len(), 1);

    let rotated = detector()
        .detect_rotated(&image, Rotation::Cw90)
        .expect("a 90-degree rotation should be accepted");
    assert_ne!(
        rotated.first().map(|f| f.bounding_box),
        upright.first().map(|f| f.bounding_box),
        "rotation was accepted but had no effect"
    );

    // Rotation::None must match the plain call exactly.
    let none = detector()
        .detect_rotated(&image, Rotation::None)
        .expect("a zero rotation should be accepted");
    assert_eq!(none[0].bounding_box, upright[0].bounding_box);
}

#[test]
fn invalid_inputs_are_rejected_before_reaching_c() {
    // Out-of-range values are rejected where they are written, not deep inside
    // build(), and the two 0..1 quantities report themselves distinctly.
    match Confidence::new(1.5).expect_err("1.5 should be rejected, confidences are 0..=1") {
        Error::OutOfRange {
            quantity, value, ..
        } => {
            assert_eq!(quantity, "confidence");
            assert_eq!(value, 1.5);
        }
        other => panic!("got {other:?}"),
    }
    match IouThreshold::new(-0.1).expect_err("-0.1 should be rejected, overlap ratios are 0..=1") {
        Error::OutOfRange { quantity, .. } => assert_eq!(quantity, "IoU threshold"),
        other => panic!("got {other:?}"),
    }

    // A short pixel buffer would be an out-of-bounds read inside MediaPipe.
    let err = Image::from_rgb(
        Size {
            width: 4,
            height: 4,
        },
        &[0u8; 10],
    )
    .expect_err("a 10-byte buffer should be rejected for a 4x4 RGB image");
    assert!(
        matches!(
            err,
            Error::BufferSize {
                expected: 48,
                got: 10,
                ..
            }
        ),
        "got {err:?}"
    );
}

#[test]
fn missing_model_reports_a_message() {
    let err = FaceDetector::builder(ModelSource::path("does_not_exist.tflite"))
        .build()
        .expect_err("a nonexistent model path should fail to build");
    match err {
        Error::Mp { message, .. } => assert!(!message.is_empty(), "empty error message"),
        other => panic!("expected Error::Mp, got {other:?}"),
    }
}

/// Same-space geometry is available on the types, so callers have no reason to
/// pull raw floats out and risk mixing spaces by hand.
#[test]
fn same_space_geometry() {
    if !fixtures() {
        return;
    }
    let image = portrait();
    let faces = detector()
        .detect(&image)
        .expect("an image-mode detector should accept an image");
    let kp = &faces[0].keypoints;

    // Keypoints 0 and 1 are the two eyes: measurably apart, and well under the
    // width of the frame.
    let eye_gap_norm = kp[0].point.distance_to(kp[1].point);
    assert!(
        (0.05..0.5).contains(&eye_gap_norm),
        "normalized eye separation {eye_gap_norm} is implausible"
    );

    let eye_gap_px = kp[0]
        .point
        .to_pixels(image.size())
        .distance_to(kp[1].point.to_pixels(image.size()));
    assert!(
        eye_gap_px > eye_gap_norm,
        "a pixel distance on an 820x1024 image should exceed the normalized one"
    );
    assert!(eye_gap_px < faces[0].bounding_box.width() as f32);
}

/// A landmark's `z` is a model-defined depth, not an image coordinate, so `xy()`
/// exists to drop it rather than have callers reach past it.
#[test]
fn landmark_xy_drops_the_incomparable_z() {
    if !fixtures() {
        return;
    }
    let mut landmarker = FaceLandmarker::builder(ModelSource::path(LANDMARKER))
        .build()
        .expect("the fetched face_landmarker.task should build a landmarker");
    let result = landmarker
        .detect(&portrait())
        .expect("an image-mode landmarker should accept an image");

    let lm = result.landmarks[0][0].point;
    assert_eq!(lm.xy().x(), lm.x());
    assert_eq!(lm.xy().y(), lm.y());
    assert_eq!(
        lm.to_pixels(Size {
            width: 100,
            height: 100
        }),
        lm.xy().to_pixels(Size {
            width: 100,
            height: 100
        })
    );
}

#[test]
fn image_from_rgb_roundtrips_dimensions() {
    let size = Size {
        width: 64,
        height: 48,
    };
    let image = Image::from_rgb(size, &vec![128u8; 64 * 48 * 3])
        .expect("a correctly sized RGB buffer should be accepted");
    assert_eq!(image.size(), size);
    assert_eq!(image.channels(), 3);
}

/// Reports which ABI was detected, and asserts the two option layouts really do
/// differ by the 16 bytes that `BaseOptions` grew. The `const` assertions in
/// `sys::compat` cover the exact offsets at compile time.
#[test]
fn abi_detection() {
    let lib = mediapipe::loader::lib()
        .expect("libmediapipe should be loadable, the other tests depend on it");
    eprintln!("abi={:?} source={:?}", lib.abi, lib.source);

    use mediapipe::sys;
    assert_eq!(size_of::<sys::MpBaseOptions>(), 72);
    assert_eq!(size_of::<sys::compat::MpBaseOptionsV35>(), 56);
    assert_eq!(
        size_of::<sys::MpFaceDetectorOptions>()
            - size_of::<sys::compat::MpFaceDetectorOptionsV35>(),
        16
    );
    assert_eq!(
        size_of::<sys::MpFaceLandmarkerOptions>()
            - size_of::<sys::compat::MpFaceLandmarkerOptionsV35>(),
        16
    );
}
