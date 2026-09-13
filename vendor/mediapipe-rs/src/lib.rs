//! Safe Rust bindings to the [MediaPipe Tasks C API][c-api].
//!
//! The native library is not linked; it is `dlopen`ed on first use. With the
//! default `download` feature it is fetched once from the official PyPI wheel
//! into the user cache, so nothing has to be installed or built beforehand.
//! Set `$MEDIAPIPE_LIB` to use a specific `libmediapipe` instead.
//!
//! ```no_run
//! use mediapipe::{Confidence, FaceDetector, Image, ModelSource};
//!
//! let mut detector = FaceDetector::builder(ModelSource::path("blaze_face_short_range.tflite"))
//!     .min_detection_confidence(Confidence::new(0.5)?)
//!     .build()?;
//! let image = Image::from_file("portrait.jpg")?;
//! for face in detector.detect(&image)? {
//!     println!("{:?} {:?}", face.bounding_box, face.score());
//! }
//! # Ok::<(), mediapipe::Error>(())
//! ```
//!
//! [c-api]: https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/tasks/c

mod error;
mod face_detector;
mod face_landmarker;
mod image;
mod pose_landmarker;
mod stream;
mod types;

pub mod loader;
pub mod sys;

pub use error::{Error, Result, StatusCode};
pub use face_detector::{Detection, FaceDetector, FaceDetectorBuilder, FaceDetectorStream};
pub use face_landmarker::{
    FaceLandmarker, FaceLandmarkerBuilder, FaceLandmarkerResult, FaceLandmarkerStream,
};
pub use image::{Image, ImageRef};
pub use loader::{LibrarySource, MEDIAPIPE_VERSION};
pub use pose_landmarker::{
    PoseLandmarker, PoseLandmarkerBuilder, PoseLandmarkerResult, PoseLandmarkerVideo,
};
pub use stream::{AsyncTask, Stream};
pub use types::{
    Category, Confidence, Delegate, IouThreshold, Keypoint, ModelSource, NormalizedLandmark,
    NormalizedPoint2, NormalizedPoint3, PixelPoint, PixelRect, Rotation, Size, Timestamp,
    Transform4x4, WorldLandmark, WorldPoint3,
};
