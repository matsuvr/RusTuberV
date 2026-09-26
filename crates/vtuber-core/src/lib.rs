// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]
//! `vtuber-core`: platform- and engine-independent data and synchronization contracts.
//!
//! This crate does not depend on Bevy, VRM loaders, inference runtimes, or OS APIs.
//! Slots retain the latest value; each reader owns its own generation cursor.
//!
//! ```
//! use vtuber_core::{LatestSlot, ReadResult};
//! let slot = LatestSlot::new();
//! assert!(slot.publish(42));
//! let Some(ReadResult::New { generation, value }) = slot.try_read_after(0) else {
//!     return;
//! };
//! assert_eq!(value, 42);
//! assert!(slot.try_read_after(generation).is_none());
//! slot.close();
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// Engine-neutral pose-arm observations and targets.
pub mod arm_tracking;

/// Re-export core types used across worker boundaries.
pub mod types;

/// Control settings and commands for the tracking pipeline.
pub mod control;

/// Fixed-size metrics collection for acceptance testing.
pub mod metrics;

/// Engine-neutral ARKit 52 blendshape contract.
pub mod arkit;
/// Canonical MediaPipe face-tracking contract.
pub mod face_tracking;
/// Raw observation contract between inference and tracking.
pub mod observation;
/// Latest-value slot for single-producer / single-consumer communication.
pub mod slot;
/// Worker stop token.
pub mod stop;
/// Process-wide monotonic clock.
pub mod time;
/// Transport-neutral contract for transparent avatar video output.
pub mod video_output;
/// Deterministic worker supervision helpers.
pub mod worker;

pub use arkit::{
    ARKIT_NON_TONGUE_CHANNEL_COUNT, ARKIT_NON_TONGUE_LEFT_RIGHT_PAIRS, ARKIT52_CHANNEL_COUNT,
    Arkit52Coefficients, Arkit52NameError, Arkit52ValueError, ArkitBlendshape,
    arkit_non_tongue_values, arkit52_with_zero_tongue,
};
pub use control::{CalibrationError, CalibrationSettings};
pub use face_tracking::{
    CameraFaceTransform, FaceBlendshapeSet, FaceLandmark, FaceTrackingContractError,
    FaceTrackingOutcome, FaceTrackingQuality, FaceTrackingSample, MEDIAPIPE_FACE_BLENDSHAPE_COUNT,
    MEDIAPIPE_FACE_LANDMARK_COUNT, MediaPipeBlendshape,
};
pub use observation::RawExpressionObservation;
pub use slot::{LatestSlot, ReadResult, skipped_generations};
pub use stop::StopToken;
pub use time::now as monotonic_now;
pub use types::{
    AvatarControlFrame, ExpressionCoefficients, FrameSeq, GazeSignal, GazeTrackingState, HeadPose,
    HeadTranslationSignal, HeadTranslationState, InferenceOutput, Landmark3, LandmarkSchemaId,
    MonoTimeNs, NamedCoefficient, NormalizedRect, PixelFormat, RawFaceObservation, TrackingState,
    VideoFrame,
};
pub use video_output::{
    VideoOutputFrame, VideoOutputFrameError, VideoOutputPixelFormat, VideoOutputProfile,
    unpremultiply_bgra8_in_place,
};
pub use worker::{WorkerHandle, WorkerResult};
