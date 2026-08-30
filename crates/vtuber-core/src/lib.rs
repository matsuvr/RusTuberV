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
//! This crate must not depend on Bevy, `bevy_vrm1`, `nokhwa`, tract, or OS-specific APIs.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

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

pub use arkit::*;
pub use control::{CalibrationError, CalibrationSettings};
pub use face_tracking::*;
pub use observation::RawExpressionObservation;
pub use slot::{LatestSlot, ReadResult};
pub use stop::StopToken;
pub use time::now as monotonic_now;
pub use types::*;
pub use video_output::*;
pub use worker::{WorkerHandle, WorkerResult};
