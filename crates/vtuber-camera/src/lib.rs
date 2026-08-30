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
//! `vtuber-camera`: OS camera backends and capture worker.
//!
//! Native camera objects are constructed, opened, used, stopped, and dropped
//! inside the capture worker. Backend buffers and OS handles are never exposed.

#![deny(unsafe_code)]
#![warn(missing_docs)]

/// Production capture service.
pub mod capture;
/// Camera device and format domain types.
pub mod device;
/// Format negotiation logic.
pub mod format;
/// Mock backend for tests.
pub mod mock;
/// Placeholder for camera subsystem.
pub mod placeholder;

/// Platform-specific camera backends.
pub mod backend;

pub use capture::{CaptureController, CaptureMetrics, CaptureServiceState};
pub use device::{CameraDescriptor, CameraError, CameraFormat, CameraRequest};
pub use format::select_format;
