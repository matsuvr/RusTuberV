//! `vtuber-camera`: OS camera backends and capture worker.
//!
//! Native camera objects are constructed, opened, used, stopped, and dropped
//! inside the capture worker. Backend buffers and OS handles are never exposed.
//! Windows uses the Media Foundation backend. Other desktop targets, including
//! macOS, currently select an explicit development mock, not a real camera.
//! A failed real-camera request never silently changes into a mock request.
//!
//! ```no_run
//! use vtuber_camera::{CaptureController, mock::MockBackend};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut capture = CaptureController::new();
//! capture.start_worker(MockBackend::default())?;
//! // Starting the worker does not yet select or open a device.
//! let _metrics = capture.shutdown()?;
//! # Ok(())
//! # }
//! ```

#![deny(unsafe_code)]
#![warn(missing_docs)]

/// Production capture service.
pub mod capture;
/// Camera device and format domain types.
pub mod device;
/// Format negotiation logic.
pub mod format;
/// Explicit Mock input for development and deterministic test fixtures.
pub mod mock;
/// Placeholder for camera subsystem.
pub mod placeholder;

/// Platform-specific camera backends.
pub mod backend;

pub use capture::{CaptureController, CaptureMetrics, CaptureServiceState};
pub use device::{CameraDescriptor, CameraError, CameraFormat, CameraRequest};
pub use format::select_format;
