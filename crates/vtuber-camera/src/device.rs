//! Camera device and format domain types.

use std::fmt;
use thiserror::Error;
use vtuber_core::{PixelFormat, StopToken, VideoFrame};

/// Errors that can occur when enumerating or opening a camera.
#[derive(Debug, Error)]
pub enum CameraError {
    /// Device enumeration failed.
    #[error("CAMERA_ENUM_FAILED: {0}")]
    EnumFailed(String),
    /// Permission was denied.
    #[error("CAMERA_PERMISSION_DENIED")]
    PermissionDenied,
    /// Opening the device failed.
    #[error("CAMERA_OPEN_FAILED: {0}")]
    OpenFailed(String),
    /// The camera disconnected.
    #[error("CAMERA_DISCONNECTED")]
    Disconnected,
    /// Frame decode failed.
    #[error("CAMERA_FRAME_DECODE_FAILED: {0}")]
    FrameDecodeFailed(String),
    /// No suitable format was found.
    #[error("CAMERA_OPEN_FAILED: no suitable format")]
    NoSuitableFormat,
}

/// Describes a camera device.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CameraDescriptor {
    /// Stable identifier used for selection.
    ///
    /// The index alone must not be used as the persistent key.
    pub id: String,
    /// Human-readable label.
    pub label: String,
}

impl fmt::Display for CameraDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.label, self.id)
    }
}

/// Requested camera configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CameraRequest {
    /// Target width in pixels.
    pub width: u32,
    /// Target height in pixels.
    pub height: u32,
    /// Target frame rate numerator (frames per `fps_denominator` units).
    pub fps_numerator: u32,
    /// Target frame rate denominator.
    pub fps_denominator: u32,
    /// Preferred pixel format.
    pub format: RequestedFormat,
}

impl Default for CameraRequest {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            fps_numerator: 30,
            fps_denominator: 1,
            format: RequestedFormat::Any,
        }
    }
}

/// Preferred pixel format requested by the caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RequestedFormat {
    /// No preference.
    Any,
    /// Prefer MJPEG.
    Mjpeg,
    /// Prefer YUYV or equivalent uncompressed.
    Yuyv,
}

/// Actual camera format chosen by negotiation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CameraFormat {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Frame rate numerator.
    pub fps_numerator: u32,
    /// Frame rate denominator.
    pub fps_denominator: u32,
    /// Pixel format used by the backend.
    pub format: PixelFormat,
}

impl fmt::Display for CameraFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}x{} @ {}/{} {:?}",
            self.width, self.height, self.fps_numerator, self.fps_denominator, self.format
        )
    }
}

/// Trait for camera backends.
pub trait CameraBackend {
    /// Enumerates available camera devices.
    fn enumerate(&self) -> Result<Vec<CameraDescriptor>, CameraError>;
    /// Opens the selected camera device and returns a stream.
    ///
    /// The `descriptor` identifies which physical device to open. The
    /// `request` specifies the desired capture format.
    fn open(
        &self,
        descriptor: &CameraDescriptor,
        request: &CameraRequest,
    ) -> Result<Box<dyn CameraStream>, CameraError>;
}

/// Trait for an opened camera stream.
pub trait CameraStream {
    /// Returns the actual negotiated format.
    fn actual_format(&self) -> CameraFormat;
    /// Captures the next frame, respecting the stop token.
    fn next_frame(&mut self, stop: &StopToken) -> Result<VideoFrame, CameraError>;
    /// Stops the stream.
    fn stop(&mut self) -> Result<(), CameraError>;
}
