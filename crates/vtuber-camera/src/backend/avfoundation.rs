//! macOS AVFoundation capture through nokhwa. Camera objects stay on the worker.

use std::sync::{Arc, OnceLock};

use nokhwa::Camera;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{
    ApiBackend, CameraFormat as NokhwaFormat, CameraIndex, FrameFormat, RequestedFormat,
    RequestedFormatType, Resolution,
};
use vtuber_core::{FrameSeq, PixelFormat, StopToken, VideoFrame};

use crate::device::{
    CameraBackend, CameraDescriptor, CameraError, CameraFormat, CameraRequest, CameraStream,
};
use crate::format::{FormatCandidate, select_format};

/// Asynchronous camera authorization; the UI keeps running while macOS prompts.
#[derive(Debug, Default)]
pub enum CameraPermission {
    /// Authorization has not been requested by this runtime.
    #[default]
    NotChecked,
    /// Waiting for nokhwa's authorization callback.
    Requesting(Arc<OnceLock<bool>>),
    /// macOS authorized camera access.
    Granted,
    /// macOS denied or restricted camera access.
    DeniedOrRestricted,
}

impl CameraPermission {
    /// Requests access once and polls its result without blocking the UI thread.
    pub fn poll(&mut self) -> Option<Result<(), CameraError>> {
        match self {
            Self::NotChecked => {
                let result = Arc::new(OnceLock::new());
                let callback_result = Arc::clone(&result);
                nokhwa::nokhwa_initialize(move |granted| {
                    let _ = callback_result.set(granted);
                });
                *self = Self::Requesting(result);
                None
            }
            Self::Requesting(result) => {
                let granted = *result.get()?;
                *self = if granted {
                    Self::Granted
                } else {
                    Self::DeniedOrRestricted
                };
                Some(if granted {
                    Ok(())
                } else {
                    Err(CameraError::PermissionDenied)
                })
            }
            Self::Granted => Some(Ok(())),
            Self::DeniedOrRestricted => {
                // A user may enable access in System Settings and press Refresh.
                if nokhwa::nokhwa_check() {
                    *self = Self::Granted;
                    Some(Ok(()))
                } else {
                    Some(Err(CameraError::PermissionDenied))
                }
            }
        }
    }
}

/// AVFoundation backend, selected by default on macOS.
pub struct AvFoundationBackend;

impl CameraBackend for AvFoundationBackend {
    fn power_state(&self) -> crate::device::CameraPowerState {
        let power = vtuber_platform::power_state();
        crate::device::CameraPowerState {
            sleeping: power.sleeping,
            generation: power.generation,
        }
    }
    fn enumerate(&self) -> Result<Vec<CameraDescriptor>, CameraError> {
        nokhwa::query(ApiBackend::AVFoundation)
            .map_err(|error| CameraError::EnumFailed(error.to_string()))?
            .into_iter()
            .map(|info| {
                let id = info.misc(); // AVCaptureDevice.uniqueID, not an enumeration index.
                if id.is_empty() {
                    return Err(CameraError::EnumFailed(format!(
                        "AVFoundation device `{}` has no unique ID",
                        info.human_name()
                    )));
                }
                Ok(CameraDescriptor {
                    id: format!("avfoundation:{id}"),
                    label: info.human_name(),
                })
            })
            .collect()
    }

    fn open(
        &self,
        descriptor: &CameraDescriptor,
        request: &CameraRequest,
    ) -> Result<Box<dyn CameraStream>, CameraError> {
        if !nokhwa::nokhwa_check() {
            return Err(CameraError::PermissionDenied);
        }
        let id = descriptor
            .id
            .strip_prefix("avfoundation:")
            .filter(|id| !id.is_empty())
            .ok_or_else(|| {
                CameraError::OpenFailed(format!(
                    "not an AVFoundation descriptor: {}",
                    descriptor.id
                ))
            })?;
        let index = CameraIndex::String(id.to_owned());
        // Read the native format list: nokhwa 0.10.11's resolution query has an
        // inverted fourcc filter, and Closest fails for unlisted requested sizes.
        let formats = nokhwa_bindings_macos::AVCaptureDevice::new(&index)
            .and_then(|device| device.supported_formats())
            .map_err(|error| CameraError::OpenFailed(error.to_string()))?;
        // nokhwa configures AVFoundation to deliver packed YUYV, including for
        // Apple cameras whose sensor formats are bi-planar. Decode that output.
        let candidates: Vec<_> = formats
            .into_iter()
            .filter(|format| format.format() == FrameFormat::YUYV)
            .map(|format| FormatCandidate {
                width: format.width(),
                height: format.height(),
                fps_numerator: format.frame_rate(),
                fps_denominator: 1,
                format: PixelFormat::Rgb8,
            })
            .collect();
        let chosen = select_format(request, &candidates)?;
        let requested = NokhwaFormat::new(
            Resolution::new(chosen.width, chosen.height),
            FrameFormat::YUYV,
            chosen.fps_numerator,
        );
        let mut camera = Camera::with_backend(
            index,
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::Exact(requested)),
            ApiBackend::AVFoundation,
        )
        .map_err(|error| CameraError::OpenFailed(error.to_string()))?;
        camera
            .open_stream()
            .map_err(|error| CameraError::OpenFailed(error.to_string()))?;
        let actual = camera.camera_format();
        Ok(Box::new(AvFoundationStream {
            camera,
            format: CameraFormat {
                width: actual.width(),
                height: actual.height(),
                fps_numerator: actual.frame_rate(),
                fps_denominator: 1,
                format: PixelFormat::Rgb8,
            },
            seq: 0,
        }))
    }
}

struct AvFoundationStream {
    camera: Camera,
    format: CameraFormat,
    seq: u64,
}

impl CameraStream for AvFoundationStream {
    fn actual_format(&self) -> CameraFormat {
        self.format
    }

    fn next_frame(&mut self, stop: &StopToken) -> Result<VideoFrame, CameraError> {
        if stop.is_stopped() {
            return Err(CameraError::Disconnected);
        }
        let buffer = self
            .camera
            .frame()
            .map_err(|error| CameraError::FrameReadFailed(error.to_string()))?;
        let captured_at = vtuber_core::monotonic_now();
        let rgb = buffer
            .decode_image::<RgbFormat>()
            .map_err(|error| CameraError::FrameDecodeFailed(error.to_string()))?;
        self.seq = self.seq.saturating_add(1);
        Ok(VideoFrame {
            seq: FrameSeq(self.seq),
            captured_at,
            width: rgb.width(),
            height: rgb.height(),
            stride_bytes: rgb.width() as usize * 3,
            format: PixelFormat::Rgb8,
            data: rgb.into_raw().into(),
        })
    }

    fn stop(&mut self) -> Result<(), CameraError> {
        self.camera
            .stop_stream()
            .map_err(|error| CameraError::StopFailed(error.to_string()))
    }
}
