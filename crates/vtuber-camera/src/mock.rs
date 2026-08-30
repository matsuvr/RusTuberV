//! Mock camera backend for unit tests.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::{
    CameraBackend, CameraDescriptor, CameraError, CameraFormat, CameraRequest, CameraStream,
};
use vtuber_core::{FrameSeq, MonoTimeNs, PixelFormat, StopToken, VideoFrame};

/// Mock backend with a configurable format and disconnect behavior.
pub struct MockBackend {
    /// Formats returned by enumeration.
    pub descriptors: Vec<CameraDescriptor>,
    /// Formats available for `open`.
    pub formats: Vec<CameraFormat>,
    /// Number of frames to produce before disconnecting.
    pub disconnect_after: Option<u64>,
    /// IDs opened by the backend, useful for selection-contract tests.
    pub opened_devices: Arc<std::sync::Mutex<Vec<String>>>,
}

impl Default for MockBackend {
    fn default() -> Self {
        Self {
            descriptors: vec![CameraDescriptor {
                id: "mock-0".into(),
                label: "Mock Camera".into(),
            }],
            formats: vec![CameraFormat {
                width: 1280,
                height: 720,
                fps_numerator: 30,
                fps_denominator: 1,
                format: PixelFormat::Rgb8,
            }],
            disconnect_after: None,
            opened_devices: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }
}

impl CameraBackend for MockBackend {
    fn enumerate(&self) -> Result<Vec<CameraDescriptor>, CameraError> {
        Ok(self.descriptors.clone())
    }

    fn open(
        &self,
        descriptor: &CameraDescriptor,
        _request: &CameraRequest,
    ) -> Result<Box<dyn CameraStream>, CameraError> {
        if !self.descriptors.iter().any(|d| d.id == descriptor.id) {
            return Err(CameraError::OpenFailed(format!(
                "unknown mock device {}",
                descriptor.id
            )));
        }
        if let Ok(mut opened) = self.opened_devices.lock() {
            opened.push(descriptor.id.clone());
        }
        Ok(Box::new(MockStream {
            format: self.formats.first().copied().unwrap_or(CameraFormat {
                width: 640,
                height: 480,
                fps_numerator: 30,
                fps_denominator: 1,
                format: PixelFormat::Rgb8,
            }),
            counter: Arc::new(AtomicU64::new(0)),
            disconnect_after: self.disconnect_after,
        }))
    }
}

struct MockStream {
    format: CameraFormat,
    counter: Arc<AtomicU64>,
    disconnect_after: Option<u64>,
}

impl CameraStream for MockStream {
    fn actual_format(&self) -> CameraFormat {
        self.format
    }

    fn next_frame(&mut self, _stop: &StopToken) -> Result<VideoFrame, CameraError> {
        let count = self.counter.fetch_add(1, Ordering::SeqCst);
        if self.disconnect_after == Some(count) {
            return Err(CameraError::Disconnected);
        }
        Ok(VideoFrame {
            seq: FrameSeq(count),
            captured_at: MonoTimeNs(0),
            width: self.format.width,
            height: self.format.height,
            stride_bytes: (self.format.width * 3) as usize,
            format: PixelFormat::Rgb8,
            data: vec![0u8; (self.format.width * self.format.height * 3) as usize].into(),
        })
    }

    fn stop(&mut self) -> Result<(), CameraError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_enumerates_devices() {
        let backend = MockBackend::default();
        let devices = backend.enumerate().unwrap();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].id, "mock-0");
    }

    #[test]
    fn mock_produces_frames() {
        let backend = MockBackend::default();
        let descriptor = CameraDescriptor {
            id: "mock-0".into(),
            label: "Mock".into(),
        };
        let request = CameraRequest::default();
        let mut stream = backend.open(&descriptor, &request).unwrap();
        let stop = StopToken::new();
        let frame = stream.next_frame(&stop).unwrap();
        assert_eq!(frame.width, 1280);
        assert_eq!(frame.height, 720);
    }

    #[test]
    fn mock_disconnects_after_n_frames() {
        let backend = MockBackend {
            disconnect_after: Some(2),
            ..Default::default()
        };
        let descriptor = CameraDescriptor {
            id: "mock-0".into(),
            label: "Mock".into(),
        };
        let request = CameraRequest::default();
        let mut stream = backend.open(&descriptor, &request).unwrap();
        let stop = StopToken::new();
        assert!(stream.next_frame(&stop).is_ok());
        assert!(stream.next_frame(&stop).is_ok());
        let err = stream.next_frame(&stop).unwrap_err();
        assert!(matches!(err, CameraError::Disconnected));
    }

    #[test]
    fn mock_opens_the_selected_device() {
        let backend = MockBackend {
            descriptors: vec![
                CameraDescriptor {
                    id: "mock-0".into(),
                    label: "First".into(),
                },
                CameraDescriptor {
                    id: "mock-1".into(),
                    label: "Second".into(),
                },
            ],
            ..Default::default()
        };
        let descriptor = backend.descriptors[1].clone();
        let _stream = backend
            .open(&descriptor, &CameraRequest::default())
            .expect("selected mock device should open");
        let opened = backend
            .opened_devices
            .lock()
            .expect("test mutex is healthy");
        assert_eq!(opened.as_slice(), ["mock-1"]);
    }
}
