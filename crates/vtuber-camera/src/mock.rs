//! Explicit black-frame input for development and deterministic test fixtures.

use std::sync::Arc;
use std::time::{Duration, Instant};

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
        let format = self
            .formats
            .first()
            .copied()
            .ok_or(CameraError::NoSuitableFormat)?;
        let frame_interval = if format.fps_numerator == 0 || format.fps_denominator == 0 {
            return Err(CameraError::NoSuitableFormat);
        } else {
            Duration::from_secs_f64(
                f64::from(format.fps_denominator) / f64::from(format.fps_numerator),
            )
        };
        let stride = format.width as usize * channels(format.format);
        let len = stride
            .checked_mul(format.height as usize)
            .ok_or(CameraError::NoSuitableFormat)?;
        Ok(Box::new(MockStream {
            format,
            next_seq: 0,
            data: vec![0; len].into(),
            next_frame_at: Instant::now(),
            frame_interval,
            disconnect_after: self.disconnect_after,
        }))
    }
}

const fn channels(format: PixelFormat) -> usize {
    match format {
        PixelFormat::Rgb8 | PixelFormat::Bgr8 => 3,
        PixelFormat::Rgba8 => 4,
        PixelFormat::Gray8 => 1,
    }
}

/// Builds a deterministic fixture from caller-supplied time and shared pixels.
/// No clock, allocation or sleep is performed here.
#[must_use]
pub fn make_mock_frame(
    format: CameraFormat,
    seq: FrameSeq,
    captured_at: MonoTimeNs,
    data: Arc<[u8]>,
) -> VideoFrame {
    VideoFrame {
        seq,
        captured_at,
        width: format.width,
        height: format.height,
        stride_bytes: format.width as usize * channels(format.format),
        format: format.format,
        data,
    }
}

struct MockStream {
    format: CameraFormat,
    next_seq: u64,
    data: Arc<[u8]>,
    next_frame_at: Instant,
    frame_interval: Duration,
    disconnect_after: Option<u64>,
}

impl CameraStream for MockStream {
    fn actual_format(&self) -> CameraFormat {
        self.format
    }

    fn next_frame(&mut self, stop: &StopToken) -> Result<VideoFrame, CameraError> {
        while !stop.is_stopped() {
            let remaining = self.next_frame_at.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            std::thread::sleep(remaining.min(Duration::from_millis(5)));
        }
        if stop.is_stopped() || self.disconnect_after == Some(self.next_seq) {
            return Err(CameraError::Disconnected);
        }
        let frame = make_mock_frame(
            self.format,
            FrameSeq(self.next_seq),
            vtuber_core::monotonic_now(),
            Arc::clone(&self.data),
        );
        self.next_seq += 1;
        self.next_frame_at = Instant::now() + self.frame_interval;
        Ok(frame)
    }

    fn stop(&mut self) -> Result<(), CameraError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_preserves_explicit_time_sequence_and_shared_pixels() {
        let format = CameraFormat {
            width: 2,
            height: 1,
            fps_numerator: 30,
            fps_denominator: 1,
            format: PixelFormat::Rgb8,
        };
        let pixels: Arc<[u8]> = Arc::from([0; 6]);
        let frame = make_mock_frame(format, FrameSeq(42), MonoTimeNs(123), Arc::clone(&pixels));
        assert_eq!(frame.seq, FrameSeq(42));
        assert_eq!(frame.captured_at, MonoTimeNs(123));
        assert_eq!(frame.stride_bytes, 6);
        assert!(Arc::ptr_eq(&frame.data, &pixels));
    }

    #[test]
    fn live_mock_uses_the_process_clock_and_reuses_its_black_buffer() {
        let backend = MockBackend::default();
        let mut stream = backend
            .open(&backend.descriptors[0], &CameraRequest::default())
            .unwrap();
        let stop = StopToken::new();
        let before = vtuber_core::monotonic_now();
        let first = stream.next_frame(&stop).unwrap();
        let second = stream.next_frame(&stop).unwrap();
        assert!(before <= first.captured_at);
        assert!(first.captured_at < second.captured_at);
        assert!(second.captured_at <= vtuber_core::monotonic_now());
        assert_eq!(second.seq.0, first.seq.0 + 1);
        assert!(Arc::ptr_eq(&first.data, &second.data));
        assert!(second.data.iter().all(|byte| *byte == 0));
        stop.stop();
        assert!(matches!(
            stream.next_frame(&stop),
            Err(CameraError::Disconnected)
        ));
    }

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
