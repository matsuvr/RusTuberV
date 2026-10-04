//! Paced RGB replay through the same capture worker used by physical cameras.
//!
//! Input is packed RGB8, 640x360 at 30 Hz, prepared offline from a video.
//! The stream loops until stopped. Late reads skip to the current video time;
//! they never speed up the video to catch up with a delayed consumer.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use crate::device::{
    CameraBackend, CameraDescriptor, CameraError, CameraFormat, CameraRequest, CameraStream,
};
use vtuber_core::{FrameSeq, MonoTimeNs, PixelFormat, StopToken, VideoFrame, monotonic_now};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;
const FPS: u64 = 30;
const FRAME_BYTES: usize = WIDTH as usize * HEIGHT as usize * 3;

/// A validated RGB recording and the clock origin of its capture session.
#[derive(Clone, Debug)]
pub struct ReplaySource {
    path: PathBuf,
    frames: u64,
    started_at: Arc<OnceLock<MonoTimeNs>>,
}

impl ReplaySource {
    /// Opens a prepared recording and checks its complete frame layout.
    pub fn load(path: &Path) -> Result<Self, CameraError> {
        let bytes = File::open(path)
            .and_then(|f| f.metadata())
            .map_err(|e| CameraError::OpenFailed(e.to_string()))?
            .len();
        if bytes == 0 || bytes % FRAME_BYTES as u64 != 0 {
            return Err(CameraError::OpenFailed(
                "replay must contain complete 640x360 RGB8 frames".into(),
            ));
        }
        Ok(Self {
            path: path.to_owned(),
            frames: bytes / FRAME_BYTES as u64,
            started_at: Arc::new(OnceLock::new()),
        })
    }

    /// Duration of one pass at the recorded 30 Hz cadence.
    pub fn duration(&self) -> Duration {
        Duration::from_secs_f64(self.frames as f64 / FPS as f64)
    }

    /// Capture clock origin, populated when the worker opens the recording.
    pub fn started_at(&self) -> Option<MonoTimeNs> {
        self.started_at.get().copied()
    }
}

impl CameraBackend for ReplaySource {
    fn enumerate(&self) -> Result<Vec<CameraDescriptor>, CameraError> {
        Ok(vec![CameraDescriptor {
            id: "replay-rgb".into(),
            label: "Recorded video (640x360, 30 Hz)".into(),
        }])
    }

    fn open(
        &self,
        descriptor: &CameraDescriptor,
        _: &CameraRequest,
    ) -> Result<Box<dyn CameraStream>, CameraError> {
        if descriptor.id != "replay-rgb" {
            return Err(CameraError::OpenFailed("unknown replay input".into()));
        }
        let file = File::open(&self.path).map_err(|e| CameraError::OpenFailed(e.to_string()))?;
        let start = Instant::now();
        let epoch = monotonic_now();
        // A source represents one capture session. Reopening it gets a new
        // backend instance in the next measurement process.
        self.started_at
            .set(epoch)
            .map_err(|_| CameraError::OpenFailed("replay session is already open".into()))?;
        Ok(Box::new(ReplayStream {
            file,
            frames: self.frames,
            next: 0,
            start,
            epoch,
        }))
    }
}

struct ReplayStream {
    file: File,
    frames: u64,
    next: u64,
    start: Instant,
    epoch: MonoTimeNs,
}

fn scheduled_index(next: u64, elapsed: Duration) -> u64 {
    next.max((elapsed.as_secs_f64() * FPS as f64).floor() as u64)
}

impl CameraStream for ReplayStream {
    fn actual_format(&self) -> CameraFormat {
        CameraFormat {
            width: WIDTH,
            height: HEIGHT,
            fps_numerator: FPS as u32,
            fps_denominator: 1,
            format: PixelFormat::Rgb8,
        }
    }

    fn next_frame(&mut self, stop: &StopToken) -> Result<VideoFrame, CameraError> {
        let due = Duration::from_secs_f64(self.next as f64 / FPS as f64);
        while !stop.is_stopped() {
            let wait = due.saturating_sub(self.start.elapsed());
            if wait.is_zero() {
                break;
            }
            std::thread::sleep(wait.min(Duration::from_millis(5)));
        }
        if stop.is_stopped() {
            return Err(CameraError::Disconnected);
        }
        let index = scheduled_index(self.next, self.start.elapsed());
        let offset = (index % self.frames)
            .checked_mul(FRAME_BYTES as u64)
            .ok_or(CameraError::NoSuitableFormat)?;
        let mut data = vec![0; FRAME_BYTES];
        self.file
            .seek(SeekFrom::Start(offset))
            .and_then(|_| self.file.read_exact(&mut data))
            .map_err(|e| CameraError::FrameReadFailed(e.to_string()))?;
        self.next = index.saturating_add(1);
        let elapsed_ns = index.saturating_mul(1_000_000_000) / FPS;
        Ok(VideoFrame {
            seq: FrameSeq(index),
            captured_at: MonoTimeNs(self.epoch.0.saturating_add(elapsed_ns)),
            width: WIDTH,
            height: HEIGHT,
            stride_bytes: WIDTH as usize * 3,
            format: PixelFormat::Rgb8,
            data: data.into(),
        })
    }

    fn stop(&mut self) -> Result<(), CameraError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]
    use super::*;
    use std::io::Write;

    #[test]
    fn delayed_reader_skips_video_time_without_catch_up_bursts() {
        assert_eq!(scheduled_index(1, Duration::from_millis(5)), 1);
        assert_eq!(scheduled_index(1, Duration::from_millis(500)), 15);
        assert_eq!(scheduled_index(16, Duration::from_millis(501)), 16);
    }

    #[test]
    fn recorded_pixels_keep_video_timestamps_and_stop_cooperatively() {
        let path = std::env::temp_dir().join(format!(
            "rustuberv-replay-{}-{}.rgb",
            std::process::id(),
            monotonic_now().0
        ));
        let mut file = File::create_new(&path).unwrap();
        file.write_all(&vec![0; FRAME_BYTES]).unwrap();
        file.write_all(&vec![1; FRAME_BYTES]).unwrap();
        drop(file);
        let source = ReplaySource::load(&path).unwrap();
        let descriptor = source.enumerate().unwrap().remove(0);
        let mut stream = source.open(&descriptor, &CameraRequest::default()).unwrap();
        let stop = StopToken::new();
        for _ in 0..2 {
            let frame = stream.next_frame(&stop).unwrap();
            assert_eq!(frame.data.len(), FRAME_BYTES);
            assert_eq!(frame.data[0], (frame.seq.0 % 2) as u8);
            assert_eq!(
                frame.captured_at.0 - source.started_at().unwrap().0,
                frame.seq.0 * 1_000_000_000 / FPS
            );
        }
        stop.stop();
        assert!(matches!(
            stream.next_frame(&stop),
            Err(CameraError::Disconnected)
        ));
        drop(stream);
        std::fs::write(&path, [0]).unwrap();
        assert!(ReplaySource::load(&path).is_err());
        std::fs::remove_file(path).unwrap();
    }
}
