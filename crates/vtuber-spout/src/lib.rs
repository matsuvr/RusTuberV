//! Optional Windows Spout2 sender. Native handles live and die on one worker.
//! The existing asynchronous BGRA readback is uploaded to a D3D11 shared texture;
//! this is not a zero-copy renderer integration. The mailbox holds one frame.
#![forbid(unsafe_code)]

use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use vtuber_core::VideoOutputFrame;

/// Sender state. `Live` means an SDK submission succeeded, not receiver delivery.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Status {
    /// No sender requested.
    #[default]
    Off,
    /// Device initialization or waiting for the first frame.
    Starting,
    /// Frames are being submitted to the SDK.
    Live,
    /// Recoverable failure; another explicit start retries.
    Error(String),
}

/// Whether this platform includes the backend.
pub const fn available() -> bool {
    cfg!(windows)
}

#[derive(Default)]
struct Mailbox {
    stop: bool,
    frame: Option<VideoOutputFrame>,
    status: Status,
    replaced: u64,
}

/// Nonblocking application-facing controller. No unbounded frame queue.
#[derive(Default)]
pub struct Controller {
    shared: Arc<(Mutex<Mailbox>, Condvar)>,
    worker: Option<JoinHandle<()>>,
}

impl Controller {
    /// Status snapshot; poisoned native-worker state is reported rather than panicking.
    pub fn status(&self) -> Status {
        self.shared
            .0
            .lock()
            .map(|s| s.status.clone())
            .unwrap_or_else(|_| Status::Error("Spout worker state poisoned".into()))
    }

    /// Number of frames replaced before submission (not a delivery counter).
    pub fn replaced_frames(&self) -> u64 {
        self.shared.0.lock().map_or(0, |s| s.replaced)
    }

    /// Reap only a completed worker. Never wait for a native call on the UI thread.
    pub fn poll(&mut self) {
        if self.worker.as_ref().is_some_and(JoinHandle::is_finished)
            && let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            self.set_error("Spout worker terminated unexpectedly");
        }
    }

    /// True while a previous sender is still being released.
    pub fn is_stopping(&self) -> bool {
        self.worker.is_some() && self.shared.0.lock().map_or(true, |s| s.stop)
    }

    /// Start once. A pending teardown must finish before a replacement starts.
    pub fn start(&mut self) {
        self.poll();
        if self.worker.is_some() {
            return;
        }
        if !available() {
            self.set_error("Spout2 is available on Windows only");
            return;
        }
        if let Ok(mut state) = self.shared.0.lock() {
            *state = Mailbox {
                status: Status::Starting,
                ..Default::default()
            };
        } else {
            return;
        }
        let shared = Arc::clone(&self.shared);
        match std::thread::Builder::new()
            .name("spout-output".into())
            .spawn(move || {
                let result = send_native(&shared);
                if let Ok(mut state) = shared.0.lock() {
                    state.frame = None;
                    state.status = if state.stop {
                        Status::Off
                    } else {
                        match result {
                            Ok(()) => Status::Off,
                            Err(e) => Status::Error(e),
                        }
                    };
                }
            }) {
            Ok(worker) => self.worker = Some(worker),
            Err(e) => self.set_error(&format!("Cannot start Spout worker: {e}")),
        }
    }

    fn set_error(&self, message: &str) {
        if let Ok(mut state) = self.shared.0.lock() {
            state.status = Status::Error(message.into());
        }
    }

    /// Request stop, discard queued data and wake the worker. No blocking join.
    pub fn stop(&mut self) {
        if let Ok(mut state) = self.shared.0.lock() {
            state.stop = true;
            state.frame = None;
            if !matches!(state.status, Status::Error(_)) {
                state.status = Status::Off;
            }
        }
        self.shared.1.notify_one();
    }

    /// Clear old-generation data while a model is being replaced.
    pub fn discard_pending(&self) {
        if let Ok(mut state) = self.shared.0.lock() {
            state.frame = None;
        }
    }

    /// Publish the latest validated BGRA frame, sharing its Arc allocation with NDI.
    pub fn submit(&self, frame: VideoOutputFrame) {
        if let Ok(mut state) = self.shared.0.lock() {
            if state.stop || !matches!(state.status, Status::Starting | Status::Live) {
                return;
            }
            if state.frame.replace(frame).is_some() {
                state.replaced = state.replaced.saturating_add(1);
            }
        }
        self.shared.1.notify_one();
    }
}

impl Drop for Controller {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(windows)]
fn send_native(shared: &Arc<(Mutex<Mailbox>, Condvar)>) -> Result<(), String> {
    let mut sender = spout2::dx::Sender::new("RusTuberV").map_err(|e| e.to_string())?;
    sender.set_format(spout2::dx::format::B8G8R8A8_UNORM);
    send_loop(shared, |frame| {
        // The SDK recreates its shared texture if dimensions change. No retained
        // pointer to Bevy resources crosses this thread or the FFI boundary.
        sender
            .send_image(frame.data(), frame.width(), frame.height())
            .map_err(|e| e.to_string())
    })
}

#[cfg(not(windows))]
fn send_native(_: &Arc<(Mutex<Mailbox>, Condvar)>) -> Result<(), String> {
    Err("Spout2 is available on Windows only".into())
}

#[cfg(any(windows, test))]
fn send_loop(
    shared: &Arc<(Mutex<Mailbox>, Condvar)>,
    mut send: impl FnMut(&VideoOutputFrame) -> Result<(), String>,
) -> Result<(), String> {
    loop {
        let frame = {
            let state = shared.0.lock().map_err(|_| "Spout worker state poisoned")?;
            let mut state = shared
                .1
                .wait_while(state, |s| !s.stop && s.frame.is_none())
                .map_err(|_| "Spout worker state poisoned")?;
            if state.stop {
                return Ok(());
            }
            state.frame.take()
        };
        if let Some(frame) = frame {
            send(&frame)?;
            let mut state = shared.0.lock().map_err(|_| "Spout worker state poisoned")?;
            if !state.stop {
                state.status = Status::Live;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;
    fn frame(seq: u64, width: u32) -> VideoOutputFrame {
        VideoOutputFrame::new_bgra8(
            width,
            1,
            vtuber_core::FrameSeq(seq),
            vtuber_core::MonoTimeNs(0),
            vec![0; width as usize * 4],
        )
        .unwrap()
    }
    fn active() -> Controller {
        let c = Controller::default();
        c.shared.0.lock().unwrap().status = Status::Starting;
        c
    }
    #[test]
    fn mailbox_replaces_old_frames_and_stop_discards() {
        let mut c = active();
        c.submit(frame(1, 2));
        c.submit(frame(2, 3));
        assert_eq!(c.replaced_frames(), 1);
        assert_eq!(
            c.shared.0.lock().unwrap().frame.as_ref().unwrap().width(),
            3
        );
        c.stop();
        c.submit(frame(3, 2));
        assert!(c.shared.0.lock().unwrap().frame.is_none());
        assert_eq!(c.status(), Status::Off);
    }
    #[test]
    fn sender_handles_new_dimensions_without_reformatting_bgra() {
        let c = active();
        c.submit(frame(1, 2));
        let shared = Arc::clone(&c.shared);
        let mut widths = Vec::new();
        send_loop(&shared, |f| {
            widths.push(f.width());
            assert_eq!(f.data().len(), f.width() as usize * 4);
            if f.width() == 2 {
                c.submit(frame(2, 3));
            } else {
                shared.0.lock().unwrap().stop = true;
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(widths, [2, 3]);
    }
    #[test]
    fn failure_is_returned_and_no_queued_frame_is_retried() {
        let c = active();
        c.submit(frame(1, 2));
        assert_eq!(
            send_loop(&c.shared, |_| Err("device lost".into())),
            Err("device lost".into())
        );
        assert!(c.shared.0.lock().unwrap().frame.is_none());
    }
    #[test]
    fn stop_wakes_idle_worker() {
        let mut c = active();
        let shared = Arc::clone(&c.shared);
        let worker = std::thread::spawn(move || send_loop(&shared, |_| Ok(())));
        c.stop();
        assert!(worker.join().unwrap().is_ok());
    }
    #[cfg(not(windows))]
    #[test]
    fn unsupported_platform_is_explicit() {
        let mut c = Controller::default();
        c.start();
        assert!(matches!(c.status(), Status::Error(_)));
        assert!(c.worker.is_none());
    }
}
