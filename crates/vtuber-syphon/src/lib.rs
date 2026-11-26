//! Optional macOS Syphon sender. Native handles live and die on one worker.
//! The existing asynchronous BGRA readback is uploaded to a Metal texture;
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
    /// Device initialization, waiting for a frame or for a receiver.
    Starting,
    /// Frames are being submitted to the SDK.
    Live,
    /// Recoverable failure; another explicit start retries.
    Error(String),
}

/// Whether this platform includes the backend.
pub const fn available() -> bool {
    cfg!(target_os = "macos")
}

#[derive(Default)]
struct Mailbox {
    stop: bool,
    has_clients: bool,
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
            .unwrap_or_else(|_| Status::Error("Syphon worker state poisoned".into()))
    }

    /// Number of frames replaced before submission (not a delivery counter).
    pub fn replaced_frames(&self) -> u64 {
        self.shared.0.lock().map_or(0, |s| s.replaced)
    }

    /// Whether the running server has a receiver that needs readback.
    pub fn has_clients(&self) -> bool {
        self.shared.0.lock().is_ok_and(|s| s.has_clients)
    }

    /// Reap only a completed worker. Never wait for a native call on the UI thread.
    pub fn poll(&mut self) {
        if self.worker.as_ref().is_some_and(JoinHandle::is_finished)
            && let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            self.set_error("Syphon worker terminated unexpectedly");
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
            self.set_error("Syphon is available on macOS only");
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
            .name("syphon-output".into())
            .spawn(move || {
                let result = send_native(&shared);
                if let Ok(mut state) = shared.0.lock() {
                    state.has_clients = false;
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
            Err(e) => self.set_error(&format!("Cannot start Syphon worker: {e}")),
        }
    }

    fn set_error(&self, message: &str) {
        if let Ok(mut state) = self.shared.0.lock() {
            state.has_clients = false;
            state.frame = None;
            state.status = Status::Error(message.into());
        }
    }

    /// Request stop, discard queued data and wake the worker. No blocking join.
    pub fn stop(&mut self) {
        if let Ok(mut state) = self.shared.0.lock() {
            state.stop = true;
            state.has_clients = false;
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
            if !state.has_clients {
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

#[cfg(target_os = "macos")]
fn send_native(shared: &Arc<(Mutex<Mailbox>, Condvar)>) -> Result<(), String> {
    let mut sender = vtuber_syphon_native::Sender::new()?;
    send_loop(shared, |frame| match frame {
        Some(frame) => sender.send(frame.data(), frame.width(), frame.height()),
        None => Ok(sender.has_clients()),
    })
}

#[cfg(not(target_os = "macos"))]
fn send_native(_: &Arc<(Mutex<Mailbox>, Condvar)>) -> Result<(), String> {
    Err("Syphon is available on macOS only".into())
}

#[cfg(any(target_os = "macos", test))]
fn send_loop(
    shared: &Arc<(Mutex<Mailbox>, Condvar)>,
    mut poll_or_send: impl FnMut(Option<&VideoOutputFrame>) -> Result<bool, String>,
) -> Result<(), String> {
    const CLIENT_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(16);
    loop {
        let frame = {
            let state = shared
                .0
                .lock()
                .map_err(|_| "Syphon worker state poisoned")?;
            // Even with readback paused, keep the discoverable server and check
            // for a receiver independently of frame arrivals.
            let (mut state, _) = shared
                .1
                .wait_timeout_while(state, CLIENT_POLL_INTERVAL, |s| {
                    s.frame.is_none() && !s.stop
                })
                .map_err(|_| "Syphon worker state poisoned")?;
            if state.stop {
                return Ok(());
            }
            state.frame.take()
        };
        let has_clients = poll_or_send(frame.as_ref())?;
        let mut state = shared
            .0
            .lock()
            .map_err(|_| "Syphon worker state poisoned")?;
        if !state.stop {
            state.has_clients = has_clients;
            if !has_clients {
                state.frame = None;
                state.status = Status::Starting;
            } else if frame.is_some() {
                state.status = Status::Live;
            }
        }
    }
}

#[cfg(test)]
#[path = "../../../verification/replay.rs"]
mod quint_replay;

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
    #[test]
    #[ignore = "generated Quint ITF traces"]
    fn quint_syphon() {
        use crate::quint_replay::{number, traces};
        traces("syphon", |states| {
            let mut c = Controller::default();
            c.shared.0.lock().unwrap().status = Status::Starting;
            let shared = Arc::clone(&c.shared);
            let mut rows = states.iter();
            let mut previous = rows.next().unwrap();
            let check = |c: &Controller, row: &serde_json::Value| {
                assert_eq!(c.has_clients(), row["clients"]);
                assert_eq!(format!("{:?}", c.status()), row["status"]);
            };
            let producer = |c: &mut Controller, row: &serde_json::Value| match row["event"]
                .as_str()
                .unwrap()
            {
                "none" => {}
                "submit" => c.submit(frame(1, 1)),
                "replace" => {
                    c.submit(frame(1, 1));
                    c.submit(frame(2, 2));
                }
                "discard" => {
                    c.submit(frame(1, 1));
                    c.discard_pending();
                }
                "stop" => {
                    c.submit(frame(1, 1));
                    c.stop();
                }
                other => panic!("{other}"),
            };
            // The callback is the point where the native call has taken the old frame
            // and new submissions can arrive, before the worker learns the client count.
            send_loop(&shared, |offered| {
                check(&c, previous);
                let seq = offered.map_or(0, |f| f.frame_seq().0 as i64);
                assert_eq!(seq, number(&previous["pending"]));
                let Some(row) = rows.next() else {
                    c.stop();
                    return Ok(false);
                };
                assert_eq!(seq, number(&row["offered"]));
                producer(&mut c, row);
                previous = row;
                Ok(row["connected"].as_bool().unwrap())
            })
            .unwrap();
            if previous["status"] == "Off" {
                check(&c, previous);
                assert!(shared.0.lock().unwrap().frame.is_none());
                // A stopped worker cannot consume another frame, even if producers publish.
                for row in rows {
                    producer(&mut c, row);
                    check(&c, row);
                    assert_eq!(number(&row["offered"]), 0);
                    assert_eq!(number(&row["pending"]), 0);
                    assert!(shared.0.lock().unwrap().frame.is_none());
                }
            }
        });
    }

    fn active() -> Controller {
        let c = Controller::default();
        {
            let mut state = c.shared.0.lock().unwrap();
            state.status = Status::Starting;
            state.has_clients = true;
        }
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
    fn native_upload_borrows_the_original_readback_pixels() {
        let c = active();
        // Two 2-pixel rows with GPU padding and distinct colors at each corner.
        let pixels = [
            11, 32, 73, 128, 5, 7, 9, 0, 255, 255, 255, 255, 101, 93, 28, 255, 17, 51, 30, 64, 255,
            255, 255, 255,
        ]
        .to_vec();
        let allocation = pixels.as_ptr();
        let frame = VideoOutputFrame::from_rendered_bgra8(
            2,
            2,
            12,
            vtuber_core::FrameSeq(3),
            vtuber_core::MonoTimeNs(9),
            pixels,
        )
        .unwrap();
        c.submit(frame.clone());
        // The second consumer must neither change Spout's pixels nor detach its buffer.
        let ndi = frame.into_straight_alpha();
        assert_ne!(ndi.data().as_ptr(), allocation);
        assert_eq!(
            ndi.data(),
            &[
                22, 64, 145, 128, 0, 0, 0, 0, 101, 93, 28, 255, 68, 203, 120, 64
            ]
        );
        send_loop(&c.shared, |frame| {
            let frame = frame.unwrap();
            assert_eq!(frame.data().as_ptr(), allocation);
            assert_eq!(
                frame.data(),
                &[
                    11, 32, 73, 128, 5, 7, 9, 0, 101, 93, 28, 255, 17, 51, 30, 64
                ]
            );
            c.shared.0.lock().unwrap().stop = true;
            Ok(true)
        })
        .unwrap();
    }
    #[test]
    fn sender_handles_new_dimensions_without_reformatting_bgra() {
        let c = active();
        c.submit(frame(1, 2));
        let shared = Arc::clone(&c.shared);
        let mut widths = Vec::new();
        send_loop(&shared, |f| {
            let f = f.unwrap();
            widths.push(f.width());
            assert_eq!(f.data().len(), f.width() as usize * 4);
            if f.width() == 2 {
                c.submit(frame(2, 3));
            } else {
                shared.0.lock().unwrap().stop = true;
            }
            Ok(true)
        })
        .unwrap();
        assert_eq!(widths, [2, 3]);
    }
    #[test]
    fn skips_without_client_and_resumes_after_reconnection() {
        let c = active();
        c.shared.0.lock().unwrap().has_clients = false;
        let connected = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let shared = Arc::clone(&c.shared);
        let (published, received) = std::sync::mpsc::channel();
        let worker = std::thread::spawn({
            let connected = Arc::clone(&connected);
            move || {
                send_loop(&shared, |frame| {
                    let connected = connected.load(std::sync::atomic::Ordering::Relaxed);
                    if connected && let Some(frame) = frame {
                        published.send(frame.width()).unwrap();
                    }
                    Ok(connected)
                })
            }
        });
        let timeout = std::time::Duration::from_secs(2);
        let change_connection = |expected| {
            connected.store(expected, std::sync::atomic::Ordering::Relaxed);
            let deadline = std::time::Instant::now() + timeout;
            while c.has_clients() != expected {
                assert!(std::time::Instant::now() < deadline, "client state stalled");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        };

        for width in [2, 4] {
            c.submit(frame(1, width - 1));
            assert!(c.shared.0.lock().unwrap().frame.is_none());
            change_connection(true);
            assert_eq!(c.status(), Status::Starting);
            c.submit(frame(2, width));
            assert_eq!(received.recv_timeout(timeout).unwrap(), width);
            change_connection(false);
            assert_eq!(c.status(), Status::Starting);
        }
        drop(c);
        assert!(worker.join().unwrap().is_ok());
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
        let worker = std::thread::spawn(move || send_loop(&shared, |_| Ok(true)));
        c.stop();
        assert!(worker.join().unwrap().is_ok());
    }
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn unsupported_platform_is_explicit() {
        let mut c = Controller::default();
        c.start();
        assert!(matches!(c.status(), Status::Error(_)));
        assert!(c.worker.is_none());
    }
}
