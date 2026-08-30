//! Production capture service.
//!
//! [`CaptureController`] owns the lifecycle of the capture worker: selecting a
//! device, starting and stopping capture, and requesting clean shutdown. The
//! native camera object is constructed, opened, used, stopped, and dropped
//! inside the worker thread.

use std::sync::Arc;
use std::time::Duration;

use vtuber_core::{FrameSeq, LatestSlot, StopToken, VideoFrame, WorkerHandle};

use crate::device::{CameraBackend, CameraDescriptor, CameraError, CameraFormat, CameraRequest};

/// Maximum number of consecutive reconnect attempts before giving up.
const MAX_RECONNECT_ATTEMPTS: u32 = 5;

/// Initial delay before the first reconnect attempt.
const RECONNECT_DELAY_BASE: Duration = Duration::from_millis(100);

/// Maximum delay between reconnect attempts.
const RECONNECT_DELAY_MAX: Duration = Duration::from_secs(5);

/// Current state of the capture service as observed by the controller.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CaptureServiceState {
    /// No device selected.
    #[default]
    Idle,
    /// A device is selected but capture is not running.
    Selected,
    /// Opening the camera and starting the worker.
    Starting,
    /// Actively capturing frames.
    Running,
    /// Device was lost; waiting to reconnect.
    Reconnecting,
    /// A recoverable error occurred too many times.
    BackOff,
    /// Capture is stopping.
    Stopping,
}

/// Snapshot of capture metrics exposed to callers.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CaptureMetrics {
    /// Number of frames produced by the backend.
    pub frames_captured: u64,
    /// Number of frames dropped because the consumer had not read the previous
    /// value (the capacity-one slot was overwritten).
    pub frames_dropped: u64,
    /// Number of reconnect attempts made.
    pub reconnect_attempts: u32,
    /// Negotiated format, if known.
    pub format: Option<CameraFormat>,
    /// Last observed error code, if any.
    pub last_error: Option<String>,
}

/// Control commands sent from [`CaptureController`] to the capture worker.
#[derive(Clone, Debug, PartialEq)]
enum ControlCommand {
    /// Start capture with the given device and request.
    Start(CameraDescriptor, CameraRequest),
    /// Stop capture but keep the selected device.
    Stop,
    /// Stop capture and clear the selected device.
    Reset,
}

/// Shared mutable state guarded by a mutex so the controller and any UI can
/// read it without message passing.
#[derive(Debug, Default)]
struct SharedState {
    state: CaptureServiceState,
    selected_device: Option<CameraDescriptor>,
    requested_format: Option<CameraRequest>,
    metrics: CaptureMetrics,
}

/// Production capture service controller.
///
/// The controller lives on the application main thread. It spawns a single
/// capture worker thread that owns the backend stream. Frames are published to
/// a [`LatestSlot<VideoFrame>`] so consumers always see the most recent frame.
pub struct CaptureController {
    state: Arc<std::sync::Mutex<SharedState>>,
    command_tx: Option<std::sync::mpsc::Sender<ControlCommand>>,
    frame_slot: Arc<LatestSlot<VideoFrame>>,
    worker: Option<WorkerHandle<CaptureWorkerResult>>,
}

impl Drop for CaptureController {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            worker.stop();
            // Closing the slot wakes a worker waiting for its next frame.
            self.frame_slot.close();
            let _ = worker.join();
        }
    }
}

/// Result returned by the capture worker when it finishes.
#[derive(Clone, Debug, Default, PartialEq)]
struct CaptureWorkerResult {
    final_metrics: CaptureMetrics,
}

impl CaptureController {
    /// Creates a new controller in the idle state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Arc::new(std::sync::Mutex::new(SharedState::default())),
            command_tx: None,
            frame_slot: Arc::new(LatestSlot::new()),
            worker: None,
        }
    }

    /// Returns the capacity-one slot used for frame transport.
    #[must_use]
    pub fn frame_slot(&self) -> Arc<LatestSlot<VideoFrame>> {
        Arc::clone(&self.frame_slot)
    }

    /// Returns the current service state.
    #[must_use]
    pub fn state(&self) -> CaptureServiceState {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.state
    }

    /// Returns the currently selected device, if any.
    #[must_use]
    pub fn selected_device(&self) -> Option<CameraDescriptor> {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.selected_device.clone()
    }

    /// Returns a snapshot of current metrics.
    #[must_use]
    pub fn metrics(&self) -> CaptureMetrics {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.metrics.clone()
    }

    /// Returns whether the supervised worker has exited unexpectedly or is
    /// finishing a requested shutdown.
    #[must_use]
    pub fn worker_finished(&self) -> bool {
        self.worker.as_ref().is_some_and(WorkerHandle::is_finished)
    }

    /// Starts the capture worker.
    ///
    /// The worker owns the backend and must be started before any device command
    /// is accepted. Calling start when already started returns an error.
    pub fn start_worker<B>(&mut self, backend: B) -> Result<(), CameraError>
    where
        B: CameraBackend + Send + 'static,
    {
        if self.worker.is_some() {
            return Err(CameraError::OpenFailed(
                "capture worker already running".into(),
            ));
        }

        let (tx, rx) = std::sync::mpsc::channel::<ControlCommand>();
        self.command_tx = Some(tx);

        let state = Arc::clone(&self.state);
        let slot = Arc::clone(&self.frame_slot);

        let worker = WorkerHandle::spawn("capture-worker", move |stop| {
            run_capture_worker(backend, rx, stop, state, slot)
        });

        self.worker = Some(worker);
        Ok(())
    }

    /// Selects a device and starts capture.
    ///
    /// If a device is already running, it is stopped first. If no worker has
    /// been started, this method returns an error.
    pub fn select_and_start(
        &mut self,
        device: CameraDescriptor,
        request: CameraRequest,
    ) -> Result<(), CameraError> {
        let tx = self
            .command_tx
            .as_ref()
            .ok_or_else(|| CameraError::OpenFailed("capture worker not started".into()))?;

        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.selected_device = Some(device.clone());
            state.requested_format = Some(request);
            state.state = CaptureServiceState::Starting;
        }

        tx.send(ControlCommand::Start(device, request))
            .map_err(|_| CameraError::OpenFailed("capture worker command channel closed".into()))?;
        Ok(())
    }

    /// Stops capture but keeps the selected device.
    pub fn stop(&mut self) {
        if let Some(tx) = self.command_tx.as_ref() {
            let _ = tx.send(ControlCommand::Stop);
        }
        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.state != CaptureServiceState::Idle
                && state.state != CaptureServiceState::Selected
            {
                state.state = CaptureServiceState::Stopping;
            }
        }
    }

    /// Stops capture and clears the selected device.
    pub fn reset(&mut self) {
        if let Some(tx) = self.command_tx.as_ref() {
            let _ = tx.send(ControlCommand::Reset);
        }
        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.selected_device = None;
            state.requested_format = None;
            state.state = CaptureServiceState::Stopping;
        }
    }

    /// Requests graceful shutdown and joins the worker.
    ///
    /// This consumes the controller. After this call returns, no worker thread
    /// is running and the frame slot is closed.
    pub fn shutdown(mut self) -> CaptureMetrics {
        if let Some(worker) = self.worker.take() {
            worker.stop();
            let _ = worker.join();
        }
        self.frame_slot.close();

        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.state = CaptureServiceState::Idle;
            state.metrics.clone()
        }
    }
}

impl Default for CaptureController {
    fn default() -> Self {
        Self::new()
    }
}

/// Runs the capture worker loop until the stop token is set and all commands
/// have been processed.
fn run_capture_worker<B>(
    backend: B,
    command_rx: std::sync::mpsc::Receiver<ControlCommand>,
    stop: StopToken,
    state: Arc<std::sync::Mutex<SharedState>>,
    slot: Arc<LatestSlot<VideoFrame>>,
) -> CaptureWorkerResult
where
    B: CameraBackend,
{
    let mut active_stream: Option<Box<dyn crate::device::CameraStream>> = None;
    let mut selected_device: Option<CameraDescriptor> = None;
    let mut requested_format: Option<CameraRequest> = None;
    let mut reconnect_attempts: u32 = 0;
    let mut next_frame_seq = 0u64;
    let mut metrics = CaptureMetrics::default();

    while !stop.is_stopped() {
        // Drain control commands first so state changes take effect immediately.
        loop {
            match command_rx.try_recv() {
                Ok(ControlCommand::Start(device, request)) => {
                    if let Some(mut stream) = active_stream.take() {
                        let _ = stream.stop();
                    }
                    selected_device = Some(device);
                    requested_format = Some(request);
                    reconnect_attempts = 0;
                    update_state(&state, |s| {
                        s.state = CaptureServiceState::Starting;
                        s.metrics.reconnect_attempts = 0;
                        s.metrics.last_error = None;
                    });

                    // Invariant: `selected_device` was assigned `Some(device)` above.
                    let Some(device) = selected_device.as_ref() else {
                        continue;
                    };
                    match open_and_stream(
                        &backend,
                        device,
                        request,
                        &stop,
                        &state,
                        &slot,
                        &mut metrics,
                        &mut next_frame_seq,
                    ) {
                        Ok(stream) => {
                            active_stream = Some(stream);
                            reconnect_attempts = 0;
                            update_state(&state, |s| {
                                s.state = CaptureServiceState::Running;
                                s.metrics.reconnect_attempts = 0;
                            });
                        }
                        Err(err) => {
                            metrics.last_error = Some(format!("{err:?}"));
                            update_state(&state, |s| {
                                s.state = CaptureServiceState::BackOff;
                                s.metrics.last_error.clone_from(&metrics.last_error);
                            });
                            active_stream = None;
                        }
                    }
                }
                Ok(ControlCommand::Stop) => {
                    if let Some(mut stream) = active_stream.take() {
                        let _ = stream.stop();
                    }
                    slot.clear();
                    update_state(&state, |s| {
                        s.state = CaptureServiceState::Selected;
                    });
                }
                Ok(ControlCommand::Reset) => {
                    if let Some(mut stream) = active_stream.take() {
                        let _ = stream.stop();
                    }
                    slot.clear();
                    selected_device = None;
                    requested_format = None;
                    reconnect_attempts = 0;
                    update_state(&state, |s| {
                        s.state = CaptureServiceState::Idle;
                        s.selected_device = None;
                        s.requested_format = None;
                    });
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    stop.stop();
                    break;
                }
            }
        }

        // If a stream is active, capture one frame with a short timeout so we
        // remain responsive to stop/commands.
        if let Some(stream) = active_stream.as_mut() {
            match stream.next_frame(&stop) {
                Ok(frame) => {
                    reconnect_attempts = 0;
                    let frame = stamp_frame_sequence(frame, &mut next_frame_seq);
                    metrics.frames_captured = metrics.frames_captured.saturating_add(1);
                    if !slot.publish(frame) {
                        metrics.frames_dropped = metrics.frames_dropped.saturating_add(1);
                    }
                    update_state(&state, |s| {
                        s.metrics.frames_captured = metrics.frames_captured;
                        s.metrics.frames_dropped = metrics.frames_dropped;
                        s.metrics.format = metrics.format;
                        s.state = CaptureServiceState::Running;
                    });
                }
                Err(CameraError::Disconnected) => {
                    active_stream = None;
                    slot.clear();
                    metrics.last_error = Some("CAMERA_DISCONNECTED".into());
                    if reconnect_attempts < MAX_RECONNECT_ATTEMPTS && selected_device.is_some() {
                        reconnect_attempts += 1;
                        update_state(&state, |s| {
                            s.state = CaptureServiceState::Reconnecting;
                            s.metrics.reconnect_attempts = reconnect_attempts;
                            s.metrics.last_error.clone_from(&metrics.last_error);
                        });
                        std::thread::sleep(reconnect_delay(reconnect_attempts));

                        if let (Some(device), Some(request)) =
                            (selected_device.as_ref(), requested_format)
                        {
                            match open_and_stream(
                                &backend,
                                device,
                                request,
                                &stop,
                                &state,
                                &slot,
                                &mut metrics,
                                &mut next_frame_seq,
                            ) {
                                Ok(stream) => {
                                    active_stream = Some(stream);
                                    update_state(&state, |s| {
                                        s.state = CaptureServiceState::Running;
                                    });
                                }
                                Err(err) => {
                                    metrics.last_error = Some(format!("{err:?}"));
                                }
                            }
                        }
                    } else {
                        update_state(&state, |s| {
                            s.state = CaptureServiceState::BackOff;
                            s.metrics.last_error.clone_from(&metrics.last_error);
                        });
                    }
                }
                Err(err) => {
                    metrics.last_error = Some(format!("{err:?}"));
                    update_state(&state, |s| {
                        s.metrics.last_error.clone_from(&metrics.last_error);
                    });
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        } else {
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    if let Some(mut stream) = active_stream.take() {
        let _ = stream.stop();
    }

    let final_metrics = metrics.clone();
    update_state(&state, |s| {
        s.state = CaptureServiceState::Idle;
        s.metrics = metrics;
    });

    CaptureWorkerResult { final_metrics }
}

/// Opens the requested camera and returns the stream.
#[allow(clippy::too_many_arguments)]
fn open_and_stream<B>(
    backend: &B,
    device: &CameraDescriptor,
    request: CameraRequest,
    stop: &StopToken,
    state: &Arc<std::sync::Mutex<SharedState>>,
    slot: &Arc<LatestSlot<VideoFrame>>,
    metrics: &mut CaptureMetrics,
    next_frame_seq: &mut u64,
) -> Result<Box<dyn crate::device::CameraStream>, CameraError>
where
    B: CameraBackend,
{
    let mut stream = backend.open(device, &request)?;
    metrics.format = Some(stream.actual_format());

    // Discard stale slot contents so the consumer does not see an old frame
    // after a reconnect.
    slot.clear();

    // Capture one frame immediately to confirm the device is really alive.
    match stream.next_frame(stop) {
        Ok(frame) => {
            let frame = stamp_frame_sequence(frame, next_frame_seq);
            metrics.frames_captured = metrics.frames_captured.saturating_add(1);
            if !slot.publish(frame) {
                metrics.frames_dropped = metrics.frames_dropped.saturating_add(1);
            }
            update_state(state, |s| {
                s.metrics.frames_captured = metrics.frames_captured;
                s.metrics.frames_dropped = metrics.frames_dropped;
                s.metrics.format = metrics.format;
            });
            Ok(stream)
        }
        Err(err) => {
            let _ = stream.stop();
            Err(err)
        }
    }
}

/// Assigns the capture-owned sequence number to a frame.
///
/// Backend streams may restart their local counters after a reconnect. The
/// capture worker owns the cross-session sequence contract, so the sequence
/// is overwritten here before a frame crosses the worker boundary.
fn stamp_frame_sequence(mut frame: VideoFrame, next_frame_seq: &mut u64) -> VideoFrame {
    *next_frame_seq = next_frame_seq.saturating_add(1);
    frame.seq = FrameSeq(*next_frame_seq);
    frame
}

/// Computes an exponential-backoff delay capped at [`RECONNECT_DELAY_MAX`].
fn reconnect_delay(attempt: u32) -> Duration {
    let base = RECONNECT_DELAY_BASE.as_millis() as u64;
    let delay_ms = base.saturating_mul(2_u64.saturating_pow(attempt.min(10)));
    let delay_ms = delay_ms.min(RECONNECT_DELAY_MAX.as_millis() as u64);
    Duration::from_millis(delay_ms)
}

/// Helper to update the shared state under the mutex.
fn update_state<F>(state: &Arc<std::sync::Mutex<SharedState>>, f: F)
where
    F: FnOnce(&mut SharedState),
{
    let mut guard = state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    f(&mut guard);
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )] // tests may panic (AGENTS.md)

    use super::*;
    use crate::device::{CameraDescriptor, CameraRequest};
    use crate::mock::MockBackend;
    use std::time::Duration;

    #[test]
    fn controller_starts_and_stops() {
        let mut controller = CaptureController::new();
        controller.start_worker(MockBackend::default()).unwrap();

        let device = CameraDescriptor {
            id: "mock-0".into(),
            label: "Mock".into(),
        };
        controller
            .select_and_start(device, CameraRequest::default())
            .unwrap();

        // Wait briefly for the worker to produce at least one frame.
        let slot = controller.frame_slot();
        let result = slot.wait_read_after(0, Duration::from_secs(2));
        assert!(matches!(result, Some(vtuber_core::ReadResult::New(_))));

        let metrics = controller.shutdown();
        assert!(metrics.frames_captured > 0);
    }

    #[test]
    fn stop_keeps_selection() {
        let mut controller = CaptureController::new();
        controller.start_worker(MockBackend::default()).unwrap();

        let device = CameraDescriptor {
            id: "mock-0".into(),
            label: "Mock".into(),
        };
        controller
            .select_and_start(device.clone(), CameraRequest::default())
            .unwrap();

        let slot = controller.frame_slot();
        let _ = slot.wait_read_after(0, Duration::from_secs(2));

        controller.stop();
        std::thread::sleep(Duration::from_millis(50));

        assert_eq!(controller.selected_device(), Some(device));
        assert_eq!(slot.try_read_after(0), None);
        let _ = controller.shutdown();
    }

    #[test]
    fn reset_clears_selection() {
        let mut controller = CaptureController::new();
        controller.start_worker(MockBackend::default()).unwrap();

        let device = CameraDescriptor {
            id: "mock-0".into(),
            label: "Mock".into(),
        };
        controller
            .select_and_start(device, CameraRequest::default())
            .unwrap();

        controller.reset();
        std::thread::sleep(Duration::from_millis(50));

        assert_eq!(controller.selected_device(), None);
        let _ = controller.shutdown();
    }

    #[test]
    fn worker_stops_on_shutdown() {
        let mut controller = CaptureController::new();
        controller.start_worker(MockBackend::default()).unwrap();

        let device = CameraDescriptor {
            id: "mock-0".into(),
            label: "Mock".into(),
        };
        controller
            .select_and_start(device, CameraRequest::default())
            .unwrap();

        std::thread::sleep(Duration::from_millis(30));
        let slot = controller.frame_slot();
        let metrics = controller.shutdown();

        assert!(slot.is_closed());
        assert!(metrics.frames_captured > 0);
    }

    #[test]
    fn reconnect_after_disconnect() {
        let mut controller = CaptureController::new();
        let backend = MockBackend {
            disconnect_after: Some(1),
            ..Default::default()
        };
        controller.start_worker(backend).unwrap();

        let device = CameraDescriptor {
            id: "mock-0".into(),
            label: "Mock".into(),
        };
        controller
            .select_and_start(device, CameraRequest::default())
            .unwrap();

        // Wait long enough for at least one disconnect and reconnect. The
        // mock resets its per-stream counter, so `frames_captured` is the
        // reliable signal that reconnect happened.
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while std::time::Instant::now() < deadline {
            let metrics = controller.metrics();
            if metrics.frames_captured >= 3 {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }

        let metrics = controller.shutdown();
        assert!(
            metrics.frames_captured >= 3,
            "expected reconnect to produce more frames, got {:?}",
            metrics
        );
    }

    #[test]
    fn reconnect_keeps_capture_sequence_monotonic() {
        let mut next_frame_seq = 0;
        let frame = || VideoFrame {
            seq: FrameSeq(0),
            captured_at: vtuber_core::MonoTimeNs(0),
            width: 1,
            height: 1,
            stride_bytes: 1,
            format: vtuber_core::PixelFormat::Gray8,
            data: vec![0].into(),
        };

        // A backend is allowed to restart its local sequence at zero after a
        // reconnect; the capture-owned stamp must not do so.
        assert_eq!(stamp_frame_sequence(frame(), &mut next_frame_seq).seq.0, 1);
        assert_eq!(stamp_frame_sequence(frame(), &mut next_frame_seq).seq.0, 2);
    }

    #[test]
    fn double_start_worker_fails() {
        let mut controller = CaptureController::new();
        controller.start_worker(MockBackend::default()).unwrap();
        let result = controller.start_worker(MockBackend::default());
        assert!(result.is_err());
        let _ = controller.shutdown();
    }

    #[test]
    fn select_without_worker_fails() {
        let mut controller = CaptureController::new();
        let device = CameraDescriptor {
            id: "mock-0".into(),
            label: "Mock".into(),
        };
        let result = controller.select_and_start(device, CameraRequest::default());
        assert!(result.is_err());
    }
}
