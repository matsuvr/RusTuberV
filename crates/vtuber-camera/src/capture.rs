//! Production capture service.
//!
//! [`CaptureController`] owns the lifecycle of the capture worker: selecting a
//! device, starting and stopping capture, and requesting clean shutdown. The
//! native camera object is constructed, opened, used, stopped, and dropped
//! inside the worker thread.

use std::sync::Arc;
use std::time::{Duration, Instant};

use vtuber_core::{FrameSeq, LatestSlot, StopToken, VideoFrame, WorkerHandle, WorkerResult};

use crate::device::{CameraBackend, CameraDescriptor, CameraError, CameraFormat, CameraRequest};

/// Maximum number of consecutive reconnect attempts before giving up.
const MAX_RECONNECT_ATTEMPTS: u32 = 5;

/// Initial delay before the first reconnect attempt.
const RECONNECT_DELAY_BASE: Duration = Duration::from_millis(100);

/// Maximum delay between reconnect attempts.
const RECONNECT_DELAY_MAX: Duration = Duration::from_secs(5);

/// A pending reconnect episode; attempts counts actual calls to reopen.
#[derive(Clone, Copy, Debug)]
struct ReconnectPlan {
    attempts: u32,
    next_attempt_at: Instant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReconnectDecision {
    Wait,
    Attempt,
    Exhausted,
}

fn reconnect_decision(plan: &ReconnectPlan, now: Instant) -> ReconnectDecision {
    if plan.attempts >= MAX_RECONNECT_ATTEMPTS {
        ReconnectDecision::Exhausted
    } else if now < plan.next_attempt_at {
        ReconnectDecision::Wait
    } else {
        ReconnectDecision::Attempt
    }
}

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
    /// Number of face-frame publications rejected because the slot was closed.
    /// Reader skips and decode failures are not included.
    pub publish_rejected_frames: u64,
    /// Actual reopen attempts in the current or most recent reconnect episode.
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
#[derive(Clone, Debug, Default, PartialEq)]
struct SharedState {
    state: CaptureServiceState,
    selected_device: Option<CameraDescriptor>,
    requested_format: Option<CameraRequest>,
    metrics: CaptureMetrics,
}

/// Publishes one captured frame to the face consumer and, when enabled, the
/// Pose consumer.
///
/// Both slots receive the same `VideoFrame`, whose pixel buffer is an `Arc`, so
/// the image body is never copied and the camera is never opened twice. Returns
/// whether the face slot accepted the frame; the Pose slot is allowed to be
/// overwritten because it runs at its own cadence. Each slot keeps its own
/// single-producer/single-consumer contract.
#[must_use]
pub fn publish_tracking_frame(
    frame: VideoFrame,
    face_slot: &LatestSlot<VideoFrame>,
    pose_slot: Option<&LatestSlot<VideoFrame>>,
) -> bool {
    let sent = face_slot.publish(frame.clone());
    if let Some(pose_slot) = pose_slot {
        let _ = pose_slot.publish(frame);
    }
    sent
}

/// Discards retained frames from both consumers, e.g. across a stop or reconnect.
fn clear_frame_slots(
    face_slot: &LatestSlot<VideoFrame>,
    pose_slot: Option<&LatestSlot<VideoFrame>>,
) {
    face_slot.clear();
    if let Some(pose_slot) = pose_slot {
        pose_slot.clear();
    }
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
    pose_slot: Option<Arc<LatestSlot<VideoFrame>>>,
    worker: Option<WorkerHandle<CaptureWorkerResult>>,
}

impl std::fmt::Debug for CaptureController {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CaptureController")
            .field("state", &self.state)
            .field("has_worker", &self.worker.is_some())
            .field("has_command_channel", &self.command_tx.is_some())
            .finish()
    }
}

impl Drop for CaptureController {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            worker.stop();
            // Closing the slot wakes a worker waiting for its next frame.
            self.frame_slot.close();
            // Drop cannot return a shutdown error; explicit shutdown can.
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
            pose_slot: None,
            worker: None,
        }
    }

    /// Returns the capacity-one slot used for frame transport.
    #[must_use]
    pub fn frame_slot(&self) -> Arc<LatestSlot<VideoFrame>> {
        Arc::clone(&self.frame_slot)
    }

    /// Enables or disables the second capacity-one slot used by the Pose worker.
    ///
    /// Must be called before [`CaptureController::start_worker`]. Passing `None`
    /// disables the arm-tracking fan-out, so a disabled Pose worker is never
    /// fed frames.
    pub fn set_pose_output(&mut self, pose_slot: Option<Arc<LatestSlot<VideoFrame>>>) {
        self.pose_slot = pose_slot;
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
    /// is accepted. `Ok` acknowledges thread creation, not device opening.
    ///
    /// # Errors
    /// Returns `OpenFailed` when a worker already exists, or `WorkerSpawnFailed`
    /// when the OS cannot create the thread. No worker is retained on spawn failure.
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

        let state = Arc::clone(&self.state);
        let slot = Arc::clone(&self.frame_slot);
        let pose_slot = self.pose_slot.clone();

        let worker = WorkerHandle::spawn("capture-worker", move |stop| {
            run_capture_worker(backend, rx, stop, state, slot, pose_slot)
        })
        .map_err(CameraError::WorkerSpawnFailed)?;

        self.command_tx = Some(tx);
        self.worker = Some(worker);
        Ok(())
    }

    /// Selects a device and starts capture.
    ///
    /// If a device is already running, it is stopped first. If no worker has
    /// been started, this method returns an error. Success means enqueued, not
    /// that the camera has opened. The worker owns the completed state.
    ///
    /// # Errors
    /// Returns an error if no worker is started or its command channel is
    /// closed. The selected device, request and state stay unchanged.
    pub fn select_and_start(
        &mut self,
        device: CameraDescriptor,
        request: CameraRequest,
    ) -> Result<(), CameraError> {
        let tx = self
            .command_tx
            .as_ref()
            .ok_or_else(|| CameraError::OpenFailed("capture worker not started".into()))?;

        // Clone before locking. The unbounded send does not wait for a reader;
        // holding the state lock prevents the worker's completion from racing
        // ahead of the accepted-request state written below.
        let selected_device = device.clone();
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        tx.send(ControlCommand::Start(device, request))
            .map_err(|_| CameraError::CommandChannelClosed)?;
        state.selected_device = Some(selected_device);
        state.requested_format = Some(request);
        state.state = CaptureServiceState::Starting;
        Ok(())
    }

    /// Requests capture stop, keeping the selected device.
    ///
    /// `Ok(())` means accepted, not completed. The worker changes Stopping to
    /// Selected (or Idle without a selection). Before startup this is a no-op.
    ///
    /// # Errors
    /// Returns [`CameraError::CommandChannelClosed`] without changing state if
    /// the worker can no longer receive the request.
    pub fn stop(&mut self) -> Result<(), CameraError> {
        let Some(tx) = self.command_tx.as_ref() else {
            return Ok(());
        };
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        tx.send(ControlCommand::Stop)
            .map_err(|_| CameraError::CommandChannelClosed)?;
        state.state = CaptureServiceState::Stopping;
        Ok(())
    }

    /// Requests capture stop and selection reset.
    ///
    /// `Ok(())` means accepted, not completed. The worker clears selection and
    /// returns to Idle. With no worker, selection is cleared immediately.
    ///
    /// # Errors
    /// Returns [`CameraError::CommandChannelClosed`] without changing state if
    /// the worker can no longer receive the request.
    pub fn reset(&mut self) -> Result<(), CameraError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(tx) = self.command_tx.as_ref() else {
            state.selected_device = None;
            state.requested_format = None;
            state.state = CaptureServiceState::Idle;
            return Ok(());
        };
        tx.send(ControlCommand::Reset)
            .map_err(|_| CameraError::CommandChannelClosed)?;
        state.state = CaptureServiceState::Stopping;
        Ok(())
    }

    /// Requests graceful shutdown and joins the worker.
    ///
    /// This consumes the controller. After this call returns, no worker thread
    /// is running and the frame slot is closed.
    ///
    /// # Errors
    /// Returns `WorkerPanicked` after closing the owned frame slot. Joining can
    /// block. Dropping a live controller also stops and joins, but cannot report
    /// that error; use explicit shutdown when its result matters.
    pub fn shutdown(mut self) -> Result<CaptureMetrics, CameraError> {
        let result = if let Some(worker) = self.worker.take() {
            worker.stop();
            worker.join()
        } else {
            WorkerResult::Completed(CaptureWorkerResult {
                final_metrics: self.metrics(),
            })
        };
        // Cleanup must also run after a panicked worker.
        self.frame_slot.close();
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match result {
            WorkerResult::Completed(result) => {
                state.state = CaptureServiceState::Idle;
                Ok(result.final_metrics)
            }
            WorkerResult::Panicked => {
                state.state = CaptureServiceState::BackOff;
                state.metrics.last_error = Some(CameraError::WorkerPanicked.to_string());
                Err(CameraError::WorkerPanicked)
            }
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
    pose_slot: Option<Arc<LatestSlot<VideoFrame>>>,
) -> CaptureWorkerResult
where
    B: CameraBackend,
{
    let mut active_stream: Option<Box<dyn crate::device::CameraStream>> = None;
    let mut selected_device: Option<CameraDescriptor> = None;
    let mut requested_format: Option<CameraRequest> = None;
    let mut reconnect_plan: Option<ReconnectPlan> = None;
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
                    reconnect_plan = None;
                    metrics.reconnect_attempts = 0;
                    metrics.last_error = None;
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
                        pose_slot.as_deref(),
                        &mut metrics,
                        &mut next_frame_seq,
                    ) {
                        Ok(stream) => {
                            active_stream = Some(stream);
                            reconnect_plan = None;
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
                    reconnect_plan = None;
                    if let Some(mut stream) = active_stream.take() {
                        let _ = stream.stop();
                    }
                    clear_frame_slots(&slot, pose_slot.as_deref());
                    update_state(&state, |s| {
                        s.state = if selected_device.is_some() {
                            CaptureServiceState::Selected
                        } else {
                            CaptureServiceState::Idle
                        };
                    });
                }
                Ok(ControlCommand::Reset) => {
                    if let Some(mut stream) = active_stream.take() {
                        let _ = stream.stop();
                    }
                    clear_frame_slots(&slot, pose_slot.as_deref());
                    selected_device = None;
                    requested_format = None;
                    reconnect_plan = None;
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
                    reconnect_plan = None;
                    let frame = stamp_frame_sequence(frame, &mut next_frame_seq);
                    metrics.frames_captured = metrics.frames_captured.saturating_add(1);
                    if !publish_tracking_frame(frame, &slot, pose_slot.as_deref()) {
                        metrics.publish_rejected_frames =
                            metrics.publish_rejected_frames.saturating_add(1);
                    }
                    update_state(&state, |s| {
                        s.metrics.frames_captured = metrics.frames_captured;
                        s.metrics.publish_rejected_frames = metrics.publish_rejected_frames;
                        s.metrics.format = metrics.format;
                        s.state = CaptureServiceState::Running;
                    });
                }
                Err(CameraError::Disconnected) => {
                    active_stream = None;
                    clear_frame_slots(&slot, pose_slot.as_deref());
                    metrics.last_error = Some("CAMERA_DISCONNECTED".into());
                    metrics.reconnect_attempts = 0;
                    reconnect_plan = selected_device.as_ref().map(|_| ReconnectPlan {
                        attempts: 0,
                        next_attempt_at: Instant::now() + reconnect_delay(1),
                    });
                    update_state(&state, |s| {
                        s.state = if reconnect_plan.is_some() {
                            CaptureServiceState::Reconnecting
                        } else {
                            CaptureServiceState::BackOff
                        };
                        s.metrics.reconnect_attempts = 0;
                        s.metrics.last_error.clone_from(&metrics.last_error);
                    });
                }
                Err(err) => {
                    metrics.last_error = Some(format!("{err:?}"));
                    update_state(&state, |s| {
                        s.metrics.last_error.clone_from(&metrics.last_error);
                    });
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        } else if let Some(mut plan) = reconnect_plan {
            match reconnect_decision(&plan, Instant::now()) {
                ReconnectDecision::Wait => std::thread::sleep(Duration::from_millis(10)),
                ReconnectDecision::Exhausted => {
                    reconnect_plan = None;
                    update_state(&state, |s| {
                        s.state = CaptureServiceState::BackOff;
                        s.metrics.last_error.clone_from(&metrics.last_error);
                    });
                }
                ReconnectDecision::Attempt => {
                    if let (Some(device), Some(request)) =
                        (selected_device.as_ref(), requested_format)
                    {
                        plan.attempts += 1;
                        metrics.reconnect_attempts = plan.attempts;
                        update_state(&state, |s| s.metrics.reconnect_attempts = plan.attempts);
                        match open_and_stream(
                            &backend,
                            device,
                            request,
                            &stop,
                            &state,
                            &slot,
                            pose_slot.as_deref(),
                            &mut metrics,
                            &mut next_frame_seq,
                        ) {
                            Ok(stream) => {
                                active_stream = Some(stream);
                                reconnect_plan = None;
                                update_state(&state, |s| s.state = CaptureServiceState::Running);
                            }
                            Err(error) => {
                                metrics.last_error = Some(format!("{error:?}"));
                                plan.next_attempt_at =
                                    Instant::now() + reconnect_delay(plan.attempts + 1);
                                reconnect_plan = Some(plan);
                                update_state(&state, |s| {
                                    s.metrics.last_error.clone_from(&metrics.last_error)
                                });
                            }
                        }
                    }
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
    pose_slot: Option<&LatestSlot<VideoFrame>>,
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
    clear_frame_slots(slot, pose_slot);

    // Capture one frame immediately to confirm the device is really alive.
    match stream.next_frame(stop) {
        Ok(frame) => {
            let frame = stamp_frame_sequence(frame, next_frame_seq);
            metrics.frames_captured = metrics.frames_captured.saturating_add(1);
            if !publish_tracking_frame(frame, slot, pose_slot) {
                metrics.publish_rejected_frames = metrics.publish_rejected_frames.saturating_add(1);
            }
            update_state(state, |s| {
                s.metrics.frames_captured = metrics.frames_captured;
                s.metrics.publish_rejected_frames = metrics.publish_rejected_frames;
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

    struct ScriptedReconnectBackend {
        opens: std::sync::Mutex<std::collections::VecDeque<Result<bool, &'static str>>>,
        opened: std::sync::mpsc::Sender<()>,
    }

    struct ScriptedStream {
        disconnect: bool,
        frames: u64,
    }

    impl crate::device::CameraBackend for ScriptedReconnectBackend {
        fn enumerate(&self) -> Result<Vec<CameraDescriptor>, CameraError> {
            Ok(vec![test_device()])
        }

        fn open(
            &self,
            _: &CameraDescriptor,
            _: &CameraRequest,
        ) -> Result<Box<dyn crate::device::CameraStream>, CameraError> {
            self.opened.send(()).unwrap();
            let disconnect = self
                .opens
                .lock()
                .unwrap()
                .pop_front()
                .expect("unexpected extra open")
                .map_err(|message| CameraError::OpenFailed(message.to_owned()))?;
            Ok(Box::new(ScriptedStream {
                disconnect,
                frames: 0,
            }))
        }
    }

    impl crate::device::CameraStream for ScriptedStream {
        fn actual_format(&self) -> CameraFormat {
            CameraFormat {
                width: 1,
                height: 1,
                fps_numerator: 30,
                fps_denominator: 1,
                format: vtuber_core::PixelFormat::Rgb8,
            }
        }

        fn next_frame(&mut self, stop: &StopToken) -> Result<VideoFrame, CameraError> {
            if stop.is_stopped() || (self.disconnect && self.frames > 0) {
                return Err(CameraError::Disconnected);
            }
            self.frames += 1;
            Ok(VideoFrame {
                seq: FrameSeq(self.frames),
                captured_at: vtuber_core::monotonic_now(),
                width: 1,
                height: 1,
                stride_bytes: 3,
                format: vtuber_core::PixelFormat::Rgb8,
                data: Arc::from([0; 3]),
            })
        }

        fn stop(&mut self) -> Result<(), CameraError> {
            Ok(())
        }
    }

    fn test_device() -> CameraDescriptor {
        CameraDescriptor {
            id: "scripted".into(),
            label: "Scripted".into(),
        }
    }

    fn scripted_controller(
        opens: Vec<Result<bool, &'static str>>,
    ) -> (CaptureController, std::sync::mpsc::Receiver<()>) {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut controller = CaptureController::new();
        controller
            .start_worker(ScriptedReconnectBackend {
                opens: std::sync::Mutex::new(opens.into()),
                opened: tx,
            })
            .unwrap();
        controller
            .select_and_start(test_device(), CameraRequest::default())
            .unwrap();
        (controller, rx)
    }

    fn wait_for_state(controller: &CaptureController, expected: CaptureServiceState) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while controller.state() != expected {
            assert!(
                Instant::now() < deadline,
                "worker did not reach {expected:?}"
            );
            std::thread::yield_now();
        }
    }

    #[test]
    fn debug_only_reads_the_controller_summary() {
        let controller = CaptureController::new();
        let text = format!("{controller:?}");
        assert!(text.contains("CaptureController"));
        assert!(text.contains("Idle"));
        assert!(text.contains("has_worker: false"));
        assert!(text.contains("has_command_channel: false"));
        assert!(controller.worker.is_none());
        assert!(!controller.frame_slot.is_closed());
    }

    #[test]
    fn explicit_shutdown_reports_panic_and_closes_the_frame_slot() {
        let mut controller = CaptureController::new();
        let slot = controller.frame_slot();
        controller.worker = Some(
            WorkerHandle::spawn("capture-panic-test", |_| {
                panic!("scripted worker panic");
            })
            .unwrap(),
        );
        assert!(matches!(
            controller.shutdown(),
            Err(CameraError::WorkerPanicked)
        ));
        assert!(slot.is_closed());
    }

    #[test]
    fn reconnect_decision_uses_only_the_supplied_clock_and_attempt_count() {
        let now = Instant::now();
        let mut plan = ReconnectPlan {
            attempts: 0,
            next_attempt_at: now + Duration::from_secs(1),
        };
        assert_eq!(reconnect_decision(&plan, now), ReconnectDecision::Wait);
        assert_eq!(
            reconnect_decision(&plan, plan.next_attempt_at),
            ReconnectDecision::Attempt
        );
        plan.attempts = MAX_RECONNECT_ATTEMPTS;
        assert_eq!(reconnect_decision(&plan, now), ReconnectDecision::Exhausted);
        assert_eq!(reconnect_delay(1), Duration::from_millis(200));
        assert_eq!(reconnect_delay(100), RECONNECT_DELAY_MAX);
    }

    #[test]
    fn reconnect_retries_two_failures_then_returns_to_running() {
        let (controller, opened) =
            scripted_controller(vec![Ok(true), Err("first"), Err("second"), Ok(false)]);
        for _ in 0..4 {
            opened.recv_timeout(Duration::from_secs(3)).unwrap();
        }
        wait_for_state(&controller, CaptureServiceState::Running);
        assert_eq!(controller.metrics().reconnect_attempts, 3);
        assert!(!controller.worker_finished());
        let _ = controller.shutdown();
    }

    #[test]
    fn reconnect_exhaustion_keeps_the_last_error_and_stops_opening() {
        let mut script = vec![Ok(true)];
        script.extend((0..MAX_RECONNECT_ATTEMPTS).map(|_| Err("last reopen failure")));
        let (controller, opened) = scripted_controller(script);
        for _ in 0..=MAX_RECONNECT_ATTEMPTS {
            opened.recv_timeout(Duration::from_secs(4)).unwrap();
        }
        wait_for_state(&controller, CaptureServiceState::BackOff);
        assert_eq!(
            controller.metrics().reconnect_attempts,
            MAX_RECONNECT_ATTEMPTS
        );
        assert!(
            controller
                .metrics()
                .last_error
                .unwrap()
                .contains("last reopen failure")
        );
        assert!(matches!(
            opened.recv_timeout(Duration::from_millis(50)),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout)
        ));
        assert!(!controller.worker_finished());
        let _ = controller.shutdown();
    }

    #[test]
    fn stop_reset_and_new_start_cancel_a_pending_reconnect() {
        for command in [
            ControlCommand::Stop,
            ControlCommand::Reset,
            ControlCommand::Start(test_device(), CameraRequest::default()),
        ] {
            let (mut controller, opened) = scripted_controller(vec![Ok(true), Ok(false)]);
            opened.recv_timeout(Duration::from_secs(2)).unwrap();
            wait_for_state(&controller, CaptureServiceState::Reconnecting);
            match command {
                ControlCommand::Stop => {
                    controller.stop().unwrap();
                    wait_for_state(&controller, CaptureServiceState::Selected);
                    assert!(controller.frame_slot().try_read_after(0).is_none());
                }
                ControlCommand::Reset => {
                    controller.reset().unwrap();
                    wait_for_state(&controller, CaptureServiceState::Idle);
                    assert!(controller.frame_slot().try_read_after(0).is_none());
                }
                ControlCommand::Start(device, request) => {
                    controller.select_and_start(device, request).unwrap();
                    opened.recv_timeout(Duration::from_secs(2)).unwrap();
                    wait_for_state(&controller, CaptureServiceState::Running);
                }
            }
            assert!(matches!(
                opened.recv_timeout(reconnect_delay(1) + Duration::from_millis(50)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ));
            assert_eq!(controller.metrics().reconnect_attempts, 0);
            assert!(!controller.worker_finished());
            let _ = controller.shutdown();
        }
    }

    #[test]
    fn commands_before_worker_start_do_not_leave_stopping_state() {
        let mut controller = CaptureController::new();
        controller.stop().unwrap();
        assert_eq!(controller.state(), CaptureServiceState::Idle);
        controller.reset().unwrap();
        assert_eq!(controller.state(), CaptureServiceState::Idle);
        assert_eq!(controller.selected_device(), None);
    }

    #[test]
    fn failed_control_send_preserves_the_entire_snapshot() {
        let mut controller = CaptureController::new();
        let device = CameraDescriptor {
            id: "original".into(),
            label: "Original".into(),
        };
        update_state(&controller.state, |s| {
            s.state = CaptureServiceState::Running;
            s.selected_device = Some(device.clone());
            s.requested_format = Some(CameraRequest::default());
            s.metrics.frames_captured = 7;
        });
        let (tx, rx) = std::sync::mpsc::channel();
        controller.command_tx = Some(tx);
        drop(rx);
        let before = controller.state.lock().unwrap().clone();
        assert!(matches!(
            controller.stop(),
            Err(CameraError::CommandChannelClosed)
        ));
        assert_eq!(*controller.state.lock().unwrap(), before);
        assert!(matches!(
            controller.reset(),
            Err(CameraError::CommandChannelClosed)
        ));
        assert_eq!(*controller.state.lock().unwrap(), before);
        assert!(matches!(
            controller.select_and_start(
                CameraDescriptor {
                    id: "replacement".into(),
                    label: "Replacement".into()
                },
                CameraRequest::default()
            ),
            Err(CameraError::CommandChannelClosed)
        ));
        assert_eq!(*controller.state.lock().unwrap(), before);
    }

    #[test]
    fn accepted_commands_expose_requested_state_before_completion() {
        let mut controller = CaptureController::new();
        let (tx, rx) = std::sync::mpsc::channel();
        controller.command_tx = Some(tx);
        let device = CameraDescriptor {
            id: "mock-0".into(),
            label: "Mock".into(),
        };
        controller
            .select_and_start(device.clone(), CameraRequest::default())
            .unwrap();
        assert!(matches!(rx.try_recv().unwrap(), ControlCommand::Start(..)));
        assert_eq!(controller.state(), CaptureServiceState::Starting);
        update_state(&controller.state, |s| {
            s.state = CaptureServiceState::Running
        });
        controller.stop().unwrap();
        assert_eq!(rx.try_recv().unwrap(), ControlCommand::Stop);
        assert_eq!(controller.state(), CaptureServiceState::Stopping);
        update_state(&controller.state, |s| {
            s.state = CaptureServiceState::Selected
        });
        assert_eq!(controller.state(), CaptureServiceState::Selected);
        controller.reset().unwrap();
        assert_eq!(rx.try_recv().unwrap(), ControlCommand::Reset);
        assert_eq!(controller.selected_device(), Some(device));
        update_state(&controller.state, |s| {
            s.state = CaptureServiceState::Idle;
            s.selected_device = None;
            s.requested_format = None;
        });
        assert_eq!(controller.state(), CaptureServiceState::Idle);
        assert_eq!(controller.selected_device(), None);
    }

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
        assert!(matches!(result, Some(vtuber_core::ReadResult::New { .. })));

        let metrics = controller.shutdown().unwrap();
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

        controller.stop().unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while controller.state() != CaptureServiceState::Selected {
            assert!(
                std::time::Instant::now() < deadline,
                "worker did not complete Stop"
            );
            std::thread::yield_now();
        }
        assert_eq!(controller.selected_device(), Some(device));
        assert!(slot.try_read_after(0).is_none());
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

        controller.reset().unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while controller.state() != CaptureServiceState::Idle {
            assert!(
                std::time::Instant::now() < deadline,
                "worker did not complete Reset"
            );
            std::thread::yield_now();
        }
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
        let metrics = controller.shutdown().unwrap();

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

        let metrics = controller.shutdown().unwrap();
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
    fn tracking_fan_out_shares_one_pixel_buffer() {
        let face: LatestSlot<VideoFrame> = LatestSlot::new();
        let pose: LatestSlot<VideoFrame> = LatestSlot::new();
        let frame = VideoFrame {
            seq: FrameSeq(1),
            captured_at: vtuber_core::MonoTimeNs(0),
            width: 1,
            height: 1,
            stride_bytes: 1,
            format: vtuber_core::PixelFormat::Gray8,
            data: vec![7_u8].into(),
        };
        let pointer = Arc::as_ptr(&frame.data);
        assert!(publish_tracking_frame(frame, &face, Some(&pose)));

        let vtuber_core::ReadResult::New {
            value: face_frame, ..
        } = face.try_read_after(0).expect("face frame published")
        else {
            panic!("face slot should contain a new frame");
        };
        let vtuber_core::ReadResult::New {
            value: pose_frame, ..
        } = pose.try_read_after(0).expect("pose frame published")
        else {
            panic!("pose slot should contain a new frame");
        };
        assert_eq!(Arc::as_ptr(&face_frame.data), pointer);
        assert!(Arc::ptr_eq(&face_frame.data, &pose_frame.data));
    }

    #[test]
    fn only_a_closed_slot_rejects_a_face_frame_publication() {
        let face = LatestSlot::new();
        let frame = VideoFrame {
            seq: FrameSeq(1),
            captured_at: vtuber_core::MonoTimeNs(0),
            width: 1,
            height: 1,
            stride_bytes: 1,
            format: vtuber_core::PixelFormat::Gray8,
            data: vec![0_u8].into(),
        };
        let mut rejected = 0;
        for _ in 0..3 {
            rejected += u64::from(!publish_tracking_frame(frame.clone(), &face, None));
        }
        assert_eq!(rejected, 0);
        face.close();
        rejected += u64::from(!publish_tracking_frame(frame, &face, None));
        assert_eq!(rejected, 1);
    }

    #[test]
    fn disabled_pose_output_never_publishes_a_second_slot() {
        let face: LatestSlot<VideoFrame> = LatestSlot::new();
        let frame = VideoFrame {
            seq: FrameSeq(1),
            captured_at: vtuber_core::MonoTimeNs(0),
            width: 1,
            height: 1,
            stride_bytes: 1,
            format: vtuber_core::PixelFormat::Gray8,
            data: vec![0_u8].into(),
        };
        assert!(publish_tracking_frame(frame, &face, None));
        assert!(matches!(
            face.try_read_after(0),
            Some(vtuber_core::ReadResult::New { .. })
        ));
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
