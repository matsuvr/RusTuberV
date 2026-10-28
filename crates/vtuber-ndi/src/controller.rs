//! Sender controller, bounded mailbox, metrics, and scripted backend controls.
use crate::{
    NdiErrorCode, NdiOutputCommand, NdiOutputConfig, NdiOutputError, NdiOutputMetrics,
    NdiOutputStatus, NdiSubmitResult,
};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use vtuber_core::{FrameSeq, VideoOutputFrame};

mod worker;
#[cfg(feature = "ndi-sdk")]
use worker::run_ndi_worker;
use worker::run_scripted_worker;

/// Deterministic in-process sender used by tests and orchestration harnesses.
///
/// This backend never loads the NDI SDK or performs network I/O. Production
/// [`NdiOutputController::new`] does not use it.
#[derive(Clone, Debug)]
/// Used by app integration tests without an installed native NDI SDK.
pub struct NdiScriptedBackend {
    inner: Arc<ScriptedInner>,
}

impl NdiScriptedBackend {
    fn with_errors(init_error: Option<NdiErrorCode>, create_error: Option<NdiErrorCode>) -> Self {
        Self {
            inner: Arc::new(ScriptedInner {
                init_error,
                create_error,
                startup_allowed: Mutex::new(true),
                startup_allowed_signal: Condvar::new(),
                ready: Mutex::new(false),
                ready_signal: Condvar::new(),
                send_allowed: Mutex::new(true),
                send_allowed_signal: Condvar::new(),
                send_waiting: Mutex::new(false),
                send_waiting_signal: Condvar::new(),
                connections: std::sync::atomic::AtomicU32::new(0),
                sent_frames: AtomicU64::new(0),
                live_senders: AtomicU64::new(0),
            }),
        }
    }

    /// A backend that reaches Live and accepts BGRA frames.
    #[must_use]
    pub fn successful() -> Self {
        Self::with_errors(None, None)
    }

    /// A backend whose runtime initialization fails with the supplied code.
    #[must_use]
    pub fn fail_initialize(code: NdiErrorCode) -> Self {
        Self::with_errors(Some(code), None)
    }

    /// A backend that initializes but cannot create a sender.
    #[must_use]
    pub fn fail_create_sender() -> Self {
        Self::with_errors(None, Some(NdiErrorCode::SenderCreateFailed))
    }

    /// Holds startup until [`Self::release_startup`] so Starting can be observed.
    pub fn hold_startup(&self) -> Result<(), NdiOutputError> {
        *checked_lock(self.inner.startup_allowed.lock())? = false;
        Ok(())
    }

    /// Allows a held startup to continue.
    pub fn release_startup(&self) -> Result<(), NdiOutputError> {
        *checked_lock(self.inner.startup_allowed.lock())? = true;
        self.inner.startup_allowed_signal.notify_all();
        Ok(())
    }

    /// Blocks [`Self`] send calls until [`Self::release_send`].
    pub fn hold_send(&self) -> Result<(), NdiOutputError> {
        *checked_lock(self.inner.send_allowed.lock())? = false;
        Ok(())
    }

    /// Releases a blocked send so the worker can continue.
    pub fn release_send(&self) -> Result<(), NdiOutputError> {
        *checked_lock(self.inner.send_allowed.lock())? = true;
        self.inner.send_allowed_signal.notify_all();
        Ok(())
    }

    /// Waits until the worker is blocked inside a send call.
    pub fn wait_until_send_blocked(&self) -> Result<(), NdiOutputError> {
        let mut waiting = checked_lock(self.inner.send_waiting.lock())?;
        while !*waiting {
            waiting = checked_lock(self.inner.send_waiting_signal.wait(waiting))?;
        }
        Ok(())
    }

    /// Waits until the worker has published Live or Error.
    pub fn wait_until_ready(&self) -> Result<(), NdiOutputError> {
        let mut ready = checked_lock(self.inner.ready.lock())?;
        while !*ready {
            ready = checked_lock(self.inner.ready_signal.wait(ready))?;
        }
        Ok(())
    }

    /// Number of frames accepted by the fake sender.
    #[must_use]
    pub fn sent_frames(&self) -> u64 {
        self.inner.sent_frames.load(Ordering::Relaxed)
    }

    /// Number of live fake sender handles that have not been dropped.
    #[must_use]
    pub fn live_senders(&self) -> u64 {
        self.inner.live_senders.load(Ordering::Relaxed)
    }

    fn mark_ready(&self) -> Result<(), NdiOutputError> {
        *checked_lock(self.inner.ready.lock())? = true;
        self.inner.ready_signal.notify_all();
        Ok(())
    }

    fn wait_for_startup(&self, stop: &vtuber_core::StopToken) -> Result<bool, NdiOutputError> {
        let mut allowed = checked_lock(self.inner.startup_allowed.lock())?;
        while !*allowed {
            if stop.is_stopped() {
                return Ok(false);
            }
            allowed = self
                .inner
                .startup_allowed_signal
                .wait_timeout(allowed, std::time::Duration::from_millis(50))
                .map_err(|_| {
                    NdiOutputError::new(
                        NdiErrorCode::StatePoisoned,
                        "NDI condition variable mutex poisoned",
                    )
                })?
                .0;
        }
        Ok(true)
    }

    fn send_frame(&self, stop: &vtuber_core::StopToken) -> Result<(), NdiOutputError> {
        {
            *checked_lock(self.inner.send_waiting.lock())? = true;
            self.inner.send_waiting_signal.notify_all();
        }
        let mut allowed = checked_lock(self.inner.send_allowed.lock())?;
        while !*allowed {
            if stop.is_stopped() {
                *checked_lock(self.inner.send_waiting.lock())? = false;
                return Err(NdiOutputError::new(
                    NdiErrorCode::NotRunning,
                    "scripted sender stopped while a send was held",
                ));
            }
            allowed = self
                .inner
                .send_allowed_signal
                .wait_timeout(allowed, std::time::Duration::from_millis(50))
                .map_err(|_| {
                    NdiOutputError::new(
                        NdiErrorCode::StatePoisoned,
                        "NDI condition variable mutex poisoned",
                    )
                })?
                .0;
        }
        *checked_lock(self.inner.send_waiting.lock())? = false;
        self.inner.sent_frames.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

/// Owns at most one sender worker and its bounded latest-frame mailbox.
pub struct NdiOutputController {
    shared: Arc<SharedState>,
    worker: Option<vtuber_core::WorkerHandle<Result<WorkerExit, NdiOutputError>>>,
    backend: ControllerBackend,
}

impl fmt::Debug for NdiOutputController {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NdiOutputController")
            .field("status", &self.status())
            .field("worker_present", &self.worker.is_some())
            .finish()
    }
}

impl Default for NdiOutputController {
    fn default() -> Self {
        Self::new()
    }
}

impl NdiOutputController {
    /// Creates an inactive controller without touching the NDI runtime.
    #[must_use]
    pub fn new() -> Self {
        #[cfg(feature = "ndi-sdk")]
        {
            Self::with_backend(ControllerBackend::Sdk)
        }
        #[cfg(not(feature = "ndi-sdk"))]
        {
            Self::with_backend(ControllerBackend::FeatureDisabled)
        }
    }

    /// Creates a controller that uses a deterministic in-process sender.
    ///
    /// The resulting API is identical to the SDK-backed controller, but no NDI
    /// type or network call is involved. This is the test seam for lifecycle,
    /// mailbox, and orchestration contracts.
    #[must_use]
    pub fn with_scripted_backend(backend: NdiScriptedBackend) -> Self {
        Self::with_backend(ControllerBackend::Scripted(backend))
    }

    fn with_backend(backend: ControllerBackend) -> Self {
        Self {
            shared: Arc::new(SharedState {
                status: Mutex::new(NdiOutputStatus::Off),
                mailbox: Mutex::new(None),
                metrics: MetricsInner::default(),
            }),
            worker: None,
            backend,
        }
    }

    /// Returns whether a sender worker is currently retained.
    #[must_use]
    pub fn has_worker(&self) -> bool {
        self.worker.is_some()
    }

    /// Applies one start/stop command.
    pub fn apply(&mut self, command: NdiOutputCommand) -> Result<(), NdiOutputError> {
        match command {
            NdiOutputCommand::Start(config) => self.start(config),
            NdiOutputCommand::Stop => self.stop(),
        }
    }

    /// Returns the current worker status.
    #[must_use]
    pub fn status(&self) -> NdiOutputStatus {
        let status = self.shared.status();
        if self.is_stopping() && !matches!(status, NdiOutputStatus::Error { .. }) {
            NdiOutputStatus::Off
        } else {
            status
        }
    }

    /// Whether the previous sender is still releasing its native resources.
    #[must_use]
    pub fn is_stopping(&self) -> bool {
        self.worker
            .as_ref()
            .is_some_and(vtuber_core::WorkerHandle::is_stopped)
    }

    /// Requests shutdown without waiting for an SDK call or joining the worker.
    /// A subsequent start must wait until [`Self::poll`] reaps this worker.
    pub fn request_stop(&mut self) -> Result<(), NdiOutputError> {
        let result = (|| {
            if let Some(worker) = &self.worker {
                worker.stop();
            }
            self.close_mailbox()?;
            self.shared.set_status(NdiOutputStatus::Off)
        })();
        if let Err(error) = &result {
            self.shared.replace_status_error(error);
        }
        result
    }

    /// Reaps an already finished worker without blocking the caller.
    pub fn poll(&mut self) -> Result<(), NdiOutputError> {
        self.reap_finished_worker()
    }

    /// Returns a bounded snapshot of sender metrics.
    #[must_use]
    pub fn metrics(&self) -> NdiOutputMetrics {
        self.shared.metrics.snapshot()
    }

    /// Starts one sender worker.
    ///
    /// With the default feature set this transitions to a typed
    /// `NDI_FEATURE_DISABLED` error without spawning a thread. With
    /// `ndi-sdk`, runtime initialization and sender creation occur inside the
    /// worker so no SDK handle crosses the application boundary. `Ok` means the
    /// thread started, not that NDI is live; inspect status for the worker result.
    ///
    /// # Errors
    /// Reports an already-running sender, invalid configuration, disabled SDK,
    /// thread-spawn failure, or a panic reaped from the preceding worker. Native
    /// initialization/sender-creation failures are reported in worker status.
    pub fn start(&mut self, config: NdiOutputConfig) -> Result<(), NdiOutputError> {
        self.reap_finished_worker()?;
        if self.worker.is_some() {
            return Err(NdiOutputError::new(
                NdiErrorCode::AlreadyRunning,
                "NDI output is already starting or live",
            ));
        }
        if !matches!(
            self.status(),
            NdiOutputStatus::Off | NdiOutputStatus::Error { .. }
        ) {
            return Err(NdiOutputError::new(
                NdiErrorCode::AlreadyRunning,
                "NDI output is already starting or live",
            ));
        }
        if let Err(error) = config.validate() {
            self.shared.replace_status_error(&error);
            return Err(error);
        }
        self.shared.metrics.reset();
        self.shared.set_status(NdiOutputStatus::Starting)?;
        let mailbox = Arc::new(LatestFrameMailbox::new());
        match checked_lock(self.shared.mailbox.lock()) {
            Ok(mut slot) => *slot = Some(Arc::clone(&mailbox)),
            Err(error) => {
                self.shared.replace_status_error(&error);
                return Err(error);
            }
        }

        match &self.backend {
            #[cfg(not(feature = "ndi-sdk"))]
            ControllerBackend::FeatureDisabled => {
                mailbox.close()?;
                *checked_lock(self.shared.mailbox.lock())? = None;
                let error = NdiOutputError::new(
                    NdiErrorCode::FeatureDisabled,
                    "NDI output was not enabled for this build",
                );
                self.shared
                    .metrics
                    .start_failures
                    .fetch_add(1, Ordering::Relaxed);
                self.shared.replace_status_error(&error);
                Err(error)
            }
            ControllerBackend::Scripted(backend) => {
                let shared = Arc::clone(&self.shared);
                let backend = backend.clone();
                let spawned = vtuber_core::WorkerHandle::spawn("ndi-output-sender", move |stop| {
                    let result =
                        run_scripted_worker(Arc::clone(&shared), mailbox, config, backend, stop);
                    if let Err(error) = &result {
                        shared.replace_status_error(error);
                    }
                    result
                });
                match spawned {
                    Ok(worker) => {
                        self.worker = Some(worker);
                        Ok(())
                    }
                    Err(error) => Err(self.finish_spawn_failure(error)),
                }
            }
            #[cfg(feature = "ndi-sdk")]
            ControllerBackend::Sdk => {
                let shared = Arc::clone(&self.shared);
                let spawned = vtuber_core::WorkerHandle::spawn("ndi-output-sender", move |stop| {
                    let result = run_ndi_worker(Arc::clone(&shared), mailbox, config, stop);
                    if let Err(error) = &result {
                        shared.replace_status_error(error);
                    }
                    result
                });
                match spawned {
                    Ok(worker) => {
                        self.worker = Some(worker);
                        Ok(())
                    }
                    Err(error) => Err(self.finish_spawn_failure(error)),
                }
            }
        }
    }

    /// Stops and joins the sender worker. The operation is idempotent.
    ///
    /// Blocks until the worker exits, closes the mailbox and discards its pending
    /// frame. UI callers should use `request_stop` and `poll` instead.
    ///
    /// # Errors
    /// Returns `WorkerStopFailed` for a worker panic; mailbox/worker resources are
    /// released even on that error.
    pub fn stop(&mut self) -> Result<(), NdiOutputError> {
        let cleanup = self.request_stop();
        let joined = self.worker.take().map(|worker| {
            worker.stop();
            worker.join()
        });
        let result = self.finish_join(joined, true).and(cleanup);
        if let Err(error) = &result {
            self.shared.replace_status_error(error);
        }
        result
    }

    /// Submits a frame without waiting for the network sender.
    ///
    /// Returns `Submitted` for a new pending frame, `Replaced` when replacing
    /// one, or `RejectedNotRunning` when no accepting mailbox is available.
    /// Acceptance is not delivery: the worker validates the frame's output
    /// profile and performs the send later.
    pub fn submit_frame(&self, frame: VideoOutputFrame) -> Result<NdiSubmitResult, NdiOutputError> {
        let result = (|| {
            let status = self.status();
            if let NdiOutputStatus::Error {
                code: NdiErrorCode::StatePoisoned,
                message,
            } = status
            {
                return Err(NdiOutputError::new(NdiErrorCode::StatePoisoned, message));
            }
            if !matches!(
                status,
                NdiOutputStatus::Starting | NdiOutputStatus::Live { .. }
            ) {
                return Ok(NdiSubmitResult::RejectedNotRunning);
            }
            let result = checked_lock(self.shared.mailbox.lock())?
                .as_ref()
                .map_or(Ok(NdiSubmitResult::RejectedNotRunning), |mailbox| {
                    mailbox.submit(frame)
                })?;
            match result {
                NdiSubmitResult::Submitted => {
                    self.shared
                        .metrics
                        .submitted_frames
                        .fetch_add(1, Ordering::Relaxed);
                }
                NdiSubmitResult::Replaced => {
                    self.shared
                        .metrics
                        .submitted_frames
                        .fetch_add(1, Ordering::Relaxed);
                    self.shared
                        .metrics
                        .replaced_frames
                        .fetch_add(1, Ordering::Relaxed);
                }
                NdiSubmitResult::RejectedNotRunning => {}
            }
            Ok(result)
        })();
        if let Err(error) = &result {
            if let Some(worker) = &self.worker {
                worker.stop();
            }
            self.shared.replace_status_error(error);
        }
        result
    }

    /// Rolls the controller back after the OS refused a worker spawn: the
    /// mailbox taken from the shared slot is closed and cleared, the failure
    /// is counted, and the Starting status is replaced by the typed error.
    fn finish_spawn_failure(&self, spawn_error: std::io::Error) -> NdiOutputError {
        let cleanup = self.close_mailbox();
        self.shared
            .metrics
            .start_failures
            .fetch_add(1, Ordering::Relaxed);
        let error = cleanup.err().unwrap_or_else(|| {
            NdiOutputError::new(
                NdiErrorCode::WorkerSpawnFailed,
                format!("NDI sender worker could not be started: {spawn_error}"),
            )
        });
        self.shared.replace_status_error(&error);
        error
    }

    fn close_mailbox(&self) -> Result<(), NdiOutputError> {
        if let Some(mailbox) = checked_lock(self.shared.mailbox.lock())?.take()
            && mailbox.close()?
        {
            self.shared
                .metrics
                .dropped_frames
                .fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    fn finish_join(
        &self,
        joined: Option<vtuber_core::WorkerResult<Result<WorkerExit, NdiOutputError>>>,
        stopping: bool,
    ) -> Result<(), NdiOutputError> {
        match joined {
            Some(vtuber_core::WorkerResult::Completed(Err(error))) => Err(error),
            Some(vtuber_core::WorkerResult::Panicked) => Err(NdiOutputError::new(
                NdiErrorCode::WorkerStopFailed,
                "NDI sender worker did not join cleanly",
            )),
            Some(vtuber_core::WorkerResult::Completed(Ok(WorkerExit::StartupFailed)))
                if !stopping =>
            {
                Ok(())
            }
            _ => self.shared.set_status(NdiOutputStatus::Off),
        }
    }

    fn reap_finished_worker(&mut self) -> Result<(), NdiOutputError> {
        if !self
            .worker
            .as_ref()
            .is_some_and(vtuber_core::WorkerHandle::is_finished)
        {
            return Ok(());
        }
        let Some(worker) = self.worker.take() else {
            return Ok(());
        };
        let cleanup = self.close_mailbox();
        let stopping = worker.is_stopped();
        let result = self.finish_join(Some(worker.join()), stopping).and(cleanup);
        if let Err(error) = &result {
            self.shared.replace_status_error(error);
        }
        result
    }
}

impl Drop for NdiOutputController {
    fn drop(&mut self) {
        let _ = self.request_stop();
        // Native teardown belongs to the worker. A stuck SDK must not hold the
        // application event loop (including exit) hostage. No replacement is
        // spawned while this controller retains a stopping worker.
    }
}

#[derive(Debug)]
struct MetricsInner {
    submitted_frames: AtomicU64,
    sent_frames: AtomicU64,
    replaced_frames: AtomicU64,
    dropped_frames: AtomicU64,
    rejected_frames: AtomicU64,
    start_failures: AtomicU64,
    last_frame_seq: AtomicU64,
}

impl Default for MetricsInner {
    fn default() -> Self {
        Self {
            submitted_frames: AtomicU64::new(0),
            sent_frames: AtomicU64::new(0),
            replaced_frames: AtomicU64::new(0),
            dropped_frames: AtomicU64::new(0),
            rejected_frames: AtomicU64::new(0),
            start_failures: AtomicU64::new(0),
            last_frame_seq: AtomicU64::new(u64::MAX),
        }
    }
}

impl MetricsInner {
    fn reset(&self) {
        for counter in [
            &self.submitted_frames,
            &self.sent_frames,
            &self.replaced_frames,
            &self.dropped_frames,
            &self.rejected_frames,
            &self.start_failures,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
        self.last_frame_seq.store(u64::MAX, Ordering::Relaxed);
    }

    fn snapshot(&self) -> NdiOutputMetrics {
        let last_frame_seq = match self.last_frame_seq.load(Ordering::Relaxed) {
            u64::MAX => None,
            seq => Some(FrameSeq(seq)),
        };
        NdiOutputMetrics {
            submitted_frames: self.submitted_frames.load(Ordering::Relaxed),
            sent_frames: self.sent_frames.load(Ordering::Relaxed),
            replaced_frames: self.replaced_frames.load(Ordering::Relaxed),
            dropped_frames: self.dropped_frames.load(Ordering::Relaxed),
            rejected_frames: self.rejected_frames.load(Ordering::Relaxed),
            start_failures: self.start_failures.load(Ordering::Relaxed),
            last_frame_seq,
        }
    }
}

#[derive(Debug)]
struct MailboxState {
    latest: Option<VideoOutputFrame>,
    closed: bool,
}

#[derive(Debug)]
struct LatestFrameMailbox {
    state: Mutex<MailboxState>,
    available: Condvar,
}

impl LatestFrameMailbox {
    fn new() -> Self {
        Self {
            state: Mutex::new(MailboxState {
                latest: None,
                closed: false,
            }),
            available: Condvar::new(),
        }
    }

    fn submit(&self, frame: VideoOutputFrame) -> Result<NdiSubmitResult, NdiOutputError> {
        let mut state = checked_lock(self.state.lock())?;
        if state.closed {
            return Ok(NdiSubmitResult::RejectedNotRunning);
        }
        let result = if state.latest.replace(frame).is_some() {
            NdiSubmitResult::Replaced
        } else {
            NdiSubmitResult::Submitted
        };
        self.available.notify_one();
        Ok(result)
    }

    fn take(
        &self,
        stop_requested: impl Fn() -> bool,
    ) -> Result<Option<VideoOutputFrame>, NdiOutputError> {
        let mut state = checked_lock(self.state.lock())?;
        loop {
            if let Some(frame) = state.latest.take() {
                return Ok(Some(frame));
            }
            if state.closed || stop_requested() {
                return Ok(None);
            }
            state = self
                .available
                .wait_timeout(state, std::time::Duration::from_millis(50))
                .map_err(|_| {
                    NdiOutputError::new(
                        NdiErrorCode::StatePoisoned,
                        "NDI condition variable mutex poisoned",
                    )
                })?
                .0;
        }
    }

    fn close(&self) -> Result<bool, NdiOutputError> {
        let mut state = checked_lock(self.state.lock())?;
        state.closed = true;
        let discarded = state.latest.take().is_some();
        self.available.notify_all();
        Ok(discarded)
    }
}

fn checked_lock<T>(
    result: std::sync::LockResult<MutexGuard<'_, T>>,
) -> Result<MutexGuard<'_, T>, NdiOutputError> {
    result.map_err(|_| {
        NdiOutputError::new(
            NdiErrorCode::StatePoisoned,
            "NDI shared state mutex poisoned",
        )
    })
}

#[derive(Debug)]
struct SharedState {
    status: Mutex<NdiOutputStatus>,
    mailbox: Mutex<Option<Arc<LatestFrameMailbox>>>,
    metrics: MetricsInner,
}

impl SharedState {
    fn status(&self) -> NdiOutputStatus {
        match checked_lock(self.status.lock()) {
            Ok(status) => status.clone(),
            Err(error) => NdiOutputStatus::Error {
                code: error.code,
                message: error.message,
            },
        }
    }

    fn set_status(&self, status: NdiOutputStatus) -> Result<(), NdiOutputError> {
        *checked_lock(self.status.lock())? = status;
        Ok(())
    }

    fn replace_status_error(&self, error: &NdiOutputError) {
        let _ = self.set_status(NdiOutputStatus::Error {
            code: error.code,
            message: error.message.clone(),
        });
    }
}

#[derive(Debug)]
struct ScriptedInner {
    init_error: Option<NdiErrorCode>,
    create_error: Option<NdiErrorCode>,
    startup_allowed: Mutex<bool>,
    startup_allowed_signal: Condvar,
    ready: Mutex<bool>,
    ready_signal: Condvar,
    send_allowed: Mutex<bool>,
    send_allowed_signal: Condvar,
    send_waiting: Mutex<bool>,
    send_waiting_signal: Condvar,
    connections: std::sync::atomic::AtomicU32,
    sent_frames: AtomicU64,
    live_senders: AtomicU64,
}

struct ScriptedSenderGuard {
    backend: NdiScriptedBackend,
}

impl Drop for ScriptedSenderGuard {
    fn drop(&mut self) {
        self.backend
            .inner
            .live_senders
            .fetch_sub(1, Ordering::Relaxed);
    }
}

#[derive(Clone, Debug)]
enum ControllerBackend {
    #[cfg(not(feature = "ndi-sdk"))]
    FeatureDisabled,
    Scripted(NdiScriptedBackend),
    #[cfg(feature = "ndi-sdk")]
    Sdk,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorkerExit {
    Stopped,
    StartupFailed,
}

#[cfg(test)]
mod tests;
