// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]
//! Safe, optional NDI video-output boundary.
//!
//! The public API contains only application-owned configuration, status,
//! metrics, and [`vtuber_core::VideoOutputFrame`]. NDI SDK types and the
//! binding feature remain private to this crate. The default build is a
//! deterministic feature-disabled stub so the rest of the workspace does not
//! require an installed NDI SDK.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use vtuber_core::{FrameSeq, VideoOutputFrame, VideoOutputPixelFormat, VideoOutputProfile};

/// Returns whether this build includes the explicit NDI SDK backend.
#[must_use]
pub const fn is_sdk_feature_enabled() -> bool {
    cfg!(feature = "ndi-sdk")
}

/// Stable error codes emitted by the optional output backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NdiErrorCode {
    /// The SDK feature was not enabled in this build.
    FeatureDisabled,
    /// The runtime library could not be found by the operating system.
    RuntimeNotFound,
    /// The NDI runtime failed to initialize.
    RuntimeInitFailed,
    /// The named sender could not be created.
    SenderCreateFailed,
    /// A frame could not be submitted to the sender.
    SendFailed,
    /// A worker could not be stopped and joined cleanly.
    WorkerStopFailed,
    /// A second start was requested while the sender was active.
    AlreadyRunning,
    /// The output configuration is invalid.
    InvalidConfiguration,
    /// A frame did not satisfy the fixed BGRA output contract.
    InvalidFrame,
    /// A frame was submitted while the sender was not active.
    NotRunning,
}

impl NdiErrorCode {
    /// Returns the stable machine-readable error code.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FeatureDisabled => "NDI_FEATURE_DISABLED",
            Self::RuntimeNotFound => "NDI_RUNTIME_NOT_FOUND",
            Self::RuntimeInitFailed => "NDI_RUNTIME_INIT_FAILED",
            Self::SenderCreateFailed => "NDI_SENDER_CREATE_FAILED",
            Self::SendFailed => "NDI_SEND_FAILED",
            Self::WorkerStopFailed => "NDI_WORKER_STOP_FAILED",
            Self::AlreadyRunning => "NDI_ALREADY_RUNNING",
            Self::InvalidConfiguration => "NDI_INVALID_CONFIGURATION",
            Self::InvalidFrame => "NDI_INVALID_FRAME",
            Self::NotRunning => "NDI_NOT_RUNNING",
        }
    }
}

/// A stable, user-safe backend error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NdiOutputError {
    /// Stable error classification.
    pub code: NdiErrorCode,
    /// Short diagnostic message without local paths or SDK handles.
    pub message: String,
}

impl NdiOutputError {
    fn new(code: NdiErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl fmt::Display for NdiOutputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code.as_str(), self.message)
    }
}

impl std::error::Error for NdiOutputError {}

/// Runtime state visible to the application/UI layer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum NdiOutputStatus {
    /// No sender worker exists.
    #[default]
    Off,
    /// The worker is initializing the runtime and named sender.
    Starting,
    /// The sender is publishing frames.
    Live {
        /// Number of currently connected receivers at the last worker poll.
        connections: u32,
        /// Requested stable source name.
        source_name: String,
    },
    /// The sender stopped because of a recoverable failure.
    Error {
        /// Stable error classification.
        code: NdiErrorCode,
        /// User-safe diagnostic detail.
        message: String,
    },
}

/// Configuration for one named transparent video sender.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NdiOutputConfig {
    /// UTF-8 source name shown to NDI finders and OBS.
    pub source_name: String,
    /// Fixed output dimensions and frame rate.
    pub profile: VideoOutputProfile,
}

impl Default for NdiOutputConfig {
    fn default() -> Self {
        Self {
            source_name: "RusTuberV".to_owned(),
            profile: VideoOutputProfile::DEFAULT,
        }
    }
}

impl NdiOutputConfig {
    fn validate(&self) -> Result<(), NdiOutputError> {
        if self.source_name.trim().is_empty()
            || self.source_name.chars().any(char::is_control)
            || self.source_name.contains('\0')
        {
            return Err(NdiOutputError::new(
                NdiErrorCode::InvalidConfiguration,
                "source name must be non-empty UTF-8 without control characters",
            ));
        }
        if self.profile.width == 0 || self.profile.height == 0 || self.profile.fps == 0 {
            return Err(NdiOutputError::new(
                NdiErrorCode::InvalidConfiguration,
                "output width, height, and fps must be non-zero",
            ));
        }
        if self.profile.pixel_format != VideoOutputPixelFormat::Bgra8StraightAlpha {
            return Err(NdiOutputError::new(
                NdiErrorCode::InvalidConfiguration,
                "only BGRA8 straight-alpha output is supported",
            ));
        }
        Ok(())
    }
}

/// Commands understood by the backend controller.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NdiOutputCommand {
    /// Start a sender with the supplied source name and profile.
    Start(NdiOutputConfig),
    /// Stop the current sender; stopping an already-off sender is safe.
    Stop,
}

/// Result of attempting to submit a frame to the bounded mailbox.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NdiSubmitResult {
    /// The mailbox was empty and now contains the frame.
    Submitted,
    /// A pending frame was replaced by this newer frame.
    Replaced,
    /// The sender is not active or is shutting down.
    RejectedNotRunning,
}

/// A transport-neutral description of the NDI High Bandwidth video mapping.
///
/// This type intentionally uses no NDI SDK enum. It can be tested in the
/// normal SDK-free build and is the only descriptor passed into the optional
/// binding adapter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NdiVideoFrameMapping {
    /// Frame width in pixels.
    pub width: i32,
    /// Frame height in pixels.
    pub height: i32,
    /// Validated BGRA row stride in bytes.
    pub stride_bytes: i32,
    /// Frame-rate numerator.
    pub frame_rate_n: i32,
    /// Frame-rate denominator.
    pub frame_rate_d: i32,
    /// Square-pixel picture aspect ratio.
    pub picture_aspect_ratio: f32,
    /// Standard NDI FourCC selected by this mapping.
    pub four_cc: NdiFourCc,
}

/// FourCC values exposed by the transport-neutral mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NdiFourCc {
    /// NDI BGRA with a preserved alpha byte.
    Bgra,
}

/// Validates the #46 frame contract and maps it to standard NDI video fields.
pub fn map_video_frame(
    frame: &VideoOutputFrame,
    profile: VideoOutputProfile,
) -> Result<NdiVideoFrameMapping, NdiOutputError> {
    if frame.pixel_format != VideoOutputPixelFormat::Bgra8StraightAlpha {
        return Err(NdiOutputError::new(
            NdiErrorCode::InvalidFrame,
            "frame pixel format is not BGRA8 straight alpha",
        ));
    }
    let stride = profile.packed_stride_bytes();
    let expected_len = stride
        .checked_mul(profile.height as usize)
        .ok_or_else(|| NdiOutputError::new(NdiErrorCode::InvalidFrame, "frame size overflow"))?;
    if frame.width != profile.width
        || frame.height != profile.height
        || frame.stride_bytes != stride
        || frame.data.len() != expected_len
    {
        return Err(NdiOutputError::new(
            NdiErrorCode::InvalidFrame,
            "frame dimensions, stride, or data length do not match the output profile",
        ));
    }
    let width = i32::try_from(frame.width).map_err(|_| {
        NdiOutputError::new(NdiErrorCode::InvalidFrame, "frame width exceeds NDI range")
    })?;
    let height = i32::try_from(frame.height).map_err(|_| {
        NdiOutputError::new(NdiErrorCode::InvalidFrame, "frame height exceeds NDI range")
    })?;
    let stride_bytes = i32::try_from(frame.stride_bytes).map_err(|_| {
        NdiOutputError::new(NdiErrorCode::InvalidFrame, "frame stride exceeds NDI range")
    })?;
    let frame_rate_n = i32::try_from(profile.fps).map_err(|_| {
        NdiOutputError::new(NdiErrorCode::InvalidFrame, "frame rate exceeds NDI range")
    })?;
    Ok(NdiVideoFrameMapping {
        width,
        height,
        stride_bytes,
        frame_rate_n,
        frame_rate_d: 1,
        picture_aspect_ratio: profile.width as f32 / profile.height as f32,
        four_cc: NdiFourCc::Bgra,
    })
}

/// Bounded counters collected by the sender boundary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NdiOutputMetrics {
    /// Frames accepted into the latest-value mailbox.
    pub submitted_frames: u64,
    /// Frames successfully handed to the SDK sender.
    pub sent_frames: u64,
    /// Pending frames replaced by a newer frame.
    pub replaced_frames: u64,
    /// Frames rejected or discarded during shutdown.
    pub dropped_frames: u64,
    /// Malformed frames rejected before an SDK call.
    pub rejected_frames: u64,
    /// Sender/runtime initialization failures.
    pub start_failures: u64,
    /// Most recent frame sequence successfully handed to the SDK.
    pub last_frame_seq: Option<FrameSeq>,
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

    fn submit(&self, frame: VideoOutputFrame) -> NdiSubmitResult {
        let mut state = recover_lock(self.state.lock());
        if state.closed {
            return NdiSubmitResult::RejectedNotRunning;
        }
        let result = if state.latest.replace(frame).is_some() {
            NdiSubmitResult::Replaced
        } else {
            NdiSubmitResult::Submitted
        };
        self.available.notify_one();
        result
    }

    fn take(&self, stop_requested: impl Fn() -> bool) -> Option<VideoOutputFrame> {
        let mut state = recover_lock(self.state.lock());
        loop {
            if let Some(frame) = state.latest.take() {
                return Some(frame);
            }
            if state.closed || stop_requested() {
                return None;
            }
            state = self
                .available
                .wait_timeout(state, std::time::Duration::from_millis(50))
                .map_or_else(|poisoned| poisoned.into_inner().0, |result| result.0);
        }
    }

    fn close(&self) -> bool {
        let mut state = recover_lock(self.state.lock());
        state.closed = true;
        let discarded = state.latest.take().is_some();
        self.available.notify_all();
        discarded
    }
}

fn recover_lock<T>(result: std::sync::LockResult<MutexGuard<'_, T>>) -> MutexGuard<'_, T> {
    result.unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(Debug)]
struct SharedState {
    status: Mutex<NdiOutputStatus>,
    mailbox: Mutex<Option<Arc<LatestFrameMailbox>>>,
    metrics: MetricsInner,
}

impl SharedState {
    fn status(&self) -> NdiOutputStatus {
        recover_lock(self.status.lock()).clone()
    }

    fn set_status(&self, status: NdiOutputStatus) {
        *recover_lock(self.status.lock()) = status;
    }

    fn replace_status_error(&self, error: &NdiOutputError) {
        self.set_status(NdiOutputStatus::Error {
            code: error.code,
            message: error.message.clone(),
        });
    }
}

/// Deterministic in-process sender used by tests and orchestration harnesses.
///
/// This backend never loads the NDI SDK or performs network I/O. Production
/// [`NdiOutputController::new`] does not use it.
#[derive(Clone, Debug)]
pub struct NdiScriptedBackend {
    inner: Arc<ScriptedInner>,
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
    pub fn hold_startup(&self) {
        *recover_lock(self.inner.startup_allowed.lock()) = false;
    }

    /// Allows a held startup to continue.
    pub fn release_startup(&self) {
        *recover_lock(self.inner.startup_allowed.lock()) = true;
        self.inner.startup_allowed_signal.notify_all();
    }

    /// Blocks [`Self`] send calls until [`Self::release_send`].
    pub fn hold_send(&self) {
        *recover_lock(self.inner.send_allowed.lock()) = false;
    }

    /// Releases a blocked send so the worker can continue.
    pub fn release_send(&self) {
        *recover_lock(self.inner.send_allowed.lock()) = true;
        self.inner.send_allowed_signal.notify_all();
    }

    /// Waits until the worker is blocked inside a send call.
    pub fn wait_until_send_blocked(&self) {
        let mut waiting = recover_lock(self.inner.send_waiting.lock());
        while !*waiting {
            waiting = recover_lock(self.inner.send_waiting_signal.wait(waiting));
        }
    }

    /// Waits until the worker has published Live or Error.
    pub fn wait_until_ready(&self) {
        let mut ready = recover_lock(self.inner.ready.lock());
        while !*ready {
            ready = recover_lock(self.inner.ready_signal.wait(ready));
        }
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

    fn mark_ready(&self) {
        *recover_lock(self.inner.ready.lock()) = true;
        self.inner.ready_signal.notify_all();
    }

    fn wait_for_startup(&self, stop: &vtuber_core::StopToken) -> bool {
        let mut allowed = recover_lock(self.inner.startup_allowed.lock());
        while !*allowed {
            if stop.is_stopped() {
                return false;
            }
            allowed = self
                .inner
                .startup_allowed_signal
                .wait_timeout(allowed, std::time::Duration::from_millis(50))
                .map_or_else(|poisoned| poisoned.into_inner().0, |result| result.0);
        }
        true
    }

    fn send_frame(&self, stop: &vtuber_core::StopToken) -> Result<(), NdiOutputError> {
        {
            *recover_lock(self.inner.send_waiting.lock()) = true;
            self.inner.send_waiting_signal.notify_all();
        }
        let mut allowed = recover_lock(self.inner.send_allowed.lock());
        while !*allowed {
            if stop.is_stopped() {
                *recover_lock(self.inner.send_waiting.lock()) = false;
                return Err(NdiOutputError::new(
                    NdiErrorCode::NotRunning,
                    "scripted sender stopped while a send was held",
                ));
            }
            allowed = self
                .inner
                .send_allowed_signal
                .wait_timeout(allowed, std::time::Duration::from_millis(50))
                .map_or_else(|poisoned| poisoned.into_inner().0, |result| result.0);
        }
        *recover_lock(self.inner.send_waiting.lock()) = false;
        self.inner.sent_frames.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
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

/// Owns at most one sender worker and its bounded latest-frame mailbox.
pub struct NdiOutputController {
    shared: Arc<SharedState>,
    worker: Option<vtuber_core::WorkerHandle<WorkerExit>>,
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
        self.shared.status()
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
    /// worker so no SDK handle crosses the application boundary.
    pub fn start(&mut self, config: NdiOutputConfig) -> Result<(), NdiOutputError> {
        self.reap_finished_worker();
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
        self.shared.set_status(NdiOutputStatus::Starting);
        let mailbox = Arc::new(LatestFrameMailbox::new());
        *recover_lock(self.shared.mailbox.lock()) = Some(Arc::clone(&mailbox));

        match &self.backend {
            #[cfg(not(feature = "ndi-sdk"))]
            ControllerBackend::FeatureDisabled => {
                mailbox.close();
                *recover_lock(self.shared.mailbox.lock()) = None;
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
                self.worker = Some(vtuber_core::WorkerHandle::spawn(
                    "ndi-output-sender",
                    move |stop| run_scripted_worker(shared, mailbox, config, backend, stop),
                ));
                Ok(())
            }
            #[cfg(feature = "ndi-sdk")]
            ControllerBackend::Sdk => {
                let shared = Arc::clone(&self.shared);
                self.worker = Some(vtuber_core::WorkerHandle::spawn(
                    "ndi-output-sender",
                    move |stop| run_ndi_worker(shared, mailbox, config, stop),
                ));
                Ok(())
            }
        }
    }

    /// Stops and joins the sender worker. The operation is idempotent.
    pub fn stop(&mut self) -> Result<(), NdiOutputError> {
        let mailbox = recover_lock(self.shared.mailbox.lock()).take();
        if let Some(mailbox) = mailbox
            && mailbox.close()
        {
            self.shared
                .metrics
                .dropped_frames
                .fetch_add(1, Ordering::Relaxed);
        }
        let Some(worker) = self.worker.take() else {
            self.shared.set_status(NdiOutputStatus::Off);
            return Ok(());
        };
        worker.stop();
        match worker.join() {
            vtuber_core::WorkerResult::Completed(WorkerExit::Stopped)
            | vtuber_core::WorkerResult::Completed(WorkerExit::StartupFailed) => {
                self.shared.set_status(NdiOutputStatus::Off);
                Ok(())
            }
            vtuber_core::WorkerResult::Panicked | vtuber_core::WorkerResult::SpawnFailed => {
                let error = NdiOutputError::new(
                    NdiErrorCode::WorkerStopFailed,
                    "NDI sender worker did not join cleanly",
                );
                self.shared.replace_status_error(&error);
                Err(error)
            }
        }
    }

    /// Submits a frame without waiting for the network sender.
    pub fn submit_frame(&self, frame: VideoOutputFrame) -> NdiSubmitResult {
        let status = self.status();
        if !matches!(
            status,
            NdiOutputStatus::Starting | NdiOutputStatus::Live { .. }
        ) {
            return NdiSubmitResult::RejectedNotRunning;
        }
        let result = recover_lock(self.shared.mailbox.lock())
            .as_ref()
            .map_or(NdiSubmitResult::RejectedNotRunning, |mailbox| {
                mailbox.submit(frame)
            });
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
        result
    }

    fn reap_finished_worker(&mut self) {
        let finished = self
            .worker
            .as_ref()
            .is_some_and(vtuber_core::WorkerHandle::is_finished);
        if finished {
            // Invariant: `is_finished()` returned true immediately above.
            #[allow(clippy::expect_used)]
            let worker = self.worker.take().expect("finished worker exists");
            let _ = worker.join();
            *recover_lock(self.shared.mailbox.lock()) = None;
        }
    }
}

impl Drop for NdiOutputController {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorkerExit {
    Stopped,
    StartupFailed,
}

fn run_scripted_worker(
    shared: Arc<SharedState>,
    mailbox: Arc<LatestFrameMailbox>,
    config: NdiOutputConfig,
    backend: NdiScriptedBackend,
    stop: vtuber_core::StopToken,
) -> WorkerExit {
    if !backend.wait_for_startup(&stop) {
        backend.mark_ready();
        return WorkerExit::Stopped;
    }
    if let Some(code) = backend.inner.init_error {
        let error = NdiOutputError::new(code, "NDI runtime could not be initialized");
        shared
            .metrics
            .start_failures
            .fetch_add(1, Ordering::Relaxed);
        shared.replace_status_error(&error);
        backend.mark_ready();
        return WorkerExit::StartupFailed;
    }
    if let Some(code) = backend.inner.create_error {
        let error = NdiOutputError::new(code, "could not create NDI sender");
        shared
            .metrics
            .start_failures
            .fetch_add(1, Ordering::Relaxed);
        shared.replace_status_error(&error);
        backend.mark_ready();
        return WorkerExit::StartupFailed;
    }
    backend.inner.live_senders.fetch_add(1, Ordering::Relaxed);
    let _sender_guard = ScriptedSenderGuard {
        backend: backend.clone(),
    };
    shared.set_status(NdiOutputStatus::Live {
        connections: backend.inner.connections.load(Ordering::Relaxed),
        source_name: config.source_name.clone(),
    });
    backend.mark_ready();
    loop {
        let Some(frame) = mailbox.take(|| stop.is_stopped()) else {
            return WorkerExit::Stopped;
        };
        let mapping = match map_video_frame(&frame, config.profile) {
            Ok(mapping) => mapping,
            Err(_) => {
                shared
                    .metrics
                    .rejected_frames
                    .fetch_add(1, Ordering::Relaxed);
                continue;
            }
        };
        let _ = mapping;
        if backend.send_frame(&stop).is_err() {
            return WorkerExit::Stopped;
        }
        shared.metrics.sent_frames.fetch_add(1, Ordering::Relaxed);
        shared
            .metrics
            .last_frame_seq
            .store(frame.frame_seq.0, Ordering::Relaxed);
        shared.set_status(NdiOutputStatus::Live {
            connections: backend.inner.connections.load(Ordering::Relaxed),
            source_name: config.source_name.clone(),
        });
    }
}

#[cfg(feature = "ndi-sdk")]
fn run_ndi_worker(
    shared: Arc<SharedState>,
    mailbox: Arc<LatestFrameMailbox>,
    config: NdiOutputConfig,
    stop: vtuber_core::StopToken,
) -> WorkerExit {
    use grafton_ndi::{NDI, PixelFormat, ScanType, Sender, SenderOptions, VideoFrame};

    let ndi = match NDI::new() {
        Ok(ndi) => ndi,
        Err(error) => {
            let mapped = map_runtime_error(error.to_string());
            shared
                .metrics
                .start_failures
                .fetch_add(1, Ordering::Relaxed);
            shared.replace_status_error(&mapped);
            return WorkerExit::StartupFailed;
        }
    };
    let options = SenderOptions::builder(config.source_name.clone())
        .clock_video(true)
        .clock_audio(false)
        .build();
    let sender = match Sender::new(&ndi, &options) {
        Ok(sender) => sender,
        Err(_error) => {
            let mapped = NdiOutputError::new(
                NdiErrorCode::SenderCreateFailed,
                "could not create NDI sender",
            );
            shared
                .metrics
                .start_failures
                .fetch_add(1, Ordering::Relaxed);
            shared.replace_status_error(&mapped);
            return WorkerExit::StartupFailed;
        }
    };
    shared.set_status(NdiOutputStatus::Live {
        connections: 0,
        source_name: config.source_name.clone(),
    });
    let mut last_connection_poll = std::time::Instant::now();
    loop {
        let Some(frame) = mailbox.take(|| stop.is_stopped()) else {
            return WorkerExit::Stopped;
        };
        let mapping = match map_video_frame(&frame, config.profile) {
            Ok(mapping) => mapping,
            Err(_) => {
                shared
                    .metrics
                    .rejected_frames
                    .fetch_add(1, Ordering::Relaxed);
                continue;
            }
        };
        let mut ndi_frame = match VideoFrame::builder()
            .resolution(mapping.width, mapping.height)
            .pixel_format(PixelFormat::BGRA)
            .frame_rate(mapping.frame_rate_n, mapping.frame_rate_d)
            .aspect_ratio(mapping.picture_aspect_ratio)
            .scan_type(ScanType::Progressive)
            .build()
        {
            Ok(frame) => frame,
            Err(_error) => {
                let mapped = NdiOutputError::new(
                    NdiErrorCode::SendFailed,
                    "NDI rejected the validated BGRA frame",
                );
                shared.replace_status_error(&mapped);
                return WorkerExit::StartupFailed;
            }
        };
        if ndi_frame.replace_data(frame.data.to_vec()).is_err() {
            let error = NdiOutputError::new(
                NdiErrorCode::SendFailed,
                "NDI frame storage rejected the validated BGRA frame",
            );
            shared.replace_status_error(&error);
            return WorkerExit::StartupFailed;
        }
        sender.send_video(&ndi_frame);
        shared.metrics.sent_frames.fetch_add(1, Ordering::Relaxed);
        shared
            .metrics
            .last_frame_seq
            .store(frame.frame_seq.0, Ordering::Relaxed);
        if last_connection_poll.elapsed() >= std::time::Duration::from_millis(500) {
            if let Ok(connections) = sender.connection_count(std::time::Duration::from_millis(10)) {
                shared.set_status(NdiOutputStatus::Live {
                    connections,
                    source_name: config.source_name.clone(),
                });
            }
            last_connection_poll = std::time::Instant::now();
        }
    }
}

#[cfg(feature = "ndi-sdk")]
fn map_runtime_error(message: String) -> NdiOutputError {
    let lower = message.to_ascii_lowercase();
    let code = if lower.contains("not found")
        || lower.contains("load")
        || lower.contains("library")
        || lower.contains("dll")
    {
        NdiErrorCode::RuntimeNotFound
    } else {
        NdiErrorCode::RuntimeInitFailed
    };
    NdiOutputError::new(code, "NDI runtime could not be initialized")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn frame(seq: u64) -> VideoOutputFrame {
        VideoOutputFrame::new_bgra8(
            2,
            1,
            FrameSeq(seq),
            vtuber_core::MonoTimeNs(seq),
            vec![1, 2, 3, 4, 5, 6, 7, 8],
        )
        .expect("test frame has a valid packed shape")
    }

    fn test_profile() -> VideoOutputProfile {
        VideoOutputProfile {
            width: 2,
            height: 1,
            fps: 60,
            pixel_format: VideoOutputPixelFormat::Bgra8StraightAlpha,
        }
    }

    fn test_config() -> NdiOutputConfig {
        NdiOutputConfig {
            source_name: "RusTuberV".to_owned(),
            profile: test_profile(),
        }
    }

    #[test]
    fn mapping_preserves_bgra_alpha_and_profile_fields() {
        let source = frame(7);
        let source_bytes = source.data.clone();
        let mapping = map_video_frame(&source, test_profile()).expect("valid frame maps");
        assert_eq!(mapping.width, 2);
        assert_eq!(mapping.height, 1);
        assert_eq!(mapping.four_cc, NdiFourCc::Bgra);
        assert_eq!(mapping.stride_bytes, 8);
        assert_eq!(mapping.frame_rate_n, 60);
        assert_eq!(mapping.frame_rate_d, 1);
        assert_eq!(mapping.picture_aspect_ratio, 2.0);
        assert_eq!(source.data, source_bytes);
        assert_eq!(source.data[3], 4);
        assert_eq!(source.data[7], 8);
    }

    #[test]
    fn malformed_frame_is_rejected_before_mapping() {
        let mut invalid = frame(1);
        invalid.stride_bytes = 4;
        let error = map_video_frame(
            &invalid,
            VideoOutputProfile {
                width: 2,
                height: 1,
                fps: 60,
                pixel_format: VideoOutputPixelFormat::Bgra8StraightAlpha,
            },
        )
        .expect_err("wrong stride must be rejected");
        assert_eq!(error.code, NdiErrorCode::InvalidFrame);
    }

    #[test]
    fn mailbox_replaces_old_frame_and_stays_capacity_one() {
        let mailbox = LatestFrameMailbox::new();
        assert_eq!(mailbox.submit(frame(1)), NdiSubmitResult::Submitted);
        assert_eq!(mailbox.submit(frame(2)), NdiSubmitResult::Replaced);
        assert_eq!(
            mailbox.take(|| false).expect("latest frame").frame_seq,
            FrameSeq(2)
        );
        assert!(mailbox.take(|| true).is_none());
    }

    #[test]
    fn closed_mailbox_releases_waiting_consumer() {
        let mailbox = Arc::new(LatestFrameMailbox::new());
        let worker_mailbox = Arc::clone(&mailbox);
        let started = std::thread::spawn(move || worker_mailbox.take(|| false));
        mailbox.close();
        assert!(started.join().expect("consumer joined").is_none());
    }

    #[test]
    fn burst_submission_remains_bounded_and_non_blocking() {
        let mailbox = LatestFrameMailbox::new();
        let mut last_result = NdiSubmitResult::Submitted;
        for sequence in 0..1000 {
            last_result = mailbox.submit(frame(sequence));
        }
        assert_eq!(last_result, NdiSubmitResult::Replaced);
        assert_eq!(
            mailbox.take(|| false).expect("latest frame").frame_seq,
            FrameSeq(999)
        );
        assert!(mailbox.take(|| true).is_none());
    }

    #[test]
    fn slow_consumer_does_not_block_producer() {
        let mailbox = Arc::new(LatestFrameMailbox::new());
        let consumer_mailbox = Arc::clone(&mailbox);
        let consumer_ready = Arc::new(std::sync::Barrier::new(2));
        let consumer_took_frame = Arc::new(std::sync::Barrier::new(2));
        let consumer_ready_thread = Arc::clone(&consumer_ready);
        let consumer_took_frame_thread = Arc::clone(&consumer_took_frame);
        let consumer = std::thread::spawn(move || {
            consumer_ready_thread.wait();
            let frame = consumer_mailbox.take(|| false);
            consumer_took_frame_thread.wait();
            std::thread::sleep(std::time::Duration::from_millis(20));
            frame
        });
        consumer_ready.wait();
        assert_eq!(mailbox.submit(frame(0)), NdiSubmitResult::Submitted);
        consumer_took_frame.wait();
        let begin = Instant::now();
        for sequence in 1..1000 {
            let _ = mailbox.submit(frame(sequence));
        }
        assert!(begin.elapsed() < std::time::Duration::from_secs(1));
        assert!(consumer.join().expect("slow consumer joined").is_some());
        mailbox.close();
    }

    #[test]
    fn controller_is_off_without_start() {
        let controller = NdiOutputController::new();
        assert_eq!(controller.status(), NdiOutputStatus::Off);
        assert_eq!(controller.metrics(), NdiOutputMetrics::default());
        assert_eq!(
            controller.submit_frame(frame(1)),
            NdiSubmitResult::RejectedNotRunning
        );
    }

    #[cfg(not(feature = "ndi-sdk"))]
    #[test]
    fn feature_off_start_is_typed_error_and_stop_is_idempotent() {
        let mut controller = NdiOutputController::new();
        let error = controller
            .start(NdiOutputConfig::default())
            .expect_err("feature is off");
        assert_eq!(error.code, NdiErrorCode::FeatureDisabled);
        assert!(matches!(
            controller.status(),
            NdiOutputStatus::Error {
                code: NdiErrorCode::FeatureDisabled,
                ..
            }
        ));
        controller.stop().expect("stop is idempotent");
        assert_eq!(controller.status(), NdiOutputStatus::Off);
        controller.stop().expect("second stop is idempotent");
        assert_eq!(controller.status(), NdiOutputStatus::Off);
        assert!(!controller.has_worker());
    }

    #[test]
    fn scripted_backend_walks_off_starting_live_off() {
        let backend = NdiScriptedBackend::successful();
        backend.hold_startup();
        let mut controller = NdiOutputController::with_scripted_backend(backend.clone());
        controller.start(test_config()).expect("scripted start");
        assert!(matches!(controller.status(), NdiOutputStatus::Starting));
        backend.release_startup();
        backend.wait_until_ready();
        assert!(matches!(
            controller.status(),
            NdiOutputStatus::Live {
                source_name,
                ..
            } if source_name == "RusTuberV"
        ));
        controller.stop().expect("stop");
        assert_eq!(controller.status(), NdiOutputStatus::Off);
        assert!(!controller.has_worker());
        assert_eq!(backend.live_senders(), 0);
    }

    #[test]
    fn runtime_initialization_failure_is_error_and_cleans_up() {
        let backend = NdiScriptedBackend::fail_initialize(NdiErrorCode::RuntimeInitFailed);
        let mut controller = NdiOutputController::with_scripted_backend(backend.clone());
        controller.start(test_config()).expect("worker starts");
        backend.wait_until_ready();
        assert!(matches!(
            controller.status(),
            NdiOutputStatus::Error {
                code: NdiErrorCode::RuntimeInitFailed,
                ..
            }
        ));
        assert_eq!(backend.live_senders(), 0);
        assert_eq!(
            controller.submit_frame(frame(1)),
            NdiSubmitResult::RejectedNotRunning
        );
        controller.stop().expect("stop after failure");
        assert_eq!(controller.status(), NdiOutputStatus::Off);
        assert!(!controller.has_worker());
    }

    #[test]
    fn sender_create_failure_is_error_and_cleans_up() {
        let backend = NdiScriptedBackend::fail_create_sender();
        let mut controller = NdiOutputController::with_scripted_backend(backend.clone());
        controller.start(test_config()).expect("worker starts");
        backend.wait_until_ready();
        assert!(matches!(
            controller.status(),
            NdiOutputStatus::Error {
                code: NdiErrorCode::SenderCreateFailed,
                ..
            }
        ));
        assert_eq!(backend.live_senders(), 0);
        controller.stop().expect("cleanup");
        assert!(!controller.has_worker());
        assert_eq!(
            controller.submit_frame(frame(1)),
            NdiSubmitResult::RejectedNotRunning
        );
    }

    #[test]
    fn duplicate_start_while_live_is_rejected_deterministically() {
        let backend = NdiScriptedBackend::successful();
        let mut controller = NdiOutputController::with_scripted_backend(backend.clone());
        controller.start(test_config()).expect("first start");
        backend.wait_until_ready();
        let error = controller
            .start(test_config())
            .expect_err("duplicate start");
        assert_eq!(error.code, NdiErrorCode::AlreadyRunning);
        assert!(matches!(controller.status(), NdiOutputStatus::Live { .. }));
        controller.stop().expect("stop");
    }

    #[test]
    fn repeated_start_stop_does_not_leave_a_worker() {
        let backend = NdiScriptedBackend::successful();
        let mut controller = NdiOutputController::with_scripted_backend(backend.clone());
        for _ in 0..3 {
            controller.start(test_config()).expect("start");
            backend.wait_until_ready();
            controller.stop().expect("stop");
            assert_eq!(controller.status(), NdiOutputStatus::Off);
            assert!(!controller.has_worker());
            assert_eq!(backend.live_senders(), 0);
            assert_eq!(
                controller.submit_frame(frame(1)),
                NdiSubmitResult::RejectedNotRunning
            );
        }
    }

    #[test]
    fn slow_scripted_sender_does_not_block_producer() {
        let backend = NdiScriptedBackend::successful();
        backend.hold_send();
        let mut controller = NdiOutputController::with_scripted_backend(backend.clone());
        controller.start(test_config()).expect("start");
        backend.wait_until_ready();
        assert_eq!(
            controller.submit_frame(frame(0)),
            NdiSubmitResult::Submitted
        );
        backend.wait_until_send_blocked();
        let mut replaced = 0_u64;
        for sequence in 1..1000 {
            match controller.submit_frame(frame(sequence)) {
                NdiSubmitResult::Replaced => replaced += 1,
                NdiSubmitResult::Submitted => {}
                NdiSubmitResult::RejectedNotRunning => {
                    panic!("live controller must accept frames")
                }
            }
        }
        assert!(replaced >= 1);
        assert_eq!(controller.metrics().replaced_frames, replaced);
        backend.release_send();
        controller.stop().expect("stop");
        assert!(!controller.has_worker());
    }

    #[test]
    fn worker_stop_rejects_later_frames() {
        let backend = NdiScriptedBackend::successful();
        let mut controller = NdiOutputController::with_scripted_backend(backend.clone());
        controller.start(test_config()).expect("start");
        backend.wait_until_ready();
        controller.stop().expect("stop");
        assert_eq!(
            controller.submit_frame(frame(9)),
            NdiSubmitResult::RejectedNotRunning
        );
    }

    #[test]
    fn malformed_live_frame_is_rejected_before_the_fake_sender() {
        let backend = NdiScriptedBackend::successful();
        let mut controller = NdiOutputController::with_scripted_backend(backend.clone());
        controller.start(test_config()).expect("start");
        backend.wait_until_ready();
        let mut invalid = frame(1);
        invalid.stride_bytes = 4;
        assert_eq!(controller.submit_frame(invalid), NdiSubmitResult::Submitted);
        let started = std::time::Instant::now();
        while backend.sent_frames() == 0
            && controller.metrics().rejected_frames == 0
            && started.elapsed() < std::time::Duration::from_secs(1)
        {
            std::thread::yield_now();
        }
        assert_eq!(backend.sent_frames(), 0);
        assert!(controller.metrics().rejected_frames >= 1);
        controller.stop().expect("stop");
    }

    #[cfg(feature = "ndi-sdk")]
    #[test]
    #[ignore = "requires a locally installed NDI SDK/runtime"]
    fn sdk_sender_creates_sends_synthetic_bgra_and_stops() {
        let mut controller = NdiOutputController::new();
        match controller.start(test_config()) {
            Ok(()) => {}
            Err(error)
                if matches!(
                    error.code,
                    NdiErrorCode::RuntimeNotFound | NdiErrorCode::RuntimeInitFailed
                ) =>
            {
                return;
            }
            Err(error) => panic!("unexpected SDK start failure: {error}"),
        }
        let started = std::time::Instant::now();
        while !matches!(controller.status(), NdiOutputStatus::Live { .. })
            && !matches!(controller.status(), NdiOutputStatus::Error { .. })
            && started.elapsed() < std::time::Duration::from_secs(5)
        {
            std::thread::yield_now();
        }
        match controller.status() {
            NdiOutputStatus::Live { source_name, .. } => {
                assert_eq!(source_name, "RusTuberV");
            }
            NdiOutputStatus::Error {
                code: NdiErrorCode::RuntimeNotFound | NdiErrorCode::RuntimeInitFailed,
                ..
            } => return,
            other => panic!("SDK sender did not become live: {other:?}"),
        }
        for sequence in 0..3 {
            let _ = controller.submit_frame(frame(sequence));
        }
        let sent_deadline = std::time::Instant::now();
        while controller.metrics().sent_frames == 0
            && sent_deadline.elapsed() < std::time::Duration::from_secs(2)
        {
            std::thread::yield_now();
        }
        controller.stop().expect("SDK stop");
        assert_eq!(controller.status(), NdiOutputStatus::Off);
        assert!(!controller.has_worker());
    }
}
