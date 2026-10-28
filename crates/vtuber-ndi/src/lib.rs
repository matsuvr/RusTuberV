//! Safe, optional NDI video-output boundary.
//!
//! The public API contains only application-owned configuration, status,
//! metrics, and [`vtuber_core::VideoOutputFrame`]. NDI SDK types and the
//! binding feature remain private to this crate. The default build is a
//! deterministic feature-disabled stub so the rest of the workspace does not
//! require an installed NDI SDK. Submission acknowledges the bounded mailbox,
//! not network delivery. The UI requests stop without joining; only explicit blocking stop joins.
//! Drop requests stop and lets the worker release its own native handles.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod controller;
mod frame;
pub mod runtime_identity;

pub use controller::{NdiOutputController, NdiScriptedBackend};
use frame::validate_output_profile;
pub use frame::{NDI_SEND_TIMECODE_SYNTHESIZE, NdiFourCc, NdiVideoFrameMapping, map_video_frame};

use std::fmt;

use vtuber_core::{FrameSeq, VideoOutputProfile};

/// Returns whether this build includes the explicit NDI SDK backend.
#[must_use]
pub const fn is_sdk_feature_enabled() -> bool {
    cfg!(feature = "ndi-sdk")
}

/// Returns whether an NDI Standard runtime DLL is discoverable on this machine.
///
/// Filesystem probe only; it never loads the SDK. On Windows it checks the
/// executable directory, `NDI_SDK_DIR`, `NDI_RUNTIME_DIR_V6`, and the standard
/// install locations. On other platforms it defers to the SDK init result so
/// a valid non-DLL install is never reported as missing.
#[must_use]
pub fn is_ndi_runtime_installed() -> bool {
    #[cfg(target_os = "windows")]
    {
        candidate_runtime_dirs()
            .iter()
            .any(|dir| runtime_identity::runtime_file_in(dir).is_some())
    }
    #[cfg(not(target_os = "windows"))]
    {
        true
    }
}

/// Stable error codes emitted by the optional output backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NdiErrorCode {
    /// Shared sender state was poisoned by a panic.
    StatePoisoned,
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
    /// The worker thread could not be spawned by the OS.
    WorkerSpawnFailed,
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
            Self::StatePoisoned => "NDI_STATE_POISONED",
            Self::FeatureDisabled => "NDI_FEATURE_DISABLED",
            Self::RuntimeNotFound => "NDI_RUNTIME_NOT_FOUND",
            Self::RuntimeInitFailed => "NDI_RUNTIME_INIT_FAILED",
            Self::SenderCreateFailed => "NDI_SENDER_CREATE_FAILED",
            Self::SendFailed => "NDI_SEND_FAILED",
            Self::WorkerStopFailed => "NDI_WORKER_STOP_FAILED",
            Self::WorkerSpawnFailed => "NDI_WORKER_SPAWN_FAILED",
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
        validate_output_profile(self.profile)
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

#[cfg(target_os = "windows")]
fn candidate_runtime_dirs() -> Vec<std::path::PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(exe) = std::env::current_exe()
        && let Some(parent) = exe.parent()
    {
        dirs.push(parent.to_path_buf());
    }
    if let Some(sdk_dir) = std::env::var_os("NDI_SDK_DIR") {
        let sdk_dir = std::path::PathBuf::from(sdk_dir);
        dirs.push(sdk_dir.join("Bin").join("x64"));
    }
    if let Some(runtime_dir) = std::env::var_os("NDI_RUNTIME_DIR_V6") {
        let runtime_dir = std::path::PathBuf::from(runtime_dir);
        dirs.push(runtime_dir.join("v6"));
        dirs.push(runtime_dir);
    }
    dirs.push(std::path::PathBuf::from(
        r"C:\Program Files\NDI\NDI 6 SDK\Bin\x64",
    ));
    dirs.push(std::path::PathBuf::from(
        r"C:\Program Files\NDI\NDI 6 Runtime\v6",
    ));
    dirs.push(std::path::PathBuf::from(
        r"C:\Program Files\NDI\NDI 6 Runtime",
    ));
    dirs
}
