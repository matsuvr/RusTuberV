use std::ffi::{CStr, NulError};
use std::ops::RangeInclusive;
use std::os::raw::c_char;
use std::path::PathBuf;

use crate::sys;

/// Status codes returned by the C API. Mirrors `absl::StatusCode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StatusCode {
    Cancelled,
    Unknown,
    InvalidArgument,
    DeadlineExceeded,
    NotFound,
    AlreadyExists,
    PermissionDenied,
    ResourceExhausted,
    FailedPrecondition,
    Aborted,
    OutOfRange,
    Unimplemented,
    Internal,
    Unavailable,
    DataLoss,
    Unauthenticated,
    /// A code this crate does not know about.
    Other(i32),
}

impl StatusCode {
    pub(crate) fn from_raw(raw: i32) -> Self {
        match raw {
            1 => Self::Cancelled,
            2 => Self::Unknown,
            3 => Self::InvalidArgument,
            4 => Self::DeadlineExceeded,
            5 => Self::NotFound,
            6 => Self::AlreadyExists,
            7 => Self::PermissionDenied,
            8 => Self::ResourceExhausted,
            9 => Self::FailedPrecondition,
            10 => Self::Aborted,
            11 => Self::OutOfRange,
            12 => Self::Unimplemented,
            13 => Self::Internal,
            14 => Self::Unavailable,
            15 => Self::DataLoss,
            16 => Self::Unauthenticated,
            other => Self::Other(other),
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("mediapipe: {message} ({code:?})")]
    Mp { code: StatusCode, message: String },

    #[error(
        "libmediapipe not found. Set $MEDIAPIPE_LIB to its path, or enable the \
         `download` feature to fetch it automatically. Searched: {searched:?}"
    )]
    LibraryNotFound { searched: Vec<PathBuf> },

    #[error("failed to load libmediapipe from {path}: {source}")]
    Load {
        path: PathBuf,
        #[source]
        source: libloading::Error,
    },

    #[error(
        "no prebuilt libmediapipe is published for {os}-{arch}. Build it from \
         source (`bazel build -c opt //mediapipe/tasks/c:libmediapipe.so`) and \
         point $MEDIAPIPE_LIB at the result."
    )]
    UnsupportedTarget {
        os: &'static str,
        arch: &'static str,
    },

    #[error("{quantity} must be in {}..={}, got {value}", range.start(), range.end())]
    OutOfRange {
        quantity: &'static str,
        value: f32,
        range: RangeInclusive<f32>,
    },

    #[error("pixel buffer is {got} bytes, expected {expected} for {width}x{height}x{channels}")]
    BufferSize {
        got: usize,
        expected: usize,
        width: u32,
        height: u32,
        channels: u32,
    },

    #[error("{what} is {len} bytes, which does not fit the C API's {limit}-byte limit")]
    TooLarge {
        what: &'static str,
        len: usize,
        limit: u64,
    },

    #[error("expected a {rows}x{cols} matrix, got {got_rows}x{got_cols}")]
    MatrixShape {
        rows: u32,
        cols: u32,
        got_rows: u32,
        got_cols: u32,
    },

    #[error("all {0} live-stream slots are in use; drop an existing stream first")]
    NoFreeSlot(usize),

    #[error("failed to download libmediapipe: {0}")]
    Download(String),

    #[error("checksum mismatch for downloaded wheel: expected {expected}, got {got}")]
    ChecksumMismatch { expected: String, got: String },

    #[error("path contains an interior NUL byte")]
    Nul(#[from] NulError),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Build an [`Error::Mp`] from a C status plus the `char** error_msg` the
    /// call populated, freeing the message with `MpErrorFree`.
    ///
    /// # Safety
    /// `msg` must be null or a pointer produced by the same MediaPipe call that
    /// returned `status`, not yet freed.
    pub(crate) unsafe fn from_status(
        lib: &sys::MpLib,
        status: sys::MpStatus,
        msg: *mut c_char,
    ) -> Self {
        let message = if msg.is_null() {
            String::new()
        } else {
            // SAFETY: `msg` is non-null here and points at a NUL-terminated string MediaPipe allocated.
            let owned = unsafe { CStr::from_ptr(msg) }
                .to_string_lossy()
                .into_owned();
            // SAFETY: `msg` came from MediaPipe and has not been freed yet.
            unsafe { lib.MpErrorFree(msg) };
            owned
        };

        // A struct-layout mismatch between the loaded library and the bindings
        // does not fault; it surfaces as the C side reading `running_mode` from
        // the wrong offset. Say so, rather than leaving the caller to guess.
        let message = if message.contains("running mode") {
            format!(
                "{message} — this usually means the loaded libmediapipe has a \
                 different ABI than detected; see sys::Abi"
            )
        } else {
            message
        };

        Error::Mp {
            code: StatusCode::from_raw(status as i32),
            message,
        }
    }
}
