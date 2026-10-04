//! Platform-specific camera backends.

#[cfg(target_os = "windows")]
pub mod msmf;

#[cfg(target_os = "macos")]
pub mod avfoundation;
