//! Platform-specific camera backends.

#[cfg(target_os = "windows")]
pub mod msmf;
