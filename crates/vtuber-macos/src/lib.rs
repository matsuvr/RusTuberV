//! Native macOS sleep notifications. Callbacks only publish atomic state;
//! camera and rendering work remain outside the notification callback.

#[cfg(target_os = "macos")]
mod power;
#[cfg(target_os = "macos")]
pub use power::{PowerNotifications, power_state};

/// One coherent snapshot of system sleep state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PowerState {
    /// Whether the system is preparing to sleep or sleeping.
    pub sleeping: bool,
    /// Sleep cycle counter, including cycles missed by a suspended worker.
    pub generation: u64,
}
