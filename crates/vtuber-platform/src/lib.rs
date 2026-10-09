//! Native sleep notifications shared by camera, rendering and network output.
//! Callbacks only update atomics; they never wait for workers or call drivers.

use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
pub use macos::PowerNotifications;
#[cfg(target_os = "windows")]
mod windows;
#[cfg(target_os = "windows")]
pub use windows::PowerNotifications;

/// One coherent snapshot of system sleep state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PowerState {
    /// Whether the system is preparing to sleep or sleeping.
    pub sleeping: bool,
    /// Sleep cycle counter, including cycles missed by a suspended worker.
    pub generation: u64,
}

#[derive(Default)]
struct PowerAtomic(AtomicU64);

impl PowerAtomic {
    fn snapshot(&self) -> PowerState {
        let value = self.0.load(Ordering::Acquire);
        PowerState {
            sleeping: value & 1 != 0,
            generation: value >> 1,
        }
    }

    #[cfg(any(target_os = "macos", target_os = "windows", test))]
    fn suspend(&self) {
        let _ = self
            .0
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                (value & 1 == 0).then_some((value & !1).wrapping_add(2) | 1)
            });
    }

    #[cfg(any(target_os = "macos", target_os = "windows", test))]
    fn resume(&self) {
        self.0.fetch_and(!1, Ordering::Release);
    }
}

static POWER: PowerAtomic = PowerAtomic(AtomicU64::new(0));

/// Returns sleep state without native calls or locks.
#[must_use]
pub fn power_state() -> PowerState {
    POWER.snapshot()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missed_cycles_and_duplicate_windows_resume_notifications() {
        let power = PowerAtomic::default();
        power.suspend();
        power.suspend();
        assert_eq!(
            power.snapshot(),
            PowerState {
                sleeping: true,
                generation: 1
            }
        );
        power.resume();
        power.resume();
        assert_eq!(
            power.snapshot(),
            PowerState {
                sleeping: false,
                generation: 1
            }
        );
        power.suspend();
        power.resume();
        assert_eq!(
            power.snapshot(),
            PowerState {
                sleeping: false,
                generation: 2
            }
        );
    }
}
