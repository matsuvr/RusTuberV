//! Native OS boundary for sleep notifications and startup error dialogs.
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
#[cfg(target_os = "windows")]
pub use windows::show_error_dialog;

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
            .try_update(Ordering::AcqRel, Ordering::Acquire, |value| {
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
#[path = "../../../verification/replay.rs"]
mod quint_replay;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "generated Quint ITF traces"]
    #[allow(clippy::panic, clippy::indexing_slicing)]
    fn quint_power() {
        use crate::quint_replay::{number, traces};
        let initial_epoch = (u64::MAX >> 1) - 1;
        let epoch = |cycles: i64| (initial_epoch + cycles as u64) & (u64::MAX >> 1);
        traces("power", |states| {
            let power = PowerAtomic(AtomicU64::new(initial_epoch << 1));
            let mut observed = power.snapshot();
            for row in states {
                match row["event"].as_str() {
                    Some("init") => {}
                    Some("resume") => power.resume(),
                    Some("suspend") => power.suspend(),
                    Some("observe") => observed = power.snapshot(),
                    other => panic!("unexpected power event {other:?}"),
                }
                let actual = power.snapshot();
                assert_eq!(actual.generation, epoch(number(&row["cycles"])));
                assert_eq!(actual.sleeping, row["sleeping"]);
                assert_eq!(observed.generation, epoch(number(&row["observed"])));
                assert_eq!(observed.sleeping, row["observedSleep"]);
            }
        });
    }

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
