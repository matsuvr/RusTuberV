use std::ptr::NonNull;

use block2::RcBlock;
use objc2::MainThreadMarker;
use objc2::rc::Retained;
use objc2::runtime::{NSObjectProtocol, ProtocolObject};
use objc2_app_kit::{
    NSWorkspace, NSWorkspaceDidWakeNotification, NSWorkspaceWillSleepNotification,
};
use objc2_foundation::{NSNotification, NSNotificationCenter};

use crate::POWER;
#[cfg(test)]
use crate::power_state;

/// Owns workspace observers for the duration of the desktop event loop.
/// Construct and drop this guard on the macOS main thread.
pub struct PowerNotifications {
    _observers: Observers,
    _main_thread: MainThreadMarker,
}

struct Observers {
    center: Retained<NSNotificationCenter>,
    sleep: Retained<ProtocolObject<dyn NSObjectProtocol>>,
    wake: Retained<ProtocolObject<dyn NSObjectProtocol>>,
}

impl PowerNotifications {
    /// Registers sleep and wake notifications on the main thread.
    pub fn new() -> std::io::Result<Self> {
        let main_thread = MainThreadMarker::new().ok_or_else(|| {
            std::io::Error::other("macOS power notifications require the main thread")
        })?;
        let center = NSWorkspace::sharedWorkspace().notificationCenter();
        Ok(Self {
            _observers: Observers::new(center),
            _main_thread: main_thread,
        })
    }
}

impl Observers {
    fn new(center: Retained<NSNotificationCenter>) -> Self {
        let sleep_block = RcBlock::new(|_: NonNull<NSNotification>| {
            POWER.suspend();
        });
        let wake_block = RcBlock::new(|_: NonNull<NSNotification>| {
            POWER.resume();
        });
        // SAFETY: the named notifications come from NSWorkspace. No object or
        // queue filter is supplied; both copied blocks are sendable and only
        // access a static atomic. The returned observer tokens are retained.
        let (sleep, wake) = unsafe {
            (
                center.addObserverForName_object_queue_usingBlock(
                    Some(NSWorkspaceWillSleepNotification),
                    None,
                    None,
                    &sleep_block,
                ),
                center.addObserverForName_object_queue_usingBlock(
                    Some(NSWorkspaceDidWakeNotification),
                    None,
                    None,
                    &wake_block,
                ),
            )
        };
        Self {
            center,
            sleep,
            wake,
        }
    }
}

impl Drop for Observers {
    fn drop(&mut self) {
        // SAFETY: these are exactly the tokens returned by this center, and
        // the tokens and center are still retained during teardown.
        unsafe {
            self.center.removeObserver((*self.sleep).as_ref());
            self.center.removeObserver((*self.wake).as_ref());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notifications_preserve_missed_cycles_and_observers_are_removed() {
        let center = NSNotificationCenter::new();
        let observers = Observers::new(center.clone());
        let before = power_state().generation;
        // SAFETY: the test posts the same names and workspace-shaped object
        // contract (no object filter) used by the production observers.
        unsafe {
            center.postNotificationName_object(NSWorkspaceWillSleepNotification, None);
        }
        assert!(power_state().sleeping);
        assert_eq!(power_state().generation, before + 1);
        unsafe {
            center.postNotificationName_object(NSWorkspaceDidWakeNotification, None);
        }
        assert!(!power_state().sleeping);
        assert_eq!(power_state().generation, before + 1);
        drop(observers);
        unsafe {
            center.postNotificationName_object(NSWorkspaceWillSleepNotification, None);
        }
        assert!(!power_state().sleeping);
        assert_eq!(power_state().generation, before + 1);
    }
}
