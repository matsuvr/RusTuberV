//! Capacity-one latest-value slot.

use std::sync::{Condvar, Mutex};
use std::time::Duration;

/// Internal state of a [`LatestSlot`].
struct SlotState<T> {
    generation: u64,
    value: Option<T>,
    closed: bool,
    overwritten: u64,
}

/// A capacity-one slot that always keeps the latest value.
///
/// Old unpublished values are discarded. Multiple readers, each keeping its
/// own generation cursor, can read the retained latest value: a read returns
/// the value together with the generation it was published under, so readers
/// never mark a newer publish as consumed.
pub struct LatestSlot<T> {
    inner: Mutex<SlotState<T>>,
    changed: Condvar,
}

/// Result of reading from a [`LatestSlot`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReadResult<T> {
    /// A value newer than the requested generation, together with the
    /// generation it was published under. Store `generation` as the next
    /// cursor value.
    New {
        /// Generation the value was published under.
        generation: u64,
        /// The retained value.
        value: T,
    },
    /// The slot was closed before a new value arrived.
    Closed,
}

impl<T> Default for LatestSlot<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> LatestSlot<T> {
    /// Creates a new empty slot.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(SlotState {
                generation: 0,
                value: None,
                closed: false,
                overwritten: 0,
            }),
            changed: Condvar::new(),
        }
    }

    /// Publishes a value, replacing any unread value.
    ///
    /// Returns `false` if the slot has been closed.
    pub fn publish(&self, value: T) -> bool {
        let mut state = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.closed {
            return false;
        }
        if state.value.is_some() {
            state.overwritten += 1;
        }
        state.generation += 1;
        state.value = Some(value);
        self.changed.notify_all();
        true
    }

    /// Attempts to read a value newer than `last_generation`.
    #[must_use]
    pub fn try_read_after(&self, last_generation: u64) -> Option<ReadResult<T>>
    where
        T: Clone,
    {
        let state = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Self::read_locked(&state, last_generation)
    }

    /// Waits up to `timeout` for a value newer than `last_generation`.
    pub fn wait_read_after(&self, last_generation: u64, timeout: Duration) -> Option<ReadResult<T>>
    where
        T: Clone,
    {
        let mut state = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let started = std::time::Instant::now();
        loop {
            if let Some(result) = Self::read_locked(&state, last_generation) {
                return Some(result);
            }
            let remaining = timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                return None;
            }
            let (new_state, _) = self
                .changed
                .wait_timeout(state, remaining)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state = new_state;
        }
    }

    /// Closes the slot, waking any waiters.
    pub fn close(&self) {
        let mut state = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.closed = true;
        state.value = None;
        self.changed.notify_all();
    }

    /// Removes the currently retained value without closing the slot.
    ///
    /// This is used at a session boundary, such as camera Stop or reconnect,
    /// so a consumer cannot process a frame captured by the previous session.
    /// The generation is intentionally unchanged: clearing is not a produced
    /// value and the next publish remains the only value visible to readers.
    pub fn clear(&self) {
        let mut state = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.value = None;
        self.changed.notify_all();
    }

    /// Returns the number of values that were overwritten before being read.
    #[must_use]
    pub fn overwritten_count(&self) -> u64 {
        let state = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.overwritten
    }

    /// Returns the current generation.
    #[must_use]
    pub fn generation(&self) -> u64 {
        let state = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.generation
    }

    /// Returns `true` if the slot has been closed.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        let state = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.closed
    }

    fn read_locked(state: &SlotState<T>, last_generation: u64) -> Option<ReadResult<T>>
    where
        T: Clone,
    {
        if state.closed {
            return Some(ReadResult::Closed);
        }
        if state.generation > last_generation {
            state.value.clone().map(|value| ReadResult::New {
                generation: state.generation,
                value,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )] // tests may panic (AGENTS.md)
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn publish_and_read() {
        let slot = LatestSlot::new();
        assert!(slot.publish(42));
        assert_eq!(
            slot.try_read_after(0),
            Some(ReadResult::New {
                generation: 1,
                value: 42
            })
        );
    }

    #[test]
    fn read_returns_the_newer_publish_after_the_cursor() {
        let slot = LatestSlot::new();
        slot.publish(1);
        assert_eq!(
            slot.try_read_after(0),
            Some(ReadResult::New {
                generation: 1,
                value: 1
            })
        );
        slot.publish(2);
        assert_eq!(
            slot.try_read_after(1),
            Some(ReadResult::New {
                generation: 2,
                value: 2
            })
        );
        assert_eq!(slot.try_read_after(2), None);
    }

    #[test]
    fn two_readers_with_independent_cursors_read_the_same_value() {
        let slot = LatestSlot::new();
        slot.publish(5);
        assert_eq!(
            slot.try_read_after(0),
            Some(ReadResult::New {
                generation: 1,
                value: 5
            })
        );
        assert_eq!(
            slot.try_read_after(0),
            Some(ReadResult::New {
                generation: 1,
                value: 5
            })
        );
        slot.publish(6);
        slot.publish(7);
        assert_eq!(
            slot.try_read_after(1),
            Some(ReadResult::New {
                generation: 3,
                value: 7
            })
        );
    }

    #[test]
    fn overwritten_count_increases() {
        let slot = LatestSlot::<i32>::new();
        slot.publish(1);
        slot.publish(2);
        slot.publish(3);
        assert_eq!(slot.overwritten_count(), 2);
    }

    #[test]
    fn close_wakes_waiter() {
        let slot: Arc<LatestSlot<i32>> = Arc::new(LatestSlot::new());
        let slot2 = Arc::clone(&slot);
        let handle = thread::spawn(move || slot2.wait_read_after(0, Duration::from_secs(5)));
        thread::sleep(Duration::from_millis(50));
        slot.close();
        let result = handle.join().expect("waiter panicked");
        assert_eq!(result, Some(ReadResult::Closed));
    }

    #[test]
    fn publish_after_close_is_ignored() {
        let slot = LatestSlot::new();
        slot.close();
        assert!(!slot.publish(1));
    }

    #[test]
    fn clear_removes_retained_value_without_closing_slot() {
        let slot = LatestSlot::new();
        assert!(slot.publish(42));
        let generation = slot.generation();

        slot.clear();

        assert!(!slot.is_closed());
        assert_eq!(slot.generation(), generation);
        assert_eq!(slot.try_read_after(0), None);
        assert!(slot.publish(43));
        assert_eq!(
            slot.try_read_after(generation),
            Some(ReadResult::New {
                generation: generation + 1,
                value: 43
            })
        );
    }

    #[test]
    fn capacity_one_does_not_grow() {
        const N: usize = 100_000;
        let slot = LatestSlot::new();
        for value in 0..N {
            assert!(slot.publish(value));
        }
        let result = slot.try_read_after(0);
        assert_eq!(
            result,
            Some(ReadResult::New {
                generation: N as u64,
                value: N - 1
            })
        );
        assert_eq!(slot.overwritten_count(), (N - 1) as u64);
    }

    #[test]
    fn slow_consumer_catches_up_to_latest() {
        let slot: Arc<LatestSlot<usize>> = Arc::new(LatestSlot::new());
        let slot2 = Arc::clone(&slot);

        let producer = thread::spawn(move || {
            for value in 0..1000 {
                slot2.publish(value);
                thread::sleep(Duration::from_micros(10));
            }
        });

        let mut last_seen = 0;
        let mut consumed = 0;
        while last_seen < 999 {
            if let Some(ReadResult::New {
                generation,
                value,
            }) = slot.wait_read_after(last_seen, Duration::from_secs(1))
            {
                last_seen = generation;
                consumed += 1;
                assert!(value <= 999);
            } else {
                panic!("timed out waiting for next value");
            }
        }

        producer.join().expect("producer panicked");
        assert!(
            consumed < 1000,
            "slow consumer should skip frames; consumed {consumed}"
        );
        assert_eq!(slot.try_read_after(slot.generation()), None);
    }

    #[test]
    fn closed_slot_reports_closed() {
        let slot = LatestSlot::<i32>::new();
        assert!(!slot.is_closed());
        slot.close();
        assert!(slot.is_closed());
        assert_eq!(slot.try_read_after(0), Some(ReadResult::Closed));
    }

    #[test]
    fn wait_with_max_duration_returns_immediately_when_ready_or_closed() {
        let slot = LatestSlot::new();
        slot.publish(7);
        assert_eq!(
            slot.wait_read_after(0, Duration::MAX),
            Some(ReadResult::New {
                generation: 1,
                value: 7
            })
        );
        slot.close();
        assert_eq!(slot.wait_read_after(1, Duration::MAX), Some(ReadResult::Closed));
    }
}
