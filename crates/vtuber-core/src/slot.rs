//! Capacity-one latest-value slot.

use std::sync::{Condvar, Mutex};
use std::time::Duration;

/// Counts publications skipped by one reader between two observed generations.
///
/// `None` marks a new reader/session and never counts older publications as
/// missed. Readers keep their own cursor; no consumption state is shared.
#[must_use]
pub fn skipped_generations(previous: Option<u64>, current: u64) -> u64 {
    previous.map_or(0, |previous| {
        current.saturating_sub(previous).saturating_sub(1)
    })
}

/// Internal state of a [`LatestSlot`].
struct SlotState<T> {
    generation: u64,
    value: Option<T>,
    closed: bool,
    replacements: u64,
}

/// A capacity-one slot that always keeps the latest value.
///
/// Previously retained values are replaced on publication. Multiple readers, each keeping its
/// own generation cursor, can read the retained latest value: a read returns
/// the value together with the generation it was published under, so readers
/// never mark a newer publish as consumed.
/// Every operation returns [`SlotPoisoned`] after a panic while holding the
/// mutex; poisoned state is never recovered or exposed.
pub struct LatestSlot<T> {
    inner: Mutex<SlotState<T>>,
    changed: Condvar,
}

impl<T> std::fmt::Debug for SlotState<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SlotState")
            .field("generation", &self.generation)
            .field("has_value", &self.value.is_some())
            .field("closed", &self.closed)
            .field("replacements", &self.replacements)
            .finish()
    }
}

impl<T> std::fmt::Debug for LatestSlot<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Mutex's Debug uses try_lock; formatting does not wait for a reader
        // or recursively acquire the slot lock through accessors.
        f.debug_struct("LatestSlot")
            .field("inner", &self.inner)
            .finish()
    }
}

/// A slot mutex was poisoned; its retained state must no longer be used.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SlotPoisoned;

impl std::fmt::Display for SlotPoisoned {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("latest-value slot mutex poisoned")
    }
}

impl std::error::Error for SlotPoisoned {}

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
                replacements: 0,
            }),
            changed: Condvar::new(),
        }
    }

    /// Publishes a value, replacing any retained value, even if it was read.
    ///
    /// Returns `false` if the slot has been closed.
    pub fn publish(&self, value: T) -> Result<bool, SlotPoisoned> {
        let mut state = self.inner.lock().map_err(|_| SlotPoisoned)?;
        if state.closed {
            return Ok(false);
        }
        if state.value.is_some() {
            state.replacements += 1;
        }
        state.generation += 1;
        state.value = Some(value);
        self.changed.notify_all();
        Ok(true)
    }

    /// Clones a retained value newer than this reader's `last_generation`.
    ///
    /// Returns `None` when there is no newer retained value (including after
    /// `clear`). A closed slot returns `Some(ReadResult::Closed)` regardless of
    /// the cursor. Reading does not remove a value or advance another reader.
    pub fn try_read_after(
        &self,
        last_generation: u64,
    ) -> Result<Option<ReadResult<T>>, SlotPoisoned>
    where
        T: Clone,
    {
        let state = self.inner.lock().map_err(|_| SlotPoisoned)?;
        Ok(Self::read_locked(&state, last_generation))
    }

    /// Waits up to `timeout` for a value newer than this reader's cursor.
    ///
    /// Returns `None` on timeout without a newer retained value. Closing the
    /// slot wakes the wait and returns `Some(ReadResult::Closed)`. Values are
    /// cloned rather than consumed, as in [`Self::try_read_after`].
    pub fn wait_read_after(
        &self,
        last_generation: u64,
        timeout: Duration,
    ) -> Result<Option<ReadResult<T>>, SlotPoisoned>
    where
        T: Clone,
    {
        let mut state = self.inner.lock().map_err(|_| SlotPoisoned)?;
        let started = std::time::Instant::now();
        loop {
            if let Some(result) = Self::read_locked(&state, last_generation) {
                return Ok(Some(result));
            }
            let remaining = timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                return Ok(None);
            }
            let (new_state, _) = self
                .changed
                .wait_timeout(state, remaining)
                .map_err(|_| SlotPoisoned)?;
            state = new_state;
        }
    }

    /// Permanently closes the slot, discards its value and wakes all waiters.
    ///
    /// Repeated calls are harmless. Subsequent publications return `false`;
    /// `clear` does not reopen the slot.
    pub fn close(&self) -> Result<(), SlotPoisoned> {
        let mut state = self.inner.lock().map_err(|_| {
            self.changed.notify_all();
            SlotPoisoned
        })?;
        state.closed = true;
        state.value = None;
        self.changed.notify_all();
        Ok(())
    }

    /// Removes the currently retained value without closing the slot.
    ///
    /// This is used at a session boundary, such as camera Stop or reconnect,
    /// so a consumer cannot process a frame captured by the previous session.
    /// The generation is intentionally unchanged: clearing is not a produced
    /// value and the next publish remains the only value visible to readers.
    pub fn clear(&self) -> Result<(), SlotPoisoned> {
        let mut state = self.inner.lock().map_err(|_| {
            self.changed.notify_all();
            SlotPoisoned
        })?;
        state.value = None;
        self.changed.notify_all();
        Ok(())
    }

    /// Returns the number of retained values replaced by successful publications.
    ///
    /// This is not a reader loss count: reads do not remove the retained value.
    pub fn replacement_count(&self) -> Result<u64, SlotPoisoned> {
        let state = self.inner.lock().map_err(|_| SlotPoisoned)?;
        Ok(state.replacements)
    }

    /// Returns the current generation.
    pub fn generation(&self) -> Result<u64, SlotPoisoned> {
        let state = self.inner.lock().map_err(|_| SlotPoisoned)?;
        Ok(state.generation)
    }

    /// Returns `true` if the slot has been closed.
    pub fn is_closed(&self) -> Result<bool, SlotPoisoned> {
        let state = self.inner.lock().map_err(|_| SlotPoisoned)?;
        Ok(state.closed)
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
    fn debug_summarizes_a_value_without_requiring_debug_or_dumping_it() {
        struct NoDebug([u8; 1024]);
        let slot = LatestSlot::new();
        let value = NoDebug([187; 1024]);
        assert_eq!(value.0.len(), 1024);
        slot.publish(value).expect("slot access");
        let text = format!("{slot:?}");
        assert!(text.contains("generation: 1"));
        assert!(text.contains("has_value: true"));
        assert!(!text.contains("187"));
        assert_eq!(slot.generation().expect("slot generation"), 1);
        assert!(!slot.is_closed().expect("slot access"));
        let _guard = slot.inner.lock().unwrap();
        assert!(format!("{slot:?}").contains("<locked>"));
    }

    #[test]
    fn skipped_generations_are_per_reader_and_reset_with_the_session() {
        let slot = LatestSlot::new();
        let mut frequent = None;
        let mut slow = None;
        for generation in 1..=4 {
            slot.publish(generation).expect("slot access");
            let Some(ReadResult::New {
                generation: current,
                ..
            }) = slot
                .try_read_after(frequent.unwrap_or(0))
                .expect("slot access")
            else {
                panic!("new publication")
            };
            assert_eq!(skipped_generations(frequent, current), 0);
            frequent = Some(current);
            if generation == 1 {
                slow = Some(current);
            }
        }
        let Some(ReadResult::New { generation, .. }) =
            slot.try_read_after(slow.unwrap_or(0)).expect("slot access")
        else {
            panic!("slow reader's latest publication")
        };
        assert_eq!(skipped_generations(slow, generation), 2);
        assert_eq!(skipped_generations(None, generation), 0);
        assert_eq!(skipped_generations(Some(generation), generation), 0);
        assert_eq!(skipped_generations(Some(u64::MAX), 1), 0);
    }

    #[test]
    fn publish_and_read() {
        let slot = LatestSlot::new();
        assert!(slot.publish(42).expect("slot access"));
        assert_eq!(
            slot.try_read_after(0).expect("slot access"),
            Some(ReadResult::New {
                generation: 1,
                value: 42
            })
        );
    }

    #[test]
    fn read_returns_the_newer_publish_after_the_cursor() {
        let slot = LatestSlot::new();
        slot.publish(1).expect("slot access");
        assert_eq!(
            slot.try_read_after(0).expect("slot access"),
            Some(ReadResult::New {
                generation: 1,
                value: 1
            })
        );
        slot.publish(2).expect("slot access");
        assert_eq!(
            slot.try_read_after(1).expect("slot access"),
            Some(ReadResult::New {
                generation: 2,
                value: 2
            })
        );
        assert_eq!(slot.try_read_after(2).expect("slot access"), None);
    }

    #[test]
    fn two_readers_with_independent_cursors_read_the_same_value() {
        let slot = LatestSlot::new();
        slot.publish(5).expect("slot access");
        assert_eq!(
            slot.try_read_after(0).expect("slot access"),
            Some(ReadResult::New {
                generation: 1,
                value: 5
            })
        );
        assert_eq!(
            slot.try_read_after(0).expect("slot access"),
            Some(ReadResult::New {
                generation: 1,
                value: 5
            })
        );
        slot.publish(6).expect("slot access");
        slot.publish(7).expect("slot access");
        assert_eq!(
            slot.try_read_after(1).expect("slot access"),
            Some(ReadResult::New {
                generation: 3,
                value: 7
            })
        );
    }

    #[test]
    fn replacement_count_increases_even_after_reads() {
        let slot = LatestSlot::<i32>::new();
        slot.publish(1).expect("slot access");
        let _ = slot.try_read_after(0).expect("slot access");
        slot.publish(2).expect("slot access");
        let _ = slot.try_read_after(1).expect("slot access");
        slot.publish(3).expect("slot access");
        assert_eq!(slot.replacement_count().expect("slot access"), 2);
    }

    #[test]
    fn close_wakes_waiter() {
        let slot: Arc<LatestSlot<i32>> = Arc::new(LatestSlot::new());
        let slot2 = Arc::clone(&slot);
        let handle = thread::spawn(move || {
            slot2
                .wait_read_after(0, Duration::from_secs(5))
                .expect("slot access")
        });
        thread::sleep(Duration::from_millis(50));
        slot.close().expect("slot cleanup");
        let result = handle.join().expect("waiter panicked");
        assert_eq!(result, Some(ReadResult::Closed));
    }

    #[test]
    fn publish_after_close_is_ignored() {
        let slot = LatestSlot::new();
        slot.close().expect("slot cleanup");
        assert!(!slot.publish(1).expect("slot access"));
    }

    #[test]
    fn clear_removes_retained_value_without_closing_slot() {
        let slot = LatestSlot::new();
        assert!(slot.publish(42).expect("slot access"));
        let generation = slot.generation().expect("slot generation");

        slot.clear().expect("slot cleanup");

        assert!(!slot.is_closed().expect("slot access"));
        assert_eq!(slot.generation().expect("slot generation"), generation);
        assert_eq!(slot.try_read_after(0).expect("slot access"), None);
        assert!(slot.publish(43).expect("slot access"));
        assert_eq!(
            slot.try_read_after(generation).expect("slot access"),
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
            assert!(slot.publish(value).expect("slot access"));
        }
        let result = slot.try_read_after(0).expect("slot access");
        assert_eq!(
            result,
            Some(ReadResult::New {
                generation: N as u64,
                value: N - 1
            })
        );
        assert_eq!(
            slot.replacement_count().expect("slot access"),
            (N - 1) as u64
        );
    }

    #[test]
    fn slow_consumer_catches_up_to_latest() {
        let slot: Arc<LatestSlot<usize>> = Arc::new(LatestSlot::new());
        let slot2 = Arc::clone(&slot);

        let producer = thread::spawn(move || {
            for value in 0..1000 {
                slot2.publish(value).expect("slot access");
                thread::sleep(Duration::from_micros(10));
            }
        });

        let mut last_seen = 0;
        let mut consumed = 0;
        while last_seen < 999 {
            if let Some(ReadResult::New { generation, value }) = slot
                .wait_read_after(last_seen, Duration::from_secs(1))
                .expect("slot access")
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
        assert_eq!(
            slot.try_read_after(slot.generation().expect("slot generation"))
                .expect("slot access"),
            None
        );
    }

    #[test]
    fn closed_slot_reports_closed() {
        let slot = LatestSlot::<i32>::new();
        assert!(!slot.is_closed().expect("slot access"));
        slot.close().expect("slot cleanup");
        assert!(slot.is_closed().expect("slot access"));
        assert_eq!(
            slot.try_read_after(0).expect("slot access"),
            Some(ReadResult::Closed)
        );
    }

    #[test]
    fn wait_with_max_duration_returns_immediately_when_ready_or_closed() {
        let slot = LatestSlot::new();
        slot.publish(7).expect("slot access");
        assert_eq!(
            slot.wait_read_after(0, Duration::MAX).expect("slot access"),
            Some(ReadResult::New {
                generation: 1,
                value: 7
            })
        );
        slot.close().expect("slot cleanup");
        assert_eq!(
            slot.wait_read_after(1, Duration::MAX).expect("slot access"),
            Some(ReadResult::Closed)
        );
    }

    #[test]
    fn poisoned_slot_rejects_reads_writes_and_cleanup() {
        let slot = LatestSlot::new();
        slot.publish(7).unwrap();
        let panic = std::panic::catch_unwind(|| {
            let mut state = slot.inner.lock().unwrap();
            state.generation = 999;
            panic!("interrupted slot update");
        });
        assert!(panic.is_err());
        assert_eq!(slot.publish(8), Err(SlotPoisoned));
        assert_eq!(slot.try_read_after(0), Err(SlotPoisoned));
        assert_eq!(slot.wait_read_after(0, Duration::MAX), Err(SlotPoisoned));
        assert_eq!(slot.generation(), Err(SlotPoisoned));
        assert_eq!(slot.replacement_count(), Err(SlotPoisoned));
        assert_eq!(slot.is_closed(), Err(SlotPoisoned));
        assert_eq!(slot.clear(), Err(SlotPoisoned));
        assert_eq!(slot.close(), Err(SlotPoisoned));
    }
}
