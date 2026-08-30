//! Deterministic worker supervision helpers.
//!
//! This module provides a small, std-only wrapper around named threads,
//! cooperative stop tokens, and typed join results. It is intended for
//! camera and inference workers that must own their backend objects inside
//! a single thread and shut down cleanly.

use std::thread::{self, JoinHandle};

use crate::StopToken;

/// Result of joining a supervised worker thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WorkerResult<T> {
    /// The worker completed normally and returned a value.
    Completed(T),
    /// The worker thread panicked.
    Panicked,
    /// The worker could not be spawned.
    SpawnFailed,
}

/// Handle to a named worker thread with a cooperative stop token.
///
/// The handle owns the thread's [`JoinHandle`]. Dropping the handle without
/// calling [`WorkerHandle::join`] does **not** detach the thread; it merely
/// leaks the handle and the thread will continue running until it completes.
#[derive(Debug)]
pub struct WorkerHandle<T> {
    stop: StopToken,
    join: Option<JoinHandle<T>>,
    name: String,
}

impl<T> WorkerHandle<T> {
    /// Spawns a new named worker thread.
    ///
    /// The closure receives a [`StopToken`] that becomes stopped when
    /// [`WorkerHandle::stop`] is called. The closure should poll the token
    /// or block on channels that are closed as part of shutdown.
    ///
    /// # Errors
    ///
    /// Returns [`WorkerResult::SpawnFailed`] if the OS failed to spawn the
    /// thread. In that case the stop token is still usable but no worker is
    /// running.
    pub fn spawn<F>(name: impl Into<String>, f: F) -> Self
    where
        F: FnOnce(StopToken) -> T + Send + 'static,
        T: Send + 'static,
    {
        let name = name.into();
        let stop = StopToken::new();
        let stop_for_thread = stop.clone();

        let join = thread::Builder::new()
            .name(name.clone())
            .spawn(move || f(stop_for_thread));

        Self {
            stop,
            join: join.ok(),
            name,
        }
    }

    /// Returns the worker's thread name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a clone of the stop token shared with the worker.
    #[must_use]
    pub fn stop_token(&self) -> StopToken {
        self.stop.clone()
    }

    /// Requests the worker to stop at the next opportunity.
    pub fn stop(&self) {
        self.stop.stop();
    }

    /// Returns `true` if stop has been requested.
    #[must_use]
    pub fn is_stopped(&self) -> bool {
        self.stop.is_stopped()
    }

    /// Returns whether the underlying thread has already exited.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.join
            .as_ref()
            .is_some_and(std::thread::JoinHandle::is_finished)
    }

    /// Joins the worker thread and returns its result.
    ///
    /// This call blocks until the worker returns or panics. Callers that need
    /// a timeout should arrange it externally (for example with a separate
    /// watchdog thread) because Rust standard threads do not support timed
    /// joins.
    ///
    /// After this call returns, the handle is consumed and cannot be reused.
    pub fn join(mut self) -> WorkerResult<T> {
        match self.join.take() {
            Some(handle) => match handle.join() {
                Ok(result) => WorkerResult::Completed(result),
                Err(_) => WorkerResult::Panicked,
            },
            None => WorkerResult::SpawnFailed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    use crate::slot::{LatestSlot, ReadResult};

    #[test]
    fn worker_returns_value() {
        let handle = WorkerHandle::spawn("returns-value", |_stop| 42);
        assert_eq!(handle.join(), WorkerResult::Completed(42));
    }

    #[test]
    fn worker_stops_via_token() {
        let handle = WorkerHandle::spawn("stops-via-token", |stop| {
            while !stop.is_stopped() {
                std::thread::sleep(Duration::from_millis(1));
            }
            "stopped"
        });

        std::thread::sleep(Duration::from_millis(10));
        handle.stop();
        assert_eq!(handle.join(), WorkerResult::Completed("stopped"));
    }

    #[test]
    fn worker_panic_is_detected() {
        let handle = WorkerHandle::spawn::<fn(StopToken) -> ()>("panics", |_stop| {
            panic!("expected test panic");
        });

        assert_eq!(handle.join(), WorkerResult::Panicked);
    }

    #[test]
    fn worker_shutdown_closes_slot_and_joins() {
        let slot: Arc<LatestSlot<i32>> = Arc::new(LatestSlot::new());
        let slot_for_worker = Arc::clone(&slot);

        let handle = WorkerHandle::spawn("slot-consumer", move |stop| {
            let mut last_gen = 0;
            loop {
                if stop.is_stopped() {
                    return "stop-polled";
                }
                match slot_for_worker.wait_read_after(last_gen, Duration::from_millis(50)) {
                    Some(ReadResult::New(value)) => {
                        last_gen = value as u64;
                    }
                    Some(ReadResult::Closed) => return "slot-closed",
                    None => {}
                }
            }
        });

        std::thread::sleep(Duration::from_millis(20));
        slot.close();
        assert_eq!(handle.join(), WorkerResult::Completed("slot-closed"));
    }

    #[test]
    fn stop_token_is_shared() {
        let handle = WorkerHandle::spawn("shared-token", |stop| {
            while !stop.is_stopped() {
                std::thread::sleep(Duration::from_millis(1));
            }
            "done"
        });

        let token = handle.stop_token();
        token.stop();
        assert_eq!(handle.join(), WorkerResult::Completed("done"));
    }
}
