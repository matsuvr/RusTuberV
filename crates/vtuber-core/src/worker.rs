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
}

/// Handle to a named worker thread with a cooperative stop token.
///
/// The handle always owns a real thread's [`JoinHandle`]: it can only be
/// created when the OS accepted the spawn request. Dropping the handle
/// without calling [`WorkerHandle::join`] detaches the thread: it continues
/// running, its result can no longer be joined, and no stop is requested.
/// Owners must request stop and join explicitly when completion matters.
#[derive(Debug)]
pub struct WorkerHandle<T> {
    stop: StopToken,
    join: JoinHandle<T>,
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
    /// Returns the OS [`std::io::Error`] if the thread could not be spawned.
    /// No handle exists in that case, so callers must not retain any worker
    /// state for a failed spawn.
    pub fn spawn<F>(name: impl Into<String>, f: F) -> std::io::Result<Self>
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
        worker_handle_from_result(name, stop, join)
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
        self.join.is_finished()
    }

    /// Joins the worker thread and returns its result.
    ///
    /// Blocks until the worker returns or panics. Joining does not request a
    /// stop: call [`Self::stop`] first for a worker that waits for that request.
    /// Returns [`WorkerResult::Panicked`] rather than resuming a worker panic.
    ///
    /// After this call returns, the handle is consumed and cannot be reused.
    #[must_use]
    pub fn join(self) -> WorkerResult<T> {
        match self.join.join() {
            Ok(result) => WorkerResult::Completed(result),
            Err(_) => WorkerResult::Panicked,
        }
    }
}

/// Assembles a [`WorkerHandle`] from a raw spawn result, propagating the OS
/// error when the thread was not started.
fn worker_handle_from_result<T>(
    name: String,
    stop: StopToken,
    result: std::io::Result<JoinHandle<T>>,
) -> std::io::Result<WorkerHandle<T>> {
    let join = result?;
    Ok(WorkerHandle { stop, join, name })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    use crate::slot::{LatestSlot, ReadResult};

    #[test]
    fn worker_returns_value() {
        let handle = WorkerHandle::spawn("returns-value", |_stop| 42).expect("spawn");
        assert_eq!(handle.join(), WorkerResult::Completed(42));
    }

    #[test]
    fn worker_stops_via_token() {
        let handle = WorkerHandle::spawn("stops-via-token", |stop| {
            while !stop.is_stopped() {
                std::thread::sleep(Duration::from_millis(1));
            }
            "stopped"
        })
        .expect("spawn");

        std::thread::sleep(Duration::from_millis(10));
        handle.stop();
        assert_eq!(handle.join(), WorkerResult::Completed("stopped"));
    }

    #[test]
    fn worker_panic_is_detected() {
        let handle = WorkerHandle::spawn::<fn(StopToken) -> ()>("panics", |_stop| {
            panic!("expected test panic");
        })
        .expect("spawn");

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
                    Some(ReadResult::New { generation, value }) => {
                        last_gen = generation;
                        let _ = value;
                    }
                    Some(ReadResult::Closed) => return "slot-closed",
                    None => {}
                }
            }
        })
        .expect("spawn");

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
        })
        .expect("spawn");

        let token = handle.stop_token();
        token.stop();
        assert_eq!(handle.join(), WorkerResult::Completed("done"));
    }

    #[test]
    fn failed_spawn_returns_the_io_error_and_no_handle() {
        let error = std::io::Error::other("simulated OS spawn failure");
        let result: std::io::Result<WorkerHandle<()>> =
            worker_handle_from_result("never-started".to_string(), StopToken::new(), Err(error));
        match result {
            Err(error) => assert_eq!(error.to_string(), "simulated OS spawn failure"),
            Ok(_) => panic!("expected the spawn error"),
        }
    }
}
