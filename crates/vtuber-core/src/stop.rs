//! Cooperative stop token for worker threads.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Token used to request graceful worker shutdown.
#[derive(Clone, Debug)]
pub struct StopToken {
    inner: Arc<AtomicBool>,
}

impl Default for StopToken {
    fn default() -> Self {
        Self::new()
    }
}

impl StopToken {
    /// Creates a new stop token that is not stopped.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Requests the worker to stop at the next opportunity.
    pub fn stop(&self) {
        self.inner.store(true, Ordering::SeqCst);
    }

    /// Returns `true` if stop was requested.
    #[must_use]
    pub fn is_stopped(&self) -> bool {
        self.inner.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_not_stopped() {
        let token = StopToken::new();
        assert!(!token.is_stopped());
    }

    #[test]
    fn stop_is_visible() {
        let token = StopToken::new();
        token.stop();
        assert!(token.is_stopped());
    }

    #[test]
    fn clone_shares_state() {
        let token = StopToken::new();
        let cloned = token.clone();
        token.stop();
        assert!(cloned.is_stopped());
    }
}
