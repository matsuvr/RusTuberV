//! Process-wide monotonic timestamps shared by all runtime stages.

use std::sync::OnceLock;
use std::time::Instant;

use crate::types::MonoTimeNs;

static EPOCH: OnceLock<Instant> = OnceLock::new();

/// Returns nanoseconds from one process-local monotonic epoch.
#[must_use]
pub fn now() -> MonoTimeNs {
    let epoch = EPOCH.get_or_init(Instant::now);
    MonoTimeNs(epoch.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamps_are_monotonic() {
        let first = now();
        let second = now();
        assert!(second >= first);
    }
}
