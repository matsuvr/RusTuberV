//! Fixed-size metrics collection for acceptance testing.
//!
//! Provides ring-buffer based statistics for latency, rates, and counters.
//! All values use monotonic timestamps within the same clock domain.

/// Fixed-size ring buffer for computing running statistics.
#[derive(Clone, Debug)]
pub struct FixedStats {
    buffer: Vec<f64>,
    capacity: usize,
    write_pos: usize,
    count: usize,
}

impl FixedStats {
    /// Create a new stats collector with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0.0; capacity],
            capacity,
            write_pos: 0,
            count: 0,
        }
    }

    /// Records a new value. A zero-capacity collector silently ignores values.
    pub fn record(&mut self, value: f64) {
        let Some(slot) = self.buffer.get_mut(self.write_pos) else {
            return; // capacity == 0
        };
        *slot = value;
        // The slot existed, so `capacity >= 1`; the modulo cannot divide by zero.
        self.write_pos = (self.write_pos + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    /// Number of values recorded (up to capacity).
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Mean of recorded values.
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let sum: f64 = self.buffer.iter().take(self.count).sum();
        sum / self.count as f64
    }

    /// Minimum of recorded values.
    #[must_use]
    pub fn min(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.buffer
            .iter()
            .take(self.count)
            .copied()
            .fold(f64::INFINITY, f64::min)
    }

    /// Maximum of recorded values.
    #[must_use]
    pub fn max(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.buffer
            .iter()
            .take(self.count)
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Percentile (0.0 to 1.0) of recorded values.
    /// Uses nearest-rank method.
    #[must_use]
    pub fn percentile(&self, p: f64) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let mut sorted: Vec<f64> = self.buffer.iter().take(self.count).copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let rank = (p * self.count as f64).ceil() as usize;
        let index = rank.saturating_sub(1).min(self.count - 1);
        // Invariant: `index < self.count == sorted.len()` by the clamps above.
        #[allow(clippy::indexing_slicing)]
        sorted[index]
    }

    /// p50 (median).
    #[must_use]
    pub fn p50(&self) -> f64 {
        self.percentile(0.5)
    }

    /// p95.
    #[must_use]
    pub fn p95(&self) -> f64 {
        self.percentile(0.95)
    }

    /// Reset all recorded values.
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
        self.count = 0;
    }
}

/// Pipeline stage timestamps for latency measurement.
///
/// All timestamps are in the same monotonic clock domain (nanoseconds).
#[derive(Clone, Copy, Debug, Default)]
pub struct StageTimestamps {
    /// Frame captured from camera (nanoseconds, monotonic).
    pub captured_ns: u64,
    /// Inference completed (nanoseconds, monotonic).
    pub inference_done_ns: u64,
    /// Control frame produced (nanoseconds, monotonic).
    pub control_frame_ns: u64,
    /// Avatar pose applied (nanoseconds, monotonic).
    pub applied_ns: u64,
}

impl StageTimestamps {
    /// Capture-to-apply latency in milliseconds.
    #[must_use]
    pub fn capture_to_apply_ms(&self) -> f64 {
        if self.applied_ns >= self.captured_ns {
            (self.applied_ns - self.captured_ns) as f64 / 1_000_000.0
        } else {
            0.0
        }
    }

    /// Inference duration in milliseconds.
    #[must_use]
    pub fn inference_ms(&self) -> f64 {
        if self.inference_done_ns >= self.captured_ns {
            (self.inference_done_ns - self.captured_ns) as f64 / 1_000_000.0
        } else {
            0.0
        }
    }

    /// Apply delay (control frame to apply) in milliseconds.
    #[must_use]
    pub fn apply_delay_ms(&self) -> f64 {
        if self.applied_ns >= self.control_frame_ns {
            (self.applied_ns - self.control_frame_ns) as f64 / 1_000_000.0
        } else {
            0.0
        }
    }
}

/// Rate counter using a time window.
#[derive(Clone, Debug)]
pub struct RateCounter {
    window_ns: u64,
    events: Vec<u64>,
}

impl RateCounter {
    /// Create a rate counter with the given window in nanoseconds.
    #[must_use]
    pub fn new(window_ns: u64) -> Self {
        Self {
            window_ns,
            events: Vec::new(),
        }
    }

    /// Record an event at the given timestamp.
    pub fn record(&mut self, timestamp_ns: u64) {
        self.events.push(timestamp_ns);
        self.prune(timestamp_ns);
    }

    /// Current rate in events per second.
    #[must_use]
    pub fn rate_hz(&mut self, now_ns: u64) -> f64 {
        self.prune(now_ns);
        if self.events.is_empty() {
            return 0.0;
        }
        let window = if now_ns > self.window_ns {
            self.window_ns
        } else {
            now_ns
        };
        if window == 0 {
            return 0.0;
        }
        self.events.len() as f64 * 1_000_000_000.0 / window as f64
    }

    /// Remove events outside the window.
    fn prune(&mut self, now_ns: u64) {
        let cutoff = now_ns.saturating_sub(self.window_ns);
        self.events.retain(|&t| t >= cutoff);
    }

    /// Reset the counter.
    pub fn reset(&mut self) {
        self.events.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_stats_empty() {
        let stats = FixedStats::new(10);
        assert_eq!(stats.count(), 0);
        assert_eq!(stats.mean(), 0.0);
        assert_eq!(stats.p50(), 0.0);
        assert_eq!(stats.p95(), 0.0);
    }

    #[test]
    fn fixed_stats_single_value() {
        let mut stats = FixedStats::new(10);
        stats.record(5.0);
        assert_eq!(stats.count(), 1);
        assert_eq!(stats.mean(), 5.0);
        assert_eq!(stats.min(), 5.0);
        assert_eq!(stats.max(), 5.0);
        assert_eq!(stats.p50(), 5.0);
    }

    #[test]
    fn fixed_stats_multiple_values() {
        let mut stats = FixedStats::new(10);
        for i in 1..=10 {
            stats.record(i as f64);
        }
        assert_eq!(stats.count(), 10);
        assert_eq!(stats.mean(), 5.5);
        assert_eq!(stats.min(), 1.0);
        assert_eq!(stats.max(), 10.0);
        assert_eq!(stats.p50(), 5.0);
        assert!((stats.p95() - 10.0).abs() < 0.01);
    }

    #[test]
    fn fixed_stats_ring_buffer() {
        let mut stats = FixedStats::new(5);
        for i in 1..=10 {
            stats.record(i as f64);
        }
        // Only last 5 values: 6, 7, 8, 9, 10
        assert_eq!(stats.count(), 5);
        assert_eq!(stats.mean(), 8.0);
        assert_eq!(stats.min(), 6.0);
        assert_eq!(stats.max(), 10.0);
    }

    #[test]
    fn fixed_stats_reset() {
        let mut stats = FixedStats::new(10);
        stats.record(5.0);
        stats.reset();
        assert_eq!(stats.count(), 0);
        assert_eq!(stats.mean(), 0.0);
    }

    #[test]
    fn stage_timestamps_capture_to_apply() {
        let ts = StageTimestamps {
            captured_ns: 1_000_000_000,
            inference_done_ns: 1_020_000_000,
            control_frame_ns: 1_025_000_000,
            applied_ns: 1_030_000_000,
        };
        assert!((ts.capture_to_apply_ms() - 30.0).abs() < 0.01);
        assert!((ts.inference_ms() - 20.0).abs() < 0.01);
        assert!((ts.apply_delay_ms() - 5.0).abs() < 0.01);
    }

    #[test]
    fn stage_timestamps_zero_on_invalid() {
        let ts = StageTimestamps {
            captured_ns: 100,
            inference_done_ns: 50, // before capture
            control_frame_ns: 0,
            applied_ns: 0,
        };
        assert_eq!(ts.inference_ms(), 0.0);
        assert_eq!(ts.apply_delay_ms(), 0.0);
    }

    #[test]
    fn rate_counter_empty() {
        let mut counter = RateCounter::new(1_000_000_000); // 1 second window
        assert_eq!(counter.rate_hz(0), 0.0);
    }

    #[test]
    fn rate_counter_basic() {
        let mut counter = RateCounter::new(1_000_000_000); // 1 second window
        // Record 10 events over 1 second
        for i in 0..10 {
            counter.record(i * 100_000_000);
        }
        let rate = counter.rate_hz(1_000_000_000);
        assert!((rate - 10.0).abs() < 0.1, "expected ~10 Hz, got {rate}");
    }

    #[test]
    fn rate_counter_prunes_old_events() {
        let mut counter = RateCounter::new(1_000_000_000);
        counter.record(0); // will be pruned (before cutoff at 1s)
        counter.record(1_500_000_000); // within window at now=2s
        // Now at 2 seconds, cutoff is 1s. Event at 0 is pruned, event at 1.5s remains.
        let rate = counter.rate_hz(2_000_000_000);
        assert!(
            (rate - 1.0).abs() < 0.1,
            "expected ~1 Hz after prune, got {rate}"
        );
    }

    #[test]
    fn rate_counter_reset() {
        let mut counter = RateCounter::new(1_000_000_000);
        counter.record(0);
        counter.reset();
        assert_eq!(counter.rate_hz(0), 0.0);
    }
}
