//! Fixed-size inference timing and drop accounting.
//!
//! This module records per-stage durations in fixed-size ring buffers and
//! aggregates drop counters. It is used by the inference worker to expose
//! runtime metrics without keeping unbounded history.

use std::time::Duration;

/// Number of duration samples retained per stage.
const RING_SIZE: usize = 512;

/// Inference pipeline stage for timing accounting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InferenceStage {
    /// Time spent waiting for a new input frame.
    Wait,
    /// Frame preprocessing (resize, normalize, layout conversion).
    Preprocess,
    /// Face detection.
    Detector,
    /// Detector-box crop preprocessing.
    Crop,
    /// Landmark regression.
    Landmark,
    /// Output tensor decoding to observations.
    Decode,
    /// End-to-end frame processing time.
    Total,
}

impl InferenceStage {
    /// Number of distinct inference stages.
    pub const COUNT: usize = 7;

    /// Array containing all stages in declaration order.
    pub const ALL: [Self; Self::COUNT] = [
        Self::Wait,
        Self::Preprocess,
        Self::Detector,
        Self::Crop,
        Self::Landmark,
        Self::Decode,
        Self::Total,
    ];

    const fn index(self) -> usize {
        match self {
            Self::Wait => 0,
            Self::Preprocess => 1,
            Self::Detector => 2,
            Self::Crop => 3,
            Self::Landmark => 4,
            Self::Decode => 5,
            Self::Total => 6,
        }
    }
}

/// Fixed-size ring buffer for stage duration samples.
#[derive(Clone, Debug, PartialEq)]
pub struct StageTimingRing<const N: usize> {
    samples: [Duration; N],
    head: usize,
    count: u64,
    min_ns: u64,
    max_ns: u64,
    sum_ns: u128,
}

impl<const N: usize> Default for StageTimingRing<N> {
    fn default() -> Self {
        Self {
            samples: [Duration::default(); N],
            head: 0,
            count: 0,
            min_ns: 0,
            max_ns: 0,
            sum_ns: 0,
        }
    }
}

impl<const N: usize> StageTimingRing<N> {
    /// Creates an empty ring.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a duration sample.
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub fn record(&mut self, duration: Duration) {
        let ns = duration.as_nanos() as u64;
        self.samples[self.head] = duration;
        self.head = (self.head + 1) % N;
        self.count = self.count.saturating_add(1);
        if self.count == 1 {
            self.min_ns = ns;
            self.max_ns = ns;
        } else {
            self.min_ns = self.min_ns.min(ns);
            self.max_ns = self.max_ns.max(ns);
        }
        self.sum_ns = self.sum_ns.saturating_add(ns as u128);
    }

    /// Returns a snapshot of the recorded samples.
    #[must_use]
    pub fn snapshot(&self) -> StageTimingSnapshot {
        let mut retained = self.retained_samples();
        let (p50_ns, p95_ns) = if retained.is_empty() {
            (0, 0)
        } else {
            retained.sort_unstable();
            (nearest_rank(&retained, 0.50), nearest_rank(&retained, 0.95))
        };
        StageTimingSnapshot {
            count: self.count,
            min_ns: if self.count == 0 { 0 } else { self.min_ns },
            max_ns: if self.count == 0 { 0 } else { self.max_ns },
            mean_ns: if self.count == 0 {
                0
            } else {
                (self.sum_ns / u128::from(self.count)) as u64
            },
            p50_ns,
            p95_ns,
        }
    }

    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    fn retained_samples(&self) -> Vec<u64> {
        let retained = self.count.min(N as u64) as usize;
        if retained == 0 {
            return Vec::new();
        }
        let first = if self.count >= N as u64 { self.head } else { 0 };
        (0..retained)
            .map(|offset| self.samples[(first + offset) % N].as_nanos() as u64)
            .collect()
    }

    /// Returns the fixed capacity of the ring.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        N
    }
}

/// Snapshot of timing statistics for a single stage.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StageTimingSnapshot {
    /// Number of samples recorded.
    pub count: u64,
    /// Minimum duration in nanoseconds.
    pub min_ns: u64,
    /// Maximum duration in nanoseconds.
    pub max_ns: u64,
    /// Mean duration in nanoseconds.
    pub mean_ns: u64,
    /// p50 duration in nanoseconds over the retained bounded samples.
    pub p50_ns: u64,
    /// p95 duration in nanoseconds over the retained bounded samples.
    pub p95_ns: u64,
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn nearest_rank(sorted: &[u64], percentile: f64) -> u64 {
    let rank = (percentile * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

/// Drop and skip counters for the inference pipeline.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DropCounters {
    /// Frames overwritten in the input slot before being read.
    pub input_overwritten: u64,
    /// Frames read but skipped before inference (duplicates or detector cadence).
    pub skipped_sequence: u64,
    /// Frames that completed inference.
    pub processed: u64,
    /// Frames for which the detector or landmark validity policy found no face.
    pub no_face: u64,
    /// Frames overwritten in the output slot before being consumed.
    pub output_overwritten: u64,
}

/// Public snapshot of inference metrics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct InferenceMetrics {
    /// Timing snapshots for each pipeline stage, indexed by [`InferenceStage`].
    pub stage_timings: [StageTimingSnapshot; InferenceStage::COUNT],
    /// Drop and skip counters.
    pub drops: DropCounters,
}

impl InferenceMetrics {
    /// Returns the timing snapshot for `stage`.
    #[must_use]
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub fn stage(&self, stage: InferenceStage) -> StageTimingSnapshot {
        self.stage_timings[stage.index()]
    }
}

/// Mutable internal metrics state.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct InferenceMetricsState {
    rings: [StageTimingRing<RING_SIZE>; InferenceStage::COUNT],
    drops: DropCounters,
    /// Cached snapshot, recomputed lazily after any recording mutation.
    cached: InferenceMetrics,
    dirty: bool,
}

impl InferenceMetricsState {
    /// Records a duration sample for `stage`.
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub(crate) fn record_stage_duration(&mut self, stage: InferenceStage, duration: Duration) {
        self.rings[stage.index()].record(duration);
        self.dirty = true;
    }

    /// Records frames overwritten in the input slot.
    pub(crate) fn record_input_overwritten(&mut self, count: u64) {
        self.drops.input_overwritten = self.drops.input_overwritten.saturating_add(count);
        self.dirty = true;
    }

    /// Records a frame that was skipped before inference.
    pub(crate) fn record_skipped_sequence(&mut self) {
        self.drops.skipped_sequence = self.drops.skipped_sequence.saturating_add(1);
        self.dirty = true;
    }

    /// Records a frame that completed inference.
    pub(crate) fn record_processed(&mut self) {
        self.drops.processed = self.drops.processed.saturating_add(1);
        self.dirty = true;
    }

    /// Records an ordinary no-face frame.
    pub(crate) fn record_no_face(&mut self) {
        self.drops.no_face = self.drops.no_face.saturating_add(1);
        self.dirty = true;
    }

    /// Records frames overwritten in the output slot.
    pub(crate) fn record_output_overwritten(&mut self, count: u64) {
        self.drops.output_overwritten = self.drops.output_overwritten.saturating_add(count);
        self.dirty = true;
    }

    /// Returns a snapshot of the current metrics.
    ///
    /// The snapshot is cached and only recomputed after a recording mutation,
    /// so repeated reads between frames do not re-sort the retained samples.
    #[must_use]
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub(crate) fn snapshot(&mut self) -> InferenceMetrics {
        if self.dirty {
            let mut stage_timings = [StageTimingSnapshot::default(); InferenceStage::COUNT];
            for (i, ring) in self.rings.iter().enumerate() {
                stage_timings[i] = ring.snapshot();
            }
            self.cached = InferenceMetrics {
                stage_timings,
                drops: self.drops,
            };
            self.dirty = false;
        }
        self.cached
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_ring_snapshot_is_zero() {
        let ring = StageTimingRing::<4>::new();
        let snap = ring.snapshot();
        assert_eq!(snap.count, 0);
        assert_eq!(snap.min_ns, 0);
        assert_eq!(snap.max_ns, 0);
        assert_eq!(snap.mean_ns, 0);
    }

    #[test]
    fn ring_records_samples() {
        let mut ring = StageTimingRing::<4>::new();
        ring.record(Duration::from_nanos(100));
        ring.record(Duration::from_nanos(200));
        ring.record(Duration::from_nanos(300));
        let snap = ring.snapshot();
        assert_eq!(snap.count, 3);
        assert_eq!(snap.min_ns, 100);
        assert_eq!(snap.max_ns, 300);
        assert_eq!(snap.mean_ns, 200);
    }

    #[test]
    fn ring_overwrites_old_samples() {
        let mut ring = StageTimingRing::<4>::new();
        for ns in [100, 200, 300, 400, 500] {
            ring.record(Duration::from_nanos(ns));
        }
        let snap = ring.snapshot();
        assert_eq!(snap.count, 5);
        // Min/max/mean reflect all historical samples, not just the ring contents.
        assert_eq!(snap.min_ns, 100);
        assert_eq!(snap.max_ns, 500);
        assert_eq!(snap.mean_ns, 300);
    }

    #[test]
    fn ring_capacity_is_fixed() {
        let ring = StageTimingRing::<8>::new();
        assert_eq!(ring.capacity(), 8);
    }

    #[test]
    fn ring_percentiles_use_only_the_bounded_retained_window() {
        let mut ring = StageTimingRing::<4>::new();
        for ns in [100, 200, 300, 400, 500] {
            ring.record(Duration::from_nanos(ns));
        }
        let snap = ring.snapshot();
        assert_eq!(snap.p50_ns, 300);
        assert_eq!(snap.p95_ns, 500);
    }

    #[test]
    fn metrics_state_records_and_snapshots() {
        let mut state = InferenceMetricsState::default();
        state.record_stage_duration(InferenceStage::Wait, Duration::from_millis(5));
        state.record_stage_duration(InferenceStage::Preprocess, Duration::from_millis(2));
        state.record_stage_duration(InferenceStage::Detector, Duration::from_millis(8));
        state.record_input_overwritten(3);
        state.record_skipped_sequence();
        state.record_skipped_sequence();
        state.record_processed();
        state.record_no_face();
        state.record_output_overwritten(1);

        let snap = state.snapshot();
        assert_eq!(snap.stage(InferenceStage::Wait).count, 1);
        assert_eq!(snap.stage(InferenceStage::Preprocess).mean_ns, 2_000_000);
        assert_eq!(snap.stage(InferenceStage::Landmark).count, 0);
        assert_eq!(snap.drops.input_overwritten, 3);
        assert_eq!(snap.drops.skipped_sequence, 2);
        assert_eq!(snap.drops.processed, 1);
        assert_eq!(snap.drops.no_face, 1);
        assert_eq!(snap.drops.output_overwritten, 1);
    }

    #[test]
    fn drop_counters_saturate() {
        let mut state = InferenceMetricsState::default();
        state.drops.input_overwritten = u64::MAX;
        state.record_input_overwritten(1);
        assert_eq!(state.drops.input_overwritten, u64::MAX);
    }

    #[test]
    fn stage_enum_index_matches_array_order() {
        for (i, stage) in InferenceStage::ALL.iter().enumerate() {
            assert_eq!(stage.index(), i);
        }
    }
}
