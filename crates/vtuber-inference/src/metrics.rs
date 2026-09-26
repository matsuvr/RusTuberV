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
///
/// Internal to this module: the only user is [`InferenceMetricsState`], and
/// there is no call site that needs a different capacity. The capacity is
/// therefore the [`RING_SIZE`] constant, and no public contract can build a
/// ring that `record` would index out of bounds.
#[derive(Clone, Debug, PartialEq)]
struct StageTimingRing {
    samples: [Duration; RING_SIZE],
    /// Next write position: starts at 0 and `record` keeps it below
    /// [`RING_SIZE`], which is also the length of `samples`.
    head: usize,
    count: u64,
    min_ns: u64,
    max_ns: u64,
    sum_ns: u128,
}

impl Default for StageTimingRing {
    fn default() -> Self {
        Self {
            samples: [Duration::default(); RING_SIZE],
            head: 0,
            count: 0,
            min_ns: 0,
            max_ns: 0,
            sum_ns: 0,
        }
    }
}

impl StageTimingRing {
    /// Records a duration sample.
    // Bounds hold by construction: `samples` has RING_SIZE elements, `head`
    // starts at 0, and the two wraparound updates below keep it below
    // RING_SIZE. See the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    fn record(&mut self, duration: Duration) {
        let ns = duration.as_nanos() as u64;
        self.samples[self.head] = duration;
        self.head = (self.head + 1) % RING_SIZE;
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
    fn snapshot(&self) -> StageTimingSnapshot {
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

    // Bounds hold by construction for the same reason as `record`.
    #[allow(clippy::indexing_slicing)]
    fn retained_samples(&self) -> Vec<u64> {
        let retained = self.count.min(RING_SIZE as u64) as usize;
        if retained == 0 {
            return Vec::new();
        }
        let first = if self.count >= RING_SIZE as u64 {
            self.head
        } else {
            0
        };
        (0..retained)
            .map(|offset| self.samples[(first + offset) % RING_SIZE].as_nanos() as u64)
            .collect()
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

/// Publication, per-reader skip and processed-frame counters for inference.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameCounters {
    /// Input publications skipped by this inference reader between observed generations.
    pub input_skipped: u64,
    /// Frames read but skipped before inference (duplicates or detector cadence).
    pub skipped_sequence: u64,
    /// Frames that completed inference.
    pub processed: u64,
    /// Frames for which the detector or landmark validity policy found no face.
    pub no_face: u64,
    /// Retained output values replaced by publication, regardless of reader activity.
    pub output_replacements: u64,
}

/// Public snapshot of inference metrics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct InferenceMetrics {
    /// Timing snapshots for each pipeline stage, indexed by [`InferenceStage`].
    pub stage_timings: [StageTimingSnapshot; InferenceStage::COUNT],
    /// Publication, per-reader skip and processed-frame counters.
    pub frames: FrameCounters,
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
    rings: [StageTimingRing; InferenceStage::COUNT],
    frames: FrameCounters,
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

    /// Records input generations skipped by this reader.
    pub(crate) fn record_input_skipped(&mut self, count: u64) {
        self.frames.input_skipped = self.frames.input_skipped.saturating_add(count);
        self.dirty = true;
    }

    /// Records a frame that was skipped before inference.
    pub(crate) fn record_skipped_sequence(&mut self) {
        self.frames.skipped_sequence = self.frames.skipped_sequence.saturating_add(1);
        self.dirty = true;
    }

    /// Records a frame that completed inference.
    pub(crate) fn record_processed(&mut self) {
        self.frames.processed = self.frames.processed.saturating_add(1);
        self.dirty = true;
    }

    /// Records an ordinary no-face frame.
    pub(crate) fn record_no_face(&mut self) {
        self.frames.no_face = self.frames.no_face.saturating_add(1);
        self.dirty = true;
    }

    /// Records retained output replacements, not reader losses.
    pub(crate) fn record_output_replacements(&mut self, count: u64) {
        self.frames.output_replacements = self.frames.output_replacements.saturating_add(count);
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
                frames: self.frames,
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
        let snap = StageTimingRing::default().snapshot();
        assert_eq!(snap.count, 0);
        assert_eq!(snap.min_ns, 0);
        assert_eq!(snap.max_ns, 0);
        assert_eq!(snap.mean_ns, 0);
        assert_eq!(snap.p50_ns, 0);
        assert_eq!(snap.p95_ns, 0);
    }

    #[test]
    fn ring_records_samples() {
        let mut ring = StageTimingRing::default();
        for ns in [100, 200, 300] {
            ring.record(Duration::from_nanos(ns));
        }
        let snap = ring.snapshot();
        assert_eq!(snap.count, 3);
        assert_eq!(snap.min_ns, 100);
        assert_eq!(snap.max_ns, 300);
        assert_eq!(snap.mean_ns, 200);
        // Below the fixed capacity every sample is retained.
        assert_eq!(snap.p50_ns, 200);
        assert_eq!(snap.p95_ns, 300);
    }

    #[test]
    fn ring_overwrites_the_oldest_samples_once_full() {
        let mut ring = StageTimingRing::default();
        // A distinctive first sample must be the one that gets overwritten.
        ring.record(Duration::from_nanos(999_999));
        for ns in 1..=RING_SIZE as u64 {
            ring.record(Duration::from_nanos(ns));
        }
        let snap = ring.snapshot();
        assert_eq!(snap.count, RING_SIZE as u64 + 1);
        // Min/max/mean reflect all historical samples, not just the ring
        // contents, so the overwritten value is still in min and mean.
        assert_eq!(snap.min_ns, 1);
        assert_eq!(snap.max_ns, 999_999);
        assert_eq!(snap.mean_ns, (999_999 + 131_328) / (RING_SIZE as u64 + 1));
        // The percentile window holds exactly one ring of samples with the
        // oldest dropped, so 999_999 never reaches p50 or p95.
        assert_eq!(snap.p50_ns, 256);
        assert_eq!(snap.p95_ns, 487);
    }

    #[test]
    fn ring_capacity_is_the_module_constant() {
        let mut ring = StageTimingRing::default();
        for _ in 0..RING_SIZE {
            ring.record(Duration::from_nanos(7));
        }
        // Exactly full: the wraparound has not dropped anything yet and the
        // head is back at the start of the fixed buffer.
        let snap = ring.snapshot();
        assert_eq!(snap.count, RING_SIZE as u64);
        assert_eq!(snap.min_ns, 7);
        assert_eq!(snap.p50_ns, 7);
        assert_eq!(snap.p95_ns, 7);
        assert_eq!(ring.head, 0);
    }

    #[test]
    fn metrics_state_records_and_snapshots() {
        let mut state = InferenceMetricsState::default();
        state.record_stage_duration(InferenceStage::Wait, Duration::from_millis(5));
        state.record_stage_duration(InferenceStage::Preprocess, Duration::from_millis(2));
        state.record_stage_duration(InferenceStage::Detector, Duration::from_millis(8));
        state.record_input_skipped(3);
        state.record_skipped_sequence();
        state.record_skipped_sequence();
        state.record_processed();
        state.record_no_face();
        state.record_output_replacements(1);

        let snap = state.snapshot();
        assert_eq!(snap.stage(InferenceStage::Wait).count, 1);
        assert_eq!(snap.stage(InferenceStage::Preprocess).mean_ns, 2_000_000);
        assert_eq!(snap.stage(InferenceStage::Landmark).count, 0);
        assert_eq!(snap.frames.input_skipped, 3);
        assert_eq!(snap.frames.skipped_sequence, 2);
        assert_eq!(snap.frames.processed, 1);
        assert_eq!(snap.frames.no_face, 1);
        assert_eq!(snap.frames.output_replacements, 1);
    }

    #[test]
    fn drop_counters_saturate() {
        let mut state = InferenceMetricsState::default();
        state.frames.input_skipped = u64::MAX;
        state.record_input_skipped(1);
        assert_eq!(state.frames.input_skipped, u64::MAX);
    }

    #[test]
    fn stage_enum_index_matches_array_order() {
        for (i, stage) in InferenceStage::ALL.iter().enumerate() {
            assert_eq!(stage.index(), i);
        }
    }
}
