//! Time-based smoothing for the neutral-relative head translation signal.
//!
//! The face-transform translation carries millimeter-scale per-observation
//! noise, and the body-motion consumer applies it directly (root offset and
//! torso lean). Without temporal filtering, that noise reaches the avatar as
//! a step signal at the observation rate, which the spring bones then
//! faithfully amplify into visible hair and cloth jitter. This filter runs on
//! every pipeline tick, so its output is a continuous signal at the consumer
//! frame rate — the translation counterpart of the rotation-side
//! [`HeadRotationFilter`](super::HeadRotationFilter).

use vtuber_core::types::{HeadTranslationSignal, MonoTimeNs};

/// Parameters for the translation filter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TranslationFilterParams {
    /// Smoothing time constant in seconds.
    ///
    /// One constant covers the whole channel. A sitting subject's monocular
    /// head translation wobbles by several centimeters at around 1 Hz, and the
    /// body-follow filter downstream cannot remove what it cannot see, so the
    /// observation itself is smoothed harder than the rotation filter. It is
    /// not stretched by how fast the observation moves: a real lean is not an
    /// outlier, and an unavailable observation is already passed through
    /// instead of being integrated.
    pub time_constant_sec: f32,
    /// Maximum accepted delta-time in seconds.
    ///
    /// Larger gaps are clamped so that a stale observation cannot fully snap
    /// the output.
    pub max_dt_sec: f32,
}

impl Default for TranslationFilterParams {
    fn default() -> Self {
        Self {
            time_constant_sec: 0.10,
            max_dt_sec: 0.5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct FilterState {
    translation: HeadTranslationSignal,
    last_time: MonoTimeNs,
}

/// Exponential smoothing filter for the head translation signal.
///
/// The first available observation snaps the filter to its target so a fresh
/// session does not glide in from a stale position. While the observation is
/// unavailable the filter keeps its state untouched and passes the
/// unavailable signal through; the next available observation then resumes
/// from the stored value instead of snapping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TranslationFilter {
    params: TranslationFilterParams,
    state: Option<FilterState>,
}

impl TranslationFilter {
    /// Creates a new filter with the given parameters.
    #[must_use]
    pub fn new(params: TranslationFilterParams) -> Self {
        Self {
            params,
            state: None,
        }
    }

    /// Resets the filter, discarding all state.
    pub fn reset(&mut self) {
        self.state = None;
    }

    /// Updates the filter with a new target translation.
    ///
    /// An unavailable target passes through unchanged while the internal
    /// state is retained, so the filtered output never steps across an
    /// observation gap. The returned signal carries the target's availability
    /// state.
    #[must_use]
    pub fn update(
        &mut self,
        target: HeadTranslationSignal,
        timestamp: MonoTimeNs,
    ) -> HeadTranslationSignal {
        if !target.is_available() {
            return HeadTranslationSignal::UNAVAILABLE;
        }

        let Some(state) = self.state else {
            self.state = Some(FilterState {
                translation: target,
                last_time: timestamp,
            });
            return target;
        };

        let dt_ns = timestamp.0.saturating_sub(state.last_time.0);
        let dt_sec = ((dt_ns as f32) / 1_000_000_000.0).min(self.params.max_dt_sec);
        if dt_sec <= 0.0 {
            return state.translation;
        }

        let tau = self.params.time_constant_sec.max(f32::EPSILON);
        let alpha = 1.0 - (-dt_sec / tau).exp();
        let blend_axis = |current: f32, goal: f32| current + (goal - current) * alpha;
        let coordinates = (
            blend_axis(state.translation.x_meters(), target.x_meters()),
            blend_axis(state.translation.y_meters(), target.y_meters()),
            blend_axis(state.translation.z_meters(), target.z_meters()),
        );
        let smoothed = match target.state() {
            vtuber_core::HeadTranslationState::Tracked => {
                HeadTranslationSignal::tracked(coordinates.0, coordinates.1, coordinates.2)
            }
            vtuber_core::HeadTranslationState::Degraded => {
                HeadTranslationSignal::degraded(coordinates.0, coordinates.1, coordinates.2)
            }
            vtuber_core::HeadTranslationState::Unavailable => HeadTranslationSignal::UNAVAILABLE,
        };
        self.state = Some(FilterState {
            translation: smoothed,
            last_time: timestamp,
        });
        smoothed
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
    use vtuber_core::types::HeadTranslationState;

    const TAU: f32 = 0.1;
    fn params() -> TranslationFilterParams {
        TranslationFilterParams {
            time_constant_sec: TAU,
            max_dt_sec: 0.5,
        }
    }

    #[test]
    fn the_fraction_followed_does_not_depend_on_the_step_size() {
        // The response is linear in the error, so one observation moves the
        // output by the same fraction whatever the displacement was. A deep
        // lean and a small wobble are therefore treated the same way, and a
        // large displacement is not mistaken for a jump that needs absorbing.
        let step = |value: f32| HeadTranslationSignal::tracked(value, 0.0, 0.0);
        let followed = |distance: f32| {
            let mut filter = TranslationFilter::new(params());
            let _ = filter.update(step(0.0), MonoTimeNs(0));
            filter
                .update(step(distance), MonoTimeNs(16_666_667))
                .x_meters()
                / distance
        };
        let wobble = followed(0.005);
        let lean = followed(0.3);
        assert!(
            (wobble - lean).abs() < 1.0e-5,
            "a deep lean must be followed exactly like a wobble: {wobble} vs {lean}"
        );
    }

    #[test]
    fn first_available_observation_snaps() {
        let mut filter = TranslationFilter::new(params());
        let target = HeadTranslationSignal::tracked(0.05, -0.02, 0.1);
        let output = filter.update(target, MonoTimeNs(0));
        assert_eq!(output, target);
    }

    #[test]
    fn held_re_feed_converges_toward_the_target() {
        let mut filter = TranslationFilter::new(params());
        let start = HeadTranslationSignal::tracked(0.0, 0.0, 0.0);
        let _ = filter.update(start, MonoTimeNs(0));

        let target = HeadTranslationSignal::tracked(0.1, 0.0, 0.0);
        let first = filter.update(target, MonoTimeNs(16_666_667));
        let second = filter.update(target, MonoTimeNs(33_333_333));
        let third = filter.update(target, MonoTimeNs(50_000_000));

        let distance = |value: HeadTranslationSignal| (value.x_meters() - 0.1).abs();
        assert!(distance(first) > 0.0);
        assert!(distance(first) > distance(second));
        assert!(distance(second) > distance(third));
        assert_eq!(second.state(), HeadTranslationState::Tracked);
    }

    #[test]
    fn unavailable_target_passes_through_and_resumes_smoothly() {
        let mut filter = TranslationFilter::new(params());
        let start = HeadTranslationSignal::tracked(0.0, 0.0, 0.0);
        let _ = filter.update(start, MonoTimeNs(0));

        assert_eq!(
            filter.update(HeadTranslationSignal::UNAVAILABLE, MonoTimeNs(16_666_667)),
            HeadTranslationSignal::UNAVAILABLE
        );

        let target = HeadTranslationSignal::tracked(0.1, 0.0, 0.0);
        let resumed = filter.update(target, MonoTimeNs(33_333_333));
        assert!(
            resumed.x_meters() > 0.0 && resumed.x_meters() < 0.1,
            "resume must continue from the stored value, got {}",
            resumed.x_meters()
        );
    }

    #[test]
    fn degraded_target_keeps_its_state_label() {
        let mut filter = TranslationFilter::new(params());
        let start = HeadTranslationSignal::tracked(0.0, 0.0, 0.0);
        let _ = filter.update(start, MonoTimeNs(0));

        let degraded = HeadTranslationSignal::degraded(0.1, 0.0, 0.0);
        let output = filter.update(degraded, MonoTimeNs(16_666_667));
        assert_eq!(output.state(), HeadTranslationState::Degraded);
    }

    #[test]
    fn reset_discards_state_so_the_next_target_snaps() {
        let mut filter = TranslationFilter::new(params());
        let _ = filter.update(HeadTranslationSignal::tracked(0.0, 0.0, 0.0), MonoTimeNs(0));
        filter.reset();

        let target = HeadTranslationSignal::tracked(0.1, 0.0, 0.0);
        assert_eq!(filter.update(target, MonoTimeNs(16_666_667)), target);
    }
}
