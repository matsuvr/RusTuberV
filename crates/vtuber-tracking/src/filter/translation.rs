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
    /// Smoothing time constant in seconds while observations continue the
    /// motion the filter is already following.
    ///
    /// Larger values suppress more noise at the cost of a softer body
    /// follow.
    pub time_constant_sec: f32,
    /// Smoothing time constant used when an observation jumps away from the
    /// previous one.
    ///
    /// Never smaller than [`Self::time_constant_sec`].
    pub slow_time_constant_sec: f32,
    /// Observation rate in meters per second at which the response has fully
    /// slowed to [`Self::slow_time_constant_sec`].
    pub jump_rate_meters_per_sec: f32,
    /// Maximum accepted delta-time in seconds.
    ///
    /// Larger gaps are clamped so that a stale observation cannot fully snap
    /// the output.
    pub max_dt_sec: f32,
}

impl Default for TranslationFilterParams {
    fn default() -> Self {
        Self {
            // A sitting subject's monocular head translation wobbles by
            // several centimeters at around 1 Hz; the body-follow filter
            // downstream cannot remove what it cannot see, so the observation
            // itself is smoothed harder than the rotation filter. The fast
            // constant keeps a genuine lean responsive while the slow one
            // absorbs a monocular depth jump.
            time_constant_sec: 0.10,
            slow_time_constant_sec: 0.25,
            jump_rate_meters_per_sec: 1.5,
            max_dt_sec: 0.5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct FilterState {
    translation: HeadTranslationSignal,
    last_time: MonoTimeNs,
    /// The observation the filter last received, used to measure how far the
    /// next one departs from it.
    last_target: HeadTranslationSignal,
    /// When `last_target` first changed, so the departure is measured over the
    /// observation interval rather than over one render tick.
    last_target_time: MonoTimeNs,
}

/// Euclidean distance between two translation observations, in meters.
fn departure_distance(from: HeadTranslationSignal, to: HeadTranslationSignal) -> f32 {
    (to.x_meters - from.x_meters)
        .hypot(to.y_meters - from.y_meters)
        .hypot(to.z_meters - from.z_meters)
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
                last_target: target,
                last_target_time: timestamp,
            });
            return target;
        };

        let dt_ns = timestamp.0.saturating_sub(state.last_time.0);
        let dt_sec = ((dt_ns as f32) / 1_000_000_000.0).min(self.params.max_dt_sec);
        if dt_sec <= 0.0 {
            return state.translation;
        }

        // How fast this observation departs from the previous one. A held
        // sample repeats `last_target`, so it stays on the fast response.
        let observation_changed = target != state.last_target;
        let rate = if observation_changed {
            super::damped::observation_rate(
                departure_distance(state.last_target, target),
                (timestamp.0.saturating_sub(state.last_target_time.0) as f32) * 1.0e-9,
            )
        } else {
            0.0
        };
        let response = super::damped::ResidualResponse::new(
            self.params.time_constant_sec,
            self.params.slow_time_constant_sec,
            self.params.jump_rate_meters_per_sec,
        );
        let tau = response.time_constant_sec(rate);

        let alpha = 1.0 - (-dt_sec / tau).exp();
        let blend_axis = |current: f32, goal: f32| current + (goal - current) * alpha;
        let smoothed = HeadTranslationSignal {
            x_meters: blend_axis(state.translation.x_meters, target.x_meters),
            y_meters: blend_axis(state.translation.y_meters, target.y_meters),
            z_meters: blend_axis(state.translation.z_meters, target.z_meters),
            state: target.state,
        };
        self.state = Some(FilterState {
            translation: smoothed,
            last_time: timestamp,
            last_target: if observation_changed {
                target
            } else {
                state.last_target
            },
            last_target_time: if observation_changed {
                timestamp
            } else {
                state.last_target_time
            },
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
            slow_time_constant_sec: TAU,
            jump_rate_meters_per_sec: 1.5,
            max_dt_sec: 0.5,
        }
    }

    #[test]
    fn a_small_step_uses_the_fast_response_and_a_jump_the_slow_one() {
        let step = |value: f32| HeadTranslationSignal::tracked(value, 0.0, 0.0);
        let run = |target: HeadTranslationSignal| {
            let mut filter = TranslationFilter::new(TranslationFilterParams::default());
            let _ = filter.update(step(0.0), MonoTimeNs(0));
            filter.update(target, MonoTimeNs(16_666_667)).x_meters
        };

        // A millimeter-scale wobble is normal; a 30 cm one-observation change
        // is a monocular depth jump.
        let small = run(step(0.005));
        let jump = run(step(0.3));
        assert!(
            small / 0.005 > (jump / 0.3) * 1.5,
            "a small wobble must follow faster than a depth jump: small={}, jump={}",
            small / 0.005,
            jump / 0.3
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

        let distance = |value: HeadTranslationSignal| (value.x_meters - 0.1).abs();
        assert!(distance(first) > 0.0);
        assert!(distance(first) > distance(second));
        assert!(distance(second) > distance(third));
        assert_eq!(second.state, HeadTranslationState::Tracked);
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
            resumed.x_meters > 0.0 && resumed.x_meters < 0.1,
            "resume must continue from the stored value, got {}",
            resumed.x_meters
        );
    }

    #[test]
    fn degraded_target_keeps_its_state_label() {
        let mut filter = TranslationFilter::new(params());
        let start = HeadTranslationSignal::tracked(0.0, 0.0, 0.0);
        let _ = filter.update(start, MonoTimeNs(0));

        let degraded = HeadTranslationSignal::degraded(0.1, 0.0, 0.0);
        let output = filter.update(degraded, MonoTimeNs(16_666_667));
        assert_eq!(output.state, HeadTranslationState::Degraded);
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
