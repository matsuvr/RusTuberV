//! Frame-rate-independent smoothing and neutral return for eye-in-head gaze.

use std::time::Duration;

use vtuber_core::{GazeSignal, GazeTrackingState};

/// Default tracked-motion half-life.
pub const DEFAULT_TRACKED_HALF_LIFE: Duration = Duration::from_millis(55);
/// Default neutral-return half-life after gaze becomes unavailable.
pub const DEFAULT_RETURN_HALF_LIFE: Duration = Duration::from_millis(150);
/// Default short hold before neutral return.
pub const DEFAULT_UNAVAILABLE_HOLD: Duration = Duration::from_millis(80);

/// Parameters for [`GazeFilter`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GazeFilterParams {
    /// Half-life while following tracked motion. Zero means immediate.
    pub tracked_half_life: Duration,
    /// Half-life while returning to neutral. Zero means immediate.
    pub return_half_life: Duration,
    /// How long to hold the last value when input is unavailable.
    pub unavailable_hold: Duration,
}

impl Default for GazeFilterParams {
    fn default() -> Self {
        Self {
            tracked_half_life: DEFAULT_TRACKED_HALF_LIFE,
            return_half_life: DEFAULT_RETURN_HALF_LIFE,
            unavailable_hold: DEFAULT_UNAVAILABLE_HOLD,
        }
    }
}

/// Lightweight gaze smoother with explicit unavailable hold and neutral return.
#[derive(Clone, Debug, PartialEq)]
pub struct GazeFilter {
    params: GazeFilterParams,
    current: GazeSignal,
    unavailable_elapsed: Duration,
    initialized: bool,
}

impl GazeFilter {
    /// Creates an empty filter.
    #[must_use]
    pub fn new(params: GazeFilterParams) -> Self {
        Self {
            params,
            current: GazeSignal::UNAVAILABLE,
            unavailable_elapsed: Duration::ZERO,
            initialized: false,
        }
    }

    /// Clears all retained gaze state.
    pub fn reset(&mut self) {
        self.current = GazeSignal::UNAVAILABLE;
        self.unavailable_elapsed = Duration::ZERO;
        self.initialized = false;
    }

    /// Updates the filter using elapsed wall time.
    #[must_use]
    pub fn update(&mut self, input: GazeSignal, dt: Duration) -> GazeSignal {
        if input.is_available() {
            self.unavailable_elapsed = Duration::ZERO;
            if !self.initialized {
                self.current = GazeSignal::degraded(0.0, 0.0, 0.0);
                self.initialized = true;
            }
            let alpha = smoothing_alpha(dt, self.params.tracked_half_life);
            self.current = blend(self.current, input, alpha, input.state);
            return self.current;
        }

        if !self.initialized {
            return GazeSignal::UNAVAILABLE;
        }
        self.unavailable_elapsed = self.unavailable_elapsed.saturating_add(dt);
        if self.unavailable_elapsed < self.params.unavailable_hold {
            self.current.state = GazeTrackingState::Degraded;
            return self.current;
        }

        let alpha = smoothing_alpha(dt, self.params.return_half_life);
        self.current = blend(
            self.current,
            GazeSignal::degraded(0.0, 0.0, 0.0),
            alpha,
            GazeTrackingState::Degraded,
        );
        if self.current.horizontal.abs() < 1.0e-4
            && self.current.vertical.abs() < 1.0e-4
            && self.current.confidence < 1.0e-4
        {
            self.current = GazeSignal::UNAVAILABLE;
            self.initialized = false;
        }
        self.current
    }
}

fn smoothing_alpha(dt: Duration, half_life: Duration) -> f32 {
    if half_life.is_zero() {
        return 1.0;
    }
    let dt = dt.as_secs_f32().max(0.0);
    let half_life = half_life.as_secs_f32();
    (1.0 - (-std::f32::consts::LN_2 * dt / half_life).exp()).clamp(0.0, 1.0)
}

fn blend(from: GazeSignal, to: GazeSignal, t: f32, state: GazeTrackingState) -> GazeSignal {
    let horizontal = from.horizontal + (to.horizontal - from.horizontal) * t;
    let vertical = from.vertical + (to.vertical - from.vertical) * t;
    let confidence = from.confidence + (to.confidence - from.confidence) * t;
    match state {
        GazeTrackingState::Tracked => GazeSignal::tracked(horizontal, vertical, confidence),
        GazeTrackingState::Degraded | GazeTrackingState::Unavailable => {
            GazeSignal::degraded(horizontal, vertical, confidence)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response(fps: u32) -> f32 {
        let mut filter = GazeFilter::new(GazeFilterParams::default());
        let dt = Duration::from_secs_f32(1.0 / fps as f32);
        let mut value = GazeSignal::UNAVAILABLE;
        for _ in 0..fps {
            value = filter.update(GazeSignal::tracked(1.0, 0.0, 1.0), dt);
        }
        value.horizontal
    }

    #[test]
    fn response_is_frame_rate_independent() {
        let r30 = response(30);
        let r60 = response(60);
        let r120 = response(120);
        assert!((r30 - r60).abs() < 1.0e-5);
        assert!((r60 - r120).abs() < 1.0e-5);
    }

    #[test]
    fn unavailable_holds_then_returns_neutral_and_reacquires_continuously() {
        let mut filter = GazeFilter::new(GazeFilterParams::default());
        let dt = Duration::from_millis(20);
        let tracked = filter.update(GazeSignal::tracked(1.0, 0.0, 1.0), dt);
        let held = filter.update(GazeSignal::UNAVAILABLE, dt);
        assert_eq!(tracked.horizontal, held.horizontal);
        for _ in 0..6 {
            let _ = filter.update(GazeSignal::UNAVAILABLE, dt);
        }
        let returning = filter.update(GazeSignal::UNAVAILABLE, dt);
        assert!(returning.horizontal < held.horizontal);
        let reacquired = filter.update(GazeSignal::tracked(-1.0, 0.0, 1.0), dt);
        assert!(reacquired.horizontal > -1.0);
        assert!(reacquired.horizontal < returning.horizontal);
    }

    #[test]
    fn zero_half_life_tracks_immediately_and_non_finite_is_safe() {
        let mut filter = GazeFilter::new(GazeFilterParams {
            tracked_half_life: Duration::ZERO,
            return_half_life: Duration::ZERO,
            unavailable_hold: Duration::ZERO,
        });
        assert_eq!(
            filter.update(GazeSignal::tracked(0.5, -0.25, 1.0), Duration::ZERO),
            GazeSignal::tracked(0.5, -0.25, 1.0)
        );
        assert_eq!(
            filter.update(GazeSignal::tracked(f32::NAN, 0.0, 1.0), Duration::ZERO),
            GazeSignal::UNAVAILABLE
        );
    }
}
