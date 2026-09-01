//! Lightweight smoothing for the 52 ARKit Perfect Sync channels.
//!
//! The MediaPipe coefficients are already validated and bounded when they
//! reach this filter, so it only has to damp per-frame jitter.  Every channel
//! shares the attack/release time constants of the standard expression
//! filter; there are no per-channel parameters, no history window, and no
//! temporal model.

use vtuber_core::{ARKIT52_CHANNEL_COUNT, Arkit52Coefficients, ArkitBlendshape, MonoTimeNs};

use super::expression::ExpressionFilterParams;

/// Exponential attack/release smoother for all ARKit52 channels.
///
/// `TongueOut` has no tracking source and is pinned to `0.0` on every frame,
/// so it never carries a value from an earlier frame or from the input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetailedExpressionFilter {
    values: [f32; ARKIT52_CHANNEL_COUNT],
    last_time: Option<MonoTimeNs>,
    params: ExpressionFilterParams,
}

impl DetailedExpressionFilter {
    /// Creates a filter using the shared expression time constants.
    #[must_use]
    pub fn new(params: ExpressionFilterParams) -> Self {
        Self {
            values: [0.0; ARKIT52_CHANNEL_COUNT],
            last_time: None,
            params,
        }
    }

    /// Clears every channel so no coefficient survives a session boundary.
    pub fn reset(&mut self) {
        self.values = [0.0; ARKIT52_CHANNEL_COUNT];
        self.last_time = None;
    }

    /// Smooths `input` toward the stored state and returns the new values.
    pub fn update(&mut self, input: &Arkit52Coefficients, now: MonoTimeNs) -> Arkit52Coefficients {
        let Some(last_time) = self.last_time else {
            self.snap_to(input);
            self.last_time = Some(now);
            return self.coefficients();
        };
        // Backwards or repeated timestamps leave no elapsed time to smooth
        // over, so adopt the input the same way the standard filter does.
        if now.0 <= last_time.0 {
            self.snap_to(input);
            self.last_time = Some(now);
            return self.coefficients();
        }
        let dt_sec = ((now.0 - last_time.0) as f32 / 1_000_000_000.0)
            .min(self.params.max_dt_sec)
            .max(0.0);
        self.last_time = Some(now);
        if dt_sec > 0.0 {
            self.smooth(input, dt_sec);
        }
        self.coefficients()
    }

    fn snap_to(&mut self, input: &Arkit52Coefficients) {
        for (channel, slot) in ArkitBlendshape::ALL.into_iter().zip(self.values.iter_mut()) {
            *slot = coefficient(channel, input);
        }
    }

    fn smooth(&mut self, input: &Arkit52Coefficients, dt_sec: f32) {
        for (channel, slot) in ArkitBlendshape::ALL.into_iter().zip(self.values.iter_mut()) {
            let target = coefficient(channel, input);
            let tau = if target > *slot {
                self.params.attack_time_constant_sec
            } else {
                self.params.release_time_constant_sec
            }
            .max(f32::EPSILON);
            let alpha = (1.0 - (-dt_sec / tau).exp()).clamp(0.0, 1.0);
            *slot = (*slot + alpha * (target - *slot)).clamp(0.0, 1.0);
        }
    }

    // Invariant: `snap_to` copies validated `[0, 1]` coefficients, `smooth`
    // clamps every update, and the state starts at `0.0`, so validation
    // cannot reject the vector.
    #[allow(clippy::expect_used)]
    fn coefficients(&self) -> Arkit52Coefficients {
        Arkit52Coefficients::try_from_array(self.values)
            .expect("detailed expression state is clamped to [0, 1]")
    }
}

/// Reads one channel, pinning `TongueOut` to `0.0`.
fn coefficient(channel: ArkitBlendshape, input: &Arkit52Coefficients) -> f32 {
    if channel == ArkitBlendshape::TongueOut {
        0.0
    } else {
        input.get(channel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Two 60 fps frames make one 30 fps frame, so the two step counts below
    // cover exactly the same elapsed time.
    const FRAME_60FPS_NS: u64 = 16_666_667;
    const FRAME_30FPS_NS: u64 = 2 * FRAME_60FPS_NS;

    fn params() -> ExpressionFilterParams {
        ExpressionFilterParams::with_time_constants(0.03, 0.10)
    }

    fn input(value: f32, tongue_out: f32) -> Arkit52Coefficients {
        let mut values = [value; ARKIT52_CHANNEL_COUNT];
        values[ArkitBlendshape::TongueOut.index()] = tongue_out;
        Arkit52Coefficients::try_from_array(values).expect("test values are in [0, 1]")
    }

    /// Primes the state at `0.0`, then steps `target` in for `steps` frames.
    fn run_to(filter: &mut DetailedExpressionFilter, target: f32, steps: u64, step_ns: u64) -> f32 {
        filter.update(&input(0.0, 0.0), MonoTimeNs(0));
        let mut now = 0u64;
        let mut value = 0.0;
        for _ in 0..steps {
            now += step_ns;
            value = filter
                .update(&input(target, 1.0), MonoTimeNs(now))
                .get(ArkitBlendshape::JawOpen);
        }
        value
    }

    #[test]
    fn constant_input_converges_monotonically_within_bounds() {
        let mut filter = DetailedExpressionFilter::new(params());
        filter.update(&input(0.0, 0.0), MonoTimeNs(0));

        let mut previous = 0.0;
        for step in 1..=30u64 {
            let value = filter
                .update(&input(1.0, 0.0), MonoTimeNs(step * FRAME_30FPS_NS))
                .get(ArkitBlendshape::JawOpen);
            assert!(value >= previous, "step {step} went backwards");
            assert!(value.is_finite() && (0.0..=1.0).contains(&value));
            previous = value;
        }
        assert!(previous > 0.99, "did not converge: {previous}");
    }

    #[test]
    fn equal_elapsed_time_reaches_the_same_value_at_30_and_60_fps() {
        // Two 30 fps frames stay well below saturation, so the comparison
        // measures the filter rather than the `[0, 1]` clamp.
        let slow_frames = 2u64;
        let elapsed_ns = slow_frames * FRAME_30FPS_NS;
        let slow = run_to(
            &mut DetailedExpressionFilter::new(params()),
            1.0,
            slow_frames,
            FRAME_30FPS_NS,
        );
        let fast = run_to(
            &mut DetailedExpressionFilter::new(params()),
            1.0,
            elapsed_ns / FRAME_60FPS_NS,
            FRAME_60FPS_NS,
        );
        assert!(slow < 1.0, "test must not saturate: {slow}");
        assert!(
            (slow - fast).abs() < 0.01,
            "dt invariance broken: 30fps={slow}, 60fps={fast}"
        );
    }

    #[test]
    fn attack_uses_the_attack_constant_and_release_the_release_constant() {
        let dt_sec = FRAME_30FPS_NS as f32 / 1_000_000_000.0;
        let mut filter = DetailedExpressionFilter::new(params());
        filter.update(&input(0.0, 0.0), MonoTimeNs(FRAME_30FPS_NS));

        let attacked = filter
            .update(&input(1.0, 0.0), MonoTimeNs(2 * FRAME_30FPS_NS))
            .get(ArkitBlendshape::JawOpen);
        let attack_alpha = 1.0 - (-dt_sec / 0.03).exp();
        assert!(
            (attacked - attack_alpha).abs() < 1.0e-6,
            "attack used the wrong time constant: {attacked}"
        );

        let relaxed = filter
            .update(&input(0.0, 0.0), MonoTimeNs(3 * FRAME_30FPS_NS))
            .get(ArkitBlendshape::JawOpen);
        let expected = attacked * (-dt_sec / 0.10).exp();
        assert!(
            (relaxed - expected).abs() < 1.0e-6,
            "release used the wrong time constant: {relaxed}"
        );
    }

    #[test]
    fn tongue_out_stays_zero() {
        let mut filter = DetailedExpressionFilter::new(params());
        for step in 1..=5u64 {
            let coefficients = filter.update(&input(0.5, 1.0), MonoTimeNs(step * FRAME_30FPS_NS));
            assert_eq!(coefficients.get(ArkitBlendshape::TongueOut), 0.0);
        }
    }

    #[test]
    fn reset_drops_previous_session_coefficients() {
        let mut filter = DetailedExpressionFilter::new(params());
        for step in 1..=10u64 {
            filter.update(&input(1.0, 0.0), MonoTimeNs(step * FRAME_30FPS_NS));
        }
        filter.reset();
        let coefficients = filter.update(&input(0.0, 0.0), MonoTimeNs(11 * FRAME_30FPS_NS));
        assert_eq!(coefficients, Arkit52Coefficients::default());
    }
}
