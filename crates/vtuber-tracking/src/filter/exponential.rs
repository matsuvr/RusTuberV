//! Exact first-order smoothing shared by motion and expression channels.
//!
//! A time constant and a half-life are different units of response. Callers
//! select their existing clock, target and channel policy before this step.

/// Fraction followed over `dt_sec` with a positive time constant.
#[must_use]
pub fn time_constant_alpha(time_constant_sec: f32, dt_sec: f32) -> f32 {
    decay_alpha(dt_sec / time_constant_sec)
}

/// Fraction followed over `dt_sec` with a half-life. A disabled/non-positive
/// half-life follows immediately; a non-positive elapsed time holds the state.
#[must_use]
pub fn half_life_alpha(half_life_sec: f32, dt_sec: f32) -> f32 {
    if !half_life_sec.is_finite() || half_life_sec <= 0.0 {
        return 1.0;
    }
    if !dt_sec.is_finite() || dt_sec <= 0.0 {
        return 0.0;
    }
    decay_alpha(std::f32::consts::LN_2 * dt_sec / half_life_sec)
}

fn decay_alpha(exponent: f32) -> f32 {
    1.0 - (-exponent).exp()
}

/// Shortest signed difference of two finite angles, in radians.
#[must_use]
pub fn shortest_angle_delta(current: f32, target: f32) -> f32 {
    let current = if current.is_finite() { current } else { 0.0 };
    let target = if target.is_finite() { target } else { 0.0 };
    (target - current + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
        - std::f32::consts::PI
}

/// First-order angular follow without wrapping through the long arc.
#[must_use]
pub fn smooth_angle_half_life(current: f32, target: f32, half_life: f32, dt: f32) -> f32 {
    let current = if current.is_finite() { current } else { 0.0 };
    let target = if target.is_finite() { target } else { 0.0 };
    let alpha = half_life_alpha(half_life, dt);
    if alpha >= 1.0 {
        return target;
    }
    current + shortest_angle_delta(current, target) * alpha
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_units_and_frame_partition_are_preserved() {
        assert!((half_life_alpha(0.2, 0.2) - 0.5).abs() < 1.0e-6);
        assert!((time_constant_alpha(0.2, 0.2) - (1.0 - (-1.0_f32).exp())).abs() < 1.0e-6);
        for fps in [30, 60, 120] {
            for (alpha, expected) in [
                (half_life_alpha(0.2, 1.0 / fps as f32), 2.0_f32.powf(-5.0)),
                (time_constant_alpha(0.2, 1.0 / fps as f32), (-5.0_f32).exp()),
            ] {
                let mut residual = 1.0;
                for _ in 0..fps {
                    residual *= 1.0 - alpha;
                }
                assert!((residual - expected).abs() < 1.0e-6);
            }
        }
    }

    #[test]
    fn angular_follow_crosses_the_wrap_on_the_short_arc() {
        let current = 179.0_f32.to_radians();
        let target = -179.0_f32.to_radians();
        let halfway = smooth_angle_half_life(current, target, 0.2, 0.2);
        assert!((halfway - std::f32::consts::PI).abs() < 1.0e-6);
        assert_eq!(smooth_angle_half_life(current, target, 0.2, 0.0), current);
        assert_eq!(smooth_angle_half_life(current, target, 0.0, 0.2), target);
    }
}
