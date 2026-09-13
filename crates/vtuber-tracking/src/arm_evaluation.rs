//! Pure arm-sequence evaluation for the 5/5 quality comparison (Issue #50).
//!
//! The application captures one [`ArmEvaluationSample`] per frame and hands the
//! recorded slice to [`evaluate_arm_sequence`]. Everything here is a pure
//! function over recorded numbers: it reads no clock, no ECS, and no native
//! handle, so the same recording can be compared across the virtual, raw
//! tracked, and final tracked paths.

use nalgebra::Vector3;
use vtuber_core::MonoTimeNs;

/// Final-target movement below this is treated as a stationary frame.
const STATIONARY_EPSILON: f32 = 1.0e-3;

/// One recorded frame of one arm path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmEvaluationSample {
    /// Camera capture time of the observation that produced this frame.
    pub captured_at: MonoTimeNs,
    /// Time this frame was displayed.
    pub displayed_at: MonoTimeNs,
    /// Target before any minimum-distance shaping.
    pub shaped_target: [f32; 3],
    /// Target actually handed to the solver after shaping.
    pub final_target: [f32; 3],
    /// Solved shoulder origin in rest space.
    pub shoulder: [f32; 3],
    /// Solved elbow origin in rest space.
    pub elbow: [f32; 3],
    /// Solved wrist origin in rest space.
    pub wrist: [f32; 3],
    /// Reference upper-arm length for this model.
    pub upper_arm_length: f32,
    /// Reference forearm length for this model.
    pub forearm_length: f32,
}

/// Aggregate quality of one recorded arm sequence.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ArmEvaluationSummary {
    /// Mean wrist distance from the shaped final target, in meters.
    pub wrist_reach_error_meters: f32,
    /// Mean absolute upper-arm length error, in meters.
    pub upper_arm_length_error_meters: f32,
    /// Mean absolute forearm length error, in meters.
    pub forearm_length_error_meters: f32,
    /// Mean wrist spread on auto-detected stationary frames, in meters.
    pub stationary_wrist_variance_meters: f32,
    /// Largest per-frame bend-plane turn, in radians.
    pub pole_max_step_radians: f32,
    /// Mean capture-to-display latency in nanoseconds.
    pub capture_to_display_mean_ns: f64,
    /// 95th-percentile capture-to-display latency in nanoseconds.
    pub capture_to_display_p95_ns: u64,
    /// Number of frames that informed the summary.
    pub sample_count: usize,
}

/// Evaluates one recorded arm sequence.
///
/// Empty input yields a zero summary. Latencies are computed as the positive
/// difference `displayed_at - captured_at`; negative values are clamped to zero
/// rather than shifting the mean.
#[must_use]
pub fn evaluate_arm_sequence(samples: &[ArmEvaluationSample]) -> ArmEvaluationSummary {
    if samples.is_empty() {
        return ArmEvaluationSummary::default();
    }
    let count = samples.len();
    let count_f = count as f32;

    let wrist_reach_error_meters = samples
        .iter()
        .map(|sample| {
            (vector(sample.wrist) - vector(sample.final_target))
                .norm()
                .max(0.0)
        })
        .sum::<f32>()
        / count_f;
    let upper_arm_length_error_meters = samples
        .iter()
        .map(|sample| {
            ((vector(sample.elbow) - vector(sample.shoulder)).norm() - sample.upper_arm_length)
                .abs()
        })
        .sum::<f32>()
        / count_f;
    let forearm_length_error_meters = samples
        .iter()
        .map(|sample| {
            ((vector(sample.wrist) - vector(sample.elbow)).norm() - sample.forearm_length).abs()
        })
        .sum::<f32>()
        / count_f;

    let stationary_wrist_variance_meters = stationary_wrist_variance(samples);
    let pole_max_step_radians = pole_max_step(samples);

    let mut latencies: Vec<u64> = samples
        .iter()
        .map(|sample| sample.displayed_at.0.saturating_sub(sample.captured_at.0))
        .collect();
    latencies.sort_unstable();
    let capture_to_display_mean_ns =
        latencies.iter().map(|value| *value as f64).sum::<f64>() / count as f64;
    let p95_index = ((count as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(count - 1);
    let capture_to_display_p95_ns = latencies.get(p95_index).copied().unwrap_or(0);

    ArmEvaluationSummary {
        wrist_reach_error_meters,
        upper_arm_length_error_meters,
        forearm_length_error_meters,
        stationary_wrist_variance_meters,
        pole_max_step_radians,
        capture_to_display_mean_ns,
        capture_to_display_p95_ns,
        sample_count: count,
    }
}

fn stationary_wrist_variance(samples: &[ArmEvaluationSample]) -> f32 {
    let mut wrists: Vec<Vector3<f32>> = Vec::new();
    let mut previous: Option<Vector3<f32>> = None;
    for sample in samples {
        let target = vector(sample.final_target);
        let stationary = previous.is_none_or(|value| (target - value).norm() < STATIONARY_EPSILON);
        if stationary {
            wrists.push(vector(sample.wrist));
        }
        previous = Some(target);
    }
    if wrists.is_empty() {
        return 0.0;
    }
    let count = wrists.len() as f32;
    let mean = wrists
        .iter()
        .fold(Vector3::zeros(), |sum, value| sum + value)
        / count;
    wrists
        .iter()
        .map(|value| (value - mean).norm_squared())
        .sum::<f32>()
        / count
}

fn pole_max_step(samples: &[ArmEvaluationSample]) -> f32 {
    let mut maximum = 0.0_f32;
    for pair in samples.windows(2) {
        let [previous, current] = pair else {
            continue;
        };
        let previous_direction = vector(previous.elbow) - vector(previous.shoulder);
        let current_direction = vector(current.elbow) - vector(current.shoulder);
        let Some(previous_direction) = finite_normalized(previous_direction) else {
            continue;
        };
        let Some(current_direction) = finite_normalized(current_direction) else {
            continue;
        };
        let angle = previous_direction
            .dot(&current_direction)
            .clamp(-1.0, 1.0)
            .acos();
        maximum = maximum.max(angle);
    }
    maximum
}

fn finite_normalized(value: Vector3<f32>) -> Option<Vector3<f32>> {
    let length = value.norm();
    if value.iter().all(|component| component.is_finite()) && length > f32::EPSILON {
        Some(value / length)
    } else {
        None
    }
}

fn vector([x, y, z]: [f32; 3]) -> Vector3<f32> {
    Vector3::new(x, y, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(seq: u64, target: [f32; 3], wrist: [f32; 3]) -> ArmEvaluationSample {
        ArmEvaluationSample {
            captured_at: MonoTimeNs(seq * 1_000_000),
            displayed_at: MonoTimeNs(seq * 1_000_000 + 4_000_000),
            shaped_target: target,
            final_target: target,
            shoulder: [0.0, 0.0, 0.0],
            elbow: [0.0, 0.0, 0.0],
            wrist,
            upper_arm_length: 0.0,
            forearm_length: 0.0,
        }
    }

    #[test]
    fn empty_input_is_a_zero_summary() {
        assert_eq!(evaluate_arm_sequence(&[]), ArmEvaluationSummary::default());
    }

    #[test]
    fn reach_and_length_errors_match_known_values() {
        let samples = [ArmEvaluationSample {
            captured_at: MonoTimeNs(0),
            displayed_at: MonoTimeNs(5_000_000),
            shaped_target: [0.0, 0.0, 0.0],
            final_target: [1.0, 0.0, 0.0],
            shoulder: [0.0, 0.0, 0.0],
            elbow: [0.3, 0.0, 0.0],
            wrist: [1.1, 0.0, 0.0],
            upper_arm_length: 0.3,
            forearm_length: 0.8,
        }];
        let summary = evaluate_arm_sequence(&samples);
        assert!((summary.wrist_reach_error_meters - 0.1).abs() < 1.0e-6);
        assert!((summary.upper_arm_length_error_meters - 0.0).abs() < 1.0e-6);
        assert!((summary.forearm_length_error_meters - 0.0).abs() < 1.0e-6);
        assert_eq!(summary.capture_to_display_p95_ns, 5_000_000);
        assert_eq!(summary.sample_count, 1);
    }

    #[test]
    fn stationary_variance_only_counts_frames_that_do_not_move() {
        // Three stationary frames plus one moving frame; the moving frame's
        // wrist must not inflate the stationary spread.
        let samples = [
            sample(0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            sample(1, [0.0, 0.0, 0.0], [0.001, 0.0, 0.0]),
            sample(2, [0.0, 0.0, 0.0], [-0.001, 0.0, 0.0]),
            sample(3, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ];
        let summary = evaluate_arm_sequence(&samples);
        assert!(summary.stationary_wrist_variance_meters < 1.0e-6);
    }

    #[test]
    fn pole_step_reports_a_flip() {
        let mut first = sample(0, [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]);
        first.elbow = [0.0, 0.5, 0.0];
        let mut second = sample(1, [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]);
        second.elbow = [0.0, -0.5, 0.0];
        let summary = evaluate_arm_sequence(&[first, second]);
        assert!((summary.pole_max_step_radians - std::f32::consts::PI).abs() < 1.0e-4);
    }
}
