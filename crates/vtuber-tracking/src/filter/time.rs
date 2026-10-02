//! Shared elapsed-time arithmetic; sample adoption remains a channel policy.

use vtuber_core::MonoTimeNs;

/// Monotonic elapsed seconds, bounded by the caller's existing maximum gap.
#[must_use]
pub fn elapsed_seconds(now: MonoTimeNs, previous: MonoTimeNs, max_dt_sec: f32) -> f32 {
    bounded_dt(
        now.0.saturating_sub(previous.0) as f32 / 1_000_000_000.0,
        max_dt_sec,
    )
}

/// Bound a render delta without changing the clock which produced it.
#[must_use]
pub fn bounded_dt(dt_sec: f32, max_dt_sec: f32) -> f32 {
    dt_sec.min(max_dt_sec).max(0.0)
}

/// Unit progress of a finite-duration transition. Zero duration completes it.
#[must_use]
pub fn transition_progress(elapsed_sec: f32, duration_sec: f32) -> f32 {
    if duration_sec <= 0.0 {
        1.0
    } else {
        (elapsed_sec / duration_sec).clamp(0.0, 1.0)
    }
}

/// Smoothstep easing on a unit interval.
#[must_use]
pub fn smoothstep(progress: f32) -> f32 {
    let t = progress.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
