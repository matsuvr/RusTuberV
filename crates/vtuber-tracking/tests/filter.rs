// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Integration tests for `vtuber-tracking` filters.

use approx::assert_relative_eq;
use nalgebra::{UnitQuaternion, Vector3};
use vtuber_core::types::MonoTimeNs;
use vtuber_tracking::filter::{HeadFilterParams, HeadRotationFilter};

fn ts(ns: u64) -> MonoTimeNs {
    MonoTimeNs(ns)
}

fn assert_quat_eq(a: UnitQuaternion<f32>, b: UnitQuaternion<f32>) {
    assert_relative_eq!(a.quaternion().w, b.quaternion().w, epsilon = 1e-5);
    assert_relative_eq!(a.quaternion().i, b.quaternion().i, epsilon = 1e-5);
    assert_relative_eq!(a.quaternion().j, b.quaternion().j, epsilon = 1e-5);
    assert_relative_eq!(a.quaternion().k, b.quaternion().k, epsilon = 1e-5);
}

#[test]
fn head_filter_constant_input_converges() {
    let params = HeadFilterParams::with_time_constant(0.05);
    let mut filter = HeadRotationFilter::new(params);
    let target = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.5);

    let _ = filter.update(target, ts(0));
    for i in 1..=300 {
        let out = filter.update(target, ts(i * 16_666_667));
        let diff = (out.quaternion().coords - target.quaternion().coords).norm();
        assert!(
            diff < 1e-4,
            "did not converge at frame {i}: diff={diff}, out={out:?}, target={target:?}"
        );
    }
}

#[test]
fn head_filter_sign_flip_does_not_jump() {
    let params = HeadFilterParams::with_time_constant(0.05);
    let mut filter = HeadRotationFilter::new(params);
    let q = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.3);

    let _ = filter.update(q, ts(0));
    let neg_q = UnitQuaternion::from_quaternion(-*q.quaternion());
    let out = filter.update(neg_q, ts(16_666_667));

    // The filter should stay near `q` (same physical rotation) and should
    // not travel the long way around the sphere toward `-q`.
    let dot = out.quaternion().coords.dot(&q.quaternion().coords).abs();
    assert!(
        dot > 0.99,
        "filter jumped after sign flip: dot={dot}, out={out:?}, q={q:?}"
    );
}

#[test]
fn head_filter_large_dt_does_not_panic() {
    let params = HeadFilterParams::with_time_constant(0.05);
    let mut filter = HeadRotationFilter::new(params);
    let q = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.3);

    let _ = filter.update(q, ts(0));
    // One year in nanoseconds.
    let year_ns = 365u64 * 24 * 60 * 60 * 1_000_000_000;
    let out = filter.update(q, ts(year_ns));

    assert!(out.quaternion().w.is_finite());
    assert!(out.quaternion().i.is_finite());
    assert!(out.quaternion().j.is_finite());
    assert!(out.quaternion().k.is_finite());
}

#[test]
fn head_filter_zero_dt_does_not_panic() {
    let params = HeadFilterParams::with_time_constant(0.05);
    let mut filter = HeadRotationFilter::new(params);
    let q = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.3);

    let _ = filter.update(q, ts(1_000_000_000));
    let out = filter.update(q, ts(1_000_000_000));

    assert!(out.quaternion().w.is_finite());
    assert!(out.quaternion().i.is_finite());
    assert!(out.quaternion().j.is_finite());
    assert!(out.quaternion().k.is_finite());
}

#[test]
fn head_filter_backwards_timestamp_does_not_panic() {
    let params = HeadFilterParams::with_time_constant(0.05);
    let mut filter = HeadRotationFilter::new(params);
    let q = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.3);

    let _ = filter.update(q, ts(2_000_000_000));
    let out = filter.update(q, ts(1_000_000_000));

    assert!(out.quaternion().w.is_finite());
    assert!(out.quaternion().i.is_finite());
    assert!(out.quaternion().j.is_finite());
    assert!(out.quaternion().k.is_finite());
}

#[test]
fn head_filter_reset_reacquire_initializes_with_next_input() {
    let params = HeadFilterParams::with_time_constant(0.05);
    let mut filter = HeadRotationFilter::new(params);
    let q1 = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.3);
    let q2 = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), -0.5);

    let out = filter.update(q1, ts(0));
    assert_quat_eq(out, q1);

    filter.reset();
    assert!(!filter.is_initialized());

    let out = filter.update(q2, ts(16_666_667));
    assert_quat_eq(out, q2);

    // reacquire behaves the same way.
    filter.reacquire();
    assert!(!filter.is_initialized());

    let out = filter.update(q1, ts(33_333_334));
    assert_quat_eq(out, q1);
}

#[test]
fn head_filter_smoothing_moves_toward_target() {
    let params = HeadFilterParams::with_time_constant(0.05);
    let mut filter = HeadRotationFilter::new(params);
    let identity = UnitQuaternion::identity();
    let target = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 1.0);

    let out1 = filter.update(identity, ts(0));
    assert_quat_eq(out1, identity);

    let out2 = filter.update(target, ts(16_666_667));

    // After one 60 Hz frame the output should be strictly between identity
    // and target.
    let dot_identity_target = identity
        .quaternion()
        .coords
        .dot(&target.quaternion().coords)
        .abs();
    let dot_out_target = out2
        .quaternion()
        .coords
        .dot(&target.quaternion().coords)
        .abs();
    assert!(
        dot_out_target > dot_identity_target,
        "filter did not move toward target: dot_identity_target={dot_identity_target}, dot_out_target={dot_out_target}"
    );
}
