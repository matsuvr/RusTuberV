// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Integration tests for the avatar-output authority gate (GNM #57.3).
//!
//! These tests simulate the runtime avatar application loop over source-aligned
//! backend outputs: every frame carries a Direct payload and an optional
//! same-source GNM payload, and the gate decides which single payload reaches
//! the avatar. Verified invariants:
//!
//! * Exactly one backend's coefficients are published per frame; channels are
//!   never summed or blended across backends.
//! * Hard failures (`Unavailable`) fall back to Direct immediately.
//! * Transient issues follow hysteresis without frame-by-frame thrashing.
//! * Every authority change flags `clear_previous_output` so stale detailed
//!   expression state is explicitly cleared/coalesced.

use vtuber_core::{ARKIT52_CHANNEL_COUNT, Arkit52Coefficients, ArkitBlendshape};
use vtuber_tracking::{
    AlignedBackendOutputs, AuthorityGate, BackendOutputTiming, BackendSelectionDecision,
    FaceTrackingBackend, FaceTrackingMode, GnmRuntimeHealth, GnmTransientIssue,
    GnmUnavailableReason, SourceFrameStamp, StampedBackendOutput,
};

/// Two consecutive transients trigger fallback; three ready frames recover.
const TRANSIENT_LIMIT: u32 = 2;
const READY_LIMIT: u32 = 3;

fn direct_coefficients() -> Arkit52Coefficients {
    let mut values = [0.0; ARKIT52_CHANNEL_COUNT];
    values[ArkitBlendshape::JawOpen.index()] = 0.25;
    values[ArkitBlendshape::MouthSmileLeft.index()] = 0.5;
    Arkit52Coefficients::try_from_array(values).expect("valid direct coefficients")
}

fn gnm_coefficients() -> Arkit52Coefficients {
    let mut values = [0.0; ARKIT52_CHANNEL_COUNT];
    values[ArkitBlendshape::JawOpen.index()] = 0.75;
    values[ArkitBlendshape::BrowDownLeft.index()] = 0.6;
    Arkit52Coefficients::try_from_array(values).expect("valid gnm coefficients")
}

fn gate() -> AuthorityGate {
    AuthorityGate::new(TRANSIENT_LIMIT, READY_LIMIT).expect("valid hysteresis")
}

/// One simulated frame of the avatar path: both backends publish for the same
/// source stamp and the gate selects the sole authoritative payload.
fn advance_frame(
    authority: &mut AuthorityGate,
    seq: u64,
    health: GnmRuntimeHealth,
) -> (
    Arkit52Coefficients,
    FaceTrackingBackend,
    BackendSelectionDecision,
) {
    let stamp = SourceFrameStamp::new(seq, 1_000_000 + seq, None).expect("monotonic stamp");
    let direct_timing = BackendOutputTiming::new(
        stamp,
        FaceTrackingBackend::DirectMediaPipe,
        None,
        None,
        stamp.capture_micros() + 100,
    )
    .expect("direct timing");
    let gnm_timing = BackendOutputTiming::new(
        stamp,
        FaceTrackingBackend::GnmTemporal,
        Some(stamp.capture_micros() + 200),
        Some(stamp.capture_micros() + 300),
        stamp.capture_micros() + 400,
    )
    .expect("gnm timing");
    // Source alignment is validated before arbitration consumes either side.
    let aligned = AlignedBackendOutputs::new(
        StampedBackendOutput::new(direct_timing, direct_coefficients()),
        StampedBackendOutput::new(gnm_timing, gnm_coefficients()),
    );
    assert!(aligned.is_ok(), "same-source pair must align");

    let outcome = authority.advance(
        FaceTrackingMode::GnmTemporal,
        health,
        direct_coefficients(),
        Some(gnm_coefficients()),
    );
    (
        outcome.avatar_output,
        outcome.authority_backend,
        outcome.decision,
    )
}

#[test]
fn hard_failure_falls_back_to_direct_immediately_and_clears_state() {
    let mut authority = gate();
    let (_, backend, decision) = advance_frame(&mut authority, 1, GnmRuntimeHealth::Ready);
    assert_eq!(backend, FaceTrackingBackend::GnmTemporal);
    assert!(!decision.clear_previous_output);

    // A hard/unavailable condition never waits for hysteresis.
    let (output, backend, decision) = advance_frame(
        &mut authority,
        2,
        GnmRuntimeHealth::Unavailable(GnmUnavailableReason::ModelInvalid),
    );
    assert_eq!(backend, FaceTrackingBackend::DirectMediaPipe);
    assert!(decision.clear_previous_output);
    assert_eq!(
        output.get(ArkitBlendshape::JawOpen),
        direct_coefficients().get(ArkitBlendshape::JawOpen)
    );
    assert_eq!(output.get(ArkitBlendshape::JawOpen), 0.25);
}

#[test]
fn published_payload_is_never_a_sum_of_both_backends() {
    let mut authority = gate();
    let (output, backend, _) = advance_frame(&mut authority, 1, GnmRuntimeHealth::Ready);
    assert_eq!(backend, FaceTrackingBackend::GnmTemporal);
    // Byte-equal to the GNM payload alone; the Direct-only BrowDownLeft value
    // must not leak in and JawOpen must not become 0.25 + 0.75.
    assert_eq!(output.get(ArkitBlendshape::JawOpen), 0.75);
    assert_eq!(output.get(ArkitBlendshape::BrowDownLeft), 0.6);
    assert_eq!(output.get(ArkitBlendshape::MouthSmileLeft), 0.0);

    let (output, backend, _) = advance_frame(&mut authority, 2, GnmRuntimeHealth::Ready);
    assert_eq!(backend, FaceTrackingBackend::GnmTemporal);
    let _ = output;

    let (output, backend, _) = advance_frame(
        &mut authority,
        3,
        GnmRuntimeHealth::Unavailable(GnmUnavailableReason::DecoderUnavailable),
    );
    assert_eq!(backend, FaceTrackingBackend::DirectMediaPipe);
    assert_eq!(output.get(ArkitBlendshape::JawOpen), 0.25);
    assert_eq!(output.get(ArkitBlendshape::MouthSmileLeft), 0.5);
    assert_eq!(output.get(ArkitBlendshape::BrowDownLeft), 0.0);
}

#[test]
fn transient_hysteresis_does_not_thrash_and_recovery_requires_ready_streak() {
    let mut authority = gate();

    // Establish GNM authority before any instability.
    let (_, backend, _) = advance_frame(&mut authority, 1, GnmRuntimeHealth::Ready);
    assert_eq!(backend, FaceTrackingBackend::GnmTemporal);

    // First transient stays within hysteresis.
    let (_, backend, _) = advance_frame(
        &mut authority,
        2,
        GnmRuntimeHealth::Transient(GnmTransientIssue::ResidualSpike),
    );
    assert_eq!(backend, FaceTrackingBackend::GnmTemporal);

    // Crossing the threshold flips once and clears previous detailed state.
    let (_, backend, decision) = advance_frame(
        &mut authority,
        3,
        GnmRuntimeHealth::Transient(GnmTransientIssue::ResidualSpike),
    );
    assert_eq!(backend, FaceTrackingBackend::DirectMediaPipe);
    assert!(decision.clear_previous_output);

    // A single ready frame does not flip back.
    let (_, backend, _) = advance_frame(&mut authority, 4, GnmRuntimeHealth::Ready);
    assert_eq!(backend, FaceTrackingBackend::DirectMediaPipe);

    // The configured ready streak restores GNM exactly once, with clear.
    let (_, backend, _) = advance_frame(&mut authority, 5, GnmRuntimeHealth::Ready);
    assert_eq!(backend, FaceTrackingBackend::DirectMediaPipe);
    let (_, backend, decision) = advance_frame(&mut authority, 6, GnmRuntimeHealth::Ready);
    assert_eq!(backend, FaceTrackingBackend::GnmTemporal);
    assert!(decision.clear_previous_output);
}
