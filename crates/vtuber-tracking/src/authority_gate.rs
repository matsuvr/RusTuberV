//! Avatar-output authority gate connecting backend arbitration to the
//! canonical ARKit52 application path (GNM #57.3).
//!
//! This module applies [`advance_backend_selection`] decisions to concrete
//! outputs so that exactly one backend owns the published coefficients every
//! frame:
//!
//! - The two backends' channels are never summed or blended; the published
//!   payload is byte-equal to one backend's output.
//! - When GNM loses authority or is unavailable, Direct output passes through
//!   unchanged.
//! - On an authority change the caller receives
//!   [`AuthorityOutcome::clear_previous_output`], which instructs the avatar
//!   application to explicitly clear or coalesce the previous backend's
//!   detailed expression state instead of letting stale channels linger.
//! - Transient failures follow the existing hysteresis thresholds, so the
//!   backend does not thrash frame by frame.

use crate::ab_backend::{
    BackendSelectionConfig, BackendSelectionDecision, BackendSelectionState, FaceTrackingBackend,
    FaceTrackingMode, GnmRuntimeHealth, advance_backend_selection,
};

/// Result of applying arbitration to one frame's outputs.
#[derive(Clone, Debug, PartialEq)]
pub struct AuthorityOutcome<T> {
    /// The single authoritative payload to publish to the avatar.
    pub avatar_output: T,
    /// Backend that owns this payload.
    pub authority_backend: FaceTrackingBackend,
    /// True when authority changed this frame; the avatar application must
    /// explicitly clear/coalesce the previous detailed expression state.
    pub clear_previous_output: bool,
    /// Full arbitration decision carried forward for diagnostics.
    pub decision: BackendSelectionDecision,
}

/// Stateful authority gate over the pure arbitration state machine.
#[derive(Clone, Copy, Debug)]
pub struct AuthorityGate {
    selection_state: Option<BackendSelectionState>,
    config: BackendSelectionConfig,
}

impl AuthorityGate {
    /// Creates a gate with validated hysteresis configuration.
    ///
    /// # Errors
    ///
    /// Propagates invalid hysteresis thresholds from
    /// [`BackendSelectionConfig::new`].
    pub fn new(
        transient_failures_before_fallback: u32,
        ready_frames_before_recover: u32,
    ) -> Result<Self, crate::ab_backend::AbBackendError> {
        Ok(Self {
            selection_state: None,
            config: BackendSelectionConfig::new(
                transient_failures_before_fallback,
                ready_frames_before_recover,
            )?,
        })
    }

    /// Applies one frame of arbitration and selects the sole output payload.
    ///
    /// `gnm_output` is the decoded GNM coefficients for this exact frame when
    /// available. When GNM holds authority but its frame output is missing,
    /// the gate publishes the Direct payload for that frame rather than
    /// fabricating or summing channels; sustained unavailability is reported
    /// through the carried decision's fallback reason.
    pub fn advance<T>(
        &mut self,
        requested_mode: FaceTrackingMode,
        gnm_health: GnmRuntimeHealth,
        direct_output: T,
        gnm_output: Option<T>,
    ) -> AuthorityOutcome<T>
    where
        T: Clone,
    {
        let decision = advance_backend_selection(
            self.selection_state,
            requested_mode,
            gnm_health,
            self.config,
        );
        self.selection_state = Some(decision.state);

        let authority_backend = decision.state.avatar_backend;
        match (authority_backend, gnm_output) {
            (FaceTrackingBackend::GnmTemporal, Some(gnm)) => AuthorityOutcome {
                avatar_output: gnm,
                authority_backend,
                clear_previous_output: decision.clear_previous_output,
                decision,
            },
            // Direct authority, or GNM authority with a missing frame output:
            // publish the Direct payload untouched either way. Channels are
            // never combined.
            _ => AuthorityOutcome {
                avatar_output: direct_output,
                authority_backend: FaceTrackingBackend::DirectMediaPipe,
                clear_previous_output: decision.clear_previous_output,
                decision,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ab_backend::GnmRuntimeHealth;
    use crate::ab_backend::{GnmTransientIssue, GnmUnavailableReason};
    use vtuber_core::{ARKIT52_CHANNEL_COUNT, Arkit52Coefficients, ArkitBlendshape};

    fn coefficients(jaw_open: f32) -> Arkit52Coefficients {
        let mut values = [0.0; ARKIT52_CHANNEL_COUNT];
        values[ArkitBlendshape::JawOpen.index()] = jaw_open;
        Arkit52Coefficients::try_from_array(values).expect("valid coefficients")
    }

    fn gate() -> AuthorityGate {
        // 2 consecutive transients trigger fallback; 3 consecutive ready
        // frames recover GNM authority.
        AuthorityGate::new(2, 3).expect("valid hysteresis")
    }

    fn advance(
        authority: &mut AuthorityGate,
        health: GnmRuntimeHealth,
    ) -> AuthorityOutcome<Arkit52Coefficients> {
        authority.advance(
            FaceTrackingMode::GnmTemporal,
            health,
            coefficients(0.25),
            Some(coefficients(0.75)),
        )
    }

    #[test]
    fn gnm_not_ready_or_invalid_falls_back_to_direct() {
        for health in [
            GnmRuntimeHealth::Unavailable(GnmUnavailableReason::CalibrationUnavailable),
            GnmRuntimeHealth::Unavailable(GnmUnavailableReason::ModelInvalid),
            GnmRuntimeHealth::Unavailable(GnmUnavailableReason::MappingInvalid),
            GnmRuntimeHealth::Unavailable(GnmUnavailableReason::DecoderUnavailable),
        ] {
            let mut authority = gate();
            let outcome = advance(&mut authority, health);
            assert_eq!(
                outcome.authority_backend,
                FaceTrackingBackend::DirectMediaPipe
            );
            assert_eq!(outcome.avatar_output.get(ArkitBlendshape::JawOpen), 0.25);
        }
    }

    #[test]
    fn ready_gnm_holds_authority_and_publishes_only_gnm_channels() {
        let mut authority = gate();
        let outcome = advance(&mut authority, GnmRuntimeHealth::Ready);
        assert_eq!(outcome.authority_backend, FaceTrackingBackend::GnmTemporal);
        assert_eq!(outcome.avatar_output.get(ArkitBlendshape::JawOpen), 0.75);
        assert!(!outcome.clear_previous_output);
    }

    #[test]
    fn transient_failures_follow_hysteresis_without_frame_thrashing() {
        let mut authority = gate();
        // Establish GNM authority first.
        let outcome = advance(&mut authority, GnmRuntimeHealth::Ready);
        assert_eq!(outcome.authority_backend, FaceTrackingBackend::GnmTemporal);
        // First transient: hysteresis keeps GNM in authority.
        let outcome = advance(
            &mut authority,
            GnmRuntimeHealth::Transient(GnmTransientIssue::ResidualSpike),
        );
        assert_eq!(outcome.authority_backend, FaceTrackingBackend::GnmTemporal);
        assert_eq!(outcome.avatar_output.get(ArkitBlendshape::JawOpen), 0.75);
        // Second consecutive transient crosses the threshold exactly once.
        let outcome = advance(
            &mut authority,
            GnmRuntimeHealth::Transient(GnmTransientIssue::ResidualSpike),
        );
        assert_eq!(
            outcome.authority_backend,
            FaceTrackingBackend::DirectMediaPipe
        );
        assert!(outcome.clear_previous_output);
        // Recovery must not happen on a single ready frame.
        let outcome = advance(&mut authority, GnmRuntimeHealth::Ready);
        assert_eq!(
            outcome.authority_backend,
            FaceTrackingBackend::DirectMediaPipe
        );
        // ...but three consecutive ready frames restore GNM once.
        let outcome = advance(&mut authority, GnmRuntimeHealth::Ready);
        assert_eq!(
            outcome.authority_backend,
            FaceTrackingBackend::DirectMediaPipe
        );
        let outcome = advance(&mut authority, GnmRuntimeHealth::Ready);
        assert_eq!(outcome.authority_backend, FaceTrackingBackend::GnmTemporal);
        assert!(outcome.clear_previous_output);
        assert_eq!(outcome.avatar_output.get(ArkitBlendshape::JawOpen), 0.75);
    }

    #[test]
    fn gnm_authority_with_missing_frame_output_publishes_direct_untouched() {
        let mut authority = gate();
        let outcome = authority.advance(
            FaceTrackingMode::GnmTemporal,
            GnmRuntimeHealth::Ready,
            coefficients(0.25),
            None,
        );
        // Never fabricate or sum channels: the Direct payload passes through.
        assert_eq!(outcome.avatar_output, coefficients(0.25));
    }

    #[test]
    fn authority_changes_flag_explicit_clear_in_both_directions() {
        // GNM -> Direct.
        let mut authority = gate();
        let _ = advance(&mut authority, GnmRuntimeHealth::Ready);
        let outcome = advance(
            &mut authority,
            GnmRuntimeHealth::Unavailable(GnmUnavailableReason::StaleOutput),
        );
        assert!(outcome.clear_previous_output);

        // Direct -> GNM.
        let mut authority = gate();
        let outcome = authority.advance(
            FaceTrackingMode::GnmTemporal,
            GnmRuntimeHealth::Unavailable(GnmUnavailableReason::CalibrationUnavailable),
            coefficients(0.25),
            None,
        );
        assert_eq!(
            outcome.authority_backend,
            FaceTrackingBackend::DirectMediaPipe
        );
        for _ in 0..2 {
            let _ = advance(&mut authority, GnmRuntimeHealth::Ready);
        }
        let outcome = advance(&mut authority, GnmRuntimeHealth::Ready);
        assert_eq!(outcome.authority_backend, FaceTrackingBackend::GnmTemporal);
        assert!(outcome.clear_previous_output);
    }

    #[test]
    fn direct_mode_never_uses_gnm_output() {
        let mut authority = gate();
        let outcome = authority.advance(
            FaceTrackingMode::DirectMediaPipe,
            GnmRuntimeHealth::Ready,
            coefficients(0.25),
            Some(coefficients(0.75)),
        );
        assert_eq!(
            outcome.authority_backend,
            FaceTrackingBackend::DirectMediaPipe
        );
        assert_eq!(outcome.avatar_output.get(ArkitBlendshape::JawOpen), 0.25);
    }
}
