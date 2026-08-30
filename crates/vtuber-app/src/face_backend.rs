//! Face-tracking backend selection: requested mode, persistence, authority.
//!
//! This module owns the minimal user-facing selector state for the three face
//! tracking backends (`Direct MediaPipe`, `GNM Temporal (Experimental)`,
//! `GNM Shadow`), maps the selection uniquely onto the runtime
//! [`FaceTrackingMode`], and keeps the actual output authority visible via
//! [`crate::diagnostics::DiagnosticsSnapshot`].
//!
//! The selection reuses the application settings persistence boundary; an
//! invalid or legacy persisted value falls back to Direct instead of blocking
//! startup.

use bevy::prelude::*;
use vtuber_tracking::{
    AuthorityGate, FaceTrackingBackend, FaceTrackingMode, GnmRuntimeHealth, GnmUnavailableReason,
};

use crate::settings::{ArmPoseSettings, load_face_tracking_mode};

/// User-selectable face tracking backends shown in the settings UI.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FaceTrackingBackendSelection {
    /// The current production MediaPipe path.
    #[default]
    DirectMediaPipe,
    /// GNM temporal fitting as the avatar-output authority (experimental).
    GnmTemporal,
    /// GNM evaluated side-by-side while Direct keeps authority.
    GnmShadow,
}

impl FaceTrackingBackendSelection {
    /// All selections in UI order.
    #[must_use]
    pub fn all() -> [Self; 3] {
        [Self::DirectMediaPipe, Self::GnmTemporal, Self::GnmShadow]
    }

    /// Stable UI label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::DirectMediaPipe => "Direct MediaPipe",
            Self::GnmTemporal => "GNM Temporal (Experimental)",
            Self::GnmShadow => "GNM Shadow",
        }
    }

    /// Stable persistence token written to settings.
    #[must_use]
    pub fn as_persisted_str(self) -> &'static str {
        match self {
            Self::DirectMediaPipe => "direct",
            Self::GnmTemporal => "gnm_temporal",
            Self::GnmShadow => "gnm_shadow",
        }
    }

    /// Parses a persistence token. Unknown tokens yield `None` so callers can
    /// fall back to Direct without failing startup.
    #[must_use]
    pub fn from_persisted_str(value: &str) -> Option<Self> {
        match value {
            "direct" => Some(Self::DirectMediaPipe),
            "gnm_temporal" => Some(Self::GnmTemporal),
            "gnm_shadow" => Some(Self::GnmShadow),
            _ => None,
        }
    }

    /// Unique conversion into the runtime arbitration request.
    ///
    /// The shadow selection requests evaluation of the GNM side while Direct
    /// keeps sole authority, exactly matching [`FaceTrackingMode`]'s meaning.
    #[must_use]
    pub fn to_runtime_mode(self) -> FaceTrackingMode {
        match self {
            Self::DirectMediaPipe => FaceTrackingMode::DirectMediaPipe,
            Self::GnmTemporal | Self::GnmShadow => FaceTrackingMode::GnmTemporal,
        }
    }
}

/// Requested face tracking backend plus the observed runtime authority.
#[derive(Resource)]
pub struct FaceTrackingBackendState {
    /// Backend the user selected.
    pub requested: FaceTrackingBackendSelection,
    /// Backend that currently owns avatar output.
    pub authority: FaceTrackingBackend,
    /// Why `authority` differs from the GNM request, when applicable.
    pub fallback_reason: Option<String>,
    /// Hysteresis gate connecting the request to published authority.
    gate: AuthorityGate,
}

impl Default for FaceTrackingBackendState {
    fn default() -> Self {
        // Invariant: (2, 3) are positive hysteresis thresholds accepted by
        // construction; repository tests pin this constructor.
        #[allow(clippy::expect_used)]
        let gate = AuthorityGate::new(2, 3).expect("fixed hysteresis thresholds are valid");
        Self {
            requested: FaceTrackingBackendSelection::DirectMediaPipe,
            authority: FaceTrackingBackend::DirectMediaPipe,
            fallback_reason: None,
            gate,
        }
    }
}

impl FaceTrackingBackendState {
    /// Applies a user selection.
    pub fn set_requested(&mut self, requested: FaceTrackingBackendSelection) {
        self.requested = requested;
    }

    /// Advances one frame of authority arbitration with the current health.
    ///
    /// The GNM decoder is not configured in this build yet, so its health is
    /// reported as unavailable rather than fabricated as ready; Direct stays
    /// the sole authority until a real GNM runtime exists behind the gate.
    pub fn advance(&mut self, gnm_decoder_available: bool) {
        let health = if gnm_decoder_available {
            GnmRuntimeHealth::Ready
        } else {
            GnmRuntimeHealth::Unavailable(GnmUnavailableReason::DecoderUnavailable)
        };
        let outcome = self.gate.advance(
            self.requested.to_runtime_mode(),
            health,
            (),
            if matches!(health, GnmRuntimeHealth::Ready) {
                Some(())
            } else {
                None
            },
        );
        self.authority = outcome.authority_backend;
        self.fallback_reason = decision_fallback_reason(&outcome.decision);
    }
}

fn decision_fallback_reason(
    decision: &vtuber_tracking::BackendSelectionDecision,
) -> Option<String> {
    decision
        .state
        .fallback_reason
        .map(|reason| format!("{reason:?}"))
}

/// Restores the persisted backend selection before the first frame.
pub fn restore_face_backend_selection_system(
    settings: Res<ArmPoseSettings>,
    mut state: ResMut<FaceTrackingBackendState>,
) {
    let Some(path) = settings.path() else {
        return;
    };
    match load_face_tracking_mode(path) {
        Ok(Some(selection)) => state.requested = selection,
        Ok(None) => {}
        Err(error) => bevy::log::warn!("face tracking selection ignored: {error}"),
    }
}

/// Publishes the requested mode, actual authority, and fallback reason into
/// the diagnostics snapshot each frame.
pub fn sync_face_backend_diagnostics_system(
    mut state: ResMut<FaceTrackingBackendState>,
    mut snapshot: ResMut<crate::diagnostics::DiagnosticsSnapshot>,
) {
    state.advance(false);
    snapshot.face_tracking_requested = Some(state.requested.label().to_owned());
    snapshot.face_tracking_authority = Some(format!("{:?}", state.authority));
    snapshot.face_tracking_fallback_reason = state.fallback_reason.clone();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_selection_maps_uniquely_onto_runtime_modes() {
        assert_eq!(
            FaceTrackingBackendSelection::DirectMediaPipe.to_runtime_mode(),
            FaceTrackingMode::DirectMediaPipe
        );
        assert_eq!(
            FaceTrackingBackendSelection::GnmTemporal.to_runtime_mode(),
            FaceTrackingMode::GnmTemporal
        );
        // Shadow evaluates GNM but never takes avatar authority by request.
        assert_eq!(
            FaceTrackingBackendSelection::GnmShadow.to_runtime_mode(),
            FaceTrackingMode::GnmTemporal
        );
    }

    #[test]
    fn persisted_tokens_round_trip_and_unknown_values_fall_back() {
        for selection in FaceTrackingBackendSelection::all() {
            assert_eq!(
                FaceTrackingBackendSelection::from_persisted_str(selection.as_persisted_str()),
                Some(selection)
            );
        }
        assert_eq!(
            FaceTrackingBackendSelection::from_persisted_str("experimental_gnm_v9"),
            None
        );
    }

    #[test]
    fn unavailable_decoder_keeps_direct_authority_with_reason() {
        let mut state = FaceTrackingBackendState::default();
        state.set_requested(FaceTrackingBackendSelection::GnmTemporal);
        state.advance(false);
        assert_eq!(state.requested, FaceTrackingBackendSelection::GnmTemporal);
        assert_eq!(state.authority, FaceTrackingBackend::DirectMediaPipe);
        let reason = state.fallback_reason.expect("fallback reason");
        assert!(
            reason.contains("DecoderUnavailable") || reason.contains("decoder"),
            "unexpected reason: {reason}"
        );
    }
}
