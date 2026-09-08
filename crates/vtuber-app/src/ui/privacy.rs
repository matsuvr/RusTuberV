//! Session-only consent for displaying camera pixels in the application window.
//! Navigation is never consent, and hiding the preview never stops tracking.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum CameraPreviewConsent {
    #[default]
    Hidden,
    Confirming,
    Visible,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CameraPreviewEvent {
    Request,
    Confirm,
    Hide,
}

/// Pure transition: only confirmation of an outstanding request reveals pixels.
pub(crate) fn next_camera_preview_consent(
    current: CameraPreviewConsent,
    event: CameraPreviewEvent,
) -> CameraPreviewConsent {
    use CameraPreviewConsent::{Confirming, Hidden, Visible};
    use CameraPreviewEvent::{Confirm, Hide, Request};
    match (current, event) {
        (_, Hide) => Hidden,
        (Hidden, Request) => Confirming,
        (Confirming, Confirm) => Visible,
        _ => current,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_a_page_or_requesting_a_preview_does_not_reveal_pixels() {
        let hidden = CameraPreviewConsent::default();
        assert_eq!(hidden, CameraPreviewConsent::Hidden);
        assert_eq!(next_camera_preview_consent(hidden, CameraPreviewEvent::Confirm), hidden);
        let requested = next_camera_preview_consent(hidden, CameraPreviewEvent::Request);
        assert_eq!(requested, CameraPreviewConsent::Confirming);
        assert_eq!(next_camera_preview_consent(requested, CameraPreviewEvent::Confirm), CameraPreviewConsent::Visible);
    }

    #[test]
    fn navigation_close_escape_and_camera_change_revoke_consent() {
        for state in [CameraPreviewConsent::Hidden, CameraPreviewConsent::Confirming, CameraPreviewConsent::Visible] {
            assert_eq!(next_camera_preview_consent(state, CameraPreviewEvent::Hide), CameraPreviewConsent::Hidden);
        }
    }
}
