//! Camera format negotiation.

use crate::device::{CameraError, CameraFormat, CameraRequest, RequestedFormat};
use vtuber_core::PixelFormat;

/// A format candidate reported by the backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FormatCandidate {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Frame rate numerator.
    pub fps_numerator: u32,
    /// Frame rate denominator.
    pub fps_denominator: u32,
    /// Pixel format used by the backend.
    pub format: PixelFormat,
}

/// Selects the candidate closest to the caller's requested size and frame rate.
///
/// There are no hidden resolution tiers: an exact request wins with a zero
/// distance, and an unavailable request chooses the nearest format the camera
/// actually reports. An explicit MJPEG or YUYV request is respected rather than
/// silently replaced. With [`RequestedFormat::Any`], MJPEG (`Rgb8`) wins only as
/// the final tie-breaker because it carries less USB traffic than YUYV.
pub fn select_format(
    request: &CameraRequest,
    candidates: &[FormatCandidate],
) -> Result<CameraFormat, CameraError> {
    let candidate = candidates
        .iter()
        .filter(|candidate| format_matches_request(candidate.format, request.format))
        .min_by_key(|candidate| (score(request, candidate), format_rank(candidate.format)))
        .copied()
        .ok_or(CameraError::NoSuitableFormat)?;
    Ok(candidate_to_format(candidate))
}

fn format_matches_request(candidate: PixelFormat, request: RequestedFormat) -> bool {
    match request {
        RequestedFormat::Any => true,
        RequestedFormat::Mjpeg => candidate == PixelFormat::Rgb8,
        RequestedFormat::Yuyv => candidate == PixelFormat::Bgr8,
    }
}

fn format_rank(candidate: PixelFormat) -> u8 {
    match candidate {
        PixelFormat::Rgb8 => 0,
        PixelFormat::Bgr8 => 1,
        _ => 2,
    }
}

fn score(request: &CameraRequest, candidate: &FormatCandidate) -> u64 {
    let dx = i64::from(candidate.width) - i64::from(request.width);
    let dy = i64::from(candidate.height) - i64::from(request.height);
    let candidate_fps =
        i64::from(candidate.fps_numerator) / i64::from(candidate.fps_denominator.max(1));
    let requested_fps =
        i64::from(request.fps_numerator) / i64::from(request.fps_denominator.max(1));
    let dfps = candidate_fps - requested_fps;
    (dx * dx + dy * dy) as u64 + (dfps * dfps) as u64 * 100
}

fn candidate_to_format(candidate: FormatCandidate) -> CameraFormat {
    CameraFormat {
        width: candidate.width,
        height: candidate.height,
        fps_numerator: candidate.fps_numerator,
        fps_denominator: candidate.fps_denominator,
        format: candidate.format,
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )] // tests may panic (AGENTS.md)
    use super::*;

    fn candidate(width: u32, height: u32, fps: u32, format: PixelFormat) -> FormatCandidate {
        FormatCandidate {
            width,
            height,
            fps_numerator: fps,
            fps_denominator: 1,
            format,
        }
    }

    #[test]
    fn default_request_is_360p30() {
        let request = CameraRequest::default();
        assert_eq!((request.width, request.height), (640, 360));
        assert_eq!((request.fps_numerator, request.fps_denominator), (30, 1));
    }

    #[test]
    fn exact_default_beats_the_old_720p_tier() {
        let request = CameraRequest::default();
        let candidates = [
            candidate(1280, 720, 30, PixelFormat::Rgb8),
            candidate(640, 360, 30, PixelFormat::Rgb8),
        ];
        let format = select_format(&request, &candidates).unwrap();
        assert_eq!((format.width, format.height), (640, 360));
        assert_eq!((format.fps_numerator, format.fps_denominator), (30, 1));
    }

    #[test]
    fn nearest_reported_format_is_used_when_360p_is_unavailable() {
        let request = CameraRequest::default();
        let candidates = [
            candidate(1280, 720, 30, PixelFormat::Rgb8),
            candidate(640, 480, 30, PixelFormat::Rgb8),
        ];
        let format = select_format(&request, &candidates).unwrap();
        assert_eq!((format.width, format.height), (640, 480));
    }

    #[test]
    fn any_uses_mjpeg_only_as_an_exact_tie_breaker() {
        let request = CameraRequest::default();
        let candidates = [
            candidate(640, 360, 30, PixelFormat::Bgr8),
            candidate(640, 360, 30, PixelFormat::Rgb8),
        ];
        let format = select_format(&request, &candidates).unwrap();
        assert_eq!(format.format, PixelFormat::Rgb8);
    }

    #[test]
    fn explicit_format_is_not_silently_replaced() {
        let request = CameraRequest {
            format: RequestedFormat::Mjpeg,
            ..CameraRequest::default()
        };
        let candidates = [candidate(640, 360, 30, PixelFormat::Bgr8)];
        let error = select_format(&request, &candidates).unwrap_err();
        assert!(matches!(error, CameraError::NoSuitableFormat));
    }

    #[test]
    fn empty_candidates_fails() {
        let error = select_format(&CameraRequest::default(), &[]).unwrap_err();
        assert!(matches!(error, CameraError::NoSuitableFormat));
    }
}
