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

/// Selects the best matching format from the available candidates.
///
/// Priority:
/// 1. 1280x720 @ 30fps MJPEG
/// 2. 1280x720 @ 30fps YUYV
/// 3. 640x480 @ 30fps MJPEG
/// 4. 640x480 @ 30fps YUYV
/// 5. Closest 30fps format available
///
/// The requested format in `CameraRequest` biases the choice.
pub fn select_format(
    request: &CameraRequest,
    candidates: &[FormatCandidate],
) -> Result<CameraFormat, CameraError> {
    if candidates.is_empty() {
        return Err(CameraError::NoSuitableFormat);
    }

    let preferred = [
        (1280, 720, RequestedFormat::Mjpeg),
        (1280, 720, RequestedFormat::Yuyv),
        (640, 480, RequestedFormat::Mjpeg),
        (640, 480, RequestedFormat::Yuyv),
    ];

    // First pass: exact preferred format at 30fps.
    for (w, h, fmt) in &preferred {
        if let Some(c) = candidates.iter().find(|c| {
            c.width == *w
                && c.height == *h
                && c.fps_numerator == 30
                && c.fps_denominator == 1
                && format_matches_preference(c.format, *fmt, request.format)
        }) {
            return Ok(candidate_to_format(*c));
        }
    }

    // Second pass: closest 30fps format.
    if let Some(c) = candidates
        .iter()
        .filter(|c| c.fps_numerator == 30 && c.fps_denominator == 1)
        .min_by_key(|c| score(request, c))
    {
        return Ok(candidate_to_format(*c));
    }

    // Fallback: closest format at any frame rate.
    // Invariant: `candidates` is non-empty (checked at the top), so `first`
    // always exists.
    let fallback = candidates.first().copied();
    let c = candidates
        .iter()
        .min_by_key(|c| score(request, c))
        .copied()
        .or(fallback)
        .ok_or(CameraError::NoSuitableFormat)?;
    Ok(candidate_to_format(c))
}

fn format_matches_preference(
    candidate: PixelFormat,
    preference: RequestedFormat,
    request: RequestedFormat,
) -> bool {
    let effective = if request == RequestedFormat::Any {
        preference
    } else {
        request
    };
    match effective {
        RequestedFormat::Any => true,
        RequestedFormat::Mjpeg => candidate == PixelFormat::Rgb8,
        RequestedFormat::Yuyv => candidate == PixelFormat::Bgr8,
    }
}

fn score(request: &CameraRequest, candidate: &FormatCandidate) -> u64 {
    let dx = i64::from(candidate.width) - i64::from(request.width);
    let dy = i64::from(candidate.height) - i64::from(request.height);
    let dfps = i64::from(candidate.fps_numerator) / i64::from(candidate.fps_denominator.max(1))
        - i64::from(request.fps_numerator) / i64::from(request.fps_denominator.max(1));
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

    #[test]
    fn prefers_720p30_mjpeg() {
        let request = CameraRequest::default();
        let candidates = vec![
            FormatCandidate {
                width: 640,
                height: 480,
                fps_numerator: 30,
                fps_denominator: 1,
                format: PixelFormat::Rgb8,
            },
            FormatCandidate {
                width: 1280,
                height: 720,
                fps_numerator: 30,
                fps_denominator: 1,
                format: PixelFormat::Rgb8,
            },
        ];
        let format = select_format(&request, &candidates).unwrap();
        assert_eq!(format.width, 1280);
        assert_eq!(format.height, 720);
    }

    #[test]
    fn falls_back_to_480p30() {
        let request = CameraRequest::default();
        let candidates = vec![FormatCandidate {
            width: 640,
            height: 480,
            fps_numerator: 30,
            fps_denominator: 1,
            format: PixelFormat::Rgb8,
        }];
        let format = select_format(&request, &candidates).unwrap();
        assert_eq!(format.width, 640);
        assert_eq!(format.height, 480);
    }

    #[test]
    fn empty_candidates_fails() {
        let request = CameraRequest::default();
        let err = select_format(&request, &[]).unwrap_err();
        assert!(matches!(err, CameraError::NoSuitableFormat));
    }
}
