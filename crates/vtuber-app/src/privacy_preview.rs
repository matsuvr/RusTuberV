//! Pure low-resolution conversion for the privacy-preserving camera preview.
//!
//! This module deliberately has no Bevy, ECS, or renderer integration. It
//! reads the captured frame directly and writes the small RGBA result in one
//! pass, so no source-sized intermediate image is created.

use std::ops::Range;

use thiserror::Error;
use vtuber_core::{PixelFormat, VideoFrame};

/// Maximum length of either output image edge.
pub const PRIVACY_PREVIEW_MAX_EDGE: u32 = 48;

/// A privacy-preview image containing tightly packed RGBA8 pixels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrivacyPreviewFrame {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Tightly packed row-major RGBA8 pixels.
    pub rgba: Vec<u8>,
}

/// Errors reported when a source frame cannot be converted safely.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum PrivacyPreviewError {
    /// The source has a zero width or height.
    #[error("privacy preview source dimensions must be non-zero")]
    ZeroDimension,
    /// A source dimension cannot be represented by the current platform.
    #[error("privacy preview source dimensions are not representable")]
    DimensionOverflow,
    /// The source row size overflowed while being calculated.
    #[error("privacy preview source row size overflowed")]
    RowBytesOverflow,
    /// The source stride is shorter than one complete row.
    #[error("privacy preview source stride {actual} is smaller than row size {minimum}")]
    InvalidStride {
        /// Actual source row stride.
        actual: usize,
        /// Minimum stride required for the declared format and width.
        minimum: usize,
    },
    /// The source buffer is shorter than the declared strided image.
    #[error("privacy preview source buffer is truncated: need {required} bytes, got {actual}")]
    DataTooShort {
        /// Number of bytes required by the declared dimensions and stride.
        required: usize,
        /// Number of bytes supplied by the source frame.
        actual: usize,
    },
    /// The declared strided image size overflowed while being calculated.
    #[error("privacy preview source buffer size overflowed")]
    BufferSizeOverflow,
    /// An internal pixel offset calculation could not be represented.
    #[error("privacy preview pixel offset overflowed")]
    PixelOffsetOverflow,
}

/// Converts a camera frame into a strongly downsampled RGBA privacy preview.
///
/// Every output pixel is the arithmetic mean of its corresponding source
/// rectangle. The source format and stride are read in place; the function
/// never constructs a source-sized RGBA buffer and never mutates the input.
pub fn build_privacy_preview(
    frame: &VideoFrame,
) -> Result<PrivacyPreviewFrame, PrivacyPreviewError> {
    let (source_width, source_height, channels) = validate_source(frame)?;
    let (output_width, output_height) = output_dimensions(source_width, source_height);
    let output_pixels = output_width
        .checked_mul(output_height)
        .ok_or(PrivacyPreviewError::PixelOffsetOverflow)?;
    let output_len = output_pixels
        .checked_mul(4)
        .ok_or(PrivacyPreviewError::PixelOffsetOverflow)?;
    let mut rgba = Vec::with_capacity(output_len);

    for output_y in 0..output_height {
        let source_y_range = block_range(output_y, source_height, output_height);
        for output_x in 0..output_width {
            let source_x_range = block_range(output_x, source_width, output_width);
            let mut sums = [0_u128; 4];
            let mut sample_count = 0_u128;

            for source_y in source_y_range.clone() {
                for source_x in source_x_range.clone() {
                    accumulate_pixel(frame, source_x, source_y, channels, &mut sums)?;
                    sample_count += 1;
                }
            }

            // Every block contains at least one source pixel because output
            // dimensions never exceed the corresponding source dimensions.
            debug_assert!(sample_count > 0);
            for sum in sums {
                rgba.push(average_channel(sum, sample_count));
            }
        }
    }

    Ok(PrivacyPreviewFrame {
        width: output_width as u32,
        height: output_height as u32,
        rgba,
    })
}

fn validate_source(frame: &VideoFrame) -> Result<(usize, usize, usize), PrivacyPreviewError> {
    if frame.width == 0 || frame.height == 0 {
        return Err(PrivacyPreviewError::ZeroDimension);
    }

    let width = usize::try_from(frame.width).map_err(|_| PrivacyPreviewError::DimensionOverflow)?;
    let height =
        usize::try_from(frame.height).map_err(|_| PrivacyPreviewError::DimensionOverflow)?;
    let channels = channels_for(frame.format);
    let row_bytes = width
        .checked_mul(channels)
        .ok_or(PrivacyPreviewError::RowBytesOverflow)?;
    if frame.stride_bytes < row_bytes {
        return Err(PrivacyPreviewError::InvalidStride {
            actual: frame.stride_bytes,
            minimum: row_bytes,
        });
    }
    let required = frame
        .stride_bytes
        .checked_mul(height)
        .ok_or(PrivacyPreviewError::BufferSizeOverflow)?;
    if frame.data.len() < required {
        return Err(PrivacyPreviewError::DataTooShort {
            required,
            actual: frame.data.len(),
        });
    }
    Ok((width, height, channels))
}

fn channels_for(format: PixelFormat) -> usize {
    match format {
        PixelFormat::Gray8 => 1,
        PixelFormat::Rgb8 | PixelFormat::Bgr8 => 3,
        PixelFormat::Rgba8 => 4,
    }
}

fn output_dimensions(source_width: usize, source_height: usize) -> (usize, usize) {
    // Infallible: the constant is a small `u32` literal.
    let max_edge = PRIVACY_PREVIEW_MAX_EDGE as usize;
    if source_width >= source_height {
        let width = source_width.min(max_edge);
        let height = rounded_ratio(source_height, width, source_width).max(1);
        (width, height)
    } else {
        let height = source_height.min(max_edge);
        let width = rounded_ratio(source_width, height, source_height).max(1);
        (width, height)
    }
}

// Invariant: inputs derive from validated `VideoFrame` dimensions, which are
// bounded well below `u64`, so every conversion below is lossless.
#[allow(clippy::expect_used)]
fn rounded_ratio(numerator: usize, scale: usize, denominator: usize) -> usize {
    // The scaled numerator is bounded by u32::MAX * 48, so u64 is sufficient
    // even on platforms where usize is wider than u32.
    let numerator = u64::try_from(numerator).expect("VideoFrame dimensions fit in u64");
    let scale = u64::try_from(scale).expect("preview dimensions fit in u64");
    let denominator = u64::try_from(denominator).expect("VideoFrame dimensions fit in u64");
    usize::try_from(((numerator * scale) + denominator / 2) / denominator)
        .expect("rounded preview dimension fits in usize")
}

// Invariant: inputs derive from validated `VideoFrame` and preview
// dimensions, bounded well below `u64`.
#[allow(clippy::expect_used)]
fn block_range(index: usize, source: usize, target: usize) -> Range<usize> {
    let index = u64::try_from(index).expect("preview dimensions fit in u64");
    let source = u64::try_from(source).expect("VideoFrame dimensions fit in u64");
    let target = u64::try_from(target).expect("preview dimensions fit in u64");
    let start = (index * source / target) as usize;
    let end_numerator = (index + 1) * source;
    let end = end_numerator.div_ceil(target).min(source) as usize;
    start..end.max(start + 1).min(source as usize)
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn accumulate_pixel(
    frame: &VideoFrame,
    source_x: usize,
    source_y: usize,
    channels: usize,
    sums: &mut [u128; 4],
) -> Result<(), PrivacyPreviewError> {
    let pixel_offset = source_y
        .checked_mul(frame.stride_bytes)
        .and_then(|row_offset| {
            source_x
                .checked_mul(channels)
                .and_then(|x| row_offset.checked_add(x))
        })
        .ok_or(PrivacyPreviewError::PixelOffsetOverflow)?;
    let pixel_end = pixel_offset
        .checked_add(channels)
        .ok_or(PrivacyPreviewError::PixelOffsetOverflow)?;
    let pixel =
        frame
            .data
            .get(pixel_offset..pixel_end)
            .ok_or(PrivacyPreviewError::DataTooShort {
                required: pixel_end,
                actual: frame.data.len(),
            })?;

    match frame.format {
        PixelFormat::Gray8 => {
            let value = u128::from(pixel[0]);
            sums[0] += value;
            sums[1] += value;
            sums[2] += value;
            sums[3] += 255;
        }
        PixelFormat::Rgb8 => {
            sums[0] += u128::from(pixel[0]);
            sums[1] += u128::from(pixel[1]);
            sums[2] += u128::from(pixel[2]);
            sums[3] += 255;
        }
        PixelFormat::Bgr8 => {
            sums[0] += u128::from(pixel[2]);
            sums[1] += u128::from(pixel[1]);
            sums[2] += u128::from(pixel[0]);
            sums[3] += 255;
        }
        PixelFormat::Rgba8 => {
            sums[0] += u128::from(pixel[0]);
            sums[1] += u128::from(pixel[1]);
            sums[2] += u128::from(pixel[2]);
            sums[3] += u128::from(pixel[3]);
        }
    }
    Ok(())
}

fn average_channel(sum: u128, sample_count: u128) -> u8 {
    ((sum + sample_count / 2) / sample_count).min(255) as u8
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use vtuber_core::{FrameSeq, MonoTimeNs};

    fn frame(
        width: u32,
        height: u32,
        stride_bytes: usize,
        format: PixelFormat,
        data: Vec<u8>,
    ) -> VideoFrame {
        VideoFrame {
            seq: FrameSeq(1),
            captured_at: MonoTimeNs(1),
            width,
            height,
            stride_bytes,
            format,
            data: Arc::from(data),
        }
    }

    fn solid_frame(width: u32, height: u32, pixel: &[u8]) -> VideoFrame {
        let row_bytes = width as usize * pixel.len();
        let mut data = Vec::with_capacity(row_bytes * height as usize);
        for _ in 0..(width as usize * height as usize) {
            data.extend_from_slice(pixel);
        }
        frame(width, height, row_bytes, PixelFormat::Rgba8, data)
    }

    #[test]
    fn widescreen_preview_is_48_by_27_without_source_sized_output() {
        let output = build_privacy_preview(&solid_frame(1920, 1080, &[12, 34, 56, 255]))
            .expect("valid widescreen frame");

        assert_eq!((output.width, output.height), (48, 27));
        assert_eq!(output.rgba.len(), 48 * 27 * 4);
        assert!(output.rgba.len() < 1920 * 1080 * 4);
        assert!(
            output
                .rgba
                .chunks_exact(4)
                .all(|pixel| pixel == [12, 34, 56, 255])
        );
    }

    #[test]
    fn aspect_ratio_is_preserved_for_common_orientations() {
        let cases = [
            (1280, 720, (48, 27)),
            (640, 480, (48, 36)),
            (100, 100, (48, 48)),
            (360, 640, (27, 48)),
        ];

        for (width, height, expected) in cases {
            let output = build_privacy_preview(&solid_frame(width, height, &[1, 2, 3, 255]))
                .expect("valid frame");
            assert_eq!((output.width, output.height), expected);
            assert!(output.width.max(output.height) <= PRIVACY_PREVIEW_MAX_EDGE);
        }
    }

    #[test]
    fn supported_formats_and_padded_stride_are_converted_without_mutating_input() {
        let cases = [
            (
                PixelFormat::Gray8,
                1,
                vec![42, 0, 0, 0],
                vec![42, 42, 42, 255],
            ),
            (
                PixelFormat::Rgb8,
                3,
                vec![10, 20, 30, 0, 0],
                vec![10, 20, 30, 255],
            ),
            (
                PixelFormat::Bgr8,
                3,
                vec![30, 20, 10, 0, 0],
                vec![10, 20, 30, 255],
            ),
            (
                PixelFormat::Rgba8,
                4,
                vec![10, 20, 30, 40, 0, 0],
                vec![10, 20, 30, 40],
            ),
        ];

        for (format, row_bytes, data, expected) in cases {
            let input = frame(1, 1, row_bytes.max(data.len()), format, data);
            let before = input.data.clone();
            let output = build_privacy_preview(&input).expect("valid one-pixel frame");
            assert_eq!(output.rgba, expected);
            assert_eq!(input.data.as_ref(), before.as_ref());
        }
    }

    #[test]
    fn box_average_blurs_high_frequency_detail_and_preserves_alpha_average() {
        let width = 96;
        let height = 48;
        let mut data = Vec::with_capacity(width as usize * height as usize * 4);
        for y in 0..height {
            for x in 0..width {
                let value = if (x + y) % 2 == 0 { 0 } else { 255 };
                data.extend_from_slice(&[value, 255 - value, 10, value]);
            }
        }
        let output = build_privacy_preview(&frame(
            width,
            height,
            width as usize * 4,
            PixelFormat::Rgba8,
            data,
        ))
        .expect("valid checkerboard frame");

        assert_eq!((output.width, output.height), (48, 24));
        assert_eq!(&output.rgba[..4], &[128, 128, 10, 128]);
    }

    #[test]
    fn invalid_dimensions_stride_buffer_and_overflow_are_rejected() {
        assert_eq!(
            build_privacy_preview(&frame(0, 1, 0, PixelFormat::Gray8, Vec::new())),
            Err(PrivacyPreviewError::ZeroDimension)
        );
        assert!(matches!(
            build_privacy_preview(&frame(2, 1, 1, PixelFormat::Rgb8, vec![0])),
            Err(PrivacyPreviewError::InvalidStride { .. })
        ));
        assert!(matches!(
            build_privacy_preview(&frame(2, 2, 6, PixelFormat::Rgb8, vec![0; 6])),
            Err(PrivacyPreviewError::DataTooShort { .. })
        ));
        assert!(matches!(
            build_privacy_preview(&frame(
                u32::MAX,
                u32::MAX,
                usize::MAX,
                PixelFormat::Rgba8,
                Vec::new(),
            )),
            Err(PrivacyPreviewError::BufferSizeOverflow)
                | Err(PrivacyPreviewError::DataTooShort { .. })
        ));
    }
}
