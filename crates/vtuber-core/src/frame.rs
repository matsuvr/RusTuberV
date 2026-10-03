//! Decoded frame layout and channel reading, shared by image consumers.

use crate::{PixelFormat, VideoFrame};

/// Failure to read the declared decoded image layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrameLayoutError {
    /// Both dimensions must be nonzero.
    ZeroDimension,
    /// A dimension, row or buffer size is not representable.
    SizeOverflow,
    /// A row stride cannot contain the declared pixels.
    StrideTooSmall {
        /// Minimum row bytes.
        expected: usize,
        /// Supplied stride.
        actual: usize,
    },
    /// The final pixel is outside the supplied bytes.
    BufferTooSmall {
        /// Required bytes, excluding final-row padding.
        expected: usize,
        /// Supplied bytes.
        actual: usize,
    },
    /// The requested pixel is outside the image.
    PixelOutOfBounds,
}

impl std::fmt::Display for FrameLayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "zero frame dimension"),
            Self::SizeOverflow => write!(f, "frame layout size overflow"),
            Self::StrideTooSmall { expected, actual } => {
                write!(f, "frame stride {actual} is smaller than {expected}")
            }
            Self::BufferTooSmall { expected, actual } => {
                write!(f, "frame buffer has {actual} bytes but requires {expected}")
            }
            Self::PixelOutOfBounds => write!(f, "pixel outside frame dimensions"),
        }
    }
}
impl std::error::Error for FrameLayoutError {}

impl PixelFormat {
    /// Number of bytes in one decoded pixel.
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgb8 | Self::Bgr8 => 3,
            Self::Rgba8 => 4,
            Self::Gray8 => 1,
        }
    }
}

impl VideoFrame {
    /// Validates dimensions, row stride and the last pixel's byte range once.
    /// Padding between rows must exist; padding after the final row is optional.
    pub fn pixels(&self) -> Result<FramePixels<'_>, FrameLayoutError> {
        if self.width == 0 || self.height == 0 {
            return Err(FrameLayoutError::ZeroDimension);
        }
        let width = usize::try_from(self.width).map_err(|_| FrameLayoutError::SizeOverflow)?;
        let height = usize::try_from(self.height).map_err(|_| FrameLayoutError::SizeOverflow)?;
        let row = width
            .checked_mul(self.format.bytes_per_pixel())
            .ok_or(FrameLayoutError::SizeOverflow)?;
        if self.stride_bytes < row {
            return Err(FrameLayoutError::StrideTooSmall {
                expected: row,
                actual: self.stride_bytes,
            });
        }
        let required = self
            .stride_bytes
            .checked_mul(height - 1)
            .and_then(|base| base.checked_add(row))
            .ok_or(FrameLayoutError::SizeOverflow)?;
        if self.data.len() < required {
            return Err(FrameLayoutError::BufferTooSmall {
                expected: required,
                actual: self.data.len(),
            });
        }
        Ok(FramePixels { frame: self })
    }
}

/// Borrowed pixels with a validated layout; channel reads do not revalidate it.
#[derive(Clone, Copy)]
pub struct FramePixels<'a> {
    frame: &'a VideoFrame,
}

impl FramePixels<'_> {
    /// Image width in pixels.
    pub fn width(self) -> usize {
        self.frame.width as usize
    }
    /// Image height in pixels.
    pub fn height(self) -> usize {
        self.frame.height as usize
    }

    /// Reads RGBA, expanding luminance and using opaque alpha for RGB/BGR.
    pub fn rgba(self, x: usize, y: usize) -> Result<[u8; 4], FrameLayoutError> {
        if x >= self.width() || y >= self.height() {
            return Err(FrameLayoutError::PixelOutOfBounds);
        }
        let offset = y * self.frame.stride_bytes + x * self.frame.format.bytes_per_pixel();
        let bytes = self
            .frame
            .data
            .get(offset..)
            .ok_or(FrameLayoutError::PixelOutOfBounds)?;
        let rgba = match (self.frame.format, bytes) {
            (PixelFormat::Rgb8, [r, g, b, ..]) => [*r, *g, *b, 255],
            (PixelFormat::Bgr8, [b, g, r, ..]) => [*r, *g, *b, 255],
            (PixelFormat::Rgba8, [r, g, b, a, ..]) => [*r, *g, *b, *a],
            (PixelFormat::Gray8, [v, ..]) => [*v, *v, *v, 255],
            _ => return Err(FrameLayoutError::PixelOutOfBounds),
        };
        Ok(rgba)
    }

    /// Returns packed RGB, borrowing packed RGB input or filling reusable storage.
    pub fn packed_rgb<'a>(self, staging: &'a mut Vec<u8>) -> Result<&'a [u8], FrameLayoutError>
    where
        Self: 'a,
    {
        let width = self.width();
        let length = width
            .checked_mul(self.height())
            .and_then(|n| n.checked_mul(3))
            .ok_or(FrameLayoutError::SizeOverflow)?;
        let row_bytes = width * 3;
        if self.frame.format == PixelFormat::Rgb8 && self.frame.stride_bytes == row_bytes {
            return self
                .frame
                .data
                .get(..length)
                .ok_or(FrameLayoutError::PixelOutOfBounds);
        }
        staging.resize(length, 0);
        for (y, row) in staging.chunks_exact_mut(row_bytes).enumerate() {
            for (x, destination) in row.chunks_exact_mut(3).enumerate() {
                let [r, g, b, _] = self.rgba(x, y)?;
                destination.copy_from_slice(&[r, g, b]);
            }
        }
        Ok(staging)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]
    use super::*;
    use crate::{FrameSeq, MonoTimeNs};
    use std::sync::Arc;

    #[test]
    fn formats_share_colour_order_and_optional_final_padding() {
        for (format, pixel, expected) in [
            (PixelFormat::Rgb8, vec![1, 2, 3], [1, 2, 3, 255]),
            (PixelFormat::Bgr8, vec![3, 2, 1], [1, 2, 3, 255]),
            (PixelFormat::Rgba8, vec![1, 2, 3, 4], [1, 2, 3, 4]),
            (PixelFormat::Gray8, vec![5], [5, 5, 5, 255]),
        ] {
            let mut data = pixel.clone();
            data.extend_from_slice(&[99; 3]);
            data.extend_from_slice(&pixel);
            let mut frame = VideoFrame {
                seq: FrameSeq(0),
                captured_at: MonoTimeNs(0),
                width: 1,
                height: 2,
                stride_bytes: pixel.len() + 3,
                format,
                data: Arc::from(data.clone()),
            };
            let mut staging = Vec::new();
            let expected_rgb = vec![
                expected[0],
                expected[1],
                expected[2],
                expected[0],
                expected[1],
                expected[2],
            ];
            for trailing_padding in [false, true] {
                if trailing_padding {
                    data.extend_from_slice(&[88; 3]);
                }
                frame.data = Arc::from(data.clone());
                let pixels = frame.pixels().unwrap();
                assert_eq!(pixels.rgba(0, 1).unwrap(), expected);
                assert_eq!(pixels.packed_rgb(&mut staging).unwrap(), expected_rgb);
                assert_eq!(pixels.rgba(1, 0), Err(FrameLayoutError::PixelOutOfBounds));
            }
            frame.data = Arc::from(vec![0; frame.stride_bytes + pixel.len() - 1]);
            assert!(matches!(
                frame.pixels(),
                Err(FrameLayoutError::BufferTooSmall { .. })
            ));
            frame.stride_bytes = pixel.len() - 1;
            assert!(matches!(
                frame.pixels(),
                Err(FrameLayoutError::StrideTooSmall { .. })
            ));
            frame.height = 0;
            assert!(matches!(
                frame.pixels(),
                Err(FrameLayoutError::ZeroDimension)
            ));
        }
    }
}
