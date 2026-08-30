//! UltraFace's fixed RGB/NCHW input preprocessing.

use thiserror::Error;
use vtuber_core::types::{PixelFormat, VideoFrame};

/// Width of the fixed UltraFace RFB-320 input.
pub const ULTRAFACE_INPUT_WIDTH: usize = 320;
/// Height of the fixed UltraFace RFB-320 input.
pub const ULTRAFACE_INPUT_HEIGHT: usize = 240;

/// Per-channel normalization used by the accepted UltraFace artifact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectorNormalization {
    /// RGB channel means subtracted before scaling.
    pub mean: [f32; 3],
    /// RGB channel scales applied after subtracting the mean.
    pub scale: [f32; 3],
}

impl Default for DetectorNormalization {
    fn default() -> Self {
        Self {
            mean: [127.0; 3],
            scale: [128.0; 3],
        }
    }
}

/// Typed failures from detector-frame preprocessing.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum DetectorPreprocessError {
    /// The source frame has no pixels.
    #[error("source frame has a zero dimension: width={width} height={height}")]
    ZeroDimension {
        /// Source width.
        width: u32,
        /// Source height.
        height: u32,
    },
    /// The source pixel format is not one of the canonical decoded formats.
    #[error("unsupported detector pixel format: {format}")]
    UnsupportedPixelFormat {
        /// Debug name of the unsupported format.
        format: String,
    },
    /// The source stride cannot contain one complete row.
    #[error("source stride is too small: stride={actual} required={expected}")]
    StrideTooSmall {
        /// Actual source row stride.
        actual: usize,
        /// Minimum row stride for the selected format.
        expected: usize,
    },
    /// The owned frame buffer does not contain all rows described by its layout.
    #[error("source frame buffer is too short: length={actual} required={required}")]
    FrameBufferTooSmall {
        /// Actual buffer length.
        actual: usize,
        /// Required buffer length including row padding.
        required: usize,
    },
    /// A checked frame-layout multiplication overflowed.
    #[error("source frame layout is too large")]
    FrameLayoutOverflow,
    /// A normalization mean or scale is not finite.
    #[error("normalization setting is not finite at channel {channel}: mean={mean} scale={scale}")]
    NonFiniteNormalization {
        /// RGB channel index.
        channel: usize,
        /// Invalid mean value.
        mean: f32,
        /// Invalid scale value.
        scale: f32,
    },
    /// A normalization scale is zero and therefore cannot normalize a channel.
    #[error("normalization scale is zero at channel {channel}")]
    ZeroNormalizationScale {
        /// RGB channel index.
        channel: usize,
    },
}

#[derive(Clone, Copy, Debug)]
struct AxisSample {
    low: usize,
    high: usize,
    fraction: f32,
}

/// Reusable detector tensor storage owned by the inference worker.
///
/// Construct one instance per worker and pass it to [`Self::preprocess`] for
/// every frame. The tensor and resize coordinate tables are allocated only at
/// construction time; preprocessing itself writes into the existing tensor.
#[derive(Debug)]
pub struct UltraFacePreprocessBuffers {
    output_width: usize,
    output_height: usize,
    normalization: DetectorNormalization,
    source_width: usize,
    source_height: usize,
    x_samples: Vec<AxisSample>,
    y_samples: Vec<AxisSample>,
    tensor: Vec<f32>,
}

impl UltraFacePreprocessBuffers {
    /// Construct buffers for the manifest's fixed `[1, 3, 240, 320]` input.
    pub fn new() -> Self {
        Self::with_dimensions(
            ULTRAFACE_INPUT_WIDTH,
            ULTRAFACE_INPUT_HEIGHT,
            DetectorNormalization::default(),
        )
    }

    /// Construct buffers for a chosen output size.
    ///
    /// The production detector uses [`Self::new`]. This constructor keeps the
    /// resize algorithm independently testable with small synthetic tensors.
    pub fn with_dimensions(
        output_width: usize,
        output_height: usize,
        normalization: DetectorNormalization,
    ) -> Self {
        let tensor_len = output_width.saturating_mul(output_height).saturating_mul(3);
        Self {
            output_width,
            output_height,
            normalization,
            source_width: 0,
            source_height: 0,
            x_samples: Vec::new(),
            y_samples: Vec::new(),
            tensor: vec![0.0; tensor_len],
        }
    }

    /// Return the fixed tensor shape represented by these buffers.
    pub fn shape(&self) -> [usize; 4] {
        [1, 3, self.output_height, self.output_width]
    }

    /// Return the configured normalization contract.
    pub fn normalization(&self) -> DetectorNormalization {
        self.normalization
    }

    /// Return the reusable NCHW tensor data.
    pub fn tensor(&self) -> &[f32] {
        &self.tensor
    }

    /// Convert and resize one source frame into the reusable NCHW tensor.
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub fn preprocess(&mut self, frame: &VideoFrame) -> Result<&[f32], DetectorPreprocessError> {
        self.validate_normalization()?;
        let bytes_per_pixel = bytes_per_pixel(frame.format)?;
        if frame.width == 0 || frame.height == 0 {
            return Err(DetectorPreprocessError::ZeroDimension {
                width: frame.width,
                height: frame.height,
            });
        }

        let source_width = frame.width as usize;
        let source_height = frame.height as usize;
        let expected_stride = source_width
            .checked_mul(bytes_per_pixel)
            .ok_or(DetectorPreprocessError::FrameLayoutOverflow)?;
        if frame.stride_bytes < expected_stride {
            return Err(DetectorPreprocessError::StrideTooSmall {
                actual: frame.stride_bytes,
                expected: expected_stride,
            });
        }
        let required_len = frame
            .stride_bytes
            .checked_mul(source_height)
            .ok_or(DetectorPreprocessError::FrameLayoutOverflow)?;
        if frame.data.len() < required_len {
            return Err(DetectorPreprocessError::FrameBufferTooSmall {
                actual: frame.data.len(),
                required: required_len,
            });
        }

        if self.source_width != source_width || self.source_height != source_height {
            self.x_samples.clear();
            self.x_samples
                .extend(axis_samples(source_width, self.output_width));
            self.y_samples.clear();
            self.y_samples
                .extend(axis_samples(source_height, self.output_height));
            self.source_width = source_width;
            self.source_height = source_height;
        }
        let plane_len = self
            .output_width
            .checked_mul(self.output_height)
            .ok_or(DetectorPreprocessError::FrameLayoutOverflow)?;
        let expected_tensor_len = plane_len
            .checked_mul(3)
            .ok_or(DetectorPreprocessError::FrameLayoutOverflow)?;
        if self.tensor.len() != expected_tensor_len {
            return Err(DetectorPreprocessError::FrameLayoutOverflow);
        }

        for (output_y, y_sample) in self.y_samples.iter().copied().enumerate() {
            let row0 = y_sample.low * frame.stride_bytes;
            let row1 = y_sample.high * frame.stride_bytes;
            for (output_x, x_sample) in self.x_samples.iter().copied().enumerate() {
                let top_left = rgb_at(
                    &frame.data,
                    row0 + x_sample.low * bytes_per_pixel,
                    frame.format,
                );
                let top_right = rgb_at(
                    &frame.data,
                    row0 + x_sample.high * bytes_per_pixel,
                    frame.format,
                );
                let bottom_left = rgb_at(
                    &frame.data,
                    row1 + x_sample.low * bytes_per_pixel,
                    frame.format,
                );
                let bottom_right = rgb_at(
                    &frame.data,
                    row1 + x_sample.high * bytes_per_pixel,
                    frame.format,
                );
                let tensor_index = output_y * self.output_width + output_x;
                for channel in 0..3 {
                    let top = lerp(top_left[channel], top_right[channel], x_sample.fraction);
                    let bottom = lerp(
                        bottom_left[channel],
                        bottom_right[channel],
                        x_sample.fraction,
                    );
                    let pixel = lerp(top, bottom, y_sample.fraction);
                    self.tensor[channel * plane_len + tensor_index] = (pixel
                        - self.normalization.mean[channel])
                        / self.normalization.scale[channel];
                }
            }
        }

        Ok(&self.tensor)
    }

    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    fn validate_normalization(&self) -> Result<(), DetectorPreprocessError> {
        for channel in 0..3 {
            let mean = self.normalization.mean[channel];
            let scale = self.normalization.scale[channel];
            if !mean.is_finite() || !scale.is_finite() {
                return Err(DetectorPreprocessError::NonFiniteNormalization {
                    channel,
                    mean,
                    scale,
                });
            }
            if scale == 0.0 {
                return Err(DetectorPreprocessError::ZeroNormalizationScale { channel });
            }
        }
        Ok(())
    }
}

impl Default for UltraFacePreprocessBuffers {
    fn default() -> Self {
        Self::new()
    }
}

fn bytes_per_pixel(format: PixelFormat) -> Result<usize, DetectorPreprocessError> {
    match format {
        PixelFormat::Rgb8 | PixelFormat::Bgr8 => Ok(3),
        PixelFormat::Rgba8 => Ok(4),
        PixelFormat::Gray8 => Ok(1),
    }
}

fn axis_samples(source_len: usize, output_len: usize) -> Vec<AxisSample> {
    (0..output_len)
        .map(|output| {
            let source = ((output as f32 + 0.5) * source_len as f32 / output_len as f32) - 0.5;
            let clamped = source.clamp(0.0, (source_len - 1) as f32);
            let low = clamped.floor() as usize;
            let high = (low + 1).min(source_len - 1);
            AxisSample {
                low,
                high,
                fraction: clamped - low as f32,
            }
        })
        .collect()
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn rgb_at(data: &[u8], offset: usize, format: PixelFormat) -> [f32; 3] {
    match format {
        PixelFormat::Rgb8 => [
            data[offset] as f32,
            data[offset + 1] as f32,
            data[offset + 2] as f32,
        ],
        PixelFormat::Bgr8 => [
            data[offset + 2] as f32,
            data[offset + 1] as f32,
            data[offset] as f32,
        ],
        PixelFormat::Rgba8 => [
            data[offset] as f32,
            data[offset + 1] as f32,
            data[offset + 2] as f32,
        ],
        PixelFormat::Gray8 => {
            let value = data[offset] as f32;
            [value, value, value]
        }
    }
}

fn lerp(left: f32, right: f32, fraction: f32) -> f32 {
    left + (right - left) * fraction
}
