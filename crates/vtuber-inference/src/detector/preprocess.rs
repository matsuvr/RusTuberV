//! UltraFace's fixed RGB/NCHW input preprocessing.

use thiserror::Error;
use vtuber_core::VideoFrame;

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
    /// Decoded frame layout or pixel read failure.
    #[error(transparent)]
    Frame(#[from] vtuber_core::frame::FrameLayoutError),
    /// The detector tensor size cannot be represented.
    #[error("detector tensor size overflow")]
    TensorSizeOverflow,
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
    #[expect(
        clippy::indexing_slicing,
        reason = "the stride and buffer-length checks above prove the sampled rows exist, and the tensor-length check proves the NCHW writes fit"
    )]
    pub fn preprocess(&mut self, frame: &VideoFrame) -> Result<&[f32], DetectorPreprocessError> {
        self.validate_normalization()?;
        let pixels = frame.pixels()?;
        let source_width = pixels.width();
        let source_height = pixels.height();

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
            .ok_or(DetectorPreprocessError::TensorSizeOverflow)?;
        let expected_tensor_len = plane_len
            .checked_mul(3)
            .ok_or(DetectorPreprocessError::TensorSizeOverflow)?;
        if self.tensor.len() != expected_tensor_len {
            return Err(DetectorPreprocessError::TensorSizeOverflow);
        }

        for (output_y, y_sample) in self.y_samples.iter().copied().enumerate() {
            for (output_x, x_sample) in self.x_samples.iter().copied().enumerate() {
                let top_left = pixels.rgba(x_sample.low, y_sample.low)?.map(f32::from);
                let top_right = pixels.rgba(x_sample.high, y_sample.low)?.map(f32::from);
                let bottom_left = pixels.rgba(x_sample.low, y_sample.high)?.map(f32::from);
                let bottom_right = pixels.rgba(x_sample.high, y_sample.high)?.map(f32::from);
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

    #[expect(
        clippy::indexing_slicing,
        reason = "mean and scale are fixed three-element arrays and the loop is bounded by 3"
    )]
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

fn lerp(left: f32, right: f32, fraction: f32) -> f32 {
    left + (right - left) * fraction
}
