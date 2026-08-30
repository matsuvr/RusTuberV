//! Detector-box to landmark-crop geometry and preprocessing.

use thiserror::Error;
use vtuber_core::types::{Landmark3, NormalizedRect, VideoFrame};

use crate::descriptor::{
    ChannelOrder, CropInterpolation, CropOutsideFill, FaceCropConfig, InputValueDomain,
    TensorContract, TensorLayout,
};
use crate::error::InferenceError;
use crate::preprocess::{read_rgb_pixel, validate_frame};

/// Coordinate encoding emitted by a landmark model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LandmarkCoordinateEncoding {
    /// Coordinates normalized to the crop's `[0, 1]` extent.
    Normalized0To1,
    /// Coordinates expressed in crop pixels.
    CropPixels,
}

impl LandmarkCoordinateEncoding {
    /// Parses the exact manifest spelling for this coordinate encoding.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "normalized_0_1" => Some(Self::Normalized0To1),
            "crop_pixels" => Some(Self::CropPixels),
            _ => None,
        }
    }

    /// Returns the manifest spelling for this encoding.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Normalized0To1 => "normalized_0_1",
            Self::CropPixels => "crop_pixels",
        }
    }
}

/// Typed failures from crop geometry or crop preprocessing.
#[derive(Debug, Error, PartialEq)]
pub enum CropError {
    /// A crop configuration or detector rectangle is invalid.
    #[error("invalid face crop: {0}")]
    InvalidCrop(&'static str),
    /// The frame dimensions do not match the transform dimensions.
    #[error("crop frame size mismatch: transform={transform:?}, frame={frame:?}")]
    FrameSizeMismatch {
        /// Dimensions used to construct the transform.
        transform: [u32; 2],
        /// Dimensions of the supplied frame.
        frame: [u32; 2],
    },
    /// The source frame is invalid for pixel sampling.
    #[error("invalid crop source frame: {0}")]
    Frame(#[from] InferenceError),
    /// The landmark model input contract does not match the crop.
    #[error("invalid landmark crop tensor contract: {0}")]
    TensorContract(&'static str),
    /// A landmark coordinate is not finite.
    #[error("landmark {index} has non-finite coordinate ({x}, {y})")]
    NonFiniteLandmark {
        /// Landmark index.
        index: usize,
        /// X coordinate.
        x: f32,
        /// Y coordinate.
        y: f32,
    },
}

/// A square crop in source-pixel space and its coordinate transforms.
///
/// The crop is deliberately allowed to extend outside the source frame. Its
/// full square extent is retained for both geometry and inverse landmark
/// mapping; the preprocessing stage supplies mean padding for outside pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceCropTransform {
    frame_size: [u32; 2],
    source_left_px: f32,
    source_top_px: f32,
    side_px: f32,
    output_size: [usize; 2],
    source_roi: NormalizedRect,
}

impl FaceCropTransform {
    /// Builds a source-pixel square from a normalized detector box.
    ///
    /// `square_scale` expands the larger detector-box side in source pixels.
    /// `center_y_offset_fraction` is relative to the detector-box pixel
    /// height, so the manifest's `-0.05` moves the crop center upward. The
    /// returned `source_roi` is normalized independently per frame axis and
    /// may extend outside `[0, 1]` when padding is required.
    pub fn from_detector_box(
        frame_width: u32,
        frame_height: u32,
        detector_box: &NormalizedRect,
        config: FaceCropConfig,
    ) -> Result<Self, CropError> {
        validate_crop_inputs(frame_width, frame_height, detector_box, config)?;

        let frame_w = frame_width as f32;
        let frame_h = frame_height as f32;
        let box_width_px = detector_box.width * frame_w;
        let box_height_px = detector_box.height * frame_h;
        let side_px = box_width_px.max(box_height_px) * config.square_scale;
        let center_x_px = (detector_box.x + detector_box.width * 0.5) * frame_w;
        let center_y_px = (detector_box.y + detector_box.height * 0.5) * frame_h
            + config.center_y_offset_fraction * box_height_px;
        let source_left_px = center_x_px - side_px * 0.5;
        let source_top_px = center_y_px - side_px * 0.5;

        Ok(Self {
            frame_size: [frame_width, frame_height],
            source_left_px,
            source_top_px,
            side_px,
            output_size: config.output_size,
            source_roi: NormalizedRect {
                x: source_left_px / frame_w,
                y: source_top_px / frame_h,
                width: side_px / frame_w,
                height: side_px / frame_h,
                rotation_rad: 0.0,
            },
        })
    }

    /// Returns the frame dimensions used to build this transform.
    #[must_use]
    pub const fn frame_size(&self) -> [u32; 2] {
        self.frame_size
    }

    /// Returns the landmark model output size as `[width, height]`.
    #[must_use]
    pub const fn output_size(&self) -> [usize; 2] {
        self.output_size
    }

    /// Returns the full, possibly out-of-frame crop as normalized source ROI.
    #[must_use]
    pub const fn source_roi(&self) -> NormalizedRect {
        self.source_roi
    }

    /// Returns `(left, top, side)` for the crop in source pixels.
    #[must_use]
    pub const fn source_pixel_square(&self) -> (f32, f32, f32) {
        (self.source_left_px, self.source_top_px, self.side_px)
    }

    /// Converts source normalized coordinates to source pixels.
    #[must_use]
    pub fn source_normalized_to_source_pixels(&self, x: f32, y: f32) -> (f32, f32) {
        (x * self.frame_size[0] as f32, y * self.frame_size[1] as f32)
    }

    /// Converts source pixels to source normalized coordinates.
    #[must_use]
    pub fn source_pixels_to_source_normalized(&self, x: f32, y: f32) -> (f32, f32) {
        (
            x / self.frame_size[0].max(1) as f32,
            y / self.frame_size[1].max(1) as f32,
        )
    }

    /// Converts source pixels to crop pixels. Coordinates may be outside the
    /// output extent when the source point is outside the crop.
    #[must_use]
    pub fn source_pixels_to_crop_pixels(&self, x: f32, y: f32) -> (f32, f32) {
        (
            (x - self.source_left_px) * self.output_size[0] as f32 / self.side_px,
            (y - self.source_top_px) * self.output_size[1] as f32 / self.side_px,
        )
    }

    /// Converts crop pixels to source pixels without clamping padding.
    #[must_use]
    pub fn crop_pixels_to_source_pixels(&self, x: f32, y: f32) -> (f32, f32) {
        (
            self.source_left_px + x * self.side_px / self.output_size[0] as f32,
            self.source_top_px + y * self.side_px / self.output_size[1] as f32,
        )
    }

    /// Converts source normalized coordinates to crop pixels.
    #[must_use]
    pub fn source_normalized_to_crop_pixels(&self, x: f32, y: f32) -> (f32, f32) {
        let (source_x, source_y) = self.source_normalized_to_source_pixels(x, y);
        self.source_pixels_to_crop_pixels(source_x, source_y)
    }

    /// Converts crop pixels to source normalized coordinates.
    #[must_use]
    pub fn crop_pixels_to_source_normalized(&self, x: f32, y: f32) -> (f32, f32) {
        let (source_x, source_y) = self.crop_pixels_to_source_pixels(x, y);
        self.source_pixels_to_source_normalized(source_x, source_y)
    }

    /// Converts landmark model coordinates to source normalized coordinates.
    #[must_use]
    pub fn landmark_model_to_source_normalized(
        &self,
        x: f32,
        y: f32,
        encoding: LandmarkCoordinateEncoding,
    ) -> (f32, f32) {
        let (crop_x, crop_y) = match encoding {
            LandmarkCoordinateEncoding::Normalized0To1 => (
                x * self.output_size[0] as f32,
                y * self.output_size[1] as f32,
            ),
            LandmarkCoordinateEncoding::CropPixels => (x, y),
        };
        self.crop_pixels_to_source_normalized(crop_x, crop_y)
    }

    /// Converts every landmark's x/y from model crop coordinates to source
    /// normalized coordinates, preserving depth and visibility.
    pub fn map_landmarks_to_source_normalized(
        &self,
        landmarks: &mut [Landmark3],
        encoding: LandmarkCoordinateEncoding,
    ) -> Result<(), CropError> {
        for (index, landmark) in landmarks.iter_mut().enumerate() {
            if !landmark.x.is_finite() || !landmark.y.is_finite() {
                return Err(CropError::NonFiniteLandmark {
                    index,
                    x: landmark.x,
                    y: landmark.y,
                });
            }
            let (x, y) = self.landmark_model_to_source_normalized(landmark.x, landmark.y, encoding);
            landmark.x = x;
            landmark.y = y;
        }
        Ok(())
    }
}

/// Reusable worker-owned buffers for a landmark crop tensor.
#[derive(Debug)]
pub struct FaceCropPreprocessBuffers {
    rgb: Vec<f32>,
    tensor: Vec<f32>,
    output_size: [usize; 2],
}

impl FaceCropPreprocessBuffers {
    /// Allocates buffers for a crop output size.
    pub fn new(output_size: [usize; 2]) -> Result<Self, CropError> {
        let pixels = checked_pixel_count(output_size)?;
        let channels = pixels
            .checked_mul(3)
            .ok_or(CropError::InvalidCrop("crop tensor dimensions overflow"))?;
        Ok(Self {
            rgb: vec![0.0; channels],
            tensor: vec![0.0; channels],
            output_size,
        })
    }

    /// Returns the tensor shape, always NCHW `[1, 3, height, width]`.
    #[must_use]
    pub const fn tensor_shape(&self) -> [usize; 4] {
        [1, 3, self.output_size[1], self.output_size[0]]
    }

    /// Returns the current tensor contents in NCHW order.
    #[must_use]
    pub fn tensor(&self) -> &[f32] {
        &self.tensor
    }

    /// Returns the current backing capacities for allocation-reuse tests.
    #[must_use]
    pub fn capacities(&self) -> (usize, usize) {
        (self.rgb.capacity(), self.tensor.capacity())
    }

    /// Samples a source frame into a padded, bilinear RGB Peppa tensor.
    ///
    /// The supplied contract must describe the manifest's unit-float RGB
    /// NCHW input. Values outside the source frame use the normalization mean
    /// before `(value - mean) / scale`, which therefore produces zero padding.
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub fn preprocess(
        &mut self,
        frame: &VideoFrame,
        transform: &FaceCropTransform,
        input: &TensorContract,
        crop_config: FaceCropConfig,
    ) -> Result<&[f32], CropError> {
        if transform.frame_size != [frame.width, frame.height] {
            return Err(CropError::FrameSizeMismatch {
                transform: transform.frame_size,
                frame: [frame.width, frame.height],
            });
        }
        if self.output_size != transform.output_size {
            return Err(CropError::TensorContract(
                "preprocess buffers and transform output sizes differ",
            ));
        }
        validate_frame(frame)?;
        validate_tensor_contract(transform, input, crop_config)?;
        let fill = match crop_config.outside_fill {
            CropOutsideFill::NormalizationMean => input.normalization.mean,
        };
        let [output_w, output_h] = self.output_size;
        let (_, _, side_px) = transform.source_pixel_square();
        let (left_px, top_px, _) = transform.source_pixel_square();

        for y in 0..output_h {
            let source_y = top_px + ((y as f32 + 0.5) * side_px / output_h as f32) - 0.5;
            for x in 0..output_w {
                let source_x = left_px + ((x as f32 + 0.5) * side_px / output_w as f32) - 0.5;
                let rgb = bilinear_rgb(frame, source_x, source_y, fill);
                let index = (y * output_w + x) * 3;
                self.rgb[index..index + 3].copy_from_slice(&rgb);
            }
        }

        let pixels = output_w * output_h;
        for index in 0..pixels {
            for channel in 0..3 {
                self.tensor[channel * pixels + index] = (self.rgb[index * 3 + channel]
                    - input.normalization.mean[channel])
                    / input.normalization.scale[channel];
            }
        }
        Ok(&self.tensor)
    }
}

fn validate_crop_inputs(
    frame_width: u32,
    frame_height: u32,
    detector_box: &NormalizedRect,
    config: FaceCropConfig,
) -> Result<(), CropError> {
    if frame_width == 0 || frame_height == 0 {
        return Err(CropError::InvalidCrop("frame dimensions must be positive"));
    }
    if !detector_box.x.is_finite()
        || !detector_box.y.is_finite()
        || !detector_box.width.is_finite()
        || !detector_box.height.is_finite()
        || detector_box.width <= 0.0
        || detector_box.height <= 0.0
    {
        return Err(CropError::InvalidCrop(
            "detector box must be finite and non-empty",
        ));
    }
    if !config.square_scale.is_finite()
        || config.square_scale <= 0.0
        || !config.center_y_offset_fraction.is_finite()
    {
        return Err(CropError::InvalidCrop(
            "crop scale and offset must be finite",
        ));
    }
    checked_pixel_count(config.output_size)?;
    Ok(())
}

fn checked_pixel_count(output_size: [usize; 2]) -> Result<usize, CropError> {
    if output_size.contains(&0) {
        return Err(CropError::InvalidCrop(
            "crop output dimensions must be positive",
        ));
    }
    output_size[0]
        .checked_mul(output_size[1])
        .ok_or(CropError::InvalidCrop("crop output dimensions overflow"))
}

fn validate_tensor_contract(
    transform: &FaceCropTransform,
    input: &TensorContract,
    crop_config: FaceCropConfig,
) -> Result<(), CropError> {
    if transform.output_size != crop_config.output_size {
        return Err(CropError::TensorContract(
            "transform and crop output sizes differ",
        ));
    }
    let [output_w, output_h] = transform.output_size;
    if input.shape != [1, 3, output_h, output_w]
        || input.dtype != "float32"
        || input.layout != TensorLayout::Nchw
        || input.channel_order != ChannelOrder::Rgb
        || input.value_domain != InputValueDomain::UnitFloat
    {
        return Err(CropError::TensorContract(
            "expected float32 unit-float RGB NCHW input",
        ));
    }
    if input
        .normalization
        .mean
        .iter()
        .any(|value| !value.is_finite())
        || input
            .normalization
            .scale
            .iter()
            .any(|value| !value.is_finite() || *value == 0.0)
    {
        return Err(CropError::TensorContract(
            "normalization must be finite and non-zero",
        ));
    }
    if !matches!(crop_config.interpolation, CropInterpolation::Bilinear)
        || !matches!(crop_config.outside_fill, CropOutsideFill::NormalizationMean)
    {
        return Err(CropError::TensorContract("unsupported crop policy"));
    }
    Ok(())
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn bilinear_rgb(frame: &VideoFrame, x: f32, y: f32, fill: [f32; 3]) -> [f32; 3] {
    let x0 = x.floor();
    let y0 = y.floor();
    let tx = x - x0;
    let ty = y - y0;
    let x0 = x0 as i64;
    let y0 = y0 as i64;

    let top_left = sample_rgb(frame, x0, y0, fill);
    let top_right = sample_rgb(frame, x0 + 1, y0, fill);
    let bottom_left = sample_rgb(frame, x0, y0 + 1, fill);
    let bottom_right = sample_rgb(frame, x0 + 1, y0 + 1, fill);

    let mut output = [0.0; 3];
    for channel in 0..3 {
        let top = top_left[channel] * (1.0 - tx) + top_right[channel] * tx;
        let bottom = bottom_left[channel] * (1.0 - tx) + bottom_right[channel] * tx;
        output[channel] = top * (1.0 - ty) + bottom * ty;
    }
    output
}

fn sample_rgb(frame: &VideoFrame, x: i64, y: i64, fill: [f32; 3]) -> [f32; 3] {
    if x < 0 || y < 0 || x >= i64::from(frame.width) || y >= i64::from(frame.height) {
        return fill;
    }
    let [r, g, b] = read_rgb_pixel(frame, x as u32, y as u32);
    [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]
}
