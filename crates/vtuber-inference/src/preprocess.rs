//! Reusable preprocessing for face inference.
//!
//! Preprocessing converts a [`VideoFrame`] into the normalized float tensor
//! expected by the face model. Buffers are allocated once and reused so that
//! normal frames do not trigger allocations in the resize or normalize stages.

use vtuber_core::types::{PixelFormat, VideoFrame};

use crate::descriptor::{ChannelOrder, ModelDescriptor, Normalization};
use crate::error::{InferenceError, Result};

/// Reusable buffers for RGB resize and tensor backing.
///
/// The capacity of the internal vectors only grows when the target tensor
/// spatial size changes.
#[derive(Debug)]
pub struct PreprocessBuffers {
    resized_rgb: Vec<u8>,
    tensor: Vec<f32>,
    target_w: usize,
    target_h: usize,
}

/// Parameters describing the model's expected input tensor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreprocessParams {
    /// Full input shape, e.g. `[1, 256, 256, 3]` or `[1, 3, 256, 256]`.
    pub input_shape: [usize; 4],
    /// Channel order of the input image.
    pub channel_order: ChannelOrder,
    /// Normalization applied after converting to float.
    pub normalization: Normalization,
}

impl PreprocessParams {
    /// Builds parameters from a model descriptor.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::UnsupportedInputLayout`] if the descriptor's
    /// input shape is not a supported `[1, H, W, 3]` or `[1, 3, H, W]` layout.
    pub fn from_descriptor(descriptor: &ModelDescriptor) -> Result<Self> {
        let shape: [usize; 4] = descriptor.input_shape.as_slice().try_into().map_err(|_| {
            InferenceError::UnsupportedInputLayout {
                shape: descriptor.input_shape.clone(),
            }
        })?;
        Ok(Self {
            input_shape: shape,
            channel_order: descriptor.channel_order,
            normalization: descriptor.normalization,
        })
    }
}

impl PreprocessBuffers {
    /// Allocates buffers sized for the given target spatial resolution.
    #[must_use]
    pub fn new(target_w: usize, target_h: usize) -> Self {
        Self {
            resized_rgb: vec![0u8; target_w * target_h * 3],
            tensor: vec![0.0f32; target_w * target_h * 3],
            target_w,
            target_h,
        }
    }

    /// Returns buffers sized for `input_shape`, or an error if the layout is
    /// unsupported.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::UnsupportedInputLayout`] if `input_shape` is
    /// not `[1, H, W, 3]` or `[1, 3, H, W]`.
    pub fn for_shape(input_shape: &[usize; 4]) -> Result<Self> {
        let (target_h, target_w) = spatial_size(input_shape)?;
        Ok(Self::new(target_w, target_h))
    }

    /// Resizes the internal buffers only if the target size changed.
    pub fn ensure_size(&mut self, target_w: usize, target_h: usize) {
        if self.target_w == target_w && self.target_h == target_h {
            return;
        }
        let rgb_len = target_w
            .checked_mul(target_h)
            .and_then(|n| n.checked_mul(3));
        let tensor_len = rgb_len;
        if let (Some(rgb_len), Some(tensor_len)) = (rgb_len, tensor_len) {
            self.resized_rgb.resize(rgb_len, 0);
            self.tensor.resize(tensor_len, 0.0);
            self.target_w = target_w;
            self.target_h = target_h;
        }
    }

    /// Returns the current capacity of the RGB and tensor buffers.
    #[must_use]
    pub fn capacity(&self) -> (usize, usize) {
        (self.resized_rgb.capacity(), self.tensor.capacity())
    }

    /// Returns a reference to the tensor backing buffer.
    #[must_use]
    pub fn tensor(&self) -> &[f32] {
        &self.tensor
    }
}

/// Preprocesses `frame` into reusable buffers and returns a mutable reference
/// to the tensor backing vector.
///
/// The returned slice has length `target_h * target_w * 3` and is laid out
/// according to [`PreprocessParams::input_shape`].
///
/// # Errors
///
/// Returns a typed [`InferenceError`] if the frame stride or buffer size is
/// incompatible, or if the input layout is unsupported.
pub fn preprocess_frame<'b>(
    buffers: &'b mut PreprocessBuffers,
    frame: &VideoFrame,
    params: &PreprocessParams,
) -> Result<&'b mut Vec<f32>> {
    let (target_h, target_w) = spatial_size(&params.input_shape)?;
    buffers.ensure_size(target_w, target_h);

    validate_frame(frame)?;
    let (crop_size, offset_x, offset_y) = crop_rect(frame.width, frame.height);

    resize_to_rgb(buffers, frame, crop_size, offset_x, offset_y);
    normalize_and_layout(buffers, params);

    Ok(&mut buffers.tensor)
}

fn spatial_size(input_shape: &[usize; 4]) -> Result<(usize, usize)> {
    match input_shape[..] {
        [1, h, w, 3] => Ok((h, w)),
        [1, 3, h, w] => Ok((h, w)),
        _ => Err(InferenceError::UnsupportedInputLayout {
            shape: input_shape.to_vec(),
        }),
    }
}

pub(crate) fn validate_frame(frame: &VideoFrame) -> Result<()> {
    if frame.width == 0 || frame.height == 0 {
        return Err(InferenceError::InvalidInput("zero frame dimension".into()));
    }

    let bpp = bytes_per_pixel(frame.format);
    let expected_stride = (frame.width as usize)
        .checked_mul(bpp)
        .ok_or_else(|| InferenceError::InvalidInput("frame dimension overflow".into()))?;
    if frame.stride_bytes < expected_stride {
        return Err(InferenceError::FrameStrideMismatch {
            expected: expected_stride,
            actual: frame.stride_bytes,
        });
    }

    let expected_len = frame
        .stride_bytes
        .checked_mul((frame.height as usize).saturating_sub(1))
        .and_then(|base| base.checked_add(expected_stride))
        .ok_or_else(|| InferenceError::InvalidInput("frame size overflow".into()))?;
    if frame.data.len() < expected_len {
        return Err(InferenceError::FrameBufferTooSmall {
            expected: expected_len,
            actual: frame.data.len(),
        });
    }

    Ok(())
}

fn bytes_per_pixel(format: PixelFormat) -> usize {
    match format {
        PixelFormat::Rgb8 | PixelFormat::Bgr8 => 3,
        PixelFormat::Rgba8 => 4,
        PixelFormat::Gray8 => 1,
    }
}

fn crop_rect(width: u32, height: u32) -> (u32, u32, u32) {
    let crop_size = width.min(height);
    let offset_x = (width - crop_size) / 2;
    let offset_y = (height - crop_size) / 2;
    (crop_size, offset_x, offset_y)
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn resize_to_rgb(
    buffers: &mut PreprocessBuffers,
    frame: &VideoFrame,
    crop_size: u32,
    offset_x: u32,
    offset_y: u32,
) {
    let target_w = buffers.target_w;
    let target_h = buffers.target_h;

    for y in 0..target_h {
        let src_y = offset_y + (y as u32 * crop_size / target_h as u32);
        for x in 0..target_w {
            let src_x = offset_x + (x as u32 * crop_size / target_w as u32);
            let rgb = read_rgb_pixel(frame, src_x, src_y);
            let dst_idx = (y * target_w + x) * 3;
            buffers.resized_rgb[dst_idx] = rgb[0];
            buffers.resized_rgb[dst_idx + 1] = rgb[1];
            buffers.resized_rgb[dst_idx + 2] = rgb[2];
        }
    }
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
pub(crate) fn read_rgb_pixel(frame: &VideoFrame, x: u32, y: u32) -> [u8; 3] {
    let stride = frame.stride_bytes;
    let base = y as usize * stride
        + match frame.format {
            PixelFormat::Rgb8 | PixelFormat::Bgr8 => x as usize * 3,
            PixelFormat::Rgba8 => x as usize * 4,
            PixelFormat::Gray8 => x as usize,
        };

    match frame.format {
        PixelFormat::Rgb8 => [frame.data[base], frame.data[base + 1], frame.data[base + 2]],
        PixelFormat::Bgr8 => [frame.data[base + 2], frame.data[base + 1], frame.data[base]],
        PixelFormat::Rgba8 => [frame.data[base], frame.data[base + 1], frame.data[base + 2]],
        PixelFormat::Gray8 => {
            let v = frame.data[base];
            [v, v, v]
        }
    }
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn normalize_and_layout(buffers: &mut PreprocessBuffers, params: &PreprocessParams) {
    let (mean, std) = normalization_params(params.normalization);
    let target_w = buffers.target_w;
    let target_h = buffers.target_h;
    let count = target_w * target_h;

    match params.input_shape[..] {
        [1, 3, _, _] => {
            // NCHW layout: [batch, channel, height, width].
            for i in 0..count {
                let r = buffers.resized_rgb[i * 3] as f32 / 255.0;
                let g = buffers.resized_rgb[i * 3 + 1] as f32 / 255.0;
                let b = buffers.resized_rgb[i * 3 + 2] as f32 / 255.0;
                let (r, g, b) = reorder_channels((r, g, b), params.channel_order);
                buffers.tensor[i] = (r - mean[0]) / std[0];
                buffers.tensor[count + i] = (g - mean[1]) / std[1];
                buffers.tensor[count * 2 + i] = (b - mean[2]) / std[2];
            }
        }
        [1, _, _, 3] => {
            // NHWC layout: [batch, height, width, channel].
            for i in 0..count {
                let r = buffers.resized_rgb[i * 3] as f32 / 255.0;
                let g = buffers.resized_rgb[i * 3 + 1] as f32 / 255.0;
                let b = buffers.resized_rgb[i * 3 + 2] as f32 / 255.0;
                let (r, g, b) = reorder_channels((r, g, b), params.channel_order);
                buffers.tensor[i * 3] = (r - mean[0]) / std[0];
                buffers.tensor[i * 3 + 1] = (g - mean[1]) / std[1];
                buffers.tensor[i * 3 + 2] = (b - mean[2]) / std[2];
            }
        }
        _ => unreachable!("spatial_size already validated the input layout"),
    }
}

fn normalization_params(normalization: Normalization) -> ([f32; 3], [f32; 3]) {
    match normalization {
        Normalization::ZeroToOne => ([0.0; 3], [1.0; 3]),
        Normalization::MinusOneToOne => ([0.5; 3], [0.5; 3]),
        Normalization::MeanStd { mean, std } => (mean, std),
    }
}

fn reorder_channels((r, g, b): (f32, f32, f32), order: ChannelOrder) -> (f32, f32, f32) {
    match order {
        ChannelOrder::Bgr | ChannelOrder::Bgra => (b, g, r),
        _ => (r, g, b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vtuber_core::types::{FrameSeq, MonoTimeNs};

    fn make_frame(
        width: u32,
        height: u32,
        format: PixelFormat,
        data: Vec<u8>,
        stride: usize,
    ) -> VideoFrame {
        VideoFrame {
            seq: FrameSeq(1),
            captured_at: MonoTimeNs(0),
            width,
            height,
            stride_bytes: stride,
            format,
            data: data.into(),
        }
    }

    fn gradient_4x4_rgb() -> VideoFrame {
        let mut data = Vec::with_capacity(4 * 4 * 3);
        for y in 0..4 {
            for x in 0..4 {
                let r = (y * 4 + x) as u8;
                data.push(r);
                data.push(r.wrapping_add(16));
                data.push(r.wrapping_add(32));
            }
        }
        make_frame(4, 4, PixelFormat::Rgb8, data, 4 * 3)
    }

    fn zero_to_one_params(input_shape: [usize; 4]) -> PreprocessParams {
        PreprocessParams {
            input_shape,
            channel_order: ChannelOrder::Rgb,
            normalization: Normalization::ZeroToOne,
        }
    }

    #[test]
    fn preprocess_reuse_nchw_golden() {
        let frame = gradient_4x4_rgb();
        let params = zero_to_one_params([1, 3, 2, 2]);
        let mut buffers = PreprocessBuffers::for_shape(&params.input_shape).unwrap();
        let tensor = preprocess_frame(&mut buffers, &frame, &params).unwrap();

        // Center 2x2 crop samples (0,0), (2,0), (0,2), (2,2).
        let expected = vec![
            0.0_f32 / 255.0,
            2.0 / 255.0,
            8.0 / 255.0,
            10.0 / 255.0,
            16.0 / 255.0,
            18.0 / 255.0,
            24.0 / 255.0,
            26.0 / 255.0,
            32.0 / 255.0,
            34.0 / 255.0,
            40.0 / 255.0,
            42.0 / 255.0,
        ];
        assert_eq!(tensor[..], expected[..]);
    }

    #[test]
    fn preprocess_reuse_nhwc_golden() {
        let frame = gradient_4x4_rgb();
        let params = zero_to_one_params([1, 2, 2, 3]);
        let mut buffers = PreprocessBuffers::for_shape(&params.input_shape).unwrap();
        let tensor = preprocess_frame(&mut buffers, &frame, &params).unwrap();

        let expected = vec![
            0.0_f32 / 255.0,
            16.0 / 255.0,
            32.0 / 255.0,
            2.0 / 255.0,
            18.0 / 255.0,
            34.0 / 255.0,
            8.0 / 255.0,
            24.0 / 255.0,
            40.0 / 255.0,
            10.0 / 255.0,
            26.0 / 255.0,
            42.0 / 255.0,
        ];
        assert_eq!(tensor[..], expected[..]);
    }

    #[test]
    fn preprocess_reuse_bgr_swap() {
        // BGR8 data stores [B, G, R]. With a Bgr channel order the model
        // receives BGR layout, so the R and B tensor channels are swapped
        // relative to the RGB/Rgb result.
        let mut data = Vec::with_capacity(4 * 4 * 3);
        for y in 0..4 {
            for x in 0..4 {
                let r = (y * 4 + x) as u8;
                data.push(r.wrapping_add(32)); // B
                data.push(r.wrapping_add(16)); // G
                data.push(r); // R
            }
        }
        let frame = make_frame(4, 4, PixelFormat::Bgr8, data, 4 * 3);
        let params = PreprocessParams {
            input_shape: [1, 3, 2, 2],
            channel_order: ChannelOrder::Bgr,
            normalization: Normalization::ZeroToOne,
        };
        let mut buffers = PreprocessBuffers::for_shape(&params.input_shape).unwrap();
        let tensor = preprocess_frame(&mut buffers, &frame, &params).unwrap();

        let expected = vec![
            32.0_f32 / 255.0,
            34.0 / 255.0,
            40.0 / 255.0,
            42.0 / 255.0,
            16.0 / 255.0,
            18.0 / 255.0,
            24.0 / 255.0,
            26.0 / 255.0,
            0.0 / 255.0,
            2.0 / 255.0,
            8.0 / 255.0,
            10.0 / 255.0,
        ];
        assert_eq!(tensor[..], expected[..]);
    }

    #[test]
    fn preprocess_reuse_capacity_stable() {
        let frame = make_frame(4, 4, PixelFormat::Rgb8, vec![128u8; 4 * 4 * 3], 4 * 3);
        let params = zero_to_one_params([1, 3, 2, 2]);
        let mut buffers = PreprocessBuffers::for_shape(&params.input_shape).unwrap();
        let cap0 = buffers.capacity();

        preprocess_frame(&mut buffers, &frame, &params).unwrap();
        let cap1 = buffers.capacity();

        preprocess_frame(&mut buffers, &frame, &params).unwrap();
        let cap2 = buffers.capacity();

        assert_eq!(
            cap0, cap1,
            "capacity must not grow after first same-size frame"
        );
        assert_eq!(
            cap1, cap2,
            "capacity must stay stable across same-size frames"
        );
    }

    #[test]
    fn preprocess_reuse_invalid_stride() {
        // Stride smaller than width * bpp.
        let frame = make_frame(4, 4, PixelFormat::Rgb8, vec![0u8; 4 * 4 * 3], 4);
        let params = zero_to_one_params([1, 3, 2, 2]);
        let mut buffers = PreprocessBuffers::for_shape(&params.input_shape).unwrap();
        let err = preprocess_frame(&mut buffers, &frame, &params).unwrap_err();
        assert!(
            matches!(err, InferenceError::FrameStrideMismatch { .. }),
            "expected stride mismatch, got {err:?}"
        );
    }

    #[test]
    fn preprocess_reuse_invalid_buffer() {
        // Buffer too small for the declared resolution and stride.
        let frame = make_frame(4, 4, PixelFormat::Rgb8, vec![0u8; 4 * 3], 4 * 3);
        let params = zero_to_one_params([1, 3, 2, 2]);
        let mut buffers = PreprocessBuffers::for_shape(&params.input_shape).unwrap();
        let err = preprocess_frame(&mut buffers, &frame, &params).unwrap_err();
        assert!(
            matches!(err, InferenceError::FrameBufferTooSmall { .. }),
            "expected buffer too small, got {err:?}"
        );
    }

    #[test]
    fn preprocess_reuse_stride_padded_row() {
        // Each row is 4 RGB pixels (12 bytes) plus 4 bytes of padding.
        let mut data = Vec::with_capacity(4 * 16);
        for y in 0..4 {
            for x in 0..4 {
                let r = (y * 4 + x) as u8;
                data.push(r);
                data.push(r.wrapping_add(16));
                data.push(r.wrapping_add(32));
            }
            data.extend_from_slice(&[0, 0, 0, 0]);
        }
        let frame = make_frame(4, 4, PixelFormat::Rgb8, data, 4 * 3 + 4);
        let params = zero_to_one_params([1, 3, 2, 2]);
        let mut buffers = PreprocessBuffers::for_shape(&params.input_shape).unwrap();
        let tensor = preprocess_frame(&mut buffers, &frame, &params).unwrap();

        let expected = vec![
            0.0_f32 / 255.0,
            2.0 / 255.0,
            8.0 / 255.0,
            10.0 / 255.0,
            16.0 / 255.0,
            18.0 / 255.0,
            24.0 / 255.0,
            26.0 / 255.0,
            32.0 / 255.0,
            34.0 / 255.0,
            40.0 / 255.0,
            42.0 / 255.0,
        ];
        assert_eq!(tensor[..], expected[..]);
    }

    #[test]
    fn preprocess_reuse_resize_on_dimension_change() {
        let frame = make_frame(4, 4, PixelFormat::Rgb8, vec![128u8; 4 * 4 * 3], 4 * 3);
        let params = zero_to_one_params([1, 3, 2, 2]);
        let mut buffers = PreprocessBuffers::for_shape(&params.input_shape).unwrap();
        preprocess_frame(&mut buffers, &frame, &params).unwrap();
        let cap_small = buffers.capacity();

        let params_large = zero_to_one_params([1, 3, 4, 4]);
        preprocess_frame(&mut buffers, &frame, &params_large).unwrap();
        let cap_large = buffers.capacity();

        assert!(
            cap_large.0 >= cap_small.0 && cap_large.1 >= cap_small.1,
            "capacity must grow when target resolution increases"
        );

        // Returning to the smaller size must not reallocate above the large size.
        preprocess_frame(&mut buffers, &frame, &params).unwrap();
        let cap_returned = buffers.capacity();
        assert_eq!(cap_returned, cap_large);
    }
}
