//! Decode face landmark model outputs into engine-independent [`Landmark3`] values.
//!
//! Decoding is driven by a [`LandmarkOutputContract`] taken from the model
//! manifest, so output tensor names, shapes, and channel meanings are not
//! hard-coded in the decoder source.

use vtuber_core::types::Landmark3;

use crate::error::{InferenceError, Result};
use crate::roi::FaceRoi;

/// Data type of an output tensor element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorDtype {
    /// 32-bit IEEE-754 floating point.
    F32,
    /// 16-bit IEEE-754 floating point.
    F16,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
}

/// A borrowed view of an output tensor.
///
/// The caller is responsible for ensuring that `data` contains the elements
/// described by `shape` in row-major order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutputTensor<'a> {
    /// Tensor shape in row-major order.
    pub shape: &'a [usize],
    /// Tensor data in row-major order.
    pub data: &'a [f32],
    /// Element data type of the source tensor.
    pub dtype: TensorDtype,
}

/// Normalized or pixel coordinate range for landmark x/y values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoordinateRange {
    /// Values are already normalized to `[0, 1]`.
    ZeroToOne,
    /// Values are in `[-1, 1]` and are mapped to `[0, 1]`.
    MinusOneToOne,
    /// Values are pixel coordinates in the model input and are divided by
    /// [`LandmarkOutputContract::canonical_size`].
    Pixel,
}

/// Channels stored per landmark in the output tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LandmarkChannels {
    /// `x, y`.
    Xy,
    /// `x, y, confidence`.
    XyConfidence,
    /// `x, y, z`.
    Xyz,
    /// `x, y, z, confidence`.
    XyzConfidence,
}

impl LandmarkChannels {
    /// Number of values stored per landmark.
    #[must_use]
    pub fn values_per_landmark(self) -> usize {
        match self {
            Self::Xy => 2,
            Self::XyConfidence | Self::Xyz => 3,
            Self::XyzConfidence => 4,
        }
    }

    /// Whether the channel set includes a confidence or visibility value.
    #[must_use]
    pub fn has_confidence(self) -> bool {
        matches!(self, Self::XyConfidence | Self::XyzConfidence)
    }

    /// Whether the channel set includes a z or depth value.
    #[must_use]
    pub fn has_z(self) -> bool {
        matches!(self, Self::Xyz | Self::XyzConfidence)
    }
}

/// Layout of the landmark output tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LandmarkLayout {
    /// `[batch, landmark_count, channels]`.
    BatchLandmarkChannels,
    /// `[landmark_count, channels]`.
    LandmarkChannels,
}

/// Output contract for a face landmark model.
///
/// This contract is extracted from the model manifest and describes how to
/// interpret the landmark output tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct LandmarkOutputContract {
    /// Name of the output tensor in the runtime result map.
    pub tensor_name: String,
    /// Expected full tensor shape.
    pub expected_shape: Vec<usize>,
    /// Expected element data type.
    pub expected_dtype: TensorDtype,
    /// Number of landmarks in the output.
    pub landmark_count: usize,
    /// Coordinate range of the x/y channels.
    pub coordinate_range: CoordinateRange,
    /// Channels stored per landmark.
    pub channels: LandmarkChannels,
    /// Layout of the output tensor.
    pub layout: LandmarkLayout,
    /// Spatial size of the model input, used to convert pixel coordinates.
    pub canonical_size: u32,
}

impl LandmarkOutputContract {
    /// Number of values stored per landmark.
    #[must_use]
    pub fn values_per_landmark(&self) -> usize {
        self.channels.values_per_landmark()
    }

    /// Expected total element count derived from the manifest shape.
    #[must_use]
    pub fn expected_element_count(&self) -> usize {
        self.expected_shape.iter().product()
    }
}

/// Decodes a landmark output tensor into a vector of [`Landmark3`] values.
///
/// Landmark coordinates are converted from ROI-local normalized coordinates
/// to normalized image coordinates using the supplied `roi`.
///
/// # Errors
///
/// Returns a typed [`InferenceError`] if the tensor shape, dtype, element
/// count, or numeric values do not match the contract.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
pub fn decode_landmarks(
    tensor: &OutputTensor,
    contract: &LandmarkOutputContract,
    roi: &FaceRoi,
    frame_w: u32,
    frame_h: u32,
) -> Result<Vec<Landmark3>> {
    validate_output_tensor(tensor, contract)?;

    if frame_w == 0 || frame_h == 0 {
        return Err(InferenceError::InvalidInput(
            "cannot decode landmarks for zero-sized frame".into(),
        ));
    }

    let rank = tensor.shape.len();
    let count = tensor.shape[rank - 2];
    let row_stride = tensor.shape[rank - 1];

    let roi_transform = RoiLocalToImageTransform::new(roi, frame_w, frame_h)
        .ok_or_else(|| InferenceError::InvalidRoi("zero frame dimension".into()))?;

    let mut landmarks = Vec::with_capacity(count);

    for i in 0..count {
        let base = i * row_stride;
        let x_raw = tensor.data[base];
        let y_raw = tensor.data[base + 1];

        let (x_local, y_local) = normalize_local_coordinates(x_raw, y_raw, contract);

        let (x_img, y_img) = roi_transform.apply(x_local, y_local);

        let z = if contract.channels.has_z() {
            tensor.data[base + 2]
        } else {
            0.0
        };

        let visibility = if contract.channels.has_confidence() {
            tensor.data[base + row_stride - 1].clamp(0.0, 1.0)
        } else {
            1.0
        };

        landmarks.push(Landmark3 {
            x: x_img / frame_w as f32,
            y: y_img / frame_h as f32,
            z,
            visibility,
        });
    }

    Ok(landmarks)
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn validate_output_tensor(tensor: &OutputTensor, contract: &LandmarkOutputContract) -> Result<()> {
    if tensor.dtype != contract.expected_dtype {
        return Err(InferenceError::OutputDtypeMismatch {
            expected: format!("{:?}", contract.expected_dtype),
            actual: format!("{:?}", tensor.dtype),
        });
    }

    if tensor.shape != contract.expected_shape.as_slice() {
        return Err(InferenceError::OutputShapeMismatch {
            expected: contract.expected_shape.clone(),
            actual: tensor.shape.to_vec(),
        });
    }

    let expected = contract.expected_element_count();
    let actual = tensor.data.len();
    if actual != expected {
        return Err(InferenceError::OutputElementCountMismatch { expected, actual });
    }

    let rank = tensor.shape.len();
    if rank < 2 {
        return Err(InferenceError::OutputShapeMismatch {
            expected: contract.expected_shape.clone(),
            actual: tensor.shape.to_vec(),
        });
    }

    let channels = contract.values_per_landmark();
    let actual_channels = tensor.shape[rank - 1];
    if actual_channels != channels {
        return Err(InferenceError::OutputShapeMismatch {
            expected: contract.expected_shape.clone(),
            actual: tensor.shape.to_vec(),
        });
    }

    for (index, &value) in tensor.data.iter().enumerate() {
        if !value.is_finite() {
            return Err(InferenceError::InvalidOutputValue { index, value });
        }
    }

    Ok(())
}

fn normalize_local_coordinates(
    x_raw: f32,
    y_raw: f32,
    contract: &LandmarkOutputContract,
) -> (f32, f32) {
    match contract.coordinate_range {
        CoordinateRange::ZeroToOne => (x_raw, y_raw),
        CoordinateRange::MinusOneToOne => ((x_raw + 1.0) * 0.5, (y_raw + 1.0) * 0.5),
        CoordinateRange::Pixel => {
            let size = contract.canonical_size.max(1) as f32;
            (x_raw / size, y_raw / size)
        }
    }
}

/// Precomputed [`roi_local_to_image`] coefficients.
///
/// Hoists the trigonometry and ROI side length out of per-landmark loops;
/// the arithmetic per landmark is bit-identical to the per-call variant.
#[derive(Clone, Copy)]
pub struct RoiLocalToImageTransform {
    roi_center_x: f32,
    roi_center_y: f32,
    cos: f32,
    sin: f32,
    side: f32,
}

impl RoiLocalToImageTransform {
    /// Returns `None` if `frame_w` or `frame_h` is zero.
    #[must_use]
    pub fn new(roi: &FaceRoi, frame_w: u32, frame_h: u32) -> Option<Self> {
        if frame_w == 0 || frame_h == 0 {
            return None;
        }
        let min_dim = (frame_w.min(frame_h) as f32).max(0.0);
        Some(Self {
            roi_center_x: roi.center_x,
            roi_center_y: roi.center_y,
            cos: roi.rotation_rad.cos(),
            sin: roi.rotation_rad.sin(),
            side: roi.scale * min_dim,
        })
    }

    /// Maps ROI-local normalized coordinates to image pixel coordinates.
    #[must_use]
    pub fn apply(&self, x_local: f32, y_local: f32) -> (f32, f32) {
        let lx = (x_local - 0.5) * self.side;
        let ly = (y_local - 0.5) * self.side;
        let dx = lx * self.cos - ly * self.sin;
        let dy = lx * self.sin + ly * self.cos;
        (self.roi_center_x + dx, self.roi_center_y + dy)
    }
}

/// Converts ROI-local normalized coordinates to image pixel coordinates.
///
/// `x_local` and `y_local` are in `[0, 1]` with `(0, 0)` at the top-left of
/// the ROI crop and `(1, 1)` at the bottom-right. The ROI rotation is
/// clockwise as viewed in the unmirrored image.
///
/// Returns `None` if `frame_w` or `frame_h` is zero.
#[must_use]
pub fn roi_local_to_image(
    x_local: f32,
    y_local: f32,
    roi: &FaceRoi,
    frame_w: u32,
    frame_h: u32,
) -> Option<(f32, f32)> {
    let transform = RoiLocalToImageTransform::new(roi, frame_w, frame_h)?;
    Some(transform.apply(x_local, y_local))
}

/// Converts image pixel coordinates back to ROI-local normalized coordinates.
///
/// This is the inverse of [`roi_local_to_image`].
#[must_use]
pub fn image_to_roi_local(
    x_img: f32,
    y_img: f32,
    roi: &FaceRoi,
    frame_w: u32,
    frame_h: u32,
) -> Option<(f32, f32)> {
    if frame_w == 0 || frame_h == 0 {
        return None;
    }

    let dx = x_img - roi.center_x;
    let dy = y_img - roi.center_y;

    let cos = roi.rotation_rad.cos();
    let sin = roi.rotation_rad.sin();

    // Counter-clockwise rotation from image axes to ROI-local axes.
    let lx = dx * cos + dy * sin;
    let ly = -dx * sin + dy * cos;

    let min_dim = (frame_w.min(frame_h) as f32).max(0.0);
    let side = roi.scale * min_dim;

    if side == 0.0 {
        return None;
    }

    Some((lx / side + 0.5, ly / side + 0.5))
}

/// Converts ROI-local normalized coordinates to canonical input pixel
/// coordinates.
#[must_use]
pub fn roi_local_to_canonical(x_local: f32, y_local: f32, canonical_size: u32) -> (f32, f32) {
    let size = canonical_size.max(1) as f32;
    (x_local * size, y_local * size)
}

/// Converts canonical input pixel coordinates back to ROI-local normalized
/// coordinates.
#[must_use]
pub fn canonical_to_roi_local(x_canon: f32, y_canon: f32, canonical_size: u32) -> (f32, f32) {
    let size = canonical_size.max(1) as f32;
    (x_canon / size, y_canon / size)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peppa_contract() -> LandmarkOutputContract {
        LandmarkOutputContract {
            tensor_name: "/Concat_1".into(),
            expected_shape: vec![1, 98, 3],
            expected_dtype: TensorDtype::F32,
            landmark_count: 98,
            coordinate_range: CoordinateRange::ZeroToOne,
            channels: LandmarkChannels::XyConfidence,
            layout: LandmarkLayout::BatchLandmarkChannels,
            canonical_size: 256,
        }
    }

    fn tensor(data: Vec<f32>, shape: Vec<usize>, dtype: TensorDtype) -> OutputTensor<'static> {
        let shape = Box::leak(shape.into_boxed_slice());
        let data = Box::leak(data.into_boxed_slice());
        OutputTensor { shape, data, dtype }
    }

    fn centered_roi() -> FaceRoi {
        FaceRoi {
            center_x: 320.0,
            center_y: 240.0,
            rotation_rad: 0.0,
            scale: 0.5,
            confidence: 1.0,
        }
    }

    #[test]
    fn decode_golden_count() {
        let mut data = vec![0.5_f32; 98 * 3];
        data[2] = 0.9;
        let t = tensor(data, vec![1, 98, 3], TensorDtype::F32);

        let landmarks = decode_landmarks(&t, &peppa_contract(), &centered_roi(), 640, 480).unwrap();

        assert_eq!(landmarks.len(), 98);
        assert!((landmarks[0].x - 0.5).abs() < 1e-6);
        assert!((landmarks[0].y - 0.5).abs() < 1e-6);
        assert!((landmarks[0].visibility - 0.9).abs() < 1e-6);
        assert!(
            landmarks
                .iter()
                .all(|l| l.x.is_finite() && l.visibility.is_finite())
        );
    }

    #[test]
    fn decode_invalid_shape() {
        let t = tensor(vec![0.0_f32; 97 * 3], vec![1, 97, 3], TensorDtype::F32);
        let err = decode_landmarks(&t, &peppa_contract(), &centered_roi(), 640, 480).unwrap_err();
        assert!(
            matches!(err, InferenceError::OutputShapeMismatch { .. }),
            "expected shape mismatch, got {err:?}"
        );
    }

    #[test]
    fn decode_element_count_mismatch() {
        let mut data = vec![0.0_f32; 98 * 3];
        data.push(0.0);
        let t = tensor(data, vec![1, 98, 3], TensorDtype::F32);
        let err = decode_landmarks(&t, &peppa_contract(), &centered_roi(), 640, 480).unwrap_err();
        assert!(
            matches!(err, InferenceError::OutputElementCountMismatch { .. }),
            "expected element count mismatch, got {err:?}"
        );
    }

    #[test]
    fn decode_nan_value() {
        let mut data = vec![0.5_f32; 98 * 3];
        data[10] = f32::NAN;
        let t = tensor(data, vec![1, 98, 3], TensorDtype::F32);
        let err = decode_landmarks(&t, &peppa_contract(), &centered_roi(), 640, 480).unwrap_err();
        assert!(
            matches!(err, InferenceError::InvalidOutputValue { .. }),
            "expected invalid value, got {err:?}"
        );
    }

    #[test]
    fn decode_dtype_mismatch() {
        let t = tensor(vec![0.0_f32; 98 * 3], vec![1, 98, 3], TensorDtype::F16);
        let err = decode_landmarks(&t, &peppa_contract(), &centered_roi(), 640, 480).unwrap_err();
        assert!(
            matches!(err, InferenceError::OutputDtypeMismatch { .. }),
            "expected dtype mismatch, got {err:?}"
        );
    }

    #[test]
    fn decode_minus_one_to_one_range() {
        let contract = LandmarkOutputContract {
            coordinate_range: CoordinateRange::MinusOneToOne,
            ..peppa_contract()
        };
        let mut data = vec![0.0_f32; 98 * 3];
        data[0] = -1.0;
        data[1] = 1.0;
        data[2] = 1.0;
        let t = tensor(data, vec![1, 98, 3], TensorDtype::F32);

        let landmarks = decode_landmarks(&t, &contract, &centered_roi(), 640, 480).unwrap();

        // (-1, 1) maps to (0, 1) in ROI-local normalized space.
        // side = 0.5 * 480 = 240, so top-center of ROI is (200, 360).
        assert!((landmarks[0].x - 200.0 / 640.0).abs() < 1e-6);
        assert!((landmarks[0].y - 360.0 / 480.0).abs() < 1e-6);
    }

    #[test]
    fn decode_pixel_range() {
        let contract = LandmarkOutputContract {
            coordinate_range: CoordinateRange::Pixel,
            ..peppa_contract()
        };
        let mut data = vec![0.0_f32; 98 * 3];
        data[0] = 128.0;
        data[1] = 128.0;
        data[2] = 0.8;
        let t = tensor(data, vec![1, 98, 3], TensorDtype::F32);

        let landmarks = decode_landmarks(&t, &contract, &centered_roi(), 640, 480).unwrap();

        // (128, 128) in a 256x256 canonical input maps to the ROI center.
        assert!((landmarks[0].x - 0.5).abs() < 1e-6);
        assert!((landmarks[0].y - 0.5).abs() < 1e-6);
        assert!((landmarks[0].visibility - 0.8).abs() < 1e-6);
    }

    #[test]
    fn roi_local_image_round_trip() {
        let roi = FaceRoi {
            center_x: 123.0,
            center_y: 456.0,
            rotation_rad: 0.7,
            scale: 0.3,
            confidence: 1.0,
        };
        let local = (0.2_f32, 0.8_f32);

        let (x_img, y_img) = roi_local_to_image(local.0, local.1, &roi, 1280, 720).unwrap();
        let (x_back, y_back) = image_to_roi_local(x_img, y_img, &roi, 1280, 720).unwrap();

        assert!((x_back - local.0).abs() < 1e-5);
        assert!((y_back - local.1).abs() < 1e-5);
    }

    #[test]
    fn canonical_round_trip() {
        let local = (0.35_f32, 0.65_f32);
        let canonical = roi_local_to_canonical(local.0, local.1, 256);
        let back = canonical_to_roi_local(canonical.0, canonical.1, 256);
        assert!((back.0 - local.0).abs() < 1e-6);
        assert!((back.1 - local.1).abs() < 1e-6);
    }

    #[test]
    fn roi_local_image_center() {
        let (x, y) = roi_local_to_image(0.5, 0.5, &centered_roi(), 640, 480).unwrap();
        assert!((x - 320.0).abs() < 1e-6);
        assert!((y - 240.0).abs() < 1e-6);
    }
}
