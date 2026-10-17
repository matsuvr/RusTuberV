//! HaGRID's crop classifier, owned by the Pose worker and executed by tract.
//! Image transforms are pure Rust; the model needs no ONNX Runtime library.

use std::sync::Arc;

use tract_onnx::prelude::*;
use vtuber_core::arm_tracking::{HAND_LANDMARK_COUNT, HandGesture};
use vtuber_core::frame::FramePixels;

use crate::error::{InferenceError, Result};

const MODEL: &[u8] = include_bytes!("../../../assets/models/hagrid_crops_classifier.onnx");
const SIDE: usize = 128;
/// Landmarks lie inside the skin outline; include a margin around their bounds.
const CROP_SCALE: f32 = 1.2;
/// Output order from dynamic_gestures/utils/enums.py at revision 557a836.
const GESTURES: [HandGesture; 45] = [
    HandGesture::HandDown,
    HandGesture::HandRight,
    HandGesture::HandLeft,
    HandGesture::ThumbIndex,
    HandGesture::ThumbLeft,
    HandGesture::ThumbRight,
    HandGesture::ThumbDown,
    HandGesture::HalfUp,
    HandGesture::HalfLeft,
    HandGesture::HalfRight,
    HandGesture::HalfDown,
    HandGesture::PartHandHeart,
    HandGesture::PartHandHeart2,
    HandGesture::FistInverted,
    HandGesture::TwoLeft,
    HandGesture::TwoRight,
    HandGesture::TwoDown,
    HandGesture::Grabbing,
    HandGesture::Grip,
    HandGesture::Point,
    HandGesture::Call,
    HandGesture::Three3,
    HandGesture::LittleFinger,
    HandGesture::MiddleFinger,
    HandGesture::Dislike,
    HandGesture::Fist,
    HandGesture::Four,
    HandGesture::Like,
    HandGesture::Mute,
    HandGesture::Ok,
    HandGesture::One,
    HandGesture::Palm,
    HandGesture::Peace,
    HandGesture::PeaceInverted,
    HandGesture::Rock,
    HandGesture::Stop,
    HandGesture::StopInverted,
    HandGesture::Three,
    HandGesture::Three2,
    HandGesture::TwoUp,
    HandGesture::TwoUpInverted,
    HandGesture::ThreeGun,
    HandGesture::OneLeft,
    HandGesture::OneRight,
    HandGesture::OneDown,
];

pub(crate) struct HandGestureClassifier {
    model: Arc<TypedRunnableModel>,
}

impl HandGestureClassifier {
    pub(crate) fn new() -> Result<Self> {
        let mut bytes = MODEL;
        let model = tract_onnx::onnx()
            .model_for_read(&mut bytes)
            .map_err(|error| InferenceError::LoadFailed(format!("HaGRID: {error:?}")))?
            .with_input_fact(0, f32::fact([1, 3, SIDE, SIDE]).into())
            .map_err(|error| InferenceError::LoadFailed(format!("HaGRID input: {error:?}")))?
            .into_optimized()
            .map_err(|error| InferenceError::OptimizationFailed(format!("HaGRID: {error:?}")))?
            .into_runnable()
            .map_err(|error| InferenceError::OptimizationFailed(format!("HaGRID: {error:?}")))?;
        Ok(Self { model })
    }

    pub(crate) fn classify(
        &self,
        pixels: FramePixels<'_>,
        landmarks: &[[f32; 2]],
    ) -> Result<HandGesture> {
        let crop = hand_crop(pixels, landmarks)?;
        closest_gesture(self.logits(hand_tensor(pixels, crop)?)?)
    }

    fn logits(&self, input: Vec<f32>) -> Result<[f32; 45]> {
        let input = tract_ndarray::Array4::from_shape_vec((1, 3, SIDE, SIDE), input)
            .map_err(|error| InferenceError::InvalidInput(format!("HaGRID tensor: {error}")))?;
        let outputs = self
            .model
            .run(tvec!(input.into_tensor().into_tvalue()))
            .map_err(|error| InferenceError::ExecutionFailed(format!("HaGRID: {error:?}")))?;
        let output = outputs
            .first()
            .ok_or_else(|| InferenceError::ExecutionFailed("HaGRID returned no output".into()))?
            .to_plain_array_view::<f32>()
            .map_err(|error| {
                InferenceError::ExecutionFailed(format!("HaGRID output: {error:?}"))
            })?;
        if output.shape() != [1, GESTURES.len()] {
            return Err(InferenceError::OutputShapeMismatch {
                expected: vec![1, GESTURES.len()],
                actual: output.shape().to_vec(),
            });
        }
        let mut logits = [0.0; 45];
        for (destination, value) in logits.iter_mut().zip(output.iter()) {
            *destination = *value;
        }
        Ok(logits)
    }
}

/// Classifier logits are not probabilities. Match upstream's first-maximum
/// argmax without a threshold or an invented "no gesture" class.
fn closest_gesture(logits: [f32; 45]) -> Result<HandGesture> {
    let mut best = (HandGesture::HandDown, f32::NEG_INFINITY);
    for (index, (gesture, value)) in GESTURES.into_iter().zip(logits).enumerate() {
        if !value.is_finite() {
            return Err(InferenceError::InvalidOutputValue { index, value });
        }
        if value > best.1 {
            best = (gesture, value);
        }
    }
    Ok(best.0)
}

/// Pixel-edge coordinates: left/top inclusive, right/bottom exclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HandCrop {
    left: usize,
    top: usize,
    right: usize,
    bottom: usize,
}

/// Axis-aligned square from the hand's normalized image landmarks. Clipping
/// at the camera border can make it rectangular, as in the upstream crop path.
/// No rotation or reflection: orientation is part of the classifier's labels.
fn hand_crop(pixels: FramePixels<'_>, landmarks: &[[f32; 2]]) -> Result<HandCrop> {
    if landmarks.len() != HAND_LANDMARK_COUNT
        || landmarks.iter().flatten().any(|value| !value.is_finite())
    {
        return Err(InferenceError::InvalidInput(
            "HaGRID crop needs 21 finite image landmarks".into(),
        ));
    }
    let width = pixels.width() as f32;
    let height = pixels.height() as f32;
    let [left, top, right, bottom] = landmarks.iter().fold(
        [
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ],
        |[left, top, right, bottom], &[x, y]| {
            let (x, y) = (x * width, y * height);
            [left.min(x), top.min(y), right.max(x), bottom.max(y)]
        },
    );
    let half = (right - left).max(bottom - top) * CROP_SCALE * 0.5;
    let center_x = (left + right) * 0.5;
    let center_y = (top + bottom) * 0.5;
    let crop = HandCrop {
        left: (center_x - half).floor().clamp(0.0, width) as usize,
        top: (center_y - half).floor().clamp(0.0, height) as usize,
        right: (center_x + half).ceil().clamp(0.0, width) as usize,
        bottom: (center_y + half).ceil().clamp(0.0, height) as usize,
    };
    if crop.left >= crop.right || crop.top >= crop.bottom {
        return Err(InferenceError::InvalidInput(
            "HaGRID hand crop is empty".into(),
        ));
    }
    Ok(crop)
}

/// Bilinear half-pixel resize to RGB NCHW, then (pixel - 127) / 128.
/// Interpolated bytes are rounded before normalization like upstream's uint8
/// resize. OpenCV's fixed-point interpolation may differ by one byte value.
#[expect(
    clippy::indexing_slicing,
    reason = "the three fixed-size output planes only read RGB channels 0..3 from four-element RGBA pixels"
)]
fn hand_tensor(pixels: FramePixels<'_>, crop: HandCrop) -> Result<Vec<f32>> {
    let width = crop.right - crop.left;
    let height = crop.bottom - crop.top;
    let sample = |destination: usize, extent: usize| {
        let position = ((destination as f32 + 0.5) * extent as f32 / SIDE as f32 - 0.5)
            .clamp(0.0, (extent - 1) as f32);
        let lower = position.floor() as usize;
        (lower, (lower + 1).min(extent - 1), position - lower as f32)
    };
    let mut tensor = vec![0.0; 3 * SIDE * SIDE];
    for (channel, plane) in tensor
        .as_chunks_mut::<{ SIDE * SIDE }>()
        .0
        .iter_mut()
        .enumerate()
    {
        for (y, row) in plane.as_chunks_mut::<SIDE>().0.iter_mut().enumerate() {
            let (y0, y1, fy) = sample(y, height);
            for (x, value) in row.iter_mut().enumerate() {
                let (x0, x1, fx) = sample(x, width);
                let a = pixels.rgba(crop.left + x0, crop.top + y0)?[channel] as f32;
                let b = pixels.rgba(crop.left + x1, crop.top + y0)?[channel] as f32;
                let c = pixels.rgba(crop.left + x0, crop.top + y1)?[channel] as f32;
                let d = pixels.rgba(crop.left + x1, crop.top + y1)?[channel] as f32;
                let top = a + (b - a) * fx;
                let bottom = c + (d - c) * fx;
                *value = ((top + (bottom - top) * fy).round() - 127.0) / 128.0;
            }
        }
    }
    Ok(tensor)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]
    use super::*;
    use vtuber_core::{FrameSeq, MonoTimeNs, PixelFormat, VideoFrame};

    fn frame(
        width: u32,
        height: u32,
        format: PixelFormat,
        stride: usize,
        data: Vec<u8>,
    ) -> VideoFrame {
        VideoFrame {
            seq: FrameSeq(1),
            captured_at: MonoTimeNs(1),
            width,
            height,
            stride_bytes: stride,
            format,
            data: Arc::from(data),
        }
    }

    #[test]
    fn crop_uses_image_bounds_and_preserves_orientation() {
        let image = frame(200, 100, PixelFormat::Gray8, 200, vec![0; 20_000]);
        let mut points = [[0.5, 0.5]; HAND_LANDMARK_COUNT];
        points[0] = [0.4, 0.4];
        points[1] = [0.6, 0.6];
        let crop = hand_crop(image.pixels().unwrap(), &points).unwrap();
        // 40px-wide bounds plus a 20% margin, centered on (100, 50).
        assert!(crop.left.abs_diff(76) <= 1 && crop.right.abs_diff(124) <= 1);
        assert!(crop.top.abs_diff(26) <= 1 && crop.bottom.abs_diff(74) <= 1);
        for point in &mut points {
            point[0] -= 0.5;
        }
        let crop = hand_crop(image.pixels().unwrap(), &points).unwrap();
        assert_eq!(crop.left, 0);
        assert!(crop.right.abs_diff(24) <= 1);
        assert!(crop.top.abs_diff(26) <= 1);
    }

    #[test]
    fn tensor_matches_rgb_planes_normalization_and_padded_bgr_rows() {
        // Red/green on top, blue/white below. Final-row padding is optional.
        let rgb = frame(
            2,
            2,
            PixelFormat::Rgb8,
            6,
            vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255],
        );
        let bgr = frame(
            2,
            2,
            PixelFormat::Bgr8,
            8,
            vec![0, 0, 255, 0, 255, 0, 77, 88, 255, 0, 0, 255, 255, 255],
        );
        let crop = HandCrop {
            left: 0,
            top: 0,
            right: 2,
            bottom: 2,
        };
        let tensor = hand_tensor(rgb.pixels().unwrap(), crop).unwrap();
        assert_eq!(tensor, hand_tensor(bgr.pixels().unwrap(), crop).unwrap());
        let at = |channel, x, y| tensor[channel * SIDE * SIDE + y * SIDE + x];
        assert_eq!(
            [at(0, 0, 0), at(1, 0, 0), at(2, 0, 0)],
            [1.0, -127.0 / 128.0, -127.0 / 128.0]
        );
        assert_eq!(
            [at(0, 127, 127), at(1, 127, 127), at(2, 127, 127)],
            [1.0; 3]
        );
        // Half-pixel bilinear interpolation at destination (63, 63).
        assert_eq!(
            [at(0, 63, 63), at(1, 63, 63), at(2, 63, 63)],
            [1.0 / 128.0, -1.0 / 128.0, -1.0 / 128.0]
        );
    }

    #[test]
    fn logits_use_upstream_label_order_and_first_maximum() {
        for (index, expected) in [
            (25, HandGesture::Fist),
            (32, HandGesture::Peace),
            (31, HandGesture::Palm),
            (29, HandGesture::Ok),
            (27, HandGesture::Like),
            (44, HandGesture::OneDown),
        ] {
            let mut logits = [-3.0; 45];
            logits[index] = -1.0; // No probability threshold, even for negative logits.
            assert_eq!(closest_gesture(logits).unwrap(), expected);
        }
        assert_eq!(closest_gesture([0.0; 45]).unwrap(), HandGesture::HandDown);
    }

    #[test]
    fn embedded_model_executes_with_the_pinned_output_contract() {
        let classifier = HandGestureClassifier::new().unwrap();
        let logits = classifier.logits(vec![0.0; 3 * SIDE * SIDE]).unwrap();
        assert!(logits.iter().all(|value| value.is_finite()));
        // Reference logits for a neutral 127 RGB crop are recorded below.
        // This exercises the bundled graph rather than a mocked classifier.
        for (index, expected) in [
            (25, 0.329705_f32),
            (32, -2.396947_f32),
            (31, -5.030273_f32),
            (29, -3.74349_f32),
            (27, -2.919226_f32),
        ] {
            assert!(
                (logits[index] - expected).abs() < 0.003,
                "logit {index}: {}",
                logits[index]
            );
        }
        assert_eq!(closest_gesture(logits).unwrap(), HandGesture::Fist);
    }
}
