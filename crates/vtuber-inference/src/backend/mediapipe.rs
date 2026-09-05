//! Worker-owned MediaPipe Face Landmarker VIDEO-mode backend.
//!
//! This module is deliberately independent of Bevy. It verifies and loads the
//! approved task bundle, converts owned camera frames to packed RGB, and
//! decodes one MediaPipe result into the canonical `vtuber-core` contract.

use std::path::Path;
use std::sync::Arc;

use mediapipe::{
    Confidence, Delegate, FaceLandmarker, FaceLandmarkerResult, Image, IouThreshold, ModelSource,
    Size, Timestamp,
};
use sha2::{Digest, Sha256};
use vtuber_core::{
    CameraFaceTransform, FaceBlendshapeSet, FaceLandmark, FaceTrackingOutcome, FaceTrackingQuality,
    FaceTrackingSample, FrameSeq, MEDIAPIPE_FACE_BLENDSHAPE_COUNT, MEDIAPIPE_FACE_LANDMARK_COUNT,
    MonoTimeNs, PixelFormat, VideoFrame,
};

use crate::error::{InferenceError, Result};
use crate::runtime::FaceTrackingInference;

/// The packaged MediaPipe task filename.
pub const TASK_BUNDLE_FILE: &str = "face_landmarker.task";
/// SHA-256 of the approved MediaPipe task bundle.
pub const TASK_BUNDLE_SHA256: &str =
    "64184E229B263107BC2B804C6625DB1341FF2BB731874B0BCC2FE6544E0BC9FF";
const MATRIX_AFFINE_EPSILON: f32 = 0.1;

/// The approved task bundle compiled into this binary.
///
/// Release builds must run without depending on the process working directory
/// or a packaged `assets/models` resource tree, so the same SHA-256-verified
/// bundle is embedded at compile time. The task bundle is distributed under
/// the Apache-2.0 license (`assets/models/LICENSE.mediapipe.txt`).
#[must_use]
pub fn embedded_task_bundle() -> &'static [u8] {
    include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/models/face_landmarker.task"
    ))
}

/// A MediaPipe Face Landmarker runtime owned by one inference worker.
pub struct MediaPipeRuntime {
    landmarker: FaceLandmarker,
    staging: Vec<u8>,
    last_timestamp_ms: Option<i64>,
}

impl MediaPipeRuntime {
    /// Verifies the task bundle and constructs a CPU VIDEO-mode landmarker.
    ///
    /// The returned runtime must be constructed and dropped in the inference
    /// worker. No live MediaPipe object crosses the controller boundary.
    pub fn from_task_path(path: &Path) -> Result<Self> {
        verify_task_bundle(path)?;
        Self::from_verified_source(ModelSource::path(path))
    }

    /// Verifies an in-memory task bundle and constructs a CPU VIDEO-mode
    /// landmarker from the bundle bytes.
    ///
    /// The returned runtime must be constructed and dropped in the inference
    /// worker. No live MediaPipe object crosses the controller boundary.
    pub fn from_task_bytes(bytes: &[u8]) -> Result<Self> {
        verify_task_bundle_bytes(bytes)?;
        Self::from_verified_source(ModelSource::bytes(bytes.to_vec()))
    }

    fn from_verified_source(source: ModelSource) -> Result<Self> {
        let landmarker = FaceLandmarker::builder(source)
            .delegate(Delegate::Cpu)
            .num_faces(std::num::NonZeroU32::new(1).ok_or_else(|| {
                InferenceError::MediaPipeLoadFailed("one-face configuration is invalid".into())
            })?)
            .min_face_detection_confidence(Confidence::HALF)
            .min_face_presence_confidence(Confidence::HALF)
            .min_tracking_confidence(IouThreshold::HALF)
            .output_blendshapes(true)
            .output_transformation_matrixes(true)
            .build_for_video()
            .map_err(|error| InferenceError::MediaPipeLoadFailed(error.to_string()))?;

        Ok(Self {
            landmarker,
            staging: Vec::new(),
            last_timestamp_ms: None,
        })
    }
}

impl FaceTrackingInference for MediaPipeRuntime {
    fn infer_face_tracking(&mut self, frame: &VideoFrame) -> Result<FaceTrackingOutcome> {
        let timestamp_ms = video_timestamp_ms(frame.captured_at, &mut self.last_timestamp_ms)?;
        let image_data = frame_rgb(frame, &mut self.staging)?;
        let image = Image::from_rgb(
            Size {
                width: frame.width,
                height: frame.height,
            },
            image_data,
        )
        .map_err(|error| InferenceError::MediaPipeFrameConversion(error.to_string()))?;
        let inference_started_at = vtuber_core::monotonic_now();
        let result = self
            .landmarker
            .detect_for_video(&image, Timestamp::from_millis(timestamp_ms))
            .map_err(|error| InferenceError::MediaPipeFrameInference(error.to_string()))?;
        let inference_finished_at = vtuber_core::monotonic_now();

        decode_result(
            frame.seq,
            frame.captured_at,
            inference_started_at,
            inference_finished_at,
            result,
        )
    }
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn decode_result(
    source_seq: FrameSeq,
    captured_at: MonoTimeNs,
    inference_started_at: MonoTimeNs,
    inference_finished_at: MonoTimeNs,
    result: FaceLandmarkerResult,
) -> Result<FaceTrackingOutcome> {
    if result.landmarks.is_empty() {
        if !result.blendshapes.is_empty() || !result.transformation_matrixes.is_empty() {
            return Err(contract_error(
                "no-face result contained auxiliary face outputs",
            ));
        }
        return Ok(FaceTrackingOutcome::NoFace {
            source_seq,
            captured_at,
            inference_started_at,
            inference_finished_at,
        });
    }

    if result.landmarks.len() != 1
        || result.blendshapes.len() != 1
        || result.transformation_matrixes.len() != 1
    {
        return Err(contract_error(format!(
            "expected one face, one blendshape set, and one matrix; got faces={}, blendshape_sets={}, matrices={}",
            result.landmarks.len(),
            result.blendshapes.len(),
            result.transformation_matrixes.len()
        )));
    }

    let source_landmarks = &result.landmarks[0];
    if source_landmarks.len() != MEDIAPIPE_FACE_LANDMARK_COUNT {
        return Err(contract_error(format!(
            "expected {MEDIAPIPE_FACE_LANDMARK_COUNT} landmarks, got {}",
            source_landmarks.len()
        )));
    }
    let landmarks: Vec<FaceLandmark> = source_landmarks
        .iter()
        .enumerate()
        .map(|(index, landmark)| {
            let value = FaceLandmark {
                x: landmark.point.x(),
                y: landmark.point.y(),
                z: landmark.point.z(),
                visibility: landmark.visibility.map(|value| value.get()),
                presence: landmark.presence.map(|value| value.get()),
            };
            if value.x.is_finite()
                && value.y.is_finite()
                && value.z.is_finite()
                && value
                    .visibility
                    .is_none_or(|confidence| confidence.is_finite())
                && value
                    .presence
                    .is_none_or(|confidence| confidence.is_finite())
            {
                Ok(value)
            } else {
                Err(contract_error(format!(
                    "landmark {index} contains a non-finite value"
                )))
            }
        })
        .collect::<Result<Vec<_>>>()?;

    let source_blendshapes = &result.blendshapes[0];
    if source_blendshapes.len() != MEDIAPIPE_FACE_BLENDSHAPE_COUNT {
        return Err(contract_error(format!(
            "expected {MEDIAPIPE_FACE_BLENDSHAPE_COUNT} blendshapes, got {}",
            source_blendshapes.len()
        )));
    }
    let pairs: Vec<(&str, f32)> = source_blendshapes
        .iter()
        .map(|category| {
            let name = category.category_name.as_deref().ok_or_else(|| {
                contract_error("blendshape category is missing its official name")
            })?;
            let score = category.score.get();
            if score.is_finite() && (0.0..=1.0).contains(&score) {
                Ok((name, score))
            } else {
                Err(contract_error(format!(
                    "blendshape `{name}` has invalid score {score}"
                )))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let blendshapes =
        FaceBlendshapeSet::from_pairs(&pairs).map_err(|error| contract_error(error.to_string()))?;

    let (camera_to_face, matrix_orthogonality_error, matrix_determinant) =
        matrix_transform(&result)?;
    let face_center = face_center(&landmarks)?;
    let landmark_presence_median = median_presence(&landmarks);
    let sample = FaceTrackingSample::try_new(
        source_seq,
        captured_at,
        inference_started_at,
        inference_finished_at,
        camera_to_face,
        face_center,
        Arc::from(landmarks),
        blendshapes,
        FaceTrackingQuality {
            landmark_presence_median,
            matrix_orthogonality_error,
            matrix_determinant,
        },
    )
    .map_err(|error| contract_error(error.to_string()))?;
    Ok(FaceTrackingOutcome::Face(sample))
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn matrix_transform(result: &FaceLandmarkerResult) -> Result<(CameraFaceTransform, f32, f32)> {
    let matrix = result
        .transformation_matrixes
        .first()
        .ok_or_else(|| contract_error("missing transformation matrix"))?;
    let mut column_major = [0.0; 16];
    for column in 0..4 {
        for row in 0..4 {
            column_major[column * 4 + row] = matrix.get(row, column);
        }
    }
    matrix_from_column_major(column_major)
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn matrix_from_column_major(column_major: [f32; 16]) -> Result<(CameraFaceTransform, f32, f32)> {
    let mut values = [[0.0; 4]; 4];
    for (column, column_values) in column_major.chunks_exact(4).enumerate() {
        for (row, value) in column_values.iter().copied().enumerate() {
            values[row][column] = value;
        }
    }
    if !values.iter().flatten().all(|value| value.is_finite()) {
        return Err(contract_error(
            "transformation matrix contains non-finite data",
        ));
    }
    let affine = values[3][0].abs() <= MATRIX_AFFINE_EPSILON
        && values[3][1].abs() <= MATRIX_AFFINE_EPSILON
        && values[3][2].abs() <= MATRIX_AFFINE_EPSILON
        && (values[3][3] - 1.0).abs() <= MATRIX_AFFINE_EPSILON;
    if !affine {
        return Err(contract_error("transformation matrix is not affine"));
    }

    let determinant = determinant3(values);
    let orthogonality_error = orthogonality_error(values);
    if determinant <= 0.0 {
        return Err(contract_error(format!(
            "transformation matrix determinant must be positive, got {determinant}"
        )));
    }
    let orthonormalized = orthonormalize_rotation(values)?;
    let rotation_xyzw = rotation_to_quaternion(orthonormalized)?;
    let transform = CameraFaceTransform {
        rotation_xyzw,
        translation_xyz: [values[0][3], values[1][3], values[2][3]],
    };
    if !transform.is_valid() {
        return Err(contract_error(
            "transformation rotation is not a unit quaternion",
        ));
    }
    Ok((transform, orthogonality_error, determinant))
}

fn orthonormalize_rotation(values: [[f32; 4]; 4]) -> Result<[[f32; 4]; 4]> {
    let x = normalize([values[0][0], values[1][0], values[2][0]])?;
    let y_raw = [values[0][1], values[1][1], values[2][1]];
    let y = normalize(subtract(y_raw, scale(x, dot(x, y_raw))))?;
    let z = cross(x, y);
    let z_raw = [values[0][2], values[1][2], values[2][2]];
    if dot(z, z_raw) <= f32::EPSILON {
        return Err(contract_error(
            "transformation rotation is degenerate or reflected",
        ));
    }
    Ok([
        [x[0], y[0], z[0], 0.0],
        [x[1], y[1], z[1], 0.0],
        [x[2], y[2], z[2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn dot(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn scale(value: [f32; 3], factor: f32) -> [f32; 3] {
    [value[0] * factor, value[1] * factor, value[2] * factor]
}

fn subtract(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn cross(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn normalize(value: [f32; 3]) -> Result<[f32; 3]> {
    let norm = dot(value, value).sqrt();
    if !norm.is_finite() || norm <= f32::EPSILON {
        return Err(contract_error(
            "transformation rotation contains a zero-length axis",
        ));
    }
    Ok([value[0] / norm, value[1] / norm, value[2] / norm])
}

fn rotation_to_quaternion(values: [[f32; 4]; 4]) -> Result<[f32; 4]> {
    let trace = values[0][0] + values[1][1] + values[2][2];
    let mut quaternion = if trace > 0.0 {
        let scale = (trace + 1.0).sqrt() * 2.0;
        [
            (values[2][1] - values[1][2]) / scale,
            (values[0][2] - values[2][0]) / scale,
            (values[1][0] - values[0][1]) / scale,
            0.25 * scale,
        ]
    } else if values[0][0] > values[1][1] && values[0][0] > values[2][2] {
        let scale = (1.0 + values[0][0] - values[1][1] - values[2][2]).sqrt() * 2.0;
        [
            0.25 * scale,
            (values[0][1] + values[1][0]) / scale,
            (values[0][2] + values[2][0]) / scale,
            (values[2][1] - values[1][2]) / scale,
        ]
    } else if values[1][1] > values[2][2] {
        let scale = (1.0 + values[1][1] - values[0][0] - values[2][2]).sqrt() * 2.0;
        [
            (values[0][1] + values[1][0]) / scale,
            0.25 * scale,
            (values[1][2] + values[2][1]) / scale,
            (values[0][2] - values[2][0]) / scale,
        ]
    } else {
        let scale = (1.0 + values[2][2] - values[0][0] - values[1][1]).sqrt() * 2.0;
        [
            (values[0][2] + values[2][0]) / scale,
            (values[1][2] + values[2][1]) / scale,
            0.25 * scale,
            (values[1][0] - values[0][1]) / scale,
        ]
    };
    let norm = quaternion
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if !norm.is_finite() || norm <= f32::EPSILON {
        return Err(contract_error(
            "transformation rotation cannot be normalized",
        ));
    }
    for value in &mut quaternion {
        *value /= norm;
    }
    Ok(quaternion)
}

fn determinant3(matrix: [[f32; 4]; 4]) -> f32 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn orthogonality_error(matrix: [[f32; 4]; 4]) -> f32 {
    let mut error_squared = 0.0;
    for row in 0..3 {
        for column in 0..3 {
            let dot = (0..3)
                .map(|index| matrix[index][row] * matrix[index][column])
                .sum::<f32>();
            let expected = if row == column { 1.0 } else { 0.0 };
            error_squared += (dot - expected).powi(2);
        }
    }
    error_squared.sqrt()
}

fn face_center(landmarks: &[FaceLandmark]) -> Result<[f32; 2]> {
    let (x, y) = landmarks.iter().fold((0.0, 0.0), |(x, y), landmark| {
        (x + landmark.x, y + landmark.y)
    });
    let count = landmarks.len() as f32;
    let center = [x / count, y / count];
    if center.iter().all(|value| value.is_finite()) {
        Ok(center)
    } else {
        Err(contract_error("landmark centre is not finite"))
    }
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn median_presence(landmarks: &[FaceLandmark]) -> Option<f32> {
    let mut values: Vec<f32> = landmarks
        .iter()
        .filter_map(|landmark| landmark.presence)
        .collect();
    if values.is_empty() {
        return None;
    }
    let median_index = values.len() / 2;
    values.select_nth_unstable_by(median_index, |left, right| {
        left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
    });
    Some(values[median_index])
}

fn video_timestamp_ms(captured_at: MonoTimeNs, last_timestamp_ms: &mut Option<i64>) -> Result<i64> {
    let candidate = i64::try_from(captured_at.0 / 1_000_000)
        .map_err(|_| InferenceError::MediaPipeTimestampOutOfRange)?;
    let timestamp_ms = match *last_timestamp_ms {
        Some(last) => candidate.max(
            last.checked_add(1)
                .ok_or(InferenceError::MediaPipeTimestampOutOfRange)?,
        ),
        None => candidate,
    };
    *last_timestamp_ms = Some(timestamp_ms);
    Ok(timestamp_ms)
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn frame_rgb<'a>(frame: &'a VideoFrame, staging: &'a mut Vec<u8>) -> Result<&'a [u8]> {
    let width = usize::try_from(frame.width)
        .map_err(|_| InferenceError::MediaPipeFrameConversion("frame width is too large".into()))?;
    let height = usize::try_from(frame.height).map_err(|_| {
        InferenceError::MediaPipeFrameConversion("frame height is too large".into())
    })?;
    let channels = match frame.format {
        PixelFormat::Rgb8 | PixelFormat::Bgr8 => 3,
        PixelFormat::Rgba8 => 4,
        PixelFormat::Gray8 => 1,
    };
    let row_bytes = width.checked_mul(channels).ok_or_else(|| {
        InferenceError::MediaPipeFrameConversion("frame row size overflow".into())
    })?;
    if frame.stride_bytes < row_bytes {
        return Err(InferenceError::MediaPipeFrameConversion(format!(
            "frame stride {} is smaller than row size {row_bytes}",
            frame.stride_bytes
        )));
    }
    let required = frame.stride_bytes.checked_mul(height).ok_or_else(|| {
        InferenceError::MediaPipeFrameConversion("frame buffer size overflow".into())
    })?;
    if frame.data.len() < required {
        return Err(InferenceError::MediaPipeFrameConversion(format!(
            "frame buffer has {} bytes but requires {required}",
            frame.data.len()
        )));
    }
    if frame.format == PixelFormat::Rgb8 && frame.stride_bytes == row_bytes {
        return Ok(frame.data.as_ref());
    }

    let rgb_row_bytes = width
        .checked_mul(3)
        .ok_or_else(|| InferenceError::MediaPipeFrameConversion("RGB row size overflow".into()))?;
    let staging_len = rgb_row_bytes.checked_mul(height).ok_or_else(|| {
        InferenceError::MediaPipeFrameConversion("RGB frame size overflow".into())
    })?;
    staging.resize(staging_len, 0);
    for row in 0..height {
        let source = &frame.data[row * frame.stride_bytes..row * frame.stride_bytes + row_bytes];
        let destination = &mut staging[row * rgb_row_bytes..(row + 1) * rgb_row_bytes];
        match frame.format {
            PixelFormat::Rgb8 => destination.copy_from_slice(source),
            PixelFormat::Bgr8 => {
                for (src, dst) in source.chunks_exact(3).zip(destination.chunks_exact_mut(3)) {
                    dst.copy_from_slice(&[src[2], src[1], src[0]]);
                }
            }
            PixelFormat::Rgba8 => {
                for (src, dst) in source.chunks_exact(4).zip(destination.chunks_exact_mut(3)) {
                    dst.copy_from_slice(&src[..3]);
                }
            }
            PixelFormat::Gray8 => {
                for (value, dst) in source.iter().zip(destination.chunks_exact_mut(3)) {
                    dst.fill(*value);
                }
            }
        }
    }
    Ok(staging)
}

fn verify_task_bundle(path: &Path) -> Result<()> {
    let bytes = std::fs::read(path).map_err(|error| {
        InferenceError::MediaPipeLoadFailed(format!("task bundle read failed: {error}"))
    })?;
    verify_task_bundle_bytes(&bytes)
}

fn verify_task_bundle_bytes(bytes: &[u8]) -> Result<()> {
    let actual = Sha256::digest(bytes);
    let actual = actual
        .iter()
        .map(|byte| format!("{byte:02X}"))
        .collect::<String>();
    if actual.eq_ignore_ascii_case(TASK_BUNDLE_SHA256) {
        Ok(())
    } else {
        Err(InferenceError::HashMismatch {
            expected: TASK_BUNDLE_SHA256.into(),
            actual,
        })
    }
}

fn contract_error(message: impl Into<String>) -> InferenceError {
    InferenceError::MediaPipeOutputContract(message.into())
}

#[cfg(test)]
mod tests {
    use super::{
        embedded_task_bundle, frame_rgb, matrix_from_column_major, verify_task_bundle_bytes,
        video_timestamp_ms,
    };
    use std::sync::Arc;
    use vtuber_core::{FrameSeq, MonoTimeNs, PixelFormat, VideoFrame};

    #[test]
    fn embedded_task_bundle_matches_the_pinned_sha256() {
        assert!(verify_task_bundle_bytes(embedded_task_bundle()).is_ok());
    }

    fn frame(
        format: PixelFormat,
        width: u32,
        height: u32,
        stride_bytes: usize,
        data: &[u8],
    ) -> VideoFrame {
        VideoFrame {
            seq: FrameSeq(1),
            captured_at: MonoTimeNs(1_000_000),
            width,
            height,
            stride_bytes,
            format,
            data: Arc::from(data.to_vec()),
        }
    }

    #[test]
    fn video_timestamps_are_strictly_increasing() {
        let mut last = None;
        assert_eq!(
            video_timestamp_ms(MonoTimeNs(10_000_000), &mut last).unwrap(),
            10
        );
        assert_eq!(
            video_timestamp_ms(MonoTimeNs(10_000_000), &mut last).unwrap(),
            11
        );
        assert_eq!(
            video_timestamp_ms(MonoTimeNs(9_000_000), &mut last).unwrap(),
            12
        );
    }

    #[test]
    fn bgr_and_stride_are_converted_to_packed_rgb() {
        let source = frame(PixelFormat::Bgr8, 2, 1, 8, &[3, 2, 1, 6, 5, 4, 0, 0]);
        let mut staging = Vec::new();
        assert_eq!(
            frame_rgb(&source, &mut staging).unwrap(),
            &[1, 2, 3, 4, 5, 6]
        );
    }

    #[test]
    fn undersized_stride_is_rejected() {
        let source = frame(PixelFormat::Rgb8, 2, 1, 5, &[0; 5]);
        let mut staging = Vec::new();
        assert!(frame_rgb(&source, &mut staging).is_err());
    }

    #[test]
    fn column_major_identity_and_translation_are_decoded_without_transpose() {
        let mut matrix = [0.0; 16];
        matrix[0] = 1.0;
        matrix[5] = 1.0;
        matrix[10] = 1.0;
        matrix[15] = 1.0;
        matrix[12] = 1.0;
        matrix[13] = 2.0;
        matrix[14] = 3.0;
        let (transform, error, determinant) =
            matrix_from_column_major(matrix).expect("affine identity should decode");
        assert_eq!(transform.rotation_xyzw, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(transform.translation_xyz, [1.0, 2.0, 3.0]);
        assert_eq!(error, 0.0);
        assert_eq!(determinant, 1.0);
    }

    #[test]
    fn known_yaw_matrix_extracts_a_proper_rotation() {
        let mut matrix = [0.0; 16];
        matrix[5] = 1.0;
        matrix[2] = -1.0;
        matrix[8] = 1.0;
        matrix[15] = 1.0;
        let (transform, _, determinant) =
            matrix_from_column_major(matrix).expect("known rotation should decode");
        assert!((transform.rotation_xyzw[1] - 2.0_f32.sqrt() / 2.0).abs() < 1.0e-5);
        assert!((transform.rotation_xyzw[3] - 2.0_f32.sqrt() / 2.0).abs() < 1.0e-5);
        assert!((determinant - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn scaled_rotation_is_orthonormalized_before_quaternion_extraction() {
        let mut matrix = [0.0; 16];
        matrix[0] = 2.0;
        matrix[5] = 2.0;
        matrix[10] = 2.0;
        matrix[15] = 1.0;
        let (transform, error, determinant) =
            matrix_from_column_major(matrix).expect("positive scaled rotation should decode");
        assert_eq!(transform.rotation_xyzw, [0.0, 0.0, 0.0, 1.0]);
        assert!(error > 0.0);
        assert!((determinant - 8.0).abs() < 1.0e-5);
    }

    #[test]
    fn reflection_matrix_is_rejected() {
        let mut matrix = [0.0; 16];
        matrix[0] = -1.0;
        matrix[5] = 1.0;
        matrix[10] = 1.0;
        matrix[15] = 1.0;
        assert!(matrix_from_column_major(matrix).is_err());
    }
}
