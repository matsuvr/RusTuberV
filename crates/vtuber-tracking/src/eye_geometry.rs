//! Fixed MediaPipe lid geometry for per-eye closure features (Issue #65).
//!
//! For each eye the corner-to-corner line is used as the local axis. The
//! perpendicular distance of three upper/lower lid pairs is divided by the
//! eye width; the maximum of the three is [`max_lid_gap_ratio`]. This is a
//! dimensionless image-plane ratio, not a physical millimetre distance or a
//! closure probability, and it is invariant to translation, uniform scale,
//! and in-plane rotation only.
//!
//! The eight fixed indices per eye follow Google's official LEFT_EYE and
//! RIGHT_EYE landmark connections. The anatomical left/right naming is the
//! performer's and is not affected by the display mirror.

use vtuber_core::{FaceLandmark, FaceTrackingSample, MediaPipeBlendshape};

use crate::eye_closure::EyeSide;

const LEFT_CORNERS: [usize; 2] = [362, 263];
const LEFT_UPPER: [usize; 3] = [385, 386, 387];
const LEFT_LOWER: [usize; 3] = [380, 374, 373];
const RIGHT_CORNERS: [usize; 2] = [33, 133];
const RIGHT_UPPER: [usize; 3] = [160, 159, 158];
const RIGHT_LOWER: [usize; 3] = [144, 145, 153];

/// Fixed eye-lid points in inference-image pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LidPoints {
    /// Inner and outer eye corner `[corner_0, corner_1]`.
    pub corners: [[f32; 2]; 2],
    /// Upper-lid pair points.
    pub upper: [[f32; 2]; 3],
    /// Lower-lid pair points matching [`Self::upper`] one to one.
    pub lower: [[f32; 2]; 3],
}

impl LidPoints {
    /// Packs the eight points in the fixed order used by extracted data.
    #[must_use]
    pub fn to_array(self) -> [[f32; 2]; 8] {
        let [corner_0, corner_1] = self.corners;
        let [upper_0, upper_1, upper_2] = self.upper;
        let [lower_0, lower_1, lower_2] = self.lower;
        [
            corner_0, corner_1, upper_0, upper_1, upper_2, lower_0, lower_1, lower_2,
        ]
    }

    /// Unpacks the fixed order used by [`Self::to_array`].
    #[must_use]
    pub fn from_array(points: [[f32; 2]; 8]) -> Self {
        let [
            corner_0,
            corner_1,
            upper_0,
            upper_1,
            upper_2,
            lower_0,
            lower_1,
            lower_2,
        ] = points;
        Self {
            corners: [corner_0, corner_1],
            upper: [upper_0, upper_1, upper_2],
            lower: [lower_0, lower_1, lower_2],
        }
    }
}

/// Lid-geometry input contract failures.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
pub enum EyeGeometryError {
    /// A fixed landmark index is absent from the provided slice.
    #[error("face landmark index {index} is missing; got {len} landmarks")]
    MissingLandmark {
        /// Required fixed index.
        index: usize,
        /// Length of the provided landmark slice.
        len: usize,
    },
}

/// Extracts one eye's eight fixed lid points.
///
/// `image_size` is the width and height of the inference image whose
/// normalized `x`/`y` the landmarks carry. Only the sixteen required
/// landmarks are read; no heap allocation is performed.
///
/// # Errors
///
/// Returns [`EyeGeometryError::MissingLandmark`] when a fixed index is absent
/// from `landmarks`.
pub fn mediapipe_lid_points(
    landmarks: &[FaceLandmark],
    image_size: [u32; 2],
    eye: EyeSide,
) -> Result<LidPoints, EyeGeometryError> {
    let [width, height] = image_size;
    let (corners, upper, lower) = match eye {
        EyeSide::Left => (LEFT_CORNERS, LEFT_UPPER, LEFT_LOWER),
        EyeSide::Right => (RIGHT_CORNERS, RIGHT_UPPER, RIGHT_LOWER),
    };
    let pixel = |index: usize| {
        let landmark = landmarks
            .get(index)
            .ok_or(EyeGeometryError::MissingLandmark {
                index,
                len: landmarks.len(),
            })?;
        Ok::<[f32; 2], EyeGeometryError>([landmark.x * width as f32, landmark.y * height as f32])
    };
    let [corner_0, corner_1] = corners;
    let [upper_0, upper_1, upper_2] = upper;
    let [lower_0, lower_1, lower_2] = lower;
    Ok(LidPoints {
        corners: [pixel(corner_0)?, pixel(corner_1)?],
        upper: [pixel(upper_0)?, pixel(upper_1)?, pixel(upper_2)?],
        lower: [pixel(lower_0)?, pixel(lower_1)?, pixel(lower_2)?],
    })
}

/// Maximum of the three perpendicular lid gaps divided by the eye width.
///
/// Returns `None` when the two corners coincide, so the ratio is undefined;
/// it never fabricates `0.0` (= fully closed) for undefined geometry.
#[must_use]
pub fn max_lid_gap_ratio(points: &LidPoints) -> Option<f32> {
    let [corner_0, corner_1] = points.corners;
    let [x0, y0] = corner_0;
    let [x1, y1] = corner_1;
    let dx = x1 - x0;
    let dy = y1 - y0;
    let denominator = dx * dx + dy * dy;
    if denominator == 0.0 {
        return None;
    }
    let mut maximum = 0.0_f32;
    for (upper, lower) in points.upper.iter().zip(points.lower.iter()) {
        let [ux, uy] = *upper;
        let [lx, ly] = *lower;
        let gap = ((dx * (uy - ly) - dy * (ux - lx)).abs()) / denominator;
        if gap > maximum {
            maximum = gap;
        }
    }
    maximum.is_finite().then_some(maximum)
}

/// Per-eye closure features consumed by the geometry judgement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeClosureFeatures {
    /// Maximum lid-gap ratio from [`max_lid_gap_ratio`].
    pub lid_gap_ratio: f32,
    /// Raw MediaPipe `EyeBlinkLeft/Right` score of the same eye.
    pub raw_blink: f32,
}

/// Builds one eye's closure features from a canonical face sample.
///
/// # Errors
///
/// Returns [`EyeGeometryError::MissingLandmark`] when the sample does not
/// carry the fixed indices. Returns `Ok(None)` when the lid ratio is
/// undefined for this sample (coincident corners).
pub fn eye_closure_features(
    sample: &FaceTrackingSample,
    eye: EyeSide,
) -> Result<Option<EyeClosureFeatures>, EyeGeometryError> {
    let points = mediapipe_lid_points(&sample.landmarks, sample.image_size, eye)?;
    let Some(lid_gap_ratio) = max_lid_gap_ratio(&points) else {
        return Ok(None);
    };
    let channel = match eye {
        EyeSide::Left => MediaPipeBlendshape::EyeBlinkLeft,
        EyeSide::Right => MediaPipeBlendshape::EyeBlinkRight,
    };
    Ok(Some(EyeClosureFeatures {
        lid_gap_ratio,
        raw_blink: sample.blendshapes.get(channel),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use vtuber_core::{
        CameraFaceTransform, FaceBlendshapeSet, FaceTrackingQuality, FrameSeq,
        MEDIAPIPE_FACE_LANDMARK_COUNT, MediaPipeBlendshape, MonoTimeNs,
    };

    /// One 8-point eye at unit scale: the corners span 1.0 along x and the
    /// lid pairs are separated by the given gap ratio.
    fn unit_eye(gap: f32) -> [[f32; 2]; 8] {
        let half = gap / 2.0;
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.25, -half],
            [0.5, -half],
            [0.75, -half],
            [0.25, half],
            [0.5, half],
            [0.75, half],
        ]
    }

    fn landmarks_from(
        pixels: [[f32; 2]; 8],
        image_size: [u32; 2],
        eye: EyeSide,
    ) -> Vec<FaceLandmark> {
        let [width, height] = image_size;
        let mut landmarks = vec![FaceLandmark::default(); MEDIAPIPE_FACE_LANDMARK_COUNT];
        let indices = match eye {
            EyeSide::Left => [
                LEFT_CORNERS[0],
                LEFT_CORNERS[1],
                LEFT_UPPER[0],
                LEFT_UPPER[1],
                LEFT_UPPER[2],
                LEFT_LOWER[0],
                LEFT_LOWER[1],
                LEFT_LOWER[2],
            ],
            EyeSide::Right => [
                RIGHT_CORNERS[0],
                RIGHT_CORNERS[1],
                RIGHT_UPPER[0],
                RIGHT_UPPER[1],
                RIGHT_UPPER[2],
                RIGHT_LOWER[0],
                RIGHT_LOWER[1],
                RIGHT_LOWER[2],
            ],
        };
        for (pixel, index) in pixels.iter().zip(indices.iter()) {
            let landmark = &mut landmarks[*index];
            landmark.x = pixel[0] / width as f32;
            landmark.y = pixel[1] / height as f32;
        }
        landmarks
    }

    fn transform(
        points: [[f32; 2]; 8],
        translate: [f32; 2],
        rotate: f32,
        scale: f32,
    ) -> [[f32; 2]; 8] {
        let (sin, cos) = rotate.sin_cos();
        points.map(|[x, y]| {
            let sx = x * scale;
            let sy = y * scale;
            [
                sx * cos - sy * sin + translate[0],
                sx * sin + sy * cos + translate[1],
            ]
        })
    }

    fn ratio(pixels: [[f32; 2]; 8], image_size: [u32; 2], eye: EyeSide) -> Option<f32> {
        let landmarks = landmarks_from(pixels, image_size, eye);
        let points = mediapipe_lid_points(&landmarks, image_size, eye).unwrap();
        max_lid_gap_ratio(&points)
    }

    #[test]
    fn the_same_pixel_layout_gives_the_same_ratio_at_two_resolutions() {
        let pixels = transform(unit_eye(0.2), [0.0, 0.0], 0.0, 1.0);
        let small = ratio(pixels, [480, 640], EyeSide::Left).unwrap();
        let large = ratio(pixels, [1280, 720], EyeSide::Left).unwrap();
        assert!((small - large).abs() < 1.0e-6, "{small} vs {large}");
    }

    #[test]
    fn translation_scale_and_in_plane_rotation_preserve_the_ratio() {
        let pixels = unit_eye(0.35);
        let base = ratio(pixels, [640, 480], EyeSide::Right).unwrap();
        let moved = ratio(
            transform(pixels, [120.0, -30.0], 0.7, 3.5),
            [640, 480],
            EyeSide::Right,
        )
        .unwrap();
        assert!((base - moved).abs() < 1.0e-4, "{base} vs {moved}");
    }

    #[test]
    fn coincident_lids_give_zero_and_one_open_pair_drives_the_maximum() {
        let closed = unit_eye(0.0);
        assert_eq!(ratio(closed, [640, 480], EyeSide::Left), Some(0.0));
        let mut one_open = unit_eye(0.0);
        one_open[3] = [0.5, -0.4];
        one_open[6] = [0.5, 0.0];
        let value = ratio(one_open, [640, 480], EyeSide::Left).unwrap();
        assert!((value - 0.4).abs() < 1.0e-6, "{value}");
    }

    #[test]
    fn coincident_corners_are_undefined_not_closed() {
        let mut collapsed = unit_eye(0.2);
        collapsed[1] = collapsed[0];
        assert_eq!(ratio(collapsed, [640, 480], EyeSide::Left), None);
    }

    #[test]
    fn an_asymmetric_fixture_keeps_left_and_right_apart() {
        let open = transform(unit_eye(0.5), [0.0, 0.0], 0.0, 1.0);
        let closed = transform(unit_eye(0.0), [0.0, 0.0], 0.0, 1.0);
        let mut both = vec![[0.0; 2]; MEDIAPIPE_FACE_LANDMARK_COUNT];
        let left_indices = [
            LEFT_CORNERS[0],
            LEFT_CORNERS[1],
            LEFT_UPPER[0],
            LEFT_UPPER[1],
            LEFT_UPPER[2],
            LEFT_LOWER[0],
            LEFT_LOWER[1],
            LEFT_LOWER[2],
        ];
        let right_indices = [
            RIGHT_CORNERS[0],
            RIGHT_CORNERS[1],
            RIGHT_UPPER[0],
            RIGHT_UPPER[1],
            RIGHT_UPPER[2],
            RIGHT_LOWER[0],
            RIGHT_LOWER[1],
            RIGHT_LOWER[2],
        ];
        for (pixel, index) in closed.iter().zip(left_indices.iter()) {
            both[*index] = *pixel;
        }
        for (pixel, index) in open.iter().zip(right_indices.iter()) {
            both[*index] = *pixel;
        }
        let landmarks: Vec<FaceLandmark> = both
            .into_iter()
            .map(|[x, y]| FaceLandmark {
                x: x / 640.0,
                y: y / 480.0,
                ..FaceLandmark::default()
            })
            .collect();
        let left = max_lid_gap_ratio(
            &mediapipe_lid_points(&landmarks, [640, 480], EyeSide::Left).unwrap(),
        )
        .unwrap();
        let right = max_lid_gap_ratio(
            &mediapipe_lid_points(&landmarks, [640, 480], EyeSide::Right).unwrap(),
        )
        .unwrap();
        assert!(left.abs() < 1.0e-6, "left fixture is closed: {left}");
        assert!(
            (right - 0.5).abs() < 1.0e-6,
            "right fixture is open: {right}"
        );
    }

    #[test]
    fn missing_fixed_indices_are_an_input_error() {
        let landmarks = vec![FaceLandmark::default(); 100];
        assert_eq!(
            mediapipe_lid_points(&landmarks, [640, 480], EyeSide::Left),
            Err(EyeGeometryError::MissingLandmark {
                index: 362,
                len: 100
            })
        );
    }

    #[test]
    fn the_sample_adapter_matches_the_direct_computation() {
        let image_size = [640, 480];
        let left_landmarks = landmarks_from(unit_eye(0.3), image_size, EyeSide::Left);
        let mut blendshapes = vec![0.0_f32; 52];
        let blink_index = MediaPipeBlendshape::EyeBlinkLeft.index();
        blendshapes[blink_index] = 0.7;
        let pairs = MediaPipeBlendshape::ALL
            .iter()
            .zip(blendshapes.iter())
            .map(|(channel, value)| (channel.as_str(), *value))
            .collect::<Vec<_>>();
        let sample = FaceTrackingSample::try_new(
            FrameSeq(0),
            MonoTimeNs(0),
            MonoTimeNs(1),
            MonoTimeNs(2),
            CameraFaceTransform::identity(),
            [0.5, 0.5],
            image_size,
            Arc::from(left_landmarks),
            FaceBlendshapeSet::from_pairs(&pairs).unwrap(),
            FaceTrackingQuality {
                landmark_presence_median: Some(1.0),
                matrix_orthogonality_error: 0.0,
                matrix_determinant: 1.0,
            },
        )
        .unwrap();
        let features = eye_closure_features(&sample, EyeSide::Left)
            .unwrap()
            .unwrap();
        assert!((features.lid_gap_ratio - 0.3).abs() < 1.0e-6);
        assert!((features.raw_blink - 0.7).abs() < 1.0e-6);
        assert!(
            eye_closure_features(&sample, EyeSide::Right)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn lid_points_round_trip_through_the_stored_array() {
        let points = LidPoints {
            corners: [[1.0, 2.0], [3.0, 4.0]],
            upper: [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
            lower: [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],
        };
        assert_eq!(LidPoints::from_array(points.to_array()), points);
    }
}
