//! Head pose estimation from neutral-relative landmark sets.
//!
//! Uses weighted Kabsch to solve for rotation between a calibrated neutral
//! point cloud and a current observation. Coordinate conventions follow
//! `DESIGN.md` section 11.6.

use nalgebra::{Dyn, OMatrix, OVector, Rotation3, SVD, U3, UnitQuaternion};
use thiserror::Error;

use vtuber_core::types::HeadPose;

/// MediaPipe face-transform pose adapter.
pub mod mediapipe;
/// Image-space landmark pose adapter.
pub mod planar;

/// Minimum number of points required by the Kabsch solver.
pub const MIN_LANDMARK_POINTS: usize = 3;

/// Degeneracy threshold for singular values of the covariance matrix.
const MIN_SINGULAR_VALUE: f32 = 1e-6;

/// A single weighted point in the canonical coordinate basis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightedPoint {
    /// Point position in the canonical basis.
    pub position: [f32; 3],
    /// Weight used during centering and covariance accumulation.
    pub weight: f32,
}

/// Errors that can occur while solving for head pose.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum PoseError {
    /// Too few points were supplied.
    #[error("insufficient points: need at least {MIN_LANDMARK_POINTS}, got {0}")]
    InsufficientPoints(usize),
    /// All points are collinear or otherwise degenerate.
    #[error("point cloud is degenerate: covariance singular value below threshold")]
    DegeneratePointCloud,
    /// The solved transform is a reflection rather than a rotation.
    #[error("solved transform is a reflection, not a rotation")]
    ReflectionDetected,
    /// Total weight is zero or negative.
    #[error("total weight must be positive, got {0}")]
    ZeroWeight(f32),
}

/// Landmark set used for pose solving, paired with a neutral reference.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LandmarkSet {
    /// Weighted canonical points.
    pub points: Vec<WeightedPoint>,
}

/// Result of aligning current landmarks to a neutral reference.
#[derive(Clone, Debug, PartialEq)]
pub struct PoseAlignment {
    /// Rotation from neutral to current in the canonical basis.
    pub rotation: UnitQuaternion<f32>,
    /// Relative head pose with DESIGN.md semantic sign conventions.
    pub pose: HeadPose,
    /// Rotation-compensated residual translation of the current cloud
    /// relative to neutral, in canonical basis units. Defined with the
    /// pivot at the weighted neutral centroid, so a pure head rotation
    /// about the face centre yields zero first-order cross-talk:
    /// `d = c_current - c_neutral`.
    pub translation: [f32; 3],
    /// Weighted RMS projected (x, y) radius of the neutral cloud about its
    /// centroid. Scale evidence for depth and the X/Y unit conversion.
    pub neutral_projected_radius: f32,
    /// Weighted RMS projected (x, y) radius of the current cloud.
    pub current_projected_radius: f32,
}

impl LandmarkSet {
    /// Creates an empty set.
    #[must_use]
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Adds a weighted point.
    pub fn push(&mut self, position: [f32; 3], weight: f32) {
        self.points.push(WeightedPoint { position, weight });
    }

    /// Returns the number of points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns `true` if there are no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Weighted centroid of the point cloud.
    #[must_use]
    pub fn centroid(&self) -> Option<[f32; 3]> {
        let (weighted_sum, total_weight): ([f64; 3], f64) =
            self.points.iter().fold(([0.0; 3], 0.0), |(sum, tw), p| {
                let w = f64::from(p.weight);
                (
                    [
                        sum[0] + f64::from(p.position[0]) * w,
                        sum[1] + f64::from(p.position[1]) * w,
                        sum[2] + f64::from(p.position[2]) * w,
                    ],
                    tw + w,
                )
            });
        if total_weight <= 0.0 {
            return None;
        }
        Some([
            (weighted_sum[0] / total_weight) as f32,
            (weighted_sum[1] / total_weight) as f32,
            (weighted_sum[2] / total_weight) as f32,
        ])
    }
}

/// Solves the weighted Kabsch problem between `neutral` and `current`.
///
/// Returns the rotation that best aligns `neutral` onto `current`, together
/// with the semantic head pose derived from the relative rotation.
///
/// # Coordinate conventions
///
/// All positions are in the canonical basis used throughout the tracking
/// pipeline. The returned `HeadPose` follows `DESIGN.md`:
///
/// - `yaw > 0`: face turns right in the unmirrored image.
/// - `pitch > 0`: chin goes up.
/// - `roll > 0`: head tilts clockwise as viewed in the unmirrored image.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
pub fn solve_relative_pose(
    neutral: &LandmarkSet,
    current: &LandmarkSet,
) -> Result<PoseAlignment, PoseError> {
    if neutral.len() != current.len() {
        // The API contract requires paired points. Treat mismatch as
        // insufficient data.
        return Err(PoseError::InsufficientPoints(
            neutral.len().min(current.len()),
        ));
    }
    let n = neutral.len();
    if n < MIN_LANDMARK_POINTS {
        return Err(PoseError::InsufficientPoints(n));
    }

    let neutral_centroid = neutral.centroid().ok_or(PoseError::ZeroWeight(
        neutral.points.iter().map(|p| p.weight).sum(),
    ))?;
    let current_centroid = current.centroid().ok_or(PoseError::ZeroWeight(
        current.points.iter().map(|p| p.weight).sum(),
    ))?;

    // Build centered matrices as 3 x N dynamic matrices.
    let mut p = OMatrix::<f32, U3, Dyn>::zeros(n);
    let mut q = OMatrix::<f32, U3, Dyn>::zeros(n);
    for (i, (np, cp)) in neutral.points.iter().zip(current.points.iter()).enumerate() {
        for j in 0..3 {
            p[(j, i)] = (np.position[j] - neutral_centroid[j]) * np.weight.sqrt();
            q[(j, i)] = (cp.position[j] - current_centroid[j]) * cp.weight.sqrt();
        }
    }

    // Covariance H = P * Q^T.
    let h = &p * q.transpose();
    let svd = SVD::new(h, true, true);
    let u = svd.u.ok_or(PoseError::DegeneratePointCloud)?.cast::<f32>();
    let v_t = svd
        .v_t
        .ok_or(PoseError::DegeneratePointCloud)?
        .cast::<f32>();

    // Reflection detection: the unconstrained Kabsch solution R = V U^T is a
    // proper rotation only when det(V U^T) = +1. A negative determinant means
    // the best orthogonal map is a reflection, i.e. the input is a mirrored
    // point cloud. We reject that case rather than silently correcting it.
    let det = (u * v_t).determinant();
    if det < 0.0 {
        return Err(PoseError::ReflectionDetected);
    }
    let r = (u * v_t).transpose();

    // Validate singular values.
    let singular_values = svd.singular_values;
    if singular_values.min() < MIN_SINGULAR_VALUE {
        return Err(PoseError::DegeneratePointCloud);
    }

    let rotation = Rotation3::from_matrix_unchecked(r);
    let quat = UnitQuaternion::from_rotation_matrix(&rotation);

    let pose = quaternion_to_semantic_pose(quat);

    // Rotation-compensated residual translation plus projected size
    // evidence for Issue #166's head-translation contract. Projected radii
    // deliberately use only x/y because canonical z is model-defined depth
    // that does not follow perspective scaling.
    //
    // The rigid model is current = R * (X - pivot) + pivot + d. Choosing the
    // pivot at the weighted neutral centroid (a pure head rotation turns
    // about the face centre) collapses the compensated residual to the
    // centroid displacement, so a rotation-only observation produces no
    // first-order cross-talk:
    //   d = c_current - c_neutral.
    let translation = [
        current_centroid[0] - neutral_centroid[0],
        current_centroid[1] - neutral_centroid[1],
        current_centroid[2] - neutral_centroid[2],
    ];
    let neutral_projected_radius = projected_radius(&neutral.points, &neutral_centroid);
    let current_projected_radius = projected_radius(&current.points, &current_centroid);

    Ok(PoseAlignment {
        rotation: quat,
        pose,
        translation,
        neutral_projected_radius,
        current_projected_radius,
    })
}

fn projected_radius(points: &[WeightedPoint], centroid: &[f32; 3]) -> f32 {
    let (sum_sq, total_weight): (f64, f64) = points.iter().fold((0.0, 0.0), |(acc, tw), p| {
        let w = f64::from(p.weight);
        let dx = f64::from(p.position[0] - centroid[0]);
        let dy = f64::from(p.position[1] - centroid[1]);
        (acc + (dx * dx + dy * dy) * w, tw + w)
    });
    if total_weight <= 0.0 {
        return 0.0;
    }
    ((sum_sq / total_weight).sqrt()) as f32
}

/// Converts a canonical rotation quaternion to semantic yaw/pitch/roll.
///
/// The mapping follows `DESIGN.md`:
///
/// - yaw   > 0  -> +Y axis rotation
/// - pitch > 0  -> +X axis rotation
/// - roll  > 0  -> +Z axis rotation
pub(crate) fn quaternion_to_semantic_pose(q: UnitQuaternion<f32>) -> HeadPose {
    // Canonical basis: right +X, up +Y, forward +Z.
    // A right turn (+yaw) is a rotation around +Y.
    // Chin up (+pitch) is a rotation around +X.
    // Clockwise tilt (+roll) is a rotation around +Z.
    //
    // We want a single quaternion q_total such that
    //   q_total = q_yaw * q_pitch * q_roll
    // where the individual rotations are around the fixed canonical axes:
    //   q_yaw   = rotation(+Y, yaw)
    //   q_pitch = rotation(+X, pitch)
    //   q_roll  = rotation(+Z, roll)
    //
    // nalgebra's `from_euler_angles(roll, pitch, yaw)` composes rotations
    // around fixed axes in the order roll(X) -> pitch(Y) -> yaw(Z), which
    // does NOT match our convention. Instead we compose the three axis-angle
    // rotations explicitly and extract them by inverting that composition.

    let rotation_matrix = q.to_rotation_matrix();
    let r = rotation_matrix.matrix();
    // Our convention is the intrinsic Y-X-Z decomposition
    //   R = R_y(yaw) * R_x(pitch) * R_z(roll)
    // on the standard right-handed basis.
    //
    // Pitch and roll can be read directly from the middle row:
    //   r12 = -sin(pitch)
    //   pitch = asin(-r12)
    //   r10 = cos(pitch) * sin(roll)
    //   r11 = cos(pitch) * cos(roll)
    //   roll = atan2(r10, r11)
    //
    // Yaw is coupled with pitch/roll, so we first undo the pitch and roll
    // rotations by projecting the forward axis through R_z(-roll) * R_x(-pitch):
    //   yaw = atan2(r00*sr*sp + r01*cr*sp + r02*cp,
    //               r20*sr*sp + r21*cr*sp + r22*cp)
    // where (cp, sp) = (cos pitch, sin pitch) and (cr, sr) = (cos roll, sin roll).
    let r00 = r[(0, 0)];
    let r01 = r[(0, 1)];
    let r02 = r[(0, 2)];
    let r10 = r[(1, 0)];
    let r11 = r[(1, 1)];
    let r12 = r[(1, 2)];
    let r20 = r[(2, 0)];
    let r21 = r[(2, 1)];
    let r22 = r[(2, 2)];

    let pitch = (-r12).asin();
    let pitch_cos = pitch.cos();
    let pitch_sin = pitch.sin();
    let roll = if pitch_cos.abs() > 1e-6 {
        r10.atan2(r11)
    } else {
        // Gimbal lock: yaw and roll are coupled; report roll as zero.
        0.0
    };
    let roll_cos = roll.cos();
    let roll_sin = roll.sin();

    let yaw = (r00 * roll_sin * pitch_sin + r01 * roll_cos * pitch_sin + r02 * pitch_cos)
        .atan2(r20 * roll_sin * pitch_sin + r21 * roll_cos * pitch_sin + r22 * pitch_cos);

    HeadPose {
        yaw_rad: yaw,
        pitch_rad: pitch,
        roll_rad: roll,
    }
}

/// Builds a canonical rotation from semantic yaw/pitch/roll in radians.
///
/// This is the inverse of the conversion performed by
/// `quaternion_to_semantic_pose` and is exposed for tests and synthetic
/// fixtures.
#[must_use]
pub fn semantic_pose_to_quaternion(pose: HeadPose) -> UnitQuaternion<f32> {
    // Canonical basis: right +X, up +Y, forward +Z.
    //
    // semantic convention (from `DESIGN.md`):
    //   yaw   > 0  -> face turns right in the unmirrored image
    //   pitch > 0  -> chin goes up
    //   roll  > 0  -> head tilts clockwise as viewed by the camera
    //
    // These map to intrinsic rotations around the standard right-handed axes as
    //   R = R_y(yaw) * R_x(pitch) * R_z(roll)
    // where the rotations are applied roll-first, pitch-second, yaw-third.
    // Quaternion composition follows the same order: q = q_yaw * q_pitch * q_roll.
    let yaw_axis = OVector::<f32, U3>::y_axis();
    let pitch_axis = OVector::<f32, U3>::x_axis();
    let roll_axis = OVector::<f32, U3>::z_axis();

    let q_yaw = UnitQuaternion::from_axis_angle(&yaw_axis, pose.yaw_rad);
    let q_pitch = UnitQuaternion::from_axis_angle(&pitch_axis, pose.pitch_rad);
    let q_roll = UnitQuaternion::from_axis_angle(&roll_axis, pose.roll_rad);

    q_yaw * q_pitch * q_roll
}

/// Wraps an angle to the `[-pi, pi)` interval.
#[must_use]
pub fn normalize_angle_rad(rad: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    let mut v = rad % TAU;
    if v >= PI {
        v -= TAU;
    } else if v < -PI {
        v += TAU;
    }
    v
}

/// Returns `true` if two angles are within `tolerance_rad` after wrapping.
#[cfg(test)]
#[must_use]
pub fn angle_eq(a: f32, b: f32, tolerance_rad: f32) -> bool {
    let diff = normalize_angle_rad(a - b).abs();
    diff <= tolerance_rad
}

/// Creates a synthetic planar face-like point cloud in the canonical basis.
///
/// Points are centered around the origin. The shape is stretched along X
/// (left-right) and Y (up-down) with a small Z depth to avoid degeneracy.
#[must_use]
pub fn synthetic_face_points() -> Vec<[f32; 3]> {
    vec![
        [-1.0, 0.0, 0.05],  // left temple
        [1.0, 0.0, 0.05],   // right temple
        [0.0, 0.8, 0.0],    // forehead
        [0.0, -0.6, 0.1],   // chin
        [-0.5, 0.3, 0.02],  // left eye outer
        [0.5, 0.3, 0.02],   // right eye outer
        [-0.4, -0.3, 0.04], // left mouth corner
        [0.4, -0.3, 0.04],  // right mouth corner
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn points_to_set(points: &[[f32; 3]]) -> LandmarkSet {
        let mut set = LandmarkSet::new();
        for p in points {
            set.push(*p, 1.0);
        }
        set
    }

    fn rotate_points(points: &[[f32; 3]], quat: UnitQuaternion<f32>) -> Vec<[f32; 3]> {
        points
            .iter()
            .map(|p| {
                let v = OVector::<f32, U3>::new(p[0], p[1], p[2]);
                let r = quat * v;
                [r.x, r.y, r.z]
            })
            .collect()
    }

    #[test]
    fn identity_pose() {
        let points = synthetic_face_points();
        let neutral = points_to_set(&points);
        let current = points_to_set(&points);
        let alignment = solve_relative_pose(&neutral, &current).unwrap();
        assert_relative_eq!(alignment.pose.yaw_rad, 0.0, epsilon = 1e-4);
        assert_relative_eq!(alignment.pose.pitch_rad, 0.0, epsilon = 1e-4);
        assert_relative_eq!(alignment.pose.roll_rad, 0.0, epsilon = 1e-4);
    }

    #[test]
    fn translation_invariant() {
        let points = synthetic_face_points();
        let neutral = points_to_set(&points);
        let translated: Vec<[f32; 3]> = points
            .iter()
            .map(|p| [p[0] + 10.0, p[1] - 5.0, p[2] + 2.0])
            .collect();
        let current = points_to_set(&translated);
        let alignment = solve_relative_pose(&neutral, &current).unwrap();
        assert_relative_eq!(alignment.pose.yaw_rad, 0.0, epsilon = 1e-4);
        assert_relative_eq!(alignment.pose.pitch_rad, 0.0, epsilon = 1e-4);
        assert_relative_eq!(alignment.pose.roll_rad, 0.0, epsilon = 1e-4);
    }

    #[test]
    fn uniform_scale_invariant() {
        let points = synthetic_face_points();
        let neutral = points_to_set(&points);
        let scaled: Vec<[f32; 3]> = points
            .iter()
            .map(|p| [p[0] * 3.0, p[1] * 3.0, p[2] * 3.0])
            .collect();
        let current = points_to_set(&scaled);
        let alignment = solve_relative_pose(&neutral, &current).unwrap();
        assert_relative_eq!(alignment.pose.yaw_rad, 0.0, epsilon = 1e-4);
        assert_relative_eq!(alignment.pose.pitch_rad, 0.0, epsilon = 1e-4);
        assert_relative_eq!(alignment.pose.roll_rad, 0.0, epsilon = 1e-4);
    }

    #[test]
    fn recover_yaw_positive_right() {
        let points = synthetic_face_points();
        let neutral = points_to_set(&points);
        let expected = HeadPose {
            yaw_rad: 15.0f32.to_radians(),
            pitch_rad: 0.0,
            roll_rad: 0.0,
        };
        let rotated = rotate_points(&points, semantic_pose_to_quaternion(expected));
        let current = points_to_set(&rotated);
        let alignment = solve_relative_pose(&neutral, &current).unwrap();
        assert_relative_eq!(alignment.pose.yaw_rad, expected.yaw_rad, epsilon = 1e-3);
        assert_relative_eq!(alignment.pose.pitch_rad, 0.0, epsilon = 1e-3);
        assert_relative_eq!(alignment.pose.roll_rad, 0.0, epsilon = 1e-3);
        assert!(alignment.pose.yaw_rad > 0.0);
    }

    #[test]
    fn recover_pitch_positive_chin_up() {
        let points = synthetic_face_points();
        let neutral = points_to_set(&points);
        let expected = HeadPose {
            yaw_rad: 0.0,
            pitch_rad: 10.0f32.to_radians(),
            roll_rad: 0.0,
        };
        let rotated = rotate_points(&points, semantic_pose_to_quaternion(expected));
        let current = points_to_set(&rotated);
        let alignment = solve_relative_pose(&neutral, &current).unwrap();
        assert_relative_eq!(alignment.pose.yaw_rad, 0.0, epsilon = 1e-3);
        assert_relative_eq!(alignment.pose.pitch_rad, expected.pitch_rad, epsilon = 1e-3);
        assert_relative_eq!(alignment.pose.roll_rad, 0.0, epsilon = 1e-3);
        assert!(alignment.pose.pitch_rad > 0.0);
    }

    #[test]
    fn recover_roll_negative_clockwise() {
        let points = synthetic_face_points();
        let neutral = points_to_set(&points);
        // DESIGN.md convention: clockwise tilt (as viewed) -> roll > 0.
        // Test with a negative angle to verify the sign is preserved.
        let expected = HeadPose {
            yaw_rad: 0.0,
            pitch_rad: 0.0,
            roll_rad: -12.0f32.to_radians(),
        };
        let rotated = rotate_points(&points, semantic_pose_to_quaternion(expected));
        let current = points_to_set(&rotated);
        let alignment = solve_relative_pose(&neutral, &current).unwrap();
        assert_relative_eq!(alignment.pose.yaw_rad, 0.0, epsilon = 1e-3);
        assert_relative_eq!(alignment.pose.pitch_rad, 0.0, epsilon = 1e-3);
        assert_relative_eq!(alignment.pose.roll_rad, expected.roll_rad, epsilon = 1e-3);
        assert!(alignment.pose.roll_rad < 0.0);
    }

    #[test]
    fn recover_combined_pose() {
        let points = synthetic_face_points();
        let neutral = points_to_set(&points);
        let expected = HeadPose {
            yaw_rad: 15.0f32.to_radians(),
            pitch_rad: 10.0f32.to_radians(),
            roll_rad: -12.0f32.to_radians(),
        };
        let rotated = rotate_points(&points, semantic_pose_to_quaternion(expected));
        let current = points_to_set(&rotated);
        let alignment = solve_relative_pose(&neutral, &current).unwrap();
        assert_relative_eq!(alignment.pose.yaw_rad, expected.yaw_rad, epsilon = 1e-3);
        assert_relative_eq!(alignment.pose.pitch_rad, expected.pitch_rad, epsilon = 1e-3);
        assert_relative_eq!(alignment.pose.roll_rad, expected.roll_rad, epsilon = 1e-3);
    }

    #[test]
    fn insufficient_points_error() {
        let neutral = points_to_set(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let current = neutral.clone();
        let err = solve_relative_pose(&neutral, &current).unwrap_err();
        assert_eq!(err, PoseError::InsufficientPoints(2));
    }

    #[test]
    fn collinear_points_degenerate() {
        let points: Vec<[f32; 3]> = (0..5).map(|i| [i as f32, 0.0, 0.0]).collect();
        let neutral = points_to_set(&points);
        let current = neutral.clone();
        let err = solve_relative_pose(&neutral, &current).unwrap_err();
        assert_eq!(err, PoseError::DegeneratePointCloud);
    }

    #[test]
    fn reflection_detected_for_mirrored_input() {
        // Mirroring across X is a reflection, not a rotation.
        let points = synthetic_face_points();
        let neutral = points_to_set(&points);
        let mirrored: Vec<[f32; 3]> = points.iter().map(|p| [-p[0], p[1], p[2]]).collect();
        let current = points_to_set(&mirrored);
        let err = solve_relative_pose(&neutral, &current).unwrap_err();
        assert_eq!(err, PoseError::ReflectionDetected);
    }

    #[test]
    fn semantic_round_trip() {
        let expected = HeadPose {
            yaw_rad: 30.0f32.to_radians(),
            pitch_rad: -20.0f32.to_radians(),
            roll_rad: 15.0f32.to_radians(),
        };
        let q = semantic_pose_to_quaternion(expected);
        let actual = quaternion_to_semantic_pose(q);
        assert_relative_eq!(actual.yaw_rad, expected.yaw_rad, epsilon = 1e-4);
        assert_relative_eq!(actual.pitch_rad, expected.pitch_rad, epsilon = 1e-4);
        assert_relative_eq!(actual.roll_rad, expected.roll_rad, epsilon = 1e-4);
    }

    #[test]
    fn normalize_angle_wraps() {
        assert_relative_eq!(
            normalize_angle_rad(4.0),
            4.0 - 2.0 * std::f32::consts::PI,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            normalize_angle_rad(-4.0),
            -4.0 + 2.0 * std::f32::consts::PI,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            normalize_angle_rad(std::f32::consts::PI),
            -std::f32::consts::PI,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            normalize_angle_rad(-std::f32::consts::PI),
            -std::f32::consts::PI,
            epsilon = 1e-6
        );
    }
}
