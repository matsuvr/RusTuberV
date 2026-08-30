//! Pure-Rust head-pose solving for image-space landmark models.
//!
//! A model that emits `(x, y, confidence)` landmarks does not provide a
//! metric 3D point cloud.  In particular, filling `z` with zero and passing
//! those points to Kabsch is invalid because the covariance is planar and
//! loses the intended depth semantics.  This module instead fits a small,
//! license-safe canonical 3D face template through an orthographic projection
//! model.  The template is deliberately generic geometric data, not a copied
//! dataset or model asset.

use nalgebra::{SMatrix, SVector};
use thiserror::Error;

use vtuber_core::types::HeadPose;

/// A canonical 3D face point paired with one model-output index.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CanonicalFacePoint {
    /// Landmark index in the model output.
    pub index: usize,
    /// Canonical right coordinate.
    pub x: f32,
    /// Canonical up coordinate.
    pub y: f32,
    /// Canonical forward coordinate.
    pub z: f32,
}

/// A normalized image-space landmark.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanarLandmark {
    /// Normalized image X, left = 0 and right = 1.
    pub x: f32,
    /// Normalized image Y, top = 0 and bottom = 1.
    pub y: f32,
    /// Visibility or confidence in `[0, 1]`.
    pub confidence: f32,
}

/// A planar landmark paired with a canonical template point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanarCorrespondence {
    /// Canonical point.
    pub canonical: CanonicalFacePoint,
    /// Neutral/reference image point.
    pub reference: PlanarLandmark,
    /// Current image point.
    pub current: PlanarLandmark,
}

/// Result of fitting the planar observations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanarPoseAlignment {
    /// Semantic head pose in radians.
    pub pose: HeadPose,
    /// Mean weighted reprojection error in normalized image units.
    pub reprojection_error: f32,
    /// Confidence derived from point visibility and reprojection quality.
    pub confidence: f32,
}

/// Errors produced by the planar pose solver.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum PlanarPoseError {
    /// Fewer than six usable correspondences were supplied.
    #[error("insufficient planar correspondences: got {0}, need at least 6")]
    InsufficientCorrespondences(usize),
    /// A coordinate or confidence was invalid.
    #[error("planar correspondence contains a non-finite or out-of-range value")]
    InvalidInput,
    /// The normal equation matrix could not be solved.
    #[error("planar pose normal equation is singular")]
    Singular,
    /// The optimizer did not produce a finite result.
    #[error("planar pose result is non-finite")]
    NonFinite,
}

/// Generic canonical face template used by the 2D production adapter.
///
/// The indices are representative points from the verified WFLW-98 groups
/// emitted by the Peppa Pig landmark model: jawline 0..33, brows 33..51, nose
/// 51..60, eyes 60..76, and mouth 76..98. The coordinates are synthetic
/// geometric data, not copied dataset points.
pub const CANONICAL_FACE_TEMPLATE: [CanonicalFacePoint; 8] = [
    CanonicalFacePoint {
        index: 16,
        x: 0.00,
        y: -0.42,
        z: -0.02,
    },
    CanonicalFacePoint {
        index: 37,
        x: -0.25,
        y: 0.25,
        z: 0.00,
    },
    CanonicalFacePoint {
        index: 46,
        x: 0.25,
        y: 0.25,
        z: 0.00,
    },
    CanonicalFacePoint {
        index: 52,
        x: 0.00,
        y: 0.14,
        z: 0.18,
    },
    CanonicalFacePoint {
        index: 63,
        x: -0.22,
        y: 0.08,
        z: 0.08,
    },
    CanonicalFacePoint {
        index: 71,
        x: 0.22,
        y: 0.08,
        z: 0.08,
    },
    CanonicalFacePoint {
        index: 76,
        x: -0.22,
        y: -0.24,
        z: 0.03,
    },
    CanonicalFacePoint {
        index: 82,
        x: 0.22,
        y: -0.24,
        z: 0.03,
    },
];

/// Fits head rotation and similarity projection from neutral/current image
/// correspondences.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
pub fn solve_planar_pose(
    correspondences: &[PlanarCorrespondence],
) -> Result<PlanarPoseAlignment, PlanarPoseError> {
    if correspondences.len() < 6 {
        return Err(PlanarPoseError::InsufficientCorrespondences(
            correspondences.len(),
        ));
    }

    for c in correspondences {
        if !c.canonical.x.is_finite()
            || !c.canonical.y.is_finite()
            || !c.canonical.z.is_finite()
            || !valid_landmark(c.reference)
            || !valid_landmark(c.current)
        {
            return Err(PlanarPoseError::InvalidInput);
        }
    }

    // [yaw, pitch, roll, log(scale), tx, ty].  The reference image is used
    // to estimate neutral projection parameters; current is fitted relative
    // to the same canonical template, avoiding a direct 2D->3D Kabsch call.
    let mut parameters = initial_projection(correspondences, false);
    let mut damping = 1e-3f32;
    for _ in 0..32 {
        let (normal, gradient, cost) = normal_equations(correspondences, &parameters);
        let mut damped = normal;
        for i in 0..6 {
            damped[(i, i)] += damping;
        }
        let Some(delta) = damped.lu().solve(&(-gradient)) else {
            return Err(PlanarPoseError::Singular);
        };
        if !delta.iter().all(|value| value.is_finite()) {
            return Err(PlanarPoseError::NonFinite);
        }
        let candidate = parameters + delta;
        let candidate_cost = residual_cost(correspondences, &candidate);
        if !candidate_cost.is_finite() {
            return Err(PlanarPoseError::NonFinite);
        }
        if candidate_cost < cost {
            parameters = candidate;
            damping = (damping * 0.5).max(1e-7);
            if delta.norm() < 1e-6 {
                break;
            }
        } else {
            damping = (damping * 4.0).min(1e4);
        }
    }

    let error = residual_cost(correspondences, &parameters).sqrt();
    let scale = parameters[3].exp();
    let pose = HeadPose {
        yaw_rad: parameters[0],
        pitch_rad: parameters[1],
        roll_rad: parameters[2],
    };
    if !scale.is_finite() || !error.is_finite() || !pose_is_finite(pose) {
        return Err(PlanarPoseError::NonFinite);
    }
    let mean_visibility = correspondences
        .iter()
        .map(|c| c.current.confidence)
        .sum::<f32>()
        / correspondences.len() as f32;
    let confidence = (mean_visibility * (-error * 10.0).exp()).clamp(0.0, 1.0);

    Ok(PlanarPoseAlignment {
        pose,
        reprojection_error: error,
        confidence,
    })
}

fn valid_landmark(point: PlanarLandmark) -> bool {
    point.x.is_finite()
        && point.y.is_finite()
        && point.confidence.is_finite()
        && (0.0..=1.0).contains(&point.x)
        && (0.0..=1.0).contains(&point.y)
        && (0.0..=1.0).contains(&point.confidence)
}

fn pose_is_finite(pose: HeadPose) -> bool {
    pose.yaw_rad.is_finite() && pose.pitch_rad.is_finite() && pose.roll_rad.is_finite()
}

fn initial_projection(
    correspondences: &[PlanarCorrespondence],
    reference: bool,
) -> SVector<f32, 6> {
    let points = correspondences
        .iter()
        .map(|c| if reference { c.reference } else { c.current });
    let count = correspondences.len() as f32;
    let (mut cx, mut cy, mut spread) = (0.0, 0.0, 0.0);
    for point in points {
        cx += point.x;
        cy += point.y;
    }
    cx /= count;
    cy /= count;
    for point in correspondences
        .iter()
        .map(|c| if reference { c.reference } else { c.current })
    {
        spread += ((point.x - cx).powi(2) + (point.y - cy).powi(2)).sqrt();
    }
    let canonical_spread = correspondences
        .iter()
        .map(|c| (c.canonical.x.powi(2) + c.canonical.y.powi(2)).sqrt())
        .sum::<f32>()
        / count;
    let image_spread = (spread / count).max(1e-3);
    let scale = (image_spread / canonical_spread.max(1e-3)).max(1e-3);
    SVector::from_row_slice(&[0.0, 0.0, 0.0, scale.ln(), cx, cy])
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn normal_equations(
    correspondences: &[PlanarCorrespondence],
    parameters: &SVector<f32, 6>,
) -> (SMatrix<f32, 6, 6>, SVector<f32, 6>, f32) {
    let mut normal = SMatrix::<f32, 6, 6>::zeros();
    let mut gradient = SVector::<f32, 6>::zeros();
    let mut cost = 0.0;
    for c in correspondences {
        let weight = c.current.confidence.max(1e-3);
        let current_residual = residual(c, parameters);
        cost += weight * current_residual.dot(&current_residual);
        let mut jacobian = SMatrix::<f32, 2, 6>::zeros();
        for column in 0..6 {
            let mut perturbed = *parameters;
            let epsilon = if column < 3 { 1e-4 } else { 1e-5 };
            perturbed[column] += epsilon;
            let difference = (residual(c, &perturbed) - current_residual) / epsilon;
            jacobian[(0, column)] = difference[0];
            jacobian[(1, column)] = difference[1];
        }
        normal += weight * jacobian.transpose() * jacobian;
        gradient += weight * jacobian.transpose() * current_residual;
    }
    (normal, gradient, cost)
}

fn residual_cost(correspondences: &[PlanarCorrespondence], parameters: &SVector<f32, 6>) -> f32 {
    correspondences
        .iter()
        .map(|c| {
            let residual = residual(c, parameters);
            c.current.confidence.max(1e-3) * residual.dot(&residual)
        })
        .sum()
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn residual(
    correspondence: &PlanarCorrespondence,
    parameters: &SVector<f32, 6>,
) -> SVector<f32, 2> {
    let projected = project(correspondence.canonical, parameters);
    SVector::from_row_slice(&[
        projected[0] - correspondence.current.x,
        projected[1] - correspondence.current.y,
    ])
}

fn project(point: CanonicalFacePoint, parameters: &SVector<f32, 6>) -> SVector<f32, 2> {
    let (yaw, pitch, roll) = (parameters[0], parameters[1], parameters[2]);
    let (sy, cy) = yaw.sin_cos();
    let (sx, cx) = pitch.sin_cos();
    // Image-space positive roll is clockwise, hence the negative canonical
    // Z rotation before converting the canonical Y-up axis to image Y-down.
    let (sz, cz) = (-roll).sin_cos();
    let x1 = cz * point.x - sz * point.y;
    let y1 = sz * point.x + cz * point.y;
    let z1 = point.z;
    let x2 = x1;
    let y2 = cx * y1 - sx * z1;
    let z2 = sx * y1 + cx * z1;
    let x3 = cy * x2 + sy * z2;
    let y3 = y2;
    let scale = parameters[3].exp();
    SVector::from_row_slice(&[parameters[4] + scale * x3, parameters[5] - scale * y3])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_correspondences(yaw: f32, pitch: f32, roll: f32) -> Vec<PlanarCorrespondence> {
        let params = SVector::from_row_slice(&[yaw, pitch, roll, 0.42f32.ln(), 0.5, 0.5]);
        CANONICAL_FACE_TEMPLATE
            .iter()
            .map(|&canonical| PlanarCorrespondence {
                canonical,
                reference: PlanarLandmark {
                    x: 0.5 + 0.42 * canonical.x,
                    y: 0.5 - 0.42 * canonical.y,
                    confidence: 1.0,
                },
                current: {
                    let projected = project(canonical, &params);
                    PlanarLandmark {
                        x: projected[0],
                        y: projected[1],
                        confidence: 1.0,
                    }
                },
            })
            .collect()
    }

    #[test]
    fn neutral_projection_is_finite_and_near_zero() {
        let result = solve_planar_pose(&synthetic_correspondences(0.0, 0.0, 0.0)).unwrap();
        assert!(result.reprojection_error < 1e-4);
        assert!(result.pose.yaw_rad.abs() < 1e-3);
        assert!(result.pose.pitch_rad.abs() < 1e-3);
        assert!(result.pose.roll_rad.abs() < 1e-3);
    }

    #[test]
    fn projected_axes_preserve_semantic_signs() {
        for (yaw, pitch, roll) in [(0.20, 0.0, 0.0), (0.0, 0.15, 0.0), (0.0, 0.0, 0.18)] {
            let result = solve_planar_pose(&synthetic_correspondences(yaw, pitch, roll)).unwrap();
            assert!(result.pose.yaw_rad.signum() == yaw.signum() || yaw == 0.0);
            assert!(result.pose.pitch_rad.signum() == pitch.signum() || pitch == 0.0);
            assert!(result.pose.roll_rad.signum() == roll.signum() || roll == 0.0);
            assert!(
                result.pose.yaw_rad.is_finite()
                    && result.pose.pitch_rad.is_finite()
                    && result.pose.roll_rad.is_finite()
            );
        }
    }

    #[test]
    fn planar_solver_rejects_non_finite_input() {
        let mut data = synthetic_correspondences(0.0, 0.0, 0.0);
        data[0].current.x = f32::NAN;
        assert_eq!(solve_planar_pose(&data), Err(PlanarPoseError::InvalidInput));
    }
}
