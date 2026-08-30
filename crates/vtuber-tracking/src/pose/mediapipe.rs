//! Neutral-relative pose from MediaPipe face transforms.
//!
//! MediaPipe emits a rigid face-to-camera transform. This module composes
//! transforms before extracting semantic angles, so neutral pose is never
//! subtracted as independent Euler components.

use nalgebra::{OVector, U3, UnitQuaternion};
use thiserror::Error;

use vtuber_core::{CameraFaceTransform, HeadPose};

use super::quaternion_to_semantic_pose;

/// Errors raised while composing MediaPipe face transforms.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum MediaPipePoseError {
    /// A source transform contains a non-finite value or non-unit quaternion.
    #[error("MediaPipe face transform is invalid")]
    InvalidTransform,
    /// A relative transform contains a non-finite value.
    #[error("relative MediaPipe transform is non-finite")]
    NonFiniteRelativeTransform,
}

/// A neutral-relative rigid face transform in the MediaPipe basis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RelativeFaceTransform {
    /// Relative MediaPipe rotation from neutral to current.
    pub rotation: UnitQuaternion<f32>,
    /// Relative translation expressed in the neutral face basis.
    pub translation_xyz: [f32; 3],
}

/// Converts one MediaPipe rotation into the application's semantic basis.
///
/// The guided Windows probe established that MediaPipe's yaw already follows
/// the unmirrored image-right convention. Its pitch and roll signs are
/// inverted relative to the application's chin-up and image-clockwise
/// conventions. Keep this conversion in one named function so the
/// evidence-backed sign mapping is not scattered through UI or avatar code.
#[must_use]
pub fn mediapipe_to_application_basis(rotation: UnitQuaternion<f32>) -> HeadPose {
    let mediapipe_pose = quaternion_to_semantic_pose(rotation);
    HeadPose {
        yaw_rad: mediapipe_pose.yaw_rad,
        pitch_rad: -mediapipe_pose.pitch_rad,
        roll_rad: -mediapipe_pose.roll_rad,
    }
}

/// Composes `inverse(neutral) * current` for rigid face transforms.
pub fn relative_transform(
    neutral: CameraFaceTransform,
    current: CameraFaceTransform,
) -> Result<RelativeFaceTransform, MediaPipePoseError> {
    if !neutral.is_valid() || !current.is_valid() {
        return Err(MediaPipePoseError::InvalidTransform);
    }
    if !neutral
        .translation_xyz
        .iter()
        .all(|value| value.is_finite())
        || !current
            .translation_xyz
            .iter()
            .all(|value| value.is_finite())
    {
        return Err(MediaPipePoseError::InvalidTransform);
    }

    let neutral_rotation = quaternion(neutral.rotation_xyzw);
    let current_rotation = quaternion(current.rotation_xyzw);
    let relative_rotation = neutral_rotation.inverse() * current_rotation;
    let delta_translation = [
        current.translation_xyz[0] - neutral.translation_xyz[0],
        current.translation_xyz[1] - neutral.translation_xyz[1],
        current.translation_xyz[2] - neutral.translation_xyz[2],
    ];
    let relative_translation = neutral_rotation.inverse()
        * OVector::<f32, U3>::new(
            delta_translation[0],
            delta_translation[1],
            delta_translation[2],
        );
    let translation_xyz = [
        relative_translation.x,
        relative_translation.y,
        relative_translation.z,
    ];
    if !translation_xyz.iter().all(|value| value.is_finite()) {
        return Err(MediaPipePoseError::NonFiniteRelativeTransform);
    }

    Ok(RelativeFaceTransform {
        rotation: relative_rotation,
        translation_xyz,
    })
}

/// Converts a neutral-relative MediaPipe transform to semantic head pose.
pub fn relative_pose(
    neutral: CameraFaceTransform,
    current: CameraFaceTransform,
) -> Result<HeadPose, MediaPipePoseError> {
    let relative = relative_transform(neutral, current)?;
    Ok(mediapipe_to_application_basis(relative.rotation))
}

fn quaternion(rotation_xyzw: [f32; 4]) -> UnitQuaternion<f32> {
    UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
        rotation_xyzw[3],
        rotation_xyzw[0],
        rotation_xyzw[1],
        rotation_xyzw[2],
    ))
}

#[cfg(test)]
mod tests {
    use super::super::semantic_pose_to_quaternion;
    use super::*;
    use approx::assert_relative_eq;
    use std::f32::consts::FRAC_PI_2;

    fn transform(rotation: UnitQuaternion<f32>, translation_xyz: [f32; 3]) -> CameraFaceTransform {
        let q = rotation.quaternion();
        CameraFaceTransform {
            rotation_xyzw: [q.i, q.j, q.k, q.w],
            translation_xyz,
        }
    }

    #[test]
    fn neutral_relative_transform_is_identity_and_zero_translation() {
        let neutral = transform(UnitQuaternion::identity(), [1.0, 2.0, 3.0]);
        let current = transform(UnitQuaternion::identity(), [1.0, 2.0, 3.0]);
        let relative = relative_transform(neutral, current).expect("identity is valid");
        assert_relative_eq!(relative.rotation.angle(), 0.0, epsilon = 1.0e-6);
        assert_eq!(relative.translation_xyz, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn relative_translation_is_rotated_by_neutral_inverse() {
        let neutral_rotation = UnitQuaternion::from_euler_angles(0.0, 0.0, FRAC_PI_2);
        let neutral = transform(neutral_rotation, [2.0, 3.0, 4.0]);
        let current = transform(neutral_rotation, [3.0, 3.0, 4.0]);
        let relative = relative_transform(neutral, current).expect("rigid transforms are valid");
        assert_relative_eq!(relative.translation_xyz[0], 0.0, epsilon = 1.0e-5);
        assert_relative_eq!(relative.translation_xyz[1], -1.0, epsilon = 1.0e-5);
        assert_relative_eq!(relative.translation_xyz[2], 0.0, epsilon = 1.0e-5);
    }

    #[test]
    fn relative_pose_uses_composed_rotation_without_euler_subtraction() {
        let neutral_pose = HeadPose {
            yaw_rad: 0.2,
            pitch_rad: -0.1,
            roll_rad: 0.15,
        };
        let delta_pose = HeadPose {
            yaw_rad: 0.3,
            pitch_rad: 0.1,
            roll_rad: -0.2,
        };
        let neutral = transform(semantic_pose_to_quaternion(neutral_pose), [0.0; 3]);
        let delta = semantic_pose_to_quaternion(delta_pose);
        let current = transform(semantic_pose_to_quaternion(neutral_pose) * delta, [0.0; 3]);
        let actual = relative_pose(neutral, current).expect("relative rotation is valid");
        let expected = mediapipe_to_application_basis(delta);
        assert_relative_eq!(actual.yaw_rad, expected.yaw_rad, epsilon = 1.0e-5);
        assert_relative_eq!(actual.pitch_rad, expected.pitch_rad, epsilon = 1.0e-5);
        assert_relative_eq!(actual.roll_rad, expected.roll_rad, epsilon = 1.0e-5);
    }

    #[test]
    fn invalid_transform_is_rejected_before_composition() {
        let mut invalid = transform(UnitQuaternion::identity(), [0.0; 3]);
        invalid.rotation_xyzw[0] = f32::NAN;
        assert_eq!(
            relative_transform(invalid, transform(UnitQuaternion::identity(), [0.0; 3])),
            Err(MediaPipePoseError::InvalidTransform)
        );
    }

    #[test]
    fn basis_mapping_flips_pitch_and_roll_but_preserves_image_yaw() {
        let source_pose = HeadPose {
            yaw_rad: 0.3,
            pitch_rad: 0.2,
            roll_rad: -0.4,
        };
        let mapped_pose = mediapipe_to_application_basis(semantic_pose_to_quaternion(source_pose));
        assert_relative_eq!(mapped_pose.yaw_rad, source_pose.yaw_rad, epsilon = 1.0e-5);
        assert_relative_eq!(
            mapped_pose.pitch_rad,
            -source_pose.pitch_rad,
            epsilon = 1.0e-5
        );
        assert_relative_eq!(
            mapped_pose.roll_rad,
            -source_pose.roll_rad,
            epsilon = 1.0e-5
        );
    }
}
