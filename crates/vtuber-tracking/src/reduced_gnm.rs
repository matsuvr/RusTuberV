//! Validated teacher-aligned basis loading and MediaPipe pose seeding.

use nalgebra::{Quaternion, UnitQuaternion};
use vtuber_core::CameraFaceTransform;
use vtuber_gnm::{DenseProjection, GnmReducedExpressionBasis, GnmReprojectionError};

use crate::teacher_aligned_basis::{
    TeacherAlignedBasisError, TeacherAlignedGnmBasisArtifact,
    reconstruct_teacher_aligned_expression,
};

/// Converts a verified teacher-aligned artifact into the low-level runtime basis.
///
/// # Errors
///
/// Rejects schema, hash, model, mapping, rank, numeric, and orthogonality
/// mismatches. No basis regeneration or full-solver fallback is performed.
pub fn load_reduced_gnm_basis(
    artifact: &TeacherAlignedGnmBasisArtifact,
    expected_model_sha256: &str,
    expected_mapping_revision: u32,
) -> Result<GnmReducedExpressionBasis, TeacherAlignedBasisError> {
    reconstruct_teacher_aligned_expression(&vec![0.0; artifact.rank], artifact)?;
    if artifact.model_sha256 != expected_model_sha256 {
        return Err(TeacherAlignedBasisError::InvalidShape("model SHA-256"));
    }
    if artifact.mapping_schema_revision != expected_mapping_revision {
        return Err(TeacherAlignedBasisError::InvalidShape(
            "mapping schema revision",
        ));
    }
    Ok(GnmReducedExpressionBasis::new(
        artifact.rank,
        artifact.basis_row_major.clone(),
    )?)
}

/// Seeds the GNM `Rz(roll) * Rx(pitch) * Ry(yaw)` rotation from MediaPipe.
///
/// Translation, focal length, and principal point are retained from `base`;
/// the MediaPipe translation is not part of this initial-rotation contract.
///
/// # Errors
///
/// Rejects an invalid MediaPipe transform or an invalid reconstructed projection.
pub fn seed_gnm_projection_rotation(
    transform: &CameraFaceTransform,
    base: &DenseProjection,
) -> Result<DenseProjection, GnmReprojectionError> {
    if !transform.is_valid() {
        return Err(GnmReprojectionError::InvalidProjection(
            "MediaPipe camera-to-face rotation must be a finite unit quaternion",
        ));
    }
    let [x, y, z, w] = transform.rotation_xyzw;
    let rotation = UnitQuaternion::from_quaternion(Quaternion::new(w, x, y, z));
    let matrix = rotation.to_rotation_matrix();
    let values = matrix.matrix();
    let pitch = values[(2, 1)].clamp(-1.0, 1.0).asin();
    let yaw = (-values[(2, 0)]).atan2(values[(2, 2)]);
    let roll = (-values[(0, 1)]).atan2(values[(1, 1)]);
    DenseProjection::new(
        [yaw, pitch, roll],
        base.translation(),
        base.focal(),
        base.principal_point(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, UnitQuaternion};

    fn transform_for(yaw: f32, pitch: f32, roll: f32) -> CameraFaceTransform {
        let rotation = Rotation3::from_axis_angle(&nalgebra::Vector3::z_axis(), roll)
            * Rotation3::from_axis_angle(&nalgebra::Vector3::x_axis(), pitch)
            * Rotation3::from_axis_angle(&nalgebra::Vector3::y_axis(), yaw);
        let quaternion = UnitQuaternion::from_rotation_matrix(&rotation);
        let value = quaternion.quaternion();
        CameraFaceTransform {
            rotation_xyzw: [value.i, value.j, value.k, value.w],
            translation_xyz: [4.0, 5.0, 6.0],
        }
    }

    #[test]
    fn pose_seed_round_trips_axes_signs_and_preserves_camera_fields() {
        let base = DenseProjection::new([0.0; 3], [0.1, -0.2, 0.7], 1.4, [0.45, 0.55]).unwrap();
        for expected in [[0.3, 0.0, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, 0.4]] {
            let seeded = seed_gnm_projection_rotation(
                &transform_for(expected[0], expected[1], expected[2]),
                &base,
            )
            .unwrap();
            for (actual, expected) in seeded.yaw_pitch_roll().iter().zip(expected) {
                assert!((actual - expected).abs() < 1.0e-5);
            }
            assert_eq!(seeded.translation(), base.translation());
            assert_eq!(seeded.focal(), base.focal());
            assert_eq!(seeded.principal_point(), base.principal_point());
        }
    }

    #[test]
    fn invalid_transform_is_a_typed_projection_error() {
        let base = DenseProjection::new([0.0; 3], [0.0, 0.0, 1.0], 1.0, [0.5; 2]).unwrap();
        let mut transform = CameraFaceTransform::identity();
        transform.rotation_xyzw[0] = f32::NAN;
        assert!(matches!(
            seed_gnm_projection_rotation(&transform, &base),
            Err(GnmReprojectionError::InvalidProjection(_))
        ));
    }
}
