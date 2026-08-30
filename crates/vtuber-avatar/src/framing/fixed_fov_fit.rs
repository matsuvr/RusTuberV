//! Pure fixed-FOV perspective fitting for a world-space bounding box.

use bevy::prelude::*;

use super::head_subtree_bounds::WorldBounds;

/// The vertical field of view required by the avatar framing design.
pub const FIXED_VERTICAL_FOV: f32 = 12.0_f32.to_radians();

const MAX_NDC: f32 = 0.95;
const DISTANCE_SAFETY_SCALE: f32 = 1.001;
const DISTANCE_EPSILON: f32 = 1e-4;
const MIN_QUATERNION_LENGTH_SQUARED: f32 = 1e-12;

/// Why a fixed-FOV fit could not be produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FixedFovFitError {
    /// The input bounds contain non-finite values or have reversed extents.
    Bounds,
    /// The viewport aspect ratio is non-finite or not positive.
    AspectRatio,
    /// The camera near plane is non-finite or not positive.
    NearPlane,
    /// The camera orientation is non-finite or has no usable rotation.
    CameraOrientation,
}

/// Camera placement produced by [`solve_fixed_fov_fit`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FixedFovFit {
    /// The point placed on the camera optical axis.
    pub(crate) target: Vec3,
    /// The camera translation on the optical axis through `target`.
    pub(crate) translation: Vec3,
    /// The solved dolly distance from the target along the optical axis.
    pub(crate) distance: f32,
    /// The vertical FOV used by the fit.
    pub(crate) vertical_fov: f32,
}

/// Solves a perspective camera dolly for a fixed 12° vertical FOV.
///
/// The supplied orientation defines the camera's optical axis and roll. The
/// resulting translation is placed on that axis through the bounds center, so
/// the center is not offset by the camera's previous world position. Every
/// corner is checked against the near plane and the 0.95 NDC safe rectangle.
pub(crate) fn solve_fixed_fov_fit(
    bounds: WorldBounds,
    camera_orientation: Quat,
    aspect_ratio: f32,
    near_plane: f32,
) -> Result<FixedFovFit, FixedFovFitError> {
    if !bounds.min().is_finite()
        || !bounds.max().is_finite()
        || bounds.min().x > bounds.max().x
        || bounds.min().y > bounds.max().y
        || bounds.min().z > bounds.max().z
    {
        return Err(FixedFovFitError::Bounds);
    }
    if !aspect_ratio.is_finite() || aspect_ratio <= 0.0 {
        return Err(FixedFovFitError::AspectRatio);
    }
    if !near_plane.is_finite() || near_plane <= 0.0 {
        return Err(FixedFovFitError::NearPlane);
    }
    if !camera_orientation.is_finite()
        || camera_orientation.length_squared() <= MIN_QUATERNION_LENGTH_SQUARED
    {
        return Err(FixedFovFitError::CameraOrientation);
    }

    let orientation = camera_orientation.normalize();
    let inverse_orientation = orientation.inverse();
    let forward = orientation * -Vec3::Z;
    if !forward.is_finite() {
        return Err(FixedFovFitError::CameraOrientation);
    }

    let vertical_tangent = (FIXED_VERTICAL_FOV * 0.5).tan();
    let horizontal_tangent = vertical_tangent * aspect_ratio;
    if !vertical_tangent.is_finite() || !horizontal_tangent.is_finite() {
        return Err(FixedFovFitError::AspectRatio);
    }

    let target = bounds.center();
    let mut required_distance = near_plane;
    for corner in bounds.corners() {
        let camera_relative = inverse_orientation * (corner - target);
        if !camera_relative.is_finite() {
            return Err(FixedFovFitError::Bounds);
        }

        // Bevy's camera looks along local -Z, so positive view depth is the
        // negative local Z coordinate. With the camera at target - forward*d,
        // depth is d - camera_relative.z.
        let depth_requirement = near_plane + camera_relative.z;
        let horizontal_requirement =
            camera_relative.z + camera_relative.x.abs() / (MAX_NDC * horizontal_tangent);
        let vertical_requirement =
            camera_relative.z + camera_relative.y.abs() / (MAX_NDC * vertical_tangent);
        required_distance = required_distance
            .max(depth_requirement)
            .max(horizontal_requirement)
            .max(vertical_requirement);
    }

    if !required_distance.is_finite() || required_distance <= 0.0 {
        return Err(FixedFovFitError::Bounds);
    }
    let distance = required_distance * DISTANCE_SAFETY_SCALE + DISTANCE_EPSILON;
    let translation = target - forward * distance;
    if !distance.is_finite() || !translation.is_finite() {
        return Err(FixedFovFitError::Bounds);
    }

    Ok(FixedFovFit {
        target,
        translation,
        distance,
        vertical_fov: FIXED_VERTICAL_FOV,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(min: Vec3, max: Vec3) -> WorldBounds {
        WorldBounds::new(min, max).expect("test bounds are finite and ordered")
    }

    fn ndc_for_corner(fit: FixedFovFit, corner: Vec3, aspect_ratio: f32) -> (Vec3, f32, f32) {
        let orientation = Quat::IDENTITY;
        let camera_relative = orientation.inverse() * (corner - fit.translation);
        let depth = -camera_relative.z;
        let vertical_tangent = (fit.vertical_fov * 0.5).tan();
        let horizontal_tangent = vertical_tangent * aspect_ratio;
        (
            camera_relative,
            camera_relative.x / (depth * horizontal_tangent),
            camera_relative.y / (depth * vertical_tangent),
        )
    }

    fn assert_all_corners_fit(
        fit: FixedFovFit,
        box_bounds: WorldBounds,
        aspect_ratio: f32,
        near_plane: f32,
    ) {
        for corner in box_bounds.corners() {
            let (camera_relative, ndc_x, ndc_y) = ndc_for_corner(fit, corner, aspect_ratio);
            let depth = -camera_relative.z;
            assert!(depth >= near_plane, "depth={depth}");
            assert!((-MAX_NDC..=MAX_NDC).contains(&ndc_x), "ndc_x={ndc_x}");
            assert!((-MAX_NDC..=MAX_NDC).contains(&ndc_y), "ndc_y={ndc_y}");
        }
    }

    #[test]
    fn fits_square_bounds_in_square_viewport() {
        let box_bounds = bounds(Vec3::new(-1.0, -1.0, -0.5), Vec3::new(1.0, 1.0, 0.5));
        let fit = solve_fixed_fov_fit(box_bounds, Quat::IDENTITY, 1.0, 0.1)
            .expect("square bounds should fit");

        assert_eq!(fit.target, Vec3::ZERO);
        assert!((fit.vertical_fov - FIXED_VERTICAL_FOV).abs() < f32::EPSILON);
        assert_all_corners_fit(fit, box_bounds, 1.0, 0.1);
    }

    #[test]
    fn portrait_viewport_is_limited_by_vertical_extent() {
        let box_bounds = bounds(Vec3::new(-0.25, -2.0, -0.1), Vec3::new(0.25, 2.0, 0.1));
        let fit = solve_fixed_fov_fit(box_bounds, Quat::IDENTITY, 0.5, 0.1)
            .expect("portrait bounds should fit");
        let expected = 2.0 / (MAX_NDC * (FIXED_VERTICAL_FOV * 0.5).tan());

        assert!(fit.distance > expected);
        assert_all_corners_fit(fit, box_bounds, 0.5, 0.1);
    }

    #[test]
    fn landscape_viewport_is_limited_by_horizontal_extent() {
        let box_bounds = bounds(Vec3::new(-4.0, -0.25, -0.1), Vec3::new(4.0, 0.25, 0.1));
        let fit = solve_fixed_fov_fit(box_bounds, Quat::IDENTITY, 16.0 / 9.0, 0.1)
            .expect("landscape bounds should fit");
        let expected = 4.0 / (MAX_NDC * (FIXED_VERTICAL_FOV * 0.5).tan() * (16.0 / 9.0));

        assert!(fit.distance > expected);
        assert_all_corners_fit(fit, box_bounds, 16.0 / 9.0, 0.1);
    }

    #[test]
    fn preserves_world_offset_and_centres_target_on_optical_axis() {
        let box_bounds = bounds(Vec3::new(10.0, 20.0, -3.0), Vec3::new(12.0, 22.0, -1.0));
        let fit = solve_fixed_fov_fit(box_bounds, Quat::IDENTITY, 1.0, 0.2)
            .expect("offset bounds should fit");

        assert_eq!(fit.target, Vec3::new(11.0, 21.0, -2.0));
        assert_eq!(fit.translation.x, fit.target.x);
        assert_eq!(fit.translation.y, fit.target.y);
        assert!(fit.translation.z > fit.target.z);
        assert_all_corners_fit(fit, box_bounds, 1.0, 0.2);
    }

    #[test]
    fn accounts_for_depth_and_near_plane() {
        let box_bounds = bounds(Vec3::new(-0.25, -0.25, 5.0), Vec3::new(0.25, 0.25, 20.0));
        let fit = solve_fixed_fov_fit(box_bounds, Quat::IDENTITY, 1.0, 0.5)
            .expect("deep bounds should fit");

        assert!(fit.distance > 8.0);
        assert_all_corners_fit(fit, box_bounds, 1.0, 0.5);
    }

    #[test]
    fn supports_rotated_camera_orientation() {
        let box_bounds = bounds(Vec3::new(-1.0, -0.5, -0.5), Vec3::new(1.0, 0.5, 0.5));
        let orientation = Quat::from_rotation_y(0.6) * Quat::from_rotation_z(-0.2);
        let fit = solve_fixed_fov_fit(box_bounds, orientation, 1.0, 0.1)
            .expect("rotated camera should fit");
        let camera_position = fit.translation;
        let forward = orientation.normalize() * -Vec3::Z;

        assert!((camera_position - fit.target).dot(forward) < 0.0);
        for corner in box_bounds.corners() {
            let camera_relative = orientation.normalize().inverse() * (corner - camera_position);
            let depth = -camera_relative.z;
            let vertical_tangent = (FIXED_VERTICAL_FOV * 0.5).tan();
            assert!(depth >= 0.1);
            assert!((camera_relative.x / (depth * vertical_tangent)).abs() <= MAX_NDC);
            assert!((camera_relative.y / (depth * vertical_tangent)).abs() <= MAX_NDC);
        }
    }

    #[test]
    fn rejects_invalid_inputs_without_nan_output() {
        let valid = bounds(Vec3::splat(-1.0), Vec3::splat(1.0));
        assert_eq!(
            solve_fixed_fov_fit(valid, Quat::IDENTITY, 0.0, 0.1),
            Err(FixedFovFitError::AspectRatio)
        );
        assert_eq!(
            solve_fixed_fov_fit(valid, Quat::IDENTITY, 1.0, f32::NAN),
            Err(FixedFovFitError::NearPlane)
        );
        assert_eq!(
            solve_fixed_fov_fit(valid, Quat::from_array([f32::NAN; 4]), 1.0, 0.1),
            Err(FixedFovFitError::CameraOrientation)
        );
        assert!(WorldBounds::new(Vec3::splat(f32::NAN), Vec3::ONE).is_none());
        assert!(WorldBounds::new(Vec3::ONE, Vec3::ZERO).is_none());
    }

    #[test]
    fn every_corner_stays_inside_the_ndc_safe_rectangle() {
        let box_bounds = bounds(Vec3::new(-3.0, -2.0, -4.0), Vec3::new(5.0, 4.0, 6.0));
        let fit = solve_fixed_fov_fit(box_bounds, Quat::IDENTITY, 1.7, 0.3)
            .expect("asymmetric depth bounds should fit");

        assert_all_corners_fit(fit, box_bounds, 1.7, 0.3);
    }
}
