//! Avatar body-scale resolution from immutable rest geometry (Issue #164).
//!
//! Translation soft-cap thresholds are ratios of the avatar's body scale, so
//! the scale must come from a stable authority: the model-authored rest pose,
//! not any current animated transform. This module measures the rest-space
//! hips-to-head extent once at binding time and exposes it as an immutable
//! component.
//!
//! Optional-bone fallbacks are explicit: if the hips bone, its rest data, or
//! the root rest affine is unavailable or degenerate, the documented default
//! body scale is used instead of failing binding.

use bevy::prelude::*;

/// Body scale measured from the avatar's immutable rest geometry, in meters.
///
/// The measured value approximates the hips-to-head extent of the model. It
/// never changes after binding; later systems may read it freely but must not
/// update it from animated transforms.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct BodyScaleMeters {
    /// Avatar generation this measurement belongs to.
    pub generation: crate::lifecycle::AvatarGeneration,
    /// Positive, finite body scale in meters.
    pub scale_meters: f32,
}

/// Default body scale used when rest geometry cannot provide one.
///
/// Typical VRM models place the head joint roughly 0.7 m above the hips. This
/// constant keeps shaping usable for head-only or partially bound models and
/// for degenerate rest data; it is a documented assumption, not a measurement.
pub const DEFAULT_BODY_SCALE_METERS: f32 = 0.7;

/// Inputs needed to measure the body scale from rest geometry.
#[derive(Clone, Copy, Debug)]
pub struct RestGeometryInput {
    /// Rest-global position of the hips bone origin, if available.
    pub hips_position: Option<Vec3>,
    /// Rest-global position of the head bone origin, if available.
    pub head_position: Option<Vec3>,
    /// Uniform scale of the root's rest-global affine, if available.
    ///
    /// Dividing the measured world-space extent by this value yields the
    /// model-space scale so that root scaling does not skew the result.
    pub root_scale: Option<f32>,
}

/// Measures the body scale from rest geometry, falling back to
/// [`DEFAULT_BODY_SCALE_METERS`] whenever the input is missing, non-finite,
/// or degenerate.
#[must_use]
pub fn resolve_body_scale(input: &RestGeometryInput) -> f32 {
    let (Some(hips), Some(head)) = (input.hips_position, input.head_position) else {
        return DEFAULT_BODY_SCALE_METERS;
    };
    let extent = head.distance(hips);
    let scale = match input.root_scale {
        Some(root_scale) if root_scale.is_finite() && root_scale > f32::EPSILON => {
            extent / root_scale
        }
        _ => extent,
    };
    if scale.is_finite() && scale > f32::EPSILON {
        scale
    } else {
        DEFAULT_BODY_SCALE_METERS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measures_hips_to_head_extent_in_model_space() {
        // Root scaled 2x uniformly: world extent 1.4 m -> model space 0.7 m.
        let input = RestGeometryInput {
            hips_position: Some(Vec3::new(0.0, 1.0, 0.0)),
            head_position: Some(Vec3::new(0.0, 2.4, 0.0)),
            root_scale: Some(2.0),
        };
        let scale = resolve_body_scale(&input);
        assert!((scale - 0.7).abs() < 1e-5);
    }

    #[test]
    fn works_without_root_scale_information() {
        let input = RestGeometryInput {
            hips_position: Some(Vec3::ZERO),
            head_position: Some(Vec3::new(0.3, 0.4, 0.0)),
            root_scale: None,
        };
        assert!((resolve_body_scale(&input) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn falls_back_when_bones_or_rest_data_are_missing() {
        let base = RestGeometryInput {
            hips_position: None,
            head_position: Some(Vec3::ONE),
            root_scale: None,
        };
        assert_eq!(resolve_body_scale(&base), DEFAULT_BODY_SCALE_METERS);

        let no_head = RestGeometryInput {
            head_position: None,
            ..base
        };
        assert_eq!(resolve_body_scale(&no_head), DEFAULT_BODY_SCALE_METERS);

        let no_root = RestGeometryInput {
            hips_position: Some(Vec3::ZERO),
            head_position: Some(Vec3::Y * 0.7),
            root_scale: None,
        };
        assert!((resolve_body_scale(&no_root) - 0.7).abs() < 1e-5);
    }

    #[test]
    fn falls_back_on_degenerate_or_non_finite_geometry() {
        // Coincident bones produce zero extent -> fallback.
        let coincident = RestGeometryInput {
            hips_position: Some(Vec3::ONE),
            head_position: Some(Vec3::ONE),
            root_scale: None,
        };
        assert_eq!(resolve_body_scale(&coincident), DEFAULT_BODY_SCALE_METERS);

        // Non-finite positions -> fallback.
        let non_finite = RestGeometryInput {
            hips_position: Some(Vec3::ZERO),
            head_position: Some(Vec3::new(f32::NAN, 2.0, 0.0)),
            root_scale: None,
        };
        assert_eq!(resolve_body_scale(&non_finite), DEFAULT_BODY_SCALE_METERS);

        // Degenerate root scale -> extent used unguarded against NaN only via
        // the final finiteness check; a zero root scale must not divide.
        let zero_root = RestGeometryInput {
            hips_position: Some(Vec3::ZERO),
            head_position: Some(Vec3::Y * 0.7),
            root_scale: Some(0.0),
        };
        assert!(resolve_body_scale(&zero_root).is_finite());
    }
}
