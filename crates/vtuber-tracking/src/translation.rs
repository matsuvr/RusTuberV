//! Neutral-relative head translation from ordinary webcam face geometry.
//!
//! This module turns the geometry already produced by the Direct head-pose
//! estimators into the engine-neutral [`HeadTranslationSignal`] contract from
//! `DESIGN.md` §11.8 / Issue #163. It never re-solves a second pose problem:
//!
//! * The Kabsch landmark path reuses the rotation from
//!   [`crate::pose::solve_relative_pose`] together with the two weighted
//!   centroids. Translation is the residual `c_current - R * c_neutral`,
//!   so a pure rigid rotation produces no cross-talk by construction.
//! * The MediaPipe face-transform path reuses the relative translation of
//!   [`RelativeFaceTransform`](crate::pose::mediapipe::RelativeFaceTransform).
//!
//! Monocular webcams do not provide verified absolute distance. X/Y are
//! therefore defined as neutral-relative face-centre displacement scaled by
//! an assumed head size, and Z is signed scale evidence mapped through an
//! assumed reference webcam distance. Both anchors are explicit typed
//! constants below: outputs are approximate meters under these documented
//! assumptions, not measured absolute metric distances.
//!
//! All conversions are deterministic and defined in this one place.

use crate::pose::PoseAlignment;
use vtuber_core::types::{HeadTranslationSignal, HeadTranslationState};

/// Reference physical head half-extent used to convert normalized-image
/// displacements into approximate meters.
///
/// A typical adult face spans roughly 0.15 m between cheek boundaries; the
/// weighted landmark cloud used for pose solving covers approximately half of
/// that extent around its centre.
pub const REFERENCE_HEAD_RADIUS_METERS: f32 = 0.075;

/// Assumed camera-to-face distance at calibration time, in meters.
///
/// Z is derived from face-scale evidence (`current_size / neutral_size`), which
/// under a pinhole model measures relative distance change. This constant maps
/// that ratio onto meters and must stay a documented assumption rather than a
/// claimed measurement.
pub const REFERENCE_DISTANCE_METERS: f32 = 0.6;

/// MediaPipe face-transform camera-space units are centimeters per the
/// upstream Face Geometry convention; convert to meters.
pub const MEDIAPIPE_TRANSFORM_UNITS_TO_METERS: f32 = 0.01;

/// Projected neutral radius below which the scale evidence is degenerate.
const MIN_NEUTRAL_RADIUS: f32 = 1.0e-5;

/// Mean landmark presence below which a translation estimate is degraded.
const DEGRADED_VISIBILITY_THRESHOLD: f32 = 0.6;

/// Converts the Kabsch alignment evidence into a translation signal.
///
/// `mean_visibility` is the mean landmark visibility/presence of the current
/// observation; it selects tracked versus degraded state. Non-finite or
/// degenerate inputs produce [`HeadTranslationSignal::UNAVAILABLE`]; NaN/Inf
/// are never published.
#[must_use]
pub fn signal_from_alignment(
    alignment: &PoseAlignment,
    mean_visibility: f32,
) -> HeadTranslationSignal {
    let state = availability(mean_visibility);
    let neutral_radius = alignment.neutral_projected_radius;
    if !neutral_radius.is_finite()
        || !current_radius_is_finite(alignment)
        || neutral_radius < MIN_NEUTRAL_RADIUS
    {
        return HeadTranslationSignal::UNAVAILABLE;
    }

    // Rotation-compensated residual displacement in canonical units.
    let [tx, ty, tz] = alignment.translation;
    if !tx.is_finite() || !ty.is_finite() || !tz.is_finite() {
        return HeadTranslationSignal::UNAVAILABLE;
    }

    // Canonical x grows toward image right; canonical y grows toward the
    // image bottom (top = 0), so up-positive Y negates it.
    let scale = REFERENCE_HEAD_RADIUS_METERS / neutral_radius;
    let x_meters = tx * scale;
    let y_meters = -ty * scale;

    // Pinhole evidence: apparent size shrinks as distance grows, so
    // neutral/current > 1 means the user moved away (positive Z).
    let z_ratio = neutral_radius / alignment.current_projected_radius;
    let z_meters = REFERENCE_DISTANCE_METERS * (z_ratio - 1.0);

    build(x_meters, y_meters, z_meters, state)
}

/// Converts a MediaPipe-relative face transform translation into meters.
///
/// `relative_translation_xyz` is expressed in the calibrated neutral face
/// basis in MediaPipe camera-space units: X toward unmirrored image right,
/// Y up, Z away from the camera. The unit assumption is documented in
/// [`MEDIAPIPE_TRANSFORM_UNITS_TO_METERS`].
#[must_use]
pub fn signal_from_face_transform(
    relative_translation_xyz: [f32; 3],
    mean_visibility: f32,
) -> HeadTranslationSignal {
    let state = availability(mean_visibility);
    let [tx, ty, tz] = relative_translation_xyz;
    if !tx.is_finite() || !ty.is_finite() || !tz.is_finite() {
        return HeadTranslationSignal::UNAVAILABLE;
    }
    let k = MEDIAPIPE_TRANSFORM_UNITS_TO_METERS;
    build(tx * k, ty * k, tz * k, state)
}

fn current_radius_is_finite(alignment: &PoseAlignment) -> bool {
    alignment.current_projected_radius.is_finite()
        && alignment.current_projected_radius >= MIN_NEUTRAL_RADIUS
}

fn availability(mean_visibility: f32) -> HeadTranslationState {
    if mean_visibility < DEGRADED_VISIBILITY_THRESHOLD {
        HeadTranslationState::Degraded
    } else {
        HeadTranslationState::Tracked
    }
}

/// Builds the final signal without ever publishing non-finite values.
#[must_use]
fn build(
    x_meters: f32,
    y_meters: f32,
    z_meters: f32,
    state: HeadTranslationState,
) -> HeadTranslationSignal {
    let signal = match state {
        HeadTranslationState::Tracked => {
            HeadTranslationSignal::tracked(x_meters, y_meters, z_meters)
        }
        HeadTranslationState::Degraded | HeadTranslationState::Unavailable => {
            HeadTranslationSignal::degraded(x_meters, y_meters, z_meters)
        }
    };
    if signal.is_available() {
        signal
    } else {
        HeadTranslationSignal::UNAVAILABLE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::UnitQuaternion;

    use crate::pose::{LandmarkSet, solve_relative_pose};

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1.0e-5,
            "expected {expected}, got {actual}"
        );
    }

    /// Builds a synthetic face cloud: a flat plus-shaped pattern with depth.
    fn synthetic_cloud(center: [f32; 3], scale: f32) -> LandmarkSet {
        let mut set = LandmarkSet::new();
        let offsets = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.1],
            [-1.0, 0.0, 0.1],
            [0.0, 1.0, -0.05],
            [0.0, -1.0, -0.05],
            [0.5, 0.5, 0.0],
            [-0.5, -0.5, 0.0],
        ];
        for offset in offsets {
            set.push(
                [
                    center[0] + offset[0] * scale,
                    center[1] + offset[1] * scale,
                    center[2] + offset[2] * scale,
                ],
                1.0,
            );
        }
        set
    }

    #[test]
    fn aligned_clouds_produce_zero_translation() {
        let neutral = synthetic_cloud([0.5, 0.4, 0.0], 0.02);
        let alignment = solve_relative_pose(&neutral, &synthetic_cloud([0.5, 0.4, 0.0], 0.02))
            .expect("aligned clouds should solve");
        let signal = signal_from_alignment(&alignment, 1.0);

        assert_eq!(signal.state, HeadTranslationState::Tracked);
        assert_close(signal.x_meters, 0.0);
        assert_close(signal.y_meters, 0.0);
        assert_close(signal.z_meters, 0.0);
    }

    #[test]
    fn left_right_up_down_displacements_follow_semantic_signs() {
        let neutral = synthetic_cloud([0.5, 0.4, 0.0], 0.02);

        // Move right in the image: canonical x increases -> positive X meters.
        let right = solve_relative_pose(&neutral, &synthetic_cloud([0.55, 0.4, 0.0], 0.02))
            .expect("translated cloud should solve");
        let signal = signal_from_alignment(&right, 1.0);
        assert!(signal.x_meters > 0.01, "image-right motion is positive X");
        assert!(signal.y_meters.abs() < 1.0e-4);
        assert!(signal.z_meters.abs() < 1.0e-4);

        // Move left: negative X.
        let left = solve_relative_pose(&neutral, &synthetic_cloud([0.45, 0.4, 0.0], 0.02))
            .expect("translated cloud should solve");
        assert!(signal_from_alignment(&left, 1.0).x_meters < -0.01);

        // Image y decreases upward, so moving up yields positive Y meters.
        let up = solve_relative_pose(&neutral, &synthetic_cloud([0.5, 0.35, 0.0], 0.02))
            .expect("translated cloud should solve");
        assert!(signal_from_alignment(&up, 1.0).y_meters > 0.01);

        // Moving down yields negative Y meters.
        let down = solve_relative_pose(&neutral, &synthetic_cloud([0.5, 0.45, 0.0], 0.02))
            .expect("translated cloud should solve");
        assert!(signal_from_alignment(&down, 1.0).y_meters < -0.01);
    }

    #[test]
    fn near_far_motion_produces_signed_z_scale_evidence() {
        let neutral = synthetic_cloud([0.5, 0.4, 0.0], 0.02);

        // Approach: the face appears larger -> negative Z (toward camera).
        let near = solve_relative_pose(&neutral, &synthetic_cloud([0.5, 0.4, 0.0], 0.024))
            .expect("scaled cloud should solve");
        let near_signal = signal_from_alignment(&near, 1.0);
        assert!(near_signal.z_meters < -0.01, "approach must be negative Z");

        // Retreat: smaller apparent size -> positive Z (away from camera).
        let far = solve_relative_pose(&neutral, &synthetic_cloud([0.5, 0.4, 0.0], 0.016))
            .expect("scaled cloud should solve");
        let far_signal = signal_from_alignment(&far, 1.0);
        assert!(far_signal.z_meters > 0.01, "retreat must be positive Z");

        // Mirror flips only X; near/far Z evidence is mirror-invariant.
        assert_relative_eq!(
            near_signal.mirrored().z_meters,
            near_signal.z_meters,
            epsilon = 1.0e-6
        );
        assert_relative_eq!(
            near_signal.mirrored().y_meters,
            near_signal.y_meters,
            epsilon = 1.0e-6
        );
    }

    #[test]
    fn pure_rotation_produces_bounded_cross_talk() {
        // Rotate the neutral cloud rigidly about its own weighted centroid.
        // Because the translation residual is c_current - R*c_neutral, a pure
        // rotation about the centroid yields no cross-talk by construction;
        // rotating about an arbitrary pivot would be a real translation.
        let neutral = synthetic_cloud([0.5, 0.4, 0.0], 0.02);
        let pivot = neutral.centroid().expect("synthetic cloud has weight");
        let rotation = UnitQuaternion::from_euler_angles(0.15, -0.25, 0.10);

        let mut rotated = LandmarkSet::new();
        for point in &neutral.points {
            let v = rotation
                * nalgebra::Vector3::new(
                    point.position[0] - pivot[0],
                    point.position[1] - pivot[1],
                    point.position[2] - pivot[2],
                );
            rotated.push(
                [v.x + pivot[0], v.y + pivot[1], v.z + pivot[2]],
                point.weight,
            );
        }

        let alignment =
            solve_relative_pose(&neutral, &rotated).expect("rotated cloud should solve");
        // Confirm this fixture really exercised rotation, not identity.
        let solved_yaw = alignment.pose.yaw_rad;
        assert!(
            solved_yaw.abs() > 0.05,
            "fixture yaw {solved_yaw} too small"
        );

        let signal = signal_from_alignment(&alignment, 1.0);
        assert!(signal.is_available());
        // X/Y: a pure head rotation about its own centre does not move the
        // centroid, so first-order cross-talk is zero (float noise only).
        assert!(
            signal.x_meters.abs() < 1.0e-3 && signal.y_meters.abs() < 1.0e-3,
            "rotation-only X/Y cross-talk must stay bounded: {signal:?}"
        );
        // Z: rotation foreshortens the projected face size, so the scale
        // evidence picks up a bounded second-order error. For a rotation of
        // angle theta this stays below REFERENCE_DISTANCE * (1 - cos theta);
        // with the fixture's ~0.31 rad combined tilt that is < 0.03 m.
        assert!(
            signal.z_meters.abs() < 0.03,
            "rotation-only Z cross-talk must stay bounded: {signal:?}"
        );
    }

    #[test]
    fn degenerate_neutral_scale_is_unavailable_and_nan_never_published() {
        let neutral = synthetic_cloud([0.5, 0.4, 0.0], 0.02);
        let alignment = solve_relative_pose(&neutral, &neutral.clone()).unwrap();
        // Simulate a degenerate projected neutral radius directly on the
        // evidence carried by the alignment.
        let degenerate = PoseAlignment {
            neutral_projected_radius: 0.0,
            ..alignment
        };
        assert_eq!(
            signal_from_alignment(&degenerate, 1.0),
            HeadTranslationSignal::UNAVAILABLE
        );

        let nan_alignment = PoseAlignment {
            translation: [f32::NAN, 0.0, 0.0],
            ..alignment
        };
        assert_eq!(
            signal_from_alignment(&nan_alignment, 1.0),
            HeadTranslationSignal::UNAVAILABLE
        );
    }

    #[test]
    fn low_visibility_degrades_state_without_changing_values() {
        let neutral = synthetic_cloud([0.5, 0.4, 0.0], 0.02);
        let alignment =
            solve_relative_pose(&neutral, &synthetic_cloud([0.55, 0.4, 0.0], 0.02)).unwrap();

        let tracked = signal_from_alignment(&alignment, 1.0);
        let degraded = signal_from_alignment(&alignment, 0.2);
        assert_eq!(tracked.state, HeadTranslationState::Tracked);
        assert_eq!(degraded.state, HeadTranslationState::Degraded);
        assert_close(degraded.x_meters, tracked.x_meters);
        assert_close(degraded.y_meters, tracked.y_meters);
        assert_close(degraded.z_meters, tracked.z_meters);
    }

    #[test]
    fn face_transform_signal_converts_units_deterministically() {
        let signal = signal_from_face_transform([12.0, -8.0, 25.0], 1.0);
        assert_eq!(signal.state, HeadTranslationState::Tracked);
        assert_close(signal.x_meters, 0.12);
        assert_close(signal.y_meters, -0.08);
        assert_close(signal.z_meters, 0.25);

        // Mirror flips only X (semantic contract check at this adapter).
        let mirrored = signal.mirrored();
        assert_close(mirrored.x_meters, -0.12);
        assert_close(mirrored.y_meters, -0.08);
        assert_close(mirrored.z_meters, 0.25);

        assert_eq!(
            signal_from_face_transform([f32::INFINITY, 0.0, 0.0], 1.0),
            HeadTranslationSignal::UNAVAILABLE
        );
    }
}
