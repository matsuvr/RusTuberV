//! Fixed-FOV camera-control state and pure viewport geometry.

use bevy::prelude::*;

use crate::lifecycle::AvatarGeneration;

/// The application's immutable vertical perspective FOV, in radians.
pub use super::fixed_fov_fit::FIXED_VERTICAL_FOV;

const DEFAULT_MIN_DISTANCE: f32 = 0.15;
const DEFAULT_MAX_DISTANCE: f32 = 100.0;
const DEFAULT_DOLLY_LOG_SCALE_PER_SCROLL: f32 = 0.15;
const DEFAULT_ORBIT_RADIANS_PER_PIXEL: f32 = 0.005;
const ORBIT_POLE_EPSILON: f32 = 0.001;
const DISTANCE_EPSILON: f32 = 1e-5;
const FOV_EPSILON: f32 = 1e-6;

/// The result of a camera-control operation could not be produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraControlGeometryError {
    /// One or more vector, quaternion, scalar, or delta inputs were invalid.
    NonFiniteInput,
    /// The camera was at the target or its distance was not positive.
    InvalidDistance,
    /// The viewport dimensions were not finite and positive.
    InvalidViewport,
    /// The requested vertical FOV was not the application's fixed FOV.
    InvalidFieldOfView,
}

/// A bounded interval for perspective camera-target distance, in world units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraDistanceLimits {
    min: f32,
    max: f32,
}

impl CameraDistanceLimits {
    /// Creates validated distance limits.
    pub fn new(min: f32, max: f32) -> Result<Self, CameraControlGeometryError> {
        if !min.is_finite() || !max.is_finite() || min <= 0.0 || min > max {
            return Err(CameraControlGeometryError::NonFiniteInput);
        }
        Ok(Self { min, max })
    }

    /// The minimum allowed camera-target distance.
    #[must_use]
    pub const fn min(self) -> f32 {
        self.min
    }

    /// The maximum allowed camera-target distance.
    #[must_use]
    pub const fn max(self) -> f32 {
        self.max
    }

    fn clamp(self, distance: f32) -> f32 {
        distance.clamp(self.min, self.max)
    }
}

impl Default for CameraDistanceLimits {
    fn default() -> Self {
        Self {
            // The default Bevy near plane is 0.1 world units. Keeping the
            // minimum slightly beyond it prevents dolly from crossing the
            // near-plane safety boundary for the default viewport camera.
            min: DEFAULT_MIN_DISTANCE,
            max: DEFAULT_MAX_DISTANCE,
        }
    }
}

/// Tunable camera-control constants kept out of the input and ECS systems.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraControlConfig {
    /// Bounded camera-target distance used by dolly operations.
    pub distance_limits: CameraDistanceLimits,
    /// Exponential distance scale applied to one normalized scroll unit.
    pub dolly_log_scale_per_scroll: f32,
    /// Orbit radians produced by one input pixel for the later input layer.
    pub orbit_radians_per_pixel: f32,
}

impl Default for CameraControlConfig {
    fn default() -> Self {
        Self {
            distance_limits: CameraDistanceLimits::default(),
            dolly_log_scale_per_scroll: DEFAULT_DOLLY_LOG_SCALE_PER_SCROLL,
            orbit_radians_per_pixel: DEFAULT_ORBIT_RADIANS_PER_PIXEL,
        }
    }
}

/// Camera transform and target state used by orbit, pan, dolly, and reset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraControlPose {
    transform: Transform,
    target: Vec3,
    distance: f32,
}

impl CameraControlPose {
    /// Creates a pose and derives its distance from the transform and target.
    pub fn new(transform: Transform, target: Vec3) -> Result<Self, CameraControlGeometryError> {
        let distance = (transform.translation - target).length();
        Self::from_parts(transform, target, distance)
    }

    /// Creates a pose with an explicit distance from a fixed-FOV framing solve.
    pub(crate) fn from_parts(
        transform: Transform,
        target: Vec3,
        distance: f32,
    ) -> Result<Self, CameraControlGeometryError> {
        if !transform.translation.is_finite()
            || !transform.rotation.is_finite()
            || !transform.scale.is_finite()
            || !target.is_finite()
            || !distance.is_finite()
            || distance <= 0.0
        {
            return Err(CameraControlGeometryError::NonFiniteInput);
        }
        let measured_distance = (transform.translation - target).length();
        if !measured_distance.is_finite()
            || measured_distance <= 0.0
            || (measured_distance - distance).abs() > DISTANCE_EPSILON.max(distance * 1e-5)
        {
            return Err(CameraControlGeometryError::InvalidDistance);
        }
        Ok(Self {
            transform,
            target,
            distance,
        })
    }

    /// The camera transform.
    #[must_use]
    pub const fn transform(self) -> Transform {
        self.transform
    }

    /// The orbit and pan target.
    #[must_use]
    pub const fn target(self) -> Vec3 {
        self.target
    }

    /// The camera-target distance in world units.
    #[must_use]
    pub const fn distance(self) -> f32 {
        self.distance
    }
}

/// Pure fixed-FOV camera-control geometry.
pub mod geometry {
    use super::{
        CameraControlConfig, CameraControlGeometryError, CameraControlPose, FIXED_VERTICAL_FOV,
        FOV_EPSILON, ORBIT_POLE_EPSILON,
    };
    use bevy::prelude::*;

    /// Orbits a camera around a fixed target without changing its distance.
    ///
    /// `yaw_delta` is applied around world `+Y`. Positive `pitch_delta` raises
    /// the camera by decreasing its polar angle. The returned transform uses
    /// world `+Y` as its up direction, so orbit never introduces roll.
    pub fn orbit(
        pose: CameraControlPose,
        yaw_delta: f32,
        pitch_delta: f32,
    ) -> Result<CameraControlPose, CameraControlGeometryError> {
        if !yaw_delta.is_finite() || !pitch_delta.is_finite() {
            return Err(CameraControlGeometryError::NonFiniteInput);
        }
        let offset = pose.transform().translation - pose.target();
        let distance = offset.length();
        if !offset.is_finite() || !distance.is_finite() || distance <= 0.0 {
            return Err(CameraControlGeometryError::InvalidDistance);
        }

        let normalized_y = (offset.y / distance).clamp(-1.0, 1.0);
        let mut polar = normalized_y.acos();
        let mut azimuth = offset.z.atan2(offset.x);
        azimuth += yaw_delta;
        polar = (polar - pitch_delta).clamp(
            ORBIT_POLE_EPSILON,
            std::f32::consts::PI - ORBIT_POLE_EPSILON,
        );

        let sin_polar = polar.sin();
        let next_offset = Vec3::new(
            azimuth.cos() * sin_polar,
            polar.cos(),
            azimuth.sin() * sin_polar,
        ) * distance;
        if !next_offset.is_finite() {
            return Err(CameraControlGeometryError::NonFiniteInput);
        }

        let mut transform = Transform::from_translation(pose.target() + next_offset)
            .looking_at(pose.target(), Vec3::Y);
        transform.scale = pose.transform().scale;
        CameraControlPose::from_parts(transform, pose.target(), pose.distance())
    }

    /// Re-anchors the orbit pivot on `focus` without moving the camera.
    ///
    /// The transform is preserved exactly; only the stored target and its
    /// distance change. The following [`orbit`] call revolves the camera
    /// around `focus` and re-aims it there, so rotation stays centered on
    /// the avatar even after pan or tracking moved it away from the original
    /// framing target. Fails when `focus` is not a finite point in front of
    /// the camera.
    pub fn retarget(
        pose: CameraControlPose,
        focus: Vec3,
    ) -> Result<CameraControlPose, CameraControlGeometryError> {
        if !focus.is_finite() {
            return Err(CameraControlGeometryError::NonFiniteInput);
        }
        let transform = pose.transform();
        let forward = transform.forward();
        let axis_distance = (focus - transform.translation).dot(*forward);
        if !axis_distance.is_finite() || axis_distance <= 0.0 {
            return Err(CameraControlGeometryError::InvalidDistance);
        }
        CameraControlPose::new(transform, focus)
    }

    /// Pans camera and target together in screen space using the fixed FOV.
    ///
    /// `pixel_delta.y` follows window coordinates, where positive is downward;
    /// therefore a downward drag moves the target along camera local up so
    /// the scene follows the mouse as it is grabbed.
    pub fn pan(
        pose: CameraControlPose,
        pixel_delta: Vec2,
        viewport_size: Vec2,
    ) -> Result<CameraControlPose, CameraControlGeometryError> {
        if !pixel_delta.is_finite() {
            return Err(CameraControlGeometryError::NonFiniteInput);
        }
        if !viewport_size.is_finite() || viewport_size.x <= 0.0 || viewport_size.y <= 0.0 {
            return Err(CameraControlGeometryError::InvalidViewport);
        }

        let visible_height = 2.0 * pose.distance() * (FIXED_VERTICAL_FOV * 0.5).tan();
        let aspect_ratio = viewport_size.x / viewport_size.y;
        let world_per_pixel_x = visible_height * aspect_ratio / viewport_size.x;
        let world_per_pixel_y = visible_height / viewport_size.y;
        if !visible_height.is_finite()
            || !aspect_ratio.is_finite()
            || !world_per_pixel_x.is_finite()
            || !world_per_pixel_y.is_finite()
        {
            return Err(CameraControlGeometryError::NonFiniteInput);
        }

        let translation = -pose.transform().right() * (pixel_delta.x * world_per_pixel_x)
            + pose.transform().up() * (pixel_delta.y * world_per_pixel_y);
        if !translation.is_finite() {
            return Err(CameraControlGeometryError::NonFiniteInput);
        }
        let mut transform = pose.transform();
        transform.translation += translation;
        CameraControlPose::from_parts(transform, pose.target() + translation, pose.distance())
    }

    /// Applies multiplicative/exponential dolly to a pose without changing FOV.
    ///
    /// Positive scroll moves toward the target. The logarithmic calculation is
    /// clamped before exponentiation so extreme finite input cannot overflow
    /// into a non-finite camera transform.
    pub fn dolly(
        pose: CameraControlPose,
        normalized_scroll: f32,
        config: CameraControlConfig,
    ) -> Result<CameraControlPose, CameraControlGeometryError> {
        if !normalized_scroll.is_finite()
            || !config.dolly_log_scale_per_scroll.is_finite()
            || config.dolly_log_scale_per_scroll <= 0.0
        {
            return Err(CameraControlGeometryError::NonFiniteInput);
        }

        let limits = config.distance_limits;
        let log_distance =
            pose.distance().ln() - normalized_scroll * config.dolly_log_scale_per_scroll;
        let next_distance = if !log_distance.is_finite() || log_distance <= limits.min().ln() {
            limits.min()
        } else if log_distance >= limits.max().ln() {
            limits.max()
        } else {
            log_distance.exp()
        };
        let next_distance = limits.clamp(next_distance);
        if !next_distance.is_finite() || next_distance <= 0.0 {
            return Err(CameraControlGeometryError::InvalidDistance);
        }

        let radial = (pose.transform().translation - pose.target()).normalize();
        if !radial.is_finite() {
            return Err(CameraControlGeometryError::InvalidDistance);
        }
        let mut transform = pose.transform();
        transform.translation = pose.target() + radial * next_distance;
        CameraControlPose::from_parts(transform, pose.target(), next_distance)
    }

    /// Validates the fixed-FOV read-only value used by the pan helper.
    pub fn validate_fixed_fov(vertical_fov: f32) -> Result<(), CameraControlGeometryError> {
        if !vertical_fov.is_finite() {
            return Err(CameraControlGeometryError::NonFiniteInput);
        }
        if (vertical_fov - FIXED_VERTICAL_FOV).abs() > FOV_EPSILON {
            return Err(CameraControlGeometryError::InvalidFieldOfView);
        }
        Ok(())
    }
}

/// Whether manual camera control has a valid current avatar framing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AvatarCameraControlState {
    /// No valid current generation is available for manual mutation.
    Unavailable,
    /// A current pose and the last successful default pose are available.
    Ready {
        /// Avatar generation that owns both poses.
        generation: AvatarGeneration,
        /// Current manually controlled pose.
        current: CameraControlPose,
        /// Last successful auto-framed pose for this generation.
        default_pose: CameraControlPose,
    },
}

/// Resource owning the generation-scoped camera-control state.
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct AvatarCameraControl {
    state: AvatarCameraControlState,
    config: CameraControlConfig,
}

/// Gate published by the UI layer to distinguish egui-owned pointer input
/// from the main 3D background. The avatar crate remains independent of egui.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CameraPointerInputGate {
    egui_owns_pointer: bool,
}

impl CameraPointerInputGate {
    /// Sets whether egui currently owns the pointer.
    pub fn set_egui_owns_pointer(&mut self, owns_pointer: bool) {
        self.egui_owns_pointer = owns_pointer;
    }

    /// Returns `true` when the viewport may capture camera input.
    #[must_use]
    pub const fn allows_camera_input(&self) -> bool {
        !self.egui_owns_pointer
    }
}

impl Default for AvatarCameraControl {
    fn default() -> Self {
        Self {
            state: AvatarCameraControlState::Unavailable,
            config: CameraControlConfig::default(),
        }
    }
}

impl AvatarCameraControl {
    /// Returns the current lifecycle-scoped state.
    #[must_use]
    pub const fn state(&self) -> AvatarCameraControlState {
        self.state
    }

    /// Returns the immutable configuration used by manual geometry.
    #[must_use]
    pub const fn config(&self) -> CameraControlConfig {
        self.config
    }

    /// Returns the current pose only when it belongs to `generation`.
    #[must_use]
    pub fn current_for(&self, generation: AvatarGeneration) -> Option<CameraControlPose> {
        match self.state {
            AvatarCameraControlState::Ready {
                generation: active,
                current,
                ..
            } if active == generation => Some(current),
            _ => None,
        }
    }

    /// Returns the generation currently eligible for manual camera control.
    #[must_use]
    pub fn active_generation(&self) -> Option<AvatarGeneration> {
        match self.state {
            AvatarCameraControlState::Ready { generation, .. } => Some(generation),
            AvatarCameraControlState::Unavailable => None,
        }
    }

    /// Returns the saved auto-framed pose only when it belongs to `generation`.
    #[must_use]
    pub fn default_for(&self, generation: AvatarGeneration) -> Option<CameraControlPose> {
        match self.state {
            AvatarCameraControlState::Ready {
                generation: active,
                default_pose,
                ..
            } if active == generation => Some(default_pose),
            _ => None,
        }
    }

    /// Clears stale state when the avatar is unloading, loading, or failed.
    pub fn invalidate(&mut self) {
        self.state = AvatarCameraControlState::Unavailable;
    }

    pub(crate) fn initialize(&mut self, generation: AvatarGeneration, pose: CameraControlPose) {
        self.state = AvatarCameraControlState::Ready {
            generation,
            current: pose,
            default_pose: pose,
        };
    }

    /// Replaces the current pose for a matching generation.
    ///
    /// The input layer uses the boolean result to discard stale gestures when
    /// avatar replacement has invalidated their generation.
    pub fn set_current(&mut self, generation: AvatarGeneration, pose: CameraControlPose) -> bool {
        let AvatarCameraControlState::Ready {
            generation: active,
            current,
            ..
        } = &mut self.state
        else {
            return false;
        };
        if *active != generation {
            return false;
        }
        *current = pose;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::prelude::Projection;

    fn pose() -> CameraControlPose {
        let target = Vec3::new(0.0, 1.0, 0.0);
        let transform =
            Transform::from_translation(Vec3::new(0.0, 1.0, 5.0)).looking_at(target, Vec3::Y);
        CameraControlPose::new(transform, target).expect("test pose is valid")
    }

    #[test]
    fn orbit_preserves_target_distance_and_world_up_without_roll() {
        let before = pose();
        let after = geometry::orbit(before, 0.8, -0.3).expect("orbit should be finite");

        assert_eq!(after.target(), before.target());
        assert!((after.distance() - before.distance()).abs() < 1e-5);
        assert!(after.transform().translation.is_finite());
        assert!(after.transform().rotation.is_finite());
        assert!(after.transform().up().y > 0.0);
        assert!(
            after
                .transform()
                .forward()
                .dot((after.target() - after.transform().translation).normalize())
                > 0.999
        );
    }

    #[test]
    fn orbit_clamps_both_poles_and_repeated_operations_stay_finite() {
        let mut current = pose();
        for _ in 0..1000 {
            current = geometry::orbit(current, 0.4, 10.0).expect("orbit stays finite");
        }
        assert!(current.transform().translation.is_finite());
        assert!(current.transform().rotation.is_finite());
        assert!(current.transform().up().y > 0.0);

        current = geometry::orbit(current, -0.4, -10.0).expect("orbit stays finite");
        assert!(current.transform().translation.is_finite());
        assert!(current.transform().up().y > 0.0);
    }

    #[test]
    fn retarget_anchors_the_pivot_on_the_focus_without_moving_the_camera() {
        let before = pose();
        let focus = Vec3::new(0.6, 1.6, -0.5);

        let after = geometry::retarget(before, focus).expect("focus is in front");

        assert_eq!(after.transform(), before.transform());
        assert_eq!(after.target(), focus);
        assert!(
            (after.distance() - before.transform().translation.distance(focus)).abs() < 1e-5
        );

        let orbited = geometry::orbit(after, 0.7, -0.25).expect("orbit after retarget is valid");
        assert_eq!(orbited.target(), focus);
        assert!((orbited.distance() - after.distance()).abs() < 1e-5);
        let to_focus = (focus - orbited.transform().translation).normalize();
        assert!(orbited.transform().forward().dot(to_focus) > 0.999);
    }

    #[test]
    fn retarget_rejects_focus_behind_the_camera_or_degenerate() {
        let before = pose();
        assert_eq!(
            geometry::retarget(before, Vec3::new(0.0, 1.0, 6.0)),
            Err(CameraControlGeometryError::InvalidDistance)
        );
        assert_eq!(
            geometry::retarget(before, Vec3::splat(f32::NAN)),
            Err(CameraControlGeometryError::NonFiniteInput)
        );
        assert_eq!(
            geometry::retarget(before, before.transform().translation),
            Err(CameraControlGeometryError::InvalidDistance)
        );
    }

    #[test]
    fn pan_moves_camera_and_target_together_and_scales_with_distance() {
        let before = pose();
        let after = geometry::pan(before, Vec2::new(100.0, 50.0), Vec2::new(1600.0, 900.0))
            .expect("pan should be finite");
        let delta = after.target() - before.target();

        assert!(delta.x < 0.0, "right drag must move camera/target left");
        assert!(delta.y > 0.0, "down drag must move camera/target up");
        assert_eq!(after.transform().rotation, before.transform().rotation);
        assert!((after.distance() - before.distance()).abs() < 1e-5);
        assert_eq!(
            after.transform().translation - before.transform().translation,
            delta
        );

        let farther_transform = Transform::from_translation(Vec3::new(0.0, 1.0, 10.0))
            .looking_at(before.target(), Vec3::Y);
        let farther = CameraControlPose::new(farther_transform, before.target())
            .expect("farther pose is valid");
        let farther_after =
            geometry::pan(farther, Vec2::new(100.0, -50.0), Vec2::new(1600.0, 900.0))
                .expect("farther pan should be finite");
        assert!((farther_after.target() - farther.target()).length() > delta.length() * 1.9);
    }

    #[test]
    fn pan_uses_standard_grab_direction_for_each_screen_axis() {
        let before = pose();
        let after = geometry::pan(before, Vec2::new(100.0, 100.0), Vec2::new(1600.0, 900.0))
            .expect("pan should be finite");
        let translation = after.target() - before.target();
        let visible_height = 2.0 * before.distance() * (FIXED_VERTICAL_FOV * 0.5).tan();
        let expected = Vec3::new(
            -100.0 * visible_height * (1600.0 / 900.0) / 1600.0,
            100.0 * visible_height / 900.0,
            0.0,
        );

        assert!((translation - expected).length() < 1e-6);
        assert_eq!(
            after.transform().translation - before.transform().translation,
            translation
        );
        assert_eq!(after.transform().rotation, before.transform().rotation);
        assert_eq!(after.distance(), before.distance());
    }

    #[test]
    fn pan_preserves_screen_fraction_when_viewport_scales() {
        let before = pose();
        let small = geometry::pan(before, Vec2::new(80.0, -45.0), Vec2::new(800.0, 450.0))
            .expect("small viewport pan should be finite");
        let large = geometry::pan(before, Vec2::new(160.0, -90.0), Vec2::new(1600.0, 900.0))
            .expect("large viewport pan should be finite");

        assert!((small.target() - large.target()).length() < 1e-5);
    }

    #[test]
    fn pan_rejects_invalid_viewport_without_mutating_pose() {
        let before = pose();
        assert_eq!(
            geometry::pan(before, Vec2::X, Vec2::ZERO),
            Err(CameraControlGeometryError::InvalidViewport)
        );
        assert_eq!(
            geometry::pan(before, Vec2::X, Vec2::new(f32::NAN, 1.0)),
            Err(CameraControlGeometryError::InvalidViewport)
        );
    }

    #[test]
    fn dolly_is_multiplicative_bounded_and_does_not_cross_target() {
        let before = pose();
        let config = CameraControlConfig::default();
        let closer = geometry::dolly(before, 1.0, config).expect("dolly should be finite");
        let farther = geometry::dolly(before, -1.0, config).expect("dolly should be finite");

        assert!(closer.distance() < before.distance());
        assert!(farther.distance() > before.distance());
        assert_eq!(closer.transform().rotation, before.transform().rotation);
        assert_eq!(closer.target(), before.target());
        assert!(closer.distance() >= config.distance_limits.min());
        assert!(farther.distance() <= config.distance_limits.max());
        assert!(
            geometry::dolly(before, f32::MAX, config)
                .expect("large finite input clamps")
                .distance()
                >= config.distance_limits.min()
        );
    }

    #[test]
    fn fixed_fov_is_read_only_and_validated() {
        assert_eq!(geometry::validate_fixed_fov(FIXED_VERTICAL_FOV), Ok(()));
        assert_eq!(
            geometry::validate_fixed_fov(FIXED_VERTICAL_FOV + 0.01),
            Err(CameraControlGeometryError::InvalidFieldOfView)
        );

        let projection = Projection::Perspective(PerspectiveProjection {
            fov: FIXED_VERTICAL_FOV,
            ..default()
        });
        let Projection::Perspective(projection) = projection else {
            unreachable!("constructed perspective projection");
        };
        let before = projection.fov;
        let _ = geometry::orbit(pose(), 0.1, 0.1).expect("orbit should be finite");
        let _ =
            geometry::pan(pose(), Vec2::X, Vec2::new(1600.0, 900.0)).expect("pan should be finite");
        let _ = geometry::dolly(pose(), 1.0, CameraControlConfig::default())
            .expect("dolly should be finite");
        assert_eq!(projection.fov, before);
        assert_eq!(projection.fov, FIXED_VERTICAL_FOV);
    }

    #[test]
    fn lifecycle_state_is_generation_scoped_and_invalidates() {
        let mut control = AvatarCameraControl::default();
        let first = AvatarGeneration(1);
        let second = AvatarGeneration(2);
        let default_pose = pose();

        control.initialize(first, default_pose);
        assert_eq!(control.current_for(first), Some(default_pose));
        assert!(control.current_for(second).is_none());
        assert_eq!(control.default_for(first), Some(default_pose));

        let moved =
            geometry::dolly(default_pose, 1.0, control.config()).expect("test dolly is valid");
        assert!(control.set_current(first, moved));
        assert_eq!(control.current_for(first), Some(moved));
        assert!(!control.set_current(second, default_pose));

        control.invalidate();
        assert_eq!(control.state(), AvatarCameraControlState::Unavailable);
        assert!(control.current_for(first).is_none());
    }
}
