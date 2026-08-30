//! Generation-scoped reset of the main avatar viewport camera.

use bevy::prelude::*;

use super::AvatarViewportCamera;
use super::camera_control::AvatarCameraControl;
use super::camera_input::CameraPointerGesture;
use crate::lifecycle::{AvatarGeneration, AvatarLifecycle, AvatarLifecycleState};

/// One-shot request to restore a generation's saved auto-framed camera pose.
#[derive(Message, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResetCameraRequest {
    /// Generation observed when the UI action was processed.
    pub generation: AvatarGeneration,
}

/// System set that applies reset after pointer input, before transform
/// propagation. Applying it after input makes a same-frame stale mouse delta
/// unable to move the camera away from the restored pose.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CameraResetSet;

/// Applies a reset request only to the matching current generation.
pub(crate) fn reset_avatar_camera(
    mut requests: MessageReader<ResetCameraRequest>,
    lifecycle: Res<AvatarLifecycle>,
    mut camera_control: ResMut<AvatarCameraControl>,
    mut gesture: ResMut<CameraPointerGesture>,
    mut cameras: Query<&mut Transform, With<AvatarViewportCamera>>,
) {
    let Some(request) = requests.read().next().copied() else {
        return;
    };
    if lifecycle.state() != AvatarLifecycleState::Ready
        || lifecycle.current_generation() != request.generation
    {
        return;
    }
    let Some(default_pose) = camera_control.default_for(request.generation) else {
        return;
    };
    let Some(mut transform) = cameras.iter_mut().next() else {
        return;
    };
    if !camera_control.set_current(request.generation, default_pose) {
        return;
    }
    *transform = default_pose.transform();
    *gesture = CameraPointerGesture::None;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framing::camera_control::{CameraControlConfig, CameraControlPose, geometry};

    fn pose() -> CameraControlPose {
        let target = Vec3::new(0.0, 1.0, 0.0);
        let transform =
            Transform::from_translation(Vec3::new(0.0, 1.0, 5.0)).looking_at(target, Vec3::Y);
        CameraControlPose::new(transform, target).expect("test pose is valid")
    }

    fn ready_app() -> (App, Entity, AvatarGeneration) {
        let mut app = App::new();
        app.init_resource::<AvatarLifecycle>()
            .init_resource::<AvatarCameraControl>()
            .init_resource::<CameraPointerGesture>()
            .add_message::<ResetCameraRequest>()
            .add_systems(Update, reset_avatar_camera);

        let camera_transform = Transform::from_translation(Vec3::new(0.0, 1.0, 5.0))
            .looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y);
        let camera = app
            .world_mut()
            .spawn((
                Camera3d::default(),
                super::super::AvatarViewportCamera::from_default_transform(camera_transform),
                camera_transform,
            ))
            .id();
        let root = app.world_mut().spawn_empty().id();
        let generation = {
            let mut lifecycle = app.world_mut().resource_mut::<AvatarLifecycle>();
            lifecycle.request_load(root).expect("test load is valid");
            lifecycle.start_binding(root);
            lifecycle.finish_ready();
            lifecycle.current_generation()
        };
        app.world_mut()
            .resource_mut::<AvatarCameraControl>()
            .initialize(generation, pose());
        (app, camera, generation)
    }

    fn send_reset(app: &mut App, generation: AvatarGeneration) {
        app.world_mut()
            .resource_mut::<Messages<ResetCameraRequest>>()
            .write(ResetCameraRequest { generation });
    }

    #[test]
    fn combined_manual_pose_restores_exact_default_and_clears_gesture() {
        let (mut app, camera, generation) = ready_app();
        let default = pose();
        let combined = geometry::orbit(default, 0.4, -0.2)
            .and_then(|pose| geometry::pan(pose, Vec2::new(100.0, -40.0), Vec2::new(1600.0, 900.0)))
            .and_then(|pose| geometry::dolly(pose, 2.0, CameraControlConfig::default()))
            .expect("combined pose is valid");
        *app.world_mut()
            .get_mut::<Transform>(camera)
            .expect("camera transform") = combined.transform();
        assert!(
            app.world_mut()
                .resource_mut::<AvatarCameraControl>()
                .set_current(generation, combined)
        );
        *app.world_mut().resource_mut::<CameraPointerGesture>() =
            CameraPointerGesture::Orbit { generation };

        send_reset(&mut app, generation);
        app.update();

        let control = app.world().resource::<AvatarCameraControl>();
        assert_eq!(control.current_for(generation), Some(default));
        assert_eq!(control.default_for(generation), Some(default));
        assert_eq!(
            *app.world()
                .get::<Transform>(camera)
                .expect("camera transform"),
            default.transform()
        );
        assert_eq!(
            *app.world().resource::<CameraPointerGesture>(),
            CameraPointerGesture::None
        );
    }

    #[test]
    fn repeated_reset_is_idempotent_and_does_not_change_fov() {
        let (mut app, camera, generation) = ready_app();
        let default_pose = pose();
        let projection = PerspectiveProjection {
            fov: super::super::camera_control::FIXED_VERTICAL_FOV,
            ..default()
        };
        app.world_mut()
            .entity_mut(camera)
            .insert(Projection::Perspective(projection));

        send_reset(&mut app, generation);
        app.update();
        send_reset(&mut app, generation);
        app.update();

        assert_eq!(
            app.world()
                .resource::<AvatarCameraControl>()
                .current_for(generation),
            Some(default_pose)
        );
        let Projection::Perspective(projection) = app.world().get::<Projection>(camera).unwrap()
        else {
            panic!("test camera projection should be perspective");
        };
        assert_eq!(
            projection.fov,
            super::super::camera_control::FIXED_VERTICAL_FOV
        );
    }

    #[test]
    fn stale_generation_request_is_a_no_op() {
        let (mut app, camera, generation) = ready_app();
        let before = *app
            .world()
            .get::<Transform>(camera)
            .expect("camera transform");
        send_reset(&mut app, AvatarGeneration(generation.0.saturating_sub(1)));
        app.update();

        assert_eq!(
            *app.world()
                .get::<Transform>(camera)
                .expect("camera transform"),
            before
        );
        assert_eq!(
            app.world()
                .resource::<AvatarCameraControl>()
                .current_for(generation),
            Some(pose())
        );
    }

    #[test]
    fn non_ready_reset_is_a_no_op() {
        let (mut app, camera, generation) = ready_app();
        let before = *app
            .world()
            .get::<Transform>(camera)
            .expect("camera transform");
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .fail(crate::lifecycle::AvatarLifecycleFailure::AssetLoadFailed);
        send_reset(&mut app, generation);
        app.update();

        assert_eq!(
            *app.world()
                .get::<Transform>(camera)
                .expect("camera transform"),
            before
        );
    }
}
