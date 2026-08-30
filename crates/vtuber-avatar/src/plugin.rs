//! `VtuberAvatarPlugin` and top-level Bevy system registration.
//!
//! This is the only plugin that wires `bevy_vrm1` systems together with the
//! VTuber lifecycle domain. `bevy_vrm1` types are used internally and are not
//! re-exported from the crate facade.

use bevy::app::AnimationSystems;
use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use bevy_vrm1::prelude::*;
use bevy_vrm1::vrm::body_tracking::apply_direct_body_tracking;

use crate::arm_pose::ArmPoseOverrideStore;
use crate::arm_pose::apply_default_arm_pose;
use crate::bind::observe_initialized;
use crate::binding::bind_humanoid_bones;
use crate::body_motion::{
    LossIdleState, PositionInputMetrics, reset_position_metrics_on_lifecycle_change,
    update_body_tracking_position_input,
};
use crate::expression::apply_tracked_expressions;
use crate::framing::camera_control::AvatarCameraControl;
use crate::framing::camera_control::CameraPointerInputGate;
use crate::framing::camera_input::{
    CameraInputSet, CameraPointerGesture, apply_camera_pointer_input,
};
use crate::framing::camera_reset::{CameraResetSet, ResetCameraRequest, reset_avatar_camera};
use crate::framing::fixed_fov_fit::FIXED_VERTICAL_FOV;
use crate::framing::{AvatarViewportCamera, frame_avatar_camera};
use crate::gaze::update_direct_look_at_input;
use crate::lifecycle::{
    AvatarLifecycle, LoadAvatarRequest, LoadAvatarResult, ReplaceAvatarRequest,
    ReplaceAvatarResult, UnloadAvatarRequest, UnloadAvatarResult, apply_avatar_request_events,
};
use crate::load::{
    LoadImportedAvatarRequest, LoadImportedAvatarResult, handle_load_imported_avatar_requests,
};
use crate::mirror::AvatarMotionMirror;
use crate::pose::{
    PoseApplyMetrics, reset_pose_metrics_on_lifecycle_change, update_body_tracking_pose_input,
};
use crate::render_output::{
    AVATAR_RENDER_LAYER, VIEWPORT_ONLY_RENDER_LAYER, register_output_systems,
};
use crate::unload::{
    ActiveControlFrame, clear_control_cache_on_lifecycle_change, despawn_unloading_avatar,
};

/// Plugin that sets up the VRM avatar scene, lifecycle, and diagnostics.
#[derive(Default)]
pub struct VtuberAvatarPlugin;

impl Plugin for VtuberAvatarPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(VrmPlugin)
            .add_plugins(crate::compatibility::VrmCompatibilityPlugin)
            .init_resource::<AvatarLifecycle>()
            .init_resource::<AvatarCameraControl>()
            .init_resource::<CameraPointerInputGate>()
            .init_resource::<CameraPointerGesture>()
            .init_resource::<ArmPoseOverrideStore>()
            .init_resource::<crate::arm_pipeline::ArmSourceSelection>()
            .add_message::<crate::arm_pose::ArmPoseProfileChange>()
            .init_resource::<ActiveControlFrame>()
            .init_resource::<AvatarMotionMirror>()
            .init_resource::<PoseApplyMetrics>()
            .init_resource::<PositionInputMetrics>()
            .init_resource::<LossIdleState>()
            .add_message::<LoadAvatarRequest>()
            .add_message::<LoadAvatarResult>()
            .add_message::<UnloadAvatarRequest>()
            .add_message::<UnloadAvatarResult>()
            .add_message::<ReplaceAvatarRequest>()
            .add_message::<ReplaceAvatarResult>()
            .add_message::<LoadImportedAvatarRequest>()
            .add_message::<LoadImportedAvatarResult>()
            .add_message::<ResetCameraRequest>()
            .add_systems(Startup, setup_scene)
            .add_systems(
                Update,
                (
                    handle_load_imported_avatar_requests,
                    apply_avatar_request_events,
                    despawn_unloading_avatar,
                    observe_initialized,
                    bind_humanoid_bones,
                )
                    .chain(),
            )
            .add_systems(Update, clear_control_cache_on_lifecycle_change)
            .add_systems(Update, log_loaded_vrm)
            .add_systems(Update, log_head_bone)
            .configure_sets(
                PostUpdate,
                CameraInputSet.before(TransformSystems::Propagate),
            )
            .configure_sets(
                PostUpdate,
                CameraResetSet
                    .after(CameraInputSet)
                    .before(TransformSystems::Propagate),
            )
            .add_systems(
                PostUpdate,
                apply_camera_pointer_input.in_set(CameraInputSet),
            )
            .add_systems(PostUpdate, reset_avatar_camera.in_set(CameraResetSet))
            .add_systems(
                PostUpdate,
                frame_avatar_camera.after(TransformSystems::Propagate),
            )
            .add_systems(
                PostUpdate,
                update_body_tracking_pose_input
                    .after(AnimationSystems)
                    .before(apply_direct_body_tracking)
                    .before(VrmSystemSets::Constraints),
            )
            .add_systems(
                PostUpdate,
                update_body_tracking_position_input
                    .after(AnimationSystems)
                    .before(update_body_tracking_pose_input)
                    .before(apply_direct_body_tracking)
                    .before(VrmSystemSets::Constraints),
            )
            .add_systems(
                PostUpdate,
                crate::arm_pipeline::update_dynamic_arm_targets
                    .after(update_body_tracking_position_input)
                    .before(update_body_tracking_pose_input)
                    .before(apply_default_arm_pose),
            )
            .add_systems(
                PostUpdate,
                apply_default_arm_pose
                    .after(apply_direct_body_tracking)
                    .before(update_direct_look_at_input)
                    .before(VrmSystemSets::GazeControl)
                    .before(VrmSystemSets::Constraints),
            )
            .add_systems(
                PostUpdate,
                update_direct_look_at_input
                    .after(apply_direct_body_tracking)
                    .after(apply_default_arm_pose)
                    .before(VrmSystemSets::GazeControl),
            )
            .add_systems(
                PostUpdate,
                apply_tracked_expressions
                    .after(VrmSystemSets::GazeControl)
                    .before(VrmSystemSets::Expressions),
            )
            .add_systems(Update, reset_pose_metrics_on_lifecycle_change)
            .add_systems(Update, reset_position_metrics_on_lifecycle_change);
        register_output_systems(app);
    }
}

/// Command-line / environment path to the VRM model to load.
///
/// This resource is retained for backwards compatibility with the desktop
/// entry point. Startup model loading will be migrated to the lifecycle
/// request flow in a later subtask.
#[derive(Resource, Debug, Clone, Default)]
pub struct StartupModelPath(pub Option<String>);

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Ground plane for visual reference.
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(5.0, 5.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.3, 0.35),
            ..default()
        })),
        Transform::from_translation(Vec3::new(0.0, -1.0, 0.0)),
        RenderLayers::layer(VIEWPORT_ONLY_RENDER_LAYER),
    ));

    // Key light.
    commands.spawn((
        DirectionalLight {
            illuminance: 1500.0,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.5, 0.5, 0.0)),
        RenderLayers::from_layers(&[AVATAR_RENDER_LAYER, VIEWPORT_ONLY_RENDER_LAYER]),
    ));

    // Camera framing the upper body.
    let camera_transform = Transform::from_translation(Vec3::new(0.0, 0.0, 2.5))
        .looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y);
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: FIXED_VERTICAL_FOV,
            ..default()
        }),
        AvatarViewportCamera::from_default_transform(camera_transform),
        camera_transform,
        RenderLayers::from_layers(&[AVATAR_RENDER_LAYER, VIEWPORT_ONLY_RENDER_LAYER]),
    ));
}

fn log_loaded_vrm(vrms: Query<Entity, Added<Vrm>>) {
    for entity in vrms.iter() {
        info!("VRM runtime attached to root: {entity:?}");
    }
}

fn log_head_bone(heads: Query<Entity, Added<HeadBoneEntity>>) {
    for entity in heads.iter() {
        info!("Head bone capability found: {:?}", entity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::camera::RenderTarget;

    #[test]
    fn setup_scene_keeps_ground_off_the_output_layer() {
        let mut app = App::new();
        app.init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .add_systems(Startup, setup_scene);
        app.update();

        let mut ground = app.world_mut().query::<(&Mesh3d, &RenderLayers)>();
        let ground_layers: Vec<_> = ground
            .iter(app.world())
            .map(|(_, layers)| layers.clone())
            .collect();
        assert_eq!(ground_layers.len(), 1);
        assert_eq!(
            ground_layers[0],
            RenderLayers::layer(VIEWPORT_ONLY_RENDER_LAYER)
        );
        assert!(!ground_layers[0].intersects(&RenderLayers::layer(AVATAR_RENDER_LAYER)));

        let mut cameras = app
            .world_mut()
            .query::<(&crate::framing::AvatarViewportCamera, &RenderLayers)>();
        let viewport_layers = cameras
            .iter(app.world())
            .next()
            .expect("viewport camera")
            .1
            .clone();
        assert!(viewport_layers.intersects(&RenderLayers::layer(AVATAR_RENDER_LAYER)));
        assert!(viewport_layers.intersects(&RenderLayers::layer(VIEWPORT_ONLY_RENDER_LAYER)));

        let mut lights = app
            .world_mut()
            .query::<(&DirectionalLight, &RenderLayers)>();
        let light_layers = lights
            .iter(app.world())
            .next()
            .expect("key light")
            .1
            .clone();
        assert!(light_layers.intersects(&RenderLayers::layer(AVATAR_RENDER_LAYER)));
        assert!(light_layers.intersects(&RenderLayers::layer(VIEWPORT_ONLY_RENDER_LAYER)));
    }

    #[test]
    fn viewport_camera_renders_to_the_window_not_the_output_image() {
        let mut app = App::new();
        app.init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .add_systems(Startup, setup_scene);
        app.update();

        let mut cameras = app
            .world_mut()
            .query::<(&crate::framing::AvatarViewportCamera, Option<&RenderTarget>)>();
        let (_, target) = cameras.iter(app.world()).next().expect("viewport camera");
        assert!(
            !matches!(target, Some(RenderTarget::Image(_))),
            "egui/webcam overlay on the window must not share the offscreen image target"
        );
    }
}
