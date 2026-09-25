//! `VtuberAvatarPlugin` and top-level Bevy system registration.
//!
//! This is the only plugin that wires `bevy_vrm1` systems together with the
//! VTuber lifecycle domain. `bevy_vrm1` types are used internally and are not
//! re-exported from the crate facade.

use bevy::app::AnimationSystems;
use bevy::camera::{Exposure, visibility::RenderLayers};
use bevy::core_pipeline::tonemapping::{DebandDither, Tonemapping};
use bevy::prelude::*;
use bevy_vrm1::prelude::*;

use crate::arm_pose::ArmPoseOverrideStore;
use crate::arm_pose::apply_default_arm_pose;
use crate::bind::observe_initialized;
use crate::binding::bind_humanoid_bones;
use crate::body_motion::{
    LossIdleState, PositionInputMetrics, reset_position_metrics_on_lifecycle_change,
    update_body_tracking_position_input,
};
use crate::direct_look::register_direct_look;
use crate::direct_pose::{apply_direct_body_tracking, register_direct_pose};
use crate::direct_position::register_direct_position;
use crate::expression::apply_tracked_expressions;
use crate::expression::manual::{
    ManualExpressionRequest, ManualExpressionSelection, ManualExpressionSet,
    apply_manual_expression_requests,
};
use crate::expression::material::{
    apply_expression_materials, register_gltf_material_index_handler,
    restore_expression_materials_on_unload,
};
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
    AVATAR_RENDER_LAYER, AvatarOutputCamera, VIEWPORT_ONLY_RENDER_LAYER, register_output_systems,
};
use crate::unload::{
    ActiveControlFrame, clear_control_cache_on_lifecycle_change, despawn_unloading_avatar,
};

/// Plugin that sets up the VRM avatar scene, lifecycle, and diagnostics.
#[derive(Default)]
pub struct VtuberAvatarPlugin;

impl Plugin for VtuberAvatarPlugin {
    fn build(&self, app: &mut App) {
        // The glTF loader handler must be registered before the loader
        // plugin's `finish` snapshots the handler list; it tags mesh
        // entities with their glTF material index at load time.
        register_gltf_material_index_handler(app);
        app.add_plugins(VrmPlugin)
            .add_plugins(crate::compatibility::VrmCompatibilityPlugin);
        register_direct_pose(app);
        register_direct_position(app);
        register_direct_look(app);
        app.init_resource::<AvatarLifecycle>()
            .init_resource::<AvatarCameraControl>()
            .init_resource::<CameraPointerInputGate>()
            .init_resource::<CameraPointerGesture>()
            .init_resource::<ArmPoseOverrideStore>()
            .init_resource::<crate::arm_pipeline::ArmSourceSelection>()
            .init_resource::<crate::arm_pipeline::TrackedArmControl>()
            .add_message::<crate::arm_pose::ArmPoseProfileChange>()
            .init_resource::<ActiveControlFrame>()
            .init_resource::<ManualExpressionSelection>()
            .add_message::<ManualExpressionRequest>()
            .add_systems(
                Update,
                apply_manual_expression_requests.in_set(ManualExpressionSet),
            )
            .init_resource::<AvatarMotionMirror>()
            .init_resource::<PoseApplyMetrics>()
            .init_resource::<PositionInputMetrics>()
            .init_resource::<LossIdleState>()
            .init_resource::<crate::body_motion::BodyFollowFilter>()
            .init_resource::<crate::tracking_profile::GlobalBodyTrackingProfile>()
            .init_resource::<crate::look::AvatarLookSettings>()
            .add_message::<crate::look::LookSettingsChanged>()
            .add_systems(Update, crate::look::apply_look_settings_changes)
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
            .add_systems(PostStartup, setup_avatar_display)
            .add_systems(
                Update,
                (
                    handle_load_imported_avatar_requests,
                    apply_avatar_request_events,
                    restore_expression_materials_on_unload,
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
                align_standard_light_to_camera.after(frame_avatar_camera),
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
                crate::arm_pipeline::update_tracked_arm_targets
                    .after(apply_direct_body_tracking)
                    .after(crate::arm_pipeline::update_dynamic_arm_targets)
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
            .add_systems(
                PostUpdate,
                apply_expression_materials.after(VrmSystemSets::Expressions),
            )
            .add_systems(Update, reset_pose_metrics_on_lifecycle_change)
            .add_systems(Update, reset_position_metrics_on_lifecycle_change)
            .add_systems(Update, crate::pose::debug_propagation_probe);
        crate::look::register_look_lighting(app);
        crate::look::register_rich_mtoon(app);
        crate::look::register_rich_standard(app);
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

/// The single, fixed-strength Native directional light; never a Look light.
#[derive(Component)]
struct StandardAvatarLight;

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // No ambient fill or environment lighting in the Native baseline.
    // Author-supplied emission remains part of the unchanged materials.
    commands.insert_resource(GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 0.0,
        affects_lightmapped_meshes: false,
    });

    let camera_transform = Transform::from_translation(Vec3::new(0.0, 0.0, 2.5))
        .looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y);

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

    // Light travels along the camera's forward (-Z) direction. Keep the
    // existing illuminance rather than retuning the upstream Native look.
    // Do not spawn zero-lux Fill/Rim lights: upstream MToon still sees them.
    commands.spawn((
        DirectionalLight {
            color: Color::WHITE,
            illuminance: 650.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_rotation(camera_transform.rotation),
        StandardAvatarLight,
        RenderLayers::from_layers(&[AVATAR_RENDER_LAYER, VIEWPORT_ONLY_RENDER_LAYER]),
    ));

    // Camera framing the upper body.
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

// Both cameras exist after Startup, before the first rendered frame.
// Apply ordinary display settings once, independently of Look: fixed EV100
// 9.7, SDR (no Hdr component), no tone curve or dithering. Both cameras keep
// Bevy's default sRGB target format; do not set CompositingSpace, which would
// switch the window pass to Rgba8Unorm and conflict with the egui pipeline.
// Keep the existing transparent BGRA output and preview alpha/sRGB conversion
// unchanged. Later Look systems must not rewrite this policy.
#[allow(clippy::type_complexity)]
fn setup_avatar_display(
    mut commands: Commands,
    cameras: Query<Entity, Or<(With<AvatarViewportCamera>, With<AvatarOutputCamera>)>>,
) {
    for camera in &cameras {
        commands.entity(camera).insert((
            Exposure::BLENDER,
            Tonemapping::None,
            DebandDither::Disabled,
        ));
    }
}

// Framing writes the camera after transform propagation. Update the root
// light's GlobalTransform here too, so orbit/reset affect lighting in the same
// frame. Only the camera is read: head pose and Look strength have no authority
// over this light's direction, color, illuminance or shadow setting.
#[allow(clippy::type_complexity)]
fn align_standard_light_to_camera(
    camera: Single<&Transform, (With<AvatarViewportCamera>, Without<StandardAvatarLight>)>,
    light: Single<(&mut Transform, &mut GlobalTransform), With<StandardAvatarLight>>,
) {
    let (mut transform, mut global_transform) = light.into_inner();
    transform.rotation = camera.rotation;
    *global_transform = GlobalTransform::from(*transform);
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
    use crate::render_output::{AvatarOutputState, setup_output_camera};
    use bevy::camera::{CompositingSpace, Hdr, RenderTarget};

    #[test]
    fn setup_scene_keeps_ground_off_the_output_layer() {
        let mut app = App::new();
        app.init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .add_systems(Startup, setup_scene)
            .add_systems(PostStartup, setup_avatar_display)
            .add_systems(PostUpdate, align_standard_light_to_camera);
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
            .query_filtered::<(Entity, &Transform, &RenderLayers), With<AvatarViewportCamera>>();
        let (camera_entity, camera_transform, viewport_layers) =
            cameras.single(app.world()).expect("viewport camera");
        let camera_rotation = camera_transform.rotation;
        assert!(viewport_layers.intersects(&RenderLayers::layer(AVATAR_RENDER_LAYER)));
        assert!(viewport_layers.intersects(&RenderLayers::layer(VIEWPORT_ONLY_RENDER_LAYER)));

        let mut lights = app
            .world_mut()
            .query::<(Entity, &DirectionalLight, &Transform, &RenderLayers)>();
        let (light_entity, light, transform, light_layers) = lights
            .single(app.world())
            .expect("exactly one directional light");
        assert!(light_layers.intersects(&RenderLayers::layer(AVATAR_RENDER_LAYER)));
        assert!(light_layers.intersects(&RenderLayers::layer(VIEWPORT_ONLY_RENDER_LAYER)));
        assert_eq!(light.color, Color::WHITE);
        assert_eq!(light.illuminance, 650.0);
        assert!(!light.shadow_maps_enabled);
        assert_eq!(transform.rotation, camera_rotation);
        assert_eq!(app.world().resource::<GlobalAmbientLight>().brightness, 0.0);

        let rotation = Quat::from_euler(EulerRot::YXZ, 0.4, -0.2, 0.0);
        app.world_mut()
            .get_mut::<Transform>(camera_entity)
            .unwrap()
            .rotation = rotation;
        app.update();
        let transform = app.world().get::<Transform>(light_entity).unwrap();
        assert_eq!(transform.rotation, rotation);
        assert_eq!(
            app.world().get::<GlobalTransform>(light_entity),
            Some(&GlobalTransform::from(*transform))
        );
        let light = app.world().get::<DirectionalLight>(light_entity).unwrap();
        assert_eq!(light.illuminance, 650.0);
        assert_eq!(light.color, Color::WHITE);
        assert!(!light.shadow_maps_enabled);
    }

    #[test]
    fn look_lights_leave_the_standard_light_untouched() {
        fn standard_light(app: &mut App) -> (f32, Color, bool, Quat) {
            let mut query = app.world_mut().query::<(&DirectionalLight, &Transform)>();
            let lights: Vec<_> = query
                .iter(app.world())
                .map(|(light, transform)| {
                    (
                        light.illuminance,
                        light.color,
                        light.shadow_maps_enabled,
                        transform.rotation,
                    )
                })
                .collect();
            assert_eq!(lights.len(), 1, "exactly one standard directional light");
            lights[0]
        }

        let mut app = App::new();
        app.init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<AvatarLifecycle>()
            .init_resource::<crate::look::AvatarLookSettings>()
            .add_systems(Startup, setup_scene)
            .add_systems(PostStartup, setup_avatar_display)
            .add_systems(PostUpdate, align_standard_light_to_camera);
        crate::look::register_look_lighting(&mut app);
        app.update();

        let head = app
            .world_mut()
            .spawn(GlobalTransform::from_xyz(0.0, 1.3, 0.0))
            .id();
        let hips = app
            .world_mut()
            .spawn(GlobalTransform::from_xyz(0.0, 0.9, 0.0))
            .id();
        let root = app
            .world_mut()
            .spawn((HeadBoneEntity(head), HipsBoneEntity(hips)))
            .id();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .request_load(root)
            .expect("test load request is accepted");

        let before = standard_light(&mut app);

        app.world_mut()
            .resource_mut::<crate::look::AvatarLookSettings>()
            .0 = crate::look::RichLookSettings {
            enabled: true,
            strength: 0.5,
        };
        app.update();
        let mut spots = app.world_mut().query::<&SpotLight>();
        assert_eq!(spots.iter(app.world()).count(), 2);
        assert_eq!(standard_light(&mut app), before);

        app.world_mut()
            .resource_mut::<crate::look::AvatarLookSettings>()
            .0 = crate::look::RichLookSettings {
            enabled: false,
            strength: 1.0,
        };
        app.update();
        let mut spots = app.world_mut().query::<&SpotLight>();
        assert_eq!(spots.iter(app.world()).count(), 0);
        assert_eq!(standard_light(&mut app), before);
    }

    #[test]
    fn viewport_camera_renders_to_the_window_not_the_output_image() {
        let mut app = App::new();
        app.init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<Assets<Image>>()
            .init_resource::<AvatarOutputState>()
            .add_systems(Startup, (setup_scene, setup_output_camera))
            .add_systems(PostStartup, setup_avatar_display);
        app.update();

        let mut cameras = app
            .world_mut()
            .query::<(&crate::framing::AvatarViewportCamera, Option<&RenderTarget>)>();
        let (_, target) = cameras.iter(app.world()).next().expect("viewport camera");
        assert!(
            !matches!(target, Some(RenderTarget::Image(_))),
            "egui/webcam overlay on the window must not share the offscreen image target"
        );

        let mut displays =
            app.world_mut().query_filtered::<(
                &Exposure,
                &Tonemapping,
                &DebandDither,
                Option<&CompositingSpace>,
                Option<&Hdr>,
            ), Or<(With<AvatarViewportCamera>, With<AvatarOutputCamera>)>>(
            );
        assert_eq!(displays.iter(app.world()).count(), 2);
        for (exposure, tone, dither, compositing, hdr) in displays.iter(app.world()) {
            assert_eq!(exposure.ev100, 9.7);
            assert_eq!(*tone, Tonemapping::None);
            assert_eq!(*dither, DebandDither::Disabled);
            assert!(
                compositing.is_none(),
                "both cameras keep Bevy's default sRGB target format"
            );
            assert!(hdr.is_none(), "both cameras use the same fixed SDR policy");
        }
    }
}
