//! Additional Look lights: the fixed camera-relative key and rim spots.
//!
//! The Native standard light (`StandardAvatarLight`) is never touched here:
//! the look only adds output on top of it. OFF or 0% strength despawns both
//! spots, so the rendered result returns to the Native baseline. The rig is
//! anchored to the model's upper-body center/size and only its direction turns
//! with the camera, so head rotation never changes the lighting. Upstream
//! MToon ignores spot lights; the app-side Rich shaders (#94, #95) sum them.

use bevy::camera::visibility::RenderLayers;
use bevy::light::cluster::GlobalClusterSettings;
use bevy::prelude::*;
use bevy_vrm1::prelude::{HeadBoneEntity, HipsBoneEntity};

use crate::framing::{AvatarViewportCamera, avatar_focus_and_size};
use crate::lifecycle::AvatarLifecycle;
use crate::look::AvatarLookSettings;
use crate::render_output::{AVATAR_RENDER_LAYER, VIEWPORT_ONLY_RENDER_LAYER};

/// The spot range as a multiple of the light-to-focus distance, so the model
/// always sits well inside the range.
const RANGE_MARGIN: f32 = 3.0;

/// One fixed additional light, expressed in camera space.
///
/// `direction` points from the avatar focus toward the light with `+X` right,
/// `+Y` up and `+Z` toward the camera, so the preset turns with the camera
/// while staying anchored to the model center and size.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
struct AdditionalLookLight {
    /// Luminous power in lumens at 100% look strength.
    nominal_lumens: f32,
    /// Direction from the avatar focus toward this light, in camera space.
    direction: Vec3,
    /// Light-to-focus distance as a multiple of the upper-body size.
    distance_factor: f32,
    /// Spot cone outer half-angle in radians.
    outer_angle: f32,
}

/// The fixed preset: a diagonal front key and a back rim, both shadowless.
const ADDITIONAL_LOOK_LIGHTS: [AdditionalLookLight; 2] = [
    AdditionalLookLight {
        nominal_lumens: 8_000.0,
        direction: Vec3::new(-0.55, 0.45, 0.70),
        distance_factor: 1.5,
        outer_angle: 0.6,
    },
    AdditionalLookLight {
        nominal_lumens: 3_500.0,
        direction: Vec3::new(0.45, 0.60, -0.66),
        distance_factor: 1.6,
        outer_angle: 0.5,
    },
];

impl AdditionalLookLight {
    /// Resolves this light's transform and range from the model bounds.
    fn resolve(self, focus: Vec3, size: f32, camera_rotation: Quat) -> (Transform, f32) {
        let direction = (camera_rotation * self.direction).normalize();
        let distance = self.distance_factor * size;
        let transform = Transform::from_translation(focus + direction * distance)
            .with_rotation(Quat::from_rotation_arc(Vec3::NEG_Z, -direction));
        (transform, distance * RANGE_MARGIN)
    }
}

/// Registers the additional-light sync in `Update`.
///
/// The system runs before transform propagation so a despawn in the same
/// frame cannot race the `bevy_light` system that queues a bounding `Sphere`
/// for changed spot lights; queued commands are applied at the schedule
/// boundary, which keeps the entity alive when that insert runs.
///
/// The two spot lights are clustered, and with Bevy 0.19's default GPU light
/// clustering the avatar's clusters stay empty, so the added lights never reach
/// any material. The CPU clustering path is selected instead: with a handful of
/// lights its cost is negligible. Remove this once the GPU path delivers the
/// clusters in this scene.
pub(crate) fn register_look_lighting(app: &mut App) {
    app.add_systems(Startup, use_cpu_light_clustering)
        .add_systems(
            Update,
            sync_additional_look_lights.after(crate::look::apply_look_settings_changes),
        );
}

fn use_cpu_light_clustering(settings: Option<ResMut<GlobalClusterSettings>>) {
    if let Some(mut settings) = settings {
        settings.gpu_clustering = None;
    }
}

// Existing lights are updated in place; spawn/despawn happens only when the
// look turns on/off or when the model/bones become unavailable.
#[allow(clippy::type_complexity)]
fn sync_additional_look_lights(
    mut commands: Commands,
    lifecycle: Res<AvatarLifecycle>,
    settings: Res<AvatarLookSettings>,
    camera: Single<&Transform, (With<AvatarViewportCamera>, Without<AdditionalLookLight>)>,
    roots: Query<(&HeadBoneEntity, &HipsBoneEntity)>,
    bones: Query<&GlobalTransform, Without<AdditionalLookLight>>,
    mut lights: Query<(Entity, &AdditionalLookLight, &mut SpotLight, &mut Transform)>,
) {
    let look = settings.0;
    let effective_strength = if look.enabled { look.strength } else { 0.0 };
    let framed = (effective_strength > 0.0)
        .then(|| {
            let root = lifecycle.active_root()?;
            let (head_bone, hips_bone) = roots.get(root).ok()?;
            let head = bones.get(**head_bone).ok()?.translation();
            let hips = bones.get(**hips_bone).ok()?.translation();
            avatar_focus_and_size(head, hips)
        })
        .flatten();

    let Some((focus, size)) = framed else {
        for (entity, ..) in &mut lights {
            commands.entity(entity).despawn();
        }
        return;
    };

    if lights.is_empty() {
        for preset in ADDITIONAL_LOOK_LIGHTS {
            let (transform, range) = preset.resolve(focus, size, camera.rotation);
            commands.spawn((
                SpotLight {
                    color: Color::WHITE,
                    intensity: preset.nominal_lumens * effective_strength,
                    range,
                    outer_angle: preset.outer_angle,
                    shadow_maps_enabled: false,
                    ..default()
                },
                transform,
                preset,
                RenderLayers::from_layers(&[AVATAR_RENDER_LAYER, VIEWPORT_ONLY_RENDER_LAYER]),
            ));
        }
        return;
    }

    for (_, preset, mut light, mut transform) in &mut lights {
        let (next, range) = preset.resolve(focus, size, camera.rotation);
        *transform = next;
        light.intensity = preset.nominal_lumens * effective_strength;
        light.range = range;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::look::RichLookSettings;

    const HIPS: Vec3 = Vec3::new(0.0, 0.8, 0.0);
    const HEAD: Vec3 = Vec3::new(0.0, 1.2, 0.0);

    fn camera_transform() -> Transform {
        Transform::from_translation(Vec3::new(0.0, 1.2, 2.5)).looking_at(HEAD, Vec3::Y)
    }

    fn focus_and_size() -> (Vec3, f32) {
        avatar_focus_and_size(HEAD, HIPS).expect("test upper body is valid")
    }

    fn test_app() -> App {
        let mut app = App::new();
        app.init_resource::<AvatarLifecycle>()
            .init_resource::<AvatarLookSettings>()
            .add_systems(Startup, spawn_test_camera)
            .add_systems(Update, sync_additional_look_lights);
        app
    }

    fn spawn_test_camera(mut commands: Commands) {
        commands.spawn((
            AvatarViewportCamera::from_default_transform(camera_transform()),
            camera_transform(),
        ));
    }

    fn spawn_test_avatar(app: &mut App, head: Vec3, hips: Vec3) {
        let head_bone = app
            .world_mut()
            .spawn(GlobalTransform::from_translation(head))
            .id();
        let hips_bone = app
            .world_mut()
            .spawn(GlobalTransform::from_translation(hips))
            .id();
        let root = app
            .world_mut()
            .spawn((HeadBoneEntity(head_bone), HipsBoneEntity(hips_bone)))
            .id();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .request_load(root)
            .expect("test load request is accepted");
    }

    fn set_look(app: &mut App, enabled: bool, strength: f32) {
        app.world_mut().resource_mut::<AvatarLookSettings>().0 =
            RichLookSettings { enabled, strength };
    }

    fn set_camera_rotation(app: &mut App, rotation: Quat) {
        let mut cameras = app
            .world_mut()
            .query_filtered::<&mut Transform, With<AvatarViewportCamera>>();
        let mut camera = cameras
            .single_mut(app.world_mut())
            .expect("exactly one viewport camera");
        camera.rotation = rotation;
    }

    struct LightSnapshot {
        entity: Entity,
        preset: AdditionalLookLight,
        light: SpotLight,
        transform: Transform,
        layers: RenderLayers,
    }

    fn lights(app: &mut App) -> Vec<LightSnapshot> {
        let mut query = app.world_mut().query::<(
            Entity,
            &AdditionalLookLight,
            &SpotLight,
            &Transform,
            &RenderLayers,
        )>();
        let mut found: Vec<_> = query
            .iter(app.world())
            .map(|(entity, preset, light, transform, layers)| LightSnapshot {
                entity,
                preset: *preset,
                light: *light,
                transform: *transform,
                layers: layers.clone(),
            })
            .collect();
        found.sort_by_key(|snapshot| snapshot.entity);
        found
    }

    #[test]
    fn enabling_the_look_spawns_both_fixed_spots() {
        let mut app = test_app();
        app.update();
        spawn_test_avatar(&mut app, HEAD, HIPS);
        set_look(&mut app, true, 1.0);
        app.update();

        let (focus, size) = focus_and_size();
        let found = lights(&mut app);
        assert_eq!(found.len(), ADDITIONAL_LOOK_LIGHTS.len());
        for preset in ADDITIONAL_LOOK_LIGHTS {
            let snapshot = found
                .iter()
                .find(|snapshot| snapshot.preset == preset)
                .expect("every preset light exists");
            assert!(!snapshot.light.shadow_maps_enabled);
            assert_eq!(snapshot.light.color, Color::WHITE);
            assert_eq!(snapshot.light.outer_angle, preset.outer_angle);
            assert_eq!(snapshot.light.intensity, preset.nominal_lumens);
            assert!(
                snapshot
                    .layers
                    .intersects(&RenderLayers::layer(AVATAR_RENDER_LAYER))
            );
            assert!(
                snapshot
                    .layers
                    .intersects(&RenderLayers::layer(VIEWPORT_ONLY_RENDER_LAYER))
            );

            let expected_distance = preset.distance_factor * size;
            let to_light = snapshot.transform.translation - focus;
            assert!((to_light.length() - expected_distance).abs() < 1e-4);
            assert_eq!(snapshot.light.range, expected_distance * RANGE_MARGIN);

            let expected_direction = (camera_transform().rotation * preset.direction).normalize();
            assert!((to_light.normalize() - expected_direction).length() < 1e-5);
            let to_focus = (focus - snapshot.transform.translation).normalize();
            assert!(snapshot.transform.forward().dot(to_focus) > 0.999);
        }
    }

    #[test]
    fn strength_updates_the_output_without_respawning() {
        let mut app = test_app();
        app.update();
        spawn_test_avatar(&mut app, HEAD, HIPS);
        set_look(&mut app, true, 1.0);
        app.update();
        let before: Vec<Entity> = lights(&mut app).iter().map(|light| light.entity).collect();
        assert_eq!(before.len(), ADDITIONAL_LOOK_LIGHTS.len());

        set_look(&mut app, true, 0.25);
        app.update();

        let after = lights(&mut app);
        let after_entities: Vec<Entity> = after.iter().map(|light| light.entity).collect();
        assert_eq!(after_entities, before);
        for snapshot in &after {
            assert_eq!(
                snapshot.light.intensity,
                snapshot.preset.nominal_lumens * 0.25
            );
        }
    }

    #[test]
    fn off_and_zero_strength_despawn_the_spots() {
        let mut app = test_app();
        app.update();
        spawn_test_avatar(&mut app, HEAD, HIPS);
        set_look(&mut app, true, 1.0);
        app.update();
        assert_eq!(lights(&mut app).len(), ADDITIONAL_LOOK_LIGHTS.len());

        set_look(&mut app, true, 0.0);
        app.update();
        assert!(lights(&mut app).is_empty());

        set_look(&mut app, true, 1.0);
        app.update();
        assert_eq!(lights(&mut app).len(), ADDITIONAL_LOOK_LIGHTS.len());

        set_look(&mut app, false, 1.0);
        app.update();
        assert!(lights(&mut app).is_empty());
    }

    #[test]
    fn the_spots_need_a_bound_model() {
        let mut app = test_app();
        app.update();
        set_look(&mut app, true, 1.0);
        app.update();
        assert!(lights(&mut app).is_empty());

        spawn_test_avatar(&mut app, HEAD, HIPS);
        app.update();
        assert_eq!(lights(&mut app).len(), ADDITIONAL_LOOK_LIGHTS.len());

        let head_bone = {
            let mut roots = app.world_mut().query::<&HeadBoneEntity>();
            roots.iter(app.world()).next().map(|head| **head)
        }
        .expect("head bone entity");
        *app.world_mut()
            .get_mut::<GlobalTransform>(head_bone)
            .expect("head bone transform") = GlobalTransform::from_translation(HIPS);
        app.update();
        assert!(lights(&mut app).is_empty());
    }

    #[test]
    fn the_preset_turns_with_the_camera() {
        let mut app = test_app();
        app.update();
        spawn_test_avatar(&mut app, HEAD, HIPS);
        set_look(&mut app, true, 1.0);
        app.update();

        let rotation = Quat::from_rotation_y(0.8);
        set_camera_rotation(&mut app, rotation);
        app.update();

        let (focus, _) = focus_and_size();
        for snapshot in lights(&mut app) {
            let to_light = (snapshot.transform.translation - focus).normalize();
            let expected = (rotation * snapshot.preset.direction).normalize();
            assert!((to_light - expected).length() < 1e-5);
        }
    }

    #[test]
    fn the_preset_scales_with_the_model_size() {
        let taller_head = Vec3::new(0.0, 1.6, 0.0);
        let mut app = test_app();
        app.update();
        spawn_test_avatar(&mut app, taller_head, HIPS);
        set_look(&mut app, true, 1.0);
        app.update();

        let (focus, size) =
            avatar_focus_and_size(taller_head, HIPS).expect("test upper body is valid");
        for snapshot in lights(&mut app) {
            let expected_distance = snapshot.preset.distance_factor * size;
            let to_light = snapshot.transform.translation - focus;
            assert!((to_light.length() - expected_distance).abs() < 1e-4);
        }
    }
}
