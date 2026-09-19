//! Camera-relative portrait lighting for the rich look (Issue #71).
//!
//! A single fixed preset: a warm shadow-casting key, a weak fill and a weak
//! rim, plus a small studio environment. The rig follows the camera
//! orientation so the portrait keeps its shape while the user orbits, but it
//! never follows the head, so turning the head changes the light direction on
//! the face. Light *rotation* is the light direction; the model extent only
//! sizes the shadow range.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use bevy::render::render_resource::{
    Extent3d, TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
    TextureViewDimension,
};
use bevy_vrm1::prelude::{HeadBoneEntity, HipsBoneEntity};

use crate::framing::AvatarViewportCamera;
use crate::framing::camera_control::AvatarCameraControl;
use crate::lifecycle::{AvatarLifecycle, AvatarLifecycleState};
use crate::look::AvatarLookSettings;
use crate::look::preset::{
    STUDIO_PRESET, StudioLightPreset, StudioPreset, blend_look_scalar, effective_look_strength,
};
use crate::render_output::{AVATAR_RENDER_LAYER, AvatarOutputCamera, VIEWPORT_ONLY_RENDER_LAYER};

/// One light of the solved rig.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StudioLight {
    /// The light's world rotation; its local `-Z` is the light direction.
    pub rotation: Quat,
    /// The light color.
    pub color: LinearRgba,
    /// The illuminance in lux.
    pub illuminance: f32,
    /// Whether this light casts shadows.
    pub shadows_enabled: bool,
}

/// The solved rig applied to the scene.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StudioRig {
    /// The shadow-casting key light.
    pub key: StudioLight,
    /// The weak fill light.
    pub fill: StudioLight,
    /// The weak rim light.
    pub rim: StudioLight,
    /// The world-space point the rig is aimed around.
    pub focus: Vec3,
    /// The world-space radius the key light's shadow range must cover.
    pub shadow_extent: f32,
    /// The environment light intensity in cd/m².
    pub environment_intensity: f32,
}

/// Solves the rig for one camera orientation and subject.
#[must_use]
pub fn solve_studio_rig(
    camera_rotation: Quat,
    focus: Vec3,
    avatar_extent: f32,
    preset: &StudioPreset,
) -> StudioRig {
    StudioRig {
        key: solve_light(preset.key, camera_rotation),
        fill: solve_light(preset.fill, camera_rotation),
        rim: solve_light(preset.rim, camera_rotation),
        focus,
        shadow_extent: avatar_extent,
        environment_intensity: preset.environment_intensity,
    }
}

fn solve_light(preset: StudioLightPreset, camera_rotation: Quat) -> StudioLight {
    StudioLight {
        rotation: camera_rotation * Quat::from_rotation_arc(Vec3::NEG_Z, preset.direction.normalize()),
        color: preset.color,
        illuminance: preset.illuminance,
        shadows_enabled: preset.shadows_enabled,
    }
}

/// Interpolates the rig from the original scene state to the solved preset.
///
/// The original rig keeps the scene's own key light and zero fill/rim, so
/// strength 0 restores the standard display exactly.
#[must_use]
pub fn blend_studio_rig(original: &StudioRig, rich: &StudioRig, strength: f32) -> StudioRig {
    StudioRig {
        key: blend_light(&original.key, &rich.key, strength),
        fill: blend_light(&original.fill, &rich.fill, strength),
        rim: blend_light(&original.rim, &rich.rim, strength),
        focus: rich.focus,
        shadow_extent: blend_look_scalar(original.shadow_extent, rich.shadow_extent, strength),
        environment_intensity: blend_look_scalar(
            original.environment_intensity,
            rich.environment_intensity,
            strength,
        ),
    }
}

fn blend_light(original: &StudioLight, rich: &StudioLight, strength: f32) -> StudioLight {
    StudioLight {
        rotation: original.rotation.slerp(rich.rotation, strength),
        color: lerp_linear(original.color, rich.color, strength),
        illuminance: blend_look_scalar(original.illuminance, rich.illuminance, strength),
        shadows_enabled: if strength > 0.0 {
            rich.shadows_enabled
        } else {
            original.shadows_enabled
        },
    }
}

fn lerp_linear(from: LinearRgba, to: LinearRgba, t: f32) -> LinearRgba {
    LinearRgba::new(
        from.red + (to.red - from.red) * t,
        from.green + (to.green - from.green) * t,
        from.blue + (to.blue - from.blue) * t,
        from.alpha + (to.alpha - from.alpha) * t,
    )
}

/// Which studio light slot a light entity fills.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub enum StudioLightSlot {
    /// The scene's own light, taken over as the key.
    Key,
    /// The rig's fill light.
    Fill,
    /// The rig's rim light.
    Rim,
}

/// The scene state the rig replaces, restored while the look is off.
#[derive(Resource, Debug, Default)]
pub struct StudioLookState {
    original_key: Option<StudioLight>,
    original_ambient: Option<(Color, f32)>,
    environment_map: Option<Handle<Image>>,
    last_applied: Option<StudioRig>,
}

impl StudioLookState {
    /// The scene's own key light as it was before the rig took it over.
    #[must_use]
    pub fn original_key(&self) -> Option<StudioLight> {
        self.original_key
    }

    /// The last rig written to the scene.
    #[must_use]
    pub fn last_applied(&self) -> Option<StudioRig> {
        self.last_applied
    }
}

/// Takes over the scene's key light, builds the fill/rim lights and the
/// environment cubemap exactly once, and saves the original values.
pub fn setup_studio_lighting(
    mut commands: Commands,
    mut state: ResMut<StudioLookState>,
    mut images: ResMut<Assets<Image>>,
    ambient: Res<GlobalAmbientLight>,
    key_light: Query<(Entity, &DirectionalLight, &Transform), Without<StudioLightSlot>>,
) {
    if state.original_key.is_some() {
        return;
    }
    let Some((entity, light, transform)) = key_light.iter().next() else {
        return;
    };
    state.original_key = Some(StudioLight {
        rotation: transform.rotation,
        color: light.color.to_linear(),
        illuminance: light.illuminance,
        shadows_enabled: light.shadow_maps_enabled,
    });
    state.original_ambient = Some((ambient.color, ambient.brightness));
    state.environment_map = Some(images.add(studio_environment_cubemap()));
    commands.entity(entity).insert(StudioLightSlot::Key);

    let layers = RenderLayers::from_layers(&[AVATAR_RENDER_LAYER, VIEWPORT_ONLY_RENDER_LAYER]);
    for slot in [StudioLightSlot::Fill, StudioLightSlot::Rim] {
        commands.spawn((
            DirectionalLight {
                illuminance: 0.0,
                shadow_maps_enabled: false,
                ..default()
            },
            Transform::default(),
            layers.clone(),
            slot,
        ));
    }
}

/// Solves the rig from the current camera/focus and applies it to the lights,
/// ambient and environment. Writes nothing while the values are unchanged.
#[allow(clippy::too_many_arguments)]
pub fn sync_studio_lighting(
    settings: Res<AvatarLookSettings>,
    lifecycle: Res<AvatarLifecycle>,
    camera_control: Res<AvatarCameraControl>,
    mut state: ResMut<StudioLookState>,
    mut ambient: ResMut<GlobalAmbientLight>,
    mut lights: Query<
        (&StudioLightSlot, &mut DirectionalLight, &mut Transform, &mut GlobalTransform),
        Without<AvatarViewportCamera>,
    >,
    cameras: Query<&GlobalTransform, (With<AvatarViewportCamera>, Without<StudioLightSlot>)>,
    roots: Query<(&HeadBoneEntity, &HipsBoneEntity)>,
    bones: Query<&GlobalTransform, Without<StudioLightSlot>>,
) {
    let Some(original_key) = state.original_key else {
        return;
    };
    if lifecycle.state() != AvatarLifecycleState::Ready {
        return;
    }
    let mut original = StudioRig {
        key: original_key,
        fill: off_light(original_key.rotation),
        rim: off_light(original_key.rotation),
        focus: Vec3::ZERO,
        shadow_extent: 0.0,
        environment_intensity: 0.0,
    };
    let Some(root) = lifecycle.active_root() else {
        return;
    };
    let Ok((head_entity, hips_entity)) = roots.get(root) else {
        return;
    };
    let (Ok(head), Ok(hips)) = (
        bones.get(**head_entity),
        bones.get(**hips_entity),
    ) else {
        return;
    };
    let Some(extent) = crate::framing::avatar_lighting_extent(head.translation(), hips.translation())
    else {
        return;
    };
    let Ok(camera) = cameras.single() else {
        return;
    };
    let focus = camera_control
        .current_for(lifecycle.current_generation())
        .map_or_else(
            || (head.translation() + hips.translation()) * 0.5,
            |pose| pose.target(),
        );
    original.focus = focus;
    original.shadow_extent = extent;

    let rich = solve_studio_rig(camera.rotation(), focus, extent, &STUDIO_PRESET);
    let strength = effective_look_strength(settings.0);
    let rig = blend_studio_rig(&original, &rich, strength);
    if state.last_applied == Some(rig) {
        return;
    }

    for (slot, mut light, mut transform, mut global) in &mut lights {
        let target = match slot {
            StudioLightSlot::Key => &rig.key,
            StudioLightSlot::Fill => &rig.fill,
            StudioLightSlot::Rim => &rig.rim,
        };
        if light.illuminance != target.illuminance {
            light.illuminance = target.illuminance;
        }
        if light.color.to_linear() != target.color {
            light.color = Color::LinearRgba(target.color);
        }
        if light.shadow_maps_enabled != target.shadows_enabled {
            light.shadow_maps_enabled = target.shadows_enabled;
        }
        if transform.rotation != target.rotation {
            *transform = Transform::from_rotation(target.rotation);
            *global = GlobalTransform::from(*transform);
        }
    }

    let (original_color, original_brightness) = state.original_ambient.unwrap_or((Color::WHITE, 0.0));
    let brightness = blend_look_scalar(original_brightness, rig.environment_intensity, strength);
    if ambient.brightness != brightness {
        ambient.brightness = brightness;
    }
    if strength == 0.0 && ambient.color != original_color {
        ambient.color = original_color;
    }

    state.last_applied = Some(rig);
}

fn off_light(rotation: Quat) -> StudioLight {
    StudioLight {
        rotation,
        color: LinearRgba::BLACK,
        illuminance: 0.0,
        shadows_enabled: false,
    }
}

/// Attaches the studio environment to the cameras that actually draw the
/// avatar, and removes it when the look is off.
// The camera query is a small, fixed two-marker union.
#[allow(clippy::type_complexity)]
pub fn apply_environment_to_avatar_cameras(
    mut commands: Commands,
    settings: Res<AvatarLookSettings>,
    state: ResMut<StudioLookState>,
    cameras: Query<
        (Entity, Option<&GeneratedEnvironmentMapLight>),
        Or<(With<AvatarViewportCamera>, With<AvatarOutputCamera>)>,
    >,
) {
    let Some(map) = state.environment_map.clone() else {
        return;
    };
    let strength = effective_look_strength(settings.0);
    for (entity, existing) in &cameras {
        if strength == 0.0 {
            if existing.is_some() {
                commands
                    .entity(entity)
                    .remove::<GeneratedEnvironmentMapLight>()
                    .remove::<EnvironmentMapLight>();
            }
            continue;
        }
        let intensity = STUDIO_PRESET.environment_intensity * strength;
        if existing.is_some_and(|light| {
            light.environment_map == map && light.intensity == intensity
        }) {
            continue;
        }
        commands.entity(entity).insert(GeneratedEnvironmentMapLight {
            environment_map: map.clone(),
            intensity,
            ..default()
        });
    }
}

/// Builds the small studio environment cubemap: a bright neutral dome with a
/// darker floor. The mip chain is generated from it by Bevy's environment-map
/// filter, so the specular levels are properly roughness-prefiltered.
#[must_use]
pub fn studio_environment_cubemap() -> Image {
    const SIZE: u32 = 16;
    const FACES: usize = 6;
    const UP: LinearRgba = LinearRgba::new(0.55, 0.56, 0.60, 1.0);
    const HORIZON: LinearRgba = LinearRgba::new(0.26, 0.26, 0.30, 1.0);
    const FLOOR: LinearRgba = LinearRgba::new(0.07, 0.07, 0.08, 1.0);

    let mut data = Vec::with_capacity((SIZE * SIZE * FACES as u32 * 4) as usize);
    for face in 0..FACES {
        for y in 0..SIZE {
            for x in 0..SIZE {
                let u = (x as f32 + 0.5) / SIZE as f32 * 2.0 - 1.0;
                let v = (y as f32 + 0.5) / SIZE as f32 * 2.0 - 1.0;
                let direction = match face {
                    0 => Vec3::new(1.0, -v, -u),
                    1 => Vec3::new(-1.0, -v, u),
                    2 => Vec3::new(u, 1.0, v),
                    3 => Vec3::new(u, -1.0, -v),
                    4 => Vec3::new(u, -v, 1.0),
                    _ => Vec3::new(-u, -v, -1.0),
                }
                .normalize();
                let color = environment_color(direction.y, UP, HORIZON, FLOOR);
                data.extend_from_slice(&[
                    linear_to_srgb_u8(color.red),
                    linear_to_srgb_u8(color.green),
                    linear_to_srgb_u8(color.blue),
                    255,
                ]);
            }
        }
    }

    let mut image = Image::new_fill(
        Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: FACES as u32,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.data = Some(data);
    image.texture_descriptor.usage |= TextureUsages::TEXTURE_BINDING;
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::Cube),
        ..default()
    });
    image
}

fn environment_color(y: f32, up: LinearRgba, horizon: LinearRgba, floor: LinearRgba) -> LinearRgba {
    if y >= 0.0 {
        lerp_linear(horizon, up, y)
    } else {
        lerp_linear(horizon, floor, -y)
    }
}

fn linear_to_srgb_u8(value: f32) -> u8 {
    let encoded = if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    (encoded.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framing::camera_control::CameraControlPose;
    use crate::lifecycle::AvatarGeneration;
    use crate::look::preset::RichLookSettings;

    fn original_rig() -> StudioRig {
        StudioRig {
            key: StudioLight {
                rotation: Quat::IDENTITY,
                color: LinearRgba::WHITE,
                illuminance: 1500.0,
                shadows_enabled: false,
            },
            fill: off_light(Quat::IDENTITY),
            rim: off_light(Quat::IDENTITY),
            focus: Vec3::ZERO,
            shadow_extent: 0.0,
            environment_intensity: 0.0,
        }
    }

    #[test]
    fn solved_key_follows_the_camera_rotation() {
        let rig = solve_studio_rig(Quat::IDENTITY, Vec3::ZERO, 1.0, &STUDIO_PRESET);
        let dir = rig.key.rotation * Vec3::NEG_Z;
        assert!((dir - STUDIO_PRESET.key.direction.normalize()).length() < 1e-5);

        let yawed = Quat::from_rotation_y(0.7);
        let rotated = solve_studio_rig(yawed, Vec3::ZERO, 1.0, &STUDIO_PRESET);
        let rotated_dir = rotated.key.rotation * Vec3::NEG_Z;
        let expected = yawed * STUDIO_PRESET.key.direction.normalize();
        assert!((rotated_dir - expected).length() < 1e-5);
    }

    #[test]
    fn strength_endpoints_restore_and_apply() {
        let original = original_rig();
        let rich = solve_studio_rig(Quat::from_rotation_y(0.4), Vec3::new(0.0, 1.0, 0.0), 2.0, &STUDIO_PRESET);

        let restored = blend_studio_rig(&original, &rich, 0.0);
        assert_eq!(restored.key, original.key);
        assert_eq!(restored.fill.illuminance, 0.0);
        assert_eq!(restored.rim.illuminance, 0.0);
        assert_eq!(restored.environment_intensity, 0.0);

        let applied = blend_studio_rig(&original, &rich, 1.0);
        assert_eq!(applied.key.illuminance, STUDIO_PRESET.key.illuminance);
        assert_eq!(applied.key.illuminance, rich.key.illuminance);
        assert_eq!(applied.fill.illuminance, rich.fill.illuminance);
        assert_eq!(applied.rim.illuminance, rich.rim.illuminance);
        assert!(applied.key.shadows_enabled);
        assert!(!applied.fill.shadows_enabled);
        assert_eq!(applied.environment_intensity, STUDIO_PRESET.environment_intensity);
    }

    fn sync_app() -> (App, AvatarGeneration) {
        let mut app = App::new();
        app.init_resource::<Assets<Image>>()
            .init_resource::<AvatarLookSettings>()
            .init_resource::<StudioLookState>()
            .insert_resource(GlobalAmbientLight::default())
            .init_resource::<AvatarCameraControl>()
            .add_systems(Update, setup_studio_lighting)
            .add_systems(PostUpdate, sync_studio_lighting);

        let root = app.world_mut().spawn_empty().id();
        let head = app.world_mut().spawn(GlobalTransform::from_xyz(0.0, 1.7, 0.0)).id();
        let hips = app.world_mut().spawn(GlobalTransform::from_xyz(0.0, 1.0, 0.0)).id();
        app.world_mut()
            .entity_mut(root)
            .insert((HeadBoneEntity(head), HipsBoneEntity(hips)));

        let mut lifecycle = AvatarLifecycle::default();
        lifecycle.request_load(root).expect("load");
        let generation = lifecycle.current_generation();
        lifecycle.start_binding(root);
        lifecycle.finish_ready();
        app.insert_resource(lifecycle);

        let camera = app
            .world_mut()
            .spawn((
                AvatarViewportCamera::from_default_transform(Transform::default()),
                GlobalTransform::from(
                    Transform::from_translation(Vec3::new(0.0, 0.0, 3.0))
                        .looking_at(Vec3::ZERO, Vec3::Y),
                ),
            ))
            .id();
        let pose = CameraControlPose::new(
            Transform::from_translation(Vec3::new(0.0, 0.0, 3.0))
                .looking_at(Vec3::ZERO, Vec3::Y),
            Vec3::new(0.0, 1.2, 0.0),
        )
        .expect("pose");
        app.world_mut()
            .resource_mut::<AvatarCameraControl>()
            .set_current(generation, pose);

        app.world_mut().spawn((
            DirectionalLight {
                illuminance: 1500.0,
                ..default()
            },
            Transform::from_rotation(Quat::from_rotation_x(-0.5)),
        ));
        let _ = camera;
        (app, generation)
    }

    #[test]
    fn setup_marks_the_scene_key_and_spawns_fill_and_rim_once() {
        let (mut app, _) = sync_app();
        app.update();

        let mut lights = app
            .world_mut()
            .query::<(&StudioLightSlot, &DirectionalLight)>();
        let slots: Vec<_> = lights.iter(app.world()).map(|(slot, _)| *slot).collect();
        assert_eq!(slots.len(), 3);
        assert!(slots.contains(&StudioLightSlot::Key));
        assert!(slots.contains(&StudioLightSlot::Fill));
        assert!(slots.contains(&StudioLightSlot::Rim));
        drop(lights);

        // Repeated frames do not add more lights.
        app.update();
        let count = app.world_mut().query::<&StudioLightSlot>().iter(app.world()).count();
        assert_eq!(count, 3);
        assert!(app.world().resource::<StudioLookState>().original_key().is_some());
    }

    #[test]
    fn sync_applies_the_preset_and_restores_the_scene_key() {
        let (mut app, _) = sync_app();
        app.update();

        app.world_mut().resource_mut::<AvatarLookSettings>().0 = RichLookSettings {
            enabled: true,
            strength: 1.0,
        };
        app.update();

        let mut lights = app
            .world_mut()
            .query::<(&StudioLightSlot, &DirectionalLight, &Transform)>();
        let key = lights
            .iter(app.world())
            .find(|(slot, _, _)| **slot == StudioLightSlot::Key)
            .map(|(_, light, transform)| (light.illuminance, light.shadow_maps_enabled, transform.rotation))
            .expect("key light");
        assert_eq!(key.0, STUDIO_PRESET.key.illuminance);
        assert!(key.1);
        assert_eq!(
            app.world().resource::<GlobalAmbientLight>().brightness,
            STUDIO_PRESET.environment_intensity
        );

        app.world_mut().resource_mut::<AvatarLookSettings>().0 = RichLookSettings {
            enabled: false,
            strength: 1.0,
        };
        app.update();

        let key = lights
            .iter(app.world())
            .find(|(slot, _, _)| **slot == StudioLightSlot::Key)
            .map(|(_, light, transform)| (light.illuminance, light.shadow_maps_enabled, transform.rotation))
            .expect("key light");
        assert_eq!(key.0, 1500.0);
        assert!(!key.1);
        assert_eq!(key.2, Quat::from_rotation_x(-0.5));
        let fill = lights
            .iter(app.world())
            .find(|(slot, _, _)| **slot == StudioLightSlot::Fill)
            .map(|(_, light, _)| light.illuminance)
            .expect("fill light");
        assert_eq!(fill, 0.0);
    }

    #[test]
    fn environment_only_reaches_avatar_cameras_while_enabled() {
        let mut app = App::new();
        app.init_resource::<Assets<Image>>()
            .init_resource::<AvatarLookSettings>()
            .init_resource::<StudioLookState>()
            .init_resource::<GlobalAmbientLight>()
            .add_systems(Update, (setup_studio_lighting, apply_environment_to_avatar_cameras));
        app.world_mut()
            .resource_mut::<StudioLookState>()
            .environment_map = Some(Handle::default());

        let avatar_camera = app
            .world_mut()
            .spawn(AvatarViewportCamera::from_default_transform(Transform::default()))
            .id();
        let other_camera = app.world_mut().spawn(Camera3d::default()).id();

        app.world_mut().resource_mut::<AvatarLookSettings>().0 = RichLookSettings {
            enabled: true,
            strength: 1.0,
        };
        app.update();
        assert!(
            app.world()
                .get::<GeneratedEnvironmentMapLight>(avatar_camera)
                .is_some()
        );
        assert!(
            app.world()
                .get::<GeneratedEnvironmentMapLight>(other_camera)
                .is_none()
        );

        app.world_mut().resource_mut::<AvatarLookSettings>().0 = RichLookSettings {
            enabled: false,
            strength: 1.0,
        };
        app.update();
        assert!(
            app.world()
                .get::<GeneratedEnvironmentMapLight>(avatar_camera)
                .is_none()
        );
    }

    #[test]
    fn studio_cubemap_is_a_square_power_of_two_cube() {
        let image = studio_environment_cubemap();
        assert_eq!(image.texture_descriptor.size.width, 16);
        assert_eq!(image.texture_descriptor.size.height, 16);
        assert_eq!(image.texture_descriptor.size.depth_or_array_layers, 6);
        assert_eq!(image.data.as_ref().map(Vec::len), Some(16 * 16 * 6 * 4));
    }
}
