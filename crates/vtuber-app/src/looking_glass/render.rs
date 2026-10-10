//! Bevy resource/entity effects for the optional output. No image readback.

use bevy::asset::embedded_asset;
use bevy::camera::{
    CameraUpdateSystems, ClearColorConfig, Exposure, RenderTarget, ScalingMode,
    visibility::{RenderLayers, VisibilitySystems},
};
use bevy::core_pipeline::tonemapping::{DebandDither, Tonemapping};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::{
    render_resource::{AsBindGroup, TextureFormat},
    storage::ShaderBuffer,
};
use bevy::shader::ShaderRef;
use bevy::sprite_render::{Material2d, Material2dPlugin};
use bevy::window::{MonitorSelection, PrimaryWindow, WindowMode, WindowRef, WindowResolution};
use vtuber_avatar::{
    AVATAR_RENDER_LAYER, AvatarCameraControl, AvatarLifecycle, CameraControlPose,
    FIXED_VERTICAL_FOV,
};

use super::{
    OutputConfig,
    optics::{InterlaceUniforms, view_pose},
};

// Layers 0/1 belong to the avatar/ground. These two contain only this output's
// 2D composition meshes; none are visible to avatar, preview or NDI cameras.
const QUILT_LAYER: usize = 2;
const DISPLAY_LAYER: usize = 3;

#[derive(Component)]
struct ViewCamera(u32);
#[derive(Component)]
struct QuiltTile;
#[derive(Component)]
struct OutputWindow(Handle<InterlaceMaterial>);

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct InterlaceMaterial {
    #[uniform(0)]
    uniforms: InterlaceUniforms,
    #[texture(1)]
    #[sampler(2)]
    quilt: Handle<Image>,
    #[storage(3, read_only)]
    cells: Handle<ShaderBuffer>,
}

impl Material2d for InterlaceMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://vtuber_app/looking_glass/interlace.wgsl".into()
    }
}

pub(super) fn install(app: &mut App, config: OutputConfig) {
    embedded_asset!(app, "interlace.wgsl");
    app.insert_resource(config)
        .add_plugins(Material2dPlugin::<InterlaceMaterial>::default())
        .add_systems(Startup, setup)
        .add_systems(Update, window_keys)
        .add_systems(
            PostUpdate,
            sync_views
                .after(TransformSystems::Propagate)
                .before(CameraUpdateSystems)
                .before(VisibilitySystems::VisibilityPropagate),
        )
        .add_systems(Last, close_with_main);
}

fn target(images: &mut Assets<Image>, width: u32, height: u32) -> Handle<Image> {
    images.add(Image::new_target_texture(
        width,
        height,
        TextureFormat::Bgra8UnormSrgb,
        None,
    ))
}

fn flat_projection(width: f32, height: f32) -> Projection {
    Projection::Orthographic(OrthographicProjection {
        scaling_mode: ScalingMode::Fixed { width, height },
        ..OrthographicProjection::default_2d()
    })
}

fn setup(
    mut commands: Commands,
    config: Res<OutputConfig>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<InterlaceMaterial>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
) {
    let layout = config.layout;
    let calibration = &config.calibration;
    let quilt = target(&mut images, layout.width(), layout.height());
    let material = materials.add(InterlaceMaterial {
        uniforms: calibration.uniforms(layout),
        quilt: quilt.clone(),
        cells: buffers.add(ShaderBuffer::from(calibration.cells.clone())),
    });
    let window = commands
        .spawn((
            Window {
                title: "RusTuberV — Looking Glass (F11: fullscreen / F10: quilt / Esc: close)"
                    .into(),
                resolution: WindowResolution::new(
                    calibration.width.div_ceil(2),
                    calibration.height.div_ceil(2),
                )
                .with_scale_factor_override(1.0),
                ..default()
            },
            OutputWindow(material.clone()),
            Transform::default(),
            Visibility::Visible,
        ))
        .id();

    for index in 0..layout.count() {
        // Keep each Camera3d target small instead of giving every view the
        // entire quilt as its intermediate/depth render-target extent.
        let image = target(&mut images, layout.view_width, layout.view_height);
        commands.spawn((
            Camera3d::default(),
            Camera {
                order: -3,
                is_active: false,
                clear_color: ClearColorConfig::Custom(Color::BLACK),
                ..default()
            },
            RenderTarget::Image(image.clone().into()),
            Msaa::Off,
            Exposure::BLENDER,
            Tonemapping::None,
            DebandDither::Disabled,
            RenderLayers::layer(AVATAR_RENDER_LAYER),
            ViewCamera(index),
            ChildOf(window),
        ));
        let mut sprite = Sprite::from_image(image);
        sprite.custom_size = Some(Vec2::new(
            layout.view_width as f32,
            layout.view_height as f32,
        ));
        commands.spawn((
            sprite,
            Transform::from_translation(layout.tile_center(index)),
            Visibility::Hidden,
            RenderLayers::layer(QUILT_LAYER),
            QuiltTile,
            ChildOf(window),
        ));
    }
    commands.spawn((
        Camera2d,
        Camera {
            order: -2,
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        RenderTarget::Image(quilt.into()),
        flat_projection(layout.width() as f32, layout.height() as f32),
        Tonemapping::None,
        DebandDither::Disabled,
        Msaa::Off,
        RenderLayers::layer(QUILT_LAYER),
        ChildOf(window),
    ));
    commands.spawn((
        Camera2d,
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        RenderTarget::Window(WindowRef::Entity(window)),
        flat_projection(2.0, 2.0),
        Tonemapping::None,
        DebandDither::Disabled,
        Msaa::Off,
        RenderLayers::layer(DISPLAY_LAYER),
        ChildOf(window),
    ));
    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(2.0, 2.0))),
        MeshMaterial2d(material),
        RenderLayers::layer(DISPLAY_LAYER),
        ChildOf(window),
    ));
    info!(
        "Looking Glass prototype: {} views, {}x{} per view, {}x{} quilt; move its window to the panel and press F11",
        layout.count(),
        layout.view_width,
        layout.view_height,
        layout.width(),
        layout.height()
    );
}

fn sync_views(
    config: Res<OutputConfig>,
    lifecycle: Res<AvatarLifecycle>,
    control: Res<AvatarCameraControl>,
    mut cameras: Query<(
        &ViewCamera,
        &mut Camera,
        &mut Projection,
        &mut Transform,
        &mut GlobalTransform,
    )>,
    mut tiles: Query<&mut Visibility, With<QuiltTile>>,
    mut last_pose: Local<Option<CameraControlPose>>,
) {
    // Controls and reset run before transform propagation. Initial automatic
    // framing may become available in this or the next frame; until then the
    // quilt is black, not a retained image of a previous avatar.
    let pose = control.current_for(lifecycle.current_generation());
    if *last_pose == pose {
        return;
    }
    *last_pose = pose;
    for mut visibility in &mut tiles {
        *visibility = if pose.is_some() {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
    for (index, mut camera, mut projection, mut transform, mut global) in &mut cameras {
        camera.is_active = pose.is_some();
        let Some(pose) = pose else {
            continue;
        };
        let center = pose.transform();
        let focus_distance = (pose.target() - center.translation).dot(*center.forward());
        let perspective = PerspectiveProjection {
            fov: FIXED_VERTICAL_FOV,
            aspect_ratio: config.calibration.width as f32 / config.calibration.height as f32,
            ..default()
        };
        let (next, lens) = view_pose(
            center,
            focus_distance,
            perspective,
            index.0,
            config.layout.count(),
            config.calibration.view_cone,
            config.depth_scale,
        );
        *transform = next;
        // This system runs after propagation. All output entities are direct
        // children of an identity-transform window, so publish globals too.
        *global = GlobalTransform::from(next);
        *projection = Projection::custom(lens);
    }
}

fn window_keys(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    mut windows: Query<(Entity, &mut Window, &OutputWindow)>,
    mut materials: ResMut<Assets<InterlaceMaterial>>,
) {
    for (entity, mut window, output) in &mut windows {
        if !window.focused {
            continue;
        }
        if keys.just_pressed(KeyCode::Escape) {
            // Despawning the root drops every camera and image/material handle.
            commands.entity(entity).despawn();
            continue;
        }
        if keys.just_pressed(KeyCode::F11) {
            window.mode = match window.mode {
                WindowMode::Windowed => WindowMode::BorderlessFullscreen(MonitorSelection::Current),
                _ => WindowMode::Windowed,
            };
        }
        if keys.just_pressed(KeyCode::F10)
            && let Some(mut material) = materials.get_mut(&output.0)
        {
            material.uniforms.flags.w ^= 8;
        }
    }
}

fn close_with_main(
    mut commands: Commands,
    main: Query<(), With<PrimaryWindow>>,
    windows: Query<Entity, With<OutputWindow>>,
) {
    if main.is_empty() {
        for window in &windows {
            commands.entity(window).despawn();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closing_output_despawns_all_its_cameras_but_not_the_main_window() {
        let mut app = App::new();
        let main = app
            .world_mut()
            .spawn((Window::default(), PrimaryWindow))
            .id();
        let output = app.world_mut().spawn(OutputWindow(Handle::default())).id();
        let child = app.world_mut().spawn((ViewCamera(0), ChildOf(output))).id();
        app.world_mut().entity_mut(output).despawn();
        assert!(app.world().get_entity(child).is_err());
        assert!(app.world().get_entity(main).is_ok());
    }
}
