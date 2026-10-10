//! Bevy resource/entity effects for the optional output. No image readback.

use bevy::asset::embedded_asset;
use bevy::camera::{
    CameraOutputMode, CameraUpdateSystems, ClearColorConfig, Exposure, RenderTarget, ScalingMode,
    visibility::{RenderLayers, VisibilitySystems},
};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::view::{DebandDither, Tonemapping};
use bevy::render::{render_resource::AsBindGroup, storage::ShaderBuffer};
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
    view_array::{self, VIEW_FORMAT, ViewCamera},
};

// Layers 0/1 belong to the avatar/ground. This layer contains only the output
// mesh; it is not visible to avatar, preview or NDI cameras.
const DISPLAY_LAYER: usize = 3;

#[derive(Component)]
struct OutputMesh;
#[derive(Component)]
struct OutputWindow(Handle<InterlaceMaterial>);

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct InterlaceMaterial {
    #[uniform(0)]
    uniforms: InterlaceUniforms,
    #[texture(1, dimension = "2d_array")]
    #[sampler(2)]
    views: Handle<Image>,
    #[storage(3, read_only)]
    cells: Handle<ShaderBuffer>,
}

impl Material2d for InterlaceMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://vtuber_app/looking_glass/interlace.wesl".into()
    }
}

pub(super) fn install(app: &mut App, config: OutputConfig) {
    embedded_asset!(app, "interlace.wesl");
    view_array::install(app);
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

fn flat_projection(width: f32, height: f32) -> Projection {
    Projection::Orthographic(OrthographicProjection {
        scaling_mode: ScalingMode::Fixed { width, height },
        ..OrthographicProjection::default_2d()
    })
}

fn view_camera(index: u32, count: u32) -> Camera {
    Camera {
        // Distinct orders on one shared target prevent Bevy's ambiguous-order
        // warning, and ensure each layer is captured before scratch is reused.
        // Every view runs before the output camera (order 0).
        order: index as isize - count as isize - 1,
        is_active: false,
        clear_color: ClearColorConfig::Custom(Color::BLACK),
        // The capture system copies the completed main texture, so the normal
        // per-camera fullscreen output blit would be redundant.
        output_mode: CameraOutputMode::Skip,
        ..default()
    }
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
    let views = images.add(view_array::image(layout));
    // One logical target makes Bevy share its small main/depth textures across
    // views. The target image itself is never written (CameraOutputMode::Skip).
    // Keep it: RenderTarget::None with a clearing camera has no ViewTarget.
    let scratch = images.add(Image::new_target_texture(
        layout.view_width,
        layout.view_height,
        VIEW_FORMAT,
        None,
    ));
    let material = materials.add(InterlaceMaterial {
        uniforms: calibration.uniforms(layout),
        views: views.clone(),
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
        commands.spawn((
            Camera3d::default(),
            view_camera(index, layout.count()),
            RenderTarget::Image(scratch.clone().into()),
            Msaa::Off,
            Exposure::BLENDER,
            Tonemapping::None,
            DebandDither::Disabled,
            RenderLayers::layer(AVATAR_RENDER_LAYER),
            ViewCamera {
                index,
                array: views.clone(),
            },
            ChildOf(window),
        ));
    }
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
        Visibility::Hidden,
        OutputMesh,
        RenderLayers::layer(DISPLAY_LAYER),
        ChildOf(window),
    ));
    info!(
        "Looking Glass array output: {} views, {}x{} per view, {}x{} virtual quilt; one shared scratch target, no per-view output blits or quilt assembly; move its window to the panel and press F11",
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
    mut output: Query<&mut Visibility, With<OutputMesh>>,
    mut last_pose: Local<Option<CameraControlPose>>,
) {
    // Controls and reset run before transform propagation. Hide the output
    // mesh when framing is unavailable: disabled cameras do not overwrite the
    // array, and a previous avatar must not remain visible on the display.
    let pose = control.current_for(lifecycle.current_generation());
    if *last_pose == pose {
        return;
    }
    *last_pose = pose;
    for mut visibility in &mut output {
        *visibility = output_visibility(pose.is_some());
    }
    for (view, mut camera, mut projection, mut transform, mut global) in &mut cameras {
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
            view.index,
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

fn output_visibility(has_pose: bool) -> Visibility {
    if has_pose {
        Visibility::Visible
    } else {
        Visibility::Hidden
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
        let child = app
            .world_mut()
            .spawn((
                ViewCamera {
                    index: 0,
                    array: Handle::default(),
                },
                ChildOf(output),
            ))
            .id();
        app.world_mut().entity_mut(output).despawn();
        assert!(app.world().get_entity(child).is_err());
        assert!(app.world().get_entity(main).is_ok());
    }

    #[test]
    fn setup_has_one_array_one_shared_scratch_and_no_quilt_camera_or_sprites() {
        use super::super::optics::{Calibration, Layout};

        let mut app = App::new();
        app.init_resource::<Assets<Image>>()
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<InterlaceMaterial>>()
            .init_resource::<Assets<ShaderBuffer>>()
            .insert_resource(OutputConfig {
                layout: Layout {
                    columns: 11,
                    rows: 6,
                    view_width: 186,
                    view_height: 341,
                },
                // Entity/resource topology only; not a hardware calibration.
                calibration: Calibration {
                    width: 1440,
                    height: 2560,
                    view_cone: 40_f32.to_radians(),
                    optics: Vec4::ZERO,
                    flags: UVec4::ZERO,
                    cells: vec![Vec4::ZERO],
                },
                depth_scale: 1.0,
            })
            .add_systems(Startup, setup);
        app.update();
        let world = app.world_mut();
        let mut views = world.query::<(&ViewCamera, &RenderTarget)>();
        let mut scratch_id = None;
        let mut array_id = None;
        let mut indices = Vec::new();
        for (view, target) in views.iter(world) {
            let target_id = target.as_image().map(Handle::id);
            assert!(target_id.is_some());
            if let Some(expected) = scratch_id {
                assert_eq!(target_id, Some(expected));
            } else {
                scratch_id = target_id;
            }
            if let Some(expected) = array_id {
                assert_eq!(view.array.id(), expected);
            } else {
                array_id = Some(view.array.id());
            }
            indices.push(view.index);
        }
        indices.sort_unstable();
        assert_eq!(indices, (0..66).collect::<Vec<_>>());
        assert_ne!(scratch_id, array_id);
        assert_eq!(world.resource::<Assets<Image>>().len(), 2);
        assert_eq!(world.query::<&Camera2d>().iter(world).count(), 1);
        assert_eq!(world.query::<&Sprite>().iter(world).count(), 0);
        assert!(world
            .query_filtered::<&Visibility, With<OutputMesh>>()
            .iter(world)
            .all(|visibility| *visibility == Visibility::Hidden));
    }

    #[test]
    fn every_view_clears_scratch_and_skips_the_output_blit_in_a_distinct_order() {
        let mut previous_order = None;
        for index in 0..66 {
            let camera = view_camera(index, 66);
            assert!(camera.order < 0);
            assert!(previous_order.is_none_or(|order| order < camera.order));
            assert!(matches!(camera.output_mode, CameraOutputMode::Skip));
            assert!(matches!(
                camera.clear_color,
                ClearColorConfig::Custom(color) if color == Color::BLACK
            ));
            assert!(!camera.is_active);
            previous_order = Some(camera.order);
        }
    }

    #[test]
    fn missing_pose_hides_retained_array_pixels() {
        assert_eq!(output_visibility(true), Visibility::Visible);
        assert_eq!(output_visibility(false), Visibility::Hidden);
    }
}
