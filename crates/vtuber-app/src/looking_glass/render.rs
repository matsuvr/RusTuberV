//! Bevy resource/entity effects for the optional output. No image readback.

use bevy::asset::embedded_asset;
use bevy::camera::{
    CameraOutputMode, CameraUpdateSystems, ClearColorConfig, Exposure, RenderTarget, ScalingMode,
    visibility::{RenderLayers, VisibilitySystems},
};
use bevy::ecs::schedule::SingleThreadedExecutor;
use bevy::light::cluster::ClusterConfig;
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::renderer::RenderDevice;
use bevy::render::view::{DebandDither, NoIndirectDrawing, Tonemapping};
use bevy::render::{render_resource::AsBindGroup, storage::ShaderBuffer};
use bevy::shader::ShaderRef;
use bevy::sprite_render::{Material2d, Material2dPlugin};
use bevy::window::{MonitorSelection, PrimaryWindow, WindowMode, WindowRef, WindowResolution};
use vtuber_avatar::{
    AVATAR_RENDER_LAYER, AvatarCameraControl, AvatarLifecycle, CameraControlPose,
    FIXED_VERTICAL_FOV,
};

use super::{
    ActiveOutput, LookingGlassControl,
    optics::{InterlaceUniforms, view_pose},
    selected_output,
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

pub(super) fn install(app: &mut App) {
    embedded_asset!(app, "interlace.wesl");
    view_array::install(app);
    app.add_plugins(Material2dPlugin::<InterlaceMaterial>::default())
        .add_systems(
            Update,
            (apply_request, window_keys)
                .chain()
                .after(crate::orchestrator::process_ui_actions_system),
        )
        .add_systems(
            PostUpdate,
            sync_views
                .run_if(resource_exists::<ActiveOutput>)
                .after(TransformSystems::Propagate)
                .before(CameraUpdateSystems)
                .before(VisibilitySystems::VisibilityPropagate),
        )
        .add_systems(Last, close_with_main);
    if let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) {
        render_app.add_systems(bevy::render::ExtractSchedule, sync_executor);
    }
}

fn sync_executor(
    mut commands: Commands,
    config: bevy::render::Extract<Option<Res<ActiveOutput>>>,
    mut enabled: Local<bool>,
) {
    let multiview = matches!(config.as_deref(), Some(ActiveOutput::LightField(_)));
    if *enabled != multiview {
        *enabled = multiview;
        commands.queue(move |world: &mut World| {
            if let Some(mut schedules) = world.get_resource_mut::<bevy::ecs::schedule::Schedules>()
                && let Some(schedule) = schedules.get_mut(bevy::core_pipeline::Core3d)
            {
                if multiview {
                    schedule.set_executor(SingleThreadedExecutor::new());
                } else {
                    schedule.set_executor(bevy::ecs::schedule::MultiThreadedExecutor::new());
                }
            }
        });
    }
}

fn apply_request(world: &mut World) {
    let Some(mut control) = world.get_resource_mut::<LookingGlassControl>() else {
        return;
    };
    let pending = control.pending.take();
    let startup = control.startup.take();
    let windows: Vec<Entity> = world
        .query_filtered::<Entity, With<OutputWindow>>()
        .iter(world)
        .collect();
    if pending.is_none() && startup.is_none() {
        if windows.is_empty() {
            world.remove_resource::<ActiveOutput>();
        }
        world
            .resource_mut::<crate::ui_model::UiViewModel>()
            .looking_glass
            .running = !windows.is_empty();
        return;
    }
    if matches!(pending, Some(None)) {
        for window in windows {
            world.despawn(window);
        }
        world.remove_resource::<ActiveOutput>();
        let mut vm = world.resource_mut::<crate::ui_model::UiViewModel>();
        vm.looking_glass.running = false;
        vm.looking_glass.error = None;
        return;
    }
    let selection = pending.flatten();
    let result = if let Some(selection) = &selection {
        world
            .resource::<crate::ui_model::UiViewModel>()
            .looking_glass
            .devices
            .iter()
            .find(|preset| preset.id == selection.device_id)
            .ok_or(super::LookingGlassError::Invalid(
                "unknown Looking Glass model",
            ))
            .and_then(|preset| selected_output(preset, selection))
            .map_err(|error| error.to_string())
    } else {
        startup.ok_or_else(|| "missing Looking Glass startup configuration".to_string())
    };
    let result = result.and_then(|config| {
        let layout = config.layout();
        let device = world
            .get_resource::<RenderDevice>()
            .ok_or("render device is unavailable")?;
        let limits = device.limits();
        if layout.count() > limits.max_texture_array_layers
            || layout.view_width > limits.max_texture_dimension_2d
            || layout.view_height > limits.max_texture_dimension_2d
        {
            return Err("Looking Glass views exceed this GPU's texture limits".into());
        }
        if let Some(selection) = &selection {
            world
                .resource_mut::<crate::settings::AppSettings>()
                .save_looking_glass(selection.clone())
                .map_err(|error| error.to_string())?;
        }
        Ok(config)
    });
    match result {
        Ok(config) => {
            for window in windows {
                world.despawn(window);
            }
            world.insert_resource(config);
            let setup_result = world.run_system_cached(setup);
            let mut vm = world.resource_mut::<crate::ui_model::UiViewModel>();
            vm.looking_glass.running = setup_result.is_ok();
            vm.looking_glass.error = setup_result.err().map(|error| error.to_string());
            if let Some(selection) = selection {
                vm.looking_glass.preferences = selection;
            }
        }
        Err(error) => {
            world
                .resource_mut::<crate::ui_model::UiViewModel>()
                .looking_glass
                .error = Some(error);
        }
    }
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
        // Bevy 0.20 omits ViewTarget for a clearing target with only Skip
        // cameras. Keep the first output to establish the shared attachment;
        // the remaining views need only the capture copy, not an output blit.
        output_mode: if index == 0 {
            CameraOutputMode::default()
        } else {
            CameraOutputMode::Skip
        },
        ..default()
    }
}

fn setup(
    mut commands: Commands,
    config: Res<ActiveOutput>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<InterlaceMaterial>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
) {
    let layout = config.layout();
    let screen = config.screen();
    let (uniforms, cells) = match &*config {
        ActiveOutput::LightField(config) => (
            config.calibration.uniforms(layout),
            config.calibration.cells.clone(),
        ),
        ActiveOutput::Hld { safe_area, .. } => (
            InterlaceUniforms {
                optics: Vec4::ZERO,
                // HLD safe area: 108px sides, 100px top/bottom at 2160x3840.
                screen: Vec4::new(screen.x as f32, screen.y as f32, safe_area.x, safe_area.y),
                grid: UVec4::new(1, 1, layout.view_width, layout.view_height),
                flags: UVec4::new(0, 0, 0, 16),
            },
            vec![Vec4::ZERO],
        ),
    };
    let views = images.add(view_array::image(layout));
    // One logical target makes Bevy share its small main/depth textures across
    // views. The first camera writes it to keep the target attachment available;
    // the other cameras skip that blit. Only the array is sampled for display.
    let scratch = images.add(Image::new_target_texture(
        layout.view_width,
        layout.view_height,
        VIEW_FORMAT,
        None,
    ));
    let material = materials.add(InterlaceMaterial {
        uniforms,
        views: views.clone(),
        cells: buffers.add(ShaderBuffer::from(cells)),
    });
    let window = commands
        .spawn((
            Window {
                title: "RusTuberV — Looking Glass (F11: fullscreen / F10: quilt / Esc: close)"
                    .into(),
                resolution: WindowResolution::new(screen.x.div_ceil(2), screen.y.div_ceil(2))
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
            Camera {
                clear_color: ClearColorConfig::Custom(
                    if matches!(*config, ActiveOutput::Hld { .. }) {
                        Color::WHITE
                    } else {
                        Color::BLACK
                    },
                ),
                ..view_camera(index, layout.count())
            },
            RenderTarget::Image(scratch.clone().into()),
            Msaa::Off,
            Exposure::BLENDER,
            Tonemapping::None,
            DebandDither::Disabled,
            // Only a few lights cover these small views. Avoid constructing a
            // 3D cluster grid and GPU indirect arguments for every camera.
            ClusterConfig::Single,
            NoIndirectDrawing,
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
        "Looking Glass array output: {} views, {}x{} per view, {}x{} virtual quilt; one shared scratch target and output blit, no quilt assembly; move its window to the panel and press F11",
        layout.count(),
        layout.view_width,
        layout.view_height,
        layout.width(),
        layout.height()
    );
}

fn sync_views(
    config: Res<ActiveOutput>,
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
    if *last_pose == pose && !config.is_changed() {
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
            aspect_ratio: match &*config {
                ActiveOutput::LightField(_) => config.screen().x as f32 / config.screen().y as f32,
                ActiveOutput::Hld { screen, safe_area } => {
                    (screen.x as f32 * safe_area.x) / (screen.y as f32 * safe_area.y)
                }
            },
            ..default()
        };
        let (count, cone, depth_scale) = match &*config {
            ActiveOutput::LightField(config) => (
                config.layout.count(),
                config.calibration.view_cone,
                config.depth_scale,
            ),
            ActiveOutput::Hld { .. } => (1, 0.0, 0.0),
        };
        let (next, lens) = view_pose(
            center,
            focus_distance,
            perspective,
            view.index,
            count,
            cone,
            depth_scale,
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
            && material.uniforms.flags.w & 16 == 0
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
    use super::super::OutputConfig;
    use super::*;

    #[test]
    fn stopping_from_settings_releases_the_window_and_its_views() {
        let mut app = App::new();
        app.init_resource::<LookingGlassControl>()
            .init_resource::<crate::ui_model::UiViewModel>();
        let window = app.world_mut().spawn(OutputWindow(Handle::default())).id();
        let camera = app
            .world_mut()
            .spawn((Camera3d::default(), ChildOf(window)))
            .id();
        app.world_mut().resource_mut::<LookingGlassControl>().stop();
        apply_request(app.world_mut());
        assert!(app.world().get_entity(window).is_err());
        assert!(app.world().get_entity(camera).is_err());
        assert!(
            !app.world()
                .resource::<crate::ui_model::UiViewModel>()
                .looking_glass
                .running
        );
        assert!(!app.world().contains_resource::<ActiveOutput>());
    }

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
            .insert_resource(ActiveOutput::LightField(OutputConfig {
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
            }))
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
        assert!(
            world
                .query_filtered::<&Visibility, With<OutputMesh>>()
                .iter(world)
                .all(|visibility| *visibility == Visibility::Hidden)
        );
    }

    #[test]
    fn every_view_clears_scratch_in_order_and_first_keeps_the_target_attachment() {
        let mut previous_order = None;
        for index in 0..66 {
            let camera = view_camera(index, 66);
            assert!(camera.order < 0);
            assert!(previous_order.is_none_or(|order| order < camera.order));
            if index == 0 {
                assert!(matches!(camera.output_mode, CameraOutputMode::Write { .. }));
            } else {
                assert!(matches!(camera.output_mode, CameraOutputMode::Skip));
            }
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
