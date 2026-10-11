//! Bevy resource/entity effects for the optional output. No image readback.

use crate::{
    ActiveOutput, LookingGlassControl, LookingGlassOutput, LookingGlassStatus, LookingGlassSystems,
    LookingGlassView, ViewPose,
    optics::{InterlaceUniforms, view_pose},
    view_array::{self, VIEW_FORMAT, ViewCamera},
};
use bevy::asset::embedded_asset;
use bevy::camera::{
    CameraOutputMode, CameraUpdateSystems, ClearColorConfig, RenderTarget, ScalingMode,
    visibility::{RenderLayers, VisibilitySystems},
};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::renderer::RenderDevice;
use bevy::render::view::{DebandDither, NoIndirectDrawing, Tonemapping};
use bevy::render::{render_resource::AsBindGroup, storage::ShaderBuffer};
use bevy::shader::ShaderRef;
use bevy::sprite_render::{Material2d, Material2dPlugin};
use bevy::window::WindowRef;

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
        "embedded://bevy_looking_glass/interlace.wesl".into()
    }
}

pub(super) fn install(app: &mut App) {
    embedded_asset!(app, "interlace.wesl");
    view_array::install(app);
    app.add_plugins(Material2dPlugin::<InterlaceMaterial>::default())
        .add_systems(Update, apply_request.in_set(LookingGlassSystems::Control))
        .add_systems(
            PostUpdate,
            sync_views
                .in_set(LookingGlassSystems::SyncViews)
                .run_if(resource_exists::<ActiveOutput>)
                .after(TransformSystems::Propagate)
                .before(CameraUpdateSystems)
                .before(VisibilitySystems::VisibilityPropagate),
        );
}

fn apply_request(world: &mut World) {
    let Some(mut control) = world.get_resource_mut::<LookingGlassControl>() else {
        return;
    };
    let pending = control.pending.take();
    let preview = control.quilt_preview;
    let windows: Vec<Entity> = world
        .query_filtered::<Entity, With<OutputWindow>>()
        .iter(world)
        .collect();
    if windows.is_empty() {
        world.remove_resource::<ActiveOutput>();
    }
    if let Some(request) = pending {
        let result = request.as_ref().map_or(Ok(()), |config| {
            config
                .output
                .validate()
                .map_err(|error| error.to_string())?;
            if config
                .scene_layers
                .intersects(&RenderLayers::layer(config.display_layer))
            {
                return Err("display_layer must be separate from scene_layers".to_string());
            }
            let layout = config.output.layout();
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
            Ok(())
        });
        match result {
            Ok(()) => {
                for window in windows {
                    world.despawn(window);
                }
                world.remove_resource::<ActiveOutput>();
                world.resource_mut::<LookingGlassStatus>().error = None;
                if let Some(config) = request {
                    world.insert_resource(ActiveOutput(config));
                    if let Err(error) = world.run_system_cached(setup) {
                        world.remove_resource::<ActiveOutput>();
                        world.resource_mut::<LookingGlassStatus>().error = Some(error.to_string());
                    }
                }
            }
            Err(error) => world.resource_mut::<LookingGlassStatus>().error = Some(error),
        }
    }
    let window = world
        .query_filtered::<Entity, With<OutputWindow>>()
        .iter(world)
        .next();
    let light_field = world
        .get_resource::<ActiveOutput>()
        .is_some_and(|config| matches!(config.0.output, LookingGlassOutput::LightField { .. }));
    let material = window
        .and_then(|entity| world.get::<OutputWindow>(entity))
        .map(|o| o.0.clone());
    if let Some(handle) = material
        && let Some(mut materials) = world.get_resource_mut::<Assets<InterlaceMaterial>>()
        && materials.get(&handle).is_some_and(|material| {
            (material.uniforms.flags.w & 8 != 0) != (preview && light_field)
        })
        && let Some(mut material) = materials.get_mut(&handle)
    {
        let flags = &mut material.uniforms.flags.w;
        *flags = (*flags & !8) | if preview && light_field { 8 } else { 0 };
    }
    let mut status = world.resource_mut::<LookingGlassStatus>();
    status.window = window;
    status.light_field = light_field;
    status.quilt_preview = preview && light_field;
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
    let config = &config.0;
    let layout = config.output.layout();
    let screen = config.output.screen_size();
    let (uniforms, cells) = match &config.output {
        LookingGlassOutput::LightField { calibration, .. } => {
            (calibration.uniforms(layout), calibration.cells.clone())
        }
        LookingGlassOutput::Hld { safe_area, .. } => (
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
            config.window.clone(),
            OutputWindow(material.clone()),
            Transform::default(),
            Visibility::Visible,
        ))
        .id();

    for index in 0..layout.count() {
        let mut camera = commands.spawn((
            Camera3d::default(),
            Camera {
                clear_color: if matches!(config.output, LookingGlassOutput::Hld { .. }) {
                    ClearColorConfig::Custom(Color::WHITE)
                } else {
                    config.view_render.clear_color
                },
                ..view_camera(index, layout.count())
            },
            RenderTarget::Image(scratch.clone().into()),
            Msaa::Off,
            config.view_render.exposure,
            config.view_render.tonemapping,
            config.view_render.deband_dither,
            config.view_render.cluster_config,
            config.scene_layers.clone(),
            ViewCamera {
                index,
                array: views.clone(),
            },
            ChildOf(window),
        ));
        if config.view_render.no_indirect_drawing {
            camera.insert(NoIndirectDrawing);
        }
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
        RenderLayers::layer(config.display_layer),
        ChildOf(window),
    ));
    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(2.0, 2.0))),
        MeshMaterial2d(material),
        Visibility::Hidden,
        OutputMesh,
        RenderLayers::layer(config.display_layer),
        ChildOf(window),
    ));
    info!(
        "Looking Glass array output: {} views, {}x{} per view, {}x{} virtual quilt; one shared scratch target and output blit, no quilt assembly",
        layout.count(),
        layout.view_width,
        layout.view_height,
        layout.width(),
        layout.height()
    );
}

fn sync_views(
    config: Res<ActiveOutput>,
    view: Res<LookingGlassView>,
    mut cameras: Query<(
        &ViewCamera,
        &mut Camera,
        &mut Projection,
        &mut Transform,
        &mut GlobalTransform,
    )>,
    mut output: Query<&mut Visibility, With<OutputMesh>>,
    mut last_pose: Local<Option<ViewPose>>,
) {
    // Disabled cameras retain array pixels. Hide the mesh when the host has no pose.
    let pose = view.pose;
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
        let center = pose.transform;
        let focus_distance = pose.focus_distance;
        let perspective = PerspectiveProjection {
            fov: pose.fov_y,
            near: pose.near,
            far: pose.far,
            aspect_ratio: match &config.0.output {
                LookingGlassOutput::LightField { calibration, .. } => {
                    calibration.width as f32 / calibration.height as f32
                }
                LookingGlassOutput::Hld { screen, safe_area } => {
                    (screen.x as f32 * safe_area.x) / (screen.y as f32 * safe_area.y)
                }
            },
            ..default()
        };
        let (count, cone, depth_scale) = match &config.0.output {
            LookingGlassOutput::LightField {
                layout,
                calibration,
                depth_scale,
            } => (layout.count(), calibration.view_cone, *depth_scale),
            LookingGlassOutput::Hld { .. } => (1, 0.0, 0.0),
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;
    use crate::{Calibration, LookingGlassConfig, ViewLayout as Layout};
    use bevy::window::PrimaryWindow;

    #[test]
    fn stopping_from_settings_releases_the_window_and_its_views() {
        let mut app = App::new();
        app.init_resource::<LookingGlassControl>()
            .init_resource::<LookingGlassStatus>();
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
                .resource::<LookingGlassStatus>()
                .window
                .is_some()
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
        let mut app = App::new();
        app.init_resource::<Assets<Image>>()
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<InterlaceMaterial>>()
            .init_resource::<Assets<ShaderBuffer>>()
            .insert_resource(ActiveOutput(LookingGlassConfig::new(
                LookingGlassOutput::LightField {
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
                },
                RenderLayers::layer(0),
                3,
            )))
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

    fn hld_config() -> ActiveOutput {
        ActiveOutput(LookingGlassConfig::new(
            LookingGlassOutput::Hld {
                screen: UVec2::new(32, 64),
                safe_area: Vec2::splat(0.9),
            },
            RenderLayers::layer(2),
            5,
        ))
    }

    #[test]
    fn generic_pose_disables_retained_pixels_and_output_survives_without_primary_window() {
        let mut app = App::new();
        app.init_resource::<Assets<Image>>()
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<InterlaceMaterial>>()
            .init_resource::<Assets<ShaderBuffer>>()
            .init_resource::<LookingGlassControl>()
            .init_resource::<LookingGlassStatus>()
            .init_resource::<LookingGlassView>()
            .insert_resource(hld_config())
            .add_systems(Startup, setup)
            .add_systems(Update, (apply_request, sync_views).chain());
        let pose = ViewPose {
            transform: Transform::from_xyz(0.0, 1.0, 5.0),
            focus_distance: 5.0,
            fov_y: 0.6,
            near: 0.2,
            far: 50.0,
        };
        app.world_mut().resource_mut::<LookingGlassView>().pose = Some(pose);
        app.update();
        let window = app.world().resource::<LookingGlassStatus>().window.unwrap();
        let world = app.world_mut();
        let (camera, transform, layers) = world
            .query_filtered::<(&Camera, &Transform, &RenderLayers), With<ViewCamera>>()
            .single(world)
            .unwrap();
        assert!(camera.is_active);
        assert_eq!(*transform, pose.transform);
        assert_eq!(*layers, RenderLayers::layer(2));
        world.resource_mut::<LookingGlassView>().pose = None;
        app.update();
        let world = app.world_mut();
        assert!(world.get_entity(window).is_ok());
        assert!(
            !world
                .query_filtered::<&Camera, With<ViewCamera>>()
                .single(world)
                .unwrap()
                .is_active
        );
        assert_eq!(
            *world
                .query_filtered::<&Visibility, With<OutputMesh>>()
                .single(world)
                .unwrap(),
            Visibility::Hidden
        );
    }

    #[test]
    fn stopping_releases_owned_assets_and_preserves_host_assets() {
        use bevy::asset::AssetApp;
        let mut app = App::new();
        app.add_plugins((bevy::app::TaskPoolPlugin::default(), AssetPlugin::default()))
            .init_asset::<Image>()
            .init_asset::<Mesh>()
            .init_asset::<InterlaceMaterial>()
            .init_asset::<ShaderBuffer>()
            .init_resource::<LookingGlassControl>()
            .init_resource::<LookingGlassStatus>()
            .insert_resource(hld_config())
            .add_systems(Startup, setup)
            .add_systems(Update, apply_request);
        let host_image = app
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(Image::default());
        let host_window = app
            .world_mut()
            .spawn((Window::default(), PrimaryWindow))
            .id();
        app.update();
        assert_eq!(app.world().resource::<Assets<Image>>().len(), 3);
        app.world_mut().resource_mut::<LookingGlassControl>().stop();
        // Asset handle drops propagate from material -> array/buffer over successive frames.
        for _ in 0..5 {
            app.update();
        }
        assert!(app.world().get_entity(host_window).is_ok());
        assert!(
            app.world()
                .resource::<Assets<Image>>()
                .contains(host_image.id())
        );
        assert_eq!(app.world().resource::<Assets<Image>>().len(), 1);
        assert!(app.world().resource::<Assets<Mesh>>().is_empty());
        assert!(
            app.world()
                .resource::<Assets<InterlaceMaterial>>()
                .is_empty()
        );
        assert!(app.world().resource::<Assets<ShaderBuffer>>().is_empty());
    }

    #[test]
    fn rejected_replacement_keeps_existing_output_and_reports_error() {
        let mut app = App::new();
        app.init_resource::<LookingGlassControl>()
            .init_resource::<LookingGlassStatus>()
            .insert_resource(hld_config());
        let window = app.world_mut().spawn(OutputWindow(Handle::default())).id();
        let mut config = hld_config().0;
        config.output = LookingGlassOutput::Hld {
            screen: UVec2::ZERO,
            safe_area: Vec2::ONE,
        };
        app.world_mut()
            .resource_mut::<LookingGlassControl>()
            .start(config);
        apply_request(app.world_mut());
        let status = app.world().resource::<LookingGlassStatus>();
        assert_eq!(status.window, Some(window));
        assert!(status.error.is_some());
        assert!(app.world().contains_resource::<ActiveOutput>());
        app.world_mut().resource_mut::<LookingGlassControl>().stop();
        apply_request(app.world_mut());
        assert!(app.world().resource::<LookingGlassStatus>().error.is_none());
    }
}
