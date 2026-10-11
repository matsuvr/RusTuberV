//! A model-free scene: foreground cube, focus-plane sphere, background cube.
//! Usage: cargo run --example minimal -- /path/to/visual.json
#![forbid(unsafe_code)]

use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use bevy::window::{MonitorSelection, PrimaryWindow, WindowMode, WindowResolution};
use bevy_looking_glass::*;

#[derive(Component)]
struct PreviewCamera;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os()
        .nth(1)
        .ok_or("usage: minimal /path/to/visual.json")?;
    let calibration = parse_calibration(&std::fs::read(path)?)?;
    let screen = calibration.screen_size();
    let mut config = LookingGlassConfig::new(
        LookingGlassOutput::LightField {
            calibration,
            layout: ViewLayout::new(8, 6, 210, 280)?,
            depth_scale: 1.0,
        },
        RenderLayers::layer(0),
        1,
    );
    // A smaller window is a preview only; fullscreen on the native panel for optics.
    config.window.resolution =
        WindowResolution::new(screen.x / 2, screen.y / 2).with_scale_factor_override(1.0);
    config.window.title = "Looking Glass — F10 quilt / F11 fullscreen / Escape stop".into();
    let mut control = LookingGlassControl::default();
    control.start(config);
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(LookingGlassPlugin)
        .insert_resource(control)
        .insert_resource(bevy::winit::WinitSettings::continuous())
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                move_camera,
                output_keys.before(LookingGlassSystems::Control),
            ),
        )
        .add_systems(Last, close_with_main)
        .run();
    Ok(())
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut view: ResMut<LookingGlassView>,
) {
    let transform = Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y);
    view.pose = Some(ViewPose {
        transform,
        focus_distance: 5.0,
        fov_y: 45_f32.to_radians(),
        near: 0.1,
        far: 100.0,
    });
    commands.spawn((Camera3d::default(), transform, PreviewCamera));
    for (position, color) in [
        (Vec3::new(-0.8, 0.0, 1.3), Color::srgb(0.95, 0.2, 0.15)),
        (Vec3::new(0.8, 0.0, -1.3), Color::srgb(0.15, 0.35, 0.95)),
    ] {
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(0.8, 0.8, 0.8))),
            MeshMaterial3d(materials.add(color)),
            Transform::from_translation(position),
        ));
    }
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(0.35))),
        MeshMaterial3d(materials.add(Color::srgb(0.2, 0.9, 0.4))),
    ));
    commands.spawn((
        PointLight {
            intensity: 2_000_000.0,
            ..default()
        },
        Transform::from_xyz(2.0, 4.0, 4.0),
    ));
    info!("Arrow keys: orbit/dolly; F10: quilt; F11: fullscreen; Escape: stop output");
}

fn move_camera(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut view: ResMut<LookingGlassView>,
    mut preview: Query<&mut Transform, With<PreviewCamera>>,
) {
    let Some(pose) = view.pose.as_mut() else {
        return;
    };
    let turn =
        f32::from(keys.pressed(KeyCode::ArrowLeft)) - f32::from(keys.pressed(KeyCode::ArrowRight));
    let dolly =
        f32::from(keys.pressed(KeyCode::ArrowDown)) - f32::from(keys.pressed(KeyCode::ArrowUp));
    pose.focus_distance = (pose.focus_distance + dolly * time.delta_secs() * 2.0).clamp(2.0, 12.0);
    let direction =
        Quat::from_rotation_y(turn * time.delta_secs()) * pose.transform.translation.normalize();
    pose.transform = Transform::from_translation(direction * pose.focus_distance)
        .looking_at(Vec3::ZERO, Vec3::Y);
    for mut transform in &mut preview {
        *transform = pose.transform;
    }
}

fn output_keys(
    keys: Res<ButtonInput<KeyCode>>,
    status: Res<LookingGlassStatus>,
    mut control: ResMut<LookingGlassControl>,
    mut windows: Query<&mut Window>,
) {
    if let Some(error) = &status.error {
        error!("{error}");
    }
    let Some(entity) = status.window else {
        return;
    };
    let Ok(mut window) = windows.get_mut(entity) else {
        return;
    };
    if !window.focused {
        return;
    }
    if keys.just_pressed(KeyCode::Escape) {
        control.stop();
    }
    if keys.just_pressed(KeyCode::F10) {
        control.set_quilt_preview(!status.quilt_preview);
    }
    if keys.just_pressed(KeyCode::F11) {
        window.mode = match window.mode {
            WindowMode::Windowed => WindowMode::BorderlessFullscreen(MonitorSelection::Current),
            _ => WindowMode::Windowed,
        };
    }
}

fn close_with_main(
    mut commands: Commands,
    main: Query<(), With<PrimaryWindow>>,
    status: Res<LookingGlassStatus>,
) {
    if main.is_empty()
        && let Some(window) = status.window
        && let Ok(mut entity) = commands.get_entity(window)
    {
        entity.despawn();
    }
}
