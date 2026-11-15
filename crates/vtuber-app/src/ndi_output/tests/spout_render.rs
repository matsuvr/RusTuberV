use super::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::pipelined_rendering::PipelinedRenderingPlugin;
use bevy::winit::WinitPlugin;
use std::time::{Duration, Instant};
use vtuber_avatar::{AvatarOutputTarget, AvatarViewportCamera, register_output_systems};

#[derive(Resource, Default)]
struct ReferencePixels(Vec<u8>);

fn readers(app: &mut App) -> Vec<Entity> {
    app.world_mut()
        .query_filtered::<Entity, With<Readback>>()
        .iter(app.world())
        .collect()
}

fn test_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let camera = Transform::from_xyz(0.0, 0.0, 2.5).looking_at(Vec3::ZERO, Vec3::Y);
    commands.spawn((
        Camera3d::default(),
        Camera {
            is_active: false,
            ..default()
        },
        camera,
        AvatarViewportCamera::from_default_transform(camera),
    ));
    let mesh = meshes.add(Rectangle::new(0.5, 0.4));
    for (x, y, color, alpha_mode) in [
        (-0.35, 0.25, Color::srgb(0.2, 0.45, 0.7), AlphaMode::Opaque),
        (
            0.3,
            -0.2,
            Color::srgba(0.7, 0.1, 0.45, 0.5),
            AlphaMode::Blend,
        ),
    ] {
        commands.spawn((
            Mesh3d(mesh.clone()),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: color,
                alpha_mode,
                unlit: true,
                ..default()
            })),
            Transform::from_xyz(x, y, 0.0),
        ));
    }
}

#[test]
#[ignore = "requires a Windows GPU and the RusTuberV Spout sender name"]
fn receiver_matches_bevy_target_with_spout_alone_and_with_ndi()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = vtuber_ndi::NdiScriptedBackend::successful();
    let mut app = ndi_app(backend.clone());
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: bevy::window::ExitCondition::DontExit,
                ..default()
            })
            .disable::<WinitPlugin>()
            .disable::<PipelinedRenderingPlugin>()
            .disable::<bevy::log::LogPlugin>(),
    )
    .init_resource::<SpoutOutputRuntime>()
    .init_resource::<ReferencePixels>()
    .add_systems(Startup, test_scene);
    register_output_systems(&mut app);
    app.finish();
    app.cleanup();
    app.world_mut()
        .resource_mut::<AvatarOutputState>()
        .set_preview_visible(true);
    app.update();
    assert!(readers(&mut app).is_empty());

    // Capture the same Bevy image the studio displays, before enabling either sender.
    let target = app.world().resource::<AvatarOutputTarget>();
    let profile = target.profile();
    assert_eq!(profile.packed_stride_bytes() % 256, 0);
    let image = target.image().clone();
    let reference_entity = app
        .world_mut()
        .spawn(Readback::texture(image))
        .observe(
            |mut event: On<ReadbackComplete>, mut pixels: ResMut<ReferencePixels>| {
                pixels.0 = std::mem::take(&mut event.data);
            },
        )
        .id();
    let deadline = Instant::now() + Duration::from_secs(30);
    let expected = loop {
        app.update();
        let bytes = &mut app.world_mut().resource_mut::<ReferencePixels>().0;
        let size = profile.packed_stride_bytes() * profile.height as usize;
        if bytes.len() == size {
            let pixels = bytes.as_chunks::<4>().0;
            // Require the two materials and clear background, not a warmup/blank frame.
            if pixels.iter().any(|[b, _, r, a]| *a == 255 && b > r)
                && pixels
                    .iter()
                    .any(|[b, _, r, a]| (100..155).contains(a) && r > b)
                && pixels.first().is_some_and(|p| *p == [0; 4])
            {
                break std::mem::take(bytes);
            }
        }
        assert!(
            Instant::now() < deadline,
            "Bevy did not draw the expected reference scene"
        );
    };
    app.world_mut().despawn(reference_entity);
    assert!(readers(&mut app).is_empty());

    let mut receiver = spout2::dx::Receiver::new(Some("RusTuberV"))?;
    app.world_mut()
        .resource_mut::<NdiOutputIntent>()
        .spout_requested = true;
    app.update();
    let [reader] = readers(&mut app)[..] else {
        panic!("Spout requires one Bevy readback")
    };
    let mut received = vec![0; expected.len()];
    // ReceiveImage fills one staging texture and maps the previous one. The
    // first new-frame notification after connecting has no previous image.
    let mut primed = false;
    for with_ndi in [false, true] {
        if with_ndi {
            app.world_mut()
                .resource_mut::<NdiOutputIntent>()
                .request_start();
        }
        let deadline = Instant::now() + Duration::from_secs(12);
        loop {
            app.update();
            assert_eq!(readers(&mut app), [reader]);
            if receiver.receive_image(&mut received, profile.width, profile.height, false, false)?
                && !receiver.is_updated()
                && receiver.is_frame_new()
            {
                if !std::mem::replace(&mut primed, true) || (with_ndi && backend.sent_frames() == 0)
                {
                    continue;
                }
                if let Some((index, (actual, reference))) = received
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(expected.as_chunks::<4>().0)
                    .enumerate()
                    .find(|(_, (a, b))| a != b)
                {
                    panic!(
                        "Spout pixel {index}: received {actual:?}, Bevy {reference:?}, NDI={with_ndi}"
                    );
                }
                break;
            }
            assert!(
                Instant::now() < deadline,
                "Spout frame delivery timed out, NDI={with_ndi}"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    app.world_mut()
        .resource_mut::<NdiOutputIntent>()
        .spout_requested = false;
    app.update();
    assert_eq!(readers(&mut app), [reader]);
    app.world_mut()
        .resource_mut::<NdiOutputIntent>()
        .request_stop();
    app.update();
    assert!(readers(&mut app).is_empty());
    Ok(())
}
