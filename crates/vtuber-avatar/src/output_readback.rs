//! A single asynchronous readback for the fixed avatar output texture.
//! A stalled GPU never grows a queue of staging buffers or blocks the UI.

use std::sync::{Mutex, mpsc};

use bevy::prelude::*;
use bevy::render::gpu_readback::ReadbackComplete;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, MapMode, PollType,
    TexelCopyBufferInfo, TexelCopyBufferLayout,
};
use bevy::render::renderer::{RenderDevice, RenderQueue, render_system};
use bevy::render::texture::GpuImage;
use bevy::render::{
    ExtractSchedule, GpuResourceAppExt, MainWorld, Render, RenderApp, RenderSystems,
};

use crate::render_output::{AvatarOutputCamera, AvatarOutputState, AvatarOutputTarget};

#[derive(Clone)]
struct Request {
    entity: Entity,
    image: Handle<Image>,
    epoch: u64,
    power_generation: u64,
}

struct Completed {
    request: Request,
    data: Result<Vec<u8>, String>,
}

#[derive(Resource, Default)]
struct OutputReadback {
    request: Option<Request>,
    completed: Option<Completed>,
}

struct Pending {
    request: Request,
    receiver: Mutex<mpsc::Receiver<Result<Vec<u8>, String>>>,
}

#[derive(Resource, Default)]
struct GpuOutputReadback {
    buffer: Option<Buffer>,
    pending: Option<Pending>,
}

pub(crate) fn register(app: &mut App) {
    if let Some(render) = app.get_sub_app_mut(RenderApp) {
        render
            .init_resource::<OutputReadback>()
            .init_gpu_resource::<GpuOutputReadback>()
            .add_systems(ExtractSchedule, extract_output)
            .add_systems(
                Render,
                read_output
                    .after(render_system)
                    .in_set(RenderSystems::Render),
            );
    }
}

fn extract_output(mut world: ResMut<MainWorld>, mut output: ResMut<OutputReadback>) {
    let power = vtuber_platform::power_state();
    if let Some(value) = output.completed.take() {
        let valid = world
            .get_resource::<AvatarOutputState>()
            .is_some_and(|state| {
                state.accepts_readback(value.request.epoch, value.request.power_generation, power)
            });
        if valid {
            match value.data {
                Ok(data) => world.trigger(ReadbackComplete {
                    entity: value.request.entity,
                    data,
                }),
                Err(error) => warn!("avatar GPU readback failed: {error}"),
            }
        }
    }
    let entity = world
        .query_filtered::<Entity, With<AvatarOutputCamera>>()
        .iter(&world)
        .next();
    output.request = entity.and_then(|entity| {
        let state = world.get_resource::<AvatarOutputState>()?;
        let target = world.get_resource::<AvatarOutputTarget>()?;
        (state.is_active() && !power.sleeping).then(|| Request {
            entity,
            image: target.image().clone(),
            epoch: state.readback_epoch(),
            power_generation: power.generation,
        })
    });
}

fn read_output(
    mut gpu: ResMut<GpuOutputReadback>,
    mut output: ResMut<OutputReadback>,
    images: Res<RenderAssets<GpuImage>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    if gpu.pending.is_none() && output.request.is_none() {
        return;
    }
    // Poll only, never Wait: sleep/device loss must not strand the render thread.
    if let Err(error) = device.poll(PollType::Poll) {
        warn!("avatar GPU polling failed: {error}");
        return;
    }
    if let Some(pending) = &gpu.pending {
        let result = match pending
            .receiver
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .try_recv()
        {
            Ok(data) => data,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => {
                Err("GPU readback callback was cancelled".into())
            }
        };
        let completed = Completed {
            request: pending.request.clone(),
            data: result,
        };
        output.completed = Some(completed);
        gpu.pending = None;
    }
    let Some(request) = &output.request else {
        return;
    };
    let power = vtuber_platform::power_state();
    if power.sleeping || power.generation != request.power_generation {
        return;
    }
    let Some(image) = images.get(&request.image) else {
        return;
    };
    let size = image.texture_descriptor.size;
    let stride = RenderDevice::align_copy_bytes_per_row(size.width as usize * 4);
    let buffer_size = stride as u64 * u64::from(size.height);
    if gpu
        .buffer
        .as_ref()
        .is_none_or(|buffer| buffer.size() != buffer_size)
    {
        gpu.buffer = Some(device.create_buffer(&BufferDescriptor {
            label: Some("avatar-output-readback"),
            size: buffer_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
    }
    let Some(buffer) = &gpu.buffer else {
        return;
    };
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("avatar-output-copy"),
    });
    encoder.copy_texture_to_buffer(
        image.texture.as_image_copy(),
        TexelCopyBufferInfo {
            buffer,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride as u32),
                rows_per_image: None,
            },
        },
        size,
    );
    queue.submit([encoder.finish()]);
    let (sender, receiver) = mpsc::sync_channel(1);
    let mapped_buffer = buffer.clone();
    buffer.slice(..).map_async(MapMode::Read, move |result| {
        let data = result
            .map(|()| {
                let data = mapped_buffer.slice(..).get_mapped_range().to_vec();
                mapped_buffer.unmap();
                data
            })
            .map_err(|error| error.to_string());
        let _ = sender.try_send(data);
    });
    gpu.pending = Some(Pending {
        request: request.clone(),
        receiver: Mutex::new(receiver),
    });
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;

    #[test]
    fn failed_readback_is_consumed_without_panicking() {
        let mut main = World::new();
        let mut state = AvatarOutputState::default();
        state.activate();
        let epoch = state.readback_epoch();
        main.insert_resource(state);
        let entity = main.spawn(AvatarOutputCamera).id();
        let mut main_world = MainWorld::default();
        *main_world = main;
        let mut app = App::new();
        app.insert_resource(main_world)
            .insert_resource(OutputReadback {
                request: None,
                completed: Some(Completed {
                    request: Request {
                        entity,
                        image: Handle::default(),
                        epoch,
                        power_generation: vtuber_platform::power_state().generation,
                    },
                    data: Err("device lost during mapping".into()),
                }),
            })
            .add_systems(Update, extract_output);
        app.update();
        assert!(app.world().resource::<OutputReadback>().completed.is_none());
    }

    #[test]
    #[ignore = "requires a real GPU; run explicitly on the desktop host"]
    fn gpu_readback_resumes_after_repeated_output_restarts() {
        use crate::render_output::{AvatarOutputFrameSlot, setup_output_camera};
        let mut app = App::new();
        app.add_plugins(
            DefaultPlugins
                .set(bevy::window::WindowPlugin {
                    primary_window: None,
                    exit_condition: bevy::window::ExitCondition::DontExit,
                    ..default()
                })
                .disable::<bevy::winit::WinitPlugin>()
                .disable::<bevy::render::pipelined_rendering::PipelinedRenderingPlugin>(),
        )
        .init_resource::<AvatarOutputState>()
        .init_resource::<AvatarOutputFrameSlot>()
        .add_systems(Startup, setup_output_camera);
        register(&mut app);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while app.plugins_state() != bevy::app::PluginsState::Ready {
            assert!(
                std::time::Instant::now() < deadline,
                "renderer did not initialize"
            );
            bevy::tasks::tick_global_task_pools_on_main_thread();
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        app.finish();
        app.cleanup();
        app.update();
        for mut camera in app
            .world_mut()
            .query_filtered::<&mut Camera, With<AvatarOutputCamera>>()
            .iter_mut(app.world_mut())
        {
            camera.is_active = true;
            camera.clear_color =
                bevy::camera::ClearColorConfig::Custom(Color::srgba(1.0, 0.0, 0.0, 1.0));
        }
        for _ in 0..5 {
            app.world_mut()
                .resource_mut::<AvatarOutputState>()
                .activate();
            let before = app
                .world()
                .resource::<AvatarOutputFrameSlot>()
                .received_frames();
            while app
                .world()
                .resource::<AvatarOutputFrameSlot>()
                .received_frames()
                < before + 5
            {
                assert!(
                    std::time::Instant::now() < deadline,
                    "GPU output did not resume"
                );
                app.update();
                std::thread::sleep(std::time::Duration::from_millis(16));
            }
            let pixel = app
                .world()
                .resource::<AvatarOutputFrameSlot>()
                .latest()
                .unwrap()
                .data()
                .get(..4)
                .unwrap();
            assert!(
                pixel.get(2) > pixel.first(),
                "expected the red camera clear color"
            );
            assert_eq!(pixel.get(3), Some(&255));
            app.world_mut()
                .resource_mut::<AvatarOutputState>()
                .deactivate();
            let before = app
                .world()
                .resource::<AvatarOutputFrameSlot>()
                .received_frames();
            for _ in 0..3 {
                app.update();
            }
            assert_eq!(
                app.world()
                    .resource::<AvatarOutputFrameSlot>()
                    .received_frames(),
                before
            );
        }
    }
}
