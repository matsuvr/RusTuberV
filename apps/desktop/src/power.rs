//! Keep rendering atomic across a sleep transition, including GPU readback.

use bevy::ecs::schedule::ScheduleLabel;
use bevy::prelude::*;
use bevy::render::{ExtractSchedule, Render, RenderApp, RenderScheduleOrder};

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
struct PowerAwareRender;

pub(super) fn configure(app: &mut App) {
    app.add_systems(First, update_power_cadence.before(bevy::time::TimeSystems));
    if let Some(render) = app.get_sub_app_mut(RenderApp) {
        render.add_systems(PowerAwareRender, render_if_awake);
        // Preserve Bevy's device recovery and render-to-main time channel.
        // Wrap the entire Render schedule: skipping only Render/PrepareViews
        // still maps buffers whose copy commands were never submitted.
        for label in &mut render
            .world_mut()
            .resource_mut::<RenderScheduleOrder>()
            .labels
        {
            if *label == Render.intern() {
                *label = PowerAwareRender.intern();
            }
        }
    }
}

fn render_if_awake(world: &mut World) {
    run_render(world, vtuber_platform::power_state().sleeping);
}

fn run_render(world: &mut World, sleeping: bool) {
    if !sleeping {
        world.run_schedule(Render);
        return;
    }
    // Extraction still runs while asleep. Drain its commands every tick so
    // deferred entity/asset changes cannot accumulate until wake.
    world.resource_scope(|world, mut schedules: Mut<Schedules>| {
        if let Some(extract) = schedules.get_mut(ExtractSchedule) {
            extract.apply_deferred(world);
        }
    });
    let mut temporary =
        world.query_filtered::<Entity, With<bevy::render::sync_world::TemporaryRenderEntity>>();
    let entities: Vec<_> = temporary.iter(world).collect();
    for entity in entities {
        world.despawn(entity);
    }
}

fn update_power_cadence(
    mut settings: ResMut<bevy::winit::WinitSettings>,
    mut time: ResMut<Time<Virtual>>,
    mut previous: Local<vtuber_platform::PowerState>,
) {
    let power = vtuber_platform::power_state();
    // No catch-up simulation on the first frame after an unobserved cycle.
    // Instant includes suspend on some platforms, excludes it on others.
    if power.sleeping || power.generation != previous.generation || previous.sleeping {
        time.pause();
    } else {
        time.unpause();
    }
    *previous = power;
    let mode = if power.sleeping {
        bevy::winit::UpdateMode::reactive_low_power(std::time::Duration::from_millis(100))
    } else {
        bevy::winit::UpdateMode::Continuous
    };
    if settings.focused_mode != mode || settings.unfocused_mode != mode {
        settings.focused_mode = mode;
        settings.unfocused_mode = mode;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Resource, Default)]
    struct Submitted(u32);

    #[derive(Resource, Default)]
    struct Extracted(u32);

    #[test]
    fn sleep_skips_entire_render_but_drains_extraction() {
        let mut world = World::new();
        world.init_resource::<Submitted>();
        world.init_resource::<Extracted>();
        let mut render = Schedule::new(Render);
        render.add_systems(|mut submitted: ResMut<Submitted>| submitted.0 += 1);
        world.add_schedule(render);
        let mut extract = Schedule::new(ExtractSchedule);
        extract.set_apply_final_deferred(false);
        extract.add_systems(|mut commands: Commands| {
            commands.spawn(bevy::render::sync_world::TemporaryRenderEntity);
            commands.queue(|world: &mut World| world.resource_mut::<Extracted>().0 += 1);
        });
        world.add_schedule(extract);
        for count in 1..=3 {
            world.run_schedule(ExtractSchedule);
            run_render(&mut world, true);
            assert_eq!(world.resource::<Extracted>().0, count);
            let mut temporary = world
                .query_filtered::<Entity, With<bevy::render::sync_world::TemporaryRenderEntity>>();
            assert_eq!(temporary.iter(&world).count(), 0);
        }
        assert_eq!(world.resource::<Submitted>().0, 0);
        run_render(&mut world, false);
        assert_eq!(world.resource::<Submitted>().0, 1);
    }
}
