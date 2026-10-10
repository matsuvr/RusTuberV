use alloc::{string::String, vec::Vec};
use bevy_ecs::{component::Component, entity::Entity};
use bevy_math::{IVec2, UVec2};

#[cfg(all(feature = "serialize", feature = "bevy_reflect"))]
use bevy_reflect::{ReflectDeserialize, ReflectSerialize};
#[cfg(feature = "bevy_reflect")]
use {bevy_ecs::prelude::ReflectComponent, bevy_reflect::Reflect};

/// Represents an available monitor as reported by the user's operating system, which can be used
/// to query information about the display, such as its size, position, and video modes.
///
/// Each monitor corresponds to an entity and can be used to position a monitor using
/// [`MonitorSelection::Entity`](`crate::window::MonitorSelection::Entity`).
///
/// # Warning
///
/// This component is synchronized with `winit` through `bevy_winit`, but is effectively
/// read-only as `winit` does not support changing monitor properties.
#[derive(Component, Debug, Clone)]
#[require(HasWindows)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Component, Debug, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct Monitor {
    /// The name of the monitor
    pub name: Option<String>,
    /// The height of the monitor in physical pixels
    pub physical_height: u32,
    /// The width of the monitor in physical pixels
    pub physical_width: u32,
    /// The position of the monitor in physical pixels
    pub physical_position: IVec2,
    /// The refresh rate of the monitor in millihertz
    pub refresh_rate_millihertz: Option<u32>,
    /// The scale factor of the monitor
    pub scale_factor: f64,
    /// The video modes that the monitor supports
    pub video_modes: Vec<VideoMode>,
}

/// A marker component for the primary monitor
#[derive(Component, Debug, Clone)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Component, Debug, Clone)
)]
pub struct PrimaryMonitor;

/// A relationship for all Windows on a specific Monitor.
/// Windows outlive a monitor disconnect (including display sleep).
#[derive(Component, Debug, Default)]
#[relationship_target(relationship = crate::window::OnMonitor)]
pub struct HasWindows(Vec<Entity>);

impl Monitor {
    /// Returns the physical size of the monitor in pixels
    pub fn physical_size(&self) -> UVec2 {
        UVec2::new(self.physical_width, self.physical_height)
    }
}

/// Represents a video mode that a monitor supports
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(Reflect), reflect(Debug, Clone))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct VideoMode {
    /// The resolution of the video mode
    pub physical_size: UVec2,
    /// The bit depth of the video mode
    pub bit_depth: u16,
    /// The refresh rate in millihertz
    pub refresh_rate_millihertz: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OnMonitor, Window, WindowCloseRequested, WindowPlugin};
    use bevy_app::App;
    use bevy_ecs::message::Messages;

    #[test]
    fn monitor_disconnect_keeps_windows_and_allows_reconnect_and_explicit_close() {
        let mut app = App::new();
        app.add_plugins(WindowPlugin::default());
        let primary = app
            .world_mut()
            .query_filtered::<Entity, bevy_ecs::query::With<crate::PrimaryWindow>>()
            .single(app.world())
            .expect("primary window");
        let secondary = app.world_mut().spawn(Window::default()).id();

        for _ in 0..3 {
            let monitor = app
                .world_mut()
                .spawn(Monitor {
                    name: None,
                    physical_height: 1080,
                    physical_width: 1920,
                    physical_position: IVec2::ZERO,
                    refresh_rate_millihertz: Some(60_000),
                    scale_factor: 1.0,
                    video_modes: Vec::new(),
                })
                .id();
            for window in [primary, secondary] {
                app.world_mut()
                    .entity_mut(window)
                    .insert(OnMonitor(monitor));
            }
            app.update();
            // Winit removes the monitor entity on display sleep or unplug.
            app.world_mut().despawn(monitor);
            app.update();
            for window in [primary, secondary] {
                assert!(app.world().get::<Window>(window).is_some());
                assert!(app.world().get::<OnMonitor>(window).is_none());
            }
            assert!(app.should_exit().is_none());
        }

        // The lifetime fix must not interfere with an intentional close.
        for window in [secondary, primary] {
            app.world_mut()
                .resource_mut::<Messages<WindowCloseRequested>>()
                .write(WindowCloseRequested { window });
            app.update();
            app.update();
            assert!(app.world().get::<Window>(window).is_none());
        }
        assert!(app.should_exit().is_some());
    }
}
