//! Look settings and the additional lights and Rich material they drive.
//!
//! This module owns the look switch/strength resource the settings UI edits
//! and persists (`#96`), the fixed camera-relative additional light preset
//! (`#93`), the app-side Rich MToon material/shader with its thin outline
//! connection (`#94`), and the app-side Rich Standard material (`#95`).

mod lighting;
mod preset;
mod rich_mtoon;
mod rich_outline;
mod rich_standard;

pub(crate) use lighting::register_look_lighting;
pub use preset::RichLookSettings;
pub(crate) use rich_mtoon::{RichMtoonSwap, register_rich_mtoon};
pub(crate) use rich_standard::{RichStandardSwap, register_rich_standard};

use bevy::prelude::*;

/// Current look settings for the active avatar.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct AvatarLookSettings(pub RichLookSettings);

/// Requests that [`AvatarLookSettings`] be replaced.
#[derive(Message, Clone, Copy, Debug, PartialEq)]
pub struct LookSettingsChanged(pub RichLookSettings);

/// Copies queued settings changes into [`AvatarLookSettings`].
///
/// This system only updates the resource: it never touches materials, lights
/// or files.
pub fn apply_look_settings_changes(
    mut changes: MessageReader<LookSettingsChanged>,
    mut settings: ResMut<AvatarLookSettings>,
) {
    for change in changes.read() {
        settings.0 = change.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_changes_replace_the_resource() {
        let mut app = App::new();
        app.init_resource::<AvatarLookSettings>()
            .add_message::<LookSettingsChanged>()
            .add_systems(Update, apply_look_settings_changes);

        app.world_mut()
            .resource_mut::<Messages<LookSettingsChanged>>()
            .write(LookSettingsChanged(RichLookSettings {
                enabled: true,
                strength: 0.5,
            }));
        app.update();

        assert_eq!(
            app.world().resource::<AvatarLookSettings>().0,
            RichLookSettings {
                enabled: true,
                strength: 0.5
            }
        );
    }
}
