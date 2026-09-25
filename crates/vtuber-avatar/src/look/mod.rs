//! Look settings and the additional lights and Rich material they drive.
//!
//! This module owns the look switch/strength resource, the role-selection
//! resource the settings UI edits, the fixed camera-relative additional light
//! preset (`#93`), and the app-side Rich MToon material/shader with its thin
//! outline connection (`#94`). The Standard Rich material and the settings-UI
//! reconnection still follow.

mod lighting;
mod material;
mod preset;
mod rich_mtoon;
mod rich_outline;

pub(crate) use lighting::register_look_lighting;
pub use material::{
    AvatarMaterialRoles, MaterialRole, MaterialRoleOverride, MaterialRoleOverridesChanged,
    apply_material_role_overrides, resolve_material_role,
};
pub use preset::RichLookSettings;
pub(crate) use rich_mtoon::{RichMtoonSwap, register_rich_mtoon};

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
