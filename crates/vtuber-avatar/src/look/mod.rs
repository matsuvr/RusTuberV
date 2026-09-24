//! Look settings UI state.
//!
//! `#91` carries no rich rendering: this module owns only the look
//! switch/strength resource and the role-selection resource the settings UI
//! edits. Every system that wrote materials, lights, or the finish was removed
//! with the old rich implementation; the rendering follow-ups (`#93`–`#96`)
//! reconnect the UI state kept here.

mod material;
mod preset;

pub use material::{
    AvatarMaterialRoles, MaterialRole, MaterialRoleOverride, MaterialRoleOverridesChanged,
    apply_material_role_overrides, resolve_material_role,
};
pub use preset::RichLookSettings;

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
