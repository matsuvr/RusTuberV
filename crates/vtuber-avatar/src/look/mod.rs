//! Rich-look ("リッチ表示") settings and owned look state.
//!
//! This module owns only the look switch/strength and the small snapshot of
//! the material fields the look is allowed to write. Expression-owned fields
//! (color, emission, UV) stay owned by the existing expression writer, so the
//! two never fight over the same material values.

mod lighting;
mod material;
mod mtoon;
mod preset;

pub use lighting::{
    StudioLight, StudioLightSlot, StudioLookState, StudioRig, apply_environment_to_avatar_cameras,
    blend_studio_rig, setup_studio_lighting, solve_studio_rig, studio_environment_cubemap,
    sync_studio_lighting,
};
pub use material::{
    StandardLookBase, StandardLookBases, apply_standard_portrait_settings,
    capture_standard_look_base, clear_look_materials_on_unload, initialize_look_materials,
    resolve_standard_portrait,
};
pub use mtoon::{apply_mtoon_portrait_settings, resolve_mtoon_portrait};
pub use preset::{
    MTOON_PORTRAIT_PRESET, RichLookSettings, STUDIO_PRESET, StudioLightPreset, StudioPreset,
    blend_look_scalar, effective_look_strength,
};

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

