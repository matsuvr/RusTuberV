//! Bridge from normalized tracking gaze to direct VRM LookAt input.

use bevy::prelude::*;
use bevy_vrm1::prelude::{DirectLookAtInput, LookAtProperties, LookAtType, RangeMap};

use crate::capabilities::SelectedGazeBackend;
use crate::lifecycle::{AvatarLifecycle, AvatarLifecycleState};
use crate::mirror::AvatarMotionMirror;
use crate::unload::ActiveControlFrame;
use vtuber_core::GazeSignal;

/// Adapter-owned conservative LookAt fallback used when model metadata is absent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FallbackGazeProfile {
    /// Reference input angle for a full normalized horizontal signal.
    pub horizontal_input_degrees: f32,
    /// Reference input angle for a full normalized vertical signal.
    pub vertical_input_degrees: f32,
    /// Bone output limit in degrees.
    pub bone_output_degrees: f32,
    /// Expression output weight.
    pub expression_output: f32,
}

impl Default for FallbackGazeProfile {
    fn default() -> Self {
        Self {
            horizontal_input_degrees: 30.0,
            vertical_input_degrees: 30.0,
            bone_output_degrees: 10.0,
            expression_output: 1.0,
        }
    }
}

/// Builds explicit fallback metadata for the exclusively selected backend.
#[must_use]
pub fn fallback_look_at_properties(backend: SelectedGazeBackend) -> LookAtProperties {
    let profile = FallbackGazeProfile::default();
    let output = match backend {
        SelectedGazeBackend::Expression => profile.expression_output,
        SelectedGazeBackend::Bone | SelectedGazeBackend::None => profile.bone_output_degrees,
    };
    let horizontal = RangeMap {
        input_max_value: profile.horizontal_input_degrees,
        output_scale: output,
    };
    let vertical = RangeMap {
        input_max_value: profile.vertical_input_degrees,
        output_scale: output,
    };
    LookAtProperties {
        offset_from_head_bone: [0.0; 3],
        range_map_horizontal_inner: horizontal,
        range_map_horizontal_outer: horizontal,
        range_map_vertical_down: vertical,
        range_map_vertical_up: vertical,
        r#type: match backend {
            SelectedGazeBackend::Expression => LookAtType::Expression,
            SelectedGazeBackend::Bone | SelectedGazeBackend::None => LookAtType::Bone,
        },
    }
}

/// Updates only the direct LookAt input on the active VRM root.
///
/// The vendored runtime owns eye local rotations and expression range mapping;
/// this bridge never writes eye, head, or world transforms.
pub fn update_direct_look_at_input(
    lifecycle: Res<AvatarLifecycle>,
    control_frame: Res<ActiveControlFrame>,
    mirror: Option<Res<AvatarMotionMirror>>,
    mut roots: Query<(&LookAtProperties, &mut DirectLookAtInput)>,
) {
    if lifecycle.state() != AvatarLifecycleState::Ready {
        return;
    }
    let Some(root) = lifecycle.active_root() else {
        return;
    };
    let Ok((properties, mut input)) = roots.get_mut(root) else {
        return;
    };
    let gaze = control_frame
        .frame
        .as_ref()
        .map_or(GazeSignal::UNAVAILABLE, |frame| frame.gaze);
    *input = direct_look_at_input(
        properties,
        gaze,
        mirror.is_none_or(|mirror| mirror.is_enabled()),
    );
}

fn direct_look_at_input(
    properties: &LookAtProperties,
    gaze: GazeSignal,
    mirrored: bool,
) -> DirectLookAtInput {
    let horizontal_scale = properties
        .range_map_horizontal_inner
        .input_max_value
        .max(properties.range_map_horizontal_outer.input_max_value)
        .max(0.0);
    let vrm_pitch_sign = -gaze.vertical;
    let vertical_scale = if vrm_pitch_sign >= 0.0 {
        properties.range_map_vertical_down.input_max_value
    } else {
        properties.range_map_vertical_up.input_max_value
    }
    .max(0.0);
    let horizontal_sign = if mirrored { 1.0 } else { -1.0 };
    DirectLookAtInput {
        // DirectLookAt uses model-left-positive yaw, hence the opposite sign
        // from BodyTracking's semantic yaw. Mirroring reverses only this axis.
        yaw_degrees: finite_or_zero(horizontal_sign * gaze.horizontal * horizontal_scale),
        pitch_degrees: finite_or_zero(vrm_pitch_sign * vertical_scale),
        weight: if gaze.confidence.is_finite() {
            gaze.confidence.clamp(0.0, 1.0)
        } else {
            0.0
        },
        active: gaze.is_available(),
    }
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_profile_is_conservative_and_backend_specific() {
        let bone = fallback_look_at_properties(SelectedGazeBackend::Bone);
        assert_eq!(bone.r#type, LookAtType::Bone);
        assert_eq!(bone.range_map_horizontal_outer.output_scale, 10.0);
        let expression = fallback_look_at_properties(SelectedGazeBackend::Expression);
        assert_eq!(expression.r#type, LookAtType::Expression);
        assert_eq!(expression.range_map_horizontal_outer.output_scale, 1.0);
    }

    #[test]
    fn centered_gaze_is_active_zero_not_unavailable() {
        let properties = fallback_look_at_properties(SelectedGazeBackend::Bone);
        let input = direct_look_at_input(&properties, GazeSignal::tracked(0.0, 0.0, 0.8), true);
        assert!(input.active);
        assert_eq!(input.yaw_degrees, 0.0);
        assert_eq!(input.pitch_degrees, 0.0);
        assert_eq!(input.weight, 0.8);
    }

    #[test]
    fn mirrored_image_directions_reflect_only_horizontal_gaze() {
        let properties = fallback_look_at_properties(SelectedGazeBackend::Bone);
        let right = direct_look_at_input(&properties, GazeSignal::tracked(1.0, 0.0, 1.0), true);
        let left = direct_look_at_input(&properties, GazeSignal::tracked(-1.0, 0.0, 1.0), true);
        let up = direct_look_at_input(&properties, GazeSignal::tracked(0.0, 1.0, 1.0), true);
        let down = direct_look_at_input(&properties, GazeSignal::tracked(0.0, -1.0, 1.0), true);
        assert_eq!(right.yaw_degrees, 30.0);
        assert_eq!(left.yaw_degrees, -30.0);
        assert_eq!(up.pitch_degrees, -30.0);
        assert_eq!(down.pitch_degrees, 30.0);
    }

    #[test]
    fn unmirrored_image_directions_preserve_existing_vrm_conversion() {
        let properties = fallback_look_at_properties(SelectedGazeBackend::Bone);
        let right = direct_look_at_input(&properties, GazeSignal::tracked(1.0, 0.0, 1.0), false);
        assert_eq!(right.yaw_degrees, -30.0);
    }

    #[test]
    fn unavailable_gaze_deactivates_direct_path() {
        let properties = fallback_look_at_properties(SelectedGazeBackend::Expression);
        let input = direct_look_at_input(&properties, GazeSignal::UNAVAILABLE, true);
        assert!(!input.active);
        assert_eq!(input.weight, 0.0);
    }

    #[test]
    fn eye_local_pose_inherits_head_rotation_through_hierarchy() {
        let mut app = App::new();
        app.add_plugins(TransformPlugin);
        let root = app
            .world_mut()
            .spawn((Transform::IDENTITY, GlobalTransform::IDENTITY))
            .id();
        let head = app
            .world_mut()
            .spawn((
                Transform::from_translation(Vec3::Y),
                GlobalTransform::IDENTITY,
                ChildOf(root),
            ))
            .id();
        let eye_local =
            Transform::from_xyz(0.04, 0.08, 0.1).with_rotation(Quat::from_rotation_y(0.1));
        let eye = app
            .world_mut()
            .spawn((eye_local, GlobalTransform::IDENTITY, ChildOf(head)))
            .id();
        app.update();
        let before = *app.world().get::<GlobalTransform>(eye).unwrap();

        app.world_mut().get_mut::<Transform>(head).unwrap().rotation = Quat::from_rotation_y(0.5);
        app.update();

        let local_after = *app.world().get::<Transform>(eye).unwrap();
        let global_after = *app.world().get::<GlobalTransform>(eye).unwrap();
        assert_eq!(local_after, eye_local);
        assert!(!global_after.affine().abs_diff_eq(before.affine(), 1.0e-5));

        let head_before_counter = *app.world().get::<Transform>(head).unwrap();
        app.world_mut().get_mut::<Transform>(eye).unwrap().rotation = Quat::from_rotation_y(-0.2);
        app.update();
        let counter_local = *app.world().get::<Transform>(eye).unwrap();
        let counter_global = app.world().get::<GlobalTransform>(eye).unwrap().rotation();
        let expected_global = (head_before_counter.rotation * counter_local.rotation).normalize();
        assert!(counter_global.abs_diff_eq(expected_global, 1.0e-5));
        assert_eq!(counter_local.translation, eye_local.translation);
        assert_eq!(counter_local.scale, eye_local.scale);
        assert_eq!(
            *app.world().get::<Transform>(head).unwrap(),
            head_before_counter
        );
    }
}
