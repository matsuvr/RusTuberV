//! - [`look at specification(en)`](https://github.com/vrm-c/vrm-specification/blob/master/specification/VRMC_vrm-1.0/lookAt.md)
//! - [`look at specification(ja)`](https://github.com/vrm-c/vrm-specification/blob/master/specification/VRMC_vrm-1.0/lookAt.ja.md)

use crate::prelude::*;
use crate::system_set::VrmSystemSets;
use bevy::app::{App, Plugin};
use bevy::prelude::*;
use bevy::window::Window;

/// Controls what the VRM model looks at.
/// This component should be inserted into the root entity of the VRM.
///
/// [`LookAt::Cursor`] tracks the mouse cursor across all windows.
/// [`LookAt::Target`] looks at a specified entity.
///
/// ```no_run
/// use bevy::prelude::*;
/// use bevy_vrm1::prelude::*;
///
/// fn spawn_camera_and_vrm(
///     mut commands: Commands,
///     asset_server: Res<AssetServer>,
/// ) {
///     commands.spawn((Camera3d::default(), Transform::from_xyz(0.0, 1.3, 1.0)));
///     commands.spawn((
///         VrmHandle(asset_server.load("model.vrm")),
///         LookAt::Cursor,
///     ));
/// }
/// ```
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
#[reflect(Component)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", reflect(Serialize, Deserialize))]
pub enum LookAt {
    /// Look at the mouse cursor. Automatically finds the window with the cursor
    /// and the `Camera3d` rendering to it.
    Cursor,

    /// Look at a specific target entity.
    Target(Entity),
}

/// Direct head-relative `LookAt` input for tracker-owned eye-in-head gaze.
///
/// Unlike [`LookAt::Target`], this component requires no world-space target
/// entity. It is inserted on the VRM root and uses VRM LookAt-space degrees.
#[derive(Component, Debug, Clone, Copy, PartialEq, Reflect)]
#[reflect(Component)]
pub struct DirectLookAtInput {
    /// VRM LookAt-space yaw in degrees. Positive points toward model left.
    pub yaw_degrees: f32,
    /// VRM LookAt-space pitch in degrees. Positive points down.
    pub pitch_degrees: f32,
    /// Effective input weight in `[0, 1]`.
    pub weight: f32,
    /// Whether a tracked or returning input is active.
    pub active: bool,
}

impl Default for DirectLookAtInput {
    fn default() -> Self {
        Self {
            yaw_degrees: 0.0,
            pitch_degrees: 0.0,
            weight: 0.0,
            active: false,
        }
    }
}

/// Range-mapped look-direction weights generated for Expression `LookAt`.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq, Reflect)]
#[reflect(Component)]
pub struct LookAtExpressionWeights {
    /// `lookLeft` weight.
    pub look_left: f32,
    /// `lookRight` weight.
    pub look_right: f32,
    /// `lookUp` weight.
    pub look_up: f32,
    /// `lookDown` weight.
    pub look_down: f32,
}

#[derive(Component, Debug, Clone, Copy)]
struct AppliedEyeGaze {
    last_output: Quat,
    last_delta: Quat,
}

pub(super) struct LookAtPlugin;

impl Plugin for LookAtPlugin {
    fn build(
        &self,
        app: &mut App,
    ) {
        app.register_type::<LookAt>()
            .register_type::<DirectLookAtInput>()
            .register_type::<LookAtExpressionWeights>()
            .register_type::<LookAtProperties>()
            .register_type::<LookAtType>()
            .add_systems(
                PostUpdate,
                (track_direct_look_at, track_looking_target).in_set(VrmSystemSets::GazeControl),
            );
    }
}

pub(crate) fn track_looking_target(
    mut commands: Commands,
    vrms: Query<
        (
            &LookAt,
            &LookAtProperties,
            &HeadBoneEntity,
            &LeftEyeBoneEntity,
            &RightEyeBoneEntity,
        ),
        Without<DirectLookAtInput>,
    >,
    transforms: Query<&Transform>,
    global_transforms: Query<&GlobalTransform>,
    rests: Query<(&RestTransform, &RestGlobalTransform)>,
    windows: Query<(Entity, &Window)>,
    cameras: Cameras,
) {
    vrms.iter()
        .for_each(|(look_at, properties, head, left_eye, right_eye)| {
            let Ok(head_gtf) = global_transforms.get(head.0) else {
                return;
            };
            let Ok(head_tf) = transforms.get(head.0) else {
                return;
            };

            let look_at_space = GlobalTransform::default();
            let mut look_at_space_tf = look_at_space.reparented_to(head_gtf);
            look_at_space_tf.translation = Vec3::from(properties.offset_from_head_bone);
            look_at_space_tf.rotation = head_tf.rotation.inverse();
            let look_at_space = head_gtf.mul_transform(look_at_space_tf);

            let (yaw, pitch) = match look_at {
                LookAt::Cursor => {
                    let Some(target_pos) = find_cursor_world_position(&windows, &cameras, head_gtf)
                    else {
                        return;
                    };
                    calc_yaw_pitch(&look_at_space, target_pos)
                }
                LookAt::Target(target_entity) => {
                    let Ok(target_gtf) = global_transforms.get(*target_entity) else {
                        return;
                    };
                    calc_yaw_pitch(&look_at_space, target_gtf.translation())
                }
            };

            match properties.r#type {
                LookAtType::Bone => {
                    apply_bone(
                        &mut commands,
                        &transforms,
                        &rests,
                        left_eye,
                        right_eye,
                        properties,
                        yaw,
                        pitch,
                    );
                }
                LookAtType::Expression => {
                    todo!("Expression look at is not supported yet");
                }
            }
        });
}

fn track_direct_look_at(
    mut commands: Commands,
    vrms: Query<(
        Entity,
        &DirectLookAtInput,
        &LookAtProperties,
        Option<&LeftEyeBoneEntity>,
        Option<&RightEyeBoneEntity>,
    )>,
    eyes: Query<(
        &Transform,
        &RestTransform,
        &RestGlobalTransform,
        Option<&AppliedEyeGaze>,
    )>,
) {
    for (root, input, properties, left_eye, right_eye) in vrms.iter() {
        let (yaw, pitch, weight) = sanitized_direct_input(*input);
        match properties.r#type {
            LookAtType::Bone => {
                commands
                    .entity(root)
                    .insert(LookAtExpressionWeights::default());
                let (Some(left_eye), Some(right_eye)) = (left_eye, right_eye) else {
                    continue;
                };
                apply_direct_eye(
                    &mut commands,
                    &eyes,
                    left_eye.0,
                    properties,
                    yaw * weight,
                    pitch * weight,
                    true,
                );
                apply_direct_eye(
                    &mut commands,
                    &eyes,
                    right_eye.0,
                    properties,
                    yaw * weight,
                    pitch * weight,
                    false,
                );
            }
            LookAtType::Expression => {
                commands
                    .entity(root)
                    .insert(expression_weights(properties, yaw, pitch, weight));
            }
        }
    }
}

fn sanitized_direct_input(input: DirectLookAtInput) -> (f32, f32, f32) {
    if !input.active
        || !input.yaw_degrees.is_finite()
        || !input.pitch_degrees.is_finite()
        || !input.weight.is_finite()
    {
        return (0.0, 0.0, 0.0);
    }
    let weight = input.weight.clamp(0.0, 1.0);
    (input.yaw_degrees, input.pitch_degrees, weight)
}

#[allow(clippy::too_many_arguments)]
fn apply_direct_eye(
    commands: &mut Commands,
    eyes: &Query<(
        &Transform,
        &RestTransform,
        &RestGlobalTransform,
        Option<&AppliedEyeGaze>,
    )>,
    entity: Entity,
    properties: &LookAtProperties,
    yaw: f32,
    pitch: f32,
    is_left: bool,
) {
    let Ok((transform, rest, rest_global, applied)) = eyes.get(entity) else {
        return;
    };
    let target = if is_left {
        apply_left_eye_bone(transform, rest, rest_global, properties, yaw, pitch)
    } else {
        apply_right_eye_bone(transform, rest, rest_global, properties, yaw, pitch)
    };
    let Some((output, state)) = compose_direct_eye_rotation(
        transform.rotation,
        rest.rotation,
        target.rotation,
        applied.copied(),
    ) else {
        return;
    };
    commands
        .entity(entity)
        .insert((transform.with_rotation(output), state));
}

fn compose_direct_eye_rotation(
    current: Quat,
    rest: Quat,
    target: Quat,
    applied: Option<AppliedEyeGaze>,
) -> Option<(Quat, AppliedEyeGaze)> {
    let gaze_delta = (rest.inverse() * target).normalize();
    let animated_base = match applied {
        Some(applied) if same_rotation(current, applied.last_output, 1.0e-5) => {
            (current * applied.last_delta.inverse()).normalize()
        }
        _ => current,
    };
    let output = (animated_base * gaze_delta).normalize();
    output.is_finite().then_some((
        output,
        AppliedEyeGaze {
            last_output: output,
            last_delta: gaze_delta,
        },
    ))
}

fn same_rotation(
    a: Quat,
    b: Quat,
    epsilon: f32,
) -> bool {
    a.is_finite() && b.is_finite() && a.dot(b).abs() >= 1.0 - epsilon
}

fn expression_weights(
    properties: &LookAtProperties,
    yaw: f32,
    pitch: f32,
    weight: f32,
) -> LookAtExpressionWeights {
    let horizontal = map_range(yaw.abs(), properties.range_map_horizontal_outer) * weight;
    let vertical_map = if pitch >= 0.0 {
        properties.range_map_vertical_down
    } else {
        properties.range_map_vertical_up
    };
    let vertical = map_range(pitch.abs(), vertical_map) * weight;
    LookAtExpressionWeights {
        look_left: if yaw > 0.0 { horizontal } else { 0.0 },
        look_right: if yaw < 0.0 { horizontal } else { 0.0 },
        look_up: if pitch < 0.0 { vertical } else { 0.0 },
        look_down: if pitch > 0.0 { vertical } else { 0.0 },
    }
}

fn map_range(
    input: f32,
    range: RangeMap,
) -> f32 {
    if !input.is_finite()
        || !range.input_max_value.is_finite()
        || !range.output_scale.is_finite()
        || range.input_max_value < 0.0
    {
        return 0.0;
    }
    if range.input_max_value == 0.0 {
        return if input == 0.0 {
            0.0
        } else {
            range.output_scale.max(0.0).clamp(0.0, 1.0)
        };
    }
    (input.min(range.input_max_value) / range.input_max_value * range.output_scale).clamp(0.0, 1.0)
}

fn apply_bone(
    commands: &mut Commands,
    transforms: &Query<&Transform>,
    rests: &Query<(&RestTransform, &RestGlobalTransform)>,
    left_eye: &LeftEyeBoneEntity,
    right_eye: &RightEyeBoneEntity,
    properties: &LookAtProperties,
    yaw: f32,
    pitch: f32,
) {
    let Ok(left_eye_tf) = transforms.get(left_eye.0) else {
        return;
    };
    let Ok(right_eye_tf) = transforms.get(right_eye.0) else {
        return;
    };
    let Ok((left_eye_rest_tf, left_eye_gtf)) = rests.get(left_eye.0) else {
        return;
    };
    let Ok((right_eye_rest_tf, right_eye_gtf)) = rests.get(right_eye.0) else {
        return;
    };
    let applied_left_eye_tf = apply_left_eye_bone(
        left_eye_tf,
        left_eye_rest_tf,
        left_eye_gtf,
        properties,
        yaw,
        pitch,
    );
    let applied_right_eye_tf = apply_right_eye_bone(
        right_eye_tf,
        right_eye_rest_tf,
        right_eye_gtf,
        properties,
        yaw,
        pitch,
    );
    commands.entity(left_eye.0).insert(applied_left_eye_tf);
    commands.entity(right_eye.0).insert(applied_right_eye_tf);
}

pub(crate) fn find_cursor_world_position(
    windows: &Query<(Entity, &Window)>,
    cameras: &Cameras,
    head_gtf: &GlobalTransform,
) -> Option<Vec3> {
    let (window_entity, cursor_pos) = windows.iter().find_map(|(entity, window)| {
        let cursor = window.cursor_position();

        #[cfg(target_os = "windows")]
        let cursor = {
            let fallback = fallback_cursor_position(window);
            cursor.or(fallback)
        };

        Some((entity, cursor?))
    })?;
    cameras.to_world_by_viewport(window_entity, cursor_pos, head_gtf.translation())
}

/// Fallback cursor position using `WinAPI` `GetCursorPos` for when
/// `Window::cursor_position()` returns `None` (e.g. `hit_test = false`).
#[cfg(target_os = "windows")]
fn fallback_cursor_position(window: &Window) -> Option<Vec2> {
    use bevy::window::WindowPosition;
    use windows::Win32::Foundation::POINT;
    use windows::Win32::UI::WindowsAndMessaging::GetCursorPos;

    let mut point = POINT::default();
    // SAFETY: GetCursorPos is a safe WinAPI call that writes cursor screen coordinates.
    unsafe { GetCursorPos(&mut point).ok()? };

    let WindowPosition::At(window_pos) = window.position else {
        return None;
    };

    let scale = window.scale_factor();
    let global_logical = Vec2::new(point.x as f32 / scale, point.y as f32 / scale);
    let window_logical = global_logical - window_pos.as_vec2();

    let size = window.resolution.size();
    if window_logical.x >= 0.0
        && window_logical.y >= 0.0
        && window_logical.x <= size.x
        && window_logical.y <= size.y
    {
        Some(window_logical)
    } else {
        None
    }
}

pub(crate) fn calc_yaw_pitch(
    look_at_space: &GlobalTransform,
    target: Vec3,
) -> (f32, f32) {
    let local_target = look_at_space.to_matrix().inverse().transform_point3(target);

    let z = local_target.dot(Vec3::Z);
    let x = local_target.dot(Vec3::X);
    let yaw = (x.atan2(z)).to_degrees();

    let xz = (x * x + z * z).sqrt();
    let y = local_target.dot(Vec3::Y);
    let pitch = (-y.atan2(xz)).to_degrees();

    (yaw, pitch)
}

fn apply_left_eye_bone(
    left_eye: &Transform,
    rest_tf: &RestTransform,
    rest_gtf: &RestGlobalTransform,
    properties: &LookAtProperties,
    yaw_degrees: f32,
    pitch_degrees: f32,
) -> Transform {
    let range_map_horizontal_outer = properties.range_map_horizontal_outer;
    let range_map_horizontal_inner = properties.range_map_horizontal_inner;
    let range_map_vertical_down = properties.range_map_vertical_down;
    let range_map_vertical_up = properties.range_map_vertical_up;
    let yaw = if yaw_degrees > 0.0 {
        map_range_output(yaw_degrees, range_map_horizontal_outer)
    } else {
        -map_range_output(yaw_degrees.abs(), range_map_horizontal_inner)
    };

    let pitch = if pitch_degrees > 0.0 {
        map_range_output(pitch_degrees, range_map_vertical_down)
    } else {
        -map_range_output(pitch_degrees.abs(), range_map_vertical_up)
    };
    left_eye.with_rotation(to_eye_rotation(yaw, pitch, rest_tf, rest_gtf))
}

fn apply_right_eye_bone(
    right_eye: &Transform,
    rest_tf: &RestTransform,
    rest_gtf: &RestGlobalTransform,
    properties: &LookAtProperties,
    yaw_degrees: f32,
    pitch_degrees: f32,
) -> Transform {
    let range_map_horizontal_outer = properties.range_map_horizontal_outer;
    let range_map_horizontal_inner = properties.range_map_horizontal_inner;
    let range_map_vertical_down = properties.range_map_vertical_down;
    let range_map_vertical_up = properties.range_map_vertical_up;

    let yaw = if yaw_degrees > 0.0 {
        map_range_output(yaw_degrees, range_map_horizontal_inner)
    } else {
        -map_range_output(yaw_degrees.abs(), range_map_horizontal_outer)
    };

    let pitch = if pitch_degrees > 0.0 {
        map_range_output(pitch_degrees, range_map_vertical_down)
    } else {
        -map_range_output(pitch_degrees.abs(), range_map_vertical_up)
    };

    right_eye.with_rotation(to_eye_rotation(yaw, pitch, rest_tf, rest_gtf))
}

#[inline]
fn to_eye_rotation(
    yaw: f32,
    pitch: f32,
    rest_tf: &RestTransform,
    rest_gtf: &RestGlobalTransform,
) -> Quat {
    (rest_tf.rotation * rest_gtf.rotation().inverse())
        * Quat::from_euler(EulerRot::YXZ, yaw.to_radians(), pitch.to_radians(), 0.0)
        * rest_gtf.rotation()
}

fn map_range_output(
    input: f32,
    range: RangeMap,
) -> f32 {
    if !input.is_finite()
        || !range.input_max_value.is_finite()
        || !range.output_scale.is_finite()
        || range.input_max_value < 0.0
    {
        return 0.0;
    }
    if range.input_max_value == 0.0 {
        return if input == 0.0 {
            0.0
        } else {
            range.output_scale.max(0.0)
        };
    }
    input.min(range.input_max_value) / range.input_max_value * range.output_scale.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn properties(r#type: LookAtType) -> LookAtProperties {
        LookAtProperties {
            offset_from_head_bone: [0.0; 3],
            range_map_horizontal_inner: RangeMap {
                input_max_value: 20.0,
                output_scale: 5.0,
            },
            range_map_horizontal_outer: RangeMap {
                input_max_value: 40.0,
                output_scale: 10.0,
            },
            range_map_vertical_down: RangeMap {
                input_max_value: 30.0,
                output_scale: 12.0,
            },
            range_map_vertical_up: RangeMap {
                input_max_value: 15.0,
                output_scale: 8.0,
            },
            r#type,
        }
    }

    #[test]
    fn direct_input_needs_no_cursor_or_target_entity() {
        let input = DirectLookAtInput {
            yaw_degrees: 12.0,
            pitch_degrees: -4.0,
            weight: 0.5,
            active: true,
        };
        assert_eq!(sanitized_direct_input(input), (12.0, -4.0, 0.5));
    }

    #[test]
    fn expression_output_is_directionally_exclusive_and_range_mapped() {
        let weights = expression_weights(&properties(LookAtType::Expression), 20.0, -7.5, 1.0);
        assert_eq!(weights.look_left, 1.0);
        assert_eq!(weights.look_right, 0.0);
        assert_eq!(weights.look_up, 1.0);
        assert_eq!(weights.look_down, 0.0);

        let centered = expression_weights(&properties(LookAtType::Expression), 0.0, 0.0, 1.0);
        assert_eq!(centered, LookAtExpressionWeights::default());
    }

    #[test]
    fn zero_input_max_value_steps_nonzero_input_to_output_scale() {
        let mut properties = properties(LookAtType::Expression);
        properties.range_map_horizontal_outer.input_max_value = 0.0;
        let weights = expression_weights(&properties, 30.0, 0.0, 1.0);
        assert_eq!(weights.look_left, 1.0);
        assert!(weights.look_left.is_finite());
        assert_eq!(
            map_range_output(30.0, properties.range_map_horizontal_outer),
            properties.range_map_horizontal_outer.output_scale
        );
        assert_eq!(map_range(0.0, properties.range_map_horizontal_outer), 0.0);
        assert_eq!(
            map_range_output(0.0, properties.range_map_horizontal_outer),
            0.0
        );
    }

    #[test]
    fn left_and_right_eyes_use_outer_and_inner_maps_separately() {
        let mut properties = properties(LookAtType::Bone);
        properties.range_map_horizontal_inner.output_scale = 2.0;
        properties.range_map_horizontal_outer.output_scale = 12.0;
        let rest = RestTransform(Transform::IDENTITY);
        let rest_global = RestGlobalTransform(GlobalTransform::IDENTITY);
        let current = Transform::IDENTITY;
        let left = apply_left_eye_bone(&current, &rest, &rest_global, &properties, 20.0, 0.0);
        let right = apply_right_eye_bone(&current, &rest, &rest_global, &properties, 20.0, 0.0);
        let (_, left_yaw, _) = left.rotation.to_euler(EulerRot::XYZ);
        let (_, right_yaw, _) = right.rotation.to_euler(EulerRot::XYZ);
        assert!((left_yaw.to_degrees() - 6.0).abs() < 1.0e-3);
        assert!((right_yaw.to_degrees() - 2.0).abs() < 1.0e-3);
    }

    #[test]
    fn eye_bones_use_distinct_up_and_down_maps_without_touching_translation_or_scale() {
        let properties = properties(LookAtType::Bone);
        let rest = RestTransform(Transform::IDENTITY);
        let rest_global = RestGlobalTransform(GlobalTransform::IDENTITY);
        let current = Transform::from_xyz(1.0, 2.0, 3.0).with_scale(Vec3::splat(1.5));
        let up = apply_left_eye_bone(&current, &rest, &rest_global, &properties, 0.0, -15.0);
        let down = apply_left_eye_bone(&current, &rest, &rest_global, &properties, 0.0, 15.0);
        let (up_pitch, _, _) = up.rotation.to_euler(EulerRot::XYZ);
        let (down_pitch, _, _) = down.rotation.to_euler(EulerRot::XYZ);
        assert!((up_pitch.to_degrees() + 8.0).abs() < 1.0e-3);
        assert!((down_pitch.to_degrees() - 6.0).abs() < 1.0e-3);
        assert_eq!(up.translation, current.translation);
        assert_eq!(up.scale, current.scale);
        assert_eq!(down.translation, current.translation);
        assert_eq!(down.scale, current.scale);
    }

    #[test]
    fn non_identity_rest_orientation_produces_finite_local_rotation() {
        let rest_rotation = Quat::from_rotation_z(0.3);
        let rest = RestTransform(Transform::from_rotation(rest_rotation));
        let rest_global = RestGlobalTransform(GlobalTransform::from(Transform::from_rotation(
            Quat::from_euler(EulerRot::XYZ, 0.2, -0.1, 0.3),
        )));
        let output = apply_left_eye_bone(
            &Transform::from_rotation(rest_rotation),
            &rest,
            &rest_global,
            &properties(LookAtType::Bone),
            -10.0,
            5.0,
        );
        assert!(output.rotation.is_finite());
        assert!(!output.rotation.abs_diff_eq(rest_rotation, 1.0e-5));
    }

    #[test]
    fn direct_eye_delta_does_not_accumulate_across_frames() {
        let rest = Quat::IDENTITY;
        let target = Quat::from_rotation_y(0.2);
        let (first, state) = compose_direct_eye_rotation(rest, rest, target, None).unwrap();
        let (second, _) = compose_direct_eye_rotation(first, rest, target, Some(state)).unwrap();
        assert!(second.abs_diff_eq(first, 1.0e-5));
    }

    #[test]
    fn sign_flipped_quaternion_does_not_reapply_direct_eye_delta() {
        let rest = Quat::IDENTITY;
        let target = Quat::from_rotation_y(0.2);
        let (first, state) = compose_direct_eye_rotation(rest, rest, target, None).unwrap();
        let (output, _) = compose_direct_eye_rotation(-first, rest, target, Some(state)).unwrap();
        assert!(same_rotation(output, first, 1.0e-5));
    }

    #[test]
    fn direct_eye_returns_to_animated_base_at_center() {
        let rest = Quat::IDENTITY;
        let animated_base = Quat::from_rotation_z(0.15);
        let target = Quat::from_rotation_y(0.25);
        let (looked, state) =
            compose_direct_eye_rotation(animated_base, rest, target, None).unwrap();
        let (centered, _) = compose_direct_eye_rotation(looked, rest, rest, Some(state)).unwrap();
        assert!(centered.abs_diff_eq(animated_base, 1.0e-5));
    }

    #[test]
    fn external_animation_change_becomes_the_new_additive_base() {
        let rest = Quat::IDENTITY;
        let target = Quat::from_rotation_y(0.2);
        let (_, state) = compose_direct_eye_rotation(rest, rest, target, None).unwrap();
        let new_animation = Quat::from_rotation_z(-0.3);
        let (output, _) =
            compose_direct_eye_rotation(new_animation, rest, target, Some(state)).unwrap();
        let expected = (new_animation * target).normalize();
        assert!(output.abs_diff_eq(expected, 1.0e-5));
    }
}
