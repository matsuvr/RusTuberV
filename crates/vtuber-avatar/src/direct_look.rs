//! Head-relative gaze owned by the application.
//
// Moved from the removed vendored bevy_vrm1 patch (see #91): tracker-owned
// eye-in-head gaze needs no world-space target entity. This module owns the
// direct input component, the eye-bone writer, and the range-mapped
// expression weights, against the unmodified upstream runtime.
use bevy::app::App;
use bevy::prelude::*;
use bevy_vrm1::prelude::{
    LeftEyeBoneEntity, LookAtProperties, LookAtType, RangeMap, RestGlobalTransform, RestTransform,
    RightEyeBoneEntity, VrmSystemSets,
};

/// Registers the direct gaze writer in the upstream gaze-control set.
pub(crate) fn register_direct_look(app: &mut App) {
    app.register_type::<DirectLookAtInput>()
        .register_type::<LookAtExpressionWeights>()
        .add_systems(
            PostUpdate,
            track_direct_look_at.in_set(VrmSystemSets::GazeControl),
        );
}

/// Direct head-relative look-at input for tracker-owned eye-in-head gaze.
///
/// Unlike the upstream `LookAt::Target`, this component requires no
/// world-space target entity. It is inserted on the VRM root and uses VRM
/// LookAt-space degrees.
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

fn same_rotation(a: Quat, b: Quat, epsilon: f32) -> bool {
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

fn map_range(input: f32, range: RangeMap) -> f32 {
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

fn map_range_output(input: f32, range: RangeMap) -> f32 {
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
