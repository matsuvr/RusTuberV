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

/// Root-level gaze input, mapping and optional eye-bone bindings.
type DirectLookRoot<'a> = (
    Entity,
    &'a DirectLookAtInput,
    &'a LookAtProperties,
    Option<&'a LeftEyeBoneEntity>,
    Option<&'a RightEyeBoneEntity>,
);

fn track_direct_look_at(
    mut commands: Commands,
    vrms: Query<DirectLookRoot<'_>>,
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
                for (entity, is_left) in [(left_eye.0, true), (right_eye.0, false)] {
                    apply_direct_eye(
                        &mut commands,
                        &eyes,
                        entity,
                        properties,
                        yaw * weight,
                        pitch * weight,
                        is_left,
                    );
                }
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
    let target = eye_rotation(rest, rest_global, properties, yaw, pitch, is_left);
    let Some((output, state)) =
        compose_direct_eye_rotation(transform.rotation, rest.rotation, target, applied.copied())
    else {
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

fn eye_rotation(
    rest_tf: &RestTransform,
    rest_gtf: &RestGlobalTransform,
    properties: &LookAtProperties,
    yaw_degrees: f32,
    pitch_degrees: f32,
    is_left: bool,
) -> Quat {
    // Inner/outer are anatomical directions; the same signed yaw is outward
    // for one eye and inward for the other. Both use the same range mapping.
    let horizontal = if (yaw_degrees > 0.0) == is_left {
        properties.range_map_horizontal_outer
    } else {
        properties.range_map_horizontal_inner
    };
    let yaw = if yaw_degrees > 0.0 {
        map_range_output(yaw_degrees, horizontal)
    } else {
        -map_range_output(yaw_degrees.abs(), horizontal)
    };

    let pitch = if pitch_degrees > 0.0 {
        map_range_output(pitch_degrees, properties.range_map_vertical_down)
    } else {
        -map_range_output(pitch_degrees.abs(), properties.range_map_vertical_up)
    };
    rest_tf.rotation
        * crate::skeleton::rest_delta(
            Quat::from_euler(EulerRot::YXZ, yaw.to_radians(), pitch.to_radians(), 0.0),
            rest_gtf.rotation(),
        )
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eye_rotation_reflects_both_sides_and_preserves_inner_outer_ranges() {
        let range = |output_scale| RangeMap {
            input_max_value: 30.0,
            output_scale,
        };
        let properties = LookAtProperties {
            offset_from_head_bone: [0.0; 3],
            range_map_horizontal_inner: range(10.0),
            range_map_horizontal_outer: range(20.0),
            range_map_vertical_down: range(12.0),
            range_map_vertical_up: range(18.0),
            r#type: LookAtType::Bone,
        };
        let rest = RestTransform(Transform::IDENTITY);
        let global = RestGlobalTransform(GlobalTransform::IDENTITY);
        for yaw in [-60.0, -30.0, -15.0, 0.0, 15.0, 30.0, 60.0] {
            for pitch in [-30.0, 0.0, 30.0] {
                let left = eye_rotation(&rest, &global, &properties, yaw, pitch, true);
                let right = eye_rotation(&rest, &global, &properties, -yaw, pitch, false);
                let reflected = Quat::from_xyzw(left.x, -left.y, -left.z, left.w);
                assert!(same_rotation(right, reflected, 1.0e-6));
            }
        }
        for (is_left, input, expected) in [
            (true, 30.0, 20.0_f32),
            (true, -30.0, -10.0),
            (false, 30.0, 10.0),
            (false, -30.0, -20.0),
        ] {
            let actual = eye_rotation(&rest, &global, &properties, input, 0.0, is_left);
            assert!(same_rotation(
                actual,
                Quat::from_rotation_y(expected.to_radians()),
                1.0e-6
            ));
        }
    }
}
