//! Shared skeletal geometry and hierarchy operations.
//!
//! Targets, damping and loss blending choose joint coordinates. This module
//! alone reconstructs a fixed-length two-bone chain with a ball joint and a
//! fixed middle hinge (ozz IKTwoBoneJob). Rest orientations include helper
//! nodes; each parent rotation is inherited once through FK.

use bevy::ecs::query::QueryFilter;
use bevy::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub(crate) struct TwoBoneRest {
    pub start: Vec3,
    pub middle: Vec3,
    pub end: Vec3,
    pub lengths: Vec2,
    pub start_rotation: Quat,
    pub middle_rotation: Quat,
    pub hinge_axis: Vec3,
    pub neutral_lower: Vec3,
    pub axial_projection: f32,
    pub flexion_limit: f32,
    pub axial_limit: f32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TwoBonePose {
    pub middle: Vec3,
    pub end: Vec3,
    pub start_rotation: Quat,
    pub middle_rotation: Quat,
    pub start_delta: Quat,
    pub middle_delta: Quat,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum SolveError {
    NonFinite,
    Degenerate,
}

pub(crate) fn rest_delta(delta: Quat, rest_rotation: Quat) -> Quat {
    (rest_rotation.inverse() * delta * rest_rotation).normalize()
}

pub(crate) fn world_delta_to_local(delta: Quat, parent_rotation: Quat) -> Quat {
    rest_delta(delta, parent_rotation)
}

/// Bent rest segments define the hinge; a straight chain uses its authored
/// neutral plane, projected perpendicular to the upper segment.
pub(crate) fn rest_hinge_axis(
    upper: Vec3,
    lower: Vec3,
    neutral_plane: Option<Vec3>,
) -> Option<Vec3> {
    let upper = finite_normalized(upper)?;
    let lower = finite_normalized(lower)?;
    finite_normalized(upper.cross(lower))
        .or_else(|| {
            let normal = neutral_plane?;
            finite_normalized(normal - upper * normal.dot(upper))
        })
        .or_else(|| finite_normalized(upper.cross(Vec3::Y)))
}

/// Reconstruct joint coordinates instead of interpolating end-point positions
/// or independently rotating the two segments. Axial roll is an extra forearm
/// coordinate; knees use zero. It follows flexion around the lower long axis.
pub(crate) fn from_joints(
    rest: TwoBoneRest,
    start_rotation: Quat,
    flexion: f32,
    axial_roll: f32,
) -> Option<TwoBonePose> {
    let upper = rest.middle - rest.start;
    let lower = rest.end - rest.middle;
    let lower_dir = lower.try_normalize()?;
    let axis = rest.hinge_axis.try_normalize()?;
    // Authored segment offsets need not lie in the anatomical flexion plane.
    // Remove that rest bend before applying the fixed hinge, rather than
    // allowing a small off-plane offset to redefine the joint's axis.
    let rest_to_neutral = Quat::from_rotation_arc(lower_dir, rest.neutral_lower.try_normalize()?);
    let hinge =
        Quat::from_axis_angle(axis, flexion.clamp(0.0, rest.flexion_limit)) * rest_to_neutral;
    // The final FK reconstruction owns the limits, including calls made by
    // damping, loss blending and forearm alignment after the analytic solve.
    let relative = hinge
        * Quat::from_axis_angle(
            lower_dir,
            projected_roll(
                axial_roll.clamp(-rest.axial_limit, rest.axial_limit),
                rest.axial_projection,
            ),
        );
    let start_model = start_rotation * rest.start_rotation.inverse();
    let middle = rest.start + start_model * upper;
    let end = middle + start_model * hinge * lower;
    Some(TwoBonePose {
        middle,
        end,
        start_rotation,
        middle_rotation: (start_model * relative * rest.middle_rotation).normalize(),
        start_delta: rest_delta(start_model, rest.start_rotation),
        middle_delta: rest_delta(relative, rest.middle_rotation),
    })
}

/// Recover the flexion and axial coordinates in the immutable rest frame.
pub(crate) fn joint_coordinates(rest: TwoBoneRest, pose: TwoBonePose) -> Option<Vec2> {
    let lower = (pose.end - pose.middle).try_normalize()?;
    let start_model = pose.start_rotation * rest.start_rotation.inverse();
    let lower = start_model.inverse() * lower;
    let flexion = hinge_coordinate(rest, lower)?;
    let unrolled = from_joints(rest, pose.start_rotation, flexion, 0.0)?;
    let roll = unrolled.middle_rotation.inverse() * pose.middle_rotation;
    let roll = if roll.w < 0.0 { -roll } else { roll };
    let axis = rest.middle_rotation.inverse() * (rest.end - rest.middle).try_normalize()?;
    let projected = 2.0 * roll.xyz().dot(axis).atan2(roll.w);
    Some(Vec2::new(
        flexion,
        unprojected_roll(projected, rest.axial_projection),
    ))
}

pub(crate) fn projected_roll(angle: f32, projection: f32) -> f32 {
    let (s, c) = (0.5 * angle).sin_cos();
    2.0 * (projection * s).atan2(c)
}

pub(crate) fn unprojected_roll(angle: f32, projection: f32) -> f32 {
    let (s, c) = (0.5 * angle).sin_cos();
    2.0 * s.atan2(projection * c)
}

pub(crate) fn hinge_coordinate(rest: TwoBoneRest, lower: Vec3) -> Option<f32> {
    let axis = rest.hinge_axis.try_normalize()?;
    let source = (rest.neutral_lower - axis * rest.neutral_lower.dot(axis)).try_normalize()?;
    let target = (lower - axis * lower.dot(axis)).try_normalize()?;
    Some(axis.dot(source.cross(target)).atan2(source.dot(target)))
}

pub(crate) fn solve_two_bone(
    rest: TwoBoneRest,
    target: Vec3,
    pole: Vec3,
    extension_margin: f32,
) -> Result<TwoBonePose, SolveError> {
    if [
        rest.start,
        rest.middle,
        rest.end,
        rest.hinge_axis,
        rest.neutral_lower,
        target,
        pole,
    ]
    .iter()
    .any(|v| !v.is_finite())
        || !rest.lengths.is_finite()
        || !rest.start_rotation.is_finite()
        || !rest.middle_rotation.is_finite()
    {
        return Err(SolveError::NonFinite);
    }
    let direction = (target - rest.start)
        .try_normalize()
        .or_else(|| (rest.end - rest.start).try_normalize())
        .ok_or(SolveError::Degenerate)?;
    let flexion = flexion_for_reach(rest, target.distance(rest.start), extension_margin)?;
    let project = |v: Vec3| v - direction * v.dot(direction);
    let bend = finite_normalized(project(pole - rest.start))
        .or_else(|| finite_normalized(project(rest.middle - rest.start)))
        .or_else(|| stable_perpendicular(direction, Vec3::Y))
        .ok_or(SolveError::Degenerate)?;
    // Construct the anatomical triangle in double precision, then orient it as
    // a rigid whole. FK remains the only producer of elbow/wrist positions;
    // rounding a nearly straight triangle must not change its hinge angle.
    let upper = (rest.middle - rest.start).as_dvec3().normalize();
    let axis = rest.hinge_axis.as_dvec3().normalize();
    let lower = bevy::math::DQuat::from_axis_angle(axis, flexion)
        * rest.neutral_lower.as_dvec3().normalize();
    let reach = (upper * f64::from(rest.lengths.x) + lower * f64::from(rest.lengths.y)).normalize();
    let source_plane = upper
        .cross(lower)
        .try_normalize()
        .or_else(|| (axis - upper * upper.dot(axis)).try_normalize())
        .ok_or(SolveError::Degenerate)?;
    let source_bend = reach.cross(source_plane).normalize();
    let source = bevy::math::DMat3::from_cols(reach, source_bend, source_plane);
    let direction = direction.as_dvec3().normalize();
    let bend = (bend.as_dvec3() - direction * bend.as_dvec3().dot(direction)).normalize();
    let target = bevy::math::DMat3::from_cols(direction, bend, bend.cross(direction));
    let start_model = bevy::math::DQuat::from_mat3(&(target * source.transpose())).as_quat();
    from_joints(
        rest,
        (start_model * rest.start_rotation).normalize(),
        flexion as f32,
        0.0,
    )
    .ok_or(SolveError::Degenerate)
}

/// Inverse of the same oblique hinge used by FK. The extension-side maximum
/// follows from the source geometry, rather than assuming collinear bones.
pub(crate) fn flexion_for_reach(
    rest: TwoBoneRest,
    distance: f32,
    extension_margin: f32,
) -> Result<f64, SolveError> {
    if !distance.is_finite() || !rest.lengths.is_finite() {
        return Err(SolveError::NonFinite);
    }
    if rest.lengths.min_element() <= 1.0e-4 {
        return Err(SolveError::Degenerate);
    }
    let upper = (rest.middle - rest.start)
        .as_dvec3()
        .try_normalize()
        .ok_or(SolveError::Degenerate)?;
    let axis = rest
        .hinge_axis
        .as_dvec3()
        .try_normalize()
        .ok_or(SolveError::Degenerate)?;
    let neutral = rest
        .neutral_lower
        .as_dvec3()
        .try_normalize()
        .ok_or(SolveError::Degenerate)?;
    let c = upper.dot(axis) * neutral.dot(axis);
    let a = upper.dot(neutral) - c;
    let b = upper.dot(axis.cross(neutral));
    let radius = a.hypot(b);
    if radius <= f64::EPSILON {
        return Err(SolveError::Degenerate);
    }
    let limit = f64::from(rest.flexion_limit);
    let peak = b.atan2(a).clamp(0.0, limit);
    let lengths = rest.lengths.as_dvec2();
    let chord = |f: f64| {
        (lengths.length_squared() + 2.0 * lengths.x * lengths.y * (a * f.cos() + b * f.sin() + c))
            .max(0.0)
            .sqrt()
    };
    let min = chord(limit).min(chord(0.0));
    let max = chord(peak) - f64::from(extension_margin);
    if min >= max {
        return Err(SolveError::Degenerate);
    }
    let reach = f64::from(distance).clamp(min, max);
    let cosine = (reach * reach - lengths.length_squared()) / (2.0 * lengths.x * lengths.y);
    Ok((b.atan2(a) + ((cosine - c) / radius).clamp(-1.0, 1.0).acos()).clamp(0.0, limit))
}

pub(crate) fn finite_normalized(value: Vec3) -> Option<Vec3> {
    let length_squared = value.length_squared();
    if value.is_finite() && length_squared.is_finite() && length_squared > 1.0e-4 {
        Some(value.normalize())
    } else {
        None
    }
}

pub(crate) fn stable_perpendicular(first: Vec3, second: Vec3) -> Option<Vec3> {
    finite_normalized(first.cross(second)).or_else(|| {
        [Vec3::X, Vec3::Y, Vec3::Z]
            .into_iter()
            .filter(|axis| first.dot(*axis).abs() < 0.9)
            .find_map(|axis| finite_normalized(first.cross(axis)))
    })
}

/// Compose today's locals, including non-humanoid nodes, rather than reading
/// GlobalTransforms left by the previous propagation. The optional anchor is
/// the avatar root when the caller's query excludes that root.
pub(crate) fn current_global<F: QueryFilter>(
    entity: Entity,
    transforms: &Query<(&mut Transform, &mut GlobalTransform), F>,
    parents: &Query<&ChildOf>,
    anchor: Option<(Entity, GlobalTransform)>,
) -> Option<GlobalTransform> {
    if let Some((root, global)) = anchor
        && entity == root
    {
        return Some(global);
    }
    let local = *transforms.get(entity).ok()?.0;
    match parents.get(entity) {
        Ok(parent) => {
            Some(current_global(parent.parent(), transforms, parents, anchor)?.mul_transform(local))
        }
        Err(_) => Some(GlobalTransform::from(local)),
    }
}

pub(crate) fn parent_global<F: QueryFilter>(
    entity: Entity,
    transforms: &Query<(&mut Transform, &mut GlobalTransform), F>,
    parents: &Query<&ChildOf>,
) -> Option<GlobalTransform> {
    match parents.get(entity) {
        Ok(parent) => current_global(parent.parent(), transforms, parents, None),
        Err(_) => Some(GlobalTransform::IDENTITY),
    }
}

pub(crate) fn refresh_global<F: QueryFilter>(
    entity: Entity,
    transforms: &mut Query<(&mut Transform, &mut GlobalTransform), F>,
    parents: &Query<&ChildOf>,
    anchor: Option<(Entity, GlobalTransform)>,
) -> Option<GlobalTransform> {
    if let Some((root, global)) = anchor
        && entity == root
    {
        return Some(global);
    }
    let parent = match parents.get(entity) {
        Ok(parent) => refresh_global(parent.parent(), transforms, parents, anchor)?,
        Err(_) => GlobalTransform::IDENTITY,
    };
    let (local, mut cached) = transforms.get_mut(entity).ok()?;
    *cached = parent.mul_transform(*local);
    Some(*cached)
}

pub(crate) fn hinge_delta(
    rest_rotation: Quat,
    segment: Vec3,
    normal: Vec3,
    angle: f32,
) -> Option<Quat> {
    let axis = segment.try_normalize()?.cross(normal).try_normalize()?;
    Some(rest_delta(
        Quat::from_axis_angle(axis, angle),
        rest_rotation,
    ))
}

pub(crate) fn refresh_subtree<F: QueryFilter>(
    entity: Entity,
    global: GlobalTransform,
    transforms: &mut Query<(&mut Transform, &mut GlobalTransform), F>,
    children: &Query<&Children>,
) {
    if let Ok((_, mut cached)) = transforms.get_mut(entity) {
        *cached = global;
    }
    if let Ok(descendants) = children.get(entity) {
        for child in descendants.iter() {
            if let Ok((local, _)) = transforms.get(child) {
                let global = global.mul_transform(*local);
                refresh_subtree(child, global, transforms, children);
            }
        }
    }
}

pub(crate) fn refresh_parent_global<F: QueryFilter>(
    root: Entity,
    parent: Entity,
    root_global: GlobalTransform,
    transforms: &mut Query<(&mut Transform, &mut GlobalTransform), F>,
    child_ofs: &Query<&ChildOf>,
    computed: &mut HashMap<Entity, GlobalTransform>,
) -> Option<GlobalTransform> {
    if parent == root {
        return Some(root_global);
    }
    if let Some(global) = computed.get(&parent) {
        return Some(*global);
    }

    let global = refresh_global(parent, transforms, child_ofs, Some((root, root_global)))?;
    computed.insert(parent, global);
    Some(global)
}
