//! Skeleton-bound capsule compounds for anatomical body contacts.
//! Torso breadth and available palm breadth keep dresses and sleeves out of
//! body dimensions; remaining link shapes are fitted from the bind mesh.
//! Runtime contact is segment distance; no mesh skinning or convex-hull rebuilds.

use bevy::mesh::{
    VertexAttributeValues,
    morph::{MeshMorphWeights, MorphWeights},
    skinning::{SkinnedMesh, SkinnedMeshInverseBindposes},
};
use bevy::prelude::*;
use bevy_vrm1::prelude::*;
use parry3d::{
    math::{Pose, Vector},
    shape::{Segment, SupportMap},
};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Region {
    Torso,
    Upper(crate::arm::ArmSide),
    Forearm(crate::arm::ArmSide),
    Hand(crate::arm::ArmSide),
}

#[derive(Clone, Debug)]
pub(crate) struct CapsuleCollider {
    pub bone: Entity,
    pub region: Region,
    /// Capsule centre-line endpoints in model/rest coordinates.
    pub endpoints: [Vec3; 2],
    pub radius: f32,
}

#[derive(Clone, Debug)]
struct ShoulderSocket {
    bone: Entity,
    pivot: Vec3,
    /// Bind vertices influenced by both the thorax and this upper arm.
    radius: f32,
}

#[derive(Component, Clone, Debug, Default)]
pub(crate) struct CollisionGeometry {
    pub capsules: Vec<CapsuleCollider>,
    shoulders: Vec<ShoulderSocket>,
    pairs: Vec<[usize; 2]>,
    bones: Vec<Entity>,
}

#[derive(Component)]
/// Collision shapes consumed by the upper-limb system.
pub struct AvatarCollision(pub(crate) Result<std::sync::Arc<CollisionGeometry>, CollisionError>);

#[derive(Component, Default)]
struct CollisionMorphs(Vec<(Entity, Vec<(usize, f32)>)>);

/// Refit when a relevant body morph changes. Render vertices and skin weights
/// are read only here; candidates transform two endpoints per rigid collider.
#[expect(
    clippy::too_many_arguments,
    clippy::type_complexity,
    reason = "Bevy system parameters read the render assets and skeleton needed to bind collision shapes"
)]
pub(crate) fn bind_collision_geometry(
    mut commands: Commands,
    roots: Query<EntityRef, With<crate::binding::AvatarBinding>>,
    children: Query<&Children>,
    parents: Query<&ChildOf>,
    renderables: Query<(
        &Mesh3d,
        Option<&SkinnedMesh>,
        Option<&Visibility>,
        Option<&MeshMorphWeights>,
    )>,
    morph_weights: Query<&MorphWeights>,
    visibility: Query<&Visibility>,
    bones: Query<(&GlobalTransform, Option<&RestGlobalTransform>)>,
    meshes: Res<Assets<Mesh>>,
    inverse_binds: Res<Assets<SkinnedMeshInverseBindposes>>,
) {
    let weights = |entity| -> Result<&[f32], CollisionError> {
        match renderables
            .get(entity)
            .map_err(|_| CollisionError::InvalidMesh)?
            .3
        {
            Some(MeshMorphWeights::Value { weights }) => Ok(weights),
            Some(MeshMorphWeights::Reference(reference)) => morph_weights
                .get(*reference)
                .map(MorphWeights::weights)
                .map_err(|_| CollisionError::InvalidMesh),
            None => Ok(&[]),
        }
    };
    for root in &roots {
        if root.contains::<AvatarCollision>()
            && root.get::<CollisionMorphs>().is_some_and(|previous| {
                previous.0.iter().all(|(entity, values)| {
                    weights(*entity).is_ok_and(|current| {
                        values.iter().all(|(index, value)| {
                            current.get(*index).copied().unwrap_or(0.0) == *value
                        })
                    })
                })
            })
        {
            continue;
        }

        let Some(binding) = root.get::<crate::binding::AvatarBinding>() else {
            continue;
        };
        if binding.left_arm.is_none() && binding.right_arm.is_none() {
            commands.entity(root.id()).insert((
                AvatarCollision(Ok(std::sync::Arc::new(CollisionGeometry::default()))),
                CollisionMorphs::default(),
            ));
            continue;
        }
        let mut known = HashMap::<Entity, Option<Region>>::new();
        for bone in [
            root.get::<HipsBoneEntity>().map(|b| b.0),
            binding.spine,
            binding.chest,
            binding.upper_chest,
        ]
        .into_iter()
        .flatten()
        {
            known.insert(bone, Some(Region::Torso));
        }
        for bone in [
            Some(binding.head),
            binding.neck,
            root.get::<LeftUpperLegBoneEntity>().map(|b| b.0),
            root.get::<RightUpperLegBoneEntity>().map(|b| b.0),
        ]
        .into_iter()
        .flatten()
        {
            known.insert(bone, None);
        }
        let mut connections = Vec::new();
        let mut axes = HashMap::new();
        for chain in [binding.left_arm, binding.right_arm].into_iter().flatten() {
            let side = chain.side;
            if let Some(bone) = chain.shoulder {
                known.insert(bone, Some(Region::Torso));
            }
            known.insert(chain.upper_arm, Some(Region::Upper(side)));
            known.insert(chain.lower_arm, Some(Region::Forearm(side)));
            known.insert(chain.hand, Some(Region::Hand(side)));
            for finger in [
                chain.finger_rest.thumb,
                chain.finger_rest.index,
                chain.finger_rest.middle,
                chain.finger_rest.ring,
                chain.finger_rest.little,
            ] {
                for joint in [
                    finger.metacarpal,
                    finger.proximal,
                    finger.intermediate,
                    finger.distal,
                ]
                .into_iter()
                .flatten()
                {
                    known.insert(joint.entity, Some(Region::Hand(side)));
                }
            }
            axes.insert(
                chain.upper_arm,
                chain.rest.elbow.position - chain.rest.upper_arm.position,
            );
            axes.insert(
                chain.lower_arm,
                chain.rest.wrist.position - chain.rest.elbow.position,
            );
            axes.insert(
                chain.hand,
                chain.rest.wrist.position - chain.rest.elbow.position,
            );
            if let Some((bone, shoulder)) = chain.shoulder.zip(chain.rest.shoulder) {
                axes.insert(bone, chain.rest.upper_arm.position - shoulder.position);
                // Only the clavicle is adjacent to the humerus. The chest,
                // spine and pelvis still collide with the upper arm.
                connections.push([bone, chain.upper_arm]);
            }
        }
        let classify = |mut entity: Entity| -> Option<(Entity, Region)> {
            loop {
                if let Some(region) = known.get(&entity) {
                    return region.map(|r| (entity, r));
                }
                entity = parents.get(entity).ok()?.parent();
            }
        };
        let rest = |entity| -> Result<GlobalTransform, CollisionError> {
            let (global, rest) = bones.get(entity).map_err(|_| CollisionError::MissingBone)?;
            Ok(rest.map_or(*global, |r| r.0))
        };
        let build = || -> Result<(CollisionGeometry, CollisionMorphs), CollisionError> {
            let forearms: Vec<_> = [binding.left_arm, binding.right_arm]
                .into_iter()
                .flatten()
                .filter_map(forearm_capsule)
                .collect();
            let palms: Vec<_> = [binding.left_arm, binding.right_arm]
                .into_iter()
                .flatten()
                .flat_map(palm_capsules)
                .collect();
            // The torso follows its skeleton links. Rendered skirts, capes and
            // shoulder decorations are not measurements of the human trunk.
            let mut torso = Vec::new();
            if let Some((left, right)) = binding.left_arm.zip(binding.right_arm) {
                let mut breadth = left
                    .rest
                    .upper_arm
                    .position
                    .distance(right.rest.upper_arm.position);
                if let Some((left, right)) = root
                    .get::<LeftUpperLegBoneEntity>()
                    .zip(root.get::<RightUpperLegBoneEntity>())
                {
                    breadth = breadth.max(
                        rest(left.0)?
                            .translation()
                            .distance(rest(right.0)?.translation()),
                    );
                }
                let radius = breadth * 0.5;
                let links: Vec<_> = [
                    root.get::<HipsBoneEntity>().map(|b| b.0),
                    binding.spine,
                    binding.chest,
                    binding.upper_chest,
                ]
                .into_iter()
                .flatten()
                .collect();
                let top = (left.rest.upper_arm.position + right.rest.upper_arm.position) * 0.5;
                for (i, bone) in links.iter().enumerate() {
                    let start = rest(*bone)?.translation();
                    let end = links
                        .get(i + 1)
                        .map(|next| rest(*next).map(|t| t.translation()))
                        .transpose()?
                        .unwrap_or(top);
                    torso.push(link_capsule(*bone, Region::Torso, start, end, radius));
                }
            }
            let mut morph_inputs = CollisionMorphs::default();
            let mut points = HashMap::<(Entity, Entity), Vec<Vec3>>::new();
            let mut sockets: Vec<_> = [binding.left_arm, binding.right_arm]
                .into_iter()
                .flatten()
                .map(|c| ShoulderSocket {
                    bone: c.upper_arm,
                    pivot: c.rest.upper_arm.position,
                    radius: 0.0,
                })
                .collect();

            let mut stack = vec![root.id()];
            while let Some(entity) = stack.pop() {
                if entity != root.id() && visibility.get(entity).ok() == Some(&Visibility::Hidden) {
                    continue;
                }
                if let Ok((mesh, skin, visibility, _)) = renderables.get(entity) {
                    if visibility == Some(&Visibility::Hidden) {
                        continue;
                    }
                    let mesh = meshes.get(&mesh.0).ok_or(CollisionError::InvalidMesh)?;
                    let Some(VertexAttributeValues::Float32x3(vertices)) =
                        mesh.attribute(Mesh::ATTRIBUTE_POSITION)
                    else {
                        return Err(CollisionError::InvalidMesh);
                    };
                    let target_count = if mesh
                        .try_has_morph_targets()
                        .map_err(|_| CollisionError::InvalidMesh)?
                    {
                        mesh.try_morph_targets()
                            .map_err(|_| CollisionError::InvalidMesh)?
                            .len()
                            .checked_div(vertices.len())
                            .ok_or(CollisionError::InvalidMesh)?
                    } else {
                        0
                    };
                    let morphs = mesh.try_morph_targets().ok();
                    let current_weights = weights(entity)?;
                    let position = |index: usize| -> Result<Vec3, CollisionError> {
                        let mut position = Vec3::from_array(
                            *vertices.get(index).ok_or(CollisionError::InvalidMesh)?,
                        );
                        for target in 0..target_count {
                            let delta = morphs
                                .and_then(|m| m.get(target * vertices.len() + index))
                                .ok_or(CollisionError::InvalidMesh)?
                                .position;
                            position += delta * current_weights.get(target).copied().unwrap_or(0.0);
                        }
                        Ok(position)
                    };
                    let mut relevant = std::collections::HashSet::new();
                    let mut include_morphs = |index: usize| {
                        for target in 0..target_count {
                            if morphs
                                .and_then(|m| m.get(target * vertices.len() + index))
                                .is_some_and(|d| d.position != Vec3::ZERO)
                            {
                                relevant.insert(target);
                            }
                        }
                    };
                    if let Some(skin) = skin {
                        let binds = inverse_binds
                            .get(&skin.inverse_bindposes)
                            .ok_or(CollisionError::InvalidMesh)?;
                        let Some(VertexAttributeValues::Uint16x4(indices)) =
                            mesh.attribute(Mesh::ATTRIBUTE_JOINT_INDEX)
                        else {
                            return Err(CollisionError::InvalidMesh);
                        };
                        let Some(VertexAttributeValues::Float32x4(weights)) =
                            mesh.attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT)
                        else {
                            return Err(CollisionError::InvalidMesh);
                        };
                        if vertices.len() != indices.len() || vertices.len() != weights.len() {
                            return Err(CollisionError::InvalidMesh);
                        }
                        let used: std::collections::HashSet<_> = mesh
                            .indices()
                            .map(|i| i.iter().collect())
                            .unwrap_or_else(|| (0..vertices.len()).collect());
                        for (vertex_index, (indices, weights)) in
                            indices.iter().zip(weights).enumerate()
                        {
                            if !used.contains(&vertex_index) {
                                continue;
                            }
                            let vertex = position(vertex_index)?;
                            let mut point = Vec3::ZERO;
                            let mut strongest = (0.0, None);
                            let mut owners = Vec::new();

                            for (&index, &weight) in indices.iter().zip(weights) {
                                if weight <= 0.0 {
                                    continue;
                                }
                                let bone = *skin
                                    .joints
                                    .get(usize::from(index))
                                    .ok_or(CollisionError::InvalidMesh)?;
                                let inverse = binds
                                    .get(usize::from(index))
                                    .ok_or(CollisionError::InvalidMesh)?;
                                let rest_point =
                                    rest(bone)?.transform_point(inverse.transform_point3(vertex));
                                point += rest_point * weight;
                                let owner = classify(bone);
                                if let Some(owner) = owner {
                                    owners.push(owner);
                                }
                                if weight > strongest.0 {
                                    strongest = (weight, owner);
                                }
                            }
                            if !point.is_finite() {
                                return Err(CollisionError::InvalidMesh);
                            }
                            for socket in &mut sockets {
                                if owners.iter().any(|(b, _)| *b == socket.bone)
                                    && owners.iter().any(|(_, r)| *r == Region::Torso)
                                {
                                    include_morphs(vertex_index);
                                    socket.radius = socket.radius.max(point.distance(socket.pivot));
                                }
                            }
                            if let Some((bone, region)) = strongest.1 {
                                if forearms.iter().any(|capsule| capsule.bone == bone)
                                    || palms.iter().any(|capsule| capsule.bone == bone)
                                    || (!torso.is_empty() && region == Region::Torso)
                                {
                                    continue;
                                }
                                // Shared shoulder skin deforms across the joint;
                                // the socket above owns this region. Fitting it
                                // again as rigid upper-arm skin overstates its
                                // occupied volume when the arm crosses the chest.
                                if matches!(region, Region::Upper(_))
                                    && owners.iter().any(|(_, r)| *r == Region::Torso)
                                {
                                    continue;
                                }
                                include_morphs(vertex_index);
                                points.entry((bone, entity)).or_default().push(point);
                            }
                        }
                    } else if let Some((bone, region)) = classify(entity)
                        && !forearms.iter().any(|capsule| capsule.bone == bone)
                        && !palms.iter().any(|capsule| capsule.bone == bone)
                        && (torso.is_empty() || region != Region::Torso)
                    {
                        let transform = rest(entity)?;
                        let used: Vec<_> = mesh
                            .indices()
                            .map(|i| i.iter().collect())
                            .unwrap_or_else(|| (0..vertices.len()).collect());
                        for index in used {
                            include_morphs(index);
                            let point = transform.transform_point(position(index)?);
                            points.entry((bone, entity)).or_default().push(point);
                        }
                    }
                    if !relevant.is_empty() {
                        let mut relevant: Vec<_> = relevant.into_iter().collect();
                        relevant.sort_unstable();
                        morph_inputs.0.push((
                            entity,
                            relevant
                                .into_iter()
                                .map(|i| (i, current_weights.get(i).copied().unwrap_or(0.0)))
                                .collect(),
                        ));
                    }
                }
                if let Ok(descendants) = children.get(entity) {
                    stack.extend(descendants.iter());
                }
            }
            let mut capsules = forearms;
            capsules.extend(palms);
            capsules.extend(torso);
            // Keep separate primitives separate: a wing and a sleeve must
            // not become a single collider across their empty space.
            let mut parts: Vec<_> = points.into_iter().collect();
            parts.sort_by_key(|(key, _)| *key);
            for ((bone, _), points) in parts {
                let (_, region) = classify(bone).ok_or(CollisionError::MissingBone)?;
                // VRM1 may rotate a bone's authored axes without changing its
                // T-pose mesh. Fit in the shared model-space T-pose basis;
                // inverse bind poses already account for the authored axes.
                let orientation = if let Some(axis) = axes.get(&bone) {
                    let axis = axis.try_normalize().ok_or(CollisionError::InvalidMesh)?;
                    Quat::from_rotation_arc(Vec3::Y, axis)
                } else {
                    Quat::IDENTITY
                };
                capsules.extend(fit_capsules(bone, region, &points, orientation)?);
            }
            if !capsules.iter().any(|h| h.region == Region::Torso) {
                return Err(CollisionError::InvalidMesh);
            }
            for chain in [binding.left_arm, binding.right_arm].into_iter().flatten() {
                for region in [
                    Region::Upper(chain.side),
                    Region::Forearm(chain.side),
                    Region::Hand(chain.side),
                ] {
                    if !capsules.iter().any(|h| h.region == region) {
                        return Err(CollisionError::InvalidMesh);
                    }
                }
            }
            let mut geometry = CollisionGeometry::new(capsules, &connections);
            geometry.shoulders = sockets;
            Ok((geometry, morph_inputs))
        };

        match build() {
            Ok((geometry, morphs)) => {
                commands
                    .entity(root.id())
                    .insert((AvatarCollision(Ok(std::sync::Arc::new(geometry))), morphs));
            }
            Err(error) => {
                commands
                    .entity(root.id())
                    .insert((AvatarCollision(Err(error)), CollisionMorphs::default()));
            }
        }
        if !root.contains::<AvatarCollision>() {
            commands.entity(root.id()).insert(Visibility::Hidden);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BoneMotion {
    /// Rigid map of a point from model/rest space to the candidate pose.
    pub rotation: Quat,
    pub translation: Vec3,
}

impl BoneMotion {
    pub fn point(self, p: Vec3) -> Vec3 {
        self.rotation * p + self.translation
    }
}

/// Failure to construct or evaluate the upper-body collision shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollisionError {
    /// Missing, malformed or degenerate render mesh.
    InvalidMesh,
    /// A render skin references an unavailable bone.
    MissingBone,
}

/// Measure transverse palm breadth at the thumb CMC and four finger MCPs.
/// The thumb MCP belongs to the articulated thumb, outside the rigid palm;
/// its T-pose spread must not inflate the forearm. Neither does a sleeve.
/// Finger bones are optional in VRM; without a measurable breadth the existing
/// mesh-based binding remains responsible for that link.
fn forearm_capsule(chain: crate::arm::ArmChainBinding) -> Option<CapsuleCollider> {
    let start = chain.rest.elbow.position;
    let end = chain.rest.wrist.position;
    let direction = (end - start).try_normalize()?;
    let knuckles: Vec<_> = [
        chain.finger_rest.thumb.metacarpal,
        chain.finger_rest.index.proximal,
        chain.finger_rest.middle.proximal,
        chain.finger_rest.ring.proximal,
        chain.finger_rest.little.proximal,
    ]
    .into_iter()
    .flatten()
    .map(|joint| joint.rest.position)
    .collect();
    let mut breadth = 0.0_f32;
    for (i, a) in knuckles.iter().enumerate() {
        for b in knuckles.iter().skip(i + 1) {
            let across = *b - *a;
            breadth = breadth.max((across - direction * across.dot(direction)).length());
        }
    }
    if !breadth.is_finite() || breadth <= 0.0 {
        return None;
    }
    let radius = breadth * 0.5;
    Some(link_capsule(
        chain.lower_arm,
        Region::Forearm(chain.side),
        start,
        end,
        radius,
    ))
}

/// The rigid palm spans the wrist and four MCP centres. Their measured spacing
/// supplies its thickness, so a cuff or ornament bound to the hand bone cannot
/// block an otherwise clear arm path. Finger shapes retain their own bindings.
fn palm_capsules(chain: crate::arm::ArmChainBinding) -> Vec<CapsuleCollider> {
    let knuckles: Option<Vec<_>> = [
        chain.finger_rest.index.proximal,
        chain.finger_rest.middle.proximal,
        chain.finger_rest.ring.proximal,
        chain.finger_rest.little.proximal,
    ]
    .into_iter()
    .map(|joint| joint.map(|joint| joint.rest.position))
    .collect();
    let Some(knuckles) = knuckles else {
        return Vec::new();
    };
    let radius = knuckles
        .windows(2)
        .filter_map(|pair| {
            pair.first()
                .zip(pair.last())
                .map(|(a, b)| a.distance(*b) * 0.5)
        })
        .fold(0.0_f32, f32::max);
    knuckles
        .into_iter()
        .map(|end| {
            link_capsule(
                chain.hand,
                Region::Hand(chain.side),
                chain.rest.wrist.position,
                end,
                radius,
            )
        })
        .collect()
}

/// A skeleton link's proxy spans its two joint centres, independent of clothing.
fn link_capsule(
    bone: Entity,
    region: Region,
    start: Vec3,
    end: Vec3,
    radius: f32,
) -> CapsuleCollider {
    let half_link = (end - start) * 0.5;
    let half_length = half_link.length();
    let inset = if half_length > radius {
        half_link * (radius / half_length)
    } else {
        half_link
    };
    CapsuleCollider {
        bone,
        region,
        endpoints: [start + inset, end - inset],
        radius,
    }
}

/// Fit a rounded box using up to three parallel capsules. Its long direction
/// follows the anatomical basis. Multiple columns represent the broad, thin
/// torso/palm without a single oversized circular cross section. These are
/// collision proxies, not conservative envelopes of every clothing vertex.
fn fit_capsules(
    bone: Entity,
    region: Region,
    points: &[Vec3],
    orientation: Quat,
) -> Result<Vec<CapsuleCollider>, CollisionError> {
    let mut basis = [Vec3::Y, Vec3::X, Vec3::Z].map(|v| orientation * v);
    let bounds = |axis: Vec3| {
        points
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), p| {
                let value = p.dot(axis);
                (min.min(value), max.max(value))
            })
    };
    basis.sort_by(|a, b| {
        let (a0, a1) = bounds(*a);
        let (b0, b1) = bounds(*b);
        (b1 - b0).total_cmp(&(a1 - a0))
    });
    let [long, broad, thin] = basis;
    let (lo, hi) = bounds(long);
    let (left, right) = bounds(broad);
    let (back, front) = bounds(thin);
    let depth = front - back;
    let width = right - left;
    if !depth.is_finite()
        || depth <= 0.0
        || !width.is_finite()
        || !lo.is_finite()
        || !hi.is_finite()
    {
        return Err(CollisionError::InvalidMesh);
    }
    let count = (width / depth).ceil().clamp(1.0, 3.0) as usize;
    let radius = (depth * 0.5).max(width / (2.0 * count as f32));
    let start = (lo + radius).min((lo + hi) * 0.5);
    let end = (hi - radius).max((lo + hi) * 0.5);
    Ok((0..count)
        .map(|i| {
            let x = if count == 1 {
                (left + right) * 0.5
            } else {
                left + radius + (width - 2.0 * radius) * i as f32 / (count - 1) as f32
            };
            let offset = broad * x + thin * ((back + front) * 0.5);
            CapsuleCollider {
                bone,
                region,
                endpoints: [offset + long * start, offset + long * end],
                radius,
            }
        })
        .collect())
}

impl CollisionGeometry {
    /// Move an observed forearm in depth to the camera-facing torso surface.
    /// Both landmarks receive the same translation, preserving their view-plane
    /// positions and relative geometry. The constrained FK still decides the
    /// actual joints and separation from the other arm.
    pub fn visible_forearm_target(
        &self,
        points: [Vec3; 2],
        toward_camera: Vec3,
        side: crate::arm::ArmSide,
        body: &HashMap<Entity, BoneMotion>,
    ) -> [Vec3; 2] {
        use parry3d::{
            query::{ShapeCastOptions, details::cast_shapes_support_map_support_map},
            shape::Capsule,
        };
        let radius = self
            .capsules
            .iter()
            .filter(|c| c.region == Region::Forearm(side))
            .map(|c| c.radius)
            .fold(0.0_f32, f32::max);
        let [a, b] = points.map(|p| Vector::from_array(p.to_array().map(f64::from)));
        let axis = Vector::from_array(toward_camera.to_array().map(f64::from));
        // The body pose excludes independently solved shoulder/arm bones.
        let capsules = self
            .capsules
            .iter()
            .filter(|c| c.region == Region::Torso)
            .filter_map(|c| {
                body.get(&c.bone).map(|m| {
                    let [a, b] = c
                        .endpoints
                        .map(|p| Vector::from_array(m.point(p).to_array().map(f64::from)));
                    Capsule::new(a, b, f64::from(c.radius))
                })
            });
        let rear = a.dot(axis).min(b.dot(axis)) - f64::from(radius);
        let front = capsules
            .clone()
            .map(|c| c.segment.a.dot(axis).max(c.segment.b.dot(axis)) + c.radius)
            .fold(rear, f64::max);
        let distance = front - rear;
        let forearm = Capsule::new(a + axis * distance, b + axis * distance, f64::from(radius));
        let hit = capsules
            .filter_map(|c| {
                cast_shapes_support_map_support_map(
                    &Pose::IDENTITY,
                    -axis,
                    &c,
                    &forearm,
                    ShapeCastOptions::with_max_time_of_impact(distance),
                )
                .map(|hit| hit.time_of_impact)
            })
            .fold(distance, f64::min);
        points.map(|p| p + toward_camera * (distance - hit) as f32)
    }

    pub fn new(capsules: Vec<CapsuleCollider>, connected: &[[Entity; 2]]) -> Self {
        let mut bones: Vec<_> = capsules.iter().map(|c| c.bone).collect();
        bones.sort_unstable();
        bones.dedup();
        let mut pairs = Vec::new();
        for (i, a) in capsules.iter().enumerate() {
            for (j, b) in capsules.iter().enumerate().skip(i + 1) {
                // Standard articulated-body exclusions: a rigid link, the
                // same hand, and directly connected elbow/wrist/clavicle links.
                // Never disable the whole arm against the torso.
                let adjacent = match (a.region, b.region) {
                    (Region::Upper(x), Region::Forearm(y))
                    | (Region::Forearm(x), Region::Upper(y))
                    | (Region::Forearm(x), Region::Hand(y))
                    | (Region::Hand(x), Region::Forearm(y))
                    | (Region::Hand(x), Region::Hand(y)) => x == y,
                    (Region::Torso, Region::Torso) => true,
                    _ => false,
                };
                if a.bone != b.bone
                    && !adjacent
                    && !connected
                        .iter()
                        .any(|p| *p == [a.bone, b.bone] || *p == [b.bone, a.bone])
                {
                    pairs.push([i, j]);
                }
            }
        }
        Self {
            capsules,
            shoulders: Vec::new(),
            pairs,
            bones,
        }
    }

    pub fn bones(&self) -> impl Iterator<Item = Entity> + '_ {
        self.bones.iter().copied()
    }

    pub fn differential_pairs(
        &self,
        active: &[bool],
        moved: &std::collections::HashSet<Entity>,
    ) -> Vec<bool> {
        self.pairs
            .iter()
            .zip(active)
            .map(|(pair, active)| {
                *active
                    && pair
                        .iter()
                        .filter_map(|i| self.capsules.get(*i))
                        .any(|c| moved.contains(&c.bone))
            })
            .collect()
    }

    fn posed(
        &self,
        motion: &impl Fn(Entity) -> Option<BoneMotion>,
    ) -> Result<Vec<Segment>, CollisionError> {
        self.capsules
            .iter()
            .map(|c| {
                let motion = motion(c.bone).ok_or(CollisionError::MissingBone)?;
                let [a, b] = c
                    .endpoints
                    .map(|p| Vector::from_array(motion.point(p).to_array().map(f64::from)));
                Ok(Segment::new(a, b))
            })
            .collect()
    }

    fn pair(
        &self,
        [a, b]: [usize; 2],
    ) -> Result<(&CapsuleCollider, &CapsuleCollider), CollisionError> {
        self.capsules
            .get(a)
            .zip(self.capsules.get(b))
            .ok_or(CollisionError::InvalidMesh)
    }

    fn shoulder_socket(&self, a: &CapsuleCollider, b: &CapsuleCollider) -> Option<&ShoulderSocket> {
        let upper = match (a.region, b.region) {
            (Region::Upper(_), Region::Torso) => a.bone,
            (Region::Torso, Region::Upper(_)) => b.bone,
            _ => return None,
        };
        self.shoulders
            .iter()
            .find(|s| s.bone == upper && s.radius > 0.0)
    }

    /// Signed capsule separation. The frozen mask only skips inactive finite
    /// difference rows; every new nonlinear candidate checks all pairs.
    pub fn solver_margins(
        &self,
        motion: impl Fn(Entity) -> Option<BoneMotion>,
        margin: f32,
        frozen: Option<&[bool]>,
    ) -> Result<(Vec<f32>, Vec<bool>), CollisionError> {
        let segments = self.posed(&motion)?;
        let mut margins = Vec::with_capacity(self.pairs.len());
        let mut active = Vec::with_capacity(self.pairs.len());
        for (index, &[a, b]) in self.pairs.iter().enumerate() {
            if frozen.and_then(|mask| mask.get(index)) == Some(&false) {
                margins.push(f32::INFINITY);
                active.push(false);
                continue;
            }
            let (ca, cb) = self.pair([a, b])?;
            let (sa, sb) = segments
                .get(a)
                .zip(segments.get(b))
                .ok_or(CollisionError::InvalidMesh)?;
            let mut distance = segment_distance(sa, sb) as f32 - ca.radius - cb.radius;
            if let Some(socket) = self.shoulder_socket(ca, cb) {
                let pivot = motion(socket.bone)
                    .ok_or(CollisionError::MissingBone)?
                    .point(socket.pivot);
                let pivot = Vector::from_array(pivot.to_array().map(f64::from));
                distance = distance.max(
                    // Include both capsule rounding shells at the joint;
                    // they must not force the arm away from its attached torso.
                    socket.radius + ca.radius + cb.radius
                        - intersection_radius(
                            &[sa.a - pivot, sa.b - pivot],
                            ca.radius,
                            &[sb.a - pivot, sb.b - pivot],
                            cb.radius,
                        ),
                );
            }
            margins.push(distance);
            active.push(frozen.is_some() || distance <= margin);
        }
        Ok((margins, active))
    }

    pub fn pose_is_clear(
        &self,
        motion: &impl Fn(Entity) -> Option<BoneMotion>,
        clearance: f32,
    ) -> Result<bool, CollisionError> {
        let (margins, _) = self.solver_margins(motion, 0.0, None)?;
        Ok(margins.into_iter().all(|d| d >= clearance))
    }

    /// The four endpoint centres plus radius enclose a linearly swept capsule.
    /// Padding bounds the nonlinear FK chord error. GJK proposes a plane; its
    /// exact support gap certifies separation (no approximate GJK distance).
    pub fn sweep_is_clear(
        &self,
        motion: &impl Fn(Entity) -> Option<BoneMotion>,
        endpoint: &impl Fn(Entity) -> Option<BoneMotion>,
        padding: impl Fn(usize) -> f32,
    ) -> Result<bool, CollisionError> {
        let from = self.posed(motion)?;
        let to = self.posed(endpoint)?;
        for &[a, b] in &self.pairs {
            let (ca, cb) = self.pair([a, b])?;
            let upper_index = if matches!(ca.region, Region::Upper(_)) {
                a
            } else {
                b
            };
            let swept = |i: usize, radius: f32| -> Result<SweptCapsule, CollisionError> {
                let (from, to) = from
                    .get(i)
                    .zip(to.get(i))
                    .ok_or(CollisionError::InvalidMesh)?;
                Ok(SweptCapsule {
                    points: [from.a, from.b, to.a, to.b],
                    radius: f64::from(radius + padding(i)),
                })
            };
            let a = swept(a, ca.radius)?;
            let b = swept(b, cb.radius)?;
            if !a.separated(&b) {
                let Some(socket) = self.shoulder_socket(ca, cb) else {
                    return Ok(false);
                };
                let pivot = |motion: &dyn Fn(Entity) -> Option<BoneMotion>| {
                    motion(socket.bone)
                        .map(|m| {
                            Vector::from_array(m.point(socket.pivot).to_array().map(f64::from))
                        })
                        .ok_or(CollisionError::MissingBone)
                };
                let start = pivot(motion)?;
                let end = pivot(endpoint)?;
                // The upper capsule's acceleration bound includes the socket's
                // translation. Add it after subtracting the pivot chord.
                let joint_padding = padding(upper_index);
                let relative = |s: &SweptCapsule| {
                    let [a, b, c, d] = s.points;
                    [a - start, b - start, c - end, d - end]
                };
                if intersection_radius(
                    &relative(&a),
                    a.radius as f32 + joint_padding,
                    &relative(&b),
                    b.radius as f32 + joint_padding,
                ) > socket.radius + ca.radius + cb.radius
                {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
}

/// The intersection of two shapes lies inside the intersection of their
/// bounding boxes. Its farthest corner bounds every overlapping point,
/// including contacts other than the deepest capsule contact.
fn intersection_radius(a: &[Vector], ra: f32, b: &[Vector], rb: f32) -> f32 {
    let bounds = |points: &[Vector], radius: f32| {
        let lo = points
            .iter()
            .copied()
            .fold(Vector::splat(f64::INFINITY), |v, p| v.min(p));
        let hi = points
            .iter()
            .copied()
            .fold(Vector::splat(f64::NEG_INFINITY), |v, p| v.max(p));
        (
            lo - Vector::splat(f64::from(radius)),
            hi + Vector::splat(f64::from(radius)),
        )
    };
    let (al, ah) = bounds(a, ra);
    let (bl, bh) = bounds(b, rb);
    al.max(bl).abs().max(ah.min(bh).abs()).length() as f32
}

fn segment_distance(a: &Segment, b: &Segment) -> f64 {
    let (la, lb) = parry3d::query::details::closest_points_segment_segment_with_locations(
        &Pose::IDENTITY,
        a,
        b,
    );
    a.point_at(&la).distance(b.point_at(&lb))
}

struct SweptCapsule {
    points: [Vector; 4],
    radius: f64,
}
impl SupportMap for SweptCapsule {
    fn local_support_point(&self, direction: Vector) -> Vector {
        let [first, ..] = self.points;
        let point = self.points.iter().copied().fold(first, |best, p| {
            if p.dot(direction) > best.dot(direction) {
                p
            } else {
                best
            }
        });
        point + direction.try_normalize().unwrap_or(Vector::ZERO) * self.radius
    }
}
impl SweptCapsule {
    fn separated(&self, other: &Self) -> bool {
        let a = self.points.iter().copied().sum::<Vector>() * 0.25;
        let b = other.points.iter().copied().sum::<Vector>() * 0.25;
        let gap = |axis: Vector| {
            (other.local_support_point(-axis) - self.local_support_point(axis)).dot(axis)
        };
        if gap(b - a) > 0.0 {
            return true;
        }
        let extent = |shape: &Self, centre: Vector| {
            shape
                .points
                .iter()
                .map(|p| p.distance(centre))
                .fold(0.0_f64, f64::max)
                + shape.radius
        };
        let prediction = a.distance(b) + extent(self, a) + extent(other, b);
        match parry3d::query::details::closest_points_support_map_support_map(
            &Pose::IDENTITY,
            self,
            other,
            prediction,
        ) {
            parry3d::query::ClosestPoints::WithinMargin(a, b) => {
                (b - a).try_normalize().is_some_and(|axis| gap(axis) >= 0.0)
            }
            // No penetration depth is needed for a sweep enclosure.
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]
    use super::*;
    use crate::arm::ArmSide;
    fn entity(i: u32) -> Entity {
        Entity::from_raw_u32(i).unwrap()
    }
    fn capsule(bone: u32, region: Region, x: f32) -> CapsuleCollider {
        CapsuleCollider {
            bone: entity(bone),
            region,
            endpoints: [Vec3::new(x, -0.2, 0.0), Vec3::new(x, 0.2, 0.0)],
            radius: 0.1,
        }
    }
    fn identity(_: Entity) -> Option<BoneMotion> {
        Some(BoneMotion {
            rotation: Quat::IDENTITY,
            translation: Vec3::ZERO,
        })
    }
    #[test]
    fn forearm_proxy_uses_transverse_knuckle_breadth_and_bone_length() {
        use crate::arm::*;
        let pose = |position| RestSpaceBonePose {
            position,
            global_rotation: Quat::from_rotation_y(0.7),
            local_rotation: Quat::from_rotation_z(-0.3),
        };
        let joint = |id, position| FingerJointRestBinding {
            entity: entity(id),
            rest: pose(position),
        };
        let mut fingers = FingerRestReferences::default();
        fingers.thumb.metacarpal = Some(joint(4, Vec3::new(0.34, 0.0, 0.04)));
        // An extended articulated thumb is not the width of the rigid palm.
        fingers.thumb.proximal = Some(joint(6, Vec3::new(0.38, 0.0, 0.4)));
        fingers.little.proximal = Some(joint(5, Vec3::new(0.38, 0.0, -0.04)));
        let mut chain = ArmChainBinding {
            side: ArmSide::Left,
            shoulder: None,
            upper_arm: entity(1),
            lower_arm: entity(2),
            hand: entity(3),
            fingers: FingerReferences::default(),
            finger_rest: fingers,
            capabilities: ArmChainCapabilities::default(),
            rest: ArmRestGeometry {
                shoulder: None,
                upper_arm: pose(Vec3::ZERO),
                elbow: pose(Vec3::X * 0.1),
                wrist: pose(Vec3::X * 0.3),
                upper_arm_length: 0.1,
                forearm_length: 0.2,
                total_arm_length: 0.3,
            },
        };
        let proxy = forearm_capsule(chain).unwrap();
        assert!((proxy.radius - 0.04).abs() < 1.0e-6);
        assert!(proxy.endpoints[0].distance(Vec3::X * 0.14) < 1.0e-6);
        assert!(proxy.endpoints[1].distance(Vec3::X * 0.26) < 1.0e-6);
        assert_eq!(proxy.bone, chain.lower_arm);
        chain.finger_rest = FingerRestReferences::default();
        assert!(forearm_capsule(chain).is_none());
    }

    #[test]
    fn rigid_palm_covers_all_mcp_centres_without_a_wide_wrist_shell() {
        for side in [ArmSide::Left, ArmSide::Right] {
            let mut chain = crate::upper_limb::tests::chain(side);
            let wrist = chain.rest.wrist.position;
            let index = chain.finger_rest.index.proximal.unwrap();
            let little = chain.finger_rest.little.proximal.unwrap();
            let mut middle = index;
            middle.entity = entity(100);
            middle.rest.position = index.rest.position.lerp(little.rest.position, 1.0 / 3.0);
            let mut ring = middle;
            ring.entity = entity(101);
            ring.rest.position = index.rest.position.lerp(little.rest.position, 2.0 / 3.0);
            chain.finger_rest.middle.proximal = Some(middle);
            chain.finger_rest.ring.proximal = Some(ring);
            let palm = palm_capsules(chain);
            assert_eq!(palm.len(), 4);
            for joint in [index, middle, ring, little] {
                let point = Vector::from_array(joint.rest.position.to_array().map(f64::from));
                assert!(palm.iter().any(|c| {
                    let [a, b] = c
                        .endpoints
                        .map(|p| Vector::from_array(p.to_array().map(f64::from)));
                    segment_distance(&Segment::new(point, point), &Segment::new(a, b))
                        <= f64::from(c.radius) + 1.0e-6
                }));
            }
            // The wrist shell follows knuckle spacing, not a mesh-bound cuff.
            let forward = (index.rest.position + little.rest.position - wrist * 2.0).normalize();
            for c in palm {
                assert_eq!(c.bone, chain.hand);
                assert!(c.radius < 0.01);
                assert!(c.endpoints.iter().all(|p| (*p - wrist).dot(forward) >= 0.0));
            }
            chain.finger_rest.middle.proximal = None;
            assert!(palm_capsules(chain).is_empty());
        }
    }

    #[test]
    fn proximal_upper_arm_contact_is_checked_outside_the_shoulder_socket() {
        // The old distal-quarter trim omitted this contact entirely.
        let points: Vec<_> = [0.0, -0.3]
            .into_iter()
            .flat_map(|y| {
                [-0.03, 0.03]
                    .into_iter()
                    .flat_map(move |x| [-0.03, 0.03].into_iter().map(move |z| Vec3::new(x, y, z)))
            })
            .collect();
        let mut capsules = fit_capsules(
            entity(2),
            Region::Upper(ArmSide::Left),
            &points,
            Quat::IDENTITY,
        )
        .unwrap();
        capsules.push(CapsuleCollider {
            endpoints: [Vec3::new(0.0, -0.12, 0.0); 2],
            radius: 0.03,
            ..capsule(1, Region::Torso, 0.0)
        });
        let mut geometry = CollisionGeometry::new(capsules, &[]);
        geometry.shoulders.push(ShoulderSocket {
            bone: entity(2),
            pivot: Vec3::ZERO,
            radius: 0.05,
        });
        assert!(!geometry.pose_is_clear(&identity, 0.0).unwrap());
        assert!(
            !geometry
                .sweep_is_clear(&identity, &identity, |_| 0.0)
                .unwrap()
        );
    }

    #[test]
    fn capsule_distance_and_spherical_degeneracy() {
        let a = Segment::new(Vector::ZERO, Vector::X);
        let b = Segment::new(Vector::Y, Vector::Y);
        assert!((segment_distance(&a, &b) - 1.0).abs() < 1e-12);
        let g = CollisionGeometry::new(
            vec![
                capsule(1, Region::Torso, 0.0),
                capsule(2, Region::Upper(ArmSide::Left), 0.15),
            ],
            &[],
        );
        assert!(!g.pose_is_clear(&identity, 0.0).unwrap());
        assert!((g.solver_margins(identity, 0.0, None).unwrap().0[0] + 0.05).abs() < 1e-6);
    }

    #[test]
    fn visible_landmark_moves_only_in_depth_and_preserves_clear_targets() {
        let g = CollisionGeometry::new(
            vec![
                capsule(1, Region::Torso, 0.0),
                CapsuleCollider {
                    radius: 0.05,
                    ..capsule(2, Region::Forearm(ArmSide::Left), 0.4)
                },
            ],
            &[],
        );
        let body = HashMap::from([(entity(1), identity(entity(1)).unwrap())]);
        let input = Vec3::new(0.0, 0.0, -0.05);
        let visible = g.visible_forearm_target([input; 2], Vec3::Z, ArmSide::Left, &body)[0];
        assert!(visible.distance(Vec3::new(0.0, 0.0, 0.15)) < 1.0e-6);
        for point in [Vec3::new(0.0, 0.0, 0.25), Vec3::new(0.3, 0.0, -0.2)] {
            assert_eq!(
                g.visible_forearm_target([point; 2], Vec3::Z, ArmSide::Left, &body),
                [point; 2]
            );
        }
        // Both endpoints are outside the silhouette, but their connecting
        // forearm passes through the torso. Point raycasts would miss it.
        let across = [Vec3::new(-0.3, 0.0, 0.0), Vec3::new(0.3, 0.0, 0.0)];
        let visible = g.visible_forearm_target(across, Vec3::Z, ArmSide::Left, &body);
        for (input, output) in across.into_iter().zip(visible) {
            assert!(output.distance(input + Vec3::Z * 0.15) < 1.0e-6);
        }
        let rotation = Quat::from_euler(EulerRot::XYZ, 0.2, 0.6, -0.3);
        let body = HashMap::from([(
            entity(1),
            BoneMotion {
                rotation,
                translation: Vec3::ZERO,
            },
        )]);
        let rotated = g.visible_forearm_target(
            [rotation * input; 2],
            rotation * Vec3::Z,
            ArmSide::Left,
            &body,
        )[0];
        assert!(rotated.distance(rotation * Vec3::new(0.0, 0.0, 0.15)) < 1.0e-6);
    }
    #[test]
    fn adjacent_links_do_not_exclude_the_torso_or_opposite_arm() {
        let g = CollisionGeometry::new(
            vec![
                capsule(1, Region::Torso, 0.0),
                capsule(2, Region::Upper(ArmSide::Left), 0.0),
                capsule(3, Region::Forearm(ArmSide::Left), 0.0),
                capsule(4, Region::Upper(ArmSide::Right), 0.0),
            ],
            &[],
        );
        assert!(!g.pairs.contains(&[1, 2]));
        assert!(g.pairs.contains(&[0, 1]));
        assert!(g.pairs.contains(&[1, 3]));
    }
    #[test]
    fn shoulder_overlap_is_local_and_does_not_exempt_other_links() {
        let mut g = CollisionGeometry::new(
            vec![
                capsule(1, Region::Torso, 0.0),
                CapsuleCollider {
                    bone: entity(2),
                    region: Region::Upper(ArmSide::Left),
                    endpoints: [Vec3::new(0.1, 0.2, 0.0), Vec3::new(0.1, 0.5, 0.0)],
                    radius: 0.04,
                },
            ],
            &[],
        );
        assert!(!g.pose_is_clear(&identity, 0.0).unwrap());
        g.shoulders.push(ShoulderSocket {
            bone: entity(2),
            pivot: Vec3::new(0.1, 0.2, 0.0),
            radius: 0.12,
        });
        assert!(g.pose_is_clear(&identity, 0.0).unwrap());
        assert!(g.sweep_is_clear(&identity, &identity, |_| 0.0).unwrap());
        // A narrow blended skin seam still includes the capsule rounding
        // shells. Their contact must not be mistaken for a distant arm/body
        // intersection that requires opening the armpit.
        let mut narrow_seam = g.clone();
        narrow_seam.shoulders[0].radius = 0.04;
        assert!(narrow_seam.pose_is_clear(&identity, 0.0).unwrap());
        assert!(
            narrow_seam
                .sweep_is_clear(&identity, &identity, |_| 0.0)
                .unwrap()
        );
        // Move the arm/socket below the chest: the distal part now intersects
        // the torso outside the socket, despite involving the same two bones.
        let moved = |bone| {
            Some(BoneMotion {
                rotation: Quat::IDENTITY,
                translation: if bone == entity(2) {
                    Vec3::NEG_Y * 0.5
                } else {
                    Vec3::ZERO
                },
            })
        };
        assert!(!g.pose_is_clear(&moved, 0.0).unwrap());
        assert!(!g.sweep_is_clear(&identity, &moved, |_| 0.0).unwrap());
        g.capsules[1].region = Region::Forearm(ArmSide::Left);
        assert!(!g.pose_is_clear(&identity, 0.0).unwrap());
    }

    #[test]
    fn a_socket_does_not_admit_a_sweep_through_the_chest() {
        let mut g = CollisionGeometry::new(
            vec![
                capsule(1, Region::Torso, 0.0),
                capsule(2, Region::Upper(ArmSide::Left), -0.5),
            ],
            &[],
        );
        g.shoulders.push(ShoulderSocket {
            bone: entity(2),
            pivot: Vec3::new(-0.5, 0.2, 0.0),
            radius: 0.12,
        });
        let end = |bone| {
            Some(BoneMotion {
                rotation: Quat::IDENTITY,
                translation: if bone == entity(2) {
                    Vec3::X
                } else {
                    Vec3::ZERO
                },
            })
        };
        assert!(g.pose_is_clear(&identity, 0.0).unwrap());
        assert!(g.pose_is_clear(&end, 0.0).unwrap());
        assert!(!g.sweep_is_clear(&identity, &end, |_| 0.0).unwrap());
    }
    #[test]
    fn swept_capsules_reject_crossing_with_clear_endpoints() {
        let g = CollisionGeometry::new(
            vec![
                capsule(1, Region::Torso, 0.0),
                capsule(2, Region::Hand(ArmSide::Left), -0.5),
            ],
            &[],
        );
        let end = |b| {
            Some(BoneMotion {
                rotation: Quat::IDENTITY,
                translation: if b == entity(2) { Vec3::X } else { Vec3::ZERO },
            })
        };
        assert!(g.pose_is_clear(&identity, 0.0).unwrap());
        assert!(g.pose_is_clear(&end, 0.0).unwrap());
        assert!(!g.sweep_is_clear(&identity, &end, |_| 0.0).unwrap());
        let tangent = |b| {
            Some(BoneMotion {
                rotation: Quat::IDENTITY,
                translation: if b == entity(2) { Vec3::Z } else { Vec3::ZERO },
            })
        };
        assert!(g.sweep_is_clear(&identity, &tangent, |_| 0.0).unwrap());
    }
    #[test]
    fn bind_shapes_ignore_authored_bone_axes_for_the_same_skinned_t_pose() {
        use crate::{arm::ArmSide, binding::AvatarBinding, lifecycle::AvatarGeneration};
        use bevy::{asset::RenderAssetUsages, mesh::PrimitiveTopology};

        let bind = |rotation: Quat| {
            let mut app = App::new();
            app.add_plugins(MinimalPlugins)
                .init_resource::<Assets<Mesh>>()
                .init_resource::<Assets<SkinnedMeshInverseBindposes>>()
                .add_systems(Update, bind_collision_geometry);
            let root = app.world_mut().spawn_empty().id();
            let positions = [
                Vec3::new(0.0, 1.1, 0.0),
                Vec3::new(0.3, 1.35, 0.0),
                Vec3::new(0.6, 1.35, 0.0),
                Vec3::new(0.86, 1.35, 0.0),
            ];
            let globals = positions.map(|p| {
                GlobalTransform::from(Transform::from_translation(p).with_rotation(rotation))
            });
            let bones = globals.map(|g| app.world_mut().spawn((g, RestGlobalTransform(g))).id());
            let [hips, upper, lower, hand] = bones;
            let mut chain = crate::upper_limb::tests::chain(ArmSide::Left);
            chain.shoulder = None;
            chain.rest.shoulder = None;
            chain.upper_arm = upper;
            chain.lower_arm = lower;
            chain.hand = hand;
            chain.finger_rest = Default::default();
            for (rest, p) in [
                &mut chain.rest.upper_arm,
                &mut chain.rest.elbow,
                &mut chain.rest.wrist,
            ]
            .into_iter()
            .zip(positions.into_iter().skip(1))
            {
                rest.position = p;
                rest.global_rotation = rotation;
            }
            let mut binding = AvatarBinding::head_only(root, root, AvatarGeneration(1));
            binding.left_arm = Some(chain);
            app.world_mut()
                .entity_mut(root)
                .insert((binding, HipsBoneEntity(hips)));
            let mut vertices = Vec::new();
            let mut indices = Vec::new();
            let mut weights = Vec::new();
            let sizes = [
                Vec3::new(0.18, 0.25, 0.07),
                Vec3::new(0.13, 0.045, 0.04),
                Vec3::new(0.12, 0.035, 0.03),
                Vec3::new(0.05, 0.012, 0.03),
            ];
            for (i, (centre, size)) in positions.into_iter().zip(sizes).enumerate() {
                for x in [-1.0, 1.0] {
                    for y in [-1.0, 1.0] {
                        for z in [-1.0, 1.0] {
                            vertices.push((centre + size * Vec3::new(x, y, z)).to_array());
                            indices.push([i as u16, 0, 0, 0]);
                            weights.push([1.0, 0.0, 0.0, 0.0]);
                        }
                    }
                }
            }
            let mut mesh = Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::default(),
            );
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
            mesh.insert_attribute(
                Mesh::ATTRIBUTE_JOINT_INDEX,
                VertexAttributeValues::Uint16x4(indices),
            );
            mesh.insert_attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT, weights);
            let mesh = app.world_mut().resource_mut::<Assets<Mesh>>().add(mesh);
            let inverse_bindposes = app
                .world_mut()
                .resource_mut::<Assets<SkinnedMeshInverseBindposes>>()
                .add(SkinnedMeshInverseBindposes::from(
                    globals.map(|g| g.to_matrix().inverse()).to_vec(),
                ));
            app.world_mut().spawn((
                Mesh3d(mesh),
                SkinnedMesh {
                    inverse_bindposes,
                    joints: bones.to_vec(),
                },
                ChildOf(root),
            ));
            app.update();
            app.world()
                .get::<AvatarCollision>(root)
                .unwrap()
                .0
                .as_ref()
                .unwrap()
                .capsules
                .clone()
        };
        let normalized = bind(Quat::IDENTITY);
        let authored = bind(Quat::from_euler(EulerRot::XYZ, 0.34, -0.5, 0.7));
        assert_eq!(normalized.len(), authored.len());
        for (a, b) in normalized.iter().zip(authored) {
            assert_eq!(a.region, b.region);
            assert!((a.radius - b.radius).abs() < 2.0e-6);
            for (a, b) in a.endpoints.into_iter().zip(b.endpoints) {
                assert!(a.distance(b) < 2.0e-6);
            }
        }
    }

    #[test]
    fn broad_torso_uses_several_capsules_and_preserves_scale() {
        let points: Vec<_> = [-0.18, 0.18]
            .into_iter()
            .flat_map(|x| {
                [-0.25, 0.25]
                    .into_iter()
                    .flat_map(move |y| [-0.07, 0.07].map(move |z| Vec3::new(x, y, z)))
            })
            .collect();
        let a = fit_capsules(entity(1), Region::Torso, &points, Quat::IDENTITY).unwrap();
        let b = fit_capsules(
            entity(1),
            Region::Torso,
            &points.iter().map(|p| *p * 3.0).collect::<Vec<_>>(),
            Quat::IDENTITY,
        )
        .unwrap();
        assert_eq!(a.len(), 3);
        let rotation = Quat::from_euler(EulerRot::XYZ, 0.3, 0.7, -0.4);
        let rotated = fit_capsules(
            entity(1),
            Region::Torso,
            &points.iter().map(|p| rotation * p).collect::<Vec<_>>(),
            rotation,
        )
        .unwrap();
        for (a, b) in a.iter().zip(rotated) {
            assert!((a.radius - b.radius).abs() < 1e-6);
            assert!((rotation * a.endpoints[0]).distance(b.endpoints[0]) < 1e-6);
        }
        for (a, b) in a.iter().zip(b) {
            assert!((a.radius * 3.0 - b.radius).abs() < 1e-6);
            assert!((a.endpoints[0] * 3.0).distance(b.endpoints[0]) < 1e-6);
        }
    }
}
