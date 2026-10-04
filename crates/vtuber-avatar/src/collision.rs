//! Convex envelopes of the avatar's rendered skin/clothing. Joint allowances
//! cover only the authored skin-blend region, not a whole adjacent link pair.

use bevy::mesh::{
    VertexAttributeValues,
    morph::{MeshMorphWeights, MorphWeights},
    skinning::{SkinnedMesh, SkinnedMeshInverseBindposes},
};
use bevy::prelude::*;
use bevy_vrm1::prelude::*;
use parry3d::{math::Pose, shape::ConvexPolyhedron};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Region {
    Torso,
    Upper(crate::arm::ArmSide),
    Forearm(crate::arm::ArmSide),
    Hand(crate::arm::ArmSide),
}

#[derive(Clone, Debug)]
pub(crate) struct Hull {
    pub bone: Entity,
    pub region: Region,
    /// Model/rest-space vertices, not bone-local vertices.
    pub shape: ConvexPolyhedron,
    /// Exact skin influences of the vertices enclosing all owned triangles.
    pub skin: Vec<SkinVertex>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SkinVertex {
    pub point: Vec3,
    pub influences: Vec<(Entity, Vec3, f32)>,
}

/// Import-only coordinates. Equal joint indices/weights in one primitive
/// share one affine skin map, even when they blend several bones.
#[derive(Clone, PartialEq)]
struct BindVertex {
    source: Vec3,
    blend: ([u16; 4], [u32; 4]),
    skin: SkinVertex,
}

impl SkinVertex {
    fn posed(
        &self,
        motion: &impl Fn(Entity) -> Option<BoneMotion>,
    ) -> Result<Vec3, CollisionError> {
        let mut skinned = Vec3::ZERO;
        for &(bone, point, weight) in &self.influences {
            skinned += motion(bone)
                .ok_or(CollisionError::MissingBone)?
                .point(point)
                * weight;
        }
        Ok(skinned)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct JointAllowance {
    pub bones: [Entity; 2],
    pub pivot_bone: Entity,
    pub pivot: Vec3,
    /// Radius enclosing the skin-blend region and boundary triangles.
    pub radius: f32,
}

#[derive(Component, Clone, Debug, Default)]
pub(crate) struct CollisionGeometry {
    pub hulls: Vec<Hull>,
    pub joints: Vec<JointAllowance>,
    pairs: Vec<CollisionPair>,
    bones: Vec<Entity>,
    hull_bones: Vec<Vec<Entity>>,
}

#[derive(Clone, Debug)]
struct CollisionPair {
    a: usize,
    b: usize,
    joint: Option<usize>,
    bones: Vec<Entity>,
}

/// Shared only while differentiating one fixed configuration. A column may
/// reuse a hull iff none of its skin influences changed from that configuration.
pub(crate) struct DifferentialCache {
    hulls: Vec<std::sync::OnceLock<Result<std::sync::Arc<PosedHull>, CollisionError>>>,
}

struct SolverPairs<'a> {
    margin: f32,
    frozen: Option<&'a [bool]>,
    active: Vec<bool>,
    cache: Option<(&'a DifferentialCache, &'a std::collections::HashSet<Entity>)>,
}

#[derive(Component)]
/// Bound render-geometry result consumed by the upper-limb system.
pub struct AvatarCollision(pub(crate) Result<std::sync::Arc<CollisionGeometry>, CollisionError>);

#[derive(Component, Default)]
struct CollisionMorphs(Vec<(Entity, Vec<(usize, f32)>)>);

/// Bind imported render assets and rebind changed upper-body morphs, including visible clothing
/// primitives. Each vertex is assigned to the strongest skin influence; helper
/// joints are folded into their nearest controlled anatomical ancestor.
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
            for (a, b, pivot) in [
                (chain.upper_arm, chain.lower_arm, chain.rest.elbow.position),
                (chain.lower_arm, chain.hand, chain.rest.wrist.position),
            ] {
                connections.push(JointAllowance {
                    bones: [a, b],
                    pivot_bone: b,
                    pivot,
                    radius: 0.0,
                });
            }
            // The upper-arm socket is connected to the torso and optional
            // clavicle, not to the contralateral upper arm.
            for torso in [
                root.get::<HipsBoneEntity>().map(|b| b.0),
                binding.spine,
                binding.chest,
                binding.upper_chest,
                chain.shoulder,
            ]
            .into_iter()
            .flatten()
            {
                connections.push(JointAllowance {
                    bones: [torso, chain.upper_arm],
                    pivot_bone: chain.upper_arm,
                    pivot: chain.rest.upper_arm.position,
                    radius: 0.0,
                });
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
        // The arm compositor restores helpers below controlled arm bones.
        // Other helpers (e.g. a skirt spring) retain their actual animation.
        let skin_bone = |bone, owner: Option<(Entity, Region)>| {
            owner
                .filter(|(owner, region)| {
                    *region != Region::Torso
                        || [binding.left_arm, binding.right_arm]
                            .into_iter()
                            .flatten()
                            .any(|a| a.shoulder == Some(*owner))
                })
                .map_or(bone, |(owner, _)| owner)
        };
        let rest = |entity| -> Result<GlobalTransform, CollisionError> {
            let (global, rest) = bones.get(entity).map_err(|_| CollisionError::MissingBone)?;
            Ok(rest.map_or(*global, |r| r.0))
        };
        let build = || -> Result<(CollisionGeometry, CollisionMorphs), CollisionError> {
            let mut morph_inputs = CollisionMorphs::default();
            let mut points = HashMap::<(Entity, Entity), Vec<Vec3>>::new();
            let mut envelopes = HashMap::<(Entity, Entity), Vec<BindVertex>>::new();
            let mut joints = connections.clone();
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
                        let mut skin_vertices = Vec::with_capacity(vertices.len());
                        for (vertex_index, (indices, weights)) in
                            indices.iter().zip(weights).enumerate()
                        {
                            let vertex = position(vertex_index)?;
                            let mut point = Vec3::ZERO;
                            let mut strongest = (0.0, None);
                            let mut owners = Vec::new();
                            let mut influences = Vec::new();
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
                                if let Some((bone, _)) = owner {
                                    owners.push(bone);
                                }
                                influences.push((skin_bone(bone, owner), rest_point, weight));
                                if weight > strongest.0 {
                                    strongest = (weight, owner);
                                }
                            }
                            if !point.is_finite() {
                                return Err(CollisionError::InvalidMesh);
                            }
                            skin_vertices.push((
                                strongest.1.map(|(b, _)| b),
                                BindVertex {
                                    source: vertex,
                                    blend: (*indices, weights.map(f32::to_bits)),
                                    skin: SkinVertex { point, influences },
                                },
                            ));
                            for joint in &mut joints {
                                if joint.bones.iter().all(|bone| owners.contains(bone)) {
                                    joint.radius = joint.radius.max(point.distance(joint.pivot));
                                }
                            }
                        }
                        let triangles: Vec<_> = mesh
                            .indices()
                            .map(|i| i.iter().collect())
                            .unwrap_or_else(|| (0..vertices.len()).collect());
                        for triangle in triangles.chunks_exact(3) {
                            let corners = triangle
                                .iter()
                                .map(|i| skin_vertices.get(*i).ok_or(CollisionError::InvalidMesh))
                                .collect::<Result<Vec<_>, _>>()?;
                            let mut owners: Vec<_> =
                                corners.iter().filter_map(|(owner, _)| *owner).collect();
                            owners.sort();
                            owners.dedup();
                            if !owners.is_empty() {
                                for index in triangle {
                                    include_morphs(*index);
                                }
                            }
                            for bone in &owners {
                                for (_, vertex) in &corners {
                                    points
                                        .entry((*bone, entity))
                                        .or_default()
                                        .push(vertex.skin.point);
                                    envelopes
                                        .entry((*bone, entity))
                                        .or_default()
                                        .push(vertex.clone());
                                }
                            }
                            for joint in &mut joints {
                                if joint.bones.iter().all(|b| owners.contains(b)) {
                                    for (_, vertex) in &corners {
                                        joint.radius = joint
                                            .radius
                                            .max(vertex.skin.point.distance(joint.pivot));
                                    }
                                }
                            }
                        }
                    } else if let Some((bone, _)) = classify(entity) {
                        let transform = rest(entity)?;
                        let used: Vec<_> = mesh
                            .indices()
                            .map(|i| i.iter().collect())
                            .unwrap_or_else(|| (0..vertices.len()).collect());
                        for index in used {
                            include_morphs(index);
                            let point = transform.transform_point(position(index)?);
                            points.entry((bone, entity)).or_default().push(point);
                            envelopes
                                .entry((bone, entity))
                                .or_default()
                                .push(BindVertex {
                                    source: point,
                                    blend: ([0; 4], [1.0_f32.to_bits(), 0, 0, 0]),
                                    skin: SkinVertex {
                                        point,
                                        influences: vec![(
                                            skin_bone(entity, classify(entity)),
                                            point,
                                            1.0,
                                        )],
                                    },
                                });
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
            let mut hulls = Vec::new();
            // Separate render primitives (e.g. wings, skin, sleeves) must
            // not fill the free space between them with one convex envelope.
            let mut parts: Vec<_> = points.into_iter().collect();
            parts.sort_by_key(|(key, _)| *key);
            for (key @ (bone, _), points) in parts {
                let (_, region) = classify(bone).ok_or(CollisionError::MissingBone)?;
                let points: Vec<_> = points
                    .into_iter()
                    .map(|p| parry3d::math::Vector::from_array(p.to_array().map(f64::from)))
                    .collect();
                let shape = ConvexPolyhedron::from_convex_hull(&points)
                    .ok_or(CollisionError::InvalidMesh)?;
                let mut vertices = envelopes.remove(&key).unwrap_or_default();
                vertices.sort_by(|a, b| {
                    a.blend
                        .cmp(&b.blend)
                        .then(a.source.x.total_cmp(&b.source.x))
                        .then(a.source.y.total_cmp(&b.source.y))
                        .then(a.source.z.total_cmp(&b.source.z))
                });
                vertices.dedup();
                let mut skin = reduce_skin_vertices(vertices);
                skin.sort_by(|a, b| {
                    a.point
                        .x
                        .total_cmp(&b.point.x)
                        .then(a.point.y.total_cmp(&b.point.y))
                        .then(a.point.z.total_cmp(&b.point.z))
                });
                hulls.push(Hull {
                    bone,
                    region,
                    shape,
                    skin,
                });
            }
            hulls.sort_by_key(|h| h.bone);
            if !hulls.iter().any(|h| h.region == Region::Torso) {
                return Err(CollisionError::InvalidMesh);
            }
            for chain in [binding.left_arm, binding.right_arm].into_iter().flatten() {
                for region in [
                    Region::Upper(chain.side),
                    Region::Forearm(chain.side),
                    Region::Hand(chain.side),
                ] {
                    if !hulls.iter().any(|h| h.region == region) {
                        return Err(CollisionError::InvalidMesh);
                    }
                }
            }
            Ok((CollisionGeometry::new(hulls, joints), morph_inputs))
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

// A fixed set of skin weights is one affine map of source coordinates.
// Its convex interior stays interior under any bone pose. Group before hull
// reduction: vertices with different weights cannot share that guarantee.
fn reduce_skin_vertices(vertices: Vec<BindVertex>) -> Vec<SkinVertex> {
    let mut groups = std::collections::BTreeMap::<_, Vec<BindVertex>>::new();
    for vertex in vertices {
        groups.entry(vertex.blend).or_default().push(vertex);
    }
    let mut result = Vec::new();
    for vertices in groups.into_values() {
        let points: Vec<_> = vertices
            .iter()
            .map(|v| parry3d::math::Vector::from_array(v.source.to_array().map(f64::from)))
            .collect();
        if let Some(hull) = ConvexPolyhedron::from_convex_hull(&points) {
            let extreme: std::collections::HashSet<_> = hull
                .points()
                .iter()
                .map(|p| p.to_array().map(|v| (v as f32).to_bits()))
                .collect();
            result.extend(
                vertices
                    .into_iter()
                    .filter(|v| extreme.contains(&v.source.to_array().map(f32::to_bits)))
                    .map(|v| v.skin),
            );
        } else {
            result.extend(vertices.into_iter().map(|v| v.skin));
        }
    }
    result
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

/// Failure to construct or evaluate the rendered upper-body collision shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollisionError {
    /// Missing, malformed or degenerate render mesh.
    InvalidMesh,
    /// A render skin references an unavailable bone.
    MissingBone,
}

impl CollisionGeometry {
    pub fn new(hulls: Vec<Hull>, joints: Vec<JointAllowance>) -> Self {
        let mut seen = std::collections::HashSet::new();
        let bones = hulls
            .iter()
            .flat_map(|h| {
                std::iter::once(h.bone).chain(
                    h.skin
                        .iter()
                        .flat_map(|s| s.influences.iter().map(|(b, _, _)| *b)),
                )
            })
            .chain(joints.iter().map(|joint| joint.pivot_bone))
            .filter(|bone| seen.insert(*bone))
            .collect();
        let hull_bones = hulls
            .iter()
            .map(|h| {
                let mut bones: Vec<_> = std::iter::once(h.bone)
                    .chain(
                        h.skin
                            .iter()
                            .flat_map(|v| v.influences.iter().map(|(b, _, _)| *b)),
                    )
                    .collect();
                bones.sort_unstable();
                bones.dedup();
                bones
            })
            .collect();
        let mut geometry = Self {
            hull_bones,
            hulls,
            joints,
            pairs: Vec::new(),
            bones,
        };
        let participates = |a: &Hull, b: &Hull| {
            a.bone != b.bone
                && !(a.region == Region::Torso && b.region == Region::Torso)
                && !matches!((a.region,b.region), (Region::Hand(x),Region::Hand(y)) if x==y)
        };
        for (i, a) in geometry.hulls.iter().enumerate() {
            for (j, b) in geometry.hulls.iter().enumerate().skip(i + 1) {
                if !participates(a, b) {
                    continue;
                }
                let joint = geometry.joints.iter().enumerate().find(|(_, j)| {
                    if j.bones == [a.bone, b.bone] || j.bones == [b.bone, a.bone] {
                        return true;
                    }
                    // Finger bases share the wrist connection with the
                    // hand. Their separate hulls must not turn that
                    // authored connection into forearm self-collision.
                    let forearm = match (a.region, b.region) {
                        (Region::Hand(x), Region::Forearm(y)) if x == y => Some((b.bone, x)),
                        (Region::Forearm(x), Region::Hand(y)) if x == y => Some((a.bone, x)),
                        _ => None,
                    };
                    forearm.is_some_and(|(bone, side)| {
                        j.bones.contains(&bone)
                            && j.bones.iter().any(|b| {
                                geometry
                                    .hulls
                                    .iter()
                                    .any(|h| h.bone == *b && h.region == Region::Hand(side))
                            })
                    })
                });
                let mut bones: Vec<_> = [i, j]
                    .into_iter()
                    .filter_map(|index| geometry.hull_bones.get(index))
                    .flatten()
                    .copied()
                    .collect();
                if let Some((_, joint)) = joint {
                    bones.push(joint.pivot_bone);
                }
                bones.sort_unstable();
                bones.dedup();
                geometry.pairs.push(CollisionPair {
                    a: i,
                    b: j,
                    joint: joint.map(|(index, _)| index),
                    bones,
                });
            }
        }
        geometry
    }

    /// A contact has zero derivative when none of its skin influences or
    /// its allowed joint pivot moves. Do not rerun geometry for those rows.
    pub fn differential_pairs(
        &self,
        active: &[bool],
        moved: &std::collections::HashSet<Entity>,
    ) -> Vec<bool> {
        self.pairs
            .iter()
            .zip(active)
            .map(|(pair, active)| *active && pair.bones.iter().any(|bone| moved.contains(bone)))
            .collect()
    }

    pub fn bones(&self) -> impl Iterator<Item = Entity> + '_ {
        self.bones.iter().copied()
    }
    /// A positive result is separated/within an allowed joint connection;
    /// zero is contact, negative is forbidden volume overlap.
    #[cfg(test)]
    pub fn clearance(
        &self,
        motion: impl Fn(Entity) -> Option<BoneMotion>,
    ) -> Result<f32, CollisionError> {
        self.clearance_with_padding(motion, 0.0)
    }

    /// Enclose a path interval by adding a common displacement bound to each
    /// posed envelope and subtracting it from the allowed joint connection.
    #[cfg(test)]
    pub fn clearance_with_padding(
        &self,
        motion: impl Fn(Entity) -> Option<BoneMotion>,
        padding: f32,
    ) -> Result<f32, CollisionError> {
        Ok(self
            .margins(motion, padding)?
            .into_iter()
            .fold(f32::INFINITY, f32::min))
    }

    #[cfg(test)]
    pub fn margins(
        &self,
        motion: impl Fn(Entity) -> Option<BoneMotion>,
        padding: f32,
    ) -> Result<Vec<f32>, CollisionError> {
        self.margins_with_padding(motion, |_| padding, |_| padding)
    }

    #[cfg(test)]
    pub fn margins_with_padding(
        &self,
        motion: impl Fn(Entity) -> Option<BoneMotion>,
        padding: impl Fn(usize) -> f32,
        joint_padding: impl Fn(usize) -> f32,
    ) -> Result<Vec<f32>, CollisionError> {
        self.swept_margins(&motion, None, padding, joint_padding)
    }

    pub fn differential_cache(&self) -> DifferentialCache {
        DifferentialCache {
            hulls: (0..self.hulls.len()).map(|_| Default::default()).collect(),
        }
    }

    /// Freeze the potentially active pairs while differentiating one SQP
    /// problem. Separated broad-phase bounds are feasibility certificates,
    /// not differentiable contact constraints.
    pub fn solver_margins(
        &self,
        motion: impl Fn(Entity) -> Option<BoneMotion>,
        margin: f32,
        frozen: Option<&[bool]>,
        cache: Option<(&DifferentialCache, &std::collections::HashSet<Entity>)>,
    ) -> Result<(Vec<f32>, Vec<bool>), CollisionError> {
        let mut pairs = SolverPairs {
            margin,
            frozen,
            active: Vec::new(),
            cache,
        };
        let margins =
            self.query_margins(&motion, None, |_| 0.0, |_| 0.0, Some(&mut pairs), false)?;
        Ok((margins, pairs.active))
    }

    /// Convex enclosure of endpoint skins plus a bound on their chord error.
    #[cfg(test)]
    pub fn swept_margins(
        &self,
        motion: &impl Fn(Entity) -> Option<BoneMotion>,
        endpoint: Option<&dyn Fn(Entity) -> Option<BoneMotion>>,
        padding: impl Fn(usize) -> f32,
        joint_padding: impl Fn(usize) -> f32,
    ) -> Result<Vec<f32>, CollisionError> {
        self.query_margins(motion, endpoint, padding, joint_padding, None, false)
    }

    /// Point feasibility for path exploration, without computing unused
    /// constraint gradients or distances beyond the first blocked pair.
    pub fn pose_is_clear(
        &self,
        motion: &impl Fn(Entity) -> Option<BoneMotion>,
    ) -> Result<bool, CollisionError> {
        self.query_margins(motion, None, |_| 0.0, |_| 0.0, None, true)
            .map(|margins| margins.iter().all(|d| *d >= 0.0))
    }

    /// A failed pair suffices to subdivide this interval. Do not compute the
    /// remaining distances for an enclosure that cannot certify the path.
    pub fn sweep_is_clear(
        &self,
        motion: &impl Fn(Entity) -> Option<BoneMotion>,
        endpoint: &dyn Fn(Entity) -> Option<BoneMotion>,
        padding: impl Fn(usize) -> f32,
        joint_padding: impl Fn(usize) -> f32,
    ) -> Result<bool, CollisionError> {
        self.query_margins(motion, Some(endpoint), padding, joint_padding, None, true)
            .map(|margins| margins.iter().all(|d| *d >= 0.0))
    }

    fn query_margins(
        &self,
        motion: &impl Fn(Entity) -> Option<BoneMotion>,
        endpoint: Option<&dyn Fn(Entity) -> Option<BoneMotion>>,
        padding: impl Fn(usize) -> f32,
        joint_padding: impl Fn(usize) -> f32,
        mut solver: Option<&mut SolverPairs<'_>>,
        first_violation: bool,
    ) -> Result<Vec<f32>, CollisionError> {
        let motions = std::cell::RefCell::new(HashMap::new());
        let source_motion = motion;
        let motion = |bone| {
            if let Some(value) = motions.borrow().get(&bone).copied() {
                return Some(value);
            }
            let value = source_motion(bone)?;
            motions.borrow_mut().insert(bone, value);
            Some(value)
        };
        let mut inflated: Vec<Option<std::sync::Arc<PosedHull>>> =
            (0..self.hulls.len()).map(|_| None).collect();
        let pose_hull =
            |i: usize, h: &Hull| -> Result<PosedHull, CollisionError> {
                let owner = motion(h.bone).ok_or(CollisionError::MissingBone)?;
                let points: Result<Vec<_>, CollisionError> = if h.skin.is_empty() {
                    h.shape
                        .points()
                        .iter()
                        .map(|p| Ok(owner.point(Vec3::from_array(p.to_array().map(|v| v as f32)))))
                        .collect()
                } else {
                    h.skin.iter().map(|v| v.posed(&motion)).collect()
                };
                let mut points = points?;
                if let Some(endpoint) = endpoint {
                    let owner = endpoint(h.bone).ok_or(CollisionError::MissingBone)?;
                    if h.skin.is_empty() {
                        points.extend(h.shape.points().iter().map(|p| {
                            owner.point(Vec3::from_array(p.to_array().map(|v| v as f32)))
                        }));
                    } else {
                        for vertex in &h.skin {
                            points.push(vertex.posed(&endpoint)?);
                        }
                    }
                }
                PosedHull::new(points, padding(i))
            };
        let mut clearances = Vec::new();
        for pair in &self.pairs {
            let (i, j) = (pair.a, pair.b);
            let (a, b) = (
                self.hulls.get(i).ok_or(CollisionError::InvalidMesh)?,
                self.hulls.get(j).ok_or(CollisionError::InvalidMesh)?,
            );
            let frozen = solver
                .as_ref()
                .and_then(|s| s.frozen)
                .and_then(|pairs| pairs.get(clearances.len()))
                .copied();
            if frozen == Some(false) {
                clearances.push(1.0);
                if let Some(s) = solver.as_mut() {
                    s.active.push(false);
                }
                continue;
            }
            // Pose a skin only when an evaluated pair needs it. Frozen
            // inactive pairs are constant; a failed sweep can stop before
            // touching the remaining render primitives.
            for (index, hull) in [(i, a), (j, b)] {
                let slot = inflated.get_mut(index).ok_or(CollisionError::InvalidMesh)?;
                if slot.is_none() {
                    let cached =
                        solver
                            .as_ref()
                            .and_then(|s| s.cache)
                            .and_then(|(cache, moved)| {
                                self.hull_bones
                                    .get(index)
                                    .filter(|bones| bones.iter().all(|b| !moved.contains(b)))
                                    .and_then(|_| cache.hulls.get(index))
                            });
                    *slot = Some(if let Some(cached) = cached {
                        cached
                            .get_or_init(|| pose_hull(index, hull).map(std::sync::Arc::new))
                            .clone()?
                    } else {
                        std::sync::Arc::new(pose_hull(index, hull)?)
                    });
                }
            }
            let margin = solver.as_ref().map_or(0.0, |s| s.margin);
            let sa = inflated
                .get(i)
                .and_then(Option::as_deref)
                .ok_or(CollisionError::InvalidMesh)?;
            let sb = inflated
                .get(j)
                .and_then(Option::as_deref)
                .ok_or(CollisionError::InvalidMesh)?;
            let gap =
                sa.centre.distance(sb.centre) - sa.radius - sb.radius - sa.padding - sb.padding;
            if frozen != Some(true) && gap > margin {
                clearances.push(gap);
                if let Some(s) = solver.as_mut() {
                    s.active.push(false);
                }
                continue;
            }
            let joint = pair
                .joint
                .and_then(|index| self.joints.get(index).map(|joint| (index, joint)));
            // A natural joint allowance moves with its pivot. Enclose the
            // endpoint skins in that translating frame, including the
            // pivot's chord error, instead of eroding its ball by the
            // entire first-order travel of the joint.
            let relative_joint = if let Some(((index, joint), endpoint)) = joint.zip(endpoint) {
                let from = motion(joint.pivot_bone)
                    .ok_or(CollisionError::MissingBone)?
                    .point(joint.pivot);
                let to = endpoint(joint.pivot_bone)
                    .ok_or(CollisionError::MissingBone)?
                    .point(joint.pivot);
                Some((
                    sa.relative_sweep(from, to, joint_padding(index))?,
                    sb.relative_sweep(from, to, joint_padding(index))?,
                ))
            } else {
                None
            };
            let (sa, sb) = relative_joint.as_ref().map_or((sa, sb), |(a, b)| (a, b));
            let ma = sa.motion();
            let mb = sb.motion();
            let joint_pivot = |joint: &JointAllowance| {
                if relative_joint.is_some() {
                    Ok(Vec3::ZERO)
                } else {
                    motion(joint.pivot_bone)
                        .ok_or(CollisionError::MissingBone)
                        .map(|m| m.point(joint.pivot))
                }
            };
            let joint_padding = |index| {
                if relative_joint.is_some() {
                    0.0
                } else {
                    joint_padding(index)
                }
            };
            let lo = (sa.min - Vec3::splat(sa.padding)).max(sb.min - Vec3::splat(sb.padding));
            let hi = (sa.max + Vec3::splat(sa.padding)).min(sb.max + Vec3::splat(sb.padding));
            let box_gap = (lo - hi).max(Vec3::ZERO).length();
            if frozen != Some(true) && box_gap > margin {
                clearances.push(box_gap);
                if let Some(s) = solver.as_mut() {
                    s.active.push(false);
                }
                continue;
            }
            if frozen != Some(true)
                && let Some((index, joint)) = joint
            {
                let pivot = joint_pivot(joint)?;
                let radius = (lo - pivot).abs().max((hi - pivot).abs()).length();
                let allowance = joint.radius - joint_padding(index) - radius;
                if allowance >= margin {
                    clearances.push(allowance);
                    if let Some(s) = solver.as_mut() {
                        s.active.push(false);
                    }
                    continue;
                }
                // A 26-DOP encloses the same posed skin with a fixed number
                // of support planes (Klosowski et al., 1998). Most natural
                // joint overlaps fit inside their allowance without building
                // either full polyhedron. Active finite differences always
                // use the complete intersection below. Inflate by the stencil
                // margin too: disjoint coarse hulls must not hide a near contact
                // outside the joint ball from the solver's active set.
                let radius = intersection_radius(
                    [
                        (sa.joint_bounds(), ma, sa.padding + margin),
                        (sb.joint_bounds(), mb, sb.padding + margin),
                    ],
                    pivot,
                    joint.radius - joint_padding(index) - margin,
                );
                let allowance = joint.radius - joint_padding(index) - radius;
                if allowance > margin {
                    clearances.push(allowance);
                    if let Some(s) = solver.as_mut() {
                        s.active.push(false);
                    }
                    continue;
                }
            }
            let relative = Pose::from_translation(
                (sb.centre.as_dvec3() - sa.centre.as_dvec3())
                    .to_array()
                    .into(),
            );
            let prediction =
                sa.centre.distance(sb.centre) + sa.radius + sb.radius + sa.padding + sb.padding;
            // GJK queries the same convex envelope directly through its
            // support map. No candidate-wide QuickHull rebuild is needed.
            let contact = parry3d::query::details::contact_support_map_support_map(
                &relative,
                sa,
                sb,
                f64::from(prediction),
            );
            // No contact is also returned by EPA when penetration depth
            // cannot be resolved. Our prediction encloses both bounds,
            // so use the exact convex SAT in that case as well.
            let mut distance = contact.map_or(0.0, |c| {
                // GJK's converged contact distance is an approximation,
                // not a lower bound for swept-volume admission. Its
                // normal is a separating-plane candidate: project every
                // support vertex in f64 to certify any positive gap.
                let n = bevy::math::DVec3::from_array(c.normal1.to_array());
                let high = bevy::math::DVec3::from_array(
                    sa.support_vertex(n.to_array().into()).to_array(),
                )
                .dot(n);
                let low = bevy::math::DVec3::from_array(
                    sb.support_vertex((-n).to_array().into()).to_array(),
                )
                .dot(n);
                let gap = (((sb.centre - sa.centre).as_dvec3().dot(n) + low - high) / n.length()
                    - f64::from(sa.padding + sb.padding)) as f32;
                if gap > 0.0 {
                    gap
                } else if c.dist < 0.0 {
                    c.dist as f32
                } else {
                    0.0
                }
            });
            if distance == 0.0 && joint.is_none() {
                // GJK may report zero for a symmetric, penetrating pair
                // when its initial simplex contains the origin. SAT on
                // the actual polyhedra distinguishes volume from contact.
                distance = polyhedron_separation(sa.polyhedron()?, ma, sb.polyhedron()?, mb)
                    - sa.padding
                    - sb.padding;
            }
            if distance <= 0.0
                && let Some((joint_index, joint)) = joint
            {
                let pivot_padding = joint_padding(joint_index);
                let pivot = joint_pivot(joint)?;
                // A convex intersection is within a ball iff all its
                // vertices are. Enumerate both sets of clipped edges;
                // checking only GJK's deepest contact could hide a second
                // penetration outside the natural joint connection.
                let lo = (sa.min - Vec3::splat(sa.padding)).max(sb.min - Vec3::splat(sb.padding));
                let hi = (sa.max + Vec3::splat(sa.padding)).min(sb.max + Vec3::splat(sb.padding));
                // The intersection lies inside the intersection of its
                // AABBs. When even that box is inside the joint ball,
                // no face clipping or QuickHull is necessary.
                let box_radius = (lo - pivot).abs().max((hi - pivot).abs()).length();
                let radius = if solver.is_none() && box_radius <= joint.radius - pivot_padding {
                    box_radius
                } else {
                    intersection_radius(
                        [
                            (sa.polyhedron()?, ma, sa.padding),
                            (sb.polyhedron()?, mb, sb.padding),
                        ],
                        pivot,
                        if frozen == Some(true) {
                            0.0
                        } else {
                            joint.radius - pivot_padding - margin
                        },
                    )
                };

                let allowed = joint.radius - pivot_padding - radius;
                // A joint-contained intersection is admissible whether
                // it is a surface or a volume. SAT is needed only outside
                // that connection when GJK cannot distinguish the two.
                if distance == 0.0 && allowed < 0.0 {
                    distance = polyhedron_separation(sa.polyhedron()?, ma, sb.polyhedron()?, mb)
                        - sa.padding
                        - sb.padding;
                }
                distance = if allowed >= 0.0 {
                    allowed
                } else {
                    distance.max(allowed)
                };
            }
            if let Some(s) = solver.as_mut() {
                // An overlapping broad-phase box does not make the real
                // surfaces a contact. Freeze only actual near contacts
                // for the finite-difference stencil; all candidate poses
                // still evaluate every pair before they are accepted.
                s.active.push(frozen == Some(true) || distance <= margin);
            }
            clearances.push(distance);
            if first_violation && distance < 0.0 {
                return Ok(clearances);
            }
        }
        Ok(clearances)
    }
}

/// Implicit convex hull of the current skinned vertices. Faces are only
/// needed for the natural-joint intersection test or degenerate GJK result.
struct PosedHull {
    points: Vec<parry3d::math::Vector>,
    support_bounds: Vec<(parry3d::math::Vector, parry3d::math::Vector)>,
    first: parry3d::math::Vector,
    centre: Vec3,
    radius: f32,
    min: Vec3,
    max: Vec3,
    padding: f32,
    faces: std::sync::OnceLock<Result<HullFaces, CollisionError>>,
    joint_bounds: std::sync::OnceLock<HullFaces>,
}

impl PosedHull {
    fn new(points: Vec<Vec3>, padding: f32) -> Result<Self, CollisionError> {
        let first = *points.first().ok_or(CollisionError::InvalidMesh)?;
        let (min, max) = points
            .iter()
            .fold((first, first), |(a, b), p| (a.min(*p), b.max(*p)));
        let centre = (min + max) * 0.5;
        let radius = points
            .iter()
            .map(|p| p.distance(centre))
            .fold(0.0_f32, f32::max);
        let points: Vec<parry3d::math::Vector> = points
            .into_iter()
            .map(|p| {
                parry3d::math::Vector::from_array((p.as_dvec3() - centre.as_dvec3()).to_array())
            })
            .collect();
        let support_bounds = points
            .chunks(32)
            .map(|chunk| {
                chunk.iter().fold(
                    (
                        parry3d::math::Vector::splat(f64::INFINITY),
                        parry3d::math::Vector::splat(f64::NEG_INFINITY),
                    ),
                    |(min, max), p| (min.min(*p), max.max(*p)),
                )
            })
            .collect();
        Ok(Self {
            points,
            support_bounds,
            first: (first.as_dvec3() - centre.as_dvec3()).to_array().into(),
            centre,
            radius,
            min,
            max,
            padding,
            faces: Default::default(),
            joint_bounds: Default::default(),
        })
    }
    fn relative_sweep(&self, from: Vec3, to: Vec3, padding: f32) -> Result<Self, CollisionError> {
        let split = self.points.len() / 2;
        let points = self
            .points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                Vec3::from_array(p.to_array().map(|v| v as f32)) + self.centre
                    - if i < split { from } else { to }
            })
            .collect();
        Self::new(points, self.padding + padding)
    }
    fn motion(&self) -> BoneMotion {
        BoneMotion {
            rotation: Quat::IDENTITY,
            translation: self.centre,
        }
    }
    fn polyhedron(&self) -> Result<&HullFaces, CollisionError> {
        self.faces
            .get_or_init(|| HullFaces::from_points(&self.points))
            .as_ref()
            .map_err(|e| *e)
    }
    fn support_vertex(&self, direction: parry3d::math::Vector) -> parry3d::math::Vector {
        use parry3d::math::Vector;
        let mut best = (self.first, self.first.dot(direction), 0);
        let update = |best: &mut (Vector, f64, usize), index, p: Vector| {
            let projection = p.dot(direction);
            if projection > best.1 || (projection == best.1 && index < best.2) {
                *best = (p, projection, index);
            }
        };
        // A real point gives a lower bound; a block's positive AABB corner
        // gives an upper bound. Keep exact support without scanning every
        // skin vertex on each GJK iteration.
        for (index, p) in self.points.iter().copied().enumerate().step_by(32) {
            update(&mut best, index, p);
        }
        for (block, (points, &(min, max))) in
            self.points.chunks(32).zip(&self.support_bounds).enumerate()
        {
            let upper = Vector::select(direction.cmpge(Vector::ZERO), max, min).dot(direction);
            if upper < best.1 || (upper == best.1 && block * 32 >= best.2) {
                continue;
            }
            for (offset, p) in points.iter().copied().enumerate() {
                update(&mut best, block * 32 + offset, p);
            }
        }
        best.0
    }
    fn joint_bounds(&self) -> &HullFaces {
        self.joint_bounds.get_or_init(|| {
            use bevy::math::DVec3;
            // Only the initial enclosing box is needed by half-space clipping.
            // Keep its eight corners instead of copying/scanning the skin again.
            let (min, max) = self.support_bounds.iter().fold(
                (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY)),
                |(min, max), (a, b)| {
                    (
                        min.min(DVec3::from_array(a.to_array())),
                        max.max(DVec3::from_array(b.to_array())),
                    )
                },
            );
            let mut points = Vec::with_capacity(8);
            for x in [min.x, max.x] {
                for y in [min.y, max.y] {
                    for z in [min.z, max.z] {
                        points.push(DVec3::new(x, y, z));
                    }
                }
            }
            let mut faces = Vec::with_capacity(26);
            for x in [-1.0, 0.0, 1.0] {
                for y in [-1.0, 0.0, 1.0] {
                    for z in [-1.0, 0.0, 1.0] {
                        if let Some(normal) = DVec3::new(x, y, z).try_normalize() {
                            let extreme = self.support_vertex(normal.to_array().into());
                            let support = normal.dot(DVec3::from_array(extreme.to_array()));
                            faces.push((normal, normal * support));
                        }
                    }
                }
            }
            HullFaces {
                points,
                faces,
                edges: Vec::new(),
            }
        })
    }
}

impl parry3d::shape::SupportMap for PosedHull {
    fn local_support_point(&self, direction: parry3d::math::Vector) -> parry3d::math::Vector {
        let point = self.support_vertex(direction);
        // Preserve the original first-vertex tie rule, including flat faces.
        point
            + direction
                .try_normalize()
                .unwrap_or(parry3d::math::Vector::ZERO)
                * f64::from(self.padding)
    }
}

/// Keep the triangular hull planes. ConvexPolyhedron merges neighbouring
/// triangles by a normal-dot tolerance; those polygons need not be planar.
/// Unmerged f64 hulls keep natural-joint clipping consistent for almost
/// coincident endpoint clouds, which occur when certifying a short path.
#[derive(Clone)]
struct HullFaces {
    points: Vec<bevy::math::DVec3>,
    faces: Vec<(bevy::math::DVec3, bevy::math::DVec3)>,
    edges: Vec<[usize; 2]>,
}

impl HullFaces {
    fn from_points(points: &[parry3d::math::Vector]) -> Result<Self, CollisionError> {
        let source = points;
        let (points, triangles) = parry3d::transformation::try_convex_hull(points)
            .map_err(|_| CollisionError::InvalidMesh)?;
        let mut points: Vec<_> = points
            .iter()
            .map(|p| bevy::math::DVec3::from_array(p.to_array()))
            .collect();
        // Keep all source extrema even if QuickHull treats a near-coplanar
        // point as interior. Triangle indices still address the prefix.
        points.extend(
            source
                .iter()
                .map(|p| bevy::math::DVec3::from_array(p.to_array())),
        );
        let mut faces = Vec::new();
        let mut edges = Vec::new();
        for [a, b, c] in triangles {
            let [a, b, c] = [a as usize, b as usize, c as usize];
            let (&p, &q, &r) = (
                points.get(a).ok_or(CollisionError::InvalidMesh)?,
                points.get(b).ok_or(CollisionError::InvalidMesh)?,
                points.get(c).ok_or(CollisionError::InvalidMesh)?,
            );
            let normal = (q - p)
                .cross(r - p)
                .try_normalize()
                .ok_or(CollisionError::InvalidMesh)?;
            // A numerically thin QuickHull triangle may not define a
            // supporting plane. Its normal is only a direction: use the
            // support of every input vertex to enclose the actual shape.
            let support = points
                .iter()
                .map(|q| normal.dot(*q - p))
                .fold(0.0_f64, f64::max);
            faces.push((normal, p + normal * support));
            for [a, b] in [[a, b], [b, c], [c, a]] {
                edges.push([a.min(b), a.max(b)]);
            }
        }
        edges.sort_unstable();
        edges.dedup();
        Ok(Self {
            points,
            faces,
            edges,
        })
    }
}

fn point64(m: BoneMotion, p: bevy::math::DVec3) -> bevy::math::DVec3 {
    m.rotation.as_dquat() * p + m.translation.as_dvec3()
}

fn polyhedron_separation(a: &HullFaces, ma: BoneMotion, b: &HullFaces, mb: BoneMotion) -> f32 {
    let ap: Vec<_> = a.points.iter().map(|p| point64(ma, *p)).collect();
    let bp: Vec<_> = b.points.iter().map(|p| point64(mb, *p)).collect();
    let directions = |h: &HullFaces, m: BoneMotion| {
        h.edges
            .iter()
            .filter_map(|&[a, b]| {
                Some(m.rotation.as_dquat() * (*h.points.get(b)? - *h.points.get(a)?))
            })
            .collect::<Vec<_>>()
    };
    let ae = directions(a, ma);
    let be = directions(b, mb);
    let axes = a
        .faces
        .iter()
        .map(|(n, _)| ma.rotation.as_dquat() * *n)
        .chain(b.faces.iter().map(|(n, _)| mb.rotation.as_dquat() * *n))
        .chain(ae.iter().flat_map(|a| be.iter().map(move |b| a.cross(*b))));
    let mut separation = f64::NEG_INFINITY;
    for axis in axes {
        let Some(axis) = axis.try_normalize() else {
            continue;
        };
        let interval = |points: &[bevy::math::DVec3]| {
            points
                .iter()
                .map(|p| p.dot(axis))
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
                    (lo.min(v), hi.max(v))
                })
        };
        let (amin, amax) = interval(&ap);
        let (bmin, bmax) = interval(&bp);
        let gap = (bmin - amax).max(amin - bmax);
        if gap > 0.0 {
            return gap as f32;
        }
        separation = separation.max(gap);
    }
    separation as f32
}

/// Clip an enclosing box by both hulls' supporting half-spaces. In
/// particular, do not clip the original triangle edges against displaced
/// planes: a nearly degenerate QuickHull facet is not a reliable boundary
/// edge of that outer polytope. Every retained polygon and cut face belongs
/// to the same enclosing polytope, including after sweep inflation.
fn intersection_radius(
    [(a, ma, padding_a), (b, mb, padding_b)]: [(&HullFaces, BoneMotion, f32); 2],
    pivot: Vec3,
    stop_radius: f32,
) -> f32 {
    use bevy::math::DVec3;
    fn point_order(a: &DVec3, b: &DVec3) -> std::cmp::Ordering {
        a.x.total_cmp(&b.x)
            .then(a.y.total_cmp(&b.y))
            .then(a.z.total_cmp(&b.z))
    }
    let bounds = |h: &HullFaces, m: BoneMotion, padding: f32| {
        let (min, max) = h.points.iter().map(|p| point64(m, *p)).fold(
            (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY)),
            |(min, max), p| (min.min(p), max.max(p)),
        );
        (
            min - DVec3::splat(f64::from(padding)),
            max + DVec3::splat(f64::from(padding)),
        )
    };
    let (amin, amax) = bounds(a, ma, padding_a);
    let (bmin, bmax) = bounds(b, mb, padding_b);
    let lo = amin.max(bmin);
    let hi = amax.min(bmax);
    if lo.cmpgt(hi).any() {
        return 0.0;
    }
    let (x, y, z) = (
        DVec3::X * (hi.x - lo.x),
        DVec3::Y * (hi.y - lo.y),
        DVec3::Z * (hi.z - lo.z),
    );
    let mut polygons = vec![
        vec![lo, lo + y, lo + y + z, lo + z],
        vec![lo + x, lo + x + y, hi, lo + x + z],
        vec![lo, lo + x, lo + x + z, lo + z],
        vec![lo + y, lo + x + y, hi, lo + y + z],
        vec![lo, lo + x, lo + x + y, lo + y],
        vec![lo + z, lo + x + z, hi, lo + y + z],
    ];
    let mut bounds = (lo, hi);
    // Sutherland-Hodgman half-space clipping, with a sorted cap for each
    // new cut. Expanding each plane encloses a Euclidean sweep radius;
    // it does not require a hull rebuild with almost coincident vertices.
    for (shape, motion, padding) in [(a, ma, padding_a), (b, mb, padding_b)] {
        for &(normal, point) in &shape.faces {
            let normal = motion.rotation.as_dquat() * normal;
            let point = point64(motion, point);
            let rounding = 64.0 * f64::EPSILON * (lo.abs().max(hi.abs()) + point.abs()).length();
            let limit = f64::from(padding) + rounding;
            let positive = DVec3::select(normal.cmpge(DVec3::ZERO), bounds.1, bounds.0);
            if normal.dot(positive - point) <= limit {
                continue;
            }
            let negative = DVec3::select(normal.cmpge(DVec3::ZERO), bounds.0, bounds.1);
            if normal.dot(negative - point) > limit {
                return 0.0;
            }
            let mut clipped = Vec::new();
            let mut cap = Vec::new();
            for polygon in polygons {
                if polygon.iter().all(|p| normal.dot(*p - point) <= limit) {
                    clipped.push(polygon);
                    continue;
                }
                let mut next = Vec::new();
                for (&p, &q) in polygon.iter().zip(polygon.iter().cycle().skip(1)) {
                    let dp = normal.dot(p - point) - limit;
                    let dq = normal.dot(q - point) - limit;
                    if dp <= 0.0 {
                        next.push(p);
                    }
                    if (dp <= 0.0) != (dq <= 0.0) {
                        // Evaluate a shared edge in a canonical direction so
                        // its two incident polygons produce the same cut.
                        let cut = if point_order(&p, &q).is_le() {
                            p + (q - p) * (dp / (dp - dq))
                        } else {
                            q + (p - q) * (dq / (dq - dp))
                        };
                        next.push(cut);
                        cap.push(cut);
                    }
                }
                if !next.is_empty() {
                    clipped.push(next);
                }
            }
            cap.sort_by(point_order);
            cap.dedup();
            if cap.len() >= 3 {
                let center = cap.iter().copied().sum::<DVec3>() / cap.len() as f64;
                let u = normal.any_orthonormal_vector();
                let v = normal.cross(u);
                cap.sort_by(|a, b| {
                    let a = *a - center;
                    let b = *b - center;
                    a.dot(v)
                        .atan2(a.dot(u))
                        .total_cmp(&b.dot(v).atan2(b.dot(u)))
                });
                clipped.push(cap);
            }
            polygons = clipped;
            if polygons.is_empty() {
                return 0.0;
            }
            bounds = polygons.iter().flatten().fold(
                (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY)),
                |(min, max), p| (min.min(*p), max.max(*p)),
            );
            // Each intermediate polytope encloses the final intersection.
            // Stop once even its box is inside the allowed joint ball.
            let radius = (bounds.0 - pivot.as_dvec3())
                .abs()
                .max((bounds.1 - pivot.as_dvec3()).abs())
                .length() as f32;
            if radius < stop_radius {
                return radius;
            }
        }
    }
    polygons
        .iter()
        .flatten()
        .map(|p| p.distance(pivot.as_dvec3()))
        .fold(0.0_f64, f64::max) as f32
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic
    )]
    use super::*;
    use crate::arm::ArmSide;

    fn hull(bone: u32, region: Region, min: Vec3, max: Vec3) -> Hull {
        let mut points = Vec::new();
        for x in [min.x, max.x] {
            for y in [min.y, max.y] {
                for z in [min.z, max.z] {
                    points.push(parry3d::math::Vector::new(
                        f64::from(x),
                        f64::from(y),
                        f64::from(z),
                    ));
                }
            }
        }
        Hull {
            bone: Entity::from_raw_u32(bone).unwrap(),
            region,
            shape: ConvexPolyhedron::from_convex_hull(&points).unwrap(),
            skin: vec![],
        }
    }
    const IDENTITY: BoneMotion = BoneMotion {
        rotation: Quat::IDENTITY,
        translation: Vec3::ZERO,
    };

    #[test]
    fn sparse_contact_derivatives_keep_minor_skin_influences_and_joint_pivots() {
        let mut a = hull(1, Region::Torso, Vec3::splat(-0.5), Vec3::splat(0.5));
        let b = hull(2, Region::Upper(ArmSide::Left), Vec3::ZERO, Vec3::ONE);
        let minor = Entity::from_raw_u32(4).unwrap();
        let pivot = Entity::from_raw_u32(5).unwrap();
        a.skin = a
            .shape
            .points()
            .iter()
            .map(|p| {
                let point = Vec3::from_array(p.to_array().map(|v| v as f32));
                SkinVertex {
                    point,
                    influences: vec![(a.bone, point, 0.8), (minor, point, 0.2)],
                }
            })
            .collect();
        let joint = JointAllowance {
            bones: [a.bone, b.bone],
            pivot_bone: pivot,
            pivot: Vec3::ZERO,
            radius: 0.1,
        };
        let c = hull(
            3,
            Region::Forearm(ArmSide::Right),
            Vec3::splat(0.25),
            Vec3::splat(0.75),
        );
        let geometry = CollisionGeometry::new(vec![a, b, c], vec![joint]);
        assert!(geometry.bones().any(|bone| bone == pivot));
        let all = vec![true; geometry.pairs.len()];
        let cache = geometry.differential_cache();
        for id in 1..=6 {
            let moved = Entity::from_raw_u32(id).unwrap();
            let sparse =
                geometry.differential_pairs(&all, &std::collections::HashSet::from([moved]));
            let changed = std::collections::HashSet::from([moved]);
            let evaluate = |delta: f32, pairs: &[bool], cached: bool| {
                geometry
                    .solver_margins(
                        |bone| {
                            Some(BoneMotion {
                                translation: if bone == moved {
                                    Vec3::new(delta, delta * 0.5, 0.0)
                                } else {
                                    Vec3::ZERO
                                },
                                ..IDENTITY
                            })
                        },
                        0.01,
                        Some(pairs),
                        cached.then_some((&cache, &changed)),
                    )
                    .unwrap()
                    .0
            };
            let derivative = |pairs: &[bool], cached| {
                evaluate(0.01, pairs, cached)
                    .into_iter()
                    .zip(evaluate(-0.01, pairs, cached))
                    .map(|(a, b)| (a - b) / 0.02)
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                derivative(&all, false),
                derivative(&sparse, true),
                "bone {id}"
            );
        }
    }

    #[test]
    fn disjoint_joint_bounds_keep_near_contacts_outside_the_allowance_active() {
        let a = hull(
            1,
            Region::Upper(ArmSide::Left),
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.51, 0.01, 0.01),
        );
        let b = hull(
            2,
            Region::Forearm(ArmSide::Left),
            Vec3::new(0.5105, 0.0, 0.0),
            Vec3::new(0.5205, 0.01, 0.01),
        );
        let joint = JointAllowance {
            bones: [a.bone, b.bone],
            pivot_bone: b.bone,
            pivot: Vec3::ZERO,
            radius: 0.01,
        };
        let geometry = CollisionGeometry::new(vec![a, b], vec![joint]);
        let (margins, active) = geometry
            .solver_margins(|_| Some(IDENTITY), 0.001, None, None)
            .unwrap();
        assert_eq!(active, vec![true]);
        assert!(margins[0] > 0.0 && margins[0] < 0.001);
    }

    #[test]
    fn solver_pairs_do_not_activate_at_a_bounding_sphere_contact() {
        let geometry = CollisionGeometry::new(
            vec![
                hull(1, Region::Torso, Vec3::splat(-1.0), Vec3::ONE),
                hull(2, Region::Hand(ArmSide::Left), Vec3::splat(-1.0), Vec3::ONE),
            ],
            vec![],
        );
        for separation in [3.463, 3.464, 3.465] {
            let motion = |bone| {
                Some(BoneMotion {
                    translation: if bone == geometry.hulls[1].bone {
                        Vec3::X * separation
                    } else {
                        Vec3::ZERO
                    },
                    ..IDENTITY
                })
            };
            let (_, active) = geometry.solver_margins(motion, 1.0e-5, None, None).unwrap();
            assert_eq!(active, vec![false]);
            let (inactive, frozen) = geometry
                .solver_margins(motion, 1.0e-5, Some(&active), None)
                .unwrap();
            assert_eq!(inactive, vec![1.0]);
            assert_eq!(frozen, active);
            let (margins, _) = geometry
                .solver_margins(motion, 1.0e-5, Some(&[true]), None)
                .unwrap();
            assert!((margins[0] - (separation - 2.0)).abs() < 1.0e-5);
        }
    }

    #[test]
    fn solver_ignores_separated_surfaces_inside_overlapping_aabbs() {
        let size = Vec3::new(1.0, 0.05, 0.1);
        let geometry = CollisionGeometry::new(
            vec![
                hull(1, Region::Torso, -size, size),
                hull(2, Region::Hand(ArmSide::Left), -size, size),
            ],
            vec![],
        );
        let motion = |bone| {
            Some(BoneMotion {
                rotation: Quat::from_rotation_z(std::f32::consts::FRAC_PI_4),
                translation: if bone == geometry.hulls[1].bone {
                    Vec3::new(-0.2, 0.2, 0.0)
                } else {
                    Vec3::ZERO
                },
            })
        };
        let (distances, active) = geometry.solver_margins(motion, 1.0e-3, None, None).unwrap();
        assert!(distances[0] > 0.17);
        assert_eq!(active, vec![false]);
        let (frozen, _) = geometry
            .solver_margins(motion, 1.0e-3, Some(&active), None)
            .unwrap();
        assert_eq!(frozen, vec![1.0]);
    }

    #[test]
    fn binding_uses_rendered_morphs_and_only_rebuilds_for_shape_changes() {
        use bevy::mesh::morph::MorphAttributes;
        let mut app = App::new();
        app.init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<SkinnedMeshInverseBindposes>>()
            .add_systems(Update, bind_collision_geometry);
        let root = app
            .world_mut()
            .spawn((
                GlobalTransform::IDENTITY,
                MorphWeights::new(vec![0.0, 0.0], None).unwrap(),
            ))
            .id();
        let head = app.world_mut().spawn_empty().id();
        let mut chain = crate::upper_limb::tests::chain(ArmSide::Left);
        chain.shoulder = None;
        chain.rest.shoulder = None;
        chain.finger_rest = Default::default();
        let mut spawn = |position: Vec3, morph: bool| {
            let mut mesh = Mesh::from(Cuboid::new(0.1, 0.1, 0.1));
            if morph {
                let count = mesh.count_vertices();
                let mut deltas =
                    vec![MorphAttributes::new(Vec3::X * 0.1, Vec3::ZERO, Vec3::ZERO); count];
                deltas.extend(vec![MorphAttributes::default(); count]);
                mesh.set_morph_targets(deltas);
            }
            let handle = app.world_mut().resource_mut::<Assets<Mesh>>().add(mesh);
            let entity = app
                .world_mut()
                .spawn((
                    Mesh3d(handle),
                    GlobalTransform::from_translation(position),
                    ChildOf(root),
                ))
                .id();
            if morph {
                app.world_mut()
                    .entity_mut(entity)
                    .insert(MeshMorphWeights::Reference(root));
            }
            entity
        };
        let hips = spawn(Vec3::Y, true);
        chain.upper_arm = spawn(chain.rest.upper_arm.position, false);
        chain.lower_arm = spawn(chain.rest.elbow.position, false);
        chain.hand = spawn(chain.rest.wrist.position, false);
        let mut binding = crate::binding::AvatarBinding::head_only(
            root,
            head,
            crate::lifecycle::AvatarGeneration(1),
        );
        binding.left_arm = Some(chain);
        app.world_mut()
            .entity_mut(root)
            .insert((binding, HipsBoneEntity(hips)));
        app.update();
        let first = app
            .world()
            .get::<AvatarCollision>(root)
            .unwrap()
            .0
            .as_ref()
            .unwrap()
            .clone();
        let max_x = |g: &CollisionGeometry| {
            g.hulls
                .iter()
                .find(|h| h.bone == hips)
                .unwrap()
                .shape
                .points()
                .iter()
                .map(|p| p.x)
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let original_x = max_x(&first);
        app.world_mut()
            .get_mut::<MorphWeights>(root)
            .unwrap()
            .weights_mut()[1] = 1.0;
        app.update();
        assert!(std::sync::Arc::ptr_eq(
            &first,
            app.world()
                .get::<AvatarCollision>(root)
                .unwrap()
                .0
                .as_ref()
                .unwrap()
        ));
        app.world_mut()
            .get_mut::<MorphWeights>(root)
            .unwrap()
            .weights_mut()[0] = 0.5;
        app.update();
        let second = app
            .world()
            .get::<AvatarCollision>(root)
            .unwrap()
            .0
            .as_ref()
            .unwrap();
        assert!(!std::sync::Arc::ptr_eq(&first, second));
        assert!((max_x(second) - original_x - 0.05).abs() < 8.0 * f64::from(f32::EPSILON));
    }

    #[test]
    fn tests_volumes_and_contacts_not_only_joint_centres() {
        let body = hull(
            0,
            Region::Torso,
            Vec3::new(-0.2, 0.8, -0.12),
            Vec3::new(0.2, 1.5, 0.12),
        );
        // Both endpoints lie outside the chest but the forearm crosses it.
        let forearm = hull(
            1,
            Region::Forearm(ArmSide::Left),
            Vec3::new(-0.4, 1.05, 0.0),
            Vec3::new(0.4, 1.12, 0.07),
        );
        let geometry = CollisionGeometry::new(vec![body, forearm], vec![]);
        assert!(geometry.clearance(|_| Some(IDENTITY)).unwrap() < -0.01);
        let touch = BoneMotion {
            translation: Vec3::new(0.0, 0.0, 0.12),
            ..IDENTITY
        };
        assert!(
            geometry
                .clearance(|e| Some(if e == geometry.hulls[1].bone {
                    touch
                } else {
                    IDENTITY
                }))
                .unwrap()
                .abs()
                < 1.0e-6
        );
        let separate = BoneMotion {
            translation: Vec3::new(0.0, 0.0, 0.2),
            ..IDENTITY
        };
        assert!(
            geometry
                .clearance(|e| Some(if e == geometry.hulls[1].bone {
                    separate
                } else {
                    IDENTITY
                }))
                .unwrap()
                > 0.05
        );
    }

    #[test]
    fn touching_adjacent_tips_use_the_joint_allowance_before_separation() {
        let a = Entity::from_raw_u32(1).unwrap();
        let b = Entity::from_raw_u32(2).unwrap();
        let points = [
            Vec3::ZERO,
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, 0.0, 1.0),
        ];
        let shape = |sign: f64| {
            ConvexPolyhedron::from_convex_hull(
                &points
                    .map(|p| parry3d::math::Vector::from_array((p.as_dvec3() * sign).to_array())),
            )
            .unwrap()
        };
        let geometry = CollisionGeometry::new(
            vec![
                Hull {
                    bone: a,
                    region: Region::Upper(ArmSide::Left),
                    shape: shape(1.0),
                    skin: vec![],
                },
                Hull {
                    bone: b,
                    region: Region::Forearm(ArmSide::Left),
                    shape: shape(-1.0),
                    skin: vec![],
                },
            ],
            vec![JointAllowance {
                bones: [a, b],
                pivot_bone: b,
                pivot: Vec3::ZERO,
                radius: 0.1,
            }],
        );
        // The AABBs overlap far outside the joint; the actual hulls touch
        // only at its pivot. That permitted contact must not pin the IK.
        assert!(geometry.clearance(|_| Some(IDENTITY)).unwrap() > 0.09);
    }

    #[test]
    fn coincident_skin_boundaries_do_not_lose_intersection_vertices() {
        let points = [
            [-0.031_f64, -0.04, -0.02],
            [0.043, -0.03, 0.023],
            [-0.04, 0.035, 0.021],
            [0.28, 0.023, 0.017],
            [0.011, 0.032, -0.019],
        ];
        let hull = HullFaces::from_points(&points.map(parry3d::math::Vector::from_array)).unwrap();
        for i in 0..50 {
            let motion = BoneMotion {
                rotation: Quat::from_euler(EulerRot::XYZ, 0.013 * i as f32, 0.73, -0.21),
                translation: Vec3::new(0.18, 1.3, 0.01),
            };
            let expected = hull
                .points
                .iter()
                .map(|p| point64(motion, *p).distance(motion.translation.as_dvec3()))
                .fold(0.0_f64, f64::max);
            let actual = intersection_radius(
                [(&hull, motion, 0.0), (&hull, motion, 0.0)],
                motion.translation,
                0.0,
            );
            assert!((f64::from(actual) - expected).abs() < 4.0 * f64::from(f32::EPSILON));
            for padding in [1.0e-10, 1.0e-7, 1.0e-4] {
                let inflated = intersection_radius(
                    [(&hull, motion, padding), (&hull, motion, padding)],
                    motion.translation,
                    0.0,
                );
                assert!(inflated + 4.0 * f32::EPSILON >= actual);
            }
        }
    }

    #[test]
    fn joint_intersection_is_invariant_under_zero_length_sweeps() {
        // Almost coplanar facets must not become a different clipping plane
        // when the same contact is expressed relative to its moving pivot.
        let points: Vec<_> = [-0.04, 0.30]
            .into_iter()
            .flat_map(|x| {
                [0.0_f32, 0.01, 1.57, 3.14, 4.71].map(|angle| {
                    parry3d::math::Vector::new(
                        x,
                        f64::from(0.05 * angle.cos()),
                        f64::from(0.05 * angle.sin()),
                    )
                })
            })
            .collect();
        let a = Hull {
            bone: Entity::from_raw_u32(1).unwrap(),
            region: Region::Upper(ArmSide::Left),
            shape: ConvexPolyhedron::from_convex_hull(&points).unwrap(),
            skin: Vec::new(),
        };
        let b = Hull {
            bone: Entity::from_raw_u32(2).unwrap(),
            region: Region::Forearm(ArmSide::Left),
            ..a.clone()
        };
        let joint = JointAllowance {
            bones: [a.bone, b.bone],
            pivot_bone: b.bone,
            pivot: Vec3::ZERO,
            radius: 0.075,
        };
        let geometry = CollisionGeometry::new(vec![a, b], vec![joint]);
        for i in 0..25 {
            let motion = |bone| {
                Some(BoneMotion {
                    rotation: Quat::from_rotation_z(if bone == geometry.hulls[0].bone {
                        i as f32 * 0.0001
                    } else {
                        std::f32::consts::FRAC_PI_2
                    }),
                    translation: Vec3::new(0.2, 1.3, -0.1),
                })
            };
            let plain = geometry.margins(motion, 0.0).unwrap();
            let swept = geometry
                .swept_margins(&motion, Some(&motion), |_| 0.0, |_| 0.0)
                .unwrap();
            assert!(
                (plain[0] - swept[0]).abs() <= 64.0 * f32::EPSILON * 0.6,
                "{i}: plain={plain:?}, swept={swept:?}"
            );
        }
    }

    #[test]
    fn adjacent_allowance_does_not_hide_penetration_away_from_joint() {
        let a = hull(
            0,
            Region::Upper(ArmSide::Left),
            Vec3::new(-0.05, -0.05, -0.05),
            Vec3::new(0.35, 0.05, 0.05),
        );
        let b = hull(
            1,
            Region::Forearm(ArmSide::Left),
            Vec3::new(-0.05, -0.05, -0.05),
            Vec3::new(0.05, 0.35, 0.05),
        );
        let joint = JointAllowance {
            bones: [a.bone, b.bone],
            pivot_bone: b.bone,
            pivot: Vec3::ZERO,
            radius: 0.09,
        };
        let geometry = CollisionGeometry::new(vec![a, b], vec![joint]);
        let clearance = geometry.clearance(|_| Some(IDENTITY)).unwrap();
        assert!(
            clearance > 0.0,
            "joint clearance={clearance}, radius={}",
            intersection_radius(
                [
                    (
                        &HullFaces::from_points(geometry.hulls[0].shape.points()).unwrap(),
                        IDENTITY,
                        0.0
                    ),
                    (
                        &HullFaces::from_points(geometry.hulls[1].shape.points()).unwrap(),
                        IDENTITY,
                        0.0
                    )
                ],
                Vec3::ZERO,
                0.0,
            )
        );
        let folded = BoneMotion {
            rotation: Quat::from_rotation_z(-std::f32::consts::FRAC_PI_2),
            ..IDENTITY
        };
        assert!(
            geometry
                .clearance(|e| Some(if e == joint.pivot_bone {
                    folded
                } else {
                    IDENTITY
                }))
                .unwrap()
                < 0.0
        );
    }
    #[test]
    fn epic_counterexample_and_contralateral_crossing_have_volume_overlap() {
        let shoulder = Vec3::new(0.16, 1.3, 0.0);
        let elbow = shoulder + Vec3::new(-0.3, -0.8, 0.5).normalize() * 0.30;
        let wrist = elbow + Vec3::new(-0.6, 0.0, -0.8) * 0.26;
        assert!(elbow.z > 0.15 && wrist.z < -0.05);
        let direction = (wrist - elbow).normalize();
        let cross = direction.cross(Vec3::Y).normalize() * 0.025;
        let up = direction.cross(cross).normalize() * 0.025;
        let points: Vec<_> = [elbow, wrist]
            .into_iter()
            .flat_map(|p| {
                [-1.0, 1.0].into_iter().flat_map(move |a| {
                    [-1.0, 1.0].map(|b| {
                        parry3d::math::Vector::from_array(
                            (p + cross * a + up * b).to_array().map(f64::from),
                        )
                    })
                })
            })
            .collect();
        let arm = Hull {
            bone: Entity::from_raw_u32(1).unwrap(),
            region: Region::Forearm(ArmSide::Left),
            shape: ConvexPolyhedron::from_convex_hull(&points).unwrap(),
            skin: vec![],
        };
        let body = hull(
            0,
            Region::Torso,
            Vec3::new(-0.18, 0.9, -0.10),
            Vec3::new(0.18, 1.5, 0.10),
        );
        let geometry = CollisionGeometry::new(vec![body, arm.clone()], vec![]);
        assert!(geometry.clearance(|_| Some(IDENTITY)).unwrap() < 0.0);
        let other = hull(
            2,
            Region::Forearm(ArmSide::Right),
            Vec3::new(-0.2, 1.05, 0.01),
            Vec3::new(0.2, 1.10, 0.06),
        );
        let geometry = CollisionGeometry::new(vec![arm, other], vec![]);
        assert!(geometry.clearance(|_| Some(IDENTITY)).unwrap() < 0.0);
    }

    #[test]
    fn equal_blend_weights_preserve_the_posed_hull_after_interior_reduction() {
        use parry3d::shape::SupportMap;
        let a = Entity::from_raw_u32(1).unwrap();
        let b = Entity::from_raw_u32(2).unwrap();
        let mut vertices = Vec::new();
        for x in [-1.0, 0.0, 1.0] {
            for y in [-1.0, 0.0, 1.0] {
                for z in [-1.0, 0.0, 1.0] {
                    let source = Vec3::new(x, y, z);
                    // Different inverse binds still give one affine map for
                    // this shared weight group, including non-uniform scale.
                    let pa = source * Vec3::new(2.0, 1.0, 0.5) + Vec3::X;
                    let pb = Quat::from_rotation_y(0.4) * source + Vec3::Y;
                    vertices.push(BindVertex {
                        source,
                        blend: ([0, 1, 0, 0], [0.4_f32.to_bits(), 0.6_f32.to_bits(), 0, 0]),
                        skin: SkinVertex {
                            point: pa * 0.4 + pb * 0.6,
                            influences: vec![(a, pa, 0.4), (b, pb, 0.6)],
                        },
                    });
                }
            }
        }
        let reduced = reduce_skin_vertices(vertices.clone());
        assert_eq!(reduced.len(), 8);
        for angle in [0.0, 0.8, 2.1] {
            let motion = |bone| {
                Some(BoneMotion {
                    rotation: Quat::from_rotation_z(if bone == a { angle } else { -angle }),
                    translation: if bone == a { Vec3::X } else { Vec3::Z },
                })
            };
            let full = PosedHull::new(
                vertices
                    .iter()
                    .map(|v| v.skin.posed(&motion).unwrap())
                    .collect(),
                0.0,
            )
            .unwrap();
            let small = PosedHull::new(
                reduced.iter().map(|v| v.posed(&motion).unwrap()).collect(),
                0.0,
            )
            .unwrap();
            for direction in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::ONE, -Vec3::ONE] {
                let direction =
                    parry3d::math::Vector::from_array(direction.to_array().map(f64::from));
                let support = |h: &PosedHull| {
                    (h.local_support_point(direction)
                        + parry3d::math::Vector::from_array(h.centre.to_array().map(f64::from)))
                    .dot(direction)
                };
                assert!((support(&full) - support(&small)).abs() < 2.0e-6);
            }
        }
    }

    #[test]
    fn rest_interior_blended_vertex_remains_part_of_the_candidate_envelope() {
        use parry3d::shape::SupportMap;
        let a = Entity::from_raw_u32(1).unwrap();
        let b = Entity::from_raw_u32(2).unwrap();
        let cube = hull(
            1,
            Region::Hand(ArmSide::Left),
            Vec3::splat(-1.0),
            Vec3::splat(1.0),
        );
        let mut vertices: Vec<_> = cube
            .shape
            .points()
            .iter()
            .map(|p| {
                let point = Vec3::from_array(p.to_array().map(|v| v as f32));
                SkinVertex {
                    point,
                    influences: vec![(a, point, 1.0)],
                }
            })
            .collect();
        vertices.push(SkinVertex {
            point: Vec3::ZERO,
            influences: vec![(a, Vec3::ZERO, 0.5), (b, Vec3::ZERO, 0.5)],
        });
        let vertices = reduce_skin_vertices(
            vertices
                .into_iter()
                .map(|skin| BindVertex {
                    source: skin.point,
                    blend: if skin.influences.len() == 1 {
                        ([0; 4], [1.0_f32.to_bits(), 0, 0, 0])
                    } else {
                        ([0, 1, 0, 0], [0.5_f32.to_bits(), 0.5_f32.to_bits(), 0, 0])
                    },
                    skin,
                })
                .collect(),
        );
        let points = vertices
            .iter()
            .map(|v| {
                v.posed(&|bone| {
                    Some(BoneMotion {
                        translation: if bone == b { Vec3::X * 6.0 } else { Vec3::ZERO },
                        ..IDENTITY
                    })
                })
                .unwrap()
            })
            .collect();
        let cloud = PosedHull::new(points, 0.0).unwrap();
        let support = cloud.local_support_point(parry3d::math::Vector::X);
        assert_eq!(support.x + f64::from(cloud.centre.x), 3.0);
    }

    #[test]
    fn bounded_support_matches_the_full_scan_including_flat_face_ties() {
        use parry3d::{math::Vector, shape::SupportMap};
        let points: Vec<_> = (-5..=5)
            .flat_map(|x| {
                (-4..=4).flat_map(move |y| {
                    (-3..=3).map(move |z| Vec3::new(x as f32, y as f32, z as f32))
                })
            })
            .collect();
        let mut points = points;
        points.rotate_left(119);
        let hull = PosedHull::new(points, 0.01).unwrap();
        for x in -3..=3 {
            for y in -3..=3 {
                for z in -3..=3 {
                    let direction = Vector::new(f64::from(x), f64::from(y), f64::from(z));
                    let expected = hull.points.iter().copied().fold(hull.first, |best, p| {
                        if p.dot(direction) > best.dot(direction) {
                            p
                        } else {
                            best
                        }
                    }) + direction.try_normalize().unwrap_or(Vector::ZERO)
                        * f64::from(hull.padding);
                    assert_eq!(hull.local_support_point(direction), expected);
                }
            }
        }
        let bounds = hull.joint_bounds();
        for (normal, plane) in &bounds.faces {
            let support = hull
                .points
                .iter()
                .map(|p| normal.dot(bevy::math::DVec3::from_array(p.to_array())))
                .fold(f64::NEG_INFINITY, f64::max);
            assert_eq!(*plane, *normal * support);
        }
        let extrema = |points: Vec<bevy::math::DVec3>| {
            points.into_iter().fold(
                (
                    bevy::math::DVec3::splat(f64::INFINITY),
                    bevy::math::DVec3::splat(f64::NEG_INFINITY),
                ),
                |(min, max), p| (min.min(p), max.max(p)),
            )
        };
        assert_eq!(
            extrema(bounds.points.clone()),
            extrema(
                hull.points
                    .iter()
                    .map(|p| bevy::math::DVec3::from_array(p.to_array()))
                    .collect()
            )
        );
    }
}
