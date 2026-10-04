//! The collision-relevant body hierarchy travels on the same clock as the arms.
//! Local rotations preserve the authored links; collision FK removes the current
//! thorax transform, exactly as the arm coordinates do.

use std::{collections::HashMap, sync::Arc};

use bevy::{math::Affine3A, prelude::*};

use crate::collision::{BoneMotion, CollisionError};

#[derive(Clone)]
struct Node {
    entity: Entity,
    parent: Option<usize>,
    rest_inverse: Affine3A,
}

pub(crate) struct BodyRig {
    nodes: Vec<Node>,
    indices: HashMap<Entity, usize>,
    chest: usize,
    chest_rest: Affine3A,
    root_rest: Affine3A,
}

#[derive(Clone)]
pub(crate) struct BodyPose {
    pub rig: Arc<BodyRig>,
    locals: Vec<Transform>,
}

#[derive(Clone)]
pub(crate) struct BodyCurve {
    pub from: BodyPose,
    pub to: BodyPose,
}

impl BodyRig {
    pub fn bind(
        root: Entity,
        chest: Entity,
        bones: impl Iterator<Item = Entity>,
        read: impl Fn(Entity) -> Option<(Transform, Affine3A, Option<Entity>)>,
    ) -> Result<Arc<Self>, CollisionError> {
        let (_, root_rest, _) = read(root).ok_or(CollisionError::MissingBone)?;
        let mut nodes = Vec::new();
        let mut indices = HashMap::new();
        for bone in bones.chain(std::iter::once(chest)) {
            let mut path = Vec::new();
            let mut current = bone;
            while current != root && !indices.contains_key(&current) {
                let (_, rest, parent) = read(current).ok_or(CollisionError::MissingBone)?;
                path.push((current, rest));
                current = parent.ok_or(CollisionError::MissingBone)?;
            }
            let mut parent = indices.get(&current).copied();
            for (entity, rest) in path.into_iter().rev() {
                let index = nodes.len();
                nodes.push(Node {
                    entity,
                    parent,
                    rest_inverse: rest.inverse(),
                });
                indices.insert(entity, index);
                parent = Some(index);
            }
        }
        Ok(Arc::new(Self {
            chest: *indices.get(&chest).ok_or(CollisionError::MissingBone)?,
            chest_rest: read(chest).ok_or(CollisionError::MissingBone)?.1,
            root_rest,
            nodes,
            indices,
        }))
    }

    pub fn capture(
        self: &Arc<Self>,
        read: impl Fn(Entity) -> Option<Transform>,
    ) -> Result<BodyPose, CollisionError> {
        let locals = self
            .nodes
            .iter()
            .map(|n| read(n.entity).ok_or(CollisionError::MissingBone))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(BodyPose {
            rig: Arc::clone(self),
            locals,
        })
    }
}

fn rotation_vector(from: Quat, to: Quat) -> Vec3 {
    if from == to {
        return Vec3::ZERO;
    }
    let mut relative = (to * from.inverse()).normalize();
    if relative.w < 0.0 {
        relative = -relative;
    }
    relative.to_scaled_axis()
}

impl BodyPose {
    pub fn locals(&self) -> impl Iterator<Item = (Entity, Transform)> + '_ {
        self.rig
            .nodes
            .iter()
            .zip(&self.locals)
            .map(|(n, t)| (n.entity, *t))
    }

    pub fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.rig, &other.rig) && self.locals == other.locals
    }

    pub fn interpolate(&self, other: &Self, amount: f32) -> Self {
        if amount <= 0.0 {
            return self.clone();
        }
        if amount >= 1.0 {
            return other.clone();
        }
        let locals = self
            .locals
            .iter()
            .zip(&other.locals)
            .map(|(a, b)| Transform {
                translation: a.translation + (b.translation - a.translation) * amount,
                rotation: if a.rotation == b.rotation {
                    a.rotation
                } else {
                    (Quat::from_scaled_axis(rotation_vector(a.rotation, b.rotation) * amount)
                        * a.rotation)
                        .normalize()
                },
                // A model-scale change is handled by the avatar generation/bind.
                scale: a.scale,
            })
            .collect();
        Self {
            rig: Arc::clone(&self.rig),
            locals,
        }
    }

    pub fn motions(&self) -> Result<HashMap<Entity, BoneMotion>, CollisionError> {
        let mut globals = Vec::<Affine3A>::with_capacity(self.locals.len());
        for (node, local) in self.rig.nodes.iter().zip(&self.locals) {
            let parent = node
                .parent
                .and_then(|p| globals.get(p))
                .copied()
                .unwrap_or(self.rig.root_rest);
            globals.push(parent * local.compute_affine());
        }
        let chest = self.rig.chest_rest
            * globals
                .get(self.rig.chest)
                .ok_or(CollisionError::MissingBone)?
                .inverse();
        self.rig
            .nodes
            .iter()
            .zip(globals)
            .map(|(n, g)| {
                let (_, rotation, translation) =
                    (chest * g * n.rest_inverse).to_scale_rotation_translation();
                Ok((
                    n.entity,
                    BoneMotion {
                        rotation,
                        translation,
                    },
                ))
            })
            .collect()
    }
}

impl BodyCurve {
    pub fn at(&self, t: f32) -> BodyPose {
        self.from.interpolate(&self.to, t)
    }

    /// Bound x'' for a rest-space skin point, along linear body progress.
    /// Walk the bone to the common ancestor, then down into the moving chest
    /// frame. Each rotation is a fixed-axis slerp; translated links are affine.
    /// Product-rule bounds include both inverse rotations and translations.
    pub fn acceleration(&self, bone: Entity, point: Vec3) -> f32 {
        let rig = &self.from.rig;
        let Some(&index) = rig.indices.get(&bone) else {
            return 0.0;
        };
        let ancestors = |mut index: usize| {
            let mut result = Vec::new();
            loop {
                result.push(index);
                let Some(parent) = rig.nodes.get(index).and_then(|n| n.parent) else {
                    break;
                };
                index = parent;
            }
            result
        };
        let up = ancestors(index);
        let down = ancestors(rig.chest);
        let common = up.iter().copied().find(|i| down.contains(i));
        let Some(node) = rig.nodes.get(index) else {
            return f32::INFINITY;
        };
        let mut radius = node.rest_inverse.transform_point3(point).length();
        let mut velocity = 0.0;
        let mut acceleration = 0.0;
        let mut step = |index: usize, inverse: bool| {
            let Some((a, b)) = self.from.locals.get(index).zip(self.to.locals.get(index)) else {
                return;
            };
            let omega = rotation_vector(a.rotation, b.rotation).length();
            let translation = a.translation.length().max(b.translation.length());
            let speed = a.translation.distance(b.translation);
            let scale = if inverse {
                a.scale.recip().abs().max_element()
            } else {
                a.scale.abs().max_element()
            };
            if inverse {
                acceleration = scale
                    * (acceleration
                        + 2.0 * omega * (velocity + speed)
                        + omega * omega * (radius + translation));
                velocity = scale * (velocity + speed + omega * (radius + translation));
                radius = scale * (radius + translation);
            } else {
                acceleration =
                    scale * (acceleration + 2.0 * omega * velocity + omega * omega * radius);
                velocity = scale * (velocity + omega * radius) + speed;
                radius = scale * radius + translation;
            }
        };
        for &i in up.iter().take_while(|i| Some(**i) != common) {
            step(i, false);
        }
        for &i in down
            .iter()
            .take_while(|i| Some(**i) != common)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
        {
            step(i, true);
        }
        let (scale, _, _) = rig.chest_rest.to_scale_rotation_translation();
        acceleration * scale.abs().max_element()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]
    use super::*;

    #[test]
    fn moving_thorax_preserves_links_and_its_skin_chord_bound() {
        let root = Entity::from_raw_u32(1).unwrap();
        let hips = Entity::from_raw_u32(2).unwrap();
        let chest = Entity::from_raw_u32(3).unwrap();
        let neck = Entity::from_raw_u32(4).unwrap();
        let local = [
            Transform::IDENTITY,
            Transform::from_xyz(0.0, 0.8, 0.0),
            Transform::from_xyz(0.0, 0.4, 0.0),
            Transform::from_xyz(0.0, 0.2, 0.0),
        ];
        let global = [
            Affine3A::IDENTITY,
            local[1].compute_affine(),
            local[1].compute_affine() * local[2].compute_affine(),
            local[1].compute_affine() * local[2].compute_affine() * local[3].compute_affine(),
        ];
        let entities = [root, hips, chest, neck];
        let rig = BodyRig::bind(root, chest, [hips, chest, neck].into_iter(), |e| {
            let i = entities.iter().position(|b| *b == e)?;
            Some((local[i], global[i], i.checked_sub(1).map(|i| entities[i])))
        })
        .unwrap();
        let from = rig
            .capture(|e| entities.iter().position(|b| *b == e).map(|i| local[i]))
            .unwrap();
        let to = rig
            .capture(|e| {
                let mut t = local[entities.iter().position(|b| *b == e)?];
                t.rotation = if e == chest {
                    Quat::from_euler(EulerRot::YXZ, 0.6, -0.2, 0.1)
                } else if e == neck {
                    Quat::from_rotation_y(-0.4)
                } else {
                    Quat::IDENTITY
                };
                Some(t)
            })
            .unwrap();
        let curve = BodyCurve { from, to };
        for (bone, point) in [
            (hips, Vec3::new(0.2, 1.0, 0.05)),
            (neck, Vec3::new(-0.1, 1.43, 0.03)),
        ] {
            let a = curve.from.motions().unwrap()[&bone].point(point);
            let b = curve.to.motions().unwrap()[&bone].point(point);
            let bound = curve.acceleration(bone, point) / 8.0;
            for i in 0..=32 {
                let t = i as f32 / 32.0;
                let sample = curve.at(t);
                let position = sample.motions().unwrap()[&bone].point(point);
                assert!(position.distance(a.lerp(b, t)) <= bound + 64.0 * f32::EPSILON);
                for ((_, actual), (_, original)) in sample.locals().zip(curve.from.locals()) {
                    assert_eq!(actual.translation, original.translation);
                    assert_eq!(actual.scale, original.scale);
                }
            }
        }
        assert!(curve.at(1.0).same(&curve.to));
        let arm = crate::upper_limb::tests::chain(crate::arm::ArmSide::Left);
        let pose = [
            Some(crate::upper_limb::tests::state(0.0, 0.7, -0.2, 0.8)),
            None,
        ];
        let geometry = crate::collision::CollisionGeometry::default();
        let body = curve.to.motions().unwrap();
        let problem = crate::upper_limb_solver::Problem {
            chains: [Some(&arm), None],
            goals: [None, None],
            geometry: &geometry,
            body: &body,
            body_curve: Some(&curve),
            tolerance: 64.0 * f32::EPSILON,
        };
        let mut path = crate::upper_limb_path::JointPath::default();
        path.current = pose;
        path.body = Some(curve.from.clone());
        path.plan(&problem, pose).unwrap();
        while path.busy() {
            path.advance(&problem, pose, 1.0 / 60.0).unwrap();
        }
        assert!(path.body.unwrap().same(&curve.to));
    }
}
