//! Rest-skeleton capsule proxies, independent of clothing and render morphs.
//! Arm radii and the torso default follow Wicked Engine's humanoid capsules;
//! see THIRD_PARTY_NOTICES.md for the source and deliberate differences.

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

#[derive(Component, Clone, Debug, Default)]
pub(crate) struct CollisionGeometry {
    pub capsules: Vec<CapsuleCollider>,
    pairs: Vec<[usize; 2]>,
    bones: Vec<Entity>,
}

#[derive(Component)]
/// Collision shapes consumed by the upper-limb system.
pub struct AvatarCollision(pub(crate) Result<std::sync::Arc<CollisionGeometry>, CollisionError>);

/// Bind once from immutable rest transforms. No render assets, skin weights,
/// morph polling or shoulder-skin exceptions are needed.
pub(crate) fn bind_collision_geometry(
    mut commands: Commands,
    roots: Query<
        EntityRef,
        (
            With<crate::binding::AvatarBinding>,
            Without<AvatarCollision>,
        ),
    >,
    bones: Query<&RestGlobalTransform>,
) {
    for root in &roots {
        let Some(binding) = root.get::<crate::binding::AvatarBinding>() else {
            continue;
        };
        let build = || -> Result<CollisionGeometry, CollisionError> {
            if binding.left_arm.is_none() && binding.right_arm.is_none() {
                return Ok(CollisionGeometry::default());
            }
            let rest = |bone| {
                bones
                    .get(bone)
                    .map(|r| r.0)
                    .map_err(|_| CollisionError::MissingBone)
            };
            let root_rest = root
                .get::<RestGlobalTransform>()
                .map(|r| r.0)
                .or_else(|| root.get::<GlobalTransform>().copied())
                .ok_or(CollisionError::MissingBone)?;
            // Object scale, NOT estimated stature or shoulder breadth.
            // Rest endpoints and radii must use the same coordinate space.
            let radius = 0.1 * root_rest.affine().matrix3.x_axis.length();
            let links: Vec<_> = [
                root.get::<HipsBoneEntity>().map(|b| b.0),
                binding.spine,
                binding.chest,
                binding.upper_chest,
                Some(binding.neck.unwrap_or(binding.head)),
            ]
            .into_iter()
            .flatten()
            .collect();
            let mut capsules = Vec::new();
            for pair in links.windows(2) {
                let Some((&bone, &next)) = pair.first().zip(pair.last()) else {
                    continue;
                };
                capsules.push(link_capsule(
                    bone,
                    Region::Torso,
                    rest(bone)?.translation(),
                    rest(next)?.translation(),
                    radius,
                ));
            }
            if capsules.is_empty() {
                return Err(CollisionError::MissingBone);
            }
            for chain in [binding.left_arm, binding.right_arm].into_iter().flatten() {
                capsules.extend(arm_capsules(chain));
                capsules.extend(palm_capsules(chain));
            }
            if capsules.iter().any(|c| {
                !c.radius.is_finite()
                    || c.radius <= 0.0
                    || c.endpoints.iter().any(|p| !p.is_finite())
            }) {
                return Err(CollisionError::InvalidGeometry);
            }
            Ok(CollisionGeometry::new(capsules, &[]))
        };
        commands
            .entity(root.id())
            .insert(AvatarCollision(build().map(std::sync::Arc::new)));
        if binding.left_arm.is_some() || binding.right_arm.is_some() {
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
    /// Non-finite or degenerate collision geometry or kinematics.
    InvalidGeometry,
    /// A collision proxy references an unavailable rest bone.
    MissingBone,
}

/// Use unit fatness with Wicked Engine's upper radius = length * .20,
/// forearm radius = length * .15. Keep the elbow centre inside the forearm
/// cap, so exempting the upper arm never exempts the elbow from the torso.
fn arm_capsules(chain: crate::arm::ArmChainBinding) -> [CapsuleCollider; 2] {
    let upper = chain.rest.upper_arm.position;
    let elbow = chain.rest.elbow.position;
    let wrist = chain.rest.wrist.position;
    let radius = elbow.distance(wrist) * 0.15;
    [
        link_capsule(
            chain.upper_arm,
            Region::Upper(chain.side),
            upper,
            elbow,
            upper.distance(elbow) * 0.20,
        ),
        CapsuleCollider {
            bone: chain.lower_arm,
            region: Region::Forearm(chain.side),
            endpoints: [elbow, wrist],
            radius,
        },
    ]
}

/// The rigid palm spans the wrist and four MCP centres. Their measured spacing
/// supplies its thickness, so a cuff or ornament bound to the hand bone cannot
/// block an otherwise clear arm path. Without MCPs only the forearm wrist cap
/// is known; no mesh-derived hand or finger dimensions are invented.
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

impl CollisionGeometry {
    /// SIK-inspired elbow swivel about shoulder--wrist, applied to the task
    /// target only. Preserve the wrist and both link lengths; the existing
    /// constrained FK solve and swept path still decide admissibility.
    /// A straight arm or a penetrating wrist has no solution on this circle
    /// and keeps its original task for the joint-space solver to resolve.
    pub fn elbow_target(
        &self,
        shoulder: Vec3,
        elbow: Vec3,
        wrist: Vec3,
        side: crate::arm::ArmSide,
        body: &HashMap<Entity, BoneMotion>,
    ) -> Vec3 {
        let Some(axis) = (wrist - shoulder).try_normalize() else {
            return elbow;
        };
        let radius = self
            .capsules
            .iter()
            .filter(|c| c.region == Region::Forearm(side))
            .map(|c| c.radius)
            .fold(0.0_f32, f32::max);
        let torso: Vec<_> = self
            .capsules
            .iter()
            .filter(|c| c.region == Region::Torso)
            .filter_map(|c| {
                body.get(&c.bone).map(|m| {
                    let [a, b] = c
                        .endpoints
                        .map(|p| Vector::from_array(m.point(p).to_array().map(f64::from)));
                    (Segment::new(a, b), f64::from(c.radius + radius))
                })
            })
            .collect();
        let clear = |elbow: Vec3| {
            let [a, b] = [elbow, wrist].map(|p| Vector::from_array(p.to_array().map(f64::from)));
            let forearm = Segment::new(a, b);
            torso
                .iter()
                .all(|(s, r)| segment_distance(&forearm, s) >= *r)
        };
        if clear(elbow) {
            return elbow;
        }
        let at = |angle| shoulder + Quat::from_axis_angle(axis, angle) * (elbow - shoulder);
        // Search both directions for the smallest clear swivel. Bounded work
        // runs once per new observation, not per SQP candidate/render tick.
        let step = std::f32::consts::PI / 16.0;
        let mut best = None::<f32>;
        let signs = match side {
            crate::arm::ArmSide::Left => [-1.0, 1.0],
            crate::arm::ArmSide::Right => [1.0, -1.0],
        };
        for sign in signs {
            for i in 1..=16 {
                let angle = sign * i as f32 * step;
                if !clear(at(angle)) {
                    continue;
                }
                let mut blocked = sign * (i - 1) as f32 * step;
                let mut free = angle;
                for _ in 0..12 {
                    let mid = (blocked + free) * 0.5;
                    if clear(at(mid)) {
                        free = mid;
                    } else {
                        blocked = mid;
                    }
                }
                if best.is_none_or(|b| free.abs() < b.abs()) {
                    best = Some(free);
                }
                break;
            }
        }
        best.map(at).unwrap_or(elbow)
    }

    pub fn new(capsules: Vec<CapsuleCollider>, connected: &[[Entity; 2]]) -> Self {
        let mut bones: Vec<_> = capsules.iter().map(|c| c.bone).collect();
        bones.sort_unstable();
        bones.dedup();
        let mut pairs = Vec::new();
        for (i, a) in capsules.iter().enumerate() {
            for (j, b) in capsules.iter().enumerate().skip(i + 1) {
                // The simplified upper-arm/torso connection may overlap to
                // permit adduction. Forearm caps still protect both elbows;
                // hands and the opposite arm remain collision subjects.
                let adjacent = match (a.region, b.region) {
                    (Region::Upper(x), Region::Forearm(y))
                    | (Region::Forearm(x), Region::Upper(y))
                    | (Region::Forearm(x), Region::Hand(y))
                    | (Region::Hand(x), Region::Forearm(y))
                    | (Region::Hand(x), Region::Hand(y)) => x == y,
                    (Region::Torso, Region::Torso)
                    | (Region::Upper(_), Region::Torso)
                    | (Region::Torso, Region::Upper(_)) => true,
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
            .ok_or(CollisionError::InvalidGeometry)
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
                .ok_or(CollisionError::InvalidGeometry)?;
            let distance = segment_distance(sa, sb) as f32 - ca.radius - cb.radius;
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
            let swept = |i: usize, radius: f32| -> Result<SweptCapsule, CollisionError> {
                let (from, to) = from
                    .get(i)
                    .zip(to.get(i))
                    .ok_or(CollisionError::InvalidGeometry)?;
                Ok(SweptCapsule {
                    points: [from.a, from.b, to.a, to.b],
                    radius: f64::from(radius + padding(i)),
                })
            };
            let a = swept(a, ca.radius)?;
            let b = swept(b, cb.radius)?;
            if !a.separated(&b) {
                return Ok(false);
            }
        }
        Ok(true)
    }
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
    fn capsule_distance_and_spherical_degeneracy() {
        let a = Segment::new(Vector::ZERO, Vector::X);
        let b = Segment::new(Vector::Y, Vector::Y);
        assert!((segment_distance(&a, &b) - 1.0).abs() < 1e-12);
        let g = CollisionGeometry::new(
            vec![
                capsule(1, Region::Torso, 0.0),
                capsule(2, Region::Forearm(ArmSide::Left), 0.15),
            ],
            &[],
        );
        assert!(!g.pose_is_clear(&identity, 0.0).unwrap());
        assert!((g.solver_margins(identity, 0.0, None).unwrap().0[0] + 0.05).abs() < 1e-6);
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
    fn adduction_exempts_upper_torso_but_checks_elbow_hand_and_opposite_arm() {
        for side in [ArmSide::Left, ArmSide::Right] {
            let other = if side == ArmSide::Left {
                ArmSide::Right
            } else {
                ArmSide::Left
            };
            let g = CollisionGeometry::new(
                vec![
                    capsule(1, Region::Torso, 0.0),
                    capsule(2, Region::Upper(side), 0.0),
                    capsule(3, Region::Forearm(side), 0.5),
                    capsule(4, Region::Hand(side), 0.7),
                    capsule(5, Region::Upper(other), -0.5),
                ],
                &[],
            );
            assert!(!g.pairs.contains(&[0, 1]));
            for pair in [[0, 2], [0, 3], [1, 4], [2, 4]] {
                assert!(g.pairs.contains(&pair));
            }
            assert!(g.pose_is_clear(&identity, 0.0).unwrap());
            assert!(g.sweep_is_clear(&identity, &identity, |_| 0.0).unwrap());
            for moving in [entity(3), entity(4)] {
                let contact = |bone| {
                    Some(BoneMotion {
                        translation: if bone == moving {
                            Vec3::NEG_X * 0.65
                        } else {
                            Vec3::ZERO
                        },
                        ..identity(bone).unwrap()
                    })
                };
                assert!(!g.pose_is_clear(&contact, 0.0).unwrap());
                assert!(!g.sweep_is_clear(&identity, &contact, |_| 0.0).unwrap());
            }
        }
    }

    #[test]
    fn elbow_cap_blocks_contact_even_when_the_rest_of_the_forearm_is_outside() {
        let chain = crate::upper_limb::tests::chain(ArmSide::Left);
        let [upper, forearm] = arm_capsules(chain);
        assert!((upper.radius - chain.rest.upper_arm_length * 0.2).abs() < 1e-6);
        assert!((forearm.radius - chain.rest.forearm_length * 0.15).abs() < 1e-6);
        assert_eq!(
            forearm.endpoints,
            [chain.rest.elbow.position, chain.rest.wrist.position]
        );
        let direction = (chain.rest.wrist.position - chain.rest.elbow.position).normalize();
        let torso = CapsuleCollider {
            endpoints: [chain.rest.elbow.position - direction * 0.04; 2],
            radius: 0.02,
            ..capsule(100, Region::Torso, 0.0)
        };
        let g = CollisionGeometry::new(vec![torso, upper, forearm], &[]);
        assert!(!g.pose_is_clear(&identity, 0.0).unwrap());
        assert!(!g.sweep_is_clear(&identity, &identity, |_| 0.0).unwrap());
    }

    #[test]
    fn elbow_swivel_preserves_wrist_and_lengths_without_a_contact_gap() {
        for scale in [0.5, 1.0, 3.0] {
            for sign in [-1.0, 1.0] {
                let shoulder = Vec3::new(sign * 0.20, 0.3, 0.0) * scale;
                let elbow = Vec3::new(sign * 0.05, 0.0, 0.0) * scale;
                let wrist = Vec3::new(sign * 0.20, 0.0, 0.3) * scale;
                let g = CollisionGeometry::new(
                    vec![
                        CapsuleCollider {
                            radius: 0.10 * scale,
                            endpoints: [
                                Vec3::new(0.0, -0.3, 0.0) * scale,
                                Vec3::new(0.0, 0.3, 0.0) * scale,
                            ],
                            ..capsule(1, Region::Torso, 0.0)
                        },
                        CapsuleCollider {
                            radius: 0.04 * scale,
                            endpoints: [elbow, wrist],
                            ..capsule(2, Region::Forearm(ArmSide::Left), 0.0)
                        },
                    ],
                    &[],
                );
                let body = HashMap::from([(entity(1), identity(entity(1)).unwrap())]);
                let corrected = g.elbow_target(shoulder, elbow, wrist, ArmSide::Left, &body);
                assert!(corrected.distance(elbow) > 0.05 * scale);
                assert!(
                    (corrected.distance(shoulder) - elbow.distance(shoulder)).abs() < 1e-6 * scale
                );
                assert!((corrected.distance(wrist) - elbow.distance(wrist)).abs() < 1e-6 * scale);
                let mut corrected_g = g.clone();
                corrected_g.capsules[1].endpoints = [corrected, wrist];
                let separation = corrected_g.solver_margins(identity, 0.0, None).unwrap().0[0];
                assert!(
                    separation >= -1e-7 * scale && separation < 0.0001 * scale,
                    "{separation}"
                );
                assert_eq!(
                    g.elbow_target(shoulder, corrected, wrist, ArmSide::Left, &body),
                    corrected
                );
                // No swivel can resolve a penetrating wrist; do not fabricate
                // a translated wrist or a stretched arm in that case.
                assert_eq!(
                    g.elbow_target(shoulder, elbow, Vec3::ZERO, ArmSide::Left, &body),
                    elbow
                );
                let straight = shoulder.lerp(wrist, 0.5);
                let result = g.elbow_target(shoulder, straight, wrist, ArmSide::Left, &body);
                assert!(result.is_finite() && result.distance(straight) < 1e-6 * scale);
            }
        }
    }

    #[test]
    fn binding_needs_no_mesh_and_is_invariant_to_bone_axes_and_object_scale() {
        use crate::{binding::AvatarBinding, lifecycle::AvatarGeneration};
        let bind = |scale: f32, rotation: Quat| {
            let mut app = App::new();
            app.add_plugins(MinimalPlugins)
                .add_systems(Update, bind_collision_geometry);
            let root = app
                .world_mut()
                .spawn(GlobalTransform::from_scale(Vec3::splat(scale)))
                .id();
            let mut spawn = |position| {
                app.world_mut()
                    .spawn(RestGlobalTransform(GlobalTransform::from(
                        Transform::from_translation(position * scale).with_rotation(rotation),
                    )))
                    .id()
            };
            let hips = spawn(Vec3::Y);
            let spine = spawn(Vec3::Y * 1.15);
            let neck = spawn(Vec3::Y * 1.5);
            let mut binding = AvatarBinding::head_only(root, neck, AvatarGeneration(1));
            binding.spine = Some(spine);
            binding.neck = Some(neck);
            let mut chain = crate::upper_limb::tests::chain(ArmSide::Left);
            for pose in [
                &mut chain.rest.upper_arm,
                &mut chain.rest.elbow,
                &mut chain.rest.wrist,
            ] {
                pose.position *= scale;
                pose.global_rotation = rotation;
            }
            chain.finger_rest = Default::default();
            binding.left_arm = Some(chain);
            app.world_mut()
                .entity_mut(root)
                .insert((binding, HipsBoneEntity(hips)));
            app.update();
            let geometry = app
                .world()
                .get::<AvatarCollision>(root)
                .unwrap()
                .0
                .as_ref()
                .unwrap()
                .clone();
            // Later transforms do not refit immutable rest dimensions.
            app.world_mut()
                .entity_mut(root)
                .insert(GlobalTransform::from_scale(Vec3::splat(8.0)));
            app.update();
            assert!(std::sync::Arc::ptr_eq(
                &geometry,
                app.world()
                    .get::<AvatarCollision>(root)
                    .unwrap()
                    .0
                    .as_ref()
                    .unwrap()
            ));
            geometry
        };
        let a = bind(1.0, Quat::IDENTITY);
        assert_eq!(a.capsules.len(), 4);
        assert!(
            a.capsules
                .iter()
                .filter(|c| c.region == Region::Torso)
                .all(|c| (c.radius - 0.1).abs() < 1e-6)
        );
        let b = bind(3.0, Quat::from_euler(EulerRot::XYZ, 0.4, -0.3, 0.7));
        for (a, b) in a.capsules.iter().zip(&b.capsules) {
            assert_eq!(a.region, b.region);
            assert!((3.0 * a.radius - b.radius).abs() < 1e-6);
            for (a, b) in a.endpoints.iter().zip(b.endpoints) {
                assert!((3.0 * a).distance(b) < 1e-6);
            }
        }
    }
}
