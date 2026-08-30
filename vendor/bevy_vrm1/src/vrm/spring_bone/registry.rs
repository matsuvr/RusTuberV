use crate::vrm::gltf::extensions::vrmc_spring_bone::{
    Collider, ColliderShape, Spring, SpringJoint, VRMCSpringBone,
};
use crate::vrm::spring_bone::SpringJointProps;
use bevy::app::App;
use bevy::math::Vec3;
use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;

pub(super) struct SpringBoneRegistryPlugin;

impl Plugin for SpringBoneRegistryPlugin {
    fn build(
        &self,
        app: &mut App,
    ) {
        app.register_type::<SpringColliderRegistry>()
            .register_type::<SpringJointPropsRegistry>()
            .register_type::<SpringNodeRegistry>();
    }
}

#[derive(Component, Deref, Debug, Default, Clone, PartialEq, Reflect)]
#[reflect(Component, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", reflect(Serialize, Deserialize))]
pub(crate) struct SpringColliderRegistry(pub(crate) HashMap<usize, ColliderShape>);

impl SpringColliderRegistry {
    pub fn new(colliders: &[Collider]) -> Self {
        Self(
            colliders
                .iter()
                .map(|collider| (collider.node, collider.shape))
                .collect(),
        )
    }
}

#[derive(Component, Deref, Debug, Default, Clone, PartialEq, Reflect)]
#[reflect(Component, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", reflect(Serialize, Deserialize))]
pub(crate) struct SpringJointPropsRegistry(pub(crate) HashMap<usize, SpringJointProps>);

impl SpringJointPropsRegistry {
    pub fn new(joints: &[SpringJoint]) -> Self {
        let mut claimed = HashSet::<usize>::default();
        Self(
            joints
                .iter()
                .filter_map(|joint| {
                    let dir = joint.gravity_dir?;
                    let props = SpringJointProps {
                        drag_force: joint.drag_force?,
                        gravity_power: joint.gravity_power?,
                        hit_radius: joint.hit_radius?,
                        stiffness: joint.stiffness?,
                        gravity_dir: Vec3::new(dir[0], dir[1], dir[2]),
                    };
                    claimed.insert(joint.node).then_some((joint.node, props))
                })
                .collect(),
        )
    }
}

#[derive(Component, Debug, Default, Clone, PartialEq, Reflect)]
#[reflect(Component, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", reflect(Serialize, Deserialize))]
pub(crate) struct SpringNode {
    pub center: Option<usize>,
    pub joints: Vec<usize>,
    pub colliders: Vec<(usize, ColliderShape)>,
    pub terminal_length: Option<f32>,
}

#[derive(Component, Deref, Default, Reflect)]
#[reflect(Component, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", reflect(Serialize, Deserialize))]
pub(crate) struct SpringNodeRegistry(pub Vec<SpringNode>);

impl SpringNodeRegistry {
    pub fn new(spring_bone: &VRMCSpringBone) -> Self {
        let mut claimed_joints = HashSet::<usize>::default();
        Self(
            spring_bone
                .springs
                .iter()
                .filter_map(|spring| {
                    let joints = spring
                        .joints
                        .iter()
                        .map(|joint| joint.node)
                        .filter(|node| claimed_joints.insert(*node))
                        .collect::<Vec<_>>();
                    (!joints.is_empty()).then_some(SpringNode {
                        joints,
                        colliders: obtain_colliders(spring_bone, spring),
                        center: spring.center,
                        terminal_length: spring.terminal_length,
                    })
                })
                .collect(),
        )
    }
}

fn obtain_colliders(
    spring_bone: &VRMCSpringBone,
    spring: &Spring,
) -> Vec<(usize, ColliderShape)> {
    let Some(collider_groups) = spring.collider_groups.as_ref() else {
        return vec![];
    };
    spring_bone
        .spring_colliders(collider_groups)
        .iter()
        .map(|collider| (collider.node, collider.shape))
        .collect()
}
