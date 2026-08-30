use crate::prelude::ChildSearcher;
use crate::vrm::VrmNodeIndex;
use crate::vrm::humanoid_bone::RequestInitializeHumanoidBones;
use crate::vrm::spring_bone::registry::{
    SpringColliderRegistry, SpringJointPropsRegistry, SpringNodeRegistry,
};
use crate::vrm::spring_bone::{
    SpringCenterNode, SpringColliders, SpringJointState, SpringJoints, SpringRoot,
};
use bevy::app::{App, Update};
use bevy::prelude::*;

#[derive(EntityEvent)]
pub(crate) struct RequestInitializeSpringBone(pub(crate) Entity);

pub struct SpringBoneInitializePlugin;

impl Plugin for SpringBoneInitializePlugin {
    fn build(
        &self,
        app: &mut App,
    ) {
        app.add_systems(Update, init_spring_joint_states)
            .add_observer(apply_initialize_joint_props)
            .add_observer(apply_initialize_collider_shapes)
            .add_observer(apply_initialize_spring_roots);
    }
}

fn apply_initialize_joint_props(
    trigger: On<RequestInitializeHumanoidBones>,
    mut commands: Commands,
    child_searcher: ChildSearcher,
    models: Query<&SpringJointPropsRegistry>,
) {
    let root = trigger.event_target();
    let Ok(nodes) = models.get(root) else {
        return;
    };
    for (node_index, props) in nodes.iter() {
        let Some(joint_entity) = child_searcher.find_from_node_index(root, *node_index) else {
            continue;
        };
        commands.entity(joint_entity).insert(*props);
    }
}

fn apply_initialize_collider_shapes(
    trigger: On<RequestInitializeSpringBone>,
    mut commands: Commands,
    child_searcher: ChildSearcher,
    models: Query<&SpringColliderRegistry>,
) {
    let entity = trigger.event_target();
    let Ok(registry) = models.get(entity) else {
        return;
    };
    for (node_index, shape) in registry.iter() {
        let Some(collider_entity) = child_searcher.find_from_node_index(entity, *node_index) else {
            continue;
        };
        commands.entity(collider_entity).insert(*shape);
    }
}

fn apply_initialize_spring_roots(
    trigger: On<RequestInitializeSpringBone>,
    mut commands: Commands,
    child_searcher: ChildSearcher,
    models: Query<&SpringNodeRegistry>,
) {
    let entity = trigger.event_target();
    let Ok(registry) = models.get(entity) else {
        return;
    };
    for spring_root in registry.0.iter().map(|spring| SpringRoot {
        center_node: SpringCenterNode(
            spring
                .center
                .and_then(|center| child_searcher.find_from_node_index(entity, center)),
        ),
        joints: SpringJoints(
            spring
                .joints
                .iter()
                .filter_map(|joint| child_searcher.find_from_node_index(entity, *joint))
                .collect(),
        ),
        colliders: SpringColliders(
            spring
                .colliders
                .iter()
                .filter_map(|(collider, shape)| {
                    let entity = child_searcher.find_from_node_index(entity, *collider)?;
                    Some((entity, *shape))
                })
                .collect(),
        ),
        terminal_length: spring.terminal_length,
    }) {
        let Some(root) = spring_root.joints.first() else {
            continue;
        };
        commands.entity(*root).insert(spring_root);
    }
}

fn init_spring_joint_states(
    par_commands: ParallelCommands,
    spring_roots: Query<&SpringRoot, Added<SpringRoot>>,
    joints: Query<&Transform>,
    global_transforms: Query<&GlobalTransform>,
    children: Query<&Children>,
    node_indices: Query<Option<&VrmNodeIndex>>,
) {
    spring_roots.par_iter().for_each(|root| {
        for w in root.joints.windows(2) {
            let head_entity = w[0];
            let joint_entity = w[1];
            let Ok(head_tf) = joints.get(head_entity) else {
                continue;
            };
            let Ok(tail_tf) = joints.get(joint_entity) else {
                continue;
            };
            let Ok(tail_gtf) = global_transforms.get(joint_entity) else {
                continue;
            };
            let tail_pos = root
                .center_node
                .and_then(|center| global_transforms.get(center).ok())
                .map(|center_gtf| tail_gtf.reparented_to(center_gtf).translation)
                .unwrap_or(tail_gtf.translation());
            let bone_length = tail_tf.translation.length();
            if !bone_length.is_finite() || bone_length <= f32::EPSILON {
                continue;
            }
            let state = SpringJointState {
                prev_tail: tail_pos,
                current_tail: tail_pos,
                bone_axis: tail_tf.translation / bone_length,
                bone_length,
                initial_local_matrix: head_tf.to_matrix(),
                initial_local_rotation: head_tf.rotation,
            };
            par_commands.command_scope(|mut commands| {
                commands.entity(head_entity).insert(state);
            });
        }
        if let (Some(&last_entity), Some(terminal_length)) =
            (root.joints.last(), root.terminal_length)
            && let (Ok(last_tf), Ok(last_gtf)) =
                (joints.get(last_entity), global_transforms.get(last_entity))
            && let Some(bone_axis) =
                terminal_direction(last_entity, &children, &joints, &node_indices, last_tf)
        {
            let terminal_global =
                last_gtf.translation() + last_gtf.rotation().mul_vec3(bone_axis * terminal_length);
            let terminal_tail = root
                .center_node
                .and_then(|center| global_transforms.get(center).ok())
                .map(|center_gtf| {
                    center_gtf
                        .to_matrix()
                        .inverse()
                        .transform_point3(terminal_global)
                })
                .unwrap_or(terminal_global);
            let state = SpringJointState {
                prev_tail: terminal_tail,
                current_tail: terminal_tail,
                bone_axis,
                bone_length: terminal_length,
                initial_local_matrix: last_tf.to_matrix(),
                initial_local_rotation: last_tf.rotation,
            };
            par_commands.command_scope(|mut commands| {
                commands.entity(last_entity).insert(state);
            });
        }
    });
}

fn terminal_direction(
    entity: Entity,
    children: &Query<&Children>,
    transforms: &Query<&Transform>,
    node_indices: &Query<Option<&VrmNodeIndex>>,
    last_transform: &Transform,
) -> Option<Vec3> {
    let finite_nonzero_direction = |translation: Vec3| {
        let length_squared = translation.length_squared();
        (translation.is_finite()
            && length_squared.is_finite()
            && length_squared > f32::EPSILON)
            .then(|| translation.normalize())
    };

    // A leaf node normally has no `Children` component. That is a valid
    // glTF state, so absence of the component must fall through to the
    // parent-to-last-joint fallback instead of returning early.
    let child_direction = children
        .get(entity)
        .ok()
        .into_iter()
        .flat_map(|children| children.iter())
        .filter_map(|child| {
            let transform = transforms.get(child).ok()?;
            node_indices
                .get(child)
                .ok()
                .flatten()
                .and_then(|_| finite_nonzero_direction(transform.translation))
        })
        .next();

    child_direction.or_else(|| finite_nonzero_direction(last_transform.translation))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::app::App;

    fn terminal_state(
        last_translation: Vec3,
        child_translation: Option<Vec3>,
        terminal_length: Option<f32>,
    ) -> Option<SpringJointState> {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_systems(Update, init_spring_joint_states);

        let last_transform = Transform::from_translation(last_translation);
        let last_entity = app
            .world_mut()
            .spawn((
                last_transform,
                GlobalTransform::from(last_transform),
                VrmNodeIndex(0),
            ))
            .id();
        app.world_mut()
            .entity_mut(last_entity)
            .insert(SpringRoot {
                joints: SpringJoints(vec![last_entity]),
                colliders: SpringColliders::default(),
                center_node: SpringCenterNode::default(),
                terminal_length,
            });

        if let Some(child_translation) = child_translation {
            let child_transform = Transform::from_translation(child_translation);
            app.world_mut().spawn((
                child_transform,
                GlobalTransform::from(child_transform),
                VrmNodeIndex(1),
                ChildOf(last_entity),
            ));
        }

        app.update();
        app.world().get::<SpringJointState>(last_entity).cloned()
    }

    #[test]
    fn leaf_without_children_component_uses_last_joint_fallback() {
        let state = terminal_state(Vec3::X, None, Some(0.07))
            .expect("a leaf still needs a synthetic terminal state");

        assert_eq!(state.bone_axis, Vec3::X);
        assert_eq!(state.bone_length, 0.07);
        assert!(state.current_tail.is_finite());
    }

    #[test]
    fn valid_source_child_direction_wins_over_last_joint_fallback() {
        let state = terminal_state(Vec3::X, Some(Vec3::Z), Some(0.07))
            .expect("a valid source child supplies the terminal direction");

        assert_eq!(state.bone_axis, Vec3::Z);
        assert_eq!(state.bone_length, 0.07);
    }

    #[test]
    fn zero_or_non_finite_child_translation_uses_valid_fallback() {
        let zero_child = terminal_state(Vec3::Y, Some(Vec3::ZERO), Some(0.07))
            .expect("zero child translation should fall back");
        assert_eq!(zero_child.bone_axis, Vec3::Y);

        let non_finite_child = terminal_state(Vec3::Y, Some(Vec3::NAN), Some(0.07))
            .expect("non-finite child translation should fall back");
        assert_eq!(non_finite_child.bone_axis, Vec3::Y);
        assert!(non_finite_child.current_tail.is_finite());
    }

    #[test]
    fn zero_child_and_zero_last_joint_skip_terminal_without_nan() {
        assert!(terminal_state(Vec3::ZERO, Some(Vec3::ZERO), Some(0.07)).is_none());
    }

    #[test]
    fn missing_terminal_length_preserves_vrm1_no_terminal_state_path() {
        assert!(terminal_state(Vec3::Y, Some(Vec3::Z), None).is_none());
    }
}
