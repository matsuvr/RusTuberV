//! World-space bounds for avatar hair meshes.
//!
//! VRM hair is usually a skinned mesh sibling of the body under the avatar
//! root, not a child of the head bone. Name matching keeps the automatic
//! framing focused on hair without pulling the whole body into view.

use std::collections::HashSet;

use bevy::camera::primitives::MeshAabb;
use bevy::mesh::VertexAttributeValues;
use bevy::prelude::*;

use super::head_subtree_bounds::WorldBounds;

/// Status of hair geometry below an avatar root.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum HairBounds {
    /// No hair `Mesh3d` entity was found below the avatar root.
    Empty,
    /// A hair renderable exists, but its mesh, transform, or bounds is not ready.
    Pending,
    /// A hair renderable contains invalid geometry or a non-finite transform.
    Invalid,
    /// All hair renderables are represented by this finite union.
    Ready(WorldBounds),
}

/// Returns whether an entity name identifies hair geometry.
fn is_hair_name(name: &str) -> bool {
    name.contains('髪') || name.to_ascii_lowercase().contains("hair")
}

/// Collects the finite world-space bounds of every hair `Mesh3d` below `root`.
///
/// A mesh is treated as hair when its own [`Name`] or any ancestor's [`Name`]
/// up to `root` identifies hair. Bones, lights, and empty nodes are ignored.
pub(crate) fn collect_hair_bounds(
    root: Entity,
    children: &Query<&Children>,
    renderables: &Query<&Mesh3d>,
    names: &Query<&Name>,
    transforms: &Query<&GlobalTransform>,
    mesh_assets: &Assets<Mesh>,
) -> HairBounds {
    let Ok(root_children) = children.get(root) else {
        return HairBounds::Empty;
    };
    let mut stack: Vec<(Entity, bool)> =
        root_children.iter().map(|entity| (entity, false)).collect();
    let mut visited = HashSet::new();
    let mut renderable_count = 0;
    let mut pending = false;
    let mut invalid = false;
    let mut bounds: Option<WorldBounds> = None;

    while let Some((entity, hair_ancestor)) = stack.pop() {
        if !visited.insert(entity) {
            continue;
        }
        let is_hair = hair_ancestor
            || names
                .get(entity)
                .is_ok_and(|name| is_hair_name(name.as_str()));

        if is_hair && let Ok(mesh_3d) = renderables.get(entity) {
            renderable_count += 1;
            let Some(mesh) = mesh_assets.get(&mesh_3d.0) else {
                pending = true;
                continue;
            };
            let Ok(global_transform) = transforms.get(entity) else {
                pending = true;
                continue;
            };
            let Some(local_bounds) = mesh.compute_aabb() else {
                invalid = true;
                continue;
            };
            if !mesh_positions_are_finite(mesh) {
                invalid = true;
                continue;
            }
            let Some(local_bounds) =
                WorldBounds::new(local_bounds.min().into(), local_bounds.max().into())
            else {
                invalid = true;
                continue;
            };
            let Some(world_bounds) = world_bounds(local_bounds, global_transform) else {
                invalid = true;
                continue;
            };

            if let Some(total) = &mut bounds {
                *total = total.union(world_bounds);
            } else {
                bounds = Some(world_bounds);
            }
        }

        if let Ok(entity_children) = children.get(entity) {
            stack.extend(entity_children.iter().map(|child| (child, is_hair)));
        }
    }

    if invalid {
        HairBounds::Invalid
    } else if pending {
        HairBounds::Pending
    } else if renderable_count == 0 {
        HairBounds::Empty
    } else if let Some(bounds) = bounds {
        HairBounds::Ready(bounds)
    } else {
        HairBounds::Invalid
    }
}

fn mesh_positions_are_finite(mesh: &Mesh) -> bool {
    let Some(positions) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) else {
        return true;
    };
    match positions {
        VertexAttributeValues::Float32x3(values) => {
            values.iter().flatten().all(|value| value.is_finite())
        }
        _ => false,
    }
}

fn world_bounds(
    local_bounds: WorldBounds,
    global_transform: &GlobalTransform,
) -> Option<WorldBounds> {
    let mut corners = local_bounds.corners().into_iter();
    let first = global_transform.transform_point(corners.next()?);
    if !first.is_finite() {
        return None;
    }
    let mut min = first;
    let mut max = first;
    for local_corner in corners {
        let world_corner = global_transform.transform_point(local_corner);
        if !world_corner.is_finite() {
            return None;
        }
        min = min.min(world_corner);
        max = max.max(world_corner);
    }
    WorldBounds::new(min, max)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )] // tests may panic (AGENTS.md)
    use super::*;
    use bevy::asset::RenderAssetUsages;
    use bevy::render::render_resource::PrimitiveTopology;

    #[derive(Resource)]
    struct TestRoot(Entity);

    #[derive(Resource, Default)]
    struct TestResult(Option<HairBounds>);

    fn measure_hair(
        root: Res<TestRoot>,
        children: Query<&Children>,
        renderables: Query<&Mesh3d>,
        names: Query<&Name>,
        transforms: Query<&GlobalTransform>,
        mesh_assets: Res<Assets<Mesh>>,
        mut result: ResMut<TestResult>,
    ) {
        result.0 = Some(collect_hair_bounds(
            root.0,
            &children,
            &renderables,
            &names,
            &transforms,
            &mesh_assets,
        ));
    }

    fn mesh(app: &mut App, positions: &[[f32; 3]]) -> Handle<Mesh> {
        let asset = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions.to_vec());
        app.world_mut().resource_mut::<Assets<Mesh>>().add(asset)
    }

    fn cube() -> [[f32; 3]; 8] {
        [
            [-0.25, -0.25, -0.25],
            [-0.25, -0.25, 0.25],
            [-0.25, 0.25, -0.25],
            [-0.25, 0.25, 0.25],
            [0.25, -0.25, -0.25],
            [0.25, -0.25, 0.25],
            [0.25, 0.25, -0.25],
            [0.25, 0.25, 0.25],
        ]
    }

    fn app_with_measure() -> App {
        let mut app = App::new();
        app.init_resource::<Assets<Mesh>>()
            .init_resource::<TestResult>()
            .add_systems(Update, measure_hair);
        app
    }

    fn collect(app: &mut App, root: Entity) -> HairBounds {
        app.insert_resource(TestRoot(root));
        app.update();
        app.world()
            .resource::<TestResult>()
            .0
            .expect("measure system runs during app update")
    }

    #[test]
    fn finds_hair_mesh_by_own_name() {
        let mut app = app_with_measure();
        let root = app.world_mut().spawn_empty().id();
        let handle = mesh(&mut app, &cube());
        let hair = app
            .world_mut()
            .spawn((
                Name::new("Hair"),
                Mesh3d(handle),
                GlobalTransform::from_translation(Vec3::new(0.0, 2.0, 0.0)),
            ))
            .id();
        app.world_mut().entity_mut(root).add_child(hair);

        let HairBounds::Ready(bounds) = collect(&mut app, root) else {
            panic!("hair mesh should produce ready bounds");
        };
        assert!(bounds.min().y > 1.5);
        assert!(bounds.max().y > 2.0);
    }

    #[test]
    fn finds_hair_mesh_by_ancestor_name_case_insensitive() {
        let mut app = app_with_measure();
        let root = app.world_mut().spawn_empty().id();
        let node = app
            .world_mut()
            .spawn((Name::new("HAIR_root"), GlobalTransform::IDENTITY))
            .id();
        let handle = mesh(&mut app, &cube());
        let inner = app
            .world_mut()
            .spawn((
                Name::new("mesh"),
                Mesh3d(handle),
                GlobalTransform::from_translation(Vec3::new(3.0, 0.0, 0.0)),
            ))
            .id();
        app.world_mut().entity_mut(node).add_child(inner);
        app.world_mut().entity_mut(root).add_child(node);

        let HairBounds::Ready(bounds) = collect(&mut app, root) else {
            panic!("ancestor hair name should include descendant mesh");
        };
        assert!(bounds.min().x > 2.5);
    }

    #[test]
    fn ignores_non_hair_meshes() {
        let mut app = app_with_measure();
        let root = app.world_mut().spawn_empty().id();
        let handle = mesh(&mut app, &cube());
        let body = app
            .world_mut()
            .spawn((Name::new("Body"), Mesh3d(handle), GlobalTransform::IDENTITY))
            .id();
        app.world_mut().entity_mut(root).add_child(body);

        assert_eq!(collect(&mut app, root), HairBounds::Empty);
    }

    #[test]
    fn missing_mesh_asset_is_pending() {
        let mut app = app_with_measure();
        let root = app.world_mut().spawn_empty().id();
        let hair = app
            .world_mut()
            .spawn((
                Name::new("Hair"),
                Mesh3d(Handle::<Mesh>::default()),
                GlobalTransform::IDENTITY,
            ))
            .id();
        app.world_mut().entity_mut(root).add_child(hair);

        assert_eq!(collect(&mut app, root), HairBounds::Pending);
    }
}
