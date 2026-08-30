//! World-space bounds for renderable entities below a VRM head bone.

use std::collections::HashSet;

use bevy::camera::primitives::MeshAabb;
use bevy::mesh::VertexAttributeValues;
use bevy::prelude::*;

/// A finite world-space axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WorldBounds {
    min: Vec3,
    max: Vec3,
}

impl WorldBounds {
    pub(crate) fn new(min: Vec3, max: Vec3) -> Option<Self> {
        if !min.is_finite() || !max.is_finite() || min.x > max.x || min.y > max.y || min.z > max.z {
            return None;
        }
        Some(Self { min, max })
    }

    pub(crate) fn min(self) -> Vec3 {
        self.min
    }

    pub(crate) fn max(self) -> Vec3 {
        self.max
    }

    pub(crate) fn center(self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub(crate) fn corners(self) -> [Vec3; 8] {
        let min = self.min;
        let max = self.max;
        [
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(min.x, min.y, max.z),
            Vec3::new(min.x, max.y, min.z),
            Vec3::new(min.x, max.y, max.z),
            Vec3::new(max.x, min.y, min.z),
            Vec3::new(max.x, min.y, max.z),
            Vec3::new(max.x, max.y, min.z),
            Vec3::new(max.x, max.y, max.z),
        ]
    }

    pub(crate) fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    fn include(&mut self, other: Self) {
        *self = self.union(other);
    }
}

/// Status of the renderable geometry below a head bone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum HeadSubtreeBounds {
    /// The head subtree contains no `Mesh3d` entity.
    Empty,
    /// A renderable exists, but its mesh, transform, or bounds is not ready.
    Pending,
    /// A renderable contains invalid geometry or a non-finite transform.
    Invalid,
    /// All renderables in the subtree are represented by this finite union.
    Ready(WorldBounds),
}

/// Collects the finite world-space bounds of every `Mesh3d` below `head`.
///
/// The traversal intentionally keys inclusion only on hierarchy membership. It
/// does not inspect node names, so accessories with arbitrary names are
/// treated exactly like the head mesh. `Mesh3d` is the renderable marker;
/// bones, lights, and empty transform nodes are ignored.
pub(crate) fn collect_head_subtree_bounds(
    head: Entity,
    children: &Query<&Children>,
    renderables: &Query<&Mesh3d>,
    transforms: &Query<&GlobalTransform>,
    mesh_assets: &Assets<Mesh>,
) -> HeadSubtreeBounds {
    let mut stack = vec![head];
    let mut visited = HashSet::new();
    let mut renderable_count = 0;
    let mut pending = false;
    let mut invalid = false;
    let mut bounds: Option<WorldBounds> = None;

    while let Some(entity) = stack.pop() {
        if !visited.insert(entity) {
            continue;
        }

        if let Ok(mesh_3d) = renderables.get(entity) {
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
                total.include(world_bounds);
            } else {
                bounds = Some(world_bounds);
            }
        }

        if let Ok(entity_children) = children.get(entity) {
            stack.extend(entity_children.iter());
        }
    }

    if invalid {
        HeadSubtreeBounds::Invalid
    } else if pending {
        HeadSubtreeBounds::Pending
    } else if renderable_count == 0 {
        HeadSubtreeBounds::Empty
    } else if let Some(bounds) = bounds {
        HeadSubtreeBounds::Ready(bounds)
    } else {
        HeadSubtreeBounds::Invalid
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
    use super::*;
    use bevy::asset::RenderAssetUsages;
    use bevy::render::render_resource::PrimitiveTopology;

    #[derive(Resource)]
    struct TestHead(Entity);

    #[derive(Resource, Default)]
    struct TestResult(Option<HeadSubtreeBounds>);

    fn measure_head_subtree(
        head: Res<TestHead>,
        children: Query<&Children>,
        renderables: Query<&Mesh3d>,
        transforms: Query<&GlobalTransform>,
        mesh_assets: Res<Assets<Mesh>>,
        mut result: ResMut<TestResult>,
    ) {
        result.0 = Some(collect_head_subtree_bounds(
            head.0,
            &children,
            &renderables,
            &transforms,
            &mesh_assets,
        ));
    }

    fn app_with_measure_system() -> App {
        let mut app = App::new();
        app.init_resource::<Assets<Mesh>>()
            .init_resource::<TestResult>()
            .add_systems(Update, measure_head_subtree);
        app
    }

    fn mesh(app: &mut App, positions: &[[f32; 3]]) -> Handle<Mesh> {
        let asset = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions.to_vec());
        app.world_mut().resource_mut::<Assets<Mesh>>().add(asset)
    }

    fn set_head(app: &mut App, head: Entity) {
        app.insert_resource(TestHead(head));
        app.update();
    }

    fn result(app: &App) -> HeadSubtreeBounds {
        app.world()
            .resource::<TestResult>()
            .0
            .expect("measure system runs during app update")
    }

    fn spawn_mesh(
        app: &mut App,
        parent: Option<Entity>,
        handle: Handle<Mesh>,
        transform: GlobalTransform,
    ) -> Entity {
        let mut entity = app.world_mut().spawn((Mesh3d(handle), transform));
        if let Some(parent) = parent {
            entity.insert(ChildOf(parent));
        }
        entity.id()
    }

    #[test]
    fn unions_head_and_deep_descendants() {
        let mut app = app_with_measure_system();
        let head = app.world_mut().spawn(GlobalTransform::IDENTITY).id();
        let head_mesh = mesh(
            &mut app,
            &[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
        );
        let accessory_mesh = mesh(
            &mut app,
            &[[1.0, -0.25, -0.25], [2.0, 0.25, 0.25], [1.5, 0.0, 0.0]],
        );
        let first_child = spawn_mesh(&mut app, Some(head), head_mesh, GlobalTransform::IDENTITY);
        let second_child = app.world_mut().spawn(ChildOf(first_child)).id();
        spawn_mesh(
            &mut app,
            Some(second_child),
            accessory_mesh,
            GlobalTransform::from_translation(Vec3::new(0.0, 1.0, 0.0)),
        );

        set_head(&mut app, head);

        assert_eq!(
            result(&app),
            HeadSubtreeBounds::Ready(
                WorldBounds::new(Vec3::splat(-0.5), Vec3::new(2.0, 1.25, 0.5)).unwrap()
            )
        );
    }

    #[test]
    fn includes_wide_and_high_accessories_by_hierarchy() {
        let mut app = app_with_measure_system();
        let head = app.world_mut().spawn(GlobalTransform::IDENTITY).id();
        let accessory = mesh(
            &mut app,
            &[[-0.25, -0.25, -0.25], [0.25, 0.25, 0.25], [0.0, 0.0, 0.0]],
        );
        spawn_mesh(
            &mut app,
            Some(head),
            accessory.clone(),
            GlobalTransform::from_translation(Vec3::new(-4.0, 0.0, 0.0)),
        );
        spawn_mesh(
            &mut app,
            Some(head),
            accessory.clone(),
            GlobalTransform::from_translation(Vec3::new(4.0, 0.0, 0.0)),
        );
        spawn_mesh(
            &mut app,
            Some(head),
            accessory,
            GlobalTransform::from_translation(Vec3::new(0.0, 5.0, 0.0)),
        );

        set_head(&mut app, head);

        let HeadSubtreeBounds::Ready(bounds) = result(&app) else {
            panic!("accessories should produce ready bounds");
        };
        assert!(bounds.min().x < -4.2);
        assert!(bounds.max().x > 4.2);
        assert!(bounds.max().y > 5.2);
    }

    #[test]
    fn transforms_all_local_corners_for_rotation_and_non_uniform_scale() {
        let mut app = app_with_measure_system();
        let head = app.world_mut().spawn(GlobalTransform::IDENTITY).id();
        let local_min = Vec3::new(-1.0, -0.5, -0.25);
        let local_max = Vec3::new(1.0, 0.5, 0.25);
        let handle = mesh(
            &mut app,
            &[
                local_min.to_array(),
                local_max.to_array(),
                [local_min.x, local_max.y, local_min.z],
            ],
        );
        let rotation = Quat::from_rotation_z(0.7) * Quat::from_rotation_y(-0.35);
        let transform = GlobalTransform::from(Transform {
            translation: Vec3::new(2.0, -1.0, 3.0),
            rotation,
            scale: Vec3::new(2.0, 0.5, -3.0),
        });
        spawn_mesh(&mut app, Some(head), handle, transform);

        set_head(&mut app, head);

        let HeadSubtreeBounds::Ready(bounds) = result(&app) else {
            panic!("transformed mesh should produce ready bounds");
        };
        for corner in WorldBounds::new(local_min, local_max).unwrap().corners() {
            let transformed = transform.transform_point(corner);
            assert!(bounds.min().cmple(transformed).all());
            assert!(bounds.max().cmpge(transformed).all());
        }
    }

    #[test]
    fn excludes_head_siblings_and_ancestors() {
        let mut app = app_with_measure_system();
        let ancestor = app.world_mut().spawn(GlobalTransform::IDENTITY).id();
        let head = app.world_mut().spawn(ChildOf(ancestor)).id();
        let mesh_handle = mesh(
            &mut app,
            &[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
        );
        spawn_mesh(
            &mut app,
            Some(ancestor),
            mesh_handle.clone(),
            GlobalTransform::from_translation(Vec3::new(100.0, 0.0, 0.0)),
        );
        spawn_mesh(
            &mut app,
            None,
            mesh_handle.clone(),
            GlobalTransform::from_translation(Vec3::new(-100.0, 0.0, 0.0)),
        );
        spawn_mesh(&mut app, Some(head), mesh_handle, GlobalTransform::IDENTITY);

        set_head(&mut app, head);

        let HeadSubtreeBounds::Ready(bounds) = result(&app) else {
            panic!("head mesh should produce ready bounds");
        };
        assert!(bounds.min().x > -1.0);
        assert!(bounds.max().x < 1.0);
    }

    #[test]
    fn distinguishes_empty_pending_and_invalid_states() {
        let mut app = app_with_measure_system();
        let empty_head = app.world_mut().spawn(GlobalTransform::IDENTITY).id();
        set_head(&mut app, empty_head);
        assert_eq!(result(&app), HeadSubtreeBounds::Empty);

        let pending_head = app.world_mut().spawn(GlobalTransform::IDENTITY).id();
        let unloaded = Handle::<Mesh>::default();
        spawn_mesh(
            &mut app,
            Some(pending_head),
            unloaded,
            GlobalTransform::IDENTITY,
        );
        set_head(&mut app, pending_head);
        assert_eq!(result(&app), HeadSubtreeBounds::Pending);

        let invalid_head = app.world_mut().spawn(GlobalTransform::IDENTITY).id();
        let invalid = mesh(
            &mut app,
            &[[f32::NAN, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        );
        spawn_mesh(
            &mut app,
            Some(invalid_head),
            invalid,
            GlobalTransform::IDENTITY,
        );
        set_head(&mut app, invalid_head);
        assert_eq!(result(&app), HeadSubtreeBounds::Invalid);
    }

    #[test]
    fn missing_global_transform_is_pending_instead_of_empty() {
        let mut app = app_with_measure_system();
        let head = app.world_mut().spawn(GlobalTransform::IDENTITY).id();
        let handle = mesh(
            &mut app,
            &[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
        );
        let child = app.world_mut().spawn((Mesh3d(handle), ChildOf(head))).id();
        app.world_mut()
            .entity_mut(child)
            .remove::<GlobalTransform>();

        set_head(&mut app, head);

        assert_eq!(result(&app), HeadSubtreeBounds::Pending);
    }
}
