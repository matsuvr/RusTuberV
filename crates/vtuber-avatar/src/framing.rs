//! Avatar-aware viewport camera framing.

pub mod camera_control;
pub mod camera_input;
pub mod camera_reset;
pub(crate) mod fixed_fov_fit;
pub(crate) mod head_subtree_bounds;

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy_vrm1::prelude::{HeadBoneEntity, HipsBoneEntity};

use self::camera_control::{AvatarCameraControl, CameraControlPose};
use self::fixed_fov_fit::{FIXED_VERTICAL_FOV, solve_fixed_fov_fit};
use self::head_subtree_bounds::{HeadSubtreeBounds, WorldBounds, collect_head_subtree_bounds};
use crate::lifecycle::{AvatarLifecycle, AvatarLifecycleState};

const TARGET_FROM_HIPS: f32 = 0.60;
const VERTICAL_HALF_EXTENT_IN_HIPS_HEADS: f32 = 0.75;
const MIN_HIPS_HEAD_HEIGHT: f32 = 0.05;

/// Marks the single camera used to render the avatar viewport.
#[derive(Component)]
pub struct AvatarViewportCamera {
    /// Rotation used by every generation's automatic framing solve.
    ///
    /// This is deliberately separate from the camera's mutable transform:
    /// manual orbit changes the latter, but never this generation-independent
    /// framing authority.
    default_framing_rotation: Quat,
}

impl AvatarViewportCamera {
    /// Captures the camera's formal startup transform as immutable framing
    /// authority.
    #[must_use]
    pub fn from_default_transform(transform: Transform) -> Self {
        Self {
            default_framing_rotation: transform.rotation,
        }
    }

    fn default_framing_rotation(&self) -> Quat {
        self.default_framing_rotation
    }
}

#[derive(SystemParam)]
pub(crate) struct FramingWorld<'w, 's> {
    roots: Query<'w, 's, (&'static HeadBoneEntity, &'static HipsBoneEntity)>,
    bones: Query<'w, 's, &'static GlobalTransform>,
    children: Query<'w, 's, &'static Children>,
    renderables: Query<'w, 's, &'static Mesh3d>,
    mesh_assets: Res<'w, Assets<Mesh>>,
}

/// Frames a newly-ready avatar once, keeping live head motion visible instead
/// of making the camera follow and cancel it.
pub(crate) fn frame_avatar_camera(
    lifecycle: Res<AvatarLifecycle>,
    world: FramingWorld,
    mut cameras: Query<
        (
            &Camera,
            &AvatarViewportCamera,
            &mut Transform,
            &mut Projection,
        ),
        With<AvatarViewportCamera>,
    >,
    mut camera_control: ResMut<AvatarCameraControl>,
) {
    if lifecycle.state() != AvatarLifecycleState::Ready {
        camera_control.invalidate();
        return;
    }
    let generation = lifecycle.current_generation();
    if camera_control.current_for(generation).is_some() {
        return;
    }
    let Some(root) = lifecycle.active_root() else {
        return;
    };
    let Ok((head_entity, hips_entity)) = world.roots.get(root) else {
        return;
    };
    let (Ok(head), Ok(hips)) = (
        world.bones.get(**head_entity),
        world.bones.get(**hips_entity),
    ) else {
        return;
    };

    let subtree_bounds = collect_head_subtree_bounds(
        **head_entity,
        &world.children,
        &world.renderables,
        &world.bones,
        &world.mesh_assets,
    );
    let Some(upper_body_bounds) = upper_body_bounds(head.translation(), hips.translation()) else {
        return;
    };
    let bounds = match subtree_bounds {
        HeadSubtreeBounds::Pending => return,
        HeadSubtreeBounds::Empty | HeadSubtreeBounds::Invalid => upper_body_bounds,
        HeadSubtreeBounds::Ready(subtree_bounds) => upper_body_bounds.union(subtree_bounds),
    };

    let mut framed_pose = None;
    for (camera, viewport_camera, mut transform, mut projection) in &mut cameras {
        let Some(viewport_size) = camera.physical_viewport_size() else {
            continue;
        };
        if viewport_size.x == 0 || viewport_size.y == 0 {
            continue;
        }
        let aspect_ratio = viewport_size.x as f32 / viewport_size.y as f32;
        let Projection::Perspective(perspective) = &*projection else {
            continue;
        };

        let framing_rotation = viewport_camera.default_framing_rotation();
        let Ok(fit) = solve_fixed_fov_fit(bounds, framing_rotation, aspect_ratio, perspective.near)
        else {
            continue;
        };
        if !fit.translation.is_finite() {
            continue;
        }
        let mut framed_transform = *transform;
        framed_transform.translation = fit.translation;
        framed_transform.rotation = framing_rotation;
        let Some(pose) =
            CameraControlPose::from_parts(framed_transform, fit.target, fit.distance).ok()
        else {
            continue;
        };

        // Commit the camera and projection together only after the fit and
        // control pose have both succeeded. Pending or failed bounds must not
        // leave a partially updated camera behind.
        let Projection::Perspective(perspective) = &mut *projection else {
            continue;
        };
        perspective.fov = FIXED_VERTICAL_FOV;
        perspective.aspect_ratio = aspect_ratio;
        *transform = framed_transform;
        framed_pose = Some(pose);
        break;
    }
    if let Some(pose) = framed_pose {
        camera_control.initialize(generation, pose);
    }
}

/// The avatar-anchored orbit focus derived from live head and hips positions.
///
/// This is the same upper-body anchor the automatic framing solve uses, so
/// orbiting stays centered on the avatar as tracking moves it around the
/// scene.
pub(crate) fn avatar_focus_point(head: Vec3, hips: Vec3) -> Option<Vec3> {
    upper_body_bounds(head, hips).map(|bounds| bounds.center())
}

fn upper_body_bounds(head: Vec3, hips: Vec3) -> Option<WorldBounds> {
    if !head.is_finite() || !hips.is_finite() {
        return None;
    }
    let hips_head_height = head.y - hips.y;
    if hips_head_height < MIN_HIPS_HEAD_HEIGHT {
        return None;
    }

    let target = hips.lerp(head, TARGET_FROM_HIPS);
    let half_vertical_extent = hips_head_height * VERTICAL_HALF_EXTENT_IN_HIPS_HEADS;
    let min = Vec3::new(
        head.x.min(hips.x),
        target.y - half_vertical_extent,
        head.z.min(hips.z),
    );
    let max = Vec3::new(
        head.x.max(hips.x),
        target.y + half_vertical_extent,
        head.z.max(hips.z),
    );
    WorldBounds::new(min, max)
}

#[cfg(test)]
fn upper_body_camera_transform(head: Vec3, hips: Vec3, vertical_fov: f32) -> Option<Transform> {
    if !head.is_finite() || !hips.is_finite() || !vertical_fov.is_finite() {
        return None;
    }
    let hips_head_height = head.y - hips.y;
    if hips_head_height < MIN_HIPS_HEAD_HEIGHT
        || !(0.1..std::f32::consts::PI - 0.1).contains(&vertical_fov)
    {
        return None;
    }

    // With this target and extent, the head is at about the upper quarter and
    // the hips at about the lower tenth of the viewport.
    let target = hips.lerp(head, TARGET_FROM_HIPS);
    let half_extent = hips_head_height * VERTICAL_HALF_EXTENT_IN_HIPS_HEADS;
    let distance = half_extent / (vertical_fov * 0.5).tan();
    if !distance.is_finite() || distance <= 0.0 {
        return None;
    }
    Some(Transform::from_translation(target + Vec3::Z * distance).looking_at(target, Vec3::Y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::asset::RenderAssetUsages;
    use bevy::camera::Viewport;
    use bevy::render::render_resource::PrimitiveTopology;

    fn spawn_humanoid(app: &mut App, x: f32) -> Entity {
        spawn_humanoid_parts(app, x).0
    }

    fn spawn_humanoid_parts(app: &mut App, x: f32) -> (Entity, Entity, Entity) {
        let hips = app
            .world_mut()
            .spawn(GlobalTransform::from_translation(Vec3::new(x, 1.0, 0.0)))
            .id();
        let head = app
            .world_mut()
            .spawn(GlobalTransform::from_translation(Vec3::new(x, 1.8, 0.0)))
            .id();
        let root = app
            .world_mut()
            .spawn((HeadBoneEntity(head), HipsBoneEntity(hips)))
            .id();
        (root, head, hips)
    }

    fn make_ready(app: &mut App, root: Entity) {
        let mut lifecycle = app.world_mut().resource_mut::<AvatarLifecycle>();
        lifecycle.request_load(root).expect("load from empty state");
        lifecycle.start_binding(root);
        lifecycle.finish_ready();
    }

    fn frame_app(viewport_size: UVec2) -> App {
        let mut app = App::new();
        app.init_resource::<AvatarLifecycle>()
            .init_resource::<Assets<Mesh>>()
            .init_resource::<AvatarCameraControl>()
            .add_systems(Update, frame_avatar_camera);
        let camera_transform = Transform::from_xyz(0.0, 0.0, 2.5);
        app.world_mut().spawn((
            Camera3d::default(),
            Camera {
                viewport: Some(Viewport {
                    physical_size: viewport_size,
                    ..default()
                }),
                ..default()
            },
            AvatarViewportCamera::from_default_transform(camera_transform),
            camera_transform,
        ));
        app
    }

    fn camera_entity(app: &mut App) -> Entity {
        let mut query = app
            .world_mut()
            .query_filtered::<Entity, With<AvatarViewportCamera>>();
        query
            .iter(app.world())
            .next()
            .expect("frame test app has an avatar camera")
    }

    fn mesh_asset(app: &mut App, positions: &[[f32; 3]]) -> Handle<Mesh> {
        let mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions.to_vec());
        app.world_mut().resource_mut::<Assets<Mesh>>().add(mesh)
    }

    fn cube_positions() -> [[f32; 3]; 8] {
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

    fn spawn_renderable(
        app: &mut App,
        parent: Entity,
        mesh: Handle<Mesh>,
        transform: GlobalTransform,
    ) -> Entity {
        app.world_mut()
            .spawn((Mesh3d(mesh), ChildOf(parent), transform))
            .id()
    }

    fn assert_projected_inside(
        camera: Transform,
        projection: &PerspectiveProjection,
        bounds: WorldBounds,
    ) {
        let inverse_rotation = camera.rotation.inverse();
        let vertical_tangent = (projection.fov * 0.5).tan();
        let horizontal_tangent = vertical_tangent * projection.aspect_ratio;
        for corner in bounds.corners() {
            let camera_relative = inverse_rotation * (corner - camera.translation);
            let depth = -camera_relative.z;
            assert!(depth >= projection.near, "depth={depth}");
            let ndc_x = camera_relative.x / (depth * horizontal_tangent);
            let ndc_y = camera_relative.y / (depth * vertical_tangent);
            assert!((-0.95..=0.95).contains(&ndc_x), "ndc_x={ndc_x}");
            assert!((-0.95..=0.95).contains(&ndc_y), "ndc_y={ndc_y}");
        }
    }

    #[test]
    fn framing_places_head_high_and_hips_low() {
        let hips = Vec3::new(0.0, 1.0, 0.0);
        let head = Vec3::new(0.0, 1.8, 0.0);
        let transform = upper_body_camera_transform(head, hips, FIXED_VERTICAL_FOV)
            .expect("valid humanoid framing should be computed");
        let target_y = hips.lerp(head, TARGET_FROM_HIPS).y;
        let projected_half_extent = transform.translation.z * (FIXED_VERTICAL_FOV * 0.5).tan();
        let head_screen_y = 0.5 - (head.y - target_y) / projected_half_extent * 0.5;
        let hips_screen_y = 0.5 - (hips.y - target_y) / projected_half_extent * 0.5;

        assert!((0.15..0.35).contains(&head_screen_y), "{head_screen_y}");
        assert!((0.80..0.95).contains(&hips_screen_y), "{hips_screen_y}");
    }

    #[test]
    fn framing_centres_model_world_offset() {
        let hips = Vec3::new(2.0, 0.8, -3.0);
        let head = Vec3::new(2.2, 1.8, -2.8);
        let transform = upper_body_camera_transform(head, hips, FIXED_VERTICAL_FOV)
            .expect("valid offset humanoid framing should be computed");
        let target = hips.lerp(head, TARGET_FROM_HIPS);

        assert!((transform.translation.x - target.x).abs() < 1e-6);
        assert!(transform.translation.z > target.z);
        assert!(
            transform
                .forward()
                .dot((target - transform.translation).normalize())
                > 0.999
        );
    }

    #[test]
    fn invalid_bone_geometry_uses_existing_fixed_camera() {
        assert!(upper_body_camera_transform(Vec3::ZERO, Vec3::ZERO, FIXED_VERTICAL_FOV).is_none());
        assert!(
            upper_body_camera_transform(Vec3::splat(f32::NAN), Vec3::ZERO, FIXED_VERTICAL_FOV)
                .is_none()
        );
    }

    #[test]
    fn replacement_generation_is_reframed() {
        let mut app = App::new();
        app.init_resource::<AvatarLifecycle>()
            .init_resource::<Assets<Mesh>>()
            .init_resource::<AvatarCameraControl>()
            .init_resource::<crate::framing::camera_input::CameraPointerGesture>()
            .add_message::<crate::framing::camera_reset::ResetCameraRequest>()
            .add_systems(
                Update,
                (
                    frame_avatar_camera,
                    crate::framing::camera_reset::reset_avatar_camera,
                ),
            );
        let camera = app
            .world_mut()
            .spawn((
                Camera3d::default(),
                Camera {
                    viewport: Some(Viewport {
                        physical_size: UVec2::new(1600, 900),
                        ..default()
                    }),
                    ..default()
                },
                AvatarViewportCamera::from_default_transform(Transform::from_xyz(0.0, 0.0, 2.5)),
                Transform::from_xyz(0.0, 0.0, 2.5),
            ))
            .id();
        let first = spawn_humanoid(&mut app, 0.0);
        make_ready(&mut app, first);

        app.update();
        let first_x = app.world().get::<Transform>(camera).unwrap().translation.x;
        let first_generation = app
            .world()
            .resource::<AvatarLifecycle>()
            .current_generation();
        let controls = app.world().resource::<AvatarCameraControl>();
        let first_default = controls
            .default_for(first_generation)
            .expect("first generation has a default pose");
        assert_eq!(controls.current_for(first_generation), Some(first_default));

        let first_manual = crate::framing::camera_control::geometry::orbit(first_default, 0.7, 0.2)
            .expect("first generation manual orbit is valid");
        assert!(
            app.world_mut()
                .resource_mut::<AvatarCameraControl>()
                .set_current(first_generation, first_manual)
        );
        *app.world_mut()
            .get_mut::<Transform>(camera)
            .expect("avatar camera transform") = first_manual.transform();
        assert_eq!(
            app.world()
                .resource::<AvatarCameraControl>()
                .current_for(first_generation),
            Some(first_manual)
        );
        assert_ne!(
            first_manual.transform().rotation,
            first_default.transform().rotation
        );

        let replacement = spawn_humanoid(&mut app, 3.0);
        {
            let mut lifecycle = app.world_mut().resource_mut::<AvatarLifecycle>();
            lifecycle
                .request_replace(replacement)
                .expect("replace from ready state");
            lifecycle.finish_unload();
            lifecycle.start_binding(replacement);
            lifecycle.finish_ready();
        }
        app.update();
        let replacement_transform = *app.world().get::<Transform>(camera).unwrap();
        let replacement_generation = app
            .world()
            .resource::<AvatarLifecycle>()
            .current_generation();

        assert!(first_x.abs() < 1e-6);
        assert!((replacement_transform.translation.x - 3.0).abs() < 1e-6);
        let controls = app.world().resource::<AvatarCameraControl>();
        assert!(controls.current_for(first_generation).is_none());
        let replacement_default = controls
            .default_for(replacement_generation)
            .expect("replacement generation has a default pose");
        assert_ne!(replacement_default, first_default);
        assert_eq!(
            controls.current_for(replacement_generation),
            Some(replacement_default)
        );
        let canonical_rotation = app
            .world()
            .get::<AvatarViewportCamera>(camera)
            .expect("avatar viewport camera marker")
            .default_framing_rotation();
        assert_eq!(replacement_transform.rotation, canonical_rotation);
        assert_ne!(
            replacement_transform.rotation,
            first_manual.transform().rotation
        );
        assert_ne!(replacement_default.target(), first_manual.target());
        assert!((replacement_default.target().x - 3.0).abs() < 1e-6);

        let replacement_bounds =
            upper_body_bounds(Vec3::new(3.0, 1.8, 0.0), Vec3::new(3.0, 1.0, 0.0))
                .expect("replacement bounds");
        let expected_fit =
            solve_fixed_fov_fit(replacement_bounds, canonical_rotation, 1600.0 / 900.0, 0.1)
                .expect("replacement bounds fit");
        assert_eq!(replacement_default.target(), expected_fit.target);
        assert!((replacement_default.distance() - expected_fit.distance).abs() < 1e-5);
        let projection = app.world().get::<Projection>(camera).unwrap();
        let Projection::Perspective(perspective) = projection else {
            panic!("avatar viewport camera must use perspective projection");
        };
        assert!((perspective.fov - FIXED_VERTICAL_FOV).abs() < f32::EPSILON);

        let replacement_manual =
            crate::framing::camera_control::geometry::orbit(replacement_default, -0.4, -0.15)
                .and_then(|pose| {
                    crate::framing::camera_control::geometry::pan(
                        pose,
                        Vec2::new(80.0, 30.0),
                        Vec2::new(1600.0, 900.0),
                    )
                })
                .expect("replacement manual camera operation is valid");
        assert_ne!(replacement_manual, replacement_default);
        *app.world_mut()
            .get_mut::<Transform>(camera)
            .expect("avatar camera transform") = replacement_manual.transform();
        assert!(
            app.world_mut()
                .resource_mut::<AvatarCameraControl>()
                .set_current(replacement_generation, replacement_manual)
        );
        *app.world_mut()
            .resource_mut::<crate::framing::camera_input::CameraPointerGesture>() =
            crate::framing::camera_input::CameraPointerGesture::Orbit {
                generation: replacement_generation,
            };
        app.world_mut()
            .resource_mut::<Messages<crate::framing::camera_reset::ResetCameraRequest>>()
            .write(crate::framing::camera_reset::ResetCameraRequest {
                generation: replacement_generation,
            });
        app.update();

        assert_eq!(
            app.world()
                .resource::<AvatarCameraControl>()
                .current_for(replacement_generation),
            Some(replacement_default)
        );
        assert_eq!(
            *app.world()
                .get::<Transform>(camera)
                .expect("avatar camera transform"),
            replacement_default.transform()
        );
        assert_eq!(
            *app.world()
                .resource::<crate::framing::camera_input::CameraPointerGesture>(),
            crate::framing::camera_input::CameraPointerGesture::None
        );
        let Projection::Perspective(perspective) = app.world().get::<Projection>(camera).unwrap()
        else {
            panic!("avatar viewport camera must use perspective projection");
        };
        assert!((perspective.fov - FIXED_VERTICAL_FOV).abs() < f32::EPSILON);
    }

    #[test]
    fn does_not_frame_before_ready_but_frames_after_ready() {
        let mut app = frame_app(UVec2::new(1600, 900));
        let camera = camera_entity(&mut app);
        let root = spawn_humanoid(&mut app, 0.0);
        let before = *app
            .world()
            .get::<Transform>(camera)
            .expect("camera transform");

        app.update();
        assert_eq!(
            app.world()
                .get::<Transform>(camera)
                .expect("camera transform"),
            &before
        );
        assert_eq!(
            app.world().resource::<AvatarCameraControl>().state(),
            crate::framing::camera_control::AvatarCameraControlState::Unavailable
        );

        make_ready(&mut app, root);
        app.update();
        let after = app
            .world()
            .get::<Transform>(camera)
            .expect("camera transform");
        assert_ne!(after.translation, before.translation);
        let Projection::Perspective(projection) = app.world().get::<Projection>(camera).unwrap()
        else {
            panic!("avatar viewport camera must use perspective projection");
        };
        assert!((projection.fov - FIXED_VERTICAL_FOV).abs() < f32::EPSILON);

        let generation = app
            .world()
            .resource::<AvatarLifecycle>()
            .current_generation();
        assert!(
            app.world()
                .resource::<AvatarCameraControl>()
                .current_for(generation)
                .is_some()
        );

        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .request_unload()
            .expect("ready avatar can be unloaded");
        app.update();
        assert_eq!(
            app.world().resource::<AvatarCameraControl>().state(),
            crate::framing::camera_control::AvatarCameraControlState::Unavailable
        );
    }

    #[test]
    fn pending_bounds_retry_without_marking_generation_framed() {
        let mut app = frame_app(UVec2::new(1600, 900));
        let camera = camera_entity(&mut app);
        let (root, head, _) = spawn_humanoid_parts(&mut app, 0.0);
        let mesh = mesh_asset(&mut app, &cube_positions());
        let accessory = spawn_renderable(&mut app, head, mesh, GlobalTransform::IDENTITY);
        app.world_mut()
            .entity_mut(accessory)
            .remove::<GlobalTransform>();
        make_ready(&mut app, root);
        let before = *app
            .world()
            .get::<Transform>(camera)
            .expect("camera transform");

        app.update();
        assert_eq!(
            app.world()
                .get::<Transform>(camera)
                .expect("camera transform"),
            &before
        );

        app.world_mut()
            .entity_mut(accessory)
            .insert(GlobalTransform::IDENTITY);
        app.update();
        assert_ne!(
            app.world()
                .get::<Transform>(camera)
                .expect("camera transform"),
            &before
        );
        let generation = app
            .world()
            .resource::<AvatarLifecycle>()
            .current_generation();
        assert!(
            app.world()
                .resource::<AvatarCameraControl>()
                .default_for(generation)
                .is_some()
        );
    }

    #[test]
    fn accessory_bounds_and_head_hips_envelope_are_inside_viewport() {
        let mut app = frame_app(UVec2::new(1600, 900));
        let camera = camera_entity(&mut app);
        let (root, head, hips) = spawn_humanoid_parts(&mut app, 0.0);
        let mesh = mesh_asset(&mut app, &cube_positions());
        spawn_renderable(
            &mut app,
            head,
            mesh.clone(),
            GlobalTransform::from_translation(Vec3::new(-4.0, 0.0, 0.0)),
        );
        spawn_renderable(
            &mut app,
            head,
            mesh.clone(),
            GlobalTransform::from_translation(Vec3::new(4.0, 0.0, 0.0)),
        );
        spawn_renderable(
            &mut app,
            head,
            mesh,
            GlobalTransform::from_translation(Vec3::new(0.0, 5.0, 0.0)),
        );
        make_ready(&mut app, root);
        app.update();

        let camera_transform = *app
            .world()
            .get::<Transform>(camera)
            .expect("camera transform");
        let Projection::Perspective(projection) = app.world().get::<Projection>(camera).unwrap()
        else {
            panic!("avatar viewport camera must use perspective projection");
        };
        let base = upper_body_bounds(
            app.world()
                .get::<GlobalTransform>(head)
                .unwrap()
                .translation(),
            app.world()
                .get::<GlobalTransform>(hips)
                .unwrap()
                .translation(),
        )
        .unwrap();
        let accessory =
            WorldBounds::new(Vec3::new(-4.25, -0.25, -0.25), Vec3::new(4.25, 5.25, 0.25)).unwrap();
        assert_projected_inside(camera_transform, projection, base.union(accessory));
        assert!((projection.aspect_ratio - 1600.0 / 900.0).abs() < f32::EPSILON);
    }

    #[test]
    fn same_generation_does_not_follow_live_head_motion() {
        let mut app = frame_app(UVec2::new(1600, 900));
        let camera = camera_entity(&mut app);
        let (root, head, hips) = spawn_humanoid_parts(&mut app, 0.0);
        make_ready(&mut app, root);
        app.update();
        let first = *app
            .world()
            .get::<Transform>(camera)
            .expect("camera transform");

        app.world_mut()
            .entity_mut(head)
            .insert(GlobalTransform::from_translation(Vec3::new(2.0, 2.4, 0.0)));
        app.world_mut()
            .entity_mut(hips)
            .insert(GlobalTransform::from_translation(Vec3::new(2.0, 1.0, 0.0)));
        app.update();

        assert_eq!(
            app.world()
                .get::<Transform>(camera)
                .expect("camera transform"),
            &first
        );
    }

    #[test]
    fn zero_viewport_retries_after_viewport_becomes_available() {
        let mut app = frame_app(UVec2::ZERO);
        let camera = camera_entity(&mut app);
        let root = spawn_humanoid(&mut app, 0.0);
        make_ready(&mut app, root);
        let before = *app
            .world()
            .get::<Transform>(camera)
            .expect("camera transform");
        app.update();
        assert_eq!(
            app.world()
                .get::<Transform>(camera)
                .expect("camera transform"),
            &before
        );

        app.world_mut()
            .get_mut::<Camera>(camera)
            .expect("avatar camera")
            .viewport = Some(Viewport {
            physical_size: UVec2::new(900, 1600),
            ..default()
        });
        app.update();
        assert_ne!(
            app.world()
                .get::<Transform>(camera)
                .expect("camera transform"),
            &before
        );
    }

    #[test]
    fn invalid_or_empty_subtree_uses_finite_head_hips_fallback() {
        let mut empty = frame_app(UVec2::new(1600, 900));
        let empty_camera = camera_entity(&mut empty);
        let empty_root = spawn_humanoid(&mut empty, 0.0);
        make_ready(&mut empty, empty_root);
        empty.update();
        assert!(
            empty
                .world()
                .get::<Transform>(empty_camera)
                .unwrap()
                .translation
                .is_finite()
        );

        let mut invalid = frame_app(UVec2::new(1600, 900));
        let invalid_camera = camera_entity(&mut invalid);
        let (invalid_root, head, _) = spawn_humanoid_parts(&mut invalid, 0.0);
        let invalid_mesh = mesh_asset(
            &mut invalid,
            &[[f32::NAN, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        );
        spawn_renderable(&mut invalid, head, invalid_mesh, GlobalTransform::IDENTITY);
        make_ready(&mut invalid, invalid_root);
        invalid.update();
        let transform = invalid.world().get::<Transform>(invalid_camera).unwrap();
        assert!(transform.translation.is_finite());
        let Projection::Perspective(projection) =
            invalid.world().get::<Projection>(invalid_camera).unwrap()
        else {
            panic!("avatar viewport camera must use perspective projection");
        };
        assert!((projection.fov - FIXED_VERTICAL_FOV).abs() < f32::EPSILON);
    }

    #[test]
    fn portrait_and_landscape_viewports_use_their_actual_aspect_ratio() {
        fn run_fit(viewport: UVec2) -> (f32, f32) {
            let mut app = frame_app(viewport);
            let camera = camera_entity(&mut app);
            let (root, head, _) = spawn_humanoid_parts(&mut app, 0.0);
            let mesh = mesh_asset(&mut app, &cube_positions());
            spawn_renderable(
                &mut app,
                head,
                mesh,
                GlobalTransform::from_translation(Vec3::new(4.0, 0.0, 0.0)),
            );
            make_ready(&mut app, root);
            app.update();
            let transform = app.world().get::<Transform>(camera).unwrap();
            let Projection::Perspective(projection) =
                app.world().get::<Projection>(camera).unwrap()
            else {
                panic!("avatar viewport camera must use perspective projection");
            };
            (transform.translation.z, projection.aspect_ratio)
        }

        let (portrait_distance, portrait_aspect) = run_fit(UVec2::new(900, 1600));
        let (landscape_distance, landscape_aspect) = run_fit(UVec2::new(1600, 900));
        assert!((portrait_aspect - 900.0 / 1600.0).abs() < f32::EPSILON);
        assert!((landscape_aspect - 1600.0 / 900.0).abs() < f32::EPSILON);
        assert!(portrait_distance > landscape_distance);
    }
}
