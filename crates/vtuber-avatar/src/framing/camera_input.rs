//! Main viewport mouse input routing for camera controls.

use bevy::ecs::system::SystemParam;
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit};
use bevy::prelude::*;
use bevy::window::{PrimaryWindow, Window};
use bevy_vrm1::prelude::{HeadBoneEntity, HipsBoneEntity};

use super::camera_control::{AvatarCameraControl, CameraControlGeometryError, geometry};
use crate::lifecycle::{AvatarGeneration, AvatarLifecycle, AvatarLifecycleState};

/// System set for main viewport input before transform propagation.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CameraInputSet;

/// Deterministic mouse gesture ownership for the main viewport.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CameraPointerGesture {
    /// No camera drag owns the pointer.
    #[default]
    None,
    /// The left-button gesture owns the pointer for this generation.
    Orbit {
        /// Avatar generation captured at gesture start.
        generation: AvatarGeneration,
    },
    /// The right-button gesture owns the pointer for this generation.
    Pan {
        /// Avatar generation captured at gesture start.
        generation: AvatarGeneration,
    },
}

/// Converts Bevy's line/pixel scroll units into one normalized dolly input.
///
/// Horizontal scroll is intentionally ignored by the caller. Bevy's official
/// `MouseScrollUnit::SCROLL_UNIT_CONVERSION_FACTOR` is used for pixel input so
/// line and high-resolution trackpad input share the same scale.
pub fn normalized_vertical_scroll(
    scroll: Vec2,
    unit: MouseScrollUnit,
) -> Result<f32, CameraControlGeometryError> {
    if !scroll.is_finite() {
        return Err(CameraControlGeometryError::NonFiniteInput);
    }
    let normalized = match unit {
        MouseScrollUnit::Line => scroll.y,
        MouseScrollUnit::Pixel => scroll.y / MouseScrollUnit::SCROLL_UNIT_CONVERSION_FACTOR,
    };
    if normalized.is_finite() {
        Ok(normalized)
    } else {
        Err(CameraControlGeometryError::NonFiniteInput)
    }
}

/// Applies captured orbit/pan and background-only wheel dolly to the viewport.
///
/// The system reads only `Camera` and `Transform`; perspective projection and
/// FOV are deliberately absent from the query. Mouse deltas are already frame
/// accumulations, so no delta-time factor is applied. Orbit re-anchors its
/// pivot on the avatar's live position every frame, so left-drag rotation
/// stays centered on the avatar even after pan or tracking moved it.
#[derive(SystemParam)]
pub(crate) struct CameraInputWorld<'w, 's> {
    gate: Res<'w, super::camera_control::CameraPointerInputGate>,
    lifecycle: Res<'w, AvatarLifecycle>,
    windows: Query<'w, 's, &'static Window, With<PrimaryWindow>>,
    cameras:
        Query<'w, 's, (&'static Camera, &'static mut Transform), With<super::AvatarViewportCamera>>,
    roots: Query<'w, 's, (&'static HeadBoneEntity, &'static HipsBoneEntity)>,
    bones: Query<'w, 's, &'static GlobalTransform>,
}

/// The avatar's live upper-body anchor, or `None` while bones are unbound.
fn avatar_focus(input: &CameraInputWorld) -> Option<Vec3> {
    let root = input.lifecycle.active_root()?;
    let (head_entity, hips_entity) = input.roots.get(root).ok()?;
    let head = input.bones.get(**head_entity).ok()?.translation();
    let hips = input.bones.get(**hips_entity).ok()?.translation();
    super::avatar_focus_point(head, hips)
}

pub(crate) fn apply_camera_pointer_input(
    mut input: CameraInputWorld,
    mut gesture: ResMut<CameraPointerGesture>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    mouse_scroll: Res<AccumulatedMouseScroll>,
    mut camera_control: ResMut<AvatarCameraControl>,
) {
    let Some(generation) = camera_control.active_generation() else {
        *gesture = CameraPointerGesture::None;
        return;
    };
    if input.lifecycle.state() != AvatarLifecycleState::Ready
        || input.lifecycle.current_generation() != generation
    {
        *gesture = CameraPointerGesture::None;
        return;
    }
    if input
        .windows
        .iter()
        .next()
        .is_some_and(|window| !window.focused)
    {
        *gesture = CameraPointerGesture::None;
        return;
    }

    let focus = avatar_focus(&input);
    let Some((camera, mut transform)) = input.cameras.iter_mut().next() else {
        return;
    };
    let Some(current) = camera_control.current_for(generation) else {
        *gesture = CameraPointerGesture::None;
        return;
    };

    let active_gesture = match *gesture {
        CameraPointerGesture::None if input.gate.allows_camera_input() => {
            // Left wins if both buttons arrive in one frame. Once selected,
            // the mode cannot switch until its corresponding release.
            if mouse_buttons.just_pressed(MouseButton::Left) {
                *gesture = CameraPointerGesture::Orbit { generation };
            } else if mouse_buttons.just_pressed(MouseButton::Right) {
                *gesture = CameraPointerGesture::Pan { generation };
            }
            *gesture
        }
        existing => existing,
    };

    let mut next = current;
    match active_gesture {
        CameraPointerGesture::Orbit {
            generation: captured,
        } if captured == generation => {
            // Re-anchor the pivot on the avatar's live position every frame
            // so orbit revolves around the avatar instead of the stale panned
            // target. The camera itself only moves once an orbit delta is
            // applied, so a motionless press never jumps the view.
            if let Some(focus) = focus
                && let Ok(reanchored) = geometry::retarget(next, focus)
            {
                next = reanchored;
            }
            let sensitivity = camera_control.config().orbit_radians_per_pixel;
            if sensitivity.is_finite() && sensitivity > 0.0 && mouse_motion.delta.is_finite() {
                // Positive screen Y is downward, so dragging downward raises
                // the orbit camera and produces a positive pitch delta.
                if let Ok(candidate) = geometry::orbit(
                    next,
                    mouse_motion.delta.x * sensitivity,
                    mouse_motion.delta.y * sensitivity,
                ) {
                    next = candidate;
                }
            }
        }
        CameraPointerGesture::Pan {
            generation: captured,
        } if captured == generation => {
            if let Some(viewport_size) = camera.logical_viewport_size()
                && let Ok(candidate) = geometry::pan(next, mouse_motion.delta, viewport_size)
            {
                next = candidate;
            }
        }
        CameraPointerGesture::None => {}
        CameraPointerGesture::Orbit { .. } | CameraPointerGesture::Pan { .. } => {
            *gesture = CameraPointerGesture::None;
            return;
        }
    }

    if input.gate.allows_camera_input()
        && let Ok(scroll) = normalized_vertical_scroll(mouse_scroll.delta, mouse_scroll.unit)
        && scroll != 0.0
        && let Ok(candidate) = geometry::dolly(next, scroll, camera_control.config())
    {
        next = candidate;
    }

    if next != current {
        *transform = next.transform();
        if !camera_control.set_current(generation, next) {
            *gesture = CameraPointerGesture::None;
            return;
        }
    }

    match active_gesture {
        CameraPointerGesture::Orbit { .. } if mouse_buttons.just_released(MouseButton::Left) => {
            *gesture = CameraPointerGesture::None;
        }
        CameraPointerGesture::Pan { .. } if mouse_buttons.just_released(MouseButton::Right) => {
            *gesture = CameraPointerGesture::None;
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framing::camera_control::{
        AvatarCameraControlState, CameraControlPose, CameraPointerInputGate, FIXED_VERTICAL_FOV,
    };

    fn ready_app() -> (App, Entity, AvatarGeneration) {
        let mut app = App::new();
        app.init_resource::<CameraPointerInputGate>()
            .init_resource::<CameraPointerGesture>()
            .init_resource::<AvatarLifecycle>()
            .init_resource::<AvatarCameraControl>()
            .init_resource::<AccumulatedMouseMotion>()
            .init_resource::<AccumulatedMouseScroll>()
            .init_resource::<ButtonInput<MouseButton>>()
            .add_systems(Update, apply_camera_pointer_input);

        let camera_transform = Transform::from_translation(Vec3::new(0.0, 1.0, 5.0))
            .looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y);
        let camera = app
            .world_mut()
            .spawn((
                Camera3d::default(),
                Camera {
                    viewport: Some(bevy::camera::Viewport {
                        physical_size: UVec2::new(1600, 900),
                        ..default()
                    }),
                    ..default()
                },
                Projection::Perspective(PerspectiveProjection {
                    fov: FIXED_VERTICAL_FOV,
                    ..default()
                }),
                super::super::AvatarViewportCamera::from_default_transform(camera_transform),
                camera_transform,
            ))
            .id();
        let root = app.world_mut().spawn_empty().id();
        let generation = {
            let mut lifecycle = app.world_mut().resource_mut::<AvatarLifecycle>();
            lifecycle.request_load(root).expect("test load is valid");
            lifecycle.start_binding(root);
            lifecycle.finish_ready();
            lifecycle.current_generation()
        };
        let transform = *app
            .world()
            .get::<Transform>(camera)
            .expect("camera transform");
        let pose = CameraControlPose::new(transform, Vec3::new(0.0, 1.0, 0.0))
            .expect("test pose is valid");
        app.world_mut()
            .resource_mut::<AvatarCameraControl>()
            .initialize(generation, pose);
        (app, camera, generation)
    }

    fn press(app: &mut App, button: MouseButton) {
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .press(button);
    }

    fn release(app: &mut App, button: MouseButton) {
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .release(button);
    }

    fn clear_buttons(app: &mut App) {
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .clear();
    }

    fn pose_after_orbit_drag(delta: Vec2) -> CameraControlPose {
        let (mut app, _, generation) = ready_app();
        press(&mut app, MouseButton::Left);
        app.world_mut()
            .resource_mut::<AccumulatedMouseMotion>()
            .delta = delta;
        app.update();
        app.world()
            .resource::<AvatarCameraControl>()
            .current_for(generation)
            .expect("orbit drag updates the current pose")
    }

    fn ready_app_with_avatar(
        head: Vec3,
        hips: Vec3,
    ) -> (App, Entity, AvatarGeneration, Entity, Entity) {
        let (mut app, camera, generation) = ready_app();
        let root = app
            .world()
            .resource::<AvatarLifecycle>()
            .active_root()
            .expect("ready avatar has a root");
        let head_entity = app
            .world_mut()
            .spawn(GlobalTransform::from_translation(head))
            .id();
        let hips_entity = app
            .world_mut()
            .spawn(GlobalTransform::from_translation(hips))
            .id();
        app.world_mut()
            .entity_mut(root)
            .insert((HeadBoneEntity(head_entity), HipsBoneEntity(hips_entity)));
        (app, camera, generation, head_entity, hips_entity)
    }

    fn focus_ndc(app: &App, camera: Entity, focus: Vec3) -> Vec2 {
        let transform = app
            .world()
            .get::<Transform>(camera)
            .expect("camera transform");
        let camera_relative = transform.rotation.inverse() * (focus - transform.translation);
        let depth = -camera_relative.z;
        let tangent = (FIXED_VERTICAL_FOV * 0.5).tan();
        Vec2::new(
            camera_relative.x / (depth * tangent),
            camera_relative.y / (depth * tangent),
        )
    }

    #[test]
    fn line_and_pixel_scroll_normalize_to_the_same_physical_intent() {
        let line = normalized_vertical_scroll(Vec2::new(3.0, 2.0), MouseScrollUnit::Line)
            .expect("line scroll is valid");
        let pixel = normalized_vertical_scroll(Vec2::new(100.0, 200.0), MouseScrollUnit::Pixel)
            .expect("pixel scroll is valid");
        assert!((line - pixel).abs() < f32::EPSILON);
        assert_eq!(
            normalized_vertical_scroll(Vec2::new(f32::NAN, 0.0), MouseScrollUnit::Line),
            Err(CameraControlGeometryError::NonFiniteInput)
        );
    }

    #[test]
    fn background_left_drag_captures_orbit_and_release_clears_it() {
        let (mut app, camera, generation) = ready_app();
        press(&mut app, MouseButton::Left);
        app.world_mut()
            .resource_mut::<AccumulatedMouseMotion>()
            .delta = Vec2::new(20.0, -10.0);
        app.update();

        assert_eq!(
            *app.world().resource::<CameraPointerGesture>(),
            CameraPointerGesture::Orbit { generation }
        );
        assert_ne!(
            app.world().get::<Transform>(camera).unwrap().translation,
            Vec3::new(0.0, 1.0, 5.0)
        );

        clear_buttons(&mut app);
        release(&mut app, MouseButton::Left);
        app.update();
        assert_eq!(
            *app.world().resource::<CameraPointerGesture>(),
            CameraPointerGesture::None
        );
    }

    #[test]
    fn orbit_mouse_deltas_follow_standard_grab_direction_and_preserve_fov() {
        let canonical = CameraControlPose::new(
            Transform::from_translation(Vec3::new(0.0, 1.0, 5.0))
                .looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
            Vec3::new(0.0, 1.0, 0.0),
        )
        .expect("canonical pose is valid");
        let right = pose_after_orbit_drag(Vec2::new(20.0, 0.0));
        let left = pose_after_orbit_drag(Vec2::new(-20.0, 0.0));
        let down = pose_after_orbit_drag(Vec2::new(0.0, 20.0));
        let up = pose_after_orbit_drag(Vec2::new(0.0, -20.0));

        assert!(right.transform().translation.x < 0.0);
        assert!(left.transform().translation.x > 0.0);
        assert!(down.transform().translation.y > canonical.transform().translation.y);
        assert!(up.transform().translation.y < canonical.transform().translation.y);

        for pose in [right, left, down, up] {
            assert_eq!(pose.target(), canonical.target());
            assert!((pose.distance() - canonical.distance()).abs() < 1e-5);
            assert_eq!(pose.transform().scale, canonical.transform().scale);
        }
        assert!((FIXED_VERTICAL_FOV - 12.0_f32.to_radians()).abs() < f32::EPSILON);
    }

    #[test]
    fn simultaneous_buttons_choose_orbit_deterministically() {
        let (mut app, _, generation) = ready_app();
        press(&mut app, MouseButton::Left);
        press(&mut app, MouseButton::Right);
        app.update();

        assert_eq!(
            *app.world().resource::<CameraPointerGesture>(),
            CameraPointerGesture::Orbit { generation }
        );
    }

    #[test]
    fn orbit_after_pan_pivots_on_the_avatar_position() {
        let focus = Vec3::new(0.0, 1.48, 0.0);
        let (mut app, camera, generation, _, _) =
            ready_app_with_avatar(Vec3::new(0.0, 1.8, 0.0), Vec3::new(0.0, 1.0, 0.0));

        press(&mut app, MouseButton::Right);
        app.world_mut()
            .resource_mut::<AccumulatedMouseMotion>()
            .delta = Vec2::new(200.0, 0.0);
        app.update();
        clear_buttons(&mut app);
        release(&mut app, MouseButton::Right);
        app.update();
        let panned = app
            .world()
            .resource::<AvatarCameraControl>()
            .current_for(generation)
            .expect("panned pose");
        assert_ne!(panned.target(), focus);

        press(&mut app, MouseButton::Left);
        app.world_mut()
            .resource_mut::<AccumulatedMouseMotion>()
            .delta = Vec2::new(-40.0, 0.0);
        app.update();

        let orbited = app
            .world()
            .resource::<AvatarCameraControl>()
            .current_for(generation)
            .expect("orbited pose");
        assert_eq!(orbited.target(), focus);
        assert!(focus_ndc(&app, camera, focus).length() < 1e-4);
    }

    #[test]
    fn orbit_pivot_follows_live_avatar_motion() {
        let (mut app, camera, generation, head_entity, hips_entity) =
            ready_app_with_avatar(Vec3::new(0.0, 1.8, 0.0), Vec3::new(0.0, 1.0, 0.0));
        press(&mut app, MouseButton::Left);
        app.update();

        app.world_mut()
            .entity_mut(head_entity)
            .insert(GlobalTransform::from_translation(Vec3::new(2.0, 1.8, 0.0)));
        app.world_mut()
            .entity_mut(hips_entity)
            .insert(GlobalTransform::from_translation(Vec3::new(2.0, 1.0, 0.0)));
        app.world_mut()
            .resource_mut::<AccumulatedMouseMotion>()
            .delta = Vec2::new(-30.0, 0.0);
        app.update();

        let focus = Vec3::new(2.0, 1.48, 0.0);
        let orbited = app
            .world()
            .resource::<AvatarCameraControl>()
            .current_for(generation)
            .expect("orbited pose");
        assert_eq!(orbited.target(), focus);
        assert!(focus_ndc(&app, camera, focus).length() < 1e-4);
    }

    #[test]
    fn right_press_wins_only_when_left_is_absent_and_wheel_changes_distance() {
        let (mut app, _, generation) = ready_app();
        press(&mut app, MouseButton::Right);
        app.update();
        assert_eq!(
            *app.world().resource::<CameraPointerGesture>(),
            CameraPointerGesture::Pan { generation }
        );

        clear_buttons(&mut app);
        release(&mut app, MouseButton::Right);
        app.update();

        app.world_mut()
            .resource_mut::<AccumulatedMouseScroll>()
            .delta = Vec2::new(12.0, 1.0);
        let before = app
            .world()
            .resource::<AvatarCameraControl>()
            .current_for(generation)
            .expect("control pose");
        app.update();
        let after = app
            .world()
            .resource::<AvatarCameraControl>()
            .current_for(generation)
            .expect("control pose");
        assert!(after.distance() < before.distance());
        assert_eq!(after.transform().rotation, before.transform().rotation);
    }

    #[test]
    fn egui_owned_start_is_blocked_but_captured_drag_continues_over_ui() {
        let (mut app, camera, generation) = ready_app();
        app.world_mut()
            .resource_mut::<CameraPointerInputGate>()
            .set_egui_owns_pointer(true);
        press(&mut app, MouseButton::Left);
        app.world_mut()
            .resource_mut::<AccumulatedMouseMotion>()
            .delta = Vec2::new(20.0, 0.0);
        let before = app.world().get::<Transform>(camera).unwrap().translation;
        app.update();
        assert_eq!(
            *app.world().resource::<CameraPointerGesture>(),
            CameraPointerGesture::None
        );
        assert_eq!(
            app.world().get::<Transform>(camera).unwrap().translation,
            before
        );

        clear_buttons(&mut app);
        app.world_mut()
            .resource_mut::<CameraPointerInputGate>()
            .set_egui_owns_pointer(false);
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .reset_all();
        press(&mut app, MouseButton::Left);
        app.world_mut()
            .resource_mut::<AccumulatedMouseMotion>()
            .delta = Vec2::X;
        app.update();
        assert_eq!(
            *app.world().resource::<CameraPointerGesture>(),
            CameraPointerGesture::Orbit { generation }
        );

        clear_buttons(&mut app);
        app.world_mut()
            .resource_mut::<CameraPointerInputGate>()
            .set_egui_owns_pointer(true);
        app.world_mut()
            .resource_mut::<AccumulatedMouseMotion>()
            .delta = Vec2::new(10.0, 0.0);
        let during_ui = app.world().get::<Transform>(camera).unwrap().translation;
        app.update();
        assert_ne!(
            app.world().get::<Transform>(camera).unwrap().translation,
            during_ui
        );
        assert_eq!(
            *app.world().resource::<CameraPointerGesture>(),
            CameraPointerGesture::Orbit { generation }
        );
    }

    #[test]
    fn lifecycle_invalidation_clears_capture_without_changing_fov() {
        let (mut app, _, generation) = ready_app();
        press(&mut app, MouseButton::Left);
        app.update();
        assert_eq!(
            *app.world().resource::<CameraPointerGesture>(),
            CameraPointerGesture::Orbit { generation }
        );
        app.world_mut()
            .resource_mut::<AvatarCameraControl>()
            .invalidate();
        app.update();
        assert_eq!(
            *app.world().resource::<CameraPointerGesture>(),
            CameraPointerGesture::None
        );
        assert_eq!(
            app.world().resource::<AvatarCameraControl>().state(),
            AvatarCameraControlState::Unavailable
        );
        assert!((FIXED_VERTICAL_FOV - 12.0_f32.to_radians()).abs() < f32::EPSILON);
    }
}
