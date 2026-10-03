//! Standing contact: fixed ankle poses, two-bone legs and a small pelvis reaction.
//!
//! The authored standing pose supplies the sole height (the ankle is not the
//! sole). Keeping both the ankle position and orientation preserves that sole
//! contact without requiring a mesh-dependent ankle-to-floor offset.

use crate::skeleton::{self, TwoBoneRest, parent_global, refresh_subtree};
use bevy::app::AnimationSystems;
use bevy::ecs::world::EntityRef;
use bevy::prelude::*;
use bevy_vrm1::prelude::*;

// Artistic compliance, not an estimate of a physical ground-reaction force.
// Most of the tracked pelvis motion remains; the legs absorb the remainder.
const PELVIS_REACTION: f32 = 0.12;

#[derive(Clone, Copy)]
struct PlantedLeg {
    upper: Entity,
    lower: Entity,
    foot: Entity,
    contact: GlobalTransform,
    rest: TwoBoneRest,
}

#[derive(Component)]
pub(crate) struct GroundedFeet {
    hips: Entity,
    legs: [PlantedLeg; 2],
    forward: Vec3,
    // Restored before animation so other additive writers never adopt our
    // correction as their next frame's animation base.
    originals: Vec<(Entity, Transform)>,
}

impl GroundedFeet {
    pub(crate) fn from_rest(
        root: &EntityRef<'_>,
        rest: impl Fn(Entity) -> Option<GlobalTransform>,
    ) -> Option<Self> {
        let hips = root.get::<HipsBoneEntity>()?.0;
        let forward = root.get::<GlobalTransform>()?.rotation() * Vec3::Z;
        let leg = |upper, lower, foot| {
            let start = rest(upper)?;
            let middle = rest(lower)?;
            let end = rest(foot)?;
            let upper_direction = (middle.translation() - start.translation()).try_normalize()?;
            Some(PlantedLeg {
                upper,
                lower,
                foot,
                contact: end,
                rest: TwoBoneRest {
                    start: start.translation(),
                    middle: middle.translation(),
                    end: end.translation(),
                    lengths: Vec2::new(
                        start.translation().distance(middle.translation()),
                        middle.translation().distance(end.translation()),
                    ),
                    start_rotation: start.rotation(),
                    middle_rotation: middle.rotation(),
                    hinge_axis: skeleton::rest_hinge_axis(
                        middle.translation() - start.translation(),
                        end.translation() - middle.translation(),
                        Some(forward.cross(upper_direction)),
                    )?,
                    // Flexion extent of the OpenSim Gait2392 knee coordinate.
                    flexion_limit: 120.0_f32.to_radians(),
                    axial_limit: 0.0,
                },
            })
        };
        Some(Self {
            hips,
            legs: [
                leg(
                    root.get::<LeftUpperLegBoneEntity>()?.0,
                    root.get::<LeftLowerLegBoneEntity>()?.0,
                    root.get::<LeftFootBoneEntity>()?.0,
                )?,
                leg(
                    root.get::<RightUpperLegBoneEntity>()?.0,
                    root.get::<RightLowerLegBoneEntity>()?.0,
                    root.get::<RightFootBoneEntity>()?.0,
                )?,
            ],
            forward,
            originals: Vec::with_capacity(7),
        })
    }
}

pub(crate) fn register_grounding(app: &mut App) {
    app.add_systems(PostUpdate, restore_ungrounded_pose.before(AnimationSystems))
        .add_systems(
            PostUpdate,
            plant_feet
                .after(crate::direct_position::apply_direct_body_position)
                .before(crate::arm_pipeline::update_dynamic_arm_targets)
                .before(crate::arm_pipeline::update_tracked_arm_targets)
                .before(crate::arm_pose::apply_default_arm_pose)
                .before(crate::gaze::update_direct_look_at_input)
                .before(VrmSystemSets::GazeControl)
                .before(VrmSystemSets::Constraints)
                .before(TransformSystems::Propagate),
        );
}

fn restore_ungrounded_pose(
    mut roots: Query<&mut GroundedFeet>,
    mut transforms: Query<&mut Transform>,
) {
    for mut feet in &mut roots {
        for (entity, original) in feet.originals.drain(..) {
            if let Ok(mut transform) = transforms.get_mut(entity) {
                *transform = original;
            }
        }
    }
}

fn plant_feet(
    mut roots: Query<(Entity, &mut GroundedFeet)>,
    mut transforms: Query<(&mut Transform, &mut GlobalTransform)>,
    parents: Query<&ChildOf>,
    children: Query<&Children>,
) {
    for (root, mut feet) in &mut roots {
        if apply_contact(&mut feet, &mut transforms, &parents).is_none() {
            warn!("standing foot contact could not be solved for avatar {root:?}");
        }
        if let Some(global) = skeleton::current_global(root, &transforms, &parents, None) {
            refresh_subtree(root, global, &mut transforms, &children);
        }
    }
}

fn apply_contact(
    feet: &mut GroundedFeet,
    transforms: &mut Query<(&mut Transform, &mut GlobalTransform)>,
    parents: &Query<&ChildOf>,
) -> Option<()> {
    // Save only the seven bones this pass owns. No upper-body local rotations
    // are rewritten; the pelvis reaction reaches them through the hierarchy.
    for entity in std::iter::once(feet.hips).chain(
        feet.legs
            .iter()
            .flat_map(|leg| [leg.upper, leg.lower, leg.foot]),
    ) {
        feet.originals
            .push((entity, *transforms.get(entity).ok()?.0));
    }
    let mut translation_error = Vec3::ZERO;
    let mut rotation_error = Quat::IDENTITY;
    for (index, leg) in feet.legs.iter().enumerate() {
        let free = skeleton::current_global(leg.foot, transforms, parents, None)?;
        translation_error += (leg.contact.translation() - free.translation()) * 0.5;
        let error = leg.contact.rotation() * free.rotation().inverse();
        rotation_error = rotation_error.slerp(error, 1.0 / (index + 1) as f32);
    }
    let hips = skeleton::current_global(feet.hips, transforms, parents, None)?;
    set_world_rotation(
        feet.hips,
        Quat::IDENTITY.slerp(rotation_error, PELVIS_REACTION) * hips.rotation(),
        transforms,
        parents,
    )?;
    translate_world(
        feet.hips,
        translation_error * PELVIS_REACTION,
        transforms,
        parents,
    )?;

    // A straight leg cannot reach an ankle below its maximum extension.
    // Lower the shared pelvis just enough for BOTH planted feet to be reached,
    // rather than stretching bones or letting the higher foot float.
    let mut lowering = 0.0_f32;
    for leg in feet.legs {
        let upper = skeleton::current_global(leg.upper, transforms, parents, None)?.translation();
        let reach = leg.rest.lengths.element_sum();
        let offset = upper - leg.contact.translation();
        let height_squared = reach * reach - offset.x * offset.x - offset.z * offset.z;
        if height_squared < 0.0 {
            return None;
        }
        lowering = lowering.min(height_squared.sqrt() - offset.y);
    }
    translate_world(feet.hips, Vec3::Y * lowering, transforms, parents)?;

    for leg in feet.legs {
        let upper = skeleton::current_global(leg.upper, transforms, parents, None)?;
        // Carry the immutable rest geometry into the current upper-joint
        // frame. The solver owns knee flexion; independent rotation arcs
        // must not introduce a second axis at the knee.
        let carry = upper.rotation() * leg.rest.start_rotation.inverse();
        let origin = upper.translation();
        let rest = TwoBoneRest {
            start: origin,
            middle: origin + carry * (leg.rest.middle - leg.rest.start),
            end: origin + carry * (leg.rest.end - leg.rest.start),
            start_rotation: upper.rotation(),
            middle_rotation: carry * leg.rest.middle_rotation,
            hinge_axis: carry * leg.rest.hinge_axis,
            ..leg.rest
        };
        let solved =
            skeleton::solve_two_bone(rest, leg.contact.translation(), origin + feet.forward, 0.0)
                .ok()?;
        set_world_rotation(leg.upper, solved.start_rotation, transforms, parents)?;
        set_world_rotation(leg.lower, solved.middle_rotation, transforms, parents)?;
        set_world_rotation(leg.foot, leg.contact.rotation(), transforms, parents)?;
    }
    Some(())
}

fn set_world_rotation(
    entity: Entity,
    rotation: Quat,
    transforms: &mut Query<(&mut Transform, &mut GlobalTransform)>,
    parents: &Query<&ChildOf>,
) -> Option<()> {
    let parent = parent_global(entity, transforms, parents)?;
    transforms.get_mut(entity).ok()?.0.rotation =
        (parent.rotation().inverse() * rotation).normalize();
    Some(())
}

fn translate_world(
    entity: Entity,
    offset: Vec3,
    transforms: &mut Query<(&mut Transform, &mut GlobalTransform)>,
    parents: &Query<&ChildOf>,
) -> Option<()> {
    let parent = parent_global(entity, transforms, parents)?;
    transforms.get_mut(entity).ok()?.0.translation +=
        parent.affine().inverse().transform_vector3(offset);
    Some(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::direct_pose::{BodyTrackingPoseInput, register_direct_pose};
    use crate::direct_position::{BodyTrackingPositionInput, register_direct_position};

    fn bone(app: &mut App, parent: Entity, local: Transform) -> Entity {
        let global = app
            .world()
            .get::<GlobalTransform>(parent)
            .unwrap()
            .mul_transform(local);
        app.world_mut()
            .spawn((
                local,
                global,
                RestTransform(local),
                RestGlobalTransform(global),
                ChildOf(parent),
            ))
            .id()
    }

    fn standing_app(yaw: f32, scale: f32, rest_flexion: f32) -> (App, Entity, Entity, [Entity; 2]) {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, TransformPlugin));
        app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
            std::time::Duration::from_secs_f32(1.0 / 60.0),
        ));
        register_direct_pose(&mut app);
        register_direct_position(&mut app);
        register_grounding(&mut app);
        let placement = Transform::from_xyz(1.0, 2.0, -3.0)
            .with_rotation(Quat::from_rotation_y(yaw))
            .with_scale(Vec3::splat(scale));
        let root = app
            .world_mut()
            .spawn((
                Vrm,
                placement,
                GlobalTransform::from(placement),
                BodyTracking::default(),
                BodyTrackingPoseInput::default(),
                crate::pose::natural_body_tracking_profile(),
                BodyTrackingPositionInput::default(),
            ))
            .id();
        let hips = bone(&mut app, root, Transform::from_xyz(0.0, 1.0, 0.0));
        let spine = bone(&mut app, hips, Transform::from_xyz(0.0, 0.25, 0.0));
        let head = bone(&mut app, spine, Transform::from_xyz(0.0, 0.45, 0.0));
        let mut legs = Vec::new();
        let mut toes = Vec::new();
        for x in [-0.1, 0.1] {
            let upper = bone(
                &mut app,
                hips,
                Transform::from_xyz(x, 0.0, 0.0).with_rotation(Quat::from_rotation_y(0.3)),
            );
            // A non-humanoid parent between the two named leg bones.
            let helper = bone(
                &mut app,
                upper,
                Transform::from_rotation(Quat::from_rotation_y(-0.2)),
            );
            let lower = bone(
                &mut app,
                helper,
                Transform::from_xyz(0.0, -0.45, 0.0).with_rotation(
                    Quat::from_rotation_y(-0.1) * Quat::from_rotation_x(rest_flexion),
                ),
            );
            let foot = bone(&mut app, lower, Transform::from_xyz(0.0, -0.45, 0.0));
            let toe = bone(&mut app, foot, Transform::from_xyz(0.0, -0.1, 0.15));
            legs.push((upper, lower, foot));
            toes.push(toe);
        }
        app.world_mut().entity_mut(root).insert((
            HipsBoneEntity(hips),
            SpineBoneEntity(spine),
            HeadBoneEntity(head),
            LeftUpperLegBoneEntity(legs[0].0),
            LeftLowerLegBoneEntity(legs[0].1),
            LeftFootBoneEntity(legs[0].2),
            RightUpperLegBoneEntity(legs[1].0),
            RightLowerLegBoneEntity(legs[1].1),
            RightFootBoneEntity(legs[1].2),
        ));
        let feet = GroundedFeet::from_rest(&app.world().entity(root), |entity| {
            app.world()
                .get::<RestGlobalTransform>(entity)
                .map(|rest| rest.0)
        })
        .unwrap();
        app.world_mut().entity_mut(root).insert(feet);
        (app, root, hips, [toes[0], toes[1]])
    }

    #[test]
    fn standing_contacts_survive_tracking_translation_and_return_without_drift() {
        for (yaw, scale, rest_flexion) in [
            (0.0, 1.0, 0.0),
            (std::f32::consts::PI, 0.75, 0.0),
            (0.6, 1.4, 0.0),
            (0.0, 1.0, 0.2),
            (0.6, 1.4, 0.2),
        ] {
            let (mut app, root, hips, toes) = standing_app(yaw, scale, rest_flexion);
            for frame in 0..360 {
                let phase = if frame < 180 {
                    frame as f32 / 30.0
                } else {
                    0.0
                };
                app.world_mut().entity_mut(root).insert((
                    BodyTrackingPoseInput {
                        yaw_radians: phase.sin() * 0.9,
                        pitch_radians: phase.cos() * 0.3,
                        roll_radians: phase.sin() * 0.3,
                        active: frame < 180,
                        weight: 1.0,
                        ..Default::default()
                    },
                    BodyTrackingPositionInput {
                        head_offset: Vec3::new(phase.sin() * 0.03, 0.0, 0.02),
                        body_offset: Vec3::new(phase.sin() * 0.1, phase.cos() * 0.05, 0.08),
                        active: frame < 180,
                        weight: 1.0,
                        ..Default::default()
                    },
                ));
                app.update();
                let feet = app.world().get::<GroundedFeet>(root).unwrap();
                for entity in feet.legs.iter().map(|leg| leg.foot).chain(toes) {
                    let actual = app.world().get::<GlobalTransform>(entity).unwrap();
                    let rest = app.world().get::<RestGlobalTransform>(entity).unwrap();
                    assert!(
                        actual.translation().distance(rest.translation()) < 2.0e-4,
                        "contact drift at frame {frame}: {:?} != {:?}",
                        actual.translation(),
                        rest.translation()
                    );
                    assert!(actual.rotation().dot(rest.rotation()).abs() > 0.99999);
                }
                // Bone lengths/local offsets are never stretched to fake contact.
                for leg in feet.legs {
                    let upper = app.world().get::<GlobalTransform>(leg.upper).unwrap();
                    let lower = app.world().get::<GlobalTransform>(leg.lower).unwrap();
                    let foot = app.world().get::<GlobalTransform>(leg.foot).unwrap();
                    assert!(
                        (upper.translation().distance(lower.translation()) - leg.rest.lengths.x)
                            .abs()
                            < 2.0e-5
                    );
                    assert!(
                        (lower.translation().distance(foot.translation()) - leg.rest.lengths.y)
                            .abs()
                            < 2.0e-5
                    );
                    let knee_delta = app
                        .world()
                        .get::<RestTransform>(leg.lower)
                        .unwrap()
                        .rotation
                        .inverse()
                        * app.world().get::<Transform>(leg.lower).unwrap().rotation;
                    let knee_axis = leg.rest.middle_rotation.inverse() * leg.rest.hinge_axis;
                    assert!(
                        (knee_delta * knee_axis).distance(knee_axis) < 2.0e-5,
                        "knee acquired another axis at frame {frame}"
                    );
                    for entity in [leg.upper, leg.lower, leg.foot] {
                        assert_eq!(
                            app.world().get::<Transform>(entity).unwrap().translation,
                            app.world()
                                .get::<RestTransform>(entity)
                                .unwrap()
                                .translation
                        );
                    }
                }
            }
            let hips_now = app.world().get::<Transform>(hips).unwrap();
            let hips_rest = app.world().get::<RestTransform>(hips).unwrap();
            assert!(hips_now.translation.distance(hips_rest.translation) < 1.0e-4);
            assert!(hips_now.rotation.dot(hips_rest.rotation).abs() > 0.99999);
        }
    }

    #[test]
    fn contact_returns_a_weak_reaction_to_the_torso() {
        let (mut app, root, hips, _) = standing_app(0.0, 1.0, 0.0);
        let spine = app.world().get::<SpineBoneEntity>(root).unwrap().0;
        let ungrounded = Quat::from_rotation_z(0.12);
        app.world_mut().get_mut::<Transform>(hips).unwrap().rotation = ungrounded;
        let ungrounded_spine = GlobalTransform::from(*app.world().get::<Transform>(root).unwrap())
            .mul_transform(*app.world().get::<Transform>(hips).unwrap())
            .mul_transform(*app.world().get::<Transform>(spine).unwrap());
        app.update();
        let actual = app.world().get::<Transform>(hips).unwrap().rotation;
        let correction = actual.angle_between(ungrounded);
        assert!(correction > 0.005 && correction < 0.03, "{correction}");
        assert!(actual.angle_between(Quat::IDENTITY) > 0.08);
        let spine_now = app.world().get::<GlobalTransform>(spine).unwrap();
        assert!(
            spine_now
                .rotation()
                .angle_between(ungrounded_spine.rotation())
                > 0.005
        );
        assert!(
            spine_now
                .translation()
                .distance(ungrounded_spine.translation())
                > 0.001
        );
    }
}
