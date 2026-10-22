//! Sensor-to-render regression: filtering alone cannot certify elbow stability.
#![allow(clippy::unwrap_used, clippy::indexing_slicing)]
use super::tests::{rig, update_after_worker};
use super::*;
use vtuber_core::arm_tracking::{
    ArmLandmarks, HandWorldLandmarks, PoseArmFrame, PoseArmObservation, PoseWorldLandmark,
};
use vtuber_tracking::arm_tracking::{ArmTrackingProfile, ArmTrackingState, step_arm_tracking};

#[test]
fn stationary_camera_noise_at_30_hz() {
    stationary_camera_noise(30, false);
}

#[test]
fn stationary_camera_noise_at_60_hz() {
    stationary_camera_noise(60, false);
}

#[test]
fn stationary_camera_noise_at_120_hz() {
    stationary_camera_noise(120, false);
}

#[test]
fn stationary_camera_noise_with_head_and_torso_collision() {
    stationary_camera_noise(60, true);
}

#[test]
fn mixed_landmark_noise_does_not_gain_energy_in_the_final_bones() {
    // Same metric on both sides: squared displacement of elbow AND wrist in
    // meters, relative to the settled pose. Angular gain near an extended arm
    // is not bounded by a Cartesian gain of one; flexion steps are checked
    // separately below. This is a regression for this feasible bent posture,
    // not a global contraction claim about constrained inverse kinematics.
    for phase_offset in [0.0_f32, std::f32::consts::FRAC_PI_2, std::f32::consts::PI] {
        let noise: Vec<_> = (0..150)
            .map(|sample| {
                std::array::from_fn(|joint| {
                    std::array::from_fn(|axis| {
                        if sample >= 120 {
                            return 0.0; // Include a second of the settling tail.
                        }
                        let phase = phase_offset * (joint * 3 + axis) as f32;
                        [(0.5, 0.4), (3.0, 0.35), (7.0, 0.25)]
                            .into_iter()
                            .map(|(hz, amplitude)| {
                                amplitude
                                    * (std::f32::consts::TAU * hz * sample as f32 / 30.0 + phase)
                                        .sin()
                            })
                            .sum::<f32>()
                            * 0.01
                    })
                })
            })
            .collect();
        for fps in [30, 60, 120] {
            let (raw, rendered) = arm_noise_response(fps, true, &noise);
            assert!(
                rendered <= raw,
                "{fps} Hz, phase={phase_offset}: final Cartesian energy gain={}",
                rendered / raw
            );
            assert!(
                rendered > raw * 0.001,
                "tracking must still follow the motion"
            );
        }
    }
}

fn stationary_camera_noise(fps: u32, with_body: bool) {
    let noise: Vec<[[f32; 3]; 2]> =
        serde_json::from_str(include_str!("../tests/fixtures/stationary_arm_noise.json")).unwrap();
    arm_noise_response(fps, with_body, &noise);
}

fn arm_noise_response(fps: u32, with_body: bool, noise: &[[[f32; 3]; 2]]) -> (f64, f64) {
    let (mut app, root, left, right) = rig();
    if with_body {
        use crate::collision::{CapsuleCollider, CollisionGeometry, Region};
        use bevy_vrm1::prelude::RestTransform;
        let mut spawn = |height| {
            let local = Transform::from_translation(Vec3::Y * height);
            app.world_mut()
                .spawn((
                    local,
                    GlobalTransform::from(local),
                    RestTransform(local),
                    RestGlobalTransform(GlobalTransform::from(local)),
                    ChildOf(root),
                ))
                .id()
        };
        let chest = spawn(1.2);
        let head = spawn(1.6);
        let mut binding = app.world_mut().get_mut::<AvatarBinding>(root).unwrap();
        binding.chest = Some(chest);
        binding.head = head;
        let mut capsules = vec![
            CapsuleCollider {
                bone: chest,
                region: Region::Torso,
                endpoints: [Vec3::Y, Vec3::Y * 1.4],
                radius: 0.10,
            },
            CapsuleCollider {
                bone: head,
                region: Region::Head,
                endpoints: [Vec3::Y * 1.7; 2],
                radius: 0.11,
            },
        ];
        for chain in [left, right] {
            capsules.extend([
                CapsuleCollider {
                    bone: chain.upper_arm,
                    region: Region::Upper(chain.side),
                    endpoints: [chain.rest.upper_arm.position, chain.rest.elbow.position],
                    radius: 0.035,
                },
                CapsuleCollider {
                    bone: chain.lower_arm,
                    region: Region::Forearm(chain.side),
                    endpoints: [chain.rest.elbow.position, chain.rest.wrist.position],
                    radius: 0.03,
                },
            ]);
        }
        app.world_mut().get_mut::<AvatarCollision>(root).unwrap().0 =
            Ok(std::sync::Arc::new(CollisionGeometry::new(capsules, &[])));
    }
    app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
        std::time::Duration::from_secs_f64(1.0 / f64::from(fps)),
    ));
    let mut mirror = crate::mirror::AvatarMotionMirror::default();
    let mirrored = fps == 60;
    if !mirrored {
        mirror.toggle();
    }
    app.insert_resource(mirror);
    let (displayed, side) = if mirrored { (right, 1) } else { (left, 0) };
    app.world_mut().resource_mut::<ArmSourceSelection>().mode = ArmPoseSourceKind::TrackedPose;
    let generation = app.world().get::<AvatarBinding>(root).unwrap().generation;
    let wanted = crate::upper_limb::tests::state(0.86, 0.97, 0.18, 1.94);
    let pose = wanted.forward(&left).unwrap();
    let point = |v: Vec3| PoseWorldLandmark {
        meters: [v.x, -v.y, -v.z],
        visibility: Some(1.0),
        presence: Some(1.0),
    };
    let arm = ArmLandmarks {
        shoulder: point(pose.shoulder),
        elbow: point(pose.elbow),
        wrist: point(pose.wrist),
        // Detection presence admits the arm. A degenerate synthetic hand
        // supplies no palm/finger goal, isolating positional sensor noise.
        hand: Some(HandWorldLandmarks {
            landmarks: [point(Vec3::ZERO); 21],
            handedness_score: None,
        }),
    };
    let missing = ArmLandmarks { hand: None, ..arm };
    let profile = ArmTrackingProfile::default();
    let mut tracking = ArmTrackingState::new();
    let mut previous: Option<(Vec3, f32)> = None;
    let mut largest_elbow_step = 0.0_f32;
    let mut largest_flexion_step = 0.0_f32;
    let mut elbow_positions = Vec::new();
    let mut settled_endpoints = None;
    let mut input_energy = 0.0_f64;
    let mut output_energy = 0.0_f64;
    for tick in 0..fps * 8 {
        let sample = (tick / (fps / 30)).saturating_sub(90) as usize;
        let mut observed = arm;
        if tick >= fps * 3
            && let Some(offset) = noise.get(sample)
        {
            observed.elbow =
                point(pose.elbow + Vec3::from_array(offset[0]) * left.rest.total_arm_length);
            observed.wrist =
                point(pose.wrist + Vec3::from_array(offset[1]) * left.rest.total_arm_length);
        }
        let now = vtuber_core::MonoTimeNs((tick as u64 * 1_000_000_000) / fps as u64);
        let captured =
            vtuber_core::MonoTimeNs((tick as u64 / (fps as u64 / 30)) * 1_000_000_000 / 30);
        let frame = PoseArmFrame {
            source_seq: vtuber_core::FrameSeq(tick as u64 / (fps as u64 / 30) + 1),
            captured_at: captured,
            inference_finished_at: captured,
            observation: Some(PoseArmObservation {
                hips: None,
                left: observed,
                right: missing,
            }),
        };
        let (next, control) = step_arm_tracking(&tracking, Some(&frame), now, &profile);
        tracking = next;
        *app.world_mut().resource_mut::<TrackedArmControl>() = TrackedArmControl {
            generation: Some(generation),
            frame: control,
            view_to_model: Quat::IDENTITY,
        };
        update_after_worker(&mut app);
        if tick < fps * 3 {
            settled_endpoints = Some([
                app.world()
                    .get::<GlobalTransform>(displayed.lower_arm)
                    .unwrap()
                    .translation(),
                app.world()
                    .get::<GlobalTransform>(displayed.hand)
                    .unwrap()
                    .translation(),
            ]);
            continue;
        }
        let control = control.unwrap();
        assert_eq!(control.weights.left.wrist, 1.0);
        assert_eq!(control.weights.left.pole, 1.0);
        assert!(matches!(
            app.world().get::<SolveStatus>(root),
            Some(SolveStatus::Feasible { .. })
        ));
        let joints = app.world().resource::<UpperLimbState>().path.current[side].unwrap();
        assert!(joints.valid());
        assert!(joints.joint_domain(&displayed).unwrap().1 >= -64.0 * f32::EPSILON);
        assert!(
            joints.angles[3] > 60.0_f32.to_radians(),
            "tracking must not silently freeze at rest"
        );
        let p = |e| app.world().get::<GlobalTransform>(e).unwrap().translation();
        let elbow = p(displayed.lower_arm);
        for (current, baseline) in [elbow, p(displayed.hand)]
            .into_iter()
            .zip(settled_endpoints.unwrap())
        {
            output_energy += f64::from(current.distance_squared(baseline));
        }
        if let Some(offset) = noise.get(sample) {
            input_energy += offset
                .iter()
                .map(|point| {
                    f64::from(
                        (Vec3::from_array(*point) * left.rest.total_arm_length).length_squared(),
                    )
                })
                .sum::<f64>();
        }
        assert!(
            (p(displayed.upper_arm).distance(elbow) - displayed.rest.upper_arm_length).abs()
                < 2.0e-6
        );
        assert!((elbow.distance(p(displayed.hand)) - displayed.rest.forearm_length).abs() < 2.0e-6);
        if with_body {
            let state = app.world().resource::<UpperLimbState>();
            let mut motions = state.path.body.as_ref().unwrap().motions().unwrap();
            for (chain, joints) in [left, right].iter().zip(state.path.current) {
                motions.extend(joints.unwrap().forward(chain).unwrap().motion);
            }
            let geometry = app
                .world()
                .get::<AvatarCollision>(root)
                .unwrap()
                .0
                .as_ref()
                .unwrap();
            assert!(
                geometry
                    .pose_is_clear(&|bone| motions.get(&bone).copied(), 2.0e-6)
                    .unwrap()
            );
        }
        if let Some((prev_elbow, prev_angle)) = previous {
            largest_elbow_step =
                largest_elbow_step.max(elbow.distance(prev_elbow) * fps as f32 / 60.0);
            largest_flexion_step =
                largest_flexion_step.max((joints.angles[3] - prev_angle).abs() * fps as f32 / 60.0);
        }
        previous = Some((elbow, joints.angles[3]));
        elbow_positions.push(elbow);
    }
    let mean = elbow_positions.iter().copied().sum::<Vec3>() / elbow_positions.len() as f32;
    let rms = (elbow_positions
        .iter()
        .map(|p| p.distance_squared(mean))
        .sum::<f32>()
        / elbow_positions.len() as f32)
        .sqrt();
    eprintln!(
        "{fps} Hz, body={with_body}: elbow RMS={rms}m, step@60={largest_elbow_step}m, flexion step@60={}deg",
        largest_flexion_step.to_degrees()
    );
    // Quiet posture: no visible one-frame twitch at any render rate.
    assert!(largest_elbow_step < 0.0015);
    assert!(largest_flexion_step < 0.5_f32.to_radians());
    assert!(rms < 0.004);
    (input_energy, output_energy)
}
