//! Replay model events against the same lifecycle, tracking and output APIs as the application.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
#[path = "../../../verification/replay.rs"]
pub(crate) mod replay;
use replay::{number, traces};

use crate::{AvatarLifecycle, AvatarLifecycleFailure, AvatarOutputState};
use bevy::prelude::Entity;
use std::time::Duration;
use vtuber_core::{LatestSlot, ReadResult};
use vtuber_platform::PowerState;
use vtuber_tracking::confidence::ConfidenceSignal;
use vtuber_tracking::state_machine::{StateMachineParams, TrackingStateMachine, TransitionInput};

#[test]
#[ignore = "generated Quint ITF traces"]
fn quint_video_frame() {
    use vtuber_core::{
        FrameSeq, MonoTimeNs, VideoOutputFrame, VideoOutputFrameError, VideoOutputPixelFormat,
    };
    traces("video_frame", |states| {
        for row in states {
            let w = number(&row["width"]) as u32;
            let h = number(&row["height"]) as u32;
            let packed = w as usize * 4;
            let stride = match row["strideClass"].as_str().unwrap() {
                "short" => packed.saturating_sub(1),
                "packed" => packed,
                "padded" => packed + 3,
                "max" => usize::MAX,
                other => panic!("{other}"),
            };
            // Overflow cases use a small buffer. All ordinary sizes fit in 22 bytes.
            let len = if stride == usize::MAX {
                0
            } else {
                (stride * h as usize).saturating_add_signed(number(&row["sizeDelta"]) as isize)
            };
            let mut input = vec![0xff; len];
            let expected: Vec<u8> = (0..packed * h as usize).map(|n| n as u8).collect();
            if row["result"] == "ok" {
                for (dst, src) in input
                    .chunks_exact_mut(stride)
                    .zip(expected.chunks_exact(packed))
                {
                    dst[..packed].copy_from_slice(src);
                }
            }
            let result = VideoOutputFrame::from_rendered_bgra8(
                w,
                h,
                stride,
                FrameSeq(7),
                MonoTimeNs(12),
                input,
            );
            let kind = match result {
                Ok(frame) => {
                    assert_eq!(
                        (frame.width(), frame.height(), frame.stride_bytes()),
                        (w, h, packed)
                    );
                    assert_eq!(frame.frame_seq(), FrameSeq(7));
                    assert_eq!(frame.captured_at(), MonoTimeNs(12));
                    assert_eq!(
                        frame.pixel_format(),
                        VideoOutputPixelFormat::Bgra8PremultipliedAlpha
                    );
                    assert_eq!(frame.data().len() as i64, number(&row["packedLength"]));
                    assert_eq!(frame.data(), expected);
                    "ok"
                }
                Err(VideoOutputFrameError::ZeroDimensions) => "ZeroDimensions",
                Err(VideoOutputFrameError::InvalidStride { minimum, actual }) => {
                    assert_eq!((minimum, actual), (packed, stride));
                    "InvalidStride"
                }
                Err(VideoOutputFrameError::SizeOverflow) => "SizeOverflow",
                Err(VideoOutputFrameError::DataLength { expected, actual }) => {
                    assert_eq!((expected, actual), (stride * h as usize, len));
                    "DataLength"
                }
                Err(other) => panic!("{other}"),
            };
            assert_eq!(kind, row["result"].as_str().unwrap());
        }
    });
}

#[test]
#[ignore = "generated Quint ITF traces"]
fn quint_arm_admission() {
    let chain = crate::upper_limb::tests::chain(crate::ArmSide::Left);
    let tolerance = 64.0 * f32::EPSILON * chain.rest.total_arm_length;
    let problem = crate::upper_limb_solver::Problem {
        chains: [Some(&chain), None],
        goals: [None, None],
        geometry: &Default::default(),
        body: &Default::default(),
        body_curve: None,
        tolerance,
    };
    let mut eval = problem.kinematics([None, None]).unwrap();
    let contact_limit = chain.rest.total_arm_length * 0.002 - tolerance;
    let joint_limit = -64.0 * f32::EPSILON;
    let boundary = |threshold: f32, side: i64| match side {
        -1 => threshold.next_down(),
        0 => threshold,
        1 => threshold.next_up(),
        _ => panic!("unknown threshold partition"),
    };
    traces("arm_admission", |states| {
        for row in states {
            eval.clearance = boundary(contact_limit, number(&row["clearance"]));
            eval.joint_margin = boundary(joint_limit, number(&row["joint"]));
            assert_eq!(row["admitted"], problem.feasible(&eval));
        }
    });
}

#[test]
#[ignore = "generated Quint ITF traces"]
fn quint_slot() {
    traces("slot", |states| {
        let slot = LatestSlot::new();
        let mut readers = [0_u64; 2];
        for row in states {
            let op = row["event"].as_str().unwrap();
            let expected = &row["state"];
            let result = match op {
                "init" => "none",
                "publishA" | "publishB" => {
                    if slot.publish(if op == "publishA" { 1 } else { 2 }).unwrap() {
                        "published"
                    } else {
                        "rejected"
                    }
                }
                "face" | "pose" => {
                    let cursor = &mut readers[usize::from(op == "pose")];
                    match slot.try_read_after(*cursor).unwrap() {
                        Some(ReadResult::New { generation, value }) => {
                            assert_eq!(value, number(&expected["value"]));
                            *cursor = generation;
                            "new"
                        }
                        Some(ReadResult::Closed) => "closed",
                        None => "none",
                    }
                }
                "clear" => {
                    slot.clear().unwrap();
                    "none"
                }
                "close" => {
                    slot.close().unwrap();
                    "none"
                }
                other => panic!("unknown slot event {other}"),
            };
            assert_eq!(row["result"], result);
            assert_eq!(
                slot.generation().unwrap(),
                number(&expected["generation"]) as u64
            );
            assert_eq!(
                slot.replacement_count().unwrap(),
                number(&expected["replacements"]) as u64
            );
            assert_eq!(expected["closed"], slot.is_closed().unwrap());
            assert_eq!(
                readers,
                [
                    number(&expected["face"]) as u64,
                    number(&expected["pose"]) as u64
                ]
            );
            if !slot.is_closed().unwrap() {
                let retained = match slot.try_read_after(0).unwrap() {
                    Some(ReadResult::New { value, .. }) => value,
                    None => 0,
                    Some(ReadResult::Closed) => panic!("open slot reported closed"),
                };
                assert_eq!(retained, number(&expected["value"]));
            }
        }
    });
}

#[test]
#[ignore = "generated Quint ITF traces"]
fn quint_tracking() {
    let observation = vtuber_core::RawFaceObservation {
        source_seq: vtuber_core::FrameSeq(1),
        captured_at: vtuber_core::MonoTimeNs(0),
        inference_started_at: vtuber_core::MonoTimeNs(0),
        inference_finished_at: vtuber_core::MonoTimeNs(0),
        face_confidence: 1.0,
        landmarks: vec![],
        blendshapes: None,
        expressions: Default::default(),
        roi: Default::default(),
        schema: vtuber_core::LandmarkSchemaId("quint"),
    };
    traces("tracking", |states| {
        let mut machine = TrackingStateMachine::new(StateMachineParams {
            hold_duration: Duration::from_millis(2),
            return_duration: Duration::from_millis(3),
        })
        .unwrap();
        for (i, row) in states.iter().enumerate() {
            if i > 0 {
                let event = &row["event"];
                let result = machine.update(TransitionInput {
                    signal: match event["signal"].as_str().unwrap() {
                        "Acquire" => ConfidenceSignal::Acquire,
                        "Degrade" => ConfidenceSignal::Degrade,
                        "None" => ConfidenceSignal::None,
                        other => panic!("unknown signal {other}"),
                    },
                    observation: (event["quality"] != "missing").then_some(&observation),
                    observation_usable: event["quality"] == "usable",
                    calibration_available: event["calibrated"].as_bool().unwrap(),
                    dt: Duration::from_millis(number(&event["dt"]) as u64),
                });
                let effects: Vec<_> = result.actions.iter().map(|a| format!("{a:?}")).collect();
                let effect = row["effect"].as_str().unwrap();
                assert_eq!(
                    effects,
                    if effect == "none" {
                        vec![]
                    } else {
                        vec![effect]
                    }
                );
            }
            assert_eq!(row["phase"], format!("{:?}", machine.state()));
            assert_eq!(
                number(&row["elapsed"]),
                machine.elapsed_in_state().as_millis().min(4) as i64
            );
        }
    });
}

#[test]
#[ignore = "generated Quint ITF traces"]
fn quint_avatar() {
    let entity = |id: i64| (id > 0).then(|| Entity::from_raw_u32(id as u32).unwrap());
    traces("avatar", |states| {
        let mut lifecycle = AvatarLifecycle::new();
        let mut binding = None;
        let mut frame = None;
        for row in states {
            let event = &row["event"];
            let root = entity(number(&event["root"]));
            let accepted = match event["op"].as_str().unwrap() {
                "init" => false,
                "saveBinding" => {
                    if lifecycle.state() == crate::AvatarLifecycleState::Ready {
                        binding = Some(crate::AvatarBinding::head_only(
                            lifecycle.active_root().unwrap(),
                            Entity::from_raw_u32(3).unwrap(),
                            lifecycle.current_generation(),
                        ));
                    }
                    false
                }
                "saveFrame" => {
                    if lifecycle.has_active_generation() {
                        frame = Some(lifecycle.current_generation());
                    }
                    false
                }
                "load" => lifecycle.request_load(root.unwrap()).is_ok(),
                "replace" => lifecycle.request_replace(root.unwrap()).is_ok(),
                "unload" => lifecycle.request_unload().is_ok(),
                "bind" => {
                    lifecycle.start_binding(root.unwrap());
                    false
                }
                "ready" => {
                    lifecycle.finish_ready();
                    false
                }
                "finish" => {
                    lifecycle.finish_unload();
                    false
                }
                "fail" => {
                    lifecycle.fail(AvatarLifecycleFailure::BindingTimeout);
                    false
                }
                other => panic!("unknown avatar event {other}"),
            };
            let state = &row["state"];
            let resolved =
                crate::unload::resolve_control_target(&lifecycle, binding.as_ref(), frame);
            assert_eq!(
                row["drives"],
                frame.is_some() && resolved.is_ok_and(|target| target.frame_is_current)
            );
            assert_eq!(row["accepted"], accepted);
            assert_eq!(state["phase"], format!("{:?}", lifecycle.state()));
            assert_eq!(lifecycle.active_root(), entity(number(&state["root"])));
            assert_eq!(lifecycle.pending_root(), entity(number(&state["pending"])));
            assert_eq!(
                lifecycle.current_generation().0,
                number(&state["generation"]) as u64
            );
            assert_eq!(
                lifecycle.load_started().is_some(),
                state["phase"] == "Loading"
            );
            assert_eq!(lifecycle.failure().is_some(), state["phase"] == "Failed");
        }
    });
}

#[test]
#[ignore = "generated Quint ITF traces"]
fn quint_readback() {
    traces("readback", |states| {
        let mut output = AvatarOutputState::default();
        let mut power = PowerState::default();
        let mut request = None;
        for row in states {
            let op = row["event"].as_str().unwrap();
            match op {
                "on" | "off" => output.set_active(op == "on"),
                "show" | "hide" => output.set_preview_visible(op == "show"),
                "sleep" if !power.sleeping => {
                    power.sleeping = true;
                    power.generation += 1;
                }
                "wake" => power.sleeping = false,
                "issue" if output.is_active() && !power.sleeping => {
                    request = Some((output.readback_epoch(), power.generation));
                }
                "init" | "sleep" | "issue" | "complete" => {}
                other => panic!("unknown readback event {other}"),
            }
            let deliver = request.is_some_and(|(epoch, generation)| {
                output.accepts_readback(epoch, generation, power)
            });
            assert_eq!(row["delivered"], op == "complete" && deliver);
            assert_eq!(row["active"], output.is_active());
            assert_eq!(row["request"], request.is_some());
            assert_eq!(
                output.is_rendering(),
                output.is_active() || row["preview"].as_bool().unwrap()
            );
        }
    });
}
