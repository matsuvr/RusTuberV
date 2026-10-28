#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)] // tests may panic (AGENTS.md)
#[cfg(feature = "ndi-sdk")]
use super::worker::map_runtime_error;
use super::*;
use crate::*;
use std::time::Instant;
use vtuber_core::{MonoTimeNs, VideoOutputPixelFormat};

#[test]
fn request_stop_and_drop_do_not_wait_for_a_stuck_sdk() {
    let (release, blocked) = std::sync::mpsc::channel();
    let mut controller = NdiOutputController::new();
    let shared = Arc::clone(&controller.shared);
    controller.worker = Some(
        vtuber_core::WorkerHandle::spawn("blocked-sdk", move |_stop| {
            blocked.recv().unwrap();
            // SDK startup can return after stop was requested.
            shared
                .set_status(NdiOutputStatus::Live {
                    connections: 0,
                    source_name: "late".into(),
                })
                .expect("NDI status");
            Ok(WorkerExit::Stopped)
        })
        .unwrap(),
    );
    let started = Instant::now();
    controller.request_stop().expect("NDI operation");
    controller.poll().unwrap();
    assert!(controller.is_stopping());
    assert_eq!(controller.status(), NdiOutputStatus::Off);
    assert_eq!(
        controller
            .start(NdiOutputConfig::default())
            .unwrap_err()
            .code,
        NdiErrorCode::AlreadyRunning
    );
    drop(controller);
    assert!(started.elapsed() < std::time::Duration::from_millis(250));
    release.send(()).unwrap();
}

#[cfg(feature = "ndi-sdk")]
#[test]
fn runtime_error_words_do_not_imply_missing_runtime() {
    for detail in [
        "not found",
        "load failed",
        "library failure",
        "dll mismatch",
        "different explanation",
    ] {
        let native = grafton_ndi::Error::InitializationFailed(detail.into());
        let mapped = map_runtime_error(&native);
        assert_eq!(mapped.code, NdiErrorCode::RuntimeInitFailed);
        assert!(mapped.message.contains(detail));
    }
}

#[cfg(feature = "ndi-sdk")]
#[test]
fn unclassified_native_error_keeps_its_stage_and_detail() {
    let native = grafton_ndi::Error::InvalidConfiguration("specific setting".into());
    let mapped = map_runtime_error(&native);
    assert_eq!(mapped.code, NdiErrorCode::RuntimeInitFailed);
    assert!(mapped.message.contains("specific setting"));
    let missing_file = grafton_ndi::Error::Io(std::io::Error::from(std::io::ErrorKind::NotFound));
    assert_eq!(
        map_runtime_error(&missing_file).code,
        NdiErrorCode::RuntimeInitFailed
    );
}

fn frame(seq: u64) -> VideoOutputFrame {
    VideoOutputFrame::new_bgra8(
        2,
        1,
        FrameSeq(seq),
        vtuber_core::MonoTimeNs(seq),
        vec![1, 2, 3, 4, 5, 6, 7, 8],
    )
    .expect("test frame has a valid packed shape")
}

fn test_profile() -> VideoOutputProfile {
    VideoOutputProfile {
        width: 2,
        height: 1,
        fps: 60,
        pixel_format: VideoOutputPixelFormat::Bgra8StraightAlpha,
    }
}

fn test_config() -> NdiOutputConfig {
    NdiOutputConfig {
        source_name: "RusTuberV".to_owned(),
        profile: test_profile(),
    }
}

#[test]
fn direct_mapping_and_sender_start_both_reject_zero_profile_dimensions_or_rate() {
    for profile in [
        VideoOutputProfile {
            width: 0,
            ..test_profile()
        },
        VideoOutputProfile {
            height: 0,
            ..test_profile()
        },
        VideoOutputProfile {
            fps: 0,
            ..test_profile()
        },
    ] {
        assert_eq!(
            map_video_frame(&frame(1), profile).unwrap_err().code,
            NdiErrorCode::InvalidConfiguration
        );
        let config = NdiOutputConfig {
            profile,
            ..test_config()
        };
        assert_eq!(
            config.validate().unwrap_err().code,
            NdiErrorCode::InvalidConfiguration
        );
    }
}

#[test]
fn an_invalid_frame_cannot_reach_the_sender() {
    assert!(matches!(
        VideoOutputFrame::new_bgra8(2, 1, FrameSeq(1), MonoTimeNs(1), vec![0; 4]),
        Err(vtuber_core::VideoOutputFrameError::DataLength { .. })
    ));
    assert!(matches!(
        VideoOutputFrame::from_padded_bgra8(2, 1, 4, FrameSeq(1), MonoTimeNs(1), vec![0; 4]),
        Err(vtuber_core::VideoOutputFrameError::InvalidStride { .. })
    ));
}

#[test]
fn reaping_a_panicked_worker_preserves_the_error_and_releases_the_mailbox() {
    let mut controller = NdiOutputController::new();
    let mailbox = Arc::new(LatestFrameMailbox::new());
    *checked_lock(controller.shared.mailbox.lock()).expect("NDI lock") = Some(Arc::clone(&mailbox));
    controller.worker = Some(
        vtuber_core::WorkerHandle::spawn("ndi-panic-test", |_| {
            panic!("scripted worker panic");
        })
        .unwrap(),
    );
    while !controller.worker.as_ref().unwrap().is_finished() {
        std::thread::yield_now();
    }
    let error = controller.reap_finished_worker().unwrap_err();
    assert_eq!(error.code, NdiErrorCode::WorkerStopFailed);
    assert!(matches!(
        controller.status(),
        NdiOutputStatus::Error {
            code: NdiErrorCode::WorkerStopFailed,
            ..
        }
    ));
    assert!(controller.worker.is_none());
    assert!(
        checked_lock(controller.shared.mailbox.lock())
            .expect("NDI lock")
            .is_none()
    );
    assert_eq!(
        mailbox.submit(frame(1)).expect("mailbox access"),
        NdiSubmitResult::RejectedNotRunning
    );
}

#[test]
fn reaping_a_startup_failure_does_not_replace_its_cause() {
    let mut controller = NdiOutputController::new();
    let error = NdiOutputError::new(NdiErrorCode::SenderCreateFailed, "scripted startup failure");
    controller.shared.replace_status_error(&error);
    controller.worker = Some(
        vtuber_core::WorkerHandle::spawn("ndi-startup-failure-test", |_| {
            Ok(WorkerExit::StartupFailed)
        })
        .unwrap(),
    );
    while !controller.worker.as_ref().unwrap().is_finished() {
        std::thread::yield_now();
    }
    let before = controller.status();
    controller.reap_finished_worker().unwrap();
    assert_eq!(controller.status(), before);
}

#[test]
fn mapping_preserves_bgra_alpha_and_profile_fields() {
    let source = frame(7);
    let source_bytes = source.data().to_vec();
    let mapping = map_video_frame(&source, test_profile()).expect("valid frame maps");
    assert_eq!(mapping.width, 2);
    assert_eq!(mapping.height, 1);
    assert_eq!(mapping.four_cc, NdiFourCc::Bgra);
    assert_eq!(mapping.stride_bytes, 8);
    assert_eq!(mapping.frame_rate_n, 60);
    assert_eq!(mapping.frame_rate_d, 1);
    assert_eq!(mapping.picture_aspect_ratio, 2.0);
    assert_eq!(source.data(), source_bytes);
    assert_eq!(source.data()[3], 4);
    assert_eq!(source.data()[7], 8);
}

#[cfg(test)]
fn runtime_probe_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "vtuber-ndi-runtime-probe-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("probe dir is creatable");
    dir
}

#[test]
fn runtime_probe_finds_current_dll_name() {
    let dir = runtime_probe_dir("current");
    std::fs::write(dir.join(runtime_identity::RUNTIME_FILE_NAMES[0]), b"stub")
        .expect("current stub is writable");
    assert!(runtime_identity::runtime_file_in(&dir).is_some());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn runtime_probe_accepts_legacy_dll_name() {
    let dir = runtime_probe_dir("legacy");
    std::fs::write(dir.join(runtime_identity::RUNTIME_FILE_NAMES[1]), b"stub")
        .expect("legacy stub is writable");
    assert!(runtime_identity::runtime_file_in(&dir).is_some());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn runtime_probe_reports_missing_for_empty_dir() {
    let dir = runtime_probe_dir("empty");
    assert!(runtime_identity::runtime_file_in(&dir).is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn runtime_probe_does_not_panic() {
    let _ = is_ndi_runtime_installed();
}

#[test]
fn malformed_frame_is_rejected_before_mapping() {
    let invalid =
        VideoOutputFrame::new_bgra8(1, 1, FrameSeq(1), MonoTimeNs(1), vec![0; 4]).unwrap();
    let error = map_video_frame(
        &invalid,
        VideoOutputProfile {
            width: 2,
            height: 1,
            fps: 60,
            pixel_format: VideoOutputPixelFormat::Bgra8StraightAlpha,
        },
    )
    .expect_err("wrong stride must be rejected");
    assert_eq!(error.code, NdiErrorCode::InvalidFrame);
}

#[test]
fn mailbox_replaces_old_frame_and_stays_capacity_one() {
    let mailbox = LatestFrameMailbox::new();
    assert_eq!(
        mailbox.submit(frame(1)).expect("mailbox access"),
        NdiSubmitResult::Submitted
    );
    assert_eq!(
        mailbox.submit(frame(2)).expect("mailbox access"),
        NdiSubmitResult::Replaced
    );
    assert_eq!(
        mailbox
            .take(|| false)
            .expect("mailbox access")
            .expect("latest frame")
            .frame_seq(),
        FrameSeq(2)
    );
    assert!(mailbox.take(|| true).expect("mailbox access").is_none());
}

#[test]
fn closed_mailbox_releases_waiting_consumer() {
    let mailbox = Arc::new(LatestFrameMailbox::new());
    let worker_mailbox = Arc::clone(&mailbox);
    let started =
        std::thread::spawn(move || worker_mailbox.take(|| false).expect("mailbox access"));
    mailbox.close().expect("mailbox access");
    assert!(started.join().expect("consumer joined").is_none());
}

#[test]
fn burst_submission_remains_bounded_and_non_blocking() {
    let mailbox = LatestFrameMailbox::new();
    let mut last_result = NdiSubmitResult::Submitted;
    for sequence in 0..1000 {
        last_result = mailbox.submit(frame(sequence)).expect("mailbox access");
    }
    assert_eq!(last_result, NdiSubmitResult::Replaced);
    assert_eq!(
        mailbox
            .take(|| false)
            .expect("mailbox access")
            .expect("latest frame")
            .frame_seq(),
        FrameSeq(999)
    );
    assert!(mailbox.take(|| true).expect("mailbox access").is_none());
}

#[test]
fn slow_consumer_does_not_block_producer() {
    let mailbox = Arc::new(LatestFrameMailbox::new());
    let consumer_mailbox = Arc::clone(&mailbox);
    let consumer_ready = Arc::new(std::sync::Barrier::new(2));
    let consumer_took_frame = Arc::new(std::sync::Barrier::new(2));
    let consumer_ready_thread = Arc::clone(&consumer_ready);
    let consumer_took_frame_thread = Arc::clone(&consumer_took_frame);
    let consumer = std::thread::spawn(move || {
        consumer_ready_thread.wait();
        let frame = consumer_mailbox.take(|| false).expect("mailbox access");
        consumer_took_frame_thread.wait();
        std::thread::sleep(std::time::Duration::from_millis(20));
        frame
    });
    consumer_ready.wait();
    assert_eq!(
        mailbox.submit(frame(0)).expect("mailbox access"),
        NdiSubmitResult::Submitted
    );
    consumer_took_frame.wait();
    let begin = Instant::now();
    for sequence in 1..1000 {
        let _ = mailbox.submit(frame(sequence)).expect("mailbox access");
    }
    assert!(begin.elapsed() < std::time::Duration::from_secs(1));
    assert!(consumer.join().expect("slow consumer joined").is_some());
    mailbox.close().expect("mailbox access");
}

#[test]
fn controller_is_off_without_start() {
    let controller = NdiOutputController::new();
    assert_eq!(controller.status(), NdiOutputStatus::Off);
    assert_eq!(controller.metrics(), NdiOutputMetrics::default());
    assert_eq!(
        controller.submit_frame(frame(1)).expect("NDI operation"),
        NdiSubmitResult::RejectedNotRunning
    );
}

#[cfg(not(feature = "ndi-sdk"))]
#[test]
fn feature_off_start_is_typed_error_and_stop_is_idempotent() {
    let mut controller = NdiOutputController::new();
    let error = controller
        .start(NdiOutputConfig::default())
        .expect_err("feature is off");
    assert_eq!(error.code, NdiErrorCode::FeatureDisabled);
    assert!(matches!(
        controller.status(),
        NdiOutputStatus::Error {
            code: NdiErrorCode::FeatureDisabled,
            ..
        }
    ));
    controller.stop().expect("stop is idempotent");
    assert_eq!(controller.status(), NdiOutputStatus::Off);
    controller.stop().expect("second stop is idempotent");
    assert_eq!(controller.status(), NdiOutputStatus::Off);
    assert!(!controller.has_worker());
}

#[test]
fn scripted_backend_walks_off_starting_live_off() {
    let backend = NdiScriptedBackend::successful();
    backend.hold_startup().expect("NDI operation");
    let mut controller = NdiOutputController::with_scripted_backend(backend.clone());
    controller.start(test_config()).expect("scripted start");
    assert!(matches!(controller.status(), NdiOutputStatus::Starting));
    backend.release_startup().expect("NDI operation");
    backend.wait_until_ready().expect("NDI operation");
    assert!(matches!(
        controller.status(),
        NdiOutputStatus::Live {
            source_name,
            ..
        } if source_name == "RusTuberV"
    ));
    controller.stop().expect("stop");
    assert_eq!(controller.status(), NdiOutputStatus::Off);
    assert!(!controller.has_worker());
    assert_eq!(backend.live_senders(), 0);
}

#[test]
fn runtime_initialization_failure_is_error_and_cleans_up() {
    let backend = NdiScriptedBackend::fail_initialize(NdiErrorCode::RuntimeInitFailed);
    let mut controller = NdiOutputController::with_scripted_backend(backend.clone());
    controller.start(test_config()).expect("worker starts");
    backend.wait_until_ready().expect("NDI operation");
    assert!(matches!(
        controller.status(),
        NdiOutputStatus::Error {
            code: NdiErrorCode::RuntimeInitFailed,
            ..
        }
    ));
    assert_eq!(backend.live_senders(), 0);
    assert_eq!(
        controller.submit_frame(frame(1)).expect("NDI operation"),
        NdiSubmitResult::RejectedNotRunning
    );
    controller.stop().expect("stop after failure");
    assert_eq!(controller.status(), NdiOutputStatus::Off);
    assert!(!controller.has_worker());
}

#[test]
fn sender_create_failure_is_error_and_cleans_up() {
    let backend = NdiScriptedBackend::fail_create_sender();
    let mut controller = NdiOutputController::with_scripted_backend(backend.clone());
    controller.start(test_config()).expect("worker starts");
    backend.wait_until_ready().expect("NDI operation");
    assert!(matches!(
        controller.status(),
        NdiOutputStatus::Error {
            code: NdiErrorCode::SenderCreateFailed,
            ..
        }
    ));
    assert_eq!(backend.live_senders(), 0);
    controller.stop().expect("cleanup");
    assert!(!controller.has_worker());
    assert_eq!(
        controller.submit_frame(frame(1)).expect("NDI operation"),
        NdiSubmitResult::RejectedNotRunning
    );
}

#[test]
fn duplicate_start_while_live_is_rejected_deterministically() {
    let backend = NdiScriptedBackend::successful();
    let mut controller = NdiOutputController::with_scripted_backend(backend.clone());
    controller.start(test_config()).expect("first start");
    backend.wait_until_ready().expect("NDI operation");
    let error = controller
        .start(test_config())
        .expect_err("duplicate start");
    assert_eq!(error.code, NdiErrorCode::AlreadyRunning);
    assert!(matches!(controller.status(), NdiOutputStatus::Live { .. }));
    controller.stop().expect("stop");
}

#[test]
fn repeated_start_stop_does_not_leave_a_worker() {
    let backend = NdiScriptedBackend::successful();
    let mut controller = NdiOutputController::with_scripted_backend(backend.clone());
    for _ in 0..3 {
        controller.start(test_config()).expect("start");
        backend.wait_until_ready().expect("NDI operation");
        controller.stop().expect("stop");
        assert_eq!(controller.status(), NdiOutputStatus::Off);
        assert!(!controller.has_worker());
        assert_eq!(backend.live_senders(), 0);
        assert_eq!(
            controller.submit_frame(frame(1)).expect("NDI operation"),
            NdiSubmitResult::RejectedNotRunning
        );
    }
}

#[test]
fn slow_scripted_sender_does_not_block_producer() {
    let backend = NdiScriptedBackend::successful();
    backend.hold_send().expect("NDI operation");
    let mut controller = NdiOutputController::with_scripted_backend(backend.clone());
    controller.start(test_config()).expect("start");
    backend.wait_until_ready().expect("NDI operation");
    assert_eq!(
        controller.submit_frame(frame(0)).expect("NDI operation"),
        NdiSubmitResult::Submitted
    );
    backend.wait_until_send_blocked().expect("NDI operation");
    let mut replaced = 0_u64;
    for sequence in 1..1000 {
        match controller
            .submit_frame(frame(sequence))
            .expect("NDI operation")
        {
            NdiSubmitResult::Replaced => replaced += 1,
            NdiSubmitResult::Submitted => {}
            NdiSubmitResult::RejectedNotRunning => {
                panic!("live controller must accept frames")
            }
        }
    }
    assert!(replaced >= 1);
    assert_eq!(controller.metrics().replaced_frames, replaced);
    backend.release_send().expect("NDI operation");
    controller.stop().expect("stop");
    assert!(!controller.has_worker());
}

#[test]
fn worker_stop_rejects_later_frames() {
    let backend = NdiScriptedBackend::successful();
    let mut controller = NdiOutputController::with_scripted_backend(backend.clone());
    controller.start(test_config()).expect("start");
    backend.wait_until_ready().expect("NDI operation");
    controller.stop().expect("stop");
    assert_eq!(
        controller.submit_frame(frame(9)).expect("NDI operation"),
        NdiSubmitResult::RejectedNotRunning
    );
}

#[test]
fn valid_frame_with_wrong_profile_is_rejected_before_the_fake_sender() {
    let backend = NdiScriptedBackend::successful();
    let mut controller = NdiOutputController::with_scripted_backend(backend.clone());
    controller.start(test_config()).expect("start");
    backend.wait_until_ready().expect("NDI operation");
    let invalid =
        VideoOutputFrame::new_bgra8(1, 1, FrameSeq(1), MonoTimeNs(1), vec![0; 4]).unwrap();
    assert_eq!(
        controller.submit_frame(invalid).expect("NDI operation"),
        NdiSubmitResult::Submitted
    );
    let started = std::time::Instant::now();
    while backend.sent_frames() == 0
        && controller.metrics().rejected_frames == 0
        && started.elapsed() < std::time::Duration::from_secs(1)
    {
        std::thread::yield_now();
    }
    assert_eq!(backend.sent_frames(), 0);
    assert!(controller.metrics().rejected_frames >= 1);
    controller.stop().expect("stop");
}

#[test]
fn send_timecode_requests_sdk_synthesis() {
    assert_eq!(NDI_SEND_TIMECODE_SYNTHESIZE, i64::MAX);
}

#[cfg(feature = "ndi-sdk")]
#[test]
fn built_ndi_frame_asks_sdk_to_synthesize_timecodes() {
    let mapping = map_video_frame(&frame(1), test_profile()).expect("valid frame maps");
    let built = super::worker::build_ndi_video_frame(&mapping).expect("validated mapping builds");
    assert_eq!(built.timecode(), i64::MAX);
    assert_eq!(built.timestamp(), 0);
}

#[cfg(feature = "ndi-sdk")]
#[test]
#[ignore = "requires a locally installed NDI SDK/runtime"]
fn sdk_sender_creates_sends_synthetic_bgra_and_stops() {
    let mut controller = NdiOutputController::new();
    match controller.start(test_config()) {
        Ok(()) => {}
        Err(error)
            if matches!(
                error.code,
                NdiErrorCode::RuntimeNotFound | NdiErrorCode::RuntimeInitFailed
            ) =>
        {
            return;
        }
        Err(error) => panic!("unexpected SDK start failure: {error}"),
    }
    let started = std::time::Instant::now();
    while !matches!(controller.status(), NdiOutputStatus::Live { .. })
        && !matches!(controller.status(), NdiOutputStatus::Error { .. })
        && started.elapsed() < std::time::Duration::from_secs(5)
    {
        std::thread::yield_now();
    }
    match controller.status() {
        NdiOutputStatus::Live { source_name, .. } => {
            assert_eq!(source_name, "RusTuberV");
        }
        NdiOutputStatus::Error {
            code: NdiErrorCode::RuntimeNotFound | NdiErrorCode::RuntimeInitFailed,
            ..
        } => return,
        other => panic!("SDK sender did not become live: {other:?}"),
    }
    for sequence in 0..3 {
        let _ = controller
            .submit_frame(frame(sequence))
            .expect("NDI operation");
    }
    let sent_deadline = std::time::Instant::now();
    while controller.metrics().sent_frames == 0
        && sent_deadline.elapsed() < std::time::Duration::from_secs(2)
    {
        std::thread::yield_now();
    }
    controller.stop().expect("SDK stop");
    assert_eq!(controller.status(), NdiOutputStatus::Off);
    assert!(!controller.has_worker());
}

#[test]
fn poisoned_mailbox_is_reported_after_worker_is_joined() {
    let backend = NdiScriptedBackend::successful();
    let mut controller = NdiOutputController::with_scripted_backend(backend.clone());
    controller.start(NdiOutputConfig::default()).unwrap();
    backend.wait_until_ready().unwrap();
    let mailbox = checked_lock(controller.shared.mailbox.lock())
        .unwrap()
        .clone()
        .unwrap();
    assert!(
        std::panic::catch_unwind(|| {
            let _guard = mailbox.state.lock().unwrap();
            panic!("interrupted mailbox update");
        })
        .is_err()
    );
    assert_eq!(
        controller.submit_frame(frame(1)).unwrap_err().code,
        NdiErrorCode::StatePoisoned
    );
    assert_eq!(
        controller.stop().unwrap_err().code,
        NdiErrorCode::StatePoisoned
    );
    assert!(!controller.has_worker());
    assert_eq!(backend.live_senders(), 0);
    assert!(matches!(
        controller.status(),
        NdiOutputStatus::Error {
            code: NdiErrorCode::StatePoisoned,
            ..
        }
    ));
}

#[test]
fn poisoned_status_never_starts_a_sender_or_reads_live_state() {
    let mut controller =
        NdiOutputController::with_scripted_backend(NdiScriptedBackend::successful());
    assert!(
        std::panic::catch_unwind(|| {
            let _guard = controller.shared.status.lock().unwrap();
            panic!("interrupted status update");
        })
        .is_err()
    );
    assert!(matches!(
        controller.status(),
        NdiOutputStatus::Error {
            code: NdiErrorCode::StatePoisoned,
            ..
        }
    ));
    assert_eq!(
        controller
            .start(NdiOutputConfig::default())
            .unwrap_err()
            .code,
        NdiErrorCode::StatePoisoned
    );
    assert!(!controller.has_worker());
}

#[test]
fn poisoned_mailbox_owner_does_not_skip_join_or_leave_starting_status() {
    let backend = NdiScriptedBackend::successful();
    let mut controller = NdiOutputController::with_scripted_backend(backend.clone());
    controller.start(NdiOutputConfig::default()).unwrap();
    backend.wait_until_ready().unwrap();
    assert!(
        std::panic::catch_unwind(|| {
            let _guard = controller.shared.mailbox.lock().unwrap();
            panic!("interrupted mailbox ownership update");
        })
        .is_err()
    );
    assert_eq!(
        controller.stop().unwrap_err().code,
        NdiErrorCode::StatePoisoned
    );
    assert!(!controller.has_worker());
    assert_eq!(backend.live_senders(), 0);
    assert_eq!(
        controller
            .start(NdiOutputConfig::default())
            .unwrap_err()
            .code,
        NdiErrorCode::StatePoisoned
    );
    assert!(matches!(
        controller.status(),
        NdiOutputStatus::Error {
            code: NdiErrorCode::StatePoisoned,
            ..
        }
    ));
}
