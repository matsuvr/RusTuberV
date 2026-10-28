//! Scripted and SDK sender loops, owned by the controller.
use super::{LatestFrameMailbox, NdiScriptedBackend, ScriptedSenderGuard, SharedState, WorkerExit};
#[cfg(feature = "ndi-sdk")]
use crate::{NDI_SEND_TIMECODE_SYNTHESIZE, NdiVideoFrameMapping};
use crate::{NdiErrorCode, NdiOutputConfig, NdiOutputError, NdiOutputStatus, map_video_frame};
use std::sync::Arc;
use std::sync::atomic::Ordering;

pub(super) fn run_scripted_worker(
    shared: Arc<SharedState>,
    mailbox: Arc<LatestFrameMailbox>,
    config: NdiOutputConfig,
    backend: NdiScriptedBackend,
    stop: vtuber_core::StopToken,
) -> Result<WorkerExit, NdiOutputError> {
    if !backend.wait_for_startup(&stop)? {
        backend.mark_ready()?;
        return Ok(WorkerExit::Stopped);
    }
    if let Some(code) = backend.inner.init_error {
        let error = NdiOutputError::new(code, "NDI runtime could not be initialized");
        shared
            .metrics
            .start_failures
            .fetch_add(1, Ordering::Relaxed);
        shared.replace_status_error(&error);
        backend.mark_ready()?;
        return Ok(WorkerExit::StartupFailed);
    }
    if let Some(code) = backend.inner.create_error {
        let error = NdiOutputError::new(code, "could not create NDI sender");
        shared
            .metrics
            .start_failures
            .fetch_add(1, Ordering::Relaxed);
        shared.replace_status_error(&error);
        backend.mark_ready()?;
        return Ok(WorkerExit::StartupFailed);
    }
    backend.inner.live_senders.fetch_add(1, Ordering::Relaxed);
    let _sender_guard = ScriptedSenderGuard {
        backend: backend.clone(),
    };
    shared.set_status(NdiOutputStatus::Live {
        connections: backend.inner.connections.load(Ordering::Relaxed),
        source_name: config.source_name.clone(),
    })?;
    backend.mark_ready()?;
    loop {
        let Some(frame) = mailbox.take(|| stop.is_stopped())? else {
            return Ok(WorkerExit::Stopped);
        };
        let mapping = match map_video_frame(&frame, config.profile) {
            Ok(mapping) => mapping,
            Err(_) => {
                shared
                    .metrics
                    .rejected_frames
                    .fetch_add(1, Ordering::Relaxed);
                continue;
            }
        };
        let _ = mapping;
        if let Err(error) = backend.send_frame(&stop) {
            if error.code == NdiErrorCode::NotRunning && stop.is_stopped() {
                return Ok(WorkerExit::Stopped);
            }
            return Err(error);
        }
        shared.metrics.sent_frames.fetch_add(1, Ordering::Relaxed);
        shared
            .metrics
            .last_frame_seq
            .store(frame.frame_seq().0, Ordering::Relaxed);
        shared.set_status(NdiOutputStatus::Live {
            connections: backend.inner.connections.load(Ordering::Relaxed),
            source_name: config.source_name.clone(),
        })?;
    }
}

#[cfg(feature = "ndi-sdk")]
pub(super) fn build_ndi_video_frame(
    mapping: &NdiVideoFrameMapping,
) -> Result<grafton_ndi::VideoFrame, grafton_ndi::Error> {
    use grafton_ndi::{PixelFormat, ScanType, VideoFrame};

    VideoFrame::builder()
        .resolution(mapping.width, mapping.height)
        .pixel_format(PixelFormat::BGRA)
        .frame_rate(mapping.frame_rate_n, mapping.frame_rate_d)
        .aspect_ratio(mapping.picture_aspect_ratio)
        .scan_type(ScanType::Progressive)
        .timecode(NDI_SEND_TIMECODE_SYNTHESIZE)
        .build()
}

#[cfg(feature = "ndi-sdk")]
pub(super) fn run_ndi_worker(
    shared: Arc<SharedState>,
    mailbox: Arc<LatestFrameMailbox>,
    config: NdiOutputConfig,
    stop: vtuber_core::StopToken,
) -> Result<WorkerExit, NdiOutputError> {
    use grafton_ndi::{NDI, Sender, SenderOptions, VideoFrame};

    let power = vtuber_platform::power_state();
    let ndi = match NDI::new() {
        Ok(ndi) => ndi,
        Err(error) => {
            let mapped = map_runtime_error(&error);
            shared
                .metrics
                .start_failures
                .fetch_add(1, Ordering::Relaxed);
            shared.replace_status_error(&mapped);
            return Ok(WorkerExit::StartupFailed);
        }
    };
    let options = SenderOptions::builder(config.source_name.clone())
        .clock_video(true)
        .clock_audio(false)
        .build();
    let sender = match Sender::new(&ndi, &options) {
        Ok(sender) => sender,
        Err(error) => {
            let mapped = NdiOutputError::new(
                NdiErrorCode::SenderCreateFailed,
                format!("could not create NDI sender: {error}"),
            );
            shared
                .metrics
                .start_failures
                .fetch_add(1, Ordering::Relaxed);
            shared.replace_status_error(&mapped);
            return Ok(WorkerExit::StartupFailed);
        }
    };
    shared.set_status(NdiOutputStatus::Live {
        connections: 0,
        source_name: config.source_name.clone(),
    })?;
    let mut last_connection_poll = std::time::Instant::now();
    // The validated NDI frame is reused across sends with a stable layout;
    // only its pixel bytes are rewritten, avoiding a fresh full-frame buffer
    // allocation on every sent frame.
    let mut ndi_frame: Option<VideoFrame> = None;
    let mut ndi_frame_dims: (i32, i32) = (0, 0);
    loop {
        let Some(frame) = mailbox.take(|| stop.is_stopped())? else {
            return Ok(WorkerExit::Stopped);
        };
        let current_power = vtuber_platform::power_state();
        if stop.is_stopped()
            || current_power.sleeping
            || current_power.generation != power.generation
        {
            return Ok(WorkerExit::Stopped);
        }
        let mapping = match map_video_frame(&frame, config.profile) {
            Ok(mapping) => mapping,
            Err(_) => {
                shared
                    .metrics
                    .rejected_frames
                    .fetch_add(1, Ordering::Relaxed);
                continue;
            }
        };
        let dims = (mapping.width, mapping.height);
        if ndi_frame.is_none() || ndi_frame_dims != dims {
            let mut built = match build_ndi_video_frame(&mapping) {
                Ok(frame) => frame,
                Err(error) => {
                    let mapped = NdiOutputError::new(
                        NdiErrorCode::SendFailed,
                        format!("NDI rejected the validated BGRA frame: {error}"),
                    );
                    shared.replace_status_error(&mapped);
                    return Ok(WorkerExit::StartupFailed);
                }
            };
            if let Err(cause) = built.replace_data(frame.data().to_vec()) {
                let error = NdiOutputError::new(
                    NdiErrorCode::SendFailed,
                    format!("NDI frame storage rejected the validated BGRA frame: {cause}"),
                );
                shared.replace_status_error(&error);
                return Ok(WorkerExit::StartupFailed);
            }
            ndi_frame_dims = dims;
            ndi_frame = Some(built);
        } else if let Some(existing) = ndi_frame.as_mut()
            && existing.data().len() == frame.data().len()
        {
            existing.data_mut().copy_from_slice(frame.data());
        } else if let Some(existing) = ndi_frame.as_mut() {
            // Defensive length fallback: rebuild with a fresh buffer when the
            // validated frame data no longer matches the reused layout.
            if let Err(cause) = existing.replace_data(frame.data().to_vec()) {
                let error = NdiOutputError::new(
                    NdiErrorCode::SendFailed,
                    format!("NDI frame storage rejected the validated BGRA frame: {cause}"),
                );
                shared.replace_status_error(&error);
                return Ok(WorkerExit::StartupFailed);
            }
        }
        if let Some(existing) = &ndi_frame {
            sender.send_video(existing);
        }
        shared.metrics.sent_frames.fetch_add(1, Ordering::Relaxed);
        shared
            .metrics
            .last_frame_seq
            .store(frame.frame_seq().0, Ordering::Relaxed);
        if last_connection_poll.elapsed() >= std::time::Duration::from_millis(500) {
            if let Ok(connections) = sender.connection_count(std::time::Duration::from_millis(10)) {
                shared.set_status(NdiOutputStatus::Live {
                    connections,
                    source_name: config.source_name.clone(),
                })?;
            }
            last_connection_poll = std::time::Instant::now();
        }
    }
}

#[cfg(feature = "ndi-sdk")]
pub(super) fn map_runtime_error(error: &grafton_ndi::Error) -> NdiOutputError {
    // grafton-ndi 1.0.0 has InitializationFailed(String), but no variant that
    // proves the NDI runtime is absent. Even Io(NotFound) does not identify
    // which file failed. Classify at the known initialization stage instead
    // of guessing from words such as "library", "load", or "dll".
    NdiOutputError::new(
        NdiErrorCode::RuntimeInitFailed,
        format!("NDI runtime could not be initialized: {error}"),
    )
}
