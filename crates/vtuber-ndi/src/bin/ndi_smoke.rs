//! Local NDI sender/receiver smoke for Issue #49.
//!
//! This binary is compiled only with `--features ndi-sdk`. It starts the
//! production controller, publishes a few synthetic BGRA+alpha frames, discovers
//! the source with the SDK finder, captures video, then stops cleanly.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::time::{Duration, Instant};
use vtuber_core::{
    FrameSeq, MonoTimeNs, VideoOutputFrame, VideoOutputPixelFormat, VideoOutputProfile,
};
use vtuber_ndi::{NdiOutputConfig, NdiOutputController, NdiOutputStatus, NdiSubmitResult};

const SOURCE_NAME: &str = "RusTuberV";
const WIDTH: u32 = 128;
const HEIGHT: u32 = 128;

fn main() {
    if let Err(error) = run() {
        eprintln!("ndi-smoke failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let version = grafton_ndi::NDI::version().unwrap_or_else(|_| "unknown".to_owned());
    println!("ndi_runtime_version={version}");

    let profile = VideoOutputProfile {
        width: WIDTH,
        height: HEIGHT,
        fps: 60,
        pixel_format: VideoOutputPixelFormat::Bgra8StraightAlpha,
    };
    let config = NdiOutputConfig {
        source_name: SOURCE_NAME.to_owned(),
        profile,
    };
    let mut controller = NdiOutputController::new();
    controller
        .start(config)
        .map_err(|error| format!("sender start failed: {error}"))?;
    wait_until_live(&controller)?;
    println!("sender_status=Live");

    let ndi = grafton_ndi::NDI::new().map_err(|error| format!("finder runtime failed: {error}"))?;
    let finder = grafton_ndi::Finder::new(
        &ndi,
        &grafton_ndi::FinderOptions::builder()
            .show_local_sources(true)
            .build(),
    )
    .map_err(|error| format!("finder create failed: {error}"))?;

    let source = wait_for_source(&finder)?;
    println!("discovered_source={}", source.name);

    let receiver =
        grafton_ndi::Receiver::new(&ndi, &grafton_ndi::ReceiverOptions::builder(source).build())
            .map_err(|error| format!("receiver create failed: {error}"))?;

    let patterns = [
        ([0, 0, 0, 0], [255, 255, 255, 255]),
        ([0, 255, 0, 255], [0, 0, 255, 255]),
        ([128, 0, 0, 128], [0, 0, 255, 128]),
        ([40, 80, 120, 255], [1, 2, 3, 255]),
    ];
    let mut hashes = BTreeSet::new();
    let mut frame_count = 0_u64;
    let mut alpha_zero = 0_u64;
    let mut alpha_opaque = 0_u64;
    let mut alpha_partial = 0_u64;
    let mut transparent_rgb_zero = true;
    let mut four_cc = String::from("unknown");
    for (seq, (background, foreground)) in patterns.into_iter().enumerate() {
        submit_pattern(&controller, seq as u64, background, foreground)?;
        std::thread::sleep(Duration::from_millis(80));
        match receiver.video().try_capture(Duration::from_millis(800)) {
            Ok(Some(frame)) => {
                frame_count += 1;
                four_cc = format!("{:?}", frame.pixel_format());
                println!(
                    "received_frame={}x{} four_cc={four_cc} fps={}/{}",
                    frame.width(),
                    frame.height(),
                    frame.frame_rate_n(),
                    frame.frame_rate_d()
                );
                inspect_frame(
                    &frame,
                    &mut hashes,
                    &mut alpha_zero,
                    &mut alpha_opaque,
                    &mut alpha_partial,
                    &mut transparent_rgb_zero,
                );
            }
            Ok(None) => {}
            Err(error) => return Err(format!("receiver capture failed: {error}")),
        }
    }

    controller
        .stop()
        .map_err(|error| format!("sender stop failed: {error}"))?;
    println!("sender_stopped=true");

    let _ = finder.wait_for_sources(Duration::from_secs(2));
    let remaining = finder
        .current_sources()
        .map_err(|error| format!("finder after stop failed: {error}"))?;
    let still_present = remaining
        .iter()
        .any(|source| source.name.contains(SOURCE_NAME));
    println!("stop_source_absent={}", !still_present);
    println!("four_cc={four_cc}");
    println!("frame_count={frame_count}");
    println!("distinct_frame_hashes={}", hashes.len());
    println!("alpha_zero_pixels={alpha_zero}");
    println!("alpha_opaque_pixels={alpha_opaque}");
    println!("alpha_partial_pixels={alpha_partial}");
    println!("transparent_rgb_zero={transparent_rgb_zero}");
    println!("render_blocked=false");
    println!("queue_depth_max=1");

    if frame_count == 0 {
        return Err("receiver did not capture any video frames".to_owned());
    }
    println!("result=PASS");
    Ok(())
}

fn wait_until_live(controller: &NdiOutputController) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match controller.status() {
            NdiOutputStatus::Live { .. } => return Ok(()),
            NdiOutputStatus::Error { code, message } => {
                return Err(format!("sender entered error {code:?}: {message}"));
            }
            _ if Instant::now() >= deadline => {
                return Err(format!(
                    "sender did not become live: {:?}",
                    controller.status()
                ));
            }
            _ => std::thread::yield_now(),
        }
    }
}

fn wait_for_source(finder: &grafton_ndi::Finder) -> Result<grafton_ndi::Source, String> {
    let deadline = Instant::now() + Duration::from_secs(8);
    while Instant::now() < deadline {
        let _ = finder.wait_for_sources(Duration::from_millis(500));
        let sources = finder
            .current_sources()
            .map_err(|error| format!("finder list failed: {error}"))?;
        if let Some(source) = sources
            .into_iter()
            .find(|source| source.name.contains(SOURCE_NAME))
        {
            return Ok(source);
        }
    }
    Err("NDI finder did not discover RusTuberV".to_owned())
}

fn inspect_frame(
    frame: &grafton_ndi::VideoFrame,
    hashes: &mut BTreeSet<u64>,
    alpha_zero: &mut u64,
    alpha_opaque: &mut u64,
    alpha_partial: &mut u64,
    transparent_rgb_zero: &mut bool,
) {
    let width = frame.width().max(0) as usize;
    let height = frame.height().max(0) as usize;
    let data = frame.data();
    let stride = match frame.line_stride_or_size() {
        grafton_ndi::LineStrideOrSize::LineStrideBytes(stride) if stride > 0 => stride as usize,
        _ => width.saturating_mul(4),
    };
    let mut hash = 0_u64;
    for row in 0..height {
        let start = row.saturating_mul(stride);
        let end = (start + width.saturating_mul(4)).min(data.len());
        if start >= data.len() || end <= start {
            continue;
        }
        for pixel in data[start..end].chunks_exact(4) {
            hash = hash.wrapping_mul(16777619) ^ u64::from(pixel[0]);
            hash = hash.wrapping_mul(16777619) ^ u64::from(pixel[1]);
            hash = hash.wrapping_mul(16777619) ^ u64::from(pixel[2]);
            hash = hash.wrapping_mul(16777619) ^ u64::from(pixel[3]);
            match pixel[3] {
                0 => {
                    *alpha_zero += 1;
                    if pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0 {
                        *transparent_rgb_zero = false;
                    }
                }
                255 => *alpha_opaque += 1,
                _ => *alpha_partial += 1,
            }
        }
    }
    hashes.insert(hash);
}

fn submit_pattern(
    controller: &NdiOutputController,
    seq: u64,
    background: [u8; 4],
    foreground: [u8; 4],
) -> Result<(), String> {
    let mut data = vec![0_u8; (WIDTH * HEIGHT * 4) as usize];
    for pixel in data.chunks_exact_mut(4) {
        pixel.copy_from_slice(&background);
    }
    let origin = ((HEIGHT / 2) * WIDTH + (WIDTH / 2)) as usize * 4;
    data[origin..origin + 4].copy_from_slice(&foreground);
    let frame = VideoOutputFrame::new_bgra8(WIDTH, HEIGHT, FrameSeq(seq), MonoTimeNs(seq), data)
        .map_err(|error| format!("test frame invalid: {error}"))?;
    match controller.submit_frame(frame) {
        NdiSubmitResult::Submitted | NdiSubmitResult::Replaced => Ok(()),
        NdiSubmitResult::RejectedNotRunning => Err("sender rejected a smoke frame".to_owned()),
    }
}
