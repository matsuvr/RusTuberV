//! Candidate selection and local review sheets for labeling (Issue #52).
//!
//! The ARKit teacher is a proxy for finding interesting moments, never the
//! ground truth. Candidates deliberately include negatives (open, half, wink
//! and asymmetry) so false-close can be measured, not only high-blink peaks.

use std::fmt::Write as _;
use std::path::Path;

use image::{RgbImage, imageops};

use super::Options;
use super::decode::decode_rgb_bin;
use super::labels::{ExtractedData, FrameRow, csv_field};
use vtuber_tracking::EyeSide;

const CELL_WIDTH: u32 = 160;
const CELL_HEIGHT: u32 = 120;
const COLUMNS: u32 = 6;
const ROWS: u32 = 4;
const CANDIDATES_PER_TAKE: usize = 48;

/// Runs `eye-closure prepare-labels`.
pub(crate) fn run(options: &Options) -> Result<(), String> {
    let data_dir = options.data.as_deref().ok_or("missing --data")?;
    let data = ExtractedData::load(data_dir)?;
    std::fs::create_dir_all(&options.output)
        .map_err(|error| format!("failed to create {}: {error}", options.output.display()))?;

    let mut template = String::from("take_id,frame_seq,eye,label,label_source,tag,event_id\n");
    let mut proxy = String::from("take_id,frame_seq,eye,label,label_source,tag,event_id\n");
    let mut index = String::from("sheet,cell,take_id,frame_seq,reason\n");
    let mut report = String::from("# Eye-closure label review candidates\n\n");
    let mut sheet_number = 0_usize;
    let mut skipped_images = 0_usize;

    let takes: Vec<String> = {
        let mut ids: Vec<String> = data.takes.keys().cloned().collect();
        ids.sort();
        ids
    };
    for take_id in &takes {
        let frames: Vec<&FrameRow> = data
            .frames
            .iter()
            .filter(|frame| &frame.take_id == take_id)
            .collect();
        let candidates = select_candidates(&frames);
        let take = data.take(take_id);
        let _ = writeln!(
            report,
            "## {take_id}\n\n- frames: {}\n- candidates: {}\n- session: {:?}\n",
            frames.len(),
            candidates.len(),
            take.and_then(|take| take.session_id.as_deref())
        );
        for candidate in &candidates {
            for eye in [EyeSide::Left, EyeSide::Right] {
                let _ = writeln!(
                    template,
                    "{},{},{},,,{},{}",
                    csv_field(take_id),
                    candidate.frame_seq,
                    eye.as_str(),
                    csv_field(&candidate.reason),
                    csv_field(&format!(
                        "{take_id}:{}:{}",
                        candidate.frame_seq, candidate.reason
                    ))
                );
                if let Some(frame) = frames
                    .iter()
                    .find(|frame| frame.frame_seq == candidate.frame_seq)
                {
                    let arkit = match eye {
                        EyeSide::Left => frame.arkit_blink_left,
                        EyeSide::Right => frame.arkit_blink_right,
                    };
                    let (label, tag) = proxy_label(arkit);
                    let _ = writeln!(
                        proxy,
                        "{},{},{},{},arkit_proxy,{},proxy",
                        csv_field(take_id),
                        candidate.frame_seq,
                        eye.as_str(),
                        label,
                        tag
                    );
                }
            }
        }

        // Contact sheets are best-effort: a take without preserved raw pixels
        // still yields a candidate list and report.
        if let Some(take) = take
            && let Some(root) = take.raw_frames_root.as_deref()
        {
            let root = Path::new(root);
            let directory = options.output.join("sheets");
            for (sheet_index, chunk) in candidates.chunks((COLUMNS * ROWS) as usize).enumerate() {
                let mut montage = RgbImage::new(COLUMNS * CELL_WIDTH, ROWS * CELL_HEIGHT);
                let mut filled = 0_u32;
                for (cell, candidate) in chunk.iter().enumerate() {
                    let Some(frame) = frames
                        .iter()
                        .find(|frame| frame.frame_seq == candidate.frame_seq)
                    else {
                        continue;
                    };
                    let Some(reference) = frame.rgb_reference.as_deref() else {
                        continue;
                    };
                    let path = root.join(reference);
                    let (width, height, format) = declared_reference(frame);
                    let image = match decode_rgb_bin(
                        &path,
                        width,
                        height,
                        &format,
                        take.pixel_rotation_degrees,
                        take.mirrored,
                    ) {
                        Ok(image) => image,
                        Err(_) => {
                            skipped_images += 1;
                            continue;
                        }
                    };
                    let cell_image = eye_band(&image);
                    let cell_image = imageops::resize(
                        &cell_image,
                        CELL_WIDTH,
                        CELL_HEIGHT,
                        imageops::FilterType::Triangle,
                    );
                    paste(&mut montage, &cell_image, cell as u32);
                    let sheet = format!("{take_id}-sheet-{sheet_index:03}");
                    let _ = writeln!(
                        index,
                        "{},{},{},{},{}",
                        csv_field(&sheet),
                        cell,
                        csv_field(take_id),
                        candidate.frame_seq,
                        csv_field(&candidate.reason)
                    );
                    filled += 1;
                }
                if filled == 0 {
                    continue;
                }
                std::fs::create_dir_all(&directory).map_err(|error| {
                    format!("failed to create {}: {error}", directory.display())
                })?;
                let sheet_path = directory.join(format!("{take_id}-sheet-{sheet_number:03}.png"));
                montage
                    .save_with_format(&sheet_path, image::ImageFormat::Png)
                    .map_err(|error| {
                        format!("failed to write {}: {error}", sheet_path.display())
                    })?;
                sheet_number += 1;
            }
        }
    }

    write(&options.output.join("labels-template.csv"), &template)?;
    write(&options.output.join("arkit-proxy.csv"), &proxy)?;
    write(&options.output.join("review-index.csv"), &index)?;
    let _ = writeln!(
        report,
        "\nSkipped review crops (missing or undecodable raw pixels): {skipped_images}\n"
    );
    write(&options.output.join("report.md"), &report)?;
    println!(
        "wrote review candidates to {} ({} sheets)",
        options.output.display(),
        sheet_number
    );
    Ok(())
}

struct Candidate {
    frame_seq: u64,
    reason: String,
}

fn select_candidates(frames: &[&FrameRow]) -> Vec<Candidate> {
    let mut selected: Vec<(u64, String)> = Vec::new();
    let mut ordered: Vec<&FrameRow> = frames.to_vec();
    ordered.sort_by_key(|frame| frame.frame_seq);
    for (index, frame) in ordered.iter().enumerate() {
        let (Some(left), Some(right)) = (frame.arkit_blink_left, frame.arkit_blink_right) else {
            continue;
        };
        let max_blink = left.max(right);
        let min_blink = left.min(right);
        let asymmetry = (left - right).abs();
        let neighbors = || {
            (-3_i64..=3).filter_map(|offset| {
                let neighbor = index as i64 + offset;
                if neighbor < 0 {
                    return None;
                }
                ordered.get(neighbor as usize).and_then(|frame| {
                    match (frame.arkit_blink_left, frame.arkit_blink_right) {
                        (Some(left), Some(right)) => Some(left.max(right)),
                        _ => None,
                    }
                })
            })
        };
        let is_peak = max_blink >= 0.6 && neighbors().all(|value| max_blink >= value - 1e-6);
        let reason = if asymmetry >= 0.35 {
            Some(format!("asymmetry {asymmetry:.3}"))
        } else if is_peak {
            Some(format!("closed_peak {max_blink:.3}"))
        } else if (0.25..=0.55).contains(&max_blink) {
            Some(format!("half_open {max_blink:.3}"))
        } else if max_blink <= 0.15 && min_blink <= 0.05 {
            Some("open_negative".to_owned())
        } else if index % 60 == 0 {
            Some("interval_sample".to_owned())
        } else {
            None
        };
        if let Some(reason) = reason {
            selected.push((frame.frame_seq, reason));
        }
    }
    selected.dedup_by_key(|(seq, _)| *seq);
    if selected.len() > CANDIDATES_PER_TAKE {
        let step = selected.len() as f64 / CANDIDATES_PER_TAKE as f64;
        selected = (0..CANDIDATES_PER_TAKE)
            .filter_map(|index| selected.get((index as f64 * step) as usize).cloned())
            .collect();
    }
    selected
        .into_iter()
        .map(|(frame_seq, reason)| Candidate { frame_seq, reason })
        .collect()
}

fn proxy_label(arkit: Option<f32>) -> (&'static str, &'static str) {
    match arkit {
        None => ("unobservable", "missing_teacher"),
        Some(value) if value >= 0.6 => ("fully_closed", "arkit_high"),
        Some(value) if value <= 0.2 => ("not_closed", "arkit_low"),
        Some(_) => ("uncertain", "arkit_mid"),
    }
}

/// Declared dimensions and pixel format of a frame's RGB reference.
fn declared_reference(frame: &FrameRow) -> (u32, u32, String) {
    (
        frame.rgb_width_px.unwrap_or(1),
        frame.rgb_height_px.unwrap_or(1),
        frame
            .rgb_pixel_format
            .clone()
            .unwrap_or_else(|| "jpeg-rgb8-srgb".to_owned()),
    )
}

/// Crops the central eye band of an upright face.
fn eye_band(image: &RgbImage) -> RgbImage {
    let width = image.width() as f32;
    let height = image.height() as f32;
    let x0 = (width * 0.18).round() as u32;
    let y0 = (height * 0.22).round() as u32;
    let x1 = (width * 0.82).round() as u32;
    let y1 = (height * 0.52).round() as u32;
    let crop_width = x1
        .saturating_sub(x0)
        .max(1)
        .min(image.width().saturating_sub(x0).max(1));
    let crop_height = y1
        .saturating_sub(y0)
        .max(1)
        .min(image.height().saturating_sub(y0).max(1));
    imageops::crop_imm(image, x0, y0, crop_width, crop_height).to_image()
}

fn paste(montage: &mut RgbImage, cell: &RgbImage, cell_index: u32) {
    let column = cell_index % COLUMNS;
    let row = cell_index / COLUMNS;
    for y in 0..cell.height() {
        for x in 0..cell.width() {
            let target_x = column * CELL_WIDTH + x;
            let target_y = row * CELL_HEIGHT + y;
            if target_x < montage.width() && target_y < montage.height() {
                montage.put_pixel(target_x, target_y, *cell.get_pixel(x, y));
            }
        }
    }
}

fn write(path: &Path, text: &str) -> Result<(), String> {
    std::fs::write(path, text)
        .map_err(|error| format!("failed to write {}: {error}", path.display()))
}
