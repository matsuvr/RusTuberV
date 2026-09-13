//! Pixel decode and orientation handling for eye-closure inputs (Issue #51).
//!
//! The RGB writer stores bytes verbatim, so the extension alone never decides
//! the format. This module checks the JPEG signature, validates declared
//! dimensions against the real header, and applies an explicit read-time
//! rotation/mirror without ever writing the source file.

use std::path::Path;

use image::{RgbImage, imageops};

/// Normalizes an explicit rotation to `0`, `90`, `180`, or `270` degrees.
///
/// # Errors
///
/// Rejects any other angle instead of silently rounding it.
pub(crate) fn normalize_rotation(rotation_degrees: i32) -> Result<u16, String> {
    match rotation_degrees.rem_euclid(360) {
        0 => Ok(0),
        90 => Ok(90),
        180 => Ok(180),
        270 => Ok(270),
        other => Err(format!("unsupported rotation correction: {other} degrees")),
    }
}

/// Decodes one stored `.bin` frame into an upright, unmirrored RGB image.
///
/// # Errors
///
/// Returns a message naming `path` for read/decode failures, unknown pixel
/// formats, dimension disagreements with the declared metadata, and
/// unsupported rotations.
pub(crate) fn decode_rgb_bin(
    path: &Path,
    declared_width: u32,
    declared_height: u32,
    declared_format: &str,
    rotation_degrees: i32,
    mirrored: bool,
) -> Result<RgbImage, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    if !is_jpeg(&bytes) {
        return Err(format!(
            "{}: pixel_format {declared_format:?} is not a decodable JPEG; raw BGRA is not present in verified data",
            path.display()
        ));
    }
    let decoded = image::load_from_memory(&bytes)
        .map_err(|error| format!("failed to decode JPEG {}: {error}", path.display()))?
        .to_rgb8();
    if decoded.width() != declared_width || decoded.height() != declared_height {
        return Err(format!(
            "{}: decoded header is {}x{}, metadata declares {declared_width}x{declared_height}",
            path.display(),
            decoded.width(),
            decoded.height()
        ));
    }
    let rotated = match normalize_rotation(rotation_degrees)? {
        0 => decoded,
        90 => imageops::rotate90(&decoded),
        180 => imageops::rotate180(&decoded),
        270 => imageops::rotate270(&decoded),
        other => {
            return Err(format!(
                "{}: unsupported rotation correction {other}",
                path.display()
            ));
        }
    };
    Ok(if mirrored {
        imageops::flip_horizontal(&rotated)
    } else {
        rotated
    })
}

/// Returns `true` when `bytes` starts with a JPEG `SOI` marker.
pub(crate) fn is_jpeg(bytes: &[u8]) -> bool {
    matches!(bytes.get(0..3), Some([0xFF, 0xD8, 0xFF]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageFormat, Rgb};

    fn sample_image(width: u32, height: u32) -> RgbImage {
        let mut image = RgbImage::new(width, height);
        // Distinct top-left pixel exposes rotation and mirroring.
        image.put_pixel(0, 0, Rgb([255, 0, 0]));
        image
    }

    fn write_jpeg(directory: &Path, name: &str, image: &RgbImage) -> std::path::PathBuf {
        let path = directory.join(name);
        image
            .save_with_format(&path, ImageFormat::Jpeg)
            .expect("test writes jpeg");
        path
    }

    #[test]
    fn jpeg_signature_is_detected_without_trusting_the_extension() {
        assert!(is_jpeg(&[0xFF, 0xD8, 0xFF, 0xE0]));
        assert!(!is_jpeg(b"not jpeg"));
        assert!(!is_jpeg(&[]));
    }

    #[test]
    fn decodes_declared_dimensions_and_rejects_a_mismatch() {
        let directory = tempfile::tempdir().expect("tempdir");
        let image = sample_image(16, 8);
        let path = write_jpeg(directory.path(), "frame.bin", &image);

        let decoded = decode_rgb_bin(&path, 16, 8, "jpeg-rgb8-srgb", 0, false).expect("decodes");
        assert_eq!((decoded.width(), decoded.height()), (16, 8));

        let error = decode_rgb_bin(&path, 8, 16, "jpeg-rgb8-srgb", 0, false).expect_err("mismatch");
        assert!(error.contains("metadata declares 8x16"), "{error}");
    }

    #[test]
    fn unsupported_format_is_rejected_instead_of_guessed() {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("frame.bin");
        std::fs::write(&path, [0u8, 1, 2, 3, 4]).expect("write raw bytes");
        let error =
            decode_rgb_bin(&path, 16, 8, "bgra8888", 0, false).expect_err("raw bgra is unknown");
        assert!(error.contains("bgra8888"), "{error}");
    }

    #[test]
    fn all_four_rotations_swap_dimensions_only_when_expected() {
        let directory = tempfile::tempdir().expect("tempdir");
        let image = sample_image(16, 8);
        let path = write_jpeg(directory.path(), "frame.bin", &image);

        for (rotation, expected) in [(0, (16, 8)), (90, (8, 16)), (180, (16, 8)), (270, (8, 16))] {
            let decoded =
                decode_rgb_bin(&path, 16, 8, "jpeg-rgb8-srgb", rotation, false).expect("decodes");
            assert_eq!(
                (decoded.width(), decoded.height()),
                expected,
                "rotation {rotation}"
            );
        }
        assert!(decode_rgb_bin(&path, 16, 8, "jpeg-rgb8-srgb", 45, false).is_err());
    }

    #[test]
    fn mirror_flips_pixels_without_changing_dimensions() {
        let directory = tempfile::tempdir().expect("tempdir");
        let image = sample_image(4, 4);
        let path = write_jpeg(directory.path(), "frame.bin", &image);

        let plain = decode_rgb_bin(&path, 4, 4, "jpeg-rgb8-srgb", 0, false).expect("decodes");
        let mirrored = decode_rgb_bin(&path, 4, 4, "jpeg-rgb8-srgb", 0, true).expect("decodes");
        assert_eq!(plain.dimensions(), mirrored.dimensions());
        assert_eq!(
            plain.get_pixel(0, 0),
            mirrored.get_pixel(plain.width() - 1, 0),
            "only the horizontal axis moves"
        );
    }
}
