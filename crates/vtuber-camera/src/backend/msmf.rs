//! Windows MSMF camera backend using `nokhwa`.
//!
//! The native `nokhwa::Camera` object is constructed, opened, used, and
//! dropped entirely within the capture worker thread. It is never sent across
//! threads. The stream is deliberately not `Send`; it never crosses the
//! capture worker boundary.

use nokhwa::Camera;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{
    ApiBackend, CameraFormat as NokhwaFormat, CameraIndex, CameraInfo, FrameFormat, Resolution,
};
use vtuber_core::{FrameSeq, MonoTimeNs, PixelFormat, StopToken, VideoFrame};

use crate::device::{
    CameraBackend, CameraDescriptor, CameraError, CameraFormat, CameraRequest, CameraStream,
};
use crate::format::{FormatCandidate, select_format};

/// Windows MSMF camera backend.
pub struct MsmfBackend;

impl MsmfBackend {
    /// Creates a new MSMF backend.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for MsmfBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CameraBackend for MsmfBackend {
    fn enumerate(&self) -> Result<Vec<CameraDescriptor>, CameraError> {
        let devices = nokhwa::query(ApiBackend::MediaFoundation)
            .map_err(|e| map_nokhwa_error(CameraOperation::Enumerate, e))?;

        devices.into_iter().map(descriptor_from_info).collect()
    }

    fn open(
        &self,
        descriptor: &CameraDescriptor,
        request: &CameraRequest,
    ) -> Result<Box<dyn CameraStream>, CameraError> {
        let cam_index = parse_msmf_device_id(&descriptor.id)?;

        // Create camera with a default format to query capabilities.
        let default_fmt = NokhwaFormat::new(Resolution::new(640, 480), FrameFormat::MJPEG, 30);
        let mut camera = Camera::with_backend(
            cam_index,
            nokhwa::utils::RequestedFormat::with_formats(
                nokhwa::utils::RequestedFormatType::Closest(default_fmt),
                &[FrameFormat::MJPEG, FrameFormat::YUYV],
            ),
            ApiBackend::MediaFoundation,
        )
        .map_err(|e| map_nokhwa_error(CameraOperation::Open, e))?;

        // Enumerate available formats and pick the best match.
        let candidates = enumerate_format_candidates(&mut camera)?;
        let chosen = select_format(request, &candidates)?;

        // Apply the chosen format.
        let nokhwa_fmt = to_nokhwa_format(&chosen);
        #[allow(deprecated)]
        camera
            .set_camera_format(nokhwa_fmt)
            .map_err(|e| map_nokhwa_error(CameraOperation::Open, e))?;

        camera
            .open_stream()
            .map_err(|e| map_nokhwa_error(CameraOperation::Open, e))?;

        let source_format = camera.frame_format();

        Ok(Box::new(MsmfStream {
            camera,
            format: chosen,
            seq: 0,
            source_format,
        }))
    }
}

/// Builds a descriptor whose identity is the MSMF symbolic device link.
fn descriptor_from_info(info: CameraInfo) -> Result<CameraDescriptor, CameraError> {
    let symbolic_link = info.misc();
    if symbolic_link.is_empty() {
        return Err(CameraError::EnumFailed(format!(
            "MSMF device `{}` has no symbolic link",
            info.human_name()
        )));
    }
    Ok(CameraDescriptor {
        id: format!("msmf:{symbolic_link}"),
        label: info.human_name(),
    })
}

/// Parse the stable MSMF symbolic link from a descriptor id.
fn parse_msmf_device_id(id: &str) -> Result<CameraIndex, CameraError> {
    let symbolic_link = id
        .strip_prefix("msmf:")
        .filter(|value| !value.is_empty())
        .ok_or_else(|| CameraError::OpenFailed(format!("not an MSMF descriptor id: {id}")))?;
    Ok(CameraIndex::String(symbolic_link.to_owned()))
}

/// Enumerate format candidates from an open camera.
fn enumerate_format_candidates(camera: &mut Camera) -> Result<Vec<FormatCandidate>, CameraError> {
    let mut candidates = Vec::new();

    // Enumerate MJPEG formats.
    if let Ok(formats) = camera.compatible_list_by_resolution(FrameFormat::MJPEG) {
        for (resolution, fps_list) in &formats {
            for &fps in fps_list {
                candidates.push(FormatCandidate {
                    width: resolution.width(),
                    height: resolution.height(),
                    fps_numerator: fps,
                    fps_denominator: 1,
                    format: PixelFormat::Rgb8,
                });
            }
        }
    }

    // Enumerate YUYV formats.
    if let Ok(formats) = camera.compatible_list_by_resolution(FrameFormat::YUYV) {
        for (resolution, fps_list) in &formats {
            for &fps in fps_list {
                candidates.push(FormatCandidate {
                    width: resolution.width(),
                    height: resolution.height(),
                    fps_numerator: fps,
                    fps_denominator: 1,
                    format: PixelFormat::Bgr8,
                });
            }
        }
    }

    if candidates.is_empty() {
        return Err(CameraError::NoSuitableFormat);
    }

    Ok(candidates)
}

/// Convert our [`CameraFormat`] to a nokhwa [`NokhwaFormat`].
fn to_nokhwa_format(format: &CameraFormat) -> NokhwaFormat {
    let frame_format = match format.format {
        PixelFormat::Rgb8 => FrameFormat::MJPEG,
        PixelFormat::Bgr8 => FrameFormat::YUYV,
        _ => FrameFormat::MJPEG,
    };
    NokhwaFormat::new(
        Resolution::new(format.width, format.height),
        frame_format,
        format.fps_numerator / format.fps_denominator.max(1),
    )
}

/// The camera operation that produced a nokhwa error.
///
/// The stage is known at every call site, so it is passed in instead of being
/// recovered from the error text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CameraOperation {
    Enumerate,
    Open,
    ReadFrame,
    Stop,
}

/// Classifies a nokhwa error by operation stage.
///
/// nokhwa 0.10.11 exposes no structured cause: every `NokhwaError` payload is a
/// `String` built from a Windows error's `Display`, no variant keeps the HRESULT
/// or a `source()`, and the MSMF `stop_stream` wrapper is infallible. A refused
/// permission and a removed device are therefore not determinable here, so this
/// never claims [`CameraError::PermissionDenied`] or
/// [`CameraError::Disconnected`] from wording. It keeps the operation stage and
/// the original text and leaves the cause unclassified. The text is recorded
/// for diagnosis only; it is never searched or parsed again.
fn map_nokhwa_error(operation: CameraOperation, error: nokhwa::NokhwaError) -> CameraError {
    // The stage, not the message, selects the variant, so the same failure at
    // a different stage stays a different error. This is the single
    // normalisation point for nokhwa errors, so a stop failure is not wrapped
    // twice.
    let detail = error.to_string();
    match operation {
        CameraOperation::Enumerate => CameraError::EnumFailed(detail),
        CameraOperation::Open => CameraError::OpenFailed(detail),
        CameraOperation::ReadFrame => CameraError::FrameReadFailed(detail),
        CameraOperation::Stop => CameraError::StopFailed(detail),
    }
}

/// An opened MSMF camera stream.
///
/// The native camera is constructed, used, and dropped on the capture worker.
pub struct MsmfStream {
    camera: Camera,
    format: CameraFormat,
    seq: u64,
    source_format: FrameFormat,
}

impl CameraStream for MsmfStream {
    fn actual_format(&self) -> CameraFormat {
        self.format
    }

    fn next_frame(&mut self, stop: &StopToken) -> Result<VideoFrame, CameraError> {
        if stop.is_stopped() {
            return Err(CameraError::Disconnected);
        }

        // A frame that could not be read is not a decode failure, so it gets its
        // own variant and no decoder is run for it.
        let buffer = self
            .camera
            .frame()
            .map_err(|e| map_nokhwa_error(CameraOperation::ReadFrame, e))?;

        self.seq += 1;
        let now = vtuber_core::monotonic_now().0;

        let (data, pixel_format, stride) = decode_frame(&buffer, self.source_format, &self.format)?;

        Ok(VideoFrame {
            seq: FrameSeq(self.seq),
            captured_at: MonoTimeNs(now),
            width: self.format.width,
            height: self.format.height,
            stride_bytes: stride,
            format: pixel_format,
            data: data.into(),
        })
    }

    fn stop(&mut self) -> Result<(), CameraError> {
        self.camera
            .stop_stream()
            .map_err(|e| map_nokhwa_error(CameraOperation::Stop, e))
    }
}

/// Decode a nokhwa buffer into raw pixel data.
///
/// A format this decoder cannot read returns an error, so a truncated or
/// unsupported buffer is never published as a frame.
fn decode_frame(
    buffer: &nokhwa::Buffer,
    source_format: FrameFormat,
    format: &CameraFormat,
) -> Result<(Vec<u8>, PixelFormat, usize), CameraError> {
    match source_format {
        FrameFormat::MJPEG => {
            let decoded = buffer
                .decode_image::<RgbFormat>()
                .map_err(|e| CameraError::FrameDecodeFailed(format!("MJPEG decode: {e}")))?;
            let rgb = decoded.into_raw();
            let stride = format.width as usize * 3;
            Ok((rgb, PixelFormat::Rgb8, stride))
        }
        FrameFormat::YUYV => {
            let rgb = yuyv_to_rgb(buffer.buffer(), format.width, format.height)?;
            let stride = format.width as usize * 3;
            Ok((rgb, PixelFormat::Rgb8, stride))
        }
        FrameFormat::RAWRGB => {
            let data = buffer.buffer().to_vec();
            let stride = format.width as usize * 3;
            Ok((data, PixelFormat::Rgb8, stride))
        }
        FrameFormat::RAWBGR => {
            let data = buffer.buffer().to_vec();
            let stride = format.width as usize * 3;
            Ok((data, PixelFormat::Bgr8, stride))
        }
        FrameFormat::GRAY => {
            let data = buffer.buffer().to_vec();
            let stride = format.width as usize;
            Ok((data, PixelFormat::Gray8, stride))
        }
        FrameFormat::NV12 => Err(CameraError::FrameDecodeFailed(
            "NV12 not yet supported".into(),
        )),
    }
}

/// The packed YUY2 row layout this decoder cannot read: two pixels share one
/// four-byte `Y0 U Y1 V` group, so a row must be an even number of pixels wide.
fn odd_yuyv_row(width: u32) -> CameraError {
    CameraError::FrameDecodeFailed(format!(
        "YUYV rows must hold whole pixel pairs, got width {width}"
    ))
}

/// The packed YUY2 byte length for these dimensions is not representable.
fn yuyv_length_overflow(width: u32, height: u32) -> CameraError {
    CameraError::FrameDecodeFailed(format!(
        "YUYV byte length is not representable for {width}x{height}"
    ))
}

/// Convert packed YUYV (YUY2) data to interleaved RGB.
///
/// Each row is `width / 2` four-byte pixel pairs, so the buffer must hold at
/// least `width * 2 * height` bytes and the rows are read without ever crossing
/// a row boundary. Bytes past that length are ignored. A short buffer, an odd
/// width, or a byte length that is not representable is a decode error; the
/// decoder never pads a truncated buffer into a successful black frame.
fn yuyv_to_rgb(yuyv: &[u8], width: u32, height: u32) -> Result<Vec<u8>, CameraError> {
    let Ok(row_pixels) = usize::try_from(width) else {
        return Err(yuyv_length_overflow(width, height));
    };
    let Ok(rows) = usize::try_from(height) else {
        return Err(yuyv_length_overflow(width, height));
    };
    if row_pixels % 2 != 0 {
        return Err(odd_yuyv_row(width));
    }
    let Some(row_bytes) = row_pixels.checked_mul(2) else {
        return Err(yuyv_length_overflow(width, height));
    };
    let Some(required) = row_bytes.checked_mul(rows) else {
        return Err(yuyv_length_overflow(width, height));
    };
    if yuyv.len() < required {
        return Err(CameraError::FrameDecodeFailed(format!(
            "YUYV needs {required} bytes for {width}x{height}, got {}",
            yuyv.len()
        )));
    }
    let Some(output_len) = row_pixels.checked_mul(rows).and_then(|n| n.checked_mul(3)) else {
        return Err(yuyv_length_overflow(width, height));
    };

    let mut rgb = vec![0u8; output_len];
    // An even width makes every row a whole number of four-byte groups, so the
    // flat group order and the flat RGB pixel order stay in lockstep. `zip`
    // stops at the output side, which holds exactly `required / 4` pairs, so
    // surplus input bytes are left unread.
    let (groups, _) = yuyv.as_chunks::<4>();
    let (pixels, _) = rgb.as_chunks_mut::<3>();
    let (pixel_pairs, _) = pixels.as_chunks_mut::<2>();
    for ([y0, u_byte, y1, v_byte], pair) in groups.iter().zip(pixel_pairs) {
        let u = f32::from(*u_byte) - 128.0;
        let v = f32::from(*v_byte) - 128.0;
        let [first, second] = pair;
        yuv_to_rgb_pixel(f32::from(*y0), u, v, first);
        yuv_to_rgb_pixel(f32::from(*y1), u, v, second);
    }
    Ok(rgb)
}

fn yuv_to_rgb_pixel(y: f32, u: f32, v: f32, out: &mut [u8; 3]) {
    out[0] = (y + 1.402 * v).clamp(0.0, 255.0) as u8;
    out[1] = (y - 0.344_136 * u - 0.714_136 * v).clamp(0.0, 255.0) as u8;
    out[2] = (y + 1.772 * u).clamp(0.0, 255.0) as u8;
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )] // tests may panic (AGENTS.md)
    use super::*;

    #[test]
    fn descriptor_identity_uses_symbolic_link_not_enumeration_index() {
        let info = CameraInfo::new(
            "C922",
            "MediaFoundation Camera",
            r"\\?\usb#vid_046d&pid_085c",
            CameraIndex::Index(7),
        );
        let descriptor = descriptor_from_info(info).unwrap();
        assert_eq!(descriptor.id, r"msmf:\\?\usb#vid_046d&pid_085c");
        assert!(!descriptor.id.contains(":7:"));
    }

    #[test]
    fn parse_msmf_device_id_returns_symbolic_identity() {
        assert_eq!(
            parse_msmf_device_id(r"msmf:\\?\usb#vid_046d&pid_085c").unwrap(),
            CameraIndex::String(r"\\?\usb#vid_046d&pid_085c".to_owned())
        );
        assert!(parse_msmf_device_id("msmf:").is_err());
        assert!(parse_msmf_device_id("avf:0").is_err());
    }

    #[test]
    fn yuyv_to_rgb_produces_correct_size() {
        let yuyv = vec![128, 128, 128, 128];
        let rgb = yuyv_to_rgb(&yuyv, 2, 1).expect("one whole pixel pair decodes");
        assert_eq!(rgb.len(), 6);
    }

    #[test]
    fn yuyv_to_rgb_decodes_every_row_of_a_multi_row_pair() {
        // Two rows of two pixels; the second row must not be read as a
        // continuation of the first.
        let yuyv = [
            10, 100, 20, 150, // row 0: luma 10 and 20
            200, 100, 210, 150, // row 1: luma 200 and 210
        ];
        let rgb = yuyv_to_rgb(&yuyv, 2, 2).expect("two whole rows decode");
        assert_eq!(rgb.len(), 12);

        let expected = |y: u8| {
            let u = 100.0 - 128.0;
            let v = 150.0 - 128.0;
            let y = f32::from(y);
            [
                (y + 1.402 * v).clamp(0.0, 255.0) as u8,
                (y - 0.344_136 * u - 0.714_136 * v).clamp(0.0, 255.0) as u8,
                (y + 1.772 * u).clamp(0.0, 255.0) as u8,
            ]
        };
        assert_eq!(&rgb[0..3], &expected(10));
        assert_eq!(&rgb[3..6], &expected(20));
        assert_eq!(&rgb[6..9], &expected(200));
        assert_eq!(&rgb[9..12], &expected(210));
    }

    #[test]
    fn yuyv_to_rgb_accepts_a_real_black_frame() {
        // Luma 0 with centred chroma is a genuine black frame and must succeed,
        // unlike a truncated buffer.
        let yuyv = [0, 128, 0, 128];
        assert_eq!(
            yuyv_to_rgb(&yuyv, 2, 1).expect("black is a valid frame"),
            [0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn yuyv_to_rgb_rejects_input_shorter_than_one_row() {
        assert!(matches!(
            yuyv_to_rgb(&[], 2, 1),
            Err(CameraError::FrameDecodeFailed(_))
        ));
    }

    #[test]
    fn yuyv_to_rgb_rejects_input_one_byte_short() {
        let yuyv = [128, 128, 128];
        assert!(matches!(
            yuyv_to_rgb(&yuyv, 2, 1),
            Err(CameraError::FrameDecodeFailed(_))
        ));
    }

    #[test]
    fn yuyv_to_rgb_rejects_an_unsupported_row_layout() {
        // A single-pixel row has no YUY2 partner inside the row, and reading it
        // as if it continued into the next row is not supported.
        assert!(matches!(
            yuyv_to_rgb(&[128, 128, 128, 128], 1, 2),
            Err(CameraError::FrameDecodeFailed(_))
        ));
    }

    #[test]
    fn yuyv_to_rgb_ignores_bytes_past_the_required_length() {
        let yuyv = [0, 128, 0, 128, 7, 7, 7, 7];
        assert_eq!(
            yuyv_to_rgb(&yuyv, 2, 1).expect("surplus bytes are not rejected"),
            [0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn decode_frame_propagates_a_short_yuyv_buffer() {
        let format = CameraFormat {
            width: 2,
            height: 1,
            fps_numerator: 30,
            fps_denominator: 1,
            format: PixelFormat::Bgr8,
        };
        let buffer = nokhwa::Buffer::new(Resolution::new(2, 1), &[], FrameFormat::YUYV);
        assert!(matches!(
            decode_frame(&buffer, FrameFormat::YUYV, &format),
            Err(CameraError::FrameDecodeFailed(_))
        ));
    }

    #[test]
    fn decode_frame_decodes_a_complete_yuyv_buffer() {
        let format = CameraFormat {
            width: 2,
            height: 1,
            fps_numerator: 30,
            fps_denominator: 1,
            format: PixelFormat::Bgr8,
        };
        let buffer =
            nokhwa::Buffer::new(Resolution::new(2, 1), &[0, 128, 0, 128], FrameFormat::YUYV);
        let (data, pixel_format, stride) =
            decode_frame(&buffer, FrameFormat::YUYV, &format).expect("complete buffer decodes");
        assert_eq!(data, vec![0, 0, 0, 0, 0, 0]);
        assert_eq!(pixel_format, PixelFormat::Rgb8);
        assert_eq!(stride, 6);
    }

    #[test]
    fn yuv_to_rgb_pixel_clamps() {
        let mut out = [0u8; 3];
        // Y=0, u=0, v=0 (centered chroma) → black.
        yuv_to_rgb_pixel(0.0, 0.0, 0.0, &mut out);
        assert_eq!(out, [0, 0, 0]);

        // Y=255 with extreme chroma should not panic and produces valid output.
        yuv_to_rgb_pixel(255.0, 127.0, 127.0, &mut out);
        // Just verify the function completed without panicking.
        let _ = out;
    }

    #[test]
    fn to_nokhwa_format_rgb_maps_to_mjpeg() {
        let format = CameraFormat {
            width: 1280,
            height: 720,
            fps_numerator: 30,
            fps_denominator: 1,
            format: PixelFormat::Rgb8,
        };
        let nf = to_nokhwa_format(&format);
        assert_eq!(nf.width(), 1280);
        assert_eq!(nf.height(), 720);
        assert_eq!(nf.format(), FrameFormat::MJPEG);
    }

    #[test]
    fn the_operation_stage_selects_the_error_variant() {
        let cases = [
            (CameraOperation::Enumerate, "enum"),
            (CameraOperation::Open, "open"),
            (CameraOperation::ReadFrame, "read"),
            (CameraOperation::Stop, "stop"),
        ];
        for (operation, text) in cases {
            let error = nokhwa::NokhwaError::GeneralError(text.to_owned());
            let mapped = map_nokhwa_error(operation, error);
            let expected = match operation {
                CameraOperation::Enumerate => "CAMERA_ENUM_FAILED",
                CameraOperation::Open => "CAMERA_OPEN_FAILED",
                CameraOperation::ReadFrame => "CAMERA_FRAME_READ_FAILED",
                CameraOperation::Stop => "CAMERA_STOP_FAILED",
            };
            assert!(
                mapped.to_string().starts_with(expected),
                "{operation:?} produced {mapped:?}"
            );
        }
    }

    #[test]
    fn permission_and_disconnect_wording_is_not_reparsed() {
        // nokhwa gives no structured cause, so wording that used to be
        // substring-matched must now stay inside the stage's own variant.
        for text in [
            "Access is denied.",
            "The device has been removed.",
            "The device was disconnected.",
        ] {
            let open = map_nokhwa_error(
                CameraOperation::Open,
                nokhwa::NokhwaError::GeneralError(text.to_owned()),
            );
            assert!(matches!(open, CameraError::OpenFailed(_)), "{text}");
            let read = map_nokhwa_error(
                CameraOperation::ReadFrame,
                nokhwa::NokhwaError::GeneralError(text.to_owned()),
            );
            assert!(matches!(read, CameraError::FrameReadFailed(_)), "{text}");
        }
    }

    #[test]
    fn the_nokhwa_variant_is_carried_into_the_same_stage_variant() {
        // The variant is type information nokhwa does provide, so it is kept
        // as the diagnostic text together with its original wording.
        let mapped = map_nokhwa_error(
            CameraOperation::ReadFrame,
            nokhwa::NokhwaError::ReadFrameError("The device has been removed.".to_owned()),
        );
        assert!(
            matches!(mapped, CameraError::FrameReadFailed(_)),
            "{mapped:?}"
        );
        assert!(
            mapped.to_string().contains("The device has been removed."),
            "{mapped}"
        );
    }

    #[test]
    fn a_read_failure_stays_distinct_from_a_decode_failure() {
        let read = map_nokhwa_error(
            CameraOperation::ReadFrame,
            nokhwa::NokhwaError::GeneralError("busy".into()),
        );
        let decode = decode_frame(
            &nokhwa::Buffer::new(Resolution::new(1, 1), &[], FrameFormat::NV12),
            FrameFormat::NV12,
            &CameraFormat {
                width: 1,
                height: 1,
                fps_numerator: 30,
                fps_denominator: 1,
                format: PixelFormat::Bgr8,
            },
        );
        assert!(matches!(read, CameraError::FrameReadFailed(_)));
        assert!(matches!(decode, Err(CameraError::FrameDecodeFailed(_))));
    }
}
