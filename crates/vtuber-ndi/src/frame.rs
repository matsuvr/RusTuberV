//! Validation and mapping of the application's fixed NDI video format.
use crate::{NdiErrorCode, NdiOutputError};
use vtuber_core::{VideoOutputFrame, VideoOutputPixelFormat, VideoOutputProfile};

/// Timecode requesting per-send synthesis by the NDI SDK.
///
/// This is `NDIlib_send_timecode_synthesize` (`INT64_MAX`). A fixed `0` would
/// stamp every frame with the same epoch timecode, which freezes or stutters
/// receivers such as OBS. The send-side `timestamp` stays `0`, which is the
/// SDK default for sending.
pub const NDI_SEND_TIMECODE_SYNTHESIZE: i64 = i64::MAX;

/// A transport-neutral description of the NDI High Bandwidth video mapping.
///
/// This type intentionally uses no NDI SDK enum. It can be tested in the
/// normal SDK-free build and is the only descriptor passed into the optional
/// binding adapter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NdiVideoFrameMapping {
    /// Frame width in pixels.
    pub width: i32,
    /// Frame height in pixels.
    pub height: i32,
    /// Validated BGRA row stride in bytes.
    pub stride_bytes: i32,
    /// Frame-rate numerator.
    pub frame_rate_n: i32,
    /// Frame-rate denominator.
    pub frame_rate_d: i32,
    /// Square-pixel picture aspect ratio.
    pub picture_aspect_ratio: f32,
    /// Standard NDI FourCC selected by this mapping.
    pub four_cc: NdiFourCc,
}

/// FourCC values exposed by the transport-neutral mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NdiFourCc {
    /// NDI BGRA with a preserved alpha byte.
    Bgra,
}

/// Validates the #46 frame contract and maps it to standard NDI video fields.
///
/// `frame` is borrowed; the pixels are not copied or converted. The mapping
/// confirms the contract the transport already expects: the frame's pixel
/// format is BGRA8 straight alpha, and its width, height, stride
/// (`profile.width * 4`) and data length (`stride * profile.height`) all agree
/// with `profile`. The pixel format it produces is
/// [`NdiFourCc::Bgra`], the frame rate is `profile.fps` over a denominator of
/// 1, and the picture aspect ratio is the profile's width over its height.
///
/// This function performs no SDK call: it does not submit or send anything. An
/// `Ok` result means the frame satisfies the mapping contract, not that any
/// receiver got it. Handing a frame to the SDK is the sender's separate step,
/// and the output status and metrics describe that sender-side progress, not
/// confirmation that a receiver received the frame.
///
/// # Errors
///
/// Returns `InvalidConfiguration` with [`NdiOutputError`] when `profile` has a
/// zero width, height or fps, or a pixel format other than BGRA8 straight
/// alpha. Returns `InvalidFrame` when the frame's pixel format is not
/// BGRA8 straight alpha, when its width, height, stride or data length
/// disagrees with the profile, when the profile's stride or size is not
/// representable, or when the width, height, stride or fps does not fit the
/// NDI `i32` fields. No transport error code is produced here.
///
/// # Examples
///
/// ```
/// use vtuber_core::{FrameSeq, MonoTimeNs, VideoOutputFrame, VideoOutputProfile};
/// use vtuber_ndi::map_video_frame;
///
/// let profile = VideoOutputProfile::DEFAULT;
/// let bytes = profile.width as usize * profile.height as usize * 4;
/// let frame = VideoOutputFrame::new_bgra8(
///     profile.width,
///     profile.height,
///     FrameSeq(1),
///     MonoTimeNs(0),
///     vec![0; bytes],
/// )
/// .expect("the packed length matches the profile");
///
/// // Mapping validates the contract; it does not send the frame.
/// let mapping = map_video_frame(&frame, profile).expect("the frame matches the profile");
/// assert_eq!(mapping.four_cc, vtuber_ndi::NdiFourCc::Bgra);
/// ```
pub fn map_video_frame(
    frame: &VideoOutputFrame,
    profile: VideoOutputProfile,
) -> Result<NdiVideoFrameMapping, NdiOutputError> {
    validate_output_profile(profile)?;
    if frame.pixel_format() != VideoOutputPixelFormat::Bgra8StraightAlpha {
        return Err(NdiOutputError::new(
            NdiErrorCode::InvalidFrame,
            "frame pixel format is not BGRA8 straight alpha",
        ));
    }
    let stride = usize::try_from(profile.width)
        .ok()
        .and_then(|width| width.checked_mul(4))
        .ok_or_else(|| NdiOutputError::new(NdiErrorCode::InvalidFrame, "frame stride overflow"))?;
    let expected_len = stride
        .checked_mul(profile.height as usize)
        .ok_or_else(|| NdiOutputError::new(NdiErrorCode::InvalidFrame, "frame size overflow"))?;
    if frame.width() != profile.width
        || frame.height() != profile.height
        || frame.stride_bytes() != stride
        || frame.data().len() != expected_len
    {
        return Err(NdiOutputError::new(
            NdiErrorCode::InvalidFrame,
            "frame dimensions, stride, or data length do not match the output profile",
        ));
    }
    let width = i32::try_from(frame.width()).map_err(|_| {
        NdiOutputError::new(NdiErrorCode::InvalidFrame, "frame width exceeds NDI range")
    })?;
    let height = i32::try_from(frame.height()).map_err(|_| {
        NdiOutputError::new(NdiErrorCode::InvalidFrame, "frame height exceeds NDI range")
    })?;
    let stride_bytes = i32::try_from(frame.stride_bytes()).map_err(|_| {
        NdiOutputError::new(NdiErrorCode::InvalidFrame, "frame stride exceeds NDI range")
    })?;
    let frame_rate_n = i32::try_from(profile.fps).map_err(|_| {
        NdiOutputError::new(NdiErrorCode::InvalidFrame, "frame rate exceeds NDI range")
    })?;
    Ok(NdiVideoFrameMapping {
        width,
        height,
        stride_bytes,
        frame_rate_n,
        frame_rate_d: 1,
        picture_aspect_ratio: profile.width as f32 / profile.height as f32,
        four_cc: NdiFourCc::Bgra,
    })
}

pub(crate) fn validate_output_profile(profile: VideoOutputProfile) -> Result<(), NdiOutputError> {
    if profile.width == 0 || profile.height == 0 || profile.fps == 0 {
        return Err(NdiOutputError::new(
            NdiErrorCode::InvalidConfiguration,
            "output width, height, and fps must be non-zero",
        ));
    }
    if profile.pixel_format != VideoOutputPixelFormat::Bgra8StraightAlpha {
        return Err(NdiOutputError::new(
            NdiErrorCode::InvalidConfiguration,
            "only BGRA8 straight-alpha output is supported",
        ));
    }
    Ok(())
}
