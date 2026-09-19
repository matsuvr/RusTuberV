//! Raw FFI. Generated bindings plus the one hand-written ABI compatibility shim.

#[allow(clippy::all)]
mod bindings;
pub mod compat;
pub mod hand;

pub use bindings::*;

/// Keeps a model source alive for the duration of a `*Create` call and hands
/// out the three raw fields `BaseOptions` wants.
///
/// `BaseOptions` takes buffer *and* path *and* fd and picks whichever is set, so
/// the unused ones must be null/zero. (`file_descriptor` uses `0` as "unset" —
/// see `base_options_converter.cc`.)
pub(crate) enum ModelHold {
    Path(std::ffi::CString),
    /// The length is stored rather than recomputed from the `Vec`, so the bounds
    /// check happens exactly once, at construction, and `parts` has no cast left
    /// to get wrong.
    Buf {
        bytes: Vec<u8>,
        len: std::os::raw::c_uint,
    },
}

impl ModelHold {
    pub(crate) fn new(src: &crate::types::ModelSource) -> crate::error::Result<Self> {
        Ok(match src {
            crate::types::ModelSource::Path(p) => {
                ModelHold::Path(std::ffi::CString::new(p.as_os_str().as_encoded_bytes())?)
            }
            crate::types::ModelSource::Bytes(b) => {
                // `model_asset_buffer_count` is a C `unsigned int`; truncating it
                // would tell MediaPipe the model is shorter than it is.
                let len = std::os::raw::c_uint::try_from(b.len()).map_err(|_| {
                    crate::error::Error::TooLarge {
                        what: "model buffer",
                        len: b.len(),
                        limit: u64::from(std::os::raw::c_uint::MAX),
                    }
                })?;
                ModelHold::Buf {
                    bytes: b.clone(),
                    len,
                }
            }
        })
    }

    /// `(model_asset_buffer, model_asset_buffer_count, model_asset_path)`
    pub(crate) fn parts(
        &self,
    ) -> (
        *const std::os::raw::c_char,
        std::os::raw::c_uint,
        *const std::os::raw::c_char,
    ) {
        match self {
            ModelHold::Path(p) => (std::ptr::null(), 0, p.as_ptr()),
            ModelHold::Buf { bytes, len } => (
                bytes.as_ptr().cast::<std::os::raw::c_char>(),
                *len,
                std::ptr::null(),
            ),
        }
    }
}

/// Which `libmediapipe` ABI the loaded library speaks.
///
/// MediaPipe renamed every C type to an `Mp*` prefix in commit `75c9711a3`
/// (2026-05-14), unreleased as of v0.10.35. The rename is cosmetic, but the same
/// change set inserted `int file_descriptor` into the middle of `BaseOptions`,
/// growing it 56 -> 72 bytes and shifting every field of every task options
/// struct that follows it. Headers from the wrong side of that commit still link
/// cleanly and then misread `running_mode`, so the version has to be detected.
///
/// Detection is free: the exported symbol sets differ. v0.10.35 exports
/// `MpInteractiveSegmenter*` and `MpLlm*`; post-rename builds export
/// `MpInteractiveSegmenterLegacy*` and drop the LLM entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Abi {
    /// v0.10.35 and earlier: `BaseOptions` is 56 bytes. See [`compat`].
    V0_10_35,
    /// Post-`75c9711a3`: `BaseOptions` is 72 bytes, matching the generated bindings.
    Renamed,
}
