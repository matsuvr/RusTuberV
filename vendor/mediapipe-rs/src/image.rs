use std::ffi::CString;
use std::marker::PhantomData;
use std::path::Path;
use std::ptr;

use crate::error::{Error, Result};
use crate::loader::lib;
use crate::sys;
use crate::types::Size;

fn checked_image_len(size: Size, channels: u32) -> Result<usize> {
    let overflow = || Error::ImageSizeOverflow {
        width: size.width,
        height: size.height,
        channels,
    };
    let width = usize::try_from(size.width).map_err(|_| overflow())?;
    let height = usize::try_from(size.height).map_err(|_| overflow())?;
    let channels = usize::try_from(channels).map_err(|_| overflow())?;
    width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or_else(overflow)
}

/// An image owned by this crate, freed on drop.
///
/// MediaPipe copies pixel data on construction, so the source buffer does not
/// need to outlive the `Image`.
pub struct Image {
    pub(crate) ptr: sys::MpImagePtr,
}

// SAFETY: MpImage is a plain owned buffer handle. It is not internally
// synchronised, so it is Send but not Sync.
unsafe impl Send for Image {}

impl Image {
    /// Decodes an image file. MediaPipe handles the decoding (PNG/JPEG/…), so
    /// this needs no image crate.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let lib = lib()?;
        let path = CString::new(path.as_ref().as_os_str().as_encoded_bytes())?;
        let mut ptr: sys::MpImagePtr = ptr::null_mut();
        let mut err = ptr::null_mut();
        // SAFETY: `path` is a valid NUL-terminated string that outlives the call;
        // `ptr`/`err` are valid out-params.
        let status = unsafe {
            lib.raw
                .MpImageCreateFromFile(path.as_ptr(), &mut ptr, &mut err)
        };
        if status != sys::MpStatus::kMpOk {
            // SAFETY: `err` is the out-param of the call that just failed; it is read and freed exactly once here.
            return Err(unsafe { Error::from_status(&lib.raw, status, err) });
        }
        Ok(Image { ptr })
    }

    /// Tightly packed 8-bit RGB, `width * height * 3` bytes.
    pub fn from_rgb(size: Size, data: &[u8]) -> Result<Self> {
        Self::from_u8(size, data, 3, sys::MpImageFormat::kMpImageFormatSrgb)
    }

    /// Tightly packed 8-bit RGBA, `width * height * 4` bytes.
    pub fn from_rgba(size: Size, data: &[u8]) -> Result<Self> {
        Self::from_u8(size, data, 4, sys::MpImageFormat::kMpImageFormatSrgba)
    }

    fn from_u8(size: Size, data: &[u8], channels: u32, format: sys::MpImageFormat) -> Result<Self> {
        // The C side trusts the length it is given; a short buffer is an
        // out-of-bounds read inside MediaPipe, so check here.
        let expected = checked_image_len(size, channels)?;
        if data.len() != expected {
            return Err(Error::BufferSize {
                got: data.len(),
                expected,
                width: size.width,
                height: size.height,
                channels,
            });
        }

        // `MpImageCreateFromUint8Data` takes the length as a C `int`, and reads
        // exactly that many bytes. A silent truncation here would hand MediaPipe
        // a length longer than the buffer on the next wrap-around, so check
        // rather than cast.
        let too_large = |_| Error::TooLarge {
            what: "pixel buffer",
            len: data.len(),
            limit: i32::MAX as u64,
        };
        let len = i32::try_from(data.len()).map_err(too_large)?;
        // Implied by the check above, but converted rather than cast so the
        // reasoning does not live only in a comment.
        let width = i32::try_from(size.width).map_err(too_large)?;
        let height = i32::try_from(size.height).map_err(too_large)?;

        let lib = lib()?;
        let mut ptr: sys::MpImagePtr = ptr::null_mut();
        let mut err = ptr::null_mut();
        // SAFETY: `data` is at least `expected` bytes, which is what we declare;
        // MediaPipe copies it before returning.
        let status = unsafe {
            lib.raw.MpImageCreateFromUint8Data(
                format,
                width,
                height,
                data.as_ptr(),
                len,
                &mut ptr,
                &mut err,
            )
        };
        if status != sys::MpStatus::kMpOk {
            // SAFETY: `err` is the out-param of the call that just failed; it is read and freed exactly once here.
            return Err(unsafe { Error::from_status(&lib.raw, status, err) });
        }
        Ok(Image { ptr })
    }

    pub fn size(&self) -> Size {
        self.as_ref().size()
    }

    pub fn channels(&self) -> u32 {
        self.as_ref().channels()
    }

    /// A non-owning view, for passing around without transferring ownership.
    pub fn as_ref(&self) -> ImageRef<'_> {
        ImageRef {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }
}

impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let size = self.size();
        f.debug_struct("Image")
            .field("width", &size.width)
            .field("height", &size.height)
            .field("channels", &self.channels())
            .finish()
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        // If the library failed to load we never created an image.
        if let Ok(lib) = lib() {
            // SAFETY: `ptr` came from an MpImageCreate* call and is freed once.
            unsafe { lib.raw.MpImageFree(self.ptr) };
        }
    }
}

/// A borrowed image that this crate does **not** own.
///
/// This exists because of how live-stream callbacks work: the `MpImagePtr`
/// handed to a result callback points at a stack `MpImageInternal` constructed
/// inside MediaPipe's own lambda (see `face_landmarker.cc`). Calling
/// `MpImageFree` on it would free a stack address. `ImageRef` therefore has no
/// `Drop`, and the lifetime keeps it from escaping the callback.
#[derive(Clone, Copy)]
pub struct ImageRef<'a> {
    pub(crate) ptr: sys::MpImagePtr,
    pub(crate) _marker: PhantomData<&'a ()>,
}

impl ImageRef<'_> {
    pub(crate) fn from_raw(ptr: sys::MpImagePtr) -> Self {
        ImageRef {
            ptr,
            _marker: PhantomData,
        }
    }

    pub fn size(&self) -> Size {
        let Ok(lib) = lib() else {
            return Size {
                width: 0,
                height: 0,
            };
        };
        // SAFETY: `ptr` is a live MpImage for the lifetime of this view.
        unsafe {
            Size {
                width: lib.raw.MpImageGetWidth(self.ptr).max(0) as u32,
                height: lib.raw.MpImageGetHeight(self.ptr).max(0) as u32,
            }
        }
    }

    pub fn channels(&self) -> u32 {
        let Ok(lib) = lib() else { return 0 };
        // SAFETY: as above.
        unsafe { lib.raw.MpImageGetChannels(self.ptr).max(0) as u32 }
    }
}

impl std::fmt::Debug for ImageRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let size = self.size();
        f.debug_struct("ImageRef")
            .field("width", &size.width)
            .field("height", &size.height)
            .field("channels", &self.channels())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_lengths_preserve_rgb_and_rgba_layouts() {
        let size = Size { width: 640, height: 480 };
        assert_eq!(checked_image_len(size, 3).unwrap(), 640 * 480 * 3);
        assert_eq!(checked_image_len(size, 4).unwrap(), 640 * 480 * 4);
    }

    #[test]
    fn overflow_is_rejected_before_buffer_validation_or_ffi() {
        let size = Size { width: u32::MAX, height: u32::MAX };
        assert!(matches!(
            checked_image_len(size, 4),
            Err(Error::ImageSizeOverflow { width: u32::MAX, height: u32::MAX, channels: 4 })
        ));
        assert!(matches!(Image::from_rgba(size, &[]), Err(Error::ImageSizeOverflow { .. })));
    }

    #[test]
    fn incorrect_buffer_length_keeps_its_existing_error() {
        assert!(matches!(
            Image::from_rgb(Size { width: 2, height: 3 }, &[0; 17]),
            Err(Error::BufferSize { got: 17, expected: 18, width: 2, height: 3, channels: 3 })
        ));
    }

    #[test]
    fn c_dimension_overflow_is_rejected_without_allocating_or_loading_ffi() {
        for size in [
            Size { width: u32::MAX, height: 0 },
            Size { width: 0, height: u32::MAX },
        ] {
            assert!(matches!(Image::from_rgb(size, &[]), Err(Error::TooLarge { .. })));
        }
    }
}
