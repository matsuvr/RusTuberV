//! Native Syphon/Metal handles never escape this boundary or their owning worker.
#![cfg(target_os = "macos")]
use std::{ffi::c_void, marker::PhantomData, ptr::NonNull, rc::Rc};

unsafe extern "C" {
    fn rtv_syphon_create(out: *mut *mut c_void) -> i32;
    fn rtv_syphon_has_clients(sender: *mut c_void) -> i32;
    fn rtv_syphon_send(sender: *mut c_void, bytes: *const u8, width: u32, height: u32) -> i32;
    fn rtv_syphon_destroy(sender: *mut c_void);
}

/// Single-thread-owned sender. The native side copies pixels before returning.
pub struct Sender {
    handle: NonNull<c_void>,
    _not_send_sync: PhantomData<Rc<()>>,
}
impl Sender {
    /// Load the bundled framework and create a Metal publisher.
    pub fn new() -> Result<Self, String> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: valid out pointer; create returns an owned retained object on success.
        let status = unsafe { rtv_syphon_create(&mut raw) };
        if status != 0 {
            return Err(match status {
                1 => "Bundled Syphon.framework is missing or could not be loaded",
                2 => "SyphonMetalServer is unavailable in the bundled framework",
                _ => "Cannot initialize the Syphon Metal device/server",
            }
            .into());
        }
        let handle = NonNull::new(raw).ok_or("Syphon returned a null sender")?;
        Ok(Self {
            handle,
            _not_send_sync: PhantomData,
        })
    }
    /// Read the framework's client state independently of sending frames.
    pub fn has_clients(&self) -> bool {
        // SAFETY: the live handle is owned by this worker.
        unsafe { rtv_syphon_has_clients(self.handle.as_ptr()) != 0 }
    }

    /// Submit BGRA8 pixels; returns false when no client is connected.
    pub fn send(&mut self, bytes: &[u8], width: u32, height: u32) -> Result<bool, String> {
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(4));
        if width == 0 || height == 0 || expected != Some(bytes.len()) {
            return Err("Invalid Syphon BGRA frame size".into());
        }
        // SAFETY: owned live handle, validated buffer size; no pointer is retained.
        let status =
            unsafe { rtv_syphon_send(self.handle.as_ptr(), bytes.as_ptr(), width, height) };
        match status {
            0 => Ok(true),
            2 => Ok(false),
            _ => Err("Syphon Metal texture upload/publication failed".into()),
        }
    }
}
impl Drop for Sender {
    fn drop(&mut self) {
        // SAFETY: uniquely owned retained handle, released exactly once on this thread.
        unsafe { rtv_syphon_destroy(self.handle.as_ptr()) };
    }
}
