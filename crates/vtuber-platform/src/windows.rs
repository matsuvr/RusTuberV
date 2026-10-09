use std::ptr::null_mut;
use windows_sys::Win32::System::Power::{
    DEVICE_NOTIFY_SUBSCRIBE_PARAMETERS, HPOWERNOTIFY, PowerRegisterSuspendResumeNotification,
    PowerUnregisterSuspendResumeNotification,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    DEVICE_NOTIFY_CALLBACK, PBT_APMRESUMEAUTOMATIC, PBT_APMRESUMESUSPEND, PBT_APMSUSPEND,
};

/// Owns suspend/resume notifications without depending on window focus or HWND.
pub struct PowerNotifications {
    registration: HPOWERNOTIFY,
    _parameters: Box<DEVICE_NOTIFY_SUBSCRIBE_PARAMETERS>,
}

impl PowerNotifications {
    /// Registers the OS callback for the lifetime of the desktop event loop.
    pub fn new() -> std::io::Result<Self> {
        let mut parameters = Box::new(DEVICE_NOTIFY_SUBSCRIBE_PARAMETERS {
            Callback: Some(power_changed),
            Context: null_mut(),
        });
        let mut registration = null_mut();
        // SAFETY: the API receives a stable boxed subscription with the exact
        // callback ABI. It is retained through unregistration. No context is used.
        let error = unsafe {
            PowerRegisterSuspendResumeNotification(
                DEVICE_NOTIFY_CALLBACK,
                (&raw mut *parameters).cast(),
                &mut registration,
            )
        };
        if error != 0 {
            return Err(std::io::Error::from_raw_os_error(error as i32));
        }
        Ok(Self {
            registration: registration as HPOWERNOTIFY,
            _parameters: parameters,
        })
    }
}

unsafe extern "system" fn power_changed(
    _context: *const std::ffi::c_void,
    event: u32,
    _setting: *const std::ffi::c_void,
) -> u32 {
    match event {
        PBT_APMSUSPEND => crate::POWER.suspend(),
        PBT_APMRESUMEAUTOMATIC | PBT_APMRESUMESUSPEND => crate::POWER.resume(),
        _ => {}
    }
    0
}

impl Drop for PowerNotifications {
    fn drop(&mut self) {
        // SAFETY: this guard owns the registration returned by the matching API.
        // The callback uses only a static atomic, even if shutdown races delivery.
        unsafe {
            PowerUnregisterSuspendResumeNotification(self.registration);
        }
    }
}
