//! Live-stream plumbing.
//!
//! MediaPipe's async result callback is a bare function pointer with **no**
//! `void* user_data` field, so there is nothing to hang per-instance context on.
//! Python gets away with this because ctypes/libffi mints a fresh trampoline per
//! callable; Rust has no stable equivalent. Instead this module monomorphises a
//! fixed pool of trampolines over a const generic, and each live stream claims
//! one slot for its lifetime.

use std::os::raw::c_void;
use std::panic::AssertUnwindSafe;
use std::sync::Mutex;

use crate::error::{Error, Result};
use crate::image::Image;
use crate::sys;
use crate::types::{Rotation, Timestamp};

/// ponytail: 8 concurrent live-stream tasks. Raise this or move to libffi
/// closures if anyone ever needs more; the C API gives us no per-callback
/// context to key off, so a fixed pool is the price.
const SLOT_COUNT: usize = 8;

type RawCallback = unsafe extern "C" fn(sys::MpStatus, *const c_void, sys::MpImagePtr, i64);

/// Type-erased per-slot dispatcher: converts the C result and calls the user's
/// closure. Built by each task module, which knows the concrete result type.
pub(crate) type Dispatch =
    Box<dyn FnMut(sys::MpStatus, *const c_void, sys::MpImagePtr, i64) + Send>;

static SLOTS: [Mutex<Option<Dispatch>>; SLOT_COUNT] = [const { Mutex::new(None) }; SLOT_COUNT];

/// A claimed pool entry. Releasing it is what stops callbacks being delivered.
pub(crate) struct Slot(usize);

impl Slot {
    pub(crate) fn claim(dispatch: Dispatch) -> Result<Self> {
        for (i, slot) in SLOTS.iter().enumerate() {
            let mut guard = slot.lock().unwrap_or_else(|e| e.into_inner());
            if guard.is_none() {
                *guard = Some(dispatch);
                return Ok(Slot(i));
            }
        }
        Err(Error::NoFreeSlot(SLOT_COUNT))
    }

    pub(crate) fn raw_callback(&self) -> RawCallback {
        TRAMPOLINES[self.0]
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        *SLOTS[self.0].lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

/// One monomorphisation per `N`, so `TRAMPOLINES` holds eight distinct function
/// pointers without a declarative macro.
unsafe extern "C" fn trampoline<const N: usize>(
    status: sys::MpStatus,
    result: *const c_void,
    image: sys::MpImagePtr,
    timestamp_ms: i64,
) {
    // Unwinding across an FFI boundary aborts the process. Swallow panics from
    // user code; the alternative is taking the whole program down from a
    // MediaPipe worker thread.
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let mut guard = SLOTS[N].lock().unwrap_or_else(|e| e.into_inner());
        if let Some(dispatch) = guard.as_mut() {
            dispatch(status, result, image, timestamp_ms);
        }
        // A `None` slot means the stream was dropped between MediaPipe queueing
        // this callback and running it. Nothing to do.
    }));
}

static TRAMPOLINES: [RawCallback; SLOT_COUNT] = [
    trampoline::<0>,
    trampoline::<1>,
    trampoline::<2>,
    trampoline::<3>,
    trampoline::<4>,
    trampoline::<5>,
    trampoline::<6>,
    trampoline::<7>,
];

/// The per-task half of a live stream: how to push a frame and how to shut down.
///
/// Public only because [`Stream`] is generic over it; the `stream` module is
/// private, so this is effectively sealed to the task types in this crate.
pub trait AsyncTask: Send {
    fn send_raw(
        &mut self,
        image: &Image,
        opts: Option<&sys::MpImageProcessingOptions>,
        timestamp_ms: i64,
    ) -> Result<()>;

    fn close(&mut self) -> Result<()>;
}

/// A running live-stream task.
///
/// Results arrive on a MediaPipe worker thread, not the thread that created the
/// stream. Timestamps must strictly increase; MediaPipe drops out-of-order frames.
///
/// Dropping the stream closes the underlying task, which flushes and joins the
/// worker before the callback slot is released — so no callback can fire after
/// the drop returns. For that same reason, dropping a stream *from inside its
/// own callback* deadlocks; don't.
pub struct Stream<T: AsyncTask> {
    task: T,
    // Held purely for its Drop, which releases the callback slot. Declared after
    // `task` so field-drop order releases it only after `Drop::drop` has closed
    // the task and joined its worker.
    _slot: Slot,
}

impl<T: AsyncTask> Stream<T> {
    pub(crate) fn new(task: T, slot: Slot) -> Self {
        Stream { task, _slot: slot }
    }

    /// Queues a frame. Returns as soon as it is accepted; the result arrives via
    /// the callback.
    pub fn send(&mut self, image: &Image, timestamp: Timestamp) -> Result<()> {
        self.task.send_raw(image, None, timestamp.as_millis())
    }

    /// As [`send`](Self::send), applying a clockwise rotation first.
    pub fn send_rotated(
        &mut self,
        image: &Image,
        rotation: Rotation,
        timestamp: Timestamp,
    ) -> Result<()> {
        let raw = rotation.to_raw();
        self.task.send_raw(image, Some(&raw), timestamp.as_millis())
    }
}

impl<T: AsyncTask + std::fmt::Debug> std::fmt::Debug for Stream<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stream")
            .field("task", &self.task)
            .field("slot", &self._slot.0)
            .finish()
    }
}

impl<T: AsyncTask> Drop for Stream<T> {
    fn drop(&mut self) {
        // Close first: it joins the worker, guaranteeing no callback is running
        // by the time the slot is cleared.
        let _ = self.task.close();
    }
}
