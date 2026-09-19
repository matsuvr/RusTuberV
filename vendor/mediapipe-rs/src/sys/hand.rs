//! Hand Landmarker C API raw bindings.
//!
//! `src/sys/bindings.rs` is generated from the vendored headers by bindgen, and
//! regenerating it requires libclang. The hand task was added to this fork
//! without that toolchain, so its handful of types and entry points are
//! declared here by hand instead. The layouts mirror
//! `vendor/headers/mediapipe/tasks/c/vision/hand_landmarker/*.h` and carry the
//! same compile-time size/offset assertions bindgen emits; if the bindings are
//! ever regenerated with the hand headers included, this module can be deleted.
//!
//! libmediapipe is loaded once by [`crate::loader`]; [`HandLib`] opens the same
//! library file a second time only to resolve the hand symbols, exactly like
//! the generated `MpLib` resolves its own.

use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;

use libloading::Library;

use super::{
    MpBaseOptions, MpCategories, MpImagePtr, MpImageProcessingOptions, MpLandmarks,
    MpNormalizedLandmarks, MpRunningMode, MpStatus,
};

/// Opaque `MpHandLandmarkerInternal*`.
pub type MpHandLandmarkerPtr = *mut c_void;

/// Live-stream callback type. Always `None` here: only video mode is used.
pub type MpHandLandmarkerOptionsResultCallbackFn = Option<
    unsafe extern "C" fn(
        status: MpStatus,
        result: *const MpHandLandmarkerResult,
        image: MpImagePtr,
        timestamp_ms: i64,
    ),
>;

/// `MpHandLandmarkerOptions` as laid out by current MediaPipe.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MpHandLandmarkerOptions {
    pub base_options: MpBaseOptions,
    pub running_mode: MpRunningMode,
    pub num_hands: c_int,
    pub min_hand_detection_confidence: f32,
    pub min_hand_presence_confidence: f32,
    pub min_tracking_confidence: f32,
    pub result_callback: MpHandLandmarkerOptionsResultCallbackFn,
}

/// `MpHandLandmarkerOptions` as laid out by v0.10.35 (no `file_descriptor`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MpHandLandmarkerOptionsV35 {
    pub base_options: super::compat::MpBaseOptionsV35,
    pub running_mode: MpRunningMode,
    pub num_hands: c_int,
    pub min_hand_detection_confidence: f32,
    pub min_hand_presence_confidence: f32,
    pub min_tracking_confidence: f32,
    pub result_callback: MpHandLandmarkerOptionsResultCallbackFn,
}

/// `MpHandLandmarkerResult`, byte-identical between the two supported ABIs.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MpHandLandmarkerResult {
    pub handedness: *mut MpCategories,
    pub handedness_count: u32,
    pub hand_landmarks: *mut MpNormalizedLandmarks,
    pub hand_landmarks_count: u32,
    pub hand_world_landmarks: *mut MpLandmarks,
    pub hand_world_landmarks_count: u32,
}

#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of MpHandLandmarkerResult"][size_of::<MpHandLandmarkerResult>() - 48usize];
    ["Alignment of MpHandLandmarkerResult"][align_of::<MpHandLandmarkerResult>() - 8usize];
    ["Offset of field: MpHandLandmarkerResult::handedness"]
        [std::mem::offset_of!(MpHandLandmarkerResult, handedness) - 0usize];
    ["Offset of field: MpHandLandmarkerResult::hand_landmarks"]
        [std::mem::offset_of!(MpHandLandmarkerResult, hand_landmarks) - 16usize];
    ["Offset of field: MpHandLandmarkerResult::hand_world_landmarks"]
        [std::mem::offset_of!(MpHandLandmarkerResult, hand_world_landmarks) - 32usize];
    ["Size of MpHandLandmarkerOptions"][size_of::<MpHandLandmarkerOptions>() - 104usize];
    ["Offset of field: MpHandLandmarkerOptions::running_mode"]
        [std::mem::offset_of!(MpHandLandmarkerOptions, running_mode) - 72usize];
    ["Offset of field: MpHandLandmarkerOptions::result_callback"]
        [std::mem::offset_of!(MpHandLandmarkerOptions, result_callback) - 96usize];
    ["Size of MpHandLandmarkerOptionsV35"][size_of::<MpHandLandmarkerOptionsV35>() - 88usize];
    ["Offset of field: MpHandLandmarkerOptionsV35::running_mode"]
        [std::mem::offset_of!(MpHandLandmarkerOptionsV35, running_mode) - 56usize];
};

type CreateFn = unsafe extern "C" fn(
    options: *mut MpHandLandmarkerOptions,
    landmarker: *mut MpHandLandmarkerPtr,
    error_msg: *mut *mut c_char,
) -> MpStatus;

type DetectForVideoFn = unsafe extern "C" fn(
    landmarker: MpHandLandmarkerPtr,
    image: MpImagePtr,
    options: *const MpImageProcessingOptions,
    timestamp_ms: i64,
    result: *mut MpHandLandmarkerResult,
    error_msg: *mut *mut c_char,
) -> MpStatus;

type CloseResultFn = unsafe extern "C" fn(result: *mut MpHandLandmarkerResult);

type CloseFn =
    unsafe extern "C" fn(landmarker: MpHandLandmarkerPtr, error_msg: *mut *mut c_char) -> MpStatus;

/// The four resolved hand symbols, plus the library handle that owns them.
pub struct HandLib {
    /// Keeps the second `dlopen` alive for as long as the function pointers are.
    _library: Library,
    create: CreateFn,
    detect_for_video: DetectForVideoFn,
    close_result: CloseResultFn,
    close: CloseFn,
}

impl HandLib {
    /// Opens `path` (the same libmediapipe the loader resolved) and resolves the
    /// hand entry points. Missing symbols are a load-time error here rather than
    /// a later panic, because unlike the generated `MpLib` this table is tiny.
    ///
    /// # Safety
    /// `path` must be the libmediapipe shared library this crate was built
    /// against; the returned function pointers are called with its ABI.
    pub unsafe fn load(path: &Path) -> Result<Self, libloading::Error> {
        // SAFETY: the caller guarantees `path` is the expected shared library;
        // loading only runs normal library initialisers.
        let library = unsafe { Library::new(path) }?;
        // SAFETY: each symbol is looked up under its exact C name; a missing
        // symbol is returned as an error instead of being called.
        let (create, detect_for_video, close_result, close) = unsafe {
            (
                *library.get(b"MpHandLandmarkerCreate\0")?,
                *library.get(b"MpHandLandmarkerDetectForVideo\0")?,
                *library.get(b"MpHandLandmarkerCloseResult\0")?,
                *library.get(b"MpHandLandmarkerClose\0")?,
            )
        };
        Ok(Self {
            _library: library,
            create,
            detect_for_video,
            close_result,
            close,
        })
    }

    /// # Safety
    /// `options` must point at a live options struct of the ABI the loaded
    /// library expects, and `error_msg` must be null or a valid out-pointer.
    pub unsafe fn create(
        &self,
        options: *mut MpHandLandmarkerOptions,
        landmarker: *mut MpHandLandmarkerPtr,
        error_msg: *mut *mut c_char,
    ) -> MpStatus {
        // SAFETY: the caller guarantees the pointer contract above.
        unsafe { (self.create)(options, landmarker, error_msg) }
    }

    /// # Safety
    /// `landmarker` must be a live handle, `result` writable, and `error_msg`
    /// null or a valid out-pointer.
    pub unsafe fn detect_for_video(
        &self,
        landmarker: MpHandLandmarkerPtr,
        image: MpImagePtr,
        options: *const MpImageProcessingOptions,
        timestamp_ms: i64,
        result: *mut MpHandLandmarkerResult,
        error_msg: *mut *mut c_char,
    ) -> MpStatus {
        // SAFETY: the caller guarantees the pointer contract above.
        unsafe {
            (self.detect_for_video)(landmarker, image, options, timestamp_ms, result, error_msg)
        }
    }

    /// # Safety
    /// `result` must be a result returned by `detect_for_video` and not yet
    /// closed.
    pub unsafe fn close_result(&self, result: *mut MpHandLandmarkerResult) {
        // SAFETY: the caller guarantees the result is live and closed once.
        unsafe { (self.close_result)(result) }
    }

    /// # Safety
    /// `landmarker` must be a live handle and `error_msg` null or a valid
    /// out-pointer.
    pub unsafe fn close(
        &self,
        landmarker: MpHandLandmarkerPtr,
        error_msg: *mut *mut c_char,
    ) -> MpStatus {
        // SAFETY: the caller guarantees the pointer contract above.
        unsafe { (self.close)(landmarker, error_msg) }
    }
}

impl std::fmt::Debug for HandLib {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HandLib").finish_non_exhaustive()
    }
}
