//! Hand landmarks through the pinned MediaPipe Tasks C API.
//!
//! Only the video path is implemented, for the same reason as
//! [`crate::pose_landmarker`]: the observed-arm worker runs one synchronous
//! `detect_for_video` call per camera frame, so image and live-stream modes and
//! async callbacks are deliberately absent.
//!
//! The raw types live in [`crate::sys::hand`] because this task was added to
//! the fork after the generated bindings were produced; see that module.

use std::num::NonZeroU32;
use std::ptr;

use crate::error::{Error, Result};
use crate::image::Image;
use crate::loader::lib;
use crate::sys::hand as raw;
use crate::sys::{self, Abi, ModelHold};
use crate::types::{
    Category, Confidence, Delegate, IouThreshold, ModelSource, NormalizedLandmark, Timestamp,
    WorldLandmark,
};

/// Hands found in one frame.
///
/// The three vectors are parallel: index `i` of each describes the same hand.
/// World coordinates are in meters in the hand's own camera-aligned basis, not
/// normalized image units; they carry the palm plane's orientation, not a
/// body-relative position.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HandLandmarkerResult {
    /// Handedness classification per hand, as reported by MediaPipe.
    pub handedness: Vec<Vec<Category>>,
    /// 21 landmarks per hand in normalized image coordinates.
    pub hand_landmarks: Vec<Vec<NormalizedLandmark>>,
    /// 21 landmarks per hand in world coordinates, in meters.
    pub hand_world_landmarks: Vec<Vec<WorldLandmark>>,
}

impl HandLandmarkerResult {
    /// # Safety
    /// `raw` must be a live `MpHandLandmarkerResult`.
    unsafe fn from_raw(raw: &raw::MpHandLandmarkerResult) -> Result<Self> {
        let handedness = if raw.handedness.is_null() {
            Vec::new()
        } else {
            (0..raw.handedness_count as usize)
                .map(|h| {
                    // SAFETY: the index is below the count MediaPipe reported for this array.
                    let set = unsafe { &*raw.handedness.add(h) };
                    // SAFETY: a field of the live result this function's contract covers.
                    unsafe { Category::vec_from_raw(set.categories, set.categories_count) }
                })
                .collect()
        };

        let hand_landmarks = if raw.hand_landmarks.is_null() {
            Vec::new()
        } else {
            (0..raw.hand_landmarks_count as usize)
                .map(|h| {
                    // SAFETY: the index is below the count MediaPipe reported for this array.
                    let set = unsafe { &*raw.hand_landmarks.add(h) };
                    if set.landmarks.is_null() {
                        return Vec::new();
                    }
                    (0..set.landmarks_count as usize)
                        // SAFETY: the index is below the count MediaPipe reported for this array.
                        .map(|i| unsafe { NormalizedLandmark::from_raw(&*set.landmarks.add(i)) })
                        .collect()
                })
                .collect()
        };

        let hand_world_landmarks = if raw.hand_world_landmarks.is_null() {
            Vec::new()
        } else {
            (0..raw.hand_world_landmarks_count as usize)
                .map(|h| {
                    // SAFETY: the index is below the count MediaPipe reported for this array.
                    let set = unsafe { &*raw.hand_world_landmarks.add(h) };
                    if set.landmarks.is_null() {
                        return Vec::new();
                    }
                    (0..set.landmarks_count as usize)
                        // SAFETY: the index is below the count MediaPipe reported for this array.
                        .map(|i| unsafe { WorldLandmark::from_raw(&*set.landmarks.add(i)) })
                        .collect()
                })
                .collect()
        };

        Ok(HandLandmarkerResult {
            handedness,
            hand_landmarks,
            hand_world_landmarks,
        })
    }
}

#[derive(Debug, Clone)]
pub struct HandLandmarkerBuilder {
    source: ModelSource,
    delegate: Delegate,
    num_hands: NonZeroU32,
    min_hand_detection_confidence: Confidence,
    min_hand_presence_confidence: Confidence,
    min_tracking_confidence: IouThreshold,
}

impl HandLandmarkerBuilder {
    /// Chooses the inference backend.
    pub fn delegate(mut self, delegate: Delegate) -> Self {
        self.delegate = delegate;
        self
    }

    /// Maximum number of hands to detect.
    pub fn num_hands(mut self, n: NonZeroU32) -> Self {
        self.num_hands = n;
        self
    }

    /// Minimum score for the palm detector stage.
    pub fn min_hand_detection_confidence(mut self, v: Confidence) -> Self {
        self.min_hand_detection_confidence = v;
        self
    }

    /// Minimum score that a hand is present in the tracked region.
    pub fn min_hand_presence_confidence(mut self, v: Confidence) -> Self {
        self.min_hand_presence_confidence = v;
        self
    }

    /// How much a newly detected hand box must overlap the previous frame's box.
    pub fn min_tracking_confidence(mut self, v: IouThreshold) -> Self {
        self.min_tracking_confidence = v;
        self
    }

    /// Builds a landmarker in video mode. Timestamps must increase between calls.
    pub fn build_for_video(self) -> Result<HandLandmarkerVideo> {
        self.build_mode(sys::MpRunningMode::MP_RUNNING_MODE_VIDEO)
            .map(HandLandmarkerVideo)
    }

    fn build_mode(self, running_mode: sys::MpRunningMode) -> Result<HandLandmarker> {
        let min_hand_detection_confidence = self.min_hand_detection_confidence.get();
        let min_hand_presence_confidence = self.min_hand_presence_confidence.get();
        let min_tracking_confidence = self.min_tracking_confidence.get();

        let shared = lib()?;
        // SAFETY: the path is the libmediapipe the shared loader resolved.
        let hand_lib = unsafe { raw::HandLib::load(shared.source.path()) }.map_err(|source| {
            Error::Load {
                path: shared.source.path().to_path_buf(),
                source,
            }
        })?;
        let hold = ModelHold::new(&self.source)?;
        let (buf, buf_count, path) = hold.parts();
        let delegate = self.delegate.to_raw();
        // The C side takes this as an `int`; requesting more hands than fit is
        // meaningless rather than an error worth a variant.
        let num_hands = i32::try_from(self.num_hands.get()).unwrap_or(i32::MAX);

        let mut ptr: raw::MpHandLandmarkerPtr = ptr::null_mut();
        let mut err = ptr::null_mut();

        // SAFETY: `hold` keeps the model alive across the call; the options
        // layout is chosen from the probed ABI (see sys::compat).
        let status = unsafe {
            match shared.abi {
                Abi::Renamed => {
                    let mut opts = raw::MpHandLandmarkerOptions {
                        base_options: sys::MpBaseOptions {
                            model_asset_buffer: buf,
                            model_asset_buffer_count: buf_count,
                            model_asset_path: path,
                            file_descriptor: 0,
                            delegate,
                            host_environment: sys::MpHostEnvironment::MP_HOST_ENVIRONMENT_UNKNOWN,
                            host_system: sys::MpHostSystem::MP_HOST_SYSTEM_UNKNOWN,
                            host_version: ptr::null(),
                            ca_bundle_path: ptr::null(),
                            app_id: ptr::null(),
                            app_version: ptr::null(),
                        },
                        running_mode,
                        num_hands,
                        min_hand_detection_confidence,
                        min_hand_presence_confidence,
                        min_tracking_confidence,
                        result_callback: None,
                    };
                    hand_lib.create(&mut opts, &mut ptr, &mut err)
                }
                Abi::V0_10_35 => {
                    let mut opts = raw::MpHandLandmarkerOptionsV35 {
                        base_options: sys::compat::MpBaseOptionsV35::new(
                            buf, buf_count, path, delegate,
                        ),
                        running_mode,
                        num_hands,
                        min_hand_detection_confidence,
                        min_hand_presence_confidence,
                        min_tracking_confidence,
                        result_callback: None,
                    };
                    hand_lib.create(
                        (&raw mut opts).cast::<raw::MpHandLandmarkerOptions>(),
                        &mut ptr,
                        &mut err,
                    )
                }
            }
        };

        if status != sys::MpStatus::kMpOk {
            // SAFETY: `err` is the out-param of the call that just failed; it is read and freed exactly once here.
            return Err(unsafe { Error::from_status(&shared.raw, status, err) });
        }
        Ok(HandLandmarker {
            lib: hand_lib,
            ptr,
            closed: false,
        })
    }
}

/// Owns a `MpHandLandmarkerPtr`. Built only through [`HandLandmarker::builder`].
pub struct HandLandmarker {
    lib: raw::HandLib,
    ptr: raw::MpHandLandmarkerPtr,
    closed: bool,
}

// SAFETY: as FaceLandmarker — movable between threads, never shared.
unsafe impl Send for HandLandmarker {}

impl HandLandmarker {
    /// Starts building a hand landmarker from a model source.
    pub fn builder(source: ModelSource) -> HandLandmarkerBuilder {
        HandLandmarkerBuilder {
            source,
            delegate: Delegate::default(),
            num_hands: NonZeroU32::new(2).expect("2 should be non-zero"),
            // MediaPipe's own defaults.
            min_hand_detection_confidence: Confidence::HALF,
            min_hand_presence_confidence: Confidence::HALF,
            min_tracking_confidence: IouThreshold::HALF,
        }
    }

    fn close_once(&mut self) -> Result<()> {
        if std::mem::replace(&mut self.closed, true) {
            return Ok(());
        }
        let shared = lib()?;
        let mut err = ptr::null_mut();
        // SAFETY: runs exactly once; joins any worker thread.
        let status = unsafe { self.lib.close(self.ptr, &mut err) };
        if status != sys::MpStatus::kMpOk {
            // SAFETY: `err` is the out-param of the call that just failed; it is read and freed exactly once here.
            return Err(unsafe { Error::from_status(&shared.raw, status, err) });
        }
        Ok(())
    }
}

impl std::fmt::Debug for HandLandmarker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HandLandmarker")
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}

impl Drop for HandLandmarker {
    fn drop(&mut self) {
        let _ = self.close_once();
    }
}

/// A [`HandLandmarker`] running in video mode.
pub struct HandLandmarkerVideo(HandLandmarker);

impl HandLandmarkerVideo {
    /// Runs hand detection on one decoded video frame.
    ///
    /// Timestamps must increase between calls. The returned result owns all of
    /// its landmarks; MediaPipe frees the C result before this returns.
    pub fn detect_for_video(
        &mut self,
        image: &Image,
        timestamp: Timestamp,
    ) -> Result<HandLandmarkerResult> {
        let shared = lib()?;
        // SAFETY: the struct is only pointers and counts, so all-zero is a valid empty result; MediaPipe overwrites it.
        let mut result: raw::MpHandLandmarkerResult = unsafe { std::mem::zeroed() };
        let mut err = ptr::null_mut();

        // SAFETY: live handle and image; `result` goes straight back to
        // MpHandLandmarkerCloseResult.
        let status = unsafe {
            self.0.lib.detect_for_video(
                self.0.ptr,
                image.ptr,
                ptr::null(),
                timestamp.as_millis(),
                &mut result,
                &mut err,
            )
        };

        if status != sys::MpStatus::kMpOk {
            // SAFETY: `err` is the out-param of the call that just failed; it is read and freed exactly once here.
            return Err(unsafe { Error::from_status(&shared.raw, status, err) });
        }

        // Convert first, free second — and free even if conversion failed, as
        // in the other landmarkers.
        // SAFETY: `result` was filled by the call above and is still live.
        let converted = unsafe { HandLandmarkerResult::from_raw(&result) };
        // SAFETY: `result` was filled by the Detect call above, everything has
        // been copied out, and it is freed once.
        unsafe { self.0.lib.close_result(&mut result) };
        converted
    }
}

impl std::fmt::Debug for HandLandmarkerVideo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HandLandmarkerVideo")
            .field("landmarker", &self.0)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_hands_is_an_empty_result() {
        let raw = raw::MpHandLandmarkerResult {
            handedness: ptr::null_mut(),
            handedness_count: 0,
            hand_landmarks: ptr::null_mut(),
            hand_landmarks_count: 0,
            hand_world_landmarks: ptr::null_mut(),
            hand_world_landmarks_count: 0,
        };
        // SAFETY: `raw` is a fully initialised empty result with no pointers for
        // `from_raw` to follow.
        let result = unsafe { HandLandmarkerResult::from_raw(&raw) }.unwrap();
        assert!(result.handedness.is_empty());
        assert!(result.hand_landmarks.is_empty());
        assert!(result.hand_world_landmarks.is_empty());
    }

    #[test]
    fn handedness_and_world_landmarks_copy_without_borrowing() {
        let mut name = *b"Right\0";
        let mut categories = [sys::MpCategory {
            index: 1,
            score: 0.9,
            category_name: name.as_mut_ptr().cast(),
            display_name: ptr::null_mut(),
        }];
        let mut handedness = [sys::MpCategories {
            categories: categories.as_mut_ptr(),
            categories_count: 1,
        }];
        let mut world = [sys::MpLandmark {
            x: 0.1,
            y: -0.2,
            z: 0.3,
            has_visibility: false,
            visibility: 0.0,
            has_presence: false,
            presence: 0.0,
            name: ptr::null_mut(),
        }];
        let mut world_sets = [sys::MpLandmarks {
            landmarks: world.as_mut_ptr(),
            landmarks_count: 1,
        }];
        let raw = raw::MpHandLandmarkerResult {
            handedness: handedness.as_mut_ptr(),
            handedness_count: 1,
            hand_landmarks: ptr::null_mut(),
            hand_landmarks_count: 0,
            hand_world_landmarks: world_sets.as_mut_ptr(),
            hand_world_landmarks_count: 1,
        };
        // SAFETY: `raw` points at the live arrays above for this call, and
        // `from_raw` only copies out of them.
        let result = unsafe { HandLandmarkerResult::from_raw(&raw) }.unwrap();
        assert_eq!(result.handedness.len(), 1);
        assert_eq!(result.handedness[0][0].category_name.as_deref(), Some("Right"));
        assert_eq!(result.hand_world_landmarks.len(), 1);
        assert_eq!(
            result.hand_world_landmarks[0][0].point.to_array(),
            [0.1, -0.2, 0.3]
        );
        assert_eq!(result.hand_world_landmarks[0][0].visibility, None);
    }
}
