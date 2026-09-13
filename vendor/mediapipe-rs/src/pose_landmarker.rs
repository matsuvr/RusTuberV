//! Pose world landmarks through the pinned MediaPipe Tasks C API.
//!
//! Only the video path is implemented: the pose worker runs one synchronous
//! `detect_for_video` call per camera frame, so image and live-stream modes,
//! segmentation masks, and async callbacks are deliberately absent rather than
//! scaffolding for callers that do not exist.

use std::num::NonZeroU32;
use std::ptr;

use crate::error::{Error, Result};
use crate::image::Image;
use crate::loader::lib;
use crate::sys::{self, Abi, ModelHold};
use crate::types::{
    Confidence, Delegate, IouThreshold, ModelSource, NormalizedLandmark, Timestamp, WorldLandmark,
};

/// Landmarks for every pose found in one frame.
///
/// The two outer vectors are parallel: index `i` of each belongs to the same
/// person. [`PoseLandmarkerVideo`] is built with `num_poses = 1`, so a normal
/// result has at most one element and an empty result means no person was
/// found. World coordinates are in meters and are not normalized image units.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PoseLandmarkerResult {
    /// Normalized image landmarks per person.
    pub pose_landmarks: Vec<Vec<NormalizedLandmark>>,
    /// World landmarks per person, in meters.
    pub pose_world_landmarks: Vec<Vec<WorldLandmark>>,
}

impl PoseLandmarkerResult {
    /// # Safety
    /// `raw` must be a live `MpPoseLandmarkerResult`.
    unsafe fn from_raw(raw: &sys::MpPoseLandmarkerResult) -> Result<Self> {
        let pose_landmarks = if raw.pose_landmarks.is_null() {
            Vec::new()
        } else {
            (0..raw.pose_landmarks_count as usize)
                .map(|p| {
                    // SAFETY: the index is below the count MediaPipe reported for this array.
                    let set = unsafe { &*raw.pose_landmarks.add(p) };
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

        let pose_world_landmarks = if raw.pose_world_landmarks.is_null() {
            Vec::new()
        } else {
            (0..raw.pose_world_landmarks_count as usize)
                .map(|p| {
                    // SAFETY: the index is below the count MediaPipe reported for this array.
                    let set = unsafe { &*raw.pose_world_landmarks.add(p) };
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

        Ok(PoseLandmarkerResult {
            pose_landmarks,
            pose_world_landmarks,
        })
    }
}

/// Builds a [`PoseLandmarkerVideo`].
#[derive(Debug, Clone)]
pub struct PoseLandmarkerBuilder {
    source: ModelSource,
    delegate: Delegate,
    num_poses: NonZeroU32,
    min_pose_detection_confidence: Confidence,
    min_pose_presence_confidence: Confidence,
    min_tracking_confidence: IouThreshold,
}

impl PoseLandmarkerBuilder {
    /// Chooses the inference backend.
    pub fn delegate(mut self, delegate: Delegate) -> Self {
        self.delegate = delegate;
        self
    }

    /// Maximum number of poses to detect. The pose worker uses one.
    pub fn num_poses(mut self, n: NonZeroU32) -> Self {
        self.num_poses = n;
        self
    }

    /// Minimum score for the pose detector stage.
    pub fn min_pose_detection_confidence(mut self, v: Confidence) -> Self {
        self.min_pose_detection_confidence = v;
        self
    }

    /// Minimum score that a pose is present in the tracked region.
    pub fn min_pose_presence_confidence(mut self, v: Confidence) -> Self {
        self.min_pose_presence_confidence = v;
        self
    }

    /// How much a newly detected pose box must overlap the previous frame's box.
    ///
    /// As with the face landmarker, upstream's name says "confidence" but the
    /// value is an overlap ratio, so it takes an [`IouThreshold`].
    pub fn min_tracking_confidence(mut self, v: IouThreshold) -> Self {
        self.min_tracking_confidence = v;
        self
    }

    /// Builds a landmarker in video mode. Timestamps must increase between calls.
    pub fn build_for_video(self) -> Result<PoseLandmarkerVideo> {
        self.build_mode(sys::MpRunningMode::MP_RUNNING_MODE_VIDEO)
            .map(PoseLandmarkerVideo)
    }

    fn build_mode(self, running_mode: sys::MpRunningMode) -> Result<PoseLandmarker> {
        let min_pose_detection_confidence = self.min_pose_detection_confidence.get();
        let min_pose_presence_confidence = self.min_pose_presence_confidence.get();
        let min_tracking_confidence = self.min_tracking_confidence.get();

        let lib = lib()?;
        let hold = ModelHold::new(&self.source)?;
        let (buf, buf_count, path) = hold.parts();
        let delegate = self.delegate.to_raw();
        // The C side takes this as an `int`; requesting more poses than fit is
        // meaningless rather than an error worth a variant.
        let num_poses = i32::try_from(self.num_poses.get()).unwrap_or(i32::MAX);

        let mut ptr: sys::MpPoseLandmarkerPtr = ptr::null_mut();
        let mut err = ptr::null_mut();

        // SAFETY: `hold` keeps the model alive across the call; the options
        // layout is chosen from the probed ABI (see sys::compat).
        let status = unsafe {
            match lib.abi {
                Abi::Renamed => {
                    let mut opts = sys::MpPoseLandmarkerOptions {
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
                        num_poses,
                        min_pose_detection_confidence,
                        min_pose_presence_confidence,
                        min_tracking_confidence,
                        output_segmentation_masks: false,
                        result_callback: None,
                    };
                    lib.raw
                        .MpPoseLandmarkerCreate(&mut opts, &mut ptr, &mut err)
                }
                Abi::V0_10_35 => {
                    let mut opts = sys::compat::MpPoseLandmarkerOptionsV35 {
                        base_options: sys::compat::MpBaseOptionsV35::new(
                            buf, buf_count, path, delegate,
                        ),
                        running_mode,
                        num_poses,
                        min_pose_detection_confidence,
                        min_pose_presence_confidence,
                        min_tracking_confidence,
                        output_segmentation_masks: false,
                        result_callback: None,
                    };
                    lib.raw.MpPoseLandmarkerCreate(
                        (&raw mut opts).cast::<sys::MpPoseLandmarkerOptions>(),
                        &mut ptr,
                        &mut err,
                    )
                }
            }
        };

        if status != sys::MpStatus::kMpOk {
            // SAFETY: `err` is the out-param of the call that just failed; it is read and freed exactly once here.
            return Err(unsafe { Error::from_status(&lib.raw, status, err) });
        }
        Ok(PoseLandmarker { ptr, closed: false })
    }
}

/// Owns a `MpPoseLandmarkerPtr`. Built only through [`PoseLandmarker::builder`].
pub struct PoseLandmarker {
    ptr: sys::MpPoseLandmarkerPtr,
    closed: bool,
}

// SAFETY: as FaceLandmarker — movable between threads, never shared.
unsafe impl Send for PoseLandmarker {}

impl PoseLandmarker {
    /// Starts building a pose landmarker from a model source.
    pub fn builder(source: ModelSource) -> PoseLandmarkerBuilder {
        PoseLandmarkerBuilder {
            source,
            delegate: Delegate::default(),
            num_poses: NonZeroU32::new(1).expect("1 should be non-zero"),
            // MediaPipe's own defaults.
            min_pose_detection_confidence: Confidence::HALF,
            min_pose_presence_confidence: Confidence::HALF,
            min_tracking_confidence: IouThreshold::HALF,
        }
    }

    fn close_once(&mut self) -> Result<()> {
        if std::mem::replace(&mut self.closed, true) {
            return Ok(());
        }
        let lib = lib()?;
        let mut err = ptr::null_mut();
        // SAFETY: runs exactly once; joins any worker thread.
        let status = unsafe { lib.raw.MpPoseLandmarkerClose(self.ptr, &mut err) };
        if status != sys::MpStatus::kMpOk {
            // SAFETY: `err` is the out-param of the call that just failed; it is read and freed exactly once here.
            return Err(unsafe { Error::from_status(&lib.raw, status, err) });
        }
        Ok(())
    }
}

impl std::fmt::Debug for PoseLandmarker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoseLandmarker")
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}

impl Drop for PoseLandmarker {
    fn drop(&mut self) {
        let _ = self.close_once();
    }
}

/// A [`PoseLandmarker`] running in video mode.
pub struct PoseLandmarkerVideo(PoseLandmarker);

impl PoseLandmarkerVideo {
    /// Runs pose detection on one decoded video frame.
    ///
    /// Timestamps must increase between calls. The returned result owns all of
    /// its landmarks; MediaPipe frees the C result before this returns.
    pub fn detect_for_video(
        &mut self,
        image: &Image,
        timestamp: Timestamp,
    ) -> Result<PoseLandmarkerResult> {
        let lib = lib()?;
        // SAFETY: the struct is only pointers and counts, so all-zero is a valid empty result; MediaPipe overwrites it.
        let mut result: sys::MpPoseLandmarkerResult = unsafe { std::mem::zeroed() };
        let mut err = ptr::null_mut();

        // SAFETY: live handle and image; `result` goes straight back to
        // MpPoseLandmarkerCloseResult.
        let status = unsafe {
            lib.raw.MpPoseLandmarkerDetectForVideo(
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
            return Err(unsafe { Error::from_status(&lib.raw, status, err) });
        }

        // Convert first, free second — and free even if conversion failed.
        // SAFETY: `result` was filled by the call above and is still live.
        let converted = unsafe { PoseLandmarkerResult::from_raw(&result) };
        // SAFETY: `result` was filled by the Detect call above, everything has been copied out, and it is freed once.
        unsafe { lib.raw.MpPoseLandmarkerCloseResult(&mut result) };
        converted
    }
}

impl std::fmt::Debug for PoseLandmarkerVideo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoseLandmarkerVideo")
            .field("landmarker", &self.0)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_people_is_an_empty_result() {
        let raw = sys::MpPoseLandmarkerResult {
            segmentation_masks: ptr::null_mut(),
            segmentation_masks_count: 0,
            pose_landmarks: ptr::null_mut(),
            pose_landmarks_count: 0,
            pose_world_landmarks: ptr::null_mut(),
            pose_world_landmarks_count: 0,
        };
        // SAFETY: `raw` is a fully initialised empty result with no pointers for
        // `from_raw` to follow.
        let result = unsafe { PoseLandmarkerResult::from_raw(&raw) }.unwrap();
        assert!(result.pose_landmarks.is_empty());
        assert!(result.pose_world_landmarks.is_empty());
    }

    #[test]
    fn world_landmarks_copy_meters_and_keep_missing_scores() {
        let mut world = [sys::MpLandmark {
            x: 1.5,
            y: -2.0,
            z: 3.25,
            has_visibility: true,
            visibility: 0.8,
            has_presence: false,
            presence: 9.0,
            name: ptr::null_mut(),
        }];
        let mut sets = [sys::MpLandmarks {
            landmarks: world.as_mut_ptr(),
            landmarks_count: 1,
        }];
        let raw = sys::MpPoseLandmarkerResult {
            segmentation_masks: ptr::null_mut(),
            segmentation_masks_count: 0,
            pose_landmarks: ptr::null_mut(),
            pose_landmarks_count: 0,
            pose_world_landmarks: sets.as_mut_ptr(),
            pose_world_landmarks_count: 1,
        };
        // SAFETY: `raw` points at the live `world`/`sets` arrays for this call,
        // and `from_raw` only copies out of them.
        let result = unsafe { PoseLandmarkerResult::from_raw(&raw) }.unwrap();
        assert_eq!(result.pose_world_landmarks.len(), 1);
        let landmark = &result.pose_world_landmarks[0][0];
        assert_eq!(landmark.point.to_array(), [1.5, -2.0, 3.25]);
        assert_eq!(landmark.visibility.map(Confidence::get), Some(0.8));
        assert!(landmark.presence.is_none());
    }

    #[test]
    fn normalized_and_world_landmarks_are_kept_parallel_per_person() {
        let mut normalized = [sys::MpNormalizedLandmark {
            x: 0.1,
            y: 0.2,
            z: 0.3,
            has_visibility: false,
            visibility: 0.0,
            has_presence: false,
            presence: 0.0,
            name: ptr::null_mut(),
        }];
        let mut normalized_sets = [sys::MpNormalizedLandmarks {
            landmarks: normalized.as_mut_ptr(),
            landmarks_count: 1,
        }];
        let mut world = [sys::MpLandmark {
            x: 0.4,
            y: 0.5,
            z: 0.6,
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
        let raw = sys::MpPoseLandmarkerResult {
            segmentation_masks: ptr::null_mut(),
            segmentation_masks_count: 0,
            pose_landmarks: normalized_sets.as_mut_ptr(),
            pose_landmarks_count: 1,
            pose_world_landmarks: world_sets.as_mut_ptr(),
            pose_world_landmarks_count: 1,
        };
        // SAFETY: `raw` points at the live arrays for this call; `from_raw` only
        // copies out of them.
        let result = unsafe { PoseLandmarkerResult::from_raw(&raw) }.unwrap();
        assert_eq!(result.pose_landmarks.len(), 1);
        assert_eq!(result.pose_world_landmarks.len(), 1);
        assert_eq!(result.pose_landmarks[0][0].point.x(), 0.1);
        assert_eq!(result.pose_world_landmarks[0][0].point.x(), 0.4);
    }
}
