use std::os::raw::c_void;
use std::ptr;

use crate::error::{Error, Result};
use crate::image::{Image, ImageRef};
use crate::loader::lib;
use crate::stream::{AsyncTask, Slot, Stream};
use crate::sys::{self, Abi, ModelHold};
use crate::types::{
    Category, Confidence, Delegate, IouThreshold, Keypoint, ModelSource, PixelRect, Rotation,
    Timestamp,
};

/// MediaPipe's C header defaults this to 0.5, but its C++ and Python task layers
/// both use 0.3. Follow the task layer: that is the behaviour users see from
/// every other MediaPipe binding.
const DEFAULT_MIN_SUPPRESSION: IouThreshold = IouThreshold::from_raw(0.3);

/// One detected face.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    /// In **pixels**, unlike [`keypoints`](Self::keypoints), which are normalized.
    pub bounding_box: PixelRect,
    pub categories: Vec<Category>,
    pub keypoints: Vec<Keypoint>,
}

impl Detection {
    /// The highest category score, if the model produced any categories.
    pub fn score(&self) -> Option<Confidence> {
        self.categories
            .iter()
            .map(|c| c.score)
            .max_by(|a, b| a.get().total_cmp(&b.get()))
    }

    /// # Safety
    /// `raw` must be a live `MpDetectionResult`.
    unsafe fn vec_from_raw(raw: &sys::MpDetectionResult) -> Vec<Detection> {
        if raw.detections.is_null() {
            return Vec::new();
        }
        (0..raw.detections_count as usize)
            .map(|i| {
                // SAFETY: the index is below the count MediaPipe reported for this array.
                let d = unsafe { &*raw.detections.add(i) };
                Detection {
                    bounding_box: PixelRect::from_raw(d.bounding_box),
                    // SAFETY: a field of the live result this function's contract covers.
                    categories: unsafe { Category::vec_from_raw(d.categories, d.categories_count) },
                    keypoints: if d.keypoints.is_null() {
                        Vec::new()
                    } else {
                        (0..d.keypoints_count as usize)
                            // SAFETY: the index is below the count MediaPipe reported for this array.
                            .map(|k| unsafe { Keypoint::from_raw(&*d.keypoints.add(k)) })
                            .collect()
                    },
                }
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct FaceDetectorBuilder {
    source: ModelSource,
    delegate: Delegate,
    min_detection_confidence: Confidence,
    min_suppression_threshold: IouThreshold,
}

impl FaceDetectorBuilder {
    pub fn delegate(mut self, delegate: Delegate) -> Self {
        self.delegate = delegate;
        self
    }

    /// Minimum model score for a detection to be reported.
    ///
    /// Becomes `TensorsToDetectionsCalculator::min_score_thresh` upstream.
    pub fn min_detection_confidence(mut self, v: Confidence) -> Self {
        self.min_detection_confidence = v;
        self
    }

    /// How much two boxes must overlap before non-maximum suppression discards
    /// the weaker one.
    ///
    /// A geometric ratio, not a score: it configures a suppression node with
    /// `overlap_type = INTERSECTION_OVER_UNION`. Hence [`IouThreshold`] rather
    /// than [`Confidence`], so a score cannot be passed here by mistake:
    ///
    /// ```compile_fail,E0308
    /// use mediapipe::{Confidence, FaceDetector, ModelSource};
    /// FaceDetector::builder(ModelSource::path("m.tflite"))
    ///     .min_suppression_threshold(Confidence::HALF);  // a score is not an overlap ratio
    /// ```
    ///
    /// ```
    /// use mediapipe::{FaceDetector, IouThreshold, ModelSource};
    /// FaceDetector::builder(ModelSource::path("m.tflite"))
    ///     .min_suppression_threshold(IouThreshold::new(0.3)?);
    /// # Ok::<(), mediapipe::Error>(())
    /// ```
    pub fn min_suppression_threshold(mut self, v: IouThreshold) -> Self {
        self.min_suppression_threshold = v;
        self
    }

    /// Single images.
    pub fn build(self) -> Result<FaceDetector> {
        self.build_mode(sys::MpRunningMode::MP_RUNNING_MODE_IMAGE, None)
    }

    /// Frames of a video, decoded ahead of time. Tracks faces across calls, so
    /// timestamps must increase.
    pub fn build_for_video(self) -> Result<FaceDetector> {
        self.build_mode(sys::MpRunningMode::MP_RUNNING_MODE_VIDEO, None)
    }

    /// A live stream. Results are delivered to `callback` on a MediaPipe worker
    /// thread; see [`Stream`].
    pub fn build_stream<F>(self, callback: F) -> Result<FaceDetectorStream>
    where
        F: FnMut(Result<Vec<Detection>>, ImageRef<'_>, Timestamp) + Send + 'static,
    {
        let mut callback = callback;
        let slot = Slot::claim(Box::new(move |status, result, image, ts| {
            // MediaPipe frees `result` as soon as this returns, and `image`
            // points at a stack MpImage inside its dispatch lambda — so copy out
            // and never free either.
            let converted = if status != sys::MpStatus::kMpOk || result.is_null() {
                Err(status_only_error(status))
            } else {
                // SAFETY: status is kMpOk and `result` is non-null, so MediaPipe handed us a live result of this task's type.
                Ok(unsafe { Detection::vec_from_raw(&*result.cast::<sys::MpDetectionResult>()) })
            };
            callback(
                converted,
                ImageRef::from_raw(image),
                Timestamp::from_millis(ts),
            );
        }))?;

        let raw_cb = slot.raw_callback();
        let detector = self.build_mode(
            sys::MpRunningMode::MP_RUNNING_MODE_LIVE_STREAM,
            Some(raw_cb),
        )?;
        Ok(Stream::new(detector, slot))
    }

    fn build_mode(
        self,
        running_mode: sys::MpRunningMode,
        callback: Option<unsafe extern "C" fn(sys::MpStatus, *const c_void, sys::MpImagePtr, i64)>,
    ) -> Result<FaceDetector> {
        let min_detection_confidence = self.min_detection_confidence.get();
        let min_suppression_threshold = self.min_suppression_threshold.get();

        let lib = lib()?;
        let hold = ModelHold::new(&self.source)?;
        let (buf, buf_count, path) = hold.parts();
        let delegate = self.delegate.to_raw();

        // The C callback type differs between tasks only in the result pointer
        // type, which is ABI-identical to *const c_void.
        let cb: sys::MpFaceDetectorOptions_result_callback_fn =
            // SAFETY: the two fn types differ only in the pointee of the result pointer, which does not change the C ABI.
            callback.map(|f| unsafe { std::mem::transmute(f) });

        let mut ptr: sys::MpFaceDetectorPtr = ptr::null_mut();
        let mut err = ptr::null_mut();

        // SAFETY: `hold` owns the model path/bytes for the whole call. Which
        // options layout to build is decided by the probed ABI; see sys::compat.
        let status = unsafe {
            match lib.abi {
                Abi::Renamed => {
                    let mut opts = sys::MpFaceDetectorOptions {
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
                        min_detection_confidence,
                        min_suppression_threshold,
                        result_callback: cb,
                    };
                    lib.raw.MpFaceDetectorCreate(&mut opts, &mut ptr, &mut err)
                }
                Abi::V0_10_35 => {
                    let mut opts = sys::compat::MpFaceDetectorOptionsV35 {
                        base_options: sys::compat::MpBaseOptionsV35::new(
                            buf, buf_count, path, delegate,
                        ),
                        running_mode,
                        min_detection_confidence,
                        min_suppression_threshold,
                        result_callback: cb,
                    };
                    lib.raw.MpFaceDetectorCreate(
                        (&raw mut opts).cast::<sys::MpFaceDetectorOptions>(),
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
        Ok(FaceDetector { ptr, closed: false })
    }
}

/// Detects faces and their six keypoints.
///
/// The running mode is fixed when the detector is built. Calling the wrong
/// `detect*` method for the mode is rejected by MediaPipe with
/// [`StatusCode::InvalidArgument`](crate::StatusCode::InvalidArgument).
pub struct FaceDetector {
    ptr: sys::MpFaceDetectorPtr,
    /// `Stream` closes the task explicitly so the worker is joined before its
    /// callback slot is released; `Drop` must not then close it a second time.
    closed: bool,
}

// SAFETY: a task handle can move between threads; it is not internally
// synchronised, hence Send but not Sync, and every method takes &mut self.
unsafe impl Send for FaceDetector {}

impl FaceDetector {
    pub fn builder(source: ModelSource) -> FaceDetectorBuilder {
        FaceDetectorBuilder {
            source,
            delegate: Delegate::default(),
            // MediaPipe's own defaults.
            min_detection_confidence: Confidence::HALF,
            min_suppression_threshold: DEFAULT_MIN_SUPPRESSION,
        }
    }

    pub fn detect(&mut self, image: &Image) -> Result<Vec<Detection>> {
        self.detect_inner(image, None, None)
    }

    /// Applies a clockwise rotation before inference.
    ///
    /// There is deliberately no region-of-interest variant: MediaPipe builds
    /// both face tasks with `roi_allowed=false`, so an ROI is not a runtime
    /// error to guard against, it is a request that cannot be expressed.
    pub fn detect_rotated(&mut self, image: &Image, rotation: Rotation) -> Result<Vec<Detection>> {
        let raw = rotation.to_raw();
        self.detect_inner(image, Some(&raw), None)
    }

    /// Video mode. Timestamps must increase between calls.
    pub fn detect_for_video(
        &mut self,
        image: &Image,
        timestamp: Timestamp,
    ) -> Result<Vec<Detection>> {
        self.detect_inner(image, None, Some(timestamp.as_millis()))
    }

    fn detect_inner(
        &mut self,
        image: &Image,
        processing: Option<&sys::MpImageProcessingOptions>,
        timestamp_ms: Option<i64>,
    ) -> Result<Vec<Detection>> {
        let lib = lib()?;
        let opts = processing.map_or(ptr::null(), |p| p as *const _);
        let mut result = sys::MpDetectionResult {
            detections: ptr::null_mut(),
            detections_count: 0,
        };
        let mut err = ptr::null_mut();

        // SAFETY: `self.ptr` and `image.ptr` are live; `result` is an out-param
        // that we hand straight back to MpFaceDetectorCloseResult below.
        let status = unsafe {
            match timestamp_ms {
                None => lib.raw.MpFaceDetectorDetectImage(
                    self.ptr,
                    image.ptr,
                    opts,
                    &mut result,
                    &mut err,
                ),
                Some(ts) => lib.raw.MpFaceDetectorDetectForVideo(
                    self.ptr,
                    image.ptr,
                    opts,
                    ts,
                    &mut result,
                    &mut err,
                ),
            }
        };

        if status != sys::MpStatus::kMpOk {
            // SAFETY: `err` is the out-param of the call that just failed; it is read and freed exactly once here.
            return Err(unsafe { Error::from_status(&lib.raw, status, err) });
        }

        // Copy out, then release C memory immediately — nothing borrows it.
        // SAFETY: `result` was filled by the call above and is still live.
        let detections = unsafe { Detection::vec_from_raw(&result) };
        // SAFETY: `result` was filled by the Detect call above, everything has been copied out, and it is freed once.
        unsafe { lib.raw.MpFaceDetectorCloseResult(&mut result) };
        Ok(detections)
    }
}

impl std::fmt::Debug for FaceDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FaceDetector")
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}

impl FaceDetector {
    fn close_once(&mut self) -> Result<()> {
        if std::mem::replace(&mut self.closed, true) {
            return Ok(());
        }
        let lib = lib()?;
        let mut err = ptr::null_mut();
        // SAFETY: runs exactly once per handle; flushes and joins any worker
        // thread before returning.
        let status = unsafe { lib.raw.MpFaceDetectorClose(self.ptr, &mut err) };
        if status != sys::MpStatus::kMpOk {
            // SAFETY: `err` is the out-param of the call that just failed; it is read and freed exactly once here.
            return Err(unsafe { Error::from_status(&lib.raw, status, err) });
        }
        Ok(())
    }
}

impl Drop for FaceDetector {
    fn drop(&mut self) {
        let _ = self.close_once();
    }
}

impl AsyncTask for FaceDetector {
    fn send_raw(
        &mut self,
        image: &Image,
        opts: Option<&sys::MpImageProcessingOptions>,
        timestamp_ms: i64,
    ) -> Result<()> {
        let lib = lib()?;
        let mut err = ptr::null_mut();
        // SAFETY: as detect_inner; DetectAsync takes a reference on the image's
        // shared buffer, so `image` may be dropped once this returns.
        let status = unsafe {
            lib.raw.MpFaceDetectorDetectAsync(
                self.ptr,
                image.ptr,
                opts.map_or(ptr::null(), |p| p as *const _),
                timestamp_ms,
                &mut err,
            )
        };
        if status != sys::MpStatus::kMpOk {
            // SAFETY: `err` is the out-param of the call that just failed; it is read and freed exactly once here.
            return Err(unsafe { Error::from_status(&lib.raw, status, err) });
        }
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        self.close_once()
    }
}

/// A [`FaceDetector`] running in live-stream mode.
pub type FaceDetectorStream = Stream<FaceDetector>;

/// Live-stream callbacks report failure through the status alone; there is no
/// `error_msg` on that path.
pub(crate) fn status_only_error(status: sys::MpStatus) -> Error {
    Error::Mp {
        code: crate::error::StatusCode::from_raw(status as i32),
        message: "live-stream callback reported failure".to_owned(),
    }
}
