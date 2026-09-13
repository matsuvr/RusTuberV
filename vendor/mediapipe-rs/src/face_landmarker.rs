use std::num::NonZeroU32;
use std::os::raw::c_void;
use std::ptr;

use crate::error::{Error, Result};
use crate::face_detector::status_only_error;
use crate::image::{Image, ImageRef};
use crate::loader::lib;
use crate::stream::{AsyncTask, Slot, Stream};
use crate::sys::{self, Abi, ModelHold};
use crate::types::{
    Category, Confidence, Delegate, IouThreshold, ModelSource, NormalizedLandmark, Rotation,
    Timestamp, Transform4x4,
};

/// Landmarks for every face found in one frame.
///
/// The three vectors are parallel: index `i` of each describes the same face.
/// `blendshapes` and `transformation_matrixes` are empty unless the
/// corresponding builder option was enabled.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FaceLandmarkerResult {
    /// 478 landmarks per face for the standard model.
    pub landmarks: Vec<Vec<NormalizedLandmark>>,
    /// 52 expression coefficients per face, when enabled.
    pub blendshapes: Vec<Vec<Category>>,
    /// Face-to-camera transform per face, when enabled.
    pub transformation_matrixes: Vec<Transform4x4>,
}

impl FaceLandmarkerResult {
    /// # Safety
    /// `raw` must be a live `MpFaceLandmarkerResult`.
    unsafe fn from_raw(raw: &sys::MpFaceLandmarkerResult) -> Result<Self> {
        let landmarks = if raw.face_landmarks.is_null() {
            Vec::new()
        } else {
            (0..raw.face_landmarks_count as usize)
                .map(|f| {
                    // SAFETY: the index is below the count MediaPipe reported for this array.
                    let set = unsafe { &*raw.face_landmarks.add(f) };
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

        let blendshapes = if raw.face_blendshapes.is_null() {
            Vec::new()
        } else {
            (0..raw.face_blendshapes_count as usize)
                .map(|f| {
                    // SAFETY: the index is below the count MediaPipe reported for this array.
                    let set = unsafe { &*raw.face_blendshapes.add(f) };
                    // SAFETY: a field of the live result this function's contract covers.
                    unsafe { Category::vec_from_raw(set.categories, set.categories_count) }
                })
                .collect()
        };

        let transformation_matrixes = if raw.facial_transformation_matrixes.is_null() {
            Vec::new()
        } else {
            (0..raw.facial_transformation_matrixes_count as usize)
                .map(|i| {
                    // SAFETY: the index is below the count MediaPipe reported for this array.
                    Transform4x4::from_raw(unsafe { &*raw.facial_transformation_matrixes.add(i) })
                })
                .collect::<Result<Vec<_>>>()?
        };

        Ok(FaceLandmarkerResult {
            landmarks,
            blendshapes,
            transformation_matrixes,
        })
    }
}

#[derive(Debug, Clone)]
pub struct FaceLandmarkerBuilder {
    source: ModelSource,
    delegate: Delegate,
    num_faces: NonZeroU32,
    min_face_detection_confidence: Confidence,
    min_face_presence_confidence: Confidence,
    min_tracking_confidence: IouThreshold,
    output_blendshapes: bool,
    output_transformation_matrixes: bool,
}

impl FaceLandmarkerBuilder {
    pub fn delegate(mut self, delegate: Delegate) -> Self {
        self.delegate = delegate;
        self
    }

    /// Maximum number of faces to track.
    pub fn num_faces(mut self, n: NonZeroU32) -> Self {
        self.num_faces = n;
        self
    }

    /// Minimum model score for the face detector stage.
    pub fn min_face_detection_confidence(mut self, v: Confidence) -> Self {
        self.min_face_detection_confidence = v;
        self
    }

    /// Minimum model score that a face is present in the tracked region.
    pub fn min_face_presence_confidence(mut self, v: Confidence) -> Self {
        self.min_face_presence_confidence = v;
        self
    }

    /// How much a newly detected face box must overlap the previous frame's box
    /// to count as the same face.
    ///
    /// Upstream calls this `min_tracking_confidence`, but the name is wrong: it
    /// reaches `AssociationNormRectCalculator`, which tests it against
    /// `CalculateIou(current_rect, previous_rect)`. It is an overlap ratio, so
    /// it takes an [`IouThreshold`], not a [`Confidence`].
    pub fn min_tracking_confidence(mut self, v: IouThreshold) -> Self {
        self.min_tracking_confidence = v;
        self
    }

    /// Also produce the 52 expression coefficients.
    pub fn output_blendshapes(mut self, yes: bool) -> Self {
        self.output_blendshapes = yes;
        self
    }

    /// Also produce a 4x4 face-to-camera transform per face.
    pub fn output_transformation_matrixes(mut self, yes: bool) -> Self {
        self.output_transformation_matrixes = yes;
        self
    }

    pub fn build(self) -> Result<FaceLandmarker> {
        self.build_mode(sys::MpRunningMode::MP_RUNNING_MODE_IMAGE, None)
    }

    pub fn build_for_video(self) -> Result<FaceLandmarker> {
        self.build_mode(sys::MpRunningMode::MP_RUNNING_MODE_VIDEO, None)
    }

    pub fn build_stream<F>(self, callback: F) -> Result<FaceLandmarkerStream>
    where
        F: FnMut(Result<FaceLandmarkerResult>, ImageRef<'_>, Timestamp) + Send + 'static,
    {
        let mut callback = callback;
        let slot = Slot::claim(Box::new(move |status, result, image, ts| {
            // Copy out before returning: MediaPipe frees the result immediately
            // afterwards, and `image` is a stack MpImage in its dispatch lambda.
            let converted = if status != sys::MpStatus::kMpOk || result.is_null() {
                Err(status_only_error(status))
            } else {
                // SAFETY: status is kMpOk and `result` is non-null, so MediaPipe
                // handed us a live result of this task's type.
                unsafe {
                    FaceLandmarkerResult::from_raw(&*result.cast::<sys::MpFaceLandmarkerResult>())
                }
            };
            callback(
                converted,
                ImageRef::from_raw(image),
                Timestamp::from_millis(ts),
            );
        }))?;

        let raw_cb = slot.raw_callback();
        let landmarker = self.build_mode(
            sys::MpRunningMode::MP_RUNNING_MODE_LIVE_STREAM,
            Some(raw_cb),
        )?;
        Ok(Stream::new(landmarker, slot))
    }

    fn build_mode(
        self,
        running_mode: sys::MpRunningMode,
        callback: Option<unsafe extern "C" fn(sys::MpStatus, *const c_void, sys::MpImagePtr, i64)>,
    ) -> Result<FaceLandmarker> {
        let min_face_detection_confidence = self.min_face_detection_confidence.get();
        let min_face_presence_confidence = self.min_face_presence_confidence.get();
        let min_tracking_confidence = self.min_tracking_confidence.get();

        let lib = lib()?;
        let hold = ModelHold::new(&self.source)?;
        let (buf, buf_count, path) = hold.parts();
        let delegate = self.delegate.to_raw();
        // MediaPipe takes this as a C `int`. Requesting more faces than an i32
        // can hold is meaningless rather than an error worth a variant, so clamp
        // instead of wrapping into a negative.
        let num_faces = i32::try_from(self.num_faces.get()).unwrap_or(i32::MAX);
        let cb: sys::MpFaceLandmarkerOptions_result_callback_fn =
            // SAFETY: the two fn types differ only in the pointee of the result pointer, which does not change the C ABI.
            callback.map(|f| unsafe { std::mem::transmute(f) });

        let mut ptr: sys::MpFaceLandmarkerPtr = ptr::null_mut();
        let mut err = ptr::null_mut();

        // SAFETY: `hold` keeps the model alive across the call; the options
        // layout is chosen from the probed ABI (see sys::compat).
        let status = unsafe {
            match lib.abi {
                Abi::Renamed => {
                    let mut opts = sys::MpFaceLandmarkerOptions {
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
                        num_faces,
                        min_face_detection_confidence,
                        min_face_presence_confidence,
                        min_tracking_confidence,
                        output_face_blendshapes: self.output_blendshapes,
                        output_facial_transformation_matrixes: self.output_transformation_matrixes,
                        result_callback: cb,
                    };
                    lib.raw
                        .MpFaceLandmarkerCreate(&mut opts, &mut ptr, &mut err)
                }
                Abi::V0_10_35 => {
                    let mut opts = sys::compat::MpFaceLandmarkerOptionsV35 {
                        base_options: sys::compat::MpBaseOptionsV35::new(
                            buf, buf_count, path, delegate,
                        ),
                        running_mode,
                        num_faces,
                        min_face_detection_confidence,
                        min_face_presence_confidence,
                        min_tracking_confidence,
                        output_face_blendshapes: self.output_blendshapes,
                        output_facial_transformation_matrixes: self.output_transformation_matrixes,
                        result_callback: cb,
                    };
                    lib.raw.MpFaceLandmarkerCreate(
                        (&raw mut opts).cast::<sys::MpFaceLandmarkerOptions>(),
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
        Ok(FaceLandmarker { ptr, closed: false })
    }
}

/// Dense face landmarks, optionally with blendshapes and a face transform.
pub struct FaceLandmarker {
    ptr: sys::MpFaceLandmarkerPtr,
    closed: bool,
}

// SAFETY: as FaceDetector — movable between threads, never shared.
unsafe impl Send for FaceLandmarker {}

impl FaceLandmarker {
    pub fn builder(source: ModelSource) -> FaceLandmarkerBuilder {
        FaceLandmarkerBuilder {
            source,
            delegate: Delegate::default(),
            num_faces: NonZeroU32::new(1).expect("1 should be non-zero"),
            // MediaPipe's own defaults.
            min_face_detection_confidence: Confidence::HALF,
            min_face_presence_confidence: Confidence::HALF,
            min_tracking_confidence: IouThreshold::HALF,
            output_blendshapes: false,
            output_transformation_matrixes: false,
        }
    }

    pub fn detect(&mut self, image: &Image) -> Result<FaceLandmarkerResult> {
        self.detect_inner(image, None, None)
    }

    /// Applies a clockwise rotation before inference.
    ///
    /// There is deliberately no region-of-interest variant: MediaPipe builds
    /// both face tasks with `roi_allowed=false`, so an ROI is not a runtime
    /// error to guard against, it is a request that cannot be expressed.
    pub fn detect_rotated(
        &mut self,
        image: &Image,
        rotation: Rotation,
    ) -> Result<FaceLandmarkerResult> {
        let raw = rotation.to_raw();
        self.detect_inner(image, Some(&raw), None)
    }

    /// Video mode. Timestamps must increase between calls.
    pub fn detect_for_video(
        &mut self,
        image: &Image,
        timestamp: Timestamp,
    ) -> Result<FaceLandmarkerResult> {
        self.detect_inner(image, None, Some(timestamp.as_millis()))
    }

    fn detect_inner(
        &mut self,
        image: &Image,
        processing: Option<&sys::MpImageProcessingOptions>,
        timestamp_ms: Option<i64>,
    ) -> Result<FaceLandmarkerResult> {
        let lib = lib()?;
        let opts = processing.map_or(ptr::null(), |p| p as *const _);
        // SAFETY: the struct is only pointers and counts, so all-zero is a valid empty result; MediaPipe overwrites it.
        let mut result: sys::MpFaceLandmarkerResult = unsafe { std::mem::zeroed() };
        let mut err = ptr::null_mut();

        // SAFETY: live handle and image; `result` goes straight back to
        // MpFaceLandmarkerCloseResult.
        let status = unsafe {
            match timestamp_ms {
                None => lib.raw.MpFaceLandmarkerDetectImage(
                    self.ptr,
                    image.ptr,
                    opts,
                    &mut result,
                    &mut err,
                ),
                Some(ts) => lib.raw.MpFaceLandmarkerDetectForVideo(
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

        // Convert first, free second — and free even if conversion failed.
        // SAFETY: `result` was filled by the call above and is still live.
        let converted = unsafe { FaceLandmarkerResult::from_raw(&result) };
        // SAFETY: `result` was filled by the Detect call above, everything has been copied out, and it is freed once.
        unsafe { lib.raw.MpFaceLandmarkerCloseResult(&mut result) };
        converted
    }

    fn close_once(&mut self) -> Result<()> {
        if std::mem::replace(&mut self.closed, true) {
            return Ok(());
        }
        let lib = lib()?;
        let mut err = ptr::null_mut();
        // SAFETY: runs exactly once; joins any worker thread.
        let status = unsafe { lib.raw.MpFaceLandmarkerClose(self.ptr, &mut err) };
        if status != sys::MpStatus::kMpOk {
            // SAFETY: `err` is the out-param of the call that just failed; it is read and freed exactly once here.
            return Err(unsafe { Error::from_status(&lib.raw, status, err) });
        }
        Ok(())
    }
}

impl std::fmt::Debug for FaceLandmarker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FaceLandmarker")
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}

impl Drop for FaceLandmarker {
    fn drop(&mut self) {
        let _ = self.close_once();
    }
}

impl AsyncTask for FaceLandmarker {
    fn send_raw(
        &mut self,
        image: &Image,
        opts: Option<&sys::MpImageProcessingOptions>,
        timestamp_ms: i64,
    ) -> Result<()> {
        let lib = lib()?;
        let mut err = ptr::null_mut();
        // SAFETY: as detect_inner.
        let status = unsafe {
            lib.raw.MpFaceLandmarkerDetectAsync(
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

/// A [`FaceLandmarker`] running in live-stream mode.
pub type FaceLandmarkerStream = Stream<FaceLandmarker>;
