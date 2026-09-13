//! Domain types.
//!
//! The C API is algebraically blind in one way that matters more than all the
//! others: a detection's bounding box is in **pixels** (`MpRect`, ints) while
//! keypoints, landmarks and regions of interest are **normalized** to `0..1`
//! (`MpRectF`, floats). Nothing in the C types tells them apart, so mixing them
//! compiles and silently produces nonsense. Here the coordinate space is part of
//! the type, and the only way between them is an explicit conversion that takes
//! the image [`Size`].

use crate::error::{Error, Result};
use crate::sys;

/// Image dimensions in pixels.
///
/// Has no `new(width, height)`: two positional `u32`s are exactly the swap this
/// type exists to prevent. Construct it with named fields, which cannot be
/// transposed silently:
///
/// ```
/// # use mediapipe::Size;
/// let size = Size { width: 820, height: 1024 };
/// ```
///
/// ```compile_fail,E0599
/// # use mediapipe::Size;
/// let size = Size::new(820, 1024);  // no such constructor, on purpose
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Size {
    pub width: u32,
    pub height: u32,
}

/// A point in pixel space, measured from the top-left of the image.
///
/// Fields are private so a normalized `0..1` value cannot be dropped in here,
/// and so that reaching for a raw coordinate is a deliberate act rather than the
/// path of least resistance — [`distance_to`] covers the common reason for it.
///
/// [`distance_to`]: PixelPoint::distance_to
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PixelPoint {
    x: f32,
    y: f32,
}

impl PixelPoint {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub const fn x(self) -> f32 {
        self.x
    }

    pub const fn y(self) -> f32 {
        self.y
    }

    /// Euclidean distance, in pixels. Only defined against another point in the
    /// same space, which is the whole point of the type.
    pub fn distance_to(self, other: PixelPoint) -> f32 {
        (self.x - other.x).hypot(self.y - other.y)
    }
}

/// An axis-aligned box in pixel space. This is what detectors return.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PixelRect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

impl PixelRect {
    pub const fn left(self) -> i32 {
        self.left
    }

    pub const fn top(self) -> i32 {
        self.top
    }

    pub const fn right(self) -> i32 {
        self.right
    }

    pub const fn bottom(self) -> i32 {
        self.bottom
    }

    pub const fn width(self) -> i32 {
        self.right - self.left
    }

    pub const fn height(self) -> i32 {
        self.bottom - self.top
    }

    /// Whether a pixel-space point falls inside the box. There is deliberately
    /// no equivalent taking a normalized point: convert it first, with the image
    /// [`Size`] that gives the conversion meaning.
    ///
    /// ```compile_fail,E0308
    /// # use mediapipe::{Detection, NormalizedPoint2};
    /// # fn demo(face: &Detection) {
    /// // A keypoint is normalized; the box is in pixels. No implicit mixing.
    /// face.bounding_box.contains(face.keypoints[0].point);
    /// # }
    /// ```
    pub fn contains(self, p: PixelPoint) -> bool {
        p.x >= self.left as f32
            && p.x <= self.right as f32
            && p.y >= self.top as f32
            && p.y <= self.bottom as f32
    }

    pub(crate) const fn from_raw(r: sys::MpRect) -> Self {
        Self {
            left: r.left,
            top: r.top,
            right: r.right,
            bottom: r.bottom,
        }
    }
}

/// A 2D point normalized to the image, as produced for keypoints.
///
/// Usually within `0..1`, but MediaPipe does emit values outside it for
/// features it infers just off the edge of the frame, so this is not validated.
///
/// There is no public constructor: these only ever come out of MediaPipe, and
/// leaving one out is what makes it impossible to build one from pixel values.
///
/// ```compile_fail,E0451
/// # use mediapipe::NormalizedPoint2;
/// // 363, 184 are pixels — there is no way to smuggle them in here.
/// let p = NormalizedPoint2 { x: 363.0, y: 184.0 };
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizedPoint2 {
    x: f32,
    y: f32,
}

impl NormalizedPoint2 {
    pub const fn x(self) -> f32 {
        self.x
    }

    pub const fn y(self) -> f32 {
        self.y
    }

    /// The only route to pixel space, and it needs the image it is relative to.
    pub fn to_pixels(self, size: Size) -> PixelPoint {
        PixelPoint {
            x: self.x * size.width as f32,
            y: self.y * size.height as f32,
        }
    }

    /// Distance in normalized units. For a distance in pixels, convert both
    /// points first — the two axes scale differently unless the image is square.
    pub fn distance_to(self, other: NormalizedPoint2) -> f32 {
        (self.x - other.x).hypot(self.y - other.y)
    }

    pub(crate) const fn from_raw(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// A 3D landmark whose `x`/`y` are normalized to the image.
///
/// `z` is *not* in that space: it is a model-defined depth, roughly on the same
/// scale as `x`, with the origin at the head centre for face landmarks, smaller
/// meaning closer to the camera. It has no pixel conversion, which is why
/// [`to_pixels`] drops it.
///
/// [`to_pixels`]: NormalizedPoint3::to_pixels
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizedPoint3 {
    x: f32,
    y: f32,
    z: f32,
}

impl NormalizedPoint3 {
    pub const fn x(self) -> f32 {
        self.x
    }

    pub const fn y(self) -> f32 {
        self.y
    }

    /// Model-defined depth — not a normalized image coordinate. See the type docs.
    pub const fn z(self) -> f32 {
        self.z
    }

    /// The `x`/`y` of this landmark, without the incomparable `z`.
    pub const fn xy(self) -> NormalizedPoint2 {
        NormalizedPoint2 {
            x: self.x,
            y: self.y,
        }
    }

    /// Projects `x`/`y` only; `z` has no pixel equivalent.
    pub fn to_pixels(self, size: Size) -> PixelPoint {
        self.xy().to_pixels(size)
    }

    pub(crate) const fn from_raw(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

/// How strongly a model believes something, in `0.0..=1.0`.
///
/// Deliberately has no `From<f32>` and no arithmetic: the point is that every
/// value is checked, and that a model score never silently stands in for the
/// other `0..1` quantity in this API — see [`IouThreshold`].
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Confidence(f32);

impl Confidence {
    pub fn new(value: f32) -> Result<Self> {
        if !(0.0..=1.0).contains(&value) {
            return Err(Error::OutOfRange {
                quantity: "confidence",
                value,
                range: 0.0..=1.0,
            });
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> f32 {
        self.0
    }

    /// For scores coming *out* of MediaPipe, which are already in range.
    pub(crate) const fn from_raw(v: f32) -> Self {
        Self(v)
    }

    pub const ZERO: Self = Self(0.0);
    pub const HALF: Self = Self(0.5);
    pub const ONE: Self = Self(1.0);
}

/// How much two boxes must overlap, as intersection-over-union in `0.0..=1.0`.
///
/// A geometric ratio between two rectangles — not a model score, even though
/// MediaPipe spells both as a bare `float` in `0..1` and names one of them
/// `min_tracking_confidence`. That name is misleading: it is fed to
/// `AssociationNormRectCalculator`, which compares it against
/// `CalculateIou(current_rect, previous_rect)`. Likewise
/// `min_suppression_threshold` configures a non-maximum-suppression node with
/// `overlap_type = INTERSECTION_OVER_UNION`.
///
/// Keeping the two apart means a [`Confidence`] read off a detection cannot be
/// passed where an overlap ratio belongs.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct IouThreshold(f32);

impl IouThreshold {
    pub fn new(value: f32) -> Result<Self> {
        if !(0.0..=1.0).contains(&value) {
            return Err(Error::OutOfRange {
                quantity: "IoU threshold",
                value,
                range: 0.0..=1.0,
            });
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> f32 {
        self.0
    }

    /// For compile-time literals known to be in range.
    pub(crate) const fn from_raw(v: f32) -> Self {
        Self(v)
    }

    pub const HALF: Self = Self(0.5);
}

/// A presentation timestamp in milliseconds.
///
/// Video and live-stream modes require monotonically increasing timestamps;
/// MediaPipe rejects out-of-order frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Timestamp(i64);

impl Timestamp {
    pub const fn from_millis(ms: i64) -> Self {
        Self(ms)
    }

    pub const fn as_millis(self) -> i64 {
        self.0
    }
}

/// Clockwise rotation to apply before inference. MediaPipe accepts only
/// multiples of 90 degrees, so the type only offers those.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Rotation {
    #[default]
    None,
    Cw90,
    Cw180,
    Cw270,
}

impl Rotation {
    pub const fn degrees(self) -> i32 {
        match self {
            Rotation::None => 0,
            Rotation::Cw90 => 90,
            Rotation::Cw180 => 180,
            Rotation::Cw270 => 270,
        }
    }

    /// The C struct also carries a region of interest, but every face task is
    /// built with `roi_allowed=false` (see `face_detector.cc` and
    /// `face_landmarker.cc` upstream), so it is always left unset here. A
    /// normalized-rect type will arrive with the tasks that do accept one.
    pub(crate) fn to_raw(self) -> sys::MpImageProcessingOptions {
        sys::MpImageProcessingOptions {
            has_region_of_interest: 0,
            region_of_interest: sys::MpRectF {
                left: 0.0,
                top: 0.0,
                bottom: 0.0,
                right: 0.0,
            },
            rotation_degrees: self.degrees(),
        }
    }
}

/// A 4x4 transformation matrix.
///
/// MediaPipe hands these over in Eigen's default **column-major** order (see
/// `matrix_converter.cc`), and this type preserves that, so use [`get`] rather
/// than indexing the raw array by hand.
///
/// [`get`]: Transform4x4::get
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform4x4([f32; 16]);

impl Transform4x4 {
    /// `row` and `col` are both `0..4`.
    pub const fn get(&self, row: usize, col: usize) -> f32 {
        self.0[col * 4 + row]
    }

    /// The raw column-major buffer, ready for OpenGL-style APIs.
    pub const fn as_column_major(&self) -> &[f32; 16] {
        &self.0
    }

    pub fn to_row_major(&self) -> [f32; 16] {
        let mut out = [0.0; 16];
        for row in 0..4 {
            for col in 0..4 {
                out[row * 4 + col] = self.get(row, col);
            }
        }
        out
    }

    pub(crate) fn from_raw(m: &sys::MpMatrix) -> Result<Self> {
        if m.rows != 4 || m.cols != 4 {
            return Err(Error::MatrixShape {
                rows: 4,
                cols: 4,
                got_rows: m.rows,
                got_cols: m.cols,
            });
        }
        let mut data = [0.0f32; 16];
        // SAFETY: rows*cols == 16 floats, allocated by MpMatrix and still owned
        // by the caller's result struct.
        unsafe { std::ptr::copy_nonoverlapping(m.data, data.as_mut_ptr(), 16) };
        Ok(Self(data))
    }
}

/// Which backend runs the model graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Delegate {
    #[default]
    Cpu,
    /// Requires a GPU-capable build and an EGL/GL context. Untested by this crate.
    Gpu,
    EdgeTpuNnapi,
}

impl Delegate {
    pub(crate) fn to_raw(self) -> sys::MpDelegate {
        match self {
            Delegate::Cpu => sys::MpDelegate::MP_DELEGATE_CPU,
            Delegate::Gpu => sys::MpDelegate::MP_DELEGATE_GPU,
            Delegate::EdgeTpuNnapi => sys::MpDelegate::MP_DELEGATE_EDGETPU_NNAPI,
        }
    }
}

/// Where a task's model comes from. One field rather than two mutually
/// exclusive builder setters, so "both" and "neither" are unrepresentable.
#[derive(Debug, Clone)]
pub enum ModelSource {
    Path(std::path::PathBuf),
    Bytes(Vec<u8>),
}

impl ModelSource {
    pub fn path(p: impl Into<std::path::PathBuf>) -> Self {
        ModelSource::Path(p.into())
    }

    pub fn bytes(b: impl Into<Vec<u8>>) -> Self {
        ModelSource::Bytes(b.into())
    }
}

/// A classification result: a label with a score.
#[derive(Debug, Clone, PartialEq)]
pub struct Category {
    pub index: i32,
    pub score: Confidence,
    pub category_name: Option<String>,
    pub display_name: Option<String>,
}

/// A detected keypoint, in normalized coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct Keypoint {
    pub point: NormalizedPoint2,
    pub label: Option<String>,
    /// `None` when the model does not score keypoints (`has_score` was false).
    pub score: Option<Confidence>,
}

/// A landmark, in normalized coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedLandmark {
    pub point: NormalizedPoint3,
    /// `None` when the C side's `has_visibility` was false — the flag and the
    /// value travel together so callers cannot read an unset value.
    pub visibility: Option<Confidence>,
    /// `None` when `has_presence` was false.
    pub presence: Option<Confidence>,
    pub name: Option<String>,
}

// --- conversion out of C memory -------------------------------------------
//
// Everything below copies. MediaPipe frees each result the moment the
// corresponding `Mp*CloseResult` runs (and, for live-stream callbacks, the
// instant the callback returns), so nothing may borrow from it.

/// # Safety
/// `p` must be null or a valid NUL-terminated string owned by MediaPipe.
pub(crate) unsafe fn opt_string(p: *const std::os::raw::c_char) -> Option<String> {
    if p.is_null() {
        None
    } else {
        // SAFETY: non-null here, and the caller guarantees a NUL-terminated
        // string owned by MediaPipe.
        Some(
            unsafe { std::ffi::CStr::from_ptr(p) }
                .to_string_lossy()
                .into_owned(),
        )
    }
}

impl Category {
    /// # Safety
    /// `c` must be a live `MpCategory` from a MediaPipe result.
    pub(crate) unsafe fn from_raw(c: &sys::MpCategory) -> Self {
        Category {
            index: c.index,
            score: Confidence::from_raw(c.score),
            // SAFETY: a field of the live result this function's contract covers.
            category_name: unsafe { opt_string(c.category_name) },
            // SAFETY: a field of the live result this function's contract covers.
            display_name: unsafe { opt_string(c.display_name) },
        }
    }

    /// # Safety
    /// `ptr` must point to `count` valid `MpCategory` values, or be null when
    /// `count` is zero.
    pub(crate) unsafe fn vec_from_raw(ptr: *const sys::MpCategory, count: u32) -> Vec<Self> {
        if ptr.is_null() {
            return Vec::new();
        }
        (0..count as usize)
            // SAFETY: the index is below the count MediaPipe reported for this array.
            .map(|i| unsafe { Category::from_raw(&*ptr.add(i)) })
            .collect()
    }
}

impl Keypoint {
    /// # Safety
    /// `k` must be a live `MpNormalizedKeypoint` from a MediaPipe result.
    pub(crate) unsafe fn from_raw(k: &sys::MpNormalizedKeypoint) -> Self {
        Keypoint {
            point: NormalizedPoint2::from_raw(k.x, k.y),
            // SAFETY: a field of the live result this function's contract covers.
            label: unsafe { opt_string(k.label) },
            score: k.has_score.then(|| Confidence::from_raw(k.score)),
        }
    }
}

impl NormalizedLandmark {
    /// # Safety
    /// `l` must be a live `MpNormalizedLandmark` from a MediaPipe result.
    pub(crate) unsafe fn from_raw(l: &sys::MpNormalizedLandmark) -> Self {
        NormalizedLandmark {
            point: NormalizedPoint3::from_raw(l.x, l.y, l.z),
            visibility: l.has_visibility.then(|| Confidence::from_raw(l.visibility)),
            presence: l.has_presence.then(|| Confidence::from_raw(l.presence)),
            // SAFETY: a field of the live result this function's contract covers.
            name: unsafe { opt_string(l.name) },
        }
    }
}

/// A 3D point in the task's world coordinates, in meters.
///
/// This is NOT normalized image space: the origin is roughly the subject's hip
/// centre, and the axes are the task's unmirrored basis. There is deliberately
/// no conversion to or from [`NormalizedPoint3`], because they are different
/// spaces; mixing them is exactly what this type exists to prevent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldPoint3 {
    x: f32,
    y: f32,
    z: f32,
}

impl WorldPoint3 {
    /// Builds a world point from meters in the task's world basis.
    ///
    /// This is not a normalized image point: callers must have a real
    /// world-space measurement, which is why the constructor is named for it
    /// rather than being a generic `new`.
    pub const fn from_meters(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// X in meters, image-right.
    pub const fn x(self) -> f32 {
        self.x
    }

    /// Y in meters, down.
    pub const fn y(self) -> f32 {
        self.y
    }

    /// Z in meters, with smaller meaning nearer the camera.
    pub const fn z(self) -> f32 {
        self.z
    }

    /// The three coordinates in task order.
    pub const fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    pub(crate) const fn from_raw(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

/// A landmark in world coordinates, in meters.
///
/// Separate from [`NormalizedLandmark`] so a normalized point can never be
/// substituted for a world point. Visibility and presence keep their missing
/// state rather than being defaulted.
#[derive(Debug, Clone, PartialEq)]
pub struct WorldLandmark {
    pub point: WorldPoint3,
    /// `None` when the C side's `has_visibility` was false.
    pub visibility: Option<Confidence>,
    /// `None` when `has_presence` was false.
    pub presence: Option<Confidence>,
    pub name: Option<String>,
}

impl WorldLandmark {
    /// # Safety
    /// `l` must be a live `MpLandmark` from a MediaPipe result.
    pub(crate) unsafe fn from_raw(l: &sys::MpLandmark) -> Self {
        WorldLandmark {
            point: WorldPoint3::from_raw(l.x, l.y, l.z),
            visibility: l.has_visibility.then(|| Confidence::from_raw(l.visibility)),
            presence: l.has_presence.then(|| Confidence::from_raw(l.presence)),
            // SAFETY: a field of the live result this function's contract covers.
            name: unsafe { opt_string(l.name) },
        }
    }
}
