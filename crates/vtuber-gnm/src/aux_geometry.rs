//! Auxiliary geometry features computed from the current GNM surface and the
//! neutral identity calibration (Issue #55.1).
//!
//! These features are pure functions of validated engine-neutral geometry:
//! they never read MediaPipe blendshape scores and never depend on a preview
//! mirror. Each feature is neutral-relative (computed against
//! [`GnmIdentityCalibration::neutral_surface_reference`]) and scale-normalized
//! by a calibration-owned normalization scale, so a rigid head transform alone
//! cannot change the value.

use crate::dense_regions::{DenseRegionGroups, EyelidRing, IndexedRow};
use crate::identity_calibration::GnmIdentityCalibration;
use crate::{
    AnatomicalSide, DenseCorrespondenceSet, GnmDenseError, GnmExpressionState, GnmIdentityState,
    GnmJointState, GnmModel, GnmSparseVertices,
};

/// MediaPipe canonical indices for the iris/gaze features.
const IRIS_CENTER_RIGHT_MP: usize = 468;
const IRIS_CENTER_LEFT_MP: usize = 473;
const IRIS_APEX_RIGHT_MP: usize = 159;
const IRIS_APEX_LEFT_MP: usize = 386;
const IRIS_LOWER_MID_RIGHT_MP: usize = 145;
const IRIS_LOWER_MID_LEFT_MP: usize = 374;
const IRIS_INNER_CORNER_RIGHT_MP: usize = 173;
const IRIS_INNER_CORNER_LEFT_MP: usize = 398;
const IRIS_OUTER_CORNER_RIGHT_MP: usize = 33;
const IRIS_OUTER_CORNER_LEFT_MP: usize = 263;

/// Failure while computing an auxiliary geometry feature.
#[derive(Debug)]
pub enum GnmAuxGeometryError {
    /// The current surface could not be evaluated for the given state.
    SurfaceEvaluation(GnmDenseError),
    /// The calibration's neutral surface reference does not address the same
    /// correspondence set as the current surface.
    CalibrationSurfaceLengthMismatch {
        /// Row count of the current mapping.
        mapping_rows: usize,
        /// Length of the calibration's stored neutral surface reference.
        calibration_rows: usize,
    },
    /// A required evaluated surface point is non-finite.
    NonFiniteSurfacePoint {
        /// Correspondence-set row index of the invalid point.
        row: usize,
    },
    /// The calibration does not carry the normalization scale required by the
    /// feature.
    MissingNormalizationScale {
        /// Name of the missing scale.
        field: &'static str,
    },
    /// The calibration carries a degenerate (non-finite or non-positive)
    /// normalization scale.
    DegenerateNormalizationScale {
        /// Name of the degenerate scale.
        field: &'static str,
        /// The invalid value.
        value: f32,
    },
    /// A computed snapshot feature value is non-finite.
    NonFiniteFeature {
        /// Name of the invalid feature.
        field: &'static str,
    },
}

impl std::fmt::Display for GnmAuxGeometryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SurfaceEvaluation(error) => {
                write!(
                    formatter,
                    "auxiliary geometry surface evaluation failed: {error}"
                )
            }
            Self::CalibrationSurfaceLengthMismatch {
                mapping_rows,
                calibration_rows,
            } => write!(
                formatter,
                "calibration neutral surface has {calibration_rows} rows but mapping has {mapping_rows}"
            ),
            Self::NonFiniteSurfacePoint { row } => {
                write!(
                    formatter,
                    "auxiliary geometry surface point {row} is non-finite"
                )
            }
            Self::MissingNormalizationScale { field } => {
                write!(
                    formatter,
                    "calibration is missing `{field}` normalization scale"
                )
            }
            Self::DegenerateNormalizationScale { field, value } => write!(
                formatter,
                "calibration normalization scale `{field}` is degenerate: {value}"
            ),
            Self::NonFiniteFeature { field } => {
                write!(formatter, "facial feature `{field}` is non-finite")
            }
        }
    }
}

impl std::error::Error for GnmAuxGeometryError {}

/// One eye-aperture auxiliary feature for a single anatomical side.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeApertureFeature {
    /// Anatomical side of the eye. Never derived from image-space mirroring.
    pub side: AnatomicalSide,
    /// Current lid aperture in model-space units.
    pub current_aperture: f32,
    /// Neutral-calibration aperture in model-space units.
    pub neutral_aperture: f32,
    /// `(current - neutral) / inter-ocular scale`; negative means more closed
    /// than the calibrated neutral, positive means wider.
    pub normalized_delta: f32,
}

/// Left and right eye-aperture auxiliary features.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeAuxFeatures {
    /// Subject's anatomical right eye.
    pub right: EyeApertureFeature,
    /// Subject's anatomical left eye.
    pub left: EyeApertureFeature,
}

/// Picks the middle row of a canonical arc slice.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn arc_midpoint(arc: &[IndexedRow]) -> &IndexedRow {
    &arc[arc.len() / 2]
}

/// Vertical aperture proxy for one eyelid ring: the distance between the
/// upper-lid apex and the lower-lid midpoint in model space.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn ring_aperture(ring: &EyelidRing, surface: &[[f32; 3]]) -> (f32, usize, usize) {
    let upper = arc_midpoint(ring.upper_arc());
    let lower = arc_midpoint(ring.lower_arc());
    let upper_point = surface[upper.index];
    let lower_point = surface[lower.index];
    let delta = [
        upper_point[0] - lower_point[0],
        upper_point[1] - lower_point[1],
        upper_point[2] - lower_point[2],
    ];
    (
        (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt(),
        upper.index,
        lower.index,
    )
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn require_finite_row(surface: &[[f32; 3]], row: usize) -> Result<(), GnmAuxGeometryError> {
    if surface[row].iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(GnmAuxGeometryError::NonFiniteSurfacePoint { row })
    }
}

fn side_aperture(
    ring: &EyelidRing,
    side: AnatomicalSide,
    surface: &[[f32; 3]],
    neutral_surface: &[[f32; 3]],
) -> Result<EyeApertureFeature, GnmAuxGeometryError> {
    let (current_aperture, upper_row, _) = ring_aperture(ring, surface);
    require_finite_row(surface, upper_row)?;
    let lower_row_of_current = arc_midpoint(ring.lower_arc()).index;
    require_finite_row(surface, lower_row_of_current)?;
    let (neutral_aperture, upper_neutral_row, _) = ring_aperture(ring, neutral_surface);
    require_finite_row(neutral_surface, upper_neutral_row)?;
    let lower_neutral_row = arc_midpoint(ring.lower_arc()).index;
    require_finite_row(neutral_surface, lower_neutral_row)?;
    Ok(EyeApertureFeature {
        side,
        current_aperture,
        neutral_aperture,
        normalized_delta: 0.0,
    })
}

/// Computes left/right eye-aperture auxiliary features from the current GNM
/// surface and the neutral identity calibration.
///
/// The function is pure: it evaluates the mapped surface for the given state,
/// measures each anatomical eyelid ring against the calibration's neutral
/// surface reference, and normalizes the delta by the calibration's
/// inter-ocular scale. Rigid head transforms preserve pairwise distances, so
/// pose alone cannot change these features. Sides are keyed on the mapping's
/// anatomical topology, never on image-space orientation.
pub fn compute_eye_aperture_features(
    model: &GnmModel,
    identity: &GnmIdentityState,
    expression: &GnmExpressionState,
    joints: &GnmJointState,
    mapping: &DenseCorrespondenceSet,
    groups: &DenseRegionGroups,
    calibration: &GnmIdentityCalibration,
) -> Result<EyeAuxFeatures, GnmAuxGeometryError> {
    // Fail closed on scale availability before touching geometry.
    let scale = checked_scale(
        calibration.normalization_scales().inter_ocular,
        "inter_ocular",
    )?;

    let neutral_surface = calibration.neutral_surface_reference();
    if neutral_surface.len() != mapping.len() {
        return Err(GnmAuxGeometryError::CalibrationSurfaceLengthMismatch {
            mapping_rows: mapping.len(),
            calibration_rows: neutral_surface.len(),
        });
    }

    let mut surface = GnmSparseVertices::with_len(mapping.len());
    mapping
        .evaluate_surface(model, identity, expression, joints, &mut surface)
        .map_err(GnmAuxGeometryError::SurfaceEvaluation)?;
    eye_aperture_from_parts(surface.values(), neutral_surface, groups, scale)
}

/// Validates one optional normalization scale.
fn checked_scale(scale: Option<f32>, field: &'static str) -> Result<f32, GnmAuxGeometryError> {
    let scale = scale.ok_or(GnmAuxGeometryError::MissingNormalizationScale { field })?;
    if !scale.is_finite() || scale <= 0.0 {
        return Err(GnmAuxGeometryError::DegenerateNormalizationScale {
            field,
            value: scale,
        });
    }
    Ok(scale)
}

/// Computes the eye-aperture family from an already-evaluated surface.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
pub(crate) fn eye_aperture_from_parts(
    surface: &[[f32; 3]],
    neutral_surface: &[[f32; 3]],
    groups: &DenseRegionGroups,
    scale: f32,
) -> Result<EyeAuxFeatures, GnmAuxGeometryError> {
    let mut right = side_aperture(
        groups.eyes().right(),
        AnatomicalSide::Right,
        surface,
        neutral_surface,
    )?;
    let mut left = side_aperture(
        groups.eyes().left(),
        AnatomicalSide::Left,
        surface,
        neutral_surface,
    )?;
    right.normalized_delta = (right.current_aperture - right.neutral_aperture) / scale;
    left.normalized_delta = (left.current_aperture - left.neutral_aperture) / scale;
    Ok(EyeAuxFeatures { right, left })
}

/// Semantic landmark slots required by the jaw/mouth features (MediaPipe
/// canonical indices): upper/lower lip centers, mouth corners, chin, nose
/// tip, and the two face-oval jaw anchors.
const MOUTH_UPPER_LIP_CENTER_MP: usize = 0;
const MOUTH_LOWER_LIP_CENTER_MP: usize = 17;
const MOUTH_CORNER_RIGHT_MP: usize = 61;
const MOUTH_CORNER_LEFT_MP: usize = 291;
const JAW_CHIN_MP: usize = 152;
const JAW_NOSE_TIP_MP: usize = 4;
const JAW_ANCHOR_RIGHT_MP: usize = 234;
const JAW_ANCHOR_LEFT_MP: usize = 454;

/// Jaw and mouth auxiliary features, each neutral-relative and normalized by
/// the calibration's mouth width.
///
/// A `None` value means the mapping does not carry the semantic rows needed
/// for that feature; the value is unavailable and is never fabricated.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MouthAuxFeatures {
    /// Lip-center aperture delta; positive means more open than neutral.
    pub jaw_open: Option<f32>,
    /// Chin-to-nose-tip distance delta; negative means the jaw moved forward.
    pub jaw_forward: Option<f32>,
    /// Chin anchor-distance asymmetry delta; positive means the jaw shifted
    /// toward the subject's anatomical left. Sign is pinned by fixture.
    pub jaw_lateral: Option<f32>,
    /// Mouth-corner distance delta; positive means wider than neutral.
    pub width_delta: Option<f32>,
    /// Mean corner rise toward the upper-lip center; positive means lifted
    /// (smile-like), negative means lowered (frown-like).
    pub corner_lift: Option<f32>,
}

fn squared_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let delta = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    squared_distance(a, b).sqrt()
}

/// Looks up a MediaPipe slot in the correspondence set.
fn row_index_for(mapping: &DenseCorrespondenceSet, mediapipe: usize) -> Option<usize> {
    mapping
        .rows()
        .iter()
        .position(|row| row.mediapipe_index == mediapipe)
}

/// Measures one distance-based feature if both rows exist in the mapping.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn measured_delta(
    surface: &[[f32; 3]],
    neutral_surface: &[[f32; 3]],
    first: Option<usize>,
    second: Option<usize>,
) -> Result<Option<f32>, GnmAuxGeometryError> {
    let (first, second) = match (first, second) {
        (Some(first), Some(second)) => (first, second),
        _ => return Ok(None),
    };
    for row in [first, second] {
        require_finite_row(surface, row)?;
        require_finite_row(neutral_surface, row)?;
    }
    let current = distance(surface[first], surface[second]);
    let neutral = distance(neutral_surface[first], neutral_surface[second]);
    Ok(Some(current - neutral))
}

/// Computes jaw and mouth auxiliary features from the current GNM surface and
/// the neutral identity calibration.
///
/// Every feature is a pairwise model-space distance delta against the
/// calibration's neutral surface reference, so a rigid head transform alone
/// cannot change any of them. All deltas are divided by the calibration's
/// mouth-width scale. Features whose semantic rows are absent from the
/// mapping are returned as `None` instead of being fabricated. The lateral
/// sign convention is fixed by construction: `jaw_lateral` is positive when
/// the chin sits closer to the subject-left oval anchor than to the
/// subject-right one relative to neutral.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
pub fn compute_mouth_aux_features(
    model: &GnmModel,
    identity: &GnmIdentityState,
    expression: &GnmExpressionState,
    joints: &GnmJointState,
    mapping: &DenseCorrespondenceSet,
    calibration: &GnmIdentityCalibration,
) -> Result<MouthAuxFeatures, GnmAuxGeometryError> {
    let scale = checked_scale(
        calibration.normalization_scales().mouth_width,
        "mouth_width",
    )?;

    let neutral_surface = calibration.neutral_surface_reference();
    if neutral_surface.len() != mapping.len() {
        return Err(GnmAuxGeometryError::CalibrationSurfaceLengthMismatch {
            mapping_rows: mapping.len(),
            calibration_rows: neutral_surface.len(),
        });
    }

    let mut surface_buffer = GnmSparseVertices::with_len(mapping.len());
    mapping
        .evaluate_surface(model, identity, expression, joints, &mut surface_buffer)
        .map_err(GnmAuxGeometryError::SurfaceEvaluation)?;
    mouth_from_parts(surface_buffer.values(), neutral_surface, mapping, scale)
}

/// Computes the jaw/mouth family from an already-evaluated surface.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
pub(crate) fn mouth_from_parts(
    surface: &[[f32; 3]],
    neutral_surface: &[[f32; 3]],
    mapping: &DenseCorrespondenceSet,
    scale: f32,
) -> Result<MouthAuxFeatures, GnmAuxGeometryError> {
    let lookup = |mediapipe: usize| row_index_for(mapping, mediapipe);
    let upper_center = lookup(MOUTH_UPPER_LIP_CENTER_MP);
    let lower_center = lookup(MOUTH_LOWER_LIP_CENTER_MP);
    let corner_right = lookup(MOUTH_CORNER_RIGHT_MP);
    let corner_left = lookup(MOUTH_CORNER_LEFT_MP);
    let chin = lookup(JAW_CHIN_MP);
    let nose_tip = lookup(JAW_NOSE_TIP_MP);
    let anchor_right = lookup(JAW_ANCHOR_RIGHT_MP);
    let anchor_left = lookup(JAW_ANCHOR_LEFT_MP);

    let normalize = |value: Option<f32>| value.map(|value| value / scale);

    // Lip aperture (jaw open proxy).
    let jaw_open = normalize(measured_delta(
        surface,
        neutral_surface,
        upper_center,
        lower_center,
    )?);
    // Jaw forward: chin-to-nose-tip shrinks as the jaw moves forward.
    let jaw_forward = normalize(measured_delta(surface, neutral_surface, chin, nose_tip)?);
    // Jaw lateral: asymmetry between the chin-to-anchor distances.
    let lateral_current = match (chin, anchor_right, anchor_left) {
        (Some(chin), Some(anchor_right), Some(anchor_left)) => {
            for row in [chin, anchor_right, anchor_left] {
                require_finite_row(surface, row)?;
                require_finite_row(neutral_surface, row)?;
            }
            Some(
                distance(surface[chin], surface[anchor_right])
                    - distance(surface[chin], surface[anchor_left]),
            )
        }
        _ => None,
    };
    let lateral_neutral = match (chin, anchor_right, anchor_left) {
        (Some(chin), Some(anchor_right), Some(anchor_left)) => Some(
            distance(neutral_surface[chin], neutral_surface[anchor_right])
                - distance(neutral_surface[chin], neutral_surface[anchor_left]),
        ),
        _ => None,
    };
    let jaw_lateral = normalize(
        lateral_current
            .zip(lateral_neutral)
            .map(|(current, neutral)| current - neutral),
    );
    // Corner width.
    let width_delta = normalize(measured_delta(
        surface,
        neutral_surface,
        corner_right,
        corner_left,
    )?);
    // Corner lift: corners rise toward the upper-lip center when smiling.
    let lift_side =
        |corner: Option<usize>| measured_delta(surface, neutral_surface, corner, upper_center);
    let corner_lift = match (
        normalize(lift_side(corner_right)?),
        normalize(lift_side(corner_left)?),
    ) {
        (Some(right), Some(left)) => Some(-(right + left) / 2.0),
        _ => None,
    };

    Ok(MouthAuxFeatures {
        jaw_open,
        jaw_forward,
        jaw_lateral,
        width_delta,
        corner_lift,
    })
}

/// Semantic landmark slots required by the brow features (MediaPipe canonical
/// indices): inner/mid/outer lower-arc brow points per anatomical side and the
/// upper-lid apex used as the fixed eye reference.
const BROW_INNER_RIGHT_MP: usize = 70;
const BROW_MID_RIGHT_MP: usize = 105;
const BROW_OUTER_RIGHT_MP: usize = 65;
const BROW_INNER_LEFT_MP: usize = 300;
const BROW_MID_LEFT_MP: usize = 334;
const BROW_OUTER_LEFT_MP: usize = 295;
const LID_APEX_RIGHT_MP: usize = 159;
const LID_APEX_LEFT_MP: usize = 386;

/// Brow auxiliary features for one anatomical side, each neutral-relative and
/// normalized by the calibration's inter-ocular scale.
///
/// A `None` value means the mapping does not carry the semantic rows needed
/// for that feature; the value is unavailable and is never fabricated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrowSideAuxFeatures {
    /// Anatomical side of the brow. Never derived from image mirroring.
    pub side: AnatomicalSide,
    /// Inner-brow displacement away from the upper-lid apex relative to
    /// neutral; positive means raised (inner brow up).
    pub inner_rise: Option<f32>,
    /// Mid-brow displacement toward the upper-lid apex relative to neutral;
    /// positive means lowered (brow down).
    pub brow_lower: Option<f32>,
    /// Outer-brow displacement away from the upper-lid apex relative to
    /// neutral; positive means raised (outer brow up).
    pub outer_rise: Option<f32>,
}

/// Left and right brow auxiliary features.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrowAuxFeatures {
    /// Subject's anatomical right brow.
    pub right: BrowSideAuxFeatures,
    /// Subject's anatomical left brow.
    pub left: BrowSideAuxFeatures,
}

/// Computes brow auxiliary features from the current GNM surface and the
/// neutral identity calibration.
///
/// Every feature is a neutral-relative brow-to-upper-lid-apex distance delta,
/// divided by the calibration's inter-ocular scale, so a rigid head transform
/// or a uniform camera scale alone cannot change any of them. Sides are keyed
/// on the mapping's anatomical MediaPipe topology (70/105/65 right,
/// 300/334/295 left), never on image-space orientation. Features whose
/// semantic rows are absent from the mapping are returned as `None` instead
/// of being fabricated.
pub fn compute_brow_aux_features(
    model: &GnmModel,
    identity: &GnmIdentityState,
    expression: &GnmExpressionState,
    joints: &GnmJointState,
    mapping: &DenseCorrespondenceSet,
    calibration: &GnmIdentityCalibration,
) -> Result<BrowAuxFeatures, GnmAuxGeometryError> {
    let scale = checked_scale(
        calibration.normalization_scales().inter_ocular,
        "inter_ocular",
    )?;

    let neutral_surface = calibration.neutral_surface_reference();
    if neutral_surface.len() != mapping.len() {
        return Err(GnmAuxGeometryError::CalibrationSurfaceLengthMismatch {
            mapping_rows: mapping.len(),
            calibration_rows: neutral_surface.len(),
        });
    }

    let mut surface_buffer = GnmSparseVertices::with_len(mapping.len());
    mapping
        .evaluate_surface(model, identity, expression, joints, &mut surface_buffer)
        .map_err(GnmAuxGeometryError::SurfaceEvaluation)?;
    brow_from_parts(surface_buffer.values(), neutral_surface, mapping, scale)
}

/// Computes the brow family from an already-evaluated surface.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
pub(crate) fn brow_from_parts(
    surface: &[[f32; 3]],
    neutral_surface: &[[f32; 3]],
    mapping: &DenseCorrespondenceSet,
    scale: f32,
) -> Result<BrowAuxFeatures, GnmAuxGeometryError> {
    let lookup = |mediapipe: usize| row_index_for(mapping, mediapipe);
    let side_features = |side: AnatomicalSide,
                         inner: usize,
                         mid: usize,
                         outer: usize,
                         lid_apex: usize|
     -> Result<BrowSideAuxFeatures, GnmAuxGeometryError> {
        let apex = lookup(lid_apex);
        Ok(BrowSideAuxFeatures {
            side,
            inner_rise: measured_delta(surface, neutral_surface, lookup(inner), apex)?
                .map(|value| value / scale),
            brow_lower: measured_delta(surface, neutral_surface, lookup(mid), apex)?
                .map(|value| -value / scale),
            outer_rise: measured_delta(surface, neutral_surface, lookup(outer), apex)?
                .map(|value| value / scale),
        })
    };

    let right = side_features(
        AnatomicalSide::Right,
        BROW_INNER_RIGHT_MP,
        BROW_MID_RIGHT_MP,
        BROW_OUTER_RIGHT_MP,
        LID_APEX_RIGHT_MP,
    )?;
    let left = side_features(
        AnatomicalSide::Left,
        BROW_INNER_LEFT_MP,
        BROW_MID_LEFT_MP,
        BROW_OUTER_LEFT_MP,
        LID_APEX_LEFT_MP,
    )?;

    Ok(BrowAuxFeatures { right, left })
}

/// Iris/gaze auxiliary feature for one anatomical side.
///
/// Both values are neutral-relative pairwise-distance deltas divided by the
/// calibration's inter-ocular scale, so a rigid head transform alone cannot
/// change them. A `None` value means the mapping does not carry the rows the
/// component needs; it is unavailable and never fabricated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IrisSideAuxFeature {
    /// Anatomical side of the iris. Never derived from image mirroring.
    pub side: AnatomicalSide,
    /// `(dist(iris, lower-lid mid) - dist(iris, upper-lid apex))` delta
    /// versus neutral; positive means the iris sits higher than at neutral
    /// (gaze up).
    pub vertical_delta: Option<f32>,
    /// `(dist(iris, inner corner) - dist(iris, outer corner))` delta versus
    /// neutral; positive means the iris moved toward the outer corner
    /// (outward lateral gaze).
    pub horizontal_delta: Option<f32>,
}

/// Left and right iris/gaze auxiliary features.
///
/// A side is `None` when the mapping carries no iris-center row for it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IrisAuxFeatures {
    /// Subject's anatomical right eye.
    pub right: Option<IrisSideAuxFeature>,
    /// Subject's anatomical left eye.
    pub left: Option<IrisSideAuxFeature>,
}

/// Computes the iris family from an already-evaluated surface.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
pub(crate) fn iris_from_parts(
    surface: &[[f32; 3]],
    neutral_surface: &[[f32; 3]],
    mapping: &DenseCorrespondenceSet,
    scale: f32,
) -> Result<IrisAuxFeatures, GnmAuxGeometryError> {
    let lookup = |mediapipe: usize| row_index_for(mapping, mediapipe);
    let side_feature = |side: AnatomicalSide,
                        iris_mp: usize,
                        apex_mp: usize,
                        lower_mp: usize,
                        inner_mp: usize,
                        outer_mp: usize| {
        let Some(iris) = lookup(iris_mp) else {
            return Ok(None);
        };
        let vertical_delta = match (
            measured_delta(surface, neutral_surface, Some(iris), lookup(lower_mp))?,
            measured_delta(surface, neutral_surface, Some(iris), lookup(apex_mp))?,
        ) {
            (Some(to_lower), Some(to_apex)) => Some((to_lower - to_apex) / scale),
            _ => None,
        };
        let horizontal_delta = match (
            measured_delta(surface, neutral_surface, Some(iris), lookup(inner_mp))?,
            measured_delta(surface, neutral_surface, Some(iris), lookup(outer_mp))?,
        ) {
            (Some(to_inner), Some(to_outer)) => Some((to_inner - to_outer) / scale),
            _ => None,
        };
        Ok(Some(IrisSideAuxFeature {
            side,
            vertical_delta,
            horizontal_delta,
        }))
    };

    Ok(IrisAuxFeatures {
        right: side_feature(
            AnatomicalSide::Right,
            IRIS_CENTER_RIGHT_MP,
            IRIS_APEX_RIGHT_MP,
            IRIS_LOWER_MID_RIGHT_MP,
            IRIS_INNER_CORNER_RIGHT_MP,
            IRIS_OUTER_CORNER_RIGHT_MP,
        )?,
        left: side_feature(
            AnatomicalSide::Left,
            IRIS_CENTER_LEFT_MP,
            IRIS_APEX_LEFT_MP,
            IRIS_LOWER_MID_LEFT_MP,
            IRIS_INNER_CORNER_LEFT_MP,
            IRIS_OUTER_CORNER_LEFT_MP,
        )?,
    })
}

/// Engine-neutral facial feature snapshot consumed by the ARKit projector in
/// a single decode pass (Issue #67.1).
///
/// Every member is neutral-relative, normalized by a person-specific
/// calibration scale, invariant under rigid head transforms, and keyed on
/// anatomical sides rather than image orientation. Unavailable features are
/// `Option::None`, never fabricated. The snapshot is validated: all present
/// values are finite.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GnmFacialFeatures {
    /// Eye-aperture features.
    pub eyes: EyeAuxFeatures,
    /// Iris/gaze features (unavailable without iris rows).
    pub irises: IrisAuxFeatures,
    /// Jaw/mouth features (unavailable without mouth-width scale or rows).
    pub mouth_jaw: MouthAuxFeatures,
    /// Brow features.
    pub brows: BrowAuxFeatures,
}

impl GnmFacialFeatures {
    /// Fails closed if any present feature value is non-finite.
    ///
    /// # Errors
    ///
    /// Returns [`GnmAuxGeometryError::NonFiniteFeature`] naming the first
    /// invalid feature.
    pub fn validate(&self) -> Result<(), GnmAuxGeometryError> {
        for eye in [&self.eyes.right, &self.eyes.left] {
            for (field, value) in [
                ("current_aperture", eye.current_aperture),
                ("neutral_aperture", eye.neutral_aperture),
                ("normalized_delta", eye.normalized_delta),
            ] {
                if !value.is_finite() {
                    return Err(GnmAuxGeometryError::NonFiniteFeature { field });
                }
            }
        }
        for iris in [self.irises.right, self.irises.left].into_iter().flatten() {
            for (field, value) in [
                ("iris_vertical_delta", iris.vertical_delta),
                ("iris_horizontal_delta", iris.horizontal_delta),
            ] {
                if let Some(value) = value
                    && !value.is_finite()
                {
                    return Err(GnmAuxGeometryError::NonFiniteFeature { field });
                }
            }
        }
        for (field, value) in [
            ("jaw_open", self.mouth_jaw.jaw_open),
            ("jaw_forward", self.mouth_jaw.jaw_forward),
            ("jaw_lateral", self.mouth_jaw.jaw_lateral),
            ("width_delta", self.mouth_jaw.width_delta),
            ("corner_lift", self.mouth_jaw.corner_lift),
        ] {
            if let Some(value) = value
                && !value.is_finite()
            {
                return Err(GnmAuxGeometryError::NonFiniteFeature { field });
            }
        }
        for brow in [&self.brows.right, &self.brows.left] {
            for (field, value) in [
                ("brow_inner_rise", brow.inner_rise),
                ("brow_lower", brow.brow_lower),
                ("brow_outer_rise", brow.outer_rise),
            ] {
                if let Some(value) = value
                    && !value.is_finite()
                {
                    return Err(GnmAuxGeometryError::NonFiniteFeature { field });
                }
            }
        }
        Ok(())
    }
}

/// Aggregates every auxiliary geometry family into one validated snapshot,
/// evaluating the surface exactly once.
///
/// # Errors
///
/// Propagates typed failures from scale availability, surface evaluation,
/// calibration alignment, or snapshot validation.
pub fn compute_gnm_facial_features(
    model: &GnmModel,
    identity: &GnmIdentityState,
    expression: &GnmExpressionState,
    joints: &GnmJointState,
    mapping: &DenseCorrespondenceSet,
    groups: &DenseRegionGroups,
    calibration: &GnmIdentityCalibration,
) -> Result<GnmFacialFeatures, GnmAuxGeometryError> {
    let scales = calibration.normalization_scales();
    let inter_ocular = checked_scale(scales.inter_ocular, "inter_ocular")?;
    let mouth_width = checked_scale(scales.mouth_width, "mouth_width")?;

    let neutral_surface = calibration.neutral_surface_reference();
    if neutral_surface.len() != mapping.len() {
        return Err(GnmAuxGeometryError::CalibrationSurfaceLengthMismatch {
            mapping_rows: mapping.len(),
            calibration_rows: neutral_surface.len(),
        });
    }

    // Exactly one surface evaluation feeds every feature family.
    let mut surface_buffer = GnmSparseVertices::with_len(mapping.len());
    mapping
        .evaluate_surface(model, identity, expression, joints, &mut surface_buffer)
        .map_err(GnmAuxGeometryError::SurfaceEvaluation)?;
    let surface = surface_buffer.values();

    let snapshot = GnmFacialFeatures {
        eyes: eye_aperture_from_parts(surface, neutral_surface, groups, inter_ocular)?,
        irises: iris_from_parts(surface, neutral_surface, mapping, inter_ocular)?,
        brows: brow_from_parts(surface, neutral_surface, mapping, inter_ocular)?,
        mouth_jaw: mouth_from_parts(surface, neutral_surface, mapping, mouth_width)?,
    };
    snapshot.validate()?;
    Ok(snapshot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense_regions::topology;
    use crate::{
        AnatomicalSide, CorrespondenceProvenance, CorrespondenceReliability, DenseArray,
        DenseMappingVersion, FaceRegion, FixedGnmIdentity, GNM_HEAD_V3_EXPRESSION_DIM,
        GNM_HEAD_V3_IDENTITY_DIM, GNM_HEAD_V3_VERSION, GnmIdentityCalibration,
        GnmIdentityCalibrationError, GnmModelData, GnmVariant, IdentityFitDiagnostics,
        MEDIAPIPE_FACE_LANDMARK_COUNT, NeutralNormalizationScales, NeutralPoseDiversity,
    };

    const RIGHT_UPPER_APEX_MP: usize = 159;
    const RIGHT_LOWER_MID_MP: usize = 145;
    const LEFT_UPPER_APEX_MP: usize = 386;
    const LEFT_LOWER_MID_MP: usize = 374;

    fn region_tag(mp: usize) -> FaceRegion {
        if topology::NOSE.contains(&mp) {
            FaceRegion::Nose
        } else if topology::FACE_OVAL.contains(&mp) {
            FaceRegion::Contour
        } else if mp == topology::IRIS_CENTER_RIGHT || mp == topology::IRIS_CENTER_LEFT {
            FaceRegion::Iris
        } else if topology::is_eyelid(mp) {
            FaceRegion::Eye
        } else if topology::is_brow(mp) {
            FaceRegion::Brow
        } else if topology::LIPS.contains(&mp) {
            FaceRegion::Mouth
        } else {
            FaceRegion::Other
        }
    }

    fn anatomical_side(mp: usize) -> AnatomicalSide {
        if topology::EYE_RIGHT.contains(&mp)
            || topology::BROW_RIGHT.contains(&mp)
            || topology::LIPS.contains(&mp) && mp < 300
        {
            AnatomicalSide::Right
        } else if topology::EYE_LEFT.contains(&mp) || topology::BROW_LEFT.contains(&mp) {
            AnatomicalSide::Left
        } else {
            AnatomicalSide::Midline
        }
    }

    fn row(mp: usize) -> crate::MediaPipeGnmDenseCorrespondence {
        let target = if mp == topology::IRIS_CENTER_RIGHT || mp == topology::IRIS_CENTER_LEFT {
            crate::GnmSurfacePointRef::Barycentric {
                vertex_indices: [mp, mp + 1, mp + 2],
                weights: [0.5, 0.25, 0.25],
            }
        } else {
            crate::GnmSurfacePointRef::Vertex { vertex_index: mp }
        };
        crate::MediaPipeGnmDenseCorrespondence {
            mediapipe_index: mp,
            target,
            region: region_tag(mp),
            anatomical_side: anatomical_side(mp),
            base_weight: 1.0,
            provenance: CorrespondenceProvenance::RepositoryValidated,
            reliability: CorrespondenceReliability::High,
        }
    }

    fn mapping_version() -> DenseMappingVersion {
        DenseMappingVersion {
            schema_revision: 1,
            model_version: GNM_HEAD_V3_VERSION,
        }
    }

    /// Model whose template places every vertex at `[x, y, z] = [i % 7, i % 5, 0]`
    /// plus explicit eyelid-motion expression channels:
    /// channel 0 lowers both upper-lid apexes (closure),
    /// channel 1 raises them (wide).
    fn eyelid_motion_model() -> GnmModel {
        let vertex_count = MEDIAPIPE_FACE_LANDMARK_COUNT + 3;
        let identity_dim = GNM_HEAD_V3_IDENTITY_DIM;
        let expression_dim = GNM_HEAD_V3_EXPRESSION_DIM;
        let mut vertices = Vec::with_capacity(vertex_count * 3);
        for index in 0..vertex_count {
            vertices.extend_from_slice(&[(index % 7) as f32, (index % 5) as f32, 0.0]);
        }
        // Give both eyelid aperture pairs identical, aligned geometry so the
        // two anatomical sides are symmetric in the fixture.
        for (apex, lower, x) in [
            (RIGHT_UPPER_APEX_MP, RIGHT_LOWER_MID_MP, 5.0),
            (LEFT_UPPER_APEX_MP, LEFT_LOWER_MID_MP, 1.0),
        ] {
            vertices[apex * 3..apex * 3 + 3].copy_from_slice(&[x, 4.0, 0.0]);
            vertices[lower * 3..lower * 3 + 3].copy_from_slice(&[x, 0.0, 0.0]);
        }
        let mut expression_basis = vec![0.0f32; expression_dim * vertex_count * 3];
        for apex in [RIGHT_UPPER_APEX_MP, LEFT_UPPER_APEX_MP] {
            // Channel 0 (closure): move the apex down by 0.4.
            expression_basis[apex * 3 + 1] = -0.4;
            // Channel 1 (wide): move the apex up by 0.6.
            expression_basis[(vertex_count + apex) * 3 + 1] = 0.6;
        }
        GnmModel::from_data(GnmModelData {
            version: GNM_HEAD_V3_VERSION,
            variant: GnmVariant::Head,
            template_vertices: DenseArray::new("vertices", vec![vertex_count, 3], vertices)
                .unwrap(),
            template_joints: DenseArray::new("joints", vec![1, 3], vec![0.0; 3]).unwrap(),
            vertex_identity_basis: DenseArray::new(
                "identity",
                vec![identity_dim, vertex_count, 3],
                vec![0.0; identity_dim * vertex_count * 3],
            )
            .unwrap(),
            joint_identity_basis: DenseArray::new(
                "joint_identity",
                vec![identity_dim, 1, 3],
                vec![0.0; identity_dim * 3],
            )
            .unwrap(),
            expression_basis: DenseArray::new(
                "expression",
                vec![expression_dim, vertex_count, 3],
                expression_basis,
            )
            .unwrap(),
            joint_parent_indices: vec![-1],
            skinning_weights: DenseArray::new(
                "weights",
                vec![1, vertex_count],
                vec![1.0; vertex_count],
            )
            .unwrap(),
            pose_correctives_regressor: None,
        })
        .unwrap()
    }

    fn full_mapping(model: &GnmModel) -> crate::DenseCorrespondenceSet {
        let mut mps: Vec<usize> = topology::NOSE
            .iter()
            .copied()
            .chain(topology::FACE_OVAL.iter().copied())
            .chain(topology::LIPS.iter().copied())
            .chain(topology::EYE_RIGHT.iter().copied())
            .chain(topology::EYE_LEFT.iter().copied())
            .chain(topology::BROW_RIGHT.iter().copied())
            .chain(topology::BROW_LEFT.iter().copied())
            .chain([
                topology::IRIS_CENTER_RIGHT,
                topology::IRIS_CENTER_LEFT,
                100,
                200,
            ])
            .collect();
        mps.sort_unstable();
        mps.dedup();
        crate::DenseCorrespondenceSet::new(
            mapping_version(),
            mps.iter().map(|mp| row(*mp)).collect(),
            model,
        )
        .unwrap()
    }

    fn neutral_calibration(
        model: &GnmModel,
        mapping: &crate::DenseCorrespondenceSet,
        scales: NeutralNormalizationScales,
    ) -> Result<GnmIdentityCalibration, GnmIdentityCalibrationError> {
        let mut surface = GnmSparseVertices::with_len(mapping.len());
        mapping
            .evaluate_surface(
                model,
                &model.neutral_identity(),
                &model.neutral_expression(),
                &GnmJointState::neutral(model.joint_count()),
                &mut surface,
            )
            .unwrap();
        let fixed = FixedGnmIdentity::new(model.neutral_identity(), model).unwrap();
        GnmIdentityCalibration::new(
            model,
            mapping_version(),
            fixed,
            model.neutral_expression(),
            surface.values().to_vec(),
            scales,
            IdentityFitDiagnostics {
                accepted_samples: 8,
                rejected_samples: 0,
                reprojection_rms: 0.01,
                active_identity_dimension: 8,
                condition_number: Some(12.0),
                pose_diversity: NeutralPoseDiversity {
                    yaw_span_radians: 0.2,
                    pitch_span_radians: 0.1,
                    near_duplicate_fraction: 0.1,
                },
            },
        )
    }

    fn default_scales() -> NeutralNormalizationScales {
        NeutralNormalizationScales {
            inter_ocular: Some(2.0),
            mouth_width: None,
            eye_aperture: None,
        }
    }

    fn expression_with_channel(channel: usize, value: f32) -> GnmExpressionState {
        let mut values = vec![0.0; GNM_HEAD_V3_EXPRESSION_DIM];
        values[channel] = value;
        GnmExpressionState::new(values, GNM_HEAD_V3_EXPRESSION_DIM).unwrap()
    }

    #[test]
    fn neutral_state_gives_zero_normalized_delta() {
        let model = eyelid_motion_model();
        let mapping = full_mapping(&model);
        let groups = crate::DenseRegionGroups::from_set(&mapping).unwrap();
        let calibration = neutral_calibration(&model, &mapping, default_scales()).unwrap();

        let features = compute_eye_aperture_features(
            &model,
            &model.neutral_identity(),
            &model.neutral_expression(),
            &GnmJointState::neutral(model.joint_count()),
            &mapping,
            &groups,
            &calibration,
        )
        .unwrap();

        assert_eq!(features.right.side, AnatomicalSide::Right);
        assert_eq!(features.left.side, AnatomicalSide::Left);
        assert!(features.right.normalized_delta.abs() < 1.0e-6);
        assert!(features.left.normalized_delta.abs() < 1.0e-6);
        assert!(features.right.current_aperture > 0.0);
        assert!((features.left.current_aperture - features.right.current_aperture).abs() < 1.0e-6);
    }

    #[test]
    fn closure_lowers_and_wide_raises_both_sides() {
        let model = eyelid_motion_model();
        let mapping = full_mapping(&model);
        let groups = crate::DenseRegionGroups::from_set(&mapping).unwrap();
        let calibration = neutral_calibration(&model, &mapping, default_scales()).unwrap();
        let joints = GnmJointState::neutral(model.joint_count());

        let closure = compute_eye_aperture_features(
            &model,
            &model.neutral_identity(),
            &expression_with_channel(0, 1.0),
            &joints,
            &mapping,
            &groups,
            &calibration,
        )
        .unwrap();
        assert!((closure.right.normalized_delta + 0.2).abs() < 1.0e-5);
        assert!((closure.left.normalized_delta + 0.2).abs() < 1.0e-5);

        let wide = compute_eye_aperture_features(
            &model,
            &model.neutral_identity(),
            &expression_with_channel(1, 1.0),
            &joints,
            &mapping,
            &groups,
            &calibration,
        )
        .unwrap();
        assert!((wide.right.normalized_delta - 0.3).abs() < 1.0e-5);
        assert!((wide.left.normalized_delta - 0.3).abs() < 1.0e-5);
    }

    #[test]
    fn rigid_head_transform_alone_does_not_change_features() {
        let model = eyelid_motion_model();
        let mapping = full_mapping(&model);
        let groups = crate::DenseRegionGroups::from_set(&mapping).unwrap();
        let calibration = neutral_calibration(&model, &mapping, default_scales()).unwrap();
        let expression = expression_with_channel(1, 0.5);

        let rest = compute_eye_aperture_features(
            &model,
            &model.neutral_identity(),
            &expression,
            &GnmJointState::neutral(model.joint_count()),
            &mapping,
            &groups,
            &calibration,
        )
        .unwrap();

        // Yaw the whole head and translate it; pairwise distances are invariant.
        let posed = GnmJointState::new(vec![[0.0, 0.35, 0.0]], [10.0, -4.0, 2.5], 1).unwrap();
        let moved = compute_eye_aperture_features(
            &model,
            &model.neutral_identity(),
            &expression,
            &posed,
            &mapping,
            &groups,
            &calibration,
        )
        .unwrap();

        for (a, b) in [(rest.right, moved.right), (rest.left, moved.left)] {
            assert!((a.current_aperture - b.current_aperture).abs() < 1.0e-4);
            assert!((a.normalized_delta - b.normalized_delta).abs() < 1.0e-4);
        }
    }

    #[test]
    fn sides_are_anatomical_not_mirrored() {
        let model = eyelid_motion_model();
        let mapping = full_mapping(&model);
        let groups = crate::DenseRegionGroups::from_set(&mapping).unwrap();

        // The "right" ring must be keyed on the subject-right MediaPipe
        // topology (33/133 corners), independent of any coordinate signs or
        // preview mirroring.
        assert_eq!(groups.eyes().right().outer_corner().row.mediapipe_index, 33);
        assert_eq!(groups.eyes().left().outer_corner().row.mediapipe_index, 263);

        let calibration = neutral_calibration(&model, &mapping, default_scales()).unwrap();
        // Close only the anatomical right eye via its own apex vertex.
        let mut values = vec![0.0; GNM_HEAD_V3_EXPRESSION_DIM];
        values[0] = 1.0;
        let features = compute_eye_aperture_features(
            &model,
            &model.neutral_identity(),
            &GnmExpressionState::new(values, GNM_HEAD_V3_EXPRESSION_DIM).unwrap(),
            &GnmJointState::neutral(model.joint_count()),
            &mapping,
            &groups,
            &calibration,
        )
        .unwrap();
        // Both apexes share channel 0 in this fixture; sides stay labeled
        // anatomically regardless.
        assert_eq!(features.right.side, AnatomicalSide::Right);
        assert_eq!(features.left.side, AnatomicalSide::Left);
        assert!(
            features
                .right
                .current_aperture
                .max(features.left.current_aperture)
                .is_finite()
        );
    }

    #[test]
    fn missing_or_degenerate_scale_fails_closed() {
        let model = eyelid_motion_model();
        let mapping = full_mapping(&model);
        let groups = crate::DenseRegionGroups::from_set(&mapping).unwrap();

        let missing =
            neutral_calibration(&model, &mapping, NeutralNormalizationScales::default()).unwrap();
        assert!(matches!(
            compute_eye_aperture_features(
                &model,
                &model.neutral_identity(),
                &model.neutral_expression(),
                &GnmJointState::neutral(model.joint_count()),
                &mapping,
                &groups,
                &missing,
            ),
            Err(GnmAuxGeometryError::MissingNormalizationScale {
                field: "inter_ocular"
            })
        ));

        let degenerate = neutral_calibration(
            &model,
            &mapping,
            NeutralNormalizationScales {
                inter_ocular: Some(-1.0),
                ..NeutralNormalizationScales::default()
            },
        );
        assert!(degenerate.is_err(), "calibration itself rejects bad scales");
    }

    #[test]
    fn calibration_surface_length_mismatch_is_typed() {
        let model = eyelid_motion_model();
        let mapping = full_mapping(&model);
        let groups = crate::DenseRegionGroups::from_set(&mapping).unwrap();
        let truncated = GnmIdentityCalibration::new(
            &model,
            mapping_version(),
            FixedGnmIdentity::new(model.neutral_identity(), &model).unwrap(),
            model.neutral_expression(),
            vec![[0.0; 3]; 4],
            default_scales(),
            IdentityFitDiagnostics {
                accepted_samples: 1,
                rejected_samples: 0,
                reprojection_rms: 0.0,
                active_identity_dimension: 1,
                condition_number: None,
                pose_diversity: NeutralPoseDiversity {
                    yaw_span_radians: 0.1,
                    pitch_span_radians: 0.0,
                    near_duplicate_fraction: 0.0,
                },
            },
        )
        .unwrap();

        assert!(matches!(
            compute_eye_aperture_features(
                &model,
                &model.neutral_identity(),
                &model.neutral_expression(),
                &GnmJointState::neutral(model.joint_count()),
                &mapping,
                &groups,
                &truncated,
            ),
            Err(GnmAuxGeometryError::CalibrationSurfaceLengthMismatch { .. })
        ));
    }

    // -- Jaw/mouth auxiliary features (Issue #55.2) fixtures ------------------

    const MOUTH_SCALE: f32 = 6.0;

    /// Model whose template places the jaw/mouth semantic rows at controlled
    /// coordinates and whose expression channels 10..16 drive single semantics:
    /// 10 jaw open, 11 jaw forward, 12 smile (corner lift), 13 frown,
    /// 14 pucker (corner narrowing), 15 stretch (corner widening),
    /// 16 jaw lateral toward the subject's left.
    fn mouth_motion_model() -> GnmModel {
        let vertex_count = MEDIAPIPE_FACE_LANDMARK_COUNT;
        let identity_dim = GNM_HEAD_V3_IDENTITY_DIM;
        let expression_dim = GNM_HEAD_V3_EXPRESSION_DIM;
        let mut vertices = vec![0.0f32; vertex_count * 3];
        let place = |vertices: &mut Vec<f32>, index: usize, x: f32, y: f32| {
            vertices[index * 3..index * 3 + 3].copy_from_slice(&[x, y, 0.0]);
        };
        place(&mut vertices, MOUTH_UPPER_LIP_CENTER_MP, 0.0, 2.0);
        place(&mut vertices, MOUTH_LOWER_LIP_CENTER_MP, 0.0, 0.0);
        place(&mut vertices, MOUTH_CORNER_RIGHT_MP, -3.0, 1.0);
        place(&mut vertices, MOUTH_CORNER_LEFT_MP, 3.0, 1.0);
        place(&mut vertices, JAW_CHIN_MP, 0.0, -4.0);
        place(&mut vertices, JAW_NOSE_TIP_MP, 0.0, 3.0);
        place(&mut vertices, JAW_ANCHOR_RIGHT_MP, -5.0, 3.0);
        place(&mut vertices, JAW_ANCHOR_LEFT_MP, 5.0, 3.0);

        let mut expression_basis = vec![0.0f32; expression_dim * vertex_count * 3];
        let mut drive = |channel: usize, index: usize, dx: f32, dy: f32| {
            expression_basis[(channel * vertex_count + index) * 3] = dx;
            expression_basis[(channel * vertex_count + index) * 3 + 1] = dy;
        };
        // 10: jaw open — lower lip and chin move down.
        drive(10, MOUTH_LOWER_LIP_CENTER_MP, 0.0, -0.5);
        drive(10, JAW_CHIN_MP, 0.0, -0.5);
        // 11: jaw forward — chin moves up toward the nose tip.
        drive(11, JAW_CHIN_MP, 0.0, 0.5);
        // 12: smile — corners lift.
        drive(12, MOUTH_CORNER_RIGHT_MP, 0.0, 0.6);
        drive(12, MOUTH_CORNER_LEFT_MP, 0.0, 0.6);
        // 13: frown — corners drop.
        drive(13, MOUTH_CORNER_RIGHT_MP, 0.0, -0.6);
        drive(13, MOUTH_CORNER_LEFT_MP, 0.0, -0.6);
        // 14: pucker — corners pull inward.
        drive(14, MOUTH_CORNER_RIGHT_MP, 1.0, 0.0);
        drive(14, MOUTH_CORNER_LEFT_MP, -1.0, 0.0);
        // 15: stretch — corners push outward.
        drive(15, MOUTH_CORNER_RIGHT_MP, -1.0, 0.0);
        drive(15, MOUTH_CORNER_LEFT_MP, 1.0, 0.0);
        // 16: jaw lateral — chin shifts toward the subject's left (+x here).
        drive(16, JAW_CHIN_MP, 0.5, 0.0);

        GnmModel::from_data(GnmModelData {
            version: GNM_HEAD_V3_VERSION,
            variant: GnmVariant::Head,
            template_vertices: DenseArray::new("vertices", vec![vertex_count, 3], vertices)
                .unwrap(),
            template_joints: DenseArray::new("joints", vec![1, 3], vec![0.0; 3]).unwrap(),
            vertex_identity_basis: DenseArray::new(
                "identity",
                vec![identity_dim, vertex_count, 3],
                vec![0.0; identity_dim * vertex_count * 3],
            )
            .unwrap(),
            joint_identity_basis: DenseArray::new(
                "joint_identity",
                vec![identity_dim, 1, 3],
                vec![0.0; identity_dim * 3],
            )
            .unwrap(),
            expression_basis: DenseArray::new(
                "expression",
                vec![expression_dim, vertex_count, 3],
                expression_basis,
            )
            .unwrap(),
            joint_parent_indices: vec![-1],
            skinning_weights: DenseArray::new(
                "weights",
                vec![1, vertex_count],
                vec![1.0; vertex_count],
            )
            .unwrap(),
            pose_correctives_regressor: None,
        })
        .unwrap()
    }

    fn mouth_row(mp: usize) -> crate::MediaPipeGnmDenseCorrespondence {
        crate::MediaPipeGnmDenseCorrespondence {
            mediapipe_index: mp,
            target: crate::GnmSurfacePointRef::Vertex { vertex_index: mp },
            region: FaceRegion::Other,
            anatomical_side: AnatomicalSide::Midline,
            base_weight: 1.0,
            provenance: CorrespondenceProvenance::RepositoryValidated,
            reliability: CorrespondenceReliability::High,
        }
    }

    fn mouth_mapping(model: &GnmModel, include_lips: bool) -> crate::DenseCorrespondenceSet {
        let mut keys = vec![
            JAW_CHIN_MP,
            JAW_NOSE_TIP_MP,
            JAW_ANCHOR_RIGHT_MP,
            JAW_ANCHOR_LEFT_MP,
        ];
        if include_lips {
            keys.extend([
                MOUTH_UPPER_LIP_CENTER_MP,
                MOUTH_LOWER_LIP_CENTER_MP,
                MOUTH_CORNER_RIGHT_MP,
                MOUTH_CORNER_LEFT_MP,
            ]);
        }
        // Filler rows to satisfy the density gate while avoiding key slots.
        let filler = (60..160).filter(|mp| !keys.contains(mp));
        let mut mps: Vec<usize> = keys.clone().into_iter().chain(filler).collect();
        mps.sort_unstable();
        mps.dedup();
        crate::DenseCorrespondenceSet::new(
            mapping_version(),
            mps.iter().map(|mp| mouth_row(*mp)).collect(),
            model,
        )
        .unwrap()
    }

    fn mouth_calibration(
        model: &GnmModel,
        mapping: &crate::DenseCorrespondenceSet,
    ) -> GnmIdentityCalibration {
        neutral_calibration(
            model,
            mapping,
            NeutralNormalizationScales {
                inter_ocular: Some(2.0),
                mouth_width: Some(MOUTH_SCALE),
                eye_aperture: None,
            },
        )
        .unwrap()
    }

    fn mouth_expression(channel: usize) -> GnmExpressionState {
        let mut values = vec![0.0; GNM_HEAD_V3_EXPRESSION_DIM];
        values[channel] = 1.0;
        GnmExpressionState::new(values, GNM_HEAD_V3_EXPRESSION_DIM).unwrap()
    }

    fn mouth_features(
        model: &GnmModel,
        mapping: &crate::DenseCorrespondenceSet,
        calibration: &GnmIdentityCalibration,
        channel: usize,
    ) -> MouthAuxFeatures {
        compute_mouth_aux_features(
            model,
            &model.neutral_identity(),
            &mouth_expression(channel),
            &GnmJointState::neutral(model.joint_count()),
            mapping,
            calibration,
        )
        .unwrap()
    }

    #[test]
    fn mouth_neutral_state_gives_all_zero_deltas() {
        let model = mouth_motion_model();
        let mapping = mouth_mapping(&model, true);
        let calibration = mouth_calibration(&model, &mapping);
        let features = mouth_features(&model, &mapping, &calibration, 0);
        for (name, value) in [
            ("jaw_open", features.jaw_open),
            ("jaw_forward", features.jaw_forward),
            ("jaw_lateral", features.jaw_lateral),
            ("width_delta", features.width_delta),
            ("corner_lift", features.corner_lift),
        ] {
            let value = value.expect(name);
            assert!(value.abs() < 1.0e-6, "{name} = {value}");
        }
    }

    #[test]
    fn jaw_open_and_forward_have_fixed_signs() {
        let model = mouth_motion_model();
        let mapping = mouth_mapping(&model, true);
        let calibration = mouth_calibration(&model, &mapping);

        let open = mouth_features(&model, &mapping, &calibration, 10);
        assert!((open.jaw_open.unwrap() - 0.5 / MOUTH_SCALE).abs() < 1.0e-5);
        assert!((open.jaw_forward.unwrap() - 0.5 / MOUTH_SCALE).abs() < 1.0e-5);

        let forward = mouth_features(&model, &mapping, &calibration, 11);
        assert!((forward.jaw_forward.unwrap() + 0.5 / MOUTH_SCALE).abs() < 1.0e-5);
        assert!(forward.jaw_open.unwrap().abs() < 1.0e-6);
    }

    #[test]
    fn smile_frown_pucker_and_stretch_have_fixed_signs() {
        let model = mouth_motion_model();
        let mapping = mouth_mapping(&model, true);
        let calibration = mouth_calibration(&model, &mapping);

        let smile = mouth_features(&model, &mapping, &calibration, 12);
        assert!(smile.corner_lift.unwrap() > 0.0, "smile lifts corners");
        assert!(smile.width_delta.unwrap().abs() < 1.0e-6);

        let frown = mouth_features(&model, &mapping, &calibration, 13);
        assert!(frown.corner_lift.unwrap() < 0.0, "frown lowers corners");

        let pucker = mouth_features(&model, &mapping, &calibration, 14);
        assert!((pucker.width_delta.unwrap() + 2.0 / MOUTH_SCALE).abs() < 1.0e-5);

        let stretch = mouth_features(&model, &mapping, &calibration, 15);
        assert!((stretch.width_delta.unwrap() - 2.0 / MOUTH_SCALE).abs() < 1.0e-5);
    }

    #[test]
    fn jaw_lateral_sign_matches_subject_left_shift() {
        let model = mouth_motion_model();
        let mapping = mouth_mapping(&model, true);
        let calibration = mouth_calibration(&model, &mapping);

        let lateral = mouth_features(&model, &mapping, &calibration, 16);
        assert!(
            lateral.jaw_lateral.unwrap() > 0.0,
            "positive must mean shifted toward the subject's left"
        );
        assert!(lateral.jaw_open.unwrap().abs() < 1.0e-6);
    }

    #[test]
    fn rigid_head_transform_alone_does_not_change_mouth_features() {
        let model = mouth_motion_model();
        let mapping = mouth_mapping(&model, true);
        let calibration = mouth_calibration(&model, &mapping);

        let rest = mouth_features(&model, &mapping, &calibration, 12);
        let posed = compute_mouth_aux_features(
            &model,
            &model.neutral_identity(),
            &mouth_expression(12),
            &GnmJointState::new(vec![[0.0, 0.4, 0.0]], [7.0, -3.0, 1.5], 1).unwrap(),
            &mapping,
            &calibration,
        )
        .unwrap();
        for name in [
            "jaw_open",
            "jaw_forward",
            "jaw_lateral",
            "width_delta",
            "corner_lift",
        ] {
            let a = mouth_field(&rest, name);
            let b = mouth_field(&posed, name);
            assert!((a - b).abs() < 1.0e-4, "{name}: {a} vs {b}");
        }
    }

    fn mouth_field(features: &MouthAuxFeatures, name: &str) -> f32 {
        match name {
            "jaw_open" => features.jaw_open.unwrap(),
            "jaw_forward" => features.jaw_forward.unwrap(),
            "jaw_lateral" => features.jaw_lateral.unwrap(),
            "width_delta" => features.width_delta.unwrap(),
            "corner_lift" => features.corner_lift.unwrap(),
            _ => unreachable!(),
        }
    }

    #[test]
    fn missing_lip_rows_leave_semantics_unavailable_without_fabrication() {
        let model = mouth_motion_model();
        let mapping = mouth_mapping(&model, false);
        let calibration = mouth_calibration(&model, &mapping);
        let features = mouth_features(&model, &mapping, &calibration, 12);
        assert_eq!(features.jaw_open, None);
        assert_eq!(features.width_delta, None);
        assert_eq!(features.corner_lift, None);
        // Jaw rows are still present, so those semantics stay measurable.
        assert!(features.jaw_forward.is_some());
        assert!(features.jaw_lateral.is_some());
    }

    #[test]
    fn mouth_missing_or_degenerate_scale_fails_closed() {
        let model = mouth_motion_model();
        let mapping = mouth_mapping(&model, true);
        let no_scale = neutral_calibration(
            &model,
            &mapping,
            NeutralNormalizationScales {
                inter_ocular: Some(2.0),
                mouth_width: None,
                eye_aperture: None,
            },
        )
        .unwrap();
        assert!(matches!(
            compute_mouth_aux_features(
                &model,
                &model.neutral_identity(),
                &model.neutral_expression(),
                &GnmJointState::neutral(model.joint_count()),
                &mapping,
                &no_scale,
            ),
            Err(GnmAuxGeometryError::MissingNormalizationScale {
                field: "mouth_width"
            })
        ));
    }

    // -- Brow auxiliary features (Issue #55.3 / #88) fixtures -----------------

    /// Model whose template places brow rows directly above their upper-lid
    /// apexes (so neutral brow-to-apex distance is exactly 2.0) and whose
    /// expression channels drive single semantics:
    /// 20 inner brow up on both sides,
    /// 21 brow down on the anatomical right only,
    /// 22 outer brow up on the anatomical left only.
    fn brow_motion_model() -> GnmModel {
        let vertex_count = MEDIAPIPE_FACE_LANDMARK_COUNT;
        let identity_dim = GNM_HEAD_V3_IDENTITY_DIM;
        let expression_dim = GNM_HEAD_V3_EXPRESSION_DIM;
        let mut vertices = vec![0.0f32; vertex_count * 3];
        let place = |vertices: &mut Vec<f32>, index: usize, x: f32, y: f32| {
            vertices[index * 3..index * 3 + 3].copy_from_slice(&[x, y, 0.0]);
        };
        place(&mut vertices, LID_APEX_RIGHT_MP, -1.0, 0.0);
        place(&mut vertices, LID_APEX_LEFT_MP, 1.0, 0.0);
        place(&mut vertices, BROW_INNER_RIGHT_MP, -1.0, 2.0);
        place(&mut vertices, BROW_MID_RIGHT_MP, -1.0, 2.0);
        place(&mut vertices, BROW_OUTER_RIGHT_MP, -1.0, 2.0);
        place(&mut vertices, BROW_INNER_LEFT_MP, 1.0, 2.0);
        place(&mut vertices, BROW_MID_LEFT_MP, 1.0, 2.0);
        place(&mut vertices, BROW_OUTER_LEFT_MP, 1.0, 2.0);

        let mut expression_basis = vec![0.0f32; expression_dim * vertex_count * 3];
        let mut drive = |channel: usize, index: usize, dy: f32| {
            expression_basis[(channel * vertex_count + index) * 3 + 1] = dy;
        };
        // 20: inner brow up — both inner points move away from their apexes.
        drive(20, BROW_INNER_RIGHT_MP, 0.5);
        drive(20, BROW_INNER_LEFT_MP, 0.5);
        // 21: brow down — only the anatomical right mid point approaches its apex.
        drive(21, BROW_MID_RIGHT_MP, -0.5);
        // 22: outer brow up — only the anatomical left outer point recedes.
        drive(22, BROW_OUTER_LEFT_MP, 0.5);

        GnmModel::from_data(GnmModelData {
            version: GNM_HEAD_V3_VERSION,
            variant: GnmVariant::Head,
            template_vertices: DenseArray::new("vertices", vec![vertex_count, 3], vertices)
                .unwrap(),
            template_joints: DenseArray::new("joints", vec![1, 3], vec![0.0; 3]).unwrap(),
            vertex_identity_basis: DenseArray::new(
                "identity",
                vec![identity_dim, vertex_count, 3],
                vec![0.0; identity_dim * vertex_count * 3],
            )
            .unwrap(),
            joint_identity_basis: DenseArray::new(
                "joint_identity",
                vec![identity_dim, 1, 3],
                vec![0.0; identity_dim * 3],
            )
            .unwrap(),
            expression_basis: DenseArray::new(
                "expression",
                vec![expression_dim, vertex_count, 3],
                expression_basis,
            )
            .unwrap(),
            joint_parent_indices: vec![-1],
            skinning_weights: DenseArray::new(
                "weights",
                vec![1, vertex_count],
                vec![1.0; vertex_count],
            )
            .unwrap(),
            pose_correctives_regressor: None,
        })
        .unwrap()
    }

    const BROW_SCALE: f32 = 2.0;

    fn brow_semantic_keys() -> Vec<usize> {
        vec![
            BROW_INNER_RIGHT_MP,
            BROW_MID_RIGHT_MP,
            BROW_OUTER_RIGHT_MP,
            BROW_INNER_LEFT_MP,
            BROW_MID_LEFT_MP,
            BROW_OUTER_LEFT_MP,
            LID_APEX_RIGHT_MP,
            LID_APEX_LEFT_MP,
        ]
    }

    fn brow_mapping(model: &GnmModel, include_brows: bool) -> crate::DenseCorrespondenceSet {
        let keys = brow_semantic_keys();
        let mut mps: Vec<usize> = if include_brows {
            keys.clone()
        } else {
            vec![LID_APEX_RIGHT_MP, LID_APEX_LEFT_MP]
        }
        .into_iter()
        .chain((40..160).filter(|mp| !keys.contains(mp)))
        .collect();
        mps.sort_unstable();
        mps.dedup();
        crate::DenseCorrespondenceSet::new(
            mapping_version(),
            mps.iter().map(|mp| mouth_row(*mp)).collect(),
            model,
        )
        .unwrap()
    }

    fn brow_calibration(
        model: &GnmModel,
        mapping: &crate::DenseCorrespondenceSet,
        inter_ocular: Option<f32>,
    ) -> GnmIdentityCalibration {
        neutral_calibration(
            model,
            mapping,
            NeutralNormalizationScales {
                inter_ocular,
                mouth_width: None,
                eye_aperture: None,
            },
        )
        .unwrap()
    }

    fn brow_features(
        model: &GnmModel,
        mapping: &crate::DenseCorrespondenceSet,
        calibration: &GnmIdentityCalibration,
        channel: usize,
    ) -> BrowAuxFeatures {
        compute_brow_aux_features(
            model,
            &model.neutral_identity(),
            &mouth_expression(channel),
            &GnmJointState::neutral(model.joint_count()),
            mapping,
            calibration,
        )
        .unwrap()
    }

    #[test]
    fn brow_neutral_state_gives_all_zero_deltas() {
        let model = brow_motion_model();
        let mapping = brow_mapping(&model, true);
        let calibration = brow_calibration(&model, &mapping, Some(BROW_SCALE));
        let features = brow_features(&model, &mapping, &calibration, 0);
        assert_eq!(features.right.side, AnatomicalSide::Right);
        assert_eq!(features.left.side, AnatomicalSide::Left);
        for side in [&features.right, &features.left] {
            assert!(side.inner_rise.unwrap().abs() < 1.0e-6);
            assert!(side.brow_lower.unwrap().abs() < 1.0e-6);
            assert!(side.outer_rise.unwrap().abs() < 1.0e-6);
        }
    }

    #[test]
    fn inner_brow_up_raises_both_sides_with_fixed_sign() {
        let model = brow_motion_model();
        let mapping = brow_mapping(&model, true);
        let calibration = brow_calibration(&model, &mapping, Some(BROW_SCALE));
        let features = brow_features(&model, &mapping, &calibration, 20);
        for side in [&features.right, &features.left] {
            assert!((side.inner_rise.unwrap() - 0.5 / BROW_SCALE).abs() < 1.0e-5);
            assert!(side.brow_lower.unwrap().abs() < 1.0e-6);
            assert!(side.outer_rise.unwrap().abs() < 1.0e-6);
        }
    }

    #[test]
    fn one_side_down_and_one_side_outer_up_stay_on_their_anatomical_side() {
        let model = brow_motion_model();
        let mapping = brow_mapping(&model, true);
        let calibration = brow_calibration(&model, &mapping, Some(BROW_SCALE));

        // Brow down drives only the subject's anatomical right mid brow.
        let down = brow_features(&model, &mapping, &calibration, 21);
        assert!((down.right.brow_lower.unwrap() - 0.5 / BROW_SCALE).abs() < 1.0e-5);
        assert!(down.left.brow_lower.unwrap().abs() < 1.0e-6);
        assert!(down.right.inner_rise.unwrap().abs() < 1.0e-6);
        assert!(down.left.inner_rise.unwrap().abs() < 1.0e-6);

        // Outer brow up drives only the subject's anatomical left outer brow.
        let up = brow_features(&model, &mapping, &calibration, 22);
        assert!((up.left.outer_rise.unwrap() - 0.5 / BROW_SCALE).abs() < 1.0e-5);
        assert!(up.right.outer_rise.unwrap().abs() < 1.0e-6);
        assert!(up.left.brow_lower.unwrap().abs() < 1.0e-6);
    }

    #[test]
    fn rigid_head_transform_alone_does_not_change_brow_features() {
        let model = brow_motion_model();
        let mapping = brow_mapping(&model, true);
        let calibration = brow_calibration(&model, &mapping, Some(BROW_SCALE));

        let rest = brow_features(&model, &mapping, &calibration, 20);
        let posed = compute_brow_aux_features(
            &model,
            &model.neutral_identity(),
            &mouth_expression(20),
            &GnmJointState::new(vec![[0.0, 0.45, 0.0]], [8.0, -5.0, 2.0], 1).unwrap(),
            &mapping,
            &calibration,
        )
        .unwrap();
        for (rest_side, posed_side) in [(&rest.right, &posed.right), (&rest.left, &posed.left)] {
            assert!(
                (rest_side.inner_rise.unwrap() - posed_side.inner_rise.unwrap()).abs() < 1.0e-4
            );
            assert!(
                (rest_side.brow_lower.unwrap() - posed_side.brow_lower.unwrap()).abs() < 1.0e-4
            );
            assert!(
                (rest_side.outer_rise.unwrap() - posed_side.outer_rise.unwrap()).abs() < 1.0e-4
            );
        }
    }

    #[test]
    fn missing_brow_rows_leave_features_unavailable_without_fabrication() {
        let model = brow_motion_model();
        let mapping = brow_mapping(&model, false);
        let calibration = brow_calibration(&model, &mapping, Some(BROW_SCALE));
        let features = brow_features(&model, &mapping, &calibration, 20);
        for side in [&features.right, &features.left] {
            assert_eq!(side.inner_rise, None);
            assert_eq!(side.brow_lower, None);
            assert_eq!(side.outer_rise, None);
        }
    }

    #[test]
    fn brow_missing_scale_fails_closed() {
        let model = brow_motion_model();
        let mapping = brow_mapping(&model, true);
        let no_scale = brow_calibration(&model, &mapping, None);
        assert!(matches!(
            compute_brow_aux_features(
                &model,
                &model.neutral_identity(),
                &model.neutral_expression(),
                &GnmJointState::neutral(model.joint_count()),
                &mapping,
                &no_scale,
            ),
            Err(GnmAuxGeometryError::MissingNormalizationScale {
                field: "inter_ocular"
            })
        ));
    }

    // --- GnmFacialFeatures snapshot tests (Issue #67.1 / #96) ------------

    fn snapshot_scales() -> NeutralNormalizationScales {
        NeutralNormalizationScales {
            inter_ocular: Some(2.0),
            mouth_width: Some(3.0),
            eye_aperture: None,
        }
    }

    #[test]
    fn snapshot_neutral_state_validates_with_zero_deltas() {
        let model = eyelid_motion_model();
        let mapping = full_mapping(&model);
        let groups = crate::DenseRegionGroups::from_set(&mapping).unwrap();
        let calibration = neutral_calibration(&model, &mapping, snapshot_scales()).unwrap();

        let snapshot = compute_gnm_facial_features(
            &model,
            &model.neutral_identity(),
            &model.neutral_expression(),
            &GnmJointState::neutral(model.joint_count()),
            &mapping,
            &groups,
            &calibration,
        )
        .unwrap();

        snapshot.validate().unwrap();
        assert!(snapshot.eyes.right.normalized_delta.abs() < 1.0e-6);
        assert!(snapshot.eyes.left.normalized_delta.abs() < 1.0e-6);
        // The fixture mapping carries iris rows, so iris features exist.
        let right_iris = snapshot.irises.right.expect("right iris present");
        let left_iris = snapshot.irises.left.expect("left iris present");
        assert_eq!(right_iris.side, AnatomicalSide::Right);
        assert_eq!(left_iris.side, AnatomicalSide::Left);
        for value in [
            right_iris.vertical_delta,
            right_iris.horizontal_delta,
            left_iris.vertical_delta,
            left_iris.horizontal_delta,
        ]
        .into_iter()
        .flatten()
        {
            assert!(value.abs() < 1.0e-5, "neutral iris delta {value}");
        }
    }

    #[test]
    fn snapshot_is_invariant_under_rigid_joint_transform() {
        let model = eyelid_motion_model();
        let mapping = full_mapping(&model);
        let groups = crate::DenseRegionGroups::from_set(&mapping).unwrap();
        let calibration = neutral_calibration(&model, &mapping, snapshot_scales()).unwrap();
        let expression = expression_with_channel(0, 0.8); // non-neutral: closure
        let posed = GnmJointState::new(vec![[0.0, 0.35, 0.0]], [10.0, -4.0, 2.5], 1).unwrap();

        let compute = |joints: &GnmJointState| {
            compute_gnm_facial_features(
                &model,
                &model.neutral_identity(),
                &expression,
                joints,
                &mapping,
                &groups,
                &calibration,
            )
            .unwrap()
        };
        let reference = compute(&GnmJointState::neutral(model.joint_count()));
        let rigidly_moved = compute(&posed);

        // Rigid motion preserves pairwise distances up to f32 evaluation
        // noise; nothing may move beyond numerical noise.
        let close = |a: f32, b: f32| (a - b).abs() < 1.0e-3;
        for (reference_eye, moved_eye) in [
            (&reference.eyes.right, &rigidly_moved.eyes.right),
            (&reference.eyes.left, &rigidly_moved.eyes.left),
        ] {
            assert!(close(
                reference_eye.current_aperture,
                moved_eye.current_aperture
            ));
            assert!(close(
                reference_eye.normalized_delta,
                moved_eye.normalized_delta
            ));
        }
        let option_close = |a: Option<f32>, b: Option<f32>| match (a, b) {
            (Some(a), Some(b)) => close(a, b),
            (None, None) => true,
            _ => false,
        };
        for (reference_iris, moved_iris) in [
            (reference.irises.right, rigidly_moved.irises.right),
            (reference.irises.left, rigidly_moved.irises.left),
        ] {
            let (Some(r), Some(m)) = (reference_iris, moved_iris) else {
                panic!("iris features must be present for this mapping");
            };
            assert!(option_close(r.vertical_delta, m.vertical_delta));
            assert!(option_close(r.horizontal_delta, m.horizontal_delta));
        }
        for (field, r, m) in [
            (
                "jaw_open",
                reference.mouth_jaw.jaw_open,
                rigidly_moved.mouth_jaw.jaw_open,
            ),
            (
                "jaw_forward",
                reference.mouth_jaw.jaw_forward,
                rigidly_moved.mouth_jaw.jaw_forward,
            ),
            (
                "jaw_lateral",
                reference.mouth_jaw.jaw_lateral,
                rigidly_moved.mouth_jaw.jaw_lateral,
            ),
            (
                "width_delta",
                reference.mouth_jaw.width_delta,
                rigidly_moved.mouth_jaw.width_delta,
            ),
            (
                "corner_lift",
                reference.mouth_jaw.corner_lift,
                rigidly_moved.mouth_jaw.corner_lift,
            ),
        ] {
            assert!(option_close(r, m), "rigid drift in {field}");
        }
        for (reference_brow, moved_brow) in [
            (&reference.brows.right, &rigidly_moved.brows.right),
            (&reference.brows.left, &rigidly_moved.brows.left),
        ] {
            assert!(option_close(
                reference_brow.inner_rise,
                moved_brow.inner_rise
            ));
            assert!(option_close(
                reference_brow.brow_lower,
                moved_brow.brow_lower
            ));
            assert!(option_close(
                reference_brow.outer_rise,
                moved_brow.outer_rise
            ));
        }
    }

    #[test]
    fn snapshot_normalization_follows_calibration_scale_not_raw_geometry() {
        let model = eyelid_motion_model();
        let mapping = full_mapping(&model);
        let groups = crate::DenseRegionGroups::from_set(&mapping).unwrap();
        let narrow = neutral_calibration(&model, &mapping, snapshot_scales()).unwrap();
        let wide = neutral_calibration(
            &model,
            &mapping,
            NeutralNormalizationScales {
                inter_ocular: Some(4.0),
                mouth_width: Some(6.0),
                eye_aperture: None,
            },
        )
        .unwrap();
        let expression = expression_with_channel(0, 0.8);

        let args = (
            &model,
            &model.neutral_identity(),
            &expression,
            &GnmJointState::neutral(model.joint_count()),
            &mapping,
            &groups,
        );
        let with_narrow_scale =
            compute_gnm_facial_features(args.0, args.1, args.2, args.3, args.4, args.5, &narrow)
                .unwrap();
        let with_wide_scale =
            compute_gnm_facial_features(args.0, args.1, args.2, args.3, args.4, args.5, &wide)
                .unwrap();

        // Raw apertures are identical; normalized deltas scale inversely.
        assert!(
            (with_narrow_scale.eyes.right.current_aperture
                - with_wide_scale.eyes.right.current_aperture)
                .abs()
                < 1.0e-6
        );
        let ratio = with_narrow_scale.eyes.right.normalized_delta
            / with_wide_scale.eyes.right.normalized_delta;
        assert!((ratio - 2.0).abs() < 1.0e-4, "scale ratio {ratio}");
        assert!(with_narrow_scale.validate().is_ok());
        assert!(with_wide_scale.validate().is_ok());
    }

    #[test]
    fn mirror_symmetric_closure_moves_both_anatomical_sides_identically() {
        let model = eyelid_motion_model();
        let mapping = full_mapping(&model);
        let groups = crate::DenseRegionGroups::from_set(&mapping).unwrap();
        let calibration = neutral_calibration(&model, &mapping, snapshot_scales()).unwrap();
        let closure = expression_with_channel(0, 1.0);

        let snapshot = compute_gnm_facial_features(
            &model,
            &model.neutral_identity(),
            &closure,
            &GnmJointState::neutral(model.joint_count()),
            &mapping,
            &groups,
            &calibration,
        )
        .unwrap();

        // Sides are keyed anatomically and the fixture is symmetric.
        assert_eq!(snapshot.eyes.right.side, AnatomicalSide::Right);
        assert_eq!(snapshot.eyes.left.side, AnatomicalSide::Left);
        assert!(
            (snapshot.eyes.right.normalized_delta - snapshot.eyes.left.normalized_delta).abs()
                < 1.0e-5
        );
        // Closure narrows the aperture: negative delta.
        assert!(snapshot.eyes.right.normalized_delta < 0.0);
    }

    #[test]
    fn iris_features_are_unavailable_without_iris_rows() {
        let model = eyelid_motion_model();
        let full = full_mapping(&model);

        // Same topology minus both iris centers. Region-group construction
        // rejects such mappings, so this exercises the snapshot's internal
        // family function directly: absence is reported as `None`, never
        // fabricated.
        let rows: Vec<_> = full
            .rows()
            .iter()
            .filter(|row| {
                row.mediapipe_index != topology::IRIS_CENTER_RIGHT
                    && row.mediapipe_index != topology::IRIS_CENTER_LEFT
            })
            .cloned()
            .collect();
        let without_iris =
            crate::DenseCorrespondenceSet::new(mapping_version(), rows, &model).unwrap();

        let mut surface = crate::GnmSparseVertices::with_len(without_iris.len());
        without_iris
            .evaluate_surface(
                &model,
                &model.neutral_identity(),
                &model.neutral_expression(),
                &GnmJointState::neutral(model.joint_count()),
                &mut surface,
            )
            .unwrap();
        let calibration = neutral_calibration(&model, &without_iris, snapshot_scales()).unwrap();
        let neutral_surface = calibration.neutral_surface_reference();

        let irises = iris_from_parts(
            surface.values(),
            neutral_surface,
            &without_iris,
            snapshot_scales().inter_ocular.unwrap(),
        )
        .unwrap();
        assert!(irises.right.is_none());
        assert!(irises.left.is_none());
    }

    #[test]
    fn validate_rejects_non_finite_feature_values() {
        let model = eyelid_motion_model();
        let mapping = full_mapping(&model);
        let calibration = neutral_calibration(&model, &mapping, snapshot_scales()).unwrap();
        let _ = (&model, &mapping, &calibration);
        let mut feature = EyeApertureFeature {
            side: AnatomicalSide::Right,
            current_aperture: 1.0,
            neutral_aperture: 1.0,
            normalized_delta: 0.0,
        };
        feature.current_aperture = f32::NAN;
        let snapshot = GnmFacialFeatures {
            eyes: EyeAuxFeatures {
                right: feature,
                left: feature,
            },
            irises: IrisAuxFeatures::default(),
            mouth_jaw: MouthAuxFeatures::default(),
            brows: BrowAuxFeatures {
                right: BrowSideAuxFeatures {
                    side: AnatomicalSide::Right,
                    inner_rise: None,
                    brow_lower: None,
                    outer_rise: None,
                },
                left: BrowSideAuxFeatures {
                    side: AnatomicalSide::Left,
                    inner_rise: None,
                    brow_lower: None,
                    outer_rise: None,
                },
            },
        };
        assert!(matches!(
            snapshot.validate(),
            Err(GnmAuxGeometryError::NonFiniteFeature { .. })
        ));
    }
}
