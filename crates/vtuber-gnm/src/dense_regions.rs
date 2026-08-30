//! Region-specific typed views over the repository dense mapping
//! (Issues #77, #78, #79).
//!
//! The committed table is one text asset; this module partitions it into
//! independently validated semantic groups — central face and contour (#77),
//! eye / brow / iris (#78), and mouth / lip corners (#79). Partitioning is
//! fail-closed: every row must land in exactly one typed bucket, the table's
//! own region tag must agree with MediaPipe topology, and per-side populations
//! must be symmetric, so a regenerated table cannot drift semantically without
//! failing validation.
//!
//! Anatomical-side authority lives in geometry, never in naming conventions
//! or preview mirroring: the pinned GNM template places the subject's left at
//! +X (established during derivation), and the semantic order of every typed
//! sequence below is fixed and pinned by tests.

use crate::{
    DenseCorrespondenceSet, FaceRegion, GnmDenseError, MEDIAPIPE_FACE_LANDMARK_COUNT,
    MediaPipeGnmDenseCorrespondence,
};

/// MediaPipe canonical topology shared with the derivation example.
///
/// Membership drives region classification; the values must stay byte-stable
/// with what generated the committed table.
pub mod topology {
    /// MediaPipe canonical face-oval (jaw contour) ring.
    pub const FACE_OVAL: &[usize] = &[
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
        152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    ];
    /// MediaPipe upper outer lip arc, ordered subject-right corner to
    /// subject-left corner through the cupids-bow center.
    pub const UPPER_LIP_OUTER_ARC: &[usize] = &[61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291];
    /// MediaPipe lower outer lip arc, ordered subject-right corner to
    /// subject-left corner through the lower center.
    pub const LOWER_LIP_OUTER_ARC: &[usize] = &[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291];
    /// MediaPipe upper inner lip arc, ordered inner-right corner to inner-left
    /// corner.
    pub const UPPER_LIP_INNER_ARC: &[usize] = &[78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308];
    /// MediaPipe lower inner lip arc, ordered inner-right corner to inner-left
    /// corner.
    pub const LOWER_LIP_INNER_ARC: &[usize] = &[78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308];
    /// All MediaPipe lip landmarks (outer + inner rings), membership form.
    pub const LIPS: &[usize] = &[
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
    ];
    /// MediaPipe right eyelid ring, ordered outer corner over the upper lid to
    /// the inner corner, then back under the lower lid.
    pub const EYE_RIGHT: &[usize] = &[
        33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7,
    ];
    /// MediaPipe left eyelid ring, ordered outer corner over the upper lid to
    /// the inner corner, then back under the lower lid.
    pub const EYE_LEFT: &[usize] = &[
        263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249,
    ];
    /// MediaPipe right brow, ordered lower arc inner-to-outer then upper arc
    /// inner-to-outer.
    pub const BROW_RIGHT: &[usize] = &[70, 63, 105, 66, 107, 46, 53, 52, 65];
    /// MediaPipe left brow, mirrored ordering of [`BROW_RIGHT`].
    pub const BROW_LEFT: &[usize] = &[300, 293, 334, 296, 336, 276, 283, 282, 295];
    /// MediaPipe nose bridge, tip, and nostril region.
    pub const NOSE: &[usize] = &[
        1, 2, 4, 5, 6, 19, 94, 97, 98, 99, 164, 165, 167, 168, 195, 196, 197, 129, 358, 326, 327,
        64, 240, 49, 48, 279, 278,
    ];
    /// Iris landmark center for the subject's right eye.
    pub const IRIS_CENTER_RIGHT: usize = 468;
    /// Iris landmark center for the subject's left eye.
    pub const IRIS_CENTER_LEFT: usize = 473;

    /// Whether the index belongs to either eyelid ring.
    pub fn is_eyelid(index: usize) -> bool {
        EYE_RIGHT.contains(&index) || EYE_LEFT.contains(&index)
    }

    /// Whether the index belongs to either brow arc pair.
    pub fn is_brow(index: usize) -> bool {
        BROW_RIGHT.contains(&index) || BROW_LEFT.contains(&index)
    }
}

/// MediaPipe landmark groups deliberately excluded from the primary dense
/// mapping, recorded for deterministic retrieval (Issue #80 inventory).
#[derive(Clone, Copy, Debug)]
pub struct ExcludedMediaPipeGroup {
    /// Stable diagnostic label.
    pub label: &'static str,
    /// Excluded MediaPipe indices.
    pub indices: &'static [usize],
    /// Recorded exclusion rationale.
    pub reason: &'static str,
}

/// The exclusion inventory asserted against the committed table by tests.
pub const EXCLUDED_MEDIAPIPE_GROUPS: &[ExcludedMediaPipeGroup] = &[ExcludedMediaPipeGroup {
    label: "iris-rings",
    indices: &[469, 470, 471, 472, 474, 475, 476, 477],
    reason: "gaze-dependent eyeball surface points; they belong to gaze \
                 estimation, not surface correspondence, and would duplicate \
                 targets near each iris center",
}];

/// One correspondence row retained together with its stable index in the
/// parent [`DenseCorrespondenceSet`] (so evaluated surface positions stay
/// addressable).
#[derive(Clone, Copy, Debug)]
pub struct IndexedRow {
    /// Row position in the parent set.
    pub index: usize,
    /// The correspondence row.
    pub row: MediaPipeGnmDenseCorrespondence,
}

impl IndexedRow {
    fn new(index: usize, row: &MediaPipeGnmDenseCorrespondence) -> Self {
        Self { index, row: *row }
    }
}

/// Central rigid-face rows: nose bridge, tip, and nostril region (#77).
#[derive(Clone, Debug)]
pub struct CentralFaceRows(Vec<IndexedRow>);

impl CentralFaceRows {
    /// Rows in stable table order.
    pub fn rows(&self) -> &[IndexedRow] {
        &self.0
    }
}

/// Jaw-contour rows: the face-oval silhouette ring (#77).
///
/// These carry reduced static weight by policy; see the weight comparison
/// pinned by tests.
#[derive(Clone, Debug)]
pub struct ContourRows(Vec<IndexedRow>);

impl ContourRows {
    /// Rows in ring order starting at the chin and running subject-right
    /// first (the committed table order).
    pub fn rows(&self) -> &[IndexedRow] {
        &self.0
    }
}

/// Validated interior rows outside every named semantic region (#77).
#[derive(Clone, Debug)]
pub struct OtherValidatedRows(Vec<IndexedRow>);

impl OtherValidatedRows {
    /// Rows in stable table order.
    pub fn rows(&self) -> &[IndexedRow] {
        &self.0
    }
}

/// One eyelid ring with fixed traversal semantics (#78).
#[derive(Clone, Debug)]
pub struct EyelidRing {
    outer_corner: IndexedRow,
    inner_corner: IndexedRow,
    /// Canonical ring order: outer corner, over the upper lid, inner corner,
    /// back under the lower lid, to the start.
    ring: Vec<IndexedRow>,
}

impl EyelidRing {
    /// The temporal corner of this eye.
    pub fn outer_corner(&self) -> &IndexedRow {
        &self.outer_corner
    }

    /// The nasal corner of this eye.
    pub fn inner_corner(&self) -> &IndexedRow {
        &self.inner_corner
    }

    /// Upper-lid arc ordered outer corner to inner corner (apex in the
    /// middle).
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    // Invariant: the ring is built to contain the inner corner exactly once.
    #[allow(clippy::expect_used)]
    pub fn upper_arc(&self) -> &[IndexedRow] {
        let end = self
            .ring
            .iter()
            .position(|entry| entry.row.mediapipe_index == self.inner_corner.row.mediapipe_index);
        &self.ring[..end.expect("ring always contains its inner corner") + 1]
    }

    /// Lower-lid arc ordered inner corner toward the outer corner (the final
    /// segment of the canonical ring).
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    // Invariant: the ring is built to contain the inner corner exactly once.
    #[allow(clippy::expect_used)]
    pub fn lower_arc(&self) -> &[IndexedRow] {
        let start = self
            .ring
            .iter()
            .position(|entry| entry.row.mediapipe_index == self.inner_corner.row.mediapipe_index);
        let start = start.expect("ring always contains its inner corner");
        &self.ring[start..]
    }

    /// The whole ring in canonical order.
    pub fn rows(&self) -> &[IndexedRow] {
        &self.ring
    }
}

/// Eye-region typed rows: left and right eyelid rings (#78).
#[derive(Clone, Debug)]
pub struct EyeRegionRows {
    right: EyelidRing,
    left: EyelidRing,
}

impl EyeRegionRows {
    /// The subject's anatomical right eyelid ring.
    pub fn right(&self) -> &EyelidRing {
        &self.right
    }

    /// The subject's anatomical left eyelid ring.
    pub fn left(&self) -> &EyelidRing {
        &self.left
    }
}

/// Brow typed rows, per anatomical side, in fixed inner-to-outer order (#78).
#[derive(Clone, Debug)]
pub struct BrowRows {
    right: Vec<IndexedRow>,
    left: Vec<IndexedRow>,
}

impl BrowRows {
    /// Right brow rows in canonical order (lower arc inner-to-outer, then
    /// upper arc inner-to-outer).
    pub fn right(&self) -> &[IndexedRow] {
        &self.right
    }

    /// Left brow rows, mirrored ordering of [`BrowRows::right`].
    pub fn left(&self) -> &[IndexedRow] {
        &self.left
    }
}

/// Iris-center typed rows: the two barycentric iris centers (#78).
#[derive(Clone, Copy, Debug)]
pub struct IrisRows {
    right: IndexedRow,
    left: IndexedRow,
}

impl IrisRows {
    /// The subject's anatomical right iris center (MediaPipe 468).
    pub fn right(&self) -> &IndexedRow {
        &self.right
    }

    /// The subject's anatomical left iris center (MediaPipe 473).
    pub fn left(&self) -> &IndexedRow {
        &self.left
    }
}

/// Mouth typed rows: lip arcs and corners with fixed semantic order (#79).
///
/// Every arc runs from the subject's right to the subject's left; corners are
/// exposed individually. The outer corners (61 right / 291 left) join the
/// outer ring; the inner corners (78 right / 308 left) join the inner ring.
#[derive(Clone, Debug)]
pub struct MouthRows {
    upper_outer: Vec<IndexedRow>,
    lower_outer: Vec<IndexedRow>,
    upper_inner: Vec<IndexedRow>,
    lower_inner: Vec<IndexedRow>,
}

impl MouthRows {
    /// Upper outer lip arc, subject-right corner to subject-left corner
    /// through the philtrum center (MediaPipe 0).
    pub fn upper_outer_arc(&self) -> &[IndexedRow] {
        &self.upper_outer
    }

    /// Lower outer lip arc, subject-right corner to subject-left corner
    /// through the lower center (MediaPipe 17).
    pub fn lower_outer_arc(&self) -> &[IndexedRow] {
        &self.lower_outer
    }

    /// Upper inner lip arc, inner-right corner to inner-left corner through
    /// the upper center (MediaPipe 13).
    pub fn upper_inner_arc(&self) -> &[IndexedRow] {
        &self.upper_inner
    }

    /// Lower inner lip arc, inner-right corner to inner-left corner through
    /// the lower center (MediaPipe 14).
    pub fn lower_inner_arc(&self) -> &[IndexedRow] {
        &self.lower_inner
    }

    /// The outer lip corner on the subject's anatomical right (MediaPipe 61).
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub fn outer_corner_right(&self) -> &IndexedRow {
        &self.upper_outer[0]
    }

    /// The outer lip corner on the subject's anatomical left (MediaPipe 291).
    // Invariant: the arc is built non-empty; see `DenseRegionGroups::from_set`.
    #[allow(clippy::expect_used)]
    pub fn outer_corner_left(&self) -> &IndexedRow {
        self.upper_outer.last().expect("arc is never empty")
    }

    /// The inner lip corner on the subject's anatomical right (MediaPipe 78).
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub fn inner_corner_right(&self) -> &IndexedRow {
        &self.upper_inner[0]
    }

    /// The inner lip corner on the subject's anatomical left (MediaPipe 308).
    // Invariant: the arc is built non-empty; see `DenseRegionGroups::from_set`.
    #[allow(clippy::expect_used)]
    pub fn inner_corner_left(&self) -> &IndexedRow {
        self.upper_inner.last().expect("arc is never empty")
    }

    /// All mouth rows in stable table order.
    pub fn all(&self) -> impl Iterator<Item = IndexedRow> {
        self.upper_outer
            .iter()
            .copied()
            .chain(self.lower_outer.iter().copied())
            .chain(self.upper_inner.iter().copied())
            .chain(self.lower_inner.iter().copied())
    }

    /// Number of distinct mapped mouth points (corner rows appear on two
    /// arcs but count once).
    pub fn len(&self) -> usize {
        summarize_distinct(self.all(), FaceRegion::Mouth).rows
    }

    /// Whether there are no mouth rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Deterministic per-region statistics (Issue #80 diagnostics).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegionSummary {
    /// Facial region.
    pub region: FaceRegion,
    /// Number of rows in the region.
    pub rows: usize,
    /// Sum of static base weights across those rows.
    pub weight_sum: f32,
}

/// Typed partition of a validated dense correspondence set.
///
/// Constructed only through [`DenseRegionGroups::from_set`]; construction is
/// total (every row lands in exactly one group) and fail-closed.
#[derive(Clone, Debug)]
pub struct DenseRegionGroups {
    central_face: CentralFaceRows,
    contour: ContourRows,
    brows: BrowRows,
    eyes: EyeRegionRows,
    irises: IrisRows,
    mouth: MouthRows,
    other: OtherValidatedRows,
}

impl DenseRegionGroups {
    /// Partitions a validated set into typed region groups.
    ///
    /// Fails closed unless every row classifies into exactly one group, every
    /// row's region tag agrees with MediaPipe topology, and per-side
    /// populations are symmetric.
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub fn from_set(set: &DenseCorrespondenceSet) -> Result<Self, GnmDenseError> {
        let mut central = Vec::new();
        let mut contour = Vec::new();
        let mut brow_right = Vec::new();
        let mut brow_left = Vec::new();
        let mut eye_right_ring = Vec::new();
        let mut eye_left_ring = Vec::new();
        let mut iris_right = None;
        let mut iris_left = None;
        let mut mouth_rows: Vec<Option<IndexedRow>> = vec![None; MEDIAPIPE_FACE_LANDMARK_COUNT];
        let mut other = Vec::new();

        for (index, row) in set.rows().iter().enumerate() {
            let indexed = IndexedRow::new(index, row);
            let mediapipe = row.mediapipe_index;
            let bucket_region = if topology::NOSE.contains(&mediapipe) {
                central.push(indexed);
                FaceRegion::Nose
            } else if topology::FACE_OVAL.contains(&mediapipe) {
                contour.push(indexed);
                FaceRegion::Contour
            } else if mediapipe == topology::IRIS_CENTER_RIGHT {
                if iris_right.replace(indexed).is_some() {
                    return Err(Self::partition_error(
                        Some(index),
                        "duplicate right iris center",
                    ));
                }
                FaceRegion::Iris
            } else if mediapipe == topology::IRIS_CENTER_LEFT {
                if iris_left.replace(indexed).is_some() {
                    return Err(Self::partition_error(
                        Some(index),
                        "duplicate left iris center",
                    ));
                }
                FaceRegion::Iris
            } else if topology::is_eyelid(mediapipe) {
                if topology::EYE_RIGHT.contains(&mediapipe) {
                    eye_right_ring.push(indexed);
                } else {
                    eye_left_ring.push(indexed);
                }
                FaceRegion::Eye
            } else if topology::is_brow(mediapipe) {
                if topology::BROW_RIGHT.contains(&mediapipe) {
                    brow_right.push(indexed);
                } else {
                    brow_left.push(indexed);
                }
                FaceRegion::Brow
            } else if topology::LIPS.contains(&mediapipe) {
                if mouth_rows[mediapipe].replace(indexed).is_some() {
                    return Err(Self::partition_error(
                        Some(index),
                        "duplicate mouth landmark slot",
                    ));
                }
                FaceRegion::Mouth
            } else {
                other.push(indexed);
                FaceRegion::Other
            };
            if row.region != bucket_region {
                return Err(Self::partition_error(
                    Some(index),
                    format!(
                        "table region tag {:?} disagrees with MediaPipe topology bucket {:?}",
                        row.region, bucket_region
                    ),
                ));
            }
        }

        let irises = IrisRows {
            right: iris_right
                .ok_or_else(|| Self::partition_error(None, "missing right iris center"))?,
            left: iris_left
                .ok_or_else(|| Self::partition_error(None, "missing left iris center"))?,
        };

        if brow_right.is_empty() || brow_left.is_empty() {
            return Err(Self::partition_error(None, "brow population incomplete"));
        }
        if brow_right.len() != brow_left.len() {
            return Err(Self::partition_error(
                None,
                format!(
                    "brow sides asymmetric: {} right versus {} left",
                    brow_right.len(),
                    brow_left.len()
                ),
            ));
        }

        let eyes = EyeRegionRows {
            right: build_eyelid_ring(topology::EYE_RIGHT, &eye_right_ring, "right")?,
            left: build_eyelid_ring(topology::EYE_LEFT, &eye_left_ring, "left")?,
        };

        let take_mouth = |slot: usize| -> Result<IndexedRow, GnmDenseError> {
            mouth_rows[slot].ok_or_else(|| {
                Self::partition_error(None, format!("mouth landmark {slot} missing"))
            })
        };
        // Arcs are materialized in their documented subject-right to
        // subject-left traversal order.
        let build_arc = |slots: &[usize]| -> Result<Vec<IndexedRow>, GnmDenseError> {
            slots
                .iter()
                .map(|slot| take_mouth(*slot))
                .collect::<Result<Vec<_>, _>>()
        };
        let mouth = MouthRows {
            upper_outer: build_arc(topology::UPPER_LIP_OUTER_ARC)?,
            lower_outer: build_arc(topology::LOWER_LIP_OUTER_ARC)?,
            upper_inner: build_arc(topology::UPPER_LIP_INNER_ARC)?,
            lower_inner: build_arc(topology::LOWER_LIP_INNER_ARC)?,
        };

        Ok(Self {
            central_face: CentralFaceRows(central),
            contour: ContourRows(contour),
            brows: BrowRows {
                right: brow_right,
                left: brow_left,
            },
            eyes,
            irises,
            mouth,
            other: OtherValidatedRows(other),
        })
    }

    fn partition_error(row: Option<usize>, reason: impl std::fmt::Display) -> GnmDenseError {
        GnmDenseError::InvalidMapping {
            row,
            reason: format!("dense region partition failed: {reason}"),
        }
    }

    /// Central-face (nose) rows (#77).
    pub fn central_face(&self) -> &CentralFaceRows {
        &self.central_face
    }

    /// Jaw-contour rows (#77).
    pub fn contour(&self) -> &ContourRows {
        &self.contour
    }

    /// Brow rows per anatomical side (#78).
    pub fn brows(&self) -> &BrowRows {
        &self.brows
    }

    /// Eyelid rings per anatomical side (#78).
    pub fn eyes(&self) -> &EyeRegionRows {
        &self.eyes
    }

    /// Iris centers (#78).
    pub fn irises(&self) -> &IrisRows {
        &self.irises
    }

    /// Mouth rows with fixed arc semantics (#79).
    pub fn mouth(&self) -> &MouthRows {
        &self.mouth
    }

    /// Remaining validated interior rows (#77).
    pub fn other_validated(&self) -> &OtherValidatedRows {
        &self.other
    }

    /// Per-region row counts and weight sums in a fixed deterministic order.
    ///
    /// The excluded-group inventory ([`EXCLUDED_MEDIAPIPE_GROUPS`]) completes
    /// the Issue #80 adoption/exclusion report together with this summary.
    pub fn region_summaries(&self) -> [RegionSummary; 7] {
        fn summarize(rows: impl Iterator<Item = IndexedRow>, region: FaceRegion) -> RegionSummary {
            let mut summary = RegionSummary {
                region,
                rows: 0,
                weight_sum: 0.0,
            };
            for entry in rows {
                summary.rows += 1;
                summary.weight_sum += entry.row.base_weight;
            }
            summary
        }
        [
            summarize(self.central_face.rows().iter().copied(), FaceRegion::Nose),
            summarize(self.contour.rows().iter().copied(), FaceRegion::Contour),
            summarize(
                self.brows
                    .right()
                    .iter()
                    .copied()
                    .chain(self.brows.left().iter().copied()),
                FaceRegion::Brow,
            ),
            summarize(
                self.eyes
                    .right()
                    .rows()
                    .iter()
                    .copied()
                    .chain(self.eyes.left().rows().iter().copied()),
                FaceRegion::Eye,
            ),
            summarize(
                std::iter::once(self.irises.right).chain(std::iter::once(self.irises.left)),
                FaceRegion::Iris,
            ),
            summarize_distinct(self.mouth.all(), FaceRegion::Mouth),
            summarize(
                self.other_validated().rows().iter().copied(),
                FaceRegion::Other,
            ),
        ]
    }
}

/// Counts distinct MediaPipe slots (corner rows appear on two mouth arcs
/// but are one mapped point) and accumulates their static weight.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn summarize_distinct(rows: impl Iterator<Item = IndexedRow>, region: FaceRegion) -> RegionSummary {
    let mut seen = [false; MEDIAPIPE_FACE_LANDMARK_COUNT];
    let mut summary = RegionSummary {
        region,
        rows: 0,
        weight_sum: 0.0,
    };
    for entry in rows {
        let slot = entry.row.mediapipe_index;
        if !seen[slot] {
            seen[slot] = true;
            summary.rows += 1;
            summary.weight_sum += entry.row.base_weight;
        }
    }
    summary
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn build_eyelid_ring(
    ring_order: &[usize],
    rows: &[IndexedRow],
    side: &str,
) -> Result<EyelidRing, GnmDenseError> {
    if rows.len() != ring_order.len() {
        return Err(DenseRegionGroups::partition_error(
            None,
            format!("{side} eyelid ring population mismatch"),
        ));
    }
    let lookup = |slot: usize| -> Result<IndexedRow, GnmDenseError> {
        rows.iter()
            .find(|entry| entry.row.mediapipe_index == slot)
            .cloned()
            .ok_or_else(|| {
                DenseRegionGroups::partition_error(
                    None,
                    format!("{side} eyelid landmark {slot} missing"),
                )
            })
    };
    let ring = ring_order
        .iter()
        .map(|slot| lookup(*slot))
        .collect::<Result<Vec<_>, _>>()?;
    let outer_corner = lookup(ring_order[0])?;
    let inner_corner_slot = ring_order[8];
    let inner_corner = lookup(inner_corner_slot)?;
    Ok(EyelidRing {
        outer_corner,
        inner_corner,
        ring,
    })
}

#[cfg(test)]
mod tests {
    use super::topology::{
        self, BROW_LEFT, BROW_RIGHT, EYE_LEFT, EYE_RIGHT, FACE_OVAL, IRIS_CENTER_LEFT,
        IRIS_CENTER_RIGHT, LIPS, NOSE,
    };
    use super::*;
    use crate::dense::test_support::{synthetic_model, version};
    use crate::{
        AnatomicalSide, CorrespondenceProvenance, CorrespondenceReliability, GnmSurfacePointRef,
    };

    fn region_tag(mp: usize) -> FaceRegion {
        if NOSE.contains(&mp) {
            FaceRegion::Nose
        } else if FACE_OVAL.contains(&mp) {
            FaceRegion::Contour
        } else if mp == IRIS_CENTER_RIGHT || mp == IRIS_CENTER_LEFT {
            FaceRegion::Iris
        } else if topology::is_eyelid(mp) {
            FaceRegion::Eye
        } else if topology::is_brow(mp) {
            FaceRegion::Brow
        } else if LIPS.contains(&mp) {
            FaceRegion::Mouth
        } else {
            FaceRegion::Other
        }
    }

    fn row(mp: usize) -> MediaPipeGnmDenseCorrespondence {
        let target = if mp == IRIS_CENTER_RIGHT || mp == IRIS_CENTER_LEFT {
            GnmSurfacePointRef::Barycentric {
                vertex_indices: [mp, mp + 1, mp + 2],
                weights: [0.5, 0.25, 0.25],
            }
        } else {
            GnmSurfacePointRef::Vertex { vertex_index: mp }
        };
        MediaPipeGnmDenseCorrespondence {
            mediapipe_index: mp,
            target,
            region: region_tag(mp),
            anatomical_side: AnatomicalSide::Midline,
            base_weight: 1.0,
            provenance: CorrespondenceProvenance::RepositoryValidated,
            reliability: CorrespondenceReliability::High,
        }
    }

    fn fixture_set(model: &crate::GnmModel) -> DenseCorrespondenceSet {
        let mut mps: Vec<usize> = NOSE
            .iter()
            .copied()
            .chain(FACE_OVAL.iter().copied())
            .chain(LIPS.iter().copied())
            .chain(EYE_RIGHT.iter().copied())
            .chain(EYE_LEFT.iter().copied())
            .chain(BROW_RIGHT.iter().copied())
            .chain(BROW_LEFT.iter().copied())
            .chain([IRIS_CENTER_RIGHT, IRIS_CENTER_LEFT, 100, 200])
            .collect();
        mps.sort_unstable();
        mps.dedup();
        DenseCorrespondenceSet::new(version(), mps.iter().map(|mp| row(*mp)).collect(), model)
            .unwrap()
    }

    #[test]
    fn partition_is_total_and_typed() {
        let model = synthetic_model(480);
        let set = fixture_set(&model);
        let groups = DenseRegionGroups::from_set(&set).unwrap();

        assert_eq!(groups.central_face().rows().len(), NOSE.len());
        assert_eq!(groups.contour().rows().len(), FACE_OVAL.len());
        assert_eq!(groups.brows().right().len(), BROW_RIGHT.len());
        assert_eq!(groups.brows().left().len(), BROW_LEFT.len());
        assert_eq!(groups.eyes().right().rows().len(), EYE_RIGHT.len());
        assert_eq!(groups.eyes().left().rows().len(), EYE_LEFT.len());
        assert_eq!(
            groups.irises().right().row.mediapipe_index,
            IRIS_CENTER_RIGHT
        );
        assert_eq!(groups.irises().left().row.mediapipe_index, IRIS_CENTER_LEFT);
        assert_eq!(groups.mouth().len(), LIPS.len());
        assert_eq!(groups.other_validated().rows().len(), 2);

        // Every row lands in exactly one typed bucket.
        let classified = groups.region_summaries();
        let total: usize = classified.iter().map(|summary| summary.rows).sum();
        assert_eq!(total, set.len());

        // Weight sums are deterministic and complete.
        let weight_total: f32 = classified.iter().map(|summary| summary.weight_sum).sum();
        let expected: f32 = set.rows().iter().map(|row| row.base_weight).sum();
        assert!((weight_total - expected).abs() < 1.0e-4);
    }

    #[test]
    fn mouth_arcs_have_fixed_corners_centers_and_direction() {
        let model = synthetic_model(480);
        let set = fixture_set(&model);
        let groups = DenseRegionGroups::from_set(&set).unwrap();
        let mouth = groups.mouth();

        // Arcs run subject-right to subject-left with the documented centers.
        let assert_arc = |arc: &[IndexedRow], right: usize, center: usize, left: usize| {
            assert_eq!(arc[0].row.mediapipe_index, right);
            assert_eq!(arc[arc.len() / 2].row.mediapipe_index, center);
            assert_eq!(
                arc.last().unwrap().row.mediapipe_index,
                left,
                "arc must end at the subject-left endpoint"
            );
        };
        assert_arc(mouth.upper_outer_arc(), 61, 0, 291);
        assert_arc(mouth.upper_inner_arc(), 78, 13, 308);
        assert_arc(mouth.lower_inner_arc(), 78, 14, 308);
        assert_arc(mouth.lower_outer_arc(), 61, 17, 291);

        assert_eq!(mouth.outer_corner_right().row.mediapipe_index, 61);
        assert_eq!(mouth.outer_corner_left().row.mediapipe_index, 291);
        assert_eq!(mouth.inner_corner_right().row.mediapipe_index, 78);
        assert_eq!(mouth.inner_corner_left().row.mediapipe_index, 308);
    }

    #[test]
    fn eyelid_rings_have_fixed_corners_and_arcs() {
        let model = synthetic_model(480);
        let set = fixture_set(&model);
        let groups = DenseRegionGroups::from_set(&set).unwrap();

        assert_eq!(groups.eyes().right().outer_corner().row.mediapipe_index, 33);
        assert_eq!(
            groups.eyes().right().inner_corner().row.mediapipe_index,
            133
        );
        assert_eq!(groups.eyes().left().outer_corner().row.mediapipe_index, 263);
        assert_eq!(groups.eyes().left().inner_corner().row.mediapipe_index, 362);

        // Upper arcs run outer corner to inner corner and include the apex
        // slots; lower arcs cover the remaining ring.
        let right = groups.eyes().right();
        assert_eq!(right.upper_arc()[0].row.mediapipe_index, 33);
        assert_eq!(right.upper_arc().last().unwrap().row.mediapipe_index, 133);
        assert_eq!(right.upper_arc().len(), 9);
        assert_eq!(right.lower_arc()[0].row.mediapipe_index, 133);
        assert_eq!(right.lower_arc().len(), 8);
        assert_eq!(
            right.upper_arc().len() + right.lower_arc().len(),
            EYE_RIGHT.len() + 1
        );
    }

    #[test]
    fn partition_fails_closed_on_tag_drift_or_asymmetry() {
        let model = synthetic_model(480);

        // A region tag that disagrees with topology is a hard error.
        let drifted_rows: Vec<MediaPipeGnmDenseCorrespondence> = {
            let mut rows = fixture_set(&model).rows().to_vec();
            let contour_slot = rows
                .iter()
                .position(|entry| entry.region == FaceRegion::Contour)
                .unwrap();
            rows[contour_slot].region = FaceRegion::Other;
            rows
        };
        let drifted_set = DenseCorrespondenceSet::new(version(), drifted_rows, &model).unwrap();
        assert!(DenseRegionGroups::from_set(&drifted_set).is_err());

        // Asymmetric brow populations fail closed.
        let mut asymmetric: Vec<MediaPipeGnmDenseCorrespondence> = NOSE
            .iter()
            .chain(BROW_RIGHT.iter())
            .map(|mp| row(*mp))
            .collect();
        asymmetric.push(row(IRIS_CENTER_RIGHT));
        asymmetric.push(row(IRIS_CENTER_LEFT));
        let asymmetric_set = DenseCorrespondenceSet::new(version(), asymmetric, &model).unwrap();
        assert!(DenseRegionGroups::from_set(&asymmetric_set).is_err());

        // A missing iris center fails closed.
        let mut no_left_iris: Vec<MediaPipeGnmDenseCorrespondence> = NOSE
            .iter()
            .chain(BROW_RIGHT.iter())
            .chain(BROW_LEFT.iter())
            .map(|mp| row(*mp))
            .collect();
        no_left_iris.push(row(IRIS_CENTER_RIGHT));
        let no_iris_set = DenseCorrespondenceSet::new(version(), no_left_iris, &model).unwrap();
        assert!(DenseRegionGroups::from_set(&no_iris_set).is_err());
    }
}
