//! One-time deterministic generator for the repository-owned MediaPipe-to-GNM
//! dense correspondence table (Issue #53).
//!
//! The derivation is deliberately explicit and self-checking:
//!
//! 1. Parse the pinned MediaPipe canonical face model OBJ (Apache-2.0; see
//!    `assets/models/manifest.toml` for the upstream revision and SHA-256).
//! 2. Materialize the neutral GNM Head v3 template surface through the public
//!    sparse evaluator (no private model access).
//! 3. Verify a documented anchor set against geometric assertions on both
//!    sides, so the iBUG-style slot semantics are checked, not trusted.
//! 4. Fit a similarity transform from MediaPipe canonical space to GNM template
//!    space on the verified anchors (Kabsch with uniform scale, Horn quaternion).
//! 5. Snap each candidate MediaPipe point to its nearest template vertex and
//!    keep only rows that pass distance, side, and duplicate-target gates.
//! 6. Derive barycentric iris-center targets from the official GNM eye joints.
//!
//! Nothing here runs at runtime; the committed table under `assets/` is the
//! single typed source of truth consumed by
//! `vtuber_gnm::repository_dense_mapping`.
//!
//! Usage:
//! ```text
//! cargo run -p vtuber-gnm --example derive_mediapipe_dense_mapping -- \
//!     <canonical_face_model.obj> [--out <table path>] [--diagnose]
//! ```

use std::fmt::Write as _;
use std::path::PathBuf;

use vtuber_gnm::{
    GNM_HEAD_V3_VERSION, GnmJointState, GnmModel, GnmSparseVertices, SPARSE_BOOTSTRAP_POINT_COUNT,
    SparseLandmark, SparseLandmarkSet, head_sparse_68, load_gnm_head_v3,
};

/// Repository mapping schema revision emitted by this generator.
const SCHEMA_REVISION: u32 = 1;

/// Maximum post-fit residual (template units) for interior verified anchors.
///
/// The template spans roughly 0.25 units across the face, so this gate is a
/// few millimeters at real-world head scale.
const CENTRAL_ANCHOR_TOLERANCE: f32 = 0.010;
/// Loose gate for face-oval boundary anchors. The canonical MediaPipe model is
/// a cropped face without ears or scalp while the GNM template covers the whole
/// head, so auricular/jaw proportions legitimately differ between the two.
const CONTOUR_ANCHOR_TOLERANCE: f32 = 0.025;
/// Maximum fitted-source-to-nearest-vertex distance for a snapped row.
const SNAP_TOLERANCE: f32 = 0.008;
/// Canonical-space dead zone around the midline where side is not asserted.
const CANONICAL_SIDE_DEAD_ZONE: f32 = 0.5;
/// Template-space dead zone around the midline where side is not asserted.
const TEMPLATE_SIDE_DEAD_ZONE: f32 = 0.004;

/// Verification tier of a documented anchor pair.
#[derive(Clone, Copy, Debug, PartialEq)]
enum AnchorTier {
    /// Interior semantic point; must satisfy [`CENTRAL_ANCHOR_TOLERANCE`].
    Central,
    /// Face-oval boundary point; governed by [`CONTOUR_ANCHOR_TOLERANCE`].
    Contour,
}

impl AnchorTier {
    fn tolerance(self) -> f32 {
        match self {
            Self::Central => CENTRAL_ANCHOR_TOLERANCE,
            Self::Contour => CONTOUR_ANCHOR_TOLERANCE,
        }
    }
}

/// High-confidence MediaPipe-to-iBUG68 anchor pairs.
///
/// Each pair names a semantic point whose identity is anatomically
/// unambiguous on both sides (eye/mouth corners, nose bridge, chin, brows).
struct CoreAnchor {
    mediapipe: usize,
    dlib: usize,
    tier: AnchorTier,
}

const CORE_ANCHORS: &[CoreAnchor] = &[
    CoreAnchor {
        mediapipe: 33,
        dlib: 36,
        tier: AnchorTier::Central,
    }, // right eye outer corner
    CoreAnchor {
        mediapipe: 133,
        dlib: 39,
        tier: AnchorTier::Central,
    }, // right eye inner corner
    CoreAnchor {
        mediapipe: 263,
        dlib: 45,
        tier: AnchorTier::Central,
    }, // left eye outer corner
    CoreAnchor {
        mediapipe: 362,
        dlib: 42,
        tier: AnchorTier::Central,
    }, // left eye inner corner
    CoreAnchor {
        mediapipe: 61,
        dlib: 48,
        tier: AnchorTier::Central,
    }, // right mouth corner
    CoreAnchor {
        mediapipe: 291,
        dlib: 54,
        tier: AnchorTier::Central,
    }, // left mouth corner
    CoreAnchor {
        mediapipe: 0,
        dlib: 51,
        tier: AnchorTier::Central,
    }, // upper lip center
    CoreAnchor {
        mediapipe: 17,
        dlib: 57,
        tier: AnchorTier::Central,
    }, // lower lip lower center
    CoreAnchor {
        mediapipe: 78,
        dlib: 60,
        tier: AnchorTier::Central,
    }, // inner mouth right corner
    CoreAnchor {
        mediapipe: 308,
        dlib: 64,
        tier: AnchorTier::Central,
    }, // inner mouth left corner
    CoreAnchor {
        mediapipe: 168,
        dlib: 27,
        tier: AnchorTier::Central,
    }, // nose bridge top
    CoreAnchor {
        mediapipe: 6,
        dlib: 28,
        tier: AnchorTier::Central,
    }, // nose bridge upper
    CoreAnchor {
        mediapipe: 197,
        dlib: 29,
        tier: AnchorTier::Central,
    }, // nose bridge lower
    CoreAnchor {
        mediapipe: 4,
        dlib: 30,
        tier: AnchorTier::Central,
    }, // nose tip (pronasale)
    CoreAnchor {
        mediapipe: 2,
        dlib: 33,
        tier: AnchorTier::Central,
    }, // subnasale
    CoreAnchor {
        mediapipe: 98,
        dlib: 31,
        tier: AnchorTier::Central,
    }, // right nostril outer
    CoreAnchor {
        mediapipe: 327,
        dlib: 35,
        tier: AnchorTier::Central,
    }, // left nostril outer
    CoreAnchor {
        mediapipe: 70,
        dlib: 17,
        tier: AnchorTier::Central,
    }, // right brow outer
    CoreAnchor {
        mediapipe: 105,
        dlib: 19,
        tier: AnchorTier::Central,
    }, // right brow center
    CoreAnchor {
        mediapipe: 107,
        dlib: 21,
        tier: AnchorTier::Central,
    }, // right brow inner
    CoreAnchor {
        mediapipe: 300,
        dlib: 26,
        tier: AnchorTier::Central,
    }, // left brow outer
    CoreAnchor {
        mediapipe: 334,
        dlib: 24,
        tier: AnchorTier::Central,
    }, // left brow center
    CoreAnchor {
        mediapipe: 336,
        dlib: 22,
        tier: AnchorTier::Central,
    }, // left brow inner
    CoreAnchor {
        mediapipe: 152,
        dlib: 8,
        tier: AnchorTier::Contour,
    }, // chin bottom (gnathion)
    CoreAnchor {
        mediapipe: 234,
        dlib: 0,
        tier: AnchorTier::Contour,
    }, // right preauricular contour
    CoreAnchor {
        mediapipe: 454,
        dlib: 16,
        tier: AnchorTier::Contour,
    }, // left preauricular contour
];

/// Secondary anchor candidates kept only when the fitted residual passes
/// [`ANCHOR_FIT_TOLERANCE`]; used to strengthen the fit without trusting
/// uncertain slot semantics blindly.
const CANDIDATE_ANCHORS: &[(usize, usize)] = &[
    (246, 37), // right upper lid, outer-mid
    (161, 38), // right upper lid, inner-mid
    (163, 41), // right lower lid, outer-mid
    (7, 40),   // right lower lid, inner-mid
    (466, 43), // left upper lid, outer-mid
    (388, 44), // left upper lid, inner-mid
    (249, 47), // left lower lid, outer-mid
    (390, 46), // left lower lid, inner-mid
    (185, 49), // right upper lip ring
    (39, 50),  // right upper lip ring
    (269, 52), // left upper lip ring
    (270, 53), // left upper lip ring
    (375, 55), // left lower lip ring
    (405, 56), // left lower lip ring
    (181, 58), // right lower lip ring
    (91, 59),  // right lower lip ring
];

/// Core anchors asserted to sit on the facial midline in both spaces.
const MIDLINE_ANCHORS: &[(usize, usize)] = &[
    (168, 27),
    (6, 28),
    (197, 29),
    (4, 30),
    (2, 33),
    (0, 51),
    (17, 57),
    (152, 8),
];

/// MediaPipe face-oval contour indices (yaw-sensitive silhouette ring).
const FACE_OVAL: &[usize] = &[
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
    148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
];
/// MediaPipe lip rings (outer + inner).
const LIPS: &[usize] = &[
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 78,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
];
/// MediaPipe right eyelid ring.
const EYE_RIGHT: &[usize] = &[
    33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7,
];
/// MediaPipe left eyelid ring.
const EYE_LEFT: &[usize] = &[
    263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249,
];
/// MediaPipe right brow.
const BROW_RIGHT: &[usize] = &[70, 63, 105, 66, 107, 46, 53, 52, 65];
/// MediaPipe left brow.
const BROW_LEFT: &[usize] = &[300, 293, 334, 296, 336, 276, 283, 282, 295];
/// MediaPipe nose bridge, tip, and nostril region.
const NOSE: &[usize] = &[
    1, 2, 4, 5, 6, 19, 94, 97, 98, 99, 164, 165, 167, 168, 195, 196, 197, 129, 358, 326, 327, 64,
    240, 49, 48, 279, 278,
];

/// Iris landmark centers in the 478-point MediaPipe output.
const IRIS_CENTER_RIGHT: usize = 468;
const IRIS_CENTER_LEFT: usize = 473;

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut obj_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut diagnose = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => {
                let Some(path) = args.next() else {
                    return Err("--out requires a path".into());
                };
                out_path = Some(PathBuf::from(path));
            }
            "--diagnose" => diagnose = true,
            other => {
                if obj_path.is_none() {
                    obj_path = Some(PathBuf::from(other));
                } else {
                    return Err(format!("unexpected argument: {other}").into());
                }
            }
        }
    }
    let obj_path = obj_path.ok_or("usage: derive_mediapipe_dense_mapping <obj> [--out <path>]")?;

    let canonical = parse_obj_vertices(&obj_path)?;
    assert_eq!(
        canonical.len(),
        468,
        "pinned canonical face model must contain 468 vertices"
    );
    assert!(
        canonical.iter().all(|v| v.iter().all(|c| c.is_finite())),
        "canonical model contains non-finite coordinates"
    );

    let model = load_model()?;
    assert_eq!(model.version(), GNM_HEAD_V3_VERSION);
    let template = neutral_template_positions(&model)?;
    let dlib_targets = evaluate_dlib_targets(&model)?;

    println!(
        "canonical bounds: {:?} .. {:?}",
        axis_bounds(canonical.iter().copied()).0,
        axis_bounds(canonical.iter().copied()).1
    );
    println!(
        "template bounds:  {:?} .. {:?} ({})",
        axis_bounds(template.iter().copied()).0,
        axis_bounds(template.iter().copied()).1,
        template.len()
    );

    // ------------------------------------------------------------------
    // 1. Geometric verification of core anchors before fitting.
    // ------------------------------------------------------------------
    verify_core_anchors(&canonical, &dlib_targets);

    // ------------------------------------------------------------------
    // 2. Initial similarity fit on central anchors only.
    // ------------------------------------------------------------------
    let central_pairs: Vec<([f32; 3], [f32; 3])> = CORE_ANCHORS
        .iter()
        .filter(|anchor| anchor.tier == AnchorTier::Central)
        .map(|anchor| (canonical[anchor.mediapipe], dlib_targets[anchor.dlib]))
        .collect();
    let initial_fit = kabsch_similarity(&central_pairs);
    report_fit_residuals(
        "core anchors (initial central-only fit)",
        CORE_ANCHORS,
        &initial_fit,
        &canonical,
        &dlib_targets,
    );
    let core_tuples: Vec<(usize, usize, AnchorTier)> = CORE_ANCHORS
        .iter()
        .map(|anchor| (anchor.mediapipe, anchor.dlib, anchor.tier))
        .collect();
    assert_anchors_within_tier(&core_tuples, &initial_fit, &canonical, &dlib_targets);

    // ------------------------------------------------------------------
    // 3. Gate contour-tier cores and secondary candidates, then refine once.
    // ------------------------------------------------------------------
    let mut accepted: Vec<(usize, usize, AnchorTier)> = CORE_ANCHORS
        .iter()
        .map(|anchor| (anchor.mediapipe, anchor.dlib, anchor.tier))
        .collect();
    let mut rejected_candidates: Vec<(usize, usize, f32)> = Vec::new();
    for pair in CANDIDATE_ANCHORS {
        let residual = single_residual(*pair, &initial_fit, &canonical, &dlib_targets);
        if residual <= CENTRAL_ANCHOR_TOLERANCE {
            accepted.push((pair.0, pair.1, AnchorTier::Central));
        } else {
            rejected_candidates.push((pair.0, pair.1, residual));
        }
    }
    if !rejected_candidates.is_empty() {
        println!("\ncandidate anchors excluded by residual gate:");
        for (mp, dlib, residual) in &rejected_candidates {
            println!("  mp {mp} -> dlib {dlib}: {residual:.5}");
        }
    }
    let refined_pairs: Vec<([f32; 3], [f32; 3])> = accepted
        .iter()
        .map(|(mp, dlib, _)| (canonical[*mp], dlib_targets[*dlib]))
        .collect();
    let fit = kabsch_similarity(&refined_pairs);
    let refined_max = max_residual(
        &accepted
            .iter()
            .map(|(mp, dlib, _)| (*mp, *dlib))
            .collect::<Vec<_>>(),
        &fit,
        &canonical,
        &dlib_targets,
    );
    println!(
        "\nrefined fit on {} anchors: max residual {refined_max:.5}, scale {:.5}",
        accepted.len(),
        fit.scale
    );
    assert_anchors_within_tier(&accepted, &fit, &canonical, &dlib_targets);

    if diagnose {
        print_diagnostics(&canonical, &template, &dlib_targets, &fit);
        return Ok(());
    }

    let accepted_pairs: Vec<(usize, usize)> =
        accepted.iter().map(|(mp, dlib, _)| (*mp, *dlib)).collect();

    // ------------------------------------------------------------------
    // 4. Snap every canonical vertex to its nearest template vertex.
    // ------------------------------------------------------------------
    let rows = derive_rows(&canonical, &template, &fit, &accepted_pairs);

    // ------------------------------------------------------------------
    // 5. Emit.
    // ------------------------------------------------------------------
    assert!(
        rows.len() > SPARSE_BOOTSTRAP_POINT_COUNT * 4,
        "derived mapping must be clearly denser than the sparse bootstrap"
    );
    let text = render_table(&rows, &obj_path, accepted_pairs.len(), &rejected_candidates);
    let default_out =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/mediapipe_dense_mapping_v1.txt");
    let out_path = out_path.unwrap_or(default_out);
    std::fs::write(&out_path, &text)
        .map_err(|error| format!("cannot write {}: {error}", out_path.display()))?;
    println!("\nwrote {} rows to {}", rows.len(), out_path.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Row derivation
// ---------------------------------------------------------------------------

/// One derived correspondence row pending emission.
struct DerivedRow {
    mediapipe_index: usize,
    target_vertex: usize,
    region: &'static str,
    side: &'static str,
    weight: f32,
    provenance: &'static str,
    reliability: &'static str,
    barycentric: Option<([usize; 3], [f32; 3])>,
}

fn region_of(mp: usize) -> &'static str {
    if FACE_OVAL.contains(&mp) {
        "contour"
    } else if LIPS.contains(&mp) {
        "mouth"
    } else if EYE_RIGHT.contains(&mp) || EYE_LEFT.contains(&mp) {
        "eye"
    } else if BROW_RIGHT.contains(&mp) || BROW_LEFT.contains(&mp) {
        "brow"
    } else if NOSE.contains(&mp) {
        "nose"
    } else if mp == IRIS_CENTER_RIGHT || mp == IRIS_CENTER_LEFT {
        "iris"
    } else {
        "other"
    }
}

fn reliability_of(mp: usize, region: &str) -> &'static str {
    const MOUTH_CORNERS: [usize; 8] = [61, 291, 78, 308, 0, 17, 13, 14];
    const EYE_CORNERS: [usize; 4] = [33, 133, 263, 362];
    match region {
        "contour" => "low",
        "nose" | "brow" => "high",
        "mouth" if MOUTH_CORNERS.contains(&mp) => "high",
        "eye" if EYE_CORNERS.contains(&mp) => "high",
        _ => "medium",
    }
}

fn weight_of(reliability: &str) -> f32 {
    match reliability {
        "high" => 1.0,
        "medium" => 0.8,
        "low" => 0.5,
        _ => unreachable!("unknown reliability"),
    }
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn derive_rows(
    canonical: &[[f32; 3]],
    template: &[[f32; 3]],
    fit: &Similarity,
    anchors: &[(usize, usize)],
) -> Vec<DerivedRow> {
    let mut claimed = vec![false; template.len()];
    let mut rows = Vec::new();
    let mut excluded = ExclusionLog::default();

    // ------------------------------------------------------------------
    // Priority 1: verified anchor points claim their nearest vertices.
    // ------------------------------------------------------------------
    for (mp, _) in anchors {
        let source = fit.apply(canonical[*mp]);
        let (nearest, distance) = nearest_vertex_unclaimed(&source, template, &claimed);
        assert!(
            distance <= SNAP_TOLERANCE,
            "verified anchor mp {mp} must snap within tolerance, got {distance:.5}"
        );
        assert!(!claimed[nearest], "anchor collision at vertex {nearest}");
        claimed[nearest] = true;
        let region = region_of(*mp);
        let reliability = reliability_of(*mp, region);
        rows.push(DerivedRow {
            mediapipe_index: *mp,
            target_vertex: nearest,
            region,
            side: side_of(template[nearest][0], TEMPLATE_SIDE_DEAD_ZONE),
            weight: weight_of(reliability),
            provenance: "sparse_bootstrap",
            reliability,
            barycentric: None,
        });
    }

    // ------------------------------------------------------------------
    // Priority 2: iris centers as barycentric targets of the official GNM
    // eye-joint positions (the eyeball moves with gaze, so its surface is the
    // correct observation target for iris landmarks).
    // ------------------------------------------------------------------
    for (mp, center, side) in [
        (
            IRIS_CENTER_RIGHT,
            eye_center(&fit.apply(canonical[33]), &fit.apply(canonical[133])),
            "right",
        ),
        (
            IRIS_CENTER_LEFT,
            eye_center(&fit.apply(canonical[263]), &fit.apply(canonical[362])),
            "left",
        ),
    ] {
        match derive_barycentric_point(center, template) {
            Some((indices, weights)) if !indices.iter().any(|index| claimed[*index]) => {
                for index in indices {
                    claimed[index] = true;
                }
                rows.push(DerivedRow {
                    mediapipe_index: mp,
                    target_vertex: 0,
                    region: "iris",
                    side,
                    weight: weight_of("medium"),
                    provenance: "research_derived",
                    reliability: "medium",
                    barycentric: Some((indices, weights)),
                });
            }
            _ => excluded.iris.push(mp),
        }
    }

    // ------------------------------------------------------------------
    // Priority 3: every remaining canonical point snaps to its nearest
    // unclaimed template vertex behind distance/side gates.
    // ------------------------------------------------------------------
    for (mp, canonical_point) in canonical.iter().enumerate() {
        if anchors.iter().any(|(anchor_mp, _)| *anchor_mp == mp)
            || mp == IRIS_CENTER_RIGHT
            || mp == IRIS_CENTER_LEFT
        {
            continue;
        }
        let source = fit.apply(*canonical_point);
        let (nearest, distance) = nearest_vertex_unclaimed(&source, template, &claimed);
        if distance > SNAP_TOLERANCE {
            excluded.distance.push((mp, distance));
            continue;
        }
        let side_source = side_of(canonical_point[0], CANONICAL_SIDE_DEAD_ZONE);
        let side_target = side_of(template[nearest][0], TEMPLATE_SIDE_DEAD_ZONE);
        let Some(side) = agree_sides(side_source, side_target) else {
            excluded.side_mismatch.push((mp, nearest));
            continue;
        };
        if claimed[nearest] {
            excluded.duplicate_target.push((mp, nearest));
            continue;
        }
        claimed[nearest] = true;
        let region = region_of(mp);
        let reliability = reliability_of(mp, region);
        rows.push(DerivedRow {
            mediapipe_index: mp,
            target_vertex: nearest,
            region,
            side,
            weight: weight_of(reliability),
            provenance: "repository_validated",
            reliability,
            barycentric: None,
        });
    }

    excluded.report(rows.len());
    rows.sort_by_key(|row| row.mediapipe_index);
    rows
}

/// Midpoint of the inner and outer eye corners.
fn eye_center(outer: &[f32; 3], inner: &[f32; 3]) -> [f32; 3] {
    [
        (outer[0] + inner[0]) * 0.5,
        (outer[1] + inner[1]) * 0.5,
        (outer[2] + inner[2]) * 0.5,
    ]
}

/// Least-squares barycentric coordinates of `point` on the triangle formed by
/// three nearby template vertices, validated for finiteness, near-nonnegativity,
/// and a small planar residual. Deterministic: candidate triangles are tried in
/// ascending-vertex-index order.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn derive_barycentric_point(
    point: [f32; 3],
    template: &[[f32; 3]],
) -> Option<([usize; 3], [f32; 3])> {
    const NEARBY: usize = 6;
    let mut candidates: Vec<(usize, f32)> = (0..template.len())
        .map(|index| {
            let dx = template[index][0] - point[0];
            let dy = template[index][1] - point[1];
            let dz = template[index][2] - point[2];
            (index, dx * dx + dy * dy + dz * dz)
        })
        .collect();
    candidates.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    candidates.truncate(NEARBY);

    let mut best: Option<([usize; 3], [f32; 3], f32)> = None;
    for a in 0..NEARBY {
        for b in a + 1..NEARBY {
            for c in b + 1..NEARBY {
                let indices = [candidates[a].0, candidates[b].0, candidates[c].0];
                let pa = template[indices[0]];
                let pb = template[indices[1]];
                let pc = template[indices[2]];
                let e1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                let e2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
                let rhs = [point[0] - pa[0], point[1] - pa[1], point[2] - pa[2]];
                let dot11 = dot3(e1, e1);
                let dot12 = dot3(e1, e2);
                let dot22 = dot3(e2, e2);
                let det = dot11 * dot22 - dot12 * dot12;
                if det.abs() < 1.0e-12 {
                    continue;
                }
                let rhs1 = dot3(e1, rhs);
                let rhs2 = dot3(e2, rhs);
                let u = (dot22 * rhs1 - dot12 * rhs2) / det;
                let v = (dot11 * rhs2 - dot12 * rhs1) / det;
                if !(u.is_finite() && v.is_finite()) {
                    continue;
                }
                let weights = [1.0 - u - v, u, v];
                if weights.iter().any(|weight| *weight < -0.002) {
                    continue;
                }
                // Clamp tiny negatives and renormalize into the valid simplex.
                let clamped = weights.map(|weight| weight.max(0.0));
                let sum: f32 = clamped.iter().sum();
                let normalized = [clamped[0] / sum, clamped[1] / sum, clamped[2] / sum];
                let projected = [
                    pa[0] * normalized[0] + pb[0] * normalized[1] + pc[0] * normalized[2],
                    pa[1] * normalized[0] + pb[1] * normalized[1] + pc[1] * normalized[2],
                    pa[2] * normalized[0] + pb[2] * normalized[1] + pc[2] * normalized[2],
                ];
                let residual = distance(projected, point);
                let better_than_best = best
                    .as_ref()
                    .is_none_or(|(_, _, best_residual)| residual < *best_residual);
                if residual <= SNAP_TOLERANCE && better_than_best {
                    best = Some((indices, normalized, residual));
                }
            }
        }
    }
    best.map(|(indices, weights, _)| (indices, weights))
}

#[derive(Default)]
struct ExclusionLog {
    distance: Vec<(usize, f32)>,
    side_mismatch: Vec<(usize, usize)>,
    duplicate_target: Vec<(usize, usize)>,
    iris: Vec<usize>,
}

impl ExclusionLog {
    fn report(&self, kept: usize) {
        println!("\nderivation summary:");
        println!("  kept rows:               {kept}");
        println!("  distance-gate exclusions: {}", self.distance.len());
        println!("  side-mismatch exclusions: {}", self.side_mismatch.len());
        println!(
            "  duplicate-target drops:   {}",
            self.duplicate_target.len()
        );
        if !self.iris.is_empty() {
            println!("  unresolvable iris rows:   {:?}", self.iris);
        }
    }
}

// ---------------------------------------------------------------------------
// Similarity fit (Kabsch with uniform scale, Horn quaternion eigen-solve)
// ---------------------------------------------------------------------------

struct Similarity {
    scale: f32,
    rotation: [[f32; 3]; 3],
    translation: [f32; 3],
}

impl Similarity {
    fn apply(&self, point: [f32; 3]) -> [f32; 3] {
        let rotated = [
            self.rotation[0][0] * point[0]
                + self.rotation[0][1] * point[1]
                + self.rotation[0][2] * point[2],
            self.rotation[1][0] * point[0]
                + self.rotation[1][1] * point[1]
                + self.rotation[1][2] * point[2],
            self.rotation[2][0] * point[0]
                + self.rotation[2][1] * point[1]
                + self.rotation[2][2] * point[2],
        ];
        [
            self.scale * rotated[0] + self.translation[0],
            self.scale * rotated[1] + self.translation[1],
            self.scale * rotated[2] + self.translation[2],
        ]
    }
}

/// Fits `target ≈ scale · R · source + translation` minimizing anchor residuals.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn kabsch_similarity(pairs: &[([f32; 3], [f32; 3])]) -> Similarity {
    assert!(
        pairs.len() >= 3,
        "similarity fit needs at least three pairs"
    );
    let source_centroid = centroid(pairs.iter().map(|(source, _)| *source));
    let target_centroid = centroid(pairs.iter().map(|(_, target)| *target));

    // Cross-covariance H = Σ (s-cs)(t-ct)^T.
    let mut h = [[0.0f64; 3]; 3];
    for (source, target) in pairs {
        let s = [
            source[0] as f64 - source_centroid[0],
            source[1] as f64 - source_centroid[1],
            source[2] as f64 - source_centroid[2],
        ];
        let t = [
            target[0] as f64 - target_centroid[0],
            target[1] as f64 - target_centroid[1],
            target[2] as f64 - target_centroid[2],
        ];
        for row in 0..3 {
            for col in 0..3 {
                h[row][col] += s[row] * t[col];
            }
        }
    }

    let quaternion = largest_eigenvector_quaternion(&h);
    let rotation = quaternion_to_matrix(quaternion);

    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    for (source, target) in pairs {
        let s = [
            source[0] as f64 - source_centroid[0],
            source[1] as f64 - source_centroid[1],
            source[2] as f64 - source_centroid[2],
        ];
        let t = [
            target[0] as f64 - target_centroid[0],
            target[1] as f64 - target_centroid[1],
            target[2] as f64 - target_centroid[2],
        ];
        let rotated_s = mat_vec(&rotation, s);
        numerator += dot(rotated_s, t);
        denominator += dot(s, s);
    }
    assert!(denominator > 1.0e-12, "degenerate anchor spread");
    let scale = (numerator / denominator) as f32;

    let rotated_source_centroid = mat_vec(&rotation, source_centroid);
    let translation = [
        (target_centroid[0] - scale as f64 * rotated_source_centroid[0]) as f32,
        (target_centroid[1] - scale as f64 * rotated_source_centroid[1]) as f32,
        (target_centroid[2] - scale as f64 * rotated_source_centroid[2]) as f32,
    ];

    Similarity {
        scale,
        rotation: rotation.map(|row| row.map(|value| value as f32)),
        translation,
    }
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn centroid(points: impl Iterator<Item = [f32; 3]>) -> [f64; 3] {
    let mut sum = [0.0f64; 3];
    let mut count = 0.0f64;
    for point in points {
        for axis in 0..3 {
            sum[axis] += point[axis] as f64;
        }
        count += 1.0;
    }
    assert!(count >= 1.0);
    [sum[0] / count, sum[1] / count, sum[2] / count]
}

fn mat_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Largest eigenvector of Horn's symmetric 4×4 quaternion matrix, via cyclic
/// Jacobi rotations (deterministic, dependency-free).
#[allow(clippy::needless_range_loop)]
// fixed-size dense linear algebra reads best indexed
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn largest_eigenvector_quaternion(h: &[[f64; 3]; 3]) -> [f64; 4] {
    let [sxx, sxy, sxz] = [h[0][0], h[0][1], h[0][2]];
    let [syx, syy, syz] = [h[1][0], h[1][1], h[1][2]];
    let [szx, szy, szz] = [h[2][0], h[2][1], h[2][2]];
    let mut n = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];
    // Jacobi eigen-decomposition of the symmetric matrix `n`.
    let mut v = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for _ in 0..64 {
        let mut off = 0.0f64;
        for row in 0..4 {
            for col in row + 1..4 {
                off += n[row][col] * n[row][col];
            }
        }
        if off < 1.0e-24 {
            break;
        }
        for p in 0..4 {
            for q in p + 1..4 {
                if n[p][q].abs() < 1.0e-18 {
                    continue;
                }
                let theta = (n[q][q] - n[p][p]) / (2.0 * n[p][q]);
                let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..4 {
                    let nkp = n[k][p];
                    let nkq = n[k][q];
                    n[k][p] = c * nkp - s * nkq;
                    n[k][q] = s * nkp + c * nkq;
                }
                for k in 0..4 {
                    let npk = n[p][k];
                    let nqk = n[q][k];
                    n[p][k] = c * npk - s * nqk;
                    n[q][k] = s * npk + c * nqk;
                }
                for k in 0..4 {
                    let vkp = v[k][p];
                    let vkq = v[k][q];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    // Pick the column with the largest diagonal eigenvalue.
    let mut best = 0;
    for k in 1..4 {
        if n[k][k] > n[best][best] {
            best = k;
        }
    }
    let q = [v[0][best], v[1][best], v[2][best], v[3][best]];
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]
}

/// Converts the `[w, x, y, z]` unit quaternion into a rotation matrix.
fn quaternion_to_matrix(q: [f64; 4]) -> [[f64; 3]; 3] {
    let [w, x, y, z] = q;
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

// ---------------------------------------------------------------------------
// Anchor verification helpers
// ---------------------------------------------------------------------------

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn verify_core_anchors(canonical: &[[f32; 3]], dlib_targets: &[[f32; 3]]) {
    for anchor in CORE_ANCHORS {
        let mp = anchor.mediapipe;
        let dlib = anchor.dlib;
        let source = canonical[mp];
        let target = dlib_targets[dlib];
        if MIDLINE_ANCHORS.contains(&(mp, dlib)) {
            // Midline assertions: both sides must sit near their own midline.
            // Canonical units are tens of units wide, template units are ~0.25.
            assert!(
                source[0].abs() <= 0.35,
                "midline anchor mp {mp} is off-midline: x={}",
                source[0]
            );
            assert!(
                target[0].abs() <= 0.010,
                "midline anchor dlib {dlib} is off-midline: x={}",
                target[0]
            );
            continue;
        }
        // Side-sign agreement: +x is subject-left in both the canonical model
        // and the GNM template, so subject-side must match by sign once either
        // side leaves its own midline dead zone.
        if source[0].abs() > CANONICAL_SIDE_DEAD_ZONE && target[0].abs() > TEMPLATE_SIDE_DEAD_ZONE {
            let source_sign = source[0] > 0.0;
            let target_sign = target[0] > 0.0;
            assert_eq!(
                source_sign, target_sign,
                "anchor side mismatch: mp {mp} x={} vs dlib {dlib} x={}",
                source[0], target[0]
            );
        }
    }
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn single_residual(
    (mp, dlib): (usize, usize),
    fit: &Similarity,
    canonical: &[[f32; 3]],
    dlib_targets: &[[f32; 3]],
) -> f32 {
    let mapped = fit.apply(canonical[mp]);
    distance(mapped, dlib_targets[dlib])
}

fn max_residual(
    pairs: &[(usize, usize)],
    fit: &Similarity,
    canonical: &[[f32; 3]],
    dlib_targets: &[[f32; 3]],
) -> f32 {
    pairs
        .iter()
        .map(|pair| single_residual(*pair, fit, canonical, dlib_targets))
        .fold(0.0f32, f32::max)
}

fn report_fit_residuals(
    label: &str,
    pairs: &[CoreAnchor],
    fit: &Similarity,
    canonical: &[[f32; 3]],
    dlib_targets: &[[f32; 3]],
) {
    println!("\n{label}:");
    for anchor in pairs {
        let residual = single_residual(
            (anchor.mediapipe, anchor.dlib),
            fit,
            canonical,
            dlib_targets,
        );
        println!(
            "  mp {:3} -> dlib {:2} ({:?}): {residual:.5}",
            anchor.mediapipe, anchor.dlib, anchor.tier
        );
    }
}

/// Fails the derivation if any verified anchor drifts past its tier tolerance.
fn assert_anchors_within_tier(
    pairs: &[(usize, usize, AnchorTier)],
    fit: &Similarity,
    canonical: &[[f32; 3]],
    dlib_targets: &[[f32; 3]],
) {
    for (mp, dlib, tier) in pairs {
        let residual = single_residual((*mp, *dlib), fit, canonical, dlib_targets);
        assert!(
            residual <= tier.tolerance(),
            "anchor verification failed: mp {mp} -> dlib {dlib} residual {residual:.5} exceeds {:?} tolerance {}",
            tier,
            tier.tolerance()
        );
    }
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Nearest unclaimed vertex search; ties break toward the lower index via
/// strict `<`.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn nearest_vertex_unclaimed(
    point: &[f32; 3],
    template: &[[f32; 3]],
    claimed: &[bool],
) -> (usize, f32) {
    let mut best = usize::MAX;
    let mut best_squared = f32::MAX;
    for (index, vertex) in template.iter().enumerate() {
        if claimed[index] {
            continue;
        }
        let dx = vertex[0] - point[0];
        let dy = vertex[1] - point[1];
        let dz = vertex[2] - point[2];
        let squared = dx * dx + dy * dy + dz * dz;
        if squared < best_squared {
            best_squared = squared;
            best = index;
        }
    }
    assert!(best != usize::MAX, "no unclaimed vertex available");
    (best, best_squared.sqrt())
}

fn side_of(coordinate: f32, dead_zone: f32) -> &'static str {
    if coordinate.abs() <= dead_zone {
        "midline"
    } else if coordinate > 0.0 {
        "left"
    } else {
        "right"
    }
}

fn agree_sides(source: &'static str, target: &'static str) -> Option<&'static str> {
    match (source, target) {
        ("midline", other) | (other, "midline") => Some(other),
        (same, equal) if same == equal => Some(same),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Table emission
// ---------------------------------------------------------------------------

fn render_table(
    rows: &[DerivedRow],
    obj_path: &std::path::Path,
    anchor_count: usize,
    rejected_candidates: &[(usize, usize, f32)],
) -> String {
    let mut text = String::new();
    let _ = writeln!(
        text,
        "# Repository-owned MediaPipe(478)-to-GNM Head v3 dense correspondence table."
    );
    let _ = writeln!(
        text,
        "# Derived by crates/vtuber-gnm/examples/derive_mediapipe_dense_mapping.rs"
    );
    let _ = writeln!(
        text,
        "# Source input: pinned MediaPipe canonical_face_model.obj (468 vertices, Apache-2.0);"
    );
    let _ = writeln!(
        text,
        "# upstream revision and SHA-256 are recorded in assets/models/manifest.toml."
    );
    let _ = writeln!(
        text,
        "# Input file name: {} (content identity is pinned by SHA-256 in the manifest).",
        obj_path
            .file_name()
            .map(|name| name.to_string_lossy())
            .unwrap_or_else(|| "canonical_face_model.obj".into())
    );
    let _ = writeln!(
        text,
        "# NOT an official Google correspondence table: derivation gates, adopted counts,"
    );
    let _ = writeln!(
        text,
        "# and exclusion groups are documented in docs/gnm-dense-observation.md."
    );
    let _ = writeln!(
        text,
        "# Verified anchors feeding the similarity fit: {anchor_count}; excluded candidates: {}.",
        rejected_candidates.len()
    );
    let _ = writeln!(
        text,
        "mapping_version schema_revision={SCHEMA_REVISION} model_major=3 model_minor=0"
    );
    for row in rows {
        if row.barycentric.is_some() {
            write_barycentric_row(&mut text, row);
        } else {
            let _ = writeln!(
                text,
                "row {} vertex {} {} {} {} {} {}",
                row.mediapipe_index,
                row.target_vertex,
                row.region,
                row.side,
                format_weight(row.weight),
                row.provenance,
                row.reliability
            );
        }
    }
    text
}

fn write_barycentric_row(text: &mut String, row: &DerivedRow) {
    // Invariant: this writer is only called for rows that carry barycentric
    // data; the caller branches on exactly that field.
    #[allow(clippy::expect_used)]
    let (indices, weights) = row.barycentric.as_ref().expect("barycentric row data");
    let _ = writeln!(
        text,
        "row {} barycentric {} {} {} {} {} {} {} {} {} {} {}",
        row.mediapipe_index,
        indices[0],
        indices[1],
        indices[2],
        format_weight(weights[0]),
        format_weight(weights[1]),
        format_weight(weights[2]),
        row.region,
        row.side,
        format_weight(row.weight),
        row.provenance,
        row.reliability
    );
}

fn format_weight(weight: f32) -> String {
    format!("{weight:.6}")
}

// ---------------------------------------------------------------------------
// Pinned asset parsing and model materialization
// ---------------------------------------------------------------------------

fn parse_obj_vertices(path: &std::path::Path) -> Result<Vec<[f32; 3]>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let mut vertices = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("v ") {
            let mut parts = rest.split_whitespace();
            let parse_coord = |label: &str, raw: Option<&str>| -> Result<f32, String> {
                let raw = raw.ok_or_else(|| format!("{path:?}: OBJ vertex is missing {label}"))?;
                raw.parse::<f32>().map_err(|error| {
                    format!("{path:?}: invalid {label} coordinate `{raw}`: {error}")
                })
            };
            let x = parse_coord("x", parts.next())?;
            let y = parse_coord("y", parts.next())?;
            let z = parse_coord("z", parts.next())?;
            vertices.push([x, y, z]);
        }
    }
    Ok(vertices)
}

fn load_model() -> Result<GnmModel, Box<dyn std::error::Error>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/models/gnm_head.npz");
    Ok(load_gnm_head_v3(path)?)
}

/// Materializes every neutral template vertex through the public evaluator so
/// the generator never touches private model state.
fn neutral_template_positions(
    model: &GnmModel,
) -> Result<Vec<[f32; 3]>, Box<dyn std::error::Error>> {
    let vertex_count = model.vertex_count();
    let landmarks: Vec<SparseLandmark> = (0..vertex_count)
        .map(|vertex| {
            SparseLandmark::new([vertex, vertex, vertex], [1.0, 0.0, 0.0])
                .map_err(|error| format!("identity landmark {vertex}: {error}"))
        })
        .collect::<Result<_, _>>()?;
    let set = SparseLandmarkSet::new(landmarks).map_err(|error| error.to_string())?;
    let mut output = GnmSparseVertices::with_len(vertex_count);
    model
        .evaluate_sparse(
            &model.neutral_identity(),
            &model.neutral_expression(),
            &GnmJointState::neutral(model.joint_count()),
            &set,
            &mut output,
        )
        .map_err(|error| format!("neutral full-template evaluation failed: {error}"))?;
    Ok(output.values().to_vec())
}

/// Evaluates the official 68 sparse landmarks on the neutral template.
fn evaluate_dlib_targets(model: &GnmModel) -> Result<Vec<[f32; 3]>, Box<dyn std::error::Error>> {
    let set = head_sparse_68();
    let mut output = GnmSparseVertices::with_len(set.len());
    model
        .evaluate_sparse(
            &model.neutral_identity(),
            &model.neutral_expression(),
            &GnmJointState::neutral(model.joint_count()),
            set,
            &mut output,
        )
        .map_err(|error| format!("official 68 evaluation failed: {error}"))?;
    Ok(output.values().to_vec())
}

/// Gate-tuning diagnostics: fitted snap-distance distribution for every
/// canonical point plus per-anchor residuals under the refined fit.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn print_diagnostics(
    canonical: &[[f32; 3]],
    template: &[[f32; 3]],
    dlib_targets: &[[f32; 3]],
    fit: &Similarity,
) {
    println!("\nanchor residuals under refined fit:");
    let all_anchors = CORE_ANCHORS
        .iter()
        .map(|anchor| (anchor.mediapipe, anchor.dlib))
        .chain(CANDIDATE_ANCHORS.iter().copied());
    for (mp, dlib) in all_anchors {
        let residual = single_residual((mp, dlib), fit, canonical, dlib_targets);
        println!("  mp {mp:3} -> dlib {dlib:2}: {residual:.5}");
    }

    let unclaimed = vec![false; template.len()];
    let mut buckets = [0usize; 12];
    for point in canonical {
        let mapped = fit.apply(*point);
        let (_, distance) = nearest_vertex_unclaimed(&mapped, template, &unclaimed);
        let bucket = ((distance / 0.002) as usize).min(11);
        buckets[bucket] += 1;
    }
    println!("\nsnap-distance histogram (bucket width = 0.002 template units):");
    for (index, count) in buckets.iter().enumerate() {
        println!(
            "  {:.3}-{:.3}: {count}",
            index as f32 * 0.002,
            (index + 1) as f32 * 0.002
        );
    }
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn axis_bounds(points: impl Iterator<Item = [f32; 3]>) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for point in points {
        for axis in 0..3 {
            min[axis] = min[axis].min(point[axis]);
            max[axis] = max[axis].max(point[axis]);
        }
    }
    (min, max)
}
