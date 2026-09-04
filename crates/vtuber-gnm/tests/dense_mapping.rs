// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
#![allow(missing_docs)]

//! Integration tests binding the repository dense mapping to the pinned
//! GNM Head v3 asset (Issues #53 and #77–#81).
//!
//! Everything here runs from the checked-in npz asset alone: no Python, no
//! renderer, no network. Pose fixtures go through the public
//! `fitting_projection` helper; identity stays neutral and expression
//! fixtures are deterministic probes selected geometrically (expression
//! semantics are decoded in a later issue, #67).

use std::path::Path;
use std::sync::OnceLock;

use vtuber_gnm::{
    AnatomicalSide, ConditioningBaseline, CorrespondenceProvenance, DenseCoveragePolicy,
    DenseObservationStatus, DenseProjection, DenseRegionGroups, DenseReprojectionConfig,
    EXCLUDED_MEDIAPIPE_GROUPS, FaceRegion, GnmDenseObservation, GnmExpressionState,
    GnmIdentityState, GnmJointState, GnmModel, GnmSurfacePointRef, RigidRecoveryConfig,
    SynthesisOptions, SyntheticCase, compare_conditioning, evaluate_dense_reprojection,
    fitting_projection, head_sparse_68, load_gnm_head_v3, recover_rigid_projection,
    repository_dense_mapping, sparse_bootstrap_baseline, synthesize_observation_from_projection,
};

const DENSE_ROW_COUNT: usize = 470;
/// Provenance-anchor rows inside the committed table (a property of the
/// table, distinct from the 68-point comparison baseline below).
const ANCHOR_ROW_COUNT: usize = 37;
/// The Issue #81 comparison baseline is the official sparse bootstrap.
const BASELINE_ROW_COUNT: usize = 68;

fn shared_model() -> &'static GnmModel {
    static MODEL: OnceLock<GnmModel> = OnceLock::new();
    MODEL.get_or_init(|| {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/models/gnm_head.npz");
        load_gnm_head_v3(path).expect("checked-in GNM model must load")
    })
}

fn bound_mapping() -> vtuber_gnm::DenseCorrespondenceSet {
    repository_dense_mapping()
        .bind(shared_model())
        .expect("committed dense mapping must bind to the pinned model")
}

fn sparse_baseline() -> vtuber_gnm::DenseCorrespondenceSet {
    sparse_bootstrap_baseline(shared_model()).expect("official sparse bootstrap must bind")
}

fn neutral_states(model: &GnmModel) -> (GnmIdentityState, GnmExpressionState, GnmJointState) {
    (
        GnmIdentityState::neutral(model.identity_dimension()),
        GnmExpressionState::neutral(model.expression_dimension()),
        GnmJointState::neutral(model.joint_count()),
    )
}

fn evaluate_surface(
    mapping: &vtuber_gnm::DenseCorrespondenceSet,
    identity: &GnmIdentityState,
    expression: &GnmExpressionState,
    joints: &GnmJointState,
) -> Vec<[f32; 3]> {
    let model = shared_model();
    let mut surface = vtuber_gnm::GnmSparseVertices::with_len(mapping.len());
    mapping
        .evaluate_surface(model, identity, expression, joints, &mut surface)
        .expect("dense surface evaluation must succeed");
    surface.values().to_vec()
}

/// Deterministically wrong starting guess derived from a ground truth.
fn wrong_guess(truth: &DenseProjection) -> DenseProjection {
    DenseProjection::new(
        [
            truth.yaw_pitch_roll()[0] + 0.06,
            truth.yaw_pitch_roll()[1] - 0.05,
            truth.yaw_pitch_roll()[2],
        ],
        [
            truth.translation()[0] + 0.007,
            truth.translation()[1] - 0.006,
            truth.translation()[2] * 1.10,
        ],
        truth.focal() * 1.12,
        [0.5, 0.5],
    )
    .expect("fixture guess must be valid")
}

/// Runs one synthetic case for both the 68-point baseline and the full dense
/// mapping against a shared ground truth, returning typed stats in that
/// order. `invalidate` and `noise` shape the synthesized observations.
#[allow(clippy::too_many_arguments)]
fn run_case(
    case: SyntheticCase,
    noise_amplitude: f32,
    noise_seed: u64,
    invalidate: impl Fn(usize, &vtuber_gnm::MediaPipeGnmDenseCorrespondence) -> bool,
) -> Vec<vtuber_gnm::ConditioningStats> {
    let model = shared_model();
    let dense = bound_mapping();
    let sparse = sparse_baseline();
    let (identity, _, joints) = neutral_states(model);

    let plan: vtuber_gnm::CasePlan = case.plan(model, &dense, &identity, &joints);
    let surface = evaluate_surface(&dense, &identity, &plan.expression, &plan.joints);
    let truth =
        fitting_projection(&surface, plan.yaw_pitch_roll).expect("case truth must be valid");

    let build =
        |mapping: &vtuber_gnm::DenseCorrespondenceSet, min_points: usize| -> GnmDenseObservation {
            synthesize_observation_from_projection(
                model,
                &identity,
                &plan.expression,
                &plan.joints,
                mapping,
                &truth,
                SynthesisOptions {
                    noise_amplitude,
                    noise_seed,
                    ..SynthesisOptions::default()
                },
                DenseCoveragePolicy::new(min_points, 0.5).unwrap(),
                &invalidate,
            )
            .unwrap()
        };
    let sparse_observation = build(&sparse, 30);
    let dense_observation = build(&dense, 100);

    compare_conditioning(
        model,
        &identity,
        &plan.expression,
        &plan.joints,
        &[
            ConditioningBaseline {
                label: "sparse-68",
                mapping: &sparse,
                observation: &sparse_observation,
                initial_guess: wrong_guess(&truth),
            },
            ConditioningBaseline {
                label: "dense-470",
                mapping: &dense,
                observation: &dense_observation,
                initial_guess: wrong_guess(&truth),
            },
        ],
        &truth,
        RigidRecoveryConfig::default(),
    )
    .expect("conditioning comparison must succeed")
}

fn rotation_error_norm(stats: &vtuber_gnm::ConditioningStats) -> f32 {
    stats.rotation_error.iter().map(|error| error.abs()).sum()
}

// ---------------------------------------------------------------------------
// Committed table contracts (Issues #53 / #80)
// ---------------------------------------------------------------------------

#[test]
fn committed_table_binds_to_pinned_real_model() {
    let model = shared_model();
    let table = repository_dense_mapping();
    let mapping = bound_mapping();
    mapping
        .validate_model(model)
        .expect("committed table must stay compatible with the pinned model");

    assert_eq!(table.rows().len(), DENSE_ROW_COUNT);
    assert_eq!(
        table.version().model_version,
        model.version(),
        "mapping must be validated against the exact pinned model version"
    );
    assert!(mapping.validate_as_primary_observation().is_ok());

    let anchor_rows = table
        .rows()
        .iter()
        .filter(|row| row.provenance == CorrespondenceProvenance::SparseBootstrap)
        .count();
    assert_eq!(anchor_rows, ANCHOR_ROW_COUNT);

    let iris_rows = table
        .rows()
        .iter()
        .filter(|row| row.region == FaceRegion::Iris)
        .count();
    assert_eq!(iris_rows, 2, "only the two iris centers are mapped");
}

#[test]
fn real_dense_evaluation_is_deterministic() {
    let model = shared_model();
    let mapping = bound_mapping();
    let (identity, expression, joints) = neutral_states(model);
    let truth = fitting_projection(
        &evaluate_surface(&mapping, &identity, &expression, &joints),
        [0.10, -0.06, 0.03],
    )
    .unwrap();

    let observation = synthesize_observation_from_projection(
        model,
        &identity,
        &expression,
        &joints,
        &mapping,
        &truth,
        SynthesisOptions {
            source_seq: 42,
            captured_at_micros: 1_000_000,
            ..SynthesisOptions::default()
        },
        DenseCoveragePolicy::new(100, 0.5).unwrap(),
        |_, _| false,
    )
    .expect("synthetic observation must build");
    assert_eq!(observation.points().len(), DENSE_ROW_COUNT);
    assert_eq!(observation.coverage().status, DenseObservationStatus::Valid);

    let first = evaluate_dense_reprojection(
        model,
        &identity,
        &expression,
        &joints,
        &mapping,
        &observation,
        &truth,
        DenseReprojectionConfig::default(),
    )
    .unwrap();
    let second = evaluate_dense_reprojection(
        model,
        &identity,
        &expression,
        &joints,
        &mapping,
        &observation,
        &truth,
        DenseReprojectionConfig::default(),
    )
    .unwrap();
    assert_eq!(first, second, "evaluation must be deterministic");
    assert!(
        first.weighted_rms() < 1.0e-6,
        "noise-free self-projection residual was {}",
        first.weighted_rms()
    );

    // The two barycentric iris points participate in the objective.
    let iris_residuals = first
        .residuals()
        .iter()
        .filter(|residual| residual.region == FaceRegion::Iris)
        .count();
    assert_eq!(iris_residuals, 2);

    let replay = synthesize_observation_from_projection(
        model,
        &identity,
        &expression,
        &joints,
        &mapping,
        &truth,
        SynthesisOptions {
            source_seq: 42,
            captured_at_micros: 1_000_000,
            ..SynthesisOptions::default()
        },
        DenseCoveragePolicy::new(100, 0.5).unwrap(),
        |_, _| false,
    )
    .unwrap();
    assert_eq!(observation, replay, "synthesis must be deterministic");
}

#[test]
fn neutral_pose_recovery_recovers_ground_truth_on_real_geometry() {
    let model = shared_model();
    let mapping = bound_mapping();
    let (identity, expression, joints) = neutral_states(model);
    let truth = fitting_projection(
        &evaluate_surface(&mapping, &identity, &expression, &joints),
        [0.12, -0.07, 0.04],
    )
    .unwrap();

    let observation = synthesize_observation_from_projection(
        model,
        &identity,
        &expression,
        &joints,
        &mapping,
        &truth,
        SynthesisOptions::default(),
        DenseCoveragePolicy::new(100, 0.5).unwrap(),
        |_, _| false,
    )
    .unwrap();

    let outcome = recover_rigid_projection(
        model,
        &identity,
        &expression,
        &joints,
        &mapping,
        &observation,
        wrong_guess(&truth),
        RigidRecoveryConfig::default(),
    )
    .unwrap();
    assert!(outcome.converged, "solver must converge from a wrong guess");

    let recovered = outcome.projection;
    for axis in 0..3 {
        let rotation_error = recovered.yaw_pitch_roll()[axis] - truth.yaw_pitch_roll()[axis];
        assert!(
            rotation_error.abs() < 2.0e-3,
            "rotation error {rotation_error} exceeded tolerance on axis {axis}"
        );
        let translation_error = recovered.translation()[axis] - truth.translation()[axis];
        assert!(
            translation_error.abs() < 2.0e-3,
            "translation error {translation_error} exceeded tolerance on axis {axis}"
        );
    }
    let relative_focal_error = (recovered.focal() - truth.focal()) / truth.focal();
    assert!(relative_focal_error.abs() < 2.0e-3);
    assert!(outcome.final_report.weighted_rms() < 1.0e-5);
}

#[test]
fn expression_displacement_senses_through_the_dense_objective() {
    let model = shared_model();
    let mapping = bound_mapping();
    let (identity, neutral_expression, joints) = neutral_states(model);

    // Arbitrary deterministic coefficients: the expression basis semantics are
    // decoded in a later issue (#67), so this fixture only needs a real,
    // reproducible geometry change.
    let mut coefficients = vec![0.0f32; model.expression_dimension()];
    for index in (0..coefficients.len()).step_by(17) {
        coefficients[index] = if (index / 17) % 2 == 0 { 0.35 } else { -0.30 };
    }
    let expression = GnmExpressionState::new(coefficients, model.expression_dimension()).unwrap();

    let truth = fitting_projection(
        &evaluate_surface(&mapping, &identity, &neutral_expression, &joints),
        [0.05, 0.04, 0.0],
    )
    .unwrap();
    let policy = DenseCoveragePolicy::new(100, 0.5).unwrap();
    let neutral_observation = synthesize_observation_from_projection(
        model,
        &identity,
        &neutral_expression,
        &joints,
        &mapping,
        &truth,
        SynthesisOptions::default(),
        policy,
        |_, _| false,
    )
    .unwrap();
    let expressed_observation = synthesize_observation_from_projection(
        model,
        &identity,
        &expression,
        &joints,
        &mapping,
        &truth,
        SynthesisOptions::default(),
        policy,
        |_, _| false,
    )
    .unwrap();

    // Self-consistent evaluations are near zero...
    let neutral_self = evaluate_dense_reprojection(
        model,
        &identity,
        &neutral_expression,
        &joints,
        &mapping,
        &neutral_observation,
        &truth,
        DenseReprojectionConfig::default(),
    )
    .unwrap();
    let expressed_self = evaluate_dense_reprojection(
        model,
        &identity,
        &expression,
        &joints,
        &mapping,
        &expressed_observation,
        &truth,
        DenseReprojectionConfig::default(),
    )
    .unwrap();
    assert!(neutral_self.weighted_rms() < 1.0e-6);
    assert!(expressed_self.weighted_rms() < 1.0e-6);

    // ...while cross-evaluating the expressed observation against the neutral
    // state exposes a large, dense-wide displacement signal.
    let cross = evaluate_dense_reprojection(
        model,
        &identity,
        &neutral_expression,
        &joints,
        &mapping,
        &expressed_observation,
        &truth,
        DenseReprojectionConfig::default(),
    )
    .unwrap();
    println!(
        "expression cross rms {:.6} vs self {:.6}",
        cross.weighted_rms(),
        neutral_self.weighted_rms()
    );
    assert!(
        cross.weighted_rms() > 100.0 * neutral_self.weighted_rms() && cross.weighted_rms() > 1.0e-4,
        "cross rms {} did not expose expression displacement",
        cross.weighted_rms()
    );
}

// ---------------------------------------------------------------------------
// Region-specific typed contracts (Issues #77 / #78 / #79)
// ---------------------------------------------------------------------------

#[test]
fn region_groups_partition_the_committed_table_deterministically() {
    let mapping = bound_mapping();
    let groups = DenseRegionGroups::from_set(&mapping)
        .expect("committed table must partition into typed region groups");

    let summaries = groups.region_summaries();
    let total_rows: usize = summaries.iter().map(|summary| summary.rows).sum();
    assert_eq!(total_rows, DENSE_ROW_COUNT);

    let total_weight: f32 = summaries.iter().map(|summary| summary.weight_sum).sum();
    let expected_weight: f32 = mapping.rows().iter().map(|row| row.base_weight).sum();
    // f32 sums over 470 rows in different accumulation orders need a
    // relative, not absolute, tolerance.
    assert!(
        (total_weight - expected_weight).abs() <= 1.0e-4 * expected_weight.abs().max(1.0),
        "summary weight total {total_weight} drifted from row total {expected_weight}"
    );

    // Deterministic region order: Nose, Contour, Brow, Eye, Iris, Mouth, Other.
    let expected_order = [
        FaceRegion::Nose,
        FaceRegion::Contour,
        FaceRegion::Brow,
        FaceRegion::Eye,
        FaceRegion::Iris,
        FaceRegion::Mouth,
        FaceRegion::Other,
    ];
    for (summary, region) in summaries.iter().zip(expected_order) {
        assert_eq!(summary.region, region);
    }
    println!("{summaries:?}");

    // Excluded MediaPipe groups stay excluded (Issue #80 inventory).
    for group in EXCLUDED_MEDIAPIPE_GROUPS {
        for index in group.indices {
            assert!(
                !mapping
                    .rows()
                    .iter()
                    .any(|row| row.mediapipe_index == *index),
                "excluded index {index} of group {} appeared in the table",
                group.label
            );
        }
    }
}

#[test]
fn central_face_outweighs_contour_by_policy() {
    let mapping = bound_mapping();
    let groups = DenseRegionGroups::from_set(&mapping).unwrap();

    let contour_max = groups
        .contour()
        .rows()
        .iter()
        .map(|entry| entry.row.base_weight)
        .fold(f32::MIN, f32::max);
    let central_min = groups
        .central_face()
        .rows()
        .iter()
        .map(|entry| entry.row.base_weight)
        .fold(f32::MAX, f32::min);
    assert!(
        contour_max < central_min,
        "contour weight cap {contour_max} must sit below central-face floor {central_min}"
    );

    let contour_mean: f32 = groups
        .contour()
        .rows()
        .iter()
        .map(|entry| entry.row.base_weight)
        .sum::<f32>()
        / groups.contour().rows().len() as f32;
    let central_mean: f32 = groups
        .central_face()
        .rows()
        .iter()
        .map(|entry| entry.row.base_weight)
        .sum::<f32>()
        / groups.central_face().rows().len() as f32;
    assert!(contour_mean < central_mean);
}

#[test]
fn eye_brow_iris_sides_are_geometrically_authoritative() {
    let model = shared_model();
    let mapping = bound_mapping();
    let (identity, expression, joints) = neutral_states(model);
    let surface = evaluate_surface(&mapping, &identity, &expression, &joints);
    let groups = DenseRegionGroups::from_set(&mapping).unwrap();

    // Template +X is the subject's LEFT (established during derivation from
    // the eye joint positions); sides recorded in the table must agree with
    // that geometry, which no preview mirroring can influence.
    const DEAD_ZONE: f32 = 0.004;
    let template_x = |entry: &vtuber_gnm::IndexedRow| surface[entry.index][0];
    let assert_side_matches_geometry = |entry: &vtuber_gnm::IndexedRow, label: &str| {
        let x = template_x(entry);
        match entry.row.anatomical_side {
            AnatomicalSide::Left => assert!(
                x > DEAD_ZONE,
                "{label} tagged Left but template x {x} is not subject-left"
            ),
            AnatomicalSide::Right => assert!(
                x < -DEAD_ZONE,
                "{label} tagged Right but template x {x} is not subject-right"
            ),
            AnatomicalSide::Midline => assert!(
                x.abs() <= DEAD_ZONE.max(0.01),
                "{label} tagged Midline but template x {x} is off-axis"
            ),
        }
    };

    // Iris centers: exactly two barycentric rows, one per anatomical side.
    let irises = groups.irises();
    assert_eq!(irises.right().row.mediapipe_index, 468);
    assert_eq!(irises.left().row.mediapipe_index, 473);
    assert_eq!(irises.right().row.anatomical_side, AnatomicalSide::Right);
    assert_eq!(irises.left().row.anatomical_side, AnatomicalSide::Left);
    assert!(template_x(irises.left()) > template_x(irises.right()));
    for (entry, label) in [(irises.right(), "right iris"), (irises.left(), "left iris")] {
        assert_side_matches_geometry(entry, label);
        match entry.row.target {
            GnmSurfacePointRef::Barycentric {
                vertex_indices,
                weights,
            } => {
                assert!(weights.iter().all(|weight| weight.is_finite()));
                assert!((weights.iter().sum::<f32>() - 1.0).abs() < 0.002);
                assert!(
                    vertex_indices
                        .iter()
                        .all(|index| *index < model.vertex_count())
                );
            }
            GnmSurfacePointRef::Vertex { .. } => panic!("iris centers must be barycentric"),
        }
    }

    // Eyelid rings: symmetric populations, mirrored corners, apex above nadir.
    for (ring, side_label) in [
        (groups.eyes().right(), "right"),
        (groups.eyes().left(), "left"),
    ] {
        assert_side_matches_geometry(ring.outer_corner(), &format!("{side_label} outer corner"));
        assert_side_matches_geometry(ring.inner_corner(), &format!("{side_label} inner corner"));
        let upper_y: f32 = ring
            .upper_arc()
            .iter()
            .map(|entry| surface[entry.index][1])
            .sum::<f32>()
            / ring.upper_arc().len() as f32;
        let lower_y: f32 = ring
            .lower_arc()
            .iter()
            .map(|entry| surface[entry.index][1])
            .sum::<f32>()
            / ring.lower_arc().len() as f32;
        assert!(
            upper_y > lower_y,
            "{side_label} upper lid mean y {upper_y} must exceed lower lid {lower_y} (template y is up)"
        );
        for entry in ring.rows() {
            assert_eq!(
                entry.row.region,
                FaceRegion::Eye,
                "{side_label} ring rows must carry the Eye region tag"
            );
        }
    }
    assert_eq!(
        groups.eyes().right().rows().len(),
        groups.eyes().left().rows().len()
    );
    // The temporal corner of each ring is on that ring's own side: the right
    // eye opens toward subject-right, the left toward subject-left.
    assert_eq!(
        groups.eyes().right().outer_corner().row.anatomical_side,
        AnatomicalSide::Right
    );
    assert_eq!(
        groups.eyes().left().outer_corner().row.anatomical_side,
        AnatomicalSide::Left
    );

    // Brows: mirrored populations, every row on its own anatomical side.
    assert_eq!(groups.brows().right().len(), groups.brows().left().len());
    for (entries, side) in [
        (groups.brows().right(), AnatomicalSide::Right),
        (groups.brows().left(), AnatomicalSide::Left),
    ] {
        for entry in entries {
            assert_eq!(entry.row.anatomical_side, side);
            assert_side_matches_geometry(entry, "brow row");
        }
    }
    // Brows sit above the eyelids in template space.
    let brow_y: f32 = groups
        .brows()
        .right()
        .iter()
        .chain(groups.brows().left())
        .map(|entry| surface[entry.index][1])
        .sum::<f32>()
        / (groups.brows().right().len() + groups.brows().left().len()) as f32;
    let lid_y: f32 = (groups.eyes().right().upper_arc())
        .iter()
        .chain(groups.eyes().right().lower_arc())
        .map(|entry| surface[entry.index][1])
        .sum::<f32>()
        / (groups.eyes().right().upper_arc().len() + groups.eyes().right().lower_arc().len())
            as f32;
    assert!(
        brow_y > lid_y,
        "brows (mean y {brow_y}) must sit above lids ({lid_y})"
    );

    // No fabricated per-point confidence: observations built through this
    // mapping always report `None` unless a real source provides it.
    let truth = fitting_projection(&surface, [0.0, 0.0, 0.0]).unwrap();
    let observation = synthesize_observation_from_projection(
        model,
        &identity,
        &expression,
        &joints,
        &mapping,
        &truth,
        SynthesisOptions::default(),
        DenseCoveragePolicy::new(100, 0.5).unwrap(),
        |_, _| false,
    )
    .unwrap();
    assert!(
        observation
            .points()
            .iter()
            .all(|point| point.source_confidence.is_none()),
        "confidence must never be fabricated for points without a real source"
    );
}

#[test]
fn mouth_arcs_pin_lip_order_corner_sides_and_geometry() {
    let model = shared_model();
    let mapping = bound_mapping();
    let (identity, expression, joints) = neutral_states(model);
    let surface = evaluate_surface(&mapping, &identity, &expression, &joints);
    let groups = DenseRegionGroups::from_set(&mapping).unwrap();
    let mouth = groups.mouth();

    // Corner identities and anatomical sides are fixed typed properties.
    assert_eq!(mouth.outer_corner_right().row.mediapipe_index, 61);
    assert_eq!(mouth.outer_corner_left().row.mediapipe_index, 291);
    assert_eq!(mouth.inner_corner_right().row.mediapipe_index, 78);
    assert_eq!(mouth.inner_corner_left().row.mediapipe_index, 308);
    assert_eq!(
        mouth.outer_corner_right().row.anatomical_side,
        AnatomicalSide::Right
    );
    assert_eq!(
        mouth.outer_corner_left().row.anatomical_side,
        AnatomicalSide::Left
    );
    assert_eq!(
        mouth.inner_corner_right().row.anatomical_side,
        AnatomicalSide::Right
    );
    assert_eq!(
        mouth.inner_corner_left().row.anatomical_side,
        AnatomicalSide::Left
    );

    // Semantic order of every arc: subject-right endpoint, midline center,
    // subject-left endpoint — and the interior sides must follow position.
    let assert_arc = |arc: &[vtuber_gnm::IndexedRow], right: usize, center: usize, left: usize| {
        assert_eq!(arc[0].row.mediapipe_index, right, "arc start");
        assert_eq!(arc[arc.len() / 2].row.mediapipe_index, center, "arc center");
        assert_eq!(arc.last().unwrap().row.mediapipe_index, left, "arc end");
        let half = arc.len() / 2;
        for (position, entry) in arc.iter().enumerate() {
            let expected_side = if position < half {
                AnatomicalSide::Right
            } else if position == half {
                AnatomicalSide::Midline
            } else {
                AnatomicalSide::Left
            };
            assert_eq!(
                entry.row.anatomical_side, expected_side,
                "arc position {position} carries the wrong anatomical side"
            );
            assert_eq!(entry.row.region, FaceRegion::Mouth);
        }
    };
    assert_arc(mouth.upper_outer_arc(), 61, 0, 291);
    assert_arc(mouth.upper_inner_arc(), 78, 13, 308);
    assert_arc(mouth.lower_inner_arc(), 78, 14, 308);
    assert_arc(mouth.lower_outer_arc(), 61, 17, 291);

    // Geometry grounds the semantics: upper lip rows sit above lower lip rows
    // in template space (y up), and the left corner sits at +X versus the
    // right corner at −X.
    let arc_mean_y = |arc: &[vtuber_gnm::IndexedRow]| {
        arc.iter().map(|entry| surface[entry.index][1]).sum::<f32>() / arc.len() as f32
    };
    assert!(arc_mean_y(mouth.upper_outer_arc()) > arc_mean_y(mouth.lower_outer_arc()));
    assert!(arc_mean_y(mouth.upper_inner_arc()) > arc_mean_y(mouth.lower_inner_arc()));

    let corner_x = |entry: &vtuber_gnm::IndexedRow| surface[entry.index][0];
    assert!(corner_x(mouth.outer_corner_left()) > 0.004);
    assert!(corner_x(mouth.outer_corner_right()) < -0.004);
    assert!(corner_x(mouth.outer_corner_left()) > corner_x(mouth.outer_corner_right()));

    // Distinct mouth population: 40 unique landmarks across the four arcs.
    assert_eq!(mouth.len(), 40);
}

// ---------------------------------------------------------------------------
// Sparse 68-point baseline (Issues #80 / #81)
// ---------------------------------------------------------------------------

#[test]
fn sparse_baseline_derives_from_the_official_68_point_table() {
    let model = shared_model();
    let baseline = sparse_baseline();
    let official = head_sparse_68();

    assert_eq!(baseline.len(), BASELINE_ROW_COUNT);
    assert!(
        baseline.validate_as_primary_observation().is_err(),
        "the 68-point path stays a reference; it must never pass primary density"
    );

    // Targets come verbatim from the official table, order preserved.
    for (row, point) in baseline.rows().iter().zip(official.points()) {
        match row.target {
            GnmSurfacePointRef::Barycentric {
                vertex_indices,
                weights,
            } => {
                assert_eq!(vertex_indices, point.indices);
                assert_eq!(weights, point.weights);
            }
            GnmSurfacePointRef::Vertex { .. } => panic!("baseline targets must be barycentric"),
        }
        assert_eq!(row.provenance, CorrespondenceProvenance::SparseBootstrap);
        assert_eq!(
            row.base_weight, 1.0,
            "uniform weights: the official table defines none"
        );
    }

    // Regions follow the fixed iBUG-68 layout.
    let expected_regions = |dlib: usize| {
        if dlib <= 16 {
            FaceRegion::Contour
        } else if dlib <= 26 {
            FaceRegion::Brow
        } else if dlib <= 35 {
            FaceRegion::Nose
        } else if dlib <= 47 {
            FaceRegion::Eye
        } else {
            FaceRegion::Mouth
        }
    };
    for (dlib, row) in baseline.rows().iter().enumerate() {
        assert_eq!(row.region, expected_regions(dlib));
    }

    // Sides were derived from pinned template geometry (+X = subject left):
    // the mirror symmetry of the official layout must be recovered exactly.
    let left = baseline
        .rows()
        .iter()
        .filter(|row| row.anatomical_side == AnatomicalSide::Left)
        .count();
    let right = baseline
        .rows()
        .iter()
        .filter(|row| row.anatomical_side == AnatomicalSide::Right)
        .count();
    let midline = baseline
        .rows()
        .iter()
        .filter(|row| row.anatomical_side == AnatomicalSide::Midline)
        .count();
    assert_eq!(
        left, right,
        "official 68-point layout is left/right symmetric"
    );
    assert!(
        midline > 0,
        "chin/nose-center points should classify as midline"
    );
    assert_eq!(left + right + midline, BASELINE_ROW_COUNT);

    // And the whole set validates against the pinned model.
    baseline
        .validate_model(model)
        .expect("baseline targets must be valid on the pinned model");
}

// ---------------------------------------------------------------------------
// Issue #81 conditioning case suite (sparse-68 vs dense-470)
// ---------------------------------------------------------------------------

#[test]
fn issue_81_neutral_case_recovers_pose_for_both_paths() {
    let stats = run_case(SyntheticCase::Neutral, 0.0, 0, |_, _| false);
    assert_eq!(stats[0].label, "sparse-68");
    assert_eq!(stats[1].label, "dense-470");
    for stat in &stats {
        println!(
            "{}: points={} rms {:.6}->{:.6} rot {:.6} cond {:.1}",
            stat.label,
            stat.valid_points,
            stat.initial_rms,
            stat.final_rms,
            rotation_error_norm(stat),
            stat.condition_proxy
        );
        assert!(stat.final_rms < stat.initial_rms);
        assert!(
            rotation_error_norm(stat) < 6.0e-3,
            "case {} rotation drift",
            stat.label
        );
        assert!(stat.condition_proxy.is_finite());
    }
    // Same-format evidence from both paths; dense retains more of it.
    assert!(stats[1].valid_points > stats[0].valid_points * 4);
}

#[test]
fn issue_81_yaw_pitch_case_recovers_large_rotation() {
    let stats = run_case(SyntheticCase::YawPitch, 0.0015, 11, |_, _| false);
    for stat in &stats {
        println!(
            "{}: rot_err [{:.6}, {:.6}, {:.6}] final_rms {:.6}",
            stat.label,
            stat.rotation_error[0],
            stat.rotation_error[1],
            stat.rotation_error[2],
            stat.final_rms
        );
        assert!(stat.final_rms < stat.initial_rms);
        assert!(rotation_error_norm(stat) < 8.0e-3, "yaw/pitch case drift");
    }
    assert!(rotation_error_norm(&stats[1]) <= rotation_error_norm(&stats[0]) + 1.0e-3);
}

/// Mean per-point displacement of `region` rows versus everything else
/// between two evaluated surface clouds.
fn region_displacement(
    mapping: &vtuber_gnm::DenseCorrespondenceSet,
    moved: &[[f32; 3]],
    neutral: &[[f32; 3]],
    region: FaceRegion,
) -> (f64, f64) {
    let mut region_sum = 0.0f64;
    let mut other_sum = 0.0f64;
    let mut region_count = 0usize;
    for (index, _) in mapping.rows().iter().enumerate() {
        let delta: f32 = (0..3)
            .map(|axis| moved[index][axis] - neutral[index][axis])
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        if mapping.rows()[index].region == region {
            region_sum += delta as f64;
            region_count += 1;
        } else {
            other_sum += delta as f64;
        }
    }
    (
        region_sum / region_count as f64,
        other_sum / (mapping.len() - region_count).max(1) as f64,
    )
}

#[test]
fn issue_81_mouth_case_senses_mouth_displacement_and_recovers_pose() {
    let model = shared_model();
    let dense = bound_mapping();
    let (identity, _, joints) = neutral_states(model);

    let plan = SyntheticCase::Mouth.plan(model, &dense, &identity, &joints);

    // On the pinned model the mouth probe is an expression coefficient: the
    // expression channel beat every single-joint rotation on specificity.
    let probe_coefficients: Vec<(usize, f32)> = plan
        .expression
        .values()
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| *value != 0.0)
        .collect();
    assert_eq!(
        probe_coefficients.len(),
        1,
        "mouth probe is a single coefficient"
    );
    assert!(
        plan.joints
            .rotations()
            .iter()
            .flatten()
            .all(|value| *value == 0.0),
        "mouth case must not perturb joints"
    );
    println!("mouth probe coefficient {:?}", probe_coefficients[0]);

    // The probe must move mouth targets far more than the rest of the face.
    // Measured on this asset: mean ratio ≈ 3.3× (the expression basis has no
    // cleaner mouth isolator).
    let mouth_surface = evaluate_surface(&dense, &identity, &plan.expression, &plan.joints);
    let neutral_surface = evaluate_surface(&dense, &identity, &model.neutral_expression(), &joints);
    let (mouth_mean, other_mean) =
        region_displacement(&dense, &mouth_surface, &neutral_surface, FaceRegion::Mouth);
    println!("mouth mean displacement {mouth_mean:.6} vs rest {other_mean:.6}");
    assert!(
        mouth_mean > 2.0 * other_mean && mouth_mean > 5.0e-4,
        "probe did not concentrate displacement in the mouth region"
    );

    // Self-consistency holds and pose recovery still succeeds under the case.
    let stats = run_case(SyntheticCase::Mouth, 0.0, 0, |_, _| false);
    for stat in &stats {
        assert!(stat.final_rms < stat.initial_rms);
        assert!(rotation_error_norm(stat) < 6.0e-3, "mouth case pose drift");
    }
}

#[test]
fn issue_81_eyelid_case_senses_eyelid_displacement_and_recovers_pose() {
    let model = shared_model();
    let dense = bound_mapping();
    let (identity, _, joints) = neutral_states(model);

    let plan = SyntheticCase::Eyelid.plan(model, &dense, &identity, &joints);

    // On the pinned model the eyelid probe is a single joint-axis rotation:
    // the expression basis contains no strong isolated eyelid component, so
    // the planner's cross-channel scan picks an eyeball joint whose skinning
    // weights drag the lid rows while leaving the rest of the face fixed.
    let perturbed_joints: Vec<(usize, usize)> = plan
        .joints
        .rotations()
        .iter()
        .enumerate()
        .flat_map(|(joint, rotation)| {
            rotation
                .iter()
                .enumerate()
                .filter(|(_, value)| **value != 0.0)
                .map(move |(axis, _)| (joint, axis))
        })
        .collect();
    assert_eq!(perturbed_joints.len(), 1, "eyelid probe is one joint axis");
    assert!(
        plan.expression.values().iter().all(|value| *value == 0.0),
        "eyelid case must not perturb expression"
    );
    println!("eyelid probe joint/axis {:?}", perturbed_joints[0]);

    let moved = evaluate_surface(&dense, &identity, &plan.expression, &plan.joints);
    let neutral = evaluate_surface(&dense, &identity, &model.neutral_expression(), &joints);
    let (eye_mean, other_mean) = region_displacement(&dense, &moved, &neutral, FaceRegion::Eye);
    println!("eyelid mean displacement {eye_mean:.6} vs rest {other_mean:.6}");
    // Measured on this asset: mean ratio ≈ 27× with near-zero motion elsewhere.
    assert!(
        eye_mean > 10.0 * other_mean && eye_mean > 1.0e-4,
        "probe did not concentrate displacement in the eyelid region"
    );

    let stats = run_case(SyntheticCase::Eyelid, 0.0, 0, |_, _| false);
    for stat in &stats {
        assert!(stat.final_rms < stat.initial_rms);
        assert!(rotation_error_norm(stat) < 6.0e-3, "eyelid case pose drift");
    }
}

#[test]
fn cheek_contour_basis_specificity_scan() {
    // Cheek-puff projection question: does the Head v3 expression basis
    // contain an isolated contour direction, like the mouth probe
    // (coefficient 201, ~3.3x) or the eyelid joint probe (~27x)?
    // This scans all 383 channels for FaceRegion::Contour displacement
    // specificity and reports the best channel. It is a diagnostic: the
    // assertion below pins the measured outcome so a future model or
    // mapping change cannot silently alter the answer.
    let model = shared_model();
    let dense = bound_mapping();
    let (identity, _, joints) = neutral_states(model);
    let neutral = evaluate_surface(&dense, &identity, &model.neutral_expression(), &joints);

    let dim = model.expression_dimension();
    let mut best_channel = 0usize;
    let mut best_ratio = 0.0f64;
    let mut best_contour = 0.0f64;
    for channel in 0..dim {
        let mut values = vec![0.0f32; dim];
        values[channel] = 1.0;
        let expression = GnmExpressionState::new(values, dim).expect("valid expression");
        let moved = evaluate_surface(&dense, &identity, &expression, &joints);
        let (contour_mean, other_mean) =
            region_displacement(&dense, &moved, &neutral, FaceRegion::Contour);
        let ratio = contour_mean / other_mean.max(1.0e-12);
        if ratio > best_ratio {
            best_ratio = ratio;
            best_channel = channel;
            best_contour = contour_mean;
        }
    }
    println!(
        "cheek contour best channel {best_channel} ratio {best_ratio:.3} contour_mean {best_contour:.6}"
    );
    // Pinned on the committed asset: channel 104 at ~2.44x is weaker than
    // the mouth isolator (~3.3x) and far below the eyelid joint probe
    // (~27x), with a tiny absolute displacement (~1e-4). There is no
    // isolated cheek-puff direction in the basis, so the geometric
    // cheek gate must stay Experimental, never Reliable.
    assert_eq!(best_channel, 104, "cheek contour probe moved channels");
    assert!(
        (2.0..3.0).contains(&best_ratio),
        "cheek contour specificity drifted: {best_ratio}"
    );
    assert!(best_contour > 5.0e-5, "contour displacement vanished");
}

#[test]
fn issue_81_partial_invalidation_degrades_coverage_but_stays_solvable() {
    // Drop every fifth mapped landmark deterministically (~20% of rows).
    let invalidate =
        |index: usize, _: &vtuber_gnm::MediaPipeGnmDenseCorrespondence| index.is_multiple_of(5);
    let stats = run_case(SyntheticCase::Neutral, 0.0, 0, invalidate);

    // Coverage accounting is exact and identical in format for both paths:
    // invalidated rows never enter the observation, so the retained residual
    // count is exactly the non-multiples-of-five population. (`excluded_points`
    // counts reprojection-time exclusions, which stay zero here.)
    let dense_expected = DENSE_ROW_COUNT - DENSE_ROW_COUNT.div_ceil(5);
    let sparse_expected = BASELINE_ROW_COUNT - BASELINE_ROW_COUNT.div_ceil(5);
    assert_eq!(
        stats[1].valid_points, dense_expected,
        "dense retained points"
    );
    assert_eq!(
        stats[0].valid_points, sparse_expected,
        "sparse retained points"
    );
    assert_eq!(stats[1].excluded_points, 0);

    for stat in &stats {
        println!(
            "{}: valid={} excluded={}",
            stat.label, stat.valid_points, stat.excluded_points
        );
        assert!(stat.final_rms < stat.initial_rms);
        assert!(
            rotation_error_norm(stat) < 8.0e-3,
            "partial invalidation drift"
        );
    }
}

#[test]
fn issue_81_noise_case_reports_cross_seed_fit_variance() {
    let seeds = [7u64, 101, 2026];
    let mut sparse_rms = Vec::new();
    let mut dense_rms = Vec::new();
    for seed in seeds {
        let stats = run_case(SyntheticCase::YawPitch, 0.002, seed, |_, _| false);
        for stat in &stats {
            println!(
                "seed {seed} {}: points={} rot {:.6} trans [{:.6}, {:.6}, {:.6}] focal {:.6} rms {:.6}",
                stat.label,
                stat.valid_points,
                rotation_error_norm(stat),
                stat.translation_error[0],
                stat.translation_error[1],
                stat.translation_error[2],
                stat.relative_focal_error,
                stat.final_rms
            );
            assert!(stat.final_rms < stat.initial_rms, "noisy fit must improve");
            assert!(
                rotation_error_norm(stat) < 2.0e-2,
                "case {} rotation drift under noise",
                stat.label
            );
        }
        dense_rms.push(stats[1].final_rms);
        sparse_rms.push(stats[0].final_rms);
    }

    // Fit variance across noise iterations: deterministic inputs make this a
    // hard number, and it must be finite and small for both paths.
    let variance = |values: &[f32]| {
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        values
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f32>()
            / values.len() as f32
    };
    let sparse_variance = variance(&sparse_rms);
    let dense_variance = variance(&dense_rms);
    println!("cross-seed fit variance: sparse-68 {sparse_variance:?} dense-470 {dense_variance:?}");
    assert!(sparse_variance.is_finite() && dense_variance.is_finite());
    assert!(
        dense_variance.sqrt() < 1.0e-3,
        "dense fit variance too large"
    );
    assert!(
        sparse_variance.sqrt() < 1.0e-3,
        "sparse fit variance too large"
    );
}

#[test]
fn issue_81_case_results_are_deterministic_across_runs() {
    let first = run_case(SyntheticCase::Mouth, 0.0015, 2026, |_, _| false);
    let second = run_case(SyntheticCase::Mouth, 0.0015, 2026, |_, _| false);
    assert_eq!(
        first, second,
        "same case/seed/config must reproduce bit-identical stats"
    );
}
