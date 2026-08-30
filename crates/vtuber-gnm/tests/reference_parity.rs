// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
#![allow(missing_docs)]

use std::path::Path;

use vtuber_gnm::{
    GnmExpressionState, GnmIdentityState, GnmJointState, GnmSparseVertices, head_sparse_68,
    load_gnm_head_v3,
};

const FIXTURE: &str = include_str!("fixtures/official_gnm_head_v3_sparse.txt");
const TOLERANCE: f32 = 0.00005;

#[derive(Debug)]
struct FixtureCase {
    name: String,
    identity: Vec<(usize, f32)>,
    expression: Vec<(usize, f32)>,
    rotations: Vec<(usize, [f32; 3])>,
    translation: [f32; 3],
    values: Vec<[f32; 3]>,
}

#[test]
fn rust_sparse_evaluator_matches_pinned_official_gnm_reference() {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/models/gnm_head.npz");
    let model = load_gnm_head_v3(model_path).expect("checked-in GNM model must load");
    let landmarks = head_sparse_68();
    let cases = parse_fixture(FIXTURE);
    assert_eq!(cases.len(), 4);
    assert_eq!(cases[0].name, "neutral");
    assert_eq!(cases[1].name, "identity_joint_pose");
    assert_eq!(cases[2].name, "lower_face_expression");
    assert_eq!(cases[3].name, "eye_expression");

    let mut output = GnmSparseVertices::with_len(landmarks.len());
    for case in cases {
        let mut identity = vec![0.0; model.identity_dimension()];
        for (index, value) in case.identity {
            identity[index] = value;
        }
        let mut expression = vec![0.0; model.expression_dimension()];
        for (index, value) in case.expression {
            expression[index] = value;
        }
        let mut rotations = vec![[0.0; 3]; model.joint_count()];
        for (index, rotation) in case.rotations {
            rotations[index] = rotation;
        }
        let identity = GnmIdentityState::new(identity, model.identity_dimension()).unwrap();
        let expression = GnmExpressionState::new(expression, model.expression_dimension()).unwrap();
        let joints = GnmJointState::new(rotations, case.translation, model.joint_count()).unwrap();

        model
            .evaluate_sparse(&identity, &expression, &joints, landmarks, &mut output)
            .unwrap();
        assert_eq!(output.values().len(), case.values.len());
        for (point_index, (actual, expected)) in
            output.values().iter().zip(case.values.iter()).enumerate()
        {
            for component in 0..3 {
                let error = (actual[component] - expected[component]).abs();
                assert!(
                    error <= TOLERANCE,
                    "{} point {point_index} component {component}: actual={} expected={} error={error}",
                    case.name,
                    actual[component],
                    expected[component]
                );
            }
        }
    }
}

fn parse_fixture(text: &str) -> Vec<FixtureCase> {
    let mut cases = Vec::new();
    let mut current: Option<FixtureCase> = None;
    let mut reading_values = false;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(name) = line.strip_prefix("case ") {
            if let Some(case) = current.take() {
                cases.push(case);
            }
            current = Some(FixtureCase {
                name: name.to_owned(),
                identity: Vec::new(),
                expression: Vec::new(),
                rotations: Vec::new(),
                translation: [0.0; 3],
                values: Vec::new(),
            });
            reading_values = false;
            continue;
        }
        if line == "values" {
            reading_values = true;
            continue;
        }
        if line == "end" {
            reading_values = false;
            continue;
        }
        let case = current.as_mut().expect("fixture fields follow a case");
        if reading_values {
            let values: Vec<f32> = line
                .split_whitespace()
                .map(|value| value.parse().expect("fixture value is numeric"))
                .collect();
            assert_eq!(values.len(), 3);
            case.values.push([values[0], values[1], values[2]]);
            continue;
        }
        let mut fields = line.split_whitespace();
        match fields.next().expect("fixture field") {
            "identity_coeff" => case.identity.push((
                fields.next().unwrap().parse().unwrap(),
                fields.next().unwrap().parse().unwrap(),
            )),
            "expression_coeff" => case.expression.push((
                fields.next().unwrap().parse().unwrap(),
                fields.next().unwrap().parse().unwrap(),
            )),
            "joint_rotation" => case.rotations.push((
                fields.next().unwrap().parse().unwrap(),
                [
                    fields.next().unwrap().parse().unwrap(),
                    fields.next().unwrap().parse().unwrap(),
                    fields.next().unwrap().parse().unwrap(),
                ],
            )),
            "translation" => {
                case.translation = [
                    fields.next().unwrap().parse().unwrap(),
                    fields.next().unwrap().parse().unwrap(),
                    fields.next().unwrap().parse().unwrap(),
                ]
            }
            field => panic!("unknown fixture field {field}"),
        }
    }
    if let Some(case) = current {
        cases.push(case);
    }
    cases
}
