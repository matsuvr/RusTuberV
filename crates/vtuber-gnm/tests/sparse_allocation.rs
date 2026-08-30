// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
#![allow(missing_docs)]

use std::path::Path;

use vtuber_gnm::{GnmSparseVertices, head_sparse_68, load_gnm_head_v3};

#[test]
fn repeated_sparse_evaluation_reuses_sparse_scratch_only() {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/models/gnm_head.npz");
    let model = load_gnm_head_v3(model_path).expect("checked-in GNM model must load");
    let landmarks = head_sparse_68();
    let identity = model.neutral_identity();
    let expression = model.neutral_expression();
    let joints = vtuber_gnm::GnmJointState::neutral(model.joint_count());
    let mut output = GnmSparseVertices::with_len(landmarks.len());

    model
        .evaluate_sparse(&identity, &expression, &joints, landmarks, &mut output)
        .unwrap();
    let first_values = output.values().to_vec();
    let capacities = output.reusable_capacities();
    assert_eq!(output.vertex_scratch_len(), landmarks.unique_vertex_count());
    assert!(
        output.vertex_scratch_capacity() < model.vertex_count(),
        "sparse scratch must not have dense vertex capacity: scratch={} dense={}",
        output.vertex_scratch_capacity(),
        model.vertex_count()
    );

    for _ in 0..32 {
        model
            .evaluate_sparse(&identity, &expression, &joints, landmarks, &mut output)
            .unwrap();
        assert_eq!(output.values(), first_values.as_slice());
        assert_eq!(output.reusable_capacities(), capacities);
        assert_eq!(output.vertex_scratch_len(), landmarks.unique_vertex_count());
    }
}
