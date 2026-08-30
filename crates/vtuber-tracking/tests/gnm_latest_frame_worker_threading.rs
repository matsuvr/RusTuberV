//! Threaded latest-frame worker integration test (Issue #95).
//!
//! Runs in its own test binary so the wall-clock-sensitive worker scheduling
//! assertions are not affected by unit-test CPU contention.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // tests may panic (AGENTS.md)

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use vtuber_core::{LatestSlot, ReadResult, WorkerResult};
use vtuber_gnm::{
    DenseCoveragePolicy, DenseProjection, FixedGnmIdentity, GnmDenseObservation,
    GnmExpressionState, GnmFrameStamp, GnmJointState, PersistentGnmLifecycleConfig,
    SingleFrameFitConfig, SynthesisOptions, synthesize_observation_from_projection,
};
use vtuber_tracking::{
    GnmFaceState, GnmFitterResources, GnmFitterWorkerMetrics, GnmLatestFrameWorker,
    GnmWorkerFrameInput, GnmWorkerInput, spawn_gnm_latest_frame_worker, synthetic_head_model,
    synthetic_mapping,
};

type WorkerTestParts = (
    GnmLatestFrameWorker,
    GnmFitterResources,
    Arc<LatestSlot<GnmWorkerInput>>,
    Arc<LatestSlot<GnmFaceState>>,
    Arc<LatestSlot<GnmFitterWorkerMetrics>>,
);

fn worker() -> WorkerTestParts {
    let input = Arc::new(LatestSlot::new());
    let output = Arc::new(LatestSlot::new());
    let metrics_slot = Arc::new(LatestSlot::new());
    let solver_config = SingleFrameFitConfig::new(
        vtuber_gnm::DenseRigidStepConfig::default(),
        vtuber_gnm::DenseExpressionJointStepConfig::default(),
        16,
        1.0e-6,
    )
    .unwrap();
    let lifecycle_config = PersistentGnmLifecycleConfig::new(50_000, 250_000, 3).unwrap();
    let res = resources();
    let worker = GnmLatestFrameWorker::new(
        res.clone(),
        solver_config,
        lifecycle_config,
        Arc::clone(&input),
        Arc::clone(&output),
        Arc::clone(&metrics_slot),
    );
    (worker, res, input, output, metrics_slot)
}

/// The exact per-frame sources are internal; the smallest meaningful bound
/// is the number of completed fits (each advances the source).
fn final_processed_source(metrics: &GnmFitterWorkerMetrics) -> u64 {
    metrics.solved_frames
}

const FRAME_MICROS: u64 = 16_667;

fn resources() -> GnmFitterResources {
    let model = Arc::new(synthetic_head_model().unwrap());
    let mapping = Arc::new(synthetic_mapping(&model).unwrap());
    let identity = Arc::new(FixedGnmIdentity::new(model.neutral_identity(), &model).unwrap());
    GnmFitterResources {
        model,
        mapping,
        identity,
    }
}

fn stamped_observation(
    resources: &GnmFitterResources,
    seq: u64,
    micros: u64,
) -> Arc<GnmDenseObservation> {
    let mut values = vec![0.0_f32; resources.model.expression_dimension()];
    // Bounds: the buffer is sized by the model's validated expression
    // dimension, which is at least one.
    #[allow(clippy::indexing_slicing)]
    {
        values[0] = 0.5;
    }
    let len = values.len();
    let expression = GnmExpressionState::new(values, len).unwrap();
    let projection =
        DenseProjection::new([0.15, -0.10, 0.05], [0.02, -0.03, 0.60], 1.3, [0.5, 0.5]).unwrap();
    Arc::new(
        synthesize_observation_from_projection(
            &resources.model,
            resources.identity.state(),
            &expression,
            &GnmJointState::neutral(resources.model.joint_count()),
            &resources.mapping,
            &projection,
            SynthesisOptions {
                source_seq: seq,
                captured_at_micros: micros,
                ..SynthesisOptions::default()
            },
            DenseCoveragePolicy::new(2, 0.75).unwrap(),
            |_, _| false,
        )
        .unwrap(),
    )
}

fn frame(resources: &GnmFitterResources, seq: u64) -> GnmWorkerInput {
    GnmWorkerInput::Frame(GnmWorkerFrameInput::new(
        GnmFrameStamp {
            source_seq: seq,
            captured_at_micros: seq * FRAME_MICROS,
        },
        stamped_observation(resources, seq, seq * FRAME_MICROS),
    ))
}

fn latest_output(output: &LatestSlot<GnmFaceState>) -> Option<GnmFaceState> {
    match output.try_read_after(0) {
        Some(ReadResult::New(state)) => Some(state),
        _ => None,
    }
}

fn metrics_snapshot(slot: &LatestSlot<GnmFitterWorkerMetrics>) -> Option<GnmFitterWorkerMetrics> {
    match slot.try_read_after(0) {
        Some(ReadResult::New(metrics)) => Some(metrics),
        _ => None,
    }
}

#[test]
fn spawned_worker_skips_backlog_and_shuts_down_deterministically() {
    let (worker, res, input, output, metrics_slot) = worker();
    let handle = spawn_gnm_latest_frame_worker("gnm-test-worker", worker);

    input.publish(GnmWorkerInput::CalibrationReady);
    // Burst-publish a backlog plus a slow drip so newer frames keep
    // replacing pending ones while fits run.
    for seq in 1_u64..=10 {
        input.publish(frame(&res, seq));
    }
    thread::sleep(Duration::from_millis(200));
    let mut next_seq = 11_u64;
    // Wait until at least two fits completed (bounded wall time) while
    // continuing to drip frames.
    let started = std::time::Instant::now();
    loop {
        if metrics_snapshot(&metrics_slot).is_some_and(|m| m.solved_frames >= 2) {
            break;
        }
        // The wall-clock bound must absorb full-suite parallel load,
        // where debug-profile fits slow down by an order of magnitude.
        assert!(
            started.elapsed() < Duration::from_secs(120),
            "worker did not complete two fits in time"
        );
        if next_seq <= 80 {
            input.publish(frame(&res, next_seq));
            next_seq += 1;
        }
        thread::sleep(Duration::from_millis(25));
    }

    handle.stop();
    let joined = thread::scope(|scope| {
        scope
            .spawn(|| handle.join())
            .join()
            .expect("join thread did not panic")
    });
    assert_eq!(joined, WorkerResult::Completed(()));

    // Latest-value semantics: the worker cannot have ground through the
    // whole backlog, and the published state is near the newest frame.
    let metrics = loop {
        match metrics_slot.try_read_after(0) {
            Some(ReadResult::New(metrics)) => break metrics,
            _ => thread::sleep(Duration::from_millis(10)),
        }
    };
    assert!(metrics.solved_frames >= 2);
    assert_eq!(metrics.published_frames, metrics.solved_frames);
    // Reaching a sequence well beyond the first processed frame proves
    // stale entries were skipped by the slot instead of queued (the
    // strict property has a deterministic unit test above).
    let final_state = latest_output(&output).expect("state published");
    assert!(
        final_state.stamp().source_seq > final_processed_source(&metrics),
        "final published seq {}",
        final_state.stamp().source_seq
    );
    assert!(metrics.max_fit_latency_micros > 0);
}
