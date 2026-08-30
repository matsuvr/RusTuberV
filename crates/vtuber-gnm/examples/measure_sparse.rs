//! Opt-in local measurement helper for the sparse GNM Head v3 evaluator.

#![forbid(unsafe_code)]

use std::{error::Error, io, path::Path, time::Instant};

use vtuber_gnm::{GnmJointState, GnmSparseVertices, head_sparse_68, load_gnm_head_v3};

fn main() -> Result<(), Box<dyn Error>> {
    let iterations = parse_iterations()?;
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/models/gnm_head.npz");

    // Model loading is deliberately outside the timed section.
    let model = load_gnm_head_v3(model_path)?;
    let landmarks = head_sparse_68();
    let identity = model.neutral_identity();
    let expression = model.neutral_expression();
    let joints = GnmJointState::neutral(model.joint_count());
    let mut output = GnmSparseVertices::with_len(landmarks.len());

    for _ in 0..32 {
        model.evaluate_sparse(&identity, &expression, &joints, landmarks, &mut output)?;
    }

    let started = Instant::now();
    let mut checksum = 0.0f32;
    for _ in 0..iterations {
        model.evaluate_sparse(&identity, &expression, &joints, landmarks, &mut output)?;
        checksum += output.values().first().map_or(0.0, |point| point[0]);
    }
    let elapsed = started.elapsed();
    let total_seconds = elapsed.as_secs_f64();

    println!("model_load=excluded");
    println!("warmup_iterations=32");
    println!("iterations={iterations}");
    println!("total_elapsed_ms={:.3}", total_seconds * 1_000.0);
    println!(
        "per_iteration_us={:.3}",
        total_seconds * 1_000_000.0 / iterations as f64
    );
    println!("checksum={checksum:.9}");
    Ok(())
}

fn parse_iterations() -> Result<usize, Box<dyn Error>> {
    let mut arguments = std::env::args().skip(1);
    let Some(flag) = arguments.next() else {
        return Ok(1_000);
    };
    if flag != "--iterations" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: cargo run -p vtuber-gnm --example measure_sparse -- --iterations N",
        )
        .into());
    }
    let value = arguments.next().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "--iterations requires a positive integer",
        )
    })?;
    let iterations = value.parse::<usize>().map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid iteration count `{value}`: {error}"),
        )
    })?;
    if iterations == 0 || arguments.next().is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--iterations requires one positive integer",
        )
        .into());
    }
    Ok(iterations)
}
