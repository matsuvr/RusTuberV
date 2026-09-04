// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]
//! Repository automation entry point.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::env;
use std::path::{Path, PathBuf};
use std::process;

mod ab_report;
mod acceptance;
mod face_image_probe;
mod face_pipeline_smoke;
mod mediapipe_face_smoke;
mod mediapipe_pose_probe;
mod ndi;
mod ndi_output_render;
mod teacher_ablation;
mod teacher_fit_aligned_basis;
mod teacher_fit_gnm_decoder;
mod teacher_fit_observable_basis;
mod teacher_fit_prior;
mod teacher_fit_residual;
mod teacher_replay;
mod teacher_residual_ablation;
mod temporal_report;
mod vrm_compatibility;
mod vrm_managed_compatibility;

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        println!("usage: cargo xtask <task>");
        println!("tasks:");
        println!("  vrm-compat [fixture-dir]  run bevy_vrm1 compatibility gate");
        println!(
            "  vrm-managed-compat <path-to-model.vrm>  run the managed user:// lifecycle gate"
        );
        println!("  acceptance <command>      Windows acceptance test support");
        println!("  face-image-probe <path>  Legacy research UltraFace/Peppa probe");
        println!("  face-pipeline-smoke       Legacy research detector/crop/landmark probe");
        println!("  mediapipe-face-smoke      Windows MSMF MediaPipe Face Landmarker gate");
        println!("  mediapipe-pose-probe      Guided MediaPipe neutral-relative pose proof");
        println!("  temporal-report <json>    Direct/GNM temporal quality report (GNM #57.4)");
        println!(
            "  ab-report <json>          Direct/GNM robustness/cross-talk/performance A/B report (GNM #57.5)"
        );
        println!(
            "  teacher-replay <opts>     Offline ARKit-teacher replay into a derived trace (GNM #68.3)"
        );
        println!(
            "  teacher-fit-prior <opts>  Fit/export the causal linear prior from derived traces (GNM #68.4)"
        );
        println!(
            "  teacher-fit-observable-basis <opts> Fit geometric non-tongue GNM basis (Issue #15)"
        );
        println!(
            "  teacher-fit-aligned-basis <opts> Fit teacher-aligned observable GNM basis (Issue #16)"
        );
        println!(
            "  teacher-fit-gnm-decoder <opts> Fit GNM-only or hybrid reduced-GNM decoder (Issue #17)"
        );
        println!(
            "  teacher-fit-residual <opts> Fit/export same-frame teacher residual decoder (Issue #12)"
        );
        println!("  teacher-residual-ablation <opts> Evaluate D/G0/H0 on held-out traces (#13)");
        println!(
            "  teacher-ablation <opts>   Held-out no-prior/learned-prior ablation evaluation (GNM #68.5)"
        );
        println!("  ndi <command>             Stage or verify a Windows NDI release package");
        return;
    }

    match args[0].as_str() {
        "vrm-compat" => {
            let fixture_dir = args
                .get(1)
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("tests/fixtures/vrm"));
            match vrm_compatibility::run(&fixture_dir) {
                Ok(results) => {
                    let mut failed = 0;
                    for result in &results {
                        print_result(result);
                        if result.runner_error.is_some()
                            || result.preflight.is_err()
                            || result.runtime.as_ref().is_some_and(|r| !r.is_mvp_capable())
                        {
                            failed += 1;
                        }
                    }
                    if failed > 0 {
                        eprintln!("{failed} fixture(s) failed the compatibility gate");
                        process::exit(vrm_compatibility::EXIT_COMPAT_FAIL);
                    }
                }
                Err(e) => {
                    eprintln!("compatibility runner failed: {e}");
                    process::exit(1);
                }
            }
        }
        "vrm-managed-compat" => {
            let Some(path) = args.get(1).map(PathBuf::from) else {
                eprintln!("usage: cargo xtask -- vrm-managed-compat <path-to-model.vrm>");
                process::exit(1);
            };
            if let Err(error) = vrm_managed_compatibility::run(&path) {
                eprintln!("managed compatibility runner failed: {error}");
                process::exit(1);
            }
        }
        "acceptance" => {
            handle_acceptance(&args[1..]);
        }
        "face-image-probe" => match face_image_probe::run(&args[1..]) {
            Ok(()) => {}
            Err(error) => {
                eprintln!("face-image-probe failed: {error}");
                process::exit(1);
            }
        },
        "face-pipeline-smoke" => match face_pipeline_smoke::run(&args[1..]) {
            Ok(()) => {}
            Err(error) => {
                eprintln!("face-pipeline-smoke failed: {error}");
                process::exit(1);
            }
        },
        "mediapipe-face-smoke" => match mediapipe_face_smoke::run(&args[1..]) {
            Ok(()) => {}
            Err(error) => {
                eprintln!("mediapipe-face-smoke failed: {error}");
                process::exit(1);
            }
        },
        "mediapipe-pose-probe" => match mediapipe_pose_probe::run(&args[1..]) {
            Ok(()) => {}
            Err(error) => {
                eprintln!("mediapipe-pose-probe failed: {error}");
                std::process::exit(1);
            }
        },
        "teacher-replay" => match teacher_replay::run(&args[1..]) {
            Ok(()) => {}
            Err(error) => {
                eprintln!("teacher-replay failed: {error}");
                process::exit(1);
            }
        },
        "teacher-fit-prior" => match teacher_fit_prior::run(&args[1..]) {
            Ok(()) => {}
            Err(error) => {
                eprintln!("teacher-fit-prior failed: {error}");
                process::exit(1);
            }
        },
        "teacher-fit-observable-basis" => match teacher_fit_observable_basis::run(&args[1..]) {
            Ok(()) => {}
            Err(error) => {
                eprintln!("teacher-fit-observable-basis failed: {error}");
                process::exit(1);
            }
        },
        "teacher-fit-aligned-basis" => match teacher_fit_aligned_basis::run(&args[1..]) {
            Ok(()) => {}
            Err(error) => {
                eprintln!("teacher-fit-aligned-basis failed: {error}");
                process::exit(1);
            }
        },
        "teacher-fit-gnm-decoder" => match teacher_fit_gnm_decoder::run(&args[1..]) {
            Ok(()) => {}
            Err(error) => {
                eprintln!("teacher-fit-gnm-decoder failed: {error}");
                process::exit(1);
            }
        },
        "teacher-fit-residual" => match teacher_fit_residual::run(&args[1..]) {
            Ok(()) => {}
            Err(error) => {
                eprintln!("teacher-fit-residual failed: {error}");
                process::exit(1);
            }
        },
        "teacher-residual-ablation" => match teacher_residual_ablation::run(&args[1..]) {
            Ok(()) => {}
            Err(error) => {
                eprintln!("teacher-residual-ablation failed: {error}");
                process::exit(1);
            }
        },
        "teacher-ablation" => match teacher_ablation::run(&args[1..]) {
            Ok(()) => {}
            Err(error) => {
                eprintln!("teacher-ablation failed: {error}");
                process::exit(1);
            }
        },
        "ab-report" => match ab_report::run(&args[1..]) {
            Ok(()) => {}
            Err(error) => {
                eprintln!("ab-report failed: {error}");
                std::process::exit(1);
            }
        },
        "temporal-report" => match temporal_report::run(&args[1..]) {
            Ok(()) => {}
            Err(error) => {
                eprintln!("temporal-report failed: {error}");
                std::process::exit(1);
            }
        },
        "ndi" => {
            if let Err(error) = ndi::run(&args[1..])
                && error != "help requested"
            {
                eprintln!("ndi command failed: {error}");
                if error.starts_with("NOT RUN:") {
                    process::exit(ndi_output_render::EXIT_NOT_RUN);
                }
                process::exit(1);
            }
        }
        other => {
            eprintln!("unknown task: {other}");
            process::exit(1);
        }
    }
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn handle_acceptance(args: &[String]) {
    if args.is_empty() {
        acceptance::print_help();
        return;
    }

    match args[0].as_str() {
        "env" => {
            acceptance::print_env();
        }
        "new" => {
            let base_dir = args
                .get(1)
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("docs/acceptance/runs"));
            match acceptance::new_run(&base_dir) {
                Ok(run_dir) => println!("Created acceptance run: {}", run_dir.display()),
                Err(e) => {
                    eprintln!("failed to create run: {e}");
                    process::exit(1);
                }
            }
        }
        "verify" => {
            let manifest = args
                .get(1)
                .map(Path::new)
                .unwrap_or_else(|| Path::new("assets/models/manifest.toml"));
            if let Err(e) = acceptance::verify_models(manifest) {
                eprintln!("verify failed: {e}");
                process::exit(1);
            }
        }
        "help" | "--help" | "-h" => {
            acceptance::print_help();
        }
        other => {
            eprintln!("unknown acceptance command: {other}");
            acceptance::print_help();
            process::exit(1);
        }
    }
}

fn print_result(result: &vrm_compatibility::CompatibilityResult) {
    let name = result
        .path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();
    println!("=== {name} ===");
    println!("  machine:");
    println!("    file_size={}", result.file_size);
    println!("    sha256={}", result.sha256);
    match &result.preflight {
        Ok(summary) => {
            println!("  preflight: ok");
            println!("    generation: {:?}", summary.generation);
            println!("    name: {}", summary.name);
            println!("    specVersion: {}", summary.spec_version);
            println!("    exporterVersion: {:?}", summary.exporter_version);
            println!("    expressions: {:?}", summary.expression_presets);
            println!("    lookAt type: {:?}", summary.look_at_type);
            println!("    springBone: {}", summary.has_spring_bone);
            println!(
                "    material counts: mtoon={}, unlit={}, fallback={}",
                summary.mtoon_material_count,
                summary.unlit_material_count,
                summary.fallback_material_count
            );
            println!(
                "    spring source-declared inventory: groups_or_springs={}, joint_or_root_references={}, colliders={}, centers={}",
                summary.spring_chain_count,
                summary.spring_joint_count,
                summary.spring_collider_count,
                summary.spring_center_count
            );
            println!("    machine.parse=pass");
            println!("    machine.external_uri_gate=pass");
            println!("    machine.generation={:?}", summary.generation);
            println!("    machine.spec_version={}", summary.spec_version);
            println!(
                "    machine.exporter_version={:?}",
                summary.exporter_version
            );
            println!("    machine.name={:?}", summary.name);
            println!(
                "    machine.expression_preset_count={}",
                summary.expression_presets.len()
            );
            println!(
                "    machine.expression_presets={:?}",
                summary.expression_presets
            );
            println!("    machine.look_at_type={:?}", summary.look_at_type);
            println!("    machine.has_spring_bone={}", summary.has_spring_bone);
            println!(
                "    machine.has_node_constraint={}",
                summary.has_node_constraint
            );
            println!("    machine.has_first_person={}", summary.has_first_person);
            println!(
                "    machine.has_mtoon_materials={}",
                summary.has_mtoon_materials
            );
            println!(
                "    machine.required_humanoid_hips={}",
                summary.humanoid_nodes.hips
            );
            println!(
                "    machine.required_humanoid_head={}",
                summary.humanoid_nodes.head
            );
            println!(
                "    machine.optional_humanoid_neck={:?}",
                summary.humanoid_nodes.neck
            );
            println!(
                "    machine.material_mtoon_count={}",
                summary.mtoon_material_count
            );
            println!(
                "    machine.material_unlit_count={}",
                summary.unlit_material_count
            );
            println!(
                "    machine.material_fallback_count={}",
                summary.fallback_material_count
            );
            println!("    machine.spring_inventory_semantics=source_declared_inventory");
            println!(
                "    machine.spring_chain_count={}",
                summary.spring_chain_count
            );
            println!(
                "    machine.spring_joint_count={}",
                summary.spring_joint_count
            );
            println!(
                "    machine.spring_collider_count={}",
                summary.spring_collider_count
            );
            println!(
                "    machine.spring_center_count={}",
                summary.spring_center_count
            );
            let warning_codes = summary
                .compatibility_warnings
                .iter()
                .map(|warning| warning.code.as_str())
                .collect::<Vec<_>>();
            println!(
                "    machine.warning_count={}",
                summary.compatibility_warnings.len()
            );
            println!("    machine.warning_codes={warning_codes:?}");
        }
        Err(e) => {
            println!("  preflight: FAIL ({e})");
            println!("    machine.parse=fail");
            println!("    machine.external_uri_gate=not_evaluated");
            println!("    machine.warning_count=0");
            println!("    machine.warning_codes=[]");
        }
    }
    if let Some(report) = &result.runtime {
        println!("  runtime:");
        println!("    initialized: {}", report.initialized);
        println!("    generation: {:?}", report.generation);
        println!("    head: {}", report.has_head);
        println!("    neck: {}", report.has_neck);
        println!("    leftEye: {}", report.has_left_eye);
        println!("    rightEye: {}", report.has_right_eye);
        println!("    expressions: {:?}", report.expressions);
        println!("    spring roots: {}", report.spring_root_count);
        println!("    mvp capable: {}", report.is_mvp_capable());
        println!("    machine.initialize=pass");
        println!(
            "    machine.runtime_spring_root_count={}",
            report.spring_root_count
        );
        println!("    machine.initialized={}", report.initialized);
        println!("    machine.generation={:?}", report.generation);
        println!("    machine.head={}", report.has_head);
        println!("    machine.neck={}", report.has_neck);
        println!("    machine.left_eye={}", report.has_left_eye);
        println!("    machine.right_eye={}", report.has_right_eye);
        println!("    machine.expression_count={}", report.expressions.len());
        println!("    machine.expressions={:?}", report.expressions);
        println!("    machine.mvp_capable={}", report.is_mvp_capable());
        let warning_codes = report
            .warnings
            .iter()
            .map(|warning| warning.code.as_str())
            .collect::<Vec<_>>();
        println!("    machine.warning_count={}", report.warnings.len());
        println!("    machine.warning_codes={warning_codes:?}");
    } else {
        println!("  runtime: skipped");
        println!("    machine.initialize=not_run");
    }
    if let Some(error) = &result.runner_error {
        println!("  runner: FAIL ({error})");
    }
}
