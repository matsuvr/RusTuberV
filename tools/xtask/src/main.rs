//! Repository automation entry point.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::env;
use std::path::{Path, PathBuf};
use std::process;

mod acceptance;
mod eye_closure;
mod face_image_probe;
mod face_pipeline_smoke;
mod mediapipe_face_smoke;
mod mediapipe_pose_probe;
mod ndi;
mod ndi_output_render;
mod task_result;
mod vrm_compatibility;
mod vrm_managed_compatibility;
mod vrm_render;

use task_result::{TaskError, TaskOutcome, TaskResult, completed, decode_cli_args, task_exit_code};

fn main() {
    let result = decode_cli_args(env::args_os().skip(1)).and_then(|args| run_task(&args));
    match &result {
        Ok(TaskOutcome::NotRun { reason, .. }) => eprintln!("NOT RUN: {reason}"),
        Err(error) => eprintln!("{error}"),
        Ok(TaskOutcome::Completed | TaskOutcome::Help) => {}
    }
    process::exit(task_exit_code(&result));
}

fn print_help() {
    println!("usage: cargo xtask <task>");
    println!("tasks:");
    println!("  vrm-compat [fixture-dir]   run bevy_vrm1 compatibility gate");
    println!("  vrm-managed-compat <path> run the managed user:// lifecycle gate");
    println!("  acceptance <command>     Windows acceptance test support");
    println!("  eye-closure <command>    Eye-closure data prep and threshold fitting");
    println!("  face-image-probe <path>  Legacy research UltraFace/Peppa probe");
    println!("  face-pipeline-smoke      Legacy research detector/crop/landmark probe");
    println!("  mediapipe-face-smoke     Windows MSMF MediaPipe Face Landmarker gate");
    println!("  mediapipe-pose-probe     Guided MediaPipe neutral-relative pose proof");
    println!("  vrm-render <path> <out>  Render the rich-look switching sequence");
    println!("  ndi <command>           Stage or verify a Windows NDI release package");
}

fn run_task(args: &[String]) -> TaskResult {
    let Some((task, args)) = args.split_first() else {
        print_help();
        return Ok(TaskOutcome::Help);
    };
    match task.as_str() {
        "vrm-compat" => {
            let fixture_dir = args
                .first()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("tests/fixtures/vrm"));
            let results = vrm_compatibility::run(&fixture_dir).map_err(|error| {
                TaskError::new(format!("compatibility runner failed: {error}"), 1)
            })?;
            let mut failed = 0;
            for result in &results {
                print_result(result);
                if result.runner_error.is_some()
                    || result.preflight.is_err()
                    || result
                        .runtime
                        .as_ref()
                        .is_some_and(|runtime| !runtime.is_mvp_capable())
                {
                    failed += 1;
                }
            }
            if failed > 0 {
                return Err(TaskError::new(
                    format!("{failed} fixture(s) failed the compatibility gate"),
                    vrm_compatibility::EXIT_COMPAT_FAIL,
                ));
            }
            Ok(TaskOutcome::Completed)
        }
        "vrm-managed-compat" => {
            let path = args.first().map(PathBuf::from).ok_or_else(|| {
                TaskError::new(
                    "usage: cargo xtask -- vrm-managed-compat <path-to-model.vrm>",
                    1,
                )
            })?;
            completed(
                vrm_managed_compatibility::run(&path)
                    .map_err(|error| format!("managed compatibility runner failed: {error}")),
            )
        }
        "acceptance" => handle_acceptance(args),
        "eye-closure" => completed(eye_closure::run(args)),
        "face-image-probe" => completed(face_image_probe::run(args)),
        "face-pipeline-smoke" => completed(face_pipeline_smoke::run(args)),
        "mediapipe-face-smoke" => completed(mediapipe_face_smoke::run(args)),
        "mediapipe-pose-probe" => completed(mediapipe_pose_probe::run(args)),
        "vrm-render" => vrm_render::run(args),
        "ndi" => ndi::run(args),
        other => Err(TaskError::new(format!("unknown task: {other}"), 1)),
    }
}

fn handle_acceptance(args: &[String]) -> TaskResult {
    let Some((command, args)) = args.split_first() else {
        acceptance::print_help();
        return Ok(TaskOutcome::Help);
    };
    match command.as_str() {
        "env" => {
            acceptance::print_env();
            Ok(TaskOutcome::Completed)
        }
        "new" => {
            let base_dir = args
                .first()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("docs/acceptance/runs"));
            let run_dir = acceptance::new_run(&base_dir)?;
            println!("Created acceptance run: {}", run_dir.display());
            Ok(TaskOutcome::Completed)
        }
        "verify" => {
            let manifest = args
                .first()
                .map(Path::new)
                .unwrap_or_else(|| Path::new("assets/models/manifest.toml"));
            acceptance::verify_models(manifest)
                .map_err(|error| TaskError::new(format!("verify failed: {error}"), 1))?;
            Ok(TaskOutcome::Completed)
        }
        "help" | "--help" | "-h" => {
            acceptance::print_help();
            Ok(TaskOutcome::Help)
        }
        other => {
            acceptance::print_help();
            Err(TaskError::new(
                format!("unknown acceptance command: {other}"),
                1,
            ))
        }
    }
}

#[cfg(test)]
mod cli_tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )] // tests may panic (AGENTS.md)
    use super::*;
    #[test]
    fn absent_missing_and_unknown_arguments_are_typed() {
        assert_eq!(run_task(&[]).unwrap(), TaskOutcome::Help);
        assert!(run_task(&["unknown".into()]).is_err());
        assert!(run_task(&["vrm-managed-compat".into()]).is_err());
        assert_eq!(handle_acceptance(&[]).unwrap(), TaskOutcome::Help);
        assert!(handle_acceptance(&["unknown".into()]).is_err());
        assert_eq!(
            ndi::run(&["package".into(), "--help".into()]).unwrap(),
            TaskOutcome::Help
        );
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
        println!(
            "    perfect sync: present={} effective={}",
            report.perfect_sync.present_count(),
            report.perfect_sync.effective_count()
        );
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
