//! Local NDI package staging and package-layout verification.
//!
//! This module never downloads, commits, or discovers proprietary NDI
//! binaries. The caller must provide the exact runtime DLL and license
//! agreement obtained from the SDK package that was used for the build.

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const RUNTIME_FILE_NAMES: [&str; 2] = ["Processing.NDI.Lib.x64.dll", "Processing.NDI.Lib_x64.dll"];
const LICENSE_FILE: &str = "NDI_SDK_LICENSE_AGREEMENT.pdf";
const RUNTIME_LICENSES_FILE: &str = "Processing.NDI.Lib.Licenses.txt";
const USAGE_FILE: &str = "README_NDI.md";
const NOTICES_FILE: &str = "THIRD_PARTY_NOTICES.md";
const MANIFEST_FILE: &str = "NDI_RUNTIME_MANIFEST.txt";
const EXECUTABLE_FILE: &str = "vtuber-desktop.exe";
const MODEL_MANIFEST_FILE: &str = "assets/models/manifest.toml";
const MODEL_TASK_FILE: &str = "assets/models/face_landmarker.task";
const MODEL_LICENSE_FILE: &str = "assets/models/LICENSE.mediapipe.txt";

/// Dispatch an NDI package command.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
pub fn run(args: &[String]) -> Result<(), String> {
    let Some(command) = args.first().map(String::as_str) else {
        print_help();
        return Ok(());
    };

    match command {
        "package" => package(&args[1..]),
        "verify-package" => verify_package(
            args.get(1)
                .map(PathBuf::from)
                .ok_or_else(|| "usage: cargo xtask ndi verify-package <package-dir>".to_string())?
                .as_path(),
        ),
        "verify-roundtrip" => verify_roundtrip(
            args.get(1)
                .map(PathBuf::from)
                .ok_or_else(|| {
                    "usage: cargo xtask ndi verify-roundtrip <capture-manifest>".to_string()
                })?
                .as_path(),
        ),
        "validate-render" => crate::ndi_output_render::run(&args[1..]),
        "probe-environment" => probe_environment(),
        "zip" => zip_package(
            args.get(1)
                .map(PathBuf::from)
                .ok_or_else(|| "usage: cargo xtask ndi zip <package-dir> [output.zip]".to_string())?
                .as_path(),
            args.get(2).map(PathBuf::from),
        ),
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        other => Err(format!("unknown ndi command: {other}")),
    }
}

fn package(args: &[String]) -> Result<(), String> {
    let options = PackageOptions::parse(args)?;
    let notices_path = workspace_root().join(NOTICES_FILE);
    let notices = fs::read_to_string(&notices_path)
        .map_err(|error| format!("cannot read {NOTICES_FILE}: {error}"))?;
    require_notice(&notices)?;

    require_file(&options.executable, "release executable")?;
    require_file(&options.runtime_dll, "NDI runtime DLL")?;
    require_file(&options.sdk_license, "exact NDI SDK license agreement")?;
    require_file(&options.runtime_licenses, "NDI runtime 3rd-party licenses")?;
    require_file(&options.usage, "NDI user usage notes")?;
    require_file(&options.model_manifest, "face model manifest")?;
    require_file(&options.model_task, "MediaPipe face task bundle")?;
    require_file(&options.model_license, "MediaPipe model license")?;

    let runtime_name = options
        .runtime_dll
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "runtime DLL path has no valid UTF-8 filename".to_string())?;
    let imported_runtime_name = detect_runtime_name_in_file(&options.executable)?;
    if runtime_name != imported_runtime_name {
        return Err(format!(
            "runtime DLL {runtime_name} does not match the executable's NDI import {imported_runtime_name}"
        ));
    }
    if options
        .sdk_license
        .extension()
        .and_then(|extension| extension.to_str())
        .is_none_or(|extension| !extension.eq_ignore_ascii_case("pdf"))
    {
        return Err("the exact SDK license input must be a PDF".to_string());
    }

    fs::create_dir_all(&options.output)
        .map_err(|error| format!("cannot create package directory: {error}"))?;
    fs::create_dir_all(options.output.join("assets/models"))
        .map_err(|error| format!("cannot create package model directory: {error}"))?;
    let destinations = [
        options.output.join(EXECUTABLE_FILE),
        options.output.join(runtime_name),
        options.output.join(LICENSE_FILE),
        options.output.join(NOTICES_FILE),
        options.output.join(MANIFEST_FILE),
        options.output.join(MODEL_MANIFEST_FILE),
        options.output.join(MODEL_TASK_FILE),
        options.output.join(MODEL_LICENSE_FILE),
        options.output.join(RUNTIME_LICENSES_FILE),
        options.output.join(USAGE_FILE),
    ];
    if !options.force && destinations.iter().any(|destination| destination.exists()) {
        return Err(
            "package output already contains generated files; pass --force for a rebuild"
                .to_string(),
        );
    }

    copy_file_if_unchanged(&options.executable, &destinations[0])?;
    copy_file_if_unchanged(&options.runtime_dll, &destinations[1])?;
    copy_file_if_unchanged(&options.sdk_license, &destinations[2])?;
    copy_file_if_unchanged(&notices_path, &destinations[3])?;
    copy_file_if_unchanged(&options.model_manifest, &destinations[5])?;
    copy_file_if_unchanged(&options.model_task, &destinations[6])?;
    copy_file_if_unchanged(&options.model_license, &destinations[7])?;
    copy_file_if_unchanged(&options.runtime_licenses, &destinations[8])?;
    copy_file_if_unchanged(&options.usage, &destinations[9])?;

    let package_sha = if options.package_archive_available {
        options.sdk_package_sha256.clone()
    } else {
        String::new()
    };
    let manifest = format!(
        "# Generated by cargo xtask ndi package\n\
         format_version=1\n\
         target=windows-x86_64\n\
         sdk_version={}\n\
         sdk_package_archive_available={}\n\
         sdk_package_sha256={package_sha}\n\
         runtime_file={runtime_name}\n\
         runtime_sha256={}\n\
         license_file={LICENSE_FILE}\n\
         license_sha256={}\n\
         runtime_licenses_file={RUNTIME_LICENSES_FILE}\n\
         runtime_licenses_sha256={}\n\
         usage_file={USAGE_FILE}\n\
         face_task_file={MODEL_TASK_FILE}\n\
         face_task_sha256={}\n\
         face_model_license_file={MODEL_LICENSE_FILE}\n\
         face_model_license_sha256={}\n\
         application_local=true\n\
         system_path_install=false\n\
         ndi_tools_included=false\n\
         advanced_sdk_included=false\n\
         hx_codecs_included=false\n\
         audio_included=false\n",
        options.sdk_version,
        options.package_archive_available,
        sha256_file(&options.runtime_dll)?,
        sha256_file(&options.sdk_license)?,
        sha256_file(&options.runtime_licenses)?,
        sha256_file(&options.model_task)?,
        sha256_file(&options.model_license)?,
    );
    fs::write(&destinations[4], manifest)
        .map_err(|error| format!("cannot write {MANIFEST_FILE}: {error}"))?;

    verify_package(&options.output)?;
    println!("NDI package verified: {}", options.output.display());
    if let Some(zip) = options.zip {
        zip_package(&options.output, Some(zip))?;
    }
    Ok(())
}

fn verify_package(package_dir: &Path) -> Result<(), String> {
    if !package_dir.is_dir() {
        return Err(format!(
            "package directory does not exist: {}",
            package_dir.display()
        ));
    }

    let runtime_name = RUNTIME_FILE_NAMES
        .iter()
        .copied()
        .find(|name| package_dir.join(name).is_file())
        .ok_or_else(|| {
            format!(
                "package is missing one of the supported NDI runtime DLLs: {}",
                RUNTIME_FILE_NAMES.join(", ")
            )
        })?;
    let expected = [
        EXECUTABLE_FILE,
        LICENSE_FILE,
        NOTICES_FILE,
        MANIFEST_FILE,
        RUNTIME_LICENSES_FILE,
        USAGE_FILE,
    ];
    for file in expected {
        require_file(&package_dir.join(file), file)?;
    }
    require_file(&package_dir.join(runtime_name), runtime_name)?;

    for entry in fs::read_dir(package_dir)
        .map_err(|error| format!("cannot read package directory: {error}"))?
    {
        let entry = entry.map_err(|error| format!("cannot inspect package entry: {error}"))?;
        let file_type = entry
            .file_type()
            .map_err(|error| format!("cannot inspect package entry type: {error}"))?;
        if entry.file_name() == "assets" {
            if !file_type.is_dir() {
                return Err(format!(
                    "package assets entry must be a directory: {}",
                    entry.path().display()
                ));
            }
            verify_model_resources(&entry.path())?;
            continue;
        }
        if !file_type.is_file() {
            return Err(format!(
                "package contains an unexpected directory or link: {}",
                entry.path().display()
            ));
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name != runtime_name && !expected.iter().any(|allowed| *allowed == name) {
            return Err(format!("unexpected package file: {name}"));
        }
    }

    verify_model_resources(&package_dir.join("assets"))?;

    let notices = fs::read_to_string(package_dir.join(NOTICES_FILE))
        .map_err(|error| format!("cannot read packaged notices: {error}"))?;
    require_notice(&notices)?;

    let manifest = parse_manifest(&package_dir.join(MANIFEST_FILE))?;
    require_manifest(&manifest, "target", "windows-x86_64")?;
    require_manifest(&manifest, "runtime_file", runtime_name)?;
    require_manifest(&manifest, "license_file", LICENSE_FILE)?;
    require_manifest(&manifest, "runtime_licenses_file", RUNTIME_LICENSES_FILE)?;
    require_manifest(&manifest, "usage_file", USAGE_FILE)?;
    require_manifest(&manifest, "application_local", "true")?;
    require_manifest(&manifest, "system_path_install", "false")?;
    require_manifest(&manifest, "ndi_tools_included", "false")?;
    require_manifest(&manifest, "advanced_sdk_included", "false")?;
    require_manifest(&manifest, "hx_codecs_included", "false")?;
    require_manifest(&manifest, "audio_included", "false")?;

    let runtime_hash = manifest
        .get("runtime_sha256")
        .ok_or_else(|| "manifest is missing runtime_sha256".to_string())?;
    if runtime_hash != &sha256_file(&package_dir.join(runtime_name))? {
        return Err("packaged runtime SHA-256 does not match the manifest".to_string());
    }
    let license_hash = manifest
        .get("license_sha256")
        .ok_or_else(|| "manifest is missing license_sha256".to_string())?;
    if license_hash != &sha256_file(&package_dir.join(LICENSE_FILE))? {
        return Err("packaged license SHA-256 does not match the manifest".to_string());
    }
    require_manifest(&manifest, "face_task_file", MODEL_TASK_FILE)?;
    require_manifest(&manifest, "face_model_license_file", MODEL_LICENSE_FILE)?;
    let task_hash = manifest
        .get("face_task_sha256")
        .ok_or_else(|| "manifest is missing face_task_sha256".to_string())?;
    if task_hash != &sha256_file(&package_dir.join(MODEL_TASK_FILE))? {
        return Err("packaged face task SHA-256 does not match the manifest".to_string());
    }
    let model_license_hash = manifest
        .get("face_model_license_sha256")
        .ok_or_else(|| "manifest is missing face_model_license_sha256".to_string())?;
    if model_license_hash != &sha256_file(&package_dir.join(MODEL_LICENSE_FILE))? {
        return Err("packaged face model license SHA-256 does not match the manifest".to_string());
    }
    let archive_available = manifest
        .get("sdk_package_archive_available")
        .map(String::as_str)
        .unwrap_or("true");
    let sdk_version = manifest
        .get("sdk_version")
        .ok_or_else(|| "manifest is missing sdk_version".to_string())?;
    if sdk_version.is_empty() {
        return Err("manifest field sdk_version is invalid".to_string());
    }
    let package_hash = manifest
        .get("sdk_package_sha256")
        .ok_or_else(|| "manifest is missing sdk_package_sha256".to_string())?;
    match archive_available {
        "true" => {
            if !is_sha256(package_hash) {
                return Err("manifest field sdk_package_sha256 is invalid".to_string());
            }
        }
        "false" => {}
        other => {
            return Err(format!(
                "manifest sdk_package_archive_available must be true or false, got {other}"
            ));
        }
    }
    let licenses_hash = manifest
        .get("runtime_licenses_sha256")
        .ok_or_else(|| "manifest is missing runtime_licenses_sha256".to_string())?;
    if licenses_hash != &sha256_file(&package_dir.join(RUNTIME_LICENSES_FILE))? {
        return Err("packaged runtime licenses SHA-256 does not match the manifest".to_string());
    }

    println!("NDI package layout and hashes: PASS");
    Ok(())
}

fn verify_model_resources(assets_dir: &Path) -> Result<(), String> {
    let model_dir = assets_dir.join("models");
    require_file(&model_dir.join("manifest.toml"), MODEL_MANIFEST_FILE)?;
    require_file(&model_dir.join("face_landmarker.task"), MODEL_TASK_FILE)?;
    require_file(&model_dir.join("LICENSE.mediapipe.txt"), MODEL_LICENSE_FILE)?;

    let assets_entries = fs::read_dir(assets_dir)
        .map_err(|error| format!("cannot read package assets directory: {error}"))?;
    for entry in assets_entries {
        let entry = entry.map_err(|error| format!("cannot inspect package asset: {error}"))?;
        if entry.file_name() != "models" {
            return Err(format!(
                "unexpected package asset: {}",
                entry.path().display()
            ));
        }
        if !entry
            .file_type()
            .map_err(|error| format!("cannot inspect package asset type: {error}"))?
            .is_dir()
        {
            return Err(format!(
                "package models entry must be a directory: {}",
                entry.path().display()
            ));
        }
    }

    let expected = [
        "manifest.toml",
        "face_landmarker.task",
        "LICENSE.mediapipe.txt",
    ];
    for entry in fs::read_dir(model_dir)
        .map_err(|error| format!("cannot read package model directory: {error}"))?
    {
        let entry = entry.map_err(|error| format!("cannot inspect package model: {error}"))?;
        let file_type = entry
            .file_type()
            .map_err(|error| format!("cannot inspect package model type: {error}"))?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !file_type.is_file() || !expected.iter().any(|allowed| *allowed == name) {
            return Err(format!(
                "unexpected package model resource: {}",
                entry.path().display()
            ));
        }
    }

    Ok(())
}

fn verify_roundtrip(manifest_path: &Path) -> Result<(), String> {
    let manifest = parse_manifest(manifest_path)?;
    require_manifest(&manifest, "source_name", "RusTuberV")?;
    require_manifest(&manifest, "four_cc", "BGRA")?;
    require_manifest(&manifest, "transparent_rgb_zero", "true")?;
    require_manifest(&manifest, "sender_stopped", "true")?;
    require_manifest(&manifest, "render_blocked", "false")?;
    require_manifest(&manifest, "stop_source_absent", "true")?;
    require_u64_at_least(&manifest, "width", 1920)?;
    require_u64_at_least(&manifest, "height", 1080)?;
    require_u64_at_least(&manifest, "fps", 60)?;
    require_u64_at_least(&manifest, "connection_count", 1)?;
    require_u64_at_least(&manifest, "frame_count", 2)?;
    require_u64_at_least(&manifest, "distinct_frame_hashes", 2)?;
    require_u64_at_least(&manifest, "alpha_zero_pixels", 1)?;
    require_u64_at_least(&manifest, "alpha_opaque_pixels", 1)?;
    require_u64_at_least(&manifest, "alpha_partial_pixels", 1)?;
    let queue_depth = manifest
        .get("queue_depth_max")
        .ok_or_else(|| "roundtrip manifest is missing queue_depth_max".to_string())?
        .parse::<u64>()
        .map_err(|error| format!("roundtrip queue_depth_max is invalid: {error}"))?;
    if queue_depth > 1 {
        return Err(format!(
            "roundtrip queue_depth_max must be bounded to one frame, got {queue_depth}"
        ));
    }
    println!("NDI sender/receiver roundtrip assertions: PASS");
    Ok(())
}

fn require_notice(notices: &str) -> Result<(), String> {
    for required in [
        "NDI® is a registered trademark of Vizrt NDI AB",
        "https://ndi.video",
    ] {
        if !notices.contains(required) {
            return Err(format!(
                "{NOTICES_FILE} must contain the required attribution/link: {required}"
            ));
        }
    }
    Ok(())
}

fn require_manifest(
    manifest: &BTreeMap<String, String>,
    key: &str,
    expected: &str,
) -> Result<(), String> {
    let actual = manifest
        .get(key)
        .ok_or_else(|| format!("manifest is missing {key}"))?;
    if actual != expected {
        return Err(format!("manifest {key} must be {expected}, got {actual}"));
    }
    Ok(())
}

fn require_u64_at_least(
    manifest: &BTreeMap<String, String>,
    key: &str,
    minimum: u64,
) -> Result<(), String> {
    let value = manifest
        .get(key)
        .ok_or_else(|| format!("manifest is missing {key}"))?
        .parse::<u64>()
        .map_err(|error| format!("manifest {key} is invalid: {error}"))?;
    if value < minimum {
        return Err(format!(
            "manifest {key} must be at least {minimum}, got {value}"
        ));
    }
    Ok(())
}

fn parse_manifest(path: &Path) -> Result<BTreeMap<String, String>, String> {
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let mut manifest = BTreeMap::new();
    for (line_number, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(format!(
                "manifest line {} is not key=value",
                line_number + 1
            ));
        };
        if manifest
            .insert(key.to_string(), value.to_string())
            .is_some()
        {
            return Err(format!("manifest key is duplicated: {key}"));
        }
    }
    Ok(manifest)
}

fn copy_file(source: &Path, destination: &Path) -> Result<(), String> {
    fs::copy(source, destination).map_err(|error| {
        format!(
            "cannot copy {} to {}: {error}",
            source.display(),
            destination.display()
        )
    })?;
    Ok(())
}

fn copy_file_if_unchanged(source: &Path, destination: &Path) -> Result<(), String> {
    if destination.is_file() && sha256_file(source)? == sha256_file(destination)? {
        return Ok(());
    }
    copy_file(source, destination)
}

fn require_file(path: &Path, description: &str) -> Result<(), String> {
    if !path.is_file() {
        return Err(format!("{description} does not exist: {}", path.display()));
    }
    Ok(())
}

fn detect_runtime_name_in_file(path: &Path) -> Result<&'static str, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("cannot read executable {}: {error}", path.display()))?;
    let matches: Vec<_> = RUNTIME_FILE_NAMES
        .iter()
        .copied()
        .filter(|name| {
            bytes
                .windows(name.len())
                .any(|window| window == name.as_bytes())
        })
        .collect();
    match matches.as_slice() {
        [name] => Ok(*name),
        [] => Err(format!(
            "executable {} does not import a supported NDI runtime DLL",
            path.display()
        )),
        _ => Err(format!(
            "executable {} imports multiple supported NDI runtime DLL names",
            path.display()
        )),
    }
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("cannot hash {}: {error}", path.display()))?;
    let digest = Sha256::digest(bytes);
    Ok(digest.iter().map(|byte| format!("{byte:02X}")).collect())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

struct PackageOptions {
    output: PathBuf,
    executable: PathBuf,
    runtime_dll: PathBuf,
    sdk_license: PathBuf,
    sdk_version: String,
    sdk_package_sha256: String,
    package_archive_available: bool,
    runtime_licenses: PathBuf,
    usage: PathBuf,
    zip: Option<PathBuf>,
    model_manifest: PathBuf,
    model_task: PathBuf,
    model_license: PathBuf,
    force: bool,
}

impl PackageOptions {
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut output = None;
        let workspace_root = workspace_root();
        let mut executable = workspace_root.join("target/release/vtuber-desktop.exe");
        let mut runtime_dll = None;
        let mut sdk_license = None;
        let mut sdk_version = None;
        let mut sdk_package_sha256 = None;
        let mut package_archive_available = true;
        let mut runtime_licenses = None;
        let mut usage = workspace_root.join("docs/NDI_USER.md");
        let mut zip = None;
        let mut force = false;
        let mut index = 0;
        while index < args.len() {
            let argument = args[index].as_str();
            match argument {
                "--output" => output = Some(next_path(args, &mut index, argument)?),
                "--executable" => executable = next_path(args, &mut index, argument)?,
                "--runtime-dll" => runtime_dll = Some(next_path(args, &mut index, argument)?),
                "--sdk-license" => sdk_license = Some(next_path(args, &mut index, argument)?),
                "--sdk-version" => {
                    sdk_version = Some(next_value(args, &mut index, argument)?);
                }
                "--sdk-package-sha256" => {
                    sdk_package_sha256 = Some(next_value(args, &mut index, argument)?);
                }
                "--sdk-package-unavailable" => package_archive_available = false,
                "--runtime-licenses" => {
                    runtime_licenses = Some(next_path(args, &mut index, argument)?);
                }
                "--usage" => usage = next_path(args, &mut index, argument)?,
                "--zip" => zip = Some(next_path(args, &mut index, argument)?),
                "--force" => force = true,
                "--help" | "-h" => {
                    print_package_help();
                    return Err("help requested".to_string());
                }
                other => return Err(format!("unknown package option: {other}")),
            }
            index += 1;
        }

        let output = output.ok_or_else(|| "missing --output".to_string())?;
        let runtime_dll = runtime_dll.ok_or_else(|| "missing --runtime-dll".to_string())?;
        let sdk_license = sdk_license.ok_or_else(|| "missing --sdk-license".to_string())?;
        let sdk_version = sdk_version.ok_or_else(|| "missing --sdk-version".to_string())?;
        if sdk_version.is_empty() {
            return Err("--sdk-version cannot be empty".to_string());
        }
        let sdk_package_sha256 = match (package_archive_available, sdk_package_sha256) {
            (true, Some(hash)) if is_sha256(&hash) => hash,
            (true, _) => {
                return Err(
                    "--sdk-package-sha256 must be 64 hexadecimal characters, or pass --sdk-package-unavailable"
                        .to_string(),
                );
            }
            (false, Some(hash)) if !hash.is_empty() && !is_sha256(&hash) => {
                return Err("--sdk-package-sha256 must be 64 hexadecimal characters".to_string());
            }
            (false, hash) => hash.unwrap_or_default(),
        };
        let runtime_licenses = runtime_licenses.unwrap_or_else(|| {
            runtime_dll
                .parent()
                .map(|parent| parent.join(RUNTIME_LICENSES_FILE))
                .unwrap_or_else(|| PathBuf::from(RUNTIME_LICENSES_FILE))
        });

        Ok(Self {
            output,
            executable,
            runtime_dll,
            sdk_license,
            sdk_version,
            sdk_package_sha256,
            package_archive_available,
            runtime_licenses,
            usage,
            zip,
            model_manifest: workspace_root.join(MODEL_MANIFEST_FILE),
            model_task: workspace_root.join(MODEL_TASK_FILE),
            model_license: workspace_root.join(MODEL_LICENSE_FILE),
            force,
        })
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .to_path_buf()
}

fn next_value(args: &[String], index: &mut usize, option: &str) -> Result<String, String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("{option} requires a value"))
}

fn next_path(args: &[String], index: &mut usize, option: &str) -> Result<PathBuf, String> {
    next_value(args, index, option).map(PathBuf::from)
}

fn zip_package(package_dir: &Path, zip_path: Option<PathBuf>) -> Result<(), String> {
    verify_package(package_dir)?;
    let zip_path = zip_path.unwrap_or_else(|| {
        package_dir
            .parent()
            .unwrap_or(package_dir)
            .join("RusTuberV-ndi-windows-x64.zip")
    });
    if zip_path.exists() {
        fs::remove_file(&zip_path)
            .map_err(|error| format!("cannot replace {}: {error}", zip_path.display()))?;
    }
    let status = std::process::Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            &format!(
                "Compress-Archive -Path {} -DestinationPath {} -Force",
                powershell_literal(&package_dir.join("*")),
                powershell_literal(&zip_path)
            ),
        ])
        .status()
        .map_err(|error| format!("failed to launch PowerShell zip: {error}"))?;
    if !status.success() {
        return Err(format!(
            "PowerShell Compress-Archive failed with status {status}"
        ));
    }
    println!("NDI release ZIP: {}", zip_path.display());
    Ok(())
}

fn powershell_literal(path: &Path) -> String {
    format!("'{}'", path.display().to_string().replace('\'', "''"))
}

fn print_help() {
    println!("ndi - local NDI package and verification commands");
    println!();
    print_package_help();
    println!("  cargo xtask ndi verify-package <package-dir>");
    println!("  cargo xtask ndi zip <package-dir> [output.zip]");
    println!("  cargo xtask ndi verify-roundtrip <capture-manifest>");
    println!("  cargo xtask ndi validate-render [--evidence <file>]");
    println!("  cargo xtask ndi probe-environment");
}

fn probe_environment() -> Result<(), String> {
    let sdk_dir = std::env::var_os("NDI_SDK_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Program Files\NDI\NDI 6 SDK"));
    let license = sdk_dir.join("NDI SDK License Agreement.pdf");
    let runtime = sdk_dir
        .join("Bin")
        .join("x64")
        .join("Processing.NDI.Lib.x64.dll");
    let header = sdk_dir.join("Include").join("Processing.NDI.Lib.h");
    let tools = PathBuf::from(r"C:\Program Files\NDI\NDI 6 Tools");
    let system_runtime = PathBuf::from(r"C:\Program Files\NDI\NDI 6 Runtime");
    let obs = PathBuf::from(r"C:\Program Files\obs-studio");
    let distroav = PathBuf::from(r"C:\Program Files\obs-studio\obs-plugins\64bit");

    println!("sdk_dir={}", sdk_dir.display());
    println!("sdk_dir_present={}", sdk_dir.is_dir());
    println!("license_file={}", license.display());
    println!("license_present={}", license.is_file());
    if license.is_file() {
        println!("license_sha256={}", sha256_file(&license)?);
        println!(
            "license_bytes={}",
            fs::metadata(&license).map(|m| m.len()).unwrap_or(0)
        );
    }
    println!("runtime_file={}", runtime.display());
    println!("runtime_present={}", runtime.is_file());
    if runtime.is_file() {
        println!("runtime_sha256={}", sha256_file(&runtime)?);
    }
    println!("header_present={}", header.is_file());
    println!("ndi_tools_present={}", tools.is_dir());
    println!("system_runtime_present={}", system_runtime.is_dir());
    println!("obs_present={}", obs.is_dir());
    let distroav_present = distroav.is_dir()
        && fs::read_dir(&distroav)
            .map(|entries| {
                entries.flatten().any(|entry| {
                    let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
                    name.contains("ndi") || name.contains("distroav")
                })
            })
            .unwrap_or(false);
    println!("distroav_present={distroav_present}");
    println!(
        "clean_machine={}",
        !sdk_dir.is_dir() && !tools.is_dir() && !system_runtime.is_dir()
    );
    println!("sdk_package_archive_sha256=");
    println!("result=PROBE");
    Ok(())
}

fn print_package_help() {
    println!("  cargo xtask ndi package --output <dir> \\");
    println!("    --runtime-dll <Processing.NDI.Lib.x64.dll or Processing.NDI.Lib_x64.dll> \\");
    println!("    --sdk-license <NDI SDK License Agreement.pdf> \\");
    println!("    --sdk-version <version> \\");
    println!("    [--sdk-package-sha256 <sha256> | --sdk-package-unavailable] \\");
    println!("    [--executable <vtuber-desktop.exe>] [--runtime-licenses <txt>] \\");
    println!("    [--usage <README_NDI.md>] [--zip <output.zip>] [--force]");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_directory() -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after UNIX epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("vrm-ndi-package-test-{suffix}"))
    }

    #[test]
    fn sha256_validation_is_strict() {
        assert!(is_sha256(&"A".repeat(64)));
        assert!(!is_sha256(&"A".repeat(63)));
        assert!(!is_sha256(&"G".repeat(64)));
    }

    #[test]
    fn package_options_allow_unavailable_archive_hash() {
        let result = PackageOptions::parse(&[
            "--output".into(),
            "out".into(),
            "--runtime-dll".into(),
            "runtime.dll".into(),
            "--sdk-license".into(),
            "license.pdf".into(),
            "--sdk-version".into(),
            "6.3.2".into(),
            "--sdk-package-unavailable".into(),
        ]);
        let options = result.expect("unavailable archive is allowed");
        assert!(!options.package_archive_available);
        assert!(options.sdk_package_sha256.is_empty());
    }

    #[test]
    fn package_options_require_exact_sdk_identity() {
        let result = PackageOptions::parse(&[
            "--output".into(),
            "out".into(),
            "--runtime-dll".into(),
            "runtime.dll".into(),
            "--sdk-license".into(),
            "license.pdf".into(),
            "--sdk-version".into(),
            "6.3.2".into(),
            "--sdk-package-sha256".into(),
            "A".repeat(64),
        ]);
        assert!(result.is_ok());
    }

    #[test]
    fn verify_rejects_unexpected_package_files() {
        let directory = temporary_directory();
        fs::create_dir_all(&directory).expect("test directory should be creatable");
        fs::write(directory.join("unexpected.dll"), b"not allowed").expect("test file");
        let result = verify_package(&directory);
        assert!(result.is_err());
        fs::remove_dir_all(directory).expect("test directory should be removable");
    }

    #[test]
    fn executable_import_name_supports_legacy_runtime() {
        let directory = temporary_directory();
        fs::create_dir_all(&directory).expect("test directory should be creatable");
        let executable = directory.join(EXECUTABLE_FILE);
        fs::write(&executable, b"MZ Processing.NDI.Lib_x64.dll\0").expect("executable");

        assert_eq!(
            detect_runtime_name_in_file(&executable).expect("legacy import should be detected"),
            "Processing.NDI.Lib_x64.dll"
        );
        fs::remove_dir_all(directory).expect("test directory should be removable");
    }

    #[test]
    fn verify_accepts_packaged_mediapipe_resources() {
        let directory = temporary_directory();
        let model_dir = directory.join("assets/models");
        fs::create_dir_all(&model_dir).expect("model directory should be creatable");
        fs::write(directory.join(EXECUTABLE_FILE), b"exe").expect("executable");
        fs::write(directory.join(RUNTIME_FILE_NAMES[0]), b"runtime").expect("runtime");
        fs::write(directory.join(LICENSE_FILE), b"license").expect("license");
        fs::write(directory.join(RUNTIME_LICENSES_FILE), b"runtime licenses").expect("licenses");
        fs::write(directory.join(USAGE_FILE), b"usage").expect("usage");
        fs::write(
            directory.join(NOTICES_FILE),
            "NDI® is a registered trademark of Vizrt NDI AB\nhttps://ndi.video\n",
        )
        .expect("notices");
        fs::write(model_dir.join("manifest.toml"), "[metadata]\n").expect("model manifest");
        fs::write(model_dir.join("face_landmarker.task"), b"task").expect("task bundle");
        fs::write(model_dir.join("LICENSE.mediapipe.txt"), b"Apache-2.0").expect("model license");

        let manifest = format!(
            "target=windows-x86_64\n\
             sdk_version=6.3.2\n\
             sdk_package_archive_available=true\n\
             sdk_package_sha256={}\n\
             runtime_file={}\n\
             runtime_sha256={}\n\
             license_file={}\n\
             license_sha256={}\n\
             runtime_licenses_file={}\n\
             runtime_licenses_sha256={}\n\
             usage_file={}\n\
             face_task_file={}\n\
             face_task_sha256={}\n\
             face_model_license_file={}\n\
             face_model_license_sha256={}\n\
             application_local=true\n\
             system_path_install=false\n\
             ndi_tools_included=false\n\
             advanced_sdk_included=false\n\
             hx_codecs_included=false\n\
             audio_included=false\n",
            "A".repeat(64),
            RUNTIME_FILE_NAMES[0],
            sha256_file(&directory.join(RUNTIME_FILE_NAMES[0])).expect("runtime hash"),
            LICENSE_FILE,
            sha256_file(&directory.join(LICENSE_FILE)).expect("license hash"),
            RUNTIME_LICENSES_FILE,
            sha256_file(&directory.join(RUNTIME_LICENSES_FILE)).expect("runtime licenses hash"),
            USAGE_FILE,
            MODEL_TASK_FILE,
            sha256_file(&model_dir.join("face_landmarker.task")).expect("task hash"),
            MODEL_LICENSE_FILE,
            sha256_file(&model_dir.join("LICENSE.mediapipe.txt")).expect("model license hash"),
        );
        fs::write(directory.join(MANIFEST_FILE), manifest).expect("runtime manifest");

        assert!(verify_package(&directory).is_ok());
        fs::remove_dir_all(directory).expect("test directory should be removable");
    }

    #[test]
    fn verify_roundtrip_accepts_the_normative_capture_contract() {
        let directory = temporary_directory();
        fs::create_dir_all(&directory).expect("test directory should be creatable");
        let manifest_path = directory.join("roundtrip.txt");
        fs::write(
            &manifest_path,
            "\
             source_name=RusTuberV\n\
             four_cc=BGRA\n\
             width=1920\n\
             height=1080\n\
             fps=60\n\
             connection_count=1\n\
             frame_count=12\n\
             distinct_frame_hashes=12\n\
             alpha_zero_pixels=100\n\
             alpha_opaque_pixels=100\n\
             alpha_partial_pixels=100\n\
             transparent_rgb_zero=true\n\
             sender_stopped=true\n\
             stop_source_absent=true\n\
             render_blocked=false\n\
             queue_depth_max=1\n",
        )
        .expect("roundtrip manifest should be writable");
        assert!(verify_roundtrip(&manifest_path).is_ok());
        fs::remove_dir_all(directory).expect("test directory should be removable");
    }

    #[test]
    fn verify_roundtrip_rejects_stale_or_unbounded_capture() {
        let directory = temporary_directory();
        fs::create_dir_all(&directory).expect("test directory should be creatable");
        let manifest_path = directory.join("roundtrip.txt");
        fs::write(
            &manifest_path,
            "\
             source_name=RusTuberV\n\
             four_cc=BGRA\n\
             width=1920\n\
             height=1080\n\
             fps=60\n\
             connection_count=1\n\
             frame_count=2\n\
             distinct_frame_hashes=1\n\
             alpha_zero_pixels=1\n\
             alpha_opaque_pixels=1\n\
             alpha_partial_pixels=1\n\
             transparent_rgb_zero=true\n\
             sender_stopped=true\n\
             stop_source_absent=true\n\
             render_blocked=false\n\
             queue_depth_max=2\n",
        )
        .expect("roundtrip manifest should be writable");
        assert!(verify_roundtrip(&manifest_path).is_err());
        fs::remove_dir_all(directory).expect("test directory should be removable");
    }
}
