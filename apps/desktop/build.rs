#![allow(missing_docs)]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

// The same dependency-free source is used by runtime and packaging. No SDK dependency here.
#[path = "../../crates/vtuber-ndi/src/runtime_identity.rs"]
#[allow(
    dead_code,
    reason = "the build script uses binary identity; directory discovery is used by runtime and tools, and NDI calls are feature-gated"
)]
mod runtime_identity;
use runtime_identity::runtime_name_in_binary;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    embed_app_icon()?;

    println!("cargo:rerun-if-changed=../../crates/vtuber-ndi/src/runtime_identity.rs");
    println!("cargo:rerun-if-env-changed=NDI_SDK_DIR");
    println!("cargo:rerun-if-env-changed=NDI_RUNTIME_DLL");

    // The macOS SDK uses @rpath/libndi.dylib. Local builds load the installed
    // SDK runtime, so both cargo run and the .app work without DYLD overrides.
    if env::var_os("CARGO_FEATURE_NDI_OUTPUT").is_some()
        && env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
    {
        let sdk = env::var_os("NDI_SDK_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/Library/NDI SDK for Apple"));
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}",
            sdk.join("lib/macOS").display()
        );
    }

    if env::var_os("CARGO_FEATURE_NDI_OUTPUT").is_none()
        || env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows")
    {
        return Ok(());
    }

    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("x86_64") {
        return Err("ndi-output requires the supported Windows x86_64 target".into());
    }

    stage_runtime_dll()?;
    Ok(())
}

fn embed_app_icon() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let icon_path = manifest_dir.join("../../assets/icons/rustuberv-app.ico");
    println!("cargo:rerun-if-changed={}", icon_path.display());

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return Ok(());
    }

    if !icon_path.is_file() {
        return Err(format!("app icon not found: {}", icon_path.display()).into());
    }

    let icon_str = icon_path
        .to_str()
        .ok_or_else(|| format!("app icon path is not valid UTF-8: {}", icon_path.display()))?;

    let mut resource = winres::WindowsResource::new();
    resource.set_icon(icon_str);
    resource.compile()?;
    if env::var("CARGO_CFG_TARGET_ABI").as_deref() == Ok("llvm") {
        // LLD does not extract the symbol-free resource object from winres's
        // archive. Link it directly so the icon and version data survive.
        let object = PathBuf::from(env::var("OUT_DIR")?).join("resource.o");
        println!("cargo:rustc-link-arg={}", object.display());
    }
    Ok(())
}

fn stage_runtime_dll() -> Result<(), String> {
    let sdk_dir = env::var_os("NDI_SDK_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Program Files\NDI\NDI 6 SDK"));
    let import_library = sdk_dir
        .join("Lib")
        .join("x64")
        .join("Processing.NDI.Lib.x64.lib");
    require_file(&import_library, "NDI x64 import library")?;

    let import_library_bytes = fs::read(&import_library).map_err(|error| {
        format!(
            "cannot read NDI x64 import library {}: {error}",
            import_library.display()
        )
    })?;
    let runtime_name = runtime_name_in_binary(&import_library_bytes).map_err(|error| {
        format!(
            "NDI x64 import library {} {error}",
            import_library.display()
        )
    })?;

    let runtime_source = env::var_os("NDI_RUNTIME_DLL")
        .map(PathBuf::from)
        .unwrap_or_else(|| sdk_dir.join("Bin").join("x64").join(runtime_name));
    require_file(&runtime_source, "NDI runtime DLL")?;

    let destination = target_profile_directory()?.join(runtime_name);
    if !same_file(&runtime_source, &destination)? {
        fs::copy(&runtime_source, &destination).map_err(|error| {
            format!(
                "cannot copy NDI runtime {} to {}: {error}",
                runtime_source.display(),
                destination.display()
            )
        })?;
    }

    println!(
        "cargo:warning=staged {runtime_name} beside RusTuberV.exe in {}",
        destination.parent().map_or_else(
            || destination.display().to_string(),
            |path| path.display().to_string()
        )
    );
    println!("cargo:rerun-if-changed={}", import_library.display());
    println!("cargo:rerun-if-changed={}", runtime_source.display());
    Ok(())
}

fn target_profile_directory() -> Result<PathBuf, String> {
    let out_dir = PathBuf::from(
        env::var_os("OUT_DIR").ok_or_else(|| "OUT_DIR is not set by Cargo".to_string())?,
    );
    out_dir
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| format!("cannot derive the Cargo target profile directory from {out_dir:?}"))
}

fn same_file(left: &Path, right: &Path) -> Result<bool, String> {
    if !right.exists() {
        return Ok(false);
    }
    let left = fs::canonicalize(left)
        .map_err(|error| format!("cannot resolve {}: {error}", left.display()))?;
    let right = fs::canonicalize(right)
        .map_err(|error| format!("cannot resolve {}: {error}", right.display()))?;
    Ok(left == right)
}

fn require_file(path: &Path, description: &str) -> Result<(), String> {
    if !path.is_file() {
        return Err(format!("{description} does not exist: {}", path.display()));
    }
    Ok(())
}
