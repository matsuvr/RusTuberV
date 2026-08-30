#![allow(missing_docs)]
// Build scripts report configuration errors by panicking; Cargo turns the
// panic into a build failure message. This is not production runtime code.
#![allow(clippy::panic)]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const RUNTIME_FILE_NAMES: [&str; 2] = ["Processing.NDI.Lib.x64.dll", "Processing.NDI.Lib_x64.dll"];

fn main() {
    println!("cargo:rerun-if-env-changed=NDI_SDK_DIR");
    println!("cargo:rerun-if-env-changed=NDI_RUNTIME_DLL");

    if env::var_os("CARGO_FEATURE_NDI_OUTPUT").is_none()
        || env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows")
    {
        return;
    }

    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("x86_64") {
        panic!("ndi-output requires the supported Windows x86_64 target");
    }

    if let Err(error) = stage_runtime_dll() {
        panic!("cannot prepare the NDI-enabled desktop artifact: {error}");
    }
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
    let runtime_name = detect_runtime_name(&import_library_bytes).ok_or_else(|| {
        format!(
            "NDI x64 import library {} does not name a supported runtime DLL",
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
        "cargo:warning=staged {runtime_name} beside vtuber-desktop.exe in {}",
        destination.parent().map_or_else(
            || destination.display().to_string(),
            |path| path.display().to_string()
        )
    );
    println!("cargo:rerun-if-changed={}", import_library.display());
    println!("cargo:rerun-if-changed={}", runtime_source.display());
    Ok(())
}

fn detect_runtime_name(import_library: &[u8]) -> Option<&'static str> {
    RUNTIME_FILE_NAMES.iter().copied().find(|name| {
        import_library
            .windows(name.len())
            .any(|window| window == name.as_bytes())
    })
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

#[cfg(test)]
mod tests {
    use super::detect_runtime_name;

    #[test]
    fn detects_current_standard_sdk_runtime_name() {
        assert_eq!(
            detect_runtime_name(b"Processing.NDI.Lib.x64.dll\0"),
            Some("Processing.NDI.Lib.x64.dll")
        );
    }

    #[test]
    fn detects_legacy_standard_sdk_runtime_name() {
        assert_eq!(
            detect_runtime_name(b"Processing.NDI.Lib_x64.dll\0"),
            Some("Processing.NDI.Lib_x64.dll")
        );
    }

    #[test]
    fn rejects_unrelated_runtime_name() {
        assert_eq!(
            detect_runtime_name(b"Processing.NDI.Lib.UWP.x64.dll\0"),
            None
        );
    }
}
