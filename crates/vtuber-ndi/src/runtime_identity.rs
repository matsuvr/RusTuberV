//! Dependency-free identity of the supported Windows Standard NDI runtime.
use std::path::{Path, PathBuf};

/// Currently supported SDK runtime names, in discovery order.
pub const RUNTIME_FILE_NAMES: [&str; 2] =
    ["Processing.NDI.Lib.x64.dll", "Processing.NDI.Lib_x64.dll"];

/// Identifies the one supported runtime name embedded in an import library or executable.
pub fn runtime_name_in_binary(bytes: &[u8]) -> Result<&'static str, &'static str> {
    let mut names = RUNTIME_FILE_NAMES.into_iter().filter(|name| {
        bytes
            .windows(name.len())
            .any(|window| window == name.as_bytes())
    });
    match (names.next(), names.next()) {
        (Some(name), None) => Ok(name),
        (None, _) => Err("does not name a supported NDI runtime DLL"),
        _ => Err("names multiple supported NDI runtime DLLs"),
    }
}

/// Finds a supported runtime file in one caller-selected directory; never loads it.
pub fn runtime_file_in(dir: &Path) -> Option<(&'static str, PathBuf)> {
    RUNTIME_FILE_NAMES
        .into_iter()
        .find(|name| dir.join(name).is_file())
        .map(|name| (name, dir.join(name)))
}

#[cfg(test)]
mod tests {
    use super::runtime_name_in_binary;
    #[test]
    fn detects_current_standard_sdk_runtime_name() {
        assert_eq!(
            runtime_name_in_binary(b"Processing.NDI.Lib.x64.dll\0"),
            Ok("Processing.NDI.Lib.x64.dll")
        );
    }
    #[test]
    fn detects_legacy_standard_sdk_runtime_name() {
        assert_eq!(
            runtime_name_in_binary(b"Processing.NDI.Lib_x64.dll\0"),
            Ok("Processing.NDI.Lib_x64.dll")
        );
    }
    #[test]
    fn rejects_unrelated_or_ambiguous_runtime_name() {
        assert!(runtime_name_in_binary(b"Processing.NDI.Lib.UWP.x64.dll\0").is_err());
        assert!(
            runtime_name_in_binary(b"Processing.NDI.Lib.x64.dll\0Processing.NDI.Lib_x64.dll\0")
                .is_err()
        );
    }
}
