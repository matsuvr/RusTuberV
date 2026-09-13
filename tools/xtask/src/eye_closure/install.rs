//! Installing and removing a validated eye-closure profile (Issue #53).
//!
//! Only a `Verified` profile with fingerprints matching the current inference
//! contract can be installed. Editing a candidate's status by hand does not
//! bypass this check.

use std::path::{Path, PathBuf};

use vtuber_inference::backend::mediapipe::TASK_BUNDLE_SHA256;
use vtuber_tracking::{
    EYE_CLOSURE_FEATURE, EyeClosureProfileDocument, EyeClosureVerificationStatus,
};

use super::Options;
use super::fit::read_profile;

/// Runs `eye-closure install-profile`.
pub(crate) fn run_install(options: &Options) -> Result<(), String> {
    let profile_path = options
        .profile
        .as_deref()
        .ok_or("missing required option --profile")?;
    let document = read_profile(profile_path)?;
    ensure_installable(&document, profile_path)?;
    let destination = destination(options)?;
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    std::fs::copy(profile_path, &destination).map_err(|error| {
        format!(
            "failed to copy {} to {}: {error}",
            profile_path.display(),
            destination.display()
        )
    })?;
    println!("installed eye-closure profile at {}", destination.display());
    Ok(())
}

/// Runs `eye-closure remove-profile`.
pub(crate) fn run_remove(options: &Options) -> Result<(), String> {
    let destination = destination(options)?;
    if !destination.is_file() {
        println!("no eye-closure profile at {}", destination.display());
        return Ok(());
    }
    std::fs::remove_file(&destination)
        .map_err(|error| format!("failed to remove {}: {error}", destination.display()))?;
    println!("removed eye-closure profile at {}", destination.display());
    Ok(())
}

fn destination(options: &Options) -> Result<PathBuf, String> {
    if let Some(directory) = &options.config_dir {
        return Ok(directory.join(vtuber_app::settings::EYE_CLOSURE_PROFILE_FILE_NAME));
    }
    vtuber_app::settings::default_eye_closure_profile_path()
        .ok_or_else(|| "no per-user config directory is available; pass --config-dir".to_owned())
}

fn ensure_installable(document: &EyeClosureProfileDocument, path: &Path) -> Result<(), String> {
    if document.status != EyeClosureVerificationStatus::Verified {
        return Err(format!(
            "{}: only a verified profile can be installed, found {:?}",
            path.display(),
            document.status
        ));
    }
    if document.fingerprints.feature != EYE_CLOSURE_FEATURE
        || !document
            .fingerprints
            .task_bundle_sha256
            .as_deref()
            .is_some_and(|hash| hash.eq_ignore_ascii_case(TASK_BUNDLE_SHA256))
    {
        return Err(format!(
            "{}: inference fingerprints do not match the current runtime",
            path.display()
        ));
    }
    document
        .validate()
        .map_err(|error| format!("{}: {error}", path.display()))?;
    Ok(())
}
