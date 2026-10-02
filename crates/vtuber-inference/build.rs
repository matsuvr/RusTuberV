//! Generates the approved task identity and embedded bytes from the manifest.
use std::{env, fs, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?).join("../../assets/models");
    let manifest_path = root.join("manifest.toml");
    println!("cargo:rerun-if-changed={}", manifest_path.display());
    let manifest: toml::Value = toml::from_str(&fs::read_to_string(&manifest_path)?)?;
    let artifacts = manifest
        .get("artifacts")
        .and_then(toml::Value::as_array)
        .ok_or("missing artifacts")?;
    let mut variants = String::new();
    let mut files = String::new();
    let mut hashes = String::new();
    let mut embedded = String::new();
    for artifact in artifacts {
        let Some(task) = artifact.get("mediapipe_task").and_then(toml::Value::as_str) else {
            continue;
        };
        let file = artifact
            .get("file")
            .and_then(toml::Value::as_str)
            .ok_or("missing task file")?;
        let hash = artifact
            .get("sha256")
            .and_then(toml::Value::as_str)
            .ok_or("missing task SHA-256")?;
        variants.push_str(&format!("/// Approved {task} task.\n{task},\n"));
        files.push_str(&format!("Self::{task} => {file:?},\n"));
        hashes.push_str(&format!("Self::{task} => {hash:?},\n"));
        embedded.push_str(&format!(
            "Self::{task} => include_bytes!({:?}),\n",
            root.join(file)
        ));
    }
    let code = format!(
        "/// Approved production tasks, generated from the model manifest.\n#[derive(Clone, Copy, Debug, PartialEq, Eq)]\npub enum MediaPipeTask {{ {variants} }}\nimpl MediaPipeTask {{\n/// Packaged filename.\npub const fn file(self) -> &'static str {{ match self {{ {files} }} }}\n/// Approved SHA-256.\npub const fn sha256(self) -> &'static str {{ match self {{ {hashes} }} }}\n/// Approved task embedded for resource-free distribution.\npub fn embedded(self) -> &'static [u8] {{ match self {{ {embedded} }} }}\n}}\n"
    );
    fs::write(
        PathBuf::from(env::var("OUT_DIR")?).join("mediapipe_tasks.rs"),
        code,
    )?;
    Ok(())
}
