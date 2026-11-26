//! Test-only reader for the Informal Trace Format written by Quint.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
use serde_json::Value;
use std::{env, fs, path::PathBuf};

pub fn traces(model: &str, mut replay: impl FnMut(&[Value])) {
    let dir =
        PathBuf::from(env::var_os("RUSTUBERV_QUINT_TRACES").expect("run tools/verify-quint.ps1"))
            .join(model);
    let mut files: Vec<_> = fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    files.sort();
    assert!(files.iter().any(|p| {
        p.file_name()
            .unwrap()
            .to_string_lossy()
            .ends_with("Test.json")
    }));
    assert!(files.iter().any(|p| {
        p.file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("walk-")
    }));
    for path in files {
        eprintln!("Replaying {}", path.display());
        let trace: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        let states = trace["states"].as_array().unwrap();
        assert!(
            states.len() >= 2,
            "expected an initial state and a transition"
        );
        replay(states);
    }
    fs::write(dir.join("replayed"), b"").unwrap();
}

pub fn number(value: &Value) -> i64 {
    value["#bigint"].as_str().unwrap().parse().unwrap()
}
