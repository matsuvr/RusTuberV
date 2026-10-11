//! Optional diagnostic export; the application's normal UI queries Bridge itself.
#![forbid(unsafe_code)]

use std::{env, fs, path::PathBuf, process::ExitCode};

const DEFAULT_OUTPUT: &str = "run/looking-glass/device.json";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut output = None;
    let mut serial = None;
    let mut args = env::args_os().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" || arg == "-h" {
            println!("usage: cargo run -p looking-glass-device -- [output.json] [--serial SERIAL]");
            println!("Optional diagnostic export from running Looking Glass Bridge 2.2+.");
            println!("Normal application output retrieves calibration automatically.");
            println!("Default output: {DEFAULT_OUTPUT} (raw visual.json format).");
            return Ok(());
        } else if arg == "--serial" {
            serial = Some(
                args.next()
                    .ok_or("--serial requires the device serial number")?
                    .into_string()
                    .map_err(|_| "serial number must be valid Unicode")?,
            );
        } else if arg.to_string_lossy().starts_with('-') || output.is_some() {
            return Err(format!("unexpected argument: {}", arg.to_string_lossy()));
        } else {
            output = Some(PathBuf::from(arg));
        }
    }
    let output = output.unwrap_or_else(|| PathBuf::from(DEFAULT_OUTPUT));
    let devices = bevy_looking_glass::bridge::discover().map_err(|error| match error {
        bevy_looking_glass::bridge::Error::Unavailable => "Bridgeが見つかりません".into(),
        error => error.to_string(),
    })?;
    let available = devices
        .iter()
        .map(|device| device.serial.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let mut selected = devices.into_iter().filter(|device| {
        serial
            .as_ref()
            .is_none_or(|serial| device.serial == *serial)
    });
    let device = selected
        .next()
        .ok_or_else(|| format!("No matching Looking Glass. Available serials: {available}"))?;
    if selected.next().is_some() {
        return Err(format!(
            "Multiple Looking Glass units found; use --serial SERIAL. Available serials: {available}"
        ));
    }
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
    }
    fs::write(&output, device.calibration.as_bytes())
        .map_err(|error| format!("could not write {}: {error}", output.display()))?;
    println!("Exported {} to {}", device.serial, output.display());
    Ok(())
}
