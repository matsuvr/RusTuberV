//! Bounded performance metrics export for Windows acceptance runs.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use bevy::prelude::*;

use crate::diagnostics::DiagnosticsSnapshot;

/// Environment variable that enables the bounded CSV exporter.
pub const METRICS_CSV_ENV: &str = "VTUBER_METRICS_CSV";
/// Warm-up excluded from the measurement window.
pub const WARMUP_SECONDS: f64 = 10.0;
/// Required soak sampling cadence.
pub const SAMPLE_INTERVAL_SECONDS: f64 = 60.0;
/// Start plus 30 one-minute intervals, covering 0 through 1,800 seconds.
pub const MAX_SAMPLES: usize = 31;

const HEADER: &str = "sample,measurement_elapsed_s,render_fps,process_cpu_pct,process_memory_gib,capture_hz,inference_hz,detector_hz,landmark_hz,tracking_hz,capture_to_apply_p50_ms,capture_to_apply_p95_ms,slot_overwrites,inference_input_overwrites,no_face_frames,avatar_frames_applied,avatar_frames_skipped,capture_worker,inference_worker,tracking_state,stage_percentiles";

/// Runtime state for the opt-in bounded metrics exporter.
#[derive(Resource, Debug)]
pub struct MetricsExportState {
    output_path: Option<PathBuf>,
    phase: ExportPhase,
    file: Option<File>,
    active_started_at: Option<f64>,
    samples_written: usize,
}

#[derive(Debug)]
enum ExportPhase {
    Disabled,
    WaitingForTracking,
    WarmingUp,
    Recording,
    Complete,
    Failed(String),
}

impl Default for MetricsExportState {
    fn default() -> Self {
        let output_path = std::env::var_os(METRICS_CSV_ENV).map(PathBuf::from);
        let phase = if output_path.is_some() {
            ExportPhase::WaitingForTracking
        } else {
            ExportPhase::Disabled
        };
        Self {
            output_path,
            phase,
            file: None,
            active_started_at: None,
            samples_written: 0,
        }
    }
}

impl MetricsExportState {
    #[cfg(test)]
    fn for_path(path: PathBuf) -> Self {
        Self {
            output_path: Some(path),
            phase: ExportPhase::WaitingForTracking,
            file: None,
            active_started_at: None,
            samples_written: 0,
        }
    }

    fn tick(&mut self, now_seconds: f64, snapshot: &DiagnosticsSnapshot) {
        if matches!(
            self.phase,
            ExportPhase::Disabled | ExportPhase::Complete | ExportPhase::Failed(_)
        ) {
            return;
        }

        let active_started_at = match self.active_started_at {
            Some(started_at) => started_at,
            None if measurement_ready(snapshot) => {
                self.active_started_at = Some(now_seconds);
                self.phase = ExportPhase::WarmingUp;
                now_seconds
            }
            None => return,
        };

        let active_elapsed = (now_seconds - active_started_at).max(0.0);
        if active_elapsed < WARMUP_SECONDS {
            return;
        }

        let measurement_elapsed = active_elapsed - WARMUP_SECONDS;
        let next_due = self.samples_written as f64 * SAMPLE_INTERVAL_SECONDS;
        if measurement_elapsed < next_due {
            return;
        }

        self.phase = ExportPhase::Recording;
        if let Err(error) = self.write_sample(measurement_elapsed, snapshot) {
            self.file = None;
            self.phase = ExportPhase::Failed(error.to_string());
            return;
        }
        self.samples_written += 1;
        if self.samples_written >= MAX_SAMPLES {
            self.file = None;
            self.phase = ExportPhase::Complete;
        }
    }

    fn write_sample(
        &mut self,
        measurement_elapsed: f64,
        snapshot: &DiagnosticsSnapshot,
    ) -> io::Result<()> {
        if self.file.is_none() {
            let path = self
                .output_path
                .as_deref()
                .ok_or_else(|| io::Error::other("metrics output path is unavailable"))?;
            self.file = Some(create_output(path)?);
        }
        let file = self
            .file
            .as_mut()
            .ok_or_else(|| io::Error::other("metrics output file is unavailable"))?;
        writeln!(
            file,
            "{},{:.3},{:.3},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{},{},{},{},{},{},{},{},{},{},{}",
            self.samples_written,
            measurement_elapsed,
            snapshot.render_fps,
            option_number(snapshot.process_cpu_usage),
            option_number(snapshot.process_memory_gib),
            snapshot.capture_rate,
            snapshot.inference_rate,
            snapshot.detector_rate,
            snapshot.landmark_rate,
            snapshot.tracking_rate,
            option_number(snapshot.capture_to_apply_p50_ms),
            option_number(snapshot.capture_to_apply_p95_ms),
            snapshot.slot_overwrites,
            snapshot.inference_input_overwrites,
            snapshot.inference_no_face_frames,
            snapshot.avatar_frames_applied,
            snapshot.avatar_frames_skipped,
            csv_escape(&snapshot.capture_state),
            csv_escape(&snapshot.inference_state),
            csv_escape(&snapshot.tracking_state),
            csv_escape(&stage_percentiles(snapshot)),
        )?;
        file.flush()
    }

    fn status(&self) -> String {
        match &self.phase {
            ExportPhase::Disabled => "disabled".to_string(),
            ExportPhase::WaitingForTracking => "waiting for live tracking".to_string(),
            ExportPhase::WarmingUp => format!("warming up ({:.0}s)", WARMUP_SECONDS),
            ExportPhase::Recording => format!(
                "recording {}/{} at {:.0}s cadence",
                self.samples_written, MAX_SAMPLES, SAMPLE_INTERVAL_SECONDS
            ),
            ExportPhase::Complete => format!("complete ({MAX_SAMPLES} samples)"),
            ExportPhase::Failed(error) => format!("failed: {error}"),
        }
    }
}

/// Advances the exporter and mirrors its bounded state into diagnostics.
pub fn export_diagnostics_system(
    time: Res<Time<Real>>,
    mut snapshot: ResMut<DiagnosticsSnapshot>,
    mut exporter: ResMut<MetricsExportState>,
) {
    exporter.tick(time.elapsed_secs_f64(), &snapshot);
    snapshot.metrics_export_status = exporter.status();
    snapshot.metrics_export_samples = exporter.samples_written;
}

fn measurement_ready(snapshot: &DiagnosticsSnapshot) -> bool {
    snapshot.render_fps.is_finite()
        && snapshot.render_fps > 0.0
        && snapshot.capture_rate > 0.0
        && snapshot.inference_rate > 0.0
        && snapshot.tracking_rate > 0.0
        && snapshot.capture_to_apply_p95_ms.is_some_and(f32::is_finite)
}

fn create_output(path: &Path) -> io::Result<File> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    writeln!(file, "{HEADER}")?;
    file.flush()?;
    Ok(file)
}

fn option_number(value: Option<f32>) -> String {
    value
        .filter(|number| number.is_finite())
        .map(|number| format!("{number:.3}"))
        .unwrap_or_default()
}

fn stage_percentiles(snapshot: &DiagnosticsSnapshot) -> String {
    snapshot
        .stage_percentiles
        .iter()
        .map(|(name, p50, p95)| format!("{name}:{p50:.3}:{p95:.3}"))
        .collect::<Vec<_>>()
        .join("|")
}

fn csv_escape(value: &str) -> String {
    if value
        .chars()
        .any(|character| matches!(character, ',' | '"' | '\n' | '\r'))
    {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ready_snapshot() -> DiagnosticsSnapshot {
        DiagnosticsSnapshot {
            render_fps: 60.0,
            capture_rate: 30.0,
            inference_rate: 30.0,
            tracking_rate: 30.0,
            capture_to_apply_p95_ms: Some(48.0),
            ..Default::default()
        }
    }

    #[test]
    fn exporter_waits_for_tracking_then_excludes_warmup() {
        let temp = tempfile::tempdir().expect("temporary directory must be created");
        let path = temp.path().join("metrics.csv");
        let mut exporter = MetricsExportState::for_path(path.clone());
        let mut snapshot = ready_snapshot();
        snapshot.tracking_rate = 0.0;
        exporter.tick(100.0, &snapshot);
        assert_eq!(exporter.samples_written, 0);
        assert!(!path.exists());

        snapshot.tracking_rate = 30.0;
        exporter.tick(101.0, &snapshot);
        exporter.tick(110.9, &snapshot);
        assert_eq!(exporter.samples_written, 0);
        exporter.tick(111.0, &snapshot);
        assert_eq!(exporter.samples_written, 1);
    }

    #[test]
    fn exporter_is_fixed_cadence_and_bounded() {
        let temp = tempfile::tempdir().expect("temporary directory must be created");
        let path = temp.path().join("metrics.csv");
        let mut exporter = MetricsExportState::for_path(path.clone());
        let snapshot = ready_snapshot();
        exporter.tick(0.0, &snapshot);
        for sample in 0..(MAX_SAMPLES + 5) {
            exporter.tick(
                WARMUP_SECONDS + sample as f64 * SAMPLE_INTERVAL_SECONDS,
                &snapshot,
            );
        }
        assert_eq!(exporter.samples_written, MAX_SAMPLES);
        assert!(matches!(exporter.phase, ExportPhase::Complete));
        let output = fs::read_to_string(path).expect("metrics output must be readable");
        assert_eq!(output.lines().count(), MAX_SAMPLES + 1);
    }

    #[test]
    fn exporter_refuses_to_overwrite_existing_artifact() {
        let temp = tempfile::tempdir().expect("temporary directory must be created");
        let path = temp.path().join("metrics.csv");
        fs::write(&path, "existing").expect("test artifact must be created");
        let mut exporter = MetricsExportState::for_path(path.clone());
        exporter.tick(0.0, &ready_snapshot());
        exporter.tick(WARMUP_SECONDS, &ready_snapshot());
        assert!(matches!(exporter.phase, ExportPhase::Failed(_)));
        assert_eq!(fs::read_to_string(path).unwrap(), "existing");
    }
}
