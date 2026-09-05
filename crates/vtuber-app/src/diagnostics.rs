//! Diagnostics snapshot for the UI.
//!
//! Collects performance metrics, worker state, and model info for display
//! in the Diagnostics screen.

use bevy::diagnostic::{
    DiagnosticsStore, FrameTimeDiagnosticsPlugin, SystemInformationDiagnosticsPlugin,
};
use bevy::prelude::*;

/// Diagnostics snapshot resource.
#[derive(Resource, Debug, Default, Clone)]
pub struct DiagnosticsSnapshot {
    /// Render FPS.
    pub render_fps: f32,
    /// Current process CPU usage in percent.
    pub process_cpu_usage: Option<f32>,
    /// Current process resident memory in GiB.
    pub process_memory_gib: Option<f32>,
    /// Capture frame rate.
    pub capture_rate: f32,
    /// Inference rate.
    pub inference_rate: f32,
    /// Full-frame detector rate. This is intentionally separate from the
    /// landmark/tracking rate because detector cadence is lower by design.
    pub detector_rate: f32,
    /// Crop landmark rate.
    pub landmark_rate: f32,
    /// Tracking output rate, measured from unique source sequences.
    pub tracking_rate: f32,
    /// Current capture worker state.
    pub capture_state: String,
    /// Current inference worker state.
    pub inference_state: String,
    /// Last source sequence processed by inference.
    pub inference_last_source_seq: Option<u64>,
    /// Number of frames processed by inference.
    pub inference_frames_processed: u64,
    /// Number of frames that completed with no detected face.
    pub inference_no_face_frames: u64,
    /// Number of duplicate inference frames suppressed.
    pub inference_duplicates_suppressed: u64,
    /// Number of input-slot overwrites observed by inference.
    pub inference_input_overwrites: u64,
    /// Last inference duration in milliseconds.
    pub last_inference_ms: Option<f32>,
    /// Last source-image ROI reported by the composite runtime.
    pub inference_last_roi: Option<(f32, f32, f32, f32)>,
    /// Detector confidence for the active ROI.
    pub detector_confidence: Option<f32>,
    /// Composite ROI lifecycle state.
    pub roi_state: Option<String>,
    /// Stable manifest pipeline identifier.
    pub pipeline_id: Option<String>,
    /// Stable worker stage for the last failure, if any.
    pub inference_failure_stage: Option<String>,
    /// Tracking state description.
    pub tracking_state: String,
    /// Inference/tracking backend identity.
    pub tracking_backend: Option<String>,
    /// Canonical face-output contract summary.
    pub tracking_contract: Option<String>,
    /// Auto-neutral state exposed to diagnostics.
    pub auto_neutral_state: Option<String>,
    /// Whether an auto-neutral calibration reference is ready for tracking.
    pub face_tracking_calibration_ready: Option<bool>,
    /// Latest backend residual reported by the active tracker, if any.
    pub face_tracking_latest_residual: Option<f32>,
    /// Additional end-to-end latency added by the non-Direct backend, if any.
    pub face_tracking_added_latency_ms: Option<f32>,
    /// Slot overwrite count.
    pub slot_overwrites: u64,
    /// Stage timings (name, duration_ms).
    pub stage_timings: Vec<(String, f32)>,
    /// Bounded stage timing percentiles (name, p50_ms, p95_ms).
    pub stage_percentiles: Vec<(String, f32, f32)>,
    /// Model hash (short).
    pub model_hash: Option<String>,
    /// Camera backend name.
    pub camera_backend: Option<String>,
    /// Avatar capability summary.
    pub avatar_capabilities: Option<String>,
    /// Number of avatar pose frames successfully applied.
    pub avatar_frames_applied: u64,
    /// Number of avatar pose frames skipped because the lifecycle/binding was
    /// not ready.
    pub avatar_frames_skipped: u64,
    /// p50 capture-to-avatar-apply latency in milliseconds.
    pub capture_to_apply_p50_ms: Option<f32>,
    /// p95 capture-to-avatar-apply latency in milliseconds.
    pub capture_to_apply_p95_ms: Option<f32>,
    /// Last stable error code, if any.
    pub last_error_code: Option<String>,
    /// Last error message, if any.
    pub last_error: Option<String>,
    /// Bounded metrics exporter state.
    pub metrics_export_status: String,
    /// Number of bounded metrics rows written by the current process.
    pub metrics_export_samples: usize,
}

impl DiagnosticsSnapshot {
    /// Check if any workers are currently active.
    #[must_use]
    pub fn has_active_workers(&self) -> bool {
        self.capture_rate > 0.0
            || self.inference_rate > 0.0
            || (!self.capture_state.is_empty()
                && self.capture_state != "Idle"
                && self.capture_state != "Selected")
            || (!self.inference_state.is_empty()
                && self.inference_state != "Idle"
                && self.inference_state != "Stopping")
    }
}

/// Synchronises Bevy's engine and process diagnostics into the UI snapshot.
pub fn sync_engine_diagnostics(
    store: Option<Res<DiagnosticsStore>>,
    mut snapshot: ResMut<DiagnosticsSnapshot>,
) {
    let Some(store) = store else {
        return;
    };
    snapshot.render_fps =
        diagnostic_value(&store, &FrameTimeDiagnosticsPlugin::FPS).unwrap_or(0.0) as f32;
    snapshot.process_cpu_usage = diagnostic_value(
        &store,
        &SystemInformationDiagnosticsPlugin::PROCESS_CPU_USAGE,
    )
    .map(|value| value as f32);
    snapshot.process_memory_gib = diagnostic_value(
        &store,
        &SystemInformationDiagnosticsPlugin::PROCESS_MEM_USAGE,
    )
    .map(|value| value as f32);
}

fn diagnostic_value(
    store: &DiagnosticsStore,
    path: &bevy::diagnostic::DiagnosticPath,
) -> Option<f64> {
    let diagnostic = store.get(path)?;
    diagnostic.smoothed().or_else(|| diagnostic.value())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostics_default_has_no_workers() {
        let snap = DiagnosticsSnapshot::default();
        assert!(!snap.has_active_workers());
    }

    #[test]
    fn diagnostics_with_capture_rate_has_workers() {
        let snap = DiagnosticsSnapshot {
            capture_rate: 30.0,
            ..Default::default()
        };
        assert!(snap.has_active_workers());
    }
}
