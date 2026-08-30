//! Source-aligned Direct/GNM temporal quality report composition (GNM #57.4).
//!
//! This module composes the pure kernels from [`crate::temporal_metrics`]
//! into one comparable report per scalar channel:
//!
//! - stationary RMS and first/second difference RMS ([`crate::temporal_noise_metrics`]),
//! - onset plus 10/50/90% crossings, rise time, peak attenuation, overshoot,
//!   and settling for commanded steps ([`crate::step_response_metrics`]) and
//!   releases (a step whose target is below its baseline),
//! - blink-like pulse peak preservation and timing
//!   ([`crate::pulse_response_metrics`]),
//! - measured end-to-end latency carried alongside the smoothness metrics so
//!   a result that merely became smooth by adding delay cannot hide.
//!
//! Everything here is deterministic over numeric traces; no camera, model, or
//! runtime access exists on this path.

use crate::ab_backend::FaceTrackingBackend;
use crate::temporal_metrics::{
    PulseResponseMetrics, PulseResponseSpec, StepResponseMetrics, StepResponseSpec,
    TemporalMetricError, TemporalNoiseMetrics, TemporalTrace, pulse_response_metrics,
    step_response_metrics, temporal_noise_metrics,
};

/// Response commands measured against one channel's trace.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalChannelSpecs {
    /// Commanded step upward from baseline to target.
    pub rise: Option<StepResponseSpec>,
    /// Commanded release downward (target below baseline).
    pub release: Option<StepResponseSpec>,
    /// Short pulse such as a blink.
    pub pulse: Option<PulseResponseSpec>,
}

/// One channel's stillness/responsiveness metrics plus its measured latency.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelTemporalQuality {
    /// Human-readable channel name used in report rows.
    pub channel_name: String,
    /// Stationary RMS, first/second difference RMS.
    pub noise: TemporalNoiseMetrics,
    /// Rise response metrics when a rise command was supplied.
    pub rise: Option<StepResponseMetrics>,
    /// Release response metrics when a release command was supplied.
    pub release: Option<StepResponseMetrics>,
    /// Pulse peak preservation metrics when a pulse command was supplied.
    pub pulse: Option<PulseResponseMetrics>,
}

/// One backend's per-channel temporal quality and overall latency.
#[derive(Clone, Debug, PartialEq)]
pub struct BackendTemporalQuality {
    /// Backend this row describes.
    pub backend: FaceTrackingBackend,
    /// Measured capture-to-publish end-to-end latency in milliseconds.
    ///
    /// Reported next to smoothness so added latency cannot masquerade as
    /// improved temporal quality.
    pub end_to_end_latency_ms: f64,
    /// Per-channel metric rows in input order.
    pub channels: Vec<ChannelTemporalQuality>,
}

/// Measures one channel's full temporal quality from its validated trace.
///
/// # Errors
///
/// Propagates typed failures from the underlying step and pulse kernels.
pub fn channel_temporal_quality(
    channel_name: &str,
    trace: &TemporalTrace,
    specs: TemporalChannelSpecs,
) -> Result<ChannelTemporalQuality, TemporalMetricError> {
    let noise = temporal_noise_metrics(trace);
    let rise = match specs.rise {
        Some(spec) => Some(step_response_metrics(trace, spec)?),
        None => None,
    };
    let release = match specs.release {
        Some(spec) => Some(step_response_metrics(trace, spec)?),
        None => None,
    };
    let pulse = match specs.pulse {
        Some(spec) => Some(pulse_response_metrics(trace, spec)?),
        None => None,
    };
    Ok(ChannelTemporalQuality {
        channel_name: channel_name.to_owned(),
        noise,
        rise,
        release,
        pulse,
    })
}

/// Builds one backend's report row from validated traces.
///
/// Traces must be given in the same order as [`BackendTemporalQuality`]
/// consumers expect; each entry pairs a name, its trace, and its commands.
///
/// # Errors
///
/// Propagates typed failures from [`channel_temporal_quality`].
pub fn backend_temporal_quality(
    backend: FaceTrackingBackend,
    end_to_end_latency_ms: f64,
    channels: &[(&str, TemporalTrace, TemporalChannelSpecs)],
) -> Result<BackendTemporalQuality, TemporalMetricError> {
    if !end_to_end_latency_ms.is_finite() || end_to_end_latency_ms < 0.0 {
        return Err(TemporalMetricError::InvalidResponseSpec(
            "end-to-end latency must be finite and non-negative",
        ));
    }
    let mut rows = Vec::with_capacity(channels.len());
    for (name, trace, specs) in channels {
        rows.push(channel_temporal_quality(name, trace, specs.clone())?);
    }
    Ok(BackendTemporalQuality {
        backend,
        end_to_end_latency_ms,
        channels: rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const DT_MICROS: u64 = 16_667;

    fn sample(index: u64, value: f64) -> crate::temporal_metrics::TemporalSample {
        crate::temporal_metrics::TemporalSample {
            timestamp_micros: index * DT_MICROS,
            value,
        }
    }

    fn trace(values: &[f64]) -> TemporalTrace {
        TemporalTrace::new(
            values
                .iter()
                .enumerate()
                .map(|(index, value)| sample(index as u64, *value))
                .collect(),
        )
        .expect("valid trace")
    }

    fn rise_spec(command_index: u64) -> StepResponseSpec {
        StepResponseSpec {
            command_micros: command_index * DT_MICROS,
            baseline: 0.0,
            target: 1.0,
            settling_tolerance_fraction: 0.1,
        }
    }

    #[test]
    fn static_fixture_reports_small_stationary_noise_without_commands() {
        // Constant signal with tiny sensor jitter.
        let values = [0.0_f64, 0.001, -0.001, 0.0005, -0.0005];
        let quality = channel_temporal_quality(
            "jawOpen",
            &trace(&values),
            TemporalChannelSpecs {
                rise: None,
                release: None,
                pulse: None,
            },
        )
        .expect("static fixture measures");
        assert!(quality.noise.stationary_rms < 0.001);
        assert!(quality.noise.first_difference_rms_per_second.is_some());
        assert!(quality.rise.is_none());
        assert!(quality.release.is_none());
        assert!(quality.pulse.is_none());
    }

    #[test]
    fn ramp_fixture_crosses_10_50_90_in_order_with_measured_rise_time() {
        // A linear ramp from frame 10 to frame 110 (1.5 seconds at 60 fps).
        let values = [0.0_f64; 120]
            .iter()
            .enumerate()
            .map(|(index, _)| {
                if index <= 10 {
                    0.0
                } else if index >= 110 {
                    1.0
                } else {
                    (index - 10) as f64 / 100.0
                }
            })
            .collect::<Vec<_>>();
        let quality = channel_temporal_quality(
            "mouthSmile",
            &trace(&values),
            TemporalChannelSpecs {
                rise: Some(rise_spec(10)),
                release: None,
                pulse: None,
            },
        )
        .expect("ramp fixture measures");
        let rise = quality.rise.expect("rise measured");
        let t10 = rise.t10_ms.expect("t10 reached");
        let t50 = rise.t50_ms.expect("t50 reached");
        let t90 = rise.t90_ms.expect("t90 reached");
        assert!(t10 < t50 && t50 < t90);
        // Ramp spans 100 frames ≈ 1666.7 ms; 10→90% spans 80 frames ≈ 1333 ms.
        assert_relative_eq!(rise.rise_time_10_90_ms.unwrap(), t90 - t10);
        assert!((1300.0..=1400.0).contains(&rise.rise_time_10_90_ms.unwrap()));
        assert_eq!(rise.overshoot, 0.0);
        assert!(rise.settling_time_ms.is_some());
    }

    #[test]
    fn release_fixture_reports_attenuation_and_settling_for_a_falling_step() {
        // Hold at 1.0 for 30 frames then drop to 0.0 within 10 frames.
        let values = [0.0_f64; 60]
            .iter()
            .enumerate()
            .map(|(index, _)| if index < 40 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let spec = StepResponseSpec {
            command_micros: 40 * DT_MICROS,
            baseline: 1.0,
            target: 0.0,
            settling_tolerance_fraction: 0.1,
        };
        let quality = channel_temporal_quality(
            "jawOpen",
            &trace(&values),
            TemporalChannelSpecs {
                rise: None,
                release: Some(spec),
                pulse: None,
            },
        )
        .expect("release fixture measures");
        let release = quality.release.expect("release measured");
        assert!(release.t90_ms.is_some());
        assert_eq!(release.peak_response_ratio, 1.0);
        assert_eq!(release.peak_attenuation, 0.0);
        assert!(release.settling_time_ms.is_some());
    }

    #[test]
    fn pulse_fixture_preserves_blink_peak_and_reports_timing_error() {
        // Blink-like pulse peaking at frame 20 for three frames.
        let mut values = vec![0.0_f64; 40];
        for (index, value) in values.iter_mut().enumerate().take(23).skip(18) {
            *value = match index {
                18 => 0.3,
                19 | 21 => 0.8,
                20 => 1.0,
                22 => 0.2,
                _ => 0.0,
            };
        }
        let spec = PulseResponseSpec {
            onset_micros: 17 * DT_MICROS,
            baseline: 0.0,
            target_peak: 1.0,
            expected_peak_micros: Some(20 * DT_MICROS),
        };
        let quality = channel_temporal_quality(
            "eyeBlinkLeft",
            &trace(&values),
            TemporalChannelSpecs {
                rise: None,
                release: None,
                pulse: Some(spec),
            },
        )
        .expect("pulse fixture measures");
        let pulse = quality.pulse.expect("pulse measured");
        assert_relative_eq!(pulse.peak_response_ratio, 1.0);
        assert_eq!(pulse.peak_attenuation, 0.0);
        assert_relative_eq!(
            pulse.observed_peak_delay_ms,
            3.0 * DT_MICROS as f64 / 1_000.0
        );
        assert_relative_eq!(pulse.peak_timing_error_ms.unwrap(), 0.0);
    }

    #[test]
    fn backend_row_carries_latency_and_channel_order() {
        let values = [0.0_f64, 0.0, 0.0];
        let specs = TemporalChannelSpecs {
            rise: None,
            release: None,
            pulse: None,
        };
        let row = backend_temporal_quality(
            FaceTrackingBackend::DirectMediaPipe,
            42.5,
            &[("jawOpen", trace(&values).clone(), specs.clone())],
        )
        .expect("backend row builds");
        assert_eq!(row.backend, FaceTrackingBackend::DirectMediaPipe);
        assert_relative_eq!(row.end_to_end_latency_ms, 42.5);
        assert_eq!(row.channels.len(), 1);
        assert_eq!(row.channels[0].channel_name, "jawOpen");

        assert!(backend_temporal_quality(FaceTrackingBackend::GnmTemporal, -1.0, &[]).is_err());
        assert!(backend_temporal_quality(FaceTrackingBackend::GnmTemporal, f64::NAN, &[]).is_err());
    }
}
