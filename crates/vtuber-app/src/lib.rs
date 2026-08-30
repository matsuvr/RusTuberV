// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]
//! `vtuber-app`: orchestration, UI, settings, model import, and diagnostics.
//!
//! This crate must not contain model-specific inference math or VRM runtime internals.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// UI action commands emitted by the UI layer.
pub mod actions;

/// Bridge from tracking control frames to the active avatar generation.
pub mod avatar_bridge;

/// Capture runtime — camera backend lifecycle and frame transport.
pub mod capture_runtime;

/// Diagnostics snapshot for the UI.
pub mod diagnostics;

/// Face tracking backend selection, persistence, and authority display.
pub mod face_backend;

/// Bounded release-run metrics export for performance acceptance.
pub mod metrics_export;

/// Error presenter — maps domain errors to user-facing messages.
pub mod error_presenter;

/// VRM 0.x/1.0 import and preflight inspection.
pub mod import;

/// Application bridge for the pure-Rust face inference worker.
pub mod inference_runtime;

/// Manifest-driven inference model catalog.
pub mod model_catalog;

/// App orchestrator — processes UI actions and manages domain state.
pub mod orchestrator;

/// Optional NDI output orchestration and UI snapshot bridge.
pub mod ndi_output;

/// User-owned persistent settings, including per-model arm-pose overrides.
pub mod settings;

/// Placeholder for app subsystem.
pub mod placeholder;

/// Camera preview texture pipeline.
pub mod preview;

/// Pure low-resolution conversion for the privacy camera preview.
pub mod privacy_preview;

/// Display-only latest snapshot of canonical face landmarks.
pub mod preview_landmarks;

/// Dev-only synthetic tracking source (feature-gated).
#[cfg(feature = "dev-synthetic-input")]
pub mod synthetic_tracking;

/// UI rendering module using bevy_egui.
pub mod ui;

/// UI view models — immutable snapshots for rendering the UI.
pub mod ui_model;

/// Main-thread bridge from inference observations to tracking state.
pub mod tracking_runtime;
