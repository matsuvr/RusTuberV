//! `vtuber-app`: Bevy orchestration, UI, settings, model import, and diagnostics.
//!
//! This crate must not contain model-specific inference math or VRM runtime internals.
//!
//! # Application flow
//!
//! [`AppRuntimePlugin`] installs shared resources and worker bridges.
//! [`ui::UiShellPlugin`] installs egui rendering and input; the desktop owns
//! execution schedules.
//!
//! - [`actions::UiAction`] carries UI intent to [`orchestrator::process_ui_actions_system`].
//! - [`orchestrator::Orchestrator`] owns application state and pending requests.
//!   Its private modules handle action effects, expression bindings, and avatar loads.
//! - The capture, inference, Pose, and tracking bridges connect domain workers to
//!   Bevy resources. [`avatar_bridge`] publishes the resulting control frames.
//! - [`ui_model::UiViewModel`] is the snapshot consumed by the UI; widgets enqueue
//!   commands rather than starting workers or persisting settings themselves.
//!
//! [`import`] owns the managed model files and exposes inspection and conversion
//! through one public API. Camera devices stay in `vtuber-camera`, inference
//! backends in `vtuber-inference`, tracking math in `vtuber-tracking`, and VRM
//! rendering in `vtuber-avatar`; this crate coordinates those boundaries.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod avatar_io;
mod file_io;
mod runtime;

pub use avatar_io::{AvatarIoRuntime, prepare_avatar_io_system};
pub use runtime::AppRuntimePlugin;

/// UI action commands emitted by the UI layer.
pub mod actions;

/// Bridge from tracking control frames to the active avatar generation.
pub mod avatar_bridge;

/// Capture runtime — camera backend lifecycle and frame transport.
pub mod capture_runtime;

/// Diagnostics snapshot for the UI.
pub mod diagnostics;

/// Bounded release-run metrics export for performance acceptance.
pub mod metrics_export;

/// Error presenter — maps domain errors to user-facing messages.
pub mod error_presenter;

/// VRM 0.x/1.0 import and preflight inspection.
pub mod import;

/// Fixed 36-key expression assignment and per-model bindings.
pub mod expression_keys;

/// Application bridge for the pure-Rust face inference worker.
pub mod inference_runtime;

/// VRM license review extracted before an avatar import.
pub mod license_review;

/// Licenses of the crates this application is built from and the assets it ships.
pub mod licenses;

/// Manifest-driven inference model catalog.
pub mod model_catalog;

/// App orchestrator — processes UI actions and manages domain state.
pub mod orchestrator;

/// Optional NDI output orchestration and UI snapshot bridge.
pub mod ndi_output;

/// Optional pure-Rust Looking Glass multiview output.
pub mod looking_glass;

/// User-owned persistent settings, including per-model arm-pose overrides.
pub mod settings;

/// Placeholder for app subsystem.
pub mod placeholder;

/// Application bridge for the observed-arm Pose worker.
pub mod pose_runtime;

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

/// File boundary for the user-tunable tracking profile document.
pub mod tracking_file;
