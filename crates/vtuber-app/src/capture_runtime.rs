//! Capture runtime — bridges the orchestrator to the real camera backend.
//!
//! Manages the [`CaptureController`] lifecycle and provides Bevy systems for
//! preview texture updates and diagnostics synchronisation.

use std::sync::Arc;

use bevy::image::ImageSampler;
use bevy::prelude::*;
use bevy_egui::{EguiTextureHandle, EguiUserTextures};
use vtuber_camera::capture::{CaptureController, CaptureServiceState};
use vtuber_camera::device::{CameraBackend, CameraDescriptor, CameraRequest};
use vtuber_core::metrics::RateCounter;
use vtuber_core::{FrameSeq, LatestSlot, VideoFrame, monotonic_now};

use crate::diagnostics::DiagnosticsSnapshot;
use crate::preview::PreviewState;
use crate::privacy_preview::build_privacy_preview;

/// Resource wrapping the production [`CaptureController`].
///
/// The controller is created at app startup and lives for the entire
/// application lifetime. Individual capture sessions are started/stopped
/// via the controller's methods.
#[derive(Resource)]
pub struct CaptureRuntime {
    /// The underlying capture controller.
    controller: CaptureController,
    /// Whether the worker thread has been started.
    worker_started: bool,
    /// Last-read generation for the frame slot.
    last_generation: u64,
}

impl Default for CaptureRuntime {
    fn default() -> Self {
        Self {
            controller: CaptureController::new(),
            worker_started: false,
            last_generation: 0,
        }
    }
}

impl CaptureRuntime {
    /// Returns a reference to the underlying controller.
    #[must_use]
    pub fn controller(&self) -> &CaptureController {
        &self.controller
    }

    /// Returns a mutable reference to the underlying controller.
    pub fn controller_mut(&mut self) -> &mut CaptureController {
        &mut self.controller
    }

    /// Ensures the capture worker thread is running.
    ///
    /// This is idempotent — calling it multiple times has no effect after the
    /// first successful call.
    pub fn ensure_worker_started(&mut self) -> Result<(), String> {
        if self.worker_started {
            return Ok(());
        }

        #[cfg(target_os = "windows")]
        {
            let backend = vtuber_camera::backend::msmf::MsmfBackend::new();
            self.controller
                .start_worker(backend)
                .map_err(|e| format!("{e}"))?;
        }

        #[cfg(not(target_os = "windows"))]
        {
            let backend = vtuber_camera::mock::MockBackend::default();
            self.controller
                .start_worker(backend)
                .map_err(|e| format!("{e}"))?;
        }

        self.worker_started = true;
        Ok(())
    }

    /// Enumerates available cameras.
    pub fn enumerate_cameras(&self) -> Result<Vec<CameraDescriptor>, String> {
        #[cfg(target_os = "windows")]
        {
            let backend = vtuber_camera::backend::msmf::MsmfBackend::new();
            backend.enumerate().map_err(|error| error.to_string())
        }

        #[cfg(not(target_os = "windows"))]
        {
            let backend = vtuber_camera::mock::MockBackend::default();
            backend.enumerate().map_err(|error| error.to_string())
        }
    }

    /// Starts capture with the given device.
    pub fn start_capture(
        &mut self,
        device: CameraDescriptor,
        request: CameraRequest,
    ) -> Result<(), String> {
        self.ensure_worker_started()?;
        self.controller
            .select_and_start(device, request)
            .map_err(|e| format!("{e}"))
    }

    /// Stops capture.
    pub fn stop_capture(&mut self) {
        self.controller.stop();
    }

    /// Stops and joins the capture worker for application shutdown.
    ///
    /// The controller is replaced with a fresh idle controller so this method
    /// can be called from a Bevy exit system without moving the resource out
    /// of the world. Normal Stop uses [`Self::stop_capture`] and preserves the
    /// worker for a later Start.
    pub fn shutdown(&mut self) {
        let controller = std::mem::replace(&mut self.controller, CaptureController::new());
        let _ = controller.shutdown();
        self.worker_started = false;
        self.last_generation = 0;
    }

    /// Returns the current service state.
    #[must_use]
    pub fn state(&self) -> CaptureServiceState {
        self.controller.state()
    }

    /// Returns the frame slot for reading the latest captured frame.
    #[must_use]
    pub fn frame_slot(&self) -> Arc<LatestSlot<VideoFrame>> {
        self.controller.frame_slot()
    }

    /// Tries to read the latest frame from the slot.
    ///
    /// Returns `None` if no new frame is available.
    pub fn try_read_frame(&mut self) -> Option<VideoFrame> {
        let slot = self.controller.frame_slot();
        match slot.try_read_after(self.last_generation) {
            Some(vtuber_core::ReadResult::New(frame)) => {
                // The slot generation may advance by more than one when the
                // producer overwrites unread frames. Track the actual
                // generation so the consumer never re-reads an old value.
                self.last_generation = slot.generation();
                Some(frame)
            }
            Some(vtuber_core::ReadResult::Closed) | None => None,
        }
    }
}

/// Resource holding the latest captured video frame for preview display.
#[derive(Resource, Default)]
pub struct LatestVideoFrame {
    /// The most recent frame, if any.
    pub frame: Option<VideoFrame>,
}

/// System that reads the latest frame from the capture slot and stores it
/// for preview rendering.
pub fn read_latest_frame(
    mut capture: ResMut<CaptureRuntime>,
    mut latest: ResMut<LatestVideoFrame>,
) {
    if let Some(frame) = capture.try_read_frame() {
        latest.frame = Some(frame);
    }
}

/// System that synchronises capture metrics into [`DiagnosticsSnapshot`].
pub fn sync_capture_diagnostics(
    capture: Res<CaptureRuntime>,
    latest: Res<LatestVideoFrame>,
    mut diagnostics: ResMut<DiagnosticsSnapshot>,
    mut last_seq: Local<Option<FrameSeq>>,
    mut rate: Local<Option<RateCounter>>,
) {
    let metrics = capture.controller().metrics();
    let rate_counter = rate.get_or_insert_with(|| RateCounter::new(1_000_000_000));
    if let Some(frame) = latest.frame.as_ref()
        && *last_seq != Some(frame.seq)
    {
        rate_counter.record(frame.captured_at.0);
        *last_seq = Some(frame.seq);
    }
    diagnostics.capture_rate = rate_counter.rate_hz(monotonic_now().0) as f32;
    diagnostics.capture_state = format!("{:?}", capture.state());
    diagnostics.slot_overwrites = metrics.frames_dropped;
    diagnostics.camera_backend = Some(camera_backend_name().to_string());
    if let Some(err) = metrics.last_error {
        diagnostics.last_error = Some(err);
    }
}

fn camera_backend_name() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "MSMF"
    }
    #[cfg(target_os = "macos")]
    {
        "AVFoundation"
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        "Mock"
    }
}

/// Updates one reusable Bevy image from the latest captured frame.
///
/// This system is display-only: it never mutates the source frame and does
/// not mirror or otherwise transform the bytes used by inference. Updates are
/// throttled by [`PreviewState::target_fps`], while capture continues when the
/// preview is hidden.
pub fn update_preview_texture_system(
    latest: Res<LatestVideoFrame>,
    mut preview: ResMut<PreviewState>,
    mut images: ResMut<Assets<Image>>,
    mut last_upload: Local<Option<std::time::Instant>>,
) {
    if !preview.visible {
        return;
    }
    let Some(frame) = latest.frame.as_ref() else {
        return;
    };
    let now = std::time::Instant::now();
    if last_upload.is_some_and(|previous| now.duration_since(previous) < preview.update_interval())
    {
        return;
    }

    let Some(image) = preview_image(frame) else {
        return;
    };

    if let Some(handle) = preview.image_handle.as_ref() {
        if let Some(mut existing) = images.get_mut(handle.id()) {
            *existing = image;
        }
    } else {
        preview.image_handle = Some(images.add(image));
    }
    *last_upload = Some(now);
}

/// Registers the reusable preview image with egui once it exists.
pub fn register_preview_texture_system(
    preview: Res<PreviewState>,
    mut textures: ResMut<EguiUserTextures>,
) {
    if let Some(handle) = preview.image_handle.as_ref()
        && textures.image_id(handle.id()).is_none()
    {
        textures.add_image(EguiTextureHandle::Weak(handle.id()));
    }
}

fn preview_image(frame: &VideoFrame) -> Option<Image> {
    let privacy = build_privacy_preview(frame).ok()?;
    let size = bevy::render::render_resource::Extent3d {
        width: privacy.width,
        height: privacy.height,
        depth_or_array_layers: 1,
    };
    let mut image = Image::new(
        size,
        bevy::render::render_resource::TextureDimension::D2,
        privacy.rgba,
        bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
        // Preview pixels change every frame, so the CPU-side asset must stay
        // in the main world while also being extracted for rendering.
        bevy::asset::RenderAssetUsages::default(),
    );
    image.sampler = ImageSampler::linear();
    Some(image)
}

/// Bridge system that connects the orchestrator's capture intent to the
/// [`CaptureRuntime`].
///
/// - Enumerates cameras when the orchestrator signals a refresh.
/// - Starts capture when the orchestrator transitions to `Starting`.
/// - Stops capture when the orchestrator transitions to `Stopping`.
/// - Updates the orchestrator's pipeline state based on capture state.
pub fn capture_bridge_system(
    mut capture: ResMut<CaptureRuntime>,
    mut orchestrator: ResMut<crate::orchestrator::Orchestrator>,
) {
    use crate::orchestrator::PipelineState;

    if capture.controller().worker_finished()
        && matches!(
            capture.state(),
            CaptureServiceState::Starting | CaptureServiceState::Running
        )
        && orchestrator.capture_desired()
    {
        orchestrator.fail_camera("capture worker exited unexpectedly".to_string());
    }

    // Handle camera enumeration refresh independently from start/stop
    // acknowledgements. Refreshing while the pipeline is running must not
    // accidentally stop or restart the worker.
    if orchestrator.camera_refresh_requested() {
        match capture.enumerate_cameras() {
            Ok(cameras) => orchestrator.set_camera_list(cameras),
            Err(error) => orchestrator.set_last_error(Some(
                crate::orchestrator::OrchestratorError::CameraFailed(error),
            )),
        }
        orchestrator.clear_camera_refresh_request();
    }

    // Handle capture start.
    if orchestrator.capture_desired()
        && orchestrator.pipeline_state() == PipelineState::Starting
        && !orchestrator.capture_ack()
    {
        if let Some(device) = orchestrator.selected_camera_descriptor() {
            let request = vtuber_camera::CameraRequest::default();
            match capture.start_capture(device, request) {
                Ok(()) => {
                    orchestrator.set_capture_ack(true);
                    orchestrator.set_last_error(None);
                }
                Err(e) => {
                    orchestrator.set_pipeline_state(PipelineState::Failed);
                    orchestrator.set_capture_ack(true);
                    orchestrator.set_last_error(Some(
                        crate::orchestrator::OrchestratorError::CameraFailed(e),
                    ));
                }
            }
        } else {
            orchestrator.set_pipeline_state(PipelineState::Failed);
            orchestrator.set_capture_ack(true);
            orchestrator.set_last_error(Some(
                crate::orchestrator::OrchestratorError::NoCameraSelected,
            ));
        }
    }

    // Reflect the asynchronous worker state after a start command. A
    // successful command only means that opening was requested; Running is
    // reported after the worker has opened the selected device and produced
    // its first frame.
    if orchestrator.capture_desired()
        && orchestrator.pipeline_state() == PipelineState::Starting
        && capture.state() == CaptureServiceState::Running
    {
        orchestrator.set_pipeline_state(PipelineState::Running);
    } else if orchestrator.capture_desired() && capture.state() == CaptureServiceState::BackOff {
        orchestrator.set_pipeline_state(PipelineState::Failed);
        orchestrator.set_last_error(Some(crate::orchestrator::OrchestratorError::CameraFailed(
            "camera entered back-off".to_string(),
        )));
    }

    // Handle capture stop.
    if !orchestrator.capture_desired()
        && orchestrator.pipeline_state() == PipelineState::Stopping
        && !orchestrator.capture_ack()
    {
        capture.stop_capture();
        orchestrator.set_capture_ack(true);
    }

    if !orchestrator.capture_desired()
        && orchestrator.pipeline_state() == PipelineState::Stopping
        && matches!(
            capture.state(),
            CaptureServiceState::Selected | CaptureServiceState::Idle
        )
    {
        orchestrator.complete_capture_stop();
    }
}

#[cfg(test)]
mod preview_tests {
    use std::sync::Arc;

    use super::*;
    use vtuber_core::{MonoTimeNs, PixelFormat};

    fn rgb_frame() -> VideoFrame {
        VideoFrame {
            seq: FrameSeq(1),
            captured_at: MonoTimeNs(1),
            width: 1,
            height: 1,
            stride_bytes: 3,
            format: PixelFormat::Rgb8,
            data: Arc::from([10, 20, 30]),
        }
    }

    fn solid_rgb_frame(width: u32, height: u32, pixel: [u8; 3]) -> VideoFrame {
        let mut data = Vec::with_capacity(width as usize * height as usize * 3);
        for _ in 0..(width as usize * height as usize) {
            data.extend_from_slice(&pixel);
        }
        VideoFrame {
            seq: FrameSeq(2),
            captured_at: MonoTimeNs(2),
            width,
            height,
            stride_bytes: width as usize * 3,
            format: PixelFormat::Rgb8,
            data: Arc::from(data),
        }
    }

    fn invalid_frame() -> VideoFrame {
        VideoFrame {
            seq: FrameSeq(3),
            captured_at: MonoTimeNs(3),
            width: 2,
            height: 1,
            stride_bytes: 1,
            format: PixelFormat::Rgb8,
            data: Arc::from([0]),
        }
    }

    #[test]
    fn preview_image_remains_mutable_in_main_world() {
        let image = preview_image(&rgb_frame()).expect("valid RGB frame should produce an image");
        assert!(
            image
                .asset_usage
                .contains(bevy::asset::RenderAssetUsages::MAIN_WORLD)
        );
        assert!(
            image
                .asset_usage
                .contains(bevy::asset::RenderAssetUsages::RENDER_WORLD)
        );
    }

    #[test]
    fn preview_image_uses_privacy_dimensions_data_limit_and_linear_magnification() {
        let image = preview_image(&solid_rgb_frame(1280, 720, [10, 20, 30]))
            .expect("valid RGB frame should produce a privacy image");

        assert_eq!(image.texture_descriptor.size.width, 48);
        assert_eq!(image.texture_descriptor.size.height, 27);
        assert_eq!(image.data.as_ref().expect("image data").len(), 48 * 27 * 4);
        assert_eq!(image.sampler, ImageSampler::linear());
    }

    #[test]
    fn conversion_error_keeps_the_existing_image_unchanged() {
        let mut app = App::new();
        app.init_resource::<Assets<Image>>()
            .init_resource::<PreviewState>()
            .init_resource::<LatestVideoFrame>()
            .add_systems(Update, update_preview_texture_system);
        let original = preview_image(&rgb_frame()).expect("valid preview image");
        let handle = app
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(original);
        app.world_mut().resource_mut::<PreviewState>().image_handle = Some(handle.clone());
        app.world_mut().resource_mut::<LatestVideoFrame>().frame = Some(invalid_frame());

        app.update();

        let image = app
            .world()
            .resource::<Assets<Image>>()
            .get(handle.id())
            .expect("existing image remains registered");
        assert_eq!(image.texture_descriptor.size.width, 1);
        assert_eq!(image.texture_descriptor.size.height, 1);
        assert_eq!(image.data.as_ref().expect("image data"), &[10, 20, 30, 255]);
    }

    #[test]
    fn hidden_preview_does_not_upload_and_visible_updates_reuse_the_handle() {
        let mut app = App::new();
        app.init_resource::<Assets<Image>>()
            .init_resource::<PreviewState>()
            .init_resource::<LatestVideoFrame>()
            .add_systems(Update, update_preview_texture_system);
        let initial = preview_image(&rgb_frame()).expect("valid preview image");
        let handle = app.world_mut().resource_mut::<Assets<Image>>().add(initial);
        {
            let world = app.world_mut();
            world.resource_mut::<PreviewState>().image_handle = Some(handle.clone());
            world.resource_mut::<PreviewState>().visible = false;
            world.resource_mut::<LatestVideoFrame>().frame =
                Some(solid_rgb_frame(1280, 720, [1, 2, 3]));
        }

        app.update();
        let image = app
            .world()
            .resource::<Assets<Image>>()
            .get(handle.id())
            .expect("hidden preview keeps image");
        assert_eq!(image.texture_descriptor.size.width, 1);

        app.world_mut().resource_mut::<PreviewState>().visible = true;
        app.update();
        let preview = app.world().resource::<PreviewState>();
        assert_eq!(
            preview.image_handle.as_ref().map(Handle::id),
            Some(handle.id())
        );
        let image = app
            .world()
            .resource::<Assets<Image>>()
            .get(handle.id())
            .expect("reused image handle remains registered");
        assert_eq!(image.texture_descriptor.size.width, 48);
        assert_eq!(image.data.as_ref().expect("image data")[..3], [1, 2, 3]);
    }

    #[test]
    fn preview_upload_keeps_the_existing_target_fps_throttle() {
        let mut app = App::new();
        app.init_resource::<Assets<Image>>()
            .init_resource::<PreviewState>()
            .init_resource::<LatestVideoFrame>()
            .add_systems(Update, update_preview_texture_system);
        app.world_mut().resource_mut::<PreviewState>().target_fps = 1;
        app.world_mut().resource_mut::<LatestVideoFrame>().frame = Some(rgb_frame());
        app.update();

        app.world_mut().resource_mut::<LatestVideoFrame>().frame =
            Some(solid_rgb_frame(1, 1, [1, 2, 3]));
        app.update();

        let handle = app
            .world()
            .resource::<PreviewState>()
            .image_handle
            .as_ref()
            .expect("first upload created a handle")
            .clone();
        let image = app
            .world()
            .resource::<Assets<Image>>()
            .get(handle.id())
            .expect("throttled image remains registered");
        assert_eq!(image.data.as_ref().expect("image data"), &[10, 20, 30, 255]);
    }

    #[test]
    fn preview_handle_is_registered_for_egui() {
        let mut app = App::new();
        app.init_resource::<Assets<Image>>()
            .init_resource::<PreviewState>()
            .init_resource::<EguiUserTextures>()
            .add_systems(Update, register_preview_texture_system);
        let handle = app
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(preview_image(&rgb_frame()).expect("valid preview image"));
        app.world_mut().resource_mut::<PreviewState>().image_handle = Some(handle.clone());

        app.update();

        assert!(
            app.world()
                .resource::<EguiUserTextures>()
                .image_id(handle.id())
                .is_some()
        );
    }
}
