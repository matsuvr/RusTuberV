//! Transparent avatar-only render target and asynchronous GPU readback.
//!
//! Rendering this texture for a local preview does not activate readback or
//! NDI. The UI samples the texture; it never renders into this camera.

use crate::lifecycle::AvatarGeneration;
use bevy::camera::{CameraUpdateSystems, ClearColorConfig, RenderTarget, visibility::RenderLayers};
use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{TextureFormat, TextureUsages};
use bevy::render::renderer::RenderDevice;
use vtuber_core::{FrameSeq, VideoOutputFrame, VideoOutputProfile, monotonic_now};

/// The rendering layer containing avatar geometry and output lighting.
pub const AVATAR_RENDER_LAYER: usize = 0;
/// The main-window-only layer containing the ground plane.
pub const VIEWPORT_ONLY_RENDER_LAYER: usize = 1;

/// Fixed render target and profile used by both the preview and output camera.
#[derive(Resource, Clone, Debug)]
pub struct AvatarOutputTarget {
    image: Handle<Image>,
    profile: VideoOutputProfile,
}
impl AvatarOutputTarget {
    /// The fixed transport-neutral output profile.
    #[must_use]
    pub const fn profile(&self) -> VideoOutputProfile {
        self.profile
    }
    /// The render-target image, also sampled by the application preview.
    #[must_use]
    pub fn image(&self) -> &Handle<Image> {
        &self.image
    }
}

/// Independent activation of local rendering and transport readback.
#[derive(Resource, Clone, Debug, Default)]
pub struct AvatarOutputState {
    active: bool,
    preview_visible: bool,
    profile: VideoOutputProfile,
}
impl AvatarOutputState {
    /// Whether transport readback is active. Local preview does not change this.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.active
    }
    /// Whether either consumer needs the avatar render target this frame.
    #[must_use]
    pub const fn is_rendering(&self) -> bool {
        self.active || self.preview_visible
    }
    /// Request GPU rendering for the local preview, without any CPU readback.
    pub fn set_preview_visible(&mut self, visible: bool) {
        self.preview_visible = visible;
    }
    /// The fixed output profile.
    #[must_use]
    pub const fn profile(&self) -> VideoOutputProfile {
        self.profile
    }
    /// Activate or deactivate transport readback without hiding the local preview.
    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }
    /// Activate transparent output readback.
    pub fn activate(&mut self) {
        self.set_active(true);
    }
    /// Deactivate readback; a visible local preview keeps rendering.
    pub fn deactivate(&mut self) {
        self.set_active(false);
    }
    /// Create an inactive output state with a caller-selected profile.
    #[must_use]
    pub const fn with_profile(profile: VideoOutputProfile) -> Self {
        Self {
            active: false,
            preview_visible: false,
            profile,
        }
    }
}

/// Capacity-one slot for completed output frames.
#[derive(Resource, Default, Debug)]
pub struct AvatarOutputFrameSlot {
    latest: Option<VideoOutputFrame>,
    next_frame_seq: u64,
    received_frames: u64,
    replaced_frames: u64,
    rejected_frames: u64,
}
impl AvatarOutputFrameSlot {
    /// Take the newest completed frame.
    pub fn take_latest(&mut self) -> Option<VideoOutputFrame> {
        self.latest.take()
    }
    /// Inspect the newest completed frame.
    #[must_use]
    pub fn latest(&self) -> Option<&VideoOutputFrame> {
        self.latest.as_ref()
    }
    /// Successfully converted readback frames.
    #[must_use]
    pub const fn received_frames(&self) -> u64 {
        self.received_frames
    }
    /// Pending frames replaced before consumption.
    #[must_use]
    pub const fn replaced_frames(&self) -> u64 {
        self.replaced_frames
    }
    /// Malformed readbacks rejected at the contract boundary.
    #[must_use]
    pub const fn rejected_frames(&self) -> u64 {
        self.rejected_frames
    }
    /// Publish a completed transport-neutral frame.
    pub fn publish(&mut self, frame: VideoOutputFrame) {
        self.replace(frame);
    }
    fn replace(&mut self, frame: VideoOutputFrame) {
        self.next_frame_seq = self.next_frame_seq.saturating_add(1);
        self.received_frames = self.received_frames.saturating_add(1);
        if self.latest.replace(frame).is_some() {
            self.replaced_frames = self.replaced_frames.saturating_add(1);
        }
    }
    fn reject(&mut self) {
        self.rejected_frames = self.rejected_frames.saturating_add(1);
    }
    fn next_frame_seq(&self) -> FrameSeq {
        FrameSeq(self.next_frame_seq)
    }
}

/// Read-only camera snapshot after framing and manual controls.
#[derive(Resource, Clone, Debug, Default)]
pub struct AvatarViewportSnapshot {
    /// Associated avatar generation.
    pub generation: AvatarGeneration,
    /// Main viewport transform.
    pub transform: Option<Transform>,
    /// Main perspective projection.
    pub projection: Option<PerspectiveProjection>,
}

/// Dedicated transparent avatar-only camera; never an egui context.
#[derive(Component, Debug)]
pub struct AvatarOutputCamera;

#[derive(Component, Debug)]
struct AvatarOutputReadbackInFlight;

#[derive(SystemParam)]
struct OutputCameraQuery<'w, 's> {
    #[expect(
        clippy::type_complexity,
        reason = "Bevy's `SystemParam` query for the output camera, with no call site to change"
    )]
    cameras: Query<
        'w,
        's,
        (
            Entity,
            &'static mut Camera,
            &'static mut Projection,
            &'static mut Transform,
            &'static mut GlobalTransform,
            Option<&'static AvatarOutputReadbackInFlight>,
        ),
        (
            With<AvatarOutputCamera>,
            Without<crate::framing::AvatarViewportCamera>,
        ),
    >,
}

/// Create the fixed transparent image and output camera.
pub fn setup_output_camera(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut state: ResMut<AvatarOutputState>,
) {
    let profile = state.profile();
    let mut image = Image::new_target_texture(
        profile.width,
        profile.height,
        TextureFormat::Bgra8UnormSrgb,
        None,
    );
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC | TextureUsages::TEXTURE_BINDING;
    let image_handle = images.add(image);
    let camera_transform =
        Transform::from_translation(Vec3::new(0.0, 0.0, 2.5)).looking_at(Vec3::ZERO, Vec3::Y);
    commands.insert_resource(AvatarOutputTarget {
        image: image_handle.clone(),
        profile,
    });
    let mut camera = Camera {
        is_active: false,
        clear_color: ClearColorConfig::Custom(Color::srgba(0.0, 0.0, 0.0, 0.0)),
        ..default()
    };
    camera.order = -1;
    commands
        .spawn((
            Camera3d::default(),
            // preview and the NDI output share one finished image.
            camera,
            RenderTarget::Image(image_handle.into()),
            Projection::Perspective(PerspectiveProjection {
                fov: crate::framing::fixed_fov_fit::FIXED_VERTICAL_FOV,
                aspect_ratio: profile.width as f32 / profile.height as f32,
                ..default()
            }),
            camera_transform,
            RenderLayers::layer(AVATAR_RENDER_LAYER),
            AvatarOutputCamera,
        ))
        .observe(handle_output_readback);
    state.deactivate();
}

/// Mirror main framing into the output target. Only transport activation
/// schedules a readback; preview-only rendering remains entirely on the GPU.
#[expect(
    clippy::type_complexity,
    reason = "Bevy injects this system's resources and query filters, so the parameter list is the declared ECS contract and has no call site to restructure"
)]
fn sync_output_camera(
    lifecycle: Res<crate::lifecycle::AvatarLifecycle>,
    state: Res<AvatarOutputState>,
    target: Res<AvatarOutputTarget>,
    mut snapshot: ResMut<AvatarViewportSnapshot>,
    main_cameras: Query<
        (&Transform, &Projection),
        (
            With<crate::framing::AvatarViewportCamera>,
            Without<AvatarOutputCamera>,
        ),
    >,
    mut output_cameras: OutputCameraQuery,
    mut commands: Commands,
) {
    snapshot.generation = lifecycle.current_generation();
    let Ok((main_transform, main_projection)) = main_cameras.single() else {
        snapshot.transform = None;
        snapshot.projection = None;
        return;
    };
    let Projection::Perspective(main_projection) = main_projection else {
        snapshot.transform = None;
        snapshot.projection = None;
        return;
    };
    if snapshot.transform != Some(*main_transform) {
        snapshot.transform = Some(*main_transform);
    }
    if snapshot.projection.as_ref().is_none_or(|projection| {
        projection.fov != main_projection.fov
            || projection.aspect_ratio != main_projection.aspect_ratio
            || projection.near != main_projection.near
            || projection.far != main_projection.far
    }) {
        snapshot.projection = Some(main_projection.clone());
    }
    for (entity, mut camera, mut projection, mut transform, mut global_transform, in_flight) in
        &mut output_cameras.cameras
    {
        if *transform != *main_transform {
            *transform = *main_transform;
        }
        let main_global = GlobalTransform::from(*main_transform);
        if *global_transform != main_global {
            *global_transform = main_global;
        }
        let projection_changed = match &*projection {
            Projection::Perspective(current) => {
                current.fov != main_projection.fov
                    || current.aspect_ratio != main_projection.aspect_ratio
                    || current.near != main_projection.near
                    || current.far != main_projection.far
            }
            Projection::Orthographic(_) | Projection::Custom(_) => true,
        };
        if projection_changed {
            *projection = Projection::Perspective(main_projection.clone());
        }
        camera.is_active = state.is_rendering();
        if state.is_active() && in_flight.is_none() {
            commands.entity(entity).insert((
                Readback::texture(target.image().clone()),
                AvatarOutputReadbackInFlight,
            ));
        } else if !state.is_active() {
            commands
                .entity(entity)
                .remove::<Readback>()
                .remove::<AvatarOutputReadbackInFlight>();
        }
    }
}

fn handle_output_readback(
    event: On<ReadbackComplete>,
    mut commands: Commands,
    state: Res<AvatarOutputState>,
    target: Res<AvatarOutputTarget>,
    mut slot: ResMut<AvatarOutputFrameSlot>,
) {
    commands
        .entity(event.entity)
        .remove::<Readback>()
        .remove::<AvatarOutputReadbackInFlight>();
    if !state.is_active() {
        return;
    }
    let profile = target.profile();
    let packed_stride = profile.packed_stride_bytes();
    let source_stride = RenderDevice::align_copy_bytes_per_row(packed_stride);
    match VideoOutputFrame::from_padded_bgra8(
        profile.width,
        profile.height,
        source_stride,
        slot.next_frame_seq(),
        monotonic_now(),
        &event.data,
    ) {
        Ok(frame) => slot.publish(frame),
        Err(error) => {
            slot.reject();
            warn!("discarding malformed avatar output readback: {error}");
        }
    }
}

/// Register output resources and lifecycle systems.
pub fn register_output_systems(app: &mut App) {
    app.init_resource::<AvatarOutputState>()
        .init_resource::<AvatarOutputFrameSlot>()
        .init_resource::<AvatarViewportSnapshot>()
        .add_systems(Startup, setup_output_camera)
        .add_systems(
            PostUpdate,
            sync_output_camera
                .after(crate::framing::frame_avatar_camera)
                .after(TransformSystems::Propagate)
                .before(CameraUpdateSystems),
        );
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )] // tests may panic (AGENTS.md)
    use super::*;
    use bevy::render::render_resource::Extent3d;

    fn test_target() -> AvatarOutputTarget {
        AvatarOutputTarget {
            image: Handle::default(),
            profile: VideoOutputProfile::default(),
        }
    }

    #[test]
    fn output_starts_inactive_and_has_fixed_profile() {
        let state = AvatarOutputState::default();
        assert!(!state.is_active());
        assert!(!state.is_rendering());
        assert_eq!(state.profile(), VideoOutputProfile::DEFAULT);
    }

    #[test]
    fn output_target_keeps_the_transport_dimensions_and_srgb_bgra_format() {
        let mut app = App::new();
        app.init_resource::<Assets<Image>>()
            .insert_resource(AvatarOutputState::default())
            .add_systems(Startup, setup_output_camera);
        app.update();

        let target = app.world().resource::<AvatarOutputTarget>();
        let image = app
            .world()
            .resource::<Assets<Image>>()
            .get(target.image())
            .expect("output image asset");
        assert_eq!(
            image.texture_descriptor.format,
            TextureFormat::Bgra8UnormSrgb
        );
        assert_eq!(
            image.texture_descriptor.size,
            Extent3d {
                width: VideoOutputProfile::DEFAULT.width,
                height: VideoOutputProfile::DEFAULT.height,
                depth_or_array_layers: 1,
            }
        );
    }

    #[test]
    fn frame_slot_keeps_only_the_latest_frame() {
        let mut slot = AvatarOutputFrameSlot::default();
        let make = |seq| {
            VideoOutputFrame::new_bgra8(
                1,
                1,
                FrameSeq(seq),
                vtuber_core::MonoTimeNs(seq),
                vec![0, 0, 0, 0],
            )
            .expect("transparent pixel")
        };
        slot.publish(make(0));
        slot.publish(make(1));
        assert_eq!(slot.replaced_frames(), 1);
        assert_eq!(
            slot.take_latest().expect("latest frame").frame_seq(),
            FrameSeq(1)
        );
        assert!(slot.take_latest().is_none());
    }

    fn camera_entity(app: &mut App, main: bool) -> Entity {
        if main {
            let mut query = app
                .world_mut()
                .query_filtered::<Entity, With<crate::framing::AvatarViewportCamera>>();
            query.iter(app.world()).next().expect("main camera")
        } else {
            let mut query = app
                .world_mut()
                .query_filtered::<Entity, With<AvatarOutputCamera>>();
            query.iter(app.world()).next().expect("output camera")
        }
    }

    fn spawn_mirrored_cameras(
        app: &mut App,
        main_transform: Transform,
        main_projection: PerspectiveProjection,
    ) -> Entity {
        app.world_mut().spawn((
            main_transform,
            Projection::Perspective(main_projection),
            crate::framing::AvatarViewportCamera::from_default_transform(main_transform),
        ));
        app.world_mut()
            .spawn((
                Camera::default(),
                Projection::Perspective(PerspectiveProjection {
                    fov: crate::framing::fixed_fov_fit::FIXED_VERTICAL_FOV,
                    ..default()
                }),
                Transform::default(),
                GlobalTransform::default(),
                RenderTarget::Image(Handle::default().into()),
                RenderLayers::layer(AVATAR_RENDER_LAYER),
                AvatarOutputCamera,
            ))
            .id()
    }

    fn output_sync_app(generation: AvatarGeneration) -> App {
        let mut app = App::new();
        let mut lifecycle = crate::lifecycle::AvatarLifecycle::default();
        let root = app.world_mut().spawn_empty().id();
        lifecycle.request_load(root).expect("load from NoAvatar");
        lifecycle.start_binding(root);
        lifecycle.finish_ready();
        while lifecycle.current_generation() != generation {
            let next = app.world_mut().spawn_empty().id();
            lifecycle
                .request_replace(next)
                .expect("ready avatar can be replaced");
            lifecycle.finish_unload();
            lifecycle.start_binding(next);
            lifecycle.finish_ready();
        }
        app.insert_resource(lifecycle)
            .insert_resource(AvatarOutputState::default())
            .insert_resource(test_target())
            .insert_resource(AvatarViewportSnapshot::default())
            .insert_resource(AvatarOutputFrameSlot::default())
            .add_systems(Update, sync_output_camera);
        app
    }

    #[test]
    fn output_camera_mirrors_the_current_viewport_state() {
        let mut app = output_sync_app(AvatarGeneration(1));
        let main_transform = Transform::from_xyz(1.0, 2.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y);
        let output = spawn_mirrored_cameras(
            &mut app,
            main_transform,
            PerspectiveProjection {
                fov: 0.42,
                aspect_ratio: 1.7,
                ..default()
            },
        );
        app.update();
        assert_eq!(app.world().get::<Transform>(output), Some(&main_transform));
        assert_eq!(
            app.world().get::<GlobalTransform>(output),
            Some(&GlobalTransform::from(main_transform))
        );
        let Projection::Perspective(projection) = app.world().get::<Projection>(output).unwrap()
        else {
            panic!("perspective");
        };
        assert_eq!(projection.fov, 0.42);
        assert_eq!(projection.aspect_ratio, 1.7);
        let snapshot = app.world().resource::<AvatarViewportSnapshot>();
        assert_eq!(snapshot.transform, Some(main_transform));
        assert_eq!(snapshot.projection.as_ref().unwrap().fov, 0.42);
        assert_eq!(snapshot.projection.as_ref().unwrap().aspect_ratio, 1.7);
    }

    #[test]
    fn layer_contract_excludes_ground_from_output() {
        let output = RenderLayers::layer(AVATAR_RENDER_LAYER);
        let ground = RenderLayers::layer(VIEWPORT_ONLY_RENDER_LAYER);
        let viewport =
            RenderLayers::from_layers(&[AVATAR_RENDER_LAYER, VIEWPORT_ONLY_RENDER_LAYER]);
        assert!(!output.intersects(&ground));
        assert!(output.intersects(&viewport));
        assert!(ground.intersects(&viewport));
    }

    #[test]
    fn output_camera_does_not_invent_a_different_perspective_fov() {
        let mut app = output_sync_app(AvatarGeneration(1));
        let main_transform = Transform::from_xyz(0.0, 1.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y);
        let main_projection = PerspectiveProjection {
            fov: crate::framing::fixed_fov_fit::FIXED_VERTICAL_FOV,
            aspect_ratio: 16.0 / 9.0,
            ..default()
        };
        let output = spawn_mirrored_cameras(&mut app, main_transform, main_projection.clone());
        app.update();
        let Projection::Perspective(projection) = app.world().get::<Projection>(output).unwrap()
        else {
            panic!("perspective");
        };
        assert_eq!(
            projection.fov,
            crate::framing::fixed_fov_fit::FIXED_VERTICAL_FOV
        );
        assert_eq!(projection.fov, main_projection.fov);
        assert_eq!(projection.aspect_ratio, main_projection.aspect_ratio);
        assert_eq!(projection.near, main_projection.near);
        assert_eq!(projection.far, main_projection.far);
    }

    #[test]
    fn orbit_pan_dolly_and_reset_are_mirrored_in_the_same_frame() {
        let mut app = output_sync_app(AvatarGeneration(1));
        let initial =
            Transform::from_xyz(0.0, 1.0, 5.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y);
        let projection = PerspectiveProjection {
            fov: crate::framing::fixed_fov_fit::FIXED_VERTICAL_FOV,
            aspect_ratio: 16.0 / 9.0,
            ..default()
        };
        let output = spawn_mirrored_cameras(&mut app, initial, projection.clone());
        let main = camera_entity(&mut app, true);
        let pose = crate::framing::camera_control::CameraControlPose::new(
            initial,
            Vec3::new(0.0, 1.0, 0.0),
        )
        .expect("pose");
        let orbited =
            crate::framing::camera_control::geometry::orbit(pose, 0.3, 0.1).expect("orbit");
        *app.world_mut().get_mut::<Transform>(main).unwrap() = orbited.transform();
        app.update();
        assert_eq!(
            app.world().get::<Transform>(output),
            Some(&orbited.transform())
        );
        let panned = crate::framing::camera_control::geometry::pan(
            orbited,
            Vec2::new(40.0, -12.0),
            Vec2::new(1920.0, 1080.0),
        )
        .expect("pan");
        *app.world_mut().get_mut::<Transform>(main).unwrap() = panned.transform();
        app.update();
        assert_eq!(
            app.world().get::<Transform>(output),
            Some(&panned.transform())
        );
        let dollied = crate::framing::camera_control::geometry::dolly(
            panned,
            1.0,
            crate::framing::camera_control::CameraControlConfig::default(),
        )
        .expect("dolly");
        *app.world_mut().get_mut::<Transform>(main).unwrap() = dollied.transform();
        app.update();
        assert_eq!(
            app.world().get::<Transform>(output),
            Some(&dollied.transform())
        );
        *app.world_mut().get_mut::<Transform>(main).unwrap() = initial;
        app.update();
        assert_eq!(app.world().get::<Transform>(output), Some(&initial));
        let Projection::Perspective(mirrored) = app.world().get::<Projection>(output).unwrap()
        else {
            panic!("perspective");
        };
        assert_eq!(mirrored.fov, projection.fov);
    }

    #[test]
    fn replacement_refreshes_generation_snapshot_instead_of_reusing_stale_camera() {
        let mut app = output_sync_app(AvatarGeneration(1));
        let first = Transform::from_xyz(1.0, 2.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y);
        let projection = PerspectiveProjection {
            fov: crate::framing::fixed_fov_fit::FIXED_VERTICAL_FOV,
            ..default()
        };
        spawn_mirrored_cameras(&mut app, first, projection);
        app.update();
        assert_eq!(
            app.world().resource::<AvatarViewportSnapshot>().generation,
            AvatarGeneration(1)
        );
        assert_eq!(
            app.world().resource::<AvatarViewportSnapshot>().transform,
            Some(first)
        );
        let next_root = app.world_mut().spawn_empty().id();
        app.world_mut()
            .resource_mut::<crate::lifecycle::AvatarLifecycle>()
            .request_replace(next_root)
            .expect("replace");
        app.update();
        assert_eq!(
            app.world().resource::<AvatarViewportSnapshot>().generation,
            AvatarGeneration(2)
        );
        {
            let mut lifecycle = app
                .world_mut()
                .resource_mut::<crate::lifecycle::AvatarLifecycle>();
            lifecycle.finish_unload();
            lifecycle.start_binding(next_root);
            lifecycle.finish_ready();
        }
        let replacement = Transform::from_xyz(4.0, 5.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y);
        let main = camera_entity(&mut app, true);
        *app.world_mut().get_mut::<Transform>(main).unwrap() = replacement;
        app.update();
        let snapshot = app.world().resource::<AvatarViewportSnapshot>();
        assert_eq!(snapshot.generation, AvatarGeneration(2));
        assert_eq!(snapshot.transform, Some(replacement));
        assert_ne!(snapshot.transform, Some(first));
    }

    #[test]
    fn inactive_output_does_not_keep_a_readback_in_flight() {
        let mut app = output_sync_app(AvatarGeneration(1));
        spawn_mirrored_cameras(
            &mut app,
            Transform::from_xyz(0.0, 0.0, 2.5).looking_at(Vec3::ZERO, Vec3::Y),
            PerspectiveProjection {
                fov: crate::framing::fixed_fov_fit::FIXED_VERTICAL_FOV,
                ..default()
            },
        );
        app.world_mut()
            .resource_mut::<AvatarOutputState>()
            .activate();
        app.update();
        let output = camera_entity(&mut app, false);
        assert!(app.world().get::<Camera>(output).unwrap().is_active);
        assert!(
            app.world()
                .get::<AvatarOutputReadbackInFlight>(output)
                .is_some()
        );
        assert!(app.world().get::<Readback>(output).is_some());
        app.world_mut()
            .resource_mut::<AvatarOutputState>()
            .deactivate();
        app.update();
        assert!(!app.world().get::<Camera>(output).unwrap().is_active);
        assert!(
            app.world()
                .get::<AvatarOutputReadbackInFlight>(output)
                .is_none()
        );
        assert!(app.world().get::<Readback>(output).is_none());
        assert!(
            app.world()
                .resource::<AvatarOutputFrameSlot>()
                .latest()
                .is_none()
        );
    }

    #[test]
    fn local_preview_renders_without_readback_and_survives_stopping_ndi() {
        let mut app = output_sync_app(AvatarGeneration(1));
        let output = spawn_mirrored_cameras(
            &mut app,
            Transform::default(),
            PerspectiveProjection::default(),
        );
        app.world_mut()
            .resource_mut::<AvatarOutputState>()
            .set_preview_visible(true);
        app.update();
        assert!(app.world().get::<Camera>(output).unwrap().is_active);
        assert!(!app.world().resource::<AvatarOutputState>().is_active());
        assert!(app.world().get::<Readback>(output).is_none());
        assert!(
            app.world()
                .get::<AvatarOutputReadbackInFlight>(output)
                .is_none()
        );
        app.world_mut()
            .resource_mut::<AvatarOutputState>()
            .activate();
        app.update();
        assert!(app.world().get::<Readback>(output).is_some());
        app.world_mut()
            .resource_mut::<AvatarOutputState>()
            .deactivate();
        app.update();
        assert!(app.world().get::<Camera>(output).unwrap().is_active);
        assert!(app.world().get::<Readback>(output).is_none());
        app.world_mut()
            .resource_mut::<AvatarOutputState>()
            .set_preview_visible(false);
        app.update();
        assert!(!app.world().get::<Camera>(output).unwrap().is_active);
    }

    #[test]
    fn output_camera_uses_an_image_target_isolated_from_ui_layers() {
        let output = RenderLayers::layer(AVATAR_RENDER_LAYER);
        let ground = RenderLayers::layer(VIEWPORT_ONLY_RENDER_LAYER);
        assert!(!output.intersects(&ground));
        let mut app = output_sync_app(AvatarGeneration(1));
        let entity = spawn_mirrored_cameras(
            &mut app,
            Transform::from_xyz(0.0, 0.0, 2.5).looking_at(Vec3::ZERO, Vec3::Y),
            PerspectiveProjection::default(),
        );
        app.update();
        assert!(matches!(
            app.world().get::<RenderTarget>(entity),
            Some(RenderTarget::Image(_))
        ));
        assert_eq!(
            app.world().get::<RenderLayers>(entity),
            Some(&RenderLayers::layer(AVATAR_RENDER_LAYER))
        );
    }

    #[test]
    fn avatar_renderables_intersect_the_output_layer() {
        let avatar = RenderLayers::layer(AVATAR_RENDER_LAYER);
        let output = RenderLayers::layer(AVATAR_RENDER_LAYER);
        let ground = RenderLayers::layer(VIEWPORT_ONLY_RENDER_LAYER);
        assert!(output.intersects(&avatar));
        assert!(!output.intersects(&ground));
        assert!(!ground.intersects(&avatar));
    }
}
