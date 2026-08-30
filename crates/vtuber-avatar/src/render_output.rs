//! Transparent avatar-only render target and asynchronous GPU readback.
//!
//! This module owns the Bevy side of the output boundary. It deliberately
//! exposes only the transport-neutral [`vtuber_core::VideoOutputFrame`] and a
//! latest-value slot; no network or NDI type enters the avatar crate.

use bevy::camera::{CameraUpdateSystems, ClearColorConfig, RenderTarget, visibility::RenderLayers};
use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{TextureFormat, TextureUsages};
use bevy::render::renderer::RenderDevice;
use vtuber_core::{FrameSeq, VideoOutputFrame, VideoOutputProfile, monotonic_now};

use crate::lifecycle::AvatarGeneration;

/// The rendering layer containing avatar geometry and output lighting.
pub const AVATAR_RENDER_LAYER: usize = 0;
/// The main-window-only layer containing the ground plane.
pub const VIEWPORT_ONLY_RENDER_LAYER: usize = 1;

/// Fixed render target and profile used by the output camera.
#[derive(Resource, Clone, Debug)]
pub struct AvatarOutputTarget {
    image: Handle<Image>,
    profile: VideoOutputProfile,
}

impl AvatarOutputTarget {
    /// Returns the fixed transport-neutral output profile.
    #[must_use]
    pub const fn profile(&self) -> VideoOutputProfile {
        self.profile
    }

    /// Returns the render-target image handle for renderer integration.
    #[must_use]
    pub fn image(&self) -> &Handle<Image> {
        &self.image
    }
}

/// Runtime activation state for the transparent output camera/readback.
///
/// The default is inactive. Toggling this resource does not affect camera
/// capture, tracking, or the main avatar viewport.
#[derive(Resource, Clone, Debug, Default)]
pub struct AvatarOutputState {
    active: bool,
    profile: VideoOutputProfile,
}

impl AvatarOutputState {
    /// Returns whether the offscreen camera and readback are active.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.active
    }

    /// Returns the fixed output profile.
    #[must_use]
    pub const fn profile(&self) -> VideoOutputProfile {
        self.profile
    }

    /// Activates or deactivates the output camera/readback lifecycle.
    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    /// Activates transparent output.
    pub fn activate(&mut self) {
        self.set_active(true);
    }

    /// Deactivates transparent output.
    pub fn deactivate(&mut self) {
        self.set_active(false);
    }

    /// Creates an inactive output state with a caller-selected profile.
    ///
    /// Production uses [`VideoOutputProfile::DEFAULT`]. Validators may choose a
    /// smaller target so GPU pixel contracts can be exercised without changing
    /// the application profile.
    #[must_use]
    pub const fn with_profile(profile: VideoOutputProfile) -> Self {
        Self {
            active: false,
            profile,
        }
    }
}

/// A latest-value slot for completed output frames.
///
/// A slow consumer can replace one pending frame, but cannot grow a queue.
#[derive(Resource, Default, Debug)]
pub struct AvatarOutputFrameSlot {
    latest: Option<VideoOutputFrame>,
    next_frame_seq: u64,
    received_frames: u64,
    replaced_frames: u64,
    rejected_frames: u64,
}

impl AvatarOutputFrameSlot {
    /// Takes the newest completed frame, if one is pending.
    pub fn take_latest(&mut self) -> Option<VideoOutputFrame> {
        self.latest.take()
    }

    /// Returns the newest completed frame without removing it.
    #[must_use]
    pub fn latest(&self) -> Option<&VideoOutputFrame> {
        self.latest.as_ref()
    }

    /// Number of successfully converted readback frames.
    #[must_use]
    pub const fn received_frames(&self) -> u64 {
        self.received_frames
    }

    /// Number of completed frames replaced before a consumer took them.
    #[must_use]
    pub const fn replaced_frames(&self) -> u64 {
        self.replaced_frames
    }

    /// Number of malformed readbacks rejected at the contract boundary.
    #[must_use]
    pub const fn rejected_frames(&self) -> u64 {
        self.rejected_frames
    }

    /// Publishes a completed frame into the capacity-one slot.
    ///
    /// The GPU observer is the production caller. Tests use the same path to
    /// prove replacement and orchestration contracts without a GPU.
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

/// Read-only snapshot of the current main avatar viewport camera.
///
/// The snapshot contains no Bevy entity ID and is updated only after the
/// main camera's framing/manual controls have produced their current state.
#[derive(Resource, Clone, Debug, Default)]
pub struct AvatarViewportSnapshot {
    /// Avatar lifecycle generation associated with this camera state.
    pub generation: AvatarGeneration,
    /// Current viewport camera transform.
    pub transform: Option<Transform>,
    /// Current perspective projection values.
    pub projection: Option<PerspectiveProjection>,
}

/// Marks the dedicated transparent output camera.
#[derive(Component, Debug)]
pub struct AvatarOutputCamera;

/// Internal gate that allows only one GPU readback to be in flight.
#[derive(Component, Debug)]
struct AvatarOutputReadbackInFlight;

#[derive(SystemParam)]
struct OutputCameraQuery<'w, 's> {
    // This tuple intentionally keeps all output-camera writes in one query so
    // the camera, projection, transform, and bounded-readback gate change as
    // one synchronized boundary.
    #[allow(clippy::type_complexity)]
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

/// Creates the fixed transparent target and output camera.
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
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
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

/// Mirrors the main viewport camera into the fixed offscreen camera.
///
/// This runs after avatar framing and after transform propagation. The output
/// camera has no parent, so its global transform can be committed immediately
/// and the render extractor sees the same state in this frame.
#[allow(clippy::type_complexity)]
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

    snapshot.transform = Some(*main_transform);
    snapshot.projection = Some(main_projection.clone());

    for (entity, mut camera, mut projection, mut transform, mut global_transform, in_flight) in
        &mut output_cameras.cameras
    {
        *transform = *main_transform;
        *global_transform = GlobalTransform::from(*main_transform);
        *projection = Projection::Perspective(main_projection.clone());
        camera.is_active = state.is_active();
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

/// Adds the output lifecycle systems to an existing avatar app.
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
    use super::*;

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
        assert_eq!(state.profile(), VideoOutputProfile::DEFAULT);
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
            .expect("one transparent pixel is valid")
        };
        slot.publish(make(0));
        slot.publish(make(1));
        assert_eq!(slot.replaced_frames(), 1);
        assert_eq!(
            slot.take_latest().expect("latest frame").frame_seq,
            FrameSeq(1)
        );
        assert!(slot.take_latest().is_none());
    }

    #[test]
    fn output_camera_mirrors_the_current_viewport_state() {
        let mut app = App::new();
        app.init_resource::<crate::lifecycle::AvatarLifecycle>()
            .insert_resource(AvatarOutputState::default())
            .insert_resource(test_target())
            .insert_resource(AvatarViewportSnapshot::default());

        let main_transform = Transform::from_xyz(1.0, 2.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y);
        let main_projection = Projection::Perspective(PerspectiveProjection {
            fov: 0.42,
            aspect_ratio: 1.7,
            ..default()
        });
        app.world_mut().spawn((
            main_transform,
            main_projection,
            crate::framing::AvatarViewportCamera::from_default_transform(main_transform),
        ));
        let output = app
            .world_mut()
            .spawn((
                Camera::default(),
                Projection::Perspective(PerspectiveProjection::default()),
                Transform::default(),
                GlobalTransform::default(),
                AvatarOutputCamera,
            ))
            .id();
        app.add_systems(Update, sync_output_camera);

        app.update();

        assert_eq!(app.world().get::<Transform>(output), Some(&main_transform));
        assert_eq!(
            app.world().get::<GlobalTransform>(output),
            Some(&GlobalTransform::from(main_transform))
        );
        let Projection::Perspective(projection) = app.world().get::<Projection>(output).unwrap()
        else {
            panic!("output camera must remain perspective");
        };
        assert_eq!(projection.fov, 0.42);
        assert_eq!(projection.aspect_ratio, 1.7);
        let snapshot = app.world().resource::<AvatarViewportSnapshot>();
        assert_eq!(snapshot.transform, Some(main_transform));
        let snapshot_projection = snapshot.projection.as_ref().expect("projection snapshot");
        assert_eq!(snapshot_projection.fov, 0.42);
        assert_eq!(snapshot_projection.aspect_ratio, 1.7);
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

    fn camera_entity(app: &mut App, main: bool) -> Entity {
        if main {
            let mut query = app
                .world_mut()
                .query_filtered::<Entity, With<crate::framing::AvatarViewportCamera>>();
            query.iter(app.world()).next().expect("main camera exists")
        } else {
            let mut query = app
                .world_mut()
                .query_filtered::<Entity, With<AvatarOutputCamera>>();
            query
                .iter(app.world())
                .next()
                .expect("output camera exists")
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
        lifecycle
            .request_load(root)
            .expect("load is allowed from NoAvatar");
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
            panic!("output camera must remain perspective");
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
        .expect("initial pose is valid");
        let orbited = crate::framing::camera_control::geometry::orbit(pose, 0.3, 0.1)
            .expect("orbit is valid");
        *app.world_mut()
            .get_mut::<Transform>(main)
            .expect("main transform") = orbited.transform();
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
        .expect("pan is valid");
        *app.world_mut()
            .get_mut::<Transform>(main)
            .expect("main transform") = panned.transform();
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
        .expect("dolly is valid");
        *app.world_mut()
            .get_mut::<Transform>(main)
            .expect("main transform") = dollied.transform();
        app.update();
        assert_eq!(
            app.world().get::<Transform>(output),
            Some(&dollied.transform())
        );

        *app.world_mut()
            .get_mut::<Transform>(main)
            .expect("main transform") = initial;
        app.update();
        assert_eq!(app.world().get::<Transform>(output), Some(&initial));
        let Projection::Perspective(mirrored) = app.world().get::<Projection>(output).unwrap()
        else {
            panic!("output camera must remain perspective");
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
        spawn_mirrored_cameras(&mut app, first, projection.clone());
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
        {
            let mut lifecycle = app
                .world_mut()
                .resource_mut::<crate::lifecycle::AvatarLifecycle>();
            lifecycle
                .request_replace(next_root)
                .expect("ready avatar can be replaced");
        }
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
        *app.world_mut()
            .get_mut::<Transform>(main)
            .expect("main transform") = replacement;
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
        assert!(app.world().get::<Camera>(output).expect("camera").is_active);
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
        assert!(!app.world().get::<Camera>(output).expect("camera").is_active);
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
    fn output_camera_uses_an_image_target_isolated_from_ui_layers() {
        let output = RenderLayers::layer(AVATAR_RENDER_LAYER);
        let ground = RenderLayers::layer(VIEWPORT_ONLY_RENDER_LAYER);
        assert!(
            !output.intersects(&ground),
            "ground stays on the viewport-only layer"
        );

        let mut app = output_sync_app(AvatarGeneration(1));
        let output_entity = spawn_mirrored_cameras(
            &mut app,
            Transform::from_xyz(0.0, 0.0, 2.5).looking_at(Vec3::ZERO, Vec3::Y),
            PerspectiveProjection::default(),
        );
        app.update();
        assert!(matches!(
            app.world().get::<RenderTarget>(output_entity),
            Some(RenderTarget::Image(_))
        ));
        assert_eq!(
            app.world().get::<RenderLayers>(output_entity),
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
