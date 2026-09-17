//! GPU-backed render fixtures for the rich-look epic (Issues #69-#77).
//!
//! These commands are not substitutes for unit tests. They render synthetic
//! `MToonMaterial`/`StandardMaterial` scenes through the production offscreen
//! camera and readback path and inspect CPU pixels, so that the WGSL actually
//! compiles and that lighting/material changes reach the framebuffer.
//! When a GPU or readback completion is unavailable the command exits 2
//! (`NOT RUN`) instead of reporting success.

use bevy::app::AppExit;
use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::pipelined_rendering::PipelinedRenderingPlugin;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::winit::WinitPlugin;
use bevy_vrm1::prelude::{MToonMaterial, MtoonMaterialPlugin, Shade};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use vtuber_avatar::{
    AVATAR_RENDER_LAYER, AvatarOutputFrameSlot, AvatarOutputState, AvatarViewportCamera,
    register_output_systems,
};
use vtuber_core::{VideoOutputFrame, VideoOutputProfile};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;
const MAX_WAIT: Duration = Duration::from_secs(30);

/// Exit code used when the GPU/readback path cannot be exercised.
pub const EXIT_NOT_RUN: i32 = 2;

/// A single directional light in a fixture scene.
#[derive(Clone, Copy)]
struct LightSpec {
    /// Direction the light travels (intensity is `Color` × `illuminance`).
    direction: Vec3,
    color: Color,
    illuminance: f32,
    /// Whether this light renders shadow maps.
    shadows_enabled: bool,
}

/// Mesh used by a fixture scene.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MeshSpec {
    /// Unit plane facing the camera (normal `+Z`).
    Plane,
    /// Unit sphere centered at the origin.
    Sphere,
}

/// Declarative description of one MToon render.
#[derive(Clone)]
struct MtoonScene {
    lights: Vec<LightSpec>,
    base_color: Color,
    shade_color: Color,
    toony_factor: f32,
    shading_shift_factor: f32,
    mesh: MeshSpec,
    /// Whether a lit ground plane is placed below the subject.
    ground: bool,
    /// Optional constant normal texture, one RGBA pixel.
    normal_map: Option<[u8; 4]>,
    normal_scale: f32,
}

impl MtoonScene {
    fn lit(base_color: Color, shade_color: Color, light: LightSpec) -> Self {
        Self {
            lights: vec![light],
            base_color,
            shade_color,
            toony_factor: 0.9,
            shading_shift_factor: 0.0,
            mesh: MeshSpec::Plane,
            ground: false,
            normal_map: None,
            normal_scale: 1.0,
        }
    }
}

/// Run one rich-look fixture.
pub fn run(args: &[String]) -> Result<(), String> {
    let case = args.first().map(String::as_str).unwrap_or("help");
    let evidence = args
        .iter()
        .position(|argument| argument == "--evidence")
        .and_then(|index| args.get(index + 1))
        .map(PathBuf::from);

    let result = match case {
        "mtoon-lighting" => mtoon_lighting(),
        "mtoon-shading" => mtoon_shading(),
        "mtoon-shadow" => mtoon_shadow(),
        "mtoon-cutout-shadow" => mtoon_cutout_shadow(),
        "studio-environment" => studio_environment(),
        "mtoon-vs-standard" => mtoon_vs_standard(),
        "mtoon-normal" => mtoon_normal(),
        "help" | "--help" | "-h" => {
            println!("cargo xtask rich-look <case> [--evidence <file>]");
            println!("cases:");
            println!("  mtoon-lighting  directional-light color/intensity response");
            println!("  mtoon-shading   signed NdotL, shading shift and toony endpoints");
            println!("  mtoon-normal    normal texture, scale and TBN wiring");
            return Ok(());
        }
        other => return Err(format!("unknown rich-look case: {other}")),
    };

    let mut report = match result {
        Ok(report) => report,
        Err(RichLookError::NotRun(reason)) => {
            if let Some(path) = evidence {
                write_evidence(&path, &format!("result=NOT RUN\nreason={reason}\n"))?;
            }
            return Err(format!("NOT RUN: {reason}"));
        }
        Err(RichLookError::Failed(reason)) => {
            if let Some(path) = evidence {
                write_evidence(&path, &format!("result=FAIL\nreason={reason}\n"))?;
            }
            return Err(reason);
        }
    };
    report.push_str("result=PASS\n");
    if let Some(path) = evidence {
        write_evidence(&path, &report)?;
    }
    print!("{report}");
    println!("rich-look {case}: PASS");
    Ok(())
}

#[derive(Debug)]
enum RichLookError {
    NotRun(String),
    Failed(String),
}

fn write_evidence(path: &PathBuf, contents: &str) -> Result<(), String> {
    std::fs::write(path, contents)
        .map_err(|error| format!("cannot write rich-look evidence: {error}"))
}

/// The MToon direct term must follow each light's linear radiance.
///
/// The fixture plane's normal faces the camera (`+Z`), so a light travelling
/// toward `-Z` is a front light (`direction_to_light = +Z`).
fn mtoon_lighting() -> Result<String, RichLookError> {
    let white = |illuminance| LightSpec {
        direction: Vec3::NEG_Z,
        color: Color::WHITE,
        illuminance,
        shadows_enabled: false,
    };
    let sample = |scene: &MtoonScene| -> Result<[u8; 4], RichLookError> {
        Ok(center_pixel(&render(scene)?))
    };

    let base = MtoonScene::lit(Color::WHITE, Color::BLACK, white(0.0));
    let dark = sample(&base)?;
    let one = sample(&MtoonScene {
        lights: vec![white(200.0)],
        ..base.clone()
    })?;
    let two = sample(&MtoonScene {
        lights: vec![white(400.0)],
        ..base.clone()
    })?;
    let red = sample(&MtoonScene {
        lights: vec![LightSpec {
            direction: Vec3::NEG_Z,
            color: Color::srgb(1.0, 0.0, 0.0),
            illuminance: 200.0,
            shadows_enabled: false,
        }],
        ..base.clone()
    })?;
    let blue = sample(&MtoonScene {
        lights: vec![LightSpec {
            direction: Vec3::NEG_Z,
            color: Color::srgb(0.0, 0.0, 1.0),
            illuminance: 200.0,
            shadows_enabled: false,
        }],
        ..base.clone()
    })?;
    let split = sample(&MtoonScene {
        lights: vec![white(100.0), white(100.0)],
        ..base.clone()
    })?;
    let single_half = sample(&MtoonScene {
        lights: vec![white(100.0)],
        ..base.clone()
    })?;

    let mut report = format!(
        "case=mtoon-lighting\n\
         zero_light={dark:?}\n\
         one_light={one:?}\n\
         two_lights={two:?}\n\
         red_light={red:?}\n\
         blue_light={blue:?}\n\
         split_lights={split:?}\n\
         single_half_light={single_half:?}\n"
    );

    if luma(dark) != 0 {
        return Err(RichLookError::Failed(format!(
            "a zero-illuminance light still lit the material: {dark:?}"
        )));
    }
    if luma(one) <= luma(dark) || luma(two) <= luma(one) {
        return Err(RichLookError::Failed(format!(
            "light intensity did not increase the direct term: dark={dark:?} one={one:?} two={two:?}"
        )));
    }
    if red[2] <= red[0] || blue[0] <= blue[2] {
        return Err(RichLookError::Failed(format!(
            "light color was not applied per light: red={red:?} blue={blue:?}"
        )));
    }
    if luma(split) <= luma(single_half) {
        return Err(RichLookError::Failed(format!(
            "a second light did not add to the first: split={split:?} single={single_half:?}"
        )));
    }
    report.push_str("checks=intensity_monotonic,color_per_light,lights_additive\n");
    Ok(report)
}

/// The shading ramp follows the signed NdotL, the shift and the toony endpoint.
fn mtoon_shading() -> Result<String, RichLookError> {
    let light = LightSpec {
        direction: Vec3::NEG_Z,
        color: Color::WHITE,
        illuminance: 200.0,
        shadows_enabled: false,
    };
    let base = MtoonScene::lit(Color::WHITE, Color::BLACK, light);
    // A light travelling toward `-X` meets the plane's `+Z` normal at exactly
    // 90 degrees, so the untouched ramp sits on the middle of its range.
    let side_light = LightSpec {
        direction: Vec3::NEG_X,
        ..light
    };

    let front = center_pixel(&render(&base)?);
    let side = center_pixel(&render(&MtoonScene {
        lights: vec![side_light],
        ..base.clone()
    })?);
    let shifted_side = center_pixel(&render(&MtoonScene {
        lights: vec![side_light],
        shading_shift_factor: 0.5,
        ..base.clone()
    })?);
    let behind = center_pixel(&render(&MtoonScene {
        lights: vec![LightSpec {
            direction: Vec3::Z,
            ..light
        }],
        ..base.clone()
    })?);

    let gradient = scanline(&render(&MtoonScene {
        lights: vec![side_light],
        mesh: MeshSpec::Sphere,
        toony_factor: 0.0,
        ..base.clone()
    })?);
    let stepped = scanline(&render(&MtoonScene {
        lights: vec![side_light],
        mesh: MeshSpec::Sphere,
        toony_factor: 1.0,
        ..base.clone()
    })?);

    let mut report = format!(
        "case=mtoon-shading\n\
         front_lit={front:?}\n\
         side_90deg={side:?}\n\
         side_shifted={shifted_side:?}\n\
         light_behind={behind:?}\n\
         gradient_intermediate={gradient}\n\
         step_intermediate={stepped}\n"
    );

    if luma(front) <= luma(side) || luma(side) <= luma(behind) {
        return Err(RichLookError::Failed(format!(
            "the signed NdotL ramp was not monotonic: front={front:?} side={side:?} behind={behind:?}"
        )));
    }
    if luma(shifted_side) <= luma(side) {
        return Err(RichLookError::Failed(format!(
            "a positive shade shift did not move the boundary toward lit: side={side:?} shifted={shifted_side:?}"
        )));
    }
    if gradient < 8 {
        return Err(RichLookError::Failed(format!(
            "the toony=0 ramp produced only {gradient} intermediate samples"
        )));
    }
    if stepped * 4 >= gradient {
        return Err(RichLookError::Failed(format!(
            "the toony=1 endpoint was not a step: gradient={gradient} step={stepped}"
        )));
    }
    report.push_str("checks=signed_ndotl,shade_behind,positive_shift,toony_step\n");
    Ok(report)
}

/// A fully lit MToon surface must render at the same brightness as the
/// standard material under the same light, so switching the look off keeps the
/// standard display as the comparison basis.
fn mtoon_vs_standard() -> Result<String, RichLookError> {
    let mtoon = center_pixel(&render(&MtoonScene::lit(
        Color::WHITE,
        Color::BLACK,
        LightSpec {
            direction: Vec3::NEG_Z,
            color: Color::WHITE,
            illuminance: 200.0,
            shadows_enabled: false,
        },
    ))?);
    let standard = center_pixel(&standard_scene()?);
    let mut report = format!(
        "case=mtoon-vs-standard\n\
         mtoon_lit={mtoon:?}\n\
         standard_lit={standard:?}\n"
    );
    let difference = luma(mtoon).abs_diff(luma(standard));
    if difference > 24 {
        return Err(RichLookError::Failed(format!(
            "MToon and the standard material disagree at the same lighting: mtoon={mtoon:?} standard={standard:?}"
        )));
    }
    report.push_str("checks=lambert_normalization_matches_standard\n");
    Ok(report)
}

/// The same plane as `mtoon-vs-standard`, rendered with `StandardMaterial`.
fn standard_scene() -> Result<Vec<[u8; 4]>, RichLookError> {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: bevy::window::ExitCondition::DontExit,
                ..default()
            })
            .set(RenderPlugin { ..default() })
            .disable::<PipelinedRenderingPlugin>()
            .disable::<WinitPlugin>()
            .disable::<bevy::log::LogPlugin>(),
    )
    .insert_resource(AvatarOutputState::with_profile(VideoOutputProfile {
        width: WIDTH,
        height: HEIGHT,
        fps: 60,
        pixel_format: vtuber_core::VideoOutputPixelFormat::Bgra8StraightAlpha,
    }))
    .insert_resource(GlobalAmbientLight {
        brightness: 0.0,
        ..default()
    })
    .insert_resource(vtuber_avatar::AvatarLifecycle::default())
    .insert_resource(OutputArmed(false));
    register_output_systems(&mut app);
    app.add_systems(Startup, setup_standard_scene);
    app.add_systems(Update, activate_output_after_setup);
    app.finish();
    app.cleanup();
    collect_settled_frame(&mut app)
}

fn collect_settled_frame(app: &mut App) -> Result<Vec<[u8; 4]>, RichLookError> {
    let deadline = Instant::now() + MAX_WAIT;
    let mut satisfied = 0;
    let mut last = None;
    while Instant::now() < deadline {
        app.update();
        if let Some(frame) = app
            .world_mut()
            .resource_mut::<AvatarOutputFrameSlot>()
            .take_latest()
        {
            let sampled = pixels(&frame);
            satisfied = if sampled.iter().any(|pixel| pixel[3] > 0) {
                satisfied + 1
            } else {
                0
            };
            if satisfied >= SETTLE_FRAMES {
                return Ok(sampled);
            }
            last = Some(sampled);
        }
    }
    last.ok_or(RichLookError::NotRun(
        "GPU readback did not complete; the local renderer/GPU path is unavailable".into(),
    ))
}

fn setup_standard_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        DirectionalLight {
            illuminance: 200.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_rotation_arc(Vec3::NEG_Z, Vec3::NEG_Z)),
        RenderLayers::layer(AVATAR_RENDER_LAYER),
    ));
    commands.spawn((
        Mesh3d(meshes.add(
            Plane3d::default()
                .mesh()
                .size(4.0, 4.0)
                .build()
                .rotated_by(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
        )),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 0.9,
            ..default()
        })),
        Transform::default(),
        RenderLayers::layer(AVATAR_RENDER_LAYER),
    ));
    let camera_transform =
        Transform::from_translation(Vec3::new(0.0, 0.0, 5.0)).looking_at(Vec3::ZERO, Vec3::Y);
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: 1.0,
            ..default()
        }),
        AvatarViewportCamera::from_default_transform(camera_transform),
        camera_transform,
        RenderLayers::layer(AVATAR_RENDER_LAYER),
    ));
}

/// The bundled studio cubemap must survive Bevy's environment-map filter and
/// add image-based light to a PBR surface.
fn studio_environment() -> Result<String, RichLookError> {
    let textured = environment_scene(300.0)?;
    let unlit_environment = environment_scene(0.0)?;
    let with = center_pixel(&textured);
    let without = center_pixel(&unlit_environment);

    let mut report = format!(
        "case=studio-environment\n\
         center_with_environment={with:?}\n\
         center_without_environment={without:?}\n"
    );
    if luma(with) <= luma(without) + 8 {
        return Err(RichLookError::Failed(format!(
            "the generated studio environment did not light the surface: with={with:?} without={without:?}"
        )));
    }
    report.push_str("checks=environment_filtered_and_applied\n");
    Ok(report)
}

/// Renders a lit PBR sphere under the studio environment at the given
/// intensity. The source cubemap must be square power-of-two, which the
/// bundled one is, or Bevy's environment filter panics.
fn environment_scene(intensity: f32) -> Result<Vec<[u8; 4]>, RichLookError> {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: bevy::window::ExitCondition::DontExit,
                ..default()
            })
            .set(RenderPlugin { ..default() })
            .disable::<PipelinedRenderingPlugin>()
            .disable::<WinitPlugin>()
            .disable::<bevy::log::LogPlugin>(),
    )
    .insert_resource(AvatarOutputState::with_profile(VideoOutputProfile {
        width: WIDTH,
        height: HEIGHT,
        fps: 60,
        pixel_format: vtuber_core::VideoOutputPixelFormat::Bgra8StraightAlpha,
    }))
    .insert_resource(GlobalAmbientLight {
        brightness: 0.0,
        ..default()
    })
    .insert_resource(vtuber_avatar::AvatarLifecycle::default())
    .insert_resource(EnvironmentScene { intensity })
    .init_resource::<EnvironmentMapHandle>()
    .insert_resource(OutputArmed(false));
    register_output_systems(&mut app);
    app.add_systems(Startup, setup_environment_scene);
    app.add_systems(Update, (activate_output_after_setup, attach_environment));
    app.finish();
    app.cleanup();

    let deadline = Instant::now() + MAX_WAIT;
    let mut satisfied = 0;
    let mut last = None;
    while Instant::now() < deadline {
        app.update();
        // The filter only inserts `EnvironmentMapLight` once the GPU prefilter
        // has produced the diffuse/specular maps.
        let generated = app
            .world_mut()
            .query::<&EnvironmentMapLight>()
            .iter(app.world())
            .next()
            .is_some();
        if let Some(frame) = app
            .world_mut()
            .resource_mut::<AvatarOutputFrameSlot>()
            .take_latest()
        {
            let sampled = pixels(&frame);
            satisfied = if sampled.iter().any(|pixel| pixel[3] > 0) {
                satisfied + 1
            } else {
                0
            };
            if satisfied >= 40 && generated {
                return Ok(sampled);
            }
            last = Some(sampled);
        }
    }
    last.ok_or(RichLookError::NotRun(
        "GPU readback did not complete; the local renderer/GPU path is unavailable".into(),
    ))
}

#[derive(Resource, Clone, Copy)]
struct EnvironmentScene {
    intensity: f32,
}

#[derive(Resource, Default)]
struct EnvironmentMapHandle(Option<Handle<Image>>);

/// The attachment point: the avatar is drawn by the offscreen output camera,
/// so the environment must reach that camera too, not only the viewport one.
// The camera query is a small, fixed two-marker union.\r
#[allow(clippy::type_complexity)]
fn attach_environment(
    mut commands: Commands,
    scene: Res<EnvironmentScene>,
    map: Res<EnvironmentMapHandle>,
    cameras: Query<
        (Entity, Option<&GeneratedEnvironmentMapLight>),
        Or<(With<AvatarViewportCamera>, With<vtuber_avatar::AvatarOutputCamera>)>,
    >,
) {
    let Some(environment_map) = map.0.clone() else {
        return;
    };
    for (entity, existing) in &cameras {
        if existing.is_none() {
            commands.entity(entity).insert(GeneratedEnvironmentMapLight {
                environment_map: environment_map.clone(),
                intensity: scene.intensity,
                ..default()
            });
        }
    }
}

fn setup_environment_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    scene: Res<EnvironmentScene>,
    mut map: ResMut<EnvironmentMapHandle>,
) {
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(1.0).mesh().build())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 0.4,
            ..default()
        })),
        Transform::default(),
        RenderLayers::layer(AVATAR_RENDER_LAYER),
    ));
    let camera_transform =
        Transform::from_translation(Vec3::new(0.0, 0.0, 5.0)).looking_at(Vec3::ZERO, Vec3::Y);
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: 1.0,
            ..default()
        }),
        AvatarViewportCamera::from_default_transform(camera_transform),
        camera_transform,
        RenderLayers::layer(AVATAR_RENDER_LAYER),
    ));
    if scene.intensity > 0.0 {
        map.0 = Some(images.add(vtuber_avatar::look::studio_environment_cubemap()));
    }
}

/// A mask (cutout) MToon material must cut its shadow too: the ground under the
/// opaque half is shadowed while the ground under the transparent half stays
/// lit, so the shadow is not a solid quad.
fn mtoon_cutout_shadow() -> Result<String, RichLookError> {
    let casting = cutout_shadow_scene(true)?;
    let flat = cutout_shadow_scene(false)?;
    let (left_shadowed, right_shadowed) = (ground_band(&casting, 26), ground_band(&casting, 38));
    let (left_lit, right_lit) = (ground_band(&flat, 26), ground_band(&flat, 38));

    let mut report = format!(
        "case=mtoon-cutout-shadow\n\
         with_shadows_left={left_shadowed} right={right_shadowed}\n\
         without_shadows_left={left_lit} right={right_lit}\n"
    );
    if left_lit < 40 || right_lit < 40 {
        return Err(RichLookError::Failed(format!(
            "the ground was not lit at all: left={left_lit} right={right_lit}"
        )));
    }
    let left_dark = left_shadowed + 8 < left_lit;
    let right_dark = right_shadowed + 8 < right_lit;
    match (left_dark, right_dark) {
        (true, true) => Err(RichLookError::Failed(
            "both halves were shadowed; the cutout alpha did not reach the shadow pass".into(),
        )),
        (false, false) => Err(RichLookError::Failed(format!(
            "the quad cast no shadow on the ground: left={left_shadowed} right={right_shadowed}"
        ))),
        _ => {
            report.push_str("checks=cutout_alpha_in_shadow_pass\n");
            Ok(report)
        }
    }
}

/// Renders the cutout-shadow scene with an alpha-masked MToon quad standing on
/// a lit ground. Returns the sampled frame.
fn cutout_shadow_scene(shadows: bool) -> Result<Vec<[u8; 4]>, RichLookError> {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: bevy::window::ExitCondition::DontExit,
                ..default()
            })
            .set(RenderPlugin { ..default() })
            .disable::<PipelinedRenderingPlugin>()
            .disable::<WinitPlugin>()
            .disable::<bevy::log::LogPlugin>(),
    )
    .add_plugins(MtoonMaterialPlugin)
    .insert_resource(AvatarOutputState::with_profile(VideoOutputProfile {
        width: WIDTH,
        height: HEIGHT,
        fps: 60,
        pixel_format: vtuber_core::VideoOutputPixelFormat::Bgra8StraightAlpha,
    }))
    .insert_resource(GlobalAmbientLight {
        brightness: 0.0,
        ..default()
    })
    .insert_resource(vtuber_avatar::AvatarLifecycle::default())
    .insert_resource(CutoutScene { shadows })
    .insert_resource(OutputArmed(false));
    register_output_systems(&mut app);
    app.add_systems(Startup, setup_cutout_scene);
    app.add_systems(Update, activate_output_after_setup);
    app.finish();
    app.cleanup();

    let deadline = Instant::now() + MAX_WAIT;
    let mut satisfied = 0;
    let mut last = None;
    while Instant::now() < deadline {
        app.update();
        if let Some(frame) = app
            .world_mut()
            .resource_mut::<AvatarOutputFrameSlot>()
            .take_latest()
        {
            let sampled = pixels(&frame);
            satisfied = if sampled.iter().any(|pixel| pixel[3] > 0) {
                satisfied + 1
            } else {
                0
            };
            if satisfied >= SETTLE_FRAMES {
                return Ok(sampled);
            }
            last = Some(sampled);
        }
    }
    last.ok_or(RichLookError::NotRun(
        "GPU readback did not complete; the local renderer/GPU path is unavailable".into(),
    ))
}

#[derive(Resource, Clone, Copy)]
struct CutoutScene {
    shadows: bool,
}

fn setup_cutout_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<MToonMaterial>>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    scene: Res<CutoutScene>,
) {
    commands.spawn((
        DirectionalLight {
            illuminance: 300.0,
            shadow_maps_enabled: scene.shadows,
            ..default()
        },
        Transform::from_rotation(Quat::from_rotation_arc(
            Vec3::NEG_Z,
            Vec3::new(0.0, -1.0, 0.5).normalize(),
        )),
        RenderLayers::layer(AVATAR_RENDER_LAYER),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(8.0, 8.0).build())),
        MeshMaterial3d(standard_materials.add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 0.9,
            ..default()
        })),
        Transform::from_xyz(0.0, -1.2, 0.0),
        RenderLayers::layer(AVATAR_RENDER_LAYER),
    ));

    // A two-texel base color texture: one opaque texel, one fully transparent.
    let mut mask = Image::new_fill(
        Extent3d {
            width: 2,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[255, 255, 255, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    mask.data = Some(vec![255, 255, 255, 255, 255, 255, 255, 0]);
    let mask = images.add(mask);

    let mut quad = Plane3d::default()
        .mesh()
        .size(2.0, 2.0)
        .build()
        .rotated_by(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2));
    let _ = quad.generate_tangents();
    commands.spawn((
        Mesh3d(meshes.add(quad)),
        MeshMaterial3d(materials.add(MToonMaterial {
            base_color_texture: Some(mask),
            alpha_mode: AlphaMode::Mask(0.5),
            ..default()
        })),
        Transform::from_xyz(0.0, 1.4, 0.0),
        RenderLayers::layer(AVATAR_RENDER_LAYER),
    ));

    let camera_transform =
        Transform::from_translation(Vec3::new(0.0, 0.0, 5.0)).looking_at(Vec3::ZERO, Vec3::Y);
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: 1.0,
            ..default()
        }),
        AvatarViewportCamera::from_default_transform(camera_transform),
        camera_transform,
        RenderLayers::layer(AVATAR_RENDER_LAYER),
    ));
}

/// The darkest luminance in the ground rows around one screen column.
// Bounds are guaranteed by construction (row/column ranges are derived from\r
// the fixed output dimensions); see the AGENTS.md production panic policy.\r
#[allow(clippy::indexing_slicing)]
fn ground_band(pixels: &[[u8; 4]], column: u32) -> u32 {
    let mut darkest = u32::MAX;
    for y in 45..56u32 {
        for x in column.saturating_sub(3)..column + 4 {
            let pixel = pixels[(y * WIDTH + x) as usize];
            if pixel[3] == 0 {
                continue;
            }
            darkest = darkest.min(luma(pixel));
        }
    }
    darkest
}

/// An MToon mesh must cast into the shadow map and the lit ground must receive
/// it: the darkest ground pixel is compared with the same scene without shadow
/// maps.
fn mtoon_shadow() -> Result<String, RichLookError> {
    let light = LightSpec {
        direction: Vec3::new(0.25, -1.0, -0.15).normalize(),
        color: Color::WHITE,
        illuminance: 300.0,
        shadows_enabled: true,
    };
    let base = MtoonScene {
        lights: vec![light],
        mesh: MeshSpec::Sphere,
        ground: true,
        ..MtoonScene::lit(Color::WHITE, Color::BLACK, light)
    };

    let casting = render(&base)?;
    let flat = render(&MtoonScene {
        lights: vec![LightSpec {
            shadows_enabled: false,
            ..light
        }],
        ..base.clone()
    })?;
    let shadowed_floor = darkest_ground_luma(&casting);
    let lit_floor = darkest_ground_luma(&flat);

    let mut report = format!(
        "case=mtoon-shadow\n\
         darkest_ground_with_shadows={shadowed_floor}\n\
         darkest_ground_without_shadows={lit_floor}\n"
    );
    if lit_floor < 40 {
        return Err(RichLookError::Failed(format!(
            "the ground was not lit at all: {lit_floor}"
        )));
    }
    if shadowed_floor + 8 > lit_floor {
        return Err(RichLookError::Failed(format!(
            "the MToon sphere did not darken the ground below it: with={shadowed_floor} without={lit_floor}"
        )));
    }
    report.push_str("checks=mtoon_casts_shadow,ground_receives_shadow\n");
    Ok(report)
}

/// The darkest luminance in the rows that show the ground plane.
// Bounds are guaranteed by construction (row/column ranges are derived from\r
// the fixed output dimensions); see the AGENTS.md production panic policy.\r
#[allow(clippy::indexing_slicing)]
fn darkest_ground_luma(pixels: &[[u8; 4]]) -> u32 {
    let mut darkest = u32::MAX;
    for y in 46..58u32 {
        for x in 4..60u32 {
            let pixel = pixels[(y * WIDTH + x) as usize];
            if pixel[3] == 0 {
                continue;
            }
            darkest = darkest.min(luma(pixel));
        }
    }
    darkest
}

/// A normal texture must tilt the lighting; scale 0 must match no normal map.
fn mtoon_normal() -> Result<String, RichLookError> {
    // A smooth ramp (toony 0) keeps the response proportional to NdotL, so a
    // tilted normal changes the shading whichever way the tangent points.
    // `direction_to_light` points mostly at the camera, so a normal tilted
    // toward `-X` turns the surface away from it.
    let base = MtoonScene {
        toony_factor: 0.0,
        ..MtoonScene::lit(
            Color::WHITE,
            Color::BLACK,
            LightSpec {
                direction: -Vec3::new(0.6, 0.0, 0.8).normalize(),
                color: Color::WHITE,
                illuminance: 200.0,
                shadows_enabled: false,
            },
        )
    };

    let flat = center_pixel(&render(&base)?);
    let identity = center_pixel(&render(&MtoonScene {
        normal_map: Some([128, 128, 255, 255]),
        ..base.clone()
    })?);
    let tilted = center_pixel(&render(&MtoonScene {
        normal_map: Some([0, 128, 255, 255]),
        ..base.clone()
    })?);
    let scaled_out = center_pixel(&render(&MtoonScene {
        normal_map: Some([0, 128, 255, 255]),
        normal_scale: 0.0,
        ..base.clone()
    })?);

    let mut report = format!(
        "case=mtoon-normal\n\
         flat={flat:?}\n\
         identity_map={identity:?}\n\
         tilted_map={tilted:?}\n\
         tilted_scale_zero={scaled_out:?}\n"
    );

    if luma_diff(identity, flat) > 4 {
        return Err(RichLookError::Failed(format!(
            "an identity normal map changed the shading: flat={flat:?} identity={identity:?}"
        )));
    }
    if luma_diff(tilted, flat) <= 4 {
        return Err(RichLookError::Failed(format!(
            "a tilted normal map did not reach the lighting: flat={flat:?} tilted={tilted:?}"
        )));
    }
    if luma_diff(scaled_out, flat) > 4 {
        return Err(RichLookError::Failed(format!(
            "normal scale 0 did not restore the geometric normal: flat={flat:?} scaled={scaled_out:?}"
        )));
    }
    report.push_str("checks=identity_unchanged,tilt_applied,scale_zero_matches_flat\n");
    Ok(report)
}
fn luma(pixel: [u8; 4]) -> u32 {
    u32::from(pixel[0]) + u32::from(pixel[1]) + u32::from(pixel[2])
}

fn luma_diff(a: [u8; 4], b: [u8; 4]) -> u32 {
    luma(a).abs_diff(luma(b))
}

/// Count the pixels on the middle scanline that are neither fully lit nor
/// fully shaded, i.e. the softness of the shading ramp.
// Bounds are guaranteed by construction in this numeric kernel
// (the scanline index is derived from the fixed output dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn scanline(pixels: &[[u8; 4]]) -> usize {
    let row: Vec<u32> = (0..WIDTH)
        .map(|x| luma(pixels[(HEIGHT / 2 * WIDTH + x) as usize]))
        .collect();
    let min = row.iter().copied().min().unwrap_or(0);
    let max = row.iter().copied().max().unwrap_or(0);
    if max <= min + 2 {
        return 0;
    }
    let margin = (max - min) / 8;
    row.iter()
        .filter(|value| **value > min + margin && **value < max - margin)
        .count()
}

/// The renderer needs many frames before the first offscreen image contains
/// geometry (pipeline compilation, asset upload). The scene is static, so the
/// frame is sampled once the image has settled.
const SETTLE_FRAMES: u32 = 3;

fn render(scene: &MtoonScene) -> Result<Vec<[u8; 4]>, RichLookError> {
    let mut app = fixture_app(scene)?;
    let deadline = Instant::now() + MAX_WAIT;
    let mut satisfied = 0;
    let mut last = None;
    while Instant::now() < deadline {
        app.update();
        if let Some(frame) = app
            .world_mut()
            .resource_mut::<AvatarOutputFrameSlot>()
            .take_latest()
        {
            let sampled = pixels(&frame);
            satisfied = if sampled.iter().any(|pixel| pixel[3] > 0) {
                satisfied + 1
            } else {
                0
            };
            if satisfied >= SETTLE_FRAMES {
                return Ok(sampled);
            }
            last = Some(sampled);
        }
        if app
            .world()
            .get_resource::<Messages<AppExit>>()
            .is_some_and(|messages| !messages.is_empty())
        {
            break;
        }
    }
    last.ok_or(RichLookError::NotRun(
        "GPU readback did not complete; the local renderer/GPU path is unavailable".into(),
    ))
}

fn fixture_app(scene: &MtoonScene) -> Result<App, RichLookError> {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: bevy::window::ExitCondition::DontExit,
                ..default()
            })
            .set(RenderPlugin { ..default() })
            .disable::<PipelinedRenderingPlugin>()
            .disable::<WinitPlugin>()
            .disable::<bevy::log::LogPlugin>(),
    )
    .add_plugins(MtoonMaterialPlugin)
    .insert_resource(AvatarOutputState::with_profile(VideoOutputProfile {
        width: WIDTH,
        height: HEIGHT,
        fps: 60,
        pixel_format: vtuber_core::VideoOutputPixelFormat::Bgra8StraightAlpha,
    }))
    // The fixtures measure the direct term, so the default ambient light must
    // not lift the shaded side of the material.
    .insert_resource(GlobalAmbientLight {
        brightness: 0.0,
        ..default()
    })
    .insert_resource(ClearColor(Color::srgba(0.0, 0.0, 0.0, 0.0)))
    .insert_resource(vtuber_avatar::AvatarLifecycle::default())
    .insert_resource(FixtureScene(scene.clone()))
    .insert_resource(OutputArmed(false));
    register_output_systems(&mut app);
    app.add_systems(Startup, setup_fixture_scene);
    app.add_systems(Update, activate_output_after_setup);
    app.finish();
    app.cleanup();
    Ok(app)
}

#[derive(Resource, Clone)]
struct FixtureScene(MtoonScene);

#[derive(Resource, Clone, Copy)]
struct OutputArmed(bool);

fn setup_fixture_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<MToonMaterial>>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    scene: Res<FixtureScene>,
) {
    let scene = &scene.0;
    for light in &scene.lights {
        commands.spawn((
            DirectionalLight {
                illuminance: light.illuminance,
                color: light.color,
                shadow_maps_enabled: light.shadows_enabled,
                ..default()
            },
            Transform::from_rotation(Quat::from_rotation_arc(Vec3::NEG_Z, light.direction)),
            RenderLayers::layer(AVATAR_RENDER_LAYER),
        ));
    }

    if scene.ground {
        commands.spawn((
            Mesh3d(meshes.add(Plane3d::default().mesh().size(8.0, 8.0).build())),
            MeshMaterial3d(standard_materials.add(StandardMaterial {
                base_color: Color::WHITE,
                perceptual_roughness: 0.9,
                ..default()
            })),
            Transform::from_xyz(0.0, -1.2, 0.0),
            RenderLayers::layer(AVATAR_RENDER_LAYER),
        ));
    }

    let mut mesh = match scene.mesh {
        MeshSpec::Plane => Plane3d::default()
            .mesh()
            .size(4.0, 4.0)
            .build()
            .rotated_by(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
        MeshSpec::Sphere => Sphere::new(1.0).mesh().build(),
    };
    // The glTF loader generates tangents for meshes that use a normal map;
    // primitive meshes need the same step to reach the TBN path.
    let _ = mesh.generate_tangents();
    let mesh = meshes.add(mesh);
    let normal_texture = scene.normal_map.map(|pixel| {
        images.add(Image::new_fill(
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &pixel,
            TextureFormat::Rgba8Unorm,
            RenderAssetUsages::RENDER_WORLD,
        ))
    });
    let material = materials.add(MToonMaterial {
        base_color: scene.base_color,
        shade: Shade {
            color: LinearRgba::from(scene.shade_color),
            shading_shift_factor: scene.shading_shift_factor,
            toony_factor: scene.toony_factor,
            ..default()
        },
        normal_texture,
        normal_texture_scale: scene.normal_scale,
        ..default()
    });
    commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(material),
        Transform::default(),
        RenderLayers::layer(AVATAR_RENDER_LAYER),
    ));

    let camera_transform =
        Transform::from_translation(Vec3::new(0.0, 0.0, 5.0)).looking_at(Vec3::ZERO, Vec3::Y);
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: 1.0,
            ..default()
        }),
        AvatarViewportCamera::from_default_transform(camera_transform),
        camera_transform,
        RenderLayers::layer(AVATAR_RENDER_LAYER),
    ));
}

fn activate_output_after_setup(
    mut state: ResMut<AvatarOutputState>,
    mut armed: ResMut<OutputArmed>,
    cameras: Query<Entity, With<vtuber_avatar::AvatarOutputCamera>>,
) {
    if !armed.0 && cameras.iter().next().is_some() {
        state.activate();
        armed.0 = true;
    }
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn pixels(frame: &VideoOutputFrame) -> Vec<[u8; 4]> {
    frame
        .data
        .chunks_exact(4)
        .map(|pixel| [pixel[0], pixel[1], pixel[2], pixel[3]])
        .collect()
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn center_pixel(pixels: &[[u8; 4]]) -> [u8; 4] {
    pixels[(HEIGHT / 2 * WIDTH + WIDTH / 2) as usize]
}

