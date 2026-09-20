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
use bevy::camera::{Exposure, Hdr, RenderTarget};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::ecs::schedule::ScheduleLabel;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::pipelined_rendering::PipelinedRenderingPlugin;
use bevy::render::render_resource::{
    Extent3d, TextureDimension, TextureFormat, TextureUsages,
};
use bevy::winit::WinitPlugin;
use bevy_egui::{EguiContext, EguiMultipassSchedule, EguiPlugin};
use bevy_vrm1::prelude::{MToonMaterial, MToonPortraitParams, MtoonMaterialPlugin, Shade};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use vtuber_app::ui::{AvatarPreviewPlugin, paint_avatar_preview};
use vtuber_avatar::{
    AVATAR_RENDER_LAYER, AvatarOutputFrameSlot, AvatarOutputState, AvatarViewportCamera,
    PortraitFinishPass, register_output_systems, register_portrait_finish,
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
    /// The material's portrait values; strength 0 is the standard display.
    portrait: MToonPortraitParams,
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
            portrait: MToonPortraitParams::default(),
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
        "finish-alpha" => finish_alpha(),
        "avatar-ui-alpha" => avatar_ui_alpha(),
        "mtoon-lighting" => mtoon_lighting(),
        "mtoon-shading" => mtoon_shading(),
        "mtoon-portrait" => mtoon_portrait(),
        "mtoon-shadow" => mtoon_shadow(),
        "mtoon-cutout-shadow" => mtoon_cutout_shadow(),
        "studio-environment" => studio_environment(),
        "mtoon-standard" => mtoon_standard(),
        "mtoon-normal" => mtoon_normal(),
        "help" | "--help" | "-h" => {
            println!("cargo xtask rich-look <case> [--evidence <file>]");
            println!("cases:");
            println!("  mtoon-lighting  directional-light color/intensity response");
            println!("  mtoon-shading   signed NdotL, shading shift and toony endpoints");
            println!("  mtoon-normal    normal texture, scale and TBN wiring");
            println!("  mtoon-standard  the standard display is the plain authored display");
            println!("  finish-alpha    HDR finish, sRGB premultiplication and readback");
            println!("  avatar-ui-alpha shared avatar image through the real egui preview callback");
            println!("  mtoon-authored  (renamed to mtoon-standard)");
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

    let base = MtoonScene {
        portrait: MToonPortraitParams {
            strength: 1.0,
            ..default()
        },
        ..MtoonScene::lit(Color::WHITE, Color::BLACK, white(0.0))
    };
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
    let base = MtoonScene {
        portrait: MToonPortraitParams {
            strength: 1.0,
            ..default()
        },
        ..MtoonScene::lit(Color::WHITE, Color::BLACK, light)
    };
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

/// The standard display (`portrait.strength` 0) must be the plain display and
/// it must never blow out: a fully lit white surface stays below white, and
/// raising the light level still changes the pixel instead of saturating.
fn mtoon_standard() -> Result<String, RichLookError> {
    let scene = |illuminance: f32| MtoonScene::lit(
        Color::WHITE,
        Color::BLACK,
        LightSpec {
            direction: Vec3::NEG_Z,
            color: Color::WHITE,
            illuminance,
            shadows_enabled: false,
        },
    );
    let app_default = center_pixel(&render(&scene(650.0))?);
    let doubled = center_pixel(&render(&scene(1_300.0))?);
    let red = center_pixel(&render(&MtoonScene {
        lights: vec![LightSpec {
            direction: Vec3::NEG_Z,
            color: Color::srgb(1.0, 0.0, 0.0),
            illuminance: 650.0,
            shadows_enabled: false,
        }],
        ..scene(650.0)
    })?);

    let mut report = format!(
        "case=mtoon-standard\n\
         lit_650lx={app_default:?}\n\
         lit_1300lx={doubled:?}\n\
         lit_650lx_red={red:?}\n"
    );
    if luma(app_default) >= 750 {
        return Err(RichLookError::Failed(format!(
            "the standard display blew a fully lit white surface to white: {app_default:?}"
        )));
    }
    if luma(doubled) <= luma(app_default) {
        return Err(RichLookError::Failed(format!(
            "the standard display saturated instead of following the light: 650lx={app_default:?} 1300lx={doubled:?}"
        )));
    }
    if red[2] <= red[0] {
        return Err(RichLookError::Failed(format!(
            "the standard display did not follow the light color: {red:?}"
        )));
    }
    report.push_str("checks=standard_not_blown,standard_follows_light\n");
    Ok(report)
}

/// The added portrait terms must be layered on without touching the standard
/// display: strength 0 is the standard pixel, the rich lighting alone changes
/// it, and the gloss/environment/rim gains change it again.
fn mtoon_portrait() -> Result<String, RichLookError> {
    let light = LightSpec {
        direction: Vec3::NEG_Z,
        color: Color::WHITE,
        illuminance: 900.0,
        shadows_enabled: false,
    };
    let scene = |portrait: MToonPortraitParams| MtoonScene {
        portrait,
        ..MtoonScene::lit(Color::WHITE, Color::BLACK, light)
    };
    let standard = center_pixel(&render(&scene(MToonPortraitParams::default()))?);
    let rich_without_extras = center_pixel(&render(&scene(MToonPortraitParams {
        strength: 1.0,
        specular_gain: 0.0,
        environment_gain: 0.0,
        rim_gain: 0.0,
        ..default()
    }))?);
    let preset = vtuber_avatar::look::MTOON_PORTRAIT_PRESET;
    let rich = center_pixel(&render(&scene(MToonPortraitParams {
        strength: 1.0,
        ..preset
    }))?);
    // The added specular must follow the light: rotating the key moves the
    // highlight, so the same probe pixel changes.
    let rotated = center_pixel(&render(&scene(MToonPortraitParams {
        strength: 1.0,
        ..preset
    })
    .with_light_direction(Vec3::new(0.7, -0.3, -0.6)))?);

    let mut report = format!(
        "case=mtoon-portrait\n\
         standard={standard:?}\n\
         rich_without_extras={rich_without_extras:?}\n\
         rich={rich:?}\n\
         rich_rotated_light={rotated:?}\n"
    );
    if luma_diff(rich_without_extras, standard) > 4 {
        return Err(RichLookError::Failed(format!(
            "zeroed gains changed the standard pixel: standard={standard:?} rich={rich_without_extras:?}"
        )));
    }
    if luma_diff(rich, rich_without_extras) <= 4 {
        return Err(RichLookError::Failed(format!(
            "the added gloss/environment/rim changed nothing: without={rich_without_extras:?} with={rich:?}"
        )));
    }
    if luma_diff(rotated, rich) <= 4 {
        return Err(RichLookError::Failed(format!(
            "the added specular did not follow the light: fixed={rich:?} rotated={rotated:?}"
        )));
    }
    report.push_str("checks=zero_gains_match_standard,extras_change,specular_follows_light\n");
    Ok(report)
}

impl MtoonScene {
    /// Returns the scene with one light's travel direction replaced.
    fn with_light_direction(mut self, direction: Vec3) -> Self {
        if let Some(light) = self.lights.first_mut() {
            light.direction = direction;
        }
        self
    }
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
            portrait: MToonPortraitParams {
                strength: 1.0,
                ..default()
            },
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
        portrait: MToonPortraitParams {
            strength: 1.0,
            ..default()
        },
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
        portrait: MToonPortraitParams {
            strength: 1.0,
            ..default()
        },
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

const FINISH_ALPHA_VALUES: [f32; 4] = [0.0, 0.25, 0.5, 1.0];
const FINISH_LINEAR_COLOR: [f32; 3] = [0.5, 0.2, 0.05];

/// Exercise the actual HDR finish, sRGB target write, GPU readback and
/// `VideoOutputFrame::from_padded_bgra8` path at each alpha boundary.
#[allow(clippy::indexing_slicing)]
fn finish_alpha() -> Result<String, RichLookError> {
    let mut samples = Vec::new();
    for alpha in FINISH_ALPHA_VALUES {
        samples.push(center_pixel(&render_finish_alpha(alpha)?));
    }

    if samples[0] != [0, 0, 0, 0] {
        return Err(RichLookError::Failed(format!(
            "alpha=0 was not transparent after the finish/readback boundary: {:?}",
            samples[0]
        )));
    }

    let expected_opaque = expected_finish_bgra8(1.0);
    let mut report = format!(
        "case=finish-alpha\n\
         source_linear={FINISH_LINEAR_COLOR:?}\n\
         alphas={FINISH_ALPHA_VALUES:?}\n\
         readback_straight_bgra={samples:?}\n\
         expected_opaque_straight_bgra={expected_opaque:?}\n"
    );
    for (alpha, sample) in FINISH_ALPHA_VALUES.iter().zip(&samples) {
        let expected = expected_finish_bgra8(*alpha);
        if sample[3].abs_diff(expected[3]) > 1 {
            return Err(RichLookError::Failed(format!(
                "alpha={alpha} changed at the readback boundary: actual={sample:?} expected={expected:?}"
            )));
        }
        if *alpha > 0.0 {
            for (channel, (actual, wanted)) in
                sample[..3].iter().zip(expected[..3].iter()).enumerate()
            {
                if actual.abs_diff(*wanted) > 4 {
                    return Err(RichLookError::Failed(format!(
                        "alpha={alpha} channel={channel} has more than quantization error: actual={sample:?} expected={expected:?}"
                    )));
                }
            }
        }
    }

    for (alpha, sample) in [0.25_f32, 0.5].into_iter().zip(samples.iter().skip(1)) {
        let expected = expected_finish_bgra8(alpha);
        let legacy = legacy_finish_bgra8(alpha);
        if rgb_distance(*sample, expected) >= rgb_distance(*sample, legacy) {
            return Err(RichLookError::Failed(format!(
                "alpha={alpha} followed E(a*T(C)) instead of a*E(T(C)): actual={sample:?} expected={expected:?} legacy={legacy:?}"
            )));
        }
        report.push_str(&format!(
            "alpha={alpha}: expected={expected:?}, legacy_wrong={legacy:?}, rgb_distance_to_expected={}, rgb_distance_to_legacy={}\n",
            rgb_distance(*sample, expected),
            rgb_distance(*sample, legacy),
        ));
    }

    let backgrounds = [
        ("black", [0, 0, 0]),
        ("white", [255, 255, 255]),
        ("color", [24, 96, 180]),
    ];
    for (name, background) in backgrounds {
        for (alpha, sample) in [0.25_f32, 0.5].into_iter().zip(samples.iter().skip(1)) {
            let expected_source = [
                expected_opaque[0],
                expected_opaque[1],
                expected_opaque[2],
                quantize(alpha),
            ];
            let expected = composite_bgra8(expected_source, background);
            let actual = composite_bgra8(*sample, background);
            if actual
                .iter()
                .zip(expected)
                .any(|(actual, expected)| actual.abs_diff(expected) > 4)
            {
                return Err(RichLookError::Failed(format!(
                    "{name} background alpha={alpha} composition changed the straight color: actual={actual:?} expected={expected:?}"
                )));
            }
            report.push_str(&format!(
                "composite_{name}_alpha={alpha}: actual={actual:?}, expected={expected:?}\n"
            ));
        }
    }

    Ok(report)
}

const AVATAR_UI_SIZE: u32 = 256;
const AVATAR_UI_TILE: f32 = 64.0;
const AVATAR_UI_SOURCE_SIZE: u32 = 64;
const AVATAR_UI_SETTLE_FRAMES: u32 = 12;
const AVATAR_UI_EDGE_COLUMN: u32 = 30;
const AVATAR_UI_TONE_AFTER_LINEAR: [f32; 3] = [0.5, 0.2, 0.05];
const AVATAR_UI_ALPHA_VALUES: [f32; 4] = [0.0, 0.25, 0.5, 1.0];
const AVATAR_UI_BACKGROUNDS: [[u8; 3]; 4] = [
    [0, 0, 0],
    [255, 255, 255],
    [24, 96, 180],
    [228, 228, 228],
];

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
struct AvatarUiFixturePass;

#[derive(Clone, Copy)]
enum AvatarUiFixtureMode {
    Uniform { alpha: f32 },
    TransparentEdge,
}

#[derive(Resource, Clone, Copy)]
struct AvatarUiFixtureSpec(AvatarUiFixtureMode);

#[derive(Resource, Clone)]
struct AvatarUiFixtureImage(Handle<Image>);

#[derive(Resource, Clone)]
struct AvatarUiFixtureTarget(Handle<Image>);

#[derive(Resource, Default)]
struct AvatarUiFixtureReadback(Option<Vec<u8>>);

#[derive(Component)]
struct AvatarUiFixtureCamera;

/// Exercise the shared gamma-premultiplied avatar image through the actual
/// avatar egui paint callback, including linear-space background composition.
#[allow(clippy::indexing_slicing)]
fn avatar_ui_alpha() -> Result<String, RichLookError> {
    let mut report = format!(
        "case=avatar-ui-alpha\n\
         target=Rgba16Float_linear_egui_blend\n\
         tone_after_linear={AVATAR_UI_TONE_AFTER_LINEAR:?}\n\
         alphas={AVATAR_UI_ALPHA_VALUES:?}\n\
         backgrounds={AVATAR_UI_BACKGROUNDS:?}\n"
    );

    for alpha in AVATAR_UI_ALPHA_VALUES {
        let pixels = render_avatar_ui(AvatarUiFixtureMode::Uniform { alpha })?;
        for (background_index, background) in AVATAR_UI_BACKGROUNDS.into_iter().enumerate() {
            let x = background_index as u32 * AVATAR_UI_TILE as u32 + 32;
            let y = 32;
            let actual = encoded_ui_pixel(&pixels, x, y);
            let expected = expected_ui_pixel(
                AvatarUiFixtureMode::Uniform { alpha },
                1,
                1,
                0,
                background,
            );
            if rgb_distance(actual, expected) > 4 {
                return Err(RichLookError::Failed(format!(
                    "uniform UI composite changed at alpha={alpha} background={background:?}: actual={actual:?} expected={expected:?}"
                )));
            }
            report.push_str(&format!(
                "uniform_alpha={alpha} background={background:?}: actual={actual:?} expected={expected:?}\n"
            ));
        }
    }

    let edge_pixels = render_avatar_ui(AvatarUiFixtureMode::TransparentEdge)?;
    let normal_opaque = encoded_ui_pixel(&edge_pixels, 16 + 29, 16 + 32);
    let normal_transparent = encoded_ui_pixel(&edge_pixels, 16 + 30, 16 + 32);
    let reduced_boundary = encoded_ui_pixel(&edge_pixels, 160 + 7, 40 + 8);
    let normal_opaque_expected = expected_ui_pixel(
        AvatarUiFixtureMode::TransparentEdge,
        64,
        64,
        29,
        [228, 228, 228],
    );
    let normal_transparent_expected = expected_ui_pixel(
        AvatarUiFixtureMode::TransparentEdge,
        64,
        64,
        30,
        [228, 228, 228],
    );
    let reduced_expected = expected_ui_pixel(
        AvatarUiFixtureMode::TransparentEdge,
        16,
        16,
        7,
        [228, 228, 228],
    );
    let reduced_legacy = legacy_ui_pixel(
        AvatarUiFixtureMode::TransparentEdge,
        16,
        16,
        7,
        [228, 228, 228],
    );
    for (name, actual, expected) in [
        ("normal_opaque", normal_opaque, normal_opaque_expected),
        (
            "normal_transparent",
            normal_transparent,
            normal_transparent_expected,
        ),
        ("reduced_boundary", reduced_boundary, reduced_expected),
    ] {
        if rgb_distance(actual, expected) > 4 {
            return Err(RichLookError::Failed(format!(
                "{name} transparent boundary changed: actual={actual:?} expected={expected:?}"
            )));
        }
    }
    if rgb_distance(reduced_boundary, reduced_expected)
        >= rgb_distance(reduced_boundary, reduced_legacy)
    {
        return Err(RichLookError::Failed(format!(
            "reduced transparent boundary followed filtered gamma-premultiplied conversion: actual={reduced_boundary:?} expected={reduced_expected:?} legacy_wrong={reduced_legacy:?}"
        )));
    }
    report.push_str(&format!(
        "transparent_edge_normal_opaque={normal_opaque:?} expected={normal_opaque_expected:?}\n\
         transparent_edge_normal_transparent={normal_transparent:?} expected={normal_transparent_expected:?}\n\
         transparent_edge_reduced={reduced_boundary:?} expected={reduced_expected:?} legacy_wrong={reduced_legacy:?}\n"
    ));
    Ok(report)
}

fn avatar_ui_fixture_app(spec: AvatarUiFixtureMode) -> App {
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
    .add_plugins((EguiPlugin::default(), AvatarPreviewPlugin))
    .insert_resource(AvatarUiFixtureSpec(spec))
    .init_resource::<AvatarUiFixtureReadback>()
    .add_systems(Startup, setup_avatar_ui_fixture)
    .add_systems(AvatarUiFixturePass, draw_avatar_ui_fixture);
    app.finish();
    app.cleanup();
    app
}

fn setup_avatar_ui_fixture(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    spec: Res<AvatarUiFixtureSpec>,
) {
    let (source_width, source_height, source_data) = avatar_ui_source(*spec);
    let source = images.add(Image::new(
        Extent3d {
            width: source_width,
            height: source_height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        source_data,
        TextureFormat::Bgra8UnormSrgb,
        RenderAssetUsages::default(),
    ));
    let mut target_image = Image::new_target_texture(
        AVATAR_UI_SIZE,
        AVATAR_UI_SIZE,
        TextureFormat::Rgba16Float,
        None,
    );
    target_image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let target = images.add(target_image);
    commands.insert_resource(AvatarUiFixtureImage(source));
    commands.insert_resource(AvatarUiFixtureTarget(target.clone()));
    commands
        .spawn((
            Camera3d::default(),
            Hdr,
            RenderTarget::Image(target.clone().into()),
            Tonemapping::None,
            Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
            EguiMultipassSchedule::new(AvatarUiFixturePass),
            AvatarUiFixtureCamera,
            Readback::texture(target),
        ))
        .observe(handle_avatar_ui_readback);
}

fn draw_avatar_ui_fixture(
    context: Single<&mut EguiContext, With<AvatarUiFixtureCamera>>,
    source: Res<AvatarUiFixtureImage>,
    spec: Res<AvatarUiFixtureSpec>,
) {
    let mut context = context.into_inner();
    let ctx = context.get_mut();
    let full_rect = ctx.viewport_rect();
    let mut root = bevy_egui::egui::Ui::new(
        ctx.clone(),
        bevy_egui::egui::Id::new("avatar_ui_fixture"),
        bevy_egui::egui::UiBuilder::new()
            .layer_id(bevy_egui::egui::LayerId::background())
            .max_rect(full_rect),
    );
    match spec.0 {
        AvatarUiFixtureMode::Uniform { .. } => {
            let painter = ctx.layer_painter(bevy_egui::egui::LayerId::background());
            for row in 0..4u32 {
                for (column, background) in AVATAR_UI_BACKGROUNDS.into_iter().enumerate() {
                    let rect = bevy_egui::egui::Rect::from_min_size(
                        bevy_egui::egui::pos2(
                            column as f32 * AVATAR_UI_TILE,
                            row as f32 * AVATAR_UI_TILE,
                        ),
                        bevy_egui::egui::vec2(AVATAR_UI_TILE, AVATAR_UI_TILE),
                    );
                    painter.rect_filled(
                        rect,
                        bevy_egui::egui::CornerRadius::ZERO,
                        bevy_egui::egui::Color32::from_rgb(
                            background[0],
                            background[1],
                            background[2],
                        ),
                    );
                    paint_avatar_ui_tile(&mut root, rect, &source.0);
                }
            }
        }
        AvatarUiFixtureMode::TransparentEdge => {
            ctx.layer_painter(bevy_egui::egui::LayerId::background()).rect_filled(
                full_rect,
                bevy_egui::egui::CornerRadius::ZERO,
                bevy_egui::egui::Color32::from_gray(228),
            );
            paint_avatar_ui_tile(
                &mut root,
                bevy_egui::egui::Rect::from_min_size(
                    bevy_egui::egui::pos2(16.0, 16.0),
                    bevy_egui::egui::vec2(64.0, 64.0),
                ),
                &source.0,
            );
            paint_avatar_ui_tile(
                &mut root,
                bevy_egui::egui::Rect::from_min_size(
                    bevy_egui::egui::pos2(160.0, 40.0),
                    bevy_egui::egui::vec2(16.0, 16.0),
                ),
                &source.0,
            );
        }
    }
}

fn paint_avatar_ui_tile(
    root: &mut bevy_egui::egui::Ui,
    rect: bevy_egui::egui::Rect,
    source: &Handle<Image>,
) {
    let mut child = root.new_child(bevy_egui::egui::UiBuilder::new().max_rect(rect));
    let _ = paint_avatar_preview(&mut child, source.clone(), rect.size(), 0.0);
}

fn handle_avatar_ui_readback(
    event: On<ReadbackComplete>,
    mut commands: Commands,
    target: Res<AvatarUiFixtureTarget>,
    mut readback: ResMut<AvatarUiFixtureReadback>,
) {
    readback.0 = Some(event.data.clone());
    commands
        .entity(event.entity)
        .insert(Readback::texture(target.0.clone()));
}

fn render_avatar_ui(spec: AvatarUiFixtureMode) -> Result<Vec<u8>, RichLookError> {
    let mut app = avatar_ui_fixture_app(spec);
    let deadline = Instant::now() + MAX_WAIT;
    let mut last = None;
    let mut settled = 0;
    while Instant::now() < deadline {
        app.update();
        if let Some(data) = app
            .world_mut()
            .resource_mut::<AvatarUiFixtureReadback>()
            .0
            .take()
        {
            settled = if data.iter().any(|byte| *byte != 0) {
                settled + 1
            } else {
                0
            };
            last = Some(data.clone());
            if settled >= AVATAR_UI_SETTLE_FRAMES {
                return Ok(data);
            }
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
        "GPU avatar UI readback did not complete; the local renderer/GPU path is unavailable"
            .into(),
    ))
}

fn avatar_ui_source(spec: AvatarUiFixtureSpec) -> (u32, u32, Vec<u8>) {
    let (width, height) = match spec.0 {
        AvatarUiFixtureMode::Uniform { .. } => (1, 1),
        AvatarUiFixtureMode::TransparentEdge => (AVATAR_UI_SOURCE_SIZE, AVATAR_UI_SOURCE_SIZE),
    };
    let mut data = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let pixel = avatar_ui_source_pixel(spec.0, x, y);
            data.extend([pixel[2], pixel[1], pixel[0], pixel[3]]);
        }
    }
    (width, height, data)
}

fn avatar_ui_source_pixel(spec: AvatarUiFixtureMode, x: u32, _y: u32) -> [u8; 4] {
    let encoded = AVATAR_UI_TONE_AFTER_LINEAR.map(srgb_encode);
    match spec {
        AvatarUiFixtureMode::Uniform { alpha } => {
            let stored_alpha = quantize(alpha);
            [
                quantize(encoded[0] * alpha),
                quantize(encoded[1] * alpha),
                quantize(encoded[2] * alpha),
                stored_alpha,
            ]
        }
        AvatarUiFixtureMode::TransparentEdge => {
            if x < AVATAR_UI_EDGE_COLUMN {
                [quantize(encoded[0]), quantize(encoded[1]), quantize(encoded[2]), 255]
            } else {
                [0, 0, 0, 0]
            }
        }
    }
}

#[allow(clippy::indexing_slicing)]
fn encoded_ui_pixel(data: &[u8], x: u32, y: u32) -> [u8; 4] {
    let row_stride = AVATAR_UI_SIZE as usize * 8;
    let offset = y as usize * row_stride + x as usize * 8;
    let linear = [
        half_to_f32(u16::from_le_bytes([data[offset], data[offset + 1]])),
        half_to_f32(u16::from_le_bytes([data[offset + 2], data[offset + 3]])),
        half_to_f32(u16::from_le_bytes([data[offset + 4], data[offset + 5]])),
        half_to_f32(u16::from_le_bytes([data[offset + 6], data[offset + 7]])),
    ];
    [
        quantize(srgb_encode(linear[0])),
        quantize(srgb_encode(linear[1])),
        quantize(srgb_encode(linear[2])),
        quantize(linear[3]),
    ]
}

fn expected_ui_pixel(
    spec: AvatarUiFixtureMode,
    display_width: u32,
    display_height: u32,
    local_x: u32,
    background: [u8; 3],
) -> [u8; 4] {
    let source_size = match spec {
        AvatarUiFixtureMode::Uniform { .. } => 1,
        AvatarUiFixtureMode::TransparentEdge => AVATAR_UI_SOURCE_SIZE,
    };
    let source_x = ((local_x as f32 + 0.5) / display_width as f32 * source_size as f32 - 0.5)
        .max(-0.5);
    let source_y = ((display_height as f32 * 0.5) / display_height as f32
        * source_size as f32
        - 0.5)
    .max(-0.5);
    let base_x = source_x.floor() as i32;
    let base_y = source_y.floor() as i32;
    let fraction_x = source_x - base_x as f32;
    let fraction_y = source_y - base_y as f32;
    let top = mix_ui_source_pixels(
        linear_premultiplied_from_stored(avatar_ui_source_pixel(
            spec,
            base_x.max(0) as u32,
            base_y.max(0) as u32,
        )),
        linear_premultiplied_from_stored(avatar_ui_source_pixel(
            spec,
            (base_x + 1).max(0) as u32,
            base_y.max(0) as u32,
        )),
        fraction_x,
    );
    let bottom = mix_ui_source_pixels(
        linear_premultiplied_from_stored(avatar_ui_source_pixel(
            spec,
            base_x.max(0) as u32,
            (base_y + 1).max(0) as u32,
        )),
        linear_premultiplied_from_stored(avatar_ui_source_pixel(
            spec,
            (base_x + 1).max(0) as u32,
            (base_y + 1).max(0) as u32,
        )),
        fraction_x,
    );
    let source = mix_ui_source_pixels(top, bottom, fraction_y);
    composite_ui_pixel(source, background)
}

#[allow(clippy::indexing_slicing)]
fn legacy_ui_pixel(
    spec: AvatarUiFixtureMode,
    display_width: u32,
    _display_height: u32,
    local_x: u32,
    background: [u8; 3],
) -> [u8; 4] {
    let source_size = match spec {
        AvatarUiFixtureMode::Uniform { .. } => 1,
        AvatarUiFixtureMode::TransparentEdge => AVATAR_UI_SOURCE_SIZE,
    };
    let source_x = (local_x as f32 + 0.5) / display_width as f32 * source_size as f32 - 0.5;
    let base_x = source_x.floor() as i32;
    let fraction_x = source_x - base_x as f32;
    let left = avatar_ui_source_pixel(spec, base_x.max(0) as u32, 0);
    let right = avatar_ui_source_pixel(spec, (base_x + 1).max(0) as u32, 0);
    let left_alpha = f32::from(left[3]) / 255.0;
    let right_alpha = f32::from(right[3]) / 255.0;
    let sample = [0, 1, 2].map(|channel| {
        let left_sample = srgb_decode(f32::from(left[channel]) / 255.0);
        let right_sample = srgb_decode(f32::from(right[channel]) / 255.0);
        left_sample + (right_sample - left_sample) * fraction_x
    });
    let alpha = left_alpha + (right_alpha - left_alpha) * fraction_x;
    composite_ui_pixel(
        if alpha == 0.0 {
            [0.0, 0.0, 0.0, 0.0]
        } else {
            let straight = srgb_decode(srgb_encode(sample[0]) / alpha);
            let straight_green = srgb_decode(srgb_encode(sample[1]) / alpha);
            let straight_blue = srgb_decode(srgb_encode(sample[2]) / alpha);
            [
                straight * alpha,
                straight_green * alpha,
                straight_blue * alpha,
                alpha,
            ]
        },
        background,
    )
}

#[allow(clippy::indexing_slicing)]
fn straight_srgb_from_stored(stored: [u8; 4]) -> [f32; 4] {
    // Match VideoOutputFrame's existing byte-domain unassociation.
    let alpha = f32::from(stored[3]) / 255.0;
    if alpha == 0.0 {
        return [0.0; 4];
    }
    let straight = [0, 1, 2].map(|channel| f32::from(stored[channel]) / 255.0 / alpha);
    [straight[0], straight[1], straight[2], alpha]
}

#[allow(clippy::indexing_slicing)]
fn linear_premultiplied_from_stored(stored: [u8; 4]) -> [f32; 4] {
    let straight_srgb = straight_srgb_from_stored(stored);
    let straight = [0, 1, 2].map(|channel| srgb_decode(straight_srgb[channel]));
    [
        straight[0] * straight_srgb[3],
        straight[1] * straight_srgb[3],
        straight[2] * straight_srgb[3],
        straight_srgb[3],
    ]
}

#[allow(clippy::indexing_slicing)]
fn mix_ui_source_pixels(left: [f32; 4], right: [f32; 4], amount: f32) -> [f32; 4] {
    [0, 1, 2, 3].map(|channel| left[channel] + (right[channel] - left[channel]) * amount)
}

fn composite_ui_pixel(source: [f32; 4], background: [u8; 3]) -> [u8; 4] {
    let alpha = source[3];
    let background_linear = background.map(|channel| srgb_decode(f32::from(channel) / 255.0));
    [
        quantize(srgb_encode(source[0] + background_linear[0] * (1.0 - alpha))),
        quantize(srgb_encode(source[1] + background_linear[1] * (1.0 - alpha))),
        quantize(srgb_encode(source[2] + background_linear[2] * (1.0 - alpha))),
        255,
    ]
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x03ff);
    let value = match exponent {
        0 => {
            if fraction == 0 {
                sign
            } else {
                let mut fraction = fraction;
                let mut exponent = 0_u32;
                while fraction & 0x0400 == 0 {
                    fraction <<= 1;
                    exponent += 1;
                }
                let mantissa = fraction & 0x03ff;
                sign | ((113 - exponent) << 23) | (mantissa << 13)
            }
        }
        31 => sign | 0x7f80_0000 | (fraction << 13),
        exponent => sign | ((exponent + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(value)
}

fn srgb_encode(value: f32) -> f32 {
    if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

fn srgb_decode(value: f32) -> f32 {
    if value <= 0.040_45 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn quantize(value: f32) -> u8 {
    (value * 255.0).round().clamp(0.0, 255.0) as u8
}

fn expected_finish_bgra8(alpha: f32) -> [u8; 4] {
    let toned = FINISH_LINEAR_COLOR.map(|channel| channel / (1.0 + channel));
    let encoded = toned.map(srgb_encode);
    [
        quantize(encoded[2]),
        quantize(encoded[1]),
        quantize(encoded[0]),
        quantize(alpha),
    ]
}

fn legacy_finish_bgra8(alpha: f32) -> [u8; 4] {
    let toned = FINISH_LINEAR_COLOR.map(|channel| channel / (1.0 + channel));
    let encoded = toned.map(|channel| srgb_encode(channel * alpha) / alpha);
    [
        quantize(encoded[2]),
        quantize(encoded[1]),
        quantize(encoded[0]),
        quantize(alpha),
    ]
}

fn rgb_distance(actual: [u8; 4], expected: [u8; 4]) -> u32 {
    actual[..3]
        .iter()
        .zip(expected[..3].iter())
        .map(|(actual, expected)| u32::from(actual.abs_diff(*expected)))
        .sum()
}

fn composite_bgra8(source: [u8; 4], background: [u8; 3]) -> [u8; 3] {
    let alpha = u32::from(source[3]);
    let inverse_alpha = u32::from(u8::MAX) - alpha;
    [
        ((u32::from(source[0]) * alpha + u32::from(background[0]) * inverse_alpha + 127) / 255)
            as u8,
        ((u32::from(source[1]) * alpha + u32::from(background[1]) * inverse_alpha + 127) / 255)
            as u8,
        ((u32::from(source[2]) * alpha + u32::from(background[2]) * inverse_alpha + 127) / 255)
            as u8,
    ]
}

fn render_finish_alpha(alpha: f32) -> Result<Vec<[u8; 4]>, RichLookError> {
    let mut app = finish_fixture_app(alpha);
    let deadline = Instant::now() + MAX_WAIT;
    let expected_alpha = quantize(alpha);
    let mut settled = 0;
    let mut last = None;
    while Instant::now() < deadline {
        app.update();
        if let Some(frame) = app
            .world_mut()
            .resource_mut::<AvatarOutputFrameSlot>()
            .take_latest()
        {
            let sampled = pixels(&frame);
            let center = center_pixel(&sampled);
            let alpha_ready = center[3].abs_diff(expected_alpha) <= 1;
            let color_ready = alpha == 0.0 || sampled.iter().any(|pixel| pixel[3] > 0);
            settled = if alpha_ready && color_ready {
                settled + 1
            } else {
                0
            };
            if settled >= SETTLE_FRAMES {
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
        "GPU finish/readback did not complete; the local renderer/GPU path is unavailable".into(),
    ))
}

fn finish_fixture_app(alpha: f32) -> App {
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
    .insert_resource(ClearColor(Color::srgba(0.0, 0.0, 0.0, 0.0)))
    .insert_resource(vtuber_avatar::AvatarLifecycle::default())
    .insert_resource(FinishAlpha(alpha))
    .insert_resource(OutputArmed(false));
    register_output_systems(&mut app);
    register_portrait_finish(&mut app);
    app.add_systems(Startup, setup_finish_fixture_scene);
    app.add_systems(Update, activate_finish_output_after_setup);
    app.finish();
    app.cleanup();
    app
}

#[derive(Resource, Clone, Copy)]
struct FinishAlpha(f32);

fn setup_finish_fixture_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    alpha: Res<FinishAlpha>,
) {
    let mesh = meshes.add(
        Plane3d::default()
            .mesh()
            .size(4.0, 4.0)
            .build()
            .rotated_by(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
    );
    let material = materials.add(StandardMaterial {
        base_color: Color::LinearRgba(LinearRgba::new(
            FINISH_LINEAR_COLOR[0],
            FINISH_LINEAR_COLOR[1],
            FINISH_LINEAR_COLOR[2],
            alpha.0,
        )),
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        ..default()
    });
    commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(material),
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

fn activate_finish_output_after_setup(
    mut commands: Commands,
    mut state: ResMut<AvatarOutputState>,
    mut armed: ResMut<OutputArmed>,
    cameras: Query<Entity, With<vtuber_avatar::AvatarOutputCamera>>,
) {
    if armed.0 {
        return;
    }
    let Some(output) = cameras.iter().next() else {
        return;
    };
    commands.entity(output).insert((
        Hdr,
        Exposure {
            ev100: -0.263_034_4,
        },
        Tonemapping::None,
        PortraitFinishPass {
            tonemapping: Tonemapping::Reinhard,
        },
    ));
    state.activate();
    armed.0 = true;
}

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
        portrait: scene.portrait,
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
