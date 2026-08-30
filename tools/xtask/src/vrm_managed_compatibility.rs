//! Headless compatibility runner for the production managed-avatar route.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bevy::app::{App, PluginGroup, Startup};
use bevy::asset::io::{AssetSourceBuilder, AssetSourceBuilders};
use bevy::asset::{AssetServer, LoadState};
use bevy::prelude::{DefaultPlugins, Entity, MessageWriter, Res, Resource, Transform, Visibility};
use bevy::render::RenderPlugin;
use bevy::render::pipelined_rendering::PipelinedRenderingPlugin;
use bevy::utils::default;
use bevy::window::{ExitCondition, WindowPlugin};
use bevy_vrm1::prelude::{HipsBoneEntity, VrmAsset, VrmHandle};
use vtuber_app::import::{self, DEFAULT_SIZE_LIMIT};
use vtuber_avatar::{
    ArmChainBinding, ArmIkInput, ArmPoseBlendState, ArmPoseProfile, AvatarAssetId, AvatarBinding,
    AvatarLifecycle, AvatarLifecycleState, DefaultArmPose, IDLE_PROCEDURAL_AMPLITUDE_METERS,
    IdleMotionProfile, ImportedAvatar, LoadImportedAvatarRequest, ResolvedArmPose, UserAssetPath,
    VtuberAvatarPlugin, default_arm_target, solve_two_bone_arm,
};

const MAX_COMPAT_FRAMES: usize = 1_200;
/// Upper bound for pumping the replacement until it reaches `Ready` again.
const MAX_REPLACEMENT_FRAMES: usize = 1_200;

/// Exercises import, the named `user://` source, and the real avatar lifecycle.
pub fn run(path: &Path) -> Result<(), String> {
    let managed_root = temporary_managed_root()?;
    let result = run_with_root(path, &managed_root);
    let cleanup = fs::remove_dir_all(&managed_root);

    match (result, cleanup) {
        (Err(error), _) => Err(error),
        (Ok(()), Ok(())) => Ok(()),
        (Ok(()), Err(error)) => Err(format!(
            "managed compatibility succeeded but temporary asset cleanup failed: {error}"
        )),
    }
}

fn run_with_root(path: &Path, managed_root: &Path) -> Result<(), String> {
    let imported = import::import_vrm(path, managed_root, DEFAULT_SIZE_LIMIT)
        .map_err(|error| format!("import_vrm failed: {error}"))?;
    let asset_id = AvatarAssetId::new(&imported.id);
    let asset_path = UserAssetPath::avatar_model_path(&asset_id)
        .map_err(|error| format!("failed to construct managed asset path: {error}"))?;
    let expected_generation = match imported.summary.generation {
        vtuber_app::import::VrmGeneration::Vrm0 => vtuber_avatar::ExpectedVrmGeneration::Vrm0,
        vtuber_app::import::VrmGeneration::Vrm1 => vtuber_avatar::ExpectedVrmGeneration::Vrm1,
    };
    let imported_avatar = ImportedAvatar::new(
        asset_id,
        asset_path,
        imported.name.clone(),
        expected_generation,
    );

    let managed_root_string = managed_root
        .to_str()
        .ok_or_else(|| "temporary managed asset root is not valid UTF-8".to_owned())?;
    let mut sources = AssetSourceBuilders::default();
    sources.insert(
        "user",
        AssetSourceBuilder::platform_default(managed_root_string, None),
    );

    let mut app = App::new();
    app.insert_resource(sources)
        .insert_resource(ManagedImportedAvatar(imported_avatar))
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .set(RenderPlugin { ..default() })
                .disable::<PipelinedRenderingPlugin>(),
        )
        .add_plugins(VtuberAvatarPlugin)
        .add_systems(Startup, emit_request);

    app.finish();
    app.cleanup();

    let mut previous_state = None;
    for frame in 0..MAX_COMPAT_FRAMES {
        app.update();
        let (state, failure) = {
            let lifecycle = app.world().resource::<AvatarLifecycle>();
            (lifecycle.state(), lifecycle.failure().cloned())
        };
        if previous_state != Some(state) {
            if previous_state == Some(AvatarLifecycleState::Loading)
                && state == AvatarLifecycleState::Ready
            {
                println!("lifecycle transition: Loading -> Binding");
                println!("lifecycle transition: Binding -> Ready");
            } else {
                println!("lifecycle transition: {:?} -> {state:?}", previous_state);
            }
            previous_state = Some(state);
        }

        match state {
            AvatarLifecycleState::Ready => {
                let root = app
                    .world()
                    .resource::<AvatarLifecycle>()
                    .active_root()
                    .ok_or_else(|| "lifecycle reached Ready without an active root".to_owned())?;
                let visibility = app
                    .world()
                    .get::<Visibility>(root)
                    .ok_or_else(|| format!("Ready root {root:?} has no Visibility component"))?;
                if *visibility == Visibility::Hidden {
                    return Err(format!("Ready root {root:?} remains Visibility::Hidden"));
                }
                println!("managed avatar reached Ready: root={root:?} visibility={visibility:?}");
                verify_arm_pose(&mut app, root)?;
                verify_idle_contract(&mut app, root)?;
                verify_control_episode(&mut app, root)?;
                let (old_generation, old_root) = {
                    let lifecycle = app.world().resource::<AvatarLifecycle>();
                    (lifecycle.current_generation(), root)
                };
                trigger_replacement(&mut app, old_generation)?;
                let new_root =
                    wait_for_replacement_ready(&mut app, old_generation, MAX_REPLACEMENT_FRAMES)?;
                let new_generation = app
                    .world()
                    .resource::<AvatarLifecycle>()
                    .current_generation();
                if new_generation == old_generation {
                    return Err("replacement did not advance the avatar generation".to_owned());
                }
                if app.world().get::<AvatarBinding>(old_root).is_some() {
                    return Err(format!(
                        "stale entity: replaced root {old_root:?} still carries AvatarBinding"
                    ));
                }
                println!(
                    "replacement verified: old_root={old_root:?} new_root={new_root:?} generation {old_generation:?} -> {new_generation:?}"
                );
                verify_idle_contract(&mut app, new_root)?;
                verify_control_episode(&mut app, new_root)?;
                return Ok(());
            }
            AvatarLifecycleState::Failed => {
                println!("lifecycle failure: {failure:?}");
                print_asset_failures(&mut app);
                return Err(format!("managed avatar lifecycle failed: {failure:?}"));
            }
            _ => {
                if frame % 60 == 0 {
                    println!("lifecycle pending: frame={frame} state={state:?}");
                }
            }
        }
    }

    print_asset_failures(&mut app);
    Err(format!(
        "managed avatar did not reach Ready within {MAX_COMPAT_FRAMES} frames"
    ))
}

fn verify_arm_pose(app: &mut App, root: Entity) -> Result<(), String> {
    let (binding, default_pose, blend) = {
        let world = app.world();
        let binding = world
            .get::<AvatarBinding>(root)
            .copied()
            .ok_or_else(|| format!("Ready root {root:?} has no AvatarBinding"))?;
        let default_pose = world
            .get::<DefaultArmPose>(root)
            .copied()
            .ok_or_else(|| format!("Ready root {root:?} has no DefaultArmPose"))?;
        let blend = world
            .get::<ArmPoseBlendState>(root)
            .copied()
            .ok_or_else(|| format!("Ready root {root:?} has no ArmPoseBlendState"))?;
        (binding, default_pose, blend)
    };

    if binding.generation != default_pose.generation || binding.generation != blend.generation {
        return Err(format!(
            "Ready root {root:?} has inconsistent arm generations: binding={:?} default={:?} blend={:?}",
            binding.generation, default_pose.generation, blend.generation
        ));
    }

    verify_arm_side(
        "left",
        binding.left_arm,
        default_pose.left,
        blend.current_left(),
    )?;
    verify_arm_side(
        "right",
        binding.right_arm,
        default_pose.right,
        blend.current_right(),
    )?;
    Ok(())
}

fn verify_arm_side(
    side: &str,
    chain: Option<ArmChainBinding>,
    resolved: Option<ResolvedArmPose>,
    current: Option<ResolvedArmPose>,
) -> Result<(), String> {
    let Some(chain) = chain else {
        if resolved.is_some() || current.is_some() {
            return Err(format!(
                "{side} arm has a resolved pose without a complete cached chain"
            ));
        }
        println!("arm pose side={side} unavailable (incomplete or degenerate chain)");
        return Ok(());
    };

    let pose = resolved.ok_or_else(|| {
        format!("{side} arm has a complete cached chain but no resolved DefaultArmPose")
    })?;
    let current = current.ok_or_else(|| format!("{side} arm has no initial blend output"))?;
    if pose.upper_arm != chain.upper_arm
        || pose.lower_arm != chain.lower_arm
        || current.upper_arm != chain.upper_arm
        || current.lower_arm != chain.lower_arm
    {
        return Err(format!(
            "{side} arm resolved pose targets the wrong entities"
        ));
    }

    let target = default_arm_target(&chain, ArmPoseProfile::default())
        .map_err(|error| format!("{side} arm default target failed: {error}"))?;
    let input = ArmIkInput::from_geometry(chain.rest, target);
    let solution = solve_two_bone_arm(input)
        .map_err(|error| format!("{side} arm IK solve failed: {error}"))?;
    let upper_direction = solution.elbow - input.shoulder;
    let lower_direction = solution.wrist - solution.elbow;
    let bend_sine = upper_direction
        .try_normalize()
        .and_then(|upper| {
            lower_direction
                .try_normalize()
                .map(|lower| upper.cross(lower).length())
        })
        .ok_or_else(|| format!("{side} arm IK produced a degenerate elbow bend"))?;
    if !bend_sine.is_finite() || bend_sine <= 1.0e-4 {
        return Err(format!(
            "{side} arm IK produced no measurable elbow bend: sine={bend_sine}"
        ));
    }

    for (label, rotation) in [
        ("upper_arm_delta", pose.upper_arm_delta),
        ("lower_arm_delta", pose.lower_arm_delta),
        ("current_upper_arm_delta", current.upper_arm_delta),
        ("current_lower_arm_delta", current.lower_arm_delta),
    ] {
        if !rotation.is_finite() || rotation.length_squared() <= f32::EPSILON {
            return Err(format!("{side} arm {label} is non-finite or degenerate"));
        }
    }
    if !solution.elbow.is_finite()
        || !solution.wrist.is_finite()
        || !solution.solved_reach.is_finite()
        || !pose.upper_arm_delta.is_normalized()
        || !pose.lower_arm_delta.is_normalized()
    {
        return Err(format!(
            "{side} arm IK or resolved pose is not finite/normalized"
        ));
    }

    println!(
        "arm pose verified: side={side} upper={:?} lower={:?} hand={:?} bend_sine={bend_sine:.6} optional_shoulder={} optional_fingers={}",
        chain.upper_arm,
        chain.lower_arm,
        chain.hand,
        chain.capabilities.has_shoulder,
        chain.capabilities.has_fingers,
    );
    Ok(())
}

/// Verifies the zero-amplitude idle policy on a Ready avatar: the typed
/// profile must be bound and the hips translation must stay exactly at its
/// animation-authored value across paced frames.
fn verify_idle_contract(app: &mut App, root: Entity) -> Result<(), String> {
    const IDLE_FRAMES: usize = 30;
    const IDLE_FRAME_PACE: Duration = Duration::from_millis(16);

    let (profile, hips) = {
        let world = app.world();
        let profile = world
            .get::<IdleMotionProfile>(root)
            .copied()
            .ok_or_else(|| format!("Ready root {root:?} has no IdleMotionProfile"))?;
        let hips = world
            .get::<HipsBoneEntity>(root)
            .map(|hips| hips.0)
            .ok_or_else(|| format!("Ready root {root:?} has no HipsBoneEntity"))?;
        (profile, hips)
    };
    profile
        .validate()
        .map_err(|error| format!("idle contract invalid on {root:?}: {error}"))?;
    if profile.procedural_amplitude_meters != IDLE_PROCEDURAL_AMPLITUDE_METERS {
        return Err("idle contract amplitude must stay zero".to_owned());
    }

    let base = app
        .world()
        .get::<Transform>(hips)
        .ok_or_else(|| format!("hips entity {hips:?} has no Transform"))?
        .translation;
    if !base.is_finite() {
        return Err(format!(
            "hips entity {hips:?} has non-finite initial translation"
        ));
    }

    for _ in 0..IDLE_FRAMES {
        std::thread::sleep(IDLE_FRAME_PACE);
        app.update();
        let current = app
            .world()
            .get::<Transform>(hips)
            .ok_or_else(|| format!("hips entity {hips:?} lost its Transform"))?
            .translation;
        if !current.is_finite() {
            return Err(format!(
                "hips entity {hips:?} received a non-finite translation"
            ));
        }
        if current != base {
            return Err(format!(
                "hips translation changed without an idle writer: base={base:?} current={current:?}"
            ));
        }
    }

    println!(
        "idle contract verified: hips={hips:?} amplitude={}m",
        profile.procedural_amplitude_meters
    );
    Ok(())
}

#[derive(Resource)]
struct ManagedImportedAvatar(ImportedAvatar);

fn emit_request(
    model: Res<ManagedImportedAvatar>,
    mut requests: MessageWriter<LoadImportedAvatarRequest>,
) {
    requests.write(LoadImportedAvatarRequest {
        request_id: 1,
        imported: model.0.clone(),
    });
}

fn print_asset_failures(app: &mut App) {
    let handles: Vec<(Entity, bevy::asset::AssetId<VrmAsset>, Option<String>)> = {
        let world = app.world_mut();
        let mut query = world.query::<(Entity, &VrmHandle)>();
        query
            .iter(world)
            .map(|(entity, handle)| {
                (
                    entity,
                    handle.0.id(),
                    handle.0.path().map(|path| path.to_string()),
                )
            })
            .collect()
    };

    let asset_server = app.world().resource::<AssetServer>();
    for (entity, id, path) in handles {
        if let LoadState::Failed(error) = asset_server.load_state(id) {
            println!("underlying asset failure: root={entity:?} path={path:?} error={error:?}");
        }
    }
}

fn verify_control_episode(app: &mut App, root: Entity) -> Result<(), String> {
    use vtuber_avatar::set_active_control_frame;
    use vtuber_core::types::{FrameSeq, HeadPose, HeadTranslationSignal, TrackingState};

    const MOTION_FRAMES: usize = 45;
    const LOSS_FRAMES: usize = 40;
    const REACQUIRE_FRAMES: usize = 20;

    let generation = app
        .world()
        .resource::<AvatarLifecycle>()
        .current_generation();

    let sample_transforms = |app: &App| -> Result<Vec<(Entity, Transform)>, String> {
        let binding = app
            .world()
            .get::<AvatarBinding>(root)
            .ok_or_else(|| format!("root {root:?} lost AvatarBinding during the episode"))?;
        let mut samples = Vec::new();
        let mut collect = |entity: Entity| -> Result<(), String> {
            let transform = app
                .world()
                .get::<Transform>(entity)
                .ok_or_else(|| format!("entity {entity:?} lost Transform"))?;
            if !transform.is_finite() {
                return Err(format!("entity {entity:?} has a non-finite Transform"));
            }
            samples.push((entity, *transform));
            Ok(())
        };
        collect(binding.head)?;
        if let Some(spine) = binding.spine {
            collect(spine)?;
        }
        if let Some(chest) = binding.chest {
            collect(chest)?;
        }
        if let Some(arms) = &binding.left_arm {
            collect(arms.upper_arm)?;
            collect(arms.lower_arm)?;
        }
        if let Some(arms) = &binding.right_arm {
            collect(arms.upper_arm)?;
            collect(arms.lower_arm)?;
        }
        Ok(samples)
    };

    let push = |app: &mut App,
                seq: u64,
                state: TrackingState,
                pose: HeadPose,
                translation: Option<[f32; 3]>| {
        let frame = vtuber_core::types::AvatarControlFrame {
            source_seq: FrameSeq(seq),
            captured_at: vtuber_core::types::MonoTimeNs(0),
            produced_at: vtuber_core::types::MonoTimeNs(0),
            confidence: 1.0,
            state,
            head: pose,
            head_translation: match translation {
                Some([x, y, z]) => HeadTranslationSignal::tracked(x, y, z),
                None => HeadTranslationSignal::UNAVAILABLE,
            },
            gaze: vtuber_core::types::GazeSignal::UNAVAILABLE,
            expressions: vtuber_core::types::ExpressionCoefficients::default(),
            detailed_face: None,
        };
        let result =
            app.world_mut().resource_scope(
                |world: &mut bevy::ecs::world::World,
                 mut active: bevy::ecs::change_detection::Mut<
                    vtuber_avatar::ActiveControlFrame,
                >| {
                    let lifecycle = world.resource::<AvatarLifecycle>();
                    set_active_control_frame(lifecycle, generation, frame, &mut active)
                },
            );
        result.map_err(|error| format!("control frame injection failed: {error}"))
    };

    let mut last_head_rotation = Option::<bevy::math::Quat>::None;
    for seq in 0..MOTION_FRAMES as u64 {
        let phase = seq as f32 / MOTION_FRAMES as f32;
        let pose = HeadPose {
            yaw_rad: 0.5 * (phase * std::f32::consts::TAU).sin(),
            pitch_rad: 0.2 * (phase * std::f32::consts::TAU).cos(),
            roll_rad: 0.1 * phase,
        };
        push(
            app,
            seq + 1,
            TrackingState::Tracking,
            pose,
            Some([0.04 * (phase * std::f32::consts::TAU).sin(), 0.01, 0.02]),
        )?;
        app.update();
        let samples = sample_transforms(app)?;
        let head = samples
            .first()
            .ok_or_else(|| "trace sample list was empty".to_owned())?
            .1
            .rotation;
        if let Some(previous) = last_head_rotation
            && previous.angle_between(head) < 1.0e-9
            && seq > 2
        {
            return Err("tracked head rotation stopped responding to input".to_owned());
        }
        last_head_rotation = Some(head);
    }
    println!("control episode motion verified ({} frames)", MOTION_FRAMES);

    for seq in 0..LOSS_FRAMES as u64 {
        push(
            app,
            MOTION_FRAMES as u64 + seq + 1,
            TrackingState::LostHold,
            HeadPose::default(),
            None,
        )?;
        app.update();
        sample_transforms(app)?;
    }
    println!("control episode loss verified ({} frames)", LOSS_FRAMES);

    let before = sample_transforms(app)?;
    for seq in 0..REACQUIRE_FRAMES as u64 {
        let phase = seq as f32 / REACQUIRE_FRAMES as f32;
        push(
            app,
            MOTION_FRAMES as u64 + LOSS_FRAMES as u64 + seq + 1,
            TrackingState::Tracking,
            HeadPose {
                yaw_rad: -0.35 + 0.1 * phase,
                pitch_rad: 0.15,
                roll_rad: -0.05,
            },
            Some([-0.03, 0.0, 0.01]),
        )?;
        app.update();
        sample_transforms(app)?;
    }
    let after = sample_transforms(app)?;
    let moved = before
        .iter()
        .zip(after.iter())
        .any(|((ea, ta), (eb, tb))| ea == eb && ta.rotation.angle_between(tb.rotation) > 1.0e-4);
    if !moved {
        return Err("reacquire did not update the tracked pose".to_owned());
    }
    println!(
        "control episode reacquire verified ({} frames)",
        REACQUIRE_FRAMES
    );
    Ok(())
}

fn trigger_replacement(
    app: &mut App,
    _old_generation: vtuber_avatar::AvatarGeneration,
) -> Result<(), String> {
    use vtuber_avatar::LoadImportedAvatarRequest;

    app.world_mut().resource_scope(
        |world: &mut bevy::ecs::world::World,
         managed: bevy::ecs::change_detection::Mut<ManagedImportedAvatar>| {
            let imported = managed.0.clone();
            let mut requests =
                world.resource_mut::<bevy::ecs::message::Messages<LoadImportedAvatarRequest>>();
            requests.write(LoadImportedAvatarRequest {
                request_id: 2,
                imported,
            });
        },
    );
    Ok(())
}

fn wait_for_replacement_ready(
    app: &mut App,
    old_generation: vtuber_avatar::AvatarGeneration,
    max_frames: usize,
) -> Result<Entity, String> {
    for _frame in 0..max_frames {
        app.update();
        let lifecycle = app.world().resource::<AvatarLifecycle>();
        let (state, failure) = (lifecycle.state(), lifecycle.failure().cloned());
        match state {
            AvatarLifecycleState::Ready => {
                let root = lifecycle
                    .active_root()
                    .ok_or_else(|| "replacement reached Ready without an active root".to_owned())?;
                if lifecycle.current_generation() == old_generation {
                    return Err("replacement reached Ready with the old generation".to_owned());
                }
                return Ok(root);
            }
            AvatarLifecycleState::Failed => {
                print_asset_failures(app);
                return Err(format!("replacement lifecycle failed: {failure:?}"));
            }
            _ => {}
        }
    }
    Err(format!(
        "replacement did not reach Ready within {max_frames} frames"
    ))
}

fn temporary_managed_root() -> Result<PathBuf, String> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock is before UNIX epoch: {error}"))?;
    let root = std::env::temp_dir().join(format!(
        "vrm-managed-compat-{}-{}",
        std::process::id(),
        timestamp.as_nanos()
    ));
    fs::create_dir_all(&root)
        .map_err(|error| format!("failed to create temporary managed root: {error}"))?;
    Ok(root)
}
