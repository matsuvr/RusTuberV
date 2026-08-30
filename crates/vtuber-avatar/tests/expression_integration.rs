// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Comprehensive blink/mouth/gaze capability integration tests.
//!
//! Exercises the full expression pipeline:
//! - Per-eye blink, blink-only, no-blink
//! - Full mouth, aa-only, no-mouth
//! - Exclusive expression gaze, eye-bone gaze, no-gaze selection
//! - Coalescing, epsilon, zero-reset, generation-reset
//! - Missing capabilities don't panic

use vtuber_avatar::capabilities::{
    AvatarCapabilities, BlinkMode, BonePresence, DeclaredLookAtType, GazeFallbackReason,
    LookDirectionSet, MouthMode, PerfectSyncCapabilities, SelectedGazeBackend, select_gaze_backend,
};
use vtuber_avatar::expression::{
    ExpressionCommand, ExpressionCommandBuilder, ExpressionStateTracker, RawBlinkInput,
    RawMouthInput, build_face_commands, map_blink_to_expressions, map_blink_with_fallback,
    map_mouth_with_fallback,
};
use vtuber_core::{ARKIT52_CHANNEL_COUNT, Arkit52Coefficients, ArkitBlendshape};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn full_caps() -> AvatarCapabilities {
    AvatarCapabilities {
        bones: BonePresence {
            head: true,
            neck: true,
            left_eye: true,
            right_eye: true,
            upper_chest: false,
            chest: false,
            spine: false,
        },
        blink: BlinkMode::PerEye,
        mouth: MouthMode::Full,
        declared_look_at: DeclaredLookAtType::Expression,
        gaze_backend: SelectedGazeBackend::Expression,
        gaze_fallback: None,
        look_directions: LookDirectionSet {
            left: true,
            right: true,
            up: true,
            down: true,
        },
        spring_bone: true,
        unknown_expressions: Vec::new(),
        perfect_sync: Default::default(),
    }
}

fn blink_only_caps() -> AvatarCapabilities {
    AvatarCapabilities {
        blink: BlinkMode::Combined,
        mouth: MouthMode::None,
        declared_look_at: DeclaredLookAtType::Missing,
        gaze_backend: SelectedGazeBackend::None,
        gaze_fallback: Some(GazeFallbackReason::MetadataMissing),
        look_directions: LookDirectionSet::default(),
        ..full_caps()
    }
}

fn aa_only_caps() -> AvatarCapabilities {
    AvatarCapabilities {
        blink: BlinkMode::None,
        mouth: MouthMode::AaOnly,
        declared_look_at: DeclaredLookAtType::Missing,
        gaze_backend: SelectedGazeBackend::None,
        gaze_fallback: Some(GazeFallbackReason::MetadataMissing),
        look_directions: LookDirectionSet::default(),
        ..full_caps()
    }
}

fn no_mouth_caps() -> AvatarCapabilities {
    AvatarCapabilities {
        mouth: MouthMode::None,
        ..full_caps()
    }
}

fn no_gaze_caps() -> AvatarCapabilities {
    AvatarCapabilities {
        declared_look_at: DeclaredLookAtType::Missing,
        gaze_backend: SelectedGazeBackend::None,
        gaze_fallback: Some(GazeFallbackReason::MetadataMissing),
        look_directions: LookDirectionSet::default(),
        bones: BonePresence {
            left_eye: false,
            right_eye: false,
            ..full_caps().bones
        },
        ..full_caps()
    }
}

fn build_all_commands(
    blink: &[(String, f32)],
    mouth: &[(String, f32)],
    gaze: &[(String, f32)],
) -> Vec<ExpressionCommand> {
    let mut builder = ExpressionCommandBuilder::new();
    let all: Vec<(&str, f32)> = blink
        .iter()
        .chain(mouth.iter())
        .chain(gaze.iter())
        .map(|(n, w)| (n.as_str(), *w))
        .collect();
    builder.build(all)
}

fn detailed_coefficients(values: &[(ArkitBlendshape, f32)]) -> Arkit52Coefficients {
    let mut coefficients = [0.0; ARKIT52_CHANNEL_COUNT];
    for &(channel, value) in values {
        coefficients[channel.index()] = value;
    }
    Arkit52Coefficients::try_from_array(coefficients).expect("test coefficients are finite")
}

fn all_perfect_sync_caps() -> AvatarCapabilities {
    let mut caps = full_caps();
    caps.perfect_sync = PerfectSyncCapabilities::from_names(
        ArkitBlendshape::ALL
            .into_iter()
            .map(ArkitBlendshape::canonical_name),
    );
    caps
}

fn detailed_name(channel: ArkitBlendshape) -> Option<String> {
    Some(channel.canonical_name().to_owned())
}

// ---------------------------------------------------------------------------
// Per-eye blink model
// ---------------------------------------------------------------------------

#[test]
fn expression_integration_per_eye_blink_asymmetric() {
    let caps = full_caps();
    assert_eq!(caps.blink, BlinkMode::PerEye);

    let input = RawBlinkInput {
        left: 0.8,
        right: 0.2,
        combined: 0.0,
    };
    let blink_cmds = map_blink_to_expressions(&input, caps.blink);

    assert_eq!(blink_cmds.len(), 2);
    assert!(
        blink_cmds
            .iter()
            .any(|(n, w)| n == "blinkLeft" && (*w - 0.8).abs() < 0.01)
    );
    assert!(
        blink_cmds
            .iter()
            .any(|(n, w)| n == "blinkRight" && (*w - 0.2).abs() < 0.01)
    );
}

// ---------------------------------------------------------------------------
// Blink-only model
// ---------------------------------------------------------------------------

#[test]
fn expression_integration_blink_only_model() {
    let caps = blink_only_caps();
    assert_eq!(caps.blink, BlinkMode::Combined);
    assert_eq!(caps.mouth, MouthMode::None);

    let input = RawBlinkInput {
        left: 0.0,
        right: 0.0,
        combined: 0.7,
    };
    let blink_cmds = map_blink_with_fallback(&input, caps.blink);

    assert_eq!(blink_cmds.len(), 1);
    assert_eq!(blink_cmds[0].0, "blink");
    assert!((blink_cmds[0].1 - 0.7).abs() < 0.01);

    // No mouth commands.
    let mouth_input = RawMouthInput::default();
    let mouth_cmds = map_mouth_with_fallback(&mouth_input, caps.mouth);
    assert!(mouth_cmds.is_empty());
}

// ---------------------------------------------------------------------------
// aa-only model
// ---------------------------------------------------------------------------

#[test]
fn expression_integration_aa_only_model() {
    let caps = aa_only_caps();
    assert_eq!(caps.mouth, MouthMode::AaOnly);

    let input = RawMouthInput {
        openness: 0.6,
        aa: 0.0,
        ih: 0.0,
        ou: 0.0,
        ee: 0.0,
        oh: 0.0,
    };
    let mouth_cmds = map_mouth_with_fallback(&input, caps.mouth);

    assert_eq!(mouth_cmds.len(), 1);
    assert_eq!(mouth_cmds[0].0, "aa");
    assert!((mouth_cmds[0].1 - 0.6).abs() < 0.01);
}

// ---------------------------------------------------------------------------
// No-mouth model
// ---------------------------------------------------------------------------

#[test]
fn expression_integration_no_mouth_no_panic() {
    let caps = no_mouth_caps();
    let input = RawMouthInput {
        openness: 0.5,
        aa: 0.5,
        ih: 0.5,
        ou: 0.5,
        ee: 0.5,
        oh: 0.5,
    };
    let mouth_cmds = map_mouth_with_fallback(&input, caps.mouth);
    assert!(mouth_cmds.is_empty());
}

// ---------------------------------------------------------------------------
// Expression gaze
// ---------------------------------------------------------------------------

#[test]
fn expression_integration_expression_gaze() {
    let caps = full_caps();
    let selection = select_gaze_backend(caps.declared_look_at, &caps.look_directions, &caps.bones);
    assert_eq!(selection, (SelectedGazeBackend::Expression, None));
}

// ---------------------------------------------------------------------------
// Eye-bone gaze
// ---------------------------------------------------------------------------

#[test]
fn expression_integration_eye_bone_gaze() {
    let caps = AvatarCapabilities {
        declared_look_at: DeclaredLookAtType::Bone,
        gaze_backend: SelectedGazeBackend::Bone,
        gaze_fallback: None,
        look_directions: LookDirectionSet::default(),
        ..full_caps()
    };
    let selection = select_gaze_backend(caps.declared_look_at, &caps.look_directions, &caps.bones);
    assert_eq!(selection, (SelectedGazeBackend::Bone, None));
}

// ---------------------------------------------------------------------------
// No-gaze model
// ---------------------------------------------------------------------------

#[test]
fn expression_integration_no_gaze_model() {
    let caps = no_gaze_caps();
    let selection = select_gaze_backend(caps.declared_look_at, &caps.look_directions, &caps.bones);
    assert_eq!(selection.0, SelectedGazeBackend::None);
    assert_eq!(selection.1, Some(GazeFallbackReason::MetadataMissing));
}

// ---------------------------------------------------------------------------
// Coalescing: one event per frame
// ---------------------------------------------------------------------------

#[test]
fn expression_integration_one_event_per_frame() {
    let blink_cmds = vec![("blinkLeft".to_string(), 0.5)];
    let mouth_cmds = vec![("aa".to_string(), 0.3)];
    let gaze_cmds = vec![("lookRight".to_string(), 0.4)];

    let commands = build_all_commands(&blink_cmds, &mouth_cmds, &gaze_cmds);

    // All commands should be merged into a single list.
    assert_eq!(commands.len(), 3);
}

// ---------------------------------------------------------------------------
// Per-domain Perfect Sync fallback
// ---------------------------------------------------------------------------

#[test]
fn expression_integration_full_52_owns_all_coarse_domains() {
    let caps = all_perfect_sync_caps();
    let coefficients = detailed_coefficients(&[(ArkitBlendshape::JawOpen, 0.4)]);
    let commands = build_face_commands(
        Some(&coefficients),
        &caps,
        &[("blinkLeft".to_owned(), 0.7)],
        &[("aa".to_owned(), 0.8)],
        &[("lookLeft".to_owned(), 0.9)],
        detailed_name,
    );

    assert_eq!(commands.len(), ARKIT52_CHANNEL_COUNT);
    assert!(commands.iter().any(|command| {
        command.name == "JawOpen" && (command.weight - 0.4).abs() < f32::EPSILON
    }));
    assert!(!commands.iter().any(|command| command.name == "blinkLeft"));
    assert!(!commands.iter().any(|command| command.name == "aa"));
    assert!(!commands.iter().any(|command| command.name == "lookLeft"));
}

#[test]
fn expression_integration_tongue_only_keeps_coarse_blink_and_mouth() {
    let mut caps = full_caps();
    caps.perfect_sync = PerfectSyncCapabilities::from_names(["TongueOut"]);
    let coefficients = detailed_coefficients(&[(ArkitBlendshape::TongueOut, 1.0)]);
    let commands = build_face_commands(
        Some(&coefficients),
        &caps,
        &[("blinkLeft".to_owned(), 0.7)],
        &[("aa".to_owned(), 0.8)],
        &[],
        detailed_name,
    );

    assert!(commands.iter().any(|command| command.name == "TongueOut"));
    assert!(commands.iter().any(|command| command.name == "blinkLeft"));
    assert!(commands.iter().any(|command| command.name == "aa"));
}

#[test]
fn expression_integration_one_eye_look_channel_falls_back_to_existing_gaze() {
    let mut caps = full_caps();
    caps.perfect_sync = PerfectSyncCapabilities::from_names(["EyeLookUpLeft"]);
    let coefficients = detailed_coefficients(&[(ArkitBlendshape::EyeLookUpLeft, 1.0)]);
    let commands = build_face_commands(
        Some(&coefficients),
        &caps,
        &[],
        &[],
        &[("lookLeft".to_owned(), 0.9)],
        detailed_name,
    );

    assert!(commands.iter().any(|command| command.name == "lookLeft"));
    assert!(
        !commands
            .iter()
            .any(|command| command.name == "EyeLookUpLeft")
    );
}

#[test]
fn expression_integration_complete_eye_look_coverage_replaces_existing_gaze() {
    let mut caps = full_caps();
    caps.perfect_sync = PerfectSyncCapabilities::from_names([
        "EyeLookDownLeft",
        "EyeLookDownRight",
        "EyeLookInLeft",
        "EyeLookInRight",
        "EyeLookOutLeft",
        "EyeLookOutRight",
        "EyeLookUpLeft",
        "EyeLookUpRight",
    ]);
    let coefficients = detailed_coefficients(&[(ArkitBlendshape::EyeLookUpLeft, 1.0)]);
    let commands = build_face_commands(
        Some(&coefficients),
        &caps,
        &[],
        &[],
        &[("lookLeft".to_owned(), 0.9)],
        detailed_name,
    );

    assert!(
        commands
            .iter()
            .any(|command| command.name == "EyeLookUpLeft")
    );
    assert!(!commands.iter().any(|command| command.name == "lookLeft"));
}

#[test]
fn expression_integration_detailed_one_to_zero_emits_explicit_reset() {
    let mut tracker = ExpressionStateTracker::new();
    let active = vec![ExpressionCommand {
        name: "TongueOut".to_owned(),
        weight: 1.0,
    }];
    let neutral = vec![ExpressionCommand {
        name: "TongueOut".to_owned(),
        weight: 0.0,
    }];

    assert!(tracker.compute_commands(&active, 1).is_some());
    let reset = tracker
        .compute_commands(&neutral, 1)
        .expect("detailed 1.0 -> 0.0 must be sent");
    assert_eq!(reset, neutral);
}

// ---------------------------------------------------------------------------
// Epsilon: steady neutral skips
// ---------------------------------------------------------------------------

#[test]
fn expression_integration_steady_neutral_skips() {
    let mut tracker = ExpressionStateTracker::new();
    let commands = vec![ExpressionCommand {
        name: "blink".to_string(),
        weight: 0.5,
    }];

    // First frame sends.
    let result1 = tracker.compute_commands(&commands, 1);
    assert!(result1.is_some());

    // Same commands next frame skips.
    let result2 = tracker.compute_commands(&commands, 1);
    assert!(result2.is_none());
}

// ---------------------------------------------------------------------------
// Zero reset: previous nonzero returns to zero
// ---------------------------------------------------------------------------

#[test]
fn expression_integration_zero_reset_on_neutral() {
    let mut tracker = ExpressionStateTracker::new();

    // Frame 1: blink is active.
    let active = vec![ExpressionCommand {
        name: "blink".to_string(),
        weight: 0.8,
    }];
    let _ = tracker.compute_commands(&active, 1);

    // Frame 2: blink returns to neutral (empty commands).
    let neutral: Vec<ExpressionCommand> = vec![];
    let result = tracker.compute_commands(&neutral, 1);

    // Should send explicit zero for blink.
    assert!(result.is_some());
    let cmds = result.unwrap();
    assert!(cmds.iter().any(|c| c.name == "blink" && c.weight == 0.0));
}

// ---------------------------------------------------------------------------
// Generation reset
// ---------------------------------------------------------------------------

#[test]
fn expression_integration_generation_reset() {
    let mut tracker = ExpressionStateTracker::new();
    let commands = vec![ExpressionCommand {
        name: "blink".to_string(),
        weight: 0.5,
    }];

    // Frame 1 with generation 1.
    let _ = tracker.compute_commands(&commands, 1);

    // Same commands but generation 2 → should send again.
    let result = tracker.compute_commands(&commands, 2);
    assert!(result.is_some(), "generation change should force send");
    assert_eq!(tracker.generation(), 2);
}

// ---------------------------------------------------------------------------
// Missing capability doesn't panic
// ---------------------------------------------------------------------------

#[test]
fn expression_integration_missing_capability_no_panic() {
    // A model with no blink, no mouth, no gaze.
    let caps = AvatarCapabilities {
        blink: BlinkMode::None,
        mouth: MouthMode::None,
        declared_look_at: DeclaredLookAtType::Missing,
        gaze_backend: SelectedGazeBackend::None,
        gaze_fallback: Some(GazeFallbackReason::MetadataMissing),
        look_directions: LookDirectionSet::default(),
        bones: BonePresence {
            left_eye: false,
            right_eye: false,
            ..full_caps().bones
        },
        ..full_caps()
    };

    // All mappings should produce empty output without panic.
    let blink_input = RawBlinkInput::default();
    let blink_cmds = map_blink_with_fallback(&blink_input, caps.blink);
    assert!(blink_cmds.is_empty());

    let mouth_input = RawMouthInput::default();
    let mouth_cmds = map_mouth_with_fallback(&mouth_input, caps.mouth);
    assert!(mouth_cmds.is_empty());

    let gaze_cmds: Vec<(String, f32)> = Vec::new();

    // Building commands from all empty lists should produce empty.
    let commands = build_all_commands(&blink_cmds, &mouth_cmds, &gaze_cmds);
    assert!(commands.is_empty());
}

// ---------------------------------------------------------------------------
// M1-06 acceptance criteria verification
// ---------------------------------------------------------------------------

#[test]
fn expression_integration_m106_acceptance_criteria() {
    // 1. Per-eye blink model works asymmetrically.
    let per_eye_input = RawBlinkInput {
        left: 0.9,
        right: 0.1,
        combined: 0.0,
    };
    let per_eye_cmds = map_blink_to_expressions(&per_eye_input, BlinkMode::PerEye);
    assert_eq!(per_eye_cmds.len(), 2);

    // 2. Blink-only model works.
    let combined_input = RawBlinkInput {
        left: 0.0,
        right: 0.0,
        combined: 0.5,
    };
    let combined_cmds = map_blink_with_fallback(&combined_input, BlinkMode::Combined);
    assert_eq!(combined_cmds.len(), 1);

    // 3. No mouth preset doesn't panic.
    let no_mouth = map_mouth_with_fallback(&RawMouthInput::default(), MouthMode::None);
    assert!(no_mouth.is_empty());

    // 4. Gaze mode is visible in capabilities.
    let caps = full_caps();
    let selection = select_gaze_backend(caps.declared_look_at, &caps.look_directions, &caps.bones);
    assert_eq!(selection.0, SelectedGazeBackend::Expression);
}
