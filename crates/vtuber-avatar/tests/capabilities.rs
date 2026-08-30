// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Integration tests for expression capability discovery.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy_vrm1::prelude::{ExpressionEntityMap, VrmExpression};

use vtuber_avatar::capabilities::{
    BlinkMode, EmotionSet, ExpressionCapabilities, LookDirectionSet, MouthMode,
};

fn expression_map(names: &[&str]) -> ExpressionEntityMap {
    let mut world = World::new();
    let mut map = HashMap::default();
    for name in names {
        let entity = world.spawn_empty().id();
        map.insert(VrmExpression::from(*name), entity);
    }
    ExpressionEntityMap(map)
}

#[test]
fn expression_capabilities_map_missing_is_empty() {
    let caps = ExpressionCapabilities::from_map(None);

    assert_eq!(caps.blink, BlinkMode::None);
    assert_eq!(caps.mouth, MouthMode::None);
    assert_eq!(caps.look, LookDirectionSet::default());
    assert_eq!(caps.emotions, EmotionSet::default());
    assert!(caps.unknown.is_empty());
}

#[test]
fn expression_capabilities_no_expressions_is_empty() {
    let map = expression_map(&[]);
    let caps = ExpressionCapabilities::from_map(Some(&map));

    assert!(!caps.has_blink());
    assert!(!caps.has_mouth());
    assert!(!caps.look.any());
    assert!(!caps.emotions.any());
    assert!(caps.unknown.is_empty());
}

#[test]
fn expression_capabilities_per_eye_blink_priority() {
    let map = expression_map(&["blinkLeft", "blinkRight", "blink"]);
    let caps = ExpressionCapabilities::from_map(Some(&map));

    assert_eq!(caps.blink, BlinkMode::PerEye);
    assert!(caps.has_blink());
}

#[test]
fn expression_capabilities_combined_blink_fallback() {
    let map = expression_map(&["blink"]);
    let caps = ExpressionCapabilities::from_map(Some(&map));

    assert_eq!(caps.blink, BlinkMode::Combined);
}

#[test]
fn expression_capabilities_full_mouth_priority() {
    let map = expression_map(&["aa", "ih", "ou", "ee", "oh"]);
    let caps = ExpressionCapabilities::from_map(Some(&map));

    assert_eq!(caps.mouth, MouthMode::Full);
    assert!(caps.has_mouth());
}

#[test]
fn expression_capabilities_aa_only_fallback() {
    let map = expression_map(&["aa"]);
    let caps = ExpressionCapabilities::from_map(Some(&map));

    assert_eq!(caps.mouth, MouthMode::AaOnly);
}

#[test]
fn expression_capabilities_look_directions() {
    let map = expression_map(&["lookLeft", "lookRight", "lookUp", "lookDown"]);
    let caps = ExpressionCapabilities::from_map(Some(&map));

    assert!(caps.look.left);
    assert!(caps.look.right);
    assert!(caps.look.up);
    assert!(caps.look.down);
}

#[test]
fn expression_capabilities_emotions() {
    let map = expression_map(&["happy", "angry", "sad", "relaxed", "surprised"]);
    let caps = ExpressionCapabilities::from_map(Some(&map));

    assert!(caps.emotions.happy);
    assert!(caps.emotions.angry);
    assert!(caps.emotions.sad);
    assert!(caps.emotions.relaxed);
    assert!(caps.emotions.surprised);
    assert!(caps.emotions.any());
}

#[test]
fn expression_capabilities_custom_not_mapped() {
    let map = expression_map(&["myCustomShape", "aa"]);
    let caps = ExpressionCapabilities::from_map(Some(&map));

    assert_eq!(caps.mouth, MouthMode::AaOnly);
    assert_eq!(caps.unknown, vec!["myCustomShape"]);
}

#[test]
fn expression_capabilities_order_independent() {
    let a = expression_map(&["oh", "aa", "surprised", "ee", "ih", "ou"]);
    let b = expression_map(&["aa", "ih", "ou", "ee", "oh", "surprised"]);

    assert_eq!(
        ExpressionCapabilities::from_map(Some(&a)),
        ExpressionCapabilities::from_map(Some(&b))
    );
}
