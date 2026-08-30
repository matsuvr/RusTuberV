//! Avatar unload cleanup and stale control-frame rejection.
//!
//! This module handles:
//! - Recursive despawn of the active avatar hierarchy.
//! - Clearing binding, capability, and control caches.
//! - Rejecting control frames that target a previous avatar generation.
//!
//! SpringBone, MToon, and other `bevy_vrm1` internals are cleaned up as part
//! of Bevy's recursive scene despawn; this module does not touch them
//! individually.

use bevy::prelude::*;
use vtuber_core::types::AvatarControlFrame;

use crate::binding::AvatarBinding;
use crate::lifecycle::{ActiveAvatar, AvatarGeneration, AvatarLifecycle, AvatarLifecycleState};

/// Resource holding the latest control frame intended for the active avatar.
///
/// This cache is cleared when the active avatar generation changes or the
/// avatar is no longer ready.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct ActiveControlFrame {
    /// Generation this frame was tagged with.
    pub generation: AvatarGeneration,
    /// The frame payload, if any.
    pub frame: Option<AvatarControlFrame>,
}

/// Errors when applying a control frame to an avatar binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ControlFrameError {
    /// The frame targets a different avatar generation than the active binding.
    StaleGeneration {
        /// Generation the frame targets.
        frame_generation: AvatarGeneration,
        /// Generation of the active binding.
        binding_generation: AvatarGeneration,
    },
    /// The active avatar binding has been removed or despawned.
    StaleBinding,
    /// The lifecycle is not in the Ready state.
    NotReady {
        /// Current lifecycle state.
        state: AvatarLifecycleState,
    },
}

impl std::fmt::Display for ControlFrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StaleGeneration {
                frame_generation,
                binding_generation,
            } => {
                write!(
                    f,
                    "control frame generation {frame_generation:?} does not match binding generation {binding_generation:?}"
                )
            }
            Self::StaleBinding => write!(f, "avatar binding is stale or despawned"),
            Self::NotReady { state } => {
                write!(f, "cannot apply control frame in lifecycle state {state:?}")
            }
        }
    }
}

impl std::error::Error for ControlFrameError {}

impl ControlFrameError {
    /// Stable string code for UI mapping and logs.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::StaleGeneration { .. } => "CONTROL_FRAME_STALE_GENERATION",
            Self::StaleBinding => "CONTROL_FRAME_STALE_BINDING",
            Self::NotReady { .. } => "CONTROL_FRAME_NOT_READY",
        }
    }
}

/// Tags a raw control frame with the current active generation.
///
/// Returns `None` when no avatar is active, so the frame can be dropped.
#[must_use]
pub fn tag_control_frame(
    frame: AvatarControlFrame,
    lifecycle: &AvatarLifecycle,
) -> Option<(AvatarGeneration, AvatarControlFrame)> {
    if !lifecycle.has_active_generation() {
        return None;
    }
    Some((lifecycle.current_generation(), frame))
}

/// Sets the active control frame after validating it matches the current
/// active generation.
///
/// On mismatch, the cache is cleared and a [`ControlFrameError::StaleGeneration`]
/// error is returned.
pub fn set_active_control_frame(
    lifecycle: &AvatarLifecycle,
    generation: AvatarGeneration,
    frame: AvatarControlFrame,
    active: &mut ActiveControlFrame,
) -> Result<(), ControlFrameError> {
    let current = lifecycle.current_generation();
    if generation != current {
        active.frame = None;
        return Err(ControlFrameError::StaleGeneration {
            frame_generation: generation,
            binding_generation: current,
        });
    }
    active.generation = generation;
    active.frame = Some(frame);
    Ok(())
}

/// Applies the active control frame to the given binding if generations match.
///
/// Returns the frame when it can be applied. Returns `Ok(None)` when there is
/// no active frame. Returns an error when the binding is stale or the
/// generations do not match.
#[allow(clippy::missing_errors_doc)]
pub fn apply_active_control_frame<'a>(
    lifecycle: &'a AvatarLifecycle,
    active: &'a ActiveControlFrame,
    binding: Option<&'a AvatarBinding>,
) -> Result<Option<&'a AvatarControlFrame>, ControlFrameError> {
    if lifecycle.state() != AvatarLifecycleState::Ready {
        return Err(ControlFrameError::NotReady {
            state: lifecycle.state(),
        });
    }
    let binding = binding.ok_or(ControlFrameError::StaleBinding)?;
    let Some(frame) = active.frame.as_ref() else {
        return Ok(None);
    };
    if active.generation != binding.generation {
        return Err(ControlFrameError::StaleGeneration {
            frame_generation: active.generation,
            binding_generation: binding.generation,
        });
    }
    Ok(Some(frame))
}

/// Recursively despawns an avatar root and all its descendants.
///
/// A missing entity is treated as already cleaned up. This delegates
/// SpringBone, MToon, and other `bevy_vrm1` component cleanup to Bevy's
/// recursive scene despawn.
pub fn recursive_despawn_avatar(commands: &mut Commands, root: Entity) {
    if let Ok(mut entity_commands) = commands.get_entity(root) {
        entity_commands.despawn();
    }
}

/// Despawns the active avatar root once an unload or replacement has been
/// requested.
///
/// This system drives the `Unloading -> Loading/NoAvatar` transition. It
/// recursively removes the old root (and with it any [`AvatarBinding`],
/// expression state, and `bevy_vrm1` components), then promotes a pending
/// replacement root to active by adding the [`ActiveAvatar`] marker. The new
/// root is not marked active until the old root has been removed, preserving
/// the single-active-avatar invariant at the ECS level.
///
/// If the replacement load fails after this point, the slot moves to `Failed`
/// and remains empty; the old avatar has already been despawned and is not
/// revived.
pub fn despawn_unloading_avatar(mut commands: Commands, mut lifecycle: ResMut<AvatarLifecycle>) {
    if lifecycle.state() != AvatarLifecycleState::Unloading {
        return;
    }

    let Some(old_root) = lifecycle.active_root() else {
        // No active root to remove; finish the unload immediately.
        lifecycle.finish_unload();
        return;
    };

    recursive_despawn_avatar(&mut commands, old_root);
    lifecycle.finish_unload();

    // Promote the pending root to active now that the old root is gone.
    if let Some(new_root) = lifecycle.active_root()
        && let Ok(mut entity_commands) = commands.get_entity(new_root)
    {
        entity_commands.insert(ActiveAvatar);
    }
}

/// Clears the control cache when the active avatar generation changes or the
/// avatar is no longer ready.
pub(crate) fn clear_control_cache_on_lifecycle_change(
    lifecycle: Res<AvatarLifecycle>,
    mut active: ResMut<ActiveControlFrame>,
) {
    if lifecycle.is_changed() {
        let should_clear = lifecycle.state() != AvatarLifecycleState::Ready
            || active.generation != lifecycle.current_generation();
        if should_clear {
            active.frame = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vtuber_core::types::{
        ExpressionCoefficients, FrameSeq, HeadPose, MonoTimeNs, TrackingState,
    };

    fn dummy_frame() -> AvatarControlFrame {
        AvatarControlFrame {
            source_seq: FrameSeq(1),
            captured_at: MonoTimeNs(0),
            produced_at: MonoTimeNs(0),
            confidence: 1.0,
            state: TrackingState::Tracking,
            head: HeadPose::default(),
            head_translation: vtuber_core::HeadTranslationSignal::UNAVAILABLE,
            gaze: vtuber_core::GazeSignal::UNAVAILABLE,
            expressions: ExpressionCoefficients::default(),
            detailed_face: None,
        }
    }

    #[test]
    fn control_frame_tagged_with_current_generation() {
        let mut lifecycle = AvatarLifecycle::new();
        let root = Entity::from_raw_u32(1).unwrap();
        lifecycle.request_load(root).unwrap();

        let current = lifecycle.current_generation();
        assert_ne!(current, AvatarGeneration::default());

        let tagged = tag_control_frame(dummy_frame(), &lifecycle);
        assert!(tagged.is_some());
        let (tagged_gen, _) = tagged.unwrap();
        assert_eq!(tagged_gen, current);
    }

    #[test]
    fn control_frame_not_tagged_when_no_active_avatar() {
        let lifecycle = AvatarLifecycle::new();
        assert!(tag_control_frame(dummy_frame(), &lifecycle).is_none());
    }

    #[test]
    fn active_control_frame_rejects_stale_generation() {
        let mut lifecycle = AvatarLifecycle::new();
        let root = Entity::from_raw_u32(1).unwrap();
        lifecycle.request_load(root).unwrap();

        let mut active = ActiveControlFrame::default();
        let result =
            set_active_control_frame(&lifecycle, AvatarGeneration(0), dummy_frame(), &mut active);
        assert!(result.is_err());
        assert!(active.frame.is_none());
    }

    #[test]
    fn apply_rejects_stale_binding() {
        let mut lifecycle = AvatarLifecycle::new();
        let root = Entity::from_raw_u32(1).unwrap();
        lifecycle.request_load(root).unwrap();
        lifecycle.start_binding(root);
        lifecycle.finish_ready();

        let active = ActiveControlFrame {
            generation: lifecycle.current_generation(),
            frame: Some(dummy_frame()),
        };

        let result = apply_active_control_frame(&lifecycle, &active, None);
        assert!(matches!(result, Err(ControlFrameError::StaleBinding)));
    }

    #[test]
    fn apply_rejects_generation_mismatch() {
        let mut lifecycle = AvatarLifecycle::new();
        let root = Entity::from_raw_u32(1).unwrap();
        lifecycle.request_load(root).unwrap();

        let binding = AvatarBinding {
            root,
            head: root,
            generation: lifecycle.current_generation(),
            ..AvatarBinding::head_only(root, root, AvatarGeneration::default())
        };
        lifecycle.start_binding(root);
        lifecycle.finish_ready();

        let active = ActiveControlFrame {
            generation: AvatarGeneration(999),
            frame: Some(dummy_frame()),
        };

        let result = apply_active_control_frame(&lifecycle, &active, Some(&binding));
        assert!(matches!(
            result,
            Err(ControlFrameError::StaleGeneration {
                frame_generation: AvatarGeneration(999),
                ..
            })
        ));
    }
}
