// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]
//! `vtuber-avatar`: Bevy and `bevy_vrm1` adapter.
//!
//! This is the only crate that interacts with Bevy entities and `bevy_vrm1` APIs.
//! `bevy_vrm1` types must not leak into `vtuber-core` or `vtuber-tracking`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// Pure observed-arm adapter for the existing two-bone IK.
pub mod tracked_arm;

pub mod arm;
pub mod arm_motion_geometry;
pub mod arm_pipeline;
pub mod arm_pose;
pub mod bind;
pub mod binding;
pub mod body_motion;
pub mod body_scale;
pub mod capabilities;
pub mod compatibility;
pub mod expression;
pub mod expression_catalog;
mod framing;
pub mod gaze;
pub mod idle;
pub mod lifecycle;
pub mod load;
pub mod look;
pub mod mirror;
pub mod placeholder;
pub mod plugin;
pub mod pose;
pub mod render_output;
pub mod tracking_profile;
pub mod unload;

pub use arm::{
    ARM_POSE_PROFILE_OVERRIDE_VERSION, ArmChainBinding, ArmChainCapabilities, ArmChainReferences,
    ArmIkError, ArmIkInput, ArmIkSolution, ArmIkTarget, ArmPoseProfile, ArmPoseProfileOverride,
    ArmPoseProfileOverrideError, ArmRestGeometry, ArmSide, FingerJointReferences,
    FingerJointRestBinding, FingerJointRestReferences, FingerReferences, FingerRestReferences,
    RestSpaceBonePose, default_arm_target, solve_two_bone_arm,
};
pub use arm_motion_geometry::{
    ArmMotionGeometry, ArmMotionRestGeometry, ElbowSwivelReference, ForearmTwistAxisInfo,
    HipsAnchorFrame, build_arm_motion_rest_geometry,
};
pub use arm_pipeline::{
    ArmPipelineError, ArmPipelineInput, ArmPipelineOutcome, ArmPoseSourceKind, ArmPoseSourceUsed,
    ArmSourceSelection, DYNAMIC_ARM_PROFILE_OVERRIDE_VERSION, DynamicArmProfile,
    DynamicArmProfileOverride, DynamicArmProfileOverrideError, DynamicArmTargets,
    MAX_ARM_DROP_RADIANS, TrackedArmControl, chain_side_label, clamp_upper_arm_swing,
    resolve_arm_pose, resolve_side, update_dynamic_arm_targets, update_tracked_arm_targets,
};
pub use arm_pose::{
    ArmPoseBlendSide, ArmPoseBlendState, ArmPoseOverrideStore, ArmPoseOverrideStoreError,
    ArmPoseProfileChange, DEFAULT_ARM_RETURN_SECONDS, DEFAULT_ARM_TRANSITION_SECONDS,
    DefaultArmPose, ResolvedArmPose, ResolvedBoneDelta, ResolvedFingerJointPose,
    ResolvedFingerPose, apply_arm_pose_profile_changes, apply_default_arm_pose,
};
pub use bevy_vrm1::prelude::{
    LegacyShaderKind, Vrm0MetaDiagnostics, VrmCompatibilityWarning, VrmCompatibilityWarningCode,
    VrmRuntimeDescriptor, classify_legacy_shader, collect_legacy_compatibility_warnings,
};
pub use bind::BindTriggered;
pub use binding::{AvatarBindError, AvatarBinding, bind_humanoid_bones};
pub use body_motion::{
    BodyFollowFilter, BodyMotionProfiles, LossIdleState, PositionInputMetrics, position_channels,
    reset_position_metrics_on_lifecycle_change, update_body_tracking_position_input,
};
pub use capabilities::{
    AvatarCapabilities, BlinkMode, BonePresence, DeclaredLookAtType, EmotionSet,
    ExpressionCapabilities, GazeFallbackReason, LookDirectionSet, MouthMode,
    PerfectSyncCapabilities, SelectedGazeBackend, select_gaze_backend,
};
pub use expression::manual::{
    ManualExpressionRequest, ManualExpressionSelection, ManualExpressionSet,
    apply_manual_expression_requests, is_tracking_selection,
};
pub use expression_catalog::{
    AvatarExpressionCatalog, EMOTIONAL_PRESETS, ExpressionAvailability, ExpressionCatalogEntry,
    ExpressionCatalogInput, ExpressionKind, NEUTRAL_PRESET, build_catalog, classify_expression,
    is_excluded_expression, is_tracking_expression,
};
pub use framing::AvatarViewportCamera;
pub use framing::camera_control::geometry as camera_control_geometry;
pub use framing::camera_control::{
    AvatarCameraControl, AvatarCameraControlState, CameraControlConfig, CameraControlGeometryError,
    CameraControlPose, CameraDistanceLimits, CameraPointerInputGate, FIXED_VERTICAL_FOV,
};
pub use framing::camera_input::{CameraInputSet, CameraPointerGesture, normalized_vertical_scroll};
pub use framing::camera_reset::ResetCameraRequest;
pub use idle::{IDLE_PROCEDURAL_AMPLITUDE_METERS, IdleMotionProfile, IdleMotionProfileError};
pub use lifecycle::*;
pub use load::{
    AssetPathError, AvatarAssetId, ExpectedVrmGeneration, ImportedAvatar, LoadImportedAvatarError,
    LoadImportedAvatarRequest, LoadImportedAvatarResult, PendingAvatarLoad, UserAssetPath,
};
pub use look::{
    AvatarLookSettings, AvatarMaterialRoles, LookSettingsChanged, MaterialRole,
    MaterialRoleOverride, MaterialRoleOverridesChanged, PORTRAIT_FINISH, PortraitFinish,
    PortraitFinishPass, PortraitFinishState, RichLookSettings, StandardLookBase, StandardLookBases,
    apply_standard_portrait_settings, blend_look_scalar, capture_standard_look_base,
    effective_look_strength, face_lighting_normal, infer_material_role, register_portrait_finish,
    resolve_material_role, resolve_mtoon_role_params, resolve_mtoon_shading_mode,
    resolve_portrait_finish, resolve_standard_portrait, resolve_standard_role_params,
    studio_environment_cubemap, sync_portrait_finish,
};
pub use mirror::AvatarMotionMirror;
pub use plugin::{StartupModelPath, VtuberAvatarPlugin};
pub use pose::{PoseApplyMetrics, natural_body_tracking_profile, update_body_tracking_pose_input};
pub use render_output::{
    AVATAR_RENDER_LAYER, AvatarOutputCamera, AvatarOutputFrameSlot, AvatarOutputState,
    AvatarOutputTarget, AvatarViewportSnapshot, VIEWPORT_ONLY_RENDER_LAYER,
    register_output_systems,
};
pub use tracked_arm::{
    blend_arm_targets, resolved_tracked_arm_pose, solve_tracked_arm, tracked_arm_ik_target,
    tracking_to_rest_rotation,
};
pub use tracking_profile::{
    GlobalBodyTrackingProfile, TRACKING_PROFILE_SCHEMA_VERSION, TrackingProfileDocument,
};
pub use unload::{
    ActiveControlFrame, ControlFrameError, set_active_control_frame, tag_control_frame,
};
