//! Engine-neutral arm observations and IK targets. No camera, inference, or ECS handles.

use crate::{FrameSeq, MonoTimeNs};

/// A world landmark in a task's camera-aligned, unmirrored basis.
///
/// Coordinates are in meters: X is image-right, Y is down, and smaller Z is
/// nearer the camera. The origin depends on the task (hip centre for Pose, the
/// hand's geometric centre for the Hand Landmarker), so only differences within
/// one result are comparable across tasks. Missing quality scores remain
/// missing; they are not replaced with confidence 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoseWorldLandmark {
    /// World coordinates in meters. Never populate these from normalized landmarks.
    pub meters: [f32; 3],
    /// Reported visibility, when supplied by the task.
    pub visibility: Option<f32>,
    /// Reported presence, when supplied by the task.
    pub presence: Option<f32>,
}

/// MediaPipe Hand Landmarker's fixed landmark count per hand.
pub const HAND_LANDMARK_COUNT: usize = 21;

/// One hand's world landmarks from the Hand Landmarker.
///
/// The 21 points are the wrist, finger joints, and tips in the hand task's own
/// camera-aligned basis; only orientation is derived from them, because their
/// origin is hand-centred rather than body-centred. The full set is retained so
/// finger articulation can be consumed without a new inference model.
///
/// The presence of this value is the only hand-quality signal available: the
/// Hand Landmarker reports no per-landmark detection confidence, so a hand that
/// was paired to a wrist is treated as observed and a hand that is absent stays
/// absent. See [`Self::handedness_score`] for what is deliberately not a
/// quality signal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HandWorldLandmarks {
    /// 21 world landmarks of one hand, in meters.
    pub landmarks: [PoseWorldLandmark; HAND_LANDMARK_COUNT],
    /// MediaPipe's confidence in its Left/Right classification of this hand.
    ///
    /// This is the confidence of a *left/right label*, not of the landmark
    /// coordinates: the task derives the label from the hand's own shape, so an
    /// ambiguous pose lowers it while the landmarks stay exactly as usable. It
    /// therefore must never be read as detection quality, coordinate accuracy,
    /// or an ordering between competing hands, and no missing value is ever
    /// replaced with confidence 1.
    pub handedness_score: Option<f32>,
}

/// Three observations belonging to the same anatomical arm and source image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmLandmarks {
    /// Upper-arm origin; this is not the clavicle's origin in the avatar rig.
    pub shoulder: PoseWorldLandmark,
    /// Lower-arm origin.
    pub elbow: PoseWorldLandmark,
    /// Hand origin. No palm orientation is implied by this position alone.
    pub wrist: PoseWorldLandmark,
    /// Hand landmarker observation spanning the palm plane, when this hand was
    /// detected beside the pose wrist. A missing hand stays missing; it is never
    /// replaced with a neutral orientation.
    pub hand: Option<HandWorldLandmarks>,
}

/// Both anatomical arms from one pose. Left/right are not preview-mirror labels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoseArmObservation {
    /// Anatomical left/right hips from the same Pose result. Missing hips
    /// make thorax orientation unavailable, without invalidating the arms.
    pub hips: Option<[PoseWorldLandmark; 2]>,
    /// Subject's left arm.
    pub left: ArmLandmarks,
    /// Subject's right arm.
    pub right: ArmLandmarks,
}

/// A completed pose inference, including an explicit no-person result.
///
/// Keep this separate from the face result: a slow pose task must not hold up
/// face publication. The camera assigns the sequence and capture timestamp.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoseArmFrame {
    /// Sequence of the camera image, not an independent pose counter.
    pub source_seq: FrameSeq,
    /// Capture time, not the time the callback happened to run.
    pub captured_at: MonoTimeNs,
    /// Time inference completed, for measurement rather than filter integration.
    pub inference_finished_at: MonoTimeNs,
    /// None means a completed inference found no person, not that no new frame arrived.
    pub observation: Option<PoseArmObservation>,
}

/// Observed articulation in a hand-local palm frame, independent of arm pose.
///
/// Forward is the bisector of the unit wrist-to-index/little MCP rays. Normal
/// is their normalized index cross little; across is forward cross normal.
/// Local XYZ means (across, forward, normal). The normal is axial: reflection
/// reverses local Z and signed flexion, but preserves spread and local XY.
/// Only differences inside one hand are used, never the hand task's origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HandFingerPose {
    /// Index, middle, ring, little `[mcp, pip, dip]` radians. MCP is signed
    /// elevation from the palm plane; PIP/DIP are signed bends about the
    /// proximal segment cross palm normal. Positive bends toward the normal.
    pub fingers: [[f32; 3]; 4],
    /// Each proximal segment's in-plane angle from forward toward across.
    /// Spread is independent of MCP elevation and compared to the rig's rest.
    pub spread: [f32; 4],
    /// Thumb `[mcp, ip]` coordinates. Raw landmark features use the signed
    /// MCP-to-IP elevation and the signed IP bend. After pose selection these
    /// are authored flexion amounts applied relative to the rig's resting
    /// thumb about the anatomical MCP/IP axes, with the same handed sign as
    /// `fingers` (left negative, right positive). The Hand Landmarker CMC is
    /// not used.
    pub thumb: [f32; 2],
    /// The thumb's in-plane opening, from forward toward across. Used for pose
    /// recognition only; CMC uses the selected catalog coordinates separately.
    pub thumb_spread: f32,
    /// Authored CMC [flexion, abduction] in the right-hand model coordinates.
    /// Independent of raw landmark features; reflected by the avatar axes.
    pub thumb_cmc: [f32; 2],
}

/// Shoulder-relative target in units of the subject's calibrated total arm length.
///
/// The canonical front-view basis is +X image-right, +Y up, +Z toward the
/// camera. This is deliberately NOT the existing head-translation basis
/// (whose +Z points away). It has no avatar position or bone length embedded.
/// The avatar adapter converts it to the IK solver's rest space exactly once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmTrackingTarget {
    /// Desired wrist offset from the observed shoulder.
    pub wrist: [f32; 3],
    /// Observed elbow offset, used as a positional objective with its own
    /// confidence. Bone geometry and hard constraints can leave a residual.
    pub elbow_pole: [f32; 3],
    /// Unit normal of the observed palm plane in the canonical tracking basis,
    /// when the index/little-finger keypoints defined one.
    ///
    /// The normal points to the same anatomical hand side as the avatar rest
    /// geometry's index/pinky cross product, so no per-side sign is applied.
    /// `None` means "no palm observation", never a fabricated neutral twist.
    pub palm_normal: Option<[f32; 3]>,
    /// Palm long axis from wrist toward the index/little MCP bisector.
    /// A polar vector, unlike the axial palm normal.
    pub palm_forward: Option<[f32; 3]>,
    /// Observed finger articulation, when the hand's landmarks defined it.
    ///
    /// `None` means "no finger observation", never a fabricated rest pose; the
    /// avatar keeps its own rest fingers in that case.
    pub fingers: Option<HandFingerPose>,
}

/// Per-side targets after tracking's visibility and temporal policy.
///
/// None is an absent arm target, never a fabricated target at the origin.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ArmTrackingTargets {
    /// Target for the anatomical left arm.
    pub left: Option<ArmTrackingTarget>,
    /// Target for the anatomical right arm.
    pub right: Option<ArmTrackingTarget>,
}

/// How much an observed channel should replace the avatar's virtual arm.
///
/// `wrist`, `pole` and `palm` weight the observed wrist, elbow and palm
/// objectives in the same constrained solve. Missing channels return to the
/// admitted neutral pose; `fingers` blends the selected catalog articulation.
/// Fixed bone lengths and joint/collision limits take priority over all these
/// objectives. All weights are in `0.0..=1.0`; zero means no observation for
/// that channel, never an observation at the origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmBlendWeight {
    /// Observed arm contribution, driven by wrist presence.
    pub wrist: f32,
    /// Observed elbow-position contribution.
    pub pole: f32,
    /// Observed palm orientation contribution (normal and long axis).
    pub palm: f32,
    /// Observed finger-articulation contribution.
    pub fingers: f32,
}

impl ArmBlendWeight {
    /// Fully virtual: no channel is observed.
    pub const ZERO: Self = Self {
        wrist: 0.0,
        pole: 0.0,
        palm: 0.0,
        fingers: 0.0,
    };
    /// Fully observed: every channel is trusted.
    pub const ONE: Self = Self {
        wrist: 1.0,
        pole: 1.0,
        palm: 1.0,
        fingers: 1.0,
    };
}

impl Default for ArmBlendWeight {
    fn default() -> Self {
        Self::ZERO
    }
}

/// Per-side blend weights, anatomical left/right like [`ArmTrackingTargets`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ArmBlendWeights {
    /// Weight for the subject's left arm.
    pub left: ArmBlendWeight,
    /// Weight for the subject's right arm.
    pub right: ArmBlendWeight,
}

/// One observation's result, ready for the avatar's existing arm compositor.
///
/// The source sequence and capture time belong to the camera image the targets
/// came from. `produced_at` is when this frame was assembled. A side's weight
/// of zero pair with a target means "this is the last observed pose; render the
/// virtual arm instead", never a hand at the origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmControlFrame {
    /// Independently observed thorax and shoulder centres, with its own loss weight.
    pub thorax: Option<ThoraxTarget>,
    /// Sequence of the camera image the observation came from.
    pub source_seq: FrameSeq,
    /// Capture time of the camera image the observation came from.
    pub captured_at: MonoTimeNs,
    /// Time this control frame was produced.
    pub produced_at: MonoTimeNs,
    /// Latest per-side targets, held across short losses.
    pub targets: ArmTrackingTargets,
    /// Per-side per-channel confidence in the targets.
    pub weights: ArmBlendWeights,
}

/// Calibrated torso proxy from Pose hips and shoulders, not ISB bone markers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThoraxTarget {
    /// Neutral-relative rotation in the canonical tracking basis, quaternion XYZW.
    pub rotation: [f32; 4],
    /// Left/right displacement from the neutral shoulder coordinates, in the
    /// current thorax frame, divided by the fixed neutral shoulder width.
    /// Rigid torso rotation is excluded and is carried by `rotation` alone.
    pub shoulder_offsets: [[f32; 3]; 2],
    /// Confidence/loss weight; zero means neutral, not a fabricated observation.
    pub weight: f32,
}

impl ThoraxTarget {
    /// Reflects the rotation and exchanges the two anatomical shoulder centres.
    #[must_use]
    pub fn mirrored(self) -> Self {
        let [x, y, z, w] = self.rotation;
        let [left, right] = self.shoulder_offsets;
        let reflect = |[x, y, z]: [f32; 3]| [-x, y, z];
        Self {
            rotation: [x, -y, -z, w],
            shoulder_offsets: [reflect(right), reflect(left)],
            ..self
        }
    }
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

    #[test]
    fn mirror_reflects_and_exchanges_sides_including_absence() {
        let left = ArmTrackingTarget {
            wrist: [0.3, 0.2, 0.4],
            elbow_pole: [0.5, -0.1, 0.2],
            palm_normal: Some([0.1, 0.2, -0.9]),
            palm_forward: Some([0.2, 0.9, 0.1]),
            fingers: Some(HandFingerPose {
                fingers: [[0.1, 0.2, 0.3]; 4],
                spread: [0.2, 0.0, -0.1, -0.3],
                thumb: [0.4, 0.5],
                thumb_spread: 0.25,
                thumb_cmc: [0.0; 2],
            }),
        };
        let targets = ArmTrackingTargets {
            left: Some(left),
            right: None,
        };
        let mirrored = targets.mirrored();
        assert_eq!(mirrored.left, None);
        assert_eq!(mirrored.right.unwrap().wrist, [-0.3, 0.2, 0.4]);
        assert_eq!(mirrored.right.unwrap().elbow_pole, [-0.5, -0.1, 0.2]);
        assert_eq!(mirrored.right.unwrap().palm_forward, Some([-0.2, 0.9, 0.1]));
        assert_eq!(mirrored.right.unwrap().palm_normal, Some([0.1, -0.2, 0.9]));
        // Local normal components and signed bends reverse on reflection.
        let fingers = mirrored.right.unwrap().fingers.unwrap();
        assert_eq!(fingers.fingers, [[-0.1, -0.2, -0.3]; 4]);
        assert_eq!(fingers.spread, [0.2, 0.0, -0.1, -0.3]);
        assert_eq!(fingers.thumb, [-0.4, -0.5]);
        assert_eq!(fingers.thumb_spread, 0.25);
        assert_eq!(mirrored.mirrored(), targets);
    }

    #[test]
    fn blend_weights_exchange_sides_without_reflecting() {
        let weights = ArmBlendWeights {
            left: ArmBlendWeight {
                wrist: 0.25,
                pole: 0.5,
                palm: 0.75,
                fingers: 1.0,
            },
            right: ArmBlendWeight::ONE,
        };
        let mirrored = weights.mirrored();
        assert_eq!(mirrored.left, ArmBlendWeight::ONE);
        assert_eq!(mirrored.right.wrist, 0.25);
        assert_eq!(mirrored.right.pole, 0.5);
        assert_eq!(mirrored.right.palm, 0.75);
        assert_eq!(mirrored.right.fingers, 1.0);
        assert_eq!(mirrored.mirrored(), weights);
    }

    #[test]
    fn zero_weight_is_distinct_from_a_fully_observed_frame() {
        let target = ArmTrackingTarget {
            wrist: [0.0; 3],
            elbow_pole: [0.0; 3],
            palm_normal: None,
            palm_forward: None,
            fingers: None,
        };
        let frame = ArmControlFrame {
            thorax: None,
            source_seq: FrameSeq(1),
            captured_at: MonoTimeNs(2),
            produced_at: MonoTimeNs(3),
            targets: ArmTrackingTargets {
                left: Some(target),
                right: None,
            },
            weights: ArmBlendWeights::default(),
        };
        assert_eq!(frame.weights.left, ArmBlendWeight::ZERO);
        assert_ne!(frame.weights.left, ArmBlendWeight::ONE);
        assert!(!frame.weights.left.wrist.is_nan());
    }

    #[test]
    fn no_pose_is_distinct_from_a_pose_with_zero_visibility() {
        let point = PoseWorldLandmark {
            meters: [0.0; 3],
            visibility: Some(0.0),
            presence: None,
        };
        let arm = ArmLandmarks {
            shoulder: point,
            elbow: point,
            wrist: point,
            hand: None,
        };
        let detected = PoseArmFrame {
            source_seq: FrameSeq(7),
            captured_at: MonoTimeNs(10),
            inference_finished_at: MonoTimeNs(20),
            observation: Some(PoseArmObservation {
                hips: None,
                left: arm,
                right: arm,
            }),
        };
        let absent = PoseArmFrame {
            observation: None,
            ..detected
        };
        assert_ne!(detected, absent);
        assert_eq!(detected.observation.unwrap().left.wrist.presence, None);
    }
}
