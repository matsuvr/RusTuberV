//! Pure hand-pose policy for gestures that must not be emitted in isolation.

use vtuber_core::arm_tracking::{ArmControlFrame, ArmTrackingTarget, HandFingerPose};

/// Maximum summed absolute flexion for a finger to count as clearly extended.
///
/// This is deliberately strict: the policy must not rewrite an ordinary hand
/// opening or a partially moving finger.
const EXTENDED_MAX_TOTAL_FLEXION_RAD: f32 = 0.45;

/// Minimum summed absolute flexion for a finger to count as clearly folded.
///
/// All three non-middle fingers must exceed this value. If any one of them is
/// opening or otherwise moving out of a folded pose, the middle finger is left
/// untouched.
const FOLDED_MIN_TOTAL_FLEXION_RAD: f32 = 1.80;

/// Rewrites an isolated extended middle finger to the neighbouring folded pose.
///
/// The policy applies independently to both hands and changes only finger
/// articulation. Arm targets, palm orientation, blend weights, sequence data,
/// and timestamps are preserved.
#[must_use]
pub fn suppress_isolated_middle_extension(mut frame: ArmControlFrame) -> ArmControlFrame {
    frame.targets.left = frame.targets.left.map(sanitize_target);
    frame.targets.right = frame.targets.right.map(sanitize_target);
    frame
}

fn sanitize_target(mut target: ArmTrackingTarget) -> ArmTrackingTarget {
    target.fingers = target.fingers.map(suppress_isolated_middle_extension_pose);
    target
}

/// Rewrites one hand pose only when the middle finger is the sole extended finger.
///
/// Index, ring, and little must all be clearly folded. The thumb is deliberately
/// not used as a gate: its oblique MediaPipe solve is less stable, and a weak
/// thumb observation must not allow the isolated middle-finger pose through.
/// When the policy applies, each middle-finger joint takes the median of the
/// corresponding index, ring, and little joints. This keeps the hand curled
/// without introducing a fixed authored pose.
#[must_use]
pub fn suppress_isolated_middle_extension_pose(mut pose: HandFingerPose) -> HandFingerPose {
    let [index, middle, ring, little] = pose.fingers;
    let isolated_middle = total_flexion(middle) <= EXTENDED_MAX_TOTAL_FLEXION_RAD
        && [index, ring, little]
            .into_iter()
            .all(|finger| total_flexion(finger) >= FOLDED_MIN_TOTAL_FLEXION_RAD);
    if !isolated_middle {
        return pose;
    }

    let [index_mcp, index_pip, index_dip] = index;
    let [ring_mcp, ring_pip, ring_dip] = ring;
    let [little_mcp, little_pip, little_dip] = little;
    let folded_middle = [
        median3(index_mcp, ring_mcp, little_mcp),
        median3(index_pip, ring_pip, little_pip),
        median3(index_dip, ring_dip, little_dip),
    ];
    pose.fingers = [index, folded_middle, ring, little];
    pose
}

fn total_flexion([mcp, pip, dip]: [f32; 3]) -> f32 {
    mcp.abs() + pip.abs() + dip.abs()
}

fn median3(a: f32, b: f32, c: f32) -> f32 {
    a + b + c - a.min(b).min(c) - a.max(b).max(c)
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
    use vtuber_core::arm_tracking::{ArmBlendWeight, ArmBlendWeights, ArmTrackingTargets};
    use vtuber_core::{FrameSeq, MonoTimeNs};

    fn pose(fingers: [[f32; 3]; 4]) -> HandFingerPose {
        HandFingerPose {
            fingers,
            spread: [0.0; 4],
            thumb: [0.4, 0.4],
            thumb_direction: [1.0, 0.0, 0.0],
        }
    }

    fn target(fingers: [[f32; 3]; 4]) -> ArmTrackingTarget {
        ArmTrackingTarget {
            wrist: [0.1, 0.2, 0.3],
            elbow_pole: [0.2, 0.3, 0.4],
            palm_normal: Some([0.0, 1.0, 0.0]),
            fingers: Some(pose(fingers)),
        }
    }

    #[test]
    fn isolated_middle_extension_uses_the_neighbouring_folded_pose() {
        let input = pose([
            [0.8, 0.9, 0.4],
            [0.05, 0.05, 0.05],
            [1.0, 1.1, 0.5],
            [0.9, 1.0, 0.6],
        ]);
        let filtered = suppress_isolated_middle_extension_pose(input);
        assert_eq!(filtered.fingers[1], [0.9, 1.0, 0.5]);
        assert_eq!(filtered.spread, input.spread);
        assert_eq!(filtered.thumb, input.thumb);
        assert_eq!(filtered.thumb_direction, input.thumb_direction);
    }

    #[test]
    fn middle_extension_is_preserved_when_any_other_finger_is_moving() {
        let folded = [0.8, 0.9, 0.4];
        let moving = [0.5, 0.6, 0.4];
        let extended = [0.05, 0.05, 0.05];
        for fingers in [
            [extended, extended, folded, folded],
            [folded, extended, extended, folded],
            [folded, extended, folded, extended],
            [moving, extended, folded, folded],
            [folded, extended, moving, folded],
            [folded, extended, folded, moving],
        ] {
            let input = pose(fingers);
            assert_eq!(suppress_isolated_middle_extension_pose(input), input);
        }
    }

    #[test]
    fn signed_mirrored_flexion_is_suppressed_on_both_sides() {
        let fingers = [
            [-0.8, -0.9, -0.4],
            [-0.05, -0.05, -0.05],
            [-1.0, -1.1, -0.5],
            [-0.9, -1.0, -0.6],
        ];
        let input = ArmControlFrame {
            source_seq: FrameSeq(7),
            captured_at: MonoTimeNs(11),
            produced_at: MonoTimeNs(13),
            targets: ArmTrackingTargets {
                left: Some(target(fingers)),
                right: Some(target(fingers)),
            },
            weights: ArmBlendWeights {
                left: ArmBlendWeight::ONE,
                right: ArmBlendWeight::ONE,
            },
        };
        let filtered = suppress_isolated_middle_extension(input);
        for target in [filtered.targets.left, filtered.targets.right]
            .into_iter()
            .flatten()
        {
            assert_eq!(
                target.fingers.expect("finger pose").fingers[1],
                [-0.9, -1.0, -0.5]
            );
        }
        assert_eq!(filtered.source_seq, input.source_seq);
        assert_eq!(filtered.captured_at, input.captured_at);
        assert_eq!(filtered.produced_at, input.produced_at);
        assert_eq!(filtered.weights, input.weights);
    }
}
