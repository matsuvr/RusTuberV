//! Pure hand-pose policy for gestures that must not be emitted in isolation.

use vtuber_core::arm_tracking::{ArmControlFrame, ArmTrackingTarget, HandFingerPose};

/// Maximum summed absolute flexion for a finger to count as standing straight.
///
/// The same value decides both halves of the test: whether the middle finger is
/// standing up, and whether a neighbour is standing up beside it. One shared
/// threshold is what makes the policy stable, because a neighbour is then
/// either clearly straight or clearly not, and landmark noise cannot leave the
/// middle finger looking alone while the index is straight too.
///
/// This is not a question of how far the neighbours are folded. Asking that
/// needed a second, much larger cut-off, and a naturally held isolated
/// middle-finger pose never reached it on all three neighbours at once, so the
/// gesture passed through unfiltered.
///
/// A straight finger measures well under half a radian per joint, so 1.2 leaves
/// room for estimation error and for a middle finger held slightly bent, while
/// a curled finger passes 1.2 as soon as one knuckle is well past half of its
/// range.
const STANDING_MAX_TOTAL_FLEXION_RAD: f32 = 1.2;

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

/// Rewrites one hand pose only when the middle finger stands up on its own.
///
/// The middle finger must be straight and none of the index, ring, or little
/// may be. Any straight neighbour leaves the pose alone, which is what keeps an
/// open hand, a peace sign, and a three tracked exactly as they are seen. The
/// thumb is deliberately not used as a gate: its oblique MediaPipe solve is
/// less stable, and a weak thumb observation must not allow the isolated
/// middle-finger pose through. When the policy applies, each middle-finger joint
/// takes the median of the corresponding index, ring, and little joints. This
/// keeps the hand curled without introducing a fixed authored pose.
#[must_use]
pub fn suppress_isolated_middle_extension_pose(mut pose: HandFingerPose) -> HandFingerPose {
    let standing = |finger: [f32; 3]| total_flexion(finger) <= STANDING_MAX_TOTAL_FLEXION_RAD;
    let [index, middle, ring, little] = pose.fingers;
    if !standing(middle) || [index, ring, little].into_iter().any(standing) {
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
            thumb_spread: 0.3,
            thumb_cmc: [0.0; 2],
        }
    }

    fn target(fingers: [[f32; 3]; 4]) -> ArmTrackingTarget {
        ArmTrackingTarget {
            wrist: [0.1, 0.2, 0.3],
            elbow_pole: [0.2, 0.3, 0.4],
            palm_normal: Some([0.0, 1.0, 0.0]),
            palm_forward: None,
            fingers: Some(pose(fingers)),
        }
    }

    fn assert_approx_eq(actual: [f32; 3], expected: [f32; 3]) {
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-5, "{actual} != {expected}");
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
        assert_approx_eq(filtered.fingers[1], [0.9, 1.0, 0.5]);
        assert_eq!(filtered.spread, input.spread);
        assert_eq!(filtered.thumb, input.thumb);
        assert_eq!(filtered.thumb_spread, input.thumb_spread);
    }

    #[test]
    fn middle_extension_is_preserved_when_a_neighbour_is_also_standing() {
        let curled = [0.8, 0.9, 0.4];
        let standing = [0.05, 0.05, 0.05];
        let nearly_standing = [0.25, 0.15, 0.1];
        for fingers in [
            [standing, standing, curled, curled],
            [curled, standing, standing, curled],
            [curled, standing, curled, standing],
            [nearly_standing, standing, curled, curled],
            [curled, standing, nearly_standing, curled],
            [curled, standing, curled, nearly_standing],
        ] {
            let input = pose(fingers);
            assert_eq!(suppress_isolated_middle_extension_pose(input), input);
        }
    }

    #[test]
    fn a_bent_middle_finger_is_left_alone() {
        // Only a straight middle finger is the gesture. A middle that is bent
        // past the standing threshold, even while it is the straightest finger
        // on the hand, must still track as seen.
        let curled = [0.8, 0.9, 0.4];
        let bent_middle = [0.5, 0.45, 0.4];
        let input = pose([curled, bent_middle, curled, curled]);
        assert_eq!(suppress_isolated_middle_extension_pose(input), input);
    }

    #[test]
    fn a_closed_hand_is_left_alone() {
        let curled = [1.4, 1.5, 0.8];
        let input = pose([curled; 4]);
        assert_eq!(suppress_isolated_middle_extension_pose(input), input);
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
            thorax: None,
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
            assert_approx_eq(
                target.fingers.expect("finger pose").fingers[1],
                [-0.9, -1.0, -0.5],
            );
        }
        assert_eq!(filtered.source_seq, input.source_seq);
        assert_eq!(filtered.captured_at, input.captured_at);
        assert_eq!(filtered.produced_at, input.produced_at);
        assert_eq!(filtered.weights, input.weights);
    }
}
