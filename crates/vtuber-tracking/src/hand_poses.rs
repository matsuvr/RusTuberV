//! Nearest authored hand pose, confirmed on camera frames and blended on render ticks.
//! The catalog uses positive anatomical flexion magnitudes. In the palm frame
//! (index cross little), inward flexion is negative on the left, positive on
//! the right. Opening toward the thumb has the same sign on both hands.

use vtuber_core::arm_tracking::HandFingerPose;

use crate::filter::exponential::time_constant_alpha;

mod catalog;

/// Four consecutive *new observations*, never four draws of a retained frame.
const CONFIRM_FRAMES: u8 = 4;
/// First-order interpolation cannot overshoot the bounded authored poses.
const TRANSITION_TIME_CONSTANT_SEC: f32 = 0.06;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HandPoseSelector {
    selected: Option<usize>,
    candidate: Option<usize>,
    frames: u8,
}

impl HandPoseSelector {
    pub(crate) const fn new() -> Self {
        Self {
            selected: None,
            candidate: None,
            frames: 0,
        }
    }

    /// A missing/quarantined observation breaks the consecutive-frame run.
    /// Retain the chosen pose for the existing channel loss/recovery blend.
    pub(crate) fn interrupt(&mut self) {
        self.candidate = None;
        self.frames = 0;
    }

    pub(crate) fn observe(
        &mut self,
        observation: Option<HandFingerPose>,
        flexion_sign: f32,
    ) -> Option<HandFingerPose> {
        let Some(observed) = observation else {
            self.interrupt();
            return None;
        };
        let canonical = with_flexion_sign(observed, flexion_sign);
        let closest = catalog::POSES
            .iter()
            .enumerate()
            .min_by(|(_, (name_a, a)), (_, (name_b, b))| {
                score(canonical, name_a, *a).total_cmp(&score(canonical, name_b, *b))
            })
            .map(|(index, _)| index);
        if closest == self.selected {
            self.interrupt();
        } else {
            if closest != self.candidate {
                self.candidate = closest;
                self.frames = 0;
            }
            self.frames += 1;
            if self.frames >= CONFIRM_FRAMES {
                self.selected = closest;
                self.interrupt();
            }
        }
        let (_, pose) = catalog::POSES.get(self.selected?)?;
        Some(with_flexion_sign(*pose, flexion_sign))
    }
}

fn with_flexion_sign(mut pose: HandFingerPose, sign: f32) -> HandFingerPose {
    pose.fingers = pose.fingers.map(|joints| joints.map(|angle| angle * sign));
    pose.thumb = pose.thumb.map(|angle| angle * sign);
    pose
}

#[cfg(test)]
fn coordinates(pose: HandFingerPose) -> impl Iterator<Item = f32> {
    pose.fingers
        .into_iter()
        .flatten()
        .chain(pose.spread)
        .chain(pose.thumb)
        .chain([pose.thumb_spread])
}

/// Ranking preferences, not anatomical limits. A distinctly different shape
/// can still win; priority must not turn every detected hand into rock/paper/V.
fn priority_penalty(name: &str) -> f32 {
    match name {
        "fist" | "peace" | "open" => 0.0,
        "thumbs_up" | "point" | "horns" | "shaka" | "finger_gun" | "finger_heart" | "ok"
        | "crossed_fingers" | "vulcan" | "half_heart" | "i_love_you" => 0.15,
        _ => 0.5,
    }
}

/// Total positive flexion distinguishes an extended finger from a folded one.
/// Once folded, how the bend divides between PIP and DIP must not hide a V sign.
fn curl(joints: impl IntoIterator<Item = f32>, folded: f32) -> f32 {
    (joints.into_iter().map(|v| v.max(0.0)).sum::<f32>() / folded).clamp(0.0, 1.0)
}

fn score(a: HandFingerPose, name: &str, b: HandFingerPose) -> f32 {
    let curls_a = a.fingers.map(|joints| curl(joints, 2.4));
    let curls_b = b.fingers.map(|joints| curl(joints, 2.4));
    let mut result = priority_penalty(name);
    for ((a, b), (curl_a, curl_b)) in a
        .fingers
        .into_iter()
        .zip(b.fingers)
        .zip(curls_a.into_iter().zip(curls_b))
    {
        let [knuckle_a, _, _] = a;
        let [knuckle_b, _, _] = b;
        // The MCP keeps a hooked/claw hand distinct from a fist.
        result += (curl_a - curl_b).powi(2) + (knuckle_a - knuckle_b).powi(2);
    }
    // Compare gaps only between extended neighbours. The noisy opening of
    // folded ring/little fingers must not compete with the two visible V rays.
    let extended = curls_a
        .into_iter()
        .zip(curls_b)
        .map(|(a, b)| a < 0.5 && b < 0.5);
    for ((a, b), pair) in a
        .spread
        .windows(2)
        .zip(b.spread.windows(2))
        .zip(extended.clone().zip(extended.skip(1)))
    {
        if let ([a0, a1], [b0, b1], (true, true)) = (a, b, pair) {
            result += 2.0 * ((a0 - a1) - (b0 - b1)).powi(2);
        }
    }
    // A thumb is often occluded behind a V/fist; rank the four visible fingers
    // first, retaining enough thumb evidence to distinguish a thumbs-up/shaka.
    result
        + 0.25
            * ((curl(a.thumb, 2.1) - curl(b.thumb, 2.1)).powi(2)
                + (a.thumb_spread - b.thumb_spread).powi(2))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FingerSmootherState(HandFingerPose);

impl FingerSmootherState {
    pub(crate) fn new(pose: HandFingerPose) -> Self {
        Self(pose)
    }

    pub(crate) fn step(&mut self, pose: HandFingerPose, dt_sec: f32) {
        let alpha = time_constant_alpha(TRANSITION_TIME_CONSTANT_SEC, dt_sec);
        let blend = |current: &mut f32, target| *current += (target - *current) * alpha;
        for (current, target) in self
            .0
            .fingers
            .iter_mut()
            .flatten()
            .zip(pose.fingers.into_iter().flatten())
        {
            blend(current, target);
        }
        for (current, target) in self.0.spread.iter_mut().zip(pose.spread) {
            blend(current, target);
        }
        for (current, target) in self.0.thumb.iter_mut().zip(pose.thumb) {
            blend(current, target);
        }
        blend(&mut self.0.thumb_spread, pose.thumb_spread);
    }

    pub(crate) fn output(self) -> HandFingerPose {
        self.0
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )]

    use super::*;

    fn pose(name: &str) -> HandFingerPose {
        catalog::POSES
            .iter()
            .find(|(label, _)| *label == name)
            .unwrap()
            .1
    }

    #[test]
    fn catalog_is_distinct_bounded_and_selection_reflects_between_hands() {
        assert_eq!(catalog::POSES.len(), 32);
        for &(name, target) in catalog::POSES {
            for finger in target.fingers {
                for (angle, limit) in finger.into_iter().zip([1.75, 1.92, 1.05]) {
                    assert!((0.0..=limit).contains(&angle), "{name}: {angle}");
                }
            }
            assert!((0.0..=0.88).contains(&target.thumb[0]), "{name}");
            assert!((0.0..=1.23).contains(&target.thumb[1]), "{name}");
            assert!(
                target.spread.into_iter().all(|angle| angle.abs() <= 0.35),
                "{name}"
            );
            assert!((0.0..=1.31).contains(&target.thumb_spread), "{name}");
            let mut canonical_result = None;
            for sign in [-1.0, 1.0] {
                let input = with_flexion_sign(target, sign);
                let mut selector = HandPoseSelector::new();
                for _ in 1..CONFIRM_FRAMES {
                    assert_eq!(selector.observe(Some(input), sign), None);
                }
                let result = with_flexion_sign(selector.observe(Some(input), sign).unwrap(), sign);
                if let Some(previous) = canonical_result {
                    assert_eq!(result, previous, "{name}");
                }
                canonical_result = Some(result);
            }
            assert_eq!(
                catalog::POSES
                    .iter()
                    .filter(|(_, pose)| *pose == target)
                    .count(),
                1,
                "{name}"
            );
        }
    }

    fn selected_name(observed: HandFingerPose) -> &'static str {
        let mut selector = HandPoseSelector::new();
        for _ in 0..CONFIRM_FRAMES {
            selector.observe(Some(observed), 1.0);
        }
        catalog::POSES[selector.selected.unwrap()].0
    }

    #[test]
    fn peace_survives_thumb_occlusion_and_different_folded_joint_angles() {
        for folded in [[0.9, 1.0, 0.7], [1.3, 1.5, 0.9], [0.6, 1.4, 0.4]] {
            for thumb in [[0.0, 0.0], [0.3, 0.1], [0.7, 1.0]] {
                let observed = HandFingerPose {
                    fingers: [[0.05, 0.1, 0.0], [0.1, 0.05, 0.0], folded, folded],
                    spread: [0.15, -0.12, -0.6, 0.8],
                    thumb,
                    thumb_spread: 0.4,
                };
                assert_eq!(selected_name(observed), "peace", "{observed:?}");
            }
        }
    }

    #[test]
    fn priority_prefers_common_signs_without_hiding_distinct_emoji_or_other_shapes() {
        for name in [
            "fist",
            "peace",
            "open",
            "thumbs_up",
            "point",
            "ok",
            "crossed_fingers",
            "vulcan",
            "shaka",
            "claw",
            "ring_only",
            "little_only",
        ] {
            assert_eq!(selected_name(pose(name)), name);
        }
        let mut relaxed_fist = pose("fist");
        relaxed_fist.fingers = [[1.2, 1.3, 0.5]; 4];
        assert_eq!(selected_name(relaxed_fist), "fist");
    }

    #[test]
    fn noise_and_missing_frames_do_not_change_the_selected_pose() {
        let mut selector = HandPoseSelector::new();
        let open = pose("open");
        let fist = pose("fist");
        for _ in 0..CONFIRM_FRAMES {
            selector.observe(Some(open), 1.0);
        }
        for _ in 0..20 {
            assert_eq!(selector.observe(Some(fist), 1.0), Some(open));
            assert_eq!(selector.observe(Some(open), 1.0), Some(open));
        }
        for _ in 1..CONFIRM_FRAMES {
            assert_eq!(selector.observe(Some(fist), 1.0), Some(open));
        }
        assert_eq!(selector.observe(None, 1.0), None);
        for _ in 1..CONFIRM_FRAMES {
            assert_eq!(selector.observe(Some(fist), 1.0), Some(open));
        }
        assert_eq!(selector.observe(Some(fist), 1.0), Some(fist));
        let mut noisy = fist;
        noisy.fingers[0][0] += 0.03;
        noisy.spread[2] -= 0.02;
        for _ in 0..10 {
            assert_eq!(selector.observe(Some(noisy), 1.0), Some(fist));
        }
    }

    #[test]
    fn backward_observations_never_reach_the_display_and_transitions_stay_bounded() {
        let mut selector = HandPoseSelector::new();
        let mut backward = pose("open");
        backward.fingers = [[-1.5; 3]; 4];
        backward.thumb = [-1.0; 2];
        for _ in 0..CONFIRM_FRAMES {
            selector.observe(Some(backward), 1.0);
        }
        let result = selector.observe(Some(backward), 1.0).unwrap();
        assert!(
            result
                .fingers
                .into_iter()
                .flatten()
                .chain(result.thumb)
                .all(|a| a >= 0.0)
        );
        let mut smoother = FingerSmootherState::new(pose("open"));
        // Reverse midway through a transition, when a spring could overshoot.
        for name in ["fist", "open", "crossed_fingers", "vulcan", "fist"] {
            let target = pose(name);
            for _ in 0..8 {
                let before = smoother.output();
                smoother.step(target, 1.0 / 60.0);
                for ((a, b), c) in coordinates(before)
                    .zip(coordinates(target))
                    .zip(coordinates(smoother.output()))
                {
                    assert!(c >= a.min(b) - 1e-6 && c <= a.max(b) + 1e-6);
                }
            }
        }
    }
}
