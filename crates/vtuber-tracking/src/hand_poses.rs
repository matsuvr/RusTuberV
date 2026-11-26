//! Nearest authored hand pose, confirmed on camera frames and blended on render ticks.
//! The catalog uses positive anatomical flexion magnitudes. In the palm frame
//! (index cross little), inward flexion is negative on the left, positive on
//! the right. Opening toward the thumb has the same sign on both hands.

use vtuber_core::arm_tracking::HandFingerPose;

use crate::filter::exponential::time_constant_alpha;

mod catalog;

/// Confirmation counts new camera observations, never draws of a retained frame.
const CONFIRM_FRAMES: u8 = 4;
const CONFIDENT_FRAMES: u8 = 2;
const CONFIDENT_SCORE_GAP: f32 = 0.12;
const CATALOG_SPREAD_LIMIT: f32 = 0.35;
/// First-order interpolation cannot overshoot the bounded authored poses.
const TRANSITION_TIME_CONSTANT_SEC: f32 = 0.06;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HandPoseSelector {
    selected: Option<usize>,
    candidate: Option<usize>,
    frames: u8,
    confident_frames: u8,
}

impl HandPoseSelector {
    pub(crate) const fn new() -> Self {
        Self {
            selected: None,
            candidate: None,
            frames: 0,
            confident_frames: 0,
        }
    }

    /// A missing/quarantined observation breaks the consecutive-frame run.
    /// Retain the chosen pose for the existing channel loss/recovery blend.
    pub(crate) fn interrupt(&mut self) {
        self.candidate = None;
        self.frames = 0;
        self.confident_frames = 0;
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
        let [(best, best_score), (_, second_score)] =
            rank_poses(with_flexion_sign(observed, flexion_sign));
        let closest = Some(best);
        if closest == self.selected {
            self.interrupt();
        } else {
            if closest != self.candidate {
                self.interrupt();
                self.candidate = closest;
            }
            self.frames += 1;
            if second_score - best_score >= CONFIDENT_SCORE_GAP {
                self.confident_frames += 1;
            } else {
                self.confident_frames = 0;
            }
            if self.confident_frames >= CONFIDENT_FRAMES || self.frames >= CONFIRM_FRAMES {
                self.selected = closest;
                self.interrupt();
            }
        }
        let (_, pose) = catalog::POSES.get(self.selected?)?;
        Some(with_flexion_sign(*pose, flexion_sign))
    }
}

fn rank_poses(mut observed: HandFingerPose) -> [(usize, f32); 2] {
    // Palm-plane estimates can spread straight fingers beyond any catalog
    // pose. Keep those estimates from outweighing the measured flexion.
    observed.spread = observed
        .spread
        .map(|angle| angle.clamp(-CATALOG_SPREAD_LIMIT, CATALOG_SPREAD_LIMIT));
    let mut best = (0, f32::INFINITY);
    let mut second = best;
    for (i, &(name, pose)) in catalog::POSES.iter().enumerate() {
        let candidate = (i, score(observed, name, pose));
        if candidate.1 < best.1 {
            second = best;
            best = candidate;
        } else if candidate.1 < second.1 {
            second = candidate;
        }
    }
    [best, second]
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
        .chain(pose.thumb_cmc)
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
/// Once folded, how the bend divides between joints must not hide a sign.
fn curl(joints: impl IntoIterator<Item = f32>, folded: f32) -> f32 {
    (joints.into_iter().map(|v| v.max(0.0)).sum::<f32>() / folded).clamp(0.0, 1.0)
}

fn score(a: HandFingerPose, name: &str, b: HandFingerPose) -> f32 {
    // Keep partially folded fingers distinct from a closed fist. Saturating at
    // 2.4 radians made half-heart fingers look as closed as a thumbs-up, leaving
    // the noisier MCP estimate to decide between them.
    let curls_a = a.fingers.map(|joints| curl(joints, 3.0));
    let curls_b = b.fingers.map(|joints| curl(joints, 3.0));
    let mut distance = 0.0f32;
    for ((a, b), (curl_a, curl_b)) in a
        .fingers
        .into_iter()
        .zip(b.fingers)
        .zip(curls_a.into_iter().zip(curls_b))
    {
        let [knuckle_a, _, _] = a;
        let [knuckle_b, _, _] = b;
        // The MCP keeps a hooked/claw hand distinct from a fist, but is a
        // secondary cue. The palm-plane estimate changes how the same total
        // bend divides between MCP and PIP, including between mirrored inputs.
        distance += (curl_a - curl_b).powi(2) + 0.1 * (knuckle_a - knuckle_b).powi(2);
    }
    // Every candidate with extended neighbours must explain their observed
    // opening, even when a noisy bend estimate calls one finger folded.
    // Otherwise a V can become crossed fingers without comparing the gap
    // that distinguishes them. Candidate-folded ring/little gaps stay unused.
    // Keep the 1.2-radian spread gate independent of the curl score's scale.
    let extended = b.fingers.into_iter().map(|joints| curl(joints, 2.4) < 0.5);
    for ((a, b), pair) in a
        .spread
        .windows(2)
        .zip(b.spread.windows(2))
        .zip(extended.clone().zip(extended.skip(1)))
    {
        if let ([a0, a1], [b0, b1], (true, true)) = (a, b, pair) {
            distance += 2.0 * ((a0 - a1) - (b0 - b1)).powi(2);
        }
    }
    // With four folded fingers, thumb opening distinguishes a thumbs-up from
    // a fist even when the thumb bend is underestimated. Its influence falls
    // as the fingers extend.
    let least_curled = curls_a.into_iter().fold(1.0, f32::min);
    let thumb_spread_weight = 0.25 + 0.75 * least_curled.powi(2);
    distance += 0.5 * (curl(a.thumb, 2.1) - curl(b.thumb, 2.1)).powi(2)
        + thumb_spread_weight * (a.thumb_spread - b.thumb_spread).powi(2);
    // Keep the existing preference for noisy matches. Its influence goes to
    // zero for a close match, so every authored pose can win on its own shape.
    distance + priority_penalty(name).min(2.0 * distance)
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
        for (current, target) in self.0.thumb_cmc.iter_mut().zip(pose.thumb_cmc) {
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

    #[test]
    #[ignore = "generated Quint ITF traces"]
    fn quint_hand_pose() {
        use crate::quint_replay::{number, traces};
        let fist = pose("fist");
        let ambiguous = HandFingerPose {
            thumb: [0.32, 0.38],
            ..fist
        };
        traces("hand_pose", |states| {
            for sign in [-1.0, 1.0] {
                let mut selector = HandPoseSelector::new();
                for row in states {
                    let input = match row["event"].as_str().unwrap() {
                        "open" => Some(pose("open")),
                        "fist" => Some(fist),
                        "ambiguous" => Some(ambiguous),
                        "missing" => None,
                        other => panic!("{other}"),
                    };
                    let output = selector.observe(input.map(|p| with_flexion_sign(p, sign)), sign);
                    let name = |i: Option<usize>| i.map_or("", |idx| catalog::POSES[idx].0);
                    assert_eq!(name(selector.selected), row["selected"].as_str().unwrap());
                    assert_eq!(name(selector.candidate), row["candidate"].as_str().unwrap());
                    assert_eq!(i64::from(selector.frames), number(&row["frames"]));
                    assert_eq!(i64::from(selector.confident_frames), number(&row["strong"]));
                    let chosen = row["selected"].as_str().unwrap();
                    let expected = (input.is_some() && !chosen.is_empty())
                        .then(|| with_flexion_sign(pose(chosen), sign));
                    assert_eq!(output, expected);
                }
            }
        });
    }

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
            assert!(
                (0.0..=std::f32::consts::FRAC_PI_4).contains(&target.thumb[0]),
                "{name}"
            );
            assert!((0.0..=1.309).contains(&target.thumb[1]), "{name}");
            assert!(
                target
                    .spread
                    .into_iter()
                    .all(|angle| angle.abs() <= CATALOG_SPREAD_LIMIT),
                "{name}"
            );
            assert!((0.0..=1.31).contains(&target.thumb_spread), "{name}");
            assert!((-0.78..=0.7).contains(&target.thumb_cmc[0]), "{name}");
            assert!((-0.5..=0.78).contains(&target.thumb_cmc[1]), "{name}");
            let [(first, first_score), (second, second_score)] = rank_poses(target);
            assert_eq!(catalog::POSES[first].0, name);
            assert_eq!(first_score, 0.0, "{name}");
            assert!(second_score > first_score, "{name}");
            println!(
                "{name}: {} {first_score:.4}, {} {second_score:.4}, gap {:.4}",
                catalog::POSES[first].0,
                catalog::POSES[second].0,
                second_score - first_score,
            );
            for sign in [-1.0, 1.0] {
                let input = with_flexion_sign(target, sign);
                let mut selector = HandPoseSelector::new();
                let mut selected = None;
                for _ in 0..CONFIRM_FRAMES {
                    selected = selector.observe(Some(input), sign);
                }
                assert_eq!(selected, Some(input), "{name}, hand sign {sign}");
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
                    thumb_cmc: [0.0; 2],
                };
                assert_eq!(selected_name(observed), "peace", "{observed:?}");
            }
        }
    }

    #[test]
    fn a_v_gap_survives_the_observed_curl_threshold_on_both_hands() {
        // Left-hand features from camera frame 1024. The middle finger's
        // estimated MCP + PIP is just above the old "extended" threshold,
        // while its opening still clearly separates it from the index.
        let observed = HandFingerPose {
            fingers: [
                [0.408599, 0.138518, 0.632538],
                [0.549093, 0.652508, -0.280719],
                [1.474802, 1.382137, 0.794503],
                [1.284847, 1.411403, 0.781327],
            ],
            spread: [0.217769, -0.108644, 0.239300, 0.567489],
            thumb: [0.364734, -0.553566],
            thumb_spread: -0.538910,
            thumb_cmc: [0.0; 2],
        };
        for pip in [0.63, 0.649, 0.651, observed.fingers[1][1], 0.67] {
            let mut observed = observed;
            observed.fingers[1][1] = pip;
            for sign in [-1.0, 1.0] {
                let mut selector = HandPoseSelector::new();
                let input = with_flexion_sign(observed, sign);
                for _ in 0..CONFIRM_FRAMES {
                    selector.observe(Some(input), sign);
                }
                assert_eq!(
                    catalog::POSES[selector.selected.unwrap()].0,
                    "peace",
                    "PIP {pip}, hand sign {sign}: an open V must not become crossed fingers"
                );
            }
        }
    }

    #[test]
    fn native_left_and_right_observations_recognize_the_same_sign() {
        // Features from MediaPipe 0.10.35, the bundled hand_landmarker.task,
        // and Google's hand test images and reflections.
        // https://storage.googleapis.com/mediapipe-assets/thumb_up.jpg
        // https://storage.googleapis.com/mediapipe-assets/victory.jpg
        // https://storage.googleapis.com/mediapipe-assets/fist.jpg
        // https://storage.googleapis.com/mediapipe-assets/left_hands.jpg
        // These are independent model predictions, not a sign-flipped copy of
        // one prediction: the estimated MCP and distal bends differ by hand.
        for (name, fingers, spread, thumb, thumb_spread) in [
            (
                "open",
                [
                    [0.341302, 0.472445, 0.204666],
                    [0.349875, 0.288400, 0.376413],
                    [0.488831, 0.106997, 0.370489],
                    [0.351115, 0.083496, 0.416705],
                ],
                [0.242254, -0.391097, -0.646998, -1.246786],
                [0.320708, -0.251223],
                0.707597,
            ),
            (
                "fist",
                [
                    [1.04448, 1.61488, 1.05],
                    [1.143254, 1.57658, 1.05],
                    [1.203998, 1.755347, 1.05],
                    [0.853412, 1.085405, 1.05],
                ],
                [0.041159, 0.425530, 0.443225, 0.645296],
                [0.078473, -0.178378],
                -0.107488,
            ),
            (
                "fist",
                [
                    [1.073815, 1.513764, 1.05],
                    [1.127352, 1.527888, 1.05],
                    [1.157118, 1.513687, 1.05],
                    [0.840315, 1.495065, 1.05],
                ],
                [-0.020362, -0.094857, 0.354554, 0.034334],
                [0.183021, -0.165203],
                0.230982,
            ),
            (
                "thumbs_up",
                [
                    [0.681731, 1.51051, 1.05],
                    [0.783752, 1.623488, 1.05],
                    [0.911441, 1.559305, 1.05],
                    [0.549006, 1.224707, 1.05],
                ],
                [-0.131394, 0.083423, 0.105371, 0.108227],
                [-0.087472, -0.179561],
                0.674066,
            ),
            (
                "thumbs_up",
                [
                    [0.635825, 1.6962, 0.778160],
                    [0.853576, 1.62273, 1.05],
                    [0.969630, 1.410504, 1.05],
                    [0.650021, 1.339349, 1.05],
                ],
                [-0.195102, -0.253954, -0.272879, -0.299930],
                [-0.116946, -0.245747],
                0.846118,
            ),
            (
                "peace",
                [
                    [0.113893, -0.020865, 0.603163],
                    [-0.032353, 0.357552, 0.014910],
                    [0.805441, 1.534018, 0.421264],
                    [0.723121, 1.370764, 0.747690],
                ],
                [0.333606, 0.102163, -0.122506, -0.077124],
                [0.319804, -0.619174],
                -0.498211,
            ),
            (
                "peace",
                [
                    [-0.021583, 0.019723, 0.662764],
                    [-0.055820, 0.334034, 0.029652],
                    [0.933395, 1.362046, 0.634308],
                    [0.724971, 1.1883, 1.015948],
                ],
                [0.277230, 0.015615, 0.009066, 0.079103],
                [0.340934, -0.548893],
                -0.381891,
            ),
            // First two left-hand predictions with victory.jpg resized to
            // 180 px high, rotated 30 degrees, and placed in a 640x480 frame.
            // The right-hand clip already recognized peace on these frames;
            // the old MCP-heavy score called the left hand elegant, then claw.
            (
                "peace",
                [
                    [-0.598043, 0.044170, 0.387805],
                    [0.368448, 0.507000, 0.204420],
                    [0.715721, 1.253063, 0.662475],
                    [0.207965, 1.571466, 1.05],
                ],
                [0.294378, 0.071158, -0.264067, -0.557776],
                [0.453891, -0.354656],
                -0.291026,
            ),
            (
                "peace",
                [
                    [-0.548864, 0.200486, 0.382508],
                    [-0.588154, 0.101136, 0.128873],
                    [0.333663, 1.302241, 0.621724],
                    [0.195939, 1.291574, 1.05],
                ],
                [0.320954, 0.162180, 0.004788, -0.096815],
                [0.131309, -0.340132],
                -0.350138,
            ),
        ] {
            let canonical = HandFingerPose {
                fingers,
                spread,
                thumb,
                thumb_spread,
                thumb_cmc: [0.0; 2],
            };
            for finger in 0..4 {
                for delta in [-0.1, 0.0, 0.1] {
                    // The same total bend can move between MCP and PIP as
                    // MediaPipe's palm-plane estimate varies.
                    let mut observed = canonical;
                    observed.fingers[finger][0] += delta;
                    observed.fingers[finger][1] -= delta;
                    for sign in [-1.0, 1.0] {
                        let mut selector = HandPoseSelector::new();
                        let input = with_flexion_sign(observed, sign);
                        let mut selected = None;
                        for _ in 0..CONFIRM_FRAMES {
                            selected = selector.observe(Some(input), sign);
                        }
                        assert_eq!(
                            selected,
                            Some(with_flexion_sign(pose(name), sign)),
                            "{name}, hand sign {sign}, finger {finger}, delta {delta}, features {canonical:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_loosely_closed_fist_still_matches_fist() {
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
        assert_eq!(selector.observe(Some(fist), 1.0), Some(open));
        assert_eq!(selector.observe(None, 1.0), None);
        assert_eq!(selector.observe(Some(fist), 1.0), Some(open));
        assert_eq!(selector.observe(Some(fist), 1.0), Some(fist));
        let mut noisy = fist;
        noisy.fingers[0][0] += 0.03;
        noisy.spread[2] -= 0.02;
        for _ in 0..10 {
            assert_eq!(selector.observe(Some(noisy), 1.0), Some(fist));
        }
    }

    #[test]
    fn an_ambiguous_match_needs_four_observations() {
        let open = pose("open");
        let fist = pose("fist");
        let ambiguous = HandFingerPose {
            thumb: [0.32, 0.38],
            ..fist
        };
        let [(best, score), (_, runner_up)] = rank_poses(ambiguous);
        assert_eq!(catalog::POSES[best].0, "fist");
        assert!(runner_up - score < CONFIDENT_SCORE_GAP);

        let mut selector = HandPoseSelector::new();
        for _ in 0..CONFIRM_FRAMES {
            selector.observe(Some(open), 1.0);
        }
        // A single confident observation must not use the fast path after
        // an ambiguous one or count toward the next confident run.
        for observation in [ambiguous, fist, ambiguous] {
            assert_eq!(selector.observe(Some(observation), 1.0), Some(open));
        }
        assert_eq!(selector.observe(Some(ambiguous), 1.0), Some(fist));
    }

    #[test]
    fn four_gestures_replay_at_camera_rates_with_one_frame_outliers() {
        let gestures = [pose("open"), pose("fist"), pose("peace"), pose("thumbs_up")];
        for hz in [15, 30] {
            let samples_per_gesture = hz * 2 / 5; // 400 ms of each gesture.
            let mut selector = HandPoseSelector::new();
            let mut previous = None;
            for _ in 0..CONFIRM_FRAMES {
                previous = selector.observe(Some(gestures[3]), 1.0);
            }
            let mut correct = 0;
            let mut false_switches = 0;
            let mut delays = Vec::new();
            for (index, target) in gestures.into_iter().enumerate() {
                let mut first_match = None;
                for sample in 0..samples_per_gesture {
                    let observed = if sample == samples_per_gesture - 2 {
                        gestures[(index + 1) % gestures.len()]
                    } else {
                        let mut observed = target;
                        observed.fingers[0][0] += 0.03;
                        observed.spread[1] -= 0.02;
                        observed
                    };
                    let selected = selector.observe(Some(observed), 1.0);
                    if selected == Some(target) {
                        first_match.get_or_insert(sample);
                        correct += 1;
                    } else if selected != previous {
                        false_switches += 1;
                    }
                    previous = selected;
                }
                delays.push(first_match.unwrap() * 1000 / hz);
            }
            let total = 4 * samples_per_gesture;
            println!(
                "{hz} Hz: {correct}/{total} correct, delay {delays:?} ms, {false_switches} false switches"
            );
            assert_eq!(correct, total - 4);
            assert_eq!(delays, vec![1000 / hz; 4]);
            assert_eq!(false_switches, 0);
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
