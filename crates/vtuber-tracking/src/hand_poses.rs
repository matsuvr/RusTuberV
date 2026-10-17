//! Image-classified hand poses, confirmed on camera frames and blended on render ticks.
//! The catalog uses positive anatomical flexion magnitudes. In the palm frame
//! (index cross little), inward flexion is negative on the left, positive on
//! the right. Opening toward the thumb has the same sign on both hands.

use vtuber_core::arm_tracking::{HandFingerPose, HandGesture};

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
        observation: Option<HandGesture>,
        flexion_sign: f32,
    ) -> Option<HandFingerPose> {
        let Some(observed) = observation else {
            self.interrupt();
            return None;
        };
        let closest = catalog::POSES
            .iter()
            .position(|(name, _)| *name == pose_name(observed));
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
        .chain(pose.thumb_cmc)
        .chain([pose.thumb_spread])
}

/// Orientation variants share a pose before confirmation. Palm orientation is
/// carried by the separate landmark channel, not baked into these finger angles.
/// Heart variants use the catalog's authored approximation of one heart half.
fn pose_name(gesture: HandGesture) -> &'static str {
    use HandGesture::*;
    match gesture {
        Palm | HandDown | HandRight | HandLeft => "open",
        ThumbIndex => "finger_gun",
        Like | Dislike | ThumbLeft | ThumbRight | ThumbDown => "thumbs_up",
        HalfUp | HalfLeft | HalfRight | HalfDown => "half_closed",
        PartHandHeart | PartHandHeart2 => "half_heart",
        Fist | FistInverted => "fist",
        TwoLeft | TwoRight | TwoDown | TwoUp | TwoUpInverted => "two_together",
        Grabbing => "claw",
        Grip => "grip",
        Point => "point_click",
        Call => "shaka",
        Three3 => "i_love_you",
        LittleFinger => "little_only",
        MiddleFinger => "middle_only",
        Four => "four",
        Mute | One | OneLeft | OneRight | OneDown => "point",
        Ok => "ok",
        Peace | PeaceInverted => "peace",
        Rock => "horns",
        Stop | StopInverted => "open_together",
        Three => "three_thumb_index_middle",
        Three2 => "three",
        ThreeGun => "three_gun",
    }
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

    fn pose(name: &str) -> HandFingerPose {
        catalog::POSES
            .iter()
            .find(|(label, _)| *label == name)
            .unwrap()
            .1
    }

    #[test]
    fn catalog_is_distinct_and_anatomically_bounded() {
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
                target.spread.into_iter().all(|angle| angle.abs() <= 0.35),
                "{name}"
            );
            assert!((0.0..=1.31).contains(&target.thumb_spread), "{name}");
            assert!((-0.78..=0.7).contains(&target.thumb_cmc[0]), "{name}");
            assert!((-0.5..=0.78).contains(&target.thumb_cmc[1]), "{name}");
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

    #[test]
    fn every_gesture_selects_a_catalog_pose_with_anatomical_handedness() {
        for gesture in [
            HandGesture::HandDown,
            HandGesture::HandRight,
            HandGesture::HandLeft,
            HandGesture::ThumbIndex,
            HandGesture::ThumbLeft,
            HandGesture::ThumbRight,
            HandGesture::ThumbDown,
            HandGesture::HalfUp,
            HandGesture::HalfLeft,
            HandGesture::HalfRight,
            HandGesture::HalfDown,
            HandGesture::PartHandHeart,
            HandGesture::PartHandHeart2,
            HandGesture::FistInverted,
            HandGesture::TwoLeft,
            HandGesture::TwoRight,
            HandGesture::TwoDown,
            HandGesture::Grabbing,
            HandGesture::Grip,
            HandGesture::Point,
            HandGesture::Call,
            HandGesture::Three3,
            HandGesture::LittleFinger,
            HandGesture::MiddleFinger,
            HandGesture::Dislike,
            HandGesture::Fist,
            HandGesture::Four,
            HandGesture::Like,
            HandGesture::Mute,
            HandGesture::Ok,
            HandGesture::One,
            HandGesture::Palm,
            HandGesture::Peace,
            HandGesture::PeaceInverted,
            HandGesture::Rock,
            HandGesture::Stop,
            HandGesture::StopInverted,
            HandGesture::Three,
            HandGesture::Three2,
            HandGesture::TwoUp,
            HandGesture::TwoUpInverted,
            HandGesture::ThreeGun,
            HandGesture::OneLeft,
            HandGesture::OneRight,
            HandGesture::OneDown,
        ] {
            let expected = pose(pose_name(gesture));
            for sign in [-1.0, 1.0] {
                let mut selector = HandPoseSelector::new();
                for _ in 1..CONFIRM_FRAMES {
                    assert_eq!(selector.observe(Some(gesture), sign), None);
                }
                let selected = selector.observe(Some(gesture), sign).unwrap();
                assert_eq!(with_flexion_sign(selected, sign), expected, "{gesture:?}");
            }
        }
        for (gesture, name) in [
            (HandGesture::Fist, "fist"),
            (HandGesture::Peace, "peace"),
            (HandGesture::Palm, "open"),
            (HandGesture::Ok, "ok"),
            (HandGesture::Like, "thumbs_up"),
            (HandGesture::Three, "three_thumb_index_middle"),
            (HandGesture::Three2, "three"),
            (HandGesture::Three3, "i_love_you"),
        ] {
            assert_eq!(pose_name(gesture), name);
        }
    }

    #[test]
    fn orientation_variants_confirm_one_shape_and_noise_cannot_switch_it() {
        let mut selector = HandPoseSelector::new();
        for gesture in [
            HandGesture::Peace,
            HandGesture::PeaceInverted,
            HandGesture::Peace,
        ] {
            assert_eq!(selector.observe(Some(gesture), 1.0), None);
        }
        let peace = Some(pose("peace"));
        assert_eq!(
            selector.observe(Some(HandGesture::PeaceInverted), 1.0),
            peace
        );
        for _ in 0..10 {
            assert_eq!(selector.observe(Some(HandGesture::Fist), 1.0), peace);
            assert_eq!(selector.observe(Some(HandGesture::Peace), 1.0), peace);
        }
    }

    #[test]
    fn missing_and_quarantined_observations_break_confirmation() {
        let mut selector = HandPoseSelector::new();
        let open = Some(pose("open"));
        for _ in 0..CONFIRM_FRAMES {
            selector.observe(Some(HandGesture::Palm), 1.0);
        }
        for _ in 1..CONFIRM_FRAMES {
            assert_eq!(selector.observe(Some(HandGesture::Fist), 1.0), open);
        }
        assert_eq!(selector.observe(None, 1.0), None);
        for _ in 1..CONFIRM_FRAMES {
            assert_eq!(selector.observe(Some(HandGesture::Fist), 1.0), open);
        }
        selector.interrupt();
        for _ in 1..CONFIRM_FRAMES {
            assert_eq!(selector.observe(Some(HandGesture::Fist), 1.0), open);
        }
        assert_eq!(
            selector.observe(Some(HandGesture::Fist), 1.0),
            Some(pose("fist"))
        );
    }

    #[test]
    fn transitions_cannot_overshoot_the_authored_joint_limits() {
        let mut smoother = FingerSmootherState::new(pose("open"));
        for name in ["fist", "open", "ok", "middle_only", "fist"] {
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
