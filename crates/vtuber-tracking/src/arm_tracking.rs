//! Pure shoulder-relative retargeting and observation-rate arm smoothing.
//!
//! Visibility, loss/recovery, calibration sample selection, and render-clock
//! interpolation are separate policies. None of them is silently performed here.

use std::num::NonZeroU64;
use std::time::Duration;

use nalgebra::Vector3;
use vtuber_core::arm_tracking::{
    ArmBlendWeight, ArmBlendWeights, ArmControlFrame, ArmLandmarks, ArmTrackingTarget,
    ArmTrackingTargets, PoseArmFrame, PoseWorldLandmark,
};
use vtuber_core::{FrameSeq, MonoTimeNs};

/// A fixed subject arm length measured at calibration, never remeasured per tick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmReferenceLength(f32);

impl ArmReferenceLength {
    /// Total upper-arm plus forearm length in meters.
    #[must_use]
    pub const fn meters(self) -> f32 {
        self.0
    }
}

/// Measures one calibration sample. The caller selects a reliably observed sample.
///
/// Zero-length bones have no usable geometry and return None. This does not
/// invent an average-human arm, reuse another side, or select an idle pose.
#[must_use]
pub fn measure_arm_reference(arm: ArmLandmarks) -> Option<ArmReferenceLength> {
    let upper = (vector(arm.elbow.meters) - vector(arm.shoulder.meters)).norm();
    let lower = (vector(arm.wrist.meters) - vector(arm.elbow.meters)).norm();
    let total = upper + lower;
    (upper > 0.0 && lower > 0.0 && total.is_finite()).then_some(ArmReferenceLength(total))
}

/// Removes subject translation, converts basis once, and divides by a fixed scale.
///
/// Inputs are validated world observations, not normalized image coordinates.
/// The +Y/+Z sign change converts Pose's basis to ArmTrackingTarget's canonical
/// front view. No torso rotation, mirror, avatar scale, confidence policy, or
/// bone-length stretching is applied here.
#[must_use]
pub fn retarget_arm_landmarks(
    arm: ArmLandmarks,
    reference: ArmReferenceLength,
) -> ArmTrackingTarget {
    let shoulder = vector(arm.shoulder.meters);
    let offset = |point: [f32; 3]| {
        let v = (vector(point) - shoulder) / reference.meters();
        [v.x, -v.y, -v.z]
    };
    ArmTrackingTarget {
        wrist: offset(arm.wrist.meters),
        elbow_pole: offset(arm.elbow.meters),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PointFilterState {
    raw: Vector3<f32>,
    filtered: Vector3<f32>,
    velocity: Vector3<f32>,
}

impl PointFilterState {
    fn new(value: [f32; 3]) -> Self {
        let value = vector(value);
        Self {
            raw: value,
            filtered: value,
            velocity: Vector3::zeros(),
        }
    }
}

/// Explicit state passed between pure filter calls; it does not own a clock.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmFilterState {
    wrist: PointFilterState,
    elbow: PointFilterState,
}

impl ArmFilterState {
    /// Seeds observation filtering. Visible acquisition blending belongs to the compositor.
    #[must_use]
    pub fn new(target: ArmTrackingTarget) -> Self {
        Self {
            wrist: PointFilterState::new(target.wrist),
            elbow: PointFilterState::new(target.elbow_pole),
        }
    }
}

/// One adaptive low-pass update per NEW camera observation.
///
/// elapsed_ns is the positive difference of capture timestamps, not render dt.
/// Callers do not re-feed a retained sample on every draw. Fixed research
/// starting values use a 1 Hz derivative cutoff, 1.5 Hz wrist / 1 Hz elbow
/// minimum cutoff and less depth bandwidth. They are not validated aesthetic
/// presets. Final bone lengths are enforced by the existing analytic IK.
///
/// This is a value-in/value-out function: no thread, clock, global, or ECS writes.
#[must_use]
pub fn filter_arm_target(
    previous: ArmFilterState,
    target: ArmTrackingTarget,
    elapsed_ns: NonZeroU64,
) -> (ArmFilterState, ArmTrackingTarget) {
    let seconds = elapsed_ns.get() as f32 * 1.0e-9;
    let wrist = filter_point(previous.wrist, target.wrist, seconds, 1.5, 1.0);
    let elbow = filter_point(previous.elbow, target.elbow_pole, seconds, 1.0, 0.5);
    let filtered = ArmTrackingTarget {
        wrist: array(wrist.filtered),
        elbow_pole: array(elbow.filtered),
    };
    (ArmFilterState { wrist, elbow }, filtered)
}

fn filter_point(
    previous: PointFilterState,
    target: [f32; 3],
    seconds: f32,
    minimum_cutoff_hz: f32,
    beta: f32,
) -> PointFilterState {
    let raw = vector(target);
    let derivative = (raw - previous.raw) / seconds;
    let velocity = previous.velocity + (derivative - previous.velocity) * alpha(1.0, seconds);
    let cutoff = minimum_cutoff_hz + beta * velocity.norm();
    let gain = Vector3::new(
        alpha(cutoff, seconds),
        alpha(cutoff, seconds),
        alpha(cutoff * 0.75, seconds),
    );
    let filtered = previous.filtered + (raw - previous.filtered).component_mul(&gain);
    PointFilterState {
        raw,
        filtered,
        velocity,
    }
}

fn alpha(cutoff_hz: f32, seconds: f32) -> f32 {
    let value = std::f32::consts::TAU * cutoff_hz * seconds;
    value / (1.0 + value)
}

fn vector([x, y, z]: [f32; 3]) -> Vector3<f32> {
    Vector3::new(x, y, z)
}

fn array(value: Vector3<f32>) -> [f32; 3] {
    [value.x, value.y, value.z]
}

/// Per-side adoption and temporal policy for observed arms.
///
/// Validation belongs to the settings layer; these are research starting values
/// (hold 150 ms, return 350 ms, acquire 200 ms) that the 5/5 evaluation adjusts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmTrackingProfile {
    /// Minimum shoulder visibility to use the arm at all.
    pub shoulder_visibility: f32,
    /// Minimum wrist visibility to follow the hand.
    pub wrist_visibility: f32,
    /// Minimum elbow visibility to use the observed bend plane.
    pub elbow_visibility: f32,
    /// How long a lost channel keeps its last value before returning.
    pub hold: Duration,
    /// How long the return to the virtual arm takes after the hold.
    pub return_to_virtual: Duration,
    /// How long a reacquired channel takes to blend back in.
    pub acquire: Duration,
}

impl Default for ArmTrackingProfile {
    fn default() -> Self {
        Self {
            shoulder_visibility: 0.5,
            wrist_visibility: 0.5,
            elbow_visibility: 0.5,
            hold: Duration::from_millis(150),
            return_to_virtual: Duration::from_millis(350),
            acquire: Duration::from_millis(200),
        }
    }
}

/// Which joints of one arm are visible enough to adopt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArmObservationQuality {
    /// The shoulder origin is usable.
    pub shoulder: bool,
    /// The wrist position is usable.
    pub wrist: bool,
    /// The elbow bend plane is usable.
    pub elbow: bool,
}

impl ArmObservationQuality {
    /// Whether the shoulder and wrist together allow following the hand.
    #[must_use]
    pub const fn follows_wrist(self) -> bool {
        self.shoulder && self.wrist
    }

    /// Whether the full chain allows using the observed bend plane.
    #[must_use]
    pub const fn uses_elbow(self) -> bool {
        self.shoulder && self.wrist && self.elbow
    }
}

/// Classifies one observed arm without fabricating missing confidence.
///
/// A score that MediaPipe did not supply is not treated as `1.0`; the joint is
/// simply not adopted. Visibility is preferred over presence when both exist.
#[must_use]
pub fn assess_arm_observation(
    arm: &ArmLandmarks,
    profile: &ArmTrackingProfile,
) -> ArmObservationQuality {
    let adopt = |point: PoseWorldLandmark, threshold: f32| {
        point
            .visibility
            .or(point.presence)
            .is_some_and(|score| score >= threshold)
    };
    ArmObservationQuality {
        shoulder: adopt(arm.shoulder, profile.shoulder_visibility),
        wrist: adopt(arm.wrist, profile.wrist_visibility),
        elbow: adopt(arm.elbow, profile.elbow_visibility),
    }
}

/// Number of reliable samples kept before the arm length is fixed.
pub const ARM_CALIBRATION_CAPACITY: usize = 5;

/// A bounded set of reliable length samples and the fixed median once confirmed.
///
/// The length is frozen after confirmation, so per-frame reach changes never
/// renormalize it. The opposite arm's length is never substituted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmCalibrationState {
    samples: [f32; ARM_CALIBRATION_CAPACITY],
    count: usize,
    confirmed: Option<ArmReferenceLength>,
}

impl ArmCalibrationState {
    /// No samples yet and no fixed length.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            samples: [0.0; ARM_CALIBRATION_CAPACITY],
            count: 0,
            confirmed: None,
        }
    }

    /// The fixed total arm length, once enough reliable samples were collected.
    #[must_use]
    pub const fn confirmed_length(self) -> Option<ArmReferenceLength> {
        self.confirmed
    }

    /// How many reliable samples have been retained.
    #[must_use]
    pub const fn sample_count(self) -> usize {
        self.count
    }
}

impl Default for ArmCalibrationState {
    fn default() -> Self {
        Self::new()
    }
}

/// Adds one reliable length sample; confirms the median after enough of them.
///
/// A confirmed length is never replaced. `None` leaves the state untouched.
#[must_use]
pub fn update_arm_calibration(
    previous: &ArmCalibrationState,
    sample: Option<ArmReferenceLength>,
) -> ArmCalibrationState {
    let mut next = *previous;
    if next.confirmed.is_some() {
        return next;
    }
    let Some(length) = sample else {
        return next;
    };
    if next.count >= ARM_CALIBRATION_CAPACITY {
        return next;
    }
    if let Some(slot) = next.samples.get_mut(next.count) {
        *slot = length.meters();
    }
    next.count += 1;
    if next.count == ARM_CALIBRATION_CAPACITY {
        let mut sorted = next.samples;
        sorted.sort_by(f32::total_cmp);
        if let Some(median) = sorted.get(ARM_CALIBRATION_CAPACITY / 2)
            && median.is_finite()
            && *median > 0.0
        {
            next.confirmed = Some(ArmReferenceLength(*median));
        }
    }
    next
}

/// A stabilized bend plane plus the observed contribution that survived it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoleUpdate {
    /// The bend-plane target to feed forward, or `None` when it is undefined.
    pub pole: Option<[f32; 3]>,
    /// The observed weight after the extension falloff, in `0.0..=1.0`.
    pub observed_weight: f32,
}

/// Extension ratio at which the observed pole starts losing authority.
const POLE_EXTENSION_BAND: f32 = 0.08;

/// Keeps the elbow pole continuous and in the plane perpendicular to the arm.
///
/// The observed elbow is projected onto the shoulder-to-wrist axis, and its
/// observed weight is reduced continuously as the arm approaches full
/// extension, where the bend side is ill-conditioned. When the incoming and
/// previous planes face opposite ways the previous plane is retained rather
/// than interpolating through zero. An undefined plane returns `None` instead
/// of a fabricated world axis.
#[must_use]
pub fn stabilize_elbow_pole(
    previous: Option<[f32; 3]>,
    target: ArmTrackingTarget,
    observed_weight: f32,
) -> PoleUpdate {
    let wrist = vector(target.wrist);
    let axial = wrist.norm();
    if !axial.is_finite() || axial <= f32::EPSILON {
        return PoleUpdate {
            pole: None,
            observed_weight: 0.0,
        };
    }
    let axis = wrist / axial;
    let extension = axial.clamp(0.0, 1.0);
    let falloff = if extension >= 1.0 {
        0.0
    } else if extension <= 1.0 - POLE_EXTENSION_BAND {
        1.0
    } else {
        (1.0 - extension) / POLE_EXTENSION_BAND
    };
    let weight = observed_weight.clamp(0.0, 1.0) * falloff;

    let observed_perpendicular = perpendicular(vector(target.elbow_pole), axis);
    let observed_length = observed_perpendicular.norm();
    let observed_direction = finite_normalized(observed_perpendicular);
    let previous_perpendicular = previous.map(|value| perpendicular(vector(value), axis));
    let previous_length = previous_perpendicular
        .map(|value| value.norm())
        .unwrap_or(0.0);
    let previous_direction = previous_perpendicular.and_then(finite_normalized);

    match (observed_direction, previous_direction) {
        (Some(observed), Some(prior)) => {
            let opposing = prior.dot(&observed) < 0.0;
            let blend = if opposing { 0.0 } else { weight };
            let combined = prior * (1.0 - blend) + observed * blend;
            let direction = finite_normalized(combined).unwrap_or(prior);
            let length = if observed_length > f32::EPSILON {
                observed_length
            } else {
                previous_length.max(f32::EPSILON)
            };
            PoleUpdate {
                pole: Some(array(direction * length)),
                observed_weight: if opposing { 0.0 } else { weight },
            }
        }
        (Some(observed), None) => PoleUpdate {
            pole: Some(array(observed * observed_length.max(f32::EPSILON))),
            observed_weight: weight,
        },
        (None, Some(prior)) => PoleUpdate {
            pole: Some(array(prior * previous_length.max(f32::EPSILON))),
            observed_weight: 0.0,
        },
        (None, None) => PoleUpdate {
            pole: None,
            observed_weight: 0.0,
        },
    }
}

/// A per-channel display blend that advances on render ticks and observations.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct ChannelBlend {
    weight: f32,
    last_seen: Option<MonoTimeNs>,
}

impl ChannelBlend {
    fn advance(
        &mut self,
        now: MonoTimeNs,
        present: bool,
        render_dt_ns: Option<u64>,
        profile: &ArmTrackingProfile,
    ) {
        if present {
            self.last_seen = Some(now);
            let acquire_ns = profile.acquire.as_nanos();
            let step = match render_dt_ns {
                Some(dt) if acquire_ns > 0 => dt as f32 / acquire_ns as f32,
                _ => 1.0,
            };
            self.weight = (self.weight + step).clamp(0.0, 1.0);
        } else if let Some(seen) = self.last_seen {
            let elapsed_ms = now.0.saturating_sub(seen.0) as f32 * 1.0e-6;
            let hold_ms = profile.hold.as_secs_f32() * 1.0e3;
            let return_ms = profile.return_to_virtual.as_secs_f32() * 1.0e3;
            self.weight = if elapsed_ms <= hold_ms {
                1.0
            } else if return_ms > 0.0 && elapsed_ms <= hold_ms + return_ms {
                (1.0 - (elapsed_ms - hold_ms) / return_ms).clamp(0.0, 1.0)
            } else {
                0.0
            };
        } else {
            self.weight = 0.0;
        }
    }
}

/// Per-side observation, calibration, filter, and blend state.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ArmSideState {
    calibration: ArmCalibrationState,
    filter: Option<ArmFilterState>,
    last_pole: Option<[f32; 3]>,
    last_target: Option<ArmTrackingTarget>,
    last_wrist_capture: Option<MonoTimeNs>,
    last_elbow_capture: Option<MonoTimeNs>,
    wrist_blend: ChannelBlend,
    pole_blend: ChannelBlend,
    pole_factor: f32,
}

impl ArmSideState {
    const fn empty() -> Self {
        Self {
            calibration: ArmCalibrationState::new(),
            filter: None,
            last_pole: None,
            last_target: None,
            last_wrist_capture: None,
            last_elbow_capture: None,
            wrist_blend: ChannelBlend {
                weight: 0.0,
                last_seen: None,
            },
            pole_blend: ChannelBlend {
                weight: 0.0,
                last_seen: None,
            },
            pole_factor: 0.0,
        }
    }

    fn weights(&self) -> ArmBlendWeight {
        ArmBlendWeight {
            wrist: self.wrist_blend.weight,
            pole: self.pole_blend.weight * self.pole_factor,
        }
    }

    fn advance_without_observation(
        &mut self,
        now: MonoTimeNs,
        render_dt_ns: Option<u64>,
        profile: &ArmTrackingProfile,
    ) {
        self.wrist_blend.advance(now, false, render_dt_ns, profile);
        self.pole_blend.advance(now, false, render_dt_ns, profile);
    }

    fn step(
        &mut self,
        arm: Option<&ArmLandmarks>,
        frame: &PoseArmFrame,
        now: MonoTimeNs,
        render_dt_ns: Option<u64>,
        profile: &ArmTrackingProfile,
    ) {
        let quality = arm.map(|value| assess_arm_observation(value, profile));
        let wrist_usable = quality.is_some_and(ArmObservationQuality::follows_wrist);
        let elbow_usable = quality.is_some_and(ArmObservationQuality::uses_elbow);

        let sample = arm
            .filter(|_| elbow_usable)
            .and_then(|value| measure_arm_reference(*value));
        self.calibration = update_arm_calibration(&self.calibration, sample);

        let Some(reference) = self.calibration.confirmed_length() else {
            self.advance_without_observation(now, render_dt_ns, profile);
            return;
        };

        let mut elbow_tracked = false;
        if let Some(arm) = arm.filter(|_| wrist_usable) {
            let mut target = retarget_arm_landmarks(*arm, reference);
            if elbow_usable {
                let update = stabilize_elbow_pole(self.last_pole, target, 1.0);
                self.pole_factor = update.observed_weight;
                if let Some(pole) = update.pole {
                    target.elbow_pole = pole;
                    self.last_pole = Some(pole);
                    elbow_tracked = true;
                }
            }
            let wrist_elapsed = capture_elapsed(self.last_wrist_capture, frame.captured_at);
            let elbow_elapsed = capture_elapsed(self.last_elbow_capture, frame.captured_at);
            let (next, output) = match self.filter {
                None => (ArmFilterState::new(target), target),
                Some(filter) => filter_arm_channels(
                    filter,
                    target,
                    wrist_elapsed,
                    if elbow_tracked { elbow_elapsed } else { None },
                ),
            };
            self.filter = Some(next);
            self.last_target = Some(output);
            self.last_wrist_capture = Some(frame.captured_at);
            if elbow_tracked {
                self.last_elbow_capture = Some(frame.captured_at);
            }
        }

        self.wrist_blend
            .advance(now, wrist_usable, render_dt_ns, profile);
        self.pole_blend
            .advance(now, elbow_tracked, render_dt_ns, profile);
    }
}

/// Pure temporal state for observed arm tracking. It owns no clock or ECS.
///
/// `now` is supplied by the caller. A tick with no new inference result
/// advances only the display blend; a result with `observation: None` is a
/// completed "no person" inference and starts the hold/return timeline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmTrackingState {
    left: ArmSideState,
    right: ArmSideState,
    last_consumed: Option<(FrameSeq, MonoTimeNs)>,
    last_now: Option<MonoTimeNs>,
}

impl ArmTrackingState {
    /// No calibration, no filter state, no consumed frame.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            left: ArmSideState::empty(),
            right: ArmSideState::empty(),
            last_consumed: None,
            last_now: None,
        }
    }

    /// Drops calibration and observation state, e.g. after a camera or model change.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for ArmTrackingState {
    fn default() -> Self {
        Self::new()
    }
}

/// Advances arm tracking by one tick and emits a control frame for new results.
///
/// A stale or duplicate source sequence/capture time is ignored. Filter inputs
/// use positive capture-time differences, so a held sample is never re-fed.
/// When `observation` is `None` the state still advances its display blend but
/// no frame is produced, so no fictitious source sequence is emitted.
#[must_use]
pub fn step_arm_tracking(
    previous: &ArmTrackingState,
    observation: Option<&PoseArmFrame>,
    now: MonoTimeNs,
    profile: &ArmTrackingProfile,
) -> (ArmTrackingState, Option<ArmControlFrame>) {
    let mut state = *previous;
    let render_dt_ns = state.last_now.map(|last| now.0.saturating_sub(last.0));
    state.last_now = Some(now);

    let Some(frame) = observation else {
        state
            .left
            .advance_without_observation(now, render_dt_ns, profile);
        state
            .right
            .advance_without_observation(now, render_dt_ns, profile);
        return (state, None);
    };

    if let Some((seq, captured_at)) = state.last_consumed
        && (frame.source_seq.0 <= seq.0 || frame.captured_at.0 <= captured_at.0)
    {
        return (state, None);
    }
    state.last_consumed = Some((frame.source_seq, frame.captured_at));

    let (left, right) = match &frame.observation {
        Some(value) => (Some(&value.left), Some(&value.right)),
        None => (None, None),
    };
    state.left.step(left, frame, now, render_dt_ns, profile);
    state.right.step(right, frame, now, render_dt_ns, profile);

    let control = ArmControlFrame {
        source_seq: frame.source_seq,
        captured_at: frame.captured_at,
        produced_at: now,
        targets: ArmTrackingTargets {
            left: state.left.last_target,
            right: state.right.last_target,
        },
        weights: ArmBlendWeights {
            left: state.left.weights(),
            right: state.right.weights(),
        },
    };
    (state, Some(control))
}

/// Positive capture-time difference for one filter channel, if it advanced.
fn capture_elapsed(previous: Option<MonoTimeNs>, captured_at: MonoTimeNs) -> Option<NonZeroU64> {
    previous.and_then(|last| NonZeroU64::new(captured_at.0.saturating_sub(last.0)))
}

/// Updates each filter channel only when that channel has a new observation.
///
/// Channels without a new observation keep their previous filtered value, so an
/// occluded elbow does not freeze the wrist or inject a fabricated elbow.
fn filter_arm_channels(
    previous: ArmFilterState,
    target: ArmTrackingTarget,
    wrist_elapsed: Option<NonZeroU64>,
    elbow_elapsed: Option<NonZeroU64>,
) -> (ArmFilterState, ArmTrackingTarget) {
    let mut next = previous;
    let mut output = ArmTrackingTarget {
        wrist: array(previous.wrist.filtered),
        elbow_pole: array(previous.elbow.filtered),
    };
    if let Some(elapsed) = wrist_elapsed {
        let wrist = filter_point(
            previous.wrist,
            target.wrist,
            elapsed.get() as f32 * 1.0e-9,
            1.5,
            1.0,
        );
        output.wrist = array(wrist.filtered);
        next.wrist = wrist;
    }
    if let Some(elapsed) = elbow_elapsed {
        let elbow = filter_point(
            previous.elbow,
            target.elbow_pole,
            elapsed.get() as f32 * 1.0e-9,
            1.0,
            0.5,
        );
        output.elbow_pole = array(elbow.filtered);
        next.elbow = elbow;
    }
    (next, output)
}

fn perpendicular(value: Vector3<f32>, axis: Vector3<f32>) -> Vector3<f32> {
    value - axis * value.dot(&axis)
}

fn finite_normalized(value: Vector3<f32>) -> Option<Vector3<f32>> {
    let length_squared = value.norm_squared();
    (value.iter().all(|component| component.is_finite())
        && length_squared.is_finite()
        && length_squared > f32::EPSILON)
        .then(|| value / length_squared.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vtuber_core::arm_tracking::PoseWorldLandmark;

    fn point(meters: [f32; 3]) -> PoseWorldLandmark {
        PoseWorldLandmark {
            meters,
            visibility: Some(1.0),
            presence: Some(1.0),
        }
    }

    fn arm() -> ArmLandmarks {
        ArmLandmarks {
            shoulder: point([0.0, 0.0, 0.0]),
            elbow: point([0.3, 0.0, 0.0]),
            wrist: point([0.3, -0.4, 0.0]),
        }
    }

    fn target(value: f32) -> ArmTrackingTarget {
        ArmTrackingTarget {
            wrist: [value; 3],
            elbow_pole: [value; 3],
        }
    }

    fn dt() -> NonZeroU64 {
        NonZeroU64::new(16_666_667).unwrap()
    }

    fn near(a: [f32; 3], b: [f32; 3]) {
        assert!((vector(a) - vector(b)).norm() < 1.0e-5, "{a:?} != {b:?}");
    }

    #[test]
    fn calibration_is_total_bone_length_not_shoulder_wrist_distance() {
        let reference = measure_arm_reference(arm()).unwrap();
        assert!((reference.meters() - 0.7).abs() < 1.0e-6);
    }

    #[test]
    fn retarget_removes_translation_and_converts_pose_basis() {
        let reference = measure_arm_reference(arm()).unwrap();
        let original = retarget_arm_landmarks(arm(), reference);
        near(original.wrist, [3.0 / 7.0, 4.0 / 7.0, 0.0]);
        let shift =
            |p: PoseWorldLandmark| point(array(vector(p.meters) + Vector3::new(1.0, 2.0, 3.0)));
        let a = arm();
        let translated = ArmLandmarks {
            shoulder: shift(a.shoulder),
            elbow: shift(a.elbow),
            wrist: shift(a.wrist),
        };
        near(
            retarget_arm_landmarks(translated, reference).wrist,
            original.wrist,
        );
        let closer = ArmLandmarks {
            wrist: point([0.3, -0.4, -0.2]),
            ..a
        };
        assert!(retarget_arm_landmarks(closer, reference).wrist[2] > 0.0);
    }

    #[test]
    fn calibration_scale_is_frozen_when_observed_reach_changes() {
        let a = arm();
        let reference = measure_arm_reference(a).unwrap();
        let extended = ArmLandmarks {
            wrist: point([0.6, 0.0, 0.0]),
            ..a
        };
        near(
            retarget_arm_landmarks(extended, reference).wrist,
            [6.0 / 7.0, 0.0, 0.0],
        );
    }

    #[test]
    fn subject_scale_does_not_change_normalized_targets() {
        let a = arm();
        let twice = |p: PoseWorldLandmark| point(array(vector(p.meters) * 2.0));
        let b = ArmLandmarks {
            shoulder: twice(a.shoulder),
            elbow: twice(a.elbow),
            wrist: twice(a.wrist),
        };
        let a = retarget_arm_landmarks(a, measure_arm_reference(a).unwrap());
        let b = retarget_arm_landmarks(b, measure_arm_reference(b).unwrap());
        near(a.wrist, b.wrist);
        near(a.elbow_pole, b.elbow_pole);
    }

    #[test]
    fn zero_bone_has_no_calibration_instead_of_a_default_length() {
        let a = arm();
        assert_eq!(
            measure_arm_reference(ArmLandmarks {
                elbow: a.shoulder,
                ..a
            }),
            None
        );
        assert_eq!(
            measure_arm_reference(ArmLandmarks {
                wrist: a.elbow,
                ..a
            }),
            None
        );
    }

    #[test]
    fn constant_input_is_unchanged_at_both_observation_rates() {
        for ns in [16_666_667, 33_333_333] {
            let target = target(0.3);
            let mut state = ArmFilterState::new(target);
            for _ in 0..60 {
                let (next, value) = filter_arm_target(state, target, NonZeroU64::new(ns).unwrap());
                assert_eq!(value, target);
                state = next;
            }
        }
    }

    #[test]
    fn rapid_motion_gets_more_bandwidth_and_depth_less() {
        let initial = ArmFilterState::new(target(0.0));
        let (_, small) = filter_arm_target(initial, target(0.01), dt());
        let (_, large) = filter_arm_target(initial, target(1.0), dt());
        assert!(large.wrist[0] > small.wrist[0] / 0.01);
        assert!(large.wrist[2] < large.wrist[0]);
        assert!(large.elbow_pole[0] < large.wrist[0]);
    }

    #[test]
    fn stationary_jitter_is_reduced_and_calls_are_deterministic() {
        let mut state = ArmFilterState::new(target(0.0));
        let mut squared = 0.0;
        for frame in 0..120 {
            let raw = target(if frame % 2 == 0 { 0.01 } else { -0.01 });
            let result = filter_arm_target(state, raw, dt());
            assert_eq!(result, filter_arm_target(state, raw, dt()));
            state = result.0;
            squared += result.1.wrist[0].powi(2);
        }
        assert!((squared / 120.0).sqrt() < 0.005);
    }

    fn low_visibility(point: PoseWorldLandmark) -> PoseWorldLandmark {
        PoseWorldLandmark {
            visibility: Some(0.1),
            ..point
        }
    }

    fn arm_at(elbow: [f32; 3], wrist: [f32; 3]) -> ArmLandmarks {
        ArmLandmarks {
            shoulder: point([0.0, 0.0, 0.0]),
            elbow: point(elbow),
            wrist: point(wrist),
        }
    }

    fn observation(
        left: ArmLandmarks,
        right: ArmLandmarks,
    ) -> vtuber_core::arm_tracking::PoseArmObservation {
        vtuber_core::arm_tracking::PoseArmObservation { left, right }
    }

    fn pose_frame(
        seq: u64,
        captured_ns: u64,
        observation: Option<vtuber_core::arm_tracking::PoseArmObservation>,
    ) -> PoseArmFrame {
        PoseArmFrame {
            source_seq: FrameSeq(seq),
            captured_at: MonoTimeNs(captured_ns),
            inference_finished_at: MonoTimeNs(captured_ns),
            observation,
        }
    }

    const OBSERVATION_STEP_NS: u64 = 33_333_333;

    fn tracked_state(arm: ArmLandmarks, profile: &ArmTrackingProfile) -> ArmTrackingState {
        let mut state = ArmTrackingState::new();
        for seq in 0..(ARM_CALIBRATION_CAPACITY as u64 + 10) {
            let now = seq * OBSERVATION_STEP_NS;
            let frame = pose_frame(seq, now, Some(observation(arm, arm)));
            let (next, _) = step_arm_tracking(&state, Some(&frame), MonoTimeNs(now), profile);
            state = next;
        }
        state
    }

    #[test]
    fn calibration_confirms_only_after_reliable_samples() {
        let profile = ArmTrackingProfile::default();
        let arm = arm();
        let mut state = ArmTrackingState::new();
        for seq in 0..ARM_CALIBRATION_CAPACITY as u64 {
            let now = seq * OBSERVATION_STEP_NS;
            let frame = pose_frame(seq, now, Some(observation(arm, arm)));
            let (next, _) = step_arm_tracking(&state, Some(&frame), MonoTimeNs(now), &profile);
            state = next;
        }
        assert_eq!(
            state.left.calibration.sample_count(),
            ARM_CALIBRATION_CAPACITY
        );
        let length = state.left.calibration.confirmed_length().unwrap();
        assert!((length.meters() - 0.7).abs() < 1.0e-5);
        assert_eq!(state.right.calibration.confirmed_length(), Some(length));
    }

    #[test]
    fn low_confidence_observations_never_confirm_calibration() {
        let profile = ArmTrackingProfile::default();
        let base = arm();
        let dim = ArmLandmarks {
            shoulder: low_visibility(base.shoulder),
            elbow: low_visibility(base.elbow),
            wrist: low_visibility(base.wrist),
        };
        let mut state = ArmTrackingState::new();
        for seq in 0..40u64 {
            let now = seq * OBSERVATION_STEP_NS;
            let frame = pose_frame(seq, now, Some(observation(dim, dim)));
            let (next, _) = step_arm_tracking(&state, Some(&frame), MonoTimeNs(now), &profile);
            state = next;
        }
        assert!(state.left.calibration.confirmed_length().is_none());
        assert!(state.left.last_target.is_none());
        assert_eq!(state.left.weights().wrist, 0.0);
    }

    #[test]
    fn a_duplicate_capture_is_not_refed_into_the_filter() {
        let profile = ArmTrackingProfile::default();
        let arm = arm();
        let state = tracked_state(arm, &profile);
        let now = 10_000_000_000u64;
        let frame = pose_frame(500, now, Some(observation(arm, arm)));
        let (state, control) = step_arm_tracking(&state, Some(&frame), MonoTimeNs(now), &profile);
        assert!(control.is_some());
        let filter = state.left.filter;
        let (again, control) =
            step_arm_tracking(&state, Some(&frame), MonoTimeNs(now + 16_000_000), &profile);
        assert!(control.is_none());
        assert_eq!(again.left.filter, filter);
    }

    #[test]
    fn no_new_frame_is_distinct_from_a_no_person_result() {
        let profile = ArmTrackingProfile::default();
        let (state, control) =
            step_arm_tracking(&ArmTrackingState::new(), None, MonoTimeNs(0), &profile);
        assert!(control.is_none());
        assert!(state.last_consumed.is_none());

        let frame = pose_frame(0, 0, None);
        let (_, control) = step_arm_tracking(&state, Some(&frame), MonoTimeNs(0), &profile);
        let control = control.unwrap();
        assert_eq!(control.source_seq, FrameSeq(0));
        assert!(control.targets.left.is_none());
        assert_eq!(control.weights, ArmBlendWeights::default());
    }

    #[test]
    fn elbow_occlusion_keeps_the_wrist_tracking_while_the_pole_returns() {
        let profile = ArmTrackingProfile::default();
        let base = arm();
        let mut state = tracked_state(base, &profile);
        assert_eq!(state.left.weights().wrist, 1.0);

        let occluded = ArmLandmarks {
            elbow: low_visibility(base.elbow),
            ..base
        };
        for seq in 100u64..140 {
            let now = 10_000_000_000 + (seq - 100) * OBSERVATION_STEP_NS;
            let frame = pose_frame(seq, now, Some(observation(occluded, occluded)));
            let (next, control) =
                step_arm_tracking(&state, Some(&frame), MonoTimeNs(now), &profile);
            state = next;
            let weights = control.unwrap().weights.left;
            assert_eq!(weights.wrist, 1.0);
        }
        assert_eq!(state.left.weights().pole, 0.0);
        assert!(state.left.last_target.is_some());
    }

    #[test]
    fn losing_the_wrist_returns_to_virtual_within_a_finite_time() {
        let profile = ArmTrackingProfile::default();
        let base = arm();
        let mut state = tracked_state(base, &profile);
        let hidden = ArmLandmarks {
            shoulder: low_visibility(base.shoulder),
            elbow: low_visibility(base.elbow),
            wrist: low_visibility(base.wrist),
        };
        for seq in 100u64..140 {
            let now = 10_000_000_000 + (seq - 100) * OBSERVATION_STEP_NS;
            let frame = pose_frame(seq, now, Some(observation(hidden, hidden)));
            let (next, _) = step_arm_tracking(&state, Some(&frame), MonoTimeNs(now), &profile);
            state = next;
        }
        assert_eq!(state.left.weights().wrist, 0.0);
        assert_eq!(state.left.weights().pole, 0.0);
        assert!(state.left.last_target.is_some());
    }

    #[test]
    fn losing_one_arm_does_not_stop_the_other() {
        let profile = ArmTrackingProfile::default();
        let base = arm();
        let mut state = tracked_state(base, &profile);
        let hidden = ArmLandmarks {
            shoulder: low_visibility(base.shoulder),
            elbow: low_visibility(base.elbow),
            wrist: low_visibility(base.wrist),
        };
        for seq in 100u64..140 {
            let now = 10_000_000_000 + (seq - 100) * OBSERVATION_STEP_NS;
            let frame = pose_frame(seq, now, Some(observation(hidden, base)));
            let (next, control) =
                step_arm_tracking(&state, Some(&frame), MonoTimeNs(now), &profile);
            state = next;
            assert_eq!(control.unwrap().weights.right.wrist, 1.0);
        }
        assert_eq!(state.left.weights().wrist, 0.0);
        assert_eq!(state.right.weights().wrist, 1.0);
    }

    #[test]
    fn reacquire_ramps_the_blend_without_a_jump() {
        let profile = ArmTrackingProfile::default();
        let base = arm();
        let mut state = tracked_state(base, &profile);
        let hidden = ArmLandmarks {
            shoulder: low_visibility(base.shoulder),
            elbow: low_visibility(base.elbow),
            wrist: low_visibility(base.wrist),
        };
        for seq in 100u64..140 {
            let now = 10_000_000_000 + (seq - 100) * OBSERVATION_STEP_NS;
            let frame = pose_frame(seq, now, Some(observation(hidden, hidden)));
            let (next, _) = step_arm_tracking(&state, Some(&frame), MonoTimeNs(now), &profile);
            state = next;
        }
        assert_eq!(state.left.weights().wrist, 0.0);

        let mut previous = 0.0;
        for seq in 140u64..150 {
            let now = 10_000_000_000 + (seq - 100) * OBSERVATION_STEP_NS;
            let frame = pose_frame(seq, now, Some(observation(base, base)));
            let (next, control) =
                step_arm_tracking(&state, Some(&frame), MonoTimeNs(now), &profile);
            state = next;
            let weight = control.unwrap().weights.left.wrist;
            assert!(weight >= previous);
            assert!(weight <= 1.0);
            previous = weight;
        }
        assert_eq!(previous, 1.0);
    }

    #[test]
    fn near_full_extension_does_not_flip_the_elbow_plane() {
        let profile = ArmTrackingProfile::default();
        let theta = std::f32::consts::FRAC_PI_6;
        let upper = 0.4;
        let bent = arm_at(
            [upper * theta.cos(), upper * theta.sin(), 0.0],
            [2.0 * upper * theta.cos(), 0.0, 0.0],
        );
        let state = tracked_state(bent, &profile);
        let mut previous = state.left.last_pole.unwrap();
        let state = (0..30u64).fold(state, |state, step| {
            let noise = if step % 2 == 0 { 0.001 } else { -0.001 };
            let extended = arm_at([upper, noise, 0.0], [2.0 * upper, 0.0, 0.0]);
            let now = 10_000_000_000 + step * OBSERVATION_STEP_NS;
            let frame = pose_frame(100 + step, now, Some(observation(extended, extended)));
            let (next, _) = step_arm_tracking(&state, Some(&frame), MonoTimeNs(now), &profile);
            if let Some(current) = next.left.last_pole {
                let before = finite_normalized(vector(previous)).unwrap();
                let after = finite_normalized(vector(current)).unwrap();
                assert!(before.dot(&after) > 0.0, "pole flipped at step {step}");
                previous = current;
            }
            next
        });
        let rebent = arm_at(
            [upper * theta.cos(), upper * theta.sin(), 0.0],
            [2.0 * upper * theta.cos(), 0.0, 0.0],
        );
        let now = 10_000_000_000 + 30 * OBSERVATION_STEP_NS;
        let frame = pose_frame(130, now, Some(observation(rebent, rebent)));
        let (next, _) = step_arm_tracking(&state, Some(&frame), MonoTimeNs(now), &profile);
        let before = finite_normalized(vector(previous)).unwrap();
        let after = finite_normalized(vector(next.left.last_pole.unwrap())).unwrap();
        assert!(before.dot(&after) > 0.0);
    }

    #[test]
    fn observation_quality_does_not_invent_missing_scores() {
        let profile = ArmTrackingProfile::default();
        let mut arm = arm();
        arm.elbow.visibility = None;
        arm.elbow.presence = None;
        let quality = assess_arm_observation(&arm, &profile);
        assert!(quality.shoulder);
        assert!(quality.wrist);
        assert!(!quality.elbow);
        assert!(quality.follows_wrist());
        assert!(!quality.uses_elbow());
    }

    #[test]
    fn undefined_pole_returns_no_pole_instead_of_a_world_axis() {
        let update = stabilize_elbow_pole(
            None,
            ArmTrackingTarget {
                wrist: [0.0, 0.0, 0.0],
                elbow_pole: [0.1, 0.2, 0.0],
            },
            1.0,
        );
        assert_eq!(
            update,
            PoleUpdate {
                pole: None,
                observed_weight: 0.0
            }
        );
    }
}
