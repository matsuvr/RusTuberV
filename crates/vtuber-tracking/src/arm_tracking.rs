//! Pure shoulder-relative retargeting and render-clock arm smoothing.
//!
//! Visibility, loss/recovery, and calibration sample selection are separate
//! policies. Smoothing runs on the render clock toward the latest retained
//! observation, so the control frame is a continuous signal at the consumer
//! frame rate exactly like the head rotation and translation filters.
//!
//! Loss handling follows the face pipeline's shape: a lost wrist keeps its
//! last authority briefly, then hands authority back to the avatar's virtual
//! arm over [`ArmTrackingProfile::return_to_virtual`]; a reacquisition ramps
//! the observed authority back in from wherever the return had reached. A
//! wrist that teleports farther than [`ArmTrackingProfile::max_wrist_step`]
//! in one observation is quarantined like an outlier head sample, and the
//! first observation after a real loss is accepted as a reacquisition so a
//! hand that moved while hidden can be picked up again.

use std::time::Duration;

use nalgebra::Vector3;
use vtuber_core::arm_tracking::{
    ArmBlendWeight, ArmBlendWeights, ArmControlFrame, ArmLandmarks, ArmTrackingTarget,
    ArmTrackingTargets, PoseArmFrame, PoseWorldLandmark,
};
use vtuber_core::{FrameSeq, MonoTimeNs};

use crate::filter::damped::{
    DEFAULT_MAX_DT_SEC, DEFAULT_TIME_CONSTANT_SEC, critically_damped_step,
};

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
struct PointSmootherState {
    position: Vector3<f32>,
    velocity: Vector3<f32>,
}

impl PointSmootherState {
    fn new(value: [f32; 3]) -> Self {
        Self {
            position: vector(value),
            velocity: Vector3::zeros(),
        }
    }

    fn step(&mut self, target: [f32; 3], dt_sec: f32) {
        let error = vector(target) - self.position;
        let (correction, velocity) =
            critically_damped_step(error, self.velocity, dt_sec, DEFAULT_TIME_CONSTANT_SEC);
        self.position += correction;
        self.velocity = velocity;
    }
}

/// Render-clock critically damped state for one arm's wrist and bend plane.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ArmSmootherState {
    wrist: PointSmootherState,
    elbow: PointSmootherState,
}

impl ArmSmootherState {
    /// Seeds the smoother at the first adopted target; later ticks advance it.
    fn new(target: ArmTrackingTarget) -> Self {
        Self {
            wrist: PointSmootherState::new(target.wrist),
            elbow: PointSmootherState::new(target.elbow_pole),
        }
    }

    fn advance(&mut self, target: ArmTrackingTarget, dt_sec: f32) -> ArmTrackingTarget {
        let dt_sec = dt_sec.clamp(0.0, DEFAULT_MAX_DT_SEC);
        self.wrist.step(target.wrist, dt_sec);
        self.elbow.step(target.elbow_pole, dt_sec);
        ArmTrackingTarget {
            wrist: array(self.wrist.position),
            elbow_pole: array(self.elbow.position),
        }
    }
}

fn vector([x, y, z]: [f32; 3]) -> Vector3<f32> {
    Vector3::new(x, y, z)
}

fn array(value: Vector3<f32>) -> [f32; 3] {
    [value.x, value.y, value.z]
}

/// Per-side adoption and temporal policy for observed arms.
///
/// Validation belongs to the settings layer; these are research values from the
/// 5/5 video evaluation: a hand must return to the virtual arm slowly enough to
/// read as a relaxed drop rather than a snap, and a reacquired hand must not fly
/// to the observation. Reacquisition continues from the current authority, so
/// `acquire` is the time a fully abandoned hand takes to return to full
/// observation.
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
    /// How long a fully reacquired channel takes to blend back in.
    pub acquire: Duration,
    /// Largest accepted wrist displacement between two observations, in units
    /// of the calibrated arm length.
    ///
    /// A larger single-observation displacement is a detection teleport, not
    /// hand motion, so it is quarantined and the arm returns to the virtual
    /// arm instead of following it.
    pub max_wrist_step: f32,
}

impl Default for ArmTrackingProfile {
    fn default() -> Self {
        Self {
            shoulder_visibility: 0.5,
            wrist_visibility: 0.5,
            elbow_visibility: 0.5,
            hold: Duration::from_millis(150),
            return_to_virtual: Duration::from_secs(2),
            acquire: Duration::from_millis(500),
            max_wrist_step: 0.75,
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
///
/// A present channel rises toward full authority at the acquire rate. A lost
/// channel holds the authority it had when it was lost, then decays to zero
/// over the return time. Both directions continue from the current weight, so
/// losing a channel mid-acquire and reacquiring one mid-return never jump.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ChannelBlend {
    weight: f32,
    present: bool,
    lost_at: MonoTimeNs,
    loss_weight: f32,
}

impl ChannelBlend {
    const fn new() -> Self {
        Self {
            weight: 0.0,
            present: false,
            lost_at: MonoTimeNs(0),
            loss_weight: 0.0,
        }
    }

    fn advance(
        &mut self,
        now: MonoTimeNs,
        present: bool,
        render_dt_ns: Option<u64>,
        profile: &ArmTrackingProfile,
    ) {
        if present {
            self.present = true;
            let acquire_sec = profile.acquire.as_secs_f32().max(f32::EPSILON);
            let step = render_dt_ns.unwrap_or(0) as f32 * 1.0e-9 / acquire_sec;
            self.weight = (self.weight + step).clamp(0.0, 1.0);
        } else {
            if self.present {
                self.present = false;
                self.lost_at = now;
                self.loss_weight = self.weight;
            }
            let elapsed_ms = now.0.saturating_sub(self.lost_at.0) as f32 * 1.0e-6;
            let hold_ms = profile.hold.as_secs_f32() * 1.0e3;
            let return_ms = profile.return_to_virtual.as_secs_f32() * 1.0e3;
            let factor = if elapsed_ms <= hold_ms {
                1.0
            } else if return_ms > 0.0 && elapsed_ms <= hold_ms + return_ms {
                (1.0 - (elapsed_ms - hold_ms) / return_ms).clamp(0.0, 1.0)
            } else {
                0.0
            };
            self.weight = self.loss_weight * factor;
        }
    }
}

/// Which observed channels one intake contained.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ChannelPresence {
    wrist: bool,
    elbow: bool,
}

impl ChannelPresence {
    const NONE: Self = Self {
        wrist: false,
        elbow: false,
    };
}

/// Per-side observation, calibration, smoothing, and blend state.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ArmSideState {
    calibration: ArmCalibrationState,
    source: Option<ArmTrackingTarget>,
    smoother: Option<ArmSmootherState>,
    output: Option<ArmTrackingTarget>,
    last_pole: Option<[f32; 3]>,
    /// Channel presence of the latest consumed frame, reused on held ticks.
    presence: ChannelPresence,
    /// A real wrist gap happened since the last adopted target, so the next
    /// usable observation is a reacquisition rather than a candidate teleport.
    wrist_lost: bool,
    wrist_blend: ChannelBlend,
    pole_blend: ChannelBlend,
    pole_factor: f32,
}

impl ArmSideState {
    const fn empty() -> Self {
        Self {
            calibration: ArmCalibrationState::new(),
            source: None,
            smoother: None,
            output: None,
            last_pole: None,
            presence: ChannelPresence::NONE,
            wrist_lost: false,
            wrist_blend: ChannelBlend::new(),
            pole_blend: ChannelBlend::new(),
            pole_factor: 0.0,
        }
    }

    fn weights(&self) -> ArmBlendWeight {
        ArmBlendWeight {
            wrist: self.wrist_blend.weight,
            pole: self.pole_blend.weight * self.pole_factor,
        }
    }

    /// Adopts one new observation and records which channels it contained.
    ///
    /// An occluded elbow keeps the previous bend plane instead of freezing the
    /// wrist or injecting a fabricated one. A wrist that moved farther than the
    /// profile allows in one observation is quarantined as a detection
    /// teleport; the first usable observation after a real loss is instead
    /// accepted as a reacquisition.
    fn consume(&mut self, arm: Option<&ArmLandmarks>, profile: &ArmTrackingProfile) {
        let quality = arm.map(|value| assess_arm_observation(value, profile));
        let wrist_usable = quality.is_some_and(ArmObservationQuality::follows_wrist);
        let elbow_usable = quality.is_some_and(ArmObservationQuality::uses_elbow);

        let sample = arm
            .filter(|_| elbow_usable)
            .and_then(|value| measure_arm_reference(*value));
        self.calibration = update_arm_calibration(&self.calibration, sample);

        let Some(reference) = self.calibration.confirmed_length() else {
            self.presence = ChannelPresence::NONE;
            return;
        };
        let Some(arm) = arm.filter(|_| wrist_usable) else {
            self.wrist_lost = true;
            self.presence = ChannelPresence::NONE;
            return;
        };

        let target = retarget_arm_landmarks(*arm, reference);
        let teleported = !self.wrist_lost
            && self.source.is_some_and(|previous| {
                (vector(target.wrist) - vector(previous.wrist)).norm() > profile.max_wrist_step
            });
        if teleported {
            self.presence = ChannelPresence::NONE;
            return;
        }
        self.wrist_lost = false;

        let mut target = target;
        let mut elbow_tracked = false;
        if elbow_usable {
            let update = stabilize_elbow_pole(self.last_pole, target, 1.0);
            self.pole_factor = update.observed_weight;
            if let Some(pole) = update.pole {
                target.elbow_pole = pole;
                self.last_pole = Some(pole);
                elbow_tracked = true;
            }
        }
        if !elbow_tracked && let Some(source) = self.source {
            target.elbow_pole = source.elbow_pole;
        }
        if self.smoother.is_none() {
            self.smoother = Some(ArmSmootherState::new(target));
            self.output = Some(target);
        }
        self.source = Some(target);
        self.presence = ChannelPresence {
            wrist: true,
            elbow: elbow_tracked,
        };
    }

    /// Advances the observed blends and the render-clock smoothing by one tick.
    ///
    /// The latest consumed presence is reused so a retained observation keeps
    /// feeding the same channel state on every render tick, exactly like the
    /// face pipeline's held sample. Only a completed no-person result or a
    /// quarantined teleport advances the loss timeline.
    fn advance(
        &mut self,
        now: MonoTimeNs,
        render_dt_ns: Option<u64>,
        profile: &ArmTrackingProfile,
    ) {
        self.wrist_blend
            .advance(now, self.presence.wrist, render_dt_ns, profile);
        self.pole_blend
            .advance(now, self.presence.elbow, render_dt_ns, profile);
        if let (Some(source), Some(smoother)) = (self.source, self.smoother.as_mut()) {
            let dt_sec = render_dt_ns.unwrap_or(0) as f32 * 1.0e-9;
            self.output = Some(smoother.advance(source, dt_sec));
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
}

/// Pure temporal state for observed arm tracking. It owns no clock or ECS.
///
/// `now` is supplied by the caller on every render tick while the latest
/// observation is re-fed. A new source sequence/capture time is adopted once;
/// retargeting, calibration, and the observed blends advance with it. A result
/// with `observation: None` is a completed "no person" inference and starts the
/// hold/return timeline.
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

/// Advances arm tracking by one render tick and emits a control frame.
///
/// The caller re-feeds the retained observation on every tick, exactly like the
/// face pipeline, so the arm smoother and the observed blends advance on the
/// render clock. A stale or duplicate source sequence/capture time is not
/// re-adopted, but it keeps its channels present: only a completed no-person
/// result or a quarantined teleport starts the hold/return timeline. When
/// `observation` is `None` the state only advances its display blend and no
/// frame is produced, so no fictitious source sequence is emitted.
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

    let is_new = state.last_consumed.is_none_or(|(seq, captured_at)| {
        frame.source_seq.0 > seq.0 && frame.captured_at.0 > captured_at.0
    });
    if is_new {
        state.last_consumed = Some((frame.source_seq, frame.captured_at));
        let (left, right) = match &frame.observation {
            Some(value) => (Some(&value.left), Some(&value.right)),
            None => (None, None),
        };
        state.left.consume(left, profile);
        state.right.consume(right, profile);
    }
    state.left.advance(now, render_dt_ns, profile);
    state.right.advance(now, render_dt_ns, profile);

    let Some((source_seq, captured_at)) = state.last_consumed else {
        return (state, None);
    };
    let control = ArmControlFrame {
        source_seq,
        captured_at,
        produced_at: now,
        targets: ArmTrackingTargets {
            left: state.left.output,
            right: state.right.output,
        },
        weights: ArmBlendWeights {
            left: state.left.weights(),
            right: state.right.weights(),
        },
    };
    (state, Some(control))
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

    const RENDER_STEP_SEC: f32 = 1.0 / 60.0;

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
    fn constant_input_converges_without_bias_at_render_rates() {
        for step_sec in [1.0 / 30.0, RENDER_STEP_SEC] {
            let target = target(0.3);
            let mut smoother = ArmSmootherState::new(target);
            let mut value = target;
            for _ in 0..120 {
                value = smoother.advance(target, step_sec);
            }
            assert_eq!(value, target);
        }
    }

    #[test]
    fn a_step_target_converges_monotonically_without_overshoot() {
        let mut smoother = ArmSmootherState::new(target(0.0));
        let mut previous = 0.0;
        for _ in 0..120 {
            let value = smoother.advance(target(1.0), RENDER_STEP_SEC);
            assert!(value.wrist[0] >= previous);
            assert!(value.wrist[0] <= 1.0);
            previous = value.wrist[0];
        }
        assert!(previous > 0.99);
    }

    #[test]
    fn stationary_jitter_is_reduced_and_calls_are_deterministic() {
        let mut smoother = ArmSmootherState::new(target(0.0));
        let mut squared = 0.0;
        for frame in 0..240 {
            let raw = target(if frame % 2 == 0 { 0.01 } else { -0.01 });
            let mut copy = smoother;
            let value = smoother.advance(raw, RENDER_STEP_SEC);
            assert_eq!(value, copy.advance(raw, RENDER_STEP_SEC));
            squared += value.wrist[0].powi(2);
        }
        assert!((squared / 240.0).sqrt() < 0.005);
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

    /// Feeds one new observation and advances the state.
    fn feed(
        state: &mut ArmTrackingState,
        arm: ArmLandmarks,
        seq: u64,
        now_ns: u64,
        profile: &ArmTrackingProfile,
    ) -> Option<ArmControlFrame> {
        let frame = pose_frame(seq, now_ns, Some(observation(arm, arm)));
        let (next, control) = step_arm_tracking(state, Some(&frame), MonoTimeNs(now_ns), profile);
        *state = next;
        control
    }

    /// Re-feeds the same retained observation for one render tick.
    fn feed_held(
        state: &mut ArmTrackingState,
        arm: ArmLandmarks,
        seq: u64,
        captured_ns: u64,
        now_ns: u64,
        profile: &ArmTrackingProfile,
    ) -> Option<ArmControlFrame> {
        let frame = pose_frame(seq, captured_ns, Some(observation(arm, arm)));
        let (next, control) = step_arm_tracking(state, Some(&frame), MonoTimeNs(now_ns), profile);
        *state = next;
        control
    }

    /// Tracks a steady arm until calibration and the blends are fully settled.
    fn tracked_state(arm: ArmLandmarks, profile: &ArmTrackingProfile) -> ArmTrackingState {
        let mut state = ArmTrackingState::new();
        for seq in 0..(ARM_CALIBRATION_CAPACITY as u64 + 30) {
            let now = seq * OBSERVATION_STEP_NS;
            let _ = feed(&mut state, arm, seq, now, profile);
        }
        state
    }

    fn hidden_arm(base: ArmLandmarks) -> ArmLandmarks {
        ArmLandmarks {
            shoulder: low_visibility(base.shoulder),
            elbow: low_visibility(base.elbow),
            wrist: low_visibility(base.wrist),
        }
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
        assert!(state.left.output.is_none());
        assert_eq!(state.left.weights().wrist, 0.0);
    }

    #[test]
    fn a_duplicate_capture_is_not_readopted_but_still_emits_a_frame() {
        let profile = ArmTrackingProfile::default();
        let arm = arm();
        let state = tracked_state(arm, &profile);
        let now = 10_000_000_000u64;
        let frame = pose_frame(500, now, Some(observation(arm, arm)));
        let (state, control) = step_arm_tracking(&state, Some(&frame), MonoTimeNs(now), &profile);
        assert!(control.is_some());
        let source = state.left.source;
        let (again, control) =
            step_arm_tracking(&state, Some(&frame), MonoTimeNs(now + 16_000_000), &profile);
        assert!(control.is_some());
        assert_eq!(again.left.source, source);
    }

    #[test]
    fn render_ticks_interpolate_toward_the_retained_observation() {
        let profile = ArmTrackingProfile::default();
        let base = arm();
        let mut state = tracked_state(base, &profile);
        let moved = arm_at([0.3, 0.2, 0.0], [0.5, -0.3, 0.0]);
        let captured = 2_000_000_000u64;
        let frame = pose_frame(500, captured, Some(observation(moved, moved)));
        let (next, _) = step_arm_tracking(&state, Some(&frame), MonoTimeNs(captured), &profile);
        state = next;
        let source = state.left.source.unwrap();
        let mut previous = state.left.output.unwrap();
        assert_ne!(vector(previous.wrist), vector(source.wrist));
        for step in 1..=30u64 {
            let now = captured + step * 16_666_667;
            let (next, control) =
                step_arm_tracking(&state, Some(&frame), MonoTimeNs(now), &profile);
            state = next;
            let target = control.unwrap().targets.left.unwrap();
            let error_before = (vector(previous.wrist) - vector(source.wrist)).norm();
            let error_after = (vector(target.wrist) - vector(source.wrist)).norm();
            assert!(error_after <= error_before);
            previous = target;
        }
        assert!((vector(previous.wrist) - vector(source.wrist)).norm() < 1.0e-3);
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

    /// Observations taken long enough after calibration for the blends to settle.
    const SETTLED_NS: u64 = 10_000_000_000;

    /// A loss window longer than hold plus return for the default profile.
    const FULL_RETURN_FRAMES: u64 = 80;

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
        for seq in 0..FULL_RETURN_FRAMES {
            let now = SETTLED_NS + seq * OBSERVATION_STEP_NS;
            let control = feed(&mut state, occluded, 100 + seq, now, &profile).unwrap();
            assert_eq!(control.weights.left.wrist, 1.0);
        }
        assert_eq!(state.left.weights().pole, 0.0);
        assert!(state.left.output.is_some());
    }

    #[test]
    fn losing_the_wrist_returns_to_virtual_within_a_finite_time() {
        let profile = ArmTrackingProfile::default();
        let base = arm();
        let mut state = tracked_state(base, &profile);
        let hidden = hidden_arm(base);
        for seq in 0..FULL_RETURN_FRAMES {
            let now = SETTLED_NS + seq * OBSERVATION_STEP_NS;
            let _ = feed(&mut state, hidden, 100 + seq, now, &profile);
        }
        assert_eq!(state.left.weights().wrist, 0.0);
        assert_eq!(state.left.weights().pole, 0.0);
        assert!(state.left.output.is_some());
    }

    #[test]
    fn losing_one_arm_does_not_stop_the_other() {
        let profile = ArmTrackingProfile::default();
        let base = arm();
        let mut state = tracked_state(base, &profile);
        let hidden = hidden_arm(base);
        for seq in 0..FULL_RETURN_FRAMES {
            let now = SETTLED_NS + seq * OBSERVATION_STEP_NS;
            let frame = pose_frame(seq + 100, now, Some(observation(hidden, base)));
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
        let hidden = hidden_arm(base);
        for seq in 0..FULL_RETURN_FRAMES {
            let now = SETTLED_NS + seq * OBSERVATION_STEP_NS;
            let _ = feed(&mut state, hidden, 100 + seq, now, &profile);
        }
        assert_eq!(state.left.weights().wrist, 0.0);

        let start = SETTLED_NS + FULL_RETURN_FRAMES * OBSERVATION_STEP_NS;
        let mut previous = 0.0;
        for seq in 0..25u64 {
            let now = start + seq * OBSERVATION_STEP_NS;
            let control = feed(&mut state, base, 200 + seq, now, &profile).unwrap();
            let weight = control.weights.left.wrist;
            assert!(weight >= previous);
            assert!(weight <= 1.0);
            previous = weight;
        }
        assert_eq!(previous, 1.0);
    }

    #[test]
    fn a_lost_wrist_returns_to_virtual_over_several_seconds() {
        let profile = ArmTrackingProfile::default();
        let base = arm();
        let mut state = tracked_state(base, &profile);
        let hidden = hidden_arm(base);
        let loss_at = SETTLED_NS;

        let mut observations = Vec::new();
        for seq in 0..FULL_RETURN_FRAMES {
            let now = loss_at + seq * OBSERVATION_STEP_NS;
            let control = feed(&mut state, hidden, 100 + seq, now, &profile).unwrap();
            observations.push((now, control.weights.left.wrist));
        }

        // The hold keeps full authority, then the return is still only part
        // way back a full second after the loss: no snap to the virtual arm.
        let (_, held) = observations[0];
        assert_eq!(held, 1.0);
        let at_one_second = observations
            .iter()
            .find(|(now, _)| *now >= loss_at + 1_150_000_000)
            .expect("one second is inside the recorded window")
            .1;
        assert!(
            at_one_second > 0.3 && at_one_second < 0.7,
            "return should still be in progress after one second, got {at_one_second}"
        );

        let mut previous = 1.0;
        for (now, weight) in &observations {
            assert!(
                *weight <= previous + 1.0e-6,
                "return must be monotonic: {weight} > {previous} at {now}"
            );
            previous = *weight;
        }
        assert_eq!(previous, 0.0);
    }

    #[test]
    fn held_observation_keeps_the_channel_present_during_reacquire() {
        let profile = ArmTrackingProfile::default();
        let base = arm();
        let mut state = tracked_state(base, &profile);
        let hidden = hidden_arm(base);
        for seq in 0..FULL_RETURN_FRAMES {
            let now = SETTLED_NS + seq * OBSERVATION_STEP_NS;
            let _ = feed(&mut state, hidden, 100 + seq, now, &profile);
        }
        assert_eq!(state.left.weights().wrist, 0.0);

        // One new camera frame, then the same retained observation is re-fed
        // at render rate. The held ticks must keep ramping instead of being
        // mistaken for absence and snapping the weight to full authority.
        let captured = SETTLED_NS + (FULL_RETURN_FRAMES + 1) * OBSERVATION_STEP_NS;
        let seq = 300;
        let control = feed(&mut state, base, seq, captured, &profile).unwrap();
        let first = control.weights.left.wrist;
        assert!(first > 0.0 && first < 0.25, "got {first}");

        let mut previous = first;
        for tick in 1..=10u64 {
            let now = captured + tick * 16_666_667;
            let control = feed_held(&mut state, base, seq, captured, now, &profile).unwrap();
            let weight = control.weights.left.wrist;
            assert!(
                weight > previous,
                "held ticks must keep ramping: {weight} <= {previous}"
            );
            assert!(weight < 1.0, "held ticks must not snap to full: {weight}");
            previous = weight;
        }
    }

    #[test]
    fn a_wrist_teleport_is_quarantined_and_returns_to_virtual() {
        let profile = ArmTrackingProfile::default();
        let base = arm();
        let mut state = tracked_state(base, &profile);
        let tracked_source = state.left.source.unwrap();

        // More than one calibrated arm length away in a single observation.
        let teleport = arm_at([0.3, 0.2, 0.0], [-0.5, -0.4, 0.0]);
        for seq in 0..FULL_RETURN_FRAMES {
            let now = SETTLED_NS + seq * OBSERVATION_STEP_NS;
            let _ = feed(&mut state, teleport, 100 + seq, now, &profile);
        }

        // The teleport never becomes a target, and because the wrist was
        // never really lost it never becomes a reacquisition either.
        assert_eq!(state.left.source, Some(tracked_source));
        assert_eq!(state.left.weights().wrist, 0.0);

        // The real hand reappears near the last accepted target and reconnects
        // from zero without a jump.
        let now = SETTLED_NS + FULL_RETURN_FRAMES * OBSERVATION_STEP_NS;
        let control = feed(&mut state, base, 300, now, &profile).unwrap();
        let weight = control.weights.left.wrist;
        assert!(weight > 0.0 && weight < 0.5, "got {weight}");
        assert_eq!(state.left.source, Some(tracked_source));
    }

    #[test]
    fn a_far_detection_after_a_real_loss_reacquires_smoothly() {
        let profile = ArmTrackingProfile::default();
        let base = arm();
        let mut state = tracked_state(base, &profile);
        let tracked_source = state.left.source.unwrap();
        let hidden = hidden_arm(base);

        // A real gap long enough for the return to be under way.
        for seq in 0..20u64 {
            let now = SETTLED_NS + seq * OBSERVATION_STEP_NS;
            let _ = feed(&mut state, hidden, 100 + seq, now, &profile);
        }
        assert!(state.left.weights().wrist < 1.0);

        // The hand is detected far from where it was lost. A reacquisition may
        // be anywhere, but it must blend in from the current authority.
        let far = arm_at([0.3, 0.2, 0.0], [-0.5, -0.4, 0.0]);
        let now = SETTLED_NS + 20 * OBSERVATION_STEP_NS;
        let control = feed(&mut state, far, 120, now, &profile).unwrap();
        let weight = control.weights.left.wrist;
        assert!(
            weight > 0.0 && weight < 1.0,
            "reacquisition must ramp, got {weight}"
        );
        assert_ne!(state.left.source, Some(tracked_source));
        near(
            state.left.source.unwrap().wrist,
            [-0.5 / 0.7, 0.4 / 0.7, 0.0],
        );
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
