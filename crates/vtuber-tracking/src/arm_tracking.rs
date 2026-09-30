//! Pure shoulder-relative retargeting and render-clock arm smoothing.
//!
//! Visibility, loss/recovery, and calibration sample selection are separate
//! policies. Smoothing runs on the render clock toward the latest retained
//! observation, so the control frame is a continuous signal at the consumer
//! frame rate exactly like the head rotation and translation filters.
//!
//! Loss handling follows the shared loss-blend ramp (`crate::loss_blend`): a
//! lost wrist keeps its last authority briefly, then hands authority back to
//! the avatar's virtual arm over the profile's return duration; a
//! reacquisition ramps the observed authority back in from wherever the
//! return had reached. A wrist that teleports farther than
//! [`ArmTrackingProfile::max_wrist_step`] in one observation is quarantined
//! like an outlier head sample, and the first observation after a real loss
//! is accepted as a reacquisition so a hand that moved while hidden can be
//! picked up again.
//!
//! Reacquisition is therefore delayed by exactly one thing: the adoption gate
//! waiting for consecutive good camera frames. Nothing else is tuned against it
//! — the channel's authority comes back on the shared acquire ramp, and the
//! smoother no longer stiffens after a loss, so a second delay on the same
//! timeline would only delay the return twice.

use nalgebra::Vector3;
use vtuber_core::arm_tracking::{
    ArmBlendWeight, ArmBlendWeights, ArmControlFrame, ArmLandmarks, ArmTrackingTarget,
    ArmTrackingTargets, HandFingerPose, PoseArmFrame, PoseWorldLandmark,
};
use vtuber_core::{FrameSeq, MonoTimeNs};

use crate::filter::damped::{
    DEFAULT_MAX_DT_SEC, critically_damped_step, critically_damped_step_scalar,
};
use crate::loss_blend::{LossBlend, LossBlendProfile};

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
        palm_normal: observed_palm_normal(arm),
        fingers: observed_finger_pose(arm),
    }
}

/// Palm-plane normal from the hand landmarks, in the canonical basis.
///
/// The index/pinky MCP cross product relative to the wrist points to the same
/// anatomical hand side as the avatar's rest-space index/little cross product,
/// so no per-side sign is applied and the mirror only reflects it. Degenerate
/// or missing landmarks yield `None` instead of a fabricated axis.
fn observed_palm_normal(arm: ArmLandmarks) -> Option<[f32; 3]> {
    let hand = arm.hand?;
    let wrist = vector(hand.landmarks.get(HAND_WRIST)?.meters);
    let index = finite_normalized(vector(hand.landmarks.get(HAND_INDEX_MCP)?.meters) - wrist)?;
    let pinky = finite_normalized(vector(hand.landmarks.get(HAND_PINKY_MCP)?.meters) - wrist)?;
    let normal = index.cross(&pinky);
    finite_normalized(Vector3::new(normal.x, -normal.y, -normal.z)).map(array)
}

/// Hand Landmarker landmarks spanning the palm plane: wrist, index MCP, pinky MCP.
const HAND_WRIST: usize = 0;
const HAND_INDEX_MCP: usize = 5;
const HAND_PINKY_MCP: usize = 17;

/// Hand Landmarker landmark triples for one four-finger chain, as
/// `[mcp, pip, dip, tip]`.
const FINGER_CHAINS: [[usize; 4]; 4] = [
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
];

/// Hand Landmarker landmarks of the thumb chain, as `[cmc, mcp, ip, tip]`.
const THUMB_CHAIN: [usize; 4] = [1, 2, 3, 4];

/// Flexion limit of each four-finger joint, in radians.
///
/// These are the anatomical ranges a real finger reaches (knuckle about 100
/// degrees, middle joint about 110, last about 60). A landmark triple beyond
/// them is not a finger pose this rig can show, and applying it would fold the
/// bone back through itself.
const FINGER_FLEXION_LIMIT_RAD: [f32; 3] = [1.75, 1.92, 1.05];

/// Flexion limit of the thumb's metacarpophalangeal and interphalangeal joints,
/// in radians. Smaller than a finger's because the thumb's axes are rotated.
const THUMB_FLEXION_LIMIT_RAD: [f32; 2] = [0.96, 1.22];

/// Extract signed bends and spread in the observed palm frame before smoothing.
/// A rigid hand rotation therefore cannot become finger articulation.
fn observed_finger_pose(arm: ArmLandmarks) -> Option<HandFingerPose> {
    let landmarks = arm.hand?.landmarks;
    let point = |index: usize| landmarks.get(index).map(|value| vector(value.meters));
    let wrist = point(HAND_WRIST)?;
    let index = finite_normalized(point(HAND_INDEX_MCP)? - wrist)?;
    let little = finite_normalized(point(HAND_PINKY_MCP)? - wrist)?;
    let normal = finite_normalized(index.cross(&little))?;
    let forward = finite_normalized(index + little)?;
    let across = forward.cross(&normal);
    let local = |ray: Vector3<f32>| [ray.dot(&across), ray.dot(&forward), ray.dot(&normal)];
    let bend = |incoming: Vector3<f32>, outgoing: Vector3<f32>, axis: Vector3<f32>, limit: f32| {
        let incoming = finite_normalized(incoming)?;
        let outgoing = finite_normalized(outgoing)?;
        let angle = axis
            .dot(&incoming.cross(&outgoing))
            .atan2(incoming.dot(&outgoing));
        angle.is_finite().then(|| angle.clamp(-limit, limit))
    };
    // Thumb motion is strongly oblique to the palm. Projecting the cross
    // product onto one guessed bend axis can therefore report almost zero for
    // a visibly closed thumb. Keep the anatomical sign from that axis, but use
    // the full 3D angle as the magnitude, as established MediaPipe hand solvers
    // do for the CMC-MCP-IP and MCP-IP-tip triples. Wrist-CMC-MCP is already
    // represented by `thumb_direction` and drives the VRM thumb metacarpal, so
    // it must not be applied a second time to the proximal joint.
    let thumb_bend = |incoming: Vector3<f32>, outgoing: Vector3<f32>, limit: f32| {
        let incoming = finite_normalized(incoming)?;
        let outgoing = finite_normalized(outgoing)?;
        let cross = incoming.cross(&outgoing);
        let magnitude = cross.norm().atan2(incoming.dot(&outgoing));
        let axis = finite_normalized(incoming.cross(&normal))?;
        let sign = if axis.dot(&cross) < 0.0 { -1.0 } else { 1.0 };
        magnitude
            .is_finite()
            .then(|| (magnitude * sign).clamp(-limit, limit))
    };
    let mut fingers = [[0.0; 3]; 4];
    let mut spread = [0.0; 4];
    for ((angles, opening), &[mcp, pip, dip, tip]) in
        fingers.iter_mut().zip(&mut spread).zip(&FINGER_CHAINS)
    {
        let proximal = finite_normalized(point(pip)? - point(mcp)?)?;
        let [x, y, z] = local(proximal);
        let [mcp_limit, pip_limit, dip_limit] = FINGER_FLEXION_LIMIT_RAD;
        // A knuckle can flex past 90 degrees. Keep its in-plane opening in
        // the forward half-plane; a backward projection then means flexion
        // past 90, not a 180-degree change in spread or a reversed PIP axis.
        let facing = if y < 0.0 { -1.0 } else { 1.0 };
        let axis = finite_normalized((proximal * facing).cross(&normal))?;
        *opening = (facing * x).atan2(facing * y);
        *angles = [
            z.atan2(facing * x.hypot(y)).clamp(-mcp_limit, mcp_limit),
            bend(proximal, point(dip)? - point(pip)?, axis, pip_limit)?,
            bend(
                point(dip)? - point(pip)?,
                point(tip)? - point(dip)?,
                axis,
                dip_limit,
            )?,
        ];
    }
    let [cmc, mcp, ip, tip] = THUMB_CHAIN;
    let mcp_ray = point(mcp)? - point(cmc)?;
    let ip_ray = point(ip)? - point(mcp)?;
    let tip_ray = point(tip)? - point(ip)?;
    let ray = finite_normalized(mcp_ray)?;
    let [mcp_limit, ip_limit] = THUMB_FLEXION_LIMIT_RAD;
    Some(HandFingerPose {
        fingers,
        spread,
        thumb: [
            thumb_bend(mcp_ray, ip_ray, mcp_limit)?,
            thumb_bend(ip_ray, tip_ray, ip_limit)?,
        ],
        thumb_direction: local(ray),
    })
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

    fn step(&mut self, target: [f32; 3], dt_sec: f32, time_constant_sec: f32) {
        let error = vector(target) - self.position;
        let (correction, velocity) =
            critically_damped_step(error, self.velocity, dt_sec, time_constant_sec);
        self.position += correction;
        self.velocity = velocity;
    }
}

/// Render-clock critically damped state for one observed unit direction.
///
/// The direction is followed as a rotation rather than as a raw 3-vector. A
/// palm flip sends the observation to the opposite side of the sphere, and
/// filtering the vector would drive the state toward the origin: normalizing
/// a nearly zero vector picks an arbitrary direction, so the palm could swing
/// through a side the hand never showed. Here the error is the shortest
/// rotation between the current and observed directions, so an opposed pair
/// turns about a stable perpendicular and arrives from the side the hand
/// actually moved through. This is the same treatment the head rotation filter
/// applies in tangent space; only the shared step is reused, not the head's
/// single-frame angle limit, which has no meaning for a palm normal.
#[derive(Clone, Copy, Debug, PartialEq)]
struct DirectionSmootherState {
    value: Vector3<f32>,
    /// Retained angular derivative in the tangent plane, as a rotation vector.
    velocity: Vector3<f32>,
}

impl DirectionSmootherState {
    /// `value` must be a unit vector, so every rotation starts from a point on
    /// the sphere.
    fn new(value: Vector3<f32>) -> Self {
        Self {
            value,
            velocity: Vector3::zeros(),
        }
    }

    /// Advances one render tick. An unusable observation holds the direction.
    fn step(&mut self, target: [f32; 3], dt_sec: f32, time_constant_sec: f32) {
        let Some(target) = finite_normalized(vector(target)) else {
            return;
        };
        let Some(rotation) = shortest_arc_rotation(self.value, target) else {
            return;
        };
        let (correction, velocity) =
            critically_damped_step(rotation, self.velocity, dt_sec, time_constant_sec);
        if let Some(value) = finite_normalized(rotate(self.value, correction)) {
            self.value = value;
            self.velocity = velocity;
        }
    }

    /// The smoothed direction, or `None` if it drifted off the unit sphere.
    fn normalized(self) -> Option<[f32; 3]> {
        finite_normalized(self.value).map(array)
    }
}

/// Normalizes a candidate direction, returning `None` when it cannot be used.
fn unit_direction(value: [f32; 3]) -> Option<Vector3<f32>> {
    finite_normalized(vector(value))
}

/// Rotation vector of the shortest arc carrying one unit direction onto another.
///
/// Both inputs must be unit vectors. An exactly opposed pair leaves the axis
/// free, so a stable perpendicular of `from` is used instead of leaving the
/// rotation undefined; a `None` axis means the target is not usable.
fn shortest_arc_rotation(from: Vector3<f32>, to: Vector3<f32>) -> Option<Vector3<f32>> {
    let axis = from.cross(&to);
    let sine = axis.norm();
    let cosine = from.dot(&to);
    if sine <= f32::EPSILON {
        return (cosine >= 0.0)
            .then_some(Vector3::zeros())
            .or_else(|| stable_perpendicular(from).map(|axis| axis * std::f32::consts::PI));
    }
    Some(axis * (sine.atan2(cosine) / sine))
}

/// Rotates a vector by the rotation vector `rotation` (axis scaled by angle).
///
/// Rodrigues' formula, so a zero rotation is the identity without a degenerate
/// division. The caller renormalizes the result, which is what the rotation
/// about a unit axis already guarantees up to rounding.
fn rotate(value: Vector3<f32>, rotation: Vector3<f32>) -> Vector3<f32> {
    let angle = rotation.norm();
    if angle <= f32::EPSILON {
        return value;
    }
    let axis = rotation / angle;
    let (sine, cosine) = angle.sin_cos();
    value * cosine + axis.cross(&value) * sine + axis * axis.dot(&value) * (1.0 - cosine)
}

/// A unit vector perpendicular to `value`, chosen from the world axis least
/// aligned with it so the result does not flip between ticks.
fn stable_perpendicular(value: Vector3<f32>) -> Option<Vector3<f32>> {
    let seed = if value.x.abs() <= value.y.abs() && value.x.abs() <= value.z.abs() {
        Vector3::x()
    } else if value.y.abs() <= value.z.abs() {
        Vector3::y()
    } else {
        Vector3::z()
    };
    finite_normalized(value.cross(&seed))
}

/// Time constant of the observed wrist and elbow positions, in seconds.
///
/// An observed hand that is genuinely moving must reach the avatar without the
/// arm-length lag a slower filter would add, so this is a fixed response and
/// not one that stretches with the observation's speed: a fast hand is motion,
/// not an outlier. A detection teleport is rejected outright by
/// [`ArmTrackingProfile::max_wrist_step`] instead, and a real loss is eased back
/// by the shared loss blend.
const ARM_POSITION_TIME_CONSTANT_SEC: f32 = 0.05;

/// Time constant of the observed palm plane, in seconds.
///
/// The palm normal comes from a cross product of two hand keypoints and is the
/// noisiest observed channel, so it keeps a longer constant than the wrist
/// rather than the same one. It is deliberately only twice the wrist constant:
/// the previous 0.30 s left the orientation trailing the hand position by three
/// times as long, which read as the palm lagging behind the wrist rather than
/// as a smoother plane.
const ARM_PALM_TIME_CONSTANT_SEC: f32 = 0.10;

/// Time constant of the observed finger joints, in seconds.
///
/// Deliberately the shortest of the observed channels: a finger curl is small
/// and its endpoints are what a viewer reads, so a long response would smear a
/// fist open instead of merely adding lag. The per-joint angles are already
/// single observations of a joint angle rather than a position, so almost no
/// noise suppression is owed here.
const ARM_FINGER_TIME_CONSTANT_SEC: f32 = 0.04;

/// Render-clock critically damped state for one scalar channel.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ScalarSmootherState {
    value: f32,
    velocity: f32,
}

impl ScalarSmootherState {
    /// Advances one render tick. A non-finite observation holds the value.
    fn step(&mut self, target: f32, dt_sec: f32, time_constant_sec: f32) {
        if !target.is_finite() {
            return;
        }
        let (correction, velocity) = critically_damped_step_scalar(
            target - self.value,
            self.velocity,
            dt_sec,
            time_constant_sec,
        );
        self.value += correction;
        self.velocity = velocity;
    }
}

/// Render-clock critically damped state for one observed hand's finger joints.
#[derive(Clone, Copy, Debug, PartialEq)]
struct FingerSmootherState {
    /// Index, middle, ring, and little flexion as `(mcp, pip, dip)` radians.
    fingers: [PointSmootherState; 4],
    /// In-plane opening of each proximal segment.
    spread: [ScalarSmootherState; 4],
    /// Thumb flexion as `(mcp, ip)` radians.
    thumb: [ScalarSmootherState; 2],
    /// The thumb ray, followed as a direction for the same reason the palm
    /// normal is.
    direction: DirectionSmootherState,
}

impl FingerSmootherState {
    /// Seeds at the first adopted observation; later ticks advance it.
    fn new(pose: HandFingerPose) -> Self {
        Self {
            fingers: pose.fingers.map(PointSmootherState::new),
            spread: pose.spread.map(|value| ScalarSmootherState {
                value,
                velocity: 0.0,
            }),
            thumb: pose.thumb.map(|value| ScalarSmootherState {
                value,
                velocity: 0.0,
            }),
            // observed_finger_pose supplies a unit ray in the local palm frame.
            direction: DirectionSmootherState::new(vector(pose.thumb_direction)),
        }
    }

    fn step(&mut self, pose: HandFingerPose, dt_sec: f32) {
        for (smoother, curl) in self.fingers.iter_mut().zip(pose.fingers) {
            if curl.iter().all(|value| value.is_finite()) {
                smoother.step(curl, dt_sec, ARM_FINGER_TIME_CONSTANT_SEC);
            }
        }
        for (smoother, angle) in self.spread.iter_mut().zip(pose.spread) {
            smoother.step(angle, dt_sec, ARM_FINGER_TIME_CONSTANT_SEC);
        }
        for (smoother, angle) in self.thumb.iter_mut().zip(pose.thumb) {
            smoother.step(angle, dt_sec, ARM_FINGER_TIME_CONSTANT_SEC);
        }
        self.direction
            .step(pose.thumb_direction, dt_sec, ARM_FINGER_TIME_CONSTANT_SEC);
    }

    /// The smoothed articulation, or `None` if the thumb ray drifted off the
    /// unit sphere. A lost observation holds the last values, exactly like the
    /// palm channel holds its normal while its weight decays.
    fn output(self) -> Option<HandFingerPose> {
        Some(HandFingerPose {
            fingers: self.fingers.map(|smoother| array(smoother.position)),
            spread: self.spread.map(|smoother| smoother.value),
            thumb: self.thumb.map(|smoother| smoother.value),
            thumb_direction: self.direction.normalized()?,
        })
    }
}

/// Visibility at or above which an arm observation counts as good.
const ARM_ENTER_VISIBILITY: f32 = 0.7;
/// Visibility at or below which an arm observation counts as bad.
const ARM_EXIT_VISIBILITY: f32 = 0.4;
/// Consecutive good observations required before an arm is adopted for the
/// first time.
const ARM_GOOD_FRAMES: u32 = 4;
/// Consecutive good observations required before a previously adopted arm comes
/// back.
///
/// A cold start has to survive the calibration window, so it waits the longer
/// count. An arm that was already tracked has proved the setup works, and the
/// only thing left to guard against is a single spurious detection, which two
/// consecutive good observations already reject. Returning sooner matters
/// because the count is in camera frames: Pose and the Hand Landmarker share one
/// worker, so a frame is tens of milliseconds, and the same wait was added on
/// top of the shared `LossBlend` acquire ramp that already eases the authority
/// back over a second. Nothing here snaps, because that ramp is unchanged.
const ARM_REACQUIRE_GOOD_FRAMES: u32 = 2;
/// Consecutive bad observations required before an adopted arm is lost.
const ARM_BAD_FRAMES: u32 = 6;

/// Consecutive observation counts for one arm's adoption state.
///
/// Real visibility hovers around any single threshold, so a one-frame decision
/// makes a lost arm reacquire every other frame and the return and acquire
/// ramps fight instead of returning. A cold start needs [`ARM_GOOD_FRAMES`]
/// consecutive good observations, a return needs [`ARM_REACQUIRE_GOOD_FRAMES`],
/// and losing needs [`ARM_BAD_FRAMES`] consecutive bad ones; a score inside the
/// hysteresis band holds the current state. Missing scores count as bad rather
/// than being invented.
///
/// "Good" is the weakest of the Pose shoulder and wrist visibility scores, plus
/// the Hand Landmarker having detected a hand for that wrist at all. The
/// Hand Landmarker reports no landmark quality, so presence is the only honest
/// hand signal: its Left/Right confidence describes the label it derived from
/// the hand's shape, not how well the coordinates were localized, and must not
/// stand in for one. Requiring a detection still keeps a Pose arm without its
/// own visible hand — the usual hallucination for a hidden limb — from ever
/// being adopted, and an adopted arm loses authority a few frames after its
/// hand stops being seen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ArmAdoptionGate {
    adopted: bool,
    /// The arm was adopted at least once, so the next return is a reacquisition
    /// rather than a cold start.
    returned: bool,
    good: u32,
    bad: u32,
}

impl ArmAdoptionGate {
    const fn new() -> Self {
        Self {
            adopted: false,
            returned: false,
            good: 0,
            bad: 0,
        }
    }

    /// Feeds one observation's Pose visibility and whether a hand was detected.
    fn update(&mut self, score: Option<f32>, hand_detected: bool) -> bool {
        let score = score.unwrap_or(0.0);
        let needed = if self.returned {
            ARM_REACQUIRE_GOOD_FRAMES
        } else {
            ARM_GOOD_FRAMES
        };
        if score >= ARM_ENTER_VISIBILITY && hand_detected {
            self.good = self.good.saturating_add(1).min(needed);
            self.bad = 0;
            if self.good >= needed {
                self.adopted = true;
            }
        } else if score <= ARM_EXIT_VISIBILITY || !hand_detected {
            self.bad = self.bad.saturating_add(1).min(ARM_BAD_FRAMES);
            self.good = 0;
            if self.bad >= ARM_BAD_FRAMES {
                // Only an arm that was actually tracked counts as a return; an
                // arm that never came up is still waiting for its cold start.
                self.returned |= self.adopted;
                self.adopted = false;
            }
        }
        self.adopted
    }
}

/// Largest angular speed of the observed elbow's bend plane, in radians per
/// second.
///
/// A human elbow does not swing around the shoulder-wrist axis faster than
/// this, so a plane that turns faster is an artifact of a moving projection
/// axis or a flip in the pole reference, not arm motion. The wrist and reach
/// are untouched: this only caps the direction the elbow bends toward, which is
/// what reads as the elbow popping upward.
const ELBOW_PLANE_MAX_RATE_RAD_PER_SEC: f32 = 3.0;

/// Render-clock critically damped state for one arm's wrist, bend plane, and
/// palm plane.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ArmSmootherState {
    wrist: PointSmootherState,
    elbow: PointSmootherState,
    palm: Option<DirectionSmootherState>,
    fingers: Option<FingerSmootherState>,
    /// Bend-plane direction emitted on the previous tick, used to rate-limit
    /// the next one.
    last_plane: Option<[f32; 3]>,
}

impl ArmSmootherState {
    /// Seeds the smoother at the first adopted target; later ticks advance it.
    fn new(target: ArmTrackingTarget) -> Self {
        Self {
            wrist: PointSmootherState::new(target.wrist),
            elbow: PointSmootherState::new(target.elbow_pole),
            palm: target
                .palm_normal
                .and_then(unit_direction)
                .map(DirectionSmootherState::new),
            fingers: target.fingers.map(FingerSmootherState::new),
            last_plane: None,
        }
    }

    fn advance(&mut self, target: ArmTrackingTarget, dt_sec: f32) -> ArmTrackingTarget {
        let dt_sec = dt_sec.clamp(0.0, DEFAULT_MAX_DT_SEC);
        self.wrist
            .step(target.wrist, dt_sec, ARM_POSITION_TIME_CONSTANT_SEC);
        self.elbow
            .step(target.elbow_pole, dt_sec, ARM_POSITION_TIME_CONSTANT_SEC);
        if self.palm.is_none() {
            self.palm = target
                .palm_normal
                .and_then(unit_direction)
                .map(DirectionSmootherState::new);
        }
        if let (Some(palm), Some(normal)) = (self.palm.as_mut(), target.palm_normal) {
            palm.step(normal, dt_sec, ARM_PALM_TIME_CONSTANT_SEC);
        }
        if self.fingers.is_none() {
            self.fingers = target.fingers.map(FingerSmootherState::new);
        }
        if let (Some(fingers), Some(pose)) = (self.fingers.as_mut(), target.fingers) {
            fingers.step(pose, dt_sec);
        }
        let plane = limit_plane_rotation(
            self.last_plane,
            self.elbow.position,
            self.wrist.position,
            dt_sec,
        );
        self.last_plane = Some(array(plane));
        // A lost palm or finger observation holds the last smoothed orientation
        // while the blend decays, exactly like a held elbow pole.
        let palm_normal = self.palm.and_then(DirectionSmootherState::normalized);
        let fingers = self.fingers.and_then(FingerSmootherState::output);
        ArmTrackingTarget {
            wrist: array(self.wrist.position),
            elbow_pole: array(plane),
            palm_normal,
            fingers,
        }
    }
}

/// Caps how fast the bend plane may rotate around the shoulder-wrist axis.
///
/// Only the component perpendicular to the wrist axis is limited; the axial
/// component (how far the elbow sits along the arm) is carried through, so the
/// reach and the wrist target are exactly the smoothed observation's. The
/// first tick and any degenerate axis pass the desired plane through.
fn limit_plane_rotation(
    previous: Option<[f32; 3]>,
    desired: Vector3<f32>,
    wrist: Vector3<f32>,
    dt_sec: f32,
) -> Vector3<f32> {
    let Some(previous) = previous.map(vector) else {
        return desired;
    };
    let Some(axis) = finite_normalized(wrist) else {
        return desired;
    };
    let previous_perp = perpendicular(previous, axis);
    let desired_perp = perpendicular(desired, axis);
    let (Some(previous_dir), Some(desired_dir)) = (
        finite_normalized(previous_perp),
        finite_normalized(desired_perp),
    ) else {
        return desired;
    };
    let angle = axis
        .dot(&previous_dir.cross(&desired_dir))
        .atan2(previous_dir.dot(&desired_dir));
    let max_step = ELBOW_PLANE_MAX_RATE_RAD_PER_SEC * dt_sec;
    if !angle.is_finite() || angle.abs() <= max_step {
        return desired;
    }
    let step = max_step.copysign(angle);
    let rotated = previous_dir * step.cos() + axis.cross(&previous_dir) * step.sin();
    let radius = desired_perp.norm();
    rotated.normalize() * radius + axis * desired.dot(&axis)
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
/// [`LossBlendProfile::acquire`] is the time a fully abandoned hand takes to
/// return to full observation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmTrackingProfile {
    /// Minimum shoulder visibility to use the arm at all.
    pub shoulder_visibility: f32,
    /// Minimum wrist visibility to follow the hand.
    pub wrist_visibility: f32,
    /// Minimum elbow visibility to use the observed bend plane.
    pub elbow_visibility: f32,
    /// Unified hold / return / acquire timing shared with the face pipeline.
    pub blend: LossBlendProfile,
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
            blend: LossBlendProfile::default(),
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
    /// A hand was detected for this wrist, so the palm plane can be observed.
    ///
    /// The Hand Landmarker reports no landmark quality, so this is the plain
    /// presence of a detection. Whether those landmarks span a plane is decided
    /// later, when the normal is actually derived.
    pub palm: bool,
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

/// The visibility MediaPipe reported for one landmark, preferring visibility
/// over presence when both exist. `None` is kept as missing, never invented.
fn landmark_score(point: PoseWorldLandmark) -> Option<f32> {
    point.visibility.or(point.presence)
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
        landmark_score(point).is_some_and(|score| score >= threshold)
    };
    ArmObservationQuality {
        shoulder: adopt(arm.shoulder, profile.shoulder_visibility),
        wrist: adopt(arm.wrist, profile.wrist_visibility),
        elbow: adopt(arm.elbow, profile.elbow_visibility),
        palm: arm.hand.is_some(),
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

/// Extension ratio at which the observed pole stops following the observation.
const POLE_EXTENSION_BAND: f32 = 0.08;

/// Observed-plane weight from the shoulder-to-wrist extension ratio.
///
/// `1.0` while the arm is bent enough for the bend side to be well-conditioned,
/// falling to `0.0` at full extension, where the elbow lies on the axis.
fn extension_falloff(extension: f32) -> f32 {
    if extension >= 1.0 {
        0.0
    } else if extension <= 1.0 - POLE_EXTENSION_BAND {
        1.0
    } else {
        (1.0 - extension) / POLE_EXTENSION_BAND
    }
}

/// Keeps the elbow pole continuous and in the plane perpendicular to the arm.
///
/// The observed elbow is projected onto the shoulder-to-wrist axis and the
/// previous plane is rotated toward it around that axis with an extension
/// falloff: as the arm approaches full extension the bend side is
/// ill-conditioned, so the plane holds its last well-conditioned direction and
/// only resumes following the observation once the arm bends again. The
/// falloff damps the plane's rotation; it never drops the channel's authority,
/// so an extending arm cannot snap the elbow to the virtual pole. Rotating
/// around the arm axis (rather than linearly blending the two directions) lets
/// an opposed observation cross to the observation's side instead of being
/// pinned on the wrong side or interpolated through zero; the render clock's
/// plane rate limit keeps that crossing smooth. An undefined plane returns
/// `None` instead of a fabricated world axis.
#[must_use]
pub fn stabilize_elbow_pole(
    previous: Option<[f32; 3]>,
    target: ArmTrackingTarget,
) -> Option<[f32; 3]> {
    let wrist = vector(target.wrist);
    let axial = wrist.norm();
    if !axial.is_finite() || axial <= f32::EPSILON {
        return None;
    }
    let axis = wrist / axial;
    let weight = extension_falloff(axial);

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
            let direction = rotate_plane_toward(prior, observed, axis, weight);
            let length = if observed_length > f32::EPSILON {
                observed_length
            } else {
                previous_length.max(f32::EPSILON)
            };
            Some(array(direction * length))
        }
        (Some(observed), None) => Some(array(observed * observed_length.max(f32::EPSILON))),
        (None, Some(prior)) => Some(array(prior * previous_length.max(f32::EPSILON))),
        (None, None) => None,
    }
}

/// Rotates a unit plane direction toward another around `axis` by `amount` of
/// the angle between them.
///
/// Both inputs are expected to lie in the plane perpendicular to `axis`. The
/// rotation is the short way around the axis, so an opposed pair (angle past
/// 90 degrees) crosses to the other side instead of passing through the axis
/// origin. A zero rotation keeps `from`; a non-normalizable result falls back
/// to `to`.
fn rotate_plane_toward(
    from: Vector3<f32>,
    to: Vector3<f32>,
    axis: Vector3<f32>,
    amount: f32,
) -> Vector3<f32> {
    let angle = axis.dot(&from.cross(&to)).atan2(from.dot(&to));
    let step = angle * amount.clamp(0.0, 1.0);
    if !step.is_finite() || step.abs() <= f32::EPSILON {
        return from;
    }
    let rotated = from * step.cos() + axis.cross(&from) * step.sin();
    finite_normalized(rotated).unwrap_or(to)
}

/// Which observed channels one intake contained.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ChannelPresence {
    wrist: bool,
    elbow: bool,
    palm: bool,
    fingers: bool,
}

impl ChannelPresence {
    const NONE: Self = Self {
        wrist: false,
        elbow: false,
        palm: false,
        fingers: false,
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
    /// Hysteresis on the arm's shoulder/wrist visibility, so a score that
    /// hovers around a single threshold cannot ping-pong the whole arm.
    adoption: ArmAdoptionGate,
    wrist_blend: LossBlend,
    pole_blend: LossBlend,
    palm_blend: LossBlend,
    fingers_blend: LossBlend,
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
            adoption: ArmAdoptionGate::new(),
            wrist_blend: LossBlend::new(),
            pole_blend: LossBlend::new(),
            palm_blend: LossBlend::new(),
            fingers_blend: LossBlend::new(),
        }
    }

    fn weights(&self) -> ArmBlendWeight {
        ArmBlendWeight {
            wrist: self.wrist_blend.weight(),
            pole: self.pole_blend.weight(),
            palm: self.palm_blend.weight(),
            fingers: self.fingers_blend.weight(),
        }
    }

    /// Adopts one new observation and records which channels it contained.
    ///
    /// The arm is adopted only when the Pose chain is visible and a Hand
    /// Landmarker detection agrees: Pose keeps emitting a plausible arm for a
    /// limb it cannot see, usually mirrored onto the visible one, so a
    /// "visible" Pose arm without its own hand is not tracked. An occluded
    /// elbow keeps the previous bend plane instead of freezing the wrist or
    /// injecting a fabricated one. A wrist that moved farther than the profile
    /// allows in one observation is quarantined as a detection teleport; the
    /// first usable observation after a real loss is instead accepted as a
    /// reacquisition.
    fn consume(&mut self, arm: Option<&ArmLandmarks>, profile: &ArmTrackingProfile) {
        let quality = arm.map(|value| assess_arm_observation(value, profile));
        // Only the Pose joint visibilities rank how well located the arm is.
        // The Hand Landmarker adds presence, not a score: its handedness
        // confidence describes the left/right label, so folding it in here would
        // let an ambiguous label drop an otherwise well-observed arm.
        let adoption_score = arm.and_then(|value| {
            let shoulder = landmark_score(value.shoulder)?;
            let wrist = landmark_score(value.wrist)?;
            Some(shoulder.min(wrist))
        });
        let hand_detected = quality.is_some_and(|value| value.palm);
        let wrist_usable = self.adoption.update(adoption_score, hand_detected);
        let elbow_usable = wrist_usable && quality.is_some_and(|value| value.elbow);
        // The palm and the fingers come from the same hand detection, so they
        // share one gate. What each channel then observes is decided by the
        // landmarks, not by the gate: a hand whose palm plane is degenerate can
        // still have perfectly readable finger joints.
        let hand_usable = wrist_usable && hand_detected;

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
        let palm_observed = hand_usable && target.palm_normal.is_some();
        let fingers_observed = hand_usable && target.fingers.is_some();
        if !hand_usable {
            target.palm_normal = None;
            target.fingers = None;
        }
        let mut elbow_tracked = false;
        if elbow_usable && let Some(pole) = stabilize_elbow_pole(self.last_pole, target) {
            target.elbow_pole = pole;
            self.last_pole = Some(pole);
            elbow_tracked = true;
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
            palm: palm_observed,
            fingers: fingers_observed,
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
            .advance(now, self.presence.wrist, &profile.blend);
        self.pole_blend
            .advance(now, self.presence.elbow, &profile.blend);
        self.palm_blend
            .advance(now, self.presence.palm, &profile.blend);
        self.fingers_blend
            .advance(now, self.presence.fingers, &profile.blend);
        if let (Some(source), Some(smoother)) = (self.source, self.smoother.as_mut()) {
            let dt_sec = render_dt_ns.unwrap_or(0) as f32 * 1.0e-9;
            self.output = Some(smoother.advance(source, dt_sec));
        }
    }

    fn advance_without_observation(&mut self, now: MonoTimeNs, profile: &ArmTrackingProfile) {
        self.wrist_blend.advance(now, false, &profile.blend);
        self.pole_blend.advance(now, false, &profile.blend);
        self.palm_blend.advance(now, false, &profile.blend);
        self.fingers_blend.advance(now, false, &profile.blend);
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
        state.left.advance_without_observation(now, profile);
        state.right.advance_without_observation(now, profile);
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
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )] // tests may panic (AGENTS.md)
    use super::*;
    use vtuber_core::arm_tracking::{HandWorldLandmarks, PoseWorldLandmark};

    fn point(meters: [f32; 3]) -> PoseWorldLandmark {
        PoseWorldLandmark {
            meters,
            visibility: Some(1.0),
            presence: Some(1.0),
        }
    }

    /// A hand detection with no usable palm plane: the zeroed landmarks cannot
    /// form a normal, so the palm channel stays absent even though the hand was
    /// detected.
    fn detected_hand(handedness: f32) -> HandWorldLandmarks {
        HandWorldLandmarks {
            landmarks: [point([0.0, 0.0, 0.0]); 21],
            handedness_score: Some(handedness),
        }
    }

    fn arm() -> ArmLandmarks {
        ArmLandmarks {
            shoulder: point([0.0, 0.0, 0.0]),
            elbow: point([0.3, 0.0, 0.0]),
            wrist: point([0.3, -0.4, 0.0]),
            hand: Some(detected_hand(1.0)),
        }
    }

    fn target(value: f32) -> ArmTrackingTarget {
        ArmTrackingTarget {
            wrist: [value; 3],
            elbow_pole: [value; 3],
            palm_normal: None,
            fingers: None,
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
            hand: a.hand,
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

    /// 21 hand world landmarks with the wrist, index MCP, and pinky MCP set.
    fn hand_with(wrist: [f32; 3], index_mcp: [f32; 3], pinky_mcp: [f32; 3]) -> HandWorldLandmarks {
        let landmarks = std::array::from_fn(|index| match index {
            0 => point(wrist),
            5 => point(index_mcp),
            17 => point(pinky_mcp),
            _ => point(wrist),
        });
        HandWorldLandmarks {
            landmarks,
            handedness_score: Some(1.0),
        }
    }

    /// Hand keypoints whose canonical palm normal points toward the camera (+Z).
    fn hand_toward_camera(wrist: [f32; 3]) -> HandWorldLandmarks {
        let wrist = vector(wrist);
        hand_with(
            array(wrist),
            array(wrist + Vector3::new(0.0, 0.02, 0.0)),
            array(wrist + Vector3::new(0.01, 0.0, 0.0)),
        )
    }

    /// Hand keypoints whose canonical palm normal points away from the camera.
    fn hand_away_from_camera(wrist: [f32; 3]) -> HandWorldLandmarks {
        let wrist = vector(wrist);
        hand_with(
            array(wrist),
            array(wrist + Vector3::new(0.01, 0.0, 0.0)),
            array(wrist + Vector3::new(0.0, 0.02, 0.0)),
        )
    }

    #[test]
    fn retarget_maps_hand_landmarks_to_a_canonical_palm_normal() {
        let reference = measure_arm_reference(arm()).unwrap();
        let posed = ArmLandmarks {
            hand: Some(hand_with(
                [0.3, -0.4, 0.0],
                [0.31, -0.4, 0.0],
                [0.30, -0.38, 0.0],
            )),
            ..arm()
        };
        let target = retarget_arm_landmarks(posed, reference);
        near(target.palm_normal.unwrap(), [0.0, 0.0, -1.0]);

        // A rigid translation of the whole hand keeps the normal fixed.
        let shift =
            |p: PoseWorldLandmark| point(array(vector(p.meters) + Vector3::new(1.0, 2.0, 3.0)));
        let translated_hand = posed.hand.map(|hand| {
            let mut landmarks = hand.landmarks;
            for landmark in landmarks.iter_mut() {
                *landmark = shift(*landmark);
            }
            HandWorldLandmarks {
                landmarks,
                handedness_score: hand.handedness_score,
            }
        });
        let translated = ArmLandmarks {
            shoulder: shift(posed.shoulder),
            elbow: shift(posed.elbow),
            wrist: shift(posed.wrist),
            hand: translated_hand,
        };
        near(
            retarget_arm_landmarks(translated, reference)
                .palm_normal
                .unwrap(),
            target.palm_normal.unwrap(),
        );
        assert_eq!(retarget_arm_landmarks(arm(), reference).palm_normal, None);

        // Collinear MCPs span no plane: no fabricated normal.
        let flat = ArmLandmarks {
            hand: Some(hand_with(
                [0.3, -0.4, 0.0],
                [0.31, -0.4, 0.0],
                [0.32, -0.4, 0.0],
            )),
            ..arm()
        };
        assert_eq!(retarget_arm_landmarks(flat, reference).palm_normal, None);
    }

    #[test]
    fn palm_quality_is_the_presence_of_a_hand_detection() {
        let profile = ArmTrackingProfile::default();
        let base = arm();
        let detected = ArmLandmarks {
            hand: Some(hand_with(
                [0.3, -0.4, 0.0],
                [0.31, -0.4, 0.0],
                [0.30, -0.38, 0.0],
            )),
            ..base
        };
        // The Hand Landmarker reports no landmark quality, so any detection is
        // an observed palm — including one whose left/right label was ambiguous.
        let ambiguous = ArmLandmarks {
            hand: detected.hand.map(|mut hand| {
                hand.handedness_score = Some(0.1);
                hand
            }),
            ..base
        };
        assert!(assess_arm_observation(&detected, &profile).palm);
        assert!(assess_arm_observation(&ambiguous, &profile).palm);
        assert!(
            !assess_arm_observation(&ArmLandmarks { hand: None, ..base }, &profile).palm,
            "no detection means no palm observation"
        );
    }

    #[test]
    fn an_ambiguous_handedness_label_does_not_drop_a_visible_arm() {
        // Pose sees the whole arm and the Hand Landmarker found the hand, but
        // the left/right label was ambiguous. The arm must stay adopted: the
        // handedness score says nothing about how well located the arm is.
        let profile = ArmTrackingProfile::default();
        let base = arm();
        let ambiguous = ArmLandmarks {
            hand: base.hand.map(|mut hand| {
                hand.handedness_score = Some(0.05);
                hand
            }),
            ..base
        };
        let state = tracked_state(ambiguous, &profile);
        assert_eq!(state.left.weights().wrist, 1.0);
        assert_eq!(state.right.weights().wrist, 1.0);
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
            hand: a.hand,
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

    #[test]
    fn a_fast_hand_is_followed_at_the_same_rate_as_a_slow_one() {
        // The response is a fixed time constant, so the fraction of the
        // remaining distance covered per render tick does not depend on how far
        // the observation moved. A hand crossing the body is motion, not a jump,
        // and must not be slowed for being quick.
        let target = |value: f32| ArmTrackingTarget {
            wrist: [value, 0.0, 0.0],
            elbow_pole: [value, 0.0, 0.0],
            palm_normal: None,
            fingers: None,
        };
        let dt = 1.0 / 60.0;
        let followed = |distance: f32| {
            let mut smoother = ArmSmootherState::new(target(0.0));
            smoother.advance(target(distance), dt).wrist[0] / distance
        };
        // A slow drift and a hand thrown across the body in one observation.
        let slow = followed(0.05);
        let fast = followed(1.5);
        assert!(
            (slow - fast).abs() < 1.0e-5,
            "the followed fraction must not depend on the step: {slow} vs {fast}"
        );
    }

    #[test]
    fn a_teleport_is_rejected_before_it_reaches_the_smoother() {
        // The fixed response above follows any step, so the discontinuity guard
        // is `max_wrist_step` rather than a stiffer filter. A wrist more than
        // one calibrated arm length away in a single observation is dropped and
        // never becomes a target; a plausible step is followed.
        let profile = ArmTrackingProfile::default();
        let base = arm();
        let mut state = tracked_state(base, &profile);
        let settled = state.left.source.unwrap();

        let teleport = arm_at([0.3, 0.2, 0.0], [-0.5, -0.4, 0.0]);
        let now = TRACKED_FRAMES * OBSERVATION_STEP_NS;
        let _ = feed(&mut state, teleport, 500, now, &profile);
        assert_eq!(state.left.source, Some(settled));

        // The same distance reached over several observations is ordinary motion
        // and is adopted, because the guard looks at one step, not at speed.
        let mut state = tracked_state(base, &profile);
        for step in 1..=10u64 {
            let fraction = step as f32 / 10.0;
            let moved = arm_at(
                [0.3 * fraction, 0.2 * fraction, 0.0],
                [0.3 - 0.8 * fraction, -0.4, 0.0],
            );
            let now = TRACKED_FRAMES * OBSERVATION_STEP_NS + step * OBSERVATION_STEP_NS;
            let _ = feed(&mut state, moved, 500 + step, now, &profile);
        }
        assert_ne!(state.left.source, Some(settled));
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
            hand: Some(detected_hand(1.0)),
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

    /// Frames needed to settle calibration and the one-second acquire ramp.
    const TRACKED_FRAMES: u64 = ARM_CALIBRATION_CAPACITY as u64 + 45;

    /// Tracks a steady arm until calibration and the blends are fully settled.
    fn tracked_state(arm: ArmLandmarks, profile: &ArmTrackingProfile) -> ArmTrackingState {
        let mut state = ArmTrackingState::new();
        for seq in 0..TRACKED_FRAMES {
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
            hand: None,
        }
    }

    #[test]
    fn elbow_plane_rotation_is_rate_limited_and_converges() {
        let target = |pole: [f32; 3]| ArmTrackingTarget {
            wrist: [0.0, 0.0, 1.0],
            elbow_pole: pole,
            palm_normal: None,
            fingers: None,
        };
        let dt = 1.0 / 60.0;
        let mut smoother = ArmSmootherState::new(target([1.0, 0.0, 0.0]));
        let axis = vector([0.0, 0.0, 1.0]);
        let mut previous = smoother.advance(target([1.0, 0.0, 0.0]), dt).elbow_pole;

        // A one-observation quarter turn is paced out at the capped rate.
        let mut turned = 0.0;
        for _ in 0..240 {
            let value = smoother.advance(target([0.0, 1.0, 0.0]), dt).elbow_pole;
            let before = perpendicular(vector(previous), axis).normalize();
            let after = perpendicular(vector(value), axis).normalize();
            let step = before.dot(&after).clamp(-1.0, 1.0).acos();
            assert!(
                step <= ELBOW_PLANE_MAX_RATE_RAD_PER_SEC * dt + 1.0e-4,
                "the plane must not turn faster than the cap: {step}"
            );
            turned += step;
            previous = value;
        }
        let final_dir = perpendicular(vector(previous), axis).normalize();
        assert!(
            final_dir.dot(&Vector3::new(0.0, 1.0, 0.0)) > 0.99,
            "the plane must reach the observation: {final_dir:?}"
        );
        assert!(
            (turned - std::f32::consts::FRAC_PI_2).abs() < 0.05,
            "the total turn must equal the observation, got {turned}"
        );
    }

    #[test]
    fn arm_adoption_gate_requires_consecutive_good_frames() {
        let mut gate = ArmAdoptionGate::new();
        assert!(!gate.update(Some(0.9), true));
        assert!(!gate.update(Some(0.2), true), "a bad frame resets progress");
        for _ in 0..ARM_GOOD_FRAMES - 1 {
            assert!(!gate.update(Some(0.9), true));
        }
        assert!(
            gate.update(Some(0.9), true),
            "four consecutive good frames adopt"
        );

        // Band values hold the current state without counting either way.
        for _ in 0..10 {
            assert!(gate.update(Some(0.55), true));
        }
        for _ in 0..ARM_BAD_FRAMES - 1 {
            assert!(gate.update(Some(0.2), true));
        }
        assert!(
            !gate.update(Some(0.2), true),
            "six consecutive bad frames lose it"
        );
        assert!(!gate.update(None, true), "a missing score is not good");
    }

    #[test]
    fn a_missing_hand_detection_blocks_adoption_and_ends_it() {
        let mut gate = ArmAdoptionGate::new();
        // Perfect Pose visibility with no detected hand is the hidden-limb
        // hallucination a Pose arm produces on its own.
        for _ in 0..ARM_GOOD_FRAMES * 2 {
            assert!(!gate.update(Some(1.0), false));
        }
        assert!(
            !gate.returned,
            "an arm that was never adopted has no return"
        );
        for _ in 0..ARM_GOOD_FRAMES - 1 {
            assert!(!gate.update(Some(1.0), true));
        }
        assert!(gate.update(Some(1.0), true));
        for _ in 0..ARM_BAD_FRAMES - 1 {
            assert!(gate.update(Some(1.0), false));
        }
        assert!(!gate.update(Some(1.0), false));
        assert!(gate.returned);
    }

    #[test]
    fn a_pose_arm_without_its_own_hand_is_not_adopted() {
        let profile = ArmTrackingProfile::default();
        let base = arm();
        let no_hand = ArmLandmarks { hand: None, ..base };
        // Pose reports a perfect arm, but the Hand Landmarker never sees a
        // hand for it: this is the hidden-limb hallucination, so nothing is
        // adopted or calibrated.
        let state = tracked_state(no_hand, &profile);
        assert_eq!(state.left.calibration.confirmed_length(), None);
        assert_eq!(state.left.weights().wrist, 0.0);
        assert!(state.left.output.is_none());

        // The same Pose arm is adopted once a hand is detected for it.
        let state = tracked_state(base, &profile);
        assert_eq!(state.left.weights().wrist, 1.0);
    }

    #[test]
    fn losing_the_hand_returns_the_arm_while_pose_stays_visible() {
        let profile = ArmTrackingProfile::default();
        let base = arm();
        let mut state = tracked_state(base, &profile);
        assert_eq!(state.left.weights().wrist, 1.0);

        let no_hand = ArmLandmarks { hand: None, ..base };
        for seq in 0..FULL_RETURN_FRAMES {
            let now = SETTLED_NS + seq * OBSERVATION_STEP_NS;
            let _ = feed(&mut state, no_hand, 100 + seq, now, &profile);
        }
        assert_eq!(state.left.weights().wrist, 0.0);
    }

    #[test]
    fn visibility_band_flicker_does_not_restart_the_return() {
        let profile = ArmTrackingProfile::default();
        let base = arm();
        let mut state = tracked_state(base, &profile);
        assert_eq!(state.left.weights().wrist, 1.0);

        // The hand leaves, but Pose keeps reporting a score that flickers
        // around the old single threshold: inside the hysteresis band most
        // frames, occasionally below the exit threshold. The channel must not
        // re-ramp, and the return must stay monotonic to zero.
        let flicker = |score: f32| ArmLandmarks {
            shoulder: PoseWorldLandmark {
                visibility: Some(score),
                ..base.shoulder
            },
            elbow: PoseWorldLandmark {
                visibility: Some(score),
                ..base.elbow
            },
            wrist: PoseWorldLandmark {
                visibility: Some(score),
                ..base.wrist
            },
            hand: None,
        };
        let mut previous = 1.0;
        for seq in 0..FULL_RETURN_FRAMES {
            let now = SETTLED_NS + seq * OBSERVATION_STEP_NS;
            let score = if seq % 2 == 0 { 0.65 } else { 0.35 };
            let control = feed(&mut state, flicker(score), 100 + seq, now, &profile).unwrap();
            let weight = control.weights.left.wrist;
            assert!(
                weight <= previous + 1.0e-6,
                "the return must never re-ramp: {weight} > {previous} at {seq}"
            );
            previous = weight;
        }
        assert_eq!(previous, 0.0);
    }

    #[test]
    fn calibration_confirms_only_after_reliable_samples() {
        let profile = ArmTrackingProfile::default();
        let arm = arm();
        let mut state = ArmTrackingState::new();
        // Adoption needs ARM_GOOD_FRAMES good observations before calibration
        // samples are collected at all.
        let frames = ARM_GOOD_FRAMES as u64 + ARM_CALIBRATION_CAPACITY as u64;
        for seq in 0..frames {
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
            hand: None,
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
        for step in 1..=150u64 {
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
    const FULL_RETURN_FRAMES: u64 = 170;

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
    fn observed_palm_normal_is_smoothed_on_the_render_clock() {
        let profile = ArmTrackingProfile::default();
        let base = arm();
        let visible = ArmLandmarks {
            hand: Some(hand_toward_camera(base.wrist.meters)),
            ..base
        };
        let mut state = tracked_state(visible, &profile);
        near(
            state.left.source.unwrap().palm_normal.unwrap(),
            [0.0, 0.0, 1.0],
        );
        assert_eq!(state.left.weights().palm, 1.0);

        let flipped = ArmLandmarks {
            hand: Some(hand_away_from_camera(base.wrist.meters)),
            ..base
        };
        let captured = TRACKED_FRAMES * OBSERVATION_STEP_NS;
        let frame = pose_frame(500, captured, Some(observation(flipped, flipped)));
        let (next, control) =
            step_arm_tracking(&state, Some(&frame), MonoTimeNs(captured), &profile);
        state = next;
        near(
            state.left.source.unwrap().palm_normal.unwrap(),
            [0.0, 0.0, -1.0],
        );
        let mut previous = control.unwrap().targets.left.unwrap().palm_normal.unwrap();
        assert!(
            previous[2] > -0.99,
            "the smoothed normal must not snap to the new observation: {previous:?}"
        );

        for tick in 1..=220u64 {
            let now = captured + tick * 16_666_667;
            let (next, control) =
                step_arm_tracking(&state, Some(&frame), MonoTimeNs(now), &profile);
            state = next;
            let current = control.unwrap().targets.left.unwrap().palm_normal.unwrap();
            assert!(
                current[2] <= previous[2] + 1.0e-6,
                "the palm must turn monotonically: {current:?} after {previous:?}"
            );
            previous = current;
        }
        assert!(previous[2] < -0.99);
    }

    /// 21 hand world landmarks built from a caller-supplied position per index.
    fn hand_from(points: [PoseWorldLandmark; 21]) -> HandWorldLandmarks {
        HandWorldLandmarks {
            landmarks: points,
            handedness_score: Some(1.0),
        }
    }

    /// The ray the fixture's `finger` leaves the wrist along. Splayed so the
    /// palm plane between the index and little roots is well defined, and so a
    /// straight chain is collinear with the wrist and reports no knuckle
    /// flexion.
    fn splayed_ray(finger: usize) -> Vector3<f32> {
        finite_normalized(Vector3::new(1.0, 0.0, (1.5 - finger as f32) * 0.12))
            .expect("a splayed ray is normalizable")
    }

    /// A hand in the Hand Landmarker basis whose four fingers splay from the
    /// wrist along their own ray and fold toward +Y, so every joint's interior
    /// angle is exactly the flexion the fixture asked for.
    fn flat_hand(curl: [[f32; 3]; 4], thumb_curl: [f32; 2]) -> HandWorldLandmarks {
        let mut points = [point([0.0, 0.0, 0.0]); 21];
        let mut place = |slot: usize, value: Vector3<f32>| points[slot] = point(array(value));
        for (finger, chain) in FINGER_CHAINS.iter().enumerate() {
            let ray = splayed_ray(finger);
            let curl = curl.get(finger).copied().unwrap_or([0.0; 3]);
            let mut previous = Vector3::zeros();
            let mut heading = 0.0;
            for step in 0..4 {
                let Some(slot) = chain.get(step) else {
                    break;
                };
                // Each phalanx turns the running heading by its own joint's
                // flexion, so the last segment carries the last curl and the
                // tip is where the last joint becomes readable.
                if step > 0 {
                    heading += curl.get(step - 1).copied().unwrap_or(0.0);
                }
                let segment = ray * (0.02 * heading.cos()) + Vector3::y() * (0.02 * heading.sin());
                place(*slot, previous + segment);
                previous += segment;
            }
        }
        let (Some(cmc), Some(mcp), Some(ip), Some(tip)) = (
            THUMB_CHAIN.first().copied(),
            THUMB_CHAIN.get(1).copied(),
            THUMB_CHAIN.get(2).copied(),
            THUMB_CHAIN.get(3).copied(),
        ) else {
            return hand_from(points);
        };
        let ray = finite_normalized(Vector3::new(0.5, 0.0, 0.8)).expect("thumb ray");
        let segment = |angle: f32| ray * (0.02 * angle.cos()) + Vector3::y() * (0.02 * angle.sin());
        place(cmc, ray * 0.02);
        let ip_heading = thumb_curl.first().copied().unwrap_or(0.0);
        place(mcp, ray * 0.02 + segment(ip_heading));
        let ip_point = ray * 0.02 + segment(ip_heading) + segment(2.0 * ip_heading);
        place(ip, ip_point);
        let tip_heading = 2.0 * ip_heading + thumb_curl.get(1).copied().unwrap_or(0.0);
        place(tip, ip_point + segment(tip_heading));
        hand_from(points)
    }

    /// A thumb whose MCP bend is oblique to the palm-derived bend axis.
    ///
    /// The old scalar projection reads only a quarter of this turn even though
    /// the full CMC-MCP-IP angle is `curl`, which is how a real opposed thumb
    /// can remain visually extended in a fist.
    fn oblique_thumb_mcp_hand(curl: f32) -> HandWorldLandmarks {
        let mut hand = flat_hand([[0.0; 3]; 4], [0.0; 2]);
        let incoming = finite_normalized(Vector3::new(0.5, 0.0, 0.8)).expect("thumb ray");
        let normal = Vector3::y();
        let bend_axis = finite_normalized(incoming.cross(&normal)).expect("thumb bend axis");
        let rotation_axis =
            finite_normalized(bend_axis * 0.25 + normal * 0.968_245_8).expect("oblique thumb axis");
        let outgoing = incoming * curl.cos()
            + rotation_axis.cross(&incoming) * curl.sin()
            + rotation_axis * rotation_axis.dot(&incoming) * (1.0 - curl.cos());
        let cmc = incoming * 0.02;
        let mcp = cmc + incoming * 0.02;
        let ip = mcp + outgoing * 0.02;
        let tip = ip + outgoing * 0.02;
        hand.landmarks[1] = point(array(cmc));
        hand.landmarks[2] = point(array(mcp));
        hand.landmarks[3] = point(array(ip));
        hand.landmarks[4] = point(array(tip));
        hand
    }

    /// A hand whose only readable landmarks are the two knuckles the palm plane
    /// is derived from, so the palm channel works and the fingers do not.
    fn palm_only_hand() -> HandWorldLandmarks {
        let mut points = [point([0.0, 0.0, 0.0]); 21];
        if let Some(slot) = points.get_mut(HAND_INDEX_MCP) {
            *slot = point(array(splayed_ray(0) * 0.02));
        }
        if let Some(slot) = points.get_mut(HAND_PINKY_MCP) {
            *slot = point(array(splayed_ray(3) * 0.02));
        }
        hand_from(points)
    }

    #[test]
    fn a_flat_hand_reports_no_finger_flexion() {
        let reference = measure_arm_reference(arm()).unwrap();
        let arm = ArmLandmarks {
            hand: Some(flat_hand([[0.0; 3]; 4], [0.0; 2])),
            ..arm()
        };
        let pose = retarget_arm_landmarks(arm, reference)
            .fingers
            .expect("a flat hand defines its joints");
        for finger in pose.fingers {
            for angle in finger {
                assert!(
                    angle.abs() < 1.0e-4,
                    "a straight chain must report no flexion: {finger:?}"
                );
            }
        }
        for angle in pose.thumb {
            assert!(angle.abs() < 1.0e-4, "{:?}", pose.thumb);
        }
    }

    #[test]
    fn each_finger_reports_its_own_curl() {
        let reference = measure_arm_reference(arm()).unwrap();
        // A peace sign: the index and middle stay flat while the ring and little
        // fold, which a single shared curl amount could not produce.
        let arm = ArmLandmarks {
            hand: Some(flat_hand(
                [[0.0; 3], [0.0; 3], [1.0, 1.2, 0.6], [0.8, 0.9, 0.5]],
                [0.4, 0.5],
            )),
            ..arm()
        };
        let pose = retarget_arm_landmarks(arm, reference)
            .fingers
            .expect("a curled hand defines its joints");
        for (finger, curls) in pose.fingers.iter().enumerate() {
            for (joint, angle) in curls.iter().enumerate() {
                if finger < 2 {
                    assert!(
                        angle.abs() < 1.0e-4,
                        "finger {finger} joint {joint} must stay flat: {angle}"
                    );
                } else {
                    assert!(
                        *angle > 0.1,
                        "finger {finger} joint {joint} must report its curl: {angle}"
                    );
                }
            }
        }
        // The two folded fingers differ from each other, so the articulation is
        // per finger rather than one shared amount.
        let (ring, little) = (pose.fingers[2][0], pose.fingers[3][0]);
        assert!(
            (ring - little).abs() > 0.05,
            "each finger must keep its own knuckle curl: {ring} vs {little}"
        );
        assert!(pose.thumb[0] > 0.1 && pose.thumb[1] > 0.1);
    }

    #[test]
    fn a_finger_flexion_is_clamped_to_the_anatomical_range() {
        let reference = measure_arm_reference(arm()).unwrap();
        // A curl far past anything a finger reaches must stop at the joint's
        // limit rather than folding the bone back through itself.
        let arm = ArmLandmarks {
            hand: Some(flat_hand([[9.0; 3]; 4], [9.0; 2])),
            ..arm()
        };
        let pose = retarget_arm_landmarks(arm, reference)
            .fingers
            .expect("a curled hand defines its joints");
        for finger in pose.fingers {
            for (joint, angle) in finger.iter().enumerate() {
                let limit = FINGER_FLEXION_LIMIT_RAD.get(joint).copied().unwrap_or(0.0);
                assert!(
                    angle.abs() <= limit + 1.0e-4,
                    "joint {joint} must stop at its limit: {angle} > {limit}"
                );
            }
        }
        for (joint, angle) in pose.thumb.iter().enumerate() {
            let limit = THUMB_FLEXION_LIMIT_RAD.get(joint).copied().unwrap_or(0.0);
            assert!(angle.abs() <= limit + 1.0e-4, "{angle} > {limit}");
        }
    }

    #[test]
    fn thumb_uses_cmc_to_mcp_in_the_local_palm_frame() {
        let mut hand = flat_hand([[0.0; 3]; 4], [0.0; 2]);
        hand.landmarks[1] = point([0.012, 0.0, 0.022]);
        hand.landmarks[2] = point([0.030, 0.0, 0.035]);
        let pose = observed_finger_pose(ArmLandmarks {
            hand: Some(hand),
            ..arm()
        })
        .unwrap();
        // This fixture has forward +X, across +Z, normal +Y. Wrist-to-CMC
        // and CMC-to-MCP are deliberately non-collinear.
        let expected = Vector3::new(0.013, 0.018, 0.0).normalize();
        assert!((vector(pose.thumb_direction) - expected).norm() < 1.0e-5);
    }

    #[test]
    fn thumb_mcp_uses_the_full_3d_angle_for_oblique_closure() {
        let curl = 0.7;
        let pose = observed_finger_pose(ArmLandmarks {
            hand: Some(oblique_thumb_mcp_hand(curl)),
            ..arm()
        })
        .expect("the hand defines a thumb");
        assert!(
            (pose.thumb[0] - curl).abs() < 1.0e-4,
            "the full oblique MCP angle must close the proximal thumb: {:?}",
            pose.thumb
        );
        assert!(
            pose.thumb[1].abs() < 1.0e-4,
            "a straight IP joint must stay straight: {:?}",
            pose.thumb
        );
    }

    #[test]
    fn non_radial_extended_finger_separates_spread_from_flexion() {
        let mut hand = flat_hand([[0.0; 3]; 4], [0.0; 2]);
        hand.landmarks[5] = point([0.02, 0.05, 0.0]);
        hand.landmarks[17] = point([-0.02, 0.05, 0.0]);
        let observe = |spread: f32, curl: f32| {
            let mut posed = hand;
            // Palm normal +Z; spread goes toward +X, curl toward +Z.
            let ray = Vector3::new(
                spread.sin() * curl.cos(),
                spread.cos() * curl.cos(),
                curl.sin(),
            );
            for (step, slot) in [6, 7, 8].into_iter().enumerate() {
                posed.landmarks[slot] = point(array(
                    Vector3::new(0.02, 0.05, 0.0) + ray * (0.03 * (step + 1) as f32),
                ));
            }
            observed_finger_pose(ArmLandmarks {
                hand: Some(posed),
                ..arm()
            })
            .unwrap()
        };
        for (spread, curl) in [(0.0, 0.0), (0.4, 0.0), (0.0, 0.6), (0.4, 0.6), (0.4, 1.6)] {
            let pose = observe(spread, curl);
            assert!((pose.fingers[0][0] - curl).abs() < 1.0e-5);
            assert!((pose.spread[0] - spread).abs() < 1.0e-5);
            assert!(pose.fingers[0][1].abs() < 1.0e-5);
            assert!(pose.fingers[0][2].abs() < 1.0e-5);
        }
    }

    #[test]
    fn local_finger_shape_and_smoothing_ignore_rigid_hand_motion() {
        let hand = flat_hand([[0.3, 0.5, 0.2]; 4], [0.2, 0.4]);
        let pose = observed_finger_pose(ArmLandmarks {
            hand: Some(hand),
            ..arm()
        })
        .unwrap();
        let mut mirrored_hand = hand;
        for point in &mut mirrored_hand.landmarks {
            point.meters[0] = -point.meters[0];
        }
        let original_arm = ArmLandmarks {
            hand: Some(hand),
            ..arm()
        };
        let reflected_arm = ArmLandmarks {
            hand: Some(mirrored_hand),
            ..arm()
        };
        let reference = measure_arm_reference(original_arm).unwrap();
        assert_eq!(
            retarget_arm_landmarks(original_arm, reference)
                .mirrored()
                .fingers,
            retarget_arm_landmarks(reflected_arm, reference).fingers,
        );
        let mut smoother = FingerSmootherState::new(pose);
        for angle in [0.4, 1.2, -0.8] {
            let rotation = nalgebra::Rotation3::from_euler_angles(angle, angle * 0.7, -angle * 0.3);
            let mut moved = hand;
            for point in &mut moved.landmarks {
                point.meters =
                    array(rotation * vector(point.meters) + Vector3::new(0.1, -0.2, 0.4));
            }
            let moved_pose = observed_finger_pose(ArmLandmarks {
                hand: Some(moved),
                ..arm()
            })
            .unwrap();
            smoother.step(moved_pose, 1.0 / 60.0);
            for result in [moved_pose, smoother.output().unwrap()] {
                for (a, b) in result
                    .fingers
                    .into_iter()
                    .flatten()
                    .chain(result.spread)
                    .chain(result.thumb)
                    .chain(result.thumb_direction)
                    .zip(
                        pose.fingers
                            .into_iter()
                            .flatten()
                            .chain(pose.spread)
                            .chain(pose.thumb)
                            .chain(pose.thumb_direction),
                    )
                {
                    assert!((a - b).abs() < 1.0e-4, "{a} != {b}");
                }
            }
        }
    }

    #[test]
    fn a_hand_without_usable_joints_reports_no_fingers() {
        let reference = measure_arm_reference(arm()).unwrap();
        // Every landmark collapsed onto the wrist spans no joint, so the finger
        // channel stays absent rather than reporting fabricated zero curls.
        let arm = ArmLandmarks {
            hand: Some(detected_hand(1.0)),
            ..arm()
        };
        assert_eq!(retarget_arm_landmarks(arm, reference).fingers, None);
    }

    #[test]
    fn a_usable_palm_does_not_imply_usable_fingers() {
        let reference = measure_arm_reference(arm()).unwrap();
        // A hand whose only readable landmarks are the two knuckles still gives
        // the palm plane, but no finger joint spans anything, so the two
        // channels are decided separately rather than by one flag.
        let arm = ArmLandmarks {
            hand: Some(palm_only_hand()),
            ..arm()
        };
        let target = retarget_arm_landmarks(arm, reference);
        assert!(target.palm_normal.is_some(), "the palm plane is readable");
        assert_eq!(target.fingers, None, "the finger joints are not");
    }

    #[test]
    fn the_finger_channel_follows_the_hand_detection_and_returns_to_rest() {
        let profile = ArmTrackingProfile::default();
        let base = arm();
        let open = ArmLandmarks {
            hand: Some(flat_hand([[0.0; 3]; 4], [0.0; 2])),
            ..base
        };
        let mut state = tracked_state(open, &profile);
        assert!(
            state.left.weights().fingers > 0.99,
            "the finger channel must follow the hand detection: {}",
            state.left.weights().fingers
        );
        assert!(
            state.left.source.unwrap().fingers.is_some(),
            "an open hand still reports its joints"
        );

        // Losing the hand detection returns the whole arm, and the last
        // articulation is held while the finger weight eases to zero so the
        // avatar returns the fingers to rest instead of snapping them.
        let no_hand = ArmLandmarks { hand: None, ..base };
        for seq in 0..FULL_RETURN_FRAMES {
            let now = SETTLED_NS + seq * OBSERVATION_STEP_NS;
            let _ = feed(&mut state, no_hand, 100 + seq, now, &profile);
        }
        assert_eq!(state.left.weights().fingers, 0.0);
        assert_eq!(state.left.weights().wrist, 0.0);
        assert!(
            state.left.output.unwrap().fingers.is_some(),
            "the last articulation is held while the weight returns to the rest pose"
        );
    }

    #[test]
    fn an_exactly_opposed_palm_turns_through_a_stable_side() {
        // Turning the hand over leaves the two palm planes exactly opposed, so
        // the shortest arc has no preferred axis. The state must still turn
        // continuously and land on the observation; filtering the raw vector
        // instead would pass through a near-zero vector, whose normalization
        // picks an arbitrary side the hand never showed.
        let dt = 1.0 / 60.0;
        let start = vector([0.0, 0.0, 1.0]);
        let mut palm = DirectionSmootherState::new(start);
        let mut previous = start;
        let mut axis = Vector3::zeros();
        for _ in 0..300 {
            palm.step(array(-start), dt, ARM_PALM_TIME_CONSTANT_SEC);
            let current = vector(palm.normalized().expect("the palm stays on the sphere"));
            assert!(
                (current.norm() - 1.0).abs() < 1.0e-5,
                "the smoothed palm must stay a unit vector: {current:?}"
            );
            let turn = previous.cross(&current);
            if turn.norm() > 1.0e-6 {
                let first = if axis.norm() > 0.0 { &axis } else { &turn };
                assert!(
                    turn.dot(first) > 0.0,
                    "the palm must not reverse its turn: {turn:?}"
                );
                axis = turn;
            }
            previous = current;
        }
        assert!(
            previous.dot(&(-start)) > 1.0 - 1.0e-4,
            "the palm must reach the opposed observation: {previous:?}"
        );
    }

    #[test]
    fn a_palm_turn_never_leaves_the_plane_of_the_observation() {
        // The rotation between two palm planes is about the line where they meet,
        // so every intermediate direction stays inside the plane the two
        // observations span. A vector-space smoother can leave that plane
        // because it corrects all three components at once.
        let dt = 1.0 / 60.0;
        let from = vector([0.0, 0.0, 1.0]);
        let to = vector([1.0, 0.0, 0.0]);
        let mut palm = DirectionSmootherState::new(from);
        for _ in 0..300 {
            palm.step(array(to), dt, ARM_PALM_TIME_CONSTANT_SEC);
            let current = vector(palm.normalized().expect("the palm stays on the sphere"));
            assert!(
                current.y.abs() < 1.0e-4,
                "the turn must stay in the observation's plane: {current:?}"
            );
        }
    }

    #[test]
    fn a_degenerate_palm_observation_holds_the_last_direction() {
        let mut palm = DirectionSmootherState::new(vector([0.0, 0.0, 1.0]));
        palm.step([0.0, 0.0, 0.0], RENDER_STEP_SEC, ARM_PALM_TIME_CONSTANT_SEC);
        palm.step(
            [f32::NAN, 0.0, 0.0],
            RENDER_STEP_SEC,
            ARM_PALM_TIME_CONSTANT_SEC,
        );
        near(palm.normalized().unwrap(), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn a_palm_flip_settles_within_a_second_of_render_ticks() {
        // The palm constant is deliberately only a multiple of the wrist one, so
        // turning the hand over reaches the observation in about a second of
        // render ticks instead of trailing the wrist position for several.
        let mut palm = DirectionSmootherState::new(vector([0.0, 0.0, 1.0]));
        let target = array(-vector([0.0, 0.0, 1.0]));
        let mut ticks = 0u32;
        loop {
            palm.step(target, RENDER_STEP_SEC, ARM_PALM_TIME_CONSTANT_SEC);
            ticks += 1;
            let current = vector(palm.normalized().expect("the palm stays on the sphere"));
            if current.dot(&-vector([0.0, 0.0, 1.0])) > 1.0 - 1.0e-3 {
                break;
            }
            assert!(ticks < 60, "the palm must settle within a second: {ticks}");
        }
        assert!(
            ticks > 5,
            "the palm must still be smoothed, not snapped: {ticks} ticks"
        );
    }

    #[test]
    fn palm_loss_holds_the_normal_while_the_channel_returns_to_virtual() {
        let profile = ArmTrackingProfile::default();
        let base = arm();
        let visible = ArmLandmarks {
            hand: Some(hand_toward_camera(base.wrist.meters)),
            ..base
        };
        let mut state = tracked_state(visible, &profile);
        assert_eq!(state.left.weights().palm, 1.0);

        for seq in 0..FULL_RETURN_FRAMES {
            let now = SETTLED_NS + seq * OBSERVATION_STEP_NS;
            let control = feed(&mut state, base, 100 + seq, now, &profile).unwrap();
            assert!(control.targets.left.unwrap().palm_normal.is_some());
        }
        assert_eq!(state.left.weights().palm, 0.0);
        assert_eq!(state.left.weights().wrist, 1.0);
        assert!(state.left.output.unwrap().palm_normal.is_some());
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
        for seq in 0..40u64 {
            let now = start + seq * OBSERVATION_STEP_NS;
            let control = feed(&mut state, base, 1000 + seq, now, &profile).unwrap();
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

        // The hold keeps full authority, and the eased five-second return has
        // barely started a second after the loss: no snap to the virtual arm.
        let (_, held) = observations[0];
        assert_eq!(held, 1.0);
        let at_one_second = observations
            .iter()
            .find(|(now, _)| *now >= loss_at + 1_150_000_000)
            .expect("one second is inside the recorded window")
            .1;
        assert!(
            at_one_second > 0.8,
            "the return must start slowly, got {at_one_second}"
        );
        let at_three_seconds = observations
            .iter()
            .find(|(now, _)| *now >= loss_at + 3_150_000_000)
            .expect("three seconds are inside the recorded window")
            .1;
        assert!(
            at_three_seconds > 0.2 && at_three_seconds < 0.6,
            "the return should be in progress after three seconds, got {at_three_seconds}"
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

        // Reacquisition takes consecutive good camera frames, then the retained
        // observation is re-fed at render rate. The held ticks must keep
        // ramping instead of being mistaken for absence and snapping the
        // weight to full authority.
        let mut captured = SETTLED_NS + (FULL_RETURN_FRAMES + 1) * OBSERVATION_STEP_NS;
        let mut seq = 300;
        let mut first = 0.0;
        for step in 0..ARM_REACQUIRE_GOOD_FRAMES as u64 {
            seq = 300 + step;
            captured = SETTLED_NS + (FULL_RETURN_FRAMES + 1 + step) * OBSERVATION_STEP_NS;
            let control = feed(&mut state, base, seq, captured, &profile).unwrap();
            first = control.weights.left.wrist;
        }
        assert!((0.0..0.25).contains(&first), "got {first}");

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
        assert!((0.0..0.5).contains(&weight), "got {weight}");
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

        // The hand is detected far from where it was lost and stays good for
        // the frames reacquisition hysteresis requires. It may be anywhere,
        // but it must blend in from the current authority.
        let far = arm_at([0.3, 0.2, 0.0], [-0.5, -0.4, 0.0]);
        let mut weight = 0.0;
        for step in 0..ARM_REACQUIRE_GOOD_FRAMES as u64 {
            let now = SETTLED_NS + (20 + step) * OBSERVATION_STEP_NS;
            let control = feed(&mut state, far, 120 + step, now, &profile).unwrap();
            weight = control.weights.left.wrist;
        }
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
    fn a_returning_arm_needs_fewer_good_frames_than_a_cold_start() {
        // The count is in camera frames, and Pose and the Hand Landmarker share
        // one worker, so the wait scales with however slow inference currently
        // is. A cold start has to survive the calibration window; a return only
        // has to reject a single spurious detection, because the shared acquire
        // ramp is what brings the authority back.
        let mut gate = ArmAdoptionGate::new();
        for _ in 0..ARM_GOOD_FRAMES - 1 {
            assert!(!gate.update(Some(1.0), true));
        }
        assert!(gate.update(Some(1.0), true), "the cold start adopts");
        for _ in 0..ARM_BAD_FRAMES - 1 {
            assert!(gate.update(Some(0.0), true));
        }
        assert!(!gate.update(Some(0.0), true), "the arm is lost");

        let mut frames = 0u32;
        loop {
            frames += 1;
            if gate.update(Some(1.0), true) {
                break;
            }
            assert!(
                frames <= ARM_REACQUIRE_GOOD_FRAMES,
                "a returning arm must not wait the cold-start count: {frames}"
            );
        }
        assert_eq!(frames, ARM_REACQUIRE_GOOD_FRAMES);

        // A single good frame in the middle of the return is not enough on its
        // own, so the shorter count still rejects a flapping decision.
        let mut gate = ArmAdoptionGate::new();
        for _ in 0..ARM_GOOD_FRAMES - 1 {
            assert!(!gate.update(Some(1.0), true));
        }
        assert!(gate.update(Some(1.0), true));
        for _ in 0..ARM_BAD_FRAMES - 1 {
            assert!(gate.update(Some(0.0), true));
        }
        assert!(!gate.update(Some(0.0), true));
        assert!(!gate.update(Some(1.0), true));
        assert!(gate.update(Some(1.0), true));
    }

    #[test]
    fn a_returned_arm_still_ramps_its_authority_back_in() {
        // The shortened wait must not turn into a snap: the channel's weight
        // still comes back over the profile's acquire time.
        let profile = ArmTrackingProfile::default();
        let base = arm();
        let mut state = tracked_state(base, &profile);
        let hidden = hidden_arm(base);
        for seq in 0..FULL_RETURN_FRAMES {
            let now = SETTLED_NS + seq * OBSERVATION_STEP_NS;
            let _ = feed(&mut state, hidden, 100 + seq, now, &profile);
        }
        assert_eq!(state.left.weights().wrist, 0.0);

        let start = SETTLED_NS + (FULL_RETURN_FRAMES + 1) * OBSERVATION_STEP_NS;
        let mut previous = 0.0;
        let mut reached_full_after = None;
        for step in 0..40u64 {
            let now = start + step * OBSERVATION_STEP_NS;
            let weight = feed(&mut state, base, 1000 + step, now, &profile)
                .unwrap()
                .weights
                .left
                .wrist;
            assert!(weight >= previous, "the acquire ramp must not reverse");
            assert!(weight <= 1.0);
            if weight >= 1.0 {
                reached_full_after = Some(step);
                break;
            }
            previous = weight;
        }
        let elapsed_sec = reached_full_after.expect("the arm must come back") as f32
            * OBSERVATION_STEP_NS as f32
            * 1.0e-9;
        assert!(
            elapsed_sec > profile.blend.acquire.as_secs_f32() * 0.5,
            "the authority must still ramp over the acquire time: {elapsed_sec}"
        );
    }

    #[test]
    fn one_sides_loss_does_not_disturb_the_other_sides_recovery() {
        // Losing one arm must not consume the other arm's adoption state, and a
        // return on one side must not be read as the other side being seen.
        let profile = ArmTrackingProfile::default();
        let base = arm();
        let mut state = tracked_state(base, &profile);
        let hidden = hidden_arm(base);

        // Only the left arm goes out of frame; the right keeps good observations
        // and must stay at full authority throughout.
        for seq in 0..FULL_RETURN_FRAMES {
            let now = SETTLED_NS + seq * OBSERVATION_STEP_NS;
            let frame = pose_frame(seq + 100, now, Some(observation(hidden, base)));
            let (next, control) =
                step_arm_tracking(&state, Some(&frame), MonoTimeNs(now), &profile);
            state = next;
            assert_eq!(control.unwrap().weights.right.wrist, 1.0);
        }
        assert_eq!(state.left.weights().wrist, 0.0);
        assert!(!state.left.adoption.adopted);
        assert!(state.left.adoption.returned);
        assert!(state.right.adoption.adopted);
        assert!(!state.right.adoption.returned);
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
        arm.hand = None;
        let quality = assess_arm_observation(&arm, &profile);
        assert!(quality.shoulder);
        assert!(quality.wrist);
        assert!(!quality.elbow);
        assert!(!quality.palm);
        assert!(quality.follows_wrist());
        assert!(!quality.uses_elbow());
    }

    #[test]
    fn an_opposed_elbow_plane_crosses_to_the_observation() {
        // A bent arm establishes a bend side; the hand then moves so the
        // observed plane is on the other side of the shoulder-wrist axis. The
        // retained plane must rotate across to the observation instead of being
        // pinned on the wrong side (which wraps the upper arm past its range).
        let axis = vector([0.0, 0.0, 1.0]);
        let established = stabilize_elbow_pole(
            None,
            ArmTrackingTarget {
                wrist: [0.0, 0.0, 0.6],
                elbow_pole: [1.0, 0.0, 0.0],
                palm_normal: None,
                fingers: None,
            },
        )
        .expect("a bent arm defines a plane");

        let opposed = ArmTrackingTarget {
            wrist: [0.0, 0.0, 0.6],
            elbow_pole: [-1.0, 0.2, 0.0],
            palm_normal: None,
            fingers: None,
        };
        let crossed = stabilize_elbow_pole(Some(established), opposed).expect("plane");
        let observed = finite_normalized(perpendicular(vector(opposed.elbow_pole), axis))
            .expect("observed plane");
        let crossed_direction = finite_normalized(vector(crossed)).expect("crossed plane");
        assert!(
            crossed_direction.dot(&observed) > 0.999,
            "an opposed plane must cross to the observation: {crossed_direction:?} vs {observed:?}"
        );
    }

    #[test]
    fn undefined_pole_returns_no_pole_instead_of_a_world_axis() {
        assert_eq!(
            stabilize_elbow_pole(
                None,
                ArmTrackingTarget {
                    wrist: [0.0, 0.0, 0.0],
                    elbow_pole: [0.1, 0.2, 0.0],
                    palm_normal: None,
                    fingers: None,
                },
            ),
            None
        );
    }

    #[test]
    fn a_near_full_extension_holds_the_elbow_plane_instead_of_flipping_it() {
        // The observed plane is well-conditioned while the arm is bent. As the
        // observation extends it becomes noisy and the plane must hold its last
        // direction rather than jump to the virtual pole.
        let bent = ArmTrackingTarget {
            wrist: [0.6, 0.0, 0.0],
            elbow_pole: [0.3, 0.5, 0.0],
            palm_normal: None,
            fingers: None,
        };
        let established = stabilize_elbow_pole(None, bent).expect("a bent arm defines a plane");
        let noisy = ArmTrackingTarget {
            wrist: [1.05, 0.0, 0.0],
            elbow_pole: [0.3, -0.5, 0.0],
            palm_normal: None,
            fingers: None,
        };
        let held =
            stabilize_elbow_pole(Some(established), noisy).expect("the previous plane holds");
        let established_direction = finite_normalized(vector(established)).unwrap();
        let held_direction = finite_normalized(vector(held)).unwrap();
        assert!(
            established_direction.dot(&held_direction) > 0.0,
            "an extending arm must keep its bend side: {established:?} -> {held:?}"
        );
    }
}
