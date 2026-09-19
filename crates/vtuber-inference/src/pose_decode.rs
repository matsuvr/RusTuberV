//! Pure extraction at the Pose and Hand Landmarker result boundary.
//!
//! No model execution, avatar policy, or side assignment from the handedness
//! label: a detected hand is paired with the Pose wrist nearest to it in the
//! same normalized image, which is camera-convention independent.

use mediapipe::HandLandmarkerResult;
use mediapipe::PoseLandmarkerResult;
use mediapipe::WorldLandmark;
use vtuber_core::arm_tracking::{
    ArmLandmarks, HAND_LANDMARK_COUNT, HandWorldLandmarks, PoseArmObservation, PoseWorldLandmark,
};

/// MediaPipe Pose's fixed world-landmark count.
pub const POSE_LANDMARK_COUNT: usize = 33;

/// Largest normalized-image distance between a Pose wrist and a detected hand
/// wrist that still counts as the same physical hand.
const HAND_WRIST_PAIR_DISTANCE: f32 = 0.15;

/// Malformed task output, distinct from successful inference finding no person.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
pub enum PoseDecodeError {
    /// The task did not return its declared schema.
    #[error("expected 33 pose world landmarks, received {actual}")]
    LandmarkCount {
        /// Number actually supplied.
        actual: usize,
    },
    /// A selected point has non-finite coordinates or an invalid supplied quality score.
    #[error("invalid pose world landmark at index {index}")]
    InvalidLandmark {
        /// Index in the MediaPipe Pose schema.
        index: usize,
    },
    /// The one-person task returned a different number of people.
    #[error("expected at most one person, received {actual}")]
    PersonCount {
        /// Number of people actually returned.
        actual: usize,
    },
    /// The hand result's parallel arrays disagree on how many hands were found.
    #[error(
        "hand result arrays disagree: landmarks={landmarks}, world={world}, handedness={handedness}"
    )]
    HandParallelArrays {
        /// Number of normalized landmark sets.
        landmarks: usize,
        /// Number of world landmark sets.
        world: usize,
        /// Number of handedness classifications.
        handedness: usize,
    },
    /// A detected hand does not have the 21-landmark schema.
    #[error("expected {expected} hand landmarks for hand {hand}, received {actual}")]
    HandLandmarkCount {
        /// Detected hand index.
        hand: usize,
        /// Hand Landmarker's fixed count.
        expected: usize,
        /// Number actually supplied.
        actual: usize,
    },
    /// A hand landmark contains non-finite data or an invalid quality score.
    #[error("invalid hand landmark {index} for hand {hand}")]
    InvalidHandLandmark {
        /// Detected hand index.
        hand: usize,
        /// Index in the Hand Landmarker schema.
        index: usize,
    },
    /// A handedness classification has a non-finite or out-of-range score.
    #[error("invalid handedness score {score} for hand {hand}")]
    InvalidHandScore {
        /// Detected hand index.
        hand: usize,
        /// Offending score.
        score: f32,
    },
}

/// One decoded Pose result plus the normalized wrists used to side-pair hands.
pub struct DecodedPose {
    /// The arm chain observation, or `None` for a completed no-person result.
    pub observation: Option<PoseArmObservation>,
    /// Normalized image wrist position per side, `[left, right]`.
    pub wrist_xy: [Option<[f32; 2]>; 2],
}

/// Maps one owned MediaPipe Pose result to the engine-neutral arm observation.
///
/// The task runs with `num_poses = 1`, so an empty world-landmark array is a
/// normal no-person result and more than one person is malformed external data,
/// not a person to pick arbitrarily. Missing quality scores stay missing.
pub fn decode_pose_result(result: &PoseLandmarkerResult) -> Result<DecodedPose, PoseDecodeError> {
    let person = match result.pose_world_landmarks.as_slice() {
        [] => {
            return Ok(DecodedPose {
                observation: None,
                wrist_xy: [None, None],
            });
        }
        [person] => person,
        people => {
            return Err(PoseDecodeError::PersonCount {
                actual: people.len(),
            });
        }
    };
    let world: Vec<PoseWorldLandmark> = person
        .iter()
        .map(|landmark| PoseWorldLandmark {
            meters: landmark.point.to_array(),
            visibility: landmark.visibility.map(|value| value.get()),
            presence: landmark.presence.map(|value| value.get()),
        })
        .collect();
    let observation = decode_pose_arms(&world)?;
    let wrist_xy = result
        .pose_landmarks
        .first()
        .map(|normalized| {
            [
                normalized_wrist_xy(normalized, 15),
                normalized_wrist_xy(normalized, 16),
            ]
        })
        .unwrap_or([None, None]);
    Ok(DecodedPose {
        observation: Some(observation),
        wrist_xy,
    })
}

/// Selects shoulders 11/12, elbows 13/14, and wrists 15/16, without mirroring.
///
/// Hand landmarks are attached later by [`decode_hand_result`] from a separate
/// Hand Landmarker result over the same image.
///
/// The FFI adapter must supply WORLD landmarks, preserving optional scores.
/// It handles the zero-person case before calling this function. Image-space
/// landmarks are not accepted here and no face-shaped intermediate is used.
/// Low visibility is preserved: deciding how to animate an occluded arm belongs
/// to tracking, not inference. Only malformed external data is rejected here.
pub fn decode_pose_arms(
    world: &[PoseWorldLandmark],
) -> Result<PoseArmObservation, PoseDecodeError> {
    if world.len() != POSE_LANDMARK_COUNT {
        return Err(PoseDecodeError::LandmarkCount {
            actual: world.len(),
        });
    }
    let point = |index: usize| -> Result<PoseWorldLandmark, PoseDecodeError> {
        let value = world
            .get(index)
            .copied()
            .ok_or(PoseDecodeError::LandmarkCount {
                actual: world.len(),
            })?;
        let coordinates_valid = value.meters.iter().all(|v| v.is_finite());
        let scores_valid = [value.visibility, value.presence]
            .into_iter()
            .flatten()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(&v));
        if !coordinates_valid || !scores_valid {
            return Err(PoseDecodeError::InvalidLandmark { index });
        }
        Ok(value)
    };
    Ok(PoseArmObservation {
        left: ArmLandmarks {
            shoulder: point(11)?,
            elbow: point(13)?,
            wrist: point(15)?,
            hand: None,
        },
        right: ArmLandmarks {
            shoulder: point(12)?,
            elbow: point(14)?,
            wrist: point(16)?,
            hand: None,
        },
    })
}

/// Validates and copies one hand's world landmarks into the engine-neutral form.
///
/// # Errors
///
/// Returns [`PoseDecodeError::HandLandmarkCount`] for a schema mismatch and
/// [`PoseDecodeError::InvalidHandLandmark`] for non-finite or out-of-range
/// data.
pub fn decode_hand_world(
    hand: usize,
    world: &[WorldLandmark],
    score: Option<f32>,
) -> Result<HandWorldLandmarks, PoseDecodeError> {
    if world.len() != HAND_LANDMARK_COUNT {
        return Err(PoseDecodeError::HandLandmarkCount {
            hand,
            expected: HAND_LANDMARK_COUNT,
            actual: world.len(),
        });
    }
    let mut landmarks = [PoseWorldLandmark {
        meters: [0.0; 3],
        visibility: None,
        presence: None,
    }; HAND_LANDMARK_COUNT];
    for (index, (slot, source)) in landmarks.iter_mut().zip(world.iter()).enumerate() {
        let meters = source.point.to_array();
        let coordinates_valid = meters.iter().all(|value| value.is_finite());
        let scores_valid = [source.visibility, source.presence]
            .into_iter()
            .flatten()
            .all(|value| {
                let value = value.get();
                value.is_finite() && (0.0..=1.0).contains(&value)
            });
        if !coordinates_valid || !scores_valid {
            return Err(PoseDecodeError::InvalidHandLandmark { hand, index });
        }
        *slot = PoseWorldLandmark {
            meters,
            visibility: source.visibility.map(|value| value.get()),
            presence: source.presence.map(|value| value.get()),
        };
    }
    if let Some(score) = score
        && (!score.is_finite() || !(0.0..=1.0).contains(&score))
    {
        return Err(PoseDecodeError::InvalidHandScore { hand, score });
    }
    Ok(HandWorldLandmarks { landmarks, score })
}

/// Pairs every detected hand with the Pose wrist nearest to it and returns one
/// hand observation per anatomical side.
///
/// Pairing uses the normalized image positions of the Pose wrists and the
/// hand's own wrist, so it does not depend on MediaPipe's handedness label being
/// computed for mirrored input. Higher-scoring hands claim their nearest free
/// side first; a hand farther than [`HAND_WRIST_PAIR_DISTANCE`] from every free
/// wrist is ignored rather than assigned to a side it does not belong to.
///
/// # Errors
///
/// Returns [`PoseDecodeError`] when a detected hand's parallel arrays disagree,
/// a hand does not carry the 21-landmark schema, or a landmark or score is
/// non-finite or out of range.
pub fn decode_hand_result(
    result: &HandLandmarkerResult,
    wrist_xy: &[Option<[f32; 2]>; 2],
) -> Result<[Option<HandWorldLandmarks>; 2], PoseDecodeError> {
    if result.hand_landmarks.is_empty()
        && result.hand_world_landmarks.is_empty()
        && result.handedness.is_empty()
    {
        return Ok([None, None]);
    }
    if result.hand_landmarks.len() != result.hand_world_landmarks.len()
        || result.hand_landmarks.len() != result.handedness.len()
    {
        return Err(PoseDecodeError::HandParallelArrays {
            landmarks: result.hand_landmarks.len(),
            world: result.hand_world_landmarks.len(),
            handedness: result.handedness.len(),
        });
    }

    let mut candidates = Vec::with_capacity(result.hand_landmarks.len());
    for hand in 0..result.hand_landmarks.len() {
        let Some(normalized) = result.hand_landmarks.get(hand) else {
            continue;
        };
        let Some(world) = result.hand_world_landmarks.get(hand) else {
            continue;
        };
        let Some(normalized_wrist) = normalized.first() else {
            return Err(PoseDecodeError::HandLandmarkCount {
                hand,
                expected: HAND_LANDMARK_COUNT,
                actual: 0,
            });
        };
        let wrist = [normalized_wrist.point.x(), normalized_wrist.point.y()];
        if !wrist.iter().all(|value| value.is_finite()) {
            return Err(PoseDecodeError::InvalidHandLandmark { hand, index: 0 });
        }
        let score = result
            .handedness
            .get(hand)
            .and_then(|categories| categories.first())
            .map(|category| category.score.get());
        let hand_observation = decode_hand_world(hand, world, score)?;
        let distances = std::array::from_fn(|side| {
            wrist_xy
                .get(side)
                .copied()
                .flatten()
                .map(|pose_wrist| (pose_wrist[0] - wrist[0]).hypot(pose_wrist[1] - wrist[1]))
        });
        candidates.push(HandCandidate {
            distances,
            score: score.unwrap_or(0.0),
            hand: hand_observation,
        });
    }

    Ok(assign_hands(candidates, HAND_WRIST_PAIR_DISTANCE))
}

/// Gives each detected hand its nearest free side, best-scoring hand first.
fn assign_hands(
    mut candidates: Vec<HandCandidate>,
    max_distance: f32,
) -> [Option<HandWorldLandmarks>; 2] {
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut hands: [Option<HandWorldLandmarks>; 2] = [None, None];
    for candidate in candidates {
        let best = candidate
            .distances
            .iter()
            .enumerate()
            .filter(|(side, _)| hands.get(*side).is_some_and(|slot| slot.is_none()))
            .filter_map(|(side, distance)| distance.map(|distance| (side, distance)))
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((side, distance)) = best
            && distance <= max_distance
            && let Some(slot) = hands.get_mut(side)
        {
            *slot = Some(candidate.hand);
        }
    }
    hands
}

struct HandCandidate {
    distances: [Option<f32>; 2],
    score: f32,
    hand: HandWorldLandmarks,
}

fn normalized_wrist_xy(
    normalized: &[mediapipe::NormalizedLandmark],
    index: usize,
) -> Option<[f32; 2]> {
    let landmark = normalized.get(index)?;
    let xy = [landmark.point.x(), landmark.point.y()];
    xy.iter().all(|value| value.is_finite()).then_some(xy)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> [PoseWorldLandmark; POSE_LANDMARK_COUNT] {
        std::array::from_fn(|i| PoseWorldLandmark {
            meters: [i as f32, 2.0, -3.0],
            visibility: Some(0.7),
            presence: None,
        })
    }

    #[test]
    fn anatomical_indices_and_meter_values_are_preserved() {
        let points = fixture();
        let arms = decode_pose_arms(&points).unwrap();
        assert_eq!(arms.left.shoulder, points[11]);
        assert_eq!(arms.right.shoulder, points[12]);
        assert_eq!(arms.left.elbow, points[13]);
        assert_eq!(arms.right.elbow, points[14]);
        assert_eq!(arms.left.wrist, points[15]);
        assert_eq!(arms.right.wrist, points[16]);
        assert_eq!(arms.left.hand, None);
        assert_eq!(arms.right.hand, None);
    }

    #[test]
    fn absent_and_low_scores_are_not_replaced_with_high_confidence() {
        let mut points = fixture();
        points[13].visibility = None;
        points[15].visibility = Some(0.01);
        let arms = decode_pose_arms(&points).unwrap();
        assert_eq!(arms.left.elbow.visibility, None);
        assert_eq!(arms.left.wrist.visibility, Some(0.01));
        assert_eq!(arms.left.wrist.presence, None);
    }

    #[test]
    fn wrong_schema_is_an_error_not_a_default_arm() {
        assert_eq!(
            decode_pose_arms(&[]),
            Err(PoseDecodeError::LandmarkCount { actual: 0 })
        );
        assert!(decode_pose_arms(&fixture()[..32]).is_err());
        let mut extra = fixture().to_vec();
        extra.push(extra[0]);
        assert!(decode_pose_arms(&extra).is_err());
    }

    #[test]
    fn malformed_selected_coordinates_and_scores_are_errors() {
        for index in 11..=16 {
            let mut points = fixture();
            points[index].meters[2] = f32::NAN;
            assert_eq!(
                decode_pose_arms(&points),
                Err(PoseDecodeError::InvalidLandmark { index })
            );
            points = fixture();
            points[index].presence = Some(1.1);
            assert_eq!(
                decode_pose_arms(&points),
                Err(PoseDecodeError::InvalidLandmark { index })
            );
        }
    }

    fn world_landmark(x: f32) -> mediapipe::WorldLandmark {
        mediapipe::WorldLandmark {
            point: mediapipe::WorldPoint3::from_meters(x, 1.0, 2.0),
            visibility: Some(mediapipe::Confidence::new(0.9).unwrap()),
            presence: None,
            name: None,
        }
    }

    fn one_person() -> mediapipe::PoseLandmarkerResult {
        mediapipe::PoseLandmarkerResult {
            pose_landmarks: Vec::new(),
            pose_world_landmarks: vec![
                (0..POSE_LANDMARK_COUNT)
                    .map(|index| world_landmark(index as f32))
                    .collect(),
            ],
        }
    }

    #[test]
    fn no_person_result_maps_to_none() {
        let decoded = decode_pose_result(&mediapipe::PoseLandmarkerResult::default()).unwrap();
        assert!(decoded.observation.is_none());
        assert_eq!(decoded.wrist_xy, [None, None]);
    }

    #[test]
    fn one_person_maps_world_meters_and_keeps_missing_scores() {
        let decoded = decode_pose_result(&one_person()).unwrap();
        let arms = decoded.observation.unwrap();
        assert_eq!(arms.left.shoulder.meters, [11.0, 1.0, 2.0]);
        assert_eq!(arms.right.wrist.meters, [16.0, 1.0, 2.0]);
        assert_eq!(arms.left.shoulder.visibility, Some(0.9));
        assert_eq!(arms.left.shoulder.presence, None);
        assert_eq!(decoded.wrist_xy, [None, None]);
    }

    #[test]
    fn more_than_one_person_is_malformed_not_an_arbitrary_pick() {
        let mut result = one_person();
        result
            .pose_world_landmarks
            .push(result.pose_world_landmarks[0].clone());
        assert!(matches!(
            decode_pose_result(&result),
            Err(PoseDecodeError::PersonCount { actual: 2 })
        ));
    }

    #[test]
    fn a_person_with_the_wrong_schema_is_an_error() {
        let result = mediapipe::PoseLandmarkerResult {
            pose_landmarks: Vec::new(),
            pose_world_landmarks: vec![vec![world_landmark(0.0); 3]],
        };
        assert!(matches!(
            decode_pose_result(&result),
            Err(PoseDecodeError::LandmarkCount { actual: 3 })
        ));
    }

    fn hand_world(y: f32) -> Vec<mediapipe::WorldLandmark> {
        (0..HAND_LANDMARK_COUNT)
            .map(|index| mediapipe::WorldLandmark {
                point: mediapipe::WorldPoint3::from_meters(index as f32 * 0.01, y, 0.0),
                visibility: None,
                presence: None,
                name: None,
            })
            .collect()
    }

    #[test]
    fn empty_hand_result_attaches_nothing() {
        let hands = decode_hand_result(&HandLandmarkerResult::default(), &[None, None]).unwrap();
        assert_eq!(hands, [None, None]);
    }

    #[test]
    fn disagreeing_hand_arrays_are_malformed() {
        let result = HandLandmarkerResult {
            handedness: Vec::new(),
            hand_landmarks: Vec::new(),
            hand_world_landmarks: vec![hand_world(0.0)],
        };
        assert!(matches!(
            decode_hand_result(&result, &[None, None]),
            Err(PoseDecodeError::HandParallelArrays { .. })
        ));
    }

    #[test]
    fn hand_world_landmarks_keep_meters_and_validate_the_schema() {
        let hand = decode_hand_world(0, &hand_world(0.5), Some(0.9)).unwrap();
        let mcp = hand.landmarks[5].meters;
        assert!((mcp[0] - 0.05).abs() < 1.0e-6 && mcp[1] == 0.5 && mcp[2] == 0.0);
        assert_eq!(hand.score, Some(0.9));
        assert!(matches!(
            decode_hand_world(0, &hand_world(0.5)[..20], Some(0.9)),
            Err(PoseDecodeError::HandLandmarkCount { .. })
        ));
        assert!(matches!(
            decode_hand_world(0, &hand_world(0.5), Some(1.5)),
            Err(PoseDecodeError::InvalidHandScore { .. })
        ));
    }

    fn candidate(distances: [Option<f32>; 2], score: f32) -> HandCandidate {
        HandCandidate {
            distances,
            score,
            hand: decode_hand_world(0, &hand_world(score), Some(score)).unwrap(),
        }
    }

    #[test]
    fn pairing_claims_the_nearest_free_side_and_ignores_far_hands() {
        let hands = assign_hands(
            vec![
                candidate([Some(0.02), Some(0.40)], 0.9),
                candidate([Some(0.03), Some(0.05)], 0.8),
            ],
            0.15,
        );
        assert!(hands[0].is_some());
        assert!(hands[1].is_some());
        // The first hand takes the left side (closest); the second then takes
        // the remaining right side even though its left distance is smaller.
        assert_eq!(hands[0].unwrap().score, Some(0.9));
        assert_eq!(hands[1].unwrap().score, Some(0.8));

        let far = assign_hands(vec![candidate([Some(0.60), Some(0.70)], 0.9)], 0.15);
        assert_eq!(far, [None, None]);
    }

    #[test]
    fn higher_scoring_hand_claims_its_side_first() {
        let hands = assign_hands(
            vec![
                candidate([Some(0.10), None], 0.5),
                candidate([Some(0.02), None], 0.9),
            ],
            0.15,
        );
        assert_eq!(hands[0].unwrap().score, Some(0.9));
        assert_eq!(hands[1], None);
    }
}
