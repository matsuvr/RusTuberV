//! Pure extraction at the Pose result boundary. No model execution or avatar policy.

use mediapipe::PoseLandmarkerResult;
use vtuber_core::arm_tracking::{ArmLandmarks, PoseArmObservation, PoseWorldLandmark};

/// MediaPipe Pose's fixed world-landmark count.
pub const POSE_LANDMARK_COUNT: usize = 33;

/// Malformed task output, distinct from successful inference finding no person.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
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
}

/// Maps one owned MediaPipe Pose result to the engine-neutral arm observation.
///
/// The task runs with `num_poses = 1`, so an empty world-landmark array is a
/// normal no-person result and more than one person is malformed external data,
/// not a person to pick arbitrarily. Missing quality scores stay missing.
pub fn decode_pose_result(
    result: &PoseLandmarkerResult,
) -> Result<Option<PoseArmObservation>, PoseDecodeError> {
    let person = match result.pose_world_landmarks.as_slice() {
        [] => return Ok(None),
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
    decode_pose_arms(&world).map(Some)
}

/// Selects shoulders 11/12, elbows 13/14, and wrists 15/16, without mirroring.
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
        },
        right: ArmLandmarks {
            shoulder: point(12)?,
            elbow: point(14)?,
            wrist: point(16)?,
        },
    })
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
        assert_eq!(
            decode_pose_result(&mediapipe::PoseLandmarkerResult::default()),
            Ok(None)
        );
    }

    #[test]
    fn one_person_maps_world_meters_and_keeps_missing_scores() {
        let arms = decode_pose_result(&one_person()).unwrap().unwrap();
        assert_eq!(arms.left.shoulder.meters, [11.0, 1.0, 2.0]);
        assert_eq!(arms.right.wrist.meters, [16.0, 1.0, 2.0]);
        assert_eq!(arms.left.shoulder.visibility, Some(0.9));
        assert_eq!(arms.left.shoulder.presence, None);
    }

    #[test]
    fn more_than_one_person_is_malformed_not_an_arbitrary_pick() {
        let mut result = one_person();
        result
            .pose_world_landmarks
            .push(result.pose_world_landmarks[0].clone());
        assert_eq!(
            decode_pose_result(&result),
            Err(PoseDecodeError::PersonCount { actual: 2 })
        );
    }

    #[test]
    fn a_person_with_the_wrong_schema_is_an_error() {
        let result = mediapipe::PoseLandmarkerResult {
            pose_landmarks: Vec::new(),
            pose_world_landmarks: vec![vec![world_landmark(0.0); 3]],
        };
        assert_eq!(
            decode_pose_result(&result),
            Err(PoseDecodeError::LandmarkCount { actual: 3 })
        );
    }
}
