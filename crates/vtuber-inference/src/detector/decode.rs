//! UltraFace output decoding and single-user face selection.

use std::cmp::Ordering;

use thiserror::Error;
use vtuber_core::types::NormalizedRect;

use crate::descriptor::DetectorPostprocessConfig;
use crate::detector::nms::{FaceDetection, hard_nms, intersection_over_union};
use crate::detector::runtime::{DetectorRawOutputs, DetectorRawTensor};

const SCORES_OUTPUT_NAME: &str = "scores";
const BOXES_OUTPUT_NAME: &str = "boxes";
const SCORE_COMPONENTS: usize = 2;
const BOX_COMPONENTS: usize = 4;
const FACE_SCORE_INDEX: usize = 1;
const PRIMARY_CONTINUITY_IOU: f32 = 0.2;
const BOX_COORDINATE_TOLERANCE: f32 = 0.05;
/// Production bound for candidates entering NMS.
pub const MAX_PRE_NMS_CANDIDATES: usize = 256;
/// Production bound for detections retained after NMS.
pub const MAX_POST_NMS_DETECTIONS: usize = 16;

/// Typed failures while resolving and decoding detector outputs.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum DetectorDecodeError {
    /// The postprocess contract contains an invalid value.
    #[error("invalid detector postprocess config: {field}={value}")]
    InvalidPostprocessConfig {
        /// Configuration field name.
        field: &'static str,
        /// Invalid value rendered for diagnostics.
        value: String,
    },
    /// The raw output list does not contain exactly one required tensor.
    #[error("detector output {name} is missing or duplicated")]
    OutputNameContract {
        /// Required output name.
        name: &'static str,
    },
    /// An unexpected output tensor was supplied.
    #[error("unexpected detector output {name}")]
    UnexpectedOutput {
        /// Unexpected runtime output name.
        name: String,
    },
    /// A detector tensor shape is not compatible with the fixed contract.
    #[error("detector output {name} has invalid shape {shape:?}")]
    OutputShape {
        /// Output name.
        name: &'static str,
        /// Actual shape.
        shape: Vec<usize>,
    },
    /// A detector tensor's value count does not match its shape.
    #[error("detector output {name} has {actual} values, expected {expected}")]
    OutputLength {
        /// Output name.
        name: &'static str,
        /// Actual number of values.
        actual: usize,
        /// Expected number of values.
        expected: usize,
    },
    /// A score is not finite or outside the score contract.
    #[error("detector score at anchor {anchor_index} is invalid: {value}")]
    InvalidScore {
        /// Anchor index.
        anchor_index: usize,
        /// Invalid score.
        value: f32,
    },
    /// A selected face box is malformed or too far outside the source frame.
    #[error("detector box at anchor {anchor_index} is invalid: {reason}")]
    InvalidBox {
        /// Anchor index.
        anchor_index: usize,
        /// Validation reason.
        reason: &'static str,
    },
}

/// The normal detector observation or the ordinary no-face state.
#[derive(Clone, Debug, PartialEq)]
pub enum DetectorDecodeOutcome {
    /// No candidate met the configured face threshold.
    NoFace,
    /// Bounded, NMS-filtered face detections in descending confidence order.
    Detections(Vec<FaceDetection>),
}

/// Decodes UltraFace outputs and applies manifest-driven thresholding and NMS.
///
/// The accepted UltraFace graph already emits decoded axis-aligned boxes in
/// normalized detector-image coordinates. Because preprocessing uses direct
/// resize (not letterboxing), those normalized coordinates are also normalized
/// source-image coordinates. The returned rectangles are never mirrored.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
pub fn decode_detections(
    outputs: &DetectorRawOutputs,
    config: DetectorPostprocessConfig,
) -> Result<DetectorDecodeOutcome, DetectorDecodeError> {
    validate_config(config)?;
    let (scores, boxes) = resolve_outputs(outputs)?;
    let anchor_count = validate_output_shapes(scores, boxes)?;
    let pre_nms_limit = config.max_pre_nms_candidates.min(MAX_PRE_NMS_CANDIDATES);
    let mut candidates = Vec::with_capacity(pre_nms_limit.min(anchor_count));

    for anchor_index in 0..anchor_count {
        let score = scores.values[anchor_index * SCORE_COMPONENTS + FACE_SCORE_INDEX];
        if !score.is_finite() || !(0.0..=1.0).contains(&score) {
            return Err(DetectorDecodeError::InvalidScore {
                anchor_index,
                value: score,
            });
        }
        if score < config.score_threshold || pre_nms_limit == 0 {
            continue;
        }

        let box_offset = anchor_index * BOX_COMPONENTS;
        let rect = normalized_box(
            &boxes.values[box_offset..box_offset + BOX_COMPONENTS],
            anchor_index,
        )?;
        insert_top_candidate(
            &mut candidates,
            FaceDetection {
                rect,
                confidence: score,
                anchor_index,
            },
            pre_nms_limit,
        );
    }

    if candidates.is_empty() {
        return Ok(DetectorDecodeOutcome::NoFace);
    }

    candidates.sort_unstable_by(detection_order);
    let detections = hard_nms(
        &candidates,
        config.nms_iou,
        config.max_post_nms_detections.min(MAX_POST_NMS_DETECTIONS),
    );
    if detections.is_empty() {
        Ok(DetectorDecodeOutcome::NoFace)
    } else {
        Ok(DetectorDecodeOutcome::Detections(detections))
    }
}

/// Selects the single face used by the single-user pipeline.
///
/// When a previous ROI overlaps a candidate by at least `0.2`, continuity is
/// preferred: maximum IoU wins and confidence breaks ties. If no candidate is
/// continuous, maximum confidence wins and area breaks ties.
#[must_use]
pub fn select_primary_face(
    detections: &[FaceDetection],
    previous_roi: Option<&NormalizedRect>,
) -> Option<FaceDetection> {
    if let Some(previous) = previous_roi {
        let continuous = detections
            .iter()
            .copied()
            .filter_map(|detection| {
                let iou = intersection_over_union(&detection.rect, previous);
                (iou >= PRIMARY_CONTINUITY_IOU).then_some((iou, detection))
            })
            .max_by(|(left_iou, left), (right_iou, right)| {
                left_iou
                    .total_cmp(right_iou)
                    .then_with(|| left.confidence.total_cmp(&right.confidence))
                    .then_with(|| right.anchor_index.cmp(&left.anchor_index))
            });
        if let Some((_, detection)) = continuous {
            return Some(detection);
        }
    }

    detections.iter().copied().max_by(|left, right| {
        left.confidence
            .total_cmp(&right.confidence)
            .then_with(|| area(&left.rect).total_cmp(&area(&right.rect)))
            .then_with(|| right.anchor_index.cmp(&left.anchor_index))
    })
}

fn validate_config(config: DetectorPostprocessConfig) -> Result<(), DetectorDecodeError> {
    if !config.score_threshold.is_finite() || !(0.0..=1.0).contains(&config.score_threshold) {
        return Err(DetectorDecodeError::InvalidPostprocessConfig {
            field: "score_threshold",
            value: config.score_threshold.to_string(),
        });
    }
    if !config.nms_iou.is_finite() || !(0.0..=1.0).contains(&config.nms_iou) {
        return Err(DetectorDecodeError::InvalidPostprocessConfig {
            field: "nms_iou",
            value: config.nms_iou.to_string(),
        });
    }
    Ok(())
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn resolve_outputs(
    outputs: &DetectorRawOutputs,
) -> Result<(&DetectorRawTensor, &DetectorRawTensor), DetectorDecodeError> {
    if outputs.tensors.len() != 2 {
        return Err(DetectorDecodeError::OutputNameContract {
            name: "scores and boxes",
        });
    }

    for tensor in &outputs.tensors {
        if tensor.name != SCORES_OUTPUT_NAME && tensor.name != BOXES_OUTPUT_NAME {
            return Err(DetectorDecodeError::UnexpectedOutput {
                name: tensor.name.clone(),
            });
        }
    }

    let scores = outputs
        .tensors
        .iter()
        .filter(|tensor| tensor.name == SCORES_OUTPUT_NAME)
        .collect::<Vec<_>>();
    let boxes = outputs
        .tensors
        .iter()
        .filter(|tensor| tensor.name == BOXES_OUTPUT_NAME)
        .collect::<Vec<_>>();
    if scores.len() != 1 {
        return Err(DetectorDecodeError::OutputNameContract {
            name: SCORES_OUTPUT_NAME,
        });
    }
    if boxes.len() != 1 {
        return Err(DetectorDecodeError::OutputNameContract {
            name: BOXES_OUTPUT_NAME,
        });
    }
    Ok((scores[0], boxes[0]))
}

fn validate_output_shapes(
    scores: &DetectorRawTensor,
    boxes: &DetectorRawTensor,
) -> Result<usize, DetectorDecodeError> {
    let score_count = validate_tensor_shape(scores, SCORES_OUTPUT_NAME, SCORE_COMPONENTS)?;
    let box_count = validate_tensor_shape(boxes, BOXES_OUTPUT_NAME, BOX_COMPONENTS)?;
    if score_count != box_count {
        return Err(DetectorDecodeError::OutputShape {
            name: BOXES_OUTPUT_NAME,
            shape: boxes.shape.clone(),
        });
    }
    Ok(score_count)
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn validate_tensor_shape(
    tensor: &DetectorRawTensor,
    name: &'static str,
    components: usize,
) -> Result<usize, DetectorDecodeError> {
    if tensor.shape.len() != 3 {
        return Err(DetectorDecodeError::OutputShape {
            name,
            shape: tensor.shape.clone(),
        });
    }
    let [batch, count, width] = [tensor.shape[0], tensor.shape[1], tensor.shape[2]];
    if batch != 1 || width != components || count == 0 {
        return Err(DetectorDecodeError::OutputShape {
            name,
            shape: tensor.shape.clone(),
        });
    }
    let Some(expected) = count.checked_mul(components) else {
        return Err(DetectorDecodeError::OutputLength {
            name,
            actual: tensor.values.len(),
            expected: usize::MAX,
        });
    };
    if tensor.values.len() != expected {
        return Err(DetectorDecodeError::OutputLength {
            name,
            actual: tensor.values.len(),
            expected,
        });
    }
    Ok(count)
}

fn normalized_box(
    values: &[f32],
    anchor_index: usize,
) -> Result<NormalizedRect, DetectorDecodeError> {
    let [x_min, y_min, x_max, y_max] = values else {
        return Err(DetectorDecodeError::InvalidBox {
            anchor_index,
            reason: "box component count",
        });
    };
    if values.iter().any(|value| !value.is_finite()) {
        return Err(DetectorDecodeError::InvalidBox {
            anchor_index,
            reason: "non-finite coordinate",
        });
    }
    if *x_min >= *x_max || *y_min >= *y_max {
        return Err(DetectorDecodeError::InvalidBox {
            anchor_index,
            reason: "reversed or zero-area box",
        });
    }
    if [*x_min, *y_min, *x_max, *y_max]
        .iter()
        .any(|value| *value < -BOX_COORDINATE_TOLERANCE || *value > 1.0 + BOX_COORDINATE_TOLERANCE)
    {
        return Err(DetectorDecodeError::InvalidBox {
            anchor_index,
            reason: "coordinate too far outside normalized frame",
        });
    }

    let x_min = x_min.clamp(0.0, 1.0);
    let y_min = y_min.clamp(0.0, 1.0);
    let x_max = x_max.clamp(0.0, 1.0);
    let y_max = y_max.clamp(0.0, 1.0);
    if x_min >= x_max || y_min >= y_max {
        return Err(DetectorDecodeError::InvalidBox {
            anchor_index,
            reason: "box has no area after clamping",
        });
    }
    Ok(NormalizedRect {
        x: x_min,
        y: y_min,
        width: x_max - x_min,
        height: y_max - y_min,
        rotation_rad: 0.0,
    })
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn insert_top_candidate(
    candidates: &mut Vec<FaceDetection>,
    candidate: FaceDetection,
    limit: usize,
) {
    if limit == 0 {
        return;
    }
    if candidates.len() < limit {
        candidates.push(candidate);
        return;
    }

    let worst_index = candidates
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| detection_order(left, right))
        .map(|(index, _)| index);
    if let Some(worst_index) = worst_index
        && detection_order(&candidate, &candidates[worst_index]) == Ordering::Less
    {
        candidates[worst_index] = candidate;
    }
}

fn detection_order(left: &FaceDetection, right: &FaceDetection) -> Ordering {
    right
        .confidence
        .total_cmp(&left.confidence)
        .then_with(|| left.anchor_index.cmp(&right.anchor_index))
}

fn area(rect: &NormalizedRect) -> f32 {
    rect.width * rect.height
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> DetectorPostprocessConfig {
        DetectorPostprocessConfig {
            score_threshold: 0.7,
            nms_iou: 0.3,
            max_pre_nms_candidates: 256,
            max_post_nms_detections: 16,
        }
    }

    fn outputs(scores: &[(f32, f32)], boxes: &[[f32; 4]]) -> DetectorRawOutputs {
        let score_values = scores
            .iter()
            .flat_map(|&(background, face)| [background, face])
            .collect();
        let box_values = boxes.iter().flat_map(|item| item.iter().copied()).collect();
        DetectorRawOutputs {
            tensors: vec![
                DetectorRawTensor {
                    name: "boxes".into(),
                    shape: vec![1, boxes.len(), 4],
                    values: box_values,
                },
                DetectorRawTensor {
                    name: "scores".into(),
                    shape: vec![1, scores.len(), 2],
                    values: score_values,
                },
            ],
        }
    }

    #[test]
    fn decode_resolves_named_outputs_and_applies_threshold() {
        let raw = outputs(
            &[(0.1, 0.69), (0.1, 0.9)],
            &[[0.0, 0.0, 0.2, 0.2], [0.1, 0.2, 1.02, 0.8]],
        );

        let result = decode_detections(&raw, config()).unwrap();

        let DetectorDecodeOutcome::Detections(detections) = result else {
            panic!("expected one detection");
        };
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].anchor_index, 1);
        assert_eq!(detections[0].rect.x, 0.1);
        assert_eq!(detections[0].rect.width, 0.9);
    }

    #[test]
    fn decode_returns_no_face_without_threshold_match() {
        let raw = outputs(&[(0.9, 0.1)], &[[0.0, 0.0, 0.2, 0.2]]);

        assert_eq!(
            decode_detections(&raw, config()).unwrap(),
            DetectorDecodeOutcome::NoFace
        );
    }

    #[test]
    fn decode_rejects_malformed_selected_box() {
        let raw = outputs(&[(0.1, 0.9)], &[[0.3, 0.2, 0.1, 0.4]]);

        assert!(matches!(
            decode_detections(&raw, config()),
            Err(DetectorDecodeError::InvalidBox { .. })
        ));
    }

    #[test]
    fn primary_selection_preserves_continuity_over_confidence() {
        let detections = [
            FaceDetection {
                rect: NormalizedRect {
                    x: 0.0,
                    y: 0.0,
                    width: 0.4,
                    height: 0.4,
                    rotation_rad: 0.0,
                },
                confidence: 0.8,
                anchor_index: 1,
            },
            FaceDetection {
                rect: NormalizedRect {
                    x: 0.55,
                    y: 0.55,
                    width: 0.35,
                    height: 0.35,
                    rotation_rad: 0.0,
                },
                confidence: 0.95,
                anchor_index: 2,
            },
        ];
        let previous = NormalizedRect {
            x: 0.02,
            y: 0.02,
            width: 0.4,
            height: 0.4,
            rotation_rad: 0.0,
        };

        assert_eq!(
            select_primary_face(&detections, Some(&previous))
                .unwrap()
                .anchor_index,
            1
        );
    }

    #[test]
    fn primary_selection_falls_back_to_confidence_then_area() {
        let detections = [
            FaceDetection {
                rect: NormalizedRect {
                    x: 0.0,
                    y: 0.0,
                    width: 0.3,
                    height: 0.3,
                    rotation_rad: 0.0,
                },
                confidence: 0.8,
                anchor_index: 1,
            },
            FaceDetection {
                rect: NormalizedRect {
                    x: 0.5,
                    y: 0.5,
                    width: 0.4,
                    height: 0.4,
                    rotation_rad: 0.0,
                },
                confidence: 0.8,
                anchor_index: 2,
            },
        ];

        assert_eq!(
            select_primary_face(&detections, None).unwrap().anchor_index,
            2
        );
    }
}
