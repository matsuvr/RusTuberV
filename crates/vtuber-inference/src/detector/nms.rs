//! Bounded, deterministic non-maximum suppression for face detections.

use std::cmp::Ordering;

use vtuber_core::types::NormalizedRect;

const PRODUCTION_MAX_DETECTIONS: usize = 16;

/// One face box decoded into normalized source-image coordinates.
///
/// `rect` uses the unmirrored source image convention: `(0, 0)` is the
/// top-left, width and height are normalized to the source frame, and
/// `rotation_rad` is zero for UltraFace axis-aligned boxes. `confidence` is
/// the face-class score in `[0, 1]`. `anchor_index` is retained only to make
/// tie-breaking deterministic and for diagnostics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceDetection {
    /// Face box in normalized source-image coordinates.
    pub rect: NormalizedRect,
    /// Face-class confidence in `[0, 1]`.
    pub confidence: f32,
    /// Index of the originating UltraFace anchor.
    pub anchor_index: usize,
}

/// Computes intersection over union for two axis-aligned normalized boxes.
#[must_use]
pub fn intersection_over_union(first: &NormalizedRect, second: &NormalizedRect) -> f32 {
    if !is_finite_rect(first) || !is_finite_rect(second) {
        return 0.0;
    }

    let first_right = first.x + first.width;
    let first_bottom = first.y + first.height;
    let second_right = second.x + second.width;
    let second_bottom = second.y + second.height;
    let intersection_width = (first_right.min(second_right) - first.x.max(second.x)).max(0.0);
    let intersection_height = (first_bottom.min(second_bottom) - first.y.max(second.y)).max(0.0);
    let intersection = intersection_width * intersection_height;
    let union = first.width * first.height + second.width * second.height - intersection;

    if union > 0.0 && union.is_finite() {
        (intersection / union).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Applies hard NMS and returns at most `max_detections` boxes.
///
/// Candidates are processed by descending confidence, then ascending anchor
/// index. The returned vector has the same deterministic order and is capped
/// at the production maximum of 16 detections.
#[must_use]
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
pub fn hard_nms(
    candidates: &[FaceDetection],
    iou_threshold: f32,
    max_detections: usize,
) -> Vec<FaceDetection> {
    if max_detections == 0 || !iou_threshold.is_finite() {
        return Vec::new();
    }

    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_unstable_by(|&left, &right| detection_order(&candidates[left], &candidates[right]));

    let limit = max_detections.min(PRODUCTION_MAX_DETECTIONS);
    let mut selected: Vec<FaceDetection> = Vec::with_capacity(limit.min(order.len()));
    for index in order {
        let candidate = candidates[index];
        if selected
            .iter()
            .all(|kept| intersection_over_union(&candidate.rect, &kept.rect) <= iou_threshold)
        {
            selected.push(candidate);
            if selected.len() == limit {
                break;
            }
        }
    }
    selected
}

fn detection_order(left: &FaceDetection, right: &FaceDetection) -> Ordering {
    right
        .confidence
        .total_cmp(&left.confidence)
        .then_with(|| left.anchor_index.cmp(&right.anchor_index))
}

fn is_finite_rect(rect: &NormalizedRect) -> bool {
    rect.x.is_finite()
        && rect.y.is_finite()
        && rect.width.is_finite()
        && rect.height.is_finite()
        && rect.rotation_rad.is_finite()
        && rect.width >= 0.0
        && rect.height >= 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detection(
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        confidence: f32,
        index: usize,
    ) -> FaceDetection {
        FaceDetection {
            rect: NormalizedRect {
                x,
                y,
                width,
                height,
                rotation_rad: 0.0,
            },
            confidence,
            anchor_index: index,
        }
    }

    #[test]
    fn iou_matches_known_overlap() {
        let first = detection(0.0, 0.0, 0.5, 0.5, 0.9, 0);
        let second = detection(0.25, 0.25, 0.5, 0.5, 0.8, 1);

        assert!((intersection_over_union(&first.rect, &second.rect) - (1.0 / 7.0)).abs() < 1e-6);
    }

    #[test]
    fn hard_nms_keeps_highest_confidence_for_overlapping_boxes() {
        let candidates = [
            detection(0.0, 0.0, 0.6, 0.6, 0.8, 2),
            detection(0.05, 0.05, 0.6, 0.6, 0.9, 1),
            detection(0.7, 0.7, 0.2, 0.2, 0.7, 3),
        ];

        let kept = hard_nms(&candidates, 0.3, 16);

        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].anchor_index, 1);
        assert_eq!(kept[1].anchor_index, 3);
    }

    #[test]
    fn hard_nms_tie_breaks_by_anchor_index() {
        let candidates = [
            detection(0.0, 0.0, 0.2, 0.2, 0.8, 9),
            detection(0.3, 0.3, 0.2, 0.2, 0.8, 4),
        ];

        let kept = hard_nms(&candidates, 0.3, 16);

        assert_eq!(
            kept.iter()
                .map(|item| item.anchor_index)
                .collect::<Vec<_>>(),
            [4, 9]
        );
    }
}
