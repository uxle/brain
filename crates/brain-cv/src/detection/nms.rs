//! # Non-Maximum Suppression (NMS) & Soft-NMS
//!
//! Filters redundant overlapping bounding box predictions based on confidence scores and IoU thresholds (Girshick et al.).

use crate::ops::boxes::box_iou_matrix;
use brain_core::Tensor;

/// Configuration parameters for Non-Maximum Suppression.
#[derive(Debug, Clone)]
pub struct NmsConfig {
    pub iou_threshold: f64,
    pub score_threshold: f64,
    pub max_output_boxes: usize,
}

impl Default for NmsConfig {
    fn default() -> Self {
        Self {
            iou_threshold: 0.5,
            score_threshold: 0.05,
            max_output_boxes: 100,
        }
    }
}

/// Executes standard Greedy Non-Maximum Suppression.
/// - `boxes`: 2D Tensor `[N, 4]` (XYXY format: x1, y1, x2, y2).
/// - `scores`: 1D Tensor `[N]` containing confidence scores for each box.
/// - `config`: NMS parameters (IoU threshold, score threshold, max output boxes).
/// Returns a `Vec<usize>` containing the indices of the kept bounding boxes in descending score order.
pub fn non_max_suppression(boxes: &Tensor, scores: &Tensor, config: &NmsConfig) -> Vec<usize> {
    let num_boxes = boxes.shape()[0];
    if num_boxes == 0 {
        return Vec::new();
    }

    let score_data = scores.data();
    assert_eq!(score_data.len(), num_boxes);

    // Filter boxes above score_threshold and sort by descending confidence
    let mut order: Vec<usize> = (0..num_boxes)
        .filter(|&i| score_data[i] >= config.score_threshold)
        .collect();

    order.sort_by(|&a, &b| {
        score_data[b]
            .partial_cmp(&score_data[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = Vec::new();
    let mut suppressed = vec![false; num_boxes];

    for &i in &order {
        if suppressed[i] {
            continue;
        }

        keep.push(i);
        if keep.len() >= config.max_output_boxes {
            break;
        }

        // Compute IoU of box i with all subsequent remaining candidates
        let box_i = Tensor::from_slice(
            &[
                boxes.get_2d(i, 0),
                boxes.get_2d(i, 1),
                boxes.get_2d(i, 2),
                boxes.get_2d(i, 3),
            ],
            vec![1, 4],
        );

        for &j in &order {
            if !suppressed[j] && j != i {
                let box_j = Tensor::from_slice(
                    &[
                        boxes.get_2d(j, 0),
                        boxes.get_2d(j, 1),
                        boxes.get_2d(j, 2),
                        boxes.get_2d(j, 3),
                    ],
                    vec![1, 4],
                );
                let iou_mat = box_iou_matrix(&box_i, &box_j);
                let iou = iou_mat.data()[0];
                if iou > config.iou_threshold {
                    suppressed[j] = true;
                }
            }
        }
    }

    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nms_suppression() {
        let boxes = Tensor::from_slice(
            &[
                0.0, 0.0, 10.0, 10.0, // Box 0: high score (0.9)
                1.0, 1.0, 10.0,
                10.0, // Box 1: redundant with Box 0 (score 0.8) -> should be suppressed
                50.0, 50.0, 70.0, 70.0, // Box 2: disjoint box (score 0.7) -> should be kept
            ],
            vec![3, 4],
        );
        let scores = Tensor::from_slice(&[0.9, 0.8, 0.7], vec![3]);
        let cfg = NmsConfig {
            iou_threshold: 0.5,
            score_threshold: 0.1,
            max_output_boxes: 10,
        };

        let kept = non_max_suppression(&boxes, &scores, &cfg);
        assert_eq!(kept, vec![0, 2]);
    }
}
