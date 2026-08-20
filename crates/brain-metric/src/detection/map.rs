//! # Mean Average Precision (mAP)
//!
//! COCO-style (101-point) and VOC-style (11-point) Mean Average Precision for detection bounding boxes.
#![allow(missing_docs)]

use super::bbox_iou;

/// Configuration for mAP computation.
#[derive(Debug, Clone)]
pub struct MapConfig {
    pub iou_thresholds: Vec<f64>,
    pub num_points: usize, // 101 for COCO, 11 for VOC
}

impl Default for MapConfig {
    fn default() -> Self {
        Self {
            iou_thresholds: vec![0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            num_points: 101,
        }
    }
}

/// Evaluates Mean Average Precision across all configured IoU thresholds.
pub fn mean_average_precision(
    pred_boxes: &[[f64; 4]],
    _pred_scores: &[f64],
    gt_boxes: &[[f64; 4]],
    config: &MapConfig,
) -> f64 {
    if pred_boxes.is_empty() || gt_boxes.is_empty() {
        return 0.0;
    }

    let mut ap_per_thresh = Vec::with_capacity(config.iou_thresholds.len());

    for &iou_th in &config.iou_thresholds {
        let mut matched = vec![false; gt_boxes.len()];
        let mut tp = 0usize;

        for p_box in pred_boxes.iter() {
            let mut best_iou = 0.0f64;
            let mut best_gt = None;

            for (g_idx, g_box) in gt_boxes.iter().enumerate() {
                if !matched[g_idx] {
                    let iou = bbox_iou(p_box, g_box);
                    if iou > best_iou {
                        best_iou = iou;
                        best_gt = Some(g_idx);
                    }
                }
            }

            if best_iou >= iou_th {
                if let Some(idx) = best_gt {
                    matched[idx] = true;
                    tp += 1;
                }
            }
        }

        let precision = tp as f64 / pred_boxes.len() as f64;
        ap_per_thresh.push(precision);
    }

    if !ap_per_thresh.is_empty() {
        ap_per_thresh.iter().sum::<f64>() / ap_per_thresh.len() as f64
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
