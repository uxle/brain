//! # Object Detection Metrics
//!
//! IoU, Generalized IoU (GIoU), Distance IoU (DIoU), and Detection Precision/Recall curves.
#![allow(missing_docs)]

pub mod map;
pub use map::{mean_average_precision, MapConfig};

use crate::utils::stable_divide;

/// Configuration for detection evaluation.
#[derive(Debug, Clone, Default)]
pub struct DetMetricConfig {
    pub iou_threshold: f64,
}

/// Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2].
pub fn bbox_iou(box1: &[f64; 4], box2: &[f64; 4]) -> f64 {
    let inter_x1 = box1[0].max(box2[0]);
    let inter_y1 = box1[1].max(box2[1]);
    let inter_x2 = box1[2].min(box2[2]);
    let inter_y2 = box1[3].min(box2[3]);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let area1 = (box1[2] - box1[0]).max(0.0) * (box1[3] - box1[1]).max(0.0);
    let area2 = (box2[2] - box2[0]).max(0.0) * (box2[3] - box2[1]).max(0.0);
    let union_area = area1 + area2 - inter_area;

    stable_divide(inter_area, union_area, 0.0)
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
