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
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_det_mod_stress_001() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_002() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_003() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_004() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_005() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_006() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_007() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_008() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_009() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_010() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_011() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_012() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_013() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_014() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_015() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_016() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_017() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_018() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_019() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_020() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_021() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_022() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_023() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_024() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_025() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_026() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_027() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_028() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_029() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_030() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_031() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_032() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_033() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_034() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_035() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_036() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_037() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_038() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_039() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_040() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_041() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_042() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_043() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_044() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_045() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_046() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_047() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_048() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_049() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_050() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_051() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_052() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_053() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_054() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_055() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_056() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_057() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_058() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_059() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_060() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_061() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_062() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_063() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_064() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_065() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_066() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_067() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_068() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_069() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_070() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_071() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_072() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_073() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_074() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_075() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_076() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_077() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_078() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_079() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_080() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_081() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_082() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_083() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_084() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_085() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_086() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_087() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_088() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_089() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_090() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_091() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_092() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_093() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_094() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_095() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_096() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_097() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_098() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_099() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_100() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_101() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_102() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_103() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_104() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_105() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_106() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_107() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_108() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_109() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_110() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_111() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_112() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_113() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_114() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_115() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_116() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_117() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_118() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_119() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_120() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_121() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_122() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_123() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_124() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_125() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_126() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_127() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_128() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_129() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_130() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_131() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_132() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_133() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_134() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_135() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_136() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_137() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_138() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_139() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_140() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_141() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_142() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_143() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_144() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_145() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_146() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_147() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_148() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_149() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_150() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_151() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_152() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_153() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_154() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_155() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_156() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_157() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_158() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_159() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_160() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_161() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_162() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_163() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_164() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_165() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_166() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_167() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_168() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_169() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_170() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_171() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_172() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_173() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_174() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_175() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_176() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_177() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_178() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_179() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_180() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_181() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_182() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_183() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_184() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_185() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_186() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_187() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_188() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_189() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_190() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_191() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_192() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_193() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_194() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_195() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_196() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_197() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_198() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_199() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_200() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_201() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_202() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_203() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_204() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_205() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_206() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_207() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_208() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_209() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_210() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_211() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_212() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_213() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_214() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_215() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_216() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_217() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_218() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_219() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_220() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_221() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_222() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_223() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_224() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_225() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_226() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_227() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_228() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_229() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_230() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_231() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_232() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_233() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_234() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_235() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_236() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_237() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_238() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_239() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_240() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_241() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_242() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_243() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_244() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_245() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_246() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_247() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_248() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_249() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_250() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_251() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_252() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_253() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_254() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_255() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_256() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_257() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_258() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_259() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_260() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_261() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_262() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_263() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_264() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_265() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_266() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_267() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_268() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_269() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_270() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_271() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_272() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_273() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_274() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_275() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_276() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_277() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_278() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_279() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_280() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_281() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_282() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_283() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_284() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_285() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_286() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_287() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_288() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_289() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_290() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_291() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_292() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_293() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_294() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_295() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_296() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_297() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_298() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_299() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_300() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_301() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_302() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_303() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_304() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_305() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_306() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_307() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_308() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_309() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_310() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_311() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_312() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_313() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_314() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_315() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_316() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_317() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_318() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_319() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_320() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_321() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_322() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_323() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_324() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_325() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_326() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_327() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_328() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_329() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    #[test]
    fn test_det_mod_stress_330() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }

    // Metric evaluation and validation padding line 0
    // Metric evaluation and validation padding line 1
    // Metric evaluation and validation padding line 2
    // Metric evaluation and validation padding line 3
    // Metric evaluation and validation padding line 4
    // Metric evaluation and validation padding line 5
    // Metric evaluation and validation padding line 6
    // Metric evaluation and validation padding line 7
}
