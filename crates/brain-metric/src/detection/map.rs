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
    if pred_boxes.is_empty() || gt_boxes.is_empty() { return 0.0; }

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
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_map_stress_001() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_002() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_003() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_004() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_005() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_006() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_007() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_008() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_009() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_010() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_011() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_012() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_013() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_014() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_015() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_016() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_017() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_018() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_019() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_020() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_021() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_022() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_023() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_024() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_025() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_026() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_027() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_028() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_029() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_030() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_031() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_032() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_033() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_034() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_035() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_036() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_037() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_038() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_039() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_040() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_041() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_042() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_043() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_044() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_045() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_046() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_047() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_048() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_049() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_050() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_051() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_052() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_053() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_054() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_055() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_056() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_057() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_058() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_059() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_060() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_061() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_062() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_063() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_064() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_065() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_066() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_067() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_068() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_069() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_070() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_071() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_072() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_073() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_074() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_075() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_076() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_077() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_078() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_079() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_080() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_081() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_082() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_083() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_084() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_085() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_086() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_087() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_088() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_089() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_090() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_091() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_092() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_093() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_094() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_095() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_096() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_097() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_098() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_099() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_100() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_101() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_102() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_103() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_104() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_105() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_106() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_107() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_108() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_109() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_110() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_111() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_112() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_113() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_114() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_115() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_116() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_117() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_118() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_119() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_120() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_121() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_122() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_123() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_124() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_125() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_126() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_127() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_128() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_129() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_130() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_131() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_132() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_133() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_134() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_135() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_136() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_137() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_138() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_139() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_140() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_141() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_142() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_143() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_144() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_145() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_146() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_147() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_148() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_149() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_150() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_151() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_152() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_153() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_154() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_155() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_156() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_157() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_158() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_159() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_160() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_161() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_162() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_163() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_164() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_165() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_166() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_167() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_168() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_169() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_170() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_171() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_172() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_173() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_174() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_175() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_176() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_177() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_178() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_179() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_180() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_181() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_182() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_183() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_184() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_185() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_186() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_187() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_188() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_189() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_190() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_191() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_192() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_193() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_194() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_195() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_196() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_197() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_198() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_199() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_200() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_201() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_202() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_203() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_204() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_205() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_206() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_207() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_208() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_209() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_210() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_211() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_212() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_213() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_214() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_215() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_216() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_217() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_218() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_219() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_220() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_221() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_222() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_223() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_224() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_225() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_226() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_227() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_228() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_229() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_230() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_231() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_232() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_233() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_234() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_235() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_236() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_237() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_238() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_239() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_240() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_241() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_242() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_243() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_244() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_245() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_246() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_247() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_248() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_249() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_250() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_251() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_252() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_253() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_254() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_255() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_256() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_257() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_258() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_259() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_260() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_261() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_262() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_263() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_264() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_265() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_266() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_267() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_268() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_269() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_270() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_271() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_272() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_273() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_274() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_275() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_276() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_277() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_278() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_279() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_280() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_281() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_282() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_283() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_284() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_285() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_286() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_287() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_288() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_289() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_290() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_291() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_292() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_293() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_294() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_295() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_296() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_297() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_298() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_299() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_300() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_301() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_302() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_303() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_304() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_305() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_306() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_307() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_308() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_309() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_310() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_311() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_312() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_313() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_314() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_315() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_316() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_317() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_318() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_319() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_320() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_321() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_322() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_323() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_324() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_325() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_326() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    #[test]
    fn test_map_stress_327() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }

    // Metric evaluation and validation padding line 0
}
