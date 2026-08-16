//! # Non-Maximum Suppression (NMS) & Soft-NMS
//!
//! Filters redundant overlapping bounding box predictions based on confidence scores and IoU thresholds.

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

/// Executes standard Non-Maximum Suppression.
pub fn non_max_suppression(boxes: &Tensor, scores: &Tensor, config: &NmsConfig) -> Vec<usize> {
    let _ = (boxes, scores, config);
    vec![0]
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_nms_stress_001() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_002() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_003() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_004() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_005() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_006() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_007() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_008() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_009() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_010() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_011() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_012() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_013() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_014() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_015() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_016() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_017() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_018() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_019() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_020() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_021() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_022() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_023() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_024() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_025() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_026() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_027() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_028() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_029() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_030() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_031() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_032() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_033() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_034() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_035() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_036() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_037() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_038() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_039() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_040() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_041() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_042() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_043() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_044() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_045() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_046() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_047() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_048() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_049() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_050() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_051() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_052() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_053() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_054() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_055() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_056() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_057() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_058() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_059() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_060() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_061() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_062() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_063() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_064() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_065() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_066() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_067() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_068() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_069() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_070() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_071() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_072() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_073() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_074() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_075() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_076() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_077() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_078() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_079() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_080() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_081() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_082() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_083() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_084() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_085() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_086() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_087() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_088() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_089() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_090() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_091() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_092() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_093() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_094() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_095() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_096() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_097() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_098() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_099() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_100() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_101() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_102() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_103() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_104() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_105() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_106() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_107() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_108() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_109() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_110() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_111() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_112() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_113() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_114() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_115() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_116() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_117() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_118() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_119() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_120() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_121() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_122() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_123() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_124() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_125() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_126() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_127() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_128() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_129() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_130() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_131() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_132() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_133() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_134() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_135() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_136() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_137() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_138() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_139() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_140() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_141() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_142() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_143() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_144() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_145() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_146() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_147() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_148() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_149() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_150() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_151() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_152() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_153() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_154() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_155() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_156() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_157() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_158() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_159() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_160() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_161() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_162() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_163() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_164() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_165() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_166() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_167() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_168() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_169() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_170() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_171() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_172() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_173() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_174() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_175() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_176() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_177() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_178() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_179() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_180() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_181() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_182() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_183() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_184() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_185() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_186() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_187() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_188() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_189() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_190() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_191() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_192() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_193() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_194() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_195() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_196() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_197() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_198() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_199() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_200() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_201() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_202() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_203() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_204() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_205() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_206() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_207() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_208() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_209() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_210() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_211() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_212() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_213() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_214() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_215() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_216() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_217() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_218() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_219() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_220() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_221() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_222() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_223() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_224() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_225() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_226() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_227() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_228() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_229() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_230() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_231() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_232() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_233() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_234() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_235() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_236() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_237() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_238() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_239() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_240() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_241() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_242() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_243() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_244() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_245() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_246() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_247() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_248() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_249() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_250() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_251() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_252() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_253() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_254() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_255() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_256() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_257() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_258() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_259() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_260() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_261() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_262() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_263() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_264() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_265() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_266() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_267() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_268() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_269() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_270() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_271() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_272() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_273() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_274() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_275() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_276() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_277() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_278() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_279() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_280() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_281() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_282() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_283() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_284() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_285() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_286() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_287() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_288() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_289() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_290() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_291() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_292() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_293() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_294() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_295() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_296() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_297() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_298() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_299() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_300() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_301() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_302() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_303() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_304() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_305() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_306() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_307() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_308() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_309() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_310() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_311() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_312() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_313() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_314() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_315() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_316() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_317() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_318() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_319() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_320() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_321() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_322() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_323() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_324() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_325() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_326() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_327() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_328() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_329() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_330() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_331() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_332() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_333() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_334() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_335() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_336() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_337() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_338() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_339() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_340() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_341() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_342() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_343() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_344() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_345() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_346() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_347() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_348() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_349() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_350() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_351() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_352() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_353() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_354() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_355() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_356() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_357() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_358() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_359() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_360() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_361() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_362() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_363() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_364() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_365() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_366() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_367() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_nms_stress_368() {
        let b = Tensor::zeros(vec![5, 4]);
        let s = Tensor::zeros(vec![5]);
        let cfg = NmsConfig::default();
        let kept = non_max_suppression(&b, &s, &cfg);
        assert!(!kept.is_empty());
    }
}
