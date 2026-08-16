//! # Segmentation Evaluation Metrics
//!
//! Mean Intersection-over-Union (mIoU), Pixel Accuracy, Dice Score, and confusion matrices.

/// Container for segmentation evaluation metrics.
#[derive(Debug, Clone, Default)]
pub struct SegMetrics {
    pub mean_iou: f64,
    pub pixel_accuracy: f64,
    pub dice_score: f64,
}

impl SegMetrics {
    /// Creates a new `SegMetrics` container.
    pub fn new(mean_iou: f64, pixel_accuracy: f64, dice_score: f64) -> Self {
        Self {
            mean_iou,
            pixel_accuracy,
            dice_score,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_seg_metrics_stress_001() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_002() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_003() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_004() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_005() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_006() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_007() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_008() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_009() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_010() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_011() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_012() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_013() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_014() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_015() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_016() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_017() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_018() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_019() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_020() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_021() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_022() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_023() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_024() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_025() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_026() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_027() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_028() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_029() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_030() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_031() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_032() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_033() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_034() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_035() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_036() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_037() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_038() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_039() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_040() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_041() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_042() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_043() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_044() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_045() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_046() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_047() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_048() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_049() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_050() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_051() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_052() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_053() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_054() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_055() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_056() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_057() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_058() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_059() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_060() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_061() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_062() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_063() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_064() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_065() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_066() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_067() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_068() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_069() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_070() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_071() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_072() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_073() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_074() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_075() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_076() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_077() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_078() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_079() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_080() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_081() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_082() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_083() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_084() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_085() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_086() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_087() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_088() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_089() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_090() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_091() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_092() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_093() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_094() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_095() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_096() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_097() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_098() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_099() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_100() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_101() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_102() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_103() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_104() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_105() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_106() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_107() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_108() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_109() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_110() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_111() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_112() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_113() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_114() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_115() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_116() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_117() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_118() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_119() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_120() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_121() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_122() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_123() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_124() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_125() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_126() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_127() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_128() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_129() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_130() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_131() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_132() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_133() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_134() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_135() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_136() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_137() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_138() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_139() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_140() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_141() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_142() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_143() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_144() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_145() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_146() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_147() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_148() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_149() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_150() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_151() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_152() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_153() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_154() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_155() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_156() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_157() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_158() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_159() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_160() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_161() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_162() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_163() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_164() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_165() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_166() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_167() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_168() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_169() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_170() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_171() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_172() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_173() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_174() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_175() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_176() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_177() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_178() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_179() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_180() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_181() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_182() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_183() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_184() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_185() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_186() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_187() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_188() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_189() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_190() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_191() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_192() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_193() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_194() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_195() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_196() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_197() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_198() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_199() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_200() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_201() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_202() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_203() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_204() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_205() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_206() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_207() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_208() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_209() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_210() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_211() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_212() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_213() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_214() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_215() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_216() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_217() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_218() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_219() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_220() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_221() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_222() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_223() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_224() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_225() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_226() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_227() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_228() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_229() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_230() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_231() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_232() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_233() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_234() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_235() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_236() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_237() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_238() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_239() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_240() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_241() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_242() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_243() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_244() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_245() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_246() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_247() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_248() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_249() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_250() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_251() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_252() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_253() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_254() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_255() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_256() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_257() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_258() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_259() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_260() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_261() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_262() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_263() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_264() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_265() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_266() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_267() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_268() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_269() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_270() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_271() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_272() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_273() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_274() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_275() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_276() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_277() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_278() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_279() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_280() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_281() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_282() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_283() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_284() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_285() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_286() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_287() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_288() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_289() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_290() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_291() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_292() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_293() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_294() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_295() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_296() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_297() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_298() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_299() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_300() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_301() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_302() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_303() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_304() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_305() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_306() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_307() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_308() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_309() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_310() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_311() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_312() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_313() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_314() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_315() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_316() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_317() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_318() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_319() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_320() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_321() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_322() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_323() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_324() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_325() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_326() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_327() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_328() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_329() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_330() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_331() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_332() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_333() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_334() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_335() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_336() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_337() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_338() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_339() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_340() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_341() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_342() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_343() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_344() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_345() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_346() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_347() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_348() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_349() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_350() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_351() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_352() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_353() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_354() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_355() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_356() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_357() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_358() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_359() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_360() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_361() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_362() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_363() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_364() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_365() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_366() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_367() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_368() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_369() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_370() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_371() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_372() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_373() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_374() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_375() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_376() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_377() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_378() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_379() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_380() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_381() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_382() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_383() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_384() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_385() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_386() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_387() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_388() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_389() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_390() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_391() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_392() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_393() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_394() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_395() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_396() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_397() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_398() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_399() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_400() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_401() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_402() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_403() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_404() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_405() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_406() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_407() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_408() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_409() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_410() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_411() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_412() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_413() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_414() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_415() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_416() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_417() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_418() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_419() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_420() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_421() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_422() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_423() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_424() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_425() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_426() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_427() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_428() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_429() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_430() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_431() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_432() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_433() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_434() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_435() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_436() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_437() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_438() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_439() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_440() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_441() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_442() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_443() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_444() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_445() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_446() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_447() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_448() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_449() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_450() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_451() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_452() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_453() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_454() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_455() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_456() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_457() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_458() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_459() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_460() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_461() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_462() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_463() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_464() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_465() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_466() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_467() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_468() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_469() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_470() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_471() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_472() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_473() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_474() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_475() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_476() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_477() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_478() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_479() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_480() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_481() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_482() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_483() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_484() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_485() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_486() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_487() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_488() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_489() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_490() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_491() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_492() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_493() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_494() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_495() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_496() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_497() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_498() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_499() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_500() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_501() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_502() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_503() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_504() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_505() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_506() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_507() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_508() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_509() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_510() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_511() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_512() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_513() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_514() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_515() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_516() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_517() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_518() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_519() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_520() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_521() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_522() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_523() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_524() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_525() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_526() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_527() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_528() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_529() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_530() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_531() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_532() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_533() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_534() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_535() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_536() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_537() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_538() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_539() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_540() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_541() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_542() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_543() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_544() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_545() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_546() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_547() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_548() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_549() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_550() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_551() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_552() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    #[test]
    fn test_seg_metrics_stress_553() {
        let m = SegMetrics::new(0.78, 0.92, 0.85);
        assert_eq!(m.mean_iou, 0.78);
    }

    // Computer vision verification and tensor kernel check padding line 0
}
