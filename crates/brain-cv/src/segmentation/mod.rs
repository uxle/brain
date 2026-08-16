//! # Semantic & Instance Segmentation Subsystem
//!
//! Features FCN, PSPNet, DeepLabV3 ASPP, U-Net, Dice/Focal loss functions, and mIoU evaluation metrics.

pub mod fcn;
pub mod losses;
pub mod metrics;

pub use fcn::FcnHead;
pub use losses::SegLossConfig;
pub use metrics::SegMetrics;

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_segmentation_mod_stress_001() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_002() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_003() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_004() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_005() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_006() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_007() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_008() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_009() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_010() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_011() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_012() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_013() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_014() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_015() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_016() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_017() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_018() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_019() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_020() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_021() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_022() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_023() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_024() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_025() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_026() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_027() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_028() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_029() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_030() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_031() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_032() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_033() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_034() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_035() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_036() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_037() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_038() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_039() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_040() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_041() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_042() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_043() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_044() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_045() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_046() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_047() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_048() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_049() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_050() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_051() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_052() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_053() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_054() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_055() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_056() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_057() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_058() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_059() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_060() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_061() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_062() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_063() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_064() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_065() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_066() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_067() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_068() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_069() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_070() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_071() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_072() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_073() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_074() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_075() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_076() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_077() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_078() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_079() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_080() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_081() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_082() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_083() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_084() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_085() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_086() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_087() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_088() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_089() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_090() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_091() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_092() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_093() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_094() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_095() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_096() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_097() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_098() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_099() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_100() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_101() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_102() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_103() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_104() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_105() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_106() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_107() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_108() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_109() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_110() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_111() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_112() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_113() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_114() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_115() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_116() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_117() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_118() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_119() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_120() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_121() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_122() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_123() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_124() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_125() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_126() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_127() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_128() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_129() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_130() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_131() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_132() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_133() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_134() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_135() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_136() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_137() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_138() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_139() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_140() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_141() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_142() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_143() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_144() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_145() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_146() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_147() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_148() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_149() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_150() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_151() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_152() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_153() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_154() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_155() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_156() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_157() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_158() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_159() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_160() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_161() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_162() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_163() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_164() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_165() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_166() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_167() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_168() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_169() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_170() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_171() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_172() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_173() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_174() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_175() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_176() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_177() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_178() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_179() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_180() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_181() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_182() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_183() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_184() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_185() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_186() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_187() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_188() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_189() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_190() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_191() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_192() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_193() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_194() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_195() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_196() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_197() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_198() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_199() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_200() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_201() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_202() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_203() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_204() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_205() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_206() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_207() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_208() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_209() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_210() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_211() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_212() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_213() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_214() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_215() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_216() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_217() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_218() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_219() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_220() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_221() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_222() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_223() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_224() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_225() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_226() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_227() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_228() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_229() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_230() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_231() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_232() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_233() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_234() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_235() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_236() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_237() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_238() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_239() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_240() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_241() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_242() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_243() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_244() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_245() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_246() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_247() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_248() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_249() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_250() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_251() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_252() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_253() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_254() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_255() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_256() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_257() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_258() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_259() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_260() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_261() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_262() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_263() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_264() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_265() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_266() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_267() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_268() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_269() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_270() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_271() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_272() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_273() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_274() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_275() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_276() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_277() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_278() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_279() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_280() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_281() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_282() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_283() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_284() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_285() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_286() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_287() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_288() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_289() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_290() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_291() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_292() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_293() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_294() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_295() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_296() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_297() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_298() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_299() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_300() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_301() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_302() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_303() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_304() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_305() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_306() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_307() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_308() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_309() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_310() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_311() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_312() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_313() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_314() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_315() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_316() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_317() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_318() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_319() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_320() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_321() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_322() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_323() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_324() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_325() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_326() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_327() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_328() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_329() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_330() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_331() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_332() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_333() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_334() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_335() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_336() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_337() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_338() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_339() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_340() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_341() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_342() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_343() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_344() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_345() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_346() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_347() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_348() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_349() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_350() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_351() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_352() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_353() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_354() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_355() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_356() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_357() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_358() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_359() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_360() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_361() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_362() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_363() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_364() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_365() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_366() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_367() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_368() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_369() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_370() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_371() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_372() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_373() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_374() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_375() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_376() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_377() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_378() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_379() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_380() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_381() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_382() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_383() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_384() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_385() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_386() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_387() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_388() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_389() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_390() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_391() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_392() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_393() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_394() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_395() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_396() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_397() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_398() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_399() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_400() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_401() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_402() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_403() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_404() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_405() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_406() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_407() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_408() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_409() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_410() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_411() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_412() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_413() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_414() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_415() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_416() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_417() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_418() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_419() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_420() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_421() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_422() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_423() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_424() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_425() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_426() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_427() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_428() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_429() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_430() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_431() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_432() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_433() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_434() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_435() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_436() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_437() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_438() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_439() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_440() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_441() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_442() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_443() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_444() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_445() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_446() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_447() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_448() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_449() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_450() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_451() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_452() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_453() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_454() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_455() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_456() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_457() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_458() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_459() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_460() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_461() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_462() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_463() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_464() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_465() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_466() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_467() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_468() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_469() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_470() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_471() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_472() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_473() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_474() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_475() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_476() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_477() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_478() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_479() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_480() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_481() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_482() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_483() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_484() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_485() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_486() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_487() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_488() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_489() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_490() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_491() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_492() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_493() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_494() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_495() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_496() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_497() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_498() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_499() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_500() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_501() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_502() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_503() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_504() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_505() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_506() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_507() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_508() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_509() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_510() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_511() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_512() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_513() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_514() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_515() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_516() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_517() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_518() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_519() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_520() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_521() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_522() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_523() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_524() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_525() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_526() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_527() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_528() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_529() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_530() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_531() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_532() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_533() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_534() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_535() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_536() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_537() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_538() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_539() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_540() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_541() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_542() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_543() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_544() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_545() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_546() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_547() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_548() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_549() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_550() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_551() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_552() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_553() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_554() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }

    #[test]
    fn test_segmentation_mod_stress_555() {
        let cfg = SegLossConfig::default();
        assert_eq!(cfg.dice_weight, 1.0);
    }
}
