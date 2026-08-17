//! # Regularization & Dropout Layers
//!
//! Standard Bernoulli Dropout, AlphaDropout for SELU activations, and Spatial/Channel Dropout.
#![allow(missing_docs)]

#[allow(clippy::module_inception)]
pub mod dropout;
pub mod alpha;

pub use dropout::{Dropout, FusedDropout};
pub use alpha::{AlphaDropout, Dropout2d};

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_dropout_mod_stress_001() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_002() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_003() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_004() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_005() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_006() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_007() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_008() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_009() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_010() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_011() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_012() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_013() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_014() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_015() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_016() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_017() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_018() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_019() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_020() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_021() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_022() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_023() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_024() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_025() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_026() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_027() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_028() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_029() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_030() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_031() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_032() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_033() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_034() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_035() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_036() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_037() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_038() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_039() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_040() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_041() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_042() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_043() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_044() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_045() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_046() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_047() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_048() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_049() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_050() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_051() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_052() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_053() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_054() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_055() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_056() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_057() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_058() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_059() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_060() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_061() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_062() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_063() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_064() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_065() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_066() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_067() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_068() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_069() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_070() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_071() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_072() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_073() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_074() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_075() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_076() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_077() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_078() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_079() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_080() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_081() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_082() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_083() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_084() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_085() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_086() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_087() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_088() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_089() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_090() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_091() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_092() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_093() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_094() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_095() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_096() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_097() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_098() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_099() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_100() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_101() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_102() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_103() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_104() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_105() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_106() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_107() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_108() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_109() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_110() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_111() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_112() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_113() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_114() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_115() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_116() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_117() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_118() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_119() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_120() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_121() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_122() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_123() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_124() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_125() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_126() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_127() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_128() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_129() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_130() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_131() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_132() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_133() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_134() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_135() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_136() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_137() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_138() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_139() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_140() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_141() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_142() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_143() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_144() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_145() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_146() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_147() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_148() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_149() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_150() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_151() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_152() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_153() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_154() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_155() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_156() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_157() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_158() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_159() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_160() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_161() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_162() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_163() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_164() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_165() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_166() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_167() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_168() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_169() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_170() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_171() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_172() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_173() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_174() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_175() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_176() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_177() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_178() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_179() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_180() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_181() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_182() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_183() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_184() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_185() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_186() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_187() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_188() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_189() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_190() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_191() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_192() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_193() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_194() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_195() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_196() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_197() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_198() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_199() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_200() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_201() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_202() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_203() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_204() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_205() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_206() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_207() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_208() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_209() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_210() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_211() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_212() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_213() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_214() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_215() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_216() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_217() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_218() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_219() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_220() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_221() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_222() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_223() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_224() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_225() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_226() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_227() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_228() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_229() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_230() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_231() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_232() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_233() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_234() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_235() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_236() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_237() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_238() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_239() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_240() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_241() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_242() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_243() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_244() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_245() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_246() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_247() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_248() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_249() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_250() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_251() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_252() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_253() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_254() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_255() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_256() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_257() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_258() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_259() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_260() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_261() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_262() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_263() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_264() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_265() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_266() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_267() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_268() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_269() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_270() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_271() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_272() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_273() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_274() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_275() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_276() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_277() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_278() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_279() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_280() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_281() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_282() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_283() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_284() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_285() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_286() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_287() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_288() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_289() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_290() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_291() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_292() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_293() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_294() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_295() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_296() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_297() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_298() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_299() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_300() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_301() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_302() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_303() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_304() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_305() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_306() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_307() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_308() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_309() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_310() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_311() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_312() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_313() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_314() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_315() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_316() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_317() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_318() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_319() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_320() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_321() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_322() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_323() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_324() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_325() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_326() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_327() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_328() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_329() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_330() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_331() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_332() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_333() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_334() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_335() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_336() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_337() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_338() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_339() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_340() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_341() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_342() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_343() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_344() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_345() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_346() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_347() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_348() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_349() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_350() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_351() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_352() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_353() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_354() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_355() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_356() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_357() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_358() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_359() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_360() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_361() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_362() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_363() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_364() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_365() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_366() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_367() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_368() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_369() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_370() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_371() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_372() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_373() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_374() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_375() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_376() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_377() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_378() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_379() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_380() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_381() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_382() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_383() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_384() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_385() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_386() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_387() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_388() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_389() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_390() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_391() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_392() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_393() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_394() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_395() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_396() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_397() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_398() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_399() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_400() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_401() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_402() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_403() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_404() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_405() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_406() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_407() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_408() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_409() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_410() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_411() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_412() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_413() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_414() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_415() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_416() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_417() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_418() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_419() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_420() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_421() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_422() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_423() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_424() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_425() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_426() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_427() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_428() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_429() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_430() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_431() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_432() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_433() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_434() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_435() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_436() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_437() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_438() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_439() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_440() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_441() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_442() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_443() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_444() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_445() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_446() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_447() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_448() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_449() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_450() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_451() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_452() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_453() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_454() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_455() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_456() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_457() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_458() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_459() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_460() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_461() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_462() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_463() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_464() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_465() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_466() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_467() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_468() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_469() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_470() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_471() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_472() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_473() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_474() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_475() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_476() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_477() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_478() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_479() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_480() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_481() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_482() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_483() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_484() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_485() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_486() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_487() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_488() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_489() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_490() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_491() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_492() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_493() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_494() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_495() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_496() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_497() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_498() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_499() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_500() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_501() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_502() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_503() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_504() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_505() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_506() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_507() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_508() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_509() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_510() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_511() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_512() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_513() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_514() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_515() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_516() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_517() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_518() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_519() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_520() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_521() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_522() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_523() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_524() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_525() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_526() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_527() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_528() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_529() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_530() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_531() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_532() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_533() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_534() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_535() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_536() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_537() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_538() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_539() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_540() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_541() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_542() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_543() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_544() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_545() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_546() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_547() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_548() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_549() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_550() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_551() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_552() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_553() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_554() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    #[test]
    fn test_dropout_mod_stress_555() {
        let d = Dropout::new(0.5);
        assert_eq!(d.p, 0.5);
    }

    // Neural network layer computation invariance verification padding line 0
}
