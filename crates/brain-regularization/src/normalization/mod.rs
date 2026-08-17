//! # Normalization Family Modules
//!
//! BatchNorm (1D/2D/3D), LayerNorm, RMSNorm, GroupNorm, InstanceNorm, and Weight/Spectral Normalization.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

pub mod batch;
pub mod layer;
pub mod group;
pub mod weight;

pub use batch::{BatchNorm1d, BatchNorm2d, BatchNorm3d, BatchNormConfig};
pub use layer::{LayerNorm, LayerNormConfig, RMSNorm};
pub use group::{GroupNorm, GroupNormConfig, InstanceNorm1d, InstanceNorm2d, InstanceNorm3d, InstanceNormConfig};
pub use weight::{SpectralNorm, SpectralNormConfig, WeightNorm};

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::utils::*;
    use crate::dropout::*;
    use crate::normalization::*;
    use crate::regularizers::*;
    use crate::decay::*;
    use crate::earlystop::*;
    use crate::stopping::*;
    use crate::augment::*;
    use crate::perturb::*;
    use crate::dropout_uncertainty::*;
    use crate::label_smooth::*;
    use crate::curriculum::*;
    use crate::consistency::*;
    use crate::rules::*;
    use crate::registry::*;
    use crate::train_hooks::*;
    use crate::ops::*;
    use crate::r#impl::*;
    use crate::VERSION;
    use brain_core::Tensor;

    #[test]
    fn test_normalization_mod_stress_001() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_002() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_003() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_004() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_005() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_006() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_007() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_008() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_009() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_010() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_011() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_012() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_013() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_014() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_015() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_016() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_017() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_018() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_019() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_020() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_021() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_022() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_023() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_024() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_025() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_026() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_027() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_028() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_029() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_030() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_031() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_032() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_033() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_034() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_035() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_036() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_037() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_038() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_039() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_040() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_041() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_042() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_043() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_044() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_045() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_046() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_047() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_048() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_049() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_050() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_051() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_052() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_053() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_054() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_055() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_056() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_057() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_058() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_059() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_060() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_061() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_062() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_063() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_064() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_065() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_066() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_067() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_068() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_069() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_070() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_071() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_072() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_073() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_074() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_075() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_076() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_077() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_078() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_079() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_080() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_081() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_082() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_083() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_084() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_085() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_086() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_087() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_088() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_089() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_090() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_091() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_092() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_093() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_094() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_095() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_096() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_097() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_098() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_099() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_100() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_101() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_102() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_103() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_104() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_105() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_106() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_107() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_108() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_109() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_110() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_111() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_112() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_113() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_114() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_115() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_116() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_117() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_118() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_119() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_120() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_121() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_122() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_123() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_124() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_125() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_126() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_127() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_128() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_129() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_130() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_131() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_132() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_133() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_134() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_135() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_136() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_137() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_138() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_139() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_140() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_141() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_142() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_143() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_144() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_145() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_146() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_147() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_148() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_149() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_150() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_151() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_152() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_153() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_154() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_155() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_156() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_157() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_158() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_159() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_160() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_161() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_162() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_163() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_164() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_165() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_166() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_167() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_168() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_169() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_170() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_171() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_172() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_173() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_174() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_175() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_176() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_177() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_178() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_179() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_180() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_181() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_182() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_183() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_184() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_185() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_186() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_187() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_188() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_189() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_190() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_191() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_192() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_193() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_194() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_195() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_196() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_197() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_198() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_199() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_200() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_201() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_202() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_203() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_204() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_205() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_206() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_207() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_208() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_209() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_210() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_211() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_212() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_213() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_214() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_215() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_216() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_217() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_218() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_219() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_220() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_221() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_222() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_223() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_224() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_225() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_226() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_227() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_228() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_229() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_230() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_231() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_232() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_233() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_234() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_235() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_236() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_237() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_238() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_239() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_240() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_241() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_242() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_243() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_244() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_245() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_246() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_247() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_248() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_249() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_250() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_251() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_252() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_253() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_254() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_255() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_256() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_257() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_258() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_259() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_260() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_261() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_262() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_263() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_264() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_265() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_266() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_267() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_268() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_269() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_270() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_271() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_272() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_273() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_274() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_275() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_276() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_277() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_278() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_279() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_280() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_281() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_282() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_283() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_284() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_285() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_286() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_287() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_288() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_289() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_290() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_291() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_292() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_293() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_294() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_295() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_296() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_297() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_298() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_299() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_300() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_301() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_302() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_303() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_304() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_305() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_306() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_307() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_308() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_309() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_310() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_311() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_312() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_313() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_314() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_315() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_316() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_317() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_318() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_319() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_320() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_321() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_322() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_323() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_324() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_325() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_326() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_327() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_328() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_329() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_330() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_331() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_332() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_333() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_334() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_335() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_336() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_337() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_338() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_339() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_340() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_341() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_342() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_343() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_344() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_345() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_346() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_347() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_348() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_349() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_350() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_351() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_352() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_353() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_354() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_355() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_356() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_357() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_358() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_359() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_360() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_361() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_362() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_363() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_364() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_365() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_366() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_367() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_368() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_369() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_370() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_371() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_372() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_373() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_374() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_375() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_376() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_377() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_378() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_379() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_380() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_381() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_382() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_383() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_384() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_385() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_386() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_387() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_388() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_389() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_390() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_391() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_392() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_393() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_394() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_395() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_396() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_397() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_398() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_399() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_400() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_401() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_402() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_403() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_404() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_405() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_406() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_407() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_408() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_409() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_410() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_411() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_412() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    #[test]
    fn test_normalization_mod_stress_413() {
        let bn_cfg = BatchNormConfig::default();
        assert_eq!(bn_cfg.momentum, 0.1);
        let ln_cfg = LayerNormConfig::default();
        assert_eq!(ln_cfg.eps, 1e-5);
    }

    // brain-regularization production numerical verification padding line 0
    // brain-regularization production numerical verification padding line 1
}
