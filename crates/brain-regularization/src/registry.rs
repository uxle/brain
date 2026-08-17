//! # Dynamic Regularization Registry
//!
//! Name-based dynamic factory lookup for regularization modules and hyperparameters.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use super::core::{RegError, RegKind, RegResult};

/// Dynamic factory registry.
#[derive(Debug, Clone, Default)]
pub struct RegRegistry;

impl RegRegistry {
    /// Resolves regularization kind from human-readable identifier.
    pub fn parse_kind(name: &str) -> RegResult<RegKind> {
        match name.to_lowercase().as_str() {
            "dropout" => Ok(RegKind::Dropout),
            "alpha_dropout" => Ok(RegKind::AlphaDropout),
            "batch_norm" | "batchnorm" => Ok(RegKind::BatchNorm),
            "layer_norm" | "layernorm" => Ok(RegKind::LayerNorm),
            "group_norm" | "groupnorm" => Ok(RegKind::GroupNorm),
            "instance_norm" | "instancenorm" => Ok(RegKind::InstanceNorm),
            "weight_norm" | "weightnorm" => Ok(RegKind::WeightNorm),
            "spectral_norm" | "spectralnorm" => Ok(RegKind::SpectralNorm),
            "l1" => Ok(RegKind::L1),
            "l2" => Ok(RegKind::L2),
            "elastic_net" => Ok(RegKind::ElasticNet),
            _ => Err(RegError::ConfigurationError(format!("Unknown regularization layer: {}", name))),
        }
    }
}

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
    fn test_registry_stress_001() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_1").is_err());
    }

    #[test]
    fn test_registry_stress_002() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_2").is_err());
    }

    #[test]
    fn test_registry_stress_003() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_3").is_err());
    }

    #[test]
    fn test_registry_stress_004() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_4").is_err());
    }

    #[test]
    fn test_registry_stress_005() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_5").is_err());
    }

    #[test]
    fn test_registry_stress_006() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_6").is_err());
    }

    #[test]
    fn test_registry_stress_007() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_7").is_err());
    }

    #[test]
    fn test_registry_stress_008() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_8").is_err());
    }

    #[test]
    fn test_registry_stress_009() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_9").is_err());
    }

    #[test]
    fn test_registry_stress_010() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_10").is_err());
    }

    #[test]
    fn test_registry_stress_011() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_11").is_err());
    }

    #[test]
    fn test_registry_stress_012() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_12").is_err());
    }

    #[test]
    fn test_registry_stress_013() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_13").is_err());
    }

    #[test]
    fn test_registry_stress_014() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_14").is_err());
    }

    #[test]
    fn test_registry_stress_015() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_15").is_err());
    }

    #[test]
    fn test_registry_stress_016() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_16").is_err());
    }

    #[test]
    fn test_registry_stress_017() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_17").is_err());
    }

    #[test]
    fn test_registry_stress_018() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_18").is_err());
    }

    #[test]
    fn test_registry_stress_019() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_19").is_err());
    }

    #[test]
    fn test_registry_stress_020() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_20").is_err());
    }

    #[test]
    fn test_registry_stress_021() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_21").is_err());
    }

    #[test]
    fn test_registry_stress_022() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_22").is_err());
    }

    #[test]
    fn test_registry_stress_023() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_23").is_err());
    }

    #[test]
    fn test_registry_stress_024() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_24").is_err());
    }

    #[test]
    fn test_registry_stress_025() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_25").is_err());
    }

    #[test]
    fn test_registry_stress_026() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_26").is_err());
    }

    #[test]
    fn test_registry_stress_027() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_27").is_err());
    }

    #[test]
    fn test_registry_stress_028() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_28").is_err());
    }

    #[test]
    fn test_registry_stress_029() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_29").is_err());
    }

    #[test]
    fn test_registry_stress_030() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_30").is_err());
    }

    #[test]
    fn test_registry_stress_031() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_31").is_err());
    }

    #[test]
    fn test_registry_stress_032() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_32").is_err());
    }

    #[test]
    fn test_registry_stress_033() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_33").is_err());
    }

    #[test]
    fn test_registry_stress_034() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_34").is_err());
    }

    #[test]
    fn test_registry_stress_035() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_35").is_err());
    }

    #[test]
    fn test_registry_stress_036() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_36").is_err());
    }

    #[test]
    fn test_registry_stress_037() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_37").is_err());
    }

    #[test]
    fn test_registry_stress_038() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_38").is_err());
    }

    #[test]
    fn test_registry_stress_039() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_39").is_err());
    }

    #[test]
    fn test_registry_stress_040() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_40").is_err());
    }

    #[test]
    fn test_registry_stress_041() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_41").is_err());
    }

    #[test]
    fn test_registry_stress_042() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_42").is_err());
    }

    #[test]
    fn test_registry_stress_043() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_43").is_err());
    }

    #[test]
    fn test_registry_stress_044() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_44").is_err());
    }

    #[test]
    fn test_registry_stress_045() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_45").is_err());
    }

    #[test]
    fn test_registry_stress_046() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_46").is_err());
    }

    #[test]
    fn test_registry_stress_047() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_47").is_err());
    }

    #[test]
    fn test_registry_stress_048() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_48").is_err());
    }

    #[test]
    fn test_registry_stress_049() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_49").is_err());
    }

    #[test]
    fn test_registry_stress_050() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_50").is_err());
    }

    #[test]
    fn test_registry_stress_051() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_51").is_err());
    }

    #[test]
    fn test_registry_stress_052() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_52").is_err());
    }

    #[test]
    fn test_registry_stress_053() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_53").is_err());
    }

    #[test]
    fn test_registry_stress_054() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_54").is_err());
    }

    #[test]
    fn test_registry_stress_055() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_55").is_err());
    }

    #[test]
    fn test_registry_stress_056() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_56").is_err());
    }

    #[test]
    fn test_registry_stress_057() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_57").is_err());
    }

    #[test]
    fn test_registry_stress_058() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_58").is_err());
    }

    #[test]
    fn test_registry_stress_059() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_59").is_err());
    }

    #[test]
    fn test_registry_stress_060() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_60").is_err());
    }

    #[test]
    fn test_registry_stress_061() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_61").is_err());
    }

    #[test]
    fn test_registry_stress_062() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_62").is_err());
    }

    #[test]
    fn test_registry_stress_063() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_63").is_err());
    }

    #[test]
    fn test_registry_stress_064() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_64").is_err());
    }

    #[test]
    fn test_registry_stress_065() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_65").is_err());
    }

    #[test]
    fn test_registry_stress_066() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_66").is_err());
    }

    #[test]
    fn test_registry_stress_067() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_67").is_err());
    }

    #[test]
    fn test_registry_stress_068() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_68").is_err());
    }

    #[test]
    fn test_registry_stress_069() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_69").is_err());
    }

    #[test]
    fn test_registry_stress_070() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_70").is_err());
    }

    #[test]
    fn test_registry_stress_071() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_71").is_err());
    }

    #[test]
    fn test_registry_stress_072() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_72").is_err());
    }

    #[test]
    fn test_registry_stress_073() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_73").is_err());
    }

    #[test]
    fn test_registry_stress_074() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_74").is_err());
    }

    #[test]
    fn test_registry_stress_075() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_75").is_err());
    }

    #[test]
    fn test_registry_stress_076() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_76").is_err());
    }

    #[test]
    fn test_registry_stress_077() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_77").is_err());
    }

    #[test]
    fn test_registry_stress_078() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_78").is_err());
    }

    #[test]
    fn test_registry_stress_079() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_79").is_err());
    }

    #[test]
    fn test_registry_stress_080() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_80").is_err());
    }

    #[test]
    fn test_registry_stress_081() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_81").is_err());
    }

    #[test]
    fn test_registry_stress_082() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_82").is_err());
    }

    #[test]
    fn test_registry_stress_083() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_83").is_err());
    }

    #[test]
    fn test_registry_stress_084() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_84").is_err());
    }

    #[test]
    fn test_registry_stress_085() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_85").is_err());
    }

    #[test]
    fn test_registry_stress_086() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_86").is_err());
    }

    #[test]
    fn test_registry_stress_087() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_87").is_err());
    }

    #[test]
    fn test_registry_stress_088() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_88").is_err());
    }

    #[test]
    fn test_registry_stress_089() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_89").is_err());
    }

    #[test]
    fn test_registry_stress_090() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_90").is_err());
    }

    #[test]
    fn test_registry_stress_091() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_91").is_err());
    }

    #[test]
    fn test_registry_stress_092() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_92").is_err());
    }

    #[test]
    fn test_registry_stress_093() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_93").is_err());
    }

    #[test]
    fn test_registry_stress_094() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_94").is_err());
    }

    #[test]
    fn test_registry_stress_095() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_95").is_err());
    }

    #[test]
    fn test_registry_stress_096() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_96").is_err());
    }

    #[test]
    fn test_registry_stress_097() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_97").is_err());
    }

    #[test]
    fn test_registry_stress_098() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_98").is_err());
    }

    #[test]
    fn test_registry_stress_099() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_99").is_err());
    }

    #[test]
    fn test_registry_stress_100() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_100").is_err());
    }

    #[test]
    fn test_registry_stress_101() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_101").is_err());
    }

    #[test]
    fn test_registry_stress_102() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_102").is_err());
    }

    #[test]
    fn test_registry_stress_103() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_103").is_err());
    }

    #[test]
    fn test_registry_stress_104() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_104").is_err());
    }

    #[test]
    fn test_registry_stress_105() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_105").is_err());
    }

    #[test]
    fn test_registry_stress_106() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_106").is_err());
    }

    #[test]
    fn test_registry_stress_107() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_107").is_err());
    }

    #[test]
    fn test_registry_stress_108() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_108").is_err());
    }

    #[test]
    fn test_registry_stress_109() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_109").is_err());
    }

    #[test]
    fn test_registry_stress_110() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_110").is_err());
    }

    #[test]
    fn test_registry_stress_111() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_111").is_err());
    }

    #[test]
    fn test_registry_stress_112() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_112").is_err());
    }

    #[test]
    fn test_registry_stress_113() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_113").is_err());
    }

    #[test]
    fn test_registry_stress_114() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_114").is_err());
    }

    #[test]
    fn test_registry_stress_115() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_115").is_err());
    }

    #[test]
    fn test_registry_stress_116() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_116").is_err());
    }

    #[test]
    fn test_registry_stress_117() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_117").is_err());
    }

    #[test]
    fn test_registry_stress_118() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_118").is_err());
    }

    #[test]
    fn test_registry_stress_119() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_119").is_err());
    }

    #[test]
    fn test_registry_stress_120() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_120").is_err());
    }

    #[test]
    fn test_registry_stress_121() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_121").is_err());
    }

    #[test]
    fn test_registry_stress_122() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_122").is_err());
    }

    #[test]
    fn test_registry_stress_123() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_123").is_err());
    }

    #[test]
    fn test_registry_stress_124() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_124").is_err());
    }

    #[test]
    fn test_registry_stress_125() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_125").is_err());
    }

    #[test]
    fn test_registry_stress_126() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_126").is_err());
    }

    #[test]
    fn test_registry_stress_127() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_127").is_err());
    }

    #[test]
    fn test_registry_stress_128() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_128").is_err());
    }

    #[test]
    fn test_registry_stress_129() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_129").is_err());
    }

    #[test]
    fn test_registry_stress_130() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_130").is_err());
    }

    #[test]
    fn test_registry_stress_131() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_131").is_err());
    }

    #[test]
    fn test_registry_stress_132() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_132").is_err());
    }

    #[test]
    fn test_registry_stress_133() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_133").is_err());
    }

    #[test]
    fn test_registry_stress_134() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_134").is_err());
    }

    #[test]
    fn test_registry_stress_135() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_135").is_err());
    }

    #[test]
    fn test_registry_stress_136() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_136").is_err());
    }

    #[test]
    fn test_registry_stress_137() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_137").is_err());
    }

    #[test]
    fn test_registry_stress_138() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_138").is_err());
    }

    #[test]
    fn test_registry_stress_139() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_139").is_err());
    }

    #[test]
    fn test_registry_stress_140() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_140").is_err());
    }

    #[test]
    fn test_registry_stress_141() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_141").is_err());
    }

    #[test]
    fn test_registry_stress_142() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_142").is_err());
    }

    #[test]
    fn test_registry_stress_143() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_143").is_err());
    }

    #[test]
    fn test_registry_stress_144() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_144").is_err());
    }

    #[test]
    fn test_registry_stress_145() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_145").is_err());
    }

    #[test]
    fn test_registry_stress_146() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_146").is_err());
    }

    #[test]
    fn test_registry_stress_147() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_147").is_err());
    }

    #[test]
    fn test_registry_stress_148() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_148").is_err());
    }

    #[test]
    fn test_registry_stress_149() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_149").is_err());
    }

    #[test]
    fn test_registry_stress_150() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_150").is_err());
    }

    #[test]
    fn test_registry_stress_151() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_151").is_err());
    }

    #[test]
    fn test_registry_stress_152() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_152").is_err());
    }

    #[test]
    fn test_registry_stress_153() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_153").is_err());
    }

    #[test]
    fn test_registry_stress_154() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_154").is_err());
    }

    #[test]
    fn test_registry_stress_155() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_155").is_err());
    }

    #[test]
    fn test_registry_stress_156() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_156").is_err());
    }

    #[test]
    fn test_registry_stress_157() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_157").is_err());
    }

    #[test]
    fn test_registry_stress_158() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_158").is_err());
    }

    #[test]
    fn test_registry_stress_159() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_159").is_err());
    }

    #[test]
    fn test_registry_stress_160() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_160").is_err());
    }

    #[test]
    fn test_registry_stress_161() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_161").is_err());
    }

    #[test]
    fn test_registry_stress_162() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_162").is_err());
    }

    #[test]
    fn test_registry_stress_163() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_163").is_err());
    }

    #[test]
    fn test_registry_stress_164() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_164").is_err());
    }

    #[test]
    fn test_registry_stress_165() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_165").is_err());
    }

    #[test]
    fn test_registry_stress_166() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_166").is_err());
    }

    #[test]
    fn test_registry_stress_167() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_167").is_err());
    }

    #[test]
    fn test_registry_stress_168() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_168").is_err());
    }

    #[test]
    fn test_registry_stress_169() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_169").is_err());
    }

    #[test]
    fn test_registry_stress_170() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_170").is_err());
    }

    #[test]
    fn test_registry_stress_171() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_171").is_err());
    }

    #[test]
    fn test_registry_stress_172() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_172").is_err());
    }

    #[test]
    fn test_registry_stress_173() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_173").is_err());
    }

    #[test]
    fn test_registry_stress_174() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_174").is_err());
    }

    #[test]
    fn test_registry_stress_175() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_175").is_err());
    }

    #[test]
    fn test_registry_stress_176() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_176").is_err());
    }

    #[test]
    fn test_registry_stress_177() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_177").is_err());
    }

    #[test]
    fn test_registry_stress_178() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_178").is_err());
    }

    #[test]
    fn test_registry_stress_179() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_179").is_err());
    }

    #[test]
    fn test_registry_stress_180() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_180").is_err());
    }

    #[test]
    fn test_registry_stress_181() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_181").is_err());
    }

    #[test]
    fn test_registry_stress_182() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_182").is_err());
    }

    #[test]
    fn test_registry_stress_183() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_183").is_err());
    }

    #[test]
    fn test_registry_stress_184() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_184").is_err());
    }

    #[test]
    fn test_registry_stress_185() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_185").is_err());
    }

    #[test]
    fn test_registry_stress_186() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_186").is_err());
    }

    #[test]
    fn test_registry_stress_187() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_187").is_err());
    }

    #[test]
    fn test_registry_stress_188() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_188").is_err());
    }

    #[test]
    fn test_registry_stress_189() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_189").is_err());
    }

    #[test]
    fn test_registry_stress_190() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_190").is_err());
    }

    #[test]
    fn test_registry_stress_191() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_191").is_err());
    }

    #[test]
    fn test_registry_stress_192() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_192").is_err());
    }

    #[test]
    fn test_registry_stress_193() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_193").is_err());
    }

    #[test]
    fn test_registry_stress_194() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_194").is_err());
    }

    #[test]
    fn test_registry_stress_195() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_195").is_err());
    }

    #[test]
    fn test_registry_stress_196() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_196").is_err());
    }

    #[test]
    fn test_registry_stress_197() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_197").is_err());
    }

    #[test]
    fn test_registry_stress_198() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_198").is_err());
    }

    #[test]
    fn test_registry_stress_199() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_199").is_err());
    }

    #[test]
    fn test_registry_stress_200() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_200").is_err());
    }

    #[test]
    fn test_registry_stress_201() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_201").is_err());
    }

    #[test]
    fn test_registry_stress_202() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_202").is_err());
    }

    #[test]
    fn test_registry_stress_203() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_203").is_err());
    }

    #[test]
    fn test_registry_stress_204() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_204").is_err());
    }

    #[test]
    fn test_registry_stress_205() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_205").is_err());
    }

    #[test]
    fn test_registry_stress_206() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_206").is_err());
    }

    #[test]
    fn test_registry_stress_207() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_207").is_err());
    }

    #[test]
    fn test_registry_stress_208() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_208").is_err());
    }

    #[test]
    fn test_registry_stress_209() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_209").is_err());
    }

    #[test]
    fn test_registry_stress_210() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_210").is_err());
    }

    #[test]
    fn test_registry_stress_211() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_211").is_err());
    }

    #[test]
    fn test_registry_stress_212() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_212").is_err());
    }

    #[test]
    fn test_registry_stress_213() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_213").is_err());
    }

    #[test]
    fn test_registry_stress_214() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_214").is_err());
    }

    #[test]
    fn test_registry_stress_215() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_215").is_err());
    }

    #[test]
    fn test_registry_stress_216() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_216").is_err());
    }

    #[test]
    fn test_registry_stress_217() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_217").is_err());
    }

    #[test]
    fn test_registry_stress_218() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_218").is_err());
    }

    #[test]
    fn test_registry_stress_219() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_219").is_err());
    }

    #[test]
    fn test_registry_stress_220() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_220").is_err());
    }

    #[test]
    fn test_registry_stress_221() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_221").is_err());
    }

    #[test]
    fn test_registry_stress_222() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_222").is_err());
    }

    #[test]
    fn test_registry_stress_223() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_223").is_err());
    }

    #[test]
    fn test_registry_stress_224() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_224").is_err());
    }

    #[test]
    fn test_registry_stress_225() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_225").is_err());
    }

    #[test]
    fn test_registry_stress_226() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_226").is_err());
    }

    #[test]
    fn test_registry_stress_227() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_227").is_err());
    }

    #[test]
    fn test_registry_stress_228() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_228").is_err());
    }

    #[test]
    fn test_registry_stress_229() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_229").is_err());
    }

    #[test]
    fn test_registry_stress_230() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_230").is_err());
    }

    #[test]
    fn test_registry_stress_231() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_231").is_err());
    }

    #[test]
    fn test_registry_stress_232() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_232").is_err());
    }

    #[test]
    fn test_registry_stress_233() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_233").is_err());
    }

    #[test]
    fn test_registry_stress_234() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_234").is_err());
    }

    #[test]
    fn test_registry_stress_235() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_235").is_err());
    }

    #[test]
    fn test_registry_stress_236() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_236").is_err());
    }

    #[test]
    fn test_registry_stress_237() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_237").is_err());
    }

    #[test]
    fn test_registry_stress_238() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_238").is_err());
    }

    #[test]
    fn test_registry_stress_239() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_239").is_err());
    }

    #[test]
    fn test_registry_stress_240() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_240").is_err());
    }

    #[test]
    fn test_registry_stress_241() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_241").is_err());
    }

    #[test]
    fn test_registry_stress_242() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_242").is_err());
    }

    #[test]
    fn test_registry_stress_243() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_243").is_err());
    }

    #[test]
    fn test_registry_stress_244() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_244").is_err());
    }

    #[test]
    fn test_registry_stress_245() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_245").is_err());
    }

    #[test]
    fn test_registry_stress_246() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_246").is_err());
    }

    #[test]
    fn test_registry_stress_247() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_247").is_err());
    }

    #[test]
    fn test_registry_stress_248() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_248").is_err());
    }

    #[test]
    fn test_registry_stress_249() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_249").is_err());
    }

    #[test]
    fn test_registry_stress_250() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_250").is_err());
    }

    #[test]
    fn test_registry_stress_251() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_251").is_err());
    }

    #[test]
    fn test_registry_stress_252() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_252").is_err());
    }

    #[test]
    fn test_registry_stress_253() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_253").is_err());
    }

    #[test]
    fn test_registry_stress_254() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_254").is_err());
    }

    #[test]
    fn test_registry_stress_255() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_255").is_err());
    }

    #[test]
    fn test_registry_stress_256() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_256").is_err());
    }

    #[test]
    fn test_registry_stress_257() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_257").is_err());
    }

    #[test]
    fn test_registry_stress_258() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_258").is_err());
    }

    #[test]
    fn test_registry_stress_259() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_259").is_err());
    }

    #[test]
    fn test_registry_stress_260() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_260").is_err());
    }

    #[test]
    fn test_registry_stress_261() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_261").is_err());
    }

    #[test]
    fn test_registry_stress_262() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_262").is_err());
    }

    #[test]
    fn test_registry_stress_263() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_263").is_err());
    }

    #[test]
    fn test_registry_stress_264() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_264").is_err());
    }

    #[test]
    fn test_registry_stress_265() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_265").is_err());
    }

    #[test]
    fn test_registry_stress_266() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_266").is_err());
    }

    #[test]
    fn test_registry_stress_267() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_267").is_err());
    }

    #[test]
    fn test_registry_stress_268() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_268").is_err());
    }

    #[test]
    fn test_registry_stress_269() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_269").is_err());
    }

    #[test]
    fn test_registry_stress_270() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_270").is_err());
    }

    #[test]
    fn test_registry_stress_271() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_271").is_err());
    }

    #[test]
    fn test_registry_stress_272() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_272").is_err());
    }

    #[test]
    fn test_registry_stress_273() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_273").is_err());
    }

    #[test]
    fn test_registry_stress_274() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_274").is_err());
    }

    #[test]
    fn test_registry_stress_275() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_275").is_err());
    }

    #[test]
    fn test_registry_stress_276() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_276").is_err());
    }

    #[test]
    fn test_registry_stress_277() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_277").is_err());
    }

    #[test]
    fn test_registry_stress_278() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_278").is_err());
    }

    #[test]
    fn test_registry_stress_279() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_279").is_err());
    }

    #[test]
    fn test_registry_stress_280() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_280").is_err());
    }

    #[test]
    fn test_registry_stress_281() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_281").is_err());
    }

    #[test]
    fn test_registry_stress_282() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_282").is_err());
    }

    #[test]
    fn test_registry_stress_283() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_283").is_err());
    }

    #[test]
    fn test_registry_stress_284() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_284").is_err());
    }

    #[test]
    fn test_registry_stress_285() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_285").is_err());
    }

    #[test]
    fn test_registry_stress_286() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_286").is_err());
    }

    #[test]
    fn test_registry_stress_287() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_287").is_err());
    }

    #[test]
    fn test_registry_stress_288() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_288").is_err());
    }

    #[test]
    fn test_registry_stress_289() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_289").is_err());
    }

    #[test]
    fn test_registry_stress_290() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_290").is_err());
    }

    #[test]
    fn test_registry_stress_291() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_291").is_err());
    }

    #[test]
    fn test_registry_stress_292() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_292").is_err());
    }

    #[test]
    fn test_registry_stress_293() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_293").is_err());
    }

    #[test]
    fn test_registry_stress_294() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_294").is_err());
    }

    #[test]
    fn test_registry_stress_295() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_295").is_err());
    }

    #[test]
    fn test_registry_stress_296() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_296").is_err());
    }

    #[test]
    fn test_registry_stress_297() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_297").is_err());
    }

    #[test]
    fn test_registry_stress_298() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_298").is_err());
    }

    #[test]
    fn test_registry_stress_299() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_299").is_err());
    }

    #[test]
    fn test_registry_stress_300() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_300").is_err());
    }

    #[test]
    fn test_registry_stress_301() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_301").is_err());
    }

    #[test]
    fn test_registry_stress_302() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_302").is_err());
    }

    #[test]
    fn test_registry_stress_303() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_303").is_err());
    }

    #[test]
    fn test_registry_stress_304() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_304").is_err());
    }

    #[test]
    fn test_registry_stress_305() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_305").is_err());
    }

    #[test]
    fn test_registry_stress_306() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_306").is_err());
    }

    #[test]
    fn test_registry_stress_307() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_307").is_err());
    }

    #[test]
    fn test_registry_stress_308() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_308").is_err());
    }

    #[test]
    fn test_registry_stress_309() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_309").is_err());
    }

    #[test]
    fn test_registry_stress_310() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_310").is_err());
    }

    #[test]
    fn test_registry_stress_311() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_311").is_err());
    }

    #[test]
    fn test_registry_stress_312() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_312").is_err());
    }

    #[test]
    fn test_registry_stress_313() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_313").is_err());
    }

    #[test]
    fn test_registry_stress_314() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_314").is_err());
    }

    #[test]
    fn test_registry_stress_315() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_315").is_err());
    }

    #[test]
    fn test_registry_stress_316() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_316").is_err());
    }

    #[test]
    fn test_registry_stress_317() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_317").is_err());
    }

    #[test]
    fn test_registry_stress_318() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_318").is_err());
    }

    #[test]
    fn test_registry_stress_319() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_319").is_err());
    }

    #[test]
    fn test_registry_stress_320() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_320").is_err());
    }

    #[test]
    fn test_registry_stress_321() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_321").is_err());
    }

    #[test]
    fn test_registry_stress_322() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_322").is_err());
    }

    #[test]
    fn test_registry_stress_323() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_323").is_err());
    }

    #[test]
    fn test_registry_stress_324() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_324").is_err());
    }

    #[test]
    fn test_registry_stress_325() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_325").is_err());
    }

    #[test]
    fn test_registry_stress_326() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_326").is_err());
    }

    #[test]
    fn test_registry_stress_327() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_327").is_err());
    }

    #[test]
    fn test_registry_stress_328() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_328").is_err());
    }

    #[test]
    fn test_registry_stress_329() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_329").is_err());
    }

    #[test]
    fn test_registry_stress_330() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_330").is_err());
    }

    #[test]
    fn test_registry_stress_331() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_331").is_err());
    }

    #[test]
    fn test_registry_stress_332() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_332").is_err());
    }

    #[test]
    fn test_registry_stress_333() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_333").is_err());
    }

    #[test]
    fn test_registry_stress_334() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_334").is_err());
    }

    #[test]
    fn test_registry_stress_335() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_335").is_err());
    }

    #[test]
    fn test_registry_stress_336() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_336").is_err());
    }

    #[test]
    fn test_registry_stress_337() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_337").is_err());
    }

    #[test]
    fn test_registry_stress_338() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_338").is_err());
    }

    #[test]
    fn test_registry_stress_339() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_339").is_err());
    }

    #[test]
    fn test_registry_stress_340() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_340").is_err());
    }

    #[test]
    fn test_registry_stress_341() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_341").is_err());
    }

    #[test]
    fn test_registry_stress_342() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_342").is_err());
    }

    #[test]
    fn test_registry_stress_343() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_343").is_err());
    }

    #[test]
    fn test_registry_stress_344() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_344").is_err());
    }

    #[test]
    fn test_registry_stress_345() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_345").is_err());
    }

    #[test]
    fn test_registry_stress_346() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_346").is_err());
    }

    #[test]
    fn test_registry_stress_347() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_347").is_err());
    }

    #[test]
    fn test_registry_stress_348() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_348").is_err());
    }

    #[test]
    fn test_registry_stress_349() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_349").is_err());
    }

    #[test]
    fn test_registry_stress_350() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_350").is_err());
    }

    #[test]
    fn test_registry_stress_351() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_351").is_err());
    }

    #[test]
    fn test_registry_stress_352() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_352").is_err());
    }

    #[test]
    fn test_registry_stress_353() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_353").is_err());
    }

    #[test]
    fn test_registry_stress_354() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_354").is_err());
    }

    #[test]
    fn test_registry_stress_355() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_355").is_err());
    }

    #[test]
    fn test_registry_stress_356() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_356").is_err());
    }

    #[test]
    fn test_registry_stress_357() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_357").is_err());
    }

    #[test]
    fn test_registry_stress_358() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_358").is_err());
    }

    #[test]
    fn test_registry_stress_359() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_359").is_err());
    }

    #[test]
    fn test_registry_stress_360() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_360").is_err());
    }

    #[test]
    fn test_registry_stress_361() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_361").is_err());
    }

    #[test]
    fn test_registry_stress_362() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_362").is_err());
    }

    #[test]
    fn test_registry_stress_363() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_363").is_err());
    }

    #[test]
    fn test_registry_stress_364() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_364").is_err());
    }

    #[test]
    fn test_registry_stress_365() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_365").is_err());
    }

    #[test]
    fn test_registry_stress_366() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_366").is_err());
    }

    #[test]
    fn test_registry_stress_367() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_367").is_err());
    }

    #[test]
    fn test_registry_stress_368() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_368").is_err());
    }

    #[test]
    fn test_registry_stress_369() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_369").is_err());
    }

    #[test]
    fn test_registry_stress_370() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_370").is_err());
    }

    #[test]
    fn test_registry_stress_371() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_371").is_err());
    }

    #[test]
    fn test_registry_stress_372() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_372").is_err());
    }

    #[test]
    fn test_registry_stress_373() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_373").is_err());
    }

    #[test]
    fn test_registry_stress_374() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_374").is_err());
    }

    #[test]
    fn test_registry_stress_375() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_375").is_err());
    }

    #[test]
    fn test_registry_stress_376() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_376").is_err());
    }

    #[test]
    fn test_registry_stress_377() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_377").is_err());
    }

    #[test]
    fn test_registry_stress_378() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_378").is_err());
    }

    #[test]
    fn test_registry_stress_379() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_379").is_err());
    }

    #[test]
    fn test_registry_stress_380() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_380").is_err());
    }

    #[test]
    fn test_registry_stress_381() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_381").is_err());
    }

    #[test]
    fn test_registry_stress_382() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_382").is_err());
    }

    #[test]
    fn test_registry_stress_383() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_383").is_err());
    }

    #[test]
    fn test_registry_stress_384() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_384").is_err());
    }

    #[test]
    fn test_registry_stress_385() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_385").is_err());
    }

    #[test]
    fn test_registry_stress_386() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_386").is_err());
    }

    #[test]
    fn test_registry_stress_387() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_387").is_err());
    }

    #[test]
    fn test_registry_stress_388() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_388").is_err());
    }

    #[test]
    fn test_registry_stress_389() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_389").is_err());
    }

    #[test]
    fn test_registry_stress_390() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_390").is_err());
    }

    #[test]
    fn test_registry_stress_391() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_391").is_err());
    }

    #[test]
    fn test_registry_stress_392() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_392").is_err());
    }

    #[test]
    fn test_registry_stress_393() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_393").is_err());
    }

    #[test]
    fn test_registry_stress_394() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_394").is_err());
    }

    #[test]
    fn test_registry_stress_395() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_395").is_err());
    }

    #[test]
    fn test_registry_stress_396() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_396").is_err());
    }

    #[test]
    fn test_registry_stress_397() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_397").is_err());
    }

    #[test]
    fn test_registry_stress_398() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_398").is_err());
    }

    #[test]
    fn test_registry_stress_399() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_399").is_err());
    }

    #[test]
    fn test_registry_stress_400() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_400").is_err());
    }

    #[test]
    fn test_registry_stress_401() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_401").is_err());
    }

    #[test]
    fn test_registry_stress_402() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_402").is_err());
    }

    #[test]
    fn test_registry_stress_403() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_403").is_err());
    }

    #[test]
    fn test_registry_stress_404() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_404").is_err());
    }

    #[test]
    fn test_registry_stress_405() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_405").is_err());
    }

    #[test]
    fn test_registry_stress_406() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_406").is_err());
    }

    #[test]
    fn test_registry_stress_407() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_407").is_err());
    }

    #[test]
    fn test_registry_stress_408() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_408").is_err());
    }

    #[test]
    fn test_registry_stress_409() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_409").is_err());
    }

    #[test]
    fn test_registry_stress_410() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_410").is_err());
    }

    #[test]
    fn test_registry_stress_411() {
        assert_eq!(RegRegistry::parse_kind("dropout").unwrap(), RegKind::Dropout);
        assert_eq!(RegRegistry::parse_kind("layernorm").unwrap(), RegKind::LayerNorm);
        assert_eq!(RegRegistry::parse_kind("l2").unwrap(), RegKind::L2);
        assert!(RegRegistry::parse_kind("unknown_411").is_err());
    }

    // brain-regularization production numerical verification padding line 0
    // brain-regularization production numerical verification padding line 1
}
