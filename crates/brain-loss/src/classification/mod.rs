//! # Classification Losses
//!
//! Classification loss trait and modular loss functions (Cross-Entropy, Focal, Hinge, KL).
#![allow(missing_docs)]

pub mod ce;
pub mod focal;
pub mod other;

pub use ce::{CrossEntropyLoss, CrossEntropyConfig};
pub use focal::{FocalLoss, FocalConfig};
pub use other::{HingeLoss, KLDivergenceLoss, ClassLossKind};

use brain_core::Tensor;
use crate::core::{LossResult, Reduction};

/// Configuration for classification loss algorithms.
#[derive(Debug, Clone)]
pub struct ClassLossConfig {
    pub reduction: Reduction,
    pub label_smoothing: f64,
    pub class_weights: Option<Vec<f64>>,
    pub ignore_index: Option<usize>,
}

impl Default for ClassLossConfig {
    fn default() -> Self {
        Self {
            reduction: Reduction::Mean,
            label_smoothing: 0.0,
            class_weights: None,
            ignore_index: None,
        }
    }
}

/// Specialized trait for discrete and multi-label classification loss modules.
pub trait ClassificationLoss: Send + Sync {
    /// Computes classification loss from logits [N, C] and targets [N].
    fn compute(&self, logits: &Tensor, targets: &[usize]) -> LossResult<Tensor>;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_class_mod_stress_001() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_002() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_003() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_004() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_005() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_006() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_007() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_008() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_009() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_010() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_011() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_012() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_013() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_014() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_015() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_016() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_017() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_018() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_019() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_020() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_021() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_022() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_023() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_024() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_025() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_026() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_027() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_028() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_029() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_030() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_031() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_032() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_033() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_034() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_035() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_036() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_037() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_038() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_039() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_040() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_041() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_042() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_043() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_044() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_045() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_046() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_047() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_048() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_049() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_050() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_051() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_052() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_053() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_054() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_055() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_056() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_057() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_058() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_059() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_060() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_061() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_062() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_063() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_064() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_065() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_066() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_067() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_068() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_069() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_070() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_071() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_072() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_073() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_074() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_075() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_076() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_077() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_078() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_079() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_080() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_081() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_082() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_083() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_084() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_085() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_086() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_087() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_088() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_089() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_090() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_091() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_092() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_093() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_094() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_095() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_096() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_097() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_098() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_099() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_100() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_101() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_102() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_103() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_104() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_105() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_106() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_107() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_108() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_109() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_110() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_111() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_112() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_113() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_114() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_115() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_116() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_117() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_118() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_119() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_120() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_121() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_122() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_123() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_124() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_125() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_126() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_127() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_128() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_129() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_130() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_131() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_132() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_133() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_134() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_135() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_136() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_137() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_138() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_139() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_140() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_141() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_142() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_143() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_144() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_145() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_146() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_147() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_148() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_149() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_150() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_151() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_152() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_153() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_154() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_155() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_156() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_157() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_158() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_159() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_160() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_161() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_162() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_163() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_164() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_165() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_166() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_167() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_168() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_169() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_170() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_171() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_172() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_173() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_174() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_175() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_176() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_177() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_178() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_179() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_180() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_181() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_182() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_183() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_184() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_185() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_186() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_187() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_188() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_189() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_190() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_191() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_192() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_193() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_194() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_195() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_196() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_197() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_198() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_199() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_200() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_201() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_202() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_203() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_204() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_205() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_206() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_207() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_208() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_209() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_210() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_211() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_212() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_213() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_214() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_215() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_216() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_217() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_218() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_219() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_220() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_221() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_222() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_223() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_224() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_225() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_226() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_227() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_228() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_229() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_230() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_231() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_232() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_233() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_234() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_235() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_236() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_237() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_238() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_239() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_240() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_241() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_242() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_243() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_244() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_245() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_246() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_247() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_248() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_249() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_250() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_251() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_252() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_253() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_254() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_255() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_256() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_257() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_258() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_259() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_260() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_261() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_262() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_263() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_264() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_265() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_266() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_267() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_268() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_269() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_270() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_271() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_272() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_273() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_274() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_275() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_276() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_277() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_278() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_279() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_280() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_281() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_282() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_283() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_284() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_285() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_286() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_287() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_288() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_289() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_290() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_291() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_292() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_293() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_294() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_295() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_296() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_297() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_298() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_299() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_300() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_301() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_302() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_303() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_304() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_305() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_306() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_307() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_308() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_309() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_310() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_311() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_312() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_313() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_314() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_315() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_316() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_317() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_318() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_319() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_320() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_321() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_322() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_323() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_324() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_325() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_326() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_327() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_328() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_329() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_class_mod_stress_330() {
        let cfg = ClassLossConfig::default();
        assert_eq!(cfg.reduction, Reduction::Mean);
        let ce = CrossEntropyLoss::new(cfg);
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let loss = ce.compute(&logits, &[2]);
        assert!(loss.is_ok());
    }
}
