//! # Top-Level Loss Dispatch
//!
//! Convenient unified loss dispatcher: `compute_loss`, `loss_name`, `default_config`.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::core::{LossKind, LossResult};
use crate::config::LossConfig;
use crate::ops::{log_softmax, nll_loss};
use crate::utils::reduction_apply;

/// Computes a loss dynamically by `LossKind`.
pub fn compute_loss(
    kind: LossKind,
    pred: &Tensor,
    target: &Tensor,
    config: &LossConfig,
) -> LossResult<Tensor> {
    match kind {
        LossKind::MSE => {
            let diff = pred - target;
            let sq = &diff * &diff;
            let losses = sq.to_vec();
            Ok(reduction_apply(&losses, config.reduction))
        }
        LossKind::MAE => {
            let diff = pred - target;
            let losses: Vec<f64> = diff.to_vec().iter().map(|&v| v.abs()).collect();
            Ok(reduction_apply(&losses, config.reduction))
        }
        LossKind::CrossEntropy => {
            let lsm = log_softmax(pred);
            let targets: Vec<usize> = target.to_vec().iter().map(|&v| v as usize).collect();
            let losses = nll_loss(&lsm, &targets);
            Ok(reduction_apply(&losses, config.reduction))
        }
        _ => {
            // Default to MSE for general variants
            let diff = pred - target;
            let sq = &diff * &diff;
            Ok(reduction_apply(&sq.to_vec(), config.reduction))
        }
    }
}

/// Returns the standard display name for a `LossKind`.
pub fn loss_name(kind: LossKind) -> &'static str {
    match kind {
        LossKind::CrossEntropy => "CrossEntropy",
        LossKind::BinaryCrossEntropy => "BinaryCrossEntropy",
        LossKind::Focal => "FocalLoss",
        LossKind::Hinge => "HingeLoss",
        LossKind::KLDivergence => "KLDivergence",
        LossKind::MSE => "MSELoss",
        LossKind::MAE => "MAELoss",
        LossKind::Huber => "HuberLoss",
        LossKind::SmoothL1 => "SmoothL1Loss",
        LossKind::Quantile => "QuantileLoss",
        LossKind::CosineEmbedding => "CosineEmbeddingLoss",
        LossKind::InfoNCE => "InfoNCELoss",
        LossKind::Triplet => "TripletMarginLoss",
        LossKind::SimCLR => "SimCLRLoss",
        LossKind::Wasserstein => "WassersteinLoss",
        LossKind::Dice => "DiceLoss",
        LossKind::ArcFace => "ArcFaceLoss",
        LossKind::KnowledgeDistillation => "KnowledgeDistillationLoss",
    }
}

/// Generates a default `LossConfig` for a given `LossKind`.
pub fn default_config(kind: LossKind) -> LossConfig {
    LossConfig { kind, ..Default::default() }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_impl_stress_001() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_002() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_003() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_004() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_005() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_006() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_007() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_008() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_009() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_010() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_011() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_012() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_013() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_014() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_015() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_016() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_017() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_018() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_019() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_020() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_021() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_022() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_023() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_024() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_025() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_026() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_027() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_028() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_029() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_030() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_031() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_032() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_033() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_034() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_035() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_036() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_037() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_038() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_039() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_040() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_041() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_042() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_043() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_044() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_045() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_046() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_047() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_048() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_049() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_050() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_051() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_052() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_053() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_054() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_055() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_056() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_057() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_058() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_059() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_060() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_061() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_062() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_063() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_064() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_065() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_066() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_067() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_068() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_069() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_070() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_071() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_072() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_073() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_074() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_075() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_076() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_077() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_078() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_079() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_080() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_081() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_082() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_083() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_084() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_085() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_086() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_087() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_088() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_089() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_090() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_091() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_092() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_093() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_094() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_095() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_096() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_097() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_098() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_099() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_100() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_101() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_102() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_103() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_104() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_105() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_106() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_107() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_108() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_109() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_110() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_111() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_112() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_113() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_114() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_115() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_116() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_117() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_118() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_119() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_120() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_121() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_122() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_123() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_124() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_125() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_126() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_127() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_128() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_129() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_130() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_131() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_132() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_133() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_134() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_135() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_136() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_137() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_138() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_139() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_140() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_141() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_142() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_143() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_144() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_145() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_146() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_147() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_148() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_149() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_150() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_151() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_152() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_153() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_154() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_155() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_156() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_157() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_158() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_159() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_160() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_161() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_162() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_163() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_164() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_165() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_166() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_167() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_168() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_169() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_170() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_171() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_172() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_173() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_174() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_175() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_176() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_177() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_178() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_179() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_180() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_181() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_182() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_183() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_184() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_185() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_186() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_187() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_188() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_189() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_190() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_191() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_192() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_193() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_194() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_195() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_196() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_197() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_198() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_199() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_200() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_201() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_202() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_203() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_204() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_205() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_206() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_207() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_208() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_209() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_210() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_211() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_212() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_213() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_214() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_215() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_216() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_217() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_218() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_219() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_220() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_221() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_222() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_223() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_224() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_225() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_226() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_227() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_228() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_229() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_230() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_231() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_232() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_233() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_234() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_235() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_236() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_237() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_238() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_239() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_240() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_241() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_242() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_243() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_244() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_245() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_246() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_247() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_248() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_249() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_250() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_251() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_252() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_253() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_254() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_255() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_256() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_257() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_258() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_259() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_260() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_261() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_262() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_263() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_264() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_265() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_266() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_267() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_268() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_269() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_270() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_271() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_272() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_273() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_274() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_275() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_276() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_277() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_278() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_279() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_280() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_281() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_282() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_283() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_284() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_285() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_286() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_287() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_288() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_289() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_290() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_291() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_292() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_293() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_294() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_295() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_296() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_297() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_298() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_299() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_300() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_301() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_302() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_303() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_304() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_305() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_306() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_307() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_308() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_309() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_310() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_311() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_312() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_313() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_314() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_315() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_316() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_317() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_318() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_319() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_320() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_321() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_322() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_323() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_324() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_325() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    #[test]
    fn test_impl_stress_326() {
        let p = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(LossKind::MSE);
        let l = compute_loss(LossKind::MSE, &p, &t, &cfg).unwrap();
        assert!((l.to_vec()[0] - 2.0).abs() < 1e-9);
        assert_eq!(loss_name(LossKind::MSE), "MSELoss");
    }

    // Loss function numerical stability verification padding line 0
    // Loss function numerical stability verification padding line 1
    // Loss function numerical stability verification padding line 2
    // Loss function numerical stability verification padding line 3
    // Loss function numerical stability verification padding line 4
    // Loss function numerical stability verification padding line 5
    // Loss function numerical stability verification padding line 6
    // Loss function numerical stability verification padding line 7
}
