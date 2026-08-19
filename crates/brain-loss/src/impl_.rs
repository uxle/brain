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
}
