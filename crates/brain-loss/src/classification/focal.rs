//! # Focal Loss
//!
//! Focal Loss addressing class imbalance by down-weighting easy examples: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t).
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::core::{LossResult, Reduction};
use crate::ops::{log_softmax, softmax};
use crate::utils::reduction_apply;
use super::ClassificationLoss;

/// Configuration for Focal Loss.
#[derive(Debug, Clone)]
pub struct FocalConfig {
    pub gamma: f64,
    pub alpha: f64,
    pub reduction: Reduction,
}

impl Default for FocalConfig {
    fn default() -> Self {
        Self {
            gamma: 2.0,
            alpha: 0.25,
            reduction: Reduction::Mean,
        }
    }
}

/// Multi-class and binary Focal Loss module.
#[derive(Debug, Clone)]
pub struct FocalLoss {
    pub config: FocalConfig,
}

impl FocalLoss {
    pub fn new(config: FocalConfig) -> Self {
        Self { config }
    }

    pub fn forward_logits(&self, logits: &Tensor, targets: &[usize]) -> LossResult<Tensor> {
        let shape = logits.shape();
        let rows = shape[0];
        let cols = if shape.len() > 1 { shape[1] } else { 1 };

        let lsm = log_softmax(logits);
        let sm = softmax(logits);

        let lsm_data = lsm.to_vec();
        let sm_data = sm.to_vec();

        let n = rows.min(targets.len());
        let mut losses = vec![0.0f64; n];

        for r in 0..n {
            let c = targets[r];
            if c < cols {
                let p_t = sm_data[r * cols + c].clamp(1e-12, 1.0);
                let log_p_t = lsm_data[r * cols + c];
                let focal_weight = (1.0 - p_t).powf(self.config.gamma);
                losses[r] = -self.config.alpha * focal_weight * log_p_t;
            }
        }

        Ok(reduction_apply(&losses, self.config.reduction))
    }
}

impl ClassificationLoss for FocalLoss {
    fn compute(&self, logits: &Tensor, targets: &[usize]) -> LossResult<Tensor> {
        self.forward_logits(logits, targets)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
