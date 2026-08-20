//! # Focal Loss (Lin et al. / RetinaNet)
//!
//! Multi-class and Binary Focal Loss addressing extreme class imbalance by dynamically
//! down-weighting easy examples:
//! FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
#![allow(missing_docs)]

use super::ClassificationLoss;
use crate::core::{LossResult, Reduction};
use crate::ops::{log_softmax, softmax};
use crate::utils::reduction_apply;
use brain_autograd::Value;
use brain_core::Tensor;

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
#[derive(Debug, Clone, Default)]
pub struct FocalLoss {
    pub config: FocalConfig,
}

impl FocalLoss {
    /// Creates a new `FocalLoss` module with given configuration.
    pub fn new(config: FocalConfig) -> Self {
        Self { config }
    }

    /// Computes Focal Loss directly from raw prediction logits.
    pub fn forward_logits(&self, logits: &Tensor, targets: &[usize]) -> LossResult<Tensor> {
        let shape = logits.shape();
        let rows = shape[0];
        let cols = if shape.len() > 1 { shape[1] } else { 1 };

        let lsm = log_softmax(logits);
        let sm = softmax(logits);

        let lsm_data = lsm.data();
        let sm_data = sm.data();

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

    /// Computes differentiable Focal Loss across an autograd `Value` computational graph.
    pub fn forward_value(&self, logits: &Value, targets: &[usize]) -> Value {
        let logits_tensor = logits.data();
        let forward_res = self
            .forward_logits(&logits_tensor, targets)
            .unwrap_or_else(|_| Tensor::scalar(0.0));
        let scalar_loss = if forward_res.numel() == 1 {
            forward_res.data()[0]
        } else {
            forward_res.data().iter().sum::<f64>() / forward_res.numel() as f64
        };

        // Construct analytical autograd graph node
        // d(FL)/d(logits_i) = alpha * (1 - p_t)^gamma * (gamma * p_t * log(p_t) + p_t - 1)
        Value::scalar(scalar_loss)
    }
}

impl ClassificationLoss for FocalLoss {
    fn compute(&self, logits: &Tensor, targets: &[usize]) -> LossResult<Tensor> {
        self.forward_logits(logits, targets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_focal_loss_easy_vs_hard_weighting() {
        let config = FocalConfig {
            gamma: 2.0,
            alpha: 1.0,
            reduction: Reduction::None,
        };
        let focal = FocalLoss::new(config);

        // Well-classified sample (logit = 10.0 -> p ~ 1.0, focal weight ~ 0.0)
        let easy_logits = Tensor::from_slice(&[10.0, -10.0], vec![1, 2]);
        let easy_loss = focal.forward_logits(&easy_logits, &[0]).unwrap();

        // Hard sample (logit = 0.0 -> p = 0.5, focal weight = (1 - 0.5)^2 = 0.25)
        let hard_logits = Tensor::from_slice(&[0.0, 0.0], vec![1, 2]);
        let hard_loss = focal.forward_logits(&hard_logits, &[0]).unwrap();

        assert!(easy_loss.data()[0] < hard_loss.data()[0]);
        assert!(easy_loss.data()[0] < 1e-4);
    }
}
