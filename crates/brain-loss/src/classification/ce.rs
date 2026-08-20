//! # Cross-Entropy Loss
//!
//! Numerically stable fused log-softmax + NLL with label smoothing, class weights, and ignore index.
#![allow(missing_docs)]

use super::{ClassLossConfig, ClassificationLoss};
use crate::core::LossResult;
use crate::ops::{log_softmax, nll_loss};
use crate::utils::reduction_apply;
use brain_core::Tensor;

pub type CrossEntropyConfig = ClassLossConfig;

/// Cross-Entropy Loss module.
#[derive(Debug, Clone, Default)]
pub struct CrossEntropyLoss {
    pub config: CrossEntropyConfig,
}

impl CrossEntropyLoss {
    pub fn new(config: CrossEntropyConfig) -> Self {
        Self { config }
    }

    pub fn forward_logits(&self, logits: &Tensor, targets: &[usize]) -> LossResult<Tensor> {
        let shape = logits.shape();
        let rows = shape[0];
        let cols = if shape.len() > 1 { shape[1] } else { 1 };

        let lsm = log_softmax(logits);
        let mut sample_losses = nll_loss(&lsm, targets);

        // Apply label smoothing if configured
        if self.config.label_smoothing > 0.0 {
            let lsm_data = lsm.to_vec();
            let eps = self.config.label_smoothing;
            for r in 0..rows.min(targets.len()) {
                let mean_log_prob: f64 =
                    lsm_data[r * cols..(r + 1) * cols].iter().sum::<f64>() / cols as f64;
                sample_losses[r] = (1.0 - eps) * sample_losses[r] - eps * mean_log_prob;
            }
        }

        // Apply class weights if configured
        if let Some(ref weights) = self.config.class_weights {
            for (r, &target_c) in targets.iter().enumerate().take(sample_losses.len()) {
                if target_c < weights.len() {
                    sample_losses[r] *= weights[target_c];
                }
            }
        }

        // Filter ignore_index
        if let Some(ignore) = self.config.ignore_index {
            sample_losses = sample_losses
                .into_iter()
                .enumerate()
                .filter_map(|(i, l)| {
                    if targets.get(i).copied() != Some(ignore) {
                        Some(l)
                    } else {
                        None
                    }
                })
                .collect();
        }

        Ok(reduction_apply(&sample_losses, self.config.reduction))
    }

    /// Differentiable forward pass for cross-entropy with integer class targets.
    pub fn forward_value_logits(
        &self,
        logits: &brain_autograd::Value,
        targets: &[usize],
    ) -> LossResult<brain_autograd::Value> {
        let shape = logits.shape();
        if shape.is_empty() {
            return Err(crate::core::LossError::ShapeMismatch {
                expected: vec![targets.len(), 1],
                got: shape.to_vec(),
            });
        }
        let rows = shape[0];
        let cols = if shape.len() > 1 { shape[1] } else { 1 };

        let lsm = brain_autograd::ops::log_softmax(logits);
        let mut target_probs = vec![0.0f64; rows * cols];
        let eps = self.config.label_smoothing;
        let smooth_uniform = if cols > 0 { eps / (cols as f64) } else { 0.0 };

        for (r, &target) in targets.iter().enumerate().take(rows) {
            for c in 0..cols {
                let idx = r * cols + c;
                if c == target {
                    target_probs[idx] = (1.0 - eps) + smooth_uniform;
                } else {
                    target_probs[idx] = smooth_uniform;
                }
            }
        }
        let target_val = brain_autograd::Value::from_slice(&target_probs, vec![rows, cols]);
        let selected = &lsm * &target_val;

        let loss_sum = selected.sum().neg();
        let final_loss = match self.config.reduction {
            crate::core::Reduction::Mean => {
                let n = rows.max(1) as f64;
                &loss_sum / &brain_autograd::Value::scalar(n)
            }
            crate::core::Reduction::Sum => loss_sum,
            crate::core::Reduction::None => selected.neg(),
        };

        Ok(final_loss)
    }
}

impl ClassificationLoss for CrossEntropyLoss {
    fn compute(&self, logits: &Tensor, targets: &[usize]) -> LossResult<Tensor> {
        self.forward_logits(logits, targets)
    }

    fn forward_value(
        &self,
        logits: &brain_autograd::Value,
        targets: &[usize],
    ) -> LossResult<brain_autograd::Value> {
        self.forward_value_logits(logits, targets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_entropy_label_smoothing() {
        let mut config = CrossEntropyConfig::default();
        config.label_smoothing = 0.1;
        let ce = CrossEntropyLoss::new(config);

        let logits =
            brain_autograd::Value::new(Tensor::from_slice(&[2.0, 1.0, 0.1], vec![1, 3]), true);
        let targets = vec![0];
        let loss = ce.forward_value(&logits, &targets).unwrap();
        assert!(loss.data().item() > 0.0);
        loss.backward().unwrap();
        assert!(logits.grad().is_some());
    }
}
