//! # Binary Cross-Entropy Loss (BCE & BCEWithLogits)
//!
//! ## Mathematical Formulation
//!
//! Standard Binary Cross-Entropy:
//! $$\mathcal{L}(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
//!
//! Numerically stable BCE with Logits ($x = \text{logit}$):
//! $$\mathcal{L}(y, x) = \max(x, 0) - x \cdot y + \log(1 + e^{-|x|})$$

use brain_core::Tensor;
use crate::core::{LossError, LossResult, Reduction};
use crate::utils::reduction_apply;

/// Configuration for Binary Cross-Entropy losses.
#[derive(Debug, Clone)]
pub struct BCEConfig {
    pub reduction: Reduction,
    pub pos_weight: Option<f64>,
    pub eps: f64,
}

impl Default for BCEConfig {
    fn default() -> Self {
        Self {
            reduction: Reduction::Mean,
            pos_weight: None,
            eps: 1e-12,
        }
    }
}

/// Binary Cross-Entropy Loss over probabilities in $(0, 1)$.
#[derive(Debug, Clone, Default)]
pub struct BCELoss {
    pub config: BCEConfig,
}

impl BCELoss {
    pub fn new(reduction: Reduction) -> Self {
        Self {
            config: BCEConfig {
                reduction,
                ..Default::default()
            },
        }
    }

    pub fn compute(&self, pred: &Tensor, target: &Tensor) -> LossResult<Tensor> {
        let p_data = pred.data();
        let y_data = target.data();
        if p_data.len() != y_data.len() {
            return Err(LossError::ShapeMismatch {
                expected: pred.shape().to_vec(),
                got: target.shape().to_vec(),
            });
        }

        let eps = self.config.eps;
        let pos_w = self.config.pos_weight.unwrap_or(1.0);
        let mut losses = Vec::with_capacity(p_data.len());

        for (&p_raw, &y) in p_data.iter().zip(y_data.iter()) {
            let p = p_raw.clamp(eps, 1.0 - eps);
            let loss = -(pos_w * y * p.ln() + (1.0 - y) * (1.0 - p).ln());
            losses.push(loss);
        }

        Ok(reduction_apply(&losses, self.config.reduction))
    }
}

/// Numerically stable Binary Cross-Entropy with Logits.
#[derive(Debug, Clone, Default)]
pub struct BCEWithLogitsLoss {
    pub config: BCEConfig,
}

impl BCEWithLogitsLoss {
    pub fn new(reduction: Reduction) -> Self {
        Self {
            config: BCEConfig {
                reduction,
                ..Default::default()
            },
        }
    }

    pub fn compute(&self, logits: &Tensor, target: &Tensor) -> LossResult<Tensor> {
        let x_data = logits.data();
        let y_data = target.data();
        if x_data.len() != y_data.len() {
            return Err(LossError::ShapeMismatch {
                expected: logits.shape().to_vec(),
                got: target.shape().to_vec(),
            });
        }

        let pos_w = self.config.pos_weight.unwrap_or(1.0);
        let mut losses = Vec::with_capacity(x_data.len());

        for (&x, &y) in x_data.iter().zip(y_data.iter()) {
            // max(x, 0) - x * y + log(1 + exp(-|x|))
            let max_x = x.max(0.0);
            let neg_abs = (-x.abs()).exp();
            let log_term = (1.0 + neg_abs).ln();
            let loss = max_x - x * y * pos_w + log_term;
            losses.push(loss);
        }

        Ok(reduction_apply(&losses, self.config.reduction))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bce_loss() {
        let bce = BCELoss::default();
        let p = Tensor::from_slice(&[0.8, 0.2, 0.9], vec![3]);
        let y = Tensor::from_slice(&[1.0, 0.0, 1.0], vec![3]);
        let loss = bce.compute(&p, &y).unwrap();
        assert!(loss.item() > 0.0);
    }

    #[test]
    fn test_bce_with_logits_loss() {
        let bce_logits = BCEWithLogitsLoss::default();
        let logits = Tensor::from_slice(&[2.0, -2.0, 3.0], vec![3]);
        let y = Tensor::from_slice(&[1.0, 0.0, 1.0], vec![3]);
        let loss = bce_logits.compute(&logits, &y).unwrap();
        assert!(loss.item() > 0.0);
    }
}
