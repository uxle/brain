//! # Margin Ranking Loss
//!
//! ## Mathematical Formulation
//!
//! For score pair $(x_1, x_2)$ and target label $y \in \{1, -1\}$:
//! $$\mathcal{L}(x_1, x_2, y) = \max(0, -y \cdot (x_1 - x_2) + \text{margin})$$

use brain_core::Tensor;
use crate::core::{LossResult, Reduction};
use crate::utils::reduction_apply;

/// Margin Ranking Loss.
#[derive(Debug, Clone)]
pub struct MarginRankingLoss {
    pub margin: f64,
    pub reduction: Reduction,
}

impl Default for MarginRankingLoss {
    fn default() -> Self {
        Self { margin: 0.0, reduction: Reduction::Mean }
    }
}

impl MarginRankingLoss {
    pub fn new(margin: f64, reduction: Reduction) -> Self {
        Self { margin, reduction }
    }

    pub fn compute(&self, x1: &Tensor, x2: &Tensor, target: &[f64]) -> LossResult<Tensor> {
        let n = x1.numel().min(x2.numel()).min(target.len());
        let d1 = x1.data();
        let d2 = x2.data();
        let mut losses = Vec::with_capacity(n);

        for i in 0..n {
            let y = target[i];
            let diff = d1[i] - d2[i];
            let loss = (-y * diff + self.margin).max(0.0);
            losses.push(loss);
        }

        Ok(reduction_apply(&losses, self.reduction))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_margin_ranking_loss() {
        let loss_fn = MarginRankingLoss::new(1.0, Reduction::Mean);
        let x1 = Tensor::from_slice(&[3.0, 1.0], vec![2]);
        let x2 = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let y = [1.0, 1.0];
        let loss = loss_fn.compute(&x1, &x2, &y).unwrap();
        assert!(loss.item() >= 0.0);
    }
}
