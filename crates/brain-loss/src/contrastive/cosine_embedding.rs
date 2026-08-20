//! # Cosine Embedding Loss
//!
//! ## Mathematical Formulation
//!
//! Measures cosine distance between pairs $(x_1, x_2)$ given target $y \in \{1, -1\}$:
//! $$\mathcal{L}(x_1, x_2, y) = \begin{cases} 1 - \cos(x_1, x_2) & \text{if } y = 1 \\ \max(0, \cos(x_1, x_2) - \text{margin}) & \text{if } y = -1 \end{cases}$$

use crate::core::{LossResult, Reduction};
use crate::utils::reduction_apply;
use brain_core::Tensor;

/// Cosine Embedding Loss.
#[derive(Debug, Clone)]
pub struct CosineEmbeddingLoss {
    pub margin: f64,
    pub reduction: Reduction,
}

impl Default for CosineEmbeddingLoss {
    fn default() -> Self {
        Self {
            margin: 0.0,
            reduction: Reduction::Mean,
        }
    }
}

impl CosineEmbeddingLoss {
    pub fn new(margin: f64, reduction: Reduction) -> Self {
        Self { margin, reduction }
    }

    pub fn compute(&self, x1: &Tensor, x2: &Tensor, target: &[f64]) -> LossResult<Tensor> {
        let dim = x1.shape().last().copied().unwrap_or(1);
        let n = x1.numel() / dim;
        let mut losses = Vec::with_capacity(n);

        let d1 = x1.data();
        let d2 = x2.data();

        for i in 0..n {
            let s1 = &d1[i * dim..(i + 1) * dim];
            let s2 = &d2[i * dim..(i + 1) * dim];
            let y = target.get(i).copied().unwrap_or(1.0);

            let dot: f64 = s1.iter().zip(s2.iter()).map(|(a, b)| a * b).sum();
            let norm1: f64 = s1.iter().map(|a| a * a).sum::<f64>().sqrt().max(1e-12);
            let norm2: f64 = s2.iter().map(|b| b * b).sum::<f64>().sqrt().max(1e-12);
            let cos = dot / (norm1 * norm2);

            let loss = if y > 0.0 {
                1.0 - cos
            } else {
                (cos - self.margin).max(0.0)
            };
            losses.push(loss);
        }

        Ok(reduction_apply(&losses, self.reduction))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_embedding_loss() {
        let loss_fn = CosineEmbeddingLoss::default();
        let x1 = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let x2 = Tensor::from_slice(&[1.0, 0.0, 0.0, -1.0], vec![2, 2]);
        let y = [1.0, -1.0];
        let loss = loss_fn.compute(&x1, &x2, &y).unwrap();
        assert!(loss.item() >= 0.0);
    }
}
