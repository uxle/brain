//! # Directional & Angle Losses
//!
//! Cosine embedding loss and angular distance metrics.
#![allow(missing_docs)]

use crate::core::{LossError, LossResult, Reduction};
use crate::utils::reduction_apply;
use brain_core::Tensor;

/// Cosine Embedding Loss measuring angular similarity between embeddings.
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
    pub fn compute(&self, x1: &Tensor, x2: &Tensor, target: &[f64]) -> LossResult<Tensor> {
        let shape1 = x1.shape();
        let shape2 = x2.shape();
        if shape1 != shape2 {
            return Err(LossError::ShapeMismatch {
                expected: shape1.to_vec(),
                got: shape2.to_vec(),
            });
        }

        let rows = shape1[0];
        let cols = if shape1.len() > 1 { shape1[1] } else { 1 };
        let d1 = x1.to_vec();
        let d2 = x2.to_vec();

        let n = rows.min(target.len());
        let mut losses = vec![0.0f64; n];

        for r in 0..n {
            let mut dot = 0.0f64;
            let mut norm1_sq = 0.0f64;
            let mut norm2_sq = 0.0f64;
            for c in 0..cols {
                let v1 = d1[r * cols + c];
                let v2 = d2[r * cols + c];
                dot += v1 * v2;
                norm1_sq += v1 * v1;
                norm2_sq += v2 * v2;
            }
            let cos_sim = dot / (norm1_sq.sqrt() * norm2_sq.sqrt()).max(1e-12);
            let y = target[r];
            losses[r] = if y > 0.0 {
                1.0 - cos_sim
            } else {
                (cos_sim - self.margin).max(0.0)
            };
        }

        Ok(reduction_apply(&losses, self.reduction))
    }
}

/// Angular Distance Loss: theta / pi = arccos(cos_sim) / pi.
#[derive(Debug, Clone, Default)]
pub struct AngularDistanceLoss {
    pub reduction: Reduction,
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
