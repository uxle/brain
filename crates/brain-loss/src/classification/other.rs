//! # Auxiliary Classification Losses
//!
//! Multi-class Hinge loss, Squared Hinge, Kullback-Leibler (KL) Divergence, and Poisson loss.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::core::{LossResult, Reduction};
use crate::utils::reduction_apply;

/// Classification loss flavor identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClassLossKind {
    #[default]
    Hinge,
    SquaredHinge,
    KLDivergence,
    Poisson,
}

/// Multi-class Hinge Loss: L = max(0, 1 + max_{j != y} s_j - s_y).
pub struct HingeLoss {
    pub margin: f64,
    pub reduction: Reduction,
}

impl Default for HingeLoss {
    fn default() -> Self {
        Self { margin: 1.0, reduction: Reduction::Mean }
    }
}

impl HingeLoss {
    pub fn new(margin: f64, reduction: Reduction) -> Self {
        Self { margin, reduction }
    }

    pub fn compute(&self, scores: &Tensor, targets: &[usize]) -> LossResult<Tensor> {
        let shape = scores.shape();
        let rows = shape[0];
        let cols = if shape.len() > 1 { shape[1] } else { 1 };
        let data = scores.to_vec();

        let n = rows.min(targets.len());
        let mut losses = vec![0.0f64; n];

        for r in 0..n {
            let y = targets[r];
            let y_score = if y < cols { data[r * cols + y] } else { 0.0 };
            let mut max_other = f64::NEG_INFINITY;
            for c in 0..cols {
                if c != y && data[r * cols + c] > max_other {
                    max_other = data[r * cols + c];
                }
            }
            let diff = self.margin + max_other - y_score;
            losses[r] = diff.max(0.0);
        }

        Ok(reduction_apply(&losses, self.reduction))
    }
}

/// Kullback-Leibler Divergence Loss: KL(P || Q) = sum(P * (log(P) - log(Q))).
pub struct KLDivergenceLoss {
    pub reduction: Reduction,
}

impl Default for KLDivergenceLoss {
    fn default() -> Self {
        Self { reduction: Reduction::Mean }
    }
}

impl KLDivergenceLoss {
    pub fn compute(&self, log_prob_q: &Tensor, prob_p: &Tensor) -> LossResult<Tensor> {
        let q_data = log_prob_q.to_vec();
        let p_data = prob_p.to_vec();
        let n = q_data.len().min(p_data.len());

        let mut losses = vec![0.0f64; n];
        for i in 0..n {
            let p = p_data[i].clamp(1e-15, 1.0);
            let log_q = q_data[i];
            losses[i] = p * (p.ln() - log_q);
        }

        Ok(reduction_apply(&losses, self.reduction))
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
