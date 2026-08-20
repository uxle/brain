//! # Triplet Margin Loss
//!
//! Triplet loss: L(a, p, n) = max(0, d(a, p) - d(a, n) + margin).
#![allow(missing_docs)]

use crate::core::{LossResult, Reduction};
use crate::utils::reduction_apply;
use brain_core::Tensor;

/// Configuration for Triplet loss.
#[derive(Debug, Clone)]
pub struct TripletConfig {
    pub margin: f64,
    pub p: f64,
    pub reduction: Reduction,
}

impl Default for TripletConfig {
    fn default() -> Self {
        Self {
            margin: 1.0,
            p: 2.0,
            reduction: Reduction::Mean,
        }
    }
}

/// Triplet margin loss module.
#[derive(Debug, Clone, Default)]
pub struct TripletMarginLoss {
    pub config: TripletConfig,
}

impl TripletMarginLoss {
    pub fn new(config: TripletConfig) -> Self {
        Self { config }
    }

    pub fn compute(
        &self,
        anchor: &Tensor,
        positive: &Tensor,
        negative: &Tensor,
    ) -> LossResult<Tensor> {
        let a = anchor.to_vec();
        let p = positive.to_vec();
        let n = negative.to_vec();

        let num_items = anchor.shape()[0];
        let dim = anchor
            .shape()
            .get(1)
            .copied()
            .unwrap_or(a.len() / num_items.max(1));

        let mut losses = vec![0.0f64; num_items];

        for i in 0..num_items {
            let mut d_pos = 0.0f64;
            let mut d_neg = 0.0f64;

            for d in 0..dim {
                let diff_p = a[i * dim + d] - p[i * dim + d];
                let diff_n = a[i * dim + d] - n[i * dim + d];
                d_pos += diff_p * diff_p;
                d_neg += diff_n * diff_n;
            }

            let dist_p = d_pos.sqrt();
            let dist_n = d_neg.sqrt();

            losses[i] = (dist_p - dist_n + self.config.margin).max(0.0);
        }

        Ok(reduction_apply(&losses, self.config.reduction))
    }
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
