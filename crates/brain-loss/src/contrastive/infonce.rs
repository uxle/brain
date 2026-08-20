//! # InfoNCE Loss
//!
//! Information Noise-Contrastive Estimation loss for self-supervised representation learning.
#![allow(missing_docs)]

use crate::core::{LossResult, Reduction};
use crate::utils::reduction_apply;
use brain_core::Tensor;

/// Configuration for InfoNCE loss.
#[derive(Debug, Clone)]
pub struct InfoNceConfig {
    pub temperature: f64,
    pub reduction: Reduction,
}

impl Default for InfoNceConfig {
    fn default() -> Self {
        Self {
            temperature: 0.07,
            reduction: Reduction::Mean,
        }
    }
}

/// InfoNCE Loss module.
#[derive(Debug, Clone, Default)]
pub struct InfoNCELoss {
    pub config: InfoNceConfig,
}

impl InfoNCELoss {
    pub fn new(config: InfoNceConfig) -> Self {
        Self { config }
    }

    pub fn compute(
        &self,
        queries: &Tensor,
        pos_keys: &Tensor,
        neg_keys: &[Tensor],
    ) -> LossResult<Tensor> {
        let q_data = queries.to_vec();
        let p_data = pos_keys.to_vec();
        let dim = queries
            .shape()
            .get(1)
            .copied()
            .unwrap_or(queries.to_vec().len());
        let num_queries = queries.shape()[0];
        let t = self.config.temperature;

        let mut losses = vec![0.0f64; num_queries];

        for i in 0..num_queries {
            let q_slice = &q_data[i * dim..(i + 1) * dim];
            let p_slice = &p_data[i * dim..(i + 1) * dim];

            let pos_sim: f64 = q_slice
                .iter()
                .zip(p_slice.iter())
                .map(|(&a, &b)| a * b)
                .sum::<f64>()
                / t;
            let mut sum_exp = pos_sim.exp();

            for neg_t in neg_keys {
                let n_data = neg_t.to_vec();
                let neg_sim: f64 = q_slice
                    .iter()
                    .zip(n_data.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f64>()
                    / t;
                sum_exp += neg_sim.exp();
            }

            losses[i] = -(pos_sim - sum_exp.ln());
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
