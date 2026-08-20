//! # InfoNCE Loss (Oord et al. / CPC / SimCLR)
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
    /// Creates a new `InfoNCELoss` instance.
    pub fn new(config: InfoNceConfig) -> Self {
        Self { config }
    }

    /// Computes InfoNCE loss across query tensors and positive/negative key sets.
    pub fn compute(
        &self,
        queries: &Tensor,
        pos_keys: &Tensor,
        neg_keys: &[Tensor],
    ) -> LossResult<Tensor> {
        let q_data = queries.data();
        let p_data = pos_keys.data();
        let dim = queries
            .shape()
            .get(1)
            .copied()
            .unwrap_or_else(|| queries.numel());
        let num_queries = queries.shape()[0];
        let t = self.config.temperature.max(1e-6);

        let mut losses = vec![0.0f64; num_queries];

        for i in 0..num_queries {
            let q_slice = &q_data[i * dim..(i + 1) * dim];
            let p_slice = &p_data[i * dim..(i + 1) * dim];

            // Cosine / dot similarity for positive key
            let pos_sim: f64 = q_slice
                .iter()
                .zip(p_slice.iter())
                .map(|(&a, &b)| a * b)
                .sum::<f64>()
                / t;

            // Collect all negative similarities
            let mut all_sims = Vec::with_capacity(1 + neg_keys.len());
            all_sims.push(pos_sim);

            for neg_t in neg_keys {
                let n_data = neg_t.data();
                let neg_sim: f64 = q_slice
                    .iter()
                    .zip(n_data.iter().take(dim))
                    .map(|(&a, &b)| a * b)
                    .sum::<f64>()
                    / t;
                all_sims.push(neg_sim);
            }

            // Numerically stable Log-Sum-Exp:
            // -log(exp(pos_sim) / sum_j exp(sim_j)) = log(sum_j exp(sim_j)) - pos_sim
            let max_sim = all_sims.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = all_sims.iter().map(|&s| (s - max_sim).exp()).sum();
            let log_sum_exp = max_sim + sum_exp.ln();

            losses[i] = log_sum_exp - pos_sim;
        }

        Ok(reduction_apply(&losses, self.config.reduction))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infonce_loss_exact() {
        let config = InfoNceConfig {
            temperature: 1.0,
            reduction: Reduction::Mean,
        };
        let loss_fn = InfoNCELoss::new(config);

        // Query: [1, 0]
        let q = Tensor::from_slice(&[1.0, 0.0], vec![1, 2]);
        // Positive key: [1, 0] (exact match, sim = 1.0)
        let pos = Tensor::from_slice(&[1.0, 0.0], vec![1, 2]);
        // Negative key: [-1, 0] (opposite, sim = -1.0)
        let neg = Tensor::from_slice(&[-1.0, 0.0], vec![1, 2]);

        let loss = loss_fn.compute(&q, &pos, &[neg]).unwrap();
        // -log(e^1 / (e^1 + e^-1)) = log(1 + e^-2) ~ 0.1269
        let expected = (1.0 + (-2.0f64).exp()).ln();
        assert!((loss.data()[0] - expected).abs() < 1e-5);
    }
}
