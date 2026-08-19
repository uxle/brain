//! # SimCLR / NT-Xent Loss
//!
//! Normalized Temperature-scaled Cross Entropy loss for self-supervised contrastive learning.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::core::{LossResult, Reduction};
use crate::utils::reduction_apply;

/// Configuration for SimCLR NT-Xent loss.
#[derive(Debug, Clone)]
pub struct SimclrConfig {
    pub temperature: f64,
    pub reduction: Reduction,
}

impl Default for SimclrConfig {
    fn default() -> Self {
        Self { temperature: 0.1, reduction: Reduction::Mean }
    }
}

/// SimCLR / NT-Xent loss module.
#[derive(Debug, Clone, Default)]
pub struct SimCLRLoss {
    pub config: SimclrConfig,
}

impl SimCLRLoss {
    pub fn new(config: SimclrConfig) -> Self {
        Self { config }
    }

    pub fn compute(&self, z_i: &Tensor, z_j: &Tensor) -> LossResult<Tensor> {
        let shape = z_i.shape();
        let batch_size = shape[0];
        let dim = if shape.len() > 1 { shape[1] } else { 1 };
        let zi_data = z_i.to_vec();
        let zj_data = z_j.to_vec();
        let t = self.config.temperature;

        let mut losses = vec![0.0f64; batch_size];

        for i in 0..batch_size {
            let zi_slice = &zi_data[i * dim..(i + 1) * dim];
            let zj_slice = &zj_data[i * dim..(i + 1) * dim];

            let pos_sim: f64 = zi_slice.iter().zip(zj_slice.iter()).map(|(&a, &b)| a * b).sum::<f64>() / t;
            let mut sum_exp = pos_sim.exp();

            for k in 0..batch_size {
                if k != i {
                    let zk_slice = &zj_data[k * dim..(k + 1) * dim];
                    let neg_sim: f64 = zi_slice.iter().zip(zk_slice.iter()).map(|(&a, &b)| a * b).sum::<f64>() / t;
                    sum_exp += neg_sim.exp();
                }
            }

            losses[i] = -(pos_sim - sum_exp.ln());
        }

        Ok(reduction_apply(&losses, self.config.reduction))
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
