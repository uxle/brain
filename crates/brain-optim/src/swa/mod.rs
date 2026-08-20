//! # Stochastic Weight Averaging (SWA)
//!
//! Ensembling along the trajectory of SGD to achieve better generalization and wider minima.
#![allow(missing_docs, clippy::manual_is_multiple_of)]

use brain_core::Tensor;
use std::collections::HashMap;

/// Configuration settings for Stochastic Weight Averaging.
#[derive(Debug, Clone, PartialEq)]
pub struct SwAConfig {
    pub swa_start: usize,
    pub swa_freq: usize,
    pub swa_lr: Option<f64>,
}

impl Default for SwAConfig {
    fn default() -> Self {
        Self {
            swa_start: 10,
            swa_freq: 5,
            swa_lr: None,
        }
    }
}

/// Stochastic Weight Averaging optimizer wrapper.
#[derive(Debug, Clone)]
pub struct SwAOptimizer {
    pub config: SwAConfig,
    pub step_count: usize,
    pub num_averaged: usize,
    pub averaged_weights: HashMap<usize, Vec<f64>>,
}

impl SwAOptimizer {
    pub fn new(config: SwAConfig) -> Self {
        Self {
            config,
            step_count: 0,
            num_averaged: 0,
            averaged_weights: HashMap::new(),
        }
    }

    /// Records current model weights and accumulates SWA running average if epoch matches criteria.
    pub fn update_swa(&mut self, params: &[Tensor]) {
        self.step_count += 1;
        if self.step_count >= self.config.swa_start
            && (self.step_count - self.config.swa_start) % self.config.swa_freq == 0
        {
            self.num_averaged += 1;
            let n = self.num_averaged as f64;

            for (idx, p) in params.iter().enumerate() {
                let p_data = p.data();
                let avg = self
                    .averaged_weights
                    .entry(idx)
                    .or_insert_with(|| vec![0.0; p_data.len()]);
                if avg.len() != p_data.len() {
                    *avg = vec![0.0; p_data.len()];
                }

                for i in 0..p_data.len() {
                    avg[i] = avg[i] * ((n - 1.0) / n) + p_data[i] / n;
                }
            }
        }
    }

    /// Copies averaged weights into parameter tensors.
    pub fn swap_swa_sgd(&self, params: &mut [Tensor]) {
        if self.num_averaged == 0 {
            return;
        }
        for (idx, p) in params.iter_mut().enumerate() {
            if let Some(avg) = self.averaged_weights.get(&idx) {
                let p_data = p.data_mut();
                let len = p_data.len().min(avg.len());
                p_data[..len].copy_from_slice(&avg[..len]);
            }
        }
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
