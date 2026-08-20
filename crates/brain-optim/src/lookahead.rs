//! # Lookahead Optimizer Wrapper
//!
//! Synchronizes fast weights with slow weights every k steps to improve convergence and basin stability.
#![allow(missing_docs, clippy::manual_is_multiple_of)]

use brain_core::Tensor;
use std::collections::HashMap;

/// Configuration settings for Lookahead optimizer.
#[derive(Debug, Clone, PartialEq)]
pub struct LookaheadConfig {
    pub k: usize,
    pub alpha: f64,
}

impl Default for LookaheadConfig {
    fn default() -> Self {
        Self { k: 5, alpha: 0.5 }
    }
}

/// Lookahead Optimizer Wrapper.
#[derive(Debug, Clone)]
pub struct Lookahead {
    pub config: LookaheadConfig,
    pub step_count: usize,
    pub slow_weights: HashMap<usize, Vec<f64>>,
}

impl Lookahead {
    pub fn new(config: LookaheadConfig) -> Self {
        Self {
            config,
            step_count: 0,
            slow_weights: HashMap::new(),
        }
    }

    /// Initializes slow weights from current model parameters.
    pub fn init_slow_weights(&mut self, params: &[Tensor]) {
        self.slow_weights.clear();
        for (i, p) in params.iter().enumerate() {
            self.shadow_set(i, p.data().to_vec());
        }
    }

    fn shadow_set(&mut self, idx: usize, data: Vec<f64>) {
        self.slow_weights.insert(idx, data);
    }

    /// Steps lookahead counter and performs slow weight interpolation when step reaches multiple of k.
    pub fn step_lookahead(&mut self, params: &mut [Tensor]) {
        self.step_count += 1;
        if self.step_count % self.config.k == 0 {
            let alpha = self.config.alpha;
            for (idx, p) in params.iter_mut().enumerate() {
                let p_data = p.data_mut();
                let slow = self
                    .slow_weights
                    .entry(idx)
                    .or_insert_with(|| p_data.to_vec());
                if slow.len() != p_data.len() {
                    *slow = p_data.to_vec();
                }

                for i in 0..p_data.len() {
                    slow[i] += alpha * (p_data[i] - slow[i]);
                }
                let len = p_data.len().min(slow.len());
                p_data[..len].copy_from_slice(&slow[..len]);
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
