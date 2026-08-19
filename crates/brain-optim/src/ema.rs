//! # Exponential Moving Average of Model Parameters (EMA)
//!
//! Maintains shadow parameter copies with exponential smoothing for test-time evaluation stability.
#![allow(missing_docs)]

use std::collections::HashMap;
use brain_core::Tensor;

/// Configuration container for Model EMA.
#[derive(Debug, Clone, PartialEq)]
pub struct EmaConfig {
    pub decay: f64,
    pub warmup_steps: usize,
}

impl Default for EmaConfig {
    fn default() -> Self {
        Self {
            decay: 0.9999,
            warmup_steps: 0,
        }
    }
}

/// Model Exponential Moving Average Manager.
#[derive(Debug, Clone)]
pub struct ModelEma {
    pub config: EmaConfig,
    pub step_count: usize,
    pub shadow_params: HashMap<usize, Vec<f64>>,
}

impl ModelEma {
    pub fn new(config: EmaConfig) -> Self {
        Self {
            config,
            step_count: 0,
            shadow_params: HashMap::new(),
        }
    }

    /// Initializes shadow buffers from model parameters.
    pub fn init_from_params(&mut self, params: &[Tensor]) {
        self.shadow_params.clear();
        for (i, p) in params.iter().enumerate() {
            self.shadow_params.insert(i, p.data().to_vec());
        }
    }

    /// Updates shadow parameters using exponential smoothing.
    pub fn update(&mut self, params: &[Tensor]) {
        self.step_count += 1;
        let decay = if self.config.warmup_steps > 0 && self.step_count < self.config.warmup_steps {
            (1.0 + self.step_count as f64) / (10.0 + self.step_count as f64)
        } else {
            self.config.decay
        };

        let one_minus_decay = 1.0 - decay;

        for (idx, p) in params.iter().enumerate() {
            let p_data = p.data();
            let shadow = self.shadow_params.entry(idx).or_insert_with(|| p_data.to_vec());
            if shadow.len() != p_data.len() {
                *shadow = p_data.to_vec();
            }

            for i in 0..p_data.len() {
                shadow[i] = decay * shadow[i] + one_minus_decay * p_data[i];
            }
        }
    }

    /// Swaps shadow weights into parameters in-place.
    pub fn copy_to(&self, params: &mut [Tensor]) {
        for (idx, p) in params.iter_mut().enumerate() {
            if let Some(shadow) = self.shadow_params.get(&idx) {
                let p_data = p.data_mut();
                let len = p_data.len().min(shadow.len());
                p_data[..len].copy_from_slice(&shadow[..len]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
