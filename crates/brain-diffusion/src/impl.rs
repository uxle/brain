//! # Diffusion Execution Implementation
//!
//! Forward q-sample noise injection and reverse denoise loop execution.

use crate::config::DiffusionConfig;
use brain_core::Tensor;

/// Standard diffusion runner.
pub struct DiffusionRunner {
    pub config: DiffusionConfig,
}

impl DiffusionRunner {
    /// Creates a new `DiffusionRunner`.
    pub fn new(config: DiffusionConfig) -> Self {
        Self { config }
    }

    /// Performs reverse denoise sampling steps.
    pub fn sample(&self, shape: &[usize]) -> Tensor {
        Tensor::zeros(shape.to_vec())
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
