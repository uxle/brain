//! # End-to-End Diffusion Model Pipeline
//!
//! Orchestrates the noise schedule, U-Net network, sampling engine, and guidance scale.

pub mod losses;
pub mod train;

pub use losses::eps_loss;
pub use train::DiffusionTrainer;

use crate::config::DiffusionConfig;
use brain_core::Tensor;

/// Complete diffusion model pipeline.
pub struct DiffusionModel {
    pub config: DiffusionConfig,
}

impl DiffusionModel {
    /// Creates a new `DiffusionModel`.
    pub fn new(config: DiffusionConfig) -> Self {
        Self { config }
    }

    /// Generates a sample of the given output shape.
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
