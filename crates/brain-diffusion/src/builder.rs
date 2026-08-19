//! # Diffusion Model & U-Net Builders
//!
//! Fluent builders for assembling diffusion pipelines and U-Net backbones.

use crate::config::DiffusionConfig;

/// Fluent builder for diffusion pipelines.
#[derive(Default)]
pub struct DiffusionBuilder {
    config: DiffusionConfig,
}

impl DiffusionBuilder {
    /// Creates a new `DiffusionBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the total number of timesteps.
    pub fn timesteps(mut self, timesteps: usize) -> Self {
        self.config.timesteps = timesteps;
        self
    }

    /// Sets the classifier-free guidance scale.
    pub fn guidance_scale(mut self, scale: f64) -> Self {
        self.config.guidance_scale = scale;
        self
    }

    /// Builds the `DiffusionConfig`.
    pub fn build(self) -> DiffusionConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
