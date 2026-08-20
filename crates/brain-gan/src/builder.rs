//! # GAN System Builder
//!
//! Ergonomic builder for assembling a complete GAN: generator + discriminator + loss.
#![allow(missing_docs)]

use crate::config::{ArchVariant, GanConfig, LossVariant};

/// Builder for constructing a GAN configuration.
#[derive(Debug, Default)]
pub struct GanBuilder {
    config: GanConfig,
}

impl GanBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn generator(mut self, arch: ArchVariant) -> Self {
        self.config.arch = arch;
        self
    }

    pub fn discriminator(self, _arch: ArchVariant) -> Self {
        self
    }

    pub fn loss(mut self, loss: LossVariant) -> Self {
        self.config.loss = loss;
        self
    }

    pub fn latent_dim(mut self, dim: usize) -> Self {
        self.config.generator.latent_dim = dim;
        self
    }

    pub fn image_size(mut self, size: usize) -> Self {
        self.config.generator.image_size = size;
        self.config.discriminator.image_size = size;
        self
    }

    pub fn base_channels(mut self, ch: usize) -> Self {
        self.config.generator.base_channels = ch;
        self.config.discriminator.base_channels = ch;
        self
    }

    pub fn num_layers(mut self, n: usize) -> Self {
        self.config.generator.num_layers = n;
        self.config.discriminator.num_layers = n;
        self
    }

    pub fn n_critic(mut self, n: usize) -> Self {
        self.config.training.n_critic = n;
        self
    }

    pub fn gradient_penalty(mut self, gp: bool) -> Self {
        self.config.training.gradient_penalty = gp;
        self
    }

    pub fn label_smoothing(mut self, smooth: f64) -> Self {
        self.config.training.label_smoothing = smooth;
        self
    }

    pub fn num_classes(mut self, c: usize) -> Self {
        self.config.generator.num_classes = c;
        self.config.discriminator.num_classes = c;
        self
    }

    pub fn build(self) -> Result<GanConfig, String> {
        self.config.validate()?;
        Ok(self.config)
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
