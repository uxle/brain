//! # Unified GAN Model
//!
//! `Gan` struct: generator + discriminator + training state, forward, train_step, sample.
#![allow(missing_docs)]

use crate::config::GanConfig;
use crate::core::{GanMetrics, GanState};
use crate::eval::samples::fixed_latent_sample;
use crate::eval::{eval_gan, GanEvalReport};
use crate::train::GanTrainer;
use brain_core::Tensor;

/// Unified GAN model combining generator and discriminator.
pub struct Gan {
    pub config: GanConfig,
    pub state: GanState,
    trainer: GanTrainer,
}

impl Gan {
    pub fn new(config: GanConfig) -> Self {
        let state = GanState::new(
            vec![Tensor::zeros(vec![
                config.generator.latent_dim,
                config.generator.base_channels,
            ])],
            vec![Tensor::zeros(vec![
                config.discriminator.base_channels,
                config.discriminator.input_channels,
            ])],
        );
        let trainer = GanTrainer::new(config.training.clone());
        Self {
            config,
            state,
            trainer,
        }
    }

    /// Performs one training step.
    pub fn train_step(&mut self, real_batch: &Tensor) -> GanMetrics {
        let latent_dim = self.config.generator.latent_dim;
        self.trainer
            .train_step(&mut self.state, real_batch, latent_dim)
    }

    /// Samples `n` latent-space generated tensors.
    pub fn sample(&self, n: usize, seed: u64) -> Vec<Tensor> {
        fixed_latent_sample(self.config.generator.latent_dim, n, seed)
    }

    /// Evaluates the GAN on provided real tensors.
    pub fn evaluate(&self, real: &[Tensor]) -> GanEvalReport {
        let fake = self.sample(real.len(), 42);
        eval_gan(real, &fake)
    }

    /// Returns a configuration summary string.
    pub fn summary(&self) -> String {
        self.config.summary()
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
