//! # Training Loop
//!
//! Per-epoch training loop: D updates k times, G update, logging, checkpointing.
#![allow(missing_docs)]

use super::GanTrainer;
use crate::config::GanTrainConfig;
use crate::core::{EpochSummary, GanState};
use crate::utils::log_gan;
use brain_core::Tensor;

/// Full training loop for a GAN.
pub struct TrainLoop {
    pub trainer: GanTrainer,
    pub log_interval: usize,
}

impl TrainLoop {
    pub fn new(config: GanTrainConfig) -> Self {
        Self {
            trainer: GanTrainer::new(config),
            log_interval: 100,
        }
    }

    pub fn with_log_interval(mut self, interval: usize) -> Self {
        self.log_interval = interval;
        self
    }

    /// Runs one epoch over a dataset of batches.
    pub fn epoch(
        &mut self,
        state: &mut GanState,
        dataset: &[Tensor],
        latent_dim: usize,
    ) -> EpochSummary {
        let mut summary = EpochSummary::new(state.epoch);
        for batch in dataset {
            let metrics = self.trainer.train_step(state, batch, latent_dim);
            summary.update(&metrics);
            if metrics.step.is_multiple_of(self.log_interval.max(1)) {
                let _log = log_gan(metrics.step, metrics.d_loss, metrics.g_loss);
            }
        }
        summary.finalize();
        state.advance_epoch();
        summary
    }

    /// Runs multiple epochs and returns per-epoch summaries.
    pub fn run(
        &mut self,
        state: &mut GanState,
        dataset: &[Tensor],
        latent_dim: usize,
        num_epochs: usize,
    ) -> Vec<EpochSummary> {
        (0..num_epochs)
            .map(|_| self.epoch(state, dataset, latent_dim))
            .collect()
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
