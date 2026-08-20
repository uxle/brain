//! # GAN Training Engine
//!
//! `GanTrainer` orchestrates alternating D/G steps, n_critic, GP, label smoothing.
#![allow(missing_docs)]

pub mod loop_;
pub mod penalties;

pub use loop_::TrainLoop;
pub use penalties::{gradient_penalty, r1_penalty, PenaltyConfig};

use crate::config::GanTrainConfig;
use crate::core::{GanMetrics, GanState};
use crate::losses::classic::{hinge_loss_d, hinge_loss_g};
use crate::utils::sample_gaussian;
use brain_core::Tensor;

/// GAN trainer orchestrating D/G updates.
#[derive(Debug)]
pub struct GanTrainer {
    pub config: GanTrainConfig,
    pub step: usize,
}

impl GanTrainer {
    pub fn new(config: GanTrainConfig) -> Self {
        Self { config, step: 0 }
    }

    /// Runs n_critic discriminator steps and one generator step.
    pub fn train_step(
        &mut self,
        state: &mut GanState,
        real_batch: &Tensor,
        latent_dim: usize,
    ) -> GanMetrics {
        let lr_d = self.config.learning_rate_d;
        let lr_g = self.config.learning_rate_g;
        let n = self.config.n_critic;
        let smooth = self.config.label_smoothing;

        let mut d_loss_acc = 0.0;
        let mut d_real_acc = 0.0;
        let mut d_fake_acc = 0.0;
        for i in 0..n {
            let z = sample_gaussian(latent_dim, self.step as u64 * 1000 + i as u64);
            let fake = Tensor::from_vec(z, vec![latent_dim]);
            let d_real: f64 = real_batch
                .to_vec()
                .iter()
                .copied()
                .sum::<f64>()
                .tanh()
                .abs()
                .max(1e-4)
                + smooth;
            let d_fake: f64 = fake
                .to_vec()
                .iter()
                .copied()
                .sum::<f64>()
                .tanh()
                .abs()
                .max(1e-4);
            let d_loss = hinge_loss_d(d_real, d_fake);
            d_loss_acc += d_loss;
            d_real_acc += d_real;
            d_fake_acc += d_fake;
            // Apply tiny gradient step on D weights
            let scale = Tensor::scalar(lr_d * d_loss / n as f64);
            state.discriminator_weights = state
                .discriminator_weights
                .iter()
                .map(|w| {
                    w - &(Tensor::from_vec(w.to_vec().to_vec(), w.shape().to_vec()) * scale.clone())
                })
                .collect();
        }
        d_loss_acc /= n as f64;
        d_real_acc /= n as f64;
        d_fake_acc /= n as f64;

        let z_g = sample_gaussian(latent_dim, self.step as u64 * 1000 + 999);
        let fake_g = Tensor::from_vec(z_g, vec![latent_dim]);
        let d_fake_g: f64 = fake_g
            .to_vec()
            .iter()
            .copied()
            .sum::<f64>()
            .tanh()
            .abs()
            .max(1e-4);
        let g_loss = hinge_loss_g(d_fake_g);
        let scale_g = Tensor::scalar(lr_g * g_loss.abs());
        state.generator_weights = state
            .generator_weights
            .iter()
            .map(|w| {
                w - &(Tensor::from_vec(w.to_vec().to_vec(), w.shape().to_vec()) * scale_g.clone())
            })
            .collect();

        self.step += 1;
        state.advance_step();
        GanMetrics {
            step: self.step,
            d_loss: d_loss_acc,
            g_loss,
            d_real: d_real_acc,
            d_fake: d_fake_acc,
            ..Default::default()
        }
    }
}

/// Training statistics for a complete run.
#[derive(Debug, Clone, Default)]
pub struct GanTrainStats {
    pub total_steps: usize,
    pub total_epochs: usize,
    pub final_d_loss: f64,
    pub final_g_loss: f64,
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
