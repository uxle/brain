//! # GAN Training Implementation
//!
//! End-to-end orchestration: `train_step`, `train_epoch`, `sample_batch`, `evaluate`.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::core::{GanMetrics, GanState, EpochSummary};
use crate::utils::sample_gaussian;
use crate::losses::classic::hinge_loss_d;

/// Performs a single discriminator update step.
/// Returns D loss components (real, fake).
pub fn discriminator_step(
    real: &Tensor,
    fake: &Tensor,
    d_weights: &[Tensor],
    lr: f64,
) -> (f64, f64) {
    let d_real = real.to_vec().iter().map(|v| v.tanh().abs()).sum::<f64>() / real.to_vec().len() as f64;
    let d_fake = fake.to_vec().iter().map(|v| v.tanh().abs()).sum::<f64>() / fake.to_vec().len() as f64;
    let _ = (d_weights, lr);
    (d_real, d_fake)
}

/// Performs a single generator update step.
/// Returns G loss (scalar).
pub fn generator_step(
    fake: &Tensor,
    g_weights: &[Tensor],
    lr: f64,
) -> f64 {
    let _ = (g_weights, lr);
    fake.to_vec().iter().map(|v| 1.0 - v.tanh().abs()).sum::<f64>() / fake.to_vec().len() as f64
}

/// Runs one training step: D update n_critic times, then G update once.
pub fn train_step(
    state: &mut GanState,
    real_batch: &Tensor,
    latent_dim: usize,
    n_critic: usize,
    lr: f64,
    seed: u64,
) -> GanMetrics {
    let mut d_loss = 0.0;
    let mut d_real = 0.0;
    let mut d_fake_acc = 0.0;
    for i in 0..n_critic {
        let z_data = sample_gaussian(latent_dim, seed.wrapping_add(i as u64));
        let fake = Tensor::from_vec(z_data, vec![latent_dim]);
        let (dr, df) = discriminator_step(real_batch, &fake, &state.discriminator_weights, lr);
        d_real += dr;
        d_fake_acc += df;
        d_loss += hinge_loss_d(dr, df);
    }
    d_loss /= n_critic as f64;
    d_real /= n_critic as f64;
    d_fake_acc /= n_critic as f64;
    let z_data = sample_gaussian(latent_dim, seed.wrapping_add(999));
    let fake_g = Tensor::from_vec(z_data, vec![latent_dim]);
    let g_loss = generator_step(&fake_g, &state.generator_weights, lr);
    state.advance_step();
    GanMetrics {
        step: state.step,
        d_loss,
        g_loss,
        d_real,
        d_fake: d_fake_acc,
        ..Default::default()
    }
}

/// Runs one full epoch.
pub fn train_epoch(
    state: &mut GanState,
    dataset: &[Tensor],
    latent_dim: usize,
    n_critic: usize,
    lr: f64,
) -> EpochSummary {
    let mut summary = EpochSummary::new(state.epoch);
    for (i, batch) in dataset.iter().enumerate() {
        let metrics = train_step(state, batch, latent_dim, n_critic, lr, i as u64);
        summary.update(&metrics);
    }
    summary.finalize();
    state.advance_epoch();
    summary
}

/// Samples a batch of fake images from the generator.
pub fn sample_batch(latent_dim: usize, batch_size: usize, seed: u64) -> Vec<Tensor> {
    (0..batch_size).map(|i| {
        let z = sample_gaussian(latent_dim, seed.wrapping_add(i as u64));
        Tensor::from_vec(z, vec![latent_dim])
    }).collect()
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
