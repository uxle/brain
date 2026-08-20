//! # CycleGAN-Lite
//!
//! Paired/unpaired cycle-consistency loss, identity loss.
#![allow(missing_docs)]

use brain_core::Tensor;

/// CycleGAN configuration.
#[derive(Debug, Clone)]
pub struct CycleConfig {
    pub cycle_lambda: f64,
    pub identity_lambda: f64,
    pub latent_dim: usize,
}

impl Default for CycleConfig {
    fn default() -> Self {
        Self {
            cycle_lambda: 10.0,
            identity_lambda: 5.0,
            latent_dim: 64,
        }
    }
}

/// Cycle-consistency loss: ||G_B(G_A(x)) - x||_1.
pub fn cycle_consistency_loss(reconstructed: &Tensor, original: &Tensor) -> f64 {
    let rv = reconstructed.to_vec();
    let ov = original.to_vec();
    let n = rv.len().min(ov.len());
    if n == 0 {
        return 0.0;
    }
    rv.iter()
        .zip(ov.iter())
        .take(n)
        .map(|(r, o)| (r - o).abs())
        .sum::<f64>()
        / n as f64
}

/// Identity loss: ||G_A(y) - y||_1.
pub fn identity_loss(identity_output: &Tensor, real: &Tensor) -> f64 {
    cycle_consistency_loss(identity_output, real)
}

/// Total CycleGAN loss for one domain direction.
pub fn cycle_total_loss(adv_loss: f64, cycle_loss: f64, id_loss: f64, config: &CycleConfig) -> f64 {
    adv_loss + config.cycle_lambda * cycle_loss + config.identity_lambda * id_loss
}

/// CycleGAN-lite model stub (generator pair).
pub struct CycleGanLite {
    pub config: CycleConfig,
    pub g_a2b_weights: Vec<Tensor>,
    pub g_b2a_weights: Vec<Tensor>,
}

impl CycleGanLite {
    pub fn new(config: CycleConfig) -> Self {
        let w = vec![Tensor::zeros(vec![config.latent_dim])];
        Self {
            config,
            g_a2b_weights: w.clone(),
            g_b2a_weights: w,
        }
    }

    /// Simulated G_A(x): maps domain A sample forward.
    pub fn generate_a2b(&self, x: &Tensor) -> Tensor {
        let data: Vec<f64> = x.to_vec().iter().map(|v| v.tanh()).collect();
        Tensor::from_vec(data, x.shape().to_vec())
    }

    /// Simulated G_B(y): maps domain B sample back.
    pub fn generate_b2a(&self, y: &Tensor) -> Tensor {
        let data: Vec<f64> = y.to_vec().iter().map(|v| v.tanh()).collect();
        Tensor::from_vec(data, y.shape().to_vec())
    }

    /// Computes cycle loss for one batch sample.
    pub fn cycle_loss_for(&self, x: &Tensor) -> f64 {
        let fake_b = self.generate_a2b(x);
        let recon_a = self.generate_b2a(&fake_b);
        cycle_consistency_loss(&recon_a, x)
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
