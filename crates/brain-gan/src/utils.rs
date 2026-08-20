//! # GAN Utility Functions
//!
//! Seed management, EMA tracking, logging, and math helpers.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Sets a global seed for deterministic sampling (stored in thread-local LCG state).
static mut GLOBAL_SEED: u64 = 42;

pub fn set_seed(seed: u64) {
    unsafe {
        GLOBAL_SEED = seed;
    }
}

pub fn next_rand() -> f64 {
    let s = unsafe {
        GLOBAL_SEED = GLOBAL_SEED
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        GLOBAL_SEED
    };
    (s >> 11) as f64 / (1u64 << 53) as f64
}

/// Box-Muller transform: two uniform samples -> one standard normal.
pub fn box_muller(u1: f64, u2: f64) -> f64 {
    (-2.0 * (u1.max(1e-15)).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Samples a gaussian latent vector of size `dim` with given seed.
pub fn sample_gaussian(dim: usize, seed: u64) -> Vec<f64> {
    let mut rng = seed;
    let lcg = |s: &mut u64| -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..dim)
        .map(|_| {
            let u1 = lcg(&mut rng).max(1e-15);
            let u2 = lcg(&mut rng);
            box_muller(u1, u2)
        })
        .collect()
}

/// Updates exponential moving average: ema = decay*ema + (1-decay)*new.
pub fn track_ema(ema: &[Tensor], new_weights: &[Tensor], decay: f64) -> Vec<Tensor> {
    let d = Tensor::scalar(decay);
    let one_d = Tensor::scalar(1.0 - decay);
    ema.iter()
        .zip(new_weights.iter())
        .map(|(e, n)| &(e * &d) + &(n * &one_d))
        .collect()
}

/// Logs a GAN training step summary to a string.
pub fn log_gan(step: usize, d_loss: f64, g_loss: f64) -> String {
    format!("[step {:06}] D={:.4} G={:.4}", step, d_loss, g_loss)
}

/// Clips tensor values element-wise into [-clip, clip].
pub fn clip_weights(t: &Tensor, clip: f64) -> Tensor {
    let data: Vec<f64> = t.to_vec().iter().map(|v| v.clamp(-clip, clip)).collect();
    Tensor::from_vec(data, t.shape().to_vec())
}

/// Computes element-wise sigmoid.
pub fn sigmoid_vec(x: &[f64]) -> Vec<f64> {
    x.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect()
}

/// Computes binary cross-entropy loss for scalars: -[y*log(p) + (1-y)*log(1-p)].
pub fn bce_scalar(pred: f64, label: f64) -> f64 {
    let p = pred.clamp(1e-7, 1.0 - 1e-7);
    -(label * p.ln() + (1.0 - label) * (1.0 - p).ln())
}

/// L2 norm of a flat tensor.
pub fn l2_norm(t: &Tensor) -> f64 {
    t.to_vec().iter().map(|v| v * v).sum::<f64>().sqrt()
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
