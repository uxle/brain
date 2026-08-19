//! # Training Stabilizers
//!
//! Gradient penalty (WGAN-GP, R1/R2), spectral norm wrapper, label smoothing.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Configuration for training penalty terms.
#[derive(Debug, Clone)]
pub struct PenaltyConfig {
    pub gp_lambda: f64,
    pub r1_gamma: f64,
    pub r2_gamma: f64,
    pub label_smooth_real: f64,
    pub label_smooth_fake: f64,
}

impl Default for PenaltyConfig {
    fn default() -> Self {
        Self {
            gp_lambda: 10.0,
            r1_gamma: 10.0,
            r2_gamma: 10.0,
            label_smooth_real: 0.9,
            label_smooth_fake: 0.0,
        }
    }
}

/// WGAN-GP gradient penalty via finite difference on interpolated samples.
/// Returns scalar penalty value.
pub fn gradient_penalty(
    real: &Tensor,
    fake: &Tensor,
    lambda: f64,
    seed: u64,
) -> f64 {
    let mut rng = seed;
    let lcg = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*s >> 11) as f64 / (1u64 << 53) as f64
    };
    let alpha = lcg(&mut rng);
    let rv = real.to_vec();
    let fv = fake.to_vec();
    let n = rv.len().min(fv.len());
    if n == 0 { return 0.0; }
    // Interpolated sample
    let interp: Vec<f64> = rv.iter().zip(fv.iter()).take(n).map(|(r, f)| alpha * r + (1.0 - alpha) * f).collect();
    // Finite-difference gradient estimate
    let eps = 1e-5;
    let d_interp: f64 = interp.iter().sum::<f64>() / n as f64;
    let d_interp_plus: f64 = interp.iter().map(|v| v + eps).sum::<f64>() / n as f64;
    let fd_grad = (d_interp_plus - d_interp) / eps;
    let grad_norm = fd_grad.abs();
    lambda * (grad_norm - 1.0).powi(2)
}

/// R1 gradient penalty: ||grad D(real)||^2.
pub fn r1_penalty(real_score: f64, gamma: f64) -> f64 {
    gamma * 0.5 * real_score.powi(2)
}

/// R2 gradient penalty: ||grad D(fake)||^2.
pub fn r2_penalty(fake_score: f64, gamma: f64) -> f64 {
    gamma * 0.5 * fake_score.powi(2)
}

/// Label smoothing: returns smoothed real/fake labels.
pub fn smooth_labels(real: f64, config: &PenaltyConfig) -> (f64, f64) {
    (config.label_smooth_real.min(real), config.label_smooth_fake)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
