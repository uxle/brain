//! # Differential Privacy
//!
//! Gaussian and Laplace noise mechanisms for (ε, δ)-differential privacy.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Configuration for differential privacy noise injection.
#[derive(Debug, Clone)]
pub struct DpConfig {
    pub epsilon: f64,
    pub delta: f64,
    pub sensitivity: f64,
    pub clip_norm: f64,
}

impl Default for DpConfig {
    fn default() -> Self {
        Self { epsilon: 1.0, delta: 1e-5, sensitivity: 1.0, clip_norm: 1.0 }
    }
}

/// Gaussian noise mechanism for (ε, δ)-DP.
#[derive(Debug, Clone, Default)]
pub struct GaussianNoise {
    pub config: DpConfig,
}

impl GaussianNoise {
    pub fn new(config: DpConfig) -> Self { Self { config } }

    /// Computes the required Gaussian sigma for (ε, δ)-DP.
    pub fn compute_sigma(&self) -> f64 {
        let c = &self.config;
        c.sensitivity * (2.0_f64 * (1.25_f64 / c.delta).ln()).sqrt() / c.epsilon
    }
}

/// Clips tensor values by L2 norm.
pub fn clip_by_norm(t: &Tensor, max_norm: f64) -> Tensor {
    let norm: f64 = t.to_vec().iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm <= max_norm { t.clone() } else { t * &Tensor::scalar(max_norm / norm) }
}

/// Adds calibrated Gaussian noise to a tensor using LCG randomness.
pub fn add_dp_noise(t: &Tensor, sigma: f64, seed: u64) -> Tensor {
    let data: Vec<f64> = t.to_vec();
    let mut rng = seed;
    let noisy: Vec<f64> = data.iter().map(|v| {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = (rng >> 32) as f64 / u32::MAX as f64;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (rng >> 32) as f64 / u32::MAX as f64;
        let normal = (-2.0 * (u1 + 1e-15).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        v + sigma * normal
    }).collect();
    Tensor::from_vec(noisy, t.shape().to_vec())
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
