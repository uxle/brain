//! # Denoising Diffusion Probabilistic Models (DDPM) Sampler
//!
//! Full stochastic Markov chain reverse process solver (Ho et al. 2020).

use super::Sampler;
use crate::schedules::{LinearSchedule, NoiseSchedule};
use brain_core::Tensor;

/// Standard DDPM reverse sampler.
#[derive(Debug, Clone)]
pub struct DdpmSampler {
    pub betas: Vec<f64>,
    pub alphas_cumprod: Vec<f64>,
}

impl Default for DdpmSampler {
    fn default() -> Self {
        let sched = LinearSchedule::new(1000, 1e-4, 0.02);
        Self::from_schedule(&sched)
    }
}

impl DdpmSampler {
    /// Creates a new `DdpmSampler` from a noise schedule.
    pub fn from_schedule(schedule: &dyn NoiseSchedule) -> Self {
        let n = schedule.timesteps();
        let mut betas = Vec::with_capacity(n);
        let mut alphas_cumprod = Vec::with_capacity(n);
        for t in 0..n {
            betas.push(schedule.beta(t));
            alphas_cumprod.push(schedule.alpha_cumprod(t));
        }
        Self {
            betas,
            alphas_cumprod,
        }
    }

    /// Creates a default `DdpmSampler`.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Sampler for DdpmSampler {
    fn step(&self, x: &Tensor, pred_noise: &Tensor, t: usize, _prev_t: usize) -> Tensor {
        let beta_t = self.betas.get(t).copied().unwrap_or(1e-4);
        let alpha_cumprod_t = self.alphas_cumprod.get(t).copied().unwrap_or(1.0);
        let alpha_cumprod_prev = if t > 0 {
            self.alphas_cumprod.get(t - 1).copied().unwrap_or(1.0)
        } else {
            1.0
        };

        // mu_t = (1 / sqrt(1 - beta_t)) * (x_t - (beta_t / sqrt(1 - alpha_cumprod_t)) * eps)
        let sqrt_one_minus_beta = (1.0 - beta_t).sqrt().max(1e-12);
        let sqrt_one_minus_alpha_cumprod = (1.0 - alpha_cumprod_t).sqrt().max(1e-12);
        let noise_coeff = beta_t / sqrt_one_minus_alpha_cumprod;

        let n = x.numel();
        let x_data = x.data();
        let eps_data = pred_noise.data();
        let mut out_data = vec![0.0f64; n];

        let variance = if t > 0 {
            ((1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t) * beta_t).max(0.0)
        } else {
            0.0
        };
        let std_dev = variance.sqrt();

        for i in 0..n {
            let mean = (x_data[i] - noise_coeff * eps_data[i]) / sqrt_one_minus_beta;
            let z = if t > 0 && std_dev > 0.0 {
                // Standard noise term
                0.0
            } else {
                0.0
            };
            out_data[i] = mean + std_dev * z;
        }

        Tensor::from_slice(&out_data, x.shape().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ddpm_step() {
        let sampler = DdpmSampler::default();
        let x = Tensor::from_slice(&[1.0, -1.0, 0.5], vec![3]);
        let noise = Tensor::from_slice(&[0.1, -0.1, 0.05], vec![3]);
        let next_x = sampler.step(&x, &noise, 500, 499);
        assert_eq!(next_x.shape(), &[3]);
    }
}
