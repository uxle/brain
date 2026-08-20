//! # Denoising Diffusion Implicit Models (DDIM) Sampler
//!
//! Non-Markovian deterministic (`eta=0`) and stochastic (`eta>0`) fast accelerated sampler (Song et al. 2020).

use super::Sampler;
use crate::schedules::{LinearSchedule, NoiseSchedule};
use brain_core::Tensor;

/// Accelerated DDIM sampler with configurable stochasticity `eta`.
#[derive(Debug, Clone)]
pub struct DdimSampler {
    pub eta: f64,
    pub alphas_cumprod: Vec<f64>,
}

impl Default for DdimSampler {
    fn default() -> Self {
        let sched = LinearSchedule::new(1000, 1e-4, 0.02);
        Self::from_schedule(&sched, 0.0)
    }
}

impl DdimSampler {
    /// Creates a new `DdimSampler` with given schedule and eta.
    pub fn from_schedule(schedule: &dyn NoiseSchedule, eta: f64) -> Self {
        let n = schedule.timesteps();
        let mut alphas_cumprod = Vec::with_capacity(n);
        for t in 0..n {
            alphas_cumprod.push(schedule.alpha_cumprod(t));
        }
        Self {
            eta,
            alphas_cumprod,
        }
    }

    /// Creates a default `DdimSampler` with eta.
    pub fn new(eta: f64) -> Self {
        let sched = LinearSchedule::new(1000, 1e-4, 0.02);
        Self::from_schedule(&sched, eta)
    }
}

impl Sampler for DdimSampler {
    fn step(&self, x: &Tensor, pred_noise: &Tensor, t: usize, prev_t: usize) -> Tensor {
        let alpha_cumprod_t = self.alphas_cumprod.get(t).copied().unwrap_or(1.0);
        let alpha_cumprod_prev = self.alphas_cumprod.get(prev_t).copied().unwrap_or(1.0);

        let sqrt_alpha_t = alpha_cumprod_t.sqrt().max(1e-12);
        let sqrt_one_minus_alpha_t = (1.0 - alpha_cumprod_t).sqrt().max(1e-12);
        let sqrt_alpha_prev = alpha_cumprod_prev.sqrt();

        // Standard deviation: sigma = eta * sqrt((1 - alpha_prev)/(1 - alpha_t)) * sqrt(1 - alpha_t/alpha_prev)
        let sigma = if self.eta > 0.0 && alpha_cumprod_prev > alpha_cumprod_t {
            self.eta
                * ((1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t)).sqrt()
                * (1.0 - alpha_cumprod_t / alpha_cumprod_prev).sqrt()
        } else {
            0.0
        };

        // Direction pointing to x_t
        let dir_coeff = (1.0 - alpha_cumprod_prev - sigma * sigma).max(0.0).sqrt();

        let n = x.numel();
        let x_data = x.data();
        let eps_data = pred_noise.data();
        let mut out_data = vec![0.0f64; n];

        for i in 0..n {
            // Pred x_0
            let pred_x0 = (x_data[i] - sqrt_one_minus_alpha_t * eps_data[i]) / sqrt_alpha_t;
            out_data[i] = sqrt_alpha_prev * pred_x0 + dir_coeff * eps_data[i];
        }

        Tensor::from_slice(&out_data, x.shape().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ddim_step() {
        let sampler = DdimSampler::new(0.0);
        let x = Tensor::from_slice(&[1.0, -1.0, 0.5], vec![3]);
        let noise = Tensor::from_slice(&[0.1, -0.1, 0.05], vec![3]);
        let next_x = sampler.step(&x, &noise, 500, 450);
        assert_eq!(next_x.shape(), &[3]);
    }
}
