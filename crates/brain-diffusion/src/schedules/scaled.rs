//! # Scaled Linear & Sigmoid Schedules
//!
//! Scaled linear schedules for high-resolution latent diffusion and sigmoid schedules.

use super::NoiseSchedule;

/// Scaled linear schedule.
#[derive(Debug, Clone)]
pub struct ScaledLinearSchedule {
    pub timesteps: usize,
    pub beta_start: f64,
    pub beta_end: f64,
    pub alphas_cumprod: Vec<f64>,
}

impl ScaledLinearSchedule {
    /// Creates a new `ScaledLinearSchedule`.
    pub fn new(timesteps: usize, beta_start: f64, beta_end: f64) -> Self {
        let mut alphas_cumprod = Vec::with_capacity(timesteps);
        let mut cumprod = 1.0;

        for i in 0..timesteps {
            let frac = if timesteps > 1 {
                i as f64 / (timesteps - 1) as f64
            } else {
                0.0
            };
            let b_lin = beta_start.sqrt() + frac * (beta_end.sqrt() - beta_start.sqrt());
            let b = b_lin * b_lin;
            cumprod *= 1.0 - b;
            alphas_cumprod.push(cumprod);
        }

        Self {
            timesteps,
            beta_start,
            beta_end,
            alphas_cumprod,
        }
    }
}

impl NoiseSchedule for ScaledLinearSchedule {
    fn timesteps(&self) -> usize {
        self.timesteps
    }

    fn beta(&self, _t: usize) -> f64 {
        0.01
    }

    fn alpha_cumprod(&self, t: usize) -> f64 {
        self.alphas_cumprod.get(t).copied().unwrap_or(1.0)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
