//! # Linear Beta Noise Schedule (DDPM)
//!
//! Standard linearly spaced beta values from `beta_start` to `beta_end`.

use super::NoiseSchedule;

/// Standard linear beta schedule.
#[derive(Debug, Clone)]
pub struct LinearSchedule {
    pub timesteps: usize,
    pub beta_start: f64,
    pub beta_end: f64,
    pub betas: Vec<f64>,
    pub alphas_cumprod: Vec<f64>,
}

impl LinearSchedule {
    /// Creates a new `LinearSchedule`.
    pub fn new(timesteps: usize, beta_start: f64, beta_end: f64) -> Self {
        let mut betas = Vec::with_capacity(timesteps);
        let mut alphas_cumprod = Vec::with_capacity(timesteps);
        let mut cumprod = 1.0;

        for i in 0..timesteps {
            let frac = if timesteps > 1 {
                i as f64 / (timesteps - 1) as f64
            } else {
                0.0
            };
            let b = beta_start + frac * (beta_end - beta_start);
            betas.push(b);
            cumprod *= 1.0 - b;
            alphas_cumprod.push(cumprod);
        }

        Self {
            timesteps,
            beta_start,
            beta_end,
            betas,
            alphas_cumprod,
        }
    }
}

impl NoiseSchedule for LinearSchedule {
    fn timesteps(&self) -> usize {
        self.timesteps
    }

    fn beta(&self, t: usize) -> f64 {
        self.betas.get(t).copied().unwrap_or(0.0)
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
