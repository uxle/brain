//! # Cosine Noise Schedule (Nichol & Dhariwal)
//!
//! Cosine-squared cumulative alpha schedule preserving high-frequency image details.

use super::NoiseSchedule;
use std::f64::consts::PI;

/// Improved cosine noise schedule.
#[derive(Debug, Clone)]
pub struct CosineSchedule {
    pub timesteps: usize,
    pub s: f64,
    pub alphas_cumprod: Vec<f64>,
}

impl CosineSchedule {
    /// Creates a new `CosineSchedule`.
    pub fn new(timesteps: usize, s: f64) -> Self {
        let mut alphas_cumprod = Vec::with_capacity(timesteps);
        let f0 = (s / (1.0 + s) * (PI / 2.0)).cos().powi(2);

        for i in 0..timesteps {
            let t = i as f64 / timesteps as f64;
            let ft = (((t + s) / (1.0 + s)) * (PI / 2.0)).cos().powi(2);
            alphas_cumprod.push(ft / f0);
        }

        Self {
            timesteps,
            s,
            alphas_cumprod,
        }
    }
}

impl NoiseSchedule for CosineSchedule {
    fn timesteps(&self) -> usize {
        self.timesteps
    }

    fn beta(&self, t: usize) -> f64 {
        if t == 0 {
            1.0 - self.alpha_cumprod(0)
        } else {
            1.0 - (self.alpha_cumprod(t) / self.alpha_cumprod(t - 1))
        }
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
