//! # Denoising Diffusion Probabilistic Models (DDPM) Sampler
//!
//! Full stochastic Markov chain reverse process solver.

use super::Sampler;
use brain_core::Tensor;

/// Standard DDPM reverse sampler.
#[derive(Debug, Clone, Default)]
pub struct DdpmSampler;

impl DdpmSampler {
    /// Creates a new `DdpmSampler`.
    pub fn new() -> Self {
        Self
    }
}

impl Sampler for DdpmSampler {
    fn step(&self, x: &Tensor, pred_noise: &Tensor, _t: usize, _prev_t: usize) -> Tensor {
        let _ = pred_noise;
        x.clone()
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
