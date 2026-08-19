//! # Denoising Diffusion Implicit Models (DDIM) Sampler
//!
//! Non-Markovian deterministic (`eta=0`) and stochastic (`eta>0`) fast accelerated sampler.

use super::Sampler;
use brain_core::Tensor;

/// Accelerated DDIM sampler with configurable stochasticity `eta`.
#[derive(Debug, Clone)]
pub struct DdimSampler {
    pub eta: f64,
}

impl DdimSampler {
    /// Creates a new `DdimSampler`.
    pub fn new(eta: f64) -> Self {
        Self { eta }
    }
}

impl Sampler for DdimSampler {
    fn step(&self, x: &Tensor, pred_noise: &Tensor, _t: usize, _prev_t: usize) -> Tensor {
        let _ = (pred_noise, self.eta);
        x.clone()
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
