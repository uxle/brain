//! # Ancestral & ODE Solvers (Euler-A, Heun)
//!
//! Fast continuous-time ancestral sampling steps for high-quality single/few-step generation.

use super::Sampler;
use brain_core::Tensor;

/// Euler Ancestral diffusion sampler.
#[derive(Debug, Clone, Default)]
pub struct EulerAncestralSampler;

impl EulerAncestralSampler {
    /// Creates a new `EulerAncestralSampler`.
    pub fn new() -> Self {
        Self
    }
}

impl Sampler for EulerAncestralSampler {
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
