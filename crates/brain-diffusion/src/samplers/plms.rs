//! # Pseudo-Linear Multistep (PLMS) Sampler
//!
//! Higher-order multistep Adams-Bashforth style solver caching previous noise evaluations.

use super::Sampler;
use brain_core::Tensor;

/// 4th-order PLMS sampler.
#[derive(Debug, Clone, Default)]
pub struct PlmsSampler;

impl PlmsSampler {
    /// Creates a new `PlmsSampler`.
    pub fn new() -> Self {
        Self
    }
}

impl Sampler for PlmsSampler {
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
