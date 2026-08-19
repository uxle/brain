//! # Diffusion Samplers & Step Solvers
//!
//! Provides the primary [`Sampler`] trait, DDPM, DDIM, Euler Ancestral, and PLMS multistep solvers.

pub mod ancestral;
pub mod ddim;
pub mod ddpm;
pub mod plms;

pub use ancestral::EulerAncestralSampler;
pub use ddim::DdimSampler;
pub use ddpm::DdpmSampler;
pub use plms::PlmsSampler;

use brain_core::Tensor;

/// Abstract diffusion sampling algorithm trait.
pub trait Sampler: Send + Sync {
    fn step(&self, x: &Tensor, pred_noise: &Tensor, t: usize, prev_t: usize) -> Tensor;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
