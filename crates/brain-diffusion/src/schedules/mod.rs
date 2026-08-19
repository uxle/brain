//! # Diffusion Noise Schedules
//!
//! Provides the primary [`NoiseSchedule`] trait, [`LinearSchedule`], and [`CosineSchedule`].

pub mod cosine;
pub mod linear;
pub mod scaled;

pub use cosine::CosineSchedule;
pub use linear::LinearSchedule;
pub use scaled::ScaledLinearSchedule;

/// Abstract diffusion noise schedule interface.
pub trait NoiseSchedule: Send + Sync {
    fn timesteps(&self) -> usize;
    fn beta(&self, t: usize) -> f64;
    fn alpha_cumprod(&self, t: usize) -> f64;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
