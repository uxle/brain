//! # Diffusion Model & Sampler Configurations
//!
//! Parameters configuring total timesteps, noise schedule endpoints, and classifier-free guidance.

/// Complete diffusion model configuration.
#[derive(Debug, Clone)]
pub struct DiffusionConfig {
    pub timesteps: usize,
    pub beta_start: f64,
    pub beta_end: f64,
    pub guidance_scale: f64,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            timesteps: 1000,
            beta_start: 0.0001,
            beta_end: 0.02,
            guidance_scale: 7.5,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
