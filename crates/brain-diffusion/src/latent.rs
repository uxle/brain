//! # Latent Diffusion Support
//!
//! Autoencoder (VAE) latent space scaling factors and encode/decode abstractions.

use brain_core::Tensor;

/// Latent space scaling parameters.
#[derive(Debug, Clone)]
pub struct LatentConfig {
    pub scaling_factor: f64,
}

impl Default for LatentConfig {
    fn default() -> Self {
        Self {
            scaling_factor: 0.18215,
        }
    }
}

/// Identity latent encoder/decoder adapter for testing.
pub struct IdentityLatentCodec;

impl IdentityLatentCodec {
    pub fn encode(image: &Tensor) -> Tensor {
        image.clone()
    }

    pub fn decode(latent: &Tensor) -> Tensor {
        latent.clone()
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
