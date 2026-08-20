//! # Brain Diffusion — Production-Grade Deep Generative Diffusion Suite
//!
//! Features comprehensive noise schedules, sampling algorithms (DDPM, DDIM, Euler, PLMS),
//! Classifier-Free Guidance (CFG), full U-Net 2D backbones, and training pipelines.
//!
//! ## Subsystems
//!
//! * [`schedules`] - Linear, Cosine, Scaled Linear, and Sigmoid noise schedules
//! * [`samplers`] - DDPM, DDIM, Euler Ancestral, and PLMS multistep samplers
//! * [`guidance`] - Classifier-Free Guidance (CFG) and dynamic thresholding
//! * [`unet`] - 2D U-Net with ResBlocks, SpatialTransformers, and Timestep Embeddings
//! * [`diffusion`] - End-to-end `DiffusionModel` and training loss routines
//! * [`conditioning`] - Text cross-attention, class labels, and image inpainting conditioning
//! * [`latent`] - Latent diffusion VAE encode/decode adapters
//! * [`eval`] - Generative evaluation metrics (FID-lite, IS-lite)
//!
//! ## Quick Start Example
//!
//! ```rust
//! use brain_diffusion::prelude::*;
//!
//! let cfg = DiffusionConfig::default();
//! assert_eq!(cfg.timesteps, 1000);
//! ```

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::excessive_precision,
    clippy::identity_op,
    clippy::derivable_impls,
    clippy::manual_clamp,
    clippy::type_complexity
)]

pub mod builder;
pub mod conditioning;
pub mod config;
pub mod core;
pub mod diffusion;
pub mod eval;
pub mod guidance;
pub mod r#impl;
pub mod latent;
pub mod ops;
pub mod samplers;
pub mod schedules;
pub mod unet;
pub mod utils;

// Re-exports
pub use builder::DiffusionBuilder;
pub use config::DiffusionConfig;
pub use core::DiffusionState;
pub use diffusion::DiffusionModel;
pub use samplers::{DdimSampler, DdpmSampler, EulerAncestralSampler, PlmsSampler, Sampler};
pub use schedules::{CosineSchedule, LinearSchedule, NoiseSchedule};
pub use unet::{Unet2d, UnetConfig};

/// Package version string.
pub const VERSION: &str = "0.2.0";
pub const MAJOR_VERSION: u32 = 0;
pub const MINOR_VERSION: u32 = 2;
pub const PATCH_VERSION: u32 = 0;

/// Returns the crate version triple.
///
/// ```rust
/// use brain_diffusion::version_tuple;
/// assert_eq!(version_tuple(), (0, 2, 0));
/// ```
pub fn version_tuple() -> (u32, u32, u32) {
    (MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION)
}

/// Returns a formatted version string.
///
/// ```rust
/// use brain_diffusion::version_string;
/// assert_eq!(version_string(), "brain-diffusion v0.2.0");
/// ```
pub fn version_string() -> String {
    format!("brain-diffusion v{}", VERSION)
}

/// Standard prelude imports for diffusion models.
///
/// ```rust
/// use brain_diffusion::prelude::*;
/// let cfg = DiffusionConfig::default();
/// assert_eq!(cfg.timesteps, 1000);
/// ```
pub mod prelude {
    pub use crate::builder::DiffusionBuilder;
    pub use crate::config::DiffusionConfig;
    pub use crate::core::DiffusionState;
    pub use crate::diffusion::DiffusionModel;
    pub use crate::schedules::{CosineSchedule, LinearSchedule, NoiseSchedule};
    pub use crate::unet::{Unet2d, UnetConfig};
    pub use brain_core::Tensor;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
