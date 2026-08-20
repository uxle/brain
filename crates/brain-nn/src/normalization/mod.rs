//! # Normalization Layers & Protocols
//!
//! BatchNorm, LayerNorm, GroupNorm, InstanceNorm, and RMSNorm layers.
#![allow(missing_docs)]

pub mod batch;
pub mod group;
pub mod instance;
pub mod layer;
pub mod rms;

pub use batch::BatchNorm2d;
pub use group::GroupNorm;
pub use instance::InstanceNorm2d;
pub use layer::LayerNorm;
pub use rms::{RMSNorm, RMSNormConfig};

use brain_core::Tensor;

/// Common trait for normalization layers.
pub trait NormalizationLayer: Send + Sync {
    /// Normalizes input tensor.
    fn forward(&self, input: &Tensor) -> Tensor;
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
