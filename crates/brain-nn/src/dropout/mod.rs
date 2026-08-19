//! # Regularization & Dropout Layers
//!
//! Standard Bernoulli Dropout, AlphaDropout for SELU activations, and Spatial/Channel Dropout.
#![allow(missing_docs)]

#[allow(clippy::module_inception)]
pub mod dropout;
pub mod alpha;

pub use dropout::{Dropout, FusedDropout};
pub use alpha::{AlphaDropout, Dropout2d};

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
