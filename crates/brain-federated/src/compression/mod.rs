//! # Gradient Compression
//!
//! Communication-efficient compression techniques for federated updates.
#![allow(missing_docs)]

pub mod quantize;
pub mod sparsify;

pub use quantize::{QuantConfig, quantize_tensor, dequantize_tensor};
pub use sparsify::{SparseConfig, top_k_sparsify};

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
