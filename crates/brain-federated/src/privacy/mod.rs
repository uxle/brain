//! # Privacy Module
//!
//! Differential privacy mechanisms and secure aggregation utilities.
#![allow(missing_docs)]

pub mod dp;
pub mod secure_agg;

pub use dp::{GaussianNoise, DpConfig, add_dp_noise};
pub use secure_agg::{SecureAggregator, mask_tensor};

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
