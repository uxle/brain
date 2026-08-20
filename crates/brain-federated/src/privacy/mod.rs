//! # Privacy Module
//!
//! Differential privacy mechanisms and secure aggregation utilities.
#![allow(missing_docs)]

pub mod dp;
pub mod secure_agg;

pub use dp::{add_dp_noise, DpConfig, GaussianNoise};
pub use secure_agg::{mask_tensor, SecureAggregator};

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
