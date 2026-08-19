//! # Layer Normalization Module
//!
//! Standard LayerNorm over normalized_shape: y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta.
#![allow(missing_docs)]

pub use crate::normalization::layer::LayerNorm;
pub use crate::normalization::group::GroupNorm;
pub use crate::normalization::rms::RMSNorm;

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
