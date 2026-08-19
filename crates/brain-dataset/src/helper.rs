//! # Dataset Formatting Helpers
//!
//! Pretty printing and tensor shape formatting utilities.

use brain_core::Tensor;

/// Formats a tensor shape into a human-readable string.
pub fn format_tensor_shape(tensor: &Tensor) -> String {
    format!("{:?}", tensor.shape())
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use crate::core::Item;
    use crate::dataset::Dataset;
    use brain_core::Tensor;
}
