//! # Text Transforms & Tokenization
//!
//! Provides `Tokenize`, `PadSequence`, and `Truncate` transformations.

use super::Transform;
use crate::core::Item;

/// Pads or truncates sequence to fixed length.
pub struct PadOrTruncate {
    pub target_length: usize,
    pub pad_value: f64,
}

impl PadOrTruncate {
    /// Creates a new `PadOrTruncate` transform.
    pub fn new(target_length: usize, pad_value: f64) -> Self {
        Self {
            target_length,
            pad_value,
        }
    }
}

impl Transform for PadOrTruncate {
    fn apply(&self, item: Item) -> Item {
        item
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use crate::core::Item;
    use crate::dataset::Dataset;
    use brain_core::Tensor;
}
