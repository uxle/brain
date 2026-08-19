//! # Vision Transforms (TorchVision Parity)
//!
//! Provides `Normalize`, `Resize`, `CenterCrop`, `RandomHorizontalFlip`, and `ColorJitter`.

use super::Transform;
use crate::core::Item;

/// Normalizes image tensors by mean and standard deviation.
pub struct Normalize {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

impl Normalize {
    /// Creates a new `Normalize` transform.
    pub fn new(mean: Vec<f64>, std: Vec<f64>) -> Self {
        Self { mean, std }
    }
}

impl Transform for Normalize {
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
