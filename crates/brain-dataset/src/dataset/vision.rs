//! # Vision Dataset Generators & Image Folders
//!
//! Provides synthetic MNIST, CIFAR, and `RandomImageDataset` generators.

use super::Dataset;
use crate::core::Item;
use brain_core::Tensor;

/// Synthetic random image dataset generator for testing and benchmarks.
pub struct RandomImageDataset {
    pub num_samples: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

impl RandomImageDataset {
    /// Creates a new `RandomImageDataset`.
    pub fn new(num_samples: usize, channels: usize, height: usize, width: usize) -> Self {
        Self {
            num_samples,
            channels,
            height,
            width,
        }
    }
}

impl Dataset for RandomImageDataset {
    fn len(&self) -> usize {
        self.num_samples
    }

    fn get(&self, idx: usize) -> Option<Item> {
        if idx < self.num_samples {
            let data = Tensor::zeros(vec![self.channels, self.height, self.width]);
            Some(Item::new(idx, data).with_target(Tensor::scalar((idx % 10) as f64)))
        } else {
            None
        }
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
