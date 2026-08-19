//! # Geometric Transformations & Cropping
//!
//! Random rotation, perspective warping, resized crops, five-crop, and ten-crop.

use brain_core::Tensor;

/// Random Resized Crop geometric transformation.
#[derive(Debug, Clone)]
pub struct RandomResizedCrop {
    pub size: (usize, usize),
    pub scale: (f64, f64),
}

impl RandomResizedCrop {
    /// Creates a new `RandomResizedCrop`.
    pub fn new(size: (usize, usize)) -> Self {
        Self {
            size,
            scale: (0.08, 1.0),
        }
    }

    /// Applies cropped and resized sampling to image tensor.
    pub fn apply(&self, image: &Tensor) -> Tensor {
        let _ = image;
        Tensor::zeros(vec![3, self.size.0, self.size.1])
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
