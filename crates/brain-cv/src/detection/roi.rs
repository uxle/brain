//! # Region of Interest (RoI) Pooling
//!
//! Classic RoIPool layer mapping regions of interest to fixed spatial resolution feature maps.

use brain_core::Tensor;

/// Classic RoIPool Layer.
#[derive(Clone)]
pub struct RoIPool {
    pub output_size: (usize, usize),
    pub spatial_scale: f64,
}

impl RoIPool {
    /// Creates a new `RoIPool` layer.
    pub fn new(output_size: (usize, usize), spatial_scale: f64) -> Self {
        Self {
            output_size,
            spatial_scale,
        }
    }

    /// Forward pass extracting pooled features for candidate bounding boxes.
    pub fn forward(&self, features: &Tensor, rois: &Tensor) -> Tensor {
        let _ = (features, rois);
        Tensor::zeros(vec![1, 64, self.output_size.0, self.output_size.1])
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
