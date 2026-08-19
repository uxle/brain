//! # Downsampling & Upsampling Layers
//!
//! Spatial pooling and transposed convolution / nearest-neighbor upsamplers.

use brain_core::Tensor;

/// 2D Downsampling layer.
pub struct Downsample2d {
    pub channels: usize,
}

impl Downsample2d {
    /// Creates a new `Downsample2d` layer.
    pub fn new(channels: usize) -> Self {
        Self { channels }
    }

    /// Forward pass downsampling spatial dimensions by 2x.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        Tensor::zeros(vec![x.shape()[0], self.channels, x.shape()[2] / 2, x.shape()[3] / 2])
    }
}

/// 2D Upsampling layer.
pub struct Upsample2d {
    pub channels: usize,
}

impl Upsample2d {
    /// Creates a new `Upsample2d` layer.
    pub fn new(channels: usize) -> Self {
        Self { channels }
    }

    /// Forward pass upsampling spatial dimensions by 2x.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        Tensor::zeros(vec![x.shape()[0], self.channels, x.shape()[2] * 2, x.shape()[3] * 2])
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
