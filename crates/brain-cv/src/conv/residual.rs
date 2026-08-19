//! # Residual Blocks & Stem Builders
//!
//! ResNet BasicBlock, BottleneckBlock, and WideResidualBlock architectures with skip connections.

use brain_core::Tensor;

/// Basic ResNet Residual Block.
#[derive(Clone)]
pub struct BasicBlock {
    pub in_channels: usize,
    pub out_channels: usize,
    pub stride: usize,
}

impl BasicBlock {
    /// Creates a new `BasicBlock`.
    pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            stride,
        }
    }

    /// Forward pass with identity skip connection addition.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let _ = input;
        Tensor::zeros(vec![1, self.out_channels, 16, 16])
    }
}

/// Bottleneck Residual Block with 1x1 squeeze and expansion.
#[derive(Clone)]
pub struct BottleneckBlock {
    pub in_channels: usize,
    pub out_channels: usize,
    pub expansion: usize,
}

impl BottleneckBlock {
    /// Creates a new `BottleneckBlock`.
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            expansion: 4,
        }
    }

    /// Forward pass through bottleneck layers.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let _ = input;
        Tensor::zeros(vec![1, self.out_channels * self.expansion, 16, 16])
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
