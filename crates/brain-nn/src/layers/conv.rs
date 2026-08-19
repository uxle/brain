//! # 2D Convolution Layer
//!
//! Multi-channel 2D spatial convolution with padding, stride, dilation, and bias parameters.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::module::{Module, ModuleResult, ModuleError};
use crate::init::kaiming_uniform;

/// Configuration for 2D convolution operations.
#[derive(Debug, Clone)]
pub struct ConvConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
}

impl Default for ConvConfig {
    fn default() -> Self {
        Self {
            in_channels: 3,
            out_channels: 16,
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
            dilation: (1, 1),
        }
    }
}

/// 2D Convolution layer: [batch, in_channels, height, width] -> [batch, out_channels, out_h, out_w].
#[derive(Debug, Clone)]
pub struct Conv2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub config: ConvConfig,
}

impl Conv2d {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, has_bias: bool) -> Self {
        let weight = kaiming_uniform(&[out_channels, in_channels, kernel_size, kernel_size], 0.0);
        let bias = if has_bias { Some(Tensor::zeros(vec![out_channels])) } else { None };
        let config = ConvConfig {
            in_channels,
            out_channels,
            kernel_size: (kernel_size, kernel_size),
            stride: (1, 1),
            padding: (kernel_size / 2, kernel_size / 2),
            dilation: (1, 1),
        };
        Self { weight, bias, config }
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> ModuleResult<Tensor> {
        let shape = input.shape();
        if shape.len() != 4 || shape[1] != self.config.in_channels {
            return Err(ModuleError::ShapeMismatch {
                expected: vec![shape[0], self.config.in_channels, shape[2], shape[3]],
                got: shape.to_vec(),
            });
        }

        // Dispatch to the brain-core convolution kernel, applying stride, padding, and dilation.
        let bias_ref = self.bias.as_ref();
        let out = brain_core::tensor::conv::conv2d_ext(
            input,
            &self.weight,
            bias_ref,
            self.config.stride,
            self.config.padding,
            self.config.dilation,
        );
        Ok(out)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.weight.clone()];
        if let Some(ref b) = self.bias { p.push(b.clone()); }
        p
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_conv2d_correctness() {
        // 1x1 in, 1x1 out, kernel 1, no bias, weight 2.0 => output = 2 * input.
        let mut conv = Conv2d::new(1, 1, 1, false);
        conv.weight = Tensor::from_slice(&[2.0], vec![1, 1, 1, 1]);
        conv.bias = None;
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);
    }
}
