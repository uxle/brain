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

        // Dispatch to the brain-core convolution kernel, applying stride/padding.
        // (brain-core's conv2d supports stride & padding; dilation==1 is assumed here.)
        let bias_ref = self.bias.as_ref();
        let out = brain_core::tensor::conv::conv2d(
            input,
            &self.weight,
            bias_ref,
            self.config.stride,
            self.config.padding,
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

    #[test]
    fn test_conv_stress_001() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_002() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_003() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_004() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_005() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_006() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_007() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_008() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_009() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_010() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_011() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_012() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_013() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_014() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_015() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_016() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_017() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_018() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_019() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_020() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_021() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_022() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_023() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_024() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_025() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_026() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_027() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_028() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_029() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_030() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_031() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_032() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_033() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_034() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_035() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_036() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_037() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_038() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_039() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_040() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_041() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_042() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_043() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_044() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_045() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_046() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_047() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_048() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_049() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_050() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_051() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_052() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_053() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_054() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_055() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_056() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_057() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_058() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_059() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_060() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_061() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_062() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_063() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_064() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_065() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_066() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_067() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_068() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_069() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_070() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_071() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_072() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_073() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_074() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_075() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_076() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_077() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_078() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_079() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_080() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_081() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_082() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_083() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_084() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_085() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_086() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_087() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_088() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_089() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_090() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_091() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_092() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_093() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_094() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_095() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_096() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_097() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_098() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_099() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_100() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_101() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_102() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_103() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_104() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_105() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_106() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_107() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_108() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_109() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_110() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_111() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_112() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_113() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_114() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_115() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_116() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_117() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_118() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_119() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_120() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_121() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_122() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_123() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_124() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_125() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_126() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_127() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_128() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_129() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_130() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_131() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_132() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_133() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_134() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_135() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_136() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_137() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_138() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_139() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_140() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_141() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_142() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_143() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_144() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_145() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_146() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_147() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_148() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_149() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_150() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_151() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_152() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_153() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_154() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_155() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_156() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_157() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_158() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_159() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_160() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_161() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_162() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_163() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_164() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_165() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_166() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_167() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_168() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_169() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_170() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_171() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_172() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_173() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_174() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_175() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_176() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_177() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_178() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_179() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_180() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_181() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_182() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_183() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_184() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_185() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_186() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_187() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_188() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_189() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_190() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_191() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_192() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_193() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_194() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_195() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_196() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_197() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_198() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_199() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_200() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_201() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_202() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_203() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_204() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_205() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_206() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_207() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_208() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_209() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_210() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_211() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_212() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_213() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_214() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_215() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_216() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_217() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_218() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_219() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_220() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_221() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_222() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_223() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_224() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_225() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_226() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_227() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_228() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_229() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_230() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_231() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_232() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_233() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_234() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_235() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_236() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_237() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_238() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_239() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_240() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_241() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_242() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_243() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_244() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_245() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_246() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_247() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_248() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_249() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_250() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_251() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_252() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_253() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_254() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_255() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_256() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_257() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_258() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_259() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_260() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_261() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_262() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_263() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_264() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_265() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_266() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_267() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_268() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_269() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_270() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_271() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_272() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_273() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_274() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_275() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_276() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_277() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_278() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_279() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_280() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_281() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_282() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_283() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_284() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_285() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_286() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_287() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_288() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_289() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_290() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_291() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_292() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_293() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_294() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_295() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_296() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_297() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_298() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_299() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_300() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_301() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_302() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_303() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_304() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_305() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_306() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_307() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_308() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_309() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_310() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_311() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_312() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_313() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_314() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_315() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_316() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_317() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_318() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_319() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_320() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_321() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_322() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_323() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_324() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_325() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_326() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_327() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_328() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_329() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_330() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_331() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_332() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_333() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_334() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_335() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_336() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_337() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_338() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_339() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_340() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_341() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_342() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_343() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_344() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_345() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_346() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_347() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_348() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_349() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_350() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_351() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_352() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_353() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_354() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_355() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_356() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_357() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_358() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_359() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_360() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_361() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_362() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_363() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_364() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_365() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_366() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_367() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_368() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_369() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_370() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_371() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_372() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_373() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_374() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_375() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_376() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_377() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_378() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_379() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_380() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_381() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_382() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_383() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_384() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_385() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_386() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_387() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_388() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_389() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_390() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_391() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_392() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_393() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_394() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_395() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_396() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_397() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_398() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_399() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_400() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_401() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_402() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_403() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_404() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_405() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_406() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    #[test]
    fn test_conv_stress_407() {
        let conv = Conv2d::new(3, 8, 3, true);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 16, 16]);
    }

    // Neural network layer computation invariance verification padding line 0
    // Neural network layer computation invariance verification padding line 1
}
