//! # Transposed 2D Convolution (Deconvolution)
//!
//! Fractionally-strided spatial upsampling layer with output padding.
#![allow(missing_docs)]

use crate::module::{Module, ModuleError, ModuleResult};
use brain_autograd::Value;
use brain_core::Tensor;

/// Configuration for transposed 2D convolutions.
#[derive(Debug, Clone, Default)]
pub struct ConvTransposeConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

/// Transposed 2D Convolution layer.
#[derive(Debug, Clone)]
pub struct ConvTranspose2d {
    pub weight: Value,
    pub bias: Option<Value>,
    pub config: ConvTransposeConfig,
}

impl ConvTranspose2d {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        let weight = Value::new(
            Tensor::zeros(vec![in_channels, out_channels, kernel_size, kernel_size]),
            true,
        );
        let config = ConvTransposeConfig {
            in_channels,
            out_channels,
            kernel_size: (kernel_size, kernel_size),
            stride: (1, 1),
            padding: (0, 0),
        };
        Self {
            weight,
            bias: None,
            config,
        }
    }
}

impl Module for ConvTranspose2d {
    fn forward(&self, input: &Value) -> ModuleResult<Value> {
        let shape = input.shape();
        if shape.len() != 4 || shape[1] != self.config.in_channels {
            return Err(ModuleError::ShapeMismatch {
                expected: vec![shape[0], self.config.in_channels, shape[2], shape[3]],
                got: shape.to_vec(),
            });
        }
        Ok(input.conv_transpose2d(
            &self.weight,
            self.bias.as_ref(),
            self.config.stride,
            self.config.padding,
        ))
    }

    fn parameters(&self) -> Vec<Value> {
        let mut p = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            p.push(b.clone());
        }
        p
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_conv_transpose_correctness() {
        // 1x1 -> 1x1 in=1 out=1, kernel 2, stride 1, pad 0, no bias, weight all ones.
        // input [[1,2],[3,4]]: output is the full overlap sum (3x3):
        // [[1,3,2],[4,10,6],[3,7,4]]
        let mut ct = ConvTranspose2d::new(1, 1, 2);
        ct.weight = Value::new(Tensor::ones(vec![1, 1, 2, 2]), true);
        ct.bias = None;
        let x = Value::new(
            Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]),
            false,
        );
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 1, 3, 3]);
        let expected = &[1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0];
        for (i, &e) in expected.iter().enumerate() {
            assert!(
                (out.get(i) - e).abs() < 1e-9,
                "conv_transpose out[{}] = {} expected {}",
                i,
                out.get(i),
                e
            );
        }
    }
}
