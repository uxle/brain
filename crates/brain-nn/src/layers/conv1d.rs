//! # 1D Convolution Layer
//!
//! Multi-channel 1D temporal/sequence convolution with padding, stride, and bias parameters.

use crate::init::kaiming_uniform;
use crate::module::{Module, ModuleError, ModuleResult};
use brain_core::Tensor;

/// Configuration for 1D convolution operations.
#[derive(Debug, Clone)]
pub struct Conv1dConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}

impl Default for Conv1dConfig {
    fn default() -> Self {
        Self {
            in_channels: 1,
            out_channels: 1,
            kernel_size: 3,
            stride: 1,
            padding: 1,
        }
    }
}

/// 1D Convolution layer: [batch, in_channels, length] -> [batch, out_channels, out_length].
#[derive(Debug, Clone)]
pub struct Conv1d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub config: Conv1dConfig,
}

impl Conv1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        has_bias: bool,
    ) -> Self {
        let weight = kaiming_uniform(&[out_channels, in_channels, kernel_size], 0.0);
        let bias = if has_bias {
            Some(Tensor::zeros(vec![out_channels]))
        } else {
            None
        };
        let config = Conv1dConfig {
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            padding: kernel_size / 2,
        };
        Self {
            weight,
            bias,
            config,
        }
    }
}

use brain_autograd::Value;

impl Module for Conv1d {
    fn forward(&self, input: &Value) -> ModuleResult<Value> {
        let shape = input.shape();
        if shape.len() != 3 || shape[1] != self.config.in_channels {
            return Err(ModuleError::ShapeMismatch {
                expected: vec![
                    shape.first().copied().unwrap_or(1),
                    self.config.in_channels,
                    shape.get(2).copied().unwrap_or(1),
                ],
                got: shape.to_vec(),
            });
        }

        let bias_ref = self.bias.as_ref();
        let out = brain_core::tensor::conv::conv1d(
            input.data(),
            &self.weight,
            bias_ref,
            self.config.stride,
            self.config.padding,
        );
        Ok(Value::new(out, input.requires_grad()))
    }

    fn parameters(&self) -> Vec<Value> {
        let mut p = vec![Value::new(self.weight.clone(), true)];
        if let Some(ref b) = self.bias {
            p.push(Value::new(b.clone(), true));
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv1d_forward() {
        let conv = Conv1d::new(1, 1, 3, false);
        let x = Value::new(
            Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 1, 5]),
            false,
        );
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape().len(), 3);
        assert_eq!(out.shape()[0], 1);
        assert_eq!(out.shape()[1], 1);
    }
}
