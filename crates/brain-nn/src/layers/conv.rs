//! # 2D Convolution Layer
//!
//! Multi-channel 2D spatial convolution with padding, stride, dilation, and bias parameters.
#![allow(missing_docs)]

use crate::init::kaiming_uniform;
use crate::module::{Module, ModuleError, ModuleResult};
use brain_autograd::Value;
use brain_core::Tensor;

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

/// 2D Convolution Layer.
#[derive(Debug, Clone)]
pub struct Conv2d {
    pub weight: Value,
    pub bias: Option<Value>,
    pub config: ConvConfig,
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        has_bias: bool,
    ) -> Self {
        let weight_t = kaiming_uniform(&[out_channels, in_channels, kernel_size, kernel_size], 0.0);
        let weight = Value::new(weight_t, true);
        let bias = if has_bias {
            Some(Value::new(Tensor::zeros(vec![out_channels]), true))
        } else {
            None
        };
        let config = ConvConfig {
            in_channels,
            out_channels,
            kernel_size: (kernel_size, kernel_size),
            stride: (1, 1),
            padding: (kernel_size / 2, kernel_size / 2),
            dilation: (1, 1),
        };
        Self {
            weight,
            bias,
            config,
        }
    }

    /// Construct with a fully custom config. Returns an error immediately
    /// if `dilation != (1, 1)` -- see the module-level doc comment for why
    /// this is rejected rather than silently mishandled.
    pub fn with_config(config: ConvConfig, has_bias: bool) -> ModuleResult<Self> {
        if config.dilation != (1, 1) {
            return Err(ModuleError::InvalidParameter(format!(
                "Conv2d dilation {:?} is not yet supported: brain_autograd::Value::conv2d \
                 has no dilated-convolution gradient formula implemented yet (Phase 0.1, \
                 tracked and un-done). Use dilation=(1,1) until that lands.",
                config.dilation
            )));
        }
        let weight_tensor = kaiming_uniform(
            &[
                config.out_channels,
                config.in_channels,
                config.kernel_size.0,
                config.kernel_size.1,
            ],
            0.0,
        );
        let weight = Value::new(weight_tensor, true);
        let bias = if has_bias {
            Some(Value::new(Tensor::zeros(vec![config.out_channels]), true))
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            config,
        })
    }

    pub fn forward(&self, input: &Value) -> ModuleResult<Value> {
        let shape = input.shape();
        if shape.len() != 4 || shape[1] != self.config.in_channels {
            return Err(ModuleError::ShapeMismatch {
                expected: vec![shape[0], self.config.in_channels, shape[2], shape[3]],
                got: shape.to_vec(),
            });
        }
        // Guard again here, not just in with_config -- config is a public
        // field and could be mutated after construction.
        if self.config.dilation != (1, 1) {
            return Err(ModuleError::InvalidParameter(
                "Conv2d.config.dilation was changed to a non-(1,1) value after \
                 construction; dilated convolution has no gradient path yet (Phase 0.1)."
                    .to_string(),
            ));
        }

        Ok(input.conv2d(
            &self.weight,
            self.bias.as_ref(),
            self.config.stride,
            self.config.padding,
        ))
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Value) -> ModuleResult<Value> {
        self.forward(input)
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
    fn test_conv2d_correctness() {
        // 1x1 in, 1x1 out, kernel 1, no bias, weight 2.0 => output = 2 * input.
        let mut conv = Conv2d::new(1, 1, 1, false);
        conv.weight = Value::new(Tensor::from_slice(&[2.0], vec![1, 1, 1, 1]), true);
        conv.bias = None;
        let x = Value::new(
            Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]),
            false,
        );
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);
    }

    /// The test that was impossible before Phase 0: a real backward pass
    /// through Conv2d, verified against finite differences on the weight,
    /// via the actual tape -- not a hand-derived formula standing in for a
    /// backward implementation that didn't exist.
    #[test]
    fn test_conv2d_weight_gradient_via_real_tape_matches_finite_diff() {
        let mut conv = Conv2d::new(1, 1, 2, false);
        conv.config.padding = (0, 0);
        let w = vec![1.0, 0.5, -0.5, 2.0];
        conv.weight = Value::new(Tensor::from_slice(&w, vec![1, 1, 2, 2]), true);

        let x_data = vec![1.0, 2.0, 3.0, 4.0];
        let x = Value::new(Tensor::from_slice(&x_data, vec![1, 1, 2, 2]), false);

        let out = conv.forward(&x).unwrap();
        let loss = out.sum();
        loss.backward().unwrap();
        let analytic_grad = conv.weight.grad().unwrap().to_vec();

        let eps = 1e-5;
        for i in 0..4 {
            let mut w_plus = w.clone();
            w_plus[i] += eps;
            let mut w_minus = w.clone();
            w_minus[i] -= eps;

            let mut c_plus = conv.clone();
            c_plus.weight = Value::new(Tensor::from_slice(&w_plus, vec![1, 1, 2, 2]), true);
            let loss_plus = c_plus.forward(&x).unwrap().sum().data().item();

            let mut c_minus = conv.clone();
            c_minus.weight = Value::new(Tensor::from_slice(&w_minus, vec![1, 1, 2, 2]), true);
            let loss_minus = c_minus.forward(&x).unwrap().sum().data().item();

            let numeric = (loss_plus - loss_minus) / (2.0 * eps);
            let analytic = analytic_grad[i];

            assert!(
                (analytic - numeric).abs() < 1e-3,
                "grad mismatch at weight[{i}]: analytic={analytic}, numeric={numeric}"
            );
        }
    }

    #[test]
    fn test_conv2d_dilation_rejected_not_silently_dropped() {
        let mut config = ConvConfig::default();
        config.dilation = (2, 2);
        let result = Conv2d::with_config(config, false);
        assert!(
            result.is_err(),
            "dilation != (1,1) should error, not silently ignore dilation"
        );
    }
}
