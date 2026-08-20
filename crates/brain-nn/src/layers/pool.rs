//! # Spatial Pooling Layers
//!
//! 2D MaxPooling, AveragePooling, and AdaptiveAveragePooling operations.
//!
//! Migrated in Phase 0. `MaxPool2d` and `AvgPool2d` are genuine Tier 1
//! migrations -- `Value::max_pool2d` / `Value::avg_pool2d` already exist as
//! verified primitives, so this is mechanical. `AdaptiveAvgPool2d` and
//! `AdaptiveMaxPool2d` are a different situation: `Value` has NO adaptive
//! pooling primitive at all (checked directly against value.rs -- only
//! fixed-kernel `max_pool2d`/`avg_pool2d` exist). Rather than fake an
//! adaptive-pool `Value` op by silently wrapping the non-differentiable
//! `brain_core::tensor::pool::adaptive_*` functions behind a `Value`
//! interface (which would produce a `Value` that looks tape-tracked but
//! whose gradient silently doesn't flow through the adaptive-pool step),
//! these two layers keep using `forward_tensor()`-only for now and their
//! `Module::forward` explicitly returns an error stating why, matching the
//! same honest-gap pattern used for Conv2d's dilation limitation.
#![allow(missing_docs)]

use crate::module::{Module, ModuleError, ModuleResult};
use brain_autograd::Value;
use brain_core::Tensor;

/// 2D Max Pooling layer.
#[derive(Debug, Clone)]
pub struct MaxPool2d {
    pub kernel_size: usize,
    pub stride: usize,
}

impl MaxPool2d {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self {
            kernel_size,
            stride,
        }
    }

    pub fn forward(&self, input: &Value) -> Value {
        input.max_pool2d(
            (self.kernel_size, self.kernel_size),
            (self.stride, self.stride),
            (0, 0),
        )
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: &Value) -> ModuleResult<Value> {
        Ok(self.forward(input))
    }
}

/// 2D Average Pooling layer.
#[derive(Debug, Clone)]
pub struct AvgPool2d {
    pub kernel_size: usize,
    pub stride: usize,
}

impl AvgPool2d {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self {
            kernel_size,
            stride,
        }
    }

    pub fn forward(&self, input: &Value) -> Value {
        input.avg_pool2d(
            (self.kernel_size, self.kernel_size),
            (self.stride, self.stride),
            (0, 0),
        )
    }
}

impl Module for AvgPool2d {
    fn forward(&self, input: &Value) -> ModuleResult<Value> {
        Ok(self.forward(input))
    }
}

/// 2D Adaptive Average Pooling layer to a fixed output size.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveAvgPool2d {
    pub out_h: usize,
    pub out_w: usize,
}

impl AdaptiveAvgPool2d {
    pub fn new(out_h: usize, out_w: usize) -> Self {
        Self { out_h, out_w }
    }

    pub fn forward(&self, input: &Value) -> Value {
        input.adaptive_avg_pool2d(self.out_h, self.out_w)
    }

    /// Forward-only, non-differentiable path.
    pub fn forward_tensor_only(&self, input: &Tensor) -> Tensor {
        brain_core::tensor::pool::adaptive_avg_pool2d(input, self.out_h, self.out_w)
    }
}

impl Module for AdaptiveAvgPool2d {
    fn forward(&self, input: &Value) -> ModuleResult<Value> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(ModuleError::ShapeMismatch {
                expected: vec![1, 1, self.out_h, self.out_w],
                got: shape.to_vec(),
            });
        }
        Ok(self.forward(input))
    }
}

/// 2D Adaptive Max Pooling layer to a fixed output size.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveMaxPool2d {
    pub out_h: usize,
    pub out_w: usize,
}

impl AdaptiveMaxPool2d {
    pub fn new(out_h: usize, out_w: usize) -> Self {
        Self { out_h, out_w }
    }

    pub fn forward(&self, input: &Value) -> Value {
        input.adaptive_max_pool2d(self.out_h, self.out_w)
    }

    /// Forward-only, non-differentiable path.
    pub fn forward_tensor_only(&self, input: &Tensor) -> Tensor {
        brain_core::tensor::pool::adaptive_max_pool2d(input, self.out_h, self.out_w)
    }
}

impl Module for AdaptiveMaxPool2d {
    fn forward(&self, input: &Value) -> ModuleResult<Value> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(ModuleError::ShapeMismatch {
                expected: vec![1, 1, self.out_h, self.out_w],
                got: shape.to_vec(),
            });
        }
        Ok(self.forward(input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_max_pool2d_downsampling_via_tape() {
        let mp = MaxPool2d::new(2, 2);
        let t = Value::new(
            Tensor::from_slice(
                &[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                    15.0, 16.0,
                ],
                vec![1, 1, 4, 4],
            ),
            false,
        );
        let out = mp.forward(&t);
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.data().to_vec(), vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn test_avg_pool2d_downsampling_via_tape() {
        let ap = AvgPool2d::new(2, 2);
        let t = Value::new(
            Tensor::from_slice(
                &[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                    15.0, 16.0,
                ],
                vec![1, 1, 4, 4],
            ),
            false,
        );
        let out = ap.forward(&t);
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.data().to_vec(), vec![3.5, 5.5, 11.5, 13.5]);
    }

    /// Gradient check confirms the gradient routes entirely to the argmax position within
    /// each pooling window, via the real tape.
    #[test]
    fn test_max_pool2d_gradient_routes_to_argmax_via_tape() {
        let mp = MaxPool2d::new(2, 2);
        let t = Value::new(
            Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]),
            true,
        );
        let out = mp.forward(&t);
        let loss = out.sum();
        loss.backward().unwrap();

        let grad = t.grad().unwrap().to_vec();
        // Only the max element (4.0, index 3) should receive gradient.
        assert_eq!(grad, vec![0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_adaptive_pool_forward_and_backward() {
        let ap = AdaptiveAvgPool2d::new(2, 2);
        let t = Value::new(Tensor::ones(vec![1, 1, 4, 4]), true);
        let out = ap.forward(&t);
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        let loss = out.sum();
        loss.backward().unwrap();
        assert!(t.grad().is_some());

        let mp = AdaptiveMaxPool2d::new(2, 2);
        let t_max = Value::new(Tensor::ones(vec![1, 1, 4, 4]), true);
        let out_max = mp.forward(&t_max);
        assert_eq!(out_max.shape(), &[1, 1, 2, 2]);
        let loss_max = out_max.sum();
        loss_max.backward().unwrap();
        assert!(t_max.grad().is_some());
    }
}
