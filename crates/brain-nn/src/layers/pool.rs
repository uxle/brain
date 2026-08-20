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
///
/// NOT yet migrated onto `Value` -- see module-level doc comment. Calling
/// `Module::forward` returns an error. Use `forward_tensor_only` if you
/// need the forward-pass-only behavior (e.g. inference with no gradient
/// requirement) until Phase 0.2 (tracked follow-up: add
/// `Value::adaptive_avg_pool2d` with a real VJP) lands.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveAvgPool2d {
    pub out_h: usize,
    pub out_w: usize,
}

impl AdaptiveAvgPool2d {
    pub fn new(out_h: usize, out_w: usize) -> Self {
        Self { out_h, out_w }
    }

    /// Forward-only, non-differentiable path. Does NOT build a tape entry --
    /// do not use this inside a graph you intend to call `.backward()` on;
    /// gradients will simply stop here.
    pub fn forward_tensor_only(&self, input: &Tensor) -> Tensor {
        brain_core::tensor::pool::adaptive_avg_pool2d(input, self.out_h, self.out_w)
    }
}

impl Module for AdaptiveAvgPool2d {
    fn forward(&self, _input: &Value) -> ModuleResult<Value> {
        Err(ModuleError::InvalidParameter(
            "AdaptiveAvgPool2d has no Value-based (differentiable) forward path yet -- \
             brain_autograd::Value has no adaptive-pooling primitive (Phase 0.2, tracked \
             and un-done). Use `forward_tensor_only()` for inference-only use, or wait for \
             Phase 0.2."
                .to_string(),
        ))
    }
}

/// 2D Adaptive Max Pooling layer to a fixed output size. Same status as
/// `AdaptiveAvgPool2d` above -- see its doc comment.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveMaxPool2d {
    pub out_h: usize,
    pub out_w: usize,
}

impl AdaptiveMaxPool2d {
    pub fn new(out_h: usize, out_w: usize) -> Self {
        Self { out_h, out_w }
    }

    /// Forward-only, non-differentiable path -- see `AdaptiveAvgPool2d::forward_tensor_only`.
    pub fn forward_tensor_only(&self, input: &Tensor) -> Tensor {
        brain_core::tensor::pool::adaptive_max_pool2d(input, self.out_h, self.out_w)
    }
}

impl Module for AdaptiveMaxPool2d {
    fn forward(&self, _input: &Value) -> ModuleResult<Value> {
        Err(ModuleError::InvalidParameter(
            "AdaptiveMaxPool2d has no Value-based (differentiable) forward path yet -- \
             brain_autograd::Value has no adaptive-pooling primitive (Phase 0.2, tracked \
             and un-done). Use `forward_tensor_only()` for inference-only use, or wait for \
             Phase 0.2."
                .to_string(),
        ))
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

    /// Gradient check that was impossible before Phase 0 for MaxPool2d --
    /// confirms the gradient routes entirely to the argmax position within
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
    fn test_adaptive_pool_forward_errors_until_phase_0_2() {
        let ap = AdaptiveAvgPool2d::new(2, 2);
        let t = Value::new(Tensor::zeros(vec![1, 1, 4, 4]), false);
        let result = ap.forward(&t);
        assert!(result.is_err(), "adaptive pooling should error via Module::forward, not silently skip gradient tracking");
    }
}
