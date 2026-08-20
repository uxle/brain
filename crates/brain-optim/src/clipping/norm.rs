//! # Global Norm and Value Gradient Clipping
//!
//! Standard L1, L2, and L-infinity norm clipping for gradient tensors.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Norm type calculation enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NormType {
    L1,
    #[default]
    L2,
    LInf,
}

/// Configuration container for gradient clipping.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipConfig {
    pub max_norm: f64,
    pub norm_type: NormType,
    pub error_if_nonfinite: bool,
}

impl Default for ClipConfig {
    fn default() -> Self {
        Self {
            max_norm: 1.0,
            norm_type: NormType::L2,
            error_if_nonfinite: false,
        }
    }
}

/// Clips gradient norm of an iterable of tensors in-place.
///
/// Returns the total norm of the gradients (viewed as a single vector).
pub fn clip_grad_norm_(grads: &mut [Tensor], max_norm: f64, norm_type: NormType) -> f64 {
    if grads.is_empty() || max_norm <= 0.0 {
        return 0.0;
    }

    let total_norm = match norm_type {
        NormType::L2 => {
            let mut sum_sq = 0.0;
            for g in grads.iter() {
                for &val in g.data() {
                    if !val.is_nan() && !val.is_infinite() {
                        sum_sq += val * val;
                    }
                }
            }
            sum_sq.sqrt()
        }
        NormType::L1 => {
            let mut sum_abs = 0.0;
            for g in grads.iter() {
                for &val in g.data() {
                    if !val.is_nan() && !val.is_infinite() {
                        sum_abs += val.abs();
                    }
                }
            }
            sum_abs
        }
        NormType::LInf => {
            let mut max_abs = 0.0f64;
            for g in grads.iter() {
                for &val in g.data() {
                    if !val.is_nan() && !val.is_infinite() {
                        max_abs = max_abs.max(val.abs());
                    }
                }
            }
            max_abs
        }
    };

    let clip_coef = max_norm / (total_norm + 1e-6);
    if clip_coef < 1.0 {
        for g in grads.iter_mut() {
            for val in g.data_mut() {
                *val *= clip_coef;
            }
        }
    }

    total_norm
}

/// Clips gradient values of an iterable of tensors in-place at specified maximum absolute value.
pub fn clip_grad_value_(grads: &mut [Tensor], clip_value: f64) {
    if clip_value <= 0.0 {
        return;
    }
    for g in grads.iter_mut() {
        for val in g.data_mut() {
            *val = val.clamp(-clip_value, clip_value);
        }
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
}
