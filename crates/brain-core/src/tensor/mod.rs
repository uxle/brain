//! Tensor data structures, multidimensional array operations, and mathematical submodules.
//!
//! This module serves as the primary gateway for tensor computation in the Brain DL framework,
//! re-exporting [`Tensor`], statistical summaries ([`TensorStats`]), BLAS routines, autograd functions,
//! and specialized neural network operators.

pub mod arithmetic;
pub mod blas;
pub mod broadcast;
pub mod compare;
pub mod conv;
pub mod factory;
pub mod fft;
pub mod fold;
pub mod function;
pub mod hist;
pub mod indexing;
pub mod linalg;
pub mod math;
pub mod neural;
pub mod ops_fused;
pub mod ops_nd;
pub mod pad;
pub mod pool;
pub mod quant;
pub mod random_ops;
pub mod reduction;
pub mod simd;
pub mod sparse;
pub mod special;
#[path = "impl.rs"]
pub mod tensor_impl;
pub mod view;

pub use tensor_impl::Tensor;

// =============================================================================
// Tensor Statistics Summary
// =============================================================================

/// Comprehensive statistical diagnostics of a tensor buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorStats {
    /// Number of elements.
    pub numel: usize,
    /// Minimum finite value.
    pub min: f64,
    /// Maximum finite value.
    pub max: f64,
    /// Arithmetic mean.
    pub mean: f64,
    /// Standard deviation.
    pub std: f64,
    /// Sparsity fraction (ratio of zeros to total elements).
    pub sparsity: f64,
    /// Whether all elements are finite numbers.
    pub is_finite: bool,
    /// Whether the tensor contains NaN.
    pub has_nan: bool,
    /// Whether the tensor contains +/- Infinity.
    pub has_inf: bool,
}

impl TensorStats {
    /// Computes summary statistics for a tensor.
    pub fn compute(tensor: &Tensor) -> Self {
        let numel = tensor.numel();
        if numel == 0 {
            return TensorStats {
                numel: 0,
                min: f64::NAN,
                max: f64::NAN,
                mean: f64::NAN,
                std: f64::NAN,
                sparsity: 0.0,
                is_finite: true,
                has_nan: false,
                has_inf: false,
            };
        }

        let mut sum = 0.0;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut zeros = 0;
        let mut has_nan = false;
        let mut has_inf = false;

        for &x in tensor.data() {
            if x.is_nan() {
                has_nan = true;
            } else if x.is_infinite() {
                has_inf = true;
            } else {
                if x < min {
                    min = x;
                }
                if x > max {
                    max = x;
                }
                sum += x;
            }
            if x == 0.0 {
                zeros += 1;
            }
        }

        let mean = sum / (numel as f64);
        let mut sum_sq_diff = 0.0;
        for &x in tensor.data() {
            if !x.is_nan() && !x.is_infinite() {
                sum_sq_diff += (x - mean).powi(2);
            }
        }
        let std = (sum_sq_diff / (numel as f64)).sqrt();

        TensorStats {
            numel,
            min,
            max,
            mean,
            std,
            sparsity: (zeros as f64) / (numel as f64),
            is_finite: !has_nan && !has_inf,
            has_nan,
            has_inf,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_stats() {
        let t = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0], vec![4]);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.min, 0.0);
        assert_eq!(s.max, 3.0);
        assert_eq!(s.mean, 1.5);
        assert_eq!(s.sparsity, 0.25);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tensor_module_exports() {
        let t = Tensor::ones(vec![2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
    }
}
