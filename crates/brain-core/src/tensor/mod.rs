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
#[path = "impl.rs"]
pub mod tensor_impl;
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
                if x < min { min = x; }
                if x > max { max = x; }
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
    fn test_tmod_stress_case_001() {
        let t = Tensor::full(vec![4], 1.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 1.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_002() {
        let t = Tensor::full(vec![4], 2.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 2.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_003() {
        let t = Tensor::full(vec![4], 3.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 3.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_004() {
        let t = Tensor::full(vec![4], 4.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 4.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_005() {
        let t = Tensor::full(vec![4], 5.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 5.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_006() {
        let t = Tensor::full(vec![4], 6.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 6.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_007() {
        let t = Tensor::full(vec![4], 7.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 7.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_008() {
        let t = Tensor::full(vec![4], 8.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 8.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_009() {
        let t = Tensor::full(vec![4], 9.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 9.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_010() {
        let t = Tensor::full(vec![4], 10.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 10.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_011() {
        let t = Tensor::full(vec![4], 11.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 11.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_012() {
        let t = Tensor::full(vec![4], 12.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 12.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_013() {
        let t = Tensor::full(vec![4], 13.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 13.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_014() {
        let t = Tensor::full(vec![4], 14.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 14.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_015() {
        let t = Tensor::full(vec![4], 15.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 15.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_016() {
        let t = Tensor::full(vec![4], 16.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 16.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_017() {
        let t = Tensor::full(vec![4], 17.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 17.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_018() {
        let t = Tensor::full(vec![4], 18.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 18.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_019() {
        let t = Tensor::full(vec![4], 19.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 19.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_020() {
        let t = Tensor::full(vec![4], 20.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 20.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_021() {
        let t = Tensor::full(vec![4], 21.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 21.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_022() {
        let t = Tensor::full(vec![4], 22.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 22.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_023() {
        let t = Tensor::full(vec![4], 23.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 23.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_024() {
        let t = Tensor::full(vec![4], 24.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 24.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_025() {
        let t = Tensor::full(vec![4], 25.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 25.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_026() {
        let t = Tensor::full(vec![4], 26.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 26.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_027() {
        let t = Tensor::full(vec![4], 27.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 27.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_028() {
        let t = Tensor::full(vec![4], 28.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 28.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_029() {
        let t = Tensor::full(vec![4], 29.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 29.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_030() {
        let t = Tensor::full(vec![4], 30.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 30.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_031() {
        let t = Tensor::full(vec![4], 31.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 31.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_032() {
        let t = Tensor::full(vec![4], 32.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 32.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_033() {
        let t = Tensor::full(vec![4], 33.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 33.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_034() {
        let t = Tensor::full(vec![4], 34.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 34.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_035() {
        let t = Tensor::full(vec![4], 35.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 35.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_036() {
        let t = Tensor::full(vec![4], 36.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 36.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_037() {
        let t = Tensor::full(vec![4], 37.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 37.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_038() {
        let t = Tensor::full(vec![4], 38.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 38.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_039() {
        let t = Tensor::full(vec![4], 39.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 39.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_040() {
        let t = Tensor::full(vec![4], 40.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 40.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_041() {
        let t = Tensor::full(vec![4], 41.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 41.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_042() {
        let t = Tensor::full(vec![4], 42.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 42.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_043() {
        let t = Tensor::full(vec![4], 43.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 43.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_044() {
        let t = Tensor::full(vec![4], 44.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 44.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_045() {
        let t = Tensor::full(vec![4], 45.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 45.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_046() {
        let t = Tensor::full(vec![4], 46.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 46.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_047() {
        let t = Tensor::full(vec![4], 47.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 47.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_048() {
        let t = Tensor::full(vec![4], 48.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 48.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_049() {
        let t = Tensor::full(vec![4], 49.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 49.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_050() {
        let t = Tensor::full(vec![4], 50.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 50.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_051() {
        let t = Tensor::full(vec![4], 51.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 51.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_052() {
        let t = Tensor::full(vec![4], 52.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 52.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_053() {
        let t = Tensor::full(vec![4], 53.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 53.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_054() {
        let t = Tensor::full(vec![4], 54.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 54.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_055() {
        let t = Tensor::full(vec![4], 55.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 55.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_056() {
        let t = Tensor::full(vec![4], 56.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 56.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_057() {
        let t = Tensor::full(vec![4], 57.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 57.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_058() {
        let t = Tensor::full(vec![4], 58.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 58.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_059() {
        let t = Tensor::full(vec![4], 59.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 59.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_060() {
        let t = Tensor::full(vec![4], 60.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 60.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_061() {
        let t = Tensor::full(vec![4], 61.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 61.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_062() {
        let t = Tensor::full(vec![4], 62.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 62.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_063() {
        let t = Tensor::full(vec![4], 63.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 63.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_064() {
        let t = Tensor::full(vec![4], 64.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 64.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_065() {
        let t = Tensor::full(vec![4], 65.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 65.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_066() {
        let t = Tensor::full(vec![4], 66.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 66.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_067() {
        let t = Tensor::full(vec![4], 67.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 67.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_068() {
        let t = Tensor::full(vec![4], 68.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 68.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_069() {
        let t = Tensor::full(vec![4], 69.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 69.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_070() {
        let t = Tensor::full(vec![4], 70.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 70.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_071() {
        let t = Tensor::full(vec![4], 71.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 71.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_072() {
        let t = Tensor::full(vec![4], 72.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 72.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_073() {
        let t = Tensor::full(vec![4], 73.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 73.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_074() {
        let t = Tensor::full(vec![4], 74.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 74.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_075() {
        let t = Tensor::full(vec![4], 75.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 75.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_076() {
        let t = Tensor::full(vec![4], 76.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 76.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_077() {
        let t = Tensor::full(vec![4], 77.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 77.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_078() {
        let t = Tensor::full(vec![4], 78.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 78.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_079() {
        let t = Tensor::full(vec![4], 79.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 79.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_080() {
        let t = Tensor::full(vec![4], 80.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 80.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_081() {
        let t = Tensor::full(vec![4], 81.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 81.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_082() {
        let t = Tensor::full(vec![4], 82.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 82.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_083() {
        let t = Tensor::full(vec![4], 83.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 83.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_084() {
        let t = Tensor::full(vec![4], 84.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 84.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_085() {
        let t = Tensor::full(vec![4], 85.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 85.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_086() {
        let t = Tensor::full(vec![4], 86.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 86.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_087() {
        let t = Tensor::full(vec![4], 87.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 87.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_088() {
        let t = Tensor::full(vec![4], 88.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 88.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_089() {
        let t = Tensor::full(vec![4], 89.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 89.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_090() {
        let t = Tensor::full(vec![4], 90.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 90.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_091() {
        let t = Tensor::full(vec![4], 91.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 91.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_092() {
        let t = Tensor::full(vec![4], 92.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 92.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_093() {
        let t = Tensor::full(vec![4], 93.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 93.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_094() {
        let t = Tensor::full(vec![4], 94.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 94.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_095() {
        let t = Tensor::full(vec![4], 95.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 95.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_096() {
        let t = Tensor::full(vec![4], 96.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 96.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_097() {
        let t = Tensor::full(vec![4], 97.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 97.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_098() {
        let t = Tensor::full(vec![4], 98.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 98.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_099() {
        let t = Tensor::full(vec![4], 99.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 99.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_100() {
        let t = Tensor::full(vec![4], 100.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 100.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_101() {
        let t = Tensor::full(vec![4], 101.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 101.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_102() {
        let t = Tensor::full(vec![4], 102.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 102.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_103() {
        let t = Tensor::full(vec![4], 103.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 103.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_104() {
        let t = Tensor::full(vec![4], 104.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 104.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_105() {
        let t = Tensor::full(vec![4], 105.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 105.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_106() {
        let t = Tensor::full(vec![4], 106.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 106.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_107() {
        let t = Tensor::full(vec![4], 107.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 107.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_108() {
        let t = Tensor::full(vec![4], 108.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 108.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_109() {
        let t = Tensor::full(vec![4], 109.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 109.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_110() {
        let t = Tensor::full(vec![4], 110.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 110.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_111() {
        let t = Tensor::full(vec![4], 111.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 111.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_112() {
        let t = Tensor::full(vec![4], 112.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 112.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_113() {
        let t = Tensor::full(vec![4], 113.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 113.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_114() {
        let t = Tensor::full(vec![4], 114.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 114.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_115() {
        let t = Tensor::full(vec![4], 115.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 115.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_116() {
        let t = Tensor::full(vec![4], 116.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 116.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_117() {
        let t = Tensor::full(vec![4], 117.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 117.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_118() {
        let t = Tensor::full(vec![4], 118.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 118.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_119() {
        let t = Tensor::full(vec![4], 119.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 119.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_120() {
        let t = Tensor::full(vec![4], 120.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 120.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_121() {
        let t = Tensor::full(vec![4], 121.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 121.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_122() {
        let t = Tensor::full(vec![4], 122.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 122.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_123() {
        let t = Tensor::full(vec![4], 123.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 123.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_124() {
        let t = Tensor::full(vec![4], 124.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 124.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_125() {
        let t = Tensor::full(vec![4], 125.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 125.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_126() {
        let t = Tensor::full(vec![4], 126.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 126.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_127() {
        let t = Tensor::full(vec![4], 127.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 127.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_128() {
        let t = Tensor::full(vec![4], 128.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 128.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_129() {
        let t = Tensor::full(vec![4], 129.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 129.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_130() {
        let t = Tensor::full(vec![4], 130.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 130.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_131() {
        let t = Tensor::full(vec![4], 131.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 131.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_132() {
        let t = Tensor::full(vec![4], 132.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 132.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_133() {
        let t = Tensor::full(vec![4], 133.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 133.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_134() {
        let t = Tensor::full(vec![4], 134.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 134.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_135() {
        let t = Tensor::full(vec![4], 135.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 135.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_136() {
        let t = Tensor::full(vec![4], 136.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 136.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_137() {
        let t = Tensor::full(vec![4], 137.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 137.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_138() {
        let t = Tensor::full(vec![4], 138.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 138.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_139() {
        let t = Tensor::full(vec![4], 139.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 139.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_140() {
        let t = Tensor::full(vec![4], 140.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 140.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_141() {
        let t = Tensor::full(vec![4], 141.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 141.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_142() {
        let t = Tensor::full(vec![4], 142.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 142.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_143() {
        let t = Tensor::full(vec![4], 143.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 143.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_144() {
        let t = Tensor::full(vec![4], 144.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 144.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_145() {
        let t = Tensor::full(vec![4], 145.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 145.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_146() {
        let t = Tensor::full(vec![4], 146.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 146.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_147() {
        let t = Tensor::full(vec![4], 147.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 147.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_148() {
        let t = Tensor::full(vec![4], 148.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 148.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_149() {
        let t = Tensor::full(vec![4], 149.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 149.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_150() {
        let t = Tensor::full(vec![4], 150.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 150.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_151() {
        let t = Tensor::full(vec![4], 151.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 151.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_152() {
        let t = Tensor::full(vec![4], 152.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 152.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_153() {
        let t = Tensor::full(vec![4], 153.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 153.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_154() {
        let t = Tensor::full(vec![4], 154.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 154.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_155() {
        let t = Tensor::full(vec![4], 155.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 155.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_156() {
        let t = Tensor::full(vec![4], 156.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 156.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_157() {
        let t = Tensor::full(vec![4], 157.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 157.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_158() {
        let t = Tensor::full(vec![4], 158.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 158.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_159() {
        let t = Tensor::full(vec![4], 159.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 159.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_160() {
        let t = Tensor::full(vec![4], 160.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 160.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_161() {
        let t = Tensor::full(vec![4], 161.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 161.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_162() {
        let t = Tensor::full(vec![4], 162.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 162.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_163() {
        let t = Tensor::full(vec![4], 163.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 163.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_164() {
        let t = Tensor::full(vec![4], 164.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 164.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_165() {
        let t = Tensor::full(vec![4], 165.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 165.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_166() {
        let t = Tensor::full(vec![4], 166.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 166.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_167() {
        let t = Tensor::full(vec![4], 167.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 167.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_168() {
        let t = Tensor::full(vec![4], 168.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 168.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_169() {
        let t = Tensor::full(vec![4], 169.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 169.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_170() {
        let t = Tensor::full(vec![4], 170.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 170.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_171() {
        let t = Tensor::full(vec![4], 171.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 171.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_172() {
        let t = Tensor::full(vec![4], 172.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 172.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_173() {
        let t = Tensor::full(vec![4], 173.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 173.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_174() {
        let t = Tensor::full(vec![4], 174.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 174.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_175() {
        let t = Tensor::full(vec![4], 175.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 175.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_176() {
        let t = Tensor::full(vec![4], 176.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 176.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_177() {
        let t = Tensor::full(vec![4], 177.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 177.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_178() {
        let t = Tensor::full(vec![4], 178.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 178.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_179() {
        let t = Tensor::full(vec![4], 179.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 179.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_180() {
        let t = Tensor::full(vec![4], 180.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 180.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_181() {
        let t = Tensor::full(vec![4], 181.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 181.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_182() {
        let t = Tensor::full(vec![4], 182.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 182.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_183() {
        let t = Tensor::full(vec![4], 183.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 183.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_184() {
        let t = Tensor::full(vec![4], 184.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 184.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_185() {
        let t = Tensor::full(vec![4], 185.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 185.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_186() {
        let t = Tensor::full(vec![4], 186.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 186.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_187() {
        let t = Tensor::full(vec![4], 187.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 187.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_188() {
        let t = Tensor::full(vec![4], 188.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 188.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_189() {
        let t = Tensor::full(vec![4], 189.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 189.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_190() {
        let t = Tensor::full(vec![4], 190.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 190.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_191() {
        let t = Tensor::full(vec![4], 191.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 191.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_192() {
        let t = Tensor::full(vec![4], 192.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 192.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_193() {
        let t = Tensor::full(vec![4], 193.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 193.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_194() {
        let t = Tensor::full(vec![4], 194.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 194.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_195() {
        let t = Tensor::full(vec![4], 195.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 195.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_196() {
        let t = Tensor::full(vec![4], 196.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 196.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_197() {
        let t = Tensor::full(vec![4], 197.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 197.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_198() {
        let t = Tensor::full(vec![4], 198.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 198.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_199() {
        let t = Tensor::full(vec![4], 199.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 199.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_200() {
        let t = Tensor::full(vec![4], 200.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 200.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_201() {
        let t = Tensor::full(vec![4], 201.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 201.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_202() {
        let t = Tensor::full(vec![4], 202.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 202.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_203() {
        let t = Tensor::full(vec![4], 203.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 203.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_204() {
        let t = Tensor::full(vec![4], 204.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 204.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_205() {
        let t = Tensor::full(vec![4], 205.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 205.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_206() {
        let t = Tensor::full(vec![4], 206.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 206.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_207() {
        let t = Tensor::full(vec![4], 207.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 207.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_208() {
        let t = Tensor::full(vec![4], 208.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 208.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_209() {
        let t = Tensor::full(vec![4], 209.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 209.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_210() {
        let t = Tensor::full(vec![4], 210.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 210.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_211() {
        let t = Tensor::full(vec![4], 211.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 211.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_212() {
        let t = Tensor::full(vec![4], 212.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 212.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_213() {
        let t = Tensor::full(vec![4], 213.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 213.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_214() {
        let t = Tensor::full(vec![4], 214.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 214.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_215() {
        let t = Tensor::full(vec![4], 215.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 215.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_216() {
        let t = Tensor::full(vec![4], 216.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 216.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_217() {
        let t = Tensor::full(vec![4], 217.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 217.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_218() {
        let t = Tensor::full(vec![4], 218.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 218.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_219() {
        let t = Tensor::full(vec![4], 219.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 219.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_220() {
        let t = Tensor::full(vec![4], 220.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 220.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_221() {
        let t = Tensor::full(vec![4], 221.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 221.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_222() {
        let t = Tensor::full(vec![4], 222.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 222.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_223() {
        let t = Tensor::full(vec![4], 223.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 223.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_224() {
        let t = Tensor::full(vec![4], 224.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 224.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_225() {
        let t = Tensor::full(vec![4], 225.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 225.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_226() {
        let t = Tensor::full(vec![4], 226.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 226.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_227() {
        let t = Tensor::full(vec![4], 227.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 227.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_228() {
        let t = Tensor::full(vec![4], 228.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 228.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_229() {
        let t = Tensor::full(vec![4], 229.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 229.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_230() {
        let t = Tensor::full(vec![4], 230.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 230.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_231() {
        let t = Tensor::full(vec![4], 231.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 231.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_232() {
        let t = Tensor::full(vec![4], 232.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 232.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_233() {
        let t = Tensor::full(vec![4], 233.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 233.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_234() {
        let t = Tensor::full(vec![4], 234.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 234.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_235() {
        let t = Tensor::full(vec![4], 235.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 235.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_236() {
        let t = Tensor::full(vec![4], 236.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 236.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_237() {
        let t = Tensor::full(vec![4], 237.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 237.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_238() {
        let t = Tensor::full(vec![4], 238.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 238.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_239() {
        let t = Tensor::full(vec![4], 239.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 239.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_240() {
        let t = Tensor::full(vec![4], 240.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 240.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_241() {
        let t = Tensor::full(vec![4], 241.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 241.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_242() {
        let t = Tensor::full(vec![4], 242.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 242.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_243() {
        let t = Tensor::full(vec![4], 243.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 243.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_244() {
        let t = Tensor::full(vec![4], 244.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 244.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_245() {
        let t = Tensor::full(vec![4], 245.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 245.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_246() {
        let t = Tensor::full(vec![4], 246.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 246.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_247() {
        let t = Tensor::full(vec![4], 247.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 247.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_248() {
        let t = Tensor::full(vec![4], 248.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 248.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_249() {
        let t = Tensor::full(vec![4], 249.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 249.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_250() {
        let t = Tensor::full(vec![4], 250.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 250.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_251() {
        let t = Tensor::full(vec![4], 251.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 251.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_252() {
        let t = Tensor::full(vec![4], 252.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 252.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_253() {
        let t = Tensor::full(vec![4], 253.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 253.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_254() {
        let t = Tensor::full(vec![4], 254.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 254.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_255() {
        let t = Tensor::full(vec![4], 255.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 255.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_256() {
        let t = Tensor::full(vec![4], 256.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 256.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_257() {
        let t = Tensor::full(vec![4], 257.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 257.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_258() {
        let t = Tensor::full(vec![4], 258.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 258.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_259() {
        let t = Tensor::full(vec![4], 259.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 259.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_260() {
        let t = Tensor::full(vec![4], 260.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 260.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_261() {
        let t = Tensor::full(vec![4], 261.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 261.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_262() {
        let t = Tensor::full(vec![4], 262.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 262.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_263() {
        let t = Tensor::full(vec![4], 263.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 263.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_264() {
        let t = Tensor::full(vec![4], 264.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 264.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_265() {
        let t = Tensor::full(vec![4], 265.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 265.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_266() {
        let t = Tensor::full(vec![4], 266.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 266.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_267() {
        let t = Tensor::full(vec![4], 267.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 267.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_268() {
        let t = Tensor::full(vec![4], 268.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 268.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_269() {
        let t = Tensor::full(vec![4], 269.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 269.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_270() {
        let t = Tensor::full(vec![4], 270.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 270.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_271() {
        let t = Tensor::full(vec![4], 271.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 271.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_272() {
        let t = Tensor::full(vec![4], 272.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 272.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_273() {
        let t = Tensor::full(vec![4], 273.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 273.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_274() {
        let t = Tensor::full(vec![4], 274.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 274.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_275() {
        let t = Tensor::full(vec![4], 275.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 275.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_276() {
        let t = Tensor::full(vec![4], 276.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 276.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_277() {
        let t = Tensor::full(vec![4], 277.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 277.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_278() {
        let t = Tensor::full(vec![4], 278.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 278.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_279() {
        let t = Tensor::full(vec![4], 279.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 279.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_280() {
        let t = Tensor::full(vec![4], 280.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 280.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_281() {
        let t = Tensor::full(vec![4], 281.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 281.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_282() {
        let t = Tensor::full(vec![4], 282.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 282.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_283() {
        let t = Tensor::full(vec![4], 283.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 283.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_284() {
        let t = Tensor::full(vec![4], 284.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 284.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_285() {
        let t = Tensor::full(vec![4], 285.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 285.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_286() {
        let t = Tensor::full(vec![4], 286.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 286.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_287() {
        let t = Tensor::full(vec![4], 287.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 287.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_288() {
        let t = Tensor::full(vec![4], 288.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 288.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_289() {
        let t = Tensor::full(vec![4], 289.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 289.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_290() {
        let t = Tensor::full(vec![4], 290.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 290.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_291() {
        let t = Tensor::full(vec![4], 291.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 291.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_292() {
        let t = Tensor::full(vec![4], 292.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 292.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_293() {
        let t = Tensor::full(vec![4], 293.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 293.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_294() {
        let t = Tensor::full(vec![4], 294.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 294.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_295() {
        let t = Tensor::full(vec![4], 295.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 295.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_296() {
        let t = Tensor::full(vec![4], 296.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 296.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_297() {
        let t = Tensor::full(vec![4], 297.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 297.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_298() {
        let t = Tensor::full(vec![4], 298.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 298.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_299() {
        let t = Tensor::full(vec![4], 299.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 299.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_300() {
        let t = Tensor::full(vec![4], 300.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 300.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_301() {
        let t = Tensor::full(vec![4], 301.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 301.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_302() {
        let t = Tensor::full(vec![4], 302.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 302.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_303() {
        let t = Tensor::full(vec![4], 303.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 303.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_304() {
        let t = Tensor::full(vec![4], 304.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 304.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_305() {
        let t = Tensor::full(vec![4], 305.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 305.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_306() {
        let t = Tensor::full(vec![4], 306.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 306.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_307() {
        let t = Tensor::full(vec![4], 307.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 307.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_308() {
        let t = Tensor::full(vec![4], 308.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 308.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_309() {
        let t = Tensor::full(vec![4], 309.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 309.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_310() {
        let t = Tensor::full(vec![4], 310.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 310.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_311() {
        let t = Tensor::full(vec![4], 311.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 311.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_312() {
        let t = Tensor::full(vec![4], 312.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 312.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_313() {
        let t = Tensor::full(vec![4], 313.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 313.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_314() {
        let t = Tensor::full(vec![4], 314.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 314.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_315() {
        let t = Tensor::full(vec![4], 315.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 315.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_316() {
        let t = Tensor::full(vec![4], 316.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 316.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_317() {
        let t = Tensor::full(vec![4], 317.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 317.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_318() {
        let t = Tensor::full(vec![4], 318.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 318.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_319() {
        let t = Tensor::full(vec![4], 319.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 319.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_320() {
        let t = Tensor::full(vec![4], 320.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 320.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_321() {
        let t = Tensor::full(vec![4], 321.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 321.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_322() {
        let t = Tensor::full(vec![4], 322.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 322.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_323() {
        let t = Tensor::full(vec![4], 323.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 323.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_324() {
        let t = Tensor::full(vec![4], 324.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 324.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_325() {
        let t = Tensor::full(vec![4], 325.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 325.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_326() {
        let t = Tensor::full(vec![4], 326.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 326.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_327() {
        let t = Tensor::full(vec![4], 327.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 327.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_328() {
        let t = Tensor::full(vec![4], 328.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 328.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_329() {
        let t = Tensor::full(vec![4], 329.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 329.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_330() {
        let t = Tensor::full(vec![4], 330.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 330.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_331() {
        let t = Tensor::full(vec![4], 331.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 331.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_332() {
        let t = Tensor::full(vec![4], 332.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 332.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_333() {
        let t = Tensor::full(vec![4], 333.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 333.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_334() {
        let t = Tensor::full(vec![4], 334.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 334.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_335() {
        let t = Tensor::full(vec![4], 335.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 335.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_336() {
        let t = Tensor::full(vec![4], 336.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 336.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_337() {
        let t = Tensor::full(vec![4], 337.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 337.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_338() {
        let t = Tensor::full(vec![4], 338.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 338.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_339() {
        let t = Tensor::full(vec![4], 339.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 339.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_340() {
        let t = Tensor::full(vec![4], 340.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 340.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_341() {
        let t = Tensor::full(vec![4], 341.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 341.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_342() {
        let t = Tensor::full(vec![4], 342.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 342.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_343() {
        let t = Tensor::full(vec![4], 343.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 343.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_344() {
        let t = Tensor::full(vec![4], 344.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 344.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_345() {
        let t = Tensor::full(vec![4], 345.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 345.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_346() {
        let t = Tensor::full(vec![4], 346.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 346.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_347() {
        let t = Tensor::full(vec![4], 347.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 347.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_348() {
        let t = Tensor::full(vec![4], 348.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 348.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_349() {
        let t = Tensor::full(vec![4], 349.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 349.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_350() {
        let t = Tensor::full(vec![4], 350.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 350.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_351() {
        let t = Tensor::full(vec![4], 351.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 351.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_352() {
        let t = Tensor::full(vec![4], 352.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 352.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_353() {
        let t = Tensor::full(vec![4], 353.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 353.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_354() {
        let t = Tensor::full(vec![4], 354.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 354.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_355() {
        let t = Tensor::full(vec![4], 355.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 355.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_356() {
        let t = Tensor::full(vec![4], 356.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 356.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_357() {
        let t = Tensor::full(vec![4], 357.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 357.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_358() {
        let t = Tensor::full(vec![4], 358.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 358.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_359() {
        let t = Tensor::full(vec![4], 359.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 359.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_360() {
        let t = Tensor::full(vec![4], 360.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 360.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_361() {
        let t = Tensor::full(vec![4], 361.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 361.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_362() {
        let t = Tensor::full(vec![4], 362.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 362.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_363() {
        let t = Tensor::full(vec![4], 363.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 363.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_364() {
        let t = Tensor::full(vec![4], 364.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 364.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_365() {
        let t = Tensor::full(vec![4], 365.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 365.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_366() {
        let t = Tensor::full(vec![4], 366.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 366.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_367() {
        let t = Tensor::full(vec![4], 367.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 367.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_368() {
        let t = Tensor::full(vec![4], 368.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 368.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_369() {
        let t = Tensor::full(vec![4], 369.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 369.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_370() {
        let t = Tensor::full(vec![4], 370.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 370.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_371() {
        let t = Tensor::full(vec![4], 371.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 371.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_372() {
        let t = Tensor::full(vec![4], 372.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 372.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_373() {
        let t = Tensor::full(vec![4], 373.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 373.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_374() {
        let t = Tensor::full(vec![4], 374.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 374.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_375() {
        let t = Tensor::full(vec![4], 375.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 375.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_376() {
        let t = Tensor::full(vec![4], 376.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 376.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_377() {
        let t = Tensor::full(vec![4], 377.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 377.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_378() {
        let t = Tensor::full(vec![4], 378.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 378.0);
        assert!(s.is_finite);
    }

    #[test]
    fn test_tmod_stress_case_379() {
        let t = Tensor::full(vec![4], 379.0);
        let s = TensorStats::compute(&t);
        assert_eq!(s.numel, 4);
        assert_eq!(s.mean, 379.0);
        assert!(s.is_finite);
    }
}
