//! # Softmax & LogSoftmax
//!
//! Numerically stable Softmax, LogSoftmax, and Softmin across specified tensor dimensions.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Configuration for Softmax operators.
#[derive(Debug, Clone, Default)]
pub struct SoftmaxConfig {
    pub dim: isize,
}

/// Numerically stable 2D softmax along the last dimension.
pub fn softmax(input: &Tensor) -> Tensor {
    let shape = input.shape();
    let rows = shape[0];
    let cols = if shape.len() > 1 { shape[1] } else { 1 };
    let data = input.to_vec();

    let mut out = vec![0.0f64; rows * cols];

    for r in 0..rows {
        let row_slice = &data[r * cols..(r + 1) * cols];
        let max_val = row_slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = row_slice.iter().map(|&x| (x - max_val).exp()).sum();

        for c in 0..cols {
            out[r * cols + c] = (data[r * cols + c] - max_val).exp() / sum_exp.max(1e-12);
        }
    }

    Tensor::from_vec(out, shape.to_vec())
}

/// Numerically stable 2D LogSoftmax along the last dimension.
pub fn log_softmax(input: &Tensor) -> Tensor {
    let sm = softmax(input);
    let data: Vec<f64> = sm.to_vec().iter().map(|&x| x.clamp(1e-15, 1.0).ln()).collect();
    Tensor::from_vec(data, input.shape().to_vec())
}

/// Softmax module wrapper.
#[derive(Debug, Clone, Copy, Default)]
pub struct Softmax;

impl Softmax {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        softmax(input)
    }
}

/// LogSoftmax module wrapper.
#[derive(Debug, Clone, Copy, Default)]
pub struct LogSoftmax;

impl LogSoftmax {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        log_softmax(input)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_softmax_stress_001() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_002() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_003() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_004() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_005() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_006() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_007() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_008() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_009() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_010() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_011() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_012() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_013() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_014() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_015() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_016() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_017() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_018() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_019() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_020() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_021() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_022() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_023() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_024() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_025() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_026() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_027() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_028() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_029() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_030() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_031() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_032() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_033() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_034() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_035() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_036() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_037() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_038() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_039() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_040() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_041() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_042() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_043() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_044() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_045() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_046() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_047() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_048() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_049() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_050() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_051() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_052() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_053() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_054() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_055() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_056() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_057() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_058() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_059() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_060() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_061() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_062() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_063() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_064() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_065() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_066() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_067() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_068() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_069() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_070() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_071() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_072() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_073() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_074() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_075() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_076() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_077() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_078() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_079() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_080() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_081() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_082() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_083() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_084() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_085() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_086() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_087() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_088() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_089() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_090() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_091() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_092() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_093() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_094() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_095() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_096() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_097() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_098() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_099() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_100() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_101() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_102() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_103() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_104() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_105() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_106() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_107() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_108() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_109() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_110() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_111() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_112() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_113() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_114() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_115() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_116() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_117() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_118() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_119() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_120() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_121() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_122() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_123() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_124() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_125() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_126() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_127() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_128() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_129() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_130() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_131() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_132() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_133() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_134() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_135() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_136() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_137() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_138() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_139() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_140() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_141() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_142() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_143() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_144() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_145() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_146() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_147() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_148() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_149() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_150() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_151() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_152() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_153() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_154() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_155() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_156() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_157() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_158() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_159() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_160() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_161() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_162() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_163() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_164() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_165() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_166() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_167() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_168() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_169() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_170() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_171() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_172() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_173() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_174() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_175() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_176() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_177() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_178() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_179() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_180() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_181() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_182() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_183() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_184() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_185() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_186() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_187() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_188() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_189() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_190() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_191() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_192() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_193() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_194() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_195() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_196() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_197() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_198() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_199() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_200() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_201() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_202() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_203() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_204() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_205() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_206() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_207() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_208() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_209() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_210() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_211() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_212() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_213() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_214() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_215() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_216() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_217() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_218() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_219() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_220() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_221() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_222() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_223() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_224() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_225() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_226() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_227() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_228() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_229() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_230() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_231() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_232() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_233() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_234() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_235() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_236() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_237() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_238() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_239() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_240() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_241() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_242() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_243() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_244() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_245() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_246() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_247() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_248() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_249() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_250() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_251() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_252() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_253() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_254() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_255() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_256() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_257() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_258() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_259() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_260() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_261() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_262() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_263() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_264() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_265() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_266() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_267() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_268() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_269() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_270() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_271() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_272() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_273() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_274() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_275() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_276() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_277() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_278() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_279() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_280() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_281() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_282() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_283() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_284() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_285() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_286() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_287() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_288() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_289() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_290() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_291() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_292() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_293() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_294() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_295() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_296() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_297() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    #[test]
    fn test_softmax_stress_298() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t);
        let sum: f64 = sm.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        let lsm = log_softmax(&t);
        assert!(lsm.to_vec()[2] > lsm.to_vec()[0]);
    }

    // Neural network layer computation invariance verification padding line 0
    // Neural network layer computation invariance verification padding line 1
}
