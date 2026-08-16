//! Fused computation kernels (Linear+Activation, LayerNorm, RMSNorm, FMA).
//!
//! This module provides single-pass fused operations for neural network layers.

use crate::tensor::Tensor;

/// Fused Multiply-Add: out = a * b + c (element-wise).
pub fn fused_fma(a: &Tensor, b: &Tensor, c: &Tensor) -> Tensor {
    let numel = a.numel();
    assert_eq!(b.numel(), numel);
    assert_eq!(c.numel(), numel);
    let mut out = Vec::with_capacity(numel);
    for i in 0..numel {
        out.push(a.get(i) * b.get(i) + c.get(i));
    }
    Tensor::new(out, a.shape().to_vec())
}

/// Fused Linear layer: out = x @ w^T + bias.
pub fn fused_linear(x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> Tensor {
    let wt = w.t();
    let xw = crate::tensor::arithmetic::matmul(x, &wt);
    if let Some(b) = bias {
        crate::tensor::arithmetic::add(&xw, b)
    } else {
        xw
    }
}

/// Fused Linear + ReLU: out = ReLU(x @ w^T + bias).
pub fn fused_linear_relu(x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> Tensor {
    let lin = fused_linear(x, w, bias);
    crate::tensor::math::relu(&lin)
}

/// Fused Linear + GELU: out = GELU(x @ w^T + bias).
pub fn fused_linear_gelu(x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> Tensor {
    let lin = fused_linear(x, w, bias);
    crate::tensor::math::gelu(&lin)
}

/// Fused Layer Normalization: out = ((x - mean) / sqrt(var + eps)) * gamma + beta.
pub fn fused_layer_norm(
    x: &Tensor,
    normalized_shape: &[usize],
    gamma: Option<&Tensor>,
    beta: Option<&Tensor>,
    eps: f64,
) -> Tensor {
    assert_eq!(x.ndim(), 2);
    let (rows, cols) = (x.shape()[0], x.shape()[1]);
    let mut out = Tensor::zeros(vec![rows, cols]);

    for r in 0..rows {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for c in 0..cols {
            let val = x.get_2d(r, c);
            sum += val;
            sum_sq += val * val;
        }
        let mean = sum / (cols as f64);
        let var = (sum_sq / (cols as f64)) - mean * mean;
        let inv_std = 1.0 / (var.max(0.0) + eps).sqrt();

        for c in 0..cols {
            let g = gamma.map(|g_t| g_t.get(c)).unwrap_or(1.0);
            let b = beta.map(|b_t| b_t.get(c)).unwrap_or(0.0);
            let norm_val = (x.get_2d(r, c) - mean) * inv_std * g + b;
            out.set_2d(r, c, norm_val);
        }
    }
    out
}

/// Fused Root Mean Square Layer Normalization (RMSNorm): out = (x / sqrt(mean(x^2) + eps)) * weight.
pub fn fused_rmsnorm(x: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
    assert_eq!(x.ndim(), 2);
    let (rows, cols) = (x.shape()[0], x.shape()[1]);
    let mut out = Tensor::zeros(vec![rows, cols]);

    for r in 0..rows {
        let mut sum_sq = 0.0;
        for c in 0..cols {
            let val = x.get_2d(r, c);
            sum_sq += val * val;
        }
        let rms = 1.0 / ((sum_sq / (cols as f64)) + eps).sqrt();
        for c in 0..cols {
            let w = weight.get(c);
            out.set_2d(r, c, x.get_2d(r, c) * rms * w);
        }
    }
    out
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_linear() {
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let w = Tensor::from_slice(&[1.0, 1.0], vec![1, 2]);
        let b = Tensor::from_slice(&[5.0], vec![1, 1]);
        let out = fused_linear(&x, &w, Some(&b));
        assert_eq!(out.data(), &[8.0]);
    }

    #[test]
    fn test_layer_norm() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![1, 3]);
        let ln = fused_layer_norm(&x, &[3], None, None, 1e-5);
        assert!((crate::tensor::reduction::mean(&ln)).abs() < 1e-5);
    }

    #[test]
    fn test_fused_stress_case_001() {
        let x = Tensor::full(vec![1, 2], 1.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 1.0);
    }

    #[test]
    fn test_fused_stress_case_002() {
        let x = Tensor::full(vec![1, 2], 2.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 2.0);
    }

    #[test]
    fn test_fused_stress_case_003() {
        let x = Tensor::full(vec![1, 2], 3.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 3.0);
    }

    #[test]
    fn test_fused_stress_case_004() {
        let x = Tensor::full(vec![1, 2], 4.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 4.0);
    }

    #[test]
    fn test_fused_stress_case_005() {
        let x = Tensor::full(vec![1, 2], 5.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 5.0);
    }

    #[test]
    fn test_fused_stress_case_006() {
        let x = Tensor::full(vec![1, 2], 6.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 6.0);
    }

    #[test]
    fn test_fused_stress_case_007() {
        let x = Tensor::full(vec![1, 2], 7.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 7.0);
    }

    #[test]
    fn test_fused_stress_case_008() {
        let x = Tensor::full(vec![1, 2], 8.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 8.0);
    }

    #[test]
    fn test_fused_stress_case_009() {
        let x = Tensor::full(vec![1, 2], 9.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 9.0);
    }

    #[test]
    fn test_fused_stress_case_010() {
        let x = Tensor::full(vec![1, 2], 10.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 10.0);
    }

    #[test]
    fn test_fused_stress_case_011() {
        let x = Tensor::full(vec![1, 2], 11.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 11.0);
    }

    #[test]
    fn test_fused_stress_case_012() {
        let x = Tensor::full(vec![1, 2], 12.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 12.0);
    }

    #[test]
    fn test_fused_stress_case_013() {
        let x = Tensor::full(vec![1, 2], 13.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 13.0);
    }

    #[test]
    fn test_fused_stress_case_014() {
        let x = Tensor::full(vec![1, 2], 14.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 14.0);
    }

    #[test]
    fn test_fused_stress_case_015() {
        let x = Tensor::full(vec![1, 2], 15.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 15.0);
    }

    #[test]
    fn test_fused_stress_case_016() {
        let x = Tensor::full(vec![1, 2], 16.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 16.0);
    }

    #[test]
    fn test_fused_stress_case_017() {
        let x = Tensor::full(vec![1, 2], 17.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 17.0);
    }

    #[test]
    fn test_fused_stress_case_018() {
        let x = Tensor::full(vec![1, 2], 18.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 18.0);
    }

    #[test]
    fn test_fused_stress_case_019() {
        let x = Tensor::full(vec![1, 2], 19.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 19.0);
    }

    #[test]
    fn test_fused_stress_case_020() {
        let x = Tensor::full(vec![1, 2], 20.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 20.0);
    }

    #[test]
    fn test_fused_stress_case_021() {
        let x = Tensor::full(vec![1, 2], 21.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 21.0);
    }

    #[test]
    fn test_fused_stress_case_022() {
        let x = Tensor::full(vec![1, 2], 22.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 22.0);
    }

    #[test]
    fn test_fused_stress_case_023() {
        let x = Tensor::full(vec![1, 2], 23.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 23.0);
    }

    #[test]
    fn test_fused_stress_case_024() {
        let x = Tensor::full(vec![1, 2], 24.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 24.0);
    }

    #[test]
    fn test_fused_stress_case_025() {
        let x = Tensor::full(vec![1, 2], 25.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 25.0);
    }

    #[test]
    fn test_fused_stress_case_026() {
        let x = Tensor::full(vec![1, 2], 26.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 26.0);
    }

    #[test]
    fn test_fused_stress_case_027() {
        let x = Tensor::full(vec![1, 2], 27.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 27.0);
    }

    #[test]
    fn test_fused_stress_case_028() {
        let x = Tensor::full(vec![1, 2], 28.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 28.0);
    }

    #[test]
    fn test_fused_stress_case_029() {
        let x = Tensor::full(vec![1, 2], 29.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 29.0);
    }

    #[test]
    fn test_fused_stress_case_030() {
        let x = Tensor::full(vec![1, 2], 30.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 30.0);
    }

    #[test]
    fn test_fused_stress_case_031() {
        let x = Tensor::full(vec![1, 2], 31.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 31.0);
    }

    #[test]
    fn test_fused_stress_case_032() {
        let x = Tensor::full(vec![1, 2], 32.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 32.0);
    }

    #[test]
    fn test_fused_stress_case_033() {
        let x = Tensor::full(vec![1, 2], 33.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 33.0);
    }

    #[test]
    fn test_fused_stress_case_034() {
        let x = Tensor::full(vec![1, 2], 34.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 34.0);
    }

    #[test]
    fn test_fused_stress_case_035() {
        let x = Tensor::full(vec![1, 2], 35.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 35.0);
    }

    #[test]
    fn test_fused_stress_case_036() {
        let x = Tensor::full(vec![1, 2], 36.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 36.0);
    }

    #[test]
    fn test_fused_stress_case_037() {
        let x = Tensor::full(vec![1, 2], 37.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 37.0);
    }

    #[test]
    fn test_fused_stress_case_038() {
        let x = Tensor::full(vec![1, 2], 38.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 38.0);
    }

    #[test]
    fn test_fused_stress_case_039() {
        let x = Tensor::full(vec![1, 2], 39.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 39.0);
    }

    #[test]
    fn test_fused_stress_case_040() {
        let x = Tensor::full(vec![1, 2], 40.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 40.0);
    }

    #[test]
    fn test_fused_stress_case_041() {
        let x = Tensor::full(vec![1, 2], 41.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 41.0);
    }

    #[test]
    fn test_fused_stress_case_042() {
        let x = Tensor::full(vec![1, 2], 42.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 42.0);
    }

    #[test]
    fn test_fused_stress_case_043() {
        let x = Tensor::full(vec![1, 2], 43.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 43.0);
    }

    #[test]
    fn test_fused_stress_case_044() {
        let x = Tensor::full(vec![1, 2], 44.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 44.0);
    }

    #[test]
    fn test_fused_stress_case_045() {
        let x = Tensor::full(vec![1, 2], 45.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 45.0);
    }

    #[test]
    fn test_fused_stress_case_046() {
        let x = Tensor::full(vec![1, 2], 46.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 46.0);
    }

    #[test]
    fn test_fused_stress_case_047() {
        let x = Tensor::full(vec![1, 2], 47.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 47.0);
    }

    #[test]
    fn test_fused_stress_case_048() {
        let x = Tensor::full(vec![1, 2], 48.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 48.0);
    }

    #[test]
    fn test_fused_stress_case_049() {
        let x = Tensor::full(vec![1, 2], 49.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 49.0);
    }

    #[test]
    fn test_fused_stress_case_050() {
        let x = Tensor::full(vec![1, 2], 50.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 50.0);
    }

    #[test]
    fn test_fused_stress_case_051() {
        let x = Tensor::full(vec![1, 2], 51.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 51.0);
    }

    #[test]
    fn test_fused_stress_case_052() {
        let x = Tensor::full(vec![1, 2], 52.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 52.0);
    }

    #[test]
    fn test_fused_stress_case_053() {
        let x = Tensor::full(vec![1, 2], 53.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 53.0);
    }

    #[test]
    fn test_fused_stress_case_054() {
        let x = Tensor::full(vec![1, 2], 54.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 54.0);
    }

    #[test]
    fn test_fused_stress_case_055() {
        let x = Tensor::full(vec![1, 2], 55.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 55.0);
    }

    #[test]
    fn test_fused_stress_case_056() {
        let x = Tensor::full(vec![1, 2], 56.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 56.0);
    }

    #[test]
    fn test_fused_stress_case_057() {
        let x = Tensor::full(vec![1, 2], 57.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 57.0);
    }

    #[test]
    fn test_fused_stress_case_058() {
        let x = Tensor::full(vec![1, 2], 58.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 58.0);
    }

    #[test]
    fn test_fused_stress_case_059() {
        let x = Tensor::full(vec![1, 2], 59.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 59.0);
    }

    #[test]
    fn test_fused_stress_case_060() {
        let x = Tensor::full(vec![1, 2], 60.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 60.0);
    }

    #[test]
    fn test_fused_stress_case_061() {
        let x = Tensor::full(vec![1, 2], 61.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 61.0);
    }

    #[test]
    fn test_fused_stress_case_062() {
        let x = Tensor::full(vec![1, 2], 62.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 62.0);
    }

    #[test]
    fn test_fused_stress_case_063() {
        let x = Tensor::full(vec![1, 2], 63.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 63.0);
    }

    #[test]
    fn test_fused_stress_case_064() {
        let x = Tensor::full(vec![1, 2], 64.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 64.0);
    }

    #[test]
    fn test_fused_stress_case_065() {
        let x = Tensor::full(vec![1, 2], 65.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 65.0);
    }

    #[test]
    fn test_fused_stress_case_066() {
        let x = Tensor::full(vec![1, 2], 66.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 66.0);
    }

    #[test]
    fn test_fused_stress_case_067() {
        let x = Tensor::full(vec![1, 2], 67.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 67.0);
    }

    #[test]
    fn test_fused_stress_case_068() {
        let x = Tensor::full(vec![1, 2], 68.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 68.0);
    }

    #[test]
    fn test_fused_stress_case_069() {
        let x = Tensor::full(vec![1, 2], 69.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 69.0);
    }

    #[test]
    fn test_fused_stress_case_070() {
        let x = Tensor::full(vec![1, 2], 70.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 70.0);
    }

    #[test]
    fn test_fused_stress_case_071() {
        let x = Tensor::full(vec![1, 2], 71.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 71.0);
    }

    #[test]
    fn test_fused_stress_case_072() {
        let x = Tensor::full(vec![1, 2], 72.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 72.0);
    }

    #[test]
    fn test_fused_stress_case_073() {
        let x = Tensor::full(vec![1, 2], 73.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 73.0);
    }

    #[test]
    fn test_fused_stress_case_074() {
        let x = Tensor::full(vec![1, 2], 74.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 74.0);
    }

    #[test]
    fn test_fused_stress_case_075() {
        let x = Tensor::full(vec![1, 2], 75.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 75.0);
    }

    #[test]
    fn test_fused_stress_case_076() {
        let x = Tensor::full(vec![1, 2], 76.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 76.0);
    }

    #[test]
    fn test_fused_stress_case_077() {
        let x = Tensor::full(vec![1, 2], 77.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 77.0);
    }

    #[test]
    fn test_fused_stress_case_078() {
        let x = Tensor::full(vec![1, 2], 78.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 78.0);
    }

    #[test]
    fn test_fused_stress_case_079() {
        let x = Tensor::full(vec![1, 2], 79.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 79.0);
    }

    #[test]
    fn test_fused_stress_case_080() {
        let x = Tensor::full(vec![1, 2], 80.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 80.0);
    }

    #[test]
    fn test_fused_stress_case_081() {
        let x = Tensor::full(vec![1, 2], 81.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 81.0);
    }

    #[test]
    fn test_fused_stress_case_082() {
        let x = Tensor::full(vec![1, 2], 82.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 82.0);
    }

    #[test]
    fn test_fused_stress_case_083() {
        let x = Tensor::full(vec![1, 2], 83.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 83.0);
    }

    #[test]
    fn test_fused_stress_case_084() {
        let x = Tensor::full(vec![1, 2], 84.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 84.0);
    }

    #[test]
    fn test_fused_stress_case_085() {
        let x = Tensor::full(vec![1, 2], 85.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 85.0);
    }

    #[test]
    fn test_fused_stress_case_086() {
        let x = Tensor::full(vec![1, 2], 86.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 86.0);
    }

    #[test]
    fn test_fused_stress_case_087() {
        let x = Tensor::full(vec![1, 2], 87.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 87.0);
    }

    #[test]
    fn test_fused_stress_case_088() {
        let x = Tensor::full(vec![1, 2], 88.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 88.0);
    }

    #[test]
    fn test_fused_stress_case_089() {
        let x = Tensor::full(vec![1, 2], 89.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 89.0);
    }

    #[test]
    fn test_fused_stress_case_090() {
        let x = Tensor::full(vec![1, 2], 90.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 90.0);
    }

    #[test]
    fn test_fused_stress_case_091() {
        let x = Tensor::full(vec![1, 2], 91.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 91.0);
    }

    #[test]
    fn test_fused_stress_case_092() {
        let x = Tensor::full(vec![1, 2], 92.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 92.0);
    }

    #[test]
    fn test_fused_stress_case_093() {
        let x = Tensor::full(vec![1, 2], 93.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 93.0);
    }

    #[test]
    fn test_fused_stress_case_094() {
        let x = Tensor::full(vec![1, 2], 94.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 94.0);
    }

    #[test]
    fn test_fused_stress_case_095() {
        let x = Tensor::full(vec![1, 2], 95.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 95.0);
    }

    #[test]
    fn test_fused_stress_case_096() {
        let x = Tensor::full(vec![1, 2], 96.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 96.0);
    }

    #[test]
    fn test_fused_stress_case_097() {
        let x = Tensor::full(vec![1, 2], 97.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 97.0);
    }

    #[test]
    fn test_fused_stress_case_098() {
        let x = Tensor::full(vec![1, 2], 98.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 98.0);
    }

    #[test]
    fn test_fused_stress_case_099() {
        let x = Tensor::full(vec![1, 2], 99.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 99.0);
    }

    #[test]
    fn test_fused_stress_case_100() {
        let x = Tensor::full(vec![1, 2], 100.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 100.0);
    }

    #[test]
    fn test_fused_stress_case_101() {
        let x = Tensor::full(vec![1, 2], 101.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 101.0);
    }

    #[test]
    fn test_fused_stress_case_102() {
        let x = Tensor::full(vec![1, 2], 102.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 102.0);
    }

    #[test]
    fn test_fused_stress_case_103() {
        let x = Tensor::full(vec![1, 2], 103.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 103.0);
    }

    #[test]
    fn test_fused_stress_case_104() {
        let x = Tensor::full(vec![1, 2], 104.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 104.0);
    }

    #[test]
    fn test_fused_stress_case_105() {
        let x = Tensor::full(vec![1, 2], 105.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 105.0);
    }

    #[test]
    fn test_fused_stress_case_106() {
        let x = Tensor::full(vec![1, 2], 106.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 106.0);
    }

    #[test]
    fn test_fused_stress_case_107() {
        let x = Tensor::full(vec![1, 2], 107.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 107.0);
    }

    #[test]
    fn test_fused_stress_case_108() {
        let x = Tensor::full(vec![1, 2], 108.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 108.0);
    }

    #[test]
    fn test_fused_stress_case_109() {
        let x = Tensor::full(vec![1, 2], 109.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 109.0);
    }

    #[test]
    fn test_fused_stress_case_110() {
        let x = Tensor::full(vec![1, 2], 110.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 110.0);
    }

    #[test]
    fn test_fused_stress_case_111() {
        let x = Tensor::full(vec![1, 2], 111.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 111.0);
    }

    #[test]
    fn test_fused_stress_case_112() {
        let x = Tensor::full(vec![1, 2], 112.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 112.0);
    }

    #[test]
    fn test_fused_stress_case_113() {
        let x = Tensor::full(vec![1, 2], 113.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 113.0);
    }

    #[test]
    fn test_fused_stress_case_114() {
        let x = Tensor::full(vec![1, 2], 114.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 114.0);
    }

    #[test]
    fn test_fused_stress_case_115() {
        let x = Tensor::full(vec![1, 2], 115.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 115.0);
    }

    #[test]
    fn test_fused_stress_case_116() {
        let x = Tensor::full(vec![1, 2], 116.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 116.0);
    }

    #[test]
    fn test_fused_stress_case_117() {
        let x = Tensor::full(vec![1, 2], 117.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 117.0);
    }

    #[test]
    fn test_fused_stress_case_118() {
        let x = Tensor::full(vec![1, 2], 118.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 118.0);
    }

    #[test]
    fn test_fused_stress_case_119() {
        let x = Tensor::full(vec![1, 2], 119.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 119.0);
    }

    #[test]
    fn test_fused_stress_case_120() {
        let x = Tensor::full(vec![1, 2], 120.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 120.0);
    }

    #[test]
    fn test_fused_stress_case_121() {
        let x = Tensor::full(vec![1, 2], 121.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 121.0);
    }

    #[test]
    fn test_fused_stress_case_122() {
        let x = Tensor::full(vec![1, 2], 122.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 122.0);
    }

    #[test]
    fn test_fused_stress_case_123() {
        let x = Tensor::full(vec![1, 2], 123.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 123.0);
    }

    #[test]
    fn test_fused_stress_case_124() {
        let x = Tensor::full(vec![1, 2], 124.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 124.0);
    }

    #[test]
    fn test_fused_stress_case_125() {
        let x = Tensor::full(vec![1, 2], 125.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 125.0);
    }

    #[test]
    fn test_fused_stress_case_126() {
        let x = Tensor::full(vec![1, 2], 126.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 126.0);
    }

    #[test]
    fn test_fused_stress_case_127() {
        let x = Tensor::full(vec![1, 2], 127.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 127.0);
    }

    #[test]
    fn test_fused_stress_case_128() {
        let x = Tensor::full(vec![1, 2], 128.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 128.0);
    }

    #[test]
    fn test_fused_stress_case_129() {
        let x = Tensor::full(vec![1, 2], 129.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 129.0);
    }

    #[test]
    fn test_fused_stress_case_130() {
        let x = Tensor::full(vec![1, 2], 130.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 130.0);
    }

    #[test]
    fn test_fused_stress_case_131() {
        let x = Tensor::full(vec![1, 2], 131.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 131.0);
    }

    #[test]
    fn test_fused_stress_case_132() {
        let x = Tensor::full(vec![1, 2], 132.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 132.0);
    }

    #[test]
    fn test_fused_stress_case_133() {
        let x = Tensor::full(vec![1, 2], 133.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 133.0);
    }

    #[test]
    fn test_fused_stress_case_134() {
        let x = Tensor::full(vec![1, 2], 134.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 134.0);
    }

    #[test]
    fn test_fused_stress_case_135() {
        let x = Tensor::full(vec![1, 2], 135.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 135.0);
    }

    #[test]
    fn test_fused_stress_case_136() {
        let x = Tensor::full(vec![1, 2], 136.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 136.0);
    }

    #[test]
    fn test_fused_stress_case_137() {
        let x = Tensor::full(vec![1, 2], 137.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 137.0);
    }

    #[test]
    fn test_fused_stress_case_138() {
        let x = Tensor::full(vec![1, 2], 138.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 138.0);
    }

    #[test]
    fn test_fused_stress_case_139() {
        let x = Tensor::full(vec![1, 2], 139.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 139.0);
    }

    #[test]
    fn test_fused_stress_case_140() {
        let x = Tensor::full(vec![1, 2], 140.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 140.0);
    }

    #[test]
    fn test_fused_stress_case_141() {
        let x = Tensor::full(vec![1, 2], 141.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 141.0);
    }

    #[test]
    fn test_fused_stress_case_142() {
        let x = Tensor::full(vec![1, 2], 142.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 142.0);
    }

    #[test]
    fn test_fused_stress_case_143() {
        let x = Tensor::full(vec![1, 2], 143.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 143.0);
    }

    #[test]
    fn test_fused_stress_case_144() {
        let x = Tensor::full(vec![1, 2], 144.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 144.0);
    }

    #[test]
    fn test_fused_stress_case_145() {
        let x = Tensor::full(vec![1, 2], 145.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 145.0);
    }

    #[test]
    fn test_fused_stress_case_146() {
        let x = Tensor::full(vec![1, 2], 146.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 146.0);
    }

    #[test]
    fn test_fused_stress_case_147() {
        let x = Tensor::full(vec![1, 2], 147.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 147.0);
    }

    #[test]
    fn test_fused_stress_case_148() {
        let x = Tensor::full(vec![1, 2], 148.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 148.0);
    }

    #[test]
    fn test_fused_stress_case_149() {
        let x = Tensor::full(vec![1, 2], 149.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 149.0);
    }

    #[test]
    fn test_fused_stress_case_150() {
        let x = Tensor::full(vec![1, 2], 150.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 150.0);
    }

    #[test]
    fn test_fused_stress_case_151() {
        let x = Tensor::full(vec![1, 2], 151.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 151.0);
    }

    #[test]
    fn test_fused_stress_case_152() {
        let x = Tensor::full(vec![1, 2], 152.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 152.0);
    }

    #[test]
    fn test_fused_stress_case_153() {
        let x = Tensor::full(vec![1, 2], 153.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 153.0);
    }

    #[test]
    fn test_fused_stress_case_154() {
        let x = Tensor::full(vec![1, 2], 154.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 154.0);
    }

    #[test]
    fn test_fused_stress_case_155() {
        let x = Tensor::full(vec![1, 2], 155.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 155.0);
    }

    #[test]
    fn test_fused_stress_case_156() {
        let x = Tensor::full(vec![1, 2], 156.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 156.0);
    }

    #[test]
    fn test_fused_stress_case_157() {
        let x = Tensor::full(vec![1, 2], 157.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 157.0);
    }

    #[test]
    fn test_fused_stress_case_158() {
        let x = Tensor::full(vec![1, 2], 158.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 158.0);
    }

    #[test]
    fn test_fused_stress_case_159() {
        let x = Tensor::full(vec![1, 2], 159.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 159.0);
    }

    #[test]
    fn test_fused_stress_case_160() {
        let x = Tensor::full(vec![1, 2], 160.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 160.0);
    }

    #[test]
    fn test_fused_stress_case_161() {
        let x = Tensor::full(vec![1, 2], 161.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 161.0);
    }

    #[test]
    fn test_fused_stress_case_162() {
        let x = Tensor::full(vec![1, 2], 162.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 162.0);
    }

    #[test]
    fn test_fused_stress_case_163() {
        let x = Tensor::full(vec![1, 2], 163.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 163.0);
    }

    #[test]
    fn test_fused_stress_case_164() {
        let x = Tensor::full(vec![1, 2], 164.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 164.0);
    }

    #[test]
    fn test_fused_stress_case_165() {
        let x = Tensor::full(vec![1, 2], 165.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 165.0);
    }

    #[test]
    fn test_fused_stress_case_166() {
        let x = Tensor::full(vec![1, 2], 166.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 166.0);
    }

    #[test]
    fn test_fused_stress_case_167() {
        let x = Tensor::full(vec![1, 2], 167.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 167.0);
    }

    #[test]
    fn test_fused_stress_case_168() {
        let x = Tensor::full(vec![1, 2], 168.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 168.0);
    }

    #[test]
    fn test_fused_stress_case_169() {
        let x = Tensor::full(vec![1, 2], 169.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 169.0);
    }

    #[test]
    fn test_fused_stress_case_170() {
        let x = Tensor::full(vec![1, 2], 170.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 170.0);
    }

    #[test]
    fn test_fused_stress_case_171() {
        let x = Tensor::full(vec![1, 2], 171.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 171.0);
    }

    #[test]
    fn test_fused_stress_case_172() {
        let x = Tensor::full(vec![1, 2], 172.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 172.0);
    }

    #[test]
    fn test_fused_stress_case_173() {
        let x = Tensor::full(vec![1, 2], 173.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 173.0);
    }

    #[test]
    fn test_fused_stress_case_174() {
        let x = Tensor::full(vec![1, 2], 174.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 174.0);
    }

    #[test]
    fn test_fused_stress_case_175() {
        let x = Tensor::full(vec![1, 2], 175.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 175.0);
    }

    #[test]
    fn test_fused_stress_case_176() {
        let x = Tensor::full(vec![1, 2], 176.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 176.0);
    }

    #[test]
    fn test_fused_stress_case_177() {
        let x = Tensor::full(vec![1, 2], 177.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 177.0);
    }

    #[test]
    fn test_fused_stress_case_178() {
        let x = Tensor::full(vec![1, 2], 178.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 178.0);
    }

    #[test]
    fn test_fused_stress_case_179() {
        let x = Tensor::full(vec![1, 2], 179.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 179.0);
    }

    #[test]
    fn test_fused_stress_case_180() {
        let x = Tensor::full(vec![1, 2], 180.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 180.0);
    }

    #[test]
    fn test_fused_stress_case_181() {
        let x = Tensor::full(vec![1, 2], 181.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 181.0);
    }

    #[test]
    fn test_fused_stress_case_182() {
        let x = Tensor::full(vec![1, 2], 182.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 182.0);
    }

    #[test]
    fn test_fused_stress_case_183() {
        let x = Tensor::full(vec![1, 2], 183.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 183.0);
    }

    #[test]
    fn test_fused_stress_case_184() {
        let x = Tensor::full(vec![1, 2], 184.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 184.0);
    }

    #[test]
    fn test_fused_stress_case_185() {
        let x = Tensor::full(vec![1, 2], 185.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 185.0);
    }

    #[test]
    fn test_fused_stress_case_186() {
        let x = Tensor::full(vec![1, 2], 186.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 186.0);
    }

    #[test]
    fn test_fused_stress_case_187() {
        let x = Tensor::full(vec![1, 2], 187.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 187.0);
    }

    #[test]
    fn test_fused_stress_case_188() {
        let x = Tensor::full(vec![1, 2], 188.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 188.0);
    }

    #[test]
    fn test_fused_stress_case_189() {
        let x = Tensor::full(vec![1, 2], 189.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 189.0);
    }

    #[test]
    fn test_fused_stress_case_190() {
        let x = Tensor::full(vec![1, 2], 190.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 190.0);
    }

    #[test]
    fn test_fused_stress_case_191() {
        let x = Tensor::full(vec![1, 2], 191.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 191.0);
    }

    #[test]
    fn test_fused_stress_case_192() {
        let x = Tensor::full(vec![1, 2], 192.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 192.0);
    }

    #[test]
    fn test_fused_stress_case_193() {
        let x = Tensor::full(vec![1, 2], 193.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 193.0);
    }

    #[test]
    fn test_fused_stress_case_194() {
        let x = Tensor::full(vec![1, 2], 194.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 194.0);
    }

    #[test]
    fn test_fused_stress_case_195() {
        let x = Tensor::full(vec![1, 2], 195.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 195.0);
    }

    #[test]
    fn test_fused_stress_case_196() {
        let x = Tensor::full(vec![1, 2], 196.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 196.0);
    }

    #[test]
    fn test_fused_stress_case_197() {
        let x = Tensor::full(vec![1, 2], 197.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 197.0);
    }

    #[test]
    fn test_fused_stress_case_198() {
        let x = Tensor::full(vec![1, 2], 198.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 198.0);
    }

    #[test]
    fn test_fused_stress_case_199() {
        let x = Tensor::full(vec![1, 2], 199.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 199.0);
    }

    #[test]
    fn test_fused_stress_case_200() {
        let x = Tensor::full(vec![1, 2], 200.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 200.0);
    }

    #[test]
    fn test_fused_stress_case_201() {
        let x = Tensor::full(vec![1, 2], 201.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 201.0);
    }

    #[test]
    fn test_fused_stress_case_202() {
        let x = Tensor::full(vec![1, 2], 202.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 202.0);
    }

    #[test]
    fn test_fused_stress_case_203() {
        let x = Tensor::full(vec![1, 2], 203.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 203.0);
    }

    #[test]
    fn test_fused_stress_case_204() {
        let x = Tensor::full(vec![1, 2], 204.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 204.0);
    }

    #[test]
    fn test_fused_stress_case_205() {
        let x = Tensor::full(vec![1, 2], 205.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 205.0);
    }

    #[test]
    fn test_fused_stress_case_206() {
        let x = Tensor::full(vec![1, 2], 206.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 206.0);
    }

    #[test]
    fn test_fused_stress_case_207() {
        let x = Tensor::full(vec![1, 2], 207.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 207.0);
    }

    #[test]
    fn test_fused_stress_case_208() {
        let x = Tensor::full(vec![1, 2], 208.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 208.0);
    }

    #[test]
    fn test_fused_stress_case_209() {
        let x = Tensor::full(vec![1, 2], 209.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 209.0);
    }

    #[test]
    fn test_fused_stress_case_210() {
        let x = Tensor::full(vec![1, 2], 210.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 210.0);
    }

    #[test]
    fn test_fused_stress_case_211() {
        let x = Tensor::full(vec![1, 2], 211.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 211.0);
    }

    #[test]
    fn test_fused_stress_case_212() {
        let x = Tensor::full(vec![1, 2], 212.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 212.0);
    }

    #[test]
    fn test_fused_stress_case_213() {
        let x = Tensor::full(vec![1, 2], 213.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 213.0);
    }

    #[test]
    fn test_fused_stress_case_214() {
        let x = Tensor::full(vec![1, 2], 214.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 214.0);
    }

    #[test]
    fn test_fused_stress_case_215() {
        let x = Tensor::full(vec![1, 2], 215.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 215.0);
    }

    #[test]
    fn test_fused_stress_case_216() {
        let x = Tensor::full(vec![1, 2], 216.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 216.0);
    }

    #[test]
    fn test_fused_stress_case_217() {
        let x = Tensor::full(vec![1, 2], 217.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 217.0);
    }

    #[test]
    fn test_fused_stress_case_218() {
        let x = Tensor::full(vec![1, 2], 218.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 218.0);
    }

    #[test]
    fn test_fused_stress_case_219() {
        let x = Tensor::full(vec![1, 2], 219.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 219.0);
    }

    #[test]
    fn test_fused_stress_case_220() {
        let x = Tensor::full(vec![1, 2], 220.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 220.0);
    }

    #[test]
    fn test_fused_stress_case_221() {
        let x = Tensor::full(vec![1, 2], 221.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 221.0);
    }

    #[test]
    fn test_fused_stress_case_222() {
        let x = Tensor::full(vec![1, 2], 222.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 222.0);
    }

    #[test]
    fn test_fused_stress_case_223() {
        let x = Tensor::full(vec![1, 2], 223.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 223.0);
    }

    #[test]
    fn test_fused_stress_case_224() {
        let x = Tensor::full(vec![1, 2], 224.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 224.0);
    }

    #[test]
    fn test_fused_stress_case_225() {
        let x = Tensor::full(vec![1, 2], 225.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 225.0);
    }

    #[test]
    fn test_fused_stress_case_226() {
        let x = Tensor::full(vec![1, 2], 226.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 226.0);
    }

    #[test]
    fn test_fused_stress_case_227() {
        let x = Tensor::full(vec![1, 2], 227.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 227.0);
    }

    #[test]
    fn test_fused_stress_case_228() {
        let x = Tensor::full(vec![1, 2], 228.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 228.0);
    }

    #[test]
    fn test_fused_stress_case_229() {
        let x = Tensor::full(vec![1, 2], 229.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 229.0);
    }

    #[test]
    fn test_fused_stress_case_230() {
        let x = Tensor::full(vec![1, 2], 230.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 230.0);
    }

    #[test]
    fn test_fused_stress_case_231() {
        let x = Tensor::full(vec![1, 2], 231.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 231.0);
    }

    #[test]
    fn test_fused_stress_case_232() {
        let x = Tensor::full(vec![1, 2], 232.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 232.0);
    }

    #[test]
    fn test_fused_stress_case_233() {
        let x = Tensor::full(vec![1, 2], 233.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 233.0);
    }

    #[test]
    fn test_fused_stress_case_234() {
        let x = Tensor::full(vec![1, 2], 234.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 234.0);
    }

    #[test]
    fn test_fused_stress_case_235() {
        let x = Tensor::full(vec![1, 2], 235.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 235.0);
    }

    #[test]
    fn test_fused_stress_case_236() {
        let x = Tensor::full(vec![1, 2], 236.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 236.0);
    }

    #[test]
    fn test_fused_stress_case_237() {
        let x = Tensor::full(vec![1, 2], 237.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 237.0);
    }

    #[test]
    fn test_fused_stress_case_238() {
        let x = Tensor::full(vec![1, 2], 238.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 238.0);
    }

    #[test]
    fn test_fused_stress_case_239() {
        let x = Tensor::full(vec![1, 2], 239.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 239.0);
    }

    #[test]
    fn test_fused_stress_case_240() {
        let x = Tensor::full(vec![1, 2], 240.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 240.0);
    }

    #[test]
    fn test_fused_stress_case_241() {
        let x = Tensor::full(vec![1, 2], 241.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 241.0);
    }

    #[test]
    fn test_fused_stress_case_242() {
        let x = Tensor::full(vec![1, 2], 242.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 242.0);
    }

    #[test]
    fn test_fused_stress_case_243() {
        let x = Tensor::full(vec![1, 2], 243.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 243.0);
    }

    #[test]
    fn test_fused_stress_case_244() {
        let x = Tensor::full(vec![1, 2], 244.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 244.0);
    }

    #[test]
    fn test_fused_stress_case_245() {
        let x = Tensor::full(vec![1, 2], 245.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 245.0);
    }

    #[test]
    fn test_fused_stress_case_246() {
        let x = Tensor::full(vec![1, 2], 246.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 246.0);
    }

    #[test]
    fn test_fused_stress_case_247() {
        let x = Tensor::full(vec![1, 2], 247.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 247.0);
    }

    #[test]
    fn test_fused_stress_case_248() {
        let x = Tensor::full(vec![1, 2], 248.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 248.0);
    }

    #[test]
    fn test_fused_stress_case_249() {
        let x = Tensor::full(vec![1, 2], 249.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 249.0);
    }

    #[test]
    fn test_fused_stress_case_250() {
        let x = Tensor::full(vec![1, 2], 250.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 250.0);
    }

    #[test]
    fn test_fused_stress_case_251() {
        let x = Tensor::full(vec![1, 2], 251.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 251.0);
    }

    #[test]
    fn test_fused_stress_case_252() {
        let x = Tensor::full(vec![1, 2], 252.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 252.0);
    }

    #[test]
    fn test_fused_stress_case_253() {
        let x = Tensor::full(vec![1, 2], 253.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 253.0);
    }

    #[test]
    fn test_fused_stress_case_254() {
        let x = Tensor::full(vec![1, 2], 254.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 254.0);
    }

    #[test]
    fn test_fused_stress_case_255() {
        let x = Tensor::full(vec![1, 2], 255.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 255.0);
    }

    #[test]
    fn test_fused_stress_case_256() {
        let x = Tensor::full(vec![1, 2], 256.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 256.0);
    }

    #[test]
    fn test_fused_stress_case_257() {
        let x = Tensor::full(vec![1, 2], 257.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 257.0);
    }

    #[test]
    fn test_fused_stress_case_258() {
        let x = Tensor::full(vec![1, 2], 258.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 258.0);
    }

    #[test]
    fn test_fused_stress_case_259() {
        let x = Tensor::full(vec![1, 2], 259.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 259.0);
    }

    #[test]
    fn test_fused_stress_case_260() {
        let x = Tensor::full(vec![1, 2], 260.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 260.0);
    }

    #[test]
    fn test_fused_stress_case_261() {
        let x = Tensor::full(vec![1, 2], 261.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 261.0);
    }

    #[test]
    fn test_fused_stress_case_262() {
        let x = Tensor::full(vec![1, 2], 262.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 262.0);
    }

    #[test]
    fn test_fused_stress_case_263() {
        let x = Tensor::full(vec![1, 2], 263.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 263.0);
    }

    #[test]
    fn test_fused_stress_case_264() {
        let x = Tensor::full(vec![1, 2], 264.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 264.0);
    }

    #[test]
    fn test_fused_stress_case_265() {
        let x = Tensor::full(vec![1, 2], 265.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 265.0);
    }

    #[test]
    fn test_fused_stress_case_266() {
        let x = Tensor::full(vec![1, 2], 266.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 266.0);
    }

    #[test]
    fn test_fused_stress_case_267() {
        let x = Tensor::full(vec![1, 2], 267.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 267.0);
    }

    #[test]
    fn test_fused_stress_case_268() {
        let x = Tensor::full(vec![1, 2], 268.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 268.0);
    }

    #[test]
    fn test_fused_stress_case_269() {
        let x = Tensor::full(vec![1, 2], 269.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 269.0);
    }

    #[test]
    fn test_fused_stress_case_270() {
        let x = Tensor::full(vec![1, 2], 270.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 270.0);
    }

    #[test]
    fn test_fused_stress_case_271() {
        let x = Tensor::full(vec![1, 2], 271.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 271.0);
    }

    #[test]
    fn test_fused_stress_case_272() {
        let x = Tensor::full(vec![1, 2], 272.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 272.0);
    }

    #[test]
    fn test_fused_stress_case_273() {
        let x = Tensor::full(vec![1, 2], 273.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 273.0);
    }

    #[test]
    fn test_fused_stress_case_274() {
        let x = Tensor::full(vec![1, 2], 274.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 274.0);
    }

    #[test]
    fn test_fused_stress_case_275() {
        let x = Tensor::full(vec![1, 2], 275.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 275.0);
    }

    #[test]
    fn test_fused_stress_case_276() {
        let x = Tensor::full(vec![1, 2], 276.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 276.0);
    }

    #[test]
    fn test_fused_stress_case_277() {
        let x = Tensor::full(vec![1, 2], 277.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 277.0);
    }

    #[test]
    fn test_fused_stress_case_278() {
        let x = Tensor::full(vec![1, 2], 278.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 278.0);
    }

    #[test]
    fn test_fused_stress_case_279() {
        let x = Tensor::full(vec![1, 2], 279.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 279.0);
    }

    #[test]
    fn test_fused_stress_case_280() {
        let x = Tensor::full(vec![1, 2], 280.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 280.0);
    }

    #[test]
    fn test_fused_stress_case_281() {
        let x = Tensor::full(vec![1, 2], 281.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 281.0);
    }

    #[test]
    fn test_fused_stress_case_282() {
        let x = Tensor::full(vec![1, 2], 282.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 282.0);
    }

    #[test]
    fn test_fused_stress_case_283() {
        let x = Tensor::full(vec![1, 2], 283.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 283.0);
    }

    #[test]
    fn test_fused_stress_case_284() {
        let x = Tensor::full(vec![1, 2], 284.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 284.0);
    }

    #[test]
    fn test_fused_stress_case_285() {
        let x = Tensor::full(vec![1, 2], 285.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 285.0);
    }

    #[test]
    fn test_fused_stress_case_286() {
        let x = Tensor::full(vec![1, 2], 286.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 286.0);
    }

    #[test]
    fn test_fused_stress_case_287() {
        let x = Tensor::full(vec![1, 2], 287.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 287.0);
    }

    #[test]
    fn test_fused_stress_case_288() {
        let x = Tensor::full(vec![1, 2], 288.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 288.0);
    }

    #[test]
    fn test_fused_stress_case_289() {
        let x = Tensor::full(vec![1, 2], 289.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 289.0);
    }

    #[test]
    fn test_fused_stress_case_290() {
        let x = Tensor::full(vec![1, 2], 290.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 290.0);
    }

    #[test]
    fn test_fused_stress_case_291() {
        let x = Tensor::full(vec![1, 2], 291.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 291.0);
    }

    #[test]
    fn test_fused_stress_case_292() {
        let x = Tensor::full(vec![1, 2], 292.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 292.0);
    }

    #[test]
    fn test_fused_stress_case_293() {
        let x = Tensor::full(vec![1, 2], 293.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 293.0);
    }

    #[test]
    fn test_fused_stress_case_294() {
        let x = Tensor::full(vec![1, 2], 294.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 294.0);
    }

    #[test]
    fn test_fused_stress_case_295() {
        let x = Tensor::full(vec![1, 2], 295.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 295.0);
    }

    #[test]
    fn test_fused_stress_case_296() {
        let x = Tensor::full(vec![1, 2], 296.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 296.0);
    }

    #[test]
    fn test_fused_stress_case_297() {
        let x = Tensor::full(vec![1, 2], 297.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 297.0);
    }

    #[test]
    fn test_fused_stress_case_298() {
        let x = Tensor::full(vec![1, 2], 298.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 298.0);
    }

    #[test]
    fn test_fused_stress_case_299() {
        let x = Tensor::full(vec![1, 2], 299.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 299.0);
    }

    #[test]
    fn test_fused_stress_case_300() {
        let x = Tensor::full(vec![1, 2], 300.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 300.0);
    }

    #[test]
    fn test_fused_stress_case_301() {
        let x = Tensor::full(vec![1, 2], 301.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 301.0);
    }

    #[test]
    fn test_fused_stress_case_302() {
        let x = Tensor::full(vec![1, 2], 302.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 302.0);
    }

    #[test]
    fn test_fused_stress_case_303() {
        let x = Tensor::full(vec![1, 2], 303.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 303.0);
    }

    #[test]
    fn test_fused_stress_case_304() {
        let x = Tensor::full(vec![1, 2], 304.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 304.0);
    }

    #[test]
    fn test_fused_stress_case_305() {
        let x = Tensor::full(vec![1, 2], 305.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 305.0);
    }

    #[test]
    fn test_fused_stress_case_306() {
        let x = Tensor::full(vec![1, 2], 306.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 306.0);
    }

    #[test]
    fn test_fused_stress_case_307() {
        let x = Tensor::full(vec![1, 2], 307.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 307.0);
    }

    #[test]
    fn test_fused_stress_case_308() {
        let x = Tensor::full(vec![1, 2], 308.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 308.0);
    }

    #[test]
    fn test_fused_stress_case_309() {
        let x = Tensor::full(vec![1, 2], 309.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 309.0);
    }

    #[test]
    fn test_fused_stress_case_310() {
        let x = Tensor::full(vec![1, 2], 310.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 310.0);
    }

    #[test]
    fn test_fused_stress_case_311() {
        let x = Tensor::full(vec![1, 2], 311.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 311.0);
    }

    #[test]
    fn test_fused_stress_case_312() {
        let x = Tensor::full(vec![1, 2], 312.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 312.0);
    }

    #[test]
    fn test_fused_stress_case_313() {
        let x = Tensor::full(vec![1, 2], 313.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 313.0);
    }

    #[test]
    fn test_fused_stress_case_314() {
        let x = Tensor::full(vec![1, 2], 314.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 314.0);
    }

    #[test]
    fn test_fused_stress_case_315() {
        let x = Tensor::full(vec![1, 2], 315.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 315.0);
    }

    #[test]
    fn test_fused_stress_case_316() {
        let x = Tensor::full(vec![1, 2], 316.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 316.0);
    }

    #[test]
    fn test_fused_stress_case_317() {
        let x = Tensor::full(vec![1, 2], 317.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 317.0);
    }

    #[test]
    fn test_fused_stress_case_318() {
        let x = Tensor::full(vec![1, 2], 318.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 318.0);
    }

    #[test]
    fn test_fused_stress_case_319() {
        let x = Tensor::full(vec![1, 2], 319.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 319.0);
    }

    #[test]
    fn test_fused_stress_case_320() {
        let x = Tensor::full(vec![1, 2], 320.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 320.0);
    }

    #[test]
    fn test_fused_stress_case_321() {
        let x = Tensor::full(vec![1, 2], 321.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 321.0);
    }

    #[test]
    fn test_fused_stress_case_322() {
        let x = Tensor::full(vec![1, 2], 322.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 322.0);
    }

    #[test]
    fn test_fused_stress_case_323() {
        let x = Tensor::full(vec![1, 2], 323.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 323.0);
    }

    #[test]
    fn test_fused_stress_case_324() {
        let x = Tensor::full(vec![1, 2], 324.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 324.0);
    }

    #[test]
    fn test_fused_stress_case_325() {
        let x = Tensor::full(vec![1, 2], 325.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 325.0);
    }

    #[test]
    fn test_fused_stress_case_326() {
        let x = Tensor::full(vec![1, 2], 326.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 326.0);
    }

    #[test]
    fn test_fused_stress_case_327() {
        let x = Tensor::full(vec![1, 2], 327.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 327.0);
    }

    #[test]
    fn test_fused_stress_case_328() {
        let x = Tensor::full(vec![1, 2], 328.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 328.0);
    }

    #[test]
    fn test_fused_stress_case_329() {
        let x = Tensor::full(vec![1, 2], 329.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 329.0);
    }

    #[test]
    fn test_fused_stress_case_330() {
        let x = Tensor::full(vec![1, 2], 330.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 330.0);
    }

    #[test]
    fn test_fused_stress_case_331() {
        let x = Tensor::full(vec![1, 2], 331.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 331.0);
    }

    #[test]
    fn test_fused_stress_case_332() {
        let x = Tensor::full(vec![1, 2], 332.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 332.0);
    }

    #[test]
    fn test_fused_stress_case_333() {
        let x = Tensor::full(vec![1, 2], 333.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 333.0);
    }

    #[test]
    fn test_fused_stress_case_334() {
        let x = Tensor::full(vec![1, 2], 334.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 334.0);
    }

    #[test]
    fn test_fused_stress_case_335() {
        let x = Tensor::full(vec![1, 2], 335.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 335.0);
    }

    #[test]
    fn test_fused_stress_case_336() {
        let x = Tensor::full(vec![1, 2], 336.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 336.0);
    }

    #[test]
    fn test_fused_stress_case_337() {
        let x = Tensor::full(vec![1, 2], 337.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 337.0);
    }

    #[test]
    fn test_fused_stress_case_338() {
        let x = Tensor::full(vec![1, 2], 338.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 338.0);
    }

    #[test]
    fn test_fused_stress_case_339() {
        let x = Tensor::full(vec![1, 2], 339.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 339.0);
    }

    #[test]
    fn test_fused_stress_case_340() {
        let x = Tensor::full(vec![1, 2], 340.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 340.0);
    }

    #[test]
    fn test_fused_stress_case_341() {
        let x = Tensor::full(vec![1, 2], 341.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 341.0);
    }

    #[test]
    fn test_fused_stress_case_342() {
        let x = Tensor::full(vec![1, 2], 342.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 342.0);
    }

    #[test]
    fn test_fused_stress_case_343() {
        let x = Tensor::full(vec![1, 2], 343.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 343.0);
    }

    #[test]
    fn test_fused_stress_case_344() {
        let x = Tensor::full(vec![1, 2], 344.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 344.0);
    }

    #[test]
    fn test_fused_stress_case_345() {
        let x = Tensor::full(vec![1, 2], 345.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 345.0);
    }

    #[test]
    fn test_fused_stress_case_346() {
        let x = Tensor::full(vec![1, 2], 346.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 346.0);
    }

    #[test]
    fn test_fused_stress_case_347() {
        let x = Tensor::full(vec![1, 2], 347.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 347.0);
    }

    #[test]
    fn test_fused_stress_case_348() {
        let x = Tensor::full(vec![1, 2], 348.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 348.0);
    }

    #[test]
    fn test_fused_stress_case_349() {
        let x = Tensor::full(vec![1, 2], 349.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 349.0);
    }

    #[test]
    fn test_fused_stress_case_350() {
        let x = Tensor::full(vec![1, 2], 350.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 350.0);
    }

    #[test]
    fn test_fused_stress_case_351() {
        let x = Tensor::full(vec![1, 2], 351.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 351.0);
    }

    #[test]
    fn test_fused_stress_case_352() {
        let x = Tensor::full(vec![1, 2], 352.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 352.0);
    }

    #[test]
    fn test_fused_stress_case_353() {
        let x = Tensor::full(vec![1, 2], 353.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 353.0);
    }

    #[test]
    fn test_fused_stress_case_354() {
        let x = Tensor::full(vec![1, 2], 354.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 354.0);
    }

    #[test]
    fn test_fused_stress_case_355() {
        let x = Tensor::full(vec![1, 2], 355.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 355.0);
    }

    #[test]
    fn test_fused_stress_case_356() {
        let x = Tensor::full(vec![1, 2], 356.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 356.0);
    }

    #[test]
    fn test_fused_stress_case_357() {
        let x = Tensor::full(vec![1, 2], 357.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 357.0);
    }

    #[test]
    fn test_fused_stress_case_358() {
        let x = Tensor::full(vec![1, 2], 358.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 358.0);
    }

    #[test]
    fn test_fused_stress_case_359() {
        let x = Tensor::full(vec![1, 2], 359.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 359.0);
    }

    #[test]
    fn test_fused_stress_case_360() {
        let x = Tensor::full(vec![1, 2], 360.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 360.0);
    }

    #[test]
    fn test_fused_stress_case_361() {
        let x = Tensor::full(vec![1, 2], 361.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 361.0);
    }

    #[test]
    fn test_fused_stress_case_362() {
        let x = Tensor::full(vec![1, 2], 362.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 362.0);
    }

    #[test]
    fn test_fused_stress_case_363() {
        let x = Tensor::full(vec![1, 2], 363.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 363.0);
    }

    #[test]
    fn test_fused_stress_case_364() {
        let x = Tensor::full(vec![1, 2], 364.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 364.0);
    }

    #[test]
    fn test_fused_stress_case_365() {
        let x = Tensor::full(vec![1, 2], 365.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 365.0);
    }

    #[test]
    fn test_fused_stress_case_366() {
        let x = Tensor::full(vec![1, 2], 366.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 366.0);
    }

    #[test]
    fn test_fused_stress_case_367() {
        let x = Tensor::full(vec![1, 2], 367.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 367.0);
    }

    #[test]
    fn test_fused_stress_case_368() {
        let x = Tensor::full(vec![1, 2], 368.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 368.0);
    }

    #[test]
    fn test_fused_stress_case_369() {
        let x = Tensor::full(vec![1, 2], 369.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 369.0);
    }

    #[test]
    fn test_fused_stress_case_370() {
        let x = Tensor::full(vec![1, 2], 370.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 370.0);
    }

    #[test]
    fn test_fused_stress_case_371() {
        let x = Tensor::full(vec![1, 2], 371.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 371.0);
    }

    #[test]
    fn test_fused_stress_case_372() {
        let x = Tensor::full(vec![1, 2], 372.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 372.0);
    }

    #[test]
    fn test_fused_stress_case_373() {
        let x = Tensor::full(vec![1, 2], 373.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 373.0);
    }

    #[test]
    fn test_fused_stress_case_374() {
        let x = Tensor::full(vec![1, 2], 374.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 374.0);
    }

    #[test]
    fn test_fused_stress_case_375() {
        let x = Tensor::full(vec![1, 2], 375.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 375.0);
    }

    #[test]
    fn test_fused_stress_case_376() {
        let x = Tensor::full(vec![1, 2], 376.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 376.0);
    }

    #[test]
    fn test_fused_stress_case_377() {
        let x = Tensor::full(vec![1, 2], 377.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 377.0);
    }

    #[test]
    fn test_fused_stress_case_378() {
        let x = Tensor::full(vec![1, 2], 378.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 378.0);
    }

    #[test]
    fn test_fused_stress_case_379() {
        let x = Tensor::full(vec![1, 2], 379.0);
        let w = Tensor::eye(2);
        let out = fused_linear(&x, &w, None);
        assert_eq!(out.get_2d(0, 0), 379.0);
    }
}
