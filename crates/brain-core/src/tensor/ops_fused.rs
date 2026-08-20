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

/// Fused Residual Add + RMSNorm: out = RMSNorm(x + residual, weight, eps) in a single memory pass.
pub fn fused_add_rmsnorm(
    x: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> (Tensor, Tensor) {
    assert_eq!(x.shape(), residual.shape());
    assert_eq!(x.ndim(), 2);
    let (rows, cols) = (x.shape()[0], x.shape()[1]);
    let mut sum_out = Tensor::zeros(vec![rows, cols]);
    let mut norm_out = Tensor::zeros(vec![rows, cols]);

    for r in 0..rows {
        let mut sum_sq = 0.0;
        for c in 0..cols {
            let sum_val = x.get_2d(r, c) + residual.get_2d(r, c);
            sum_out.set_2d(r, c, sum_val);
            sum_sq += sum_val * sum_val;
        }
        let rms = 1.0 / ((sum_sq / (cols as f64)) + eps).sqrt();
        for c in 0..cols {
            let w = weight.get(c);
            norm_out.set_2d(r, c, sum_out.get_2d(r, c) * rms * w);
        }
    }
    (norm_out, sum_out)
}

/// Fused SwiGLU activation: out = (gate * sigmoid(gate)) * up.
pub fn fused_swiglu(gate: &Tensor, up: &Tensor) -> Tensor {
    assert_eq!(gate.shape(), up.shape());
    let numel = gate.numel();
    let g_data = gate.data();
    let u_data = up.data();
    let mut out = Vec::with_capacity(numel);

    for i in 0..numel {
        let g = g_data[i];
        let sig = 1.0 / (1.0 + (-g).exp());
        let swish = g * sig;
        out.push(swish * u_data[i]);
    }

    Tensor::new(out, gate.shape().to_vec())
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
    fn test_fused_ops_equivalence() {
        let a = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let b = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let c = Tensor::from_slice(&[0.5, 0.5], vec![2]);
        let fma = fused_fma(&a, &b, &c);
        assert_eq!(fma.to_vec(), vec![3.5, 8.5]);
    }

    #[test]
    fn test_fused_add_rmsnorm_and_swiglu() {
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let res = Tensor::from_slice(&[0.5, 0.5], vec![1, 2]);
        let w = Tensor::from_slice(&[1.0, 1.0], vec![2]);
        let (norm, sum) = fused_add_rmsnorm(&x, &res, &w, 1e-5);
        assert_eq!(sum.data(), &[1.5, 2.5]);
        assert_eq!(norm.shape(), &[1, 2]);

        let gate = Tensor::from_slice(&[0.0, 1.0], vec![2]);
        let up = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        let swi = fused_swiglu(&gate, &up);
        assert_eq!(swi.shape(), &[2]);
        // swish(0) = 0 * 0.5 = 0; 0 * 2 = 0
        assert!((swi.data()[0] - 0.0).abs() < 1e-6);
        // swish(1) = 1 / (1 + exp(-1)) ~= 0.73105857863; * 3 ~= 2.1931757
        assert!((swi.data()[1] - (1.0 / (1.0 + (-1.0f64).exp()) * 3.0)).abs() < 1e-5);
    }
}
