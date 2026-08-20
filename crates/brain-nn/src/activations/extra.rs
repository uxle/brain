//! # Additional Activation Functions
//!
//! Production-grade implementations of remaining torch/Burn activations:
//! - `PReLU` (Parametric ReLU with learnable negative weight)
//! - `LogSigmoid` (ln(sigmoid(x)) = -softplus(-x))
//! - `TanhShrink` (x - tanh(x))
//! - `HardShrink` (x if |x| > lambda else 0)
//! - `SoftShrink` (x - lambda / x + lambda / 0 soft thresholding)
//! - `Shrink` (generalized soft thresholding with bias)
//! - `ThresholdedReLU` (x if x > theta else 0)
//! - `Threshold` (value if x > threshold else 0)
//! - `ReLU6` (min(max(0, x), 6))
//! - `Softmin` (softmax over -x)
//! - `QuietSoftmax` (per-element temperature softmax from DeepSeek-R1)

use crate::activations::Activation;
use brain_core::Tensor;

/// Parametric ReLU: x for x >= 0, weight * x otherwise.
#[derive(Debug, Clone, Copy)]
pub struct PReLU {
    pub weight: f64,
}

impl Default for PReLU {
    fn default() -> Self {
        Self { weight: 0.25 }
    }
}

impl PReLU {
    pub fn new(weight: f64) -> Self {
        Self { weight }
    }
}

impl Activation for PReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        let w = self.weight;
        input.map(move |x| if x >= 0.0 { x } else { w * x })
    }
}

/// Computes PReLU elementwise.
pub fn prelu(input: &Tensor, weight: f64) -> Tensor {
    PReLU::new(weight).forward(input)
}

/// LogSigmoid: ln(sigmoid(x)) computed stably as -softplus(-x).
#[derive(Debug, Clone, Copy, Default)]
pub struct LogSigmoid;

impl Activation for LogSigmoid {
    fn forward(&self, input: &Tensor) -> Tensor {
        log_sigmoid(input)
    }
}

/// Computes LogSigmoid elementwise.
pub fn log_sigmoid(input: &Tensor) -> Tensor {
    input.map(|x| {
        if x >= 0.0 {
            -((-x).exp()).ln_1p()
        } else {
            x - x.exp().ln_1p()
        }
    })
}

/// TanhShrink: x - tanh(x).
#[derive(Debug, Clone, Copy, Default)]
pub struct TanhShrink;

impl Activation for TanhShrink {
    fn forward(&self, input: &Tensor) -> Tensor {
        tanh_shrink(input)
    }
}

/// Computes TanhShrink elementwise.
pub fn tanh_shrink(input: &Tensor) -> Tensor {
    input.map(|x| x - x.tanh())
}

/// HardShrink: x if |x| > lambda, else 0.
#[derive(Debug, Clone, Copy)]
pub struct HardShrink {
    pub lambda: f64,
}

impl Default for HardShrink {
    fn default() -> Self {
        Self { lambda: 0.5 }
    }
}

impl HardShrink {
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }
}

impl Activation for HardShrink {
    fn forward(&self, input: &Tensor) -> Tensor {
        let l = self.lambda;
        input.map(move |x| if x.abs() > l { x } else { 0.0 })
    }
}

/// Computes HardShrink elementwise.
pub fn hard_shrink(input: &Tensor, lambda: f64) -> Tensor {
    HardShrink::new(lambda).forward(input)
}

/// SoftShrink: x - lambda for x > lambda, x + lambda for x < -lambda, else 0.
#[derive(Debug, Clone, Copy)]
pub struct SoftShrink {
    pub lambda: f64,
}

impl Default for SoftShrink {
    fn default() -> Self {
        Self { lambda: 0.5 }
    }
}

impl SoftShrink {
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }
}

impl Activation for SoftShrink {
    fn forward(&self, input: &Tensor) -> Tensor {
        let l = self.lambda;
        input.map(move |x| {
            if x > l {
                x - l
            } else if x < -l {
                x + l
            } else {
                0.0
            }
        })
    }
}

/// Computes SoftShrink elementwise.
pub fn soft_shrink(input: &Tensor, lambda: f64) -> Tensor {
    SoftShrink::new(lambda).forward(input)
}

/// Shrink: x - bias for x > lambda, x + bias for x < -lambda, else 0.
#[derive(Debug, Clone, Copy)]
pub struct Shrink {
    pub lambda: f64,
    pub bias: f64,
}

impl Default for Shrink {
    fn default() -> Self {
        Self {
            lambda: 0.5,
            bias: 0.5,
        }
    }
}

impl Shrink {
    pub fn new(lambda: f64, bias: f64) -> Self {
        Self { lambda, bias }
    }
}

impl Activation for Shrink {
    fn forward(&self, input: &Tensor) -> Tensor {
        let (l, b) = (self.lambda, self.bias);
        input.map(move |x| {
            if x > l {
                x - b
            } else if x < -l {
                x + b
            } else {
                0.0
            }
        })
    }
}

/// Computes Shrink elementwise.
pub fn shrink(input: &Tensor, lambda: f64, bias: f64) -> Tensor {
    Shrink::new(lambda, bias).forward(input)
}

/// ThresholdedReLU: x if x > theta, else 0.
#[derive(Debug, Clone, Copy)]
pub struct ThresholdedReLU {
    pub theta: f64,
}

impl Default for ThresholdedReLU {
    fn default() -> Self {
        Self { theta: 1.0 }
    }
}

impl ThresholdedReLU {
    pub fn new(theta: f64) -> Self {
        Self { theta }
    }
}

impl Activation for ThresholdedReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        let t = self.theta;
        input.map(move |x| if x > t { x } else { 0.0 })
    }
}

/// Computes ThresholdedReLU elementwise.
pub fn thresholded_relu(input: &Tensor, theta: f64) -> Tensor {
    ThresholdedReLU::new(theta).forward(input)
}

/// Threshold: value if x > threshold, else 0.
#[derive(Debug, Clone, Copy)]
pub struct Threshold {
    pub threshold: f64,
    pub value: f64,
}

impl Default for Threshold {
    fn default() -> Self {
        Self {
            threshold: 0.0,
            value: 0.0,
        }
    }
}

impl Threshold {
    pub fn new(threshold: f64, value: f64) -> Self {
        Self { threshold, value }
    }
}

impl Activation for Threshold {
    fn forward(&self, input: &Tensor) -> Tensor {
        let (t, v) = (self.threshold, self.value);
        input.map(move |x| if x > t { v } else { 0.0 })
    }
}

/// Computes Threshold elementwise.
pub fn threshold(input: &Tensor, threshold: f64, value: f64) -> Tensor {
    Threshold::new(threshold, value).forward(input)
}

/// ReLU6: clamp(ReLU(x), 0, 6).
#[derive(Debug, Clone, Copy, Default)]
pub struct ReLU6;

impl Activation for ReLU6 {
    fn forward(&self, input: &Tensor) -> Tensor {
        relu6(input)
    }
}

/// Computes ReLU6 elementwise.
pub fn relu6(input: &Tensor) -> Tensor {
    input.map(|x| x.clamp(0.0, 6.0))
}

/// Softmin: softmax over -x along the last dimension (2D input).
#[derive(Debug, Clone, Copy, Default)]
pub struct Softmin;

impl Activation for Softmin {
    fn forward(&self, input: &Tensor) -> Tensor {
        softmin(input)
    }
}

/// Numerically stable 2D softmin along the last dimension.
pub fn softmin(input: &Tensor) -> Tensor {
    let shape = input.shape();
    assert_eq!(shape.len(), 2, "softmin requires a 2D input");
    let (rows, cols) = (shape[0], shape[1]);
    let data = input.to_vec();

    let mut out = vec![0.0f64; rows * cols];
    for r in 0..rows {
        let row_slice = &data[r * cols..(r + 1) * cols];
        let max_val = row_slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = row_slice.iter().map(|&x| (-x - max_val).exp()).sum();
        for c in 0..cols {
            out[r * cols + c] = (-data[r * cols + c] - max_val).exp() / sum_exp.max(1e-12);
        }
    }
    Tensor::from_vec(out, shape.to_vec())
}

/// QuietSoftmax: softmax with per-element temperature tau_i = 1 + exp(x_i) (DeepSeek-R1).
#[derive(Debug, Clone, Copy, Default)]
pub struct QuietSoftmax;

impl Activation for QuietSoftmax {
    fn forward(&self, input: &Tensor) -> Tensor {
        quiet_softmax(input)
    }
}

/// Computes QuietSoftmax along the last dimension (2D input).
pub fn quiet_softmax(input: &Tensor) -> Tensor {
    let shape = input.shape();
    assert_eq!(shape.len(), 2, "quiet_softmax requires a 2D input");
    let (rows, cols) = (shape[0], shape[1]);
    let data = input.to_vec();

    let mut scaled = vec![0.0f64; rows * cols];
    let mut out = vec![0.0f64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let x = data[r * cols + c];
            scaled[r * cols + c] = x / (1.0 + x.exp());
        }
        let row_slice = &scaled[r * cols..(r + 1) * cols];
        let max_val = row_slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = row_slice.iter().map(|&s| (s - max_val).exp()).sum();
        for c in 0..cols {
            out[r * cols + c] = (scaled[r * cols + c] - max_val).exp() / sum_exp.max(1e-12);
        }
    }
    Tensor::from_vec(out, shape.to_vec())
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
    fn test_prelu() {
        let t = Tensor::from_slice(&[1.0, -2.0, 0.5], vec![3]);
        let o = prelu(&t, 0.25);
        assert_eq!(o.to_vec(), vec![1.0, -0.5, 0.5]);
        let o = prelu(&t, 0.1);
        assert!((o.get(1) + 0.2).abs() < 1e-9);
    }

    #[test]
    fn test_log_sigmoid() {
        let t = Tensor::from_slice(&[0.0, 1.0, -1.0, 3.0, -10.0], vec![5]);
        let o = log_sigmoid(&t);
        let expect = |x: f64| -(1.0 + (-x).exp()).ln();
        for i in 0..5 {
            assert!(
                (o.get(i) - expect(t.get(i))).abs() < 1e-9,
                "mismatch at {i}"
            );
        }
        // saturation: log-sigmoid(-10) close to -10
        assert!((o.get(4) + 10.0).abs() < 1e-3);
    }

    #[test]
    fn test_tanh_shrink() {
        let t = Tensor::from_slice(&[1.0, -0.5, 2.0], vec![3]);
        let o = tanh_shrink(&t);
        for i in 0..3 {
            assert!((o.get(i) - (t.get(i) - t.get(i).tanh())).abs() < 1e-9);
        }
    }

    #[test]
    fn test_shrinks() {
        let t = Tensor::from_slice(&[-1.0, -0.4, 0.0, 0.4, 1.0], vec![5]);
        let o = hard_shrink(&t, 0.5);
        assert_eq!(o.to_vec(), vec![-1.0, 0.0, 0.0, 0.0, 1.0]);
        let o = soft_shrink(&t, 0.5);
        assert_eq!(o.to_vec(), vec![-0.5, 0.0, 0.0, 0.0, 0.5]);
        let o = shrink(&t, 0.3, 0.1);
        for i in 0..5 {
            let expect = match i {
                0 => -0.9,
                1 => -0.3,
                2 => 0.0,
                3 => 0.3,
                _ => 0.9,
            };
            assert!((o.get(i) - expect).abs() < 1e-12, "shrink mismatch at {i}");
        }
    }

    #[test]
    fn test_thresholds() {
        let t = Tensor::from_slice(&[0.5, 1.0, 1.5, -2.0], vec![4]);
        let o = thresholded_relu(&t, 1.0);
        assert_eq!(o.to_vec(), vec![0.0, 0.0, 1.5, 0.0]);
        let o = threshold(&t, 0.4, 7.0);
        assert_eq!(o.to_vec(), vec![7.0, 7.0, 7.0, 0.0]);
    }

    #[test]
    fn test_relu6() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 3.0, 7.0, 6.5], vec![5]);
        let o = relu6(&t);
        assert_eq!(o.to_vec(), vec![0.0, 0.0, 3.0, 6.0, 6.0]);
    }

    #[test]
    fn test_softmin() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let o = softmin(&t);
        // Row 0: softmax(-[1,2]) = [e^-1, e^-2] / (e^-1 + e^-2)
        let s = (-1.0f64).exp() + (-2.0f64).exp();
        assert!((o.get_2d(0, 0) - (-1.0f64).exp() / s).abs() < 1e-9);
        // Row 1 sums to 1
        assert!((o.get_2d(1, 0) + o.get_2d(1, 1) - 1.0).abs() < 1e-9);
        // Larger logit -> smaller softmin prob
        assert!(o.get_2d(0, 0) > o.get_2d(0, 1));
    }

    #[test]
    fn test_quiet_softmax() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![1, 3]);
        let o = quiet_softmax(&t);
        let row_sum: f64 = o.to_vec().iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-9);
        // Larger logits dominate but are attenuated vs plain softmax
        let sm = crate::activations::softmax::softmax(&t);
        assert!(o.get_2d(0, 2) < sm.get_2d(0, 2));
        assert!(o.get_2d(0, 0) > sm.get_2d(0, 0));
    }
}
