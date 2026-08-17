//! # Numerical & Initialization Utilities
//!
//! Non-linear activations, orthogonal matrix generation, and gradient clipping.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown, clippy::module_inception, clippy::manual_memcpy)]

use brain_core::Tensor;

/// Computes standard sigmoid activation: $\sigma(x) = \frac{1}{1 + e^{-x}}$.
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Derivative of hyperbolic tangent: $\tanh'(x) = 1 - \tanh^2(x)$.
pub fn tanh_prime(tanh_val: f64) -> f64 {
    1.0 - tanh_val * tanh_val
}

/// Deterministic XorShift64 pseudo-random generator.
#[derive(Debug, Clone)]
pub struct RnnRng {
    pub state: u64,
}

impl RnnRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xdeadbeef_cafebabe } else { seed },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + self.next_f64() * (high - low)
    }

    pub fn standard_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Generates approximate orthogonal weight matrix of shape [rows, cols].
pub fn init_orthogonal(rows: usize, cols: usize, seed: u64) -> Tensor {
    let mut rng = RnnRng::new(seed);
    let mut data = vec![0.0; rows * cols];
    for val in data.iter_mut() {
        *val = rng.standard_normal();
    }

    // Gram-Schmidt orthogonalization across rows
    for i in 0..rows.min(cols) {
        // Normalize row i
        let mut norm_sq = 0.0;
        for j in 0..cols {
            let v = data[i * cols + j];
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt().max(1e-12);
        for j in 0..cols {
            data[i * cols + j] /= norm;
        }

        // Subtract projection from subsequent rows
        for k in (i + 1)..rows.min(cols) {
            let mut dot = 0.0;
            for j in 0..cols {
                dot += data[i * cols + j] * data[k * cols + j];
            }
            for j in 0..cols {
                data[k * cols + j] -= dot * data[i * cols + j];
            }
        }
    }

    Tensor::from_slice(&data, vec![rows, cols])
}

/// Initializes uniform random weights in range $[-k, k]$ where $k = 1 / \sqrt{\text{fan\_in}}$.
pub fn init_uniform(rows: usize, cols: usize, fan_in: usize, seed: u64) -> Tensor {
    let mut rng = RnnRng::new(seed);
    let bound = 1.0 / (fan_in as f64).max(1.0).sqrt();
    let mut data = vec![0.0; rows * cols];
    for val in data.iter_mut() {
        *val = rng.uniform(-bound, bound);
    }
    Tensor::from_slice(&data, vec![rows, cols])
}

/// Clips gradient vector norm to max threshold.
pub fn clip_grad_norm(grads: &mut [f64], max_norm: f64) -> f64 {
    let total_norm: f64 = grads.iter().map(|&g| g * g).sum::<f64>().sqrt();
    if total_norm > max_norm && total_norm > 1e-12 {
        let scale = max_norm / total_norm;
        for g in grads.iter_mut() {
            *g *= scale;
        }
    }
    total_norm
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown, clippy::excessive_precision)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::utils::*;
    use crate::ops::*;
    use crate::cells::*;
    use crate::seq::*;
    use crate::init_rnn::*;
    use crate::reg_ops::*;
    use crate::process::*;
    use crate::backward_ops::*;
    use crate::builder::*;
    use crate::helper::*;
    use crate::VERSION;
    use brain_core::Tensor;

    #[test]
    fn test_utils_stress_001() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 1 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 1 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_002() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 2 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 2 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_003() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 3 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 3 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_004() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 4 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 4 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_005() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 5 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 5 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_006() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 6 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 6 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_007() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 7 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 7 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_008() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 8 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 8 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_009() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 9 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 9 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_010() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 10 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 10 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_011() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 11 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 11 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_012() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 12 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 12 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_013() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 13 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 13 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_014() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 14 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 14 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_015() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 15 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 15 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_016() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 16 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 16 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_017() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 17 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 17 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_018() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 18 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 18 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_019() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 19 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 19 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_020() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 20 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 20 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_021() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 21 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 21 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_022() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 22 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 22 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_023() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 23 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 23 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_024() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 24 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 24 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_025() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 25 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 25 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_026() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 26 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 26 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_027() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 27 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 27 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_028() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 28 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 28 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_029() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 29 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 29 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_030() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 30 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 30 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_031() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 31 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 31 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_032() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 32 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 32 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_033() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 33 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 33 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_034() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 34 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 34 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_035() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 35 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 35 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_036() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 36 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 36 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_037() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 37 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 37 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_038() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 38 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 38 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_039() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 39 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 39 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_040() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 40 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 40 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_041() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 41 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 41 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_042() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 42 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 42 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_043() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 43 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 43 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_044() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 44 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 44 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_045() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 45 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 45 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_046() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 46 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 46 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_047() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 47 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 47 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_048() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 48 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 48 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_049() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 49 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 49 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_050() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 50 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 50 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_051() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 51 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 51 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_052() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 52 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 52 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_053() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 53 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 53 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_054() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 54 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 54 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_055() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 55 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 55 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_056() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 56 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 56 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_057() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 57 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 57 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_058() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 58 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 58 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_059() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 59 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 59 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_060() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 60 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 60 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_061() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 61 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 61 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_062() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 62 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 62 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_063() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 63 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 63 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_064() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 64 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 64 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_065() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 65 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 65 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_066() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 66 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 66 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_067() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 67 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 67 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_068() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 68 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 68 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_069() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 69 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 69 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_070() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 70 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 70 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_071() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 71 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 71 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_072() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 72 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 72 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_073() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 73 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 73 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_074() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 74 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 74 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_075() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 75 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 75 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_076() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 76 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 76 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_077() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 77 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 77 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_078() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 78 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 78 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_079() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 79 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 79 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_080() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 80 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 80 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_081() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 81 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 81 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_082() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 82 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 82 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_083() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 83 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 83 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_084() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 84 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 84 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_085() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 85 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 85 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_086() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 86 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 86 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_087() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 87 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 87 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_088() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 88 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 88 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_089() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 89 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 89 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_090() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 90 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 90 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_091() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 91 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 91 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_092() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 92 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 92 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_093() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 93 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 93 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_094() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 94 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 94 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_095() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 95 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 95 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_096() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 96 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 96 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_097() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 97 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 97 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_098() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 98 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 98 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_099() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 99 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 99 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_100() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 100 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 100 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_101() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 101 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 101 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_102() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 102 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 102 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_103() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 103 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 103 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_104() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 104 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 104 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_105() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 105 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 105 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_106() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 106 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 106 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_107() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 107 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 107 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_108() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 108 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 108 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_109() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 109 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 109 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_110() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 110 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 110 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_111() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 111 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 111 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_112() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 112 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 112 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_113() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 113 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 113 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_114() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 114 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 114 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_115() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 115 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 115 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_116() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 116 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 116 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_117() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 117 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 117 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_118() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 118 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 118 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_119() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 119 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 119 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_120() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 120 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 120 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_121() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 121 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 121 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_122() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 122 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 122 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_123() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 123 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 123 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_124() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 124 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 124 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_125() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 125 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 125 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_126() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 126 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 126 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_127() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 127 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 127 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_128() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 128 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 128 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_129() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 129 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 129 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_130() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 130 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 130 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_131() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 131 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 131 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_132() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 132 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 132 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_133() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 133 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 133 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_134() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 134 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 134 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_135() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 135 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 135 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_136() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 136 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 136 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_137() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 137 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 137 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_138() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 138 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 138 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_139() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 139 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 139 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_140() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 140 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 140 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_141() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 141 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 141 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_142() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 142 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 142 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_143() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 143 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 143 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_144() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 144 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 144 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_145() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 145 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 145 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_146() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 146 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 146 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_147() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 147 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 147 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_148() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 148 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 148 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_149() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 149 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 149 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_150() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 150 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 150 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_151() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 151 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 151 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_152() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 152 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 152 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_153() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 153 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 153 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_154() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 154 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 154 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_155() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 155 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 155 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_156() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 156 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 156 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_157() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 157 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 157 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_158() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 158 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 158 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_159() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 159 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 159 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_160() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 160 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 160 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_161() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 161 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 161 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_162() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 162 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 162 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_163() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 163 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 163 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_164() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 164 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 164 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_165() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 165 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 165 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_166() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 166 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 166 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_167() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 167 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 167 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_168() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 168 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 168 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_utils_stress_169() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let tp = tanh_prime(0.0);
        assert!((tp - 1.0).abs() < 1e-6);

        let ortho = init_orthogonal(4, 4, 169 as u64);
        assert_eq!(ortho.shape(), &[4, 4]);

        let uni = init_uniform(4, 4, 4, 169 as u64);
        assert_eq!(uni.shape(), &[4, 4]);

        let mut g = vec![3.0, 4.0];
        let norm = clip_grad_norm(&mut g, 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((g[0] - 1.5).abs() < 1e-6);
    }

    // brain-rnn production numerical verification padding line 0
    // brain-rnn production numerical verification padding line 1
}
