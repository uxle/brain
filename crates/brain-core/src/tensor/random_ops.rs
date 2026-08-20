//! Random tensor initialization (Kaiming, Xavier, Orthogonal, Truncated Normal).
//!
//! This module provides parameter weight initializers and random tensor generation factories.

use crate::random::{self, BrainRng, NormalDist, Rng, UniformDist};
use crate::tensor::Tensor;

/// Generates a tensor filled with uniform random values in `[0.0, 1.0)`.
pub fn rand(shape: Vec<usize>) -> Tensor {
    Tensor::rand(shape)
}

/// Generates a tensor filled with standard normal random values.
pub fn randn(shape: Vec<usize>) -> Tensor {
    Tensor::randn(shape)
}

/// Fills a tensor with Kaiming / He uniform initialization.
pub fn kaiming_uniform(shape: Vec<usize>, a: f64) -> Tensor {
    assert!(shape.len() >= 2);
    let fan_in = shape[1]
        * if shape.len() > 2 {
            shape[2..].iter().product()
        } else {
            1
        };
    let gain = (2.0 / (1.0 + a * a)).sqrt();
    let std = gain / (fan_in as f64).sqrt();
    let bound = (3.0f64).sqrt() * std;

    let dist = UniformDist::new(-bound, bound);
    let numel: usize = shape.iter().product();
    let mut data = Vec::with_capacity(numel);

    random::with_rng(|rng| {
        for _ in 0..numel {
            data.push(dist.sample(rng));
        }
    });

    Tensor::new(data, shape)
}

/// Fills a tensor with Xavier / Glorot uniform initialization.
pub fn xavier_uniform(shape: Vec<usize>, gain: f64) -> Tensor {
    assert!(shape.len() >= 2);
    let fan_in = shape[1];
    let fan_out = shape[0];
    let std = gain * (2.0 / (fan_in + fan_out) as f64).sqrt();
    let bound = (3.0f64).sqrt() * std;

    let dist = UniformDist::new(-bound, bound);
    let numel: usize = shape.iter().product();
    let mut data = Vec::with_capacity(numel);

    random::with_rng(|rng| {
        for _ in 0..numel {
            data.push(dist.sample(rng));
        }
    });

    Tensor::new(data, shape)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kaiming_and_xavier() {
        let ku = kaiming_uniform(vec![10, 20], 0.0);
        assert_eq!(ku.shape(), &[10, 20]);
        let xu = xavier_uniform(vec![10, 20], 1.0);
        assert_eq!(xu.shape(), &[10, 20]);
    }

    #[test]
    fn test_random_ops_sampling() {
        let u = rand(vec![4, 4]);
        assert_eq!(u.shape(), &[4, 4]);
        for &v in u.data() {
            assert!(v >= 0.0 && v <= 1.0);
        }

        let n = randn(vec![2, 3]);
        assert_eq!(n.shape(), &[2, 3]);
    }
}
