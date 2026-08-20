//! # Kaiming (He) & Xavier (Glorot) Initialization
//!
//! Variance-preserving random weight tensor initialization for deep networks.
#![allow(missing_docs)]

use super::calculate_fan;
use brain_core::random::{NormalDist, UniformDist};
use brain_core::Tensor;

/// Configuration for Kaiming/Xavier initialization.
#[derive(Debug, Clone, Default)]
pub struct InitConfig {
    pub gain: f64,
}

/// Initializes weight tensor using Kaiming Uniform: U(-bound, bound) where bound = gain * sqrt(3 / fan_in).
pub fn kaiming_uniform(shape: &[usize], a: f64) -> Tensor {
    let (fan_in, _) = calculate_fan(shape);
    let gain = (2.0 / (1.0 + a * a)).sqrt();
    let bound = gain * (3.0 / (fan_in as f64).max(1.0)).sqrt();

    let total: usize = shape.iter().product();
    let dist = UniformDist::new(-bound, bound);
    let data = brain_core::random::with_rng(|rng| {
        let mut v = Vec::with_capacity(total);
        for _ in 0..total {
            v.push(dist.sample(rng));
        }
        v
    });

    Tensor::from_vec(data, shape.to_vec())
}

/// Initializes weight tensor using Kaiming Normal: N(0, std^2) where std = gain / sqrt(fan_in).
pub fn kaiming_normal(shape: &[usize], a: f64) -> Tensor {
    let (fan_in, _) = calculate_fan(shape);
    let gain = (2.0 / (1.0 + a * a)).sqrt();
    let std = gain / (fan_in as f64).max(1.0).sqrt();

    let total: usize = shape.iter().product();
    let dist = NormalDist::new(0.0, std);
    let data = brain_core::random::with_rng(|rng| {
        let mut v = Vec::with_capacity(total);
        for _ in 0..total {
            v.push(dist.sample(rng));
        }
        v
    });

    Tensor::from_vec(data, shape.to_vec())
}

/// Initializes weight tensor using Xavier Uniform: U(-bound, bound) where bound = sqrt(6 / (fan_in + fan_out)).
pub fn xavier_uniform(shape: &[usize]) -> Tensor {
    let (fan_in, fan_out) = calculate_fan(shape);
    let bound = (6.0 / (fan_in + fan_out) as f64).sqrt();

    let total: usize = shape.iter().product();
    let dist = UniformDist::new(-bound, bound);
    let data = brain_core::random::with_rng(|rng| {
        let mut v = Vec::with_capacity(total);
        for _ in 0..total {
            v.push(dist.sample(rng));
        }
        v
    });

    Tensor::from_vec(data, shape.to_vec())
}

/// Initializes weight tensor using Xavier Normal: N(0, std^2) where std = sqrt(2 / (fan_in + fan_out)).
pub fn xavier_normal(shape: &[usize]) -> Tensor {
    let (fan_in, fan_out) = calculate_fan(shape);
    let std = (2.0 / (fan_in + fan_out) as f64).sqrt();

    let total: usize = shape.iter().product();
    let dist = NormalDist::new(0.0, std);
    let data = brain_core::random::with_rng(|rng| {
        let mut v = Vec::with_capacity(total);
        for _ in 0..total {
            v.push(dist.sample(rng));
        }
        v
    });

    Tensor::from_vec(data, shape.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xavier_uniform_variance() {
        let t = xavier_uniform(&[256, 256]);
        let data = t.to_vec();
        let n = data.len() as f64;
        let mean: f64 = data.iter().sum::<f64>() / n;
        let var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let expected_var = 2.0 / (256.0 + 256.0); // 2 / (fan_in + fan_out)
        let rel_err = (var - expected_var).abs() / expected_var;
        assert!(
            rel_err < 0.05,
            "Xavier uniform var={}, expected={}, rel_err={}",
            var,
            expected_var,
            rel_err
        );
    }

    #[test]
    fn test_kaiming_normal_variance() {
        let t = kaiming_normal(&[256, 256], 0.0);
        let data = t.to_vec();
        let n = data.len() as f64;
        let mean: f64 = data.iter().sum::<f64>() / n;
        let var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let expected_var = 2.0 / 256.0; // 2 / fan_in for ReLU (a=0.0)
        let rel_err = (var - expected_var).abs() / expected_var;
        assert!(
            rel_err < 0.05,
            "Kaiming normal var={}, expected={}, rel_err={}",
            var,
            expected_var,
            rel_err
        );
    }
}
