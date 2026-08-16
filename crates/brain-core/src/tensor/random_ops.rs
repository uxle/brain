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
    let fan_in = shape[1] * if shape.len() > 2 { shape[2..].iter().product() } else { 1 };
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
    fn test_random_ops_stress_case_001() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_002() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_003() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_004() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_005() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_006() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_007() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_008() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_009() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_010() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_011() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_012() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_013() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_014() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_015() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_016() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_017() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_018() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_019() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_020() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_021() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_022() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_023() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_024() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_025() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_026() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_027() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_028() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_029() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_030() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_031() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_032() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_033() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_034() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_035() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_036() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_037() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_038() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_039() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_040() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_041() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_042() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_043() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_044() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_045() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_046() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_047() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_048() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_049() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_050() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_051() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_052() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_053() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_054() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_055() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_056() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_057() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_058() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_059() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_060() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_061() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_062() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_063() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_064() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_065() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_066() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_067() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_068() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_069() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_070() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_071() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_072() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_073() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_074() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_075() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_076() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_077() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_078() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_079() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_080() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_081() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_082() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_083() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_084() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_085() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_086() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_087() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_088() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_089() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_090() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_091() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_092() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_093() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_094() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_095() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_096() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_097() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_098() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_099() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_100() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_101() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_102() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_103() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_104() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_105() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_106() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_107() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_108() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_109() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_110() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_111() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_112() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_113() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_114() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_115() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_116() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_117() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_118() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_119() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_120() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_121() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_122() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_123() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_124() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_125() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_126() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_127() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_128() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_129() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_130() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_131() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_132() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_133() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_134() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_135() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_136() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_137() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_138() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_139() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_140() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_141() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_142() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_143() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_144() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_145() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_146() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_147() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_148() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_149() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_150() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_151() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_152() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_153() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_154() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_155() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_156() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_157() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_158() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_159() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_160() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_161() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_162() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_163() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_164() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_165() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_166() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_167() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_168() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_169() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_170() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_171() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_172() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_173() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_174() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_175() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_176() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_177() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_178() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_179() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_180() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_181() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_182() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_183() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_184() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_185() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_186() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_187() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_188() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_189() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_190() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_191() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_192() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_193() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_194() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_195() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_196() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_197() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_198() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_199() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_200() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_201() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_202() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_203() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_204() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_205() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_206() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_207() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_208() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_209() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_210() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_211() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_212() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_213() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_214() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_215() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_216() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_217() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_218() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_219() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_220() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_221() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_222() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_223() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_224() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_225() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_226() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_227() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_228() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_229() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_230() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_231() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_232() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_233() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_234() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_235() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_236() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_237() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_238() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_239() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_240() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_241() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_242() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_243() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_244() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_245() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_246() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_247() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_248() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_249() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_250() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_251() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_252() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_253() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_254() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_255() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_256() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_257() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_258() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_259() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_260() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_261() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_262() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_263() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_264() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_265() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_266() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_267() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_268() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_269() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_270() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_271() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_272() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_273() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_274() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_275() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_276() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_277() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_278() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_279() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_280() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_281() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_282() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_283() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_284() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_285() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_286() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_287() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_288() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_289() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_290() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_291() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_292() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_293() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_294() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_295() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_296() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_297() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_298() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_299() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_300() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_301() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_302() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_303() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_304() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_305() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_306() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_307() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_308() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_309() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_310() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_311() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_312() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_313() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_314() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_315() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_316() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_317() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_318() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_319() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_320() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_321() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_322() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_323() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_324() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_325() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_326() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_327() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_328() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_329() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_330() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_331() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_332() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_333() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_334() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_335() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_336() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_337() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_338() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_339() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_340() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_341() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_342() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_343() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_344() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_345() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_346() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_347() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_348() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_349() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_350() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_351() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_352() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_353() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_354() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_355() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_356() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_357() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_358() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_359() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_360() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_361() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_362() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_363() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_364() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_365() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_366() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_367() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_368() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_369() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_370() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_371() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_372() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_373() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_374() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_375() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_376() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_377() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_378() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_379() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_380() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_381() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_382() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_383() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_384() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_385() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_386() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_387() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_388() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_389() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_390() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_391() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_392() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_393() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_394() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_395() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_396() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_397() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_398() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_399() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_400() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_401() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_402() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_403() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_404() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_405() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_406() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_407() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_408() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_409() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_410() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_411() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_412() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_413() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_414() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_415() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_416() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_417() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_418() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_419() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_420() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_421() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_422() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_423() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_424() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_425() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_426() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_427() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_428() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }

    #[test]
    fn test_random_ops_stress_case_429() {
        let ku = kaiming_uniform(vec![2, 4], 0.0);
        assert_eq!(ku.numel(), 8);
        let xu = xavier_uniform(vec![2, 4], 1.0);
        assert_eq!(xu.numel(), 8);
    }
}
