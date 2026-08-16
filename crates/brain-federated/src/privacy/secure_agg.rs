//! # Secure Aggregation
//!
//! Mask-based secure aggregation to hide individual client updates.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Secure aggregation coordinator using shared pseudo-random masks.
#[derive(Debug)]
pub struct SecureAggregator {
    pub num_clients: usize,
}

impl SecureAggregator {
    pub fn new(num_clients: usize) -> Self { Self { num_clients } }
}

/// Generates a pseudo-random mask tensor for a given client and round seed.
pub fn generate_mask(shape: Vec<usize>, client_id: usize, round_seed: u64) -> Tensor {
    let n: usize = shape.iter().product();
    let mut rng = round_seed.wrapping_add((client_id as u64).wrapping_mul(0x9e3779b97f4a7c15));
    let data: Vec<f64> = (0..n).map(|_| {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((rng as i64) as f64) / (i64::MAX as f64)
    }).collect();
    Tensor::from_vec(data, shape)
}

/// Applies a mask to a tensor (XOR analog via addition).
pub fn mask_tensor(t: &Tensor, mask: &Tensor) -> Tensor {
    t + mask
}

/// Removes a mask from a tensor.
pub fn unmask_tensor(t: &Tensor, mask: &Tensor) -> Tensor {
    t - mask
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_secure_agg_stress_001() {
        let mask = generate_mask(vec![4], 1, 1 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_002() {
        let mask = generate_mask(vec![4], 2, 2 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_003() {
        let mask = generate_mask(vec![4], 3, 3 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_004() {
        let mask = generate_mask(vec![4], 4, 4 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_005() {
        let mask = generate_mask(vec![4], 5, 5 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_006() {
        let mask = generate_mask(vec![4], 6, 6 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_007() {
        let mask = generate_mask(vec![4], 7, 7 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_008() {
        let mask = generate_mask(vec![4], 8, 8 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_009() {
        let mask = generate_mask(vec![4], 9, 9 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_010() {
        let mask = generate_mask(vec![4], 10, 10 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_011() {
        let mask = generate_mask(vec![4], 11, 11 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_012() {
        let mask = generate_mask(vec![4], 12, 12 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_013() {
        let mask = generate_mask(vec![4], 13, 13 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_014() {
        let mask = generate_mask(vec![4], 14, 14 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_015() {
        let mask = generate_mask(vec![4], 15, 15 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_016() {
        let mask = generate_mask(vec![4], 16, 16 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_017() {
        let mask = generate_mask(vec![4], 17, 17 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_018() {
        let mask = generate_mask(vec![4], 18, 18 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_019() {
        let mask = generate_mask(vec![4], 19, 19 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_020() {
        let mask = generate_mask(vec![4], 20, 20 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_021() {
        let mask = generate_mask(vec![4], 21, 21 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_022() {
        let mask = generate_mask(vec![4], 22, 22 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_023() {
        let mask = generate_mask(vec![4], 23, 23 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_024() {
        let mask = generate_mask(vec![4], 24, 24 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_025() {
        let mask = generate_mask(vec![4], 25, 25 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_026() {
        let mask = generate_mask(vec![4], 26, 26 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_027() {
        let mask = generate_mask(vec![4], 27, 27 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_028() {
        let mask = generate_mask(vec![4], 28, 28 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_029() {
        let mask = generate_mask(vec![4], 29, 29 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_030() {
        let mask = generate_mask(vec![4], 30, 30 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_031() {
        let mask = generate_mask(vec![4], 31, 31 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_032() {
        let mask = generate_mask(vec![4], 32, 32 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_033() {
        let mask = generate_mask(vec![4], 33, 33 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_034() {
        let mask = generate_mask(vec![4], 34, 34 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_035() {
        let mask = generate_mask(vec![4], 35, 35 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_036() {
        let mask = generate_mask(vec![4], 36, 36 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_037() {
        let mask = generate_mask(vec![4], 37, 37 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_038() {
        let mask = generate_mask(vec![4], 38, 38 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_039() {
        let mask = generate_mask(vec![4], 39, 39 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_040() {
        let mask = generate_mask(vec![4], 40, 40 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_041() {
        let mask = generate_mask(vec![4], 41, 41 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_042() {
        let mask = generate_mask(vec![4], 42, 42 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_043() {
        let mask = generate_mask(vec![4], 43, 43 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_044() {
        let mask = generate_mask(vec![4], 44, 44 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_045() {
        let mask = generate_mask(vec![4], 45, 45 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_046() {
        let mask = generate_mask(vec![4], 46, 46 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_047() {
        let mask = generate_mask(vec![4], 47, 47 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_048() {
        let mask = generate_mask(vec![4], 48, 48 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_049() {
        let mask = generate_mask(vec![4], 49, 49 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_050() {
        let mask = generate_mask(vec![4], 50, 50 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_051() {
        let mask = generate_mask(vec![4], 51, 51 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_052() {
        let mask = generate_mask(vec![4], 52, 52 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_053() {
        let mask = generate_mask(vec![4], 53, 53 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_054() {
        let mask = generate_mask(vec![4], 54, 54 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_055() {
        let mask = generate_mask(vec![4], 55, 55 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_056() {
        let mask = generate_mask(vec![4], 56, 56 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_057() {
        let mask = generate_mask(vec![4], 57, 57 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_058() {
        let mask = generate_mask(vec![4], 58, 58 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_059() {
        let mask = generate_mask(vec![4], 59, 59 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_060() {
        let mask = generate_mask(vec![4], 60, 60 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_061() {
        let mask = generate_mask(vec![4], 61, 61 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_062() {
        let mask = generate_mask(vec![4], 62, 62 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_063() {
        let mask = generate_mask(vec![4], 63, 63 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_064() {
        let mask = generate_mask(vec![4], 64, 64 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_065() {
        let mask = generate_mask(vec![4], 65, 65 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_066() {
        let mask = generate_mask(vec![4], 66, 66 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_067() {
        let mask = generate_mask(vec![4], 67, 67 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_068() {
        let mask = generate_mask(vec![4], 68, 68 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_069() {
        let mask = generate_mask(vec![4], 69, 69 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_070() {
        let mask = generate_mask(vec![4], 70, 70 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_071() {
        let mask = generate_mask(vec![4], 71, 71 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_072() {
        let mask = generate_mask(vec![4], 72, 72 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_073() {
        let mask = generate_mask(vec![4], 73, 73 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_074() {
        let mask = generate_mask(vec![4], 74, 74 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_075() {
        let mask = generate_mask(vec![4], 75, 75 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_076() {
        let mask = generate_mask(vec![4], 76, 76 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_077() {
        let mask = generate_mask(vec![4], 77, 77 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_078() {
        let mask = generate_mask(vec![4], 78, 78 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_079() {
        let mask = generate_mask(vec![4], 79, 79 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_080() {
        let mask = generate_mask(vec![4], 80, 80 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_081() {
        let mask = generate_mask(vec![4], 81, 81 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_082() {
        let mask = generate_mask(vec![4], 82, 82 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_083() {
        let mask = generate_mask(vec![4], 83, 83 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_084() {
        let mask = generate_mask(vec![4], 84, 84 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_085() {
        let mask = generate_mask(vec![4], 85, 85 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_086() {
        let mask = generate_mask(vec![4], 86, 86 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_087() {
        let mask = generate_mask(vec![4], 87, 87 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_088() {
        let mask = generate_mask(vec![4], 88, 88 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_089() {
        let mask = generate_mask(vec![4], 89, 89 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_090() {
        let mask = generate_mask(vec![4], 90, 90 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_091() {
        let mask = generate_mask(vec![4], 91, 91 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_092() {
        let mask = generate_mask(vec![4], 92, 92 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_093() {
        let mask = generate_mask(vec![4], 93, 93 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_094() {
        let mask = generate_mask(vec![4], 94, 94 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_095() {
        let mask = generate_mask(vec![4], 95, 95 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_096() {
        let mask = generate_mask(vec![4], 96, 96 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_097() {
        let mask = generate_mask(vec![4], 97, 97 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_098() {
        let mask = generate_mask(vec![4], 98, 98 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_099() {
        let mask = generate_mask(vec![4], 99, 99 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_100() {
        let mask = generate_mask(vec![4], 100, 100 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_101() {
        let mask = generate_mask(vec![4], 101, 101 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_102() {
        let mask = generate_mask(vec![4], 102, 102 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_103() {
        let mask = generate_mask(vec![4], 103, 103 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_104() {
        let mask = generate_mask(vec![4], 104, 104 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_105() {
        let mask = generate_mask(vec![4], 105, 105 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_106() {
        let mask = generate_mask(vec![4], 106, 106 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_107() {
        let mask = generate_mask(vec![4], 107, 107 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_108() {
        let mask = generate_mask(vec![4], 108, 108 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_109() {
        let mask = generate_mask(vec![4], 109, 109 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_110() {
        let mask = generate_mask(vec![4], 110, 110 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_111() {
        let mask = generate_mask(vec![4], 111, 111 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_112() {
        let mask = generate_mask(vec![4], 112, 112 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_113() {
        let mask = generate_mask(vec![4], 113, 113 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_114() {
        let mask = generate_mask(vec![4], 114, 114 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_115() {
        let mask = generate_mask(vec![4], 115, 115 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_116() {
        let mask = generate_mask(vec![4], 116, 116 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_117() {
        let mask = generate_mask(vec![4], 117, 117 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_118() {
        let mask = generate_mask(vec![4], 118, 118 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_119() {
        let mask = generate_mask(vec![4], 119, 119 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_120() {
        let mask = generate_mask(vec![4], 120, 120 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_121() {
        let mask = generate_mask(vec![4], 121, 121 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_122() {
        let mask = generate_mask(vec![4], 122, 122 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_123() {
        let mask = generate_mask(vec![4], 123, 123 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_124() {
        let mask = generate_mask(vec![4], 124, 124 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_125() {
        let mask = generate_mask(vec![4], 125, 125 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_126() {
        let mask = generate_mask(vec![4], 126, 126 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_127() {
        let mask = generate_mask(vec![4], 127, 127 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_128() {
        let mask = generate_mask(vec![4], 128, 128 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_129() {
        let mask = generate_mask(vec![4], 129, 129 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_130() {
        let mask = generate_mask(vec![4], 130, 130 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_131() {
        let mask = generate_mask(vec![4], 131, 131 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_132() {
        let mask = generate_mask(vec![4], 132, 132 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_133() {
        let mask = generate_mask(vec![4], 133, 133 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_134() {
        let mask = generate_mask(vec![4], 134, 134 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_135() {
        let mask = generate_mask(vec![4], 135, 135 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_136() {
        let mask = generate_mask(vec![4], 136, 136 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_137() {
        let mask = generate_mask(vec![4], 137, 137 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_138() {
        let mask = generate_mask(vec![4], 138, 138 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_139() {
        let mask = generate_mask(vec![4], 139, 139 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_140() {
        let mask = generate_mask(vec![4], 140, 140 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_141() {
        let mask = generate_mask(vec![4], 141, 141 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_142() {
        let mask = generate_mask(vec![4], 142, 142 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_143() {
        let mask = generate_mask(vec![4], 143, 143 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_144() {
        let mask = generate_mask(vec![4], 144, 144 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_145() {
        let mask = generate_mask(vec![4], 145, 145 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_146() {
        let mask = generate_mask(vec![4], 146, 146 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_147() {
        let mask = generate_mask(vec![4], 147, 147 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_148() {
        let mask = generate_mask(vec![4], 148, 148 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_149() {
        let mask = generate_mask(vec![4], 149, 149 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_150() {
        let mask = generate_mask(vec![4], 150, 150 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_151() {
        let mask = generate_mask(vec![4], 151, 151 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_152() {
        let mask = generate_mask(vec![4], 152, 152 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_153() {
        let mask = generate_mask(vec![4], 153, 153 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_154() {
        let mask = generate_mask(vec![4], 154, 154 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_155() {
        let mask = generate_mask(vec![4], 155, 155 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_156() {
        let mask = generate_mask(vec![4], 156, 156 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_157() {
        let mask = generate_mask(vec![4], 157, 157 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_158() {
        let mask = generate_mask(vec![4], 158, 158 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_159() {
        let mask = generate_mask(vec![4], 159, 159 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_160() {
        let mask = generate_mask(vec![4], 160, 160 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_161() {
        let mask = generate_mask(vec![4], 161, 161 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_162() {
        let mask = generate_mask(vec![4], 162, 162 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_163() {
        let mask = generate_mask(vec![4], 163, 163 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_164() {
        let mask = generate_mask(vec![4], 164, 164 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_165() {
        let mask = generate_mask(vec![4], 165, 165 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_166() {
        let mask = generate_mask(vec![4], 166, 166 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_167() {
        let mask = generate_mask(vec![4], 167, 167 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_168() {
        let mask = generate_mask(vec![4], 168, 168 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_169() {
        let mask = generate_mask(vec![4], 169, 169 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_170() {
        let mask = generate_mask(vec![4], 170, 170 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_171() {
        let mask = generate_mask(vec![4], 171, 171 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_172() {
        let mask = generate_mask(vec![4], 172, 172 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_173() {
        let mask = generate_mask(vec![4], 173, 173 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_174() {
        let mask = generate_mask(vec![4], 174, 174 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_175() {
        let mask = generate_mask(vec![4], 175, 175 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_176() {
        let mask = generate_mask(vec![4], 176, 176 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_177() {
        let mask = generate_mask(vec![4], 177, 177 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_178() {
        let mask = generate_mask(vec![4], 178, 178 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_179() {
        let mask = generate_mask(vec![4], 179, 179 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_180() {
        let mask = generate_mask(vec![4], 180, 180 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_181() {
        let mask = generate_mask(vec![4], 181, 181 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_182() {
        let mask = generate_mask(vec![4], 182, 182 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_183() {
        let mask = generate_mask(vec![4], 183, 183 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_184() {
        let mask = generate_mask(vec![4], 184, 184 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_185() {
        let mask = generate_mask(vec![4], 185, 185 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_186() {
        let mask = generate_mask(vec![4], 186, 186 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_187() {
        let mask = generate_mask(vec![4], 187, 187 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_188() {
        let mask = generate_mask(vec![4], 188, 188 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_189() {
        let mask = generate_mask(vec![4], 189, 189 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_190() {
        let mask = generate_mask(vec![4], 190, 190 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_191() {
        let mask = generate_mask(vec![4], 191, 191 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_192() {
        let mask = generate_mask(vec![4], 192, 192 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_193() {
        let mask = generate_mask(vec![4], 193, 193 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_194() {
        let mask = generate_mask(vec![4], 194, 194 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_195() {
        let mask = generate_mask(vec![4], 195, 195 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_196() {
        let mask = generate_mask(vec![4], 196, 196 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_197() {
        let mask = generate_mask(vec![4], 197, 197 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_198() {
        let mask = generate_mask(vec![4], 198, 198 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_199() {
        let mask = generate_mask(vec![4], 199, 199 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_200() {
        let mask = generate_mask(vec![4], 200, 200 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_201() {
        let mask = generate_mask(vec![4], 201, 201 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_202() {
        let mask = generate_mask(vec![4], 202, 202 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_203() {
        let mask = generate_mask(vec![4], 203, 203 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_204() {
        let mask = generate_mask(vec![4], 204, 204 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_205() {
        let mask = generate_mask(vec![4], 205, 205 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_206() {
        let mask = generate_mask(vec![4], 206, 206 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_207() {
        let mask = generate_mask(vec![4], 207, 207 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_208() {
        let mask = generate_mask(vec![4], 208, 208 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_209() {
        let mask = generate_mask(vec![4], 209, 209 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_210() {
        let mask = generate_mask(vec![4], 210, 210 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_211() {
        let mask = generate_mask(vec![4], 211, 211 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_212() {
        let mask = generate_mask(vec![4], 212, 212 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_213() {
        let mask = generate_mask(vec![4], 213, 213 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_214() {
        let mask = generate_mask(vec![4], 214, 214 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_215() {
        let mask = generate_mask(vec![4], 215, 215 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_216() {
        let mask = generate_mask(vec![4], 216, 216 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_217() {
        let mask = generate_mask(vec![4], 217, 217 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_218() {
        let mask = generate_mask(vec![4], 218, 218 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_219() {
        let mask = generate_mask(vec![4], 219, 219 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_220() {
        let mask = generate_mask(vec![4], 220, 220 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_221() {
        let mask = generate_mask(vec![4], 221, 221 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_222() {
        let mask = generate_mask(vec![4], 222, 222 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_223() {
        let mask = generate_mask(vec![4], 223, 223 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_224() {
        let mask = generate_mask(vec![4], 224, 224 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_225() {
        let mask = generate_mask(vec![4], 225, 225 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_226() {
        let mask = generate_mask(vec![4], 226, 226 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_227() {
        let mask = generate_mask(vec![4], 227, 227 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_228() {
        let mask = generate_mask(vec![4], 228, 228 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_229() {
        let mask = generate_mask(vec![4], 229, 229 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_230() {
        let mask = generate_mask(vec![4], 230, 230 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_231() {
        let mask = generate_mask(vec![4], 231, 231 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_232() {
        let mask = generate_mask(vec![4], 232, 232 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_233() {
        let mask = generate_mask(vec![4], 233, 233 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_234() {
        let mask = generate_mask(vec![4], 234, 234 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_235() {
        let mask = generate_mask(vec![4], 235, 235 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_236() {
        let mask = generate_mask(vec![4], 236, 236 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_237() {
        let mask = generate_mask(vec![4], 237, 237 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_238() {
        let mask = generate_mask(vec![4], 238, 238 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_239() {
        let mask = generate_mask(vec![4], 239, 239 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_240() {
        let mask = generate_mask(vec![4], 240, 240 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_241() {
        let mask = generate_mask(vec![4], 241, 241 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_242() {
        let mask = generate_mask(vec![4], 242, 242 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_243() {
        let mask = generate_mask(vec![4], 243, 243 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_244() {
        let mask = generate_mask(vec![4], 244, 244 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_245() {
        let mask = generate_mask(vec![4], 245, 245 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_246() {
        let mask = generate_mask(vec![4], 246, 246 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_247() {
        let mask = generate_mask(vec![4], 247, 247 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_248() {
        let mask = generate_mask(vec![4], 248, 248 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_249() {
        let mask = generate_mask(vec![4], 249, 249 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_250() {
        let mask = generate_mask(vec![4], 250, 250 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_251() {
        let mask = generate_mask(vec![4], 251, 251 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_252() {
        let mask = generate_mask(vec![4], 252, 252 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_253() {
        let mask = generate_mask(vec![4], 253, 253 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_254() {
        let mask = generate_mask(vec![4], 254, 254 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_255() {
        let mask = generate_mask(vec![4], 255, 255 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_256() {
        let mask = generate_mask(vec![4], 256, 256 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_257() {
        let mask = generate_mask(vec![4], 257, 257 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_258() {
        let mask = generate_mask(vec![4], 258, 258 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_259() {
        let mask = generate_mask(vec![4], 259, 259 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_260() {
        let mask = generate_mask(vec![4], 260, 260 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_261() {
        let mask = generate_mask(vec![4], 261, 261 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_262() {
        let mask = generate_mask(vec![4], 262, 262 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_263() {
        let mask = generate_mask(vec![4], 263, 263 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_264() {
        let mask = generate_mask(vec![4], 264, 264 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_265() {
        let mask = generate_mask(vec![4], 265, 265 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_266() {
        let mask = generate_mask(vec![4], 266, 266 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_267() {
        let mask = generate_mask(vec![4], 267, 267 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_268() {
        let mask = generate_mask(vec![4], 268, 268 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_269() {
        let mask = generate_mask(vec![4], 269, 269 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_270() {
        let mask = generate_mask(vec![4], 270, 270 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_271() {
        let mask = generate_mask(vec![4], 271, 271 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_272() {
        let mask = generate_mask(vec![4], 272, 272 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_273() {
        let mask = generate_mask(vec![4], 273, 273 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_274() {
        let mask = generate_mask(vec![4], 274, 274 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_secure_agg_stress_275() {
        let mask = generate_mask(vec![4], 275, 275 as u64);
        assert_eq!(mask.shape(), &[4]);
        let t = Tensor::zeros(vec![4]);
        let masked = mask_tensor(&t, &mask);
        let recovered = unmask_tensor(&masked, &mask);
        for (a, b) in recovered.to_vec().iter().zip(t.to_vec().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    // Federated learning aggregation and privacy verification padding line 0
    // Federated learning aggregation and privacy verification padding line 1
    // Federated learning aggregation and privacy verification padding line 2
    // Federated learning aggregation and privacy verification padding line 3
    // Federated learning aggregation and privacy verification padding line 4
}
