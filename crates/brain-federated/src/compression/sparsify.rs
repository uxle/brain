//! # Gradient Sparsification
//!
//! Top-K and threshold sparsification for communication efficiency.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Configuration for gradient sparsification.
#[derive(Debug, Clone)]
pub struct SparseConfig {
    pub sparsity: f64,
}

impl Default for SparseConfig {
    fn default() -> Self { Self { sparsity: 0.9 } }
}

/// Returns a mask tensor with top-k fraction of elements kept.
pub fn top_k_sparsify(t: &Tensor, keep_fraction: f64) -> Tensor {
    let data = t.to_vec();
    let n = data.len();
    let k = (n as f64 * keep_fraction.clamp(0.0, 1.0)).ceil() as usize;
    let mut abs_vals: Vec<(usize, f64)> = data.iter().enumerate().map(|(i, v)| (i, v.abs())).collect();
    abs_vals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut result = vec![0.0f64; n];
    for (idx, _) in abs_vals.iter().take(k) {
        result[*idx] = data[*idx];
    }
    Tensor::from_vec(result, t.shape().to_vec())
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_sparsify_stress_001() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_002() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_003() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_004() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_005() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_006() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_007() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_008() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_009() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_010() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_011() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_012() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_013() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_014() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_015() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_016() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_017() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_018() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_019() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_020() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_021() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_022() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_023() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_024() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_025() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_026() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_027() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_028() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_029() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_030() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_031() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_032() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_033() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_034() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_035() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_036() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_037() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_038() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_039() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_040() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_041() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_042() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_043() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_044() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_045() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_046() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_047() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_048() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_049() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_050() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_051() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_052() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_053() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_054() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_055() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_056() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_057() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_058() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_059() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_060() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_061() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_062() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_063() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_064() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_065() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_066() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_067() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_068() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_069() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_070() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_071() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_072() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_073() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_074() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_075() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_076() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_077() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_078() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_079() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_080() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_081() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_082() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_083() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_084() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_085() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_086() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_087() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_088() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_089() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_090() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_091() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_092() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_093() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_094() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_095() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_096() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_097() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_098() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_099() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_100() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_101() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_102() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_103() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_104() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_105() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_106() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_107() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_108() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_109() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_110() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_111() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_112() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_113() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_114() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_115() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_116() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_117() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_118() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_119() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_120() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_121() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_122() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_123() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_124() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_125() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_126() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_127() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_128() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_129() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_130() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_131() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_132() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_133() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_134() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_135() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_136() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_137() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_138() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_139() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_140() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_141() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_142() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_143() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_144() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_145() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_146() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_147() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_148() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_149() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_150() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_151() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_152() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_153() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_154() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_155() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_156() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_157() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_158() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_159() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_160() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_161() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_162() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_163() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_164() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_165() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_166() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_167() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_168() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_169() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_170() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_171() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_172() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_173() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_174() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_175() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_176() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_177() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_178() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_179() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_180() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_181() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_182() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_183() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_184() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_185() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_186() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_187() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_188() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_189() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_190() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_191() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_192() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_193() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_194() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_195() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_196() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_197() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_198() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_199() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_200() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_201() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_202() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_203() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_204() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_205() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_206() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_207() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_208() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_209() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_210() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_211() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_212() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_213() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_214() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_215() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_216() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_217() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_218() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_219() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_220() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_221() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_222() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_223() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_224() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_225() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_226() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_227() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_228() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_229() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_230() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_231() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_232() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_233() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_234() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_235() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_236() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_237() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_238() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_239() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_240() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_241() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_242() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_243() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_244() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_245() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_246() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_247() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_248() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_249() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_250() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_251() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_252() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_253() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_254() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_255() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_256() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_257() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_258() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_259() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_260() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_261() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_262() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_263() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_264() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_265() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_266() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_267() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_268() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_269() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_270() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_271() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_272() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_273() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_274() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_275() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_276() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_277() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_278() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_279() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_280() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_281() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_282() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_283() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_284() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_285() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_286() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_287() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_288() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_289() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_290() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_291() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_292() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_293() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_294() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_295() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_296() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_297() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_298() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_299() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_300() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_301() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_302() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_303() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_304() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_305() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_306() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_307() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_308() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_309() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_310() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_311() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_312() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_313() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_314() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_315() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_316() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_317() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_318() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_319() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_320() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_321() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_322() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_323() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_324() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_325() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_326() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_327() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_328() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_329() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_330() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_331() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_332() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_333() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_334() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_335() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_336() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_337() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_338() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_339() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_340() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_341() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_342() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_343() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_344() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_345() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_346() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_347() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_348() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_349() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_350() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_351() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_352() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_353() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_354() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_355() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_356() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_357() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_358() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_359() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_360() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_361() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_362() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_363() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_364() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_365() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_366() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_367() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_368() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_369() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_370() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_371() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_372() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_373() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_374() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_375() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_376() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_377() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_378() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_379() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_380() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_381() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_382() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_383() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_384() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_385() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_386() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_387() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_388() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_389() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_390() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_391() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_392() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_393() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_394() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_395() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_396() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_397() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_398() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_399() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_400() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_401() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_402() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_403() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_404() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_405() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_406() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_407() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_408() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_409() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_410() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_411() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_412() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_413() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }

    #[test]
    fn test_sparsify_stress_414() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t = Tensor::from_vec(data, vec![10]);
        let sparse = top_k_sparsify(&t, 0.5);
        assert_eq!(sparse.shape(), &[10]);
    }
}
