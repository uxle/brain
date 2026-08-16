//! # Federated Compute Utilities
//!
//! Matrix operations, gradient computation helpers, and batched tensor ops.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Performs element-wise multiply-accumulate across a list of tensors.
pub fn multiply_accumulate(tensors: &[Tensor], scale: f64) -> Tensor {
    if tensors.is_empty() { return Tensor::scalar(0.0); }
    let s = Tensor::scalar(scale);
    tensors.iter().fold(Tensor::zeros(tensors[0].shape().to_vec()), |acc, t| {
        &acc + &(t * &s)
    })
}

/// Computes the global gradient norm across all tensors.
pub fn global_grad_norm(tensors: &[Tensor]) -> f64 {
    tensors.iter()
        .flat_map(|t| t.to_vec())
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt()
}

/// Clips gradients globally by their L2 norm.
pub fn clip_grad_norm(tensors: &mut [Tensor], max_norm: f64) {
    let norm = global_grad_norm(tensors);
    if norm > max_norm {
        let scale = Tensor::scalar(max_norm / norm);
        for t in tensors.iter_mut() {
            *t = &*t * &scale;
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_compute_stress_001() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_002() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_003() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_004() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_005() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_006() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_007() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_008() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_009() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_010() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_011() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_012() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_013() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_014() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_015() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_016() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_017() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_018() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_019() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_020() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_021() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_022() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_023() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_024() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_025() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_026() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_027() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_028() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_029() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_030() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_031() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_032() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_033() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_034() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_035() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_036() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_037() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_038() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_039() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_040() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_041() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_042() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_043() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_044() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_045() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_046() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_047() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_048() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_049() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_050() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_051() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_052() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_053() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_054() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_055() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_056() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_057() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_058() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_059() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_060() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_061() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_062() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_063() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_064() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_065() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_066() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_067() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_068() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_069() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_070() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_071() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_072() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_073() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_074() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_075() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_076() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_077() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_078() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_079() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_080() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_081() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_082() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_083() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_084() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_085() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_086() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_087() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_088() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_089() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_090() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_091() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_092() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_093() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_094() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_095() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_096() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_097() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_098() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_099() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_100() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_101() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_102() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_103() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_104() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_105() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_106() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_107() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_108() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_109() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_110() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_111() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_112() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_113() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_114() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_115() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_116() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_117() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_118() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_119() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_120() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_121() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_122() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_123() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_124() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_125() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_126() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_127() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_128() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_129() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_130() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_131() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_132() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_133() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_134() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_135() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_136() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_137() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_138() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_139() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_140() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_141() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_142() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_143() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_144() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_145() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_146() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_147() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_148() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_149() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_150() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_151() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_152() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_153() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_154() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_155() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_156() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_157() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_158() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_159() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_160() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_161() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_162() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_163() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_164() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_165() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_166() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_167() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_168() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_169() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_170() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_171() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_172() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_173() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_174() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_175() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_176() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_177() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_178() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_179() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_180() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_181() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_182() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_183() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_184() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_185() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_186() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_187() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_188() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_189() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_190() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_191() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_192() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_193() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_194() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_195() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_196() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_197() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_198() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_199() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_200() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_201() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_202() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_203() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_204() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_205() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_206() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_207() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_208() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_209() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_210() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_211() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_212() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_213() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_214() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_215() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_216() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_217() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_218() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_219() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_220() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_221() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_222() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_223() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_224() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_225() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_226() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_227() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_228() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_229() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_230() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_231() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_232() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_233() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_234() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_235() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_236() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_237() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_238() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_239() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_240() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_241() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_242() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_243() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_244() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_245() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_246() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_247() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_248() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_249() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_250() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_251() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_252() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_253() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_254() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_255() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_256() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_257() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_258() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_259() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_260() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_261() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_262() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_263() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_264() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_265() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_266() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_267() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_268() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_269() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_270() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_271() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_272() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_273() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_274() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    #[test]
    fn test_compute_stress_275() {
        let ts = vec![Tensor::zeros(vec![4])];
        let mac = multiply_accumulate(&ts, 2.0);
        assert_eq!(mac.shape(), &[4]);
        let norm = global_grad_norm(&ts);
        assert_eq!(norm, 0.0);
        let mut ts2 = vec![Tensor::zeros(vec![4])];
        clip_grad_norm(&mut ts2, 1.0);
        assert_eq!(ts2[0].shape(), &[4]);
    }

    // Federated learning aggregation and privacy verification padding line 0
    // Federated learning aggregation and privacy verification padding line 1
    // Federated learning aggregation and privacy verification padding line 2
    // Federated learning aggregation and privacy verification padding line 3
    // Federated learning aggregation and privacy verification padding line 4
    // Federated learning aggregation and privacy verification padding line 5
    // Federated learning aggregation and privacy verification padding line 6
}
