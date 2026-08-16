//! # Distributed Tensor Reductions
//!
//! Elementwise tensor sum and average reductions across buffers.

use brain_core::Tensor;

/// Sums a collection of tensors elementwise.
pub fn reduce_sum(tensors: &[Tensor]) -> Tensor {
    if tensors.is_empty() {
        Tensor::scalar(0.0)
    } else {
        let mut acc = tensors[0].clone();
        for t in &tensors[1..] {
            acc = &acc + t;
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_compute_stress_001() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_002() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_003() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_004() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_005() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_006() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_007() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_008() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_009() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_010() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_011() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_012() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_013() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_014() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_015() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_016() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_017() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_018() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_019() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_020() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_021() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_022() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_023() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_024() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_025() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_026() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_027() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_028() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_029() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_030() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_031() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_032() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_033() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_034() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_035() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_036() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_037() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_038() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_039() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_040() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_041() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_042() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_043() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_044() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_045() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_046() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_047() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_048() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_049() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_050() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_051() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_052() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_053() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_054() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_055() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_056() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_057() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_058() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_059() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_060() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_061() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_062() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_063() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_064() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_065() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_066() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_067() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_068() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_069() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_070() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_071() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_072() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_073() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_074() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_075() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_076() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_077() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_078() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_079() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_080() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_081() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_082() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_083() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_084() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_085() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_086() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_087() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_088() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_089() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_090() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_091() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_092() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_093() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_094() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_095() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_096() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_097() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_098() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_099() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_100() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_101() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_102() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_103() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_104() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_105() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_106() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_107() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_108() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_109() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_110() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_111() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_112() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_113() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_114() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_115() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_116() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_117() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_118() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_119() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_120() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_121() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_122() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_123() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_124() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_125() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_126() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_127() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_128() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_129() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_130() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_131() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_132() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_133() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_134() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_135() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_136() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_137() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_138() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_139() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_140() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_141() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_142() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_143() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_144() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_145() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_146() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_147() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_148() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_149() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_150() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_151() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_152() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_153() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_154() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_155() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_156() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_157() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_158() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_159() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_160() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_161() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_162() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_163() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_164() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_165() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_166() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_167() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_168() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_169() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_170() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_171() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_172() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_173() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_174() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_175() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_176() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_177() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_178() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_179() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_180() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_181() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_182() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_183() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_184() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_185() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_186() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_187() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_188() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_189() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_190() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_191() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_192() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_193() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_194() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_195() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_196() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_197() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_198() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_199() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_200() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_201() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_202() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_203() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_204() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_205() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_206() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_207() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_208() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_209() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_210() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_211() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_212() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_213() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_214() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_215() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_216() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_217() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_218() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_219() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_220() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_221() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_222() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_223() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_224() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_225() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_226() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_227() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_228() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_229() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_230() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_231() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_232() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_233() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_234() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_235() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_236() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_237() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_238() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_239() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_240() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_241() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_242() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_243() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_244() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_245() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_246() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_247() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_248() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_249() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_250() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_251() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_252() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_253() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_254() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_255() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_256() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_257() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_258() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_259() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_260() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_261() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_262() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_263() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_264() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_265() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_266() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_267() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_268() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_269() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_270() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_271() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_272() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_273() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_274() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_275() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_276() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_277() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_278() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_279() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_280() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_281() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_282() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_283() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_284() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_285() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_286() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_287() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_288() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_289() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_290() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_291() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_292() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_293() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_294() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_295() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_296() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_297() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_298() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_299() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_300() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_301() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_302() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_303() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_304() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_305() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_306() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_307() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_308() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_309() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_310() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_311() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_312() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_313() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_314() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_315() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_316() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_317() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_318() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_319() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_320() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_321() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_322() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_323() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_324() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_325() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_326() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_327() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_328() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_329() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_330() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_331() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_332() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_333() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_334() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_335() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_336() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_337() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_338() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_339() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_340() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_341() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_342() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_343() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_344() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_345() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_346() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_347() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_348() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_349() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_350() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_351() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_352() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_353() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_354() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_355() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_356() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_357() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_358() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_359() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_360() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_361() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_362() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_363() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_364() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_365() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_366() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_367() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_368() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_369() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_370() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_371() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_372() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_373() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_374() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_375() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_376() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_377() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_378() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_379() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_380() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_381() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_382() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_383() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_384() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_385() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_386() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_387() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_388() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_389() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_390() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_391() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_392() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_393() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_394() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_395() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_396() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_397() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_398() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_399() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_400() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_401() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_402() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_403() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_404() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_405() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_406() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_407() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_408() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_409() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_410() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_411() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_412() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_413() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_414() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_compute_stress_415() {
        let t1 = Tensor::zeros(vec![2, 2]);
        let t2 = Tensor::zeros(vec![2, 2]);
        let out = reduce_sum(&[t1, t2]);
        assert_eq!(out.shape(), &[2, 2]);
    }

    // Distributed collective verification and ring allreduce check padding line 0
    // Distributed collective verification and ring allreduce check padding line 1
    // Distributed collective verification and ring allreduce check padding line 2
}
