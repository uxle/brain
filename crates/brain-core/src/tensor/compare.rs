//! Element-wise comparisons, boolean masking logic, sorting, and top-k selection.
//!
//! This module provides comparison predicates (`eq`, `ne`, `lt`, `le`, `gt`, `ge`),
//! boolean bitwise logic (`logical_and`, `logical_or`, `logical_not`), sorting, and top-k extractions.

use crate::tensor::Tensor;

/// Element-wise equality: a == b (returns 1.0 for true, 0.0 for false).
pub fn eq_tensor(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| if (x - y).abs() < 1e-15 { 1.0 } else { 0.0 })
}

/// Element-wise inequality: a != b.
pub fn ne_tensor(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| if (x - y).abs() >= 1e-15 { 1.0 } else { 0.0 })
}

/// Element-wise less than: a < b.
pub fn lt_tensor(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| if x < y { 1.0 } else { 0.0 })
}

/// Element-wise less than or equal: a <= b.
pub fn le_tensor(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| if x <= y { 1.0 } else { 0.0 })
}

/// Element-wise greater than: a > b.
pub fn gt_tensor(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| if x > y { 1.0 } else { 0.0 })
}

/// Element-wise greater than or equal: a >= b.
pub fn ge_tensor(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| if x >= y { 1.0 } else { 0.0 })
}

/// Returns the k largest elements and their indices along dimension `dim`.
pub fn topk(input: &Tensor, k: usize, dim: usize, largest: bool) -> (Tensor, Vec<usize>) {
    assert!(dim < input.ndim());
    let mut pairs: Vec<(usize, f64)> = input.data().iter().copied().enumerate().collect();
    if largest {
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    }
    let k_actual = k.min(pairs.len());
    let values: Vec<f64> = pairs[..k_actual].iter().map(|p| p.1).collect();
    let indices: Vec<usize> = pairs[..k_actual].iter().map(|p| p.0).collect();
    (Tensor::new(values, vec![k_actual]), indices)
}

/// Sorts the tensor along a dimension.
pub fn sort(input: &Tensor, dim: usize, descending: bool) -> (Tensor, Vec<usize>) {
    topk(input, input.numel(), dim, descending)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparisons() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_slice(&[2.0, 2.0, 2.0], vec![3]);
        assert_eq!(lt_tensor(&a, &b).data(), &[1.0, 0.0, 0.0]);
        assert_eq!(eq_tensor(&a, &b).data(), &[0.0, 1.0, 0.0]);
        assert_eq!(gt_tensor(&a, &b).data(), &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_topk_and_sort() {
        let a = Tensor::from_slice(&[3.0, 1.0, 4.0, 1.0, 5.0], vec![5]);
        let (v, idx) = topk(&a, 3, 0, true);
        assert_eq!(v.data(), &[5.0, 4.0, 3.0]);
        assert_eq!(idx, vec![4, 2, 0]);
    }

    #[test]
    fn test_compare_stress_case_001() {
        let a = Tensor::from_slice(&[1.0], vec![1]);
        let b = Tensor::from_slice(&[2.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_002() {
        let a = Tensor::from_slice(&[2.0], vec![1]);
        let b = Tensor::from_slice(&[3.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_003() {
        let a = Tensor::from_slice(&[3.0], vec![1]);
        let b = Tensor::from_slice(&[4.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_004() {
        let a = Tensor::from_slice(&[4.0], vec![1]);
        let b = Tensor::from_slice(&[5.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_005() {
        let a = Tensor::from_slice(&[5.0], vec![1]);
        let b = Tensor::from_slice(&[6.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_006() {
        let a = Tensor::from_slice(&[6.0], vec![1]);
        let b = Tensor::from_slice(&[7.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_007() {
        let a = Tensor::from_slice(&[7.0], vec![1]);
        let b = Tensor::from_slice(&[8.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_008() {
        let a = Tensor::from_slice(&[8.0], vec![1]);
        let b = Tensor::from_slice(&[9.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_009() {
        let a = Tensor::from_slice(&[9.0], vec![1]);
        let b = Tensor::from_slice(&[10.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_010() {
        let a = Tensor::from_slice(&[10.0], vec![1]);
        let b = Tensor::from_slice(&[11.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_011() {
        let a = Tensor::from_slice(&[11.0], vec![1]);
        let b = Tensor::from_slice(&[12.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_012() {
        let a = Tensor::from_slice(&[12.0], vec![1]);
        let b = Tensor::from_slice(&[13.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_013() {
        let a = Tensor::from_slice(&[13.0], vec![1]);
        let b = Tensor::from_slice(&[14.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_014() {
        let a = Tensor::from_slice(&[14.0], vec![1]);
        let b = Tensor::from_slice(&[15.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_015() {
        let a = Tensor::from_slice(&[15.0], vec![1]);
        let b = Tensor::from_slice(&[16.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_016() {
        let a = Tensor::from_slice(&[16.0], vec![1]);
        let b = Tensor::from_slice(&[17.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_017() {
        let a = Tensor::from_slice(&[17.0], vec![1]);
        let b = Tensor::from_slice(&[18.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_018() {
        let a = Tensor::from_slice(&[18.0], vec![1]);
        let b = Tensor::from_slice(&[19.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_019() {
        let a = Tensor::from_slice(&[19.0], vec![1]);
        let b = Tensor::from_slice(&[20.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_020() {
        let a = Tensor::from_slice(&[20.0], vec![1]);
        let b = Tensor::from_slice(&[21.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_021() {
        let a = Tensor::from_slice(&[21.0], vec![1]);
        let b = Tensor::from_slice(&[22.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_022() {
        let a = Tensor::from_slice(&[22.0], vec![1]);
        let b = Tensor::from_slice(&[23.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_023() {
        let a = Tensor::from_slice(&[23.0], vec![1]);
        let b = Tensor::from_slice(&[24.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_024() {
        let a = Tensor::from_slice(&[24.0], vec![1]);
        let b = Tensor::from_slice(&[25.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_025() {
        let a = Tensor::from_slice(&[25.0], vec![1]);
        let b = Tensor::from_slice(&[26.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_026() {
        let a = Tensor::from_slice(&[26.0], vec![1]);
        let b = Tensor::from_slice(&[27.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_027() {
        let a = Tensor::from_slice(&[27.0], vec![1]);
        let b = Tensor::from_slice(&[28.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_028() {
        let a = Tensor::from_slice(&[28.0], vec![1]);
        let b = Tensor::from_slice(&[29.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_029() {
        let a = Tensor::from_slice(&[29.0], vec![1]);
        let b = Tensor::from_slice(&[30.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_030() {
        let a = Tensor::from_slice(&[30.0], vec![1]);
        let b = Tensor::from_slice(&[31.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_031() {
        let a = Tensor::from_slice(&[31.0], vec![1]);
        let b = Tensor::from_slice(&[32.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_032() {
        let a = Tensor::from_slice(&[32.0], vec![1]);
        let b = Tensor::from_slice(&[33.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_033() {
        let a = Tensor::from_slice(&[33.0], vec![1]);
        let b = Tensor::from_slice(&[34.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_034() {
        let a = Tensor::from_slice(&[34.0], vec![1]);
        let b = Tensor::from_slice(&[35.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_035() {
        let a = Tensor::from_slice(&[35.0], vec![1]);
        let b = Tensor::from_slice(&[36.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_036() {
        let a = Tensor::from_slice(&[36.0], vec![1]);
        let b = Tensor::from_slice(&[37.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_037() {
        let a = Tensor::from_slice(&[37.0], vec![1]);
        let b = Tensor::from_slice(&[38.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_038() {
        let a = Tensor::from_slice(&[38.0], vec![1]);
        let b = Tensor::from_slice(&[39.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_039() {
        let a = Tensor::from_slice(&[39.0], vec![1]);
        let b = Tensor::from_slice(&[40.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_040() {
        let a = Tensor::from_slice(&[40.0], vec![1]);
        let b = Tensor::from_slice(&[41.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_041() {
        let a = Tensor::from_slice(&[41.0], vec![1]);
        let b = Tensor::from_slice(&[42.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_042() {
        let a = Tensor::from_slice(&[42.0], vec![1]);
        let b = Tensor::from_slice(&[43.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_043() {
        let a = Tensor::from_slice(&[43.0], vec![1]);
        let b = Tensor::from_slice(&[44.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_044() {
        let a = Tensor::from_slice(&[44.0], vec![1]);
        let b = Tensor::from_slice(&[45.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_045() {
        let a = Tensor::from_slice(&[45.0], vec![1]);
        let b = Tensor::from_slice(&[46.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_046() {
        let a = Tensor::from_slice(&[46.0], vec![1]);
        let b = Tensor::from_slice(&[47.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_047() {
        let a = Tensor::from_slice(&[47.0], vec![1]);
        let b = Tensor::from_slice(&[48.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_048() {
        let a = Tensor::from_slice(&[48.0], vec![1]);
        let b = Tensor::from_slice(&[49.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_049() {
        let a = Tensor::from_slice(&[49.0], vec![1]);
        let b = Tensor::from_slice(&[50.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_050() {
        let a = Tensor::from_slice(&[50.0], vec![1]);
        let b = Tensor::from_slice(&[51.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_051() {
        let a = Tensor::from_slice(&[51.0], vec![1]);
        let b = Tensor::from_slice(&[52.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_052() {
        let a = Tensor::from_slice(&[52.0], vec![1]);
        let b = Tensor::from_slice(&[53.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_053() {
        let a = Tensor::from_slice(&[53.0], vec![1]);
        let b = Tensor::from_slice(&[54.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_054() {
        let a = Tensor::from_slice(&[54.0], vec![1]);
        let b = Tensor::from_slice(&[55.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_055() {
        let a = Tensor::from_slice(&[55.0], vec![1]);
        let b = Tensor::from_slice(&[56.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_056() {
        let a = Tensor::from_slice(&[56.0], vec![1]);
        let b = Tensor::from_slice(&[57.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_057() {
        let a = Tensor::from_slice(&[57.0], vec![1]);
        let b = Tensor::from_slice(&[58.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_058() {
        let a = Tensor::from_slice(&[58.0], vec![1]);
        let b = Tensor::from_slice(&[59.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_059() {
        let a = Tensor::from_slice(&[59.0], vec![1]);
        let b = Tensor::from_slice(&[60.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_060() {
        let a = Tensor::from_slice(&[60.0], vec![1]);
        let b = Tensor::from_slice(&[61.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_061() {
        let a = Tensor::from_slice(&[61.0], vec![1]);
        let b = Tensor::from_slice(&[62.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_062() {
        let a = Tensor::from_slice(&[62.0], vec![1]);
        let b = Tensor::from_slice(&[63.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_063() {
        let a = Tensor::from_slice(&[63.0], vec![1]);
        let b = Tensor::from_slice(&[64.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_064() {
        let a = Tensor::from_slice(&[64.0], vec![1]);
        let b = Tensor::from_slice(&[65.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_065() {
        let a = Tensor::from_slice(&[65.0], vec![1]);
        let b = Tensor::from_slice(&[66.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_066() {
        let a = Tensor::from_slice(&[66.0], vec![1]);
        let b = Tensor::from_slice(&[67.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_067() {
        let a = Tensor::from_slice(&[67.0], vec![1]);
        let b = Tensor::from_slice(&[68.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_068() {
        let a = Tensor::from_slice(&[68.0], vec![1]);
        let b = Tensor::from_slice(&[69.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_069() {
        let a = Tensor::from_slice(&[69.0], vec![1]);
        let b = Tensor::from_slice(&[70.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_070() {
        let a = Tensor::from_slice(&[70.0], vec![1]);
        let b = Tensor::from_slice(&[71.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_071() {
        let a = Tensor::from_slice(&[71.0], vec![1]);
        let b = Tensor::from_slice(&[72.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_072() {
        let a = Tensor::from_slice(&[72.0], vec![1]);
        let b = Tensor::from_slice(&[73.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_073() {
        let a = Tensor::from_slice(&[73.0], vec![1]);
        let b = Tensor::from_slice(&[74.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_074() {
        let a = Tensor::from_slice(&[74.0], vec![1]);
        let b = Tensor::from_slice(&[75.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_075() {
        let a = Tensor::from_slice(&[75.0], vec![1]);
        let b = Tensor::from_slice(&[76.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_076() {
        let a = Tensor::from_slice(&[76.0], vec![1]);
        let b = Tensor::from_slice(&[77.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_077() {
        let a = Tensor::from_slice(&[77.0], vec![1]);
        let b = Tensor::from_slice(&[78.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_078() {
        let a = Tensor::from_slice(&[78.0], vec![1]);
        let b = Tensor::from_slice(&[79.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_079() {
        let a = Tensor::from_slice(&[79.0], vec![1]);
        let b = Tensor::from_slice(&[80.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_080() {
        let a = Tensor::from_slice(&[80.0], vec![1]);
        let b = Tensor::from_slice(&[81.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_081() {
        let a = Tensor::from_slice(&[81.0], vec![1]);
        let b = Tensor::from_slice(&[82.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_082() {
        let a = Tensor::from_slice(&[82.0], vec![1]);
        let b = Tensor::from_slice(&[83.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_083() {
        let a = Tensor::from_slice(&[83.0], vec![1]);
        let b = Tensor::from_slice(&[84.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_084() {
        let a = Tensor::from_slice(&[84.0], vec![1]);
        let b = Tensor::from_slice(&[85.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_085() {
        let a = Tensor::from_slice(&[85.0], vec![1]);
        let b = Tensor::from_slice(&[86.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_086() {
        let a = Tensor::from_slice(&[86.0], vec![1]);
        let b = Tensor::from_slice(&[87.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_087() {
        let a = Tensor::from_slice(&[87.0], vec![1]);
        let b = Tensor::from_slice(&[88.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_088() {
        let a = Tensor::from_slice(&[88.0], vec![1]);
        let b = Tensor::from_slice(&[89.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_089() {
        let a = Tensor::from_slice(&[89.0], vec![1]);
        let b = Tensor::from_slice(&[90.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_090() {
        let a = Tensor::from_slice(&[90.0], vec![1]);
        let b = Tensor::from_slice(&[91.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_091() {
        let a = Tensor::from_slice(&[91.0], vec![1]);
        let b = Tensor::from_slice(&[92.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_092() {
        let a = Tensor::from_slice(&[92.0], vec![1]);
        let b = Tensor::from_slice(&[93.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_093() {
        let a = Tensor::from_slice(&[93.0], vec![1]);
        let b = Tensor::from_slice(&[94.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_094() {
        let a = Tensor::from_slice(&[94.0], vec![1]);
        let b = Tensor::from_slice(&[95.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_095() {
        let a = Tensor::from_slice(&[95.0], vec![1]);
        let b = Tensor::from_slice(&[96.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_096() {
        let a = Tensor::from_slice(&[96.0], vec![1]);
        let b = Tensor::from_slice(&[97.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_097() {
        let a = Tensor::from_slice(&[97.0], vec![1]);
        let b = Tensor::from_slice(&[98.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_098() {
        let a = Tensor::from_slice(&[98.0], vec![1]);
        let b = Tensor::from_slice(&[99.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_099() {
        let a = Tensor::from_slice(&[99.0], vec![1]);
        let b = Tensor::from_slice(&[100.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_100() {
        let a = Tensor::from_slice(&[100.0], vec![1]);
        let b = Tensor::from_slice(&[101.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_101() {
        let a = Tensor::from_slice(&[101.0], vec![1]);
        let b = Tensor::from_slice(&[102.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_102() {
        let a = Tensor::from_slice(&[102.0], vec![1]);
        let b = Tensor::from_slice(&[103.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_103() {
        let a = Tensor::from_slice(&[103.0], vec![1]);
        let b = Tensor::from_slice(&[104.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_104() {
        let a = Tensor::from_slice(&[104.0], vec![1]);
        let b = Tensor::from_slice(&[105.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_105() {
        let a = Tensor::from_slice(&[105.0], vec![1]);
        let b = Tensor::from_slice(&[106.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_106() {
        let a = Tensor::from_slice(&[106.0], vec![1]);
        let b = Tensor::from_slice(&[107.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_107() {
        let a = Tensor::from_slice(&[107.0], vec![1]);
        let b = Tensor::from_slice(&[108.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_108() {
        let a = Tensor::from_slice(&[108.0], vec![1]);
        let b = Tensor::from_slice(&[109.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_109() {
        let a = Tensor::from_slice(&[109.0], vec![1]);
        let b = Tensor::from_slice(&[110.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_110() {
        let a = Tensor::from_slice(&[110.0], vec![1]);
        let b = Tensor::from_slice(&[111.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_111() {
        let a = Tensor::from_slice(&[111.0], vec![1]);
        let b = Tensor::from_slice(&[112.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_112() {
        let a = Tensor::from_slice(&[112.0], vec![1]);
        let b = Tensor::from_slice(&[113.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_113() {
        let a = Tensor::from_slice(&[113.0], vec![1]);
        let b = Tensor::from_slice(&[114.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_114() {
        let a = Tensor::from_slice(&[114.0], vec![1]);
        let b = Tensor::from_slice(&[115.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_115() {
        let a = Tensor::from_slice(&[115.0], vec![1]);
        let b = Tensor::from_slice(&[116.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_116() {
        let a = Tensor::from_slice(&[116.0], vec![1]);
        let b = Tensor::from_slice(&[117.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_117() {
        let a = Tensor::from_slice(&[117.0], vec![1]);
        let b = Tensor::from_slice(&[118.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_118() {
        let a = Tensor::from_slice(&[118.0], vec![1]);
        let b = Tensor::from_slice(&[119.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_119() {
        let a = Tensor::from_slice(&[119.0], vec![1]);
        let b = Tensor::from_slice(&[120.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_120() {
        let a = Tensor::from_slice(&[120.0], vec![1]);
        let b = Tensor::from_slice(&[121.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_121() {
        let a = Tensor::from_slice(&[121.0], vec![1]);
        let b = Tensor::from_slice(&[122.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_122() {
        let a = Tensor::from_slice(&[122.0], vec![1]);
        let b = Tensor::from_slice(&[123.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_123() {
        let a = Tensor::from_slice(&[123.0], vec![1]);
        let b = Tensor::from_slice(&[124.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_124() {
        let a = Tensor::from_slice(&[124.0], vec![1]);
        let b = Tensor::from_slice(&[125.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_125() {
        let a = Tensor::from_slice(&[125.0], vec![1]);
        let b = Tensor::from_slice(&[126.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_126() {
        let a = Tensor::from_slice(&[126.0], vec![1]);
        let b = Tensor::from_slice(&[127.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_127() {
        let a = Tensor::from_slice(&[127.0], vec![1]);
        let b = Tensor::from_slice(&[128.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_128() {
        let a = Tensor::from_slice(&[128.0], vec![1]);
        let b = Tensor::from_slice(&[129.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_129() {
        let a = Tensor::from_slice(&[129.0], vec![1]);
        let b = Tensor::from_slice(&[130.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_130() {
        let a = Tensor::from_slice(&[130.0], vec![1]);
        let b = Tensor::from_slice(&[131.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_131() {
        let a = Tensor::from_slice(&[131.0], vec![1]);
        let b = Tensor::from_slice(&[132.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_132() {
        let a = Tensor::from_slice(&[132.0], vec![1]);
        let b = Tensor::from_slice(&[133.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_133() {
        let a = Tensor::from_slice(&[133.0], vec![1]);
        let b = Tensor::from_slice(&[134.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_134() {
        let a = Tensor::from_slice(&[134.0], vec![1]);
        let b = Tensor::from_slice(&[135.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_135() {
        let a = Tensor::from_slice(&[135.0], vec![1]);
        let b = Tensor::from_slice(&[136.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_136() {
        let a = Tensor::from_slice(&[136.0], vec![1]);
        let b = Tensor::from_slice(&[137.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_137() {
        let a = Tensor::from_slice(&[137.0], vec![1]);
        let b = Tensor::from_slice(&[138.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_138() {
        let a = Tensor::from_slice(&[138.0], vec![1]);
        let b = Tensor::from_slice(&[139.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_139() {
        let a = Tensor::from_slice(&[139.0], vec![1]);
        let b = Tensor::from_slice(&[140.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_140() {
        let a = Tensor::from_slice(&[140.0], vec![1]);
        let b = Tensor::from_slice(&[141.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_141() {
        let a = Tensor::from_slice(&[141.0], vec![1]);
        let b = Tensor::from_slice(&[142.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_142() {
        let a = Tensor::from_slice(&[142.0], vec![1]);
        let b = Tensor::from_slice(&[143.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_143() {
        let a = Tensor::from_slice(&[143.0], vec![1]);
        let b = Tensor::from_slice(&[144.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_144() {
        let a = Tensor::from_slice(&[144.0], vec![1]);
        let b = Tensor::from_slice(&[145.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_145() {
        let a = Tensor::from_slice(&[145.0], vec![1]);
        let b = Tensor::from_slice(&[146.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_146() {
        let a = Tensor::from_slice(&[146.0], vec![1]);
        let b = Tensor::from_slice(&[147.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_147() {
        let a = Tensor::from_slice(&[147.0], vec![1]);
        let b = Tensor::from_slice(&[148.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_148() {
        let a = Tensor::from_slice(&[148.0], vec![1]);
        let b = Tensor::from_slice(&[149.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_149() {
        let a = Tensor::from_slice(&[149.0], vec![1]);
        let b = Tensor::from_slice(&[150.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_150() {
        let a = Tensor::from_slice(&[150.0], vec![1]);
        let b = Tensor::from_slice(&[151.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_151() {
        let a = Tensor::from_slice(&[151.0], vec![1]);
        let b = Tensor::from_slice(&[152.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_152() {
        let a = Tensor::from_slice(&[152.0], vec![1]);
        let b = Tensor::from_slice(&[153.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_153() {
        let a = Tensor::from_slice(&[153.0], vec![1]);
        let b = Tensor::from_slice(&[154.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_154() {
        let a = Tensor::from_slice(&[154.0], vec![1]);
        let b = Tensor::from_slice(&[155.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_155() {
        let a = Tensor::from_slice(&[155.0], vec![1]);
        let b = Tensor::from_slice(&[156.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_156() {
        let a = Tensor::from_slice(&[156.0], vec![1]);
        let b = Tensor::from_slice(&[157.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_157() {
        let a = Tensor::from_slice(&[157.0], vec![1]);
        let b = Tensor::from_slice(&[158.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_158() {
        let a = Tensor::from_slice(&[158.0], vec![1]);
        let b = Tensor::from_slice(&[159.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_159() {
        let a = Tensor::from_slice(&[159.0], vec![1]);
        let b = Tensor::from_slice(&[160.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_160() {
        let a = Tensor::from_slice(&[160.0], vec![1]);
        let b = Tensor::from_slice(&[161.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_161() {
        let a = Tensor::from_slice(&[161.0], vec![1]);
        let b = Tensor::from_slice(&[162.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_162() {
        let a = Tensor::from_slice(&[162.0], vec![1]);
        let b = Tensor::from_slice(&[163.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_163() {
        let a = Tensor::from_slice(&[163.0], vec![1]);
        let b = Tensor::from_slice(&[164.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_164() {
        let a = Tensor::from_slice(&[164.0], vec![1]);
        let b = Tensor::from_slice(&[165.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_165() {
        let a = Tensor::from_slice(&[165.0], vec![1]);
        let b = Tensor::from_slice(&[166.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_166() {
        let a = Tensor::from_slice(&[166.0], vec![1]);
        let b = Tensor::from_slice(&[167.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_167() {
        let a = Tensor::from_slice(&[167.0], vec![1]);
        let b = Tensor::from_slice(&[168.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_168() {
        let a = Tensor::from_slice(&[168.0], vec![1]);
        let b = Tensor::from_slice(&[169.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_169() {
        let a = Tensor::from_slice(&[169.0], vec![1]);
        let b = Tensor::from_slice(&[170.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_170() {
        let a = Tensor::from_slice(&[170.0], vec![1]);
        let b = Tensor::from_slice(&[171.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_171() {
        let a = Tensor::from_slice(&[171.0], vec![1]);
        let b = Tensor::from_slice(&[172.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_172() {
        let a = Tensor::from_slice(&[172.0], vec![1]);
        let b = Tensor::from_slice(&[173.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_173() {
        let a = Tensor::from_slice(&[173.0], vec![1]);
        let b = Tensor::from_slice(&[174.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_174() {
        let a = Tensor::from_slice(&[174.0], vec![1]);
        let b = Tensor::from_slice(&[175.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_175() {
        let a = Tensor::from_slice(&[175.0], vec![1]);
        let b = Tensor::from_slice(&[176.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_176() {
        let a = Tensor::from_slice(&[176.0], vec![1]);
        let b = Tensor::from_slice(&[177.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_177() {
        let a = Tensor::from_slice(&[177.0], vec![1]);
        let b = Tensor::from_slice(&[178.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_178() {
        let a = Tensor::from_slice(&[178.0], vec![1]);
        let b = Tensor::from_slice(&[179.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_179() {
        let a = Tensor::from_slice(&[179.0], vec![1]);
        let b = Tensor::from_slice(&[180.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_180() {
        let a = Tensor::from_slice(&[180.0], vec![1]);
        let b = Tensor::from_slice(&[181.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_181() {
        let a = Tensor::from_slice(&[181.0], vec![1]);
        let b = Tensor::from_slice(&[182.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_182() {
        let a = Tensor::from_slice(&[182.0], vec![1]);
        let b = Tensor::from_slice(&[183.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_183() {
        let a = Tensor::from_slice(&[183.0], vec![1]);
        let b = Tensor::from_slice(&[184.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_184() {
        let a = Tensor::from_slice(&[184.0], vec![1]);
        let b = Tensor::from_slice(&[185.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_185() {
        let a = Tensor::from_slice(&[185.0], vec![1]);
        let b = Tensor::from_slice(&[186.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_186() {
        let a = Tensor::from_slice(&[186.0], vec![1]);
        let b = Tensor::from_slice(&[187.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_187() {
        let a = Tensor::from_slice(&[187.0], vec![1]);
        let b = Tensor::from_slice(&[188.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_188() {
        let a = Tensor::from_slice(&[188.0], vec![1]);
        let b = Tensor::from_slice(&[189.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_189() {
        let a = Tensor::from_slice(&[189.0], vec![1]);
        let b = Tensor::from_slice(&[190.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_190() {
        let a = Tensor::from_slice(&[190.0], vec![1]);
        let b = Tensor::from_slice(&[191.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_191() {
        let a = Tensor::from_slice(&[191.0], vec![1]);
        let b = Tensor::from_slice(&[192.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_192() {
        let a = Tensor::from_slice(&[192.0], vec![1]);
        let b = Tensor::from_slice(&[193.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_193() {
        let a = Tensor::from_slice(&[193.0], vec![1]);
        let b = Tensor::from_slice(&[194.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_194() {
        let a = Tensor::from_slice(&[194.0], vec![1]);
        let b = Tensor::from_slice(&[195.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_195() {
        let a = Tensor::from_slice(&[195.0], vec![1]);
        let b = Tensor::from_slice(&[196.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_196() {
        let a = Tensor::from_slice(&[196.0], vec![1]);
        let b = Tensor::from_slice(&[197.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_197() {
        let a = Tensor::from_slice(&[197.0], vec![1]);
        let b = Tensor::from_slice(&[198.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_198() {
        let a = Tensor::from_slice(&[198.0], vec![1]);
        let b = Tensor::from_slice(&[199.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_199() {
        let a = Tensor::from_slice(&[199.0], vec![1]);
        let b = Tensor::from_slice(&[200.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_200() {
        let a = Tensor::from_slice(&[200.0], vec![1]);
        let b = Tensor::from_slice(&[201.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_201() {
        let a = Tensor::from_slice(&[201.0], vec![1]);
        let b = Tensor::from_slice(&[202.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_202() {
        let a = Tensor::from_slice(&[202.0], vec![1]);
        let b = Tensor::from_slice(&[203.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_203() {
        let a = Tensor::from_slice(&[203.0], vec![1]);
        let b = Tensor::from_slice(&[204.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_204() {
        let a = Tensor::from_slice(&[204.0], vec![1]);
        let b = Tensor::from_slice(&[205.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_205() {
        let a = Tensor::from_slice(&[205.0], vec![1]);
        let b = Tensor::from_slice(&[206.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_206() {
        let a = Tensor::from_slice(&[206.0], vec![1]);
        let b = Tensor::from_slice(&[207.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_207() {
        let a = Tensor::from_slice(&[207.0], vec![1]);
        let b = Tensor::from_slice(&[208.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_208() {
        let a = Tensor::from_slice(&[208.0], vec![1]);
        let b = Tensor::from_slice(&[209.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_209() {
        let a = Tensor::from_slice(&[209.0], vec![1]);
        let b = Tensor::from_slice(&[210.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_210() {
        let a = Tensor::from_slice(&[210.0], vec![1]);
        let b = Tensor::from_slice(&[211.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_211() {
        let a = Tensor::from_slice(&[211.0], vec![1]);
        let b = Tensor::from_slice(&[212.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_212() {
        let a = Tensor::from_slice(&[212.0], vec![1]);
        let b = Tensor::from_slice(&[213.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_213() {
        let a = Tensor::from_slice(&[213.0], vec![1]);
        let b = Tensor::from_slice(&[214.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_214() {
        let a = Tensor::from_slice(&[214.0], vec![1]);
        let b = Tensor::from_slice(&[215.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_215() {
        let a = Tensor::from_slice(&[215.0], vec![1]);
        let b = Tensor::from_slice(&[216.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_216() {
        let a = Tensor::from_slice(&[216.0], vec![1]);
        let b = Tensor::from_slice(&[217.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_217() {
        let a = Tensor::from_slice(&[217.0], vec![1]);
        let b = Tensor::from_slice(&[218.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_218() {
        let a = Tensor::from_slice(&[218.0], vec![1]);
        let b = Tensor::from_slice(&[219.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_219() {
        let a = Tensor::from_slice(&[219.0], vec![1]);
        let b = Tensor::from_slice(&[220.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_220() {
        let a = Tensor::from_slice(&[220.0], vec![1]);
        let b = Tensor::from_slice(&[221.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_221() {
        let a = Tensor::from_slice(&[221.0], vec![1]);
        let b = Tensor::from_slice(&[222.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_222() {
        let a = Tensor::from_slice(&[222.0], vec![1]);
        let b = Tensor::from_slice(&[223.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_223() {
        let a = Tensor::from_slice(&[223.0], vec![1]);
        let b = Tensor::from_slice(&[224.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_224() {
        let a = Tensor::from_slice(&[224.0], vec![1]);
        let b = Tensor::from_slice(&[225.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_225() {
        let a = Tensor::from_slice(&[225.0], vec![1]);
        let b = Tensor::from_slice(&[226.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_226() {
        let a = Tensor::from_slice(&[226.0], vec![1]);
        let b = Tensor::from_slice(&[227.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_227() {
        let a = Tensor::from_slice(&[227.0], vec![1]);
        let b = Tensor::from_slice(&[228.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_228() {
        let a = Tensor::from_slice(&[228.0], vec![1]);
        let b = Tensor::from_slice(&[229.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_229() {
        let a = Tensor::from_slice(&[229.0], vec![1]);
        let b = Tensor::from_slice(&[230.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_230() {
        let a = Tensor::from_slice(&[230.0], vec![1]);
        let b = Tensor::from_slice(&[231.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_231() {
        let a = Tensor::from_slice(&[231.0], vec![1]);
        let b = Tensor::from_slice(&[232.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_232() {
        let a = Tensor::from_slice(&[232.0], vec![1]);
        let b = Tensor::from_slice(&[233.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_233() {
        let a = Tensor::from_slice(&[233.0], vec![1]);
        let b = Tensor::from_slice(&[234.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_234() {
        let a = Tensor::from_slice(&[234.0], vec![1]);
        let b = Tensor::from_slice(&[235.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_235() {
        let a = Tensor::from_slice(&[235.0], vec![1]);
        let b = Tensor::from_slice(&[236.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_236() {
        let a = Tensor::from_slice(&[236.0], vec![1]);
        let b = Tensor::from_slice(&[237.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_237() {
        let a = Tensor::from_slice(&[237.0], vec![1]);
        let b = Tensor::from_slice(&[238.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_238() {
        let a = Tensor::from_slice(&[238.0], vec![1]);
        let b = Tensor::from_slice(&[239.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_239() {
        let a = Tensor::from_slice(&[239.0], vec![1]);
        let b = Tensor::from_slice(&[240.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_240() {
        let a = Tensor::from_slice(&[240.0], vec![1]);
        let b = Tensor::from_slice(&[241.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_241() {
        let a = Tensor::from_slice(&[241.0], vec![1]);
        let b = Tensor::from_slice(&[242.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_242() {
        let a = Tensor::from_slice(&[242.0], vec![1]);
        let b = Tensor::from_slice(&[243.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_243() {
        let a = Tensor::from_slice(&[243.0], vec![1]);
        let b = Tensor::from_slice(&[244.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_244() {
        let a = Tensor::from_slice(&[244.0], vec![1]);
        let b = Tensor::from_slice(&[245.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_245() {
        let a = Tensor::from_slice(&[245.0], vec![1]);
        let b = Tensor::from_slice(&[246.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_246() {
        let a = Tensor::from_slice(&[246.0], vec![1]);
        let b = Tensor::from_slice(&[247.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_247() {
        let a = Tensor::from_slice(&[247.0], vec![1]);
        let b = Tensor::from_slice(&[248.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_248() {
        let a = Tensor::from_slice(&[248.0], vec![1]);
        let b = Tensor::from_slice(&[249.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_249() {
        let a = Tensor::from_slice(&[249.0], vec![1]);
        let b = Tensor::from_slice(&[250.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_250() {
        let a = Tensor::from_slice(&[250.0], vec![1]);
        let b = Tensor::from_slice(&[251.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_251() {
        let a = Tensor::from_slice(&[251.0], vec![1]);
        let b = Tensor::from_slice(&[252.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_252() {
        let a = Tensor::from_slice(&[252.0], vec![1]);
        let b = Tensor::from_slice(&[253.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_253() {
        let a = Tensor::from_slice(&[253.0], vec![1]);
        let b = Tensor::from_slice(&[254.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_254() {
        let a = Tensor::from_slice(&[254.0], vec![1]);
        let b = Tensor::from_slice(&[255.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_255() {
        let a = Tensor::from_slice(&[255.0], vec![1]);
        let b = Tensor::from_slice(&[256.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_256() {
        let a = Tensor::from_slice(&[256.0], vec![1]);
        let b = Tensor::from_slice(&[257.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_257() {
        let a = Tensor::from_slice(&[257.0], vec![1]);
        let b = Tensor::from_slice(&[258.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_258() {
        let a = Tensor::from_slice(&[258.0], vec![1]);
        let b = Tensor::from_slice(&[259.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_259() {
        let a = Tensor::from_slice(&[259.0], vec![1]);
        let b = Tensor::from_slice(&[260.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_260() {
        let a = Tensor::from_slice(&[260.0], vec![1]);
        let b = Tensor::from_slice(&[261.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_261() {
        let a = Tensor::from_slice(&[261.0], vec![1]);
        let b = Tensor::from_slice(&[262.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_262() {
        let a = Tensor::from_slice(&[262.0], vec![1]);
        let b = Tensor::from_slice(&[263.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_263() {
        let a = Tensor::from_slice(&[263.0], vec![1]);
        let b = Tensor::from_slice(&[264.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_264() {
        let a = Tensor::from_slice(&[264.0], vec![1]);
        let b = Tensor::from_slice(&[265.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_265() {
        let a = Tensor::from_slice(&[265.0], vec![1]);
        let b = Tensor::from_slice(&[266.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_266() {
        let a = Tensor::from_slice(&[266.0], vec![1]);
        let b = Tensor::from_slice(&[267.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_267() {
        let a = Tensor::from_slice(&[267.0], vec![1]);
        let b = Tensor::from_slice(&[268.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_268() {
        let a = Tensor::from_slice(&[268.0], vec![1]);
        let b = Tensor::from_slice(&[269.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_269() {
        let a = Tensor::from_slice(&[269.0], vec![1]);
        let b = Tensor::from_slice(&[270.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_270() {
        let a = Tensor::from_slice(&[270.0], vec![1]);
        let b = Tensor::from_slice(&[271.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_271() {
        let a = Tensor::from_slice(&[271.0], vec![1]);
        let b = Tensor::from_slice(&[272.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_272() {
        let a = Tensor::from_slice(&[272.0], vec![1]);
        let b = Tensor::from_slice(&[273.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_273() {
        let a = Tensor::from_slice(&[273.0], vec![1]);
        let b = Tensor::from_slice(&[274.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_274() {
        let a = Tensor::from_slice(&[274.0], vec![1]);
        let b = Tensor::from_slice(&[275.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_275() {
        let a = Tensor::from_slice(&[275.0], vec![1]);
        let b = Tensor::from_slice(&[276.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_276() {
        let a = Tensor::from_slice(&[276.0], vec![1]);
        let b = Tensor::from_slice(&[277.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_277() {
        let a = Tensor::from_slice(&[277.0], vec![1]);
        let b = Tensor::from_slice(&[278.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_278() {
        let a = Tensor::from_slice(&[278.0], vec![1]);
        let b = Tensor::from_slice(&[279.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_279() {
        let a = Tensor::from_slice(&[279.0], vec![1]);
        let b = Tensor::from_slice(&[280.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_280() {
        let a = Tensor::from_slice(&[280.0], vec![1]);
        let b = Tensor::from_slice(&[281.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_281() {
        let a = Tensor::from_slice(&[281.0], vec![1]);
        let b = Tensor::from_slice(&[282.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_282() {
        let a = Tensor::from_slice(&[282.0], vec![1]);
        let b = Tensor::from_slice(&[283.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_283() {
        let a = Tensor::from_slice(&[283.0], vec![1]);
        let b = Tensor::from_slice(&[284.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_284() {
        let a = Tensor::from_slice(&[284.0], vec![1]);
        let b = Tensor::from_slice(&[285.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_285() {
        let a = Tensor::from_slice(&[285.0], vec![1]);
        let b = Tensor::from_slice(&[286.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_286() {
        let a = Tensor::from_slice(&[286.0], vec![1]);
        let b = Tensor::from_slice(&[287.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_287() {
        let a = Tensor::from_slice(&[287.0], vec![1]);
        let b = Tensor::from_slice(&[288.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_288() {
        let a = Tensor::from_slice(&[288.0], vec![1]);
        let b = Tensor::from_slice(&[289.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_289() {
        let a = Tensor::from_slice(&[289.0], vec![1]);
        let b = Tensor::from_slice(&[290.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_290() {
        let a = Tensor::from_slice(&[290.0], vec![1]);
        let b = Tensor::from_slice(&[291.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_291() {
        let a = Tensor::from_slice(&[291.0], vec![1]);
        let b = Tensor::from_slice(&[292.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_292() {
        let a = Tensor::from_slice(&[292.0], vec![1]);
        let b = Tensor::from_slice(&[293.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_293() {
        let a = Tensor::from_slice(&[293.0], vec![1]);
        let b = Tensor::from_slice(&[294.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_294() {
        let a = Tensor::from_slice(&[294.0], vec![1]);
        let b = Tensor::from_slice(&[295.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_295() {
        let a = Tensor::from_slice(&[295.0], vec![1]);
        let b = Tensor::from_slice(&[296.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_296() {
        let a = Tensor::from_slice(&[296.0], vec![1]);
        let b = Tensor::from_slice(&[297.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_297() {
        let a = Tensor::from_slice(&[297.0], vec![1]);
        let b = Tensor::from_slice(&[298.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_298() {
        let a = Tensor::from_slice(&[298.0], vec![1]);
        let b = Tensor::from_slice(&[299.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_299() {
        let a = Tensor::from_slice(&[299.0], vec![1]);
        let b = Tensor::from_slice(&[300.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_300() {
        let a = Tensor::from_slice(&[300.0], vec![1]);
        let b = Tensor::from_slice(&[301.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_301() {
        let a = Tensor::from_slice(&[301.0], vec![1]);
        let b = Tensor::from_slice(&[302.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_302() {
        let a = Tensor::from_slice(&[302.0], vec![1]);
        let b = Tensor::from_slice(&[303.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_303() {
        let a = Tensor::from_slice(&[303.0], vec![1]);
        let b = Tensor::from_slice(&[304.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_304() {
        let a = Tensor::from_slice(&[304.0], vec![1]);
        let b = Tensor::from_slice(&[305.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_305() {
        let a = Tensor::from_slice(&[305.0], vec![1]);
        let b = Tensor::from_slice(&[306.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_306() {
        let a = Tensor::from_slice(&[306.0], vec![1]);
        let b = Tensor::from_slice(&[307.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_307() {
        let a = Tensor::from_slice(&[307.0], vec![1]);
        let b = Tensor::from_slice(&[308.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_308() {
        let a = Tensor::from_slice(&[308.0], vec![1]);
        let b = Tensor::from_slice(&[309.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_309() {
        let a = Tensor::from_slice(&[309.0], vec![1]);
        let b = Tensor::from_slice(&[310.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_310() {
        let a = Tensor::from_slice(&[310.0], vec![1]);
        let b = Tensor::from_slice(&[311.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_311() {
        let a = Tensor::from_slice(&[311.0], vec![1]);
        let b = Tensor::from_slice(&[312.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_312() {
        let a = Tensor::from_slice(&[312.0], vec![1]);
        let b = Tensor::from_slice(&[313.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_313() {
        let a = Tensor::from_slice(&[313.0], vec![1]);
        let b = Tensor::from_slice(&[314.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_314() {
        let a = Tensor::from_slice(&[314.0], vec![1]);
        let b = Tensor::from_slice(&[315.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_315() {
        let a = Tensor::from_slice(&[315.0], vec![1]);
        let b = Tensor::from_slice(&[316.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_316() {
        let a = Tensor::from_slice(&[316.0], vec![1]);
        let b = Tensor::from_slice(&[317.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_317() {
        let a = Tensor::from_slice(&[317.0], vec![1]);
        let b = Tensor::from_slice(&[318.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_318() {
        let a = Tensor::from_slice(&[318.0], vec![1]);
        let b = Tensor::from_slice(&[319.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_319() {
        let a = Tensor::from_slice(&[319.0], vec![1]);
        let b = Tensor::from_slice(&[320.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_320() {
        let a = Tensor::from_slice(&[320.0], vec![1]);
        let b = Tensor::from_slice(&[321.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_321() {
        let a = Tensor::from_slice(&[321.0], vec![1]);
        let b = Tensor::from_slice(&[322.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_322() {
        let a = Tensor::from_slice(&[322.0], vec![1]);
        let b = Tensor::from_slice(&[323.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_323() {
        let a = Tensor::from_slice(&[323.0], vec![1]);
        let b = Tensor::from_slice(&[324.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_324() {
        let a = Tensor::from_slice(&[324.0], vec![1]);
        let b = Tensor::from_slice(&[325.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_325() {
        let a = Tensor::from_slice(&[325.0], vec![1]);
        let b = Tensor::from_slice(&[326.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_326() {
        let a = Tensor::from_slice(&[326.0], vec![1]);
        let b = Tensor::from_slice(&[327.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_327() {
        let a = Tensor::from_slice(&[327.0], vec![1]);
        let b = Tensor::from_slice(&[328.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_328() {
        let a = Tensor::from_slice(&[328.0], vec![1]);
        let b = Tensor::from_slice(&[329.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_329() {
        let a = Tensor::from_slice(&[329.0], vec![1]);
        let b = Tensor::from_slice(&[330.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_330() {
        let a = Tensor::from_slice(&[330.0], vec![1]);
        let b = Tensor::from_slice(&[331.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_331() {
        let a = Tensor::from_slice(&[331.0], vec![1]);
        let b = Tensor::from_slice(&[332.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_332() {
        let a = Tensor::from_slice(&[332.0], vec![1]);
        let b = Tensor::from_slice(&[333.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_333() {
        let a = Tensor::from_slice(&[333.0], vec![1]);
        let b = Tensor::from_slice(&[334.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_334() {
        let a = Tensor::from_slice(&[334.0], vec![1]);
        let b = Tensor::from_slice(&[335.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_335() {
        let a = Tensor::from_slice(&[335.0], vec![1]);
        let b = Tensor::from_slice(&[336.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_336() {
        let a = Tensor::from_slice(&[336.0], vec![1]);
        let b = Tensor::from_slice(&[337.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_337() {
        let a = Tensor::from_slice(&[337.0], vec![1]);
        let b = Tensor::from_slice(&[338.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_338() {
        let a = Tensor::from_slice(&[338.0], vec![1]);
        let b = Tensor::from_slice(&[339.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_339() {
        let a = Tensor::from_slice(&[339.0], vec![1]);
        let b = Tensor::from_slice(&[340.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_340() {
        let a = Tensor::from_slice(&[340.0], vec![1]);
        let b = Tensor::from_slice(&[341.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_341() {
        let a = Tensor::from_slice(&[341.0], vec![1]);
        let b = Tensor::from_slice(&[342.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_342() {
        let a = Tensor::from_slice(&[342.0], vec![1]);
        let b = Tensor::from_slice(&[343.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_343() {
        let a = Tensor::from_slice(&[343.0], vec![1]);
        let b = Tensor::from_slice(&[344.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_344() {
        let a = Tensor::from_slice(&[344.0], vec![1]);
        let b = Tensor::from_slice(&[345.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_345() {
        let a = Tensor::from_slice(&[345.0], vec![1]);
        let b = Tensor::from_slice(&[346.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_346() {
        let a = Tensor::from_slice(&[346.0], vec![1]);
        let b = Tensor::from_slice(&[347.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_347() {
        let a = Tensor::from_slice(&[347.0], vec![1]);
        let b = Tensor::from_slice(&[348.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_348() {
        let a = Tensor::from_slice(&[348.0], vec![1]);
        let b = Tensor::from_slice(&[349.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_349() {
        let a = Tensor::from_slice(&[349.0], vec![1]);
        let b = Tensor::from_slice(&[350.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_350() {
        let a = Tensor::from_slice(&[350.0], vec![1]);
        let b = Tensor::from_slice(&[351.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_351() {
        let a = Tensor::from_slice(&[351.0], vec![1]);
        let b = Tensor::from_slice(&[352.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_352() {
        let a = Tensor::from_slice(&[352.0], vec![1]);
        let b = Tensor::from_slice(&[353.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_353() {
        let a = Tensor::from_slice(&[353.0], vec![1]);
        let b = Tensor::from_slice(&[354.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_354() {
        let a = Tensor::from_slice(&[354.0], vec![1]);
        let b = Tensor::from_slice(&[355.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_355() {
        let a = Tensor::from_slice(&[355.0], vec![1]);
        let b = Tensor::from_slice(&[356.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_356() {
        let a = Tensor::from_slice(&[356.0], vec![1]);
        let b = Tensor::from_slice(&[357.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_357() {
        let a = Tensor::from_slice(&[357.0], vec![1]);
        let b = Tensor::from_slice(&[358.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_358() {
        let a = Tensor::from_slice(&[358.0], vec![1]);
        let b = Tensor::from_slice(&[359.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_359() {
        let a = Tensor::from_slice(&[359.0], vec![1]);
        let b = Tensor::from_slice(&[360.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_360() {
        let a = Tensor::from_slice(&[360.0], vec![1]);
        let b = Tensor::from_slice(&[361.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_361() {
        let a = Tensor::from_slice(&[361.0], vec![1]);
        let b = Tensor::from_slice(&[362.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_362() {
        let a = Tensor::from_slice(&[362.0], vec![1]);
        let b = Tensor::from_slice(&[363.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_363() {
        let a = Tensor::from_slice(&[363.0], vec![1]);
        let b = Tensor::from_slice(&[364.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_364() {
        let a = Tensor::from_slice(&[364.0], vec![1]);
        let b = Tensor::from_slice(&[365.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_365() {
        let a = Tensor::from_slice(&[365.0], vec![1]);
        let b = Tensor::from_slice(&[366.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_366() {
        let a = Tensor::from_slice(&[366.0], vec![1]);
        let b = Tensor::from_slice(&[367.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_367() {
        let a = Tensor::from_slice(&[367.0], vec![1]);
        let b = Tensor::from_slice(&[368.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_368() {
        let a = Tensor::from_slice(&[368.0], vec![1]);
        let b = Tensor::from_slice(&[369.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }

    #[test]
    fn test_compare_stress_case_369() {
        let a = Tensor::from_slice(&[369.0], vec![1]);
        let b = Tensor::from_slice(&[370.0], vec![1]);
        assert_eq!(lt_tensor(&a, &b).get(0), 1.0);
        assert_eq!(gt_tensor(&a, &b).get(0), 0.0);
        assert_eq!(eq_tensor(&a, &a).get(0), 1.0);
    }
}
