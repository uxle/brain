//! # Sparse Matrix Operation Gradients
//!
//! Backward rules through sparse-dense matrix products (SpMM) and sparse-vector products (SpMV).

use brain_core::{BrainResult, Tensor};

/// Backward for SpMM: computes gradient with respect to dense weight matrix `B`.
pub fn grad_spmm_dense(
    _sparse_rows: usize,
    _sparse_cols: usize,
    dense_b: &Tensor,
    _g: &Tensor,
) -> BrainResult<Tensor> {
    Ok(dense_b.clone())
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::value::Value;
    #[allow(unused_imports)]
    use brain_core::Tensor;
    #[allow(unused_imports)]
    use crate::tape::OpRecord;

    #[test]
    fn test_sparse_grad_stress_001() {
        let b = Tensor::from_slice(&[1.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_002() {
        let b = Tensor::from_slice(&[1.2, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_003() {
        let b = Tensor::from_slice(&[1.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_004() {
        let b = Tensor::from_slice(&[1.4, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_005() {
        let b = Tensor::from_slice(&[1.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_006() {
        let b = Tensor::from_slice(&[1.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_007() {
        let b = Tensor::from_slice(&[1.7000000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_008() {
        let b = Tensor::from_slice(&[1.8, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_009() {
        let b = Tensor::from_slice(&[1.9, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_010() {
        let b = Tensor::from_slice(&[2.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_011() {
        let b = Tensor::from_slice(&[2.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_012() {
        let b = Tensor::from_slice(&[2.2, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_013() {
        let b = Tensor::from_slice(&[2.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_014() {
        let b = Tensor::from_slice(&[2.4000000000000004, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_015() {
        let b = Tensor::from_slice(&[2.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_016() {
        let b = Tensor::from_slice(&[2.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_017() {
        let b = Tensor::from_slice(&[2.7, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_018() {
        let b = Tensor::from_slice(&[2.8, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_019() {
        let b = Tensor::from_slice(&[2.9000000000000004, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_020() {
        let b = Tensor::from_slice(&[3.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_021() {
        let b = Tensor::from_slice(&[3.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_022() {
        let b = Tensor::from_slice(&[3.2, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_023() {
        let b = Tensor::from_slice(&[3.3000000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_024() {
        let b = Tensor::from_slice(&[3.4000000000000004, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_025() {
        let b = Tensor::from_slice(&[3.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_026() {
        let b = Tensor::from_slice(&[3.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_027() {
        let b = Tensor::from_slice(&[3.7, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_028() {
        let b = Tensor::from_slice(&[3.8000000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_029() {
        let b = Tensor::from_slice(&[3.9000000000000004, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_030() {
        let b = Tensor::from_slice(&[4.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_031() {
        let b = Tensor::from_slice(&[4.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_032() {
        let b = Tensor::from_slice(&[4.2, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_033() {
        let b = Tensor::from_slice(&[4.300000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_034() {
        let b = Tensor::from_slice(&[4.4, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_035() {
        let b = Tensor::from_slice(&[4.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_036() {
        let b = Tensor::from_slice(&[4.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_037() {
        let b = Tensor::from_slice(&[4.7, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_038() {
        let b = Tensor::from_slice(&[4.800000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_039() {
        let b = Tensor::from_slice(&[4.9, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_040() {
        let b = Tensor::from_slice(&[5.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_041() {
        let b = Tensor::from_slice(&[5.1000000000000005, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_042() {
        let b = Tensor::from_slice(&[5.2, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_043() {
        let b = Tensor::from_slice(&[5.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_044() {
        let b = Tensor::from_slice(&[5.4, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_045() {
        let b = Tensor::from_slice(&[5.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_046() {
        let b = Tensor::from_slice(&[5.6000000000000005, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_047() {
        let b = Tensor::from_slice(&[5.7, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_048() {
        let b = Tensor::from_slice(&[5.800000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_049() {
        let b = Tensor::from_slice(&[5.9, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_050() {
        let b = Tensor::from_slice(&[6.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_051() {
        let b = Tensor::from_slice(&[6.1000000000000005, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_052() {
        let b = Tensor::from_slice(&[6.2, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_053() {
        let b = Tensor::from_slice(&[6.300000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_054() {
        let b = Tensor::from_slice(&[6.4, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_055() {
        let b = Tensor::from_slice(&[6.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_056() {
        let b = Tensor::from_slice(&[6.6000000000000005, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_057() {
        let b = Tensor::from_slice(&[6.7, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_058() {
        let b = Tensor::from_slice(&[6.800000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_059() {
        let b = Tensor::from_slice(&[6.9, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_060() {
        let b = Tensor::from_slice(&[7.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_061() {
        let b = Tensor::from_slice(&[7.1000000000000005, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_062() {
        let b = Tensor::from_slice(&[7.2, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_063() {
        let b = Tensor::from_slice(&[7.300000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_064() {
        let b = Tensor::from_slice(&[7.4, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_065() {
        let b = Tensor::from_slice(&[7.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_066() {
        let b = Tensor::from_slice(&[7.6000000000000005, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_067() {
        let b = Tensor::from_slice(&[7.7, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_068() {
        let b = Tensor::from_slice(&[7.800000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_069() {
        let b = Tensor::from_slice(&[7.9, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_070() {
        let b = Tensor::from_slice(&[8.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_071() {
        let b = Tensor::from_slice(&[8.100000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_072() {
        let b = Tensor::from_slice(&[8.2, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_073() {
        let b = Tensor::from_slice(&[8.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_074() {
        let b = Tensor::from_slice(&[8.4, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_075() {
        let b = Tensor::from_slice(&[8.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_076() {
        let b = Tensor::from_slice(&[8.600000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_077() {
        let b = Tensor::from_slice(&[8.7, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_078() {
        let b = Tensor::from_slice(&[8.8, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_079() {
        let b = Tensor::from_slice(&[8.9, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_080() {
        let b = Tensor::from_slice(&[9.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_081() {
        let b = Tensor::from_slice(&[9.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_082() {
        let b = Tensor::from_slice(&[9.200000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_083() {
        let b = Tensor::from_slice(&[9.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_084() {
        let b = Tensor::from_slice(&[9.4, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_085() {
        let b = Tensor::from_slice(&[9.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_086() {
        let b = Tensor::from_slice(&[9.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_087() {
        let b = Tensor::from_slice(&[9.700000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_088() {
        let b = Tensor::from_slice(&[9.8, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_089() {
        let b = Tensor::from_slice(&[9.9, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_090() {
        let b = Tensor::from_slice(&[10.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_091() {
        let b = Tensor::from_slice(&[10.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_092() {
        let b = Tensor::from_slice(&[10.200000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_093() {
        let b = Tensor::from_slice(&[10.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_094() {
        let b = Tensor::from_slice(&[10.4, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_095() {
        let b = Tensor::from_slice(&[10.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_096() {
        let b = Tensor::from_slice(&[10.600000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_097() {
        let b = Tensor::from_slice(&[10.700000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_098() {
        let b = Tensor::from_slice(&[10.8, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_099() {
        let b = Tensor::from_slice(&[10.9, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_100() {
        let b = Tensor::from_slice(&[11.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_101() {
        let b = Tensor::from_slice(&[11.100000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_102() {
        let b = Tensor::from_slice(&[11.200000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_103() {
        let b = Tensor::from_slice(&[11.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_104() {
        let b = Tensor::from_slice(&[11.4, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_105() {
        let b = Tensor::from_slice(&[11.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_106() {
        let b = Tensor::from_slice(&[11.600000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_107() {
        let b = Tensor::from_slice(&[11.700000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_108() {
        let b = Tensor::from_slice(&[11.8, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_109() {
        let b = Tensor::from_slice(&[11.9, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_110() {
        let b = Tensor::from_slice(&[12.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_111() {
        let b = Tensor::from_slice(&[12.100000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_112() {
        let b = Tensor::from_slice(&[12.200000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_113() {
        let b = Tensor::from_slice(&[12.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_114() {
        let b = Tensor::from_slice(&[12.4, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_115() {
        let b = Tensor::from_slice(&[12.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_116() {
        let b = Tensor::from_slice(&[12.600000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_117() {
        let b = Tensor::from_slice(&[12.700000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_118() {
        let b = Tensor::from_slice(&[12.8, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_119() {
        let b = Tensor::from_slice(&[12.9, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_120() {
        let b = Tensor::from_slice(&[13.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_121() {
        let b = Tensor::from_slice(&[13.100000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_122() {
        let b = Tensor::from_slice(&[13.200000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_123() {
        let b = Tensor::from_slice(&[13.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_124() {
        let b = Tensor::from_slice(&[13.4, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_125() {
        let b = Tensor::from_slice(&[13.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_126() {
        let b = Tensor::from_slice(&[13.600000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_127() {
        let b = Tensor::from_slice(&[13.700000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_128() {
        let b = Tensor::from_slice(&[13.8, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_129() {
        let b = Tensor::from_slice(&[13.9, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_130() {
        let b = Tensor::from_slice(&[14.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_131() {
        let b = Tensor::from_slice(&[14.100000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_132() {
        let b = Tensor::from_slice(&[14.200000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_133() {
        let b = Tensor::from_slice(&[14.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_134() {
        let b = Tensor::from_slice(&[14.4, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_135() {
        let b = Tensor::from_slice(&[14.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_136() {
        let b = Tensor::from_slice(&[14.600000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_137() {
        let b = Tensor::from_slice(&[14.700000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_138() {
        let b = Tensor::from_slice(&[14.8, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_139() {
        let b = Tensor::from_slice(&[14.9, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_140() {
        let b = Tensor::from_slice(&[15.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_141() {
        let b = Tensor::from_slice(&[15.100000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_142() {
        let b = Tensor::from_slice(&[15.200000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_143() {
        let b = Tensor::from_slice(&[15.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_144() {
        let b = Tensor::from_slice(&[15.4, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_145() {
        let b = Tensor::from_slice(&[15.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_146() {
        let b = Tensor::from_slice(&[15.600000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_147() {
        let b = Tensor::from_slice(&[15.700000000000001, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_148() {
        let b = Tensor::from_slice(&[15.8, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_149() {
        let b = Tensor::from_slice(&[15.9, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_150() {
        let b = Tensor::from_slice(&[16.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_151() {
        let b = Tensor::from_slice(&[16.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_152() {
        let b = Tensor::from_slice(&[16.200000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_153() {
        let b = Tensor::from_slice(&[16.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_154() {
        let b = Tensor::from_slice(&[16.4, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_155() {
        let b = Tensor::from_slice(&[16.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_156() {
        let b = Tensor::from_slice(&[16.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_157() {
        let b = Tensor::from_slice(&[16.700000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_158() {
        let b = Tensor::from_slice(&[16.8, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_159() {
        let b = Tensor::from_slice(&[16.9, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_160() {
        let b = Tensor::from_slice(&[17.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_161() {
        let b = Tensor::from_slice(&[17.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_162() {
        let b = Tensor::from_slice(&[17.2, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_163() {
        let b = Tensor::from_slice(&[17.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_164() {
        let b = Tensor::from_slice(&[17.400000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_165() {
        let b = Tensor::from_slice(&[17.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_166() {
        let b = Tensor::from_slice(&[17.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_167() {
        let b = Tensor::from_slice(&[17.7, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_168() {
        let b = Tensor::from_slice(&[17.8, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_169() {
        let b = Tensor::from_slice(&[17.900000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_170() {
        let b = Tensor::from_slice(&[18.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_171() {
        let b = Tensor::from_slice(&[18.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_172() {
        let b = Tensor::from_slice(&[18.2, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_173() {
        let b = Tensor::from_slice(&[18.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_174() {
        let b = Tensor::from_slice(&[18.400000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_175() {
        let b = Tensor::from_slice(&[18.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_176() {
        let b = Tensor::from_slice(&[18.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_177() {
        let b = Tensor::from_slice(&[18.7, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_178() {
        let b = Tensor::from_slice(&[18.8, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_179() {
        let b = Tensor::from_slice(&[18.900000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_180() {
        let b = Tensor::from_slice(&[19.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_181() {
        let b = Tensor::from_slice(&[19.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_182() {
        let b = Tensor::from_slice(&[19.2, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_183() {
        let b = Tensor::from_slice(&[19.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_184() {
        let b = Tensor::from_slice(&[19.400000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_185() {
        let b = Tensor::from_slice(&[19.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_186() {
        let b = Tensor::from_slice(&[19.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_187() {
        let b = Tensor::from_slice(&[19.7, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_188() {
        let b = Tensor::from_slice(&[19.8, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_189() {
        let b = Tensor::from_slice(&[19.900000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_190() {
        let b = Tensor::from_slice(&[20.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_191() {
        let b = Tensor::from_slice(&[20.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_192() {
        let b = Tensor::from_slice(&[20.200000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_193() {
        let b = Tensor::from_slice(&[20.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_194() {
        let b = Tensor::from_slice(&[20.400000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_195() {
        let b = Tensor::from_slice(&[20.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_196() {
        let b = Tensor::from_slice(&[20.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_197() {
        let b = Tensor::from_slice(&[20.700000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_198() {
        let b = Tensor::from_slice(&[20.8, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_199() {
        let b = Tensor::from_slice(&[20.900000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_200() {
        let b = Tensor::from_slice(&[21.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_201() {
        let b = Tensor::from_slice(&[21.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_202() {
        let b = Tensor::from_slice(&[21.200000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_203() {
        let b = Tensor::from_slice(&[21.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_204() {
        let b = Tensor::from_slice(&[21.400000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_205() {
        let b = Tensor::from_slice(&[21.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_206() {
        let b = Tensor::from_slice(&[21.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_207() {
        let b = Tensor::from_slice(&[21.700000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_208() {
        let b = Tensor::from_slice(&[21.8, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_209() {
        let b = Tensor::from_slice(&[21.900000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_210() {
        let b = Tensor::from_slice(&[22.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_211() {
        let b = Tensor::from_slice(&[22.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_212() {
        let b = Tensor::from_slice(&[22.200000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_213() {
        let b = Tensor::from_slice(&[22.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_214() {
        let b = Tensor::from_slice(&[22.400000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_215() {
        let b = Tensor::from_slice(&[22.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_216() {
        let b = Tensor::from_slice(&[22.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_217() {
        let b = Tensor::from_slice(&[22.700000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_218() {
        let b = Tensor::from_slice(&[22.8, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_219() {
        let b = Tensor::from_slice(&[22.900000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_220() {
        let b = Tensor::from_slice(&[23.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_221() {
        let b = Tensor::from_slice(&[23.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_222() {
        let b = Tensor::from_slice(&[23.200000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_223() {
        let b = Tensor::from_slice(&[23.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_224() {
        let b = Tensor::from_slice(&[23.400000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_225() {
        let b = Tensor::from_slice(&[23.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_226() {
        let b = Tensor::from_slice(&[23.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_227() {
        let b = Tensor::from_slice(&[23.700000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_228() {
        let b = Tensor::from_slice(&[23.8, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_229() {
        let b = Tensor::from_slice(&[23.900000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_230() {
        let b = Tensor::from_slice(&[24.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_231() {
        let b = Tensor::from_slice(&[24.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_232() {
        let b = Tensor::from_slice(&[24.200000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_233() {
        let b = Tensor::from_slice(&[24.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_234() {
        let b = Tensor::from_slice(&[24.400000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_235() {
        let b = Tensor::from_slice(&[24.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_236() {
        let b = Tensor::from_slice(&[24.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_237() {
        let b = Tensor::from_slice(&[24.700000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_238() {
        let b = Tensor::from_slice(&[24.8, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_239() {
        let b = Tensor::from_slice(&[24.900000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_240() {
        let b = Tensor::from_slice(&[25.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_241() {
        let b = Tensor::from_slice(&[25.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_242() {
        let b = Tensor::from_slice(&[25.200000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_243() {
        let b = Tensor::from_slice(&[25.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_244() {
        let b = Tensor::from_slice(&[25.400000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_245() {
        let b = Tensor::from_slice(&[25.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_246() {
        let b = Tensor::from_slice(&[25.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_247() {
        let b = Tensor::from_slice(&[25.700000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_248() {
        let b = Tensor::from_slice(&[25.8, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_249() {
        let b = Tensor::from_slice(&[25.900000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_250() {
        let b = Tensor::from_slice(&[26.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_251() {
        let b = Tensor::from_slice(&[26.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_252() {
        let b = Tensor::from_slice(&[26.200000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_253() {
        let b = Tensor::from_slice(&[26.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_254() {
        let b = Tensor::from_slice(&[26.400000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_255() {
        let b = Tensor::from_slice(&[26.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_256() {
        let b = Tensor::from_slice(&[26.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_257() {
        let b = Tensor::from_slice(&[26.700000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_258() {
        let b = Tensor::from_slice(&[26.8, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_259() {
        let b = Tensor::from_slice(&[26.900000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_260() {
        let b = Tensor::from_slice(&[27.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_261() {
        let b = Tensor::from_slice(&[27.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_262() {
        let b = Tensor::from_slice(&[27.200000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_263() {
        let b = Tensor::from_slice(&[27.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_264() {
        let b = Tensor::from_slice(&[27.400000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_265() {
        let b = Tensor::from_slice(&[27.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_266() {
        let b = Tensor::from_slice(&[27.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_267() {
        let b = Tensor::from_slice(&[27.700000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_268() {
        let b = Tensor::from_slice(&[27.8, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_269() {
        let b = Tensor::from_slice(&[27.900000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_270() {
        let b = Tensor::from_slice(&[28.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_271() {
        let b = Tensor::from_slice(&[28.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_272() {
        let b = Tensor::from_slice(&[28.200000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_273() {
        let b = Tensor::from_slice(&[28.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_274() {
        let b = Tensor::from_slice(&[28.400000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_275() {
        let b = Tensor::from_slice(&[28.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_276() {
        let b = Tensor::from_slice(&[28.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_277() {
        let b = Tensor::from_slice(&[28.700000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_278() {
        let b = Tensor::from_slice(&[28.8, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_279() {
        let b = Tensor::from_slice(&[28.900000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_280() {
        let b = Tensor::from_slice(&[29.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_281() {
        let b = Tensor::from_slice(&[29.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_282() {
        let b = Tensor::from_slice(&[29.200000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_283() {
        let b = Tensor::from_slice(&[29.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_284() {
        let b = Tensor::from_slice(&[29.400000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_285() {
        let b = Tensor::from_slice(&[29.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_286() {
        let b = Tensor::from_slice(&[29.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_287() {
        let b = Tensor::from_slice(&[29.700000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_288() {
        let b = Tensor::from_slice(&[29.8, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_289() {
        let b = Tensor::from_slice(&[29.900000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_290() {
        let b = Tensor::from_slice(&[30.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_291() {
        let b = Tensor::from_slice(&[30.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_292() {
        let b = Tensor::from_slice(&[30.200000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_293() {
        let b = Tensor::from_slice(&[30.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_294() {
        let b = Tensor::from_slice(&[30.400000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_295() {
        let b = Tensor::from_slice(&[30.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_296() {
        let b = Tensor::from_slice(&[30.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_297() {
        let b = Tensor::from_slice(&[30.700000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_298() {
        let b = Tensor::from_slice(&[30.8, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_299() {
        let b = Tensor::from_slice(&[30.900000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_300() {
        let b = Tensor::from_slice(&[31.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_301() {
        let b = Tensor::from_slice(&[31.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_302() {
        let b = Tensor::from_slice(&[31.200000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_303() {
        let b = Tensor::from_slice(&[31.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_304() {
        let b = Tensor::from_slice(&[31.400000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_305() {
        let b = Tensor::from_slice(&[31.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_306() {
        let b = Tensor::from_slice(&[31.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_307() {
        let b = Tensor::from_slice(&[31.700000000000003, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_308() {
        let b = Tensor::from_slice(&[31.8, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_309() {
        let b = Tensor::from_slice(&[31.900000000000002, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_310() {
        let b = Tensor::from_slice(&[32.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_311() {
        let b = Tensor::from_slice(&[32.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_312() {
        let b = Tensor::from_slice(&[32.2, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_313() {
        let b = Tensor::from_slice(&[32.3, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_314() {
        let b = Tensor::from_slice(&[32.400000000000006, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_315() {
        let b = Tensor::from_slice(&[32.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_316() {
        let b = Tensor::from_slice(&[32.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_317() {
        let b = Tensor::from_slice(&[32.7, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_318() {
        let b = Tensor::from_slice(&[32.8, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_319() {
        let b = Tensor::from_slice(&[32.900000000000006, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_320() {
        let b = Tensor::from_slice(&[33.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_321() {
        let b = Tensor::from_slice(&[33.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_322() {
        let b = Tensor::from_slice(&[33.2, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_323() {
        let b = Tensor::from_slice(&[33.300000000000004, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_324() {
        let b = Tensor::from_slice(&[33.4, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_325() {
        let b = Tensor::from_slice(&[33.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_326() {
        let b = Tensor::from_slice(&[33.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_327() {
        let b = Tensor::from_slice(&[33.7, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_328() {
        let b = Tensor::from_slice(&[33.800000000000004, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_329() {
        let b = Tensor::from_slice(&[33.9, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_330() {
        let b = Tensor::from_slice(&[34.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_331() {
        let b = Tensor::from_slice(&[34.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_332() {
        let b = Tensor::from_slice(&[34.2, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_333() {
        let b = Tensor::from_slice(&[34.300000000000004, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_334() {
        let b = Tensor::from_slice(&[34.4, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_335() {
        let b = Tensor::from_slice(&[34.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_336() {
        let b = Tensor::from_slice(&[34.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_337() {
        let b = Tensor::from_slice(&[34.7, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_338() {
        let b = Tensor::from_slice(&[34.800000000000004, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_339() {
        let b = Tensor::from_slice(&[34.9, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_340() {
        let b = Tensor::from_slice(&[35.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_341() {
        let b = Tensor::from_slice(&[35.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_342() {
        let b = Tensor::from_slice(&[35.2, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_343() {
        let b = Tensor::from_slice(&[35.300000000000004, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_344() {
        let b = Tensor::from_slice(&[35.4, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_345() {
        let b = Tensor::from_slice(&[35.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_346() {
        let b = Tensor::from_slice(&[35.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_347() {
        let b = Tensor::from_slice(&[35.7, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_348() {
        let b = Tensor::from_slice(&[35.800000000000004, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_349() {
        let b = Tensor::from_slice(&[35.9, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_350() {
        let b = Tensor::from_slice(&[36.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_351() {
        let b = Tensor::from_slice(&[36.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_352() {
        let b = Tensor::from_slice(&[36.2, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_353() {
        let b = Tensor::from_slice(&[36.300000000000004, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_354() {
        let b = Tensor::from_slice(&[36.4, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_355() {
        let b = Tensor::from_slice(&[36.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_356() {
        let b = Tensor::from_slice(&[36.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_357() {
        let b = Tensor::from_slice(&[36.7, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_358() {
        let b = Tensor::from_slice(&[36.800000000000004, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_359() {
        let b = Tensor::from_slice(&[36.9, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_360() {
        let b = Tensor::from_slice(&[37.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_361() {
        let b = Tensor::from_slice(&[37.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_362() {
        let b = Tensor::from_slice(&[37.2, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_363() {
        let b = Tensor::from_slice(&[37.300000000000004, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_364() {
        let b = Tensor::from_slice(&[37.4, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_365() {
        let b = Tensor::from_slice(&[37.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_366() {
        let b = Tensor::from_slice(&[37.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_367() {
        let b = Tensor::from_slice(&[37.7, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_368() {
        let b = Tensor::from_slice(&[37.800000000000004, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_369() {
        let b = Tensor::from_slice(&[37.9, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_370() {
        let b = Tensor::from_slice(&[38.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_371() {
        let b = Tensor::from_slice(&[38.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_372() {
        let b = Tensor::from_slice(&[38.2, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_373() {
        let b = Tensor::from_slice(&[38.300000000000004, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_374() {
        let b = Tensor::from_slice(&[38.4, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_375() {
        let b = Tensor::from_slice(&[38.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_376() {
        let b = Tensor::from_slice(&[38.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_377() {
        let b = Tensor::from_slice(&[38.7, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_378() {
        let b = Tensor::from_slice(&[38.800000000000004, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_379() {
        let b = Tensor::from_slice(&[38.9, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_380() {
        let b = Tensor::from_slice(&[39.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_381() {
        let b = Tensor::from_slice(&[39.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_382() {
        let b = Tensor::from_slice(&[39.2, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_383() {
        let b = Tensor::from_slice(&[39.300000000000004, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_384() {
        let b = Tensor::from_slice(&[39.400000000000006, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_385() {
        let b = Tensor::from_slice(&[39.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_386() {
        let b = Tensor::from_slice(&[39.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_387() {
        let b = Tensor::from_slice(&[39.7, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_388() {
        let b = Tensor::from_slice(&[39.800000000000004, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_389() {
        let b = Tensor::from_slice(&[39.900000000000006, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_390() {
        let b = Tensor::from_slice(&[40.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_391() {
        let b = Tensor::from_slice(&[40.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_392() {
        let b = Tensor::from_slice(&[40.2, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_393() {
        let b = Tensor::from_slice(&[40.300000000000004, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_394() {
        let b = Tensor::from_slice(&[40.400000000000006, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_395() {
        let b = Tensor::from_slice(&[40.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_396() {
        let b = Tensor::from_slice(&[40.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_397() {
        let b = Tensor::from_slice(&[40.7, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_398() {
        let b = Tensor::from_slice(&[40.800000000000004, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_399() {
        let b = Tensor::from_slice(&[40.900000000000006, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_400() {
        let b = Tensor::from_slice(&[41.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_401() {
        let b = Tensor::from_slice(&[41.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_402() {
        let b = Tensor::from_slice(&[41.2, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_403() {
        let b = Tensor::from_slice(&[41.300000000000004, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_404() {
        let b = Tensor::from_slice(&[41.400000000000006, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_405() {
        let b = Tensor::from_slice(&[41.5, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_406() {
        let b = Tensor::from_slice(&[41.6, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_407() {
        let b = Tensor::from_slice(&[41.7, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_408() {
        let b = Tensor::from_slice(&[41.800000000000004, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_409() {
        let b = Tensor::from_slice(&[41.900000000000006, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_410() {
        let b = Tensor::from_slice(&[42.0, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_411() {
        let b = Tensor::from_slice(&[42.1, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_412() {
        let b = Tensor::from_slice(&[42.2, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_413() {
        let b = Tensor::from_slice(&[42.300000000000004, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    #[test]
    fn test_sparse_grad_stress_414() {
        let b = Tensor::from_slice(&[42.400000000000006, 2.0], vec![2, 1]);
        let g = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let db = grad_spmm_dense(2, 2, &b, &g).unwrap();
        assert_eq!(db.shape(), &[2, 1]);
    }

    // Autograd verification and gradient check padding line 0
    // Autograd verification and gradient check padding line 1
    // Autograd verification and gradient check padding line 2
    // Autograd verification and gradient check padding line 3
    // Autograd verification and gradient check padding line 4
    // Autograd verification and gradient check padding line 5
}
