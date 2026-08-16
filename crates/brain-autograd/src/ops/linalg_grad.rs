//! # Linear Algebra Operation Gradients
//!
//! Backward rules for matrix inversion, determinants, Cholesky, QR, SVD.

use brain_core::tensor::arithmetic as arith_t;
use brain_core::tensor::linalg as linalg_t;
use brain_core::{BrainResult, Tensor};

/// Backward pass for matrix inversion `Y = A^{-1}`: `dA = - Y^T @ G @ Y^T`.
pub fn grad_inv(a_inv: &Tensor, grad_output: &Tensor) -> BrainResult<Tensor> {
    let a_inv_t = a_inv.transpose(0, 1);
    let temp = arith_t::matmul(&a_inv_t, grad_output);
    let da = arith_t::matmul(&temp, &a_inv_t).map(|x| -x);
    Ok(da)
}

/// Backward pass for matrix determinant `y = det(A)`: `dA = g * det(A) * (A^{-1})^T`.
pub fn grad_det(a: &Tensor, det_val: f64, g: f64) -> BrainResult<Tensor> {
    let a_inv = linalg_t::inv(a);
    let a_inv_t = a_inv.transpose(0, 1);
    Ok(a_inv_t.map(|x| x * g * det_val))
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
    fn test_linalg_grad_stress_001() {
        let a_inv = Tensor::from_slice(&[1.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_002() {
        let a_inv = Tensor::from_slice(&[1.1, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_003() {
        let a_inv = Tensor::from_slice(&[1.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_004() {
        let a_inv = Tensor::from_slice(&[1.2, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_005() {
        let a_inv = Tensor::from_slice(&[1.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_006() {
        let a_inv = Tensor::from_slice(&[1.3, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_007() {
        let a_inv = Tensor::from_slice(&[1.35, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_008() {
        let a_inv = Tensor::from_slice(&[1.4, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_009() {
        let a_inv = Tensor::from_slice(&[1.45, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_010() {
        let a_inv = Tensor::from_slice(&[1.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_011() {
        let a_inv = Tensor::from_slice(&[1.55, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_012() {
        let a_inv = Tensor::from_slice(&[1.6, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_013() {
        let a_inv = Tensor::from_slice(&[1.65, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_014() {
        let a_inv = Tensor::from_slice(&[1.7000000000000002, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_015() {
        let a_inv = Tensor::from_slice(&[1.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_016() {
        let a_inv = Tensor::from_slice(&[1.8, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_017() {
        let a_inv = Tensor::from_slice(&[1.85, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_018() {
        let a_inv = Tensor::from_slice(&[1.9, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_019() {
        let a_inv = Tensor::from_slice(&[1.9500000000000002, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_020() {
        let a_inv = Tensor::from_slice(&[2.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_021() {
        let a_inv = Tensor::from_slice(&[2.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_022() {
        let a_inv = Tensor::from_slice(&[2.1, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_023() {
        let a_inv = Tensor::from_slice(&[2.1500000000000004, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_024() {
        let a_inv = Tensor::from_slice(&[2.2, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_025() {
        let a_inv = Tensor::from_slice(&[2.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_026() {
        let a_inv = Tensor::from_slice(&[2.3, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_027() {
        let a_inv = Tensor::from_slice(&[2.35, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_028() {
        let a_inv = Tensor::from_slice(&[2.4000000000000004, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_029() {
        let a_inv = Tensor::from_slice(&[2.45, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_030() {
        let a_inv = Tensor::from_slice(&[2.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_031() {
        let a_inv = Tensor::from_slice(&[2.55, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_032() {
        let a_inv = Tensor::from_slice(&[2.6, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_033() {
        let a_inv = Tensor::from_slice(&[2.6500000000000004, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_034() {
        let a_inv = Tensor::from_slice(&[2.7, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_035() {
        let a_inv = Tensor::from_slice(&[2.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_036() {
        let a_inv = Tensor::from_slice(&[2.8, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_037() {
        let a_inv = Tensor::from_slice(&[2.85, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_038() {
        let a_inv = Tensor::from_slice(&[2.9000000000000004, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_039() {
        let a_inv = Tensor::from_slice(&[2.95, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_040() {
        let a_inv = Tensor::from_slice(&[3.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_041() {
        let a_inv = Tensor::from_slice(&[3.0500000000000003, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_042() {
        let a_inv = Tensor::from_slice(&[3.1, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_043() {
        let a_inv = Tensor::from_slice(&[3.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_044() {
        let a_inv = Tensor::from_slice(&[3.2, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_045() {
        let a_inv = Tensor::from_slice(&[3.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_046() {
        let a_inv = Tensor::from_slice(&[3.3000000000000003, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_047() {
        let a_inv = Tensor::from_slice(&[3.35, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_048() {
        let a_inv = Tensor::from_slice(&[3.4000000000000004, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_049() {
        let a_inv = Tensor::from_slice(&[3.45, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_050() {
        let a_inv = Tensor::from_slice(&[3.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_051() {
        let a_inv = Tensor::from_slice(&[3.5500000000000003, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_052() {
        let a_inv = Tensor::from_slice(&[3.6, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_053() {
        let a_inv = Tensor::from_slice(&[3.6500000000000004, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_054() {
        let a_inv = Tensor::from_slice(&[3.7, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_055() {
        let a_inv = Tensor::from_slice(&[3.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_056() {
        let a_inv = Tensor::from_slice(&[3.8000000000000003, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_057() {
        let a_inv = Tensor::from_slice(&[3.85, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_058() {
        let a_inv = Tensor::from_slice(&[3.9000000000000004, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_059() {
        let a_inv = Tensor::from_slice(&[3.95, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_060() {
        let a_inv = Tensor::from_slice(&[4.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_061() {
        let a_inv = Tensor::from_slice(&[4.050000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_062() {
        let a_inv = Tensor::from_slice(&[4.1, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_063() {
        let a_inv = Tensor::from_slice(&[4.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_064() {
        let a_inv = Tensor::from_slice(&[4.2, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_065() {
        let a_inv = Tensor::from_slice(&[4.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_066() {
        let a_inv = Tensor::from_slice(&[4.300000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_067() {
        let a_inv = Tensor::from_slice(&[4.35, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_068() {
        let a_inv = Tensor::from_slice(&[4.4, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_069() {
        let a_inv = Tensor::from_slice(&[4.45, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_070() {
        let a_inv = Tensor::from_slice(&[4.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_071() {
        let a_inv = Tensor::from_slice(&[4.550000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_072() {
        let a_inv = Tensor::from_slice(&[4.6, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_073() {
        let a_inv = Tensor::from_slice(&[4.65, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_074() {
        let a_inv = Tensor::from_slice(&[4.7, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_075() {
        let a_inv = Tensor::from_slice(&[4.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_076() {
        let a_inv = Tensor::from_slice(&[4.800000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_077() {
        let a_inv = Tensor::from_slice(&[4.85, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_078() {
        let a_inv = Tensor::from_slice(&[4.9, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_079() {
        let a_inv = Tensor::from_slice(&[4.95, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_080() {
        let a_inv = Tensor::from_slice(&[5.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_081() {
        let a_inv = Tensor::from_slice(&[5.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_082() {
        let a_inv = Tensor::from_slice(&[5.1000000000000005, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_083() {
        let a_inv = Tensor::from_slice(&[5.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_084() {
        let a_inv = Tensor::from_slice(&[5.2, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_085() {
        let a_inv = Tensor::from_slice(&[5.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_086() {
        let a_inv = Tensor::from_slice(&[5.3, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_087() {
        let a_inv = Tensor::from_slice(&[5.3500000000000005, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_088() {
        let a_inv = Tensor::from_slice(&[5.4, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_089() {
        let a_inv = Tensor::from_slice(&[5.45, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_090() {
        let a_inv = Tensor::from_slice(&[5.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_091() {
        let a_inv = Tensor::from_slice(&[5.55, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_092() {
        let a_inv = Tensor::from_slice(&[5.6000000000000005, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_093() {
        let a_inv = Tensor::from_slice(&[5.65, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_094() {
        let a_inv = Tensor::from_slice(&[5.7, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_095() {
        let a_inv = Tensor::from_slice(&[5.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_096() {
        let a_inv = Tensor::from_slice(&[5.800000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_097() {
        let a_inv = Tensor::from_slice(&[5.8500000000000005, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_098() {
        let a_inv = Tensor::from_slice(&[5.9, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_099() {
        let a_inv = Tensor::from_slice(&[5.95, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_100() {
        let a_inv = Tensor::from_slice(&[6.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_101() {
        let a_inv = Tensor::from_slice(&[6.050000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_102() {
        let a_inv = Tensor::from_slice(&[6.1000000000000005, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_103() {
        let a_inv = Tensor::from_slice(&[6.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_104() {
        let a_inv = Tensor::from_slice(&[6.2, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_105() {
        let a_inv = Tensor::from_slice(&[6.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_106() {
        let a_inv = Tensor::from_slice(&[6.300000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_107() {
        let a_inv = Tensor::from_slice(&[6.3500000000000005, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_108() {
        let a_inv = Tensor::from_slice(&[6.4, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_109() {
        let a_inv = Tensor::from_slice(&[6.45, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_110() {
        let a_inv = Tensor::from_slice(&[6.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_111() {
        let a_inv = Tensor::from_slice(&[6.550000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_112() {
        let a_inv = Tensor::from_slice(&[6.6000000000000005, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_113() {
        let a_inv = Tensor::from_slice(&[6.65, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_114() {
        let a_inv = Tensor::from_slice(&[6.7, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_115() {
        let a_inv = Tensor::from_slice(&[6.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_116() {
        let a_inv = Tensor::from_slice(&[6.800000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_117() {
        let a_inv = Tensor::from_slice(&[6.8500000000000005, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_118() {
        let a_inv = Tensor::from_slice(&[6.9, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_119() {
        let a_inv = Tensor::from_slice(&[6.95, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_120() {
        let a_inv = Tensor::from_slice(&[7.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_121() {
        let a_inv = Tensor::from_slice(&[7.050000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_122() {
        let a_inv = Tensor::from_slice(&[7.1000000000000005, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_123() {
        let a_inv = Tensor::from_slice(&[7.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_124() {
        let a_inv = Tensor::from_slice(&[7.2, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_125() {
        let a_inv = Tensor::from_slice(&[7.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_126() {
        let a_inv = Tensor::from_slice(&[7.300000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_127() {
        let a_inv = Tensor::from_slice(&[7.3500000000000005, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_128() {
        let a_inv = Tensor::from_slice(&[7.4, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_129() {
        let a_inv = Tensor::from_slice(&[7.45, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_130() {
        let a_inv = Tensor::from_slice(&[7.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_131() {
        let a_inv = Tensor::from_slice(&[7.550000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_132() {
        let a_inv = Tensor::from_slice(&[7.6000000000000005, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_133() {
        let a_inv = Tensor::from_slice(&[7.65, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_134() {
        let a_inv = Tensor::from_slice(&[7.7, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_135() {
        let a_inv = Tensor::from_slice(&[7.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_136() {
        let a_inv = Tensor::from_slice(&[7.800000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_137() {
        let a_inv = Tensor::from_slice(&[7.8500000000000005, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_138() {
        let a_inv = Tensor::from_slice(&[7.9, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_139() {
        let a_inv = Tensor::from_slice(&[7.95, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_140() {
        let a_inv = Tensor::from_slice(&[8.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_141() {
        let a_inv = Tensor::from_slice(&[8.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_142() {
        let a_inv = Tensor::from_slice(&[8.100000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_143() {
        let a_inv = Tensor::from_slice(&[8.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_144() {
        let a_inv = Tensor::from_slice(&[8.2, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_145() {
        let a_inv = Tensor::from_slice(&[8.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_146() {
        let a_inv = Tensor::from_slice(&[8.3, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_147() {
        let a_inv = Tensor::from_slice(&[8.350000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_148() {
        let a_inv = Tensor::from_slice(&[8.4, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_149() {
        let a_inv = Tensor::from_slice(&[8.45, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_150() {
        let a_inv = Tensor::from_slice(&[8.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_151() {
        let a_inv = Tensor::from_slice(&[8.55, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_152() {
        let a_inv = Tensor::from_slice(&[8.600000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_153() {
        let a_inv = Tensor::from_slice(&[8.65, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_154() {
        let a_inv = Tensor::from_slice(&[8.7, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_155() {
        let a_inv = Tensor::from_slice(&[8.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_156() {
        let a_inv = Tensor::from_slice(&[8.8, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_157() {
        let a_inv = Tensor::from_slice(&[8.850000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_158() {
        let a_inv = Tensor::from_slice(&[8.9, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_159() {
        let a_inv = Tensor::from_slice(&[8.95, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_160() {
        let a_inv = Tensor::from_slice(&[9.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_161() {
        let a_inv = Tensor::from_slice(&[9.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_162() {
        let a_inv = Tensor::from_slice(&[9.1, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_163() {
        let a_inv = Tensor::from_slice(&[9.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_164() {
        let a_inv = Tensor::from_slice(&[9.200000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_165() {
        let a_inv = Tensor::from_slice(&[9.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_166() {
        let a_inv = Tensor::from_slice(&[9.3, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_167() {
        let a_inv = Tensor::from_slice(&[9.35, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_168() {
        let a_inv = Tensor::from_slice(&[9.4, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_169() {
        let a_inv = Tensor::from_slice(&[9.450000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_170() {
        let a_inv = Tensor::from_slice(&[9.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_171() {
        let a_inv = Tensor::from_slice(&[9.55, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_172() {
        let a_inv = Tensor::from_slice(&[9.6, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_173() {
        let a_inv = Tensor::from_slice(&[9.65, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_174() {
        let a_inv = Tensor::from_slice(&[9.700000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_175() {
        let a_inv = Tensor::from_slice(&[9.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_176() {
        let a_inv = Tensor::from_slice(&[9.8, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_177() {
        let a_inv = Tensor::from_slice(&[9.85, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_178() {
        let a_inv = Tensor::from_slice(&[9.9, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_179() {
        let a_inv = Tensor::from_slice(&[9.950000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_180() {
        let a_inv = Tensor::from_slice(&[10.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_181() {
        let a_inv = Tensor::from_slice(&[10.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_182() {
        let a_inv = Tensor::from_slice(&[10.1, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_183() {
        let a_inv = Tensor::from_slice(&[10.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_184() {
        let a_inv = Tensor::from_slice(&[10.200000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_185() {
        let a_inv = Tensor::from_slice(&[10.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_186() {
        let a_inv = Tensor::from_slice(&[10.3, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_187() {
        let a_inv = Tensor::from_slice(&[10.35, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_188() {
        let a_inv = Tensor::from_slice(&[10.4, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_189() {
        let a_inv = Tensor::from_slice(&[10.450000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_190() {
        let a_inv = Tensor::from_slice(&[10.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_191() {
        let a_inv = Tensor::from_slice(&[10.55, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_192() {
        let a_inv = Tensor::from_slice(&[10.600000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_193() {
        let a_inv = Tensor::from_slice(&[10.65, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_194() {
        let a_inv = Tensor::from_slice(&[10.700000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_195() {
        let a_inv = Tensor::from_slice(&[10.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_196() {
        let a_inv = Tensor::from_slice(&[10.8, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_197() {
        let a_inv = Tensor::from_slice(&[10.850000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_198() {
        let a_inv = Tensor::from_slice(&[10.9, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_199() {
        let a_inv = Tensor::from_slice(&[10.950000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_200() {
        let a_inv = Tensor::from_slice(&[11.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_201() {
        let a_inv = Tensor::from_slice(&[11.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_202() {
        let a_inv = Tensor::from_slice(&[11.100000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_203() {
        let a_inv = Tensor::from_slice(&[11.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_204() {
        let a_inv = Tensor::from_slice(&[11.200000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_205() {
        let a_inv = Tensor::from_slice(&[11.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_206() {
        let a_inv = Tensor::from_slice(&[11.3, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_207() {
        let a_inv = Tensor::from_slice(&[11.350000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_208() {
        let a_inv = Tensor::from_slice(&[11.4, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_209() {
        let a_inv = Tensor::from_slice(&[11.450000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_210() {
        let a_inv = Tensor::from_slice(&[11.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_211() {
        let a_inv = Tensor::from_slice(&[11.55, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_212() {
        let a_inv = Tensor::from_slice(&[11.600000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_213() {
        let a_inv = Tensor::from_slice(&[11.65, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_214() {
        let a_inv = Tensor::from_slice(&[11.700000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_215() {
        let a_inv = Tensor::from_slice(&[11.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_216() {
        let a_inv = Tensor::from_slice(&[11.8, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_217() {
        let a_inv = Tensor::from_slice(&[11.850000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_218() {
        let a_inv = Tensor::from_slice(&[11.9, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_219() {
        let a_inv = Tensor::from_slice(&[11.950000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_220() {
        let a_inv = Tensor::from_slice(&[12.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_221() {
        let a_inv = Tensor::from_slice(&[12.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_222() {
        let a_inv = Tensor::from_slice(&[12.100000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_223() {
        let a_inv = Tensor::from_slice(&[12.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_224() {
        let a_inv = Tensor::from_slice(&[12.200000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_225() {
        let a_inv = Tensor::from_slice(&[12.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_226() {
        let a_inv = Tensor::from_slice(&[12.3, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_227() {
        let a_inv = Tensor::from_slice(&[12.350000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_228() {
        let a_inv = Tensor::from_slice(&[12.4, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_229() {
        let a_inv = Tensor::from_slice(&[12.450000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_230() {
        let a_inv = Tensor::from_slice(&[12.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_231() {
        let a_inv = Tensor::from_slice(&[12.55, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_232() {
        let a_inv = Tensor::from_slice(&[12.600000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_233() {
        let a_inv = Tensor::from_slice(&[12.65, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_234() {
        let a_inv = Tensor::from_slice(&[12.700000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_235() {
        let a_inv = Tensor::from_slice(&[12.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_236() {
        let a_inv = Tensor::from_slice(&[12.8, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_237() {
        let a_inv = Tensor::from_slice(&[12.850000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_238() {
        let a_inv = Tensor::from_slice(&[12.9, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_239() {
        let a_inv = Tensor::from_slice(&[12.950000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_240() {
        let a_inv = Tensor::from_slice(&[13.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_241() {
        let a_inv = Tensor::from_slice(&[13.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_242() {
        let a_inv = Tensor::from_slice(&[13.100000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_243() {
        let a_inv = Tensor::from_slice(&[13.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_244() {
        let a_inv = Tensor::from_slice(&[13.200000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_245() {
        let a_inv = Tensor::from_slice(&[13.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_246() {
        let a_inv = Tensor::from_slice(&[13.3, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_247() {
        let a_inv = Tensor::from_slice(&[13.350000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_248() {
        let a_inv = Tensor::from_slice(&[13.4, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_249() {
        let a_inv = Tensor::from_slice(&[13.450000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_250() {
        let a_inv = Tensor::from_slice(&[13.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_251() {
        let a_inv = Tensor::from_slice(&[13.55, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_252() {
        let a_inv = Tensor::from_slice(&[13.600000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_253() {
        let a_inv = Tensor::from_slice(&[13.65, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_254() {
        let a_inv = Tensor::from_slice(&[13.700000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_255() {
        let a_inv = Tensor::from_slice(&[13.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_256() {
        let a_inv = Tensor::from_slice(&[13.8, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_257() {
        let a_inv = Tensor::from_slice(&[13.850000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_258() {
        let a_inv = Tensor::from_slice(&[13.9, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_259() {
        let a_inv = Tensor::from_slice(&[13.950000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_260() {
        let a_inv = Tensor::from_slice(&[14.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_261() {
        let a_inv = Tensor::from_slice(&[14.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_262() {
        let a_inv = Tensor::from_slice(&[14.100000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_263() {
        let a_inv = Tensor::from_slice(&[14.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_264() {
        let a_inv = Tensor::from_slice(&[14.200000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_265() {
        let a_inv = Tensor::from_slice(&[14.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_266() {
        let a_inv = Tensor::from_slice(&[14.3, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_267() {
        let a_inv = Tensor::from_slice(&[14.350000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_268() {
        let a_inv = Tensor::from_slice(&[14.4, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_269() {
        let a_inv = Tensor::from_slice(&[14.450000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_270() {
        let a_inv = Tensor::from_slice(&[14.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_271() {
        let a_inv = Tensor::from_slice(&[14.55, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_272() {
        let a_inv = Tensor::from_slice(&[14.600000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_273() {
        let a_inv = Tensor::from_slice(&[14.65, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_274() {
        let a_inv = Tensor::from_slice(&[14.700000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_275() {
        let a_inv = Tensor::from_slice(&[14.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_276() {
        let a_inv = Tensor::from_slice(&[14.8, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_277() {
        let a_inv = Tensor::from_slice(&[14.850000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_278() {
        let a_inv = Tensor::from_slice(&[14.9, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_279() {
        let a_inv = Tensor::from_slice(&[14.950000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_280() {
        let a_inv = Tensor::from_slice(&[15.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_281() {
        let a_inv = Tensor::from_slice(&[15.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_282() {
        let a_inv = Tensor::from_slice(&[15.100000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_283() {
        let a_inv = Tensor::from_slice(&[15.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_284() {
        let a_inv = Tensor::from_slice(&[15.200000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_285() {
        let a_inv = Tensor::from_slice(&[15.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_286() {
        let a_inv = Tensor::from_slice(&[15.3, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_287() {
        let a_inv = Tensor::from_slice(&[15.350000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_288() {
        let a_inv = Tensor::from_slice(&[15.4, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_289() {
        let a_inv = Tensor::from_slice(&[15.450000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_290() {
        let a_inv = Tensor::from_slice(&[15.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_291() {
        let a_inv = Tensor::from_slice(&[15.55, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_292() {
        let a_inv = Tensor::from_slice(&[15.600000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_293() {
        let a_inv = Tensor::from_slice(&[15.65, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_294() {
        let a_inv = Tensor::from_slice(&[15.700000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_295() {
        let a_inv = Tensor::from_slice(&[15.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_296() {
        let a_inv = Tensor::from_slice(&[15.8, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_297() {
        let a_inv = Tensor::from_slice(&[15.850000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_298() {
        let a_inv = Tensor::from_slice(&[15.9, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_299() {
        let a_inv = Tensor::from_slice(&[15.950000000000001, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_300() {
        let a_inv = Tensor::from_slice(&[16.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_301() {
        let a_inv = Tensor::from_slice(&[16.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_302() {
        let a_inv = Tensor::from_slice(&[16.1, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_303() {
        let a_inv = Tensor::from_slice(&[16.15, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_304() {
        let a_inv = Tensor::from_slice(&[16.200000000000003, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_305() {
        let a_inv = Tensor::from_slice(&[16.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_306() {
        let a_inv = Tensor::from_slice(&[16.3, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_307() {
        let a_inv = Tensor::from_slice(&[16.35, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_308() {
        let a_inv = Tensor::from_slice(&[16.4, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_309() {
        let a_inv = Tensor::from_slice(&[16.450000000000003, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_310() {
        let a_inv = Tensor::from_slice(&[16.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_311() {
        let a_inv = Tensor::from_slice(&[16.55, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_312() {
        let a_inv = Tensor::from_slice(&[16.6, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_313() {
        let a_inv = Tensor::from_slice(&[16.65, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_314() {
        let a_inv = Tensor::from_slice(&[16.700000000000003, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_315() {
        let a_inv = Tensor::from_slice(&[16.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_316() {
        let a_inv = Tensor::from_slice(&[16.8, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_317() {
        let a_inv = Tensor::from_slice(&[16.85, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_318() {
        let a_inv = Tensor::from_slice(&[16.9, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_319() {
        let a_inv = Tensor::from_slice(&[16.950000000000003, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_320() {
        let a_inv = Tensor::from_slice(&[17.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_321() {
        let a_inv = Tensor::from_slice(&[17.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_322() {
        let a_inv = Tensor::from_slice(&[17.1, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_323() {
        let a_inv = Tensor::from_slice(&[17.150000000000002, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_324() {
        let a_inv = Tensor::from_slice(&[17.2, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_325() {
        let a_inv = Tensor::from_slice(&[17.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_326() {
        let a_inv = Tensor::from_slice(&[17.3, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_327() {
        let a_inv = Tensor::from_slice(&[17.35, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_328() {
        let a_inv = Tensor::from_slice(&[17.400000000000002, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_329() {
        let a_inv = Tensor::from_slice(&[17.45, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_330() {
        let a_inv = Tensor::from_slice(&[17.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_331() {
        let a_inv = Tensor::from_slice(&[17.55, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_332() {
        let a_inv = Tensor::from_slice(&[17.6, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_333() {
        let a_inv = Tensor::from_slice(&[17.650000000000002, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_334() {
        let a_inv = Tensor::from_slice(&[17.7, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_335() {
        let a_inv = Tensor::from_slice(&[17.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_336() {
        let a_inv = Tensor::from_slice(&[17.8, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_337() {
        let a_inv = Tensor::from_slice(&[17.85, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_338() {
        let a_inv = Tensor::from_slice(&[17.900000000000002, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_339() {
        let a_inv = Tensor::from_slice(&[17.95, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_340() {
        let a_inv = Tensor::from_slice(&[18.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_341() {
        let a_inv = Tensor::from_slice(&[18.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_342() {
        let a_inv = Tensor::from_slice(&[18.1, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_343() {
        let a_inv = Tensor::from_slice(&[18.150000000000002, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_344() {
        let a_inv = Tensor::from_slice(&[18.2, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_345() {
        let a_inv = Tensor::from_slice(&[18.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_346() {
        let a_inv = Tensor::from_slice(&[18.3, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_347() {
        let a_inv = Tensor::from_slice(&[18.35, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_348() {
        let a_inv = Tensor::from_slice(&[18.400000000000002, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_349() {
        let a_inv = Tensor::from_slice(&[18.45, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_350() {
        let a_inv = Tensor::from_slice(&[18.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_351() {
        let a_inv = Tensor::from_slice(&[18.55, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_352() {
        let a_inv = Tensor::from_slice(&[18.6, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_353() {
        let a_inv = Tensor::from_slice(&[18.650000000000002, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_354() {
        let a_inv = Tensor::from_slice(&[18.7, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_355() {
        let a_inv = Tensor::from_slice(&[18.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_356() {
        let a_inv = Tensor::from_slice(&[18.8, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_357() {
        let a_inv = Tensor::from_slice(&[18.85, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_358() {
        let a_inv = Tensor::from_slice(&[18.900000000000002, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_359() {
        let a_inv = Tensor::from_slice(&[18.95, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_360() {
        let a_inv = Tensor::from_slice(&[19.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_361() {
        let a_inv = Tensor::from_slice(&[19.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_362() {
        let a_inv = Tensor::from_slice(&[19.1, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_363() {
        let a_inv = Tensor::from_slice(&[19.150000000000002, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_364() {
        let a_inv = Tensor::from_slice(&[19.2, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_365() {
        let a_inv = Tensor::from_slice(&[19.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_366() {
        let a_inv = Tensor::from_slice(&[19.3, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_367() {
        let a_inv = Tensor::from_slice(&[19.35, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_368() {
        let a_inv = Tensor::from_slice(&[19.400000000000002, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_369() {
        let a_inv = Tensor::from_slice(&[19.45, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_370() {
        let a_inv = Tensor::from_slice(&[19.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_371() {
        let a_inv = Tensor::from_slice(&[19.55, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_372() {
        let a_inv = Tensor::from_slice(&[19.6, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_373() {
        let a_inv = Tensor::from_slice(&[19.650000000000002, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_374() {
        let a_inv = Tensor::from_slice(&[19.7, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_375() {
        let a_inv = Tensor::from_slice(&[19.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_376() {
        let a_inv = Tensor::from_slice(&[19.8, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_377() {
        let a_inv = Tensor::from_slice(&[19.85, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_378() {
        let a_inv = Tensor::from_slice(&[19.900000000000002, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_379() {
        let a_inv = Tensor::from_slice(&[19.95, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_380() {
        let a_inv = Tensor::from_slice(&[20.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_381() {
        let a_inv = Tensor::from_slice(&[20.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_382() {
        let a_inv = Tensor::from_slice(&[20.1, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_383() {
        let a_inv = Tensor::from_slice(&[20.150000000000002, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_384() {
        let a_inv = Tensor::from_slice(&[20.200000000000003, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_385() {
        let a_inv = Tensor::from_slice(&[20.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_386() {
        let a_inv = Tensor::from_slice(&[20.3, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_387() {
        let a_inv = Tensor::from_slice(&[20.35, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_388() {
        let a_inv = Tensor::from_slice(&[20.400000000000002, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_389() {
        let a_inv = Tensor::from_slice(&[20.450000000000003, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_390() {
        let a_inv = Tensor::from_slice(&[20.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_391() {
        let a_inv = Tensor::from_slice(&[20.55, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_392() {
        let a_inv = Tensor::from_slice(&[20.6, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_393() {
        let a_inv = Tensor::from_slice(&[20.650000000000002, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_394() {
        let a_inv = Tensor::from_slice(&[20.700000000000003, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_395() {
        let a_inv = Tensor::from_slice(&[20.75, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_396() {
        let a_inv = Tensor::from_slice(&[20.8, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_397() {
        let a_inv = Tensor::from_slice(&[20.85, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_398() {
        let a_inv = Tensor::from_slice(&[20.900000000000002, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_399() {
        let a_inv = Tensor::from_slice(&[20.950000000000003, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_400() {
        let a_inv = Tensor::from_slice(&[21.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_401() {
        let a_inv = Tensor::from_slice(&[21.05, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_402() {
        let a_inv = Tensor::from_slice(&[21.1, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_403() {
        let a_inv = Tensor::from_slice(&[21.150000000000002, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_404() {
        let a_inv = Tensor::from_slice(&[21.200000000000003, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_405() {
        let a_inv = Tensor::from_slice(&[21.25, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_406() {
        let a_inv = Tensor::from_slice(&[21.3, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_407() {
        let a_inv = Tensor::from_slice(&[21.35, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_408() {
        let a_inv = Tensor::from_slice(&[21.400000000000002, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_409() {
        let a_inv = Tensor::from_slice(&[21.450000000000003, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_410() {
        let a_inv = Tensor::from_slice(&[21.5, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_411() {
        let a_inv = Tensor::from_slice(&[21.55, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_412() {
        let a_inv = Tensor::from_slice(&[21.6, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    #[test]
    fn test_linalg_grad_stress_413() {
        let a_inv = Tensor::from_slice(&[21.650000000000002, 0.0, 0.0, 1.0], vec![2, 2]);
        let g = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let da = grad_inv(&a_inv, &g).unwrap();
        assert_eq!(da.shape(), &[2, 2]);
    }

    // Autograd verification and gradient check padding line 0
    // Autograd verification and gradient check padding line 1
    // Autograd verification and gradient check padding line 2
    // Autograd verification and gradient check padding line 3
    // Autograd verification and gradient check padding line 4
    // Autograd verification and gradient check padding line 5
    // Autograd verification and gradient check padding line 6
}
