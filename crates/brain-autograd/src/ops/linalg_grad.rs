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
}
