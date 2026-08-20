//! # Arithmetic & Matrix VJP Rules
//!
//! Differentiable rules for binary arithmetic and matrix multiplications.

use crate::grad_fns::util::sum_to_shape;
use brain_core::tensor::arithmetic as arith_t;
use brain_core::tensor::math as math_t;
use brain_core::{BrainResult, Tensor};

/// VJP for addition operand A: `d/da (a + b) * g`.
pub fn grad_add_a(a: &Tensor, _b: &Tensor, g: &Tensor) -> BrainResult<Tensor> {
    sum_to_shape(g, a.shape())
}

/// VJP for addition operand B: `d/db (a + b) * g`.
pub fn grad_add_b(_a: &Tensor, b: &Tensor, g: &Tensor) -> BrainResult<Tensor> {
    sum_to_shape(g, b.shape())
}

/// VJP for subtraction operand A: `d/da (a - b) * g`.
pub fn grad_sub_a(a: &Tensor, _b: &Tensor, g: &Tensor) -> BrainResult<Tensor> {
    sum_to_shape(g, a.shape())
}

/// VJP for subtraction operand B: `d/db (a - b) * g = -g`.
pub fn grad_sub_b(_a: &Tensor, b: &Tensor, g: &Tensor) -> BrainResult<Tensor> {
    let neg_g = g.map(|x| -x);
    sum_to_shape(&neg_g, b.shape())
}

/// VJP for multiplication operand A: `d/da (a * b) * g = b * g`.
pub fn grad_mul_a(a: &Tensor, b: &Tensor, g: &Tensor) -> BrainResult<Tensor> {
    let prod = arith_t::mul(g, b);
    sum_to_shape(&prod, a.shape())
}

/// VJP for multiplication operand B: `d/db (a * b) * g = a * g`.
pub fn grad_mul_b(a: &Tensor, b: &Tensor, g: &Tensor) -> BrainResult<Tensor> {
    let prod = arith_t::mul(g, a);
    sum_to_shape(&prod, b.shape())
}

/// VJP for division operand A: `d/da (a / b) * g = g / b`.
pub fn grad_div_a(a: &Tensor, b: &Tensor, g: &Tensor) -> BrainResult<Tensor> {
    let div = arith_t::div(g, b);
    sum_to_shape(&div, a.shape())
}

/// VJP for division operand B: `d/db (a / b) * g = - (a * g) / b^2`.
pub fn grad_div_b(a: &Tensor, b: &Tensor, g: &Tensor) -> BrainResult<Tensor> {
    let num = arith_t::mul(g, a).map(|x| -x);
    let b_sq = arith_t::mul(b, b);
    let div = arith_t::div(&num, &b_sq);
    sum_to_shape(&div, b.shape())
}

/// VJP for power operand A: `d/da (a^b) * g = b * a^(b-1) * g`.
pub fn grad_pow_a(a: &Tensor, b: &Tensor, g: &Tensor) -> BrainResult<Tensor> {
    let ones = Tensor::full(b.shape().to_vec(), 1.0);
    let b_m_1 = arith_t::sub(b, &ones);
    let a_pow = arith_t::pow_tensors(a, &b_m_1);
    let da = arith_t::mul(b, &a_pow);
    let full = arith_t::mul(g, &da);
    sum_to_shape(&full, a.shape())
}

/// VJP for power operand B: `d/db (a^b) * g = a^b * ln(a) * g`.
pub fn grad_pow_b(a: &Tensor, b: &Tensor, g: &Tensor) -> BrainResult<Tensor> {
    let a_pow_b = arith_t::pow_tensors(a, b);
    let ln_a = math_t::log(a);
    let db = arith_t::mul(&a_pow_b, &ln_a);
    let full = arith_t::mul(g, &db);
    sum_to_shape(&full, b.shape())
}

/// VJP for matrix multiplication operand A: `d/da (A @ B) * G = G @ B^T`.
pub fn grad_matmul_a(_a: &Tensor, b: &Tensor, g: &Tensor) -> BrainResult<Tensor> {
    let bt = b.transpose(0, 1);
    Ok(arith_t::matmul(g, &bt))
}

/// VJP for matrix multiplication operand B: `d/db (A @ B) * G = A^T @ G`.
pub fn grad_matmul_b(a: &Tensor, _b: &Tensor, g: &Tensor) -> BrainResult<Tensor> {
    let at = a.transpose(0, 1);
    Ok(arith_t::matmul(&at, g))
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::tape::OpRecord;
    #[allow(unused_imports)]
    use crate::value::Value;
    #[allow(unused_imports)]
    use brain_core::Tensor;
}
