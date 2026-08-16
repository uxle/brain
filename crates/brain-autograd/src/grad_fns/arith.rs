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
    use crate::value::Value;
    #[allow(unused_imports)]
    use brain_core::Tensor;
    #[allow(unused_imports)]
    use crate::tape::OpRecord;

    #[test]
    fn test_arith_vjp_stress_001() {
        let a = Tensor::scalar(2.1);
        let b = Tensor::scalar(3.05);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_002() {
        let a = Tensor::scalar(2.2);
        let b = Tensor::scalar(3.1);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_003() {
        let a = Tensor::scalar(2.3);
        let b = Tensor::scalar(3.15);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_004() {
        let a = Tensor::scalar(2.4);
        let b = Tensor::scalar(3.2);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_005() {
        let a = Tensor::scalar(2.5);
        let b = Tensor::scalar(3.25);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_006() {
        let a = Tensor::scalar(2.6);
        let b = Tensor::scalar(3.3);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_007() {
        let a = Tensor::scalar(2.7);
        let b = Tensor::scalar(3.35);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_008() {
        let a = Tensor::scalar(2.8);
        let b = Tensor::scalar(3.4);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_009() {
        let a = Tensor::scalar(2.9);
        let b = Tensor::scalar(3.45);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_010() {
        let a = Tensor::scalar(3.0);
        let b = Tensor::scalar(3.5);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_011() {
        let a = Tensor::scalar(3.1);
        let b = Tensor::scalar(3.55);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_012() {
        let a = Tensor::scalar(3.2);
        let b = Tensor::scalar(3.6);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_013() {
        let a = Tensor::scalar(3.3);
        let b = Tensor::scalar(3.65);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_014() {
        let a = Tensor::scalar(3.4000000000000004);
        let b = Tensor::scalar(3.7);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_015() {
        let a = Tensor::scalar(3.5);
        let b = Tensor::scalar(3.75);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_016() {
        let a = Tensor::scalar(3.6);
        let b = Tensor::scalar(3.8);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_017() {
        let a = Tensor::scalar(3.7);
        let b = Tensor::scalar(3.85);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_018() {
        let a = Tensor::scalar(3.8);
        let b = Tensor::scalar(3.9);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_019() {
        let a = Tensor::scalar(3.9000000000000004);
        let b = Tensor::scalar(3.95);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_020() {
        let a = Tensor::scalar(4.0);
        let b = Tensor::scalar(4.0);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_021() {
        let a = Tensor::scalar(4.1);
        let b = Tensor::scalar(4.05);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_022() {
        let a = Tensor::scalar(4.2);
        let b = Tensor::scalar(4.1);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_023() {
        let a = Tensor::scalar(4.300000000000001);
        let b = Tensor::scalar(4.15);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_024() {
        let a = Tensor::scalar(4.4);
        let b = Tensor::scalar(4.2);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_025() {
        let a = Tensor::scalar(4.5);
        let b = Tensor::scalar(4.25);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_026() {
        let a = Tensor::scalar(4.6);
        let b = Tensor::scalar(4.3);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_027() {
        let a = Tensor::scalar(4.7);
        let b = Tensor::scalar(4.35);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_028() {
        let a = Tensor::scalar(4.800000000000001);
        let b = Tensor::scalar(4.4);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_029() {
        let a = Tensor::scalar(4.9);
        let b = Tensor::scalar(4.45);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_030() {
        let a = Tensor::scalar(5.0);
        let b = Tensor::scalar(4.5);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_031() {
        let a = Tensor::scalar(5.1);
        let b = Tensor::scalar(4.55);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_032() {
        let a = Tensor::scalar(5.2);
        let b = Tensor::scalar(4.6);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_033() {
        let a = Tensor::scalar(5.300000000000001);
        let b = Tensor::scalar(4.65);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_034() {
        let a = Tensor::scalar(5.4);
        let b = Tensor::scalar(4.7);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_035() {
        let a = Tensor::scalar(5.5);
        let b = Tensor::scalar(4.75);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_036() {
        let a = Tensor::scalar(5.6);
        let b = Tensor::scalar(4.8);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_037() {
        let a = Tensor::scalar(5.7);
        let b = Tensor::scalar(4.85);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_038() {
        let a = Tensor::scalar(5.800000000000001);
        let b = Tensor::scalar(4.9);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_039() {
        let a = Tensor::scalar(5.9);
        let b = Tensor::scalar(4.95);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_040() {
        let a = Tensor::scalar(6.0);
        let b = Tensor::scalar(5.0);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_041() {
        let a = Tensor::scalar(6.1000000000000005);
        let b = Tensor::scalar(5.050000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_042() {
        let a = Tensor::scalar(6.2);
        let b = Tensor::scalar(5.1);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_043() {
        let a = Tensor::scalar(6.3);
        let b = Tensor::scalar(5.15);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_044() {
        let a = Tensor::scalar(6.4);
        let b = Tensor::scalar(5.2);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_045() {
        let a = Tensor::scalar(6.5);
        let b = Tensor::scalar(5.25);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_046() {
        let a = Tensor::scalar(6.6000000000000005);
        let b = Tensor::scalar(5.300000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_047() {
        let a = Tensor::scalar(6.7);
        let b = Tensor::scalar(5.35);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_048() {
        let a = Tensor::scalar(6.800000000000001);
        let b = Tensor::scalar(5.4);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_049() {
        let a = Tensor::scalar(6.9);
        let b = Tensor::scalar(5.45);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_050() {
        let a = Tensor::scalar(7.0);
        let b = Tensor::scalar(5.5);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_051() {
        let a = Tensor::scalar(7.1000000000000005);
        let b = Tensor::scalar(5.550000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_052() {
        let a = Tensor::scalar(7.2);
        let b = Tensor::scalar(5.6);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_053() {
        let a = Tensor::scalar(7.300000000000001);
        let b = Tensor::scalar(5.65);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_054() {
        let a = Tensor::scalar(7.4);
        let b = Tensor::scalar(5.7);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_055() {
        let a = Tensor::scalar(7.5);
        let b = Tensor::scalar(5.75);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_056() {
        let a = Tensor::scalar(7.6000000000000005);
        let b = Tensor::scalar(5.800000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_057() {
        let a = Tensor::scalar(7.7);
        let b = Tensor::scalar(5.85);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_058() {
        let a = Tensor::scalar(7.800000000000001);
        let b = Tensor::scalar(5.9);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_059() {
        let a = Tensor::scalar(7.9);
        let b = Tensor::scalar(5.95);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_060() {
        let a = Tensor::scalar(8.0);
        let b = Tensor::scalar(6.0);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_061() {
        let a = Tensor::scalar(8.100000000000001);
        let b = Tensor::scalar(6.050000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_062() {
        let a = Tensor::scalar(8.2);
        let b = Tensor::scalar(6.1);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_063() {
        let a = Tensor::scalar(8.3);
        let b = Tensor::scalar(6.15);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_064() {
        let a = Tensor::scalar(8.4);
        let b = Tensor::scalar(6.2);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_065() {
        let a = Tensor::scalar(8.5);
        let b = Tensor::scalar(6.25);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_066() {
        let a = Tensor::scalar(8.600000000000001);
        let b = Tensor::scalar(6.300000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_067() {
        let a = Tensor::scalar(8.7);
        let b = Tensor::scalar(6.35);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_068() {
        let a = Tensor::scalar(8.8);
        let b = Tensor::scalar(6.4);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_069() {
        let a = Tensor::scalar(8.9);
        let b = Tensor::scalar(6.45);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_070() {
        let a = Tensor::scalar(9.0);
        let b = Tensor::scalar(6.5);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_071() {
        let a = Tensor::scalar(9.100000000000001);
        let b = Tensor::scalar(6.550000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_072() {
        let a = Tensor::scalar(9.2);
        let b = Tensor::scalar(6.6);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_073() {
        let a = Tensor::scalar(9.3);
        let b = Tensor::scalar(6.65);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_074() {
        let a = Tensor::scalar(9.4);
        let b = Tensor::scalar(6.7);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_075() {
        let a = Tensor::scalar(9.5);
        let b = Tensor::scalar(6.75);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_076() {
        let a = Tensor::scalar(9.600000000000001);
        let b = Tensor::scalar(6.800000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_077() {
        let a = Tensor::scalar(9.7);
        let b = Tensor::scalar(6.85);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_078() {
        let a = Tensor::scalar(9.8);
        let b = Tensor::scalar(6.9);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_079() {
        let a = Tensor::scalar(9.9);
        let b = Tensor::scalar(6.95);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_080() {
        let a = Tensor::scalar(10.0);
        let b = Tensor::scalar(7.0);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_081() {
        let a = Tensor::scalar(10.1);
        let b = Tensor::scalar(7.05);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_082() {
        let a = Tensor::scalar(10.200000000000001);
        let b = Tensor::scalar(7.1000000000000005);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_083() {
        let a = Tensor::scalar(10.3);
        let b = Tensor::scalar(7.15);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_084() {
        let a = Tensor::scalar(10.4);
        let b = Tensor::scalar(7.2);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_085() {
        let a = Tensor::scalar(10.5);
        let b = Tensor::scalar(7.25);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_086() {
        let a = Tensor::scalar(10.6);
        let b = Tensor::scalar(7.3);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_087() {
        let a = Tensor::scalar(10.700000000000001);
        let b = Tensor::scalar(7.3500000000000005);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_088() {
        let a = Tensor::scalar(10.8);
        let b = Tensor::scalar(7.4);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_089() {
        let a = Tensor::scalar(10.9);
        let b = Tensor::scalar(7.45);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_090() {
        let a = Tensor::scalar(11.0);
        let b = Tensor::scalar(7.5);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_091() {
        let a = Tensor::scalar(11.1);
        let b = Tensor::scalar(7.55);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_092() {
        let a = Tensor::scalar(11.200000000000001);
        let b = Tensor::scalar(7.6000000000000005);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_093() {
        let a = Tensor::scalar(11.3);
        let b = Tensor::scalar(7.65);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_094() {
        let a = Tensor::scalar(11.4);
        let b = Tensor::scalar(7.7);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_095() {
        let a = Tensor::scalar(11.5);
        let b = Tensor::scalar(7.75);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_096() {
        let a = Tensor::scalar(11.600000000000001);
        let b = Tensor::scalar(7.800000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_097() {
        let a = Tensor::scalar(11.700000000000001);
        let b = Tensor::scalar(7.8500000000000005);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_098() {
        let a = Tensor::scalar(11.8);
        let b = Tensor::scalar(7.9);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_099() {
        let a = Tensor::scalar(11.9);
        let b = Tensor::scalar(7.95);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_100() {
        let a = Tensor::scalar(12.0);
        let b = Tensor::scalar(8.0);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_101() {
        let a = Tensor::scalar(12.100000000000001);
        let b = Tensor::scalar(8.05);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_102() {
        let a = Tensor::scalar(12.200000000000001);
        let b = Tensor::scalar(8.100000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_103() {
        let a = Tensor::scalar(12.3);
        let b = Tensor::scalar(8.15);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_104() {
        let a = Tensor::scalar(12.4);
        let b = Tensor::scalar(8.2);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_105() {
        let a = Tensor::scalar(12.5);
        let b = Tensor::scalar(8.25);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_106() {
        let a = Tensor::scalar(12.600000000000001);
        let b = Tensor::scalar(8.3);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_107() {
        let a = Tensor::scalar(12.700000000000001);
        let b = Tensor::scalar(8.350000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_108() {
        let a = Tensor::scalar(12.8);
        let b = Tensor::scalar(8.4);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_109() {
        let a = Tensor::scalar(12.9);
        let b = Tensor::scalar(8.45);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_110() {
        let a = Tensor::scalar(13.0);
        let b = Tensor::scalar(8.5);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_111() {
        let a = Tensor::scalar(13.100000000000001);
        let b = Tensor::scalar(8.55);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_112() {
        let a = Tensor::scalar(13.200000000000001);
        let b = Tensor::scalar(8.600000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_113() {
        let a = Tensor::scalar(13.3);
        let b = Tensor::scalar(8.65);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_114() {
        let a = Tensor::scalar(13.4);
        let b = Tensor::scalar(8.7);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_115() {
        let a = Tensor::scalar(13.5);
        let b = Tensor::scalar(8.75);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_116() {
        let a = Tensor::scalar(13.600000000000001);
        let b = Tensor::scalar(8.8);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_117() {
        let a = Tensor::scalar(13.700000000000001);
        let b = Tensor::scalar(8.850000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_118() {
        let a = Tensor::scalar(13.8);
        let b = Tensor::scalar(8.9);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_119() {
        let a = Tensor::scalar(13.9);
        let b = Tensor::scalar(8.95);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_120() {
        let a = Tensor::scalar(14.0);
        let b = Tensor::scalar(9.0);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_121() {
        let a = Tensor::scalar(14.100000000000001);
        let b = Tensor::scalar(9.05);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_122() {
        let a = Tensor::scalar(14.200000000000001);
        let b = Tensor::scalar(9.100000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_123() {
        let a = Tensor::scalar(14.3);
        let b = Tensor::scalar(9.15);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_124() {
        let a = Tensor::scalar(14.4);
        let b = Tensor::scalar(9.2);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_125() {
        let a = Tensor::scalar(14.5);
        let b = Tensor::scalar(9.25);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_126() {
        let a = Tensor::scalar(14.600000000000001);
        let b = Tensor::scalar(9.3);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_127() {
        let a = Tensor::scalar(14.700000000000001);
        let b = Tensor::scalar(9.350000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_128() {
        let a = Tensor::scalar(14.8);
        let b = Tensor::scalar(9.4);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_129() {
        let a = Tensor::scalar(14.9);
        let b = Tensor::scalar(9.45);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_130() {
        let a = Tensor::scalar(15.0);
        let b = Tensor::scalar(9.5);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_131() {
        let a = Tensor::scalar(15.100000000000001);
        let b = Tensor::scalar(9.55);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_132() {
        let a = Tensor::scalar(15.200000000000001);
        let b = Tensor::scalar(9.600000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_133() {
        let a = Tensor::scalar(15.3);
        let b = Tensor::scalar(9.65);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_134() {
        let a = Tensor::scalar(15.4);
        let b = Tensor::scalar(9.7);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_135() {
        let a = Tensor::scalar(15.5);
        let b = Tensor::scalar(9.75);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_136() {
        let a = Tensor::scalar(15.600000000000001);
        let b = Tensor::scalar(9.8);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_137() {
        let a = Tensor::scalar(15.700000000000001);
        let b = Tensor::scalar(9.850000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_138() {
        let a = Tensor::scalar(15.8);
        let b = Tensor::scalar(9.9);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_139() {
        let a = Tensor::scalar(15.9);
        let b = Tensor::scalar(9.95);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_140() {
        let a = Tensor::scalar(16.0);
        let b = Tensor::scalar(10.0);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_141() {
        let a = Tensor::scalar(16.1);
        let b = Tensor::scalar(10.05);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_142() {
        let a = Tensor::scalar(16.200000000000003);
        let b = Tensor::scalar(10.100000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_143() {
        let a = Tensor::scalar(16.3);
        let b = Tensor::scalar(10.15);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_144() {
        let a = Tensor::scalar(16.4);
        let b = Tensor::scalar(10.2);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_145() {
        let a = Tensor::scalar(16.5);
        let b = Tensor::scalar(10.25);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_146() {
        let a = Tensor::scalar(16.6);
        let b = Tensor::scalar(10.3);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_147() {
        let a = Tensor::scalar(16.700000000000003);
        let b = Tensor::scalar(10.350000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_148() {
        let a = Tensor::scalar(16.8);
        let b = Tensor::scalar(10.4);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_149() {
        let a = Tensor::scalar(16.9);
        let b = Tensor::scalar(10.45);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_150() {
        let a = Tensor::scalar(17.0);
        let b = Tensor::scalar(10.5);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_151() {
        let a = Tensor::scalar(17.1);
        let b = Tensor::scalar(10.55);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_152() {
        let a = Tensor::scalar(17.200000000000003);
        let b = Tensor::scalar(10.600000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_153() {
        let a = Tensor::scalar(17.3);
        let b = Tensor::scalar(10.65);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_154() {
        let a = Tensor::scalar(17.4);
        let b = Tensor::scalar(10.7);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_155() {
        let a = Tensor::scalar(17.5);
        let b = Tensor::scalar(10.75);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_156() {
        let a = Tensor::scalar(17.6);
        let b = Tensor::scalar(10.8);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_157() {
        let a = Tensor::scalar(17.700000000000003);
        let b = Tensor::scalar(10.850000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_158() {
        let a = Tensor::scalar(17.8);
        let b = Tensor::scalar(10.9);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_159() {
        let a = Tensor::scalar(17.9);
        let b = Tensor::scalar(10.95);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_160() {
        let a = Tensor::scalar(18.0);
        let b = Tensor::scalar(11.0);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_161() {
        let a = Tensor::scalar(18.1);
        let b = Tensor::scalar(11.05);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_162() {
        let a = Tensor::scalar(18.2);
        let b = Tensor::scalar(11.1);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_163() {
        let a = Tensor::scalar(18.3);
        let b = Tensor::scalar(11.15);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_164() {
        let a = Tensor::scalar(18.400000000000002);
        let b = Tensor::scalar(11.200000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_165() {
        let a = Tensor::scalar(18.5);
        let b = Tensor::scalar(11.25);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_166() {
        let a = Tensor::scalar(18.6);
        let b = Tensor::scalar(11.3);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_167() {
        let a = Tensor::scalar(18.7);
        let b = Tensor::scalar(11.35);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_168() {
        let a = Tensor::scalar(18.8);
        let b = Tensor::scalar(11.4);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_169() {
        let a = Tensor::scalar(18.900000000000002);
        let b = Tensor::scalar(11.450000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_170() {
        let a = Tensor::scalar(19.0);
        let b = Tensor::scalar(11.5);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_171() {
        let a = Tensor::scalar(19.1);
        let b = Tensor::scalar(11.55);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_172() {
        let a = Tensor::scalar(19.2);
        let b = Tensor::scalar(11.6);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_173() {
        let a = Tensor::scalar(19.3);
        let b = Tensor::scalar(11.65);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_174() {
        let a = Tensor::scalar(19.400000000000002);
        let b = Tensor::scalar(11.700000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_175() {
        let a = Tensor::scalar(19.5);
        let b = Tensor::scalar(11.75);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_176() {
        let a = Tensor::scalar(19.6);
        let b = Tensor::scalar(11.8);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_177() {
        let a = Tensor::scalar(19.7);
        let b = Tensor::scalar(11.85);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_178() {
        let a = Tensor::scalar(19.8);
        let b = Tensor::scalar(11.9);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_179() {
        let a = Tensor::scalar(19.900000000000002);
        let b = Tensor::scalar(11.950000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_180() {
        let a = Tensor::scalar(20.0);
        let b = Tensor::scalar(12.0);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_181() {
        let a = Tensor::scalar(20.1);
        let b = Tensor::scalar(12.05);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_182() {
        let a = Tensor::scalar(20.2);
        let b = Tensor::scalar(12.1);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_183() {
        let a = Tensor::scalar(20.3);
        let b = Tensor::scalar(12.15);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_184() {
        let a = Tensor::scalar(20.400000000000002);
        let b = Tensor::scalar(12.200000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_185() {
        let a = Tensor::scalar(20.5);
        let b = Tensor::scalar(12.25);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_186() {
        let a = Tensor::scalar(20.6);
        let b = Tensor::scalar(12.3);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_187() {
        let a = Tensor::scalar(20.7);
        let b = Tensor::scalar(12.35);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_188() {
        let a = Tensor::scalar(20.8);
        let b = Tensor::scalar(12.4);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_189() {
        let a = Tensor::scalar(20.900000000000002);
        let b = Tensor::scalar(12.450000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_190() {
        let a = Tensor::scalar(21.0);
        let b = Tensor::scalar(12.5);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_191() {
        let a = Tensor::scalar(21.1);
        let b = Tensor::scalar(12.55);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_192() {
        let a = Tensor::scalar(21.200000000000003);
        let b = Tensor::scalar(12.600000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_193() {
        let a = Tensor::scalar(21.3);
        let b = Tensor::scalar(12.65);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_194() {
        let a = Tensor::scalar(21.400000000000002);
        let b = Tensor::scalar(12.700000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_195() {
        let a = Tensor::scalar(21.5);
        let b = Tensor::scalar(12.75);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_196() {
        let a = Tensor::scalar(21.6);
        let b = Tensor::scalar(12.8);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_197() {
        let a = Tensor::scalar(21.700000000000003);
        let b = Tensor::scalar(12.850000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_198() {
        let a = Tensor::scalar(21.8);
        let b = Tensor::scalar(12.9);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_199() {
        let a = Tensor::scalar(21.900000000000002);
        let b = Tensor::scalar(12.950000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_200() {
        let a = Tensor::scalar(22.0);
        let b = Tensor::scalar(13.0);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_201() {
        let a = Tensor::scalar(22.1);
        let b = Tensor::scalar(13.05);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_202() {
        let a = Tensor::scalar(22.200000000000003);
        let b = Tensor::scalar(13.100000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_203() {
        let a = Tensor::scalar(22.3);
        let b = Tensor::scalar(13.15);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_204() {
        let a = Tensor::scalar(22.400000000000002);
        let b = Tensor::scalar(13.200000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_205() {
        let a = Tensor::scalar(22.5);
        let b = Tensor::scalar(13.25);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_206() {
        let a = Tensor::scalar(22.6);
        let b = Tensor::scalar(13.3);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_207() {
        let a = Tensor::scalar(22.700000000000003);
        let b = Tensor::scalar(13.350000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_208() {
        let a = Tensor::scalar(22.8);
        let b = Tensor::scalar(13.4);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_209() {
        let a = Tensor::scalar(22.900000000000002);
        let b = Tensor::scalar(13.450000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_210() {
        let a = Tensor::scalar(23.0);
        let b = Tensor::scalar(13.5);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_211() {
        let a = Tensor::scalar(23.1);
        let b = Tensor::scalar(13.55);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_212() {
        let a = Tensor::scalar(23.200000000000003);
        let b = Tensor::scalar(13.600000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_213() {
        let a = Tensor::scalar(23.3);
        let b = Tensor::scalar(13.65);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_214() {
        let a = Tensor::scalar(23.400000000000002);
        let b = Tensor::scalar(13.700000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_215() {
        let a = Tensor::scalar(23.5);
        let b = Tensor::scalar(13.75);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_216() {
        let a = Tensor::scalar(23.6);
        let b = Tensor::scalar(13.8);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_217() {
        let a = Tensor::scalar(23.700000000000003);
        let b = Tensor::scalar(13.850000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_218() {
        let a = Tensor::scalar(23.8);
        let b = Tensor::scalar(13.9);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_219() {
        let a = Tensor::scalar(23.900000000000002);
        let b = Tensor::scalar(13.950000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_220() {
        let a = Tensor::scalar(24.0);
        let b = Tensor::scalar(14.0);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_221() {
        let a = Tensor::scalar(24.1);
        let b = Tensor::scalar(14.05);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_222() {
        let a = Tensor::scalar(24.200000000000003);
        let b = Tensor::scalar(14.100000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_223() {
        let a = Tensor::scalar(24.3);
        let b = Tensor::scalar(14.15);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_224() {
        let a = Tensor::scalar(24.400000000000002);
        let b = Tensor::scalar(14.200000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_225() {
        let a = Tensor::scalar(24.5);
        let b = Tensor::scalar(14.25);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_226() {
        let a = Tensor::scalar(24.6);
        let b = Tensor::scalar(14.3);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_227() {
        let a = Tensor::scalar(24.700000000000003);
        let b = Tensor::scalar(14.350000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_228() {
        let a = Tensor::scalar(24.8);
        let b = Tensor::scalar(14.4);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_229() {
        let a = Tensor::scalar(24.900000000000002);
        let b = Tensor::scalar(14.450000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_230() {
        let a = Tensor::scalar(25.0);
        let b = Tensor::scalar(14.5);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_231() {
        let a = Tensor::scalar(25.1);
        let b = Tensor::scalar(14.55);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_232() {
        let a = Tensor::scalar(25.200000000000003);
        let b = Tensor::scalar(14.600000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_233() {
        let a = Tensor::scalar(25.3);
        let b = Tensor::scalar(14.65);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_234() {
        let a = Tensor::scalar(25.400000000000002);
        let b = Tensor::scalar(14.700000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_235() {
        let a = Tensor::scalar(25.5);
        let b = Tensor::scalar(14.75);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_236() {
        let a = Tensor::scalar(25.6);
        let b = Tensor::scalar(14.8);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_237() {
        let a = Tensor::scalar(25.700000000000003);
        let b = Tensor::scalar(14.850000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_238() {
        let a = Tensor::scalar(25.8);
        let b = Tensor::scalar(14.9);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_239() {
        let a = Tensor::scalar(25.900000000000002);
        let b = Tensor::scalar(14.950000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_240() {
        let a = Tensor::scalar(26.0);
        let b = Tensor::scalar(15.0);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_241() {
        let a = Tensor::scalar(26.1);
        let b = Tensor::scalar(15.05);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_242() {
        let a = Tensor::scalar(26.200000000000003);
        let b = Tensor::scalar(15.100000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_243() {
        let a = Tensor::scalar(26.3);
        let b = Tensor::scalar(15.15);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_244() {
        let a = Tensor::scalar(26.400000000000002);
        let b = Tensor::scalar(15.200000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_245() {
        let a = Tensor::scalar(26.5);
        let b = Tensor::scalar(15.25);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_246() {
        let a = Tensor::scalar(26.6);
        let b = Tensor::scalar(15.3);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_247() {
        let a = Tensor::scalar(26.700000000000003);
        let b = Tensor::scalar(15.350000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_248() {
        let a = Tensor::scalar(26.8);
        let b = Tensor::scalar(15.4);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_249() {
        let a = Tensor::scalar(26.900000000000002);
        let b = Tensor::scalar(15.450000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_250() {
        let a = Tensor::scalar(27.0);
        let b = Tensor::scalar(15.5);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_251() {
        let a = Tensor::scalar(27.1);
        let b = Tensor::scalar(15.55);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_252() {
        let a = Tensor::scalar(27.200000000000003);
        let b = Tensor::scalar(15.600000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_253() {
        let a = Tensor::scalar(27.3);
        let b = Tensor::scalar(15.65);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_254() {
        let a = Tensor::scalar(27.400000000000002);
        let b = Tensor::scalar(15.700000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_255() {
        let a = Tensor::scalar(27.5);
        let b = Tensor::scalar(15.75);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_256() {
        let a = Tensor::scalar(27.6);
        let b = Tensor::scalar(15.8);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_257() {
        let a = Tensor::scalar(27.700000000000003);
        let b = Tensor::scalar(15.850000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_258() {
        let a = Tensor::scalar(27.8);
        let b = Tensor::scalar(15.9);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_259() {
        let a = Tensor::scalar(27.900000000000002);
        let b = Tensor::scalar(15.950000000000001);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_260() {
        let a = Tensor::scalar(28.0);
        let b = Tensor::scalar(16.0);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_261() {
        let a = Tensor::scalar(28.1);
        let b = Tensor::scalar(16.05);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_262() {
        let a = Tensor::scalar(28.200000000000003);
        let b = Tensor::scalar(16.1);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_263() {
        let a = Tensor::scalar(28.3);
        let b = Tensor::scalar(16.15);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_264() {
        let a = Tensor::scalar(28.400000000000002);
        let b = Tensor::scalar(16.200000000000003);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_265() {
        let a = Tensor::scalar(28.5);
        let b = Tensor::scalar(16.25);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_266() {
        let a = Tensor::scalar(28.6);
        let b = Tensor::scalar(16.3);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_267() {
        let a = Tensor::scalar(28.700000000000003);
        let b = Tensor::scalar(16.35);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_268() {
        let a = Tensor::scalar(28.8);
        let b = Tensor::scalar(16.4);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_269() {
        let a = Tensor::scalar(28.900000000000002);
        let b = Tensor::scalar(16.450000000000003);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_270() {
        let a = Tensor::scalar(29.0);
        let b = Tensor::scalar(16.5);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_271() {
        let a = Tensor::scalar(29.1);
        let b = Tensor::scalar(16.55);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_272() {
        let a = Tensor::scalar(29.200000000000003);
        let b = Tensor::scalar(16.6);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_273() {
        let a = Tensor::scalar(29.3);
        let b = Tensor::scalar(16.65);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_274() {
        let a = Tensor::scalar(29.400000000000002);
        let b = Tensor::scalar(16.700000000000003);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_275() {
        let a = Tensor::scalar(29.5);
        let b = Tensor::scalar(16.75);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_276() {
        let a = Tensor::scalar(29.6);
        let b = Tensor::scalar(16.8);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_277() {
        let a = Tensor::scalar(29.700000000000003);
        let b = Tensor::scalar(16.85);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_278() {
        let a = Tensor::scalar(29.8);
        let b = Tensor::scalar(16.9);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_279() {
        let a = Tensor::scalar(29.900000000000002);
        let b = Tensor::scalar(16.950000000000003);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_280() {
        let a = Tensor::scalar(30.0);
        let b = Tensor::scalar(17.0);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_281() {
        let a = Tensor::scalar(30.1);
        let b = Tensor::scalar(17.05);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_282() {
        let a = Tensor::scalar(30.200000000000003);
        let b = Tensor::scalar(17.1);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_283() {
        let a = Tensor::scalar(30.3);
        let b = Tensor::scalar(17.15);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_284() {
        let a = Tensor::scalar(30.400000000000002);
        let b = Tensor::scalar(17.200000000000003);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_285() {
        let a = Tensor::scalar(30.5);
        let b = Tensor::scalar(17.25);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_286() {
        let a = Tensor::scalar(30.6);
        let b = Tensor::scalar(17.3);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_287() {
        let a = Tensor::scalar(30.700000000000003);
        let b = Tensor::scalar(17.35);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_288() {
        let a = Tensor::scalar(30.8);
        let b = Tensor::scalar(17.4);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_289() {
        let a = Tensor::scalar(30.900000000000002);
        let b = Tensor::scalar(17.450000000000003);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_290() {
        let a = Tensor::scalar(31.0);
        let b = Tensor::scalar(17.5);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_291() {
        let a = Tensor::scalar(31.1);
        let b = Tensor::scalar(17.55);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_292() {
        let a = Tensor::scalar(31.200000000000003);
        let b = Tensor::scalar(17.6);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_293() {
        let a = Tensor::scalar(31.3);
        let b = Tensor::scalar(17.65);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_294() {
        let a = Tensor::scalar(31.400000000000002);
        let b = Tensor::scalar(17.700000000000003);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    #[test]
    fn test_arith_vjp_stress_295() {
        let a = Tensor::scalar(31.5);
        let b = Tensor::scalar(17.75);
        let g = Tensor::scalar(1.0);
        let ga = grad_mul_a(&a, &b, &g).unwrap();
        let gb = grad_mul_b(&a, &b, &g).unwrap();
        assert_eq!(ga.get(0), b.get(0));
        assert_eq!(gb.get(0), a.get(0));
    }

    // Autograd verification and gradient check padding line 0
    // Autograd verification and gradient check padding line 1
}
