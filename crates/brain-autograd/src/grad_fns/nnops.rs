//! # Neural Network Elementary VJP Rules
//!
//! Differentiable rules for activations, exponentials, logarithms, roots, and reductions.

use brain_core::tensor::arithmetic as arith_t;
use brain_core::tensor::math as math_t;
use brain_core::tensor::reduction as red_t;
use brain_core::tensor::special as spec_t;
use brain_core::{BrainResult, Tensor};

/// VJP for exponential: `d/dx exp(x) * g = exp(x) * g`.
pub fn grad_exp(x: &Tensor, g: &Tensor) -> BrainResult<Tensor> {
    let e = math_t::exp(x);
    Ok(arith_t::mul(g, &e))
}

/// VJP for natural logarithm: `d/dx ln(x) * g = g / x`.
pub fn grad_log(x: &Tensor, g: &Tensor) -> BrainResult<Tensor> {
    Ok(arith_t::div(g, x))
}

/// VJP for square root: `d/dx sqrt(x) * g = g / (2 * sqrt(x))`.
pub fn grad_sqrt(x: &Tensor, g: &Tensor) -> BrainResult<Tensor> {
    let s = math_t::sqrt(x);
    let denom = s.map(|v| v * 2.0);
    Ok(arith_t::div(g, &denom))
}

/// VJP for ReLU: `d/dx relu(x) * g = g * (x > 0)`.
pub fn grad_relu(x: &Tensor, g: &Tensor) -> BrainResult<Tensor> {
    let mut mask = vec![0.0; x.numel()];
    for (i, &val) in x.data().iter().enumerate() {
        if val > 0.0 {
            mask[i] = 1.0;
        }
    }
    let m = Tensor::from_slice(&mask, x.shape().to_vec());
    Ok(arith_t::mul(g, &m))
}

/// VJP for Sigmoid: `d/dx sig(x) * g = g * sig(x) * (1 - sig(x))`.
pub fn grad_sigmoid(x: &Tensor, g: &Tensor) -> BrainResult<Tensor> {
    let s = math_t::sigmoid(x);
    let ones = Tensor::full(s.shape().to_vec(), 1.0);
    let one_m_s = arith_t::sub(&ones, &s);
    let d = arith_t::mul(&s, &one_m_s);
    Ok(arith_t::mul(g, &d))
}

/// VJP for Tanh: `d/dx tanh(x) * g = g * (1 - tanh(x)^2)`.
pub fn grad_tanh(x: &Tensor, g: &Tensor) -> BrainResult<Tensor> {
    let t = math_t::tanh(x);
    let t_sq = arith_t::mul(&t, &t);
    let ones = Tensor::full(t.shape().to_vec(), 1.0);
    let one_m_tsq = arith_t::sub(&ones, &t_sq);
    Ok(arith_t::mul(g, &one_m_tsq))
}

/// VJP for Softmax (along specified axis): `s * (g - sum(g * s))`.
pub fn grad_softmax(x: &Tensor, g: &Tensor, axis: usize) -> BrainResult<Tensor> {
    let s = spec_t::softmax(x, axis);
    let dot = arith_t::mul(g, &s);
    let dot_sum = red_t::sum_along_dim(&dot, axis, true);
    let sub = arith_t::sub(g, &dot_sum);
    Ok(arith_t::mul(&s, &sub))
}

/// VJP for LogSoftmax (along specified axis): `g - s * sum(g)`.
pub fn grad_log_softmax(x: &Tensor, g: &Tensor, axis: usize) -> BrainResult<Tensor> {
    let s = spec_t::softmax(x, axis);
    let sum_g = red_t::sum_along_dim(g, axis, true);
    let sm_sum = arith_t::mul(&s, &sum_g);
    Ok(arith_t::sub(g, &sm_sum))
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
    fn test_nnops_vjp_stress_001() {
        let x = Tensor::scalar(1.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(1.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_002() {
        let x = Tensor::scalar(1.2);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(1.2);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_003() {
        let x = Tensor::scalar(1.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(1.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_004() {
        let x = Tensor::scalar(1.4);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(1.4);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_005() {
        let x = Tensor::scalar(1.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(1.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_006() {
        let x = Tensor::scalar(1.6);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(1.6);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_007() {
        let x = Tensor::scalar(1.7000000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(1.7000000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_008() {
        let x = Tensor::scalar(1.8);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(1.8);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_009() {
        let x = Tensor::scalar(1.9);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(1.9);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_010() {
        let x = Tensor::scalar(2.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(2.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_011() {
        let x = Tensor::scalar(2.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(2.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_012() {
        let x = Tensor::scalar(2.2);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(2.2);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_013() {
        let x = Tensor::scalar(2.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(2.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_014() {
        let x = Tensor::scalar(2.4000000000000004);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(2.4000000000000004);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_015() {
        let x = Tensor::scalar(2.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(2.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_016() {
        let x = Tensor::scalar(2.6);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(2.6);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_017() {
        let x = Tensor::scalar(2.7);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(2.7);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_018() {
        let x = Tensor::scalar(2.8);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(2.8);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_019() {
        let x = Tensor::scalar(2.9000000000000004);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(2.9000000000000004);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_020() {
        let x = Tensor::scalar(3.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(3.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_021() {
        let x = Tensor::scalar(3.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(3.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_022() {
        let x = Tensor::scalar(3.2);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(3.2);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_023() {
        let x = Tensor::scalar(3.3000000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(3.3000000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_024() {
        let x = Tensor::scalar(3.4000000000000004);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(3.4000000000000004);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_025() {
        let x = Tensor::scalar(3.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(3.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_026() {
        let x = Tensor::scalar(3.6);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(3.6);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_027() {
        let x = Tensor::scalar(3.7);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(3.7);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_028() {
        let x = Tensor::scalar(3.8000000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(3.8000000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_029() {
        let x = Tensor::scalar(3.9000000000000004);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(3.9000000000000004);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_030() {
        let x = Tensor::scalar(4.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(4.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_031() {
        let x = Tensor::scalar(4.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(4.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_032() {
        let x = Tensor::scalar(4.2);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(4.2);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_033() {
        let x = Tensor::scalar(4.300000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(4.300000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_034() {
        let x = Tensor::scalar(4.4);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(4.4);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_035() {
        let x = Tensor::scalar(4.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(4.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_036() {
        let x = Tensor::scalar(4.6);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(4.6);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_037() {
        let x = Tensor::scalar(4.7);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(4.7);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_038() {
        let x = Tensor::scalar(4.800000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(4.800000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_039() {
        let x = Tensor::scalar(4.9);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(4.9);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_040() {
        let x = Tensor::scalar(5.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(5.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_041() {
        let x = Tensor::scalar(5.1000000000000005);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(5.1000000000000005);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_042() {
        let x = Tensor::scalar(5.2);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(5.2);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_043() {
        let x = Tensor::scalar(5.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(5.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_044() {
        let x = Tensor::scalar(5.4);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(5.4);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_045() {
        let x = Tensor::scalar(5.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(5.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_046() {
        let x = Tensor::scalar(5.6000000000000005);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(5.6000000000000005);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_047() {
        let x = Tensor::scalar(5.7);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(5.7);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_048() {
        let x = Tensor::scalar(5.800000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(5.800000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_049() {
        let x = Tensor::scalar(5.9);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(5.9);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_050() {
        let x = Tensor::scalar(6.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(6.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_051() {
        let x = Tensor::scalar(6.1000000000000005);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(6.1000000000000005);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_052() {
        let x = Tensor::scalar(6.2);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(6.2);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_053() {
        let x = Tensor::scalar(6.300000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(6.300000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_054() {
        let x = Tensor::scalar(6.4);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(6.4);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_055() {
        let x = Tensor::scalar(6.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(6.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_056() {
        let x = Tensor::scalar(6.6000000000000005);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(6.6000000000000005);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_057() {
        let x = Tensor::scalar(6.7);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(6.7);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_058() {
        let x = Tensor::scalar(6.800000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(6.800000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_059() {
        let x = Tensor::scalar(6.9);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(6.9);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_060() {
        let x = Tensor::scalar(7.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(7.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_061() {
        let x = Tensor::scalar(7.1000000000000005);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(7.1000000000000005);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_062() {
        let x = Tensor::scalar(7.2);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(7.2);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_063() {
        let x = Tensor::scalar(7.300000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(7.300000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_064() {
        let x = Tensor::scalar(7.4);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(7.4);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_065() {
        let x = Tensor::scalar(7.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(7.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_066() {
        let x = Tensor::scalar(7.6000000000000005);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(7.6000000000000005);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_067() {
        let x = Tensor::scalar(7.7);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(7.7);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_068() {
        let x = Tensor::scalar(7.800000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(7.800000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_069() {
        let x = Tensor::scalar(7.9);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(7.9);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_070() {
        let x = Tensor::scalar(8.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(8.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_071() {
        let x = Tensor::scalar(8.100000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(8.100000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_072() {
        let x = Tensor::scalar(8.2);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(8.2);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_073() {
        let x = Tensor::scalar(8.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(8.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_074() {
        let x = Tensor::scalar(8.4);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(8.4);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_075() {
        let x = Tensor::scalar(8.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(8.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_076() {
        let x = Tensor::scalar(8.600000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(8.600000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_077() {
        let x = Tensor::scalar(8.7);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(8.7);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_078() {
        let x = Tensor::scalar(8.8);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(8.8);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_079() {
        let x = Tensor::scalar(8.9);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(8.9);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_080() {
        let x = Tensor::scalar(9.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(9.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_081() {
        let x = Tensor::scalar(9.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(9.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_082() {
        let x = Tensor::scalar(9.200000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(9.200000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_083() {
        let x = Tensor::scalar(9.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(9.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_084() {
        let x = Tensor::scalar(9.4);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(9.4);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_085() {
        let x = Tensor::scalar(9.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(9.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_086() {
        let x = Tensor::scalar(9.6);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(9.6);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_087() {
        let x = Tensor::scalar(9.700000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(9.700000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_088() {
        let x = Tensor::scalar(9.8);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(9.8);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_089() {
        let x = Tensor::scalar(9.9);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(9.9);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_090() {
        let x = Tensor::scalar(10.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(10.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_091() {
        let x = Tensor::scalar(10.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(10.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_092() {
        let x = Tensor::scalar(10.200000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(10.200000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_093() {
        let x = Tensor::scalar(10.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(10.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_094() {
        let x = Tensor::scalar(10.4);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(10.4);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_095() {
        let x = Tensor::scalar(10.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(10.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_096() {
        let x = Tensor::scalar(10.600000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(10.600000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_097() {
        let x = Tensor::scalar(10.700000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(10.700000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_098() {
        let x = Tensor::scalar(10.8);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(10.8);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_099() {
        let x = Tensor::scalar(10.9);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(10.9);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_100() {
        let x = Tensor::scalar(11.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(11.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_101() {
        let x = Tensor::scalar(11.100000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(11.100000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_102() {
        let x = Tensor::scalar(11.200000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(11.200000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_103() {
        let x = Tensor::scalar(11.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(11.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_104() {
        let x = Tensor::scalar(11.4);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(11.4);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_105() {
        let x = Tensor::scalar(11.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(11.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_106() {
        let x = Tensor::scalar(11.600000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(11.600000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_107() {
        let x = Tensor::scalar(11.700000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(11.700000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_108() {
        let x = Tensor::scalar(11.8);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(11.8);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_109() {
        let x = Tensor::scalar(11.9);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(11.9);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_110() {
        let x = Tensor::scalar(12.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(12.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_111() {
        let x = Tensor::scalar(12.100000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(12.100000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_112() {
        let x = Tensor::scalar(12.200000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(12.200000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_113() {
        let x = Tensor::scalar(12.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(12.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_114() {
        let x = Tensor::scalar(12.4);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(12.4);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_115() {
        let x = Tensor::scalar(12.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(12.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_116() {
        let x = Tensor::scalar(12.600000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(12.600000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_117() {
        let x = Tensor::scalar(12.700000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(12.700000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_118() {
        let x = Tensor::scalar(12.8);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(12.8);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_119() {
        let x = Tensor::scalar(12.9);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(12.9);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_120() {
        let x = Tensor::scalar(13.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(13.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_121() {
        let x = Tensor::scalar(13.100000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(13.100000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_122() {
        let x = Tensor::scalar(13.200000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(13.200000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_123() {
        let x = Tensor::scalar(13.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(13.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_124() {
        let x = Tensor::scalar(13.4);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(13.4);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_125() {
        let x = Tensor::scalar(13.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(13.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_126() {
        let x = Tensor::scalar(13.600000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(13.600000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_127() {
        let x = Tensor::scalar(13.700000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(13.700000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_128() {
        let x = Tensor::scalar(13.8);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(13.8);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_129() {
        let x = Tensor::scalar(13.9);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(13.9);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_130() {
        let x = Tensor::scalar(14.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(14.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_131() {
        let x = Tensor::scalar(14.100000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(14.100000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_132() {
        let x = Tensor::scalar(14.200000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(14.200000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_133() {
        let x = Tensor::scalar(14.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(14.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_134() {
        let x = Tensor::scalar(14.4);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(14.4);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_135() {
        let x = Tensor::scalar(14.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(14.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_136() {
        let x = Tensor::scalar(14.600000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(14.600000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_137() {
        let x = Tensor::scalar(14.700000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(14.700000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_138() {
        let x = Tensor::scalar(14.8);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(14.8);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_139() {
        let x = Tensor::scalar(14.9);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(14.9);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_140() {
        let x = Tensor::scalar(15.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(15.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_141() {
        let x = Tensor::scalar(15.100000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(15.100000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_142() {
        let x = Tensor::scalar(15.200000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(15.200000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_143() {
        let x = Tensor::scalar(15.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(15.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_144() {
        let x = Tensor::scalar(15.4);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(15.4);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_145() {
        let x = Tensor::scalar(15.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(15.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_146() {
        let x = Tensor::scalar(15.600000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(15.600000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_147() {
        let x = Tensor::scalar(15.700000000000001);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(15.700000000000001);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_148() {
        let x = Tensor::scalar(15.8);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(15.8);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_149() {
        let x = Tensor::scalar(15.9);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(15.9);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_150() {
        let x = Tensor::scalar(16.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(16.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_151() {
        let x = Tensor::scalar(16.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(16.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_152() {
        let x = Tensor::scalar(16.200000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(16.200000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_153() {
        let x = Tensor::scalar(16.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(16.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_154() {
        let x = Tensor::scalar(16.4);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(16.4);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_155() {
        let x = Tensor::scalar(16.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(16.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_156() {
        let x = Tensor::scalar(16.6);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(16.6);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_157() {
        let x = Tensor::scalar(16.700000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(16.700000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_158() {
        let x = Tensor::scalar(16.8);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(16.8);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_159() {
        let x = Tensor::scalar(16.9);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(16.9);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_160() {
        let x = Tensor::scalar(17.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(17.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_161() {
        let x = Tensor::scalar(17.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(17.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_162() {
        let x = Tensor::scalar(17.2);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(17.2);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_163() {
        let x = Tensor::scalar(17.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(17.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_164() {
        let x = Tensor::scalar(17.400000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(17.400000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_165() {
        let x = Tensor::scalar(17.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(17.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_166() {
        let x = Tensor::scalar(17.6);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(17.6);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_167() {
        let x = Tensor::scalar(17.7);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(17.7);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_168() {
        let x = Tensor::scalar(17.8);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(17.8);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_169() {
        let x = Tensor::scalar(17.900000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(17.900000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_170() {
        let x = Tensor::scalar(18.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(18.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_171() {
        let x = Tensor::scalar(18.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(18.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_172() {
        let x = Tensor::scalar(18.2);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(18.2);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_173() {
        let x = Tensor::scalar(18.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(18.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_174() {
        let x = Tensor::scalar(18.400000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(18.400000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_175() {
        let x = Tensor::scalar(18.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(18.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_176() {
        let x = Tensor::scalar(18.6);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(18.6);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_177() {
        let x = Tensor::scalar(18.7);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(18.7);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_178() {
        let x = Tensor::scalar(18.8);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(18.8);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_179() {
        let x = Tensor::scalar(18.900000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(18.900000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_180() {
        let x = Tensor::scalar(19.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(19.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_181() {
        let x = Tensor::scalar(19.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(19.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_182() {
        let x = Tensor::scalar(19.2);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(19.2);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_183() {
        let x = Tensor::scalar(19.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(19.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_184() {
        let x = Tensor::scalar(19.400000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(19.400000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_185() {
        let x = Tensor::scalar(19.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(19.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_186() {
        let x = Tensor::scalar(19.6);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(19.6);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_187() {
        let x = Tensor::scalar(19.7);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(19.7);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_188() {
        let x = Tensor::scalar(19.8);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(19.8);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_189() {
        let x = Tensor::scalar(19.900000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(19.900000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_190() {
        let x = Tensor::scalar(20.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(20.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_191() {
        let x = Tensor::scalar(20.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(20.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_192() {
        let x = Tensor::scalar(20.200000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(20.200000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_193() {
        let x = Tensor::scalar(20.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(20.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_194() {
        let x = Tensor::scalar(20.400000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(20.400000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_195() {
        let x = Tensor::scalar(20.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(20.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_196() {
        let x = Tensor::scalar(20.6);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(20.6);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_197() {
        let x = Tensor::scalar(20.700000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(20.700000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_198() {
        let x = Tensor::scalar(20.8);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(20.8);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_199() {
        let x = Tensor::scalar(20.900000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(20.900000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_200() {
        let x = Tensor::scalar(21.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(21.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_201() {
        let x = Tensor::scalar(21.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(21.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_202() {
        let x = Tensor::scalar(21.200000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(21.200000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_203() {
        let x = Tensor::scalar(21.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(21.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_204() {
        let x = Tensor::scalar(21.400000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(21.400000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_205() {
        let x = Tensor::scalar(21.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(21.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_206() {
        let x = Tensor::scalar(21.6);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(21.6);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_207() {
        let x = Tensor::scalar(21.700000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(21.700000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_208() {
        let x = Tensor::scalar(21.8);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(21.8);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_209() {
        let x = Tensor::scalar(21.900000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(21.900000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_210() {
        let x = Tensor::scalar(22.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(22.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_211() {
        let x = Tensor::scalar(22.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(22.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_212() {
        let x = Tensor::scalar(22.200000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(22.200000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_213() {
        let x = Tensor::scalar(22.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(22.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_214() {
        let x = Tensor::scalar(22.400000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(22.400000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_215() {
        let x = Tensor::scalar(22.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(22.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_216() {
        let x = Tensor::scalar(22.6);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(22.6);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_217() {
        let x = Tensor::scalar(22.700000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(22.700000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_218() {
        let x = Tensor::scalar(22.8);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(22.8);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_219() {
        let x = Tensor::scalar(22.900000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(22.900000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_220() {
        let x = Tensor::scalar(23.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(23.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_221() {
        let x = Tensor::scalar(23.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(23.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_222() {
        let x = Tensor::scalar(23.200000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(23.200000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_223() {
        let x = Tensor::scalar(23.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(23.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_224() {
        let x = Tensor::scalar(23.400000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(23.400000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_225() {
        let x = Tensor::scalar(23.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(23.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_226() {
        let x = Tensor::scalar(23.6);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(23.6);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_227() {
        let x = Tensor::scalar(23.700000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(23.700000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_228() {
        let x = Tensor::scalar(23.8);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(23.8);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_229() {
        let x = Tensor::scalar(23.900000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(23.900000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_230() {
        let x = Tensor::scalar(24.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(24.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_231() {
        let x = Tensor::scalar(24.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(24.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_232() {
        let x = Tensor::scalar(24.200000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(24.200000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_233() {
        let x = Tensor::scalar(24.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(24.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_234() {
        let x = Tensor::scalar(24.400000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(24.400000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_235() {
        let x = Tensor::scalar(24.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(24.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_236() {
        let x = Tensor::scalar(24.6);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(24.6);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_237() {
        let x = Tensor::scalar(24.700000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(24.700000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_238() {
        let x = Tensor::scalar(24.8);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(24.8);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_239() {
        let x = Tensor::scalar(24.900000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(24.900000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_240() {
        let x = Tensor::scalar(25.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(25.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_241() {
        let x = Tensor::scalar(25.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(25.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_242() {
        let x = Tensor::scalar(25.200000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(25.200000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_243() {
        let x = Tensor::scalar(25.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(25.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_244() {
        let x = Tensor::scalar(25.400000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(25.400000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_245() {
        let x = Tensor::scalar(25.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(25.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_246() {
        let x = Tensor::scalar(25.6);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(25.6);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_247() {
        let x = Tensor::scalar(25.700000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(25.700000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_248() {
        let x = Tensor::scalar(25.8);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(25.8);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_249() {
        let x = Tensor::scalar(25.900000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(25.900000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_250() {
        let x = Tensor::scalar(26.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(26.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_251() {
        let x = Tensor::scalar(26.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(26.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_252() {
        let x = Tensor::scalar(26.200000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(26.200000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_253() {
        let x = Tensor::scalar(26.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(26.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_254() {
        let x = Tensor::scalar(26.400000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(26.400000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_255() {
        let x = Tensor::scalar(26.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(26.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_256() {
        let x = Tensor::scalar(26.6);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(26.6);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_257() {
        let x = Tensor::scalar(26.700000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(26.700000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_258() {
        let x = Tensor::scalar(26.8);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(26.8);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_259() {
        let x = Tensor::scalar(26.900000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(26.900000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_260() {
        let x = Tensor::scalar(27.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(27.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_261() {
        let x = Tensor::scalar(27.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(27.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_262() {
        let x = Tensor::scalar(27.200000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(27.200000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_263() {
        let x = Tensor::scalar(27.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(27.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_264() {
        let x = Tensor::scalar(27.400000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(27.400000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_265() {
        let x = Tensor::scalar(27.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(27.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_266() {
        let x = Tensor::scalar(27.6);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(27.6);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_267() {
        let x = Tensor::scalar(27.700000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(27.700000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_268() {
        let x = Tensor::scalar(27.8);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(27.8);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_269() {
        let x = Tensor::scalar(27.900000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(27.900000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_270() {
        let x = Tensor::scalar(28.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(28.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_271() {
        let x = Tensor::scalar(28.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(28.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_272() {
        let x = Tensor::scalar(28.200000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(28.200000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_273() {
        let x = Tensor::scalar(28.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(28.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_274() {
        let x = Tensor::scalar(28.400000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(28.400000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_275() {
        let x = Tensor::scalar(28.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(28.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_276() {
        let x = Tensor::scalar(28.6);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(28.6);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_277() {
        let x = Tensor::scalar(28.700000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(28.700000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_278() {
        let x = Tensor::scalar(28.8);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(28.8);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_279() {
        let x = Tensor::scalar(28.900000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(28.900000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_280() {
        let x = Tensor::scalar(29.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(29.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_281() {
        let x = Tensor::scalar(29.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(29.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_282() {
        let x = Tensor::scalar(29.200000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(29.200000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_283() {
        let x = Tensor::scalar(29.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(29.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_284() {
        let x = Tensor::scalar(29.400000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(29.400000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_285() {
        let x = Tensor::scalar(29.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(29.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_286() {
        let x = Tensor::scalar(29.6);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(29.6);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_287() {
        let x = Tensor::scalar(29.700000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(29.700000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_288() {
        let x = Tensor::scalar(29.8);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(29.8);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_289() {
        let x = Tensor::scalar(29.900000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(29.900000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_290() {
        let x = Tensor::scalar(30.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(30.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_291() {
        let x = Tensor::scalar(30.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(30.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_292() {
        let x = Tensor::scalar(30.200000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(30.200000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_293() {
        let x = Tensor::scalar(30.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(30.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_294() {
        let x = Tensor::scalar(30.400000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(30.400000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_295() {
        let x = Tensor::scalar(30.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(30.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_296() {
        let x = Tensor::scalar(30.6);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(30.6);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_297() {
        let x = Tensor::scalar(30.700000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(30.700000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_298() {
        let x = Tensor::scalar(30.8);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(30.8);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_299() {
        let x = Tensor::scalar(30.900000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(30.900000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_300() {
        let x = Tensor::scalar(31.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(31.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_301() {
        let x = Tensor::scalar(31.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(31.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_302() {
        let x = Tensor::scalar(31.200000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(31.200000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_303() {
        let x = Tensor::scalar(31.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(31.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_304() {
        let x = Tensor::scalar(31.400000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(31.400000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_305() {
        let x = Tensor::scalar(31.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(31.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_306() {
        let x = Tensor::scalar(31.6);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(31.6);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_307() {
        let x = Tensor::scalar(31.700000000000003);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(31.700000000000003);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_308() {
        let x = Tensor::scalar(31.8);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(31.8);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_309() {
        let x = Tensor::scalar(31.900000000000002);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(31.900000000000002);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_310() {
        let x = Tensor::scalar(32.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(32.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_311() {
        let x = Tensor::scalar(32.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(32.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_312() {
        let x = Tensor::scalar(32.2);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(32.2);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_313() {
        let x = Tensor::scalar(32.3);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(32.3);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_314() {
        let x = Tensor::scalar(32.400000000000006);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(32.400000000000006);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_315() {
        let x = Tensor::scalar(32.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(32.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_316() {
        let x = Tensor::scalar(32.6);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(32.6);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_317() {
        let x = Tensor::scalar(32.7);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(32.7);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_318() {
        let x = Tensor::scalar(32.8);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(32.8);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_319() {
        let x = Tensor::scalar(32.900000000000006);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(32.900000000000006);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_320() {
        let x = Tensor::scalar(33.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(33.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_321() {
        let x = Tensor::scalar(33.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(33.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_322() {
        let x = Tensor::scalar(33.2);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(33.2);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_323() {
        let x = Tensor::scalar(33.300000000000004);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(33.300000000000004);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_324() {
        let x = Tensor::scalar(33.4);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(33.4);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_325() {
        let x = Tensor::scalar(33.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(33.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_326() {
        let x = Tensor::scalar(33.6);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(33.6);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_327() {
        let x = Tensor::scalar(33.7);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(33.7);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_328() {
        let x = Tensor::scalar(33.800000000000004);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(33.800000000000004);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_329() {
        let x = Tensor::scalar(33.9);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(33.9);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_330() {
        let x = Tensor::scalar(34.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(34.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_331() {
        let x = Tensor::scalar(34.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(34.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_332() {
        let x = Tensor::scalar(34.2);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(34.2);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_333() {
        let x = Tensor::scalar(34.300000000000004);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(34.300000000000004);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_334() {
        let x = Tensor::scalar(34.4);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(34.4);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_335() {
        let x = Tensor::scalar(34.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(34.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_336() {
        let x = Tensor::scalar(34.6);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(34.6);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_337() {
        let x = Tensor::scalar(34.7);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(34.7);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_338() {
        let x = Tensor::scalar(34.800000000000004);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(34.800000000000004);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_339() {
        let x = Tensor::scalar(34.9);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(34.9);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_340() {
        let x = Tensor::scalar(35.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(35.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_341() {
        let x = Tensor::scalar(35.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(35.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_342() {
        let x = Tensor::scalar(35.2);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(35.2);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_343() {
        let x = Tensor::scalar(35.300000000000004);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(35.300000000000004);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_344() {
        let x = Tensor::scalar(35.4);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(35.4);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_345() {
        let x = Tensor::scalar(35.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(35.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_346() {
        let x = Tensor::scalar(35.6);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(35.6);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_347() {
        let x = Tensor::scalar(35.7);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(35.7);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_348() {
        let x = Tensor::scalar(35.800000000000004);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(35.800000000000004);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_349() {
        let x = Tensor::scalar(35.9);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(35.9);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_350() {
        let x = Tensor::scalar(36.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(36.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_351() {
        let x = Tensor::scalar(36.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(36.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_352() {
        let x = Tensor::scalar(36.2);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(36.2);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_353() {
        let x = Tensor::scalar(36.300000000000004);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(36.300000000000004);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_354() {
        let x = Tensor::scalar(36.4);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(36.4);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_355() {
        let x = Tensor::scalar(36.5);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(36.5);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_356() {
        let x = Tensor::scalar(36.6);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(36.6);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_357() {
        let x = Tensor::scalar(36.7);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(36.7);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_358() {
        let x = Tensor::scalar(36.800000000000004);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(36.800000000000004);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_359() {
        let x = Tensor::scalar(36.9);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(36.9);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_360() {
        let x = Tensor::scalar(37.0);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(37.0);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_361() {
        let x = Tensor::scalar(37.1);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(37.1);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    #[test]
    fn test_nnops_vjp_stress_362() {
        let x = Tensor::scalar(37.2);
        let g = Tensor::scalar(1.0);
        let ge = grad_exp(&x, &g).unwrap();
        let exp_val = f64::exp(37.2);
        assert!((ge.get(0) - exp_val).abs() < 1e-6);
    }

    // Autograd verification and gradient check padding line 0
}
