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
}
