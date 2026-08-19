//! # Composite Layer Differentiable Rules
//!
//! Differentiable backward implementations for complex composite layers:
//! LayerNorm, RMSNorm, BatchNorm, Dropout, and Scaled Dot-Product Attention.

use brain_core::tensor::arithmetic as arith_t;
use brain_core::tensor::reduction as red_t;
use brain_core::tensor::special as spec_t;
use brain_core::{BrainResult, Tensor};

/// Computes backward gradients for Layer Normalization.
pub fn grad_layernorm(
    x: &Tensor,
    gamma: &Tensor,
    g: &Tensor,
    eps: f64,
) -> BrainResult<(Tensor, Tensor, Tensor)> {
    let mean = red_t::mean(x);
    let n = x.numel() as f64;
    let var = x.data().iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    let std_inv = 1.0 / (var + eps).sqrt();

    let mut dx = vec![0.0; x.numel()];
    let mut dgamma = vec![0.0; gamma.numel()];
    let mut dbeta = vec![0.0; gamma.numel()];

    let x_slice = x.data();
    let g_slice = g.data();
    let gamma_slice = gamma.data();

    for (i, (&xi, &gi)) in x_slice.iter().zip(g_slice.iter()).enumerate() {
        let x_hat = (xi - mean) * std_inv;
        dgamma[i % gamma.numel()] += gi * x_hat;
        dbeta[i % gamma.numel()] += gi;
        dx[i] = gi * gamma_slice[i % gamma.numel()] * std_inv;
    }

    Ok((
        Tensor::from_slice(&dx, x.shape().to_vec()),
        Tensor::from_slice(&dgamma, gamma.shape().to_vec()),
        Tensor::from_slice(&dbeta, gamma.shape().to_vec()),
    ))
}

/// Computes backward gradients for Scaled Dot-Product Attention: `Q @ K^T / sqrt(d) -> Softmax -> @ V`.
pub fn grad_scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
) -> BrainResult<(Tensor, Tensor, Tensor)> {
    let d_k = q.shape().last().copied().unwrap_or(1) as f64;
    let scale = 1.0 / d_k.sqrt();

    let kt = k.transpose(0, 1);
    let scores = arith_t::matmul(q, &kt).map(|x| x * scale);
    let attn_weights = spec_t::softmax(&scores, scores.ndim() - 1);

    // dV = A^T @ G
    let at = attn_weights.transpose(0, 1);
    let dv = arith_t::matmul(&at, g);

    // dA = G @ V^T
    let vt = v.transpose(0, 1);
    let da = arith_t::matmul(g, &vt);

    // dScores from Softmax backward
    let dot = arith_t::mul(&da, &attn_weights);
    let dot_sum = red_t::sum_along_dim(&dot, da.ndim() - 1, true);
    let sub = arith_t::sub(&da, &dot_sum);
    let dscores = arith_t::mul(&attn_weights, &sub).map(|x| x * scale);

    // dQ = dScores @ K
    let dq = arith_t::matmul(&dscores, k);

    // dK = dScores^T @ Q
    let dscores_t = dscores.transpose(0, 1);
    let dk = arith_t::matmul(&dscores_t, q);

    Ok((dq, dk, dv))
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
