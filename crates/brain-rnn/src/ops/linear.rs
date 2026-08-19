//! # Linear Gate Combiners & State Slicing
//!
//! Fused linear gate operations $W x + U h + b$ and vector concatenations.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown, clippy::module_inception, clippy::manual_memcpy)]

use brain_core::Tensor;

/// Computes fused gate linear projection: $y = x W_x^T + h W_h^T + b$.
pub fn gate_linear(
    x: &[f64],
    h: &[f64],
    w_x: &[f64], // [out_dim, in_dim]
    w_h: &[f64], // [out_dim, hidden_dim]
    b: Option<&[f64]>,
    in_dim: usize,
    hidden_dim: usize,
    out_dim: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; out_dim];

    if let Some(bias) = b {
        for i in 0..out_dim.min(bias.len()) {
            out[i] = bias[i];
        }
    }

    for i in 0..out_dim {
        let mut sum_x = 0.0;
        for j in 0..in_dim.min(x.len()) {
            sum_x += x[j] * w_x[i * in_dim + j];
        }

        let mut sum_h = 0.0;
        for j in 0..hidden_dim.min(h.len()) {
            sum_h += h[j] * w_h[i * hidden_dim + j];
        }

        out[i] += sum_x + sum_h;
    }

    out
}

/// Concatenates two state tensors along the last feature dimension.
pub fn concat_states(a: &Tensor, b: &Tensor) -> Tensor {
    let mut data = Vec::with_capacity(a.numel() + b.numel());
    data.extend_from_slice(a.data());
    data.extend_from_slice(b.data());
    let n = data.len();
    Tensor::from_slice(&data, vec![1, n])
}

/// Splits state tensor into two halves along feature dimension.
pub fn split_states(combined: &Tensor) -> (Tensor, Tensor) {
    let d = combined.data();
    let mid = d.len() / 2;
    let a = Tensor::from_slice(&d[..mid], vec![1, mid]);
    let b = Tensor::from_slice(&d[mid..], vec![1, d.len() - mid]);
    (a, b)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown, clippy::excessive_precision)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::utils::*;
    use crate::ops::*;
    use crate::cells::*;
    use crate::seq::*;
    use crate::init_rnn::*;
    use crate::reg_ops::*;
    use crate::process::*;
    use crate::backward_ops::*;
    use crate::builder::*;
    use crate::helper::*;
    use crate::VERSION;
    use brain_core::Tensor;
}
