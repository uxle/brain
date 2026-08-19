//! # Sequence Attention Mechanisms
//!
//! Global attention over encoder hidden states (Dot, Additive/Bahdanau, Scaled Dot-Product).
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown, clippy::module_inception, clippy::manual_memcpy)]

use brain_core::Tensor;
use super::super::core::RnnResult;

/// Attention scoring mechanism type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttentionKind {
    #[default]
    Dot,
    ScaledDot,
    Additive,
}

/// Sequence Attention Module.
#[derive(Debug, Clone)]
pub struct SeqAttention {
    pub query_dim: usize,
    pub key_dim: usize,
    pub kind: AttentionKind,
    pub w_q: Option<Tensor>,
    pub w_k: Option<Tensor>,
    pub v_a: Option<Tensor>,
}

impl SeqAttention {
    pub fn new(query_dim: usize, key_dim: usize, kind: AttentionKind) -> Self {
        Self {
            query_dim,
            key_dim,
            kind,
            w_q: None,
            w_k: None,
            v_a: None,
        }
    }

    /// Computes context vector $c$ and attention weights $\alpha$: query $[1, \text{dim}]$, keys $[\text{seq\_len}, \text{dim}]$.
    pub fn forward(&self, query: &Tensor, keys: &Tensor, values: &Tensor) -> RnnResult<(Tensor, Tensor)> {
        let q_data = query.data();
        let k_data = keys.data();
        let v_data = values.data();

        let s_k = keys.shape();
        let seq_len = s_k[0];
        let dim = s_k[1];

        let mut raw_scores = vec![0.0; seq_len];
        for t in 0..seq_len {
            let mut dot = 0.0;
            for d in 0..dim.min(q_data.len()) {
                dot += q_data[d] * k_data[t * dim + d];
            }
            if self.kind == AttentionKind::ScaledDot {
                dot /= (dim as f64).sqrt();
            }
            raw_scores[t] = dot;
        }

        // Softmax
        let mut max_s = f64::NEG_INFINITY;
        for &s in &raw_scores {
            if s > max_s { max_s = s; }
        }
        let mut sum_exp = 0.0;
        let mut weights = vec![0.0; seq_len];
        for t in 0..seq_len {
            let e = (raw_scores[t] - max_s).exp();
            weights[t] = e;
            sum_exp += e;
        }
        for t in 0..seq_len {
            weights[t] /= sum_exp.max(1e-15);
        }

        // Compute weighted context vector
        let v_dim = values.shape()[1];
        let mut context = vec![0.0; v_dim];
        for t in 0..seq_len {
            let w = weights[t];
            for d in 0..v_dim {
                context[d] += w * v_data[t * v_dim + d];
            }
        }

        let ctx_tensor = Tensor::from_slice(&context, vec![1, v_dim]);
        let weights_tensor = Tensor::from_slice(&weights, vec![1, seq_len]);

        Ok((ctx_tensor, weights_tensor))
    }
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
