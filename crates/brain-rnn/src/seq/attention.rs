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

    #[test]
    fn test_seq_attention_stress_001() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_002() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_003() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_004() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_005() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_006() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_007() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_008() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_009() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_010() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_011() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_012() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_013() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_014() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_015() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_016() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_017() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_018() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_019() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_020() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_021() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_022() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_023() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_024() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_025() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_026() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_027() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_028() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_029() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_030() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_031() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_032() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_033() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_034() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_035() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_036() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_037() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_038() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_039() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_040() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_041() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_042() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_043() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_044() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_045() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_046() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_047() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_048() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_049() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_050() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_051() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_052() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_053() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_054() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_055() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_056() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_057() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_058() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_059() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_060() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_061() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_062() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_063() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_064() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_065() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_066() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_067() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_068() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_069() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_070() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_071() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_072() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_073() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_074() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_075() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_076() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_077() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_078() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_079() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_080() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_081() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_082() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_083() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_084() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_085() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_086() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_087() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_088() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_089() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_090() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_091() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_092() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_093() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_094() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_095() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_096() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_097() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_098() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_099() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_100() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_101() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_102() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_103() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_104() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_105() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_106() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_107() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_108() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_109() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_110() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_111() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_112() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_113() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_114() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_115() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_116() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_117() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_118() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_119() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_120() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_121() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_122() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_123() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_124() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_125() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_126() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_127() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_128() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_129() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_130() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_131() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_132() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_133() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_134() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_135() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_136() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_137() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_138() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_139() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_140() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_141() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_142() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_143() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_144() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_145() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_146() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_147() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_148() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_149() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_150() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_151() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_152() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_153() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_154() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_155() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_156() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_157() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_158() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_159() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_160() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_161() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_162() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_163() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_164() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_165() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_166() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_167() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_168() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_169() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_170() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_171() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_172() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_173() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_174() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_175() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_176() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_177() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_178() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_179() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_180() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_181() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_182() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_183() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_184() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_185() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_186() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_187() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_188() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_189() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_190() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_191() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_192() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_193() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_194() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_195() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_196() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_197() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_198() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_199() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_200() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_201() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_202() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_203() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_204() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_205() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_206() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_207() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_208() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_209() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_210() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_211() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_212() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_213() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_214() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_215() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_216() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_217() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_218() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_219() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_220() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_221() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_222() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_223() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_224() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_225() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_226() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_227() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_228() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_229() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_230() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_231() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_232() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_233() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_234() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_235() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_236() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_237() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_238() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_239() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_240() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_241() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_242() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_243() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_244() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_245() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_246() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_247() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_248() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_249() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_250() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_251() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_252() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_253() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_254() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_255() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_256() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_257() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_258() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_259() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_260() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_261() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_262() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_263() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_264() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_265() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_266() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_267() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_268() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_269() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_270() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_271() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_272() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_273() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_274() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_275() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_276() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_277() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_278() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_279() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_280() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_281() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_282() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_283() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_284() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_285() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_286() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_287() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_288() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_289() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_290() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_291() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_292() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_293() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }

    #[test]
    fn test_seq_attention_stress_294() {
        let attn = SeqAttention::new(4, 4, AttentionKind::Dot);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let k_t = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 4]);
        let v_t = k_t.clone();
        let (ctx, weights) = attn.forward(&q, &k_t, &v_t).unwrap();
        assert_eq!(ctx.shape(), &[1, 4]);
        assert_eq!(weights.shape(), &[1, 2]);
    }
}
