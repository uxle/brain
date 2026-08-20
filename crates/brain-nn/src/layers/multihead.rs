//! # Multi-Head Self & Cross Attention Module
//!
//! Vaswani et al., 2017: "Attention Is All You Need"
//!
//! ## Mathematical Formulation
//!
//! Scaled Dot-Product Attention:
//! $$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} + M \right) V$$
//!
//! Multi-Head Decomposition across $h$ parallel heads:
//! $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
//! $$\text{where } \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::module::{Module, ModuleResult};
use crate::layers::linear::Linear;
use super::attention::scaled_dot_product_attention;

/// Configuration for MultiheadAttention.
#[derive(Debug, Clone)]
pub struct MhaConfig {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub dropout: f64,
}

impl Default for MhaConfig {
    fn default() -> Self {
        Self { embed_dim: 64, num_heads: 4, dropout: 0.0 }
    }
}

/// MultiheadAttention layer module.
#[derive(Debug, Clone)]
pub struct MultiheadAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub config: MhaConfig,
}

impl MultiheadAttention {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        Self {
            q_proj: Linear::new(embed_dim, embed_dim, true),
            k_proj: Linear::new(embed_dim, embed_dim, true),
            v_proj: Linear::new(embed_dim, embed_dim, true),
            out_proj: Linear::new(embed_dim, embed_dim, true),
            config: MhaConfig { embed_dim, num_heads, dropout: 0.0 },
        }
    }

    pub fn forward_mha(&self, query: &Tensor, key: &Tensor, value: &Tensor, mask: Option<&Tensor>) -> ModuleResult<Tensor> {
        let q = self.q_proj.forward_tensor(query)?;
        let k = self.k_proj.forward_tensor(key)?;
        let v = self.v_proj.forward_tensor(value)?;

        let num_heads = self.config.num_heads.max(1);
        let embed_dim = self.config.embed_dim;
        let head_dim = (embed_dim / num_heads).max(1);

        let q_shape = q.shape();
        let batch = q_shape[0];
        let seq_q = if q_shape.len() > 1 { q_shape[1] } else { 1 };
        let seq_k = if key.shape().len() > 1 { key.shape()[1] } else { 1 };

        let q_vec = q.to_vec();
        let k_vec = k.to_vec();
        let v_vec = v.to_vec();

        let mut head_outputs = Vec::with_capacity(num_heads);

        for h in 0..num_heads {
            let mut q_h = Vec::with_capacity(batch * seq_q * head_dim);
            let mut k_h = Vec::with_capacity(batch * seq_k * head_dim);
            let mut v_h = Vec::with_capacity(batch * seq_k * head_dim);

            for b in 0..batch {
                for sq in 0..seq_q {
                    let base = (b * seq_q + sq) * embed_dim + h * head_dim;
                    q_h.extend_from_slice(&q_vec[base..base + head_dim]);
                }
                for sk in 0..seq_k {
                    let base = (b * seq_k + sk) * embed_dim + h * head_dim;
                    k_h.extend_from_slice(&k_vec[base..base + head_dim]);
                    v_h.extend_from_slice(&v_vec[base..base + head_dim]);
                }
            }

            let q_t = Tensor::from_vec(q_h, vec![batch, seq_q, head_dim]);
            let k_t = Tensor::from_vec(k_h, vec![batch, seq_k, head_dim]);
            let v_t = Tensor::from_vec(v_h, vec![batch, seq_k, head_dim]);

            let out_h = scaled_dot_product_attention(&q_t, &k_t, &v_t, mask);
            head_outputs.push(out_h.to_vec());
        }

        let mut concat_out = Vec::with_capacity(batch * seq_q * embed_dim);
        for b in 0..batch {
            for sq in 0..seq_q {
                for h in 0..num_heads {
                    let base = (b * seq_q + sq) * head_dim;
                    concat_out.extend_from_slice(&head_outputs[h][base..base + head_dim]);
                }
            }
        }

        let attn_out = Tensor::from_vec(concat_out, vec![batch, seq_q, embed_dim]);
        self.out_proj.forward_tensor(&attn_out)
    }
}

use brain_autograd::Value;

impl Module for MultiheadAttention {
    fn forward(&self, input: &Value) -> ModuleResult<Value> {
        let t_out = self.forward_mha(input.data(), input.data(), input.data(), None)?;
        Ok(Value::new(t_out, input.requires_grad()))
    }

    fn parameters(&self) -> Vec<Value> {
        let mut p = Vec::new();
        p.extend(self.q_proj.parameters());
        p.extend(self.k_proj.parameters());
        p.extend(self.v_proj.parameters());
        p.extend(self.out_proj.parameters());
        p
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
