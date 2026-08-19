//! # Relative Positional Bias Attention (T5 / Shaw Style)
//!
//! Relative position bias table computing distance buckets and injecting learned biases into attention scores.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::attention::scaled::scaled_dot_product_attention;
use crate::attention::{Attention, AttentionKind};
use crate::core::{AttentionMask, LinearParams, TransformerResult};
use brain_core::Tensor;

/// Configuration for relative position bias attention.
#[derive(Debug, Clone, PartialEq)]
pub struct RelativeConfig {
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of relative position buckets (typically 32).
    pub num_buckets: usize,
    /// Maximum distance threshold before logarithmic bucketing kicks in (typically 128).
    pub max_distance: usize,
    /// Whether relative position is bidirectional (encoder) or unidirectional (decoder).
    pub bidirectional: bool,
}

impl Default for RelativeConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 768,
            num_heads: 12,
            num_buckets: 32,
            max_distance: 128,
            bidirectional: true,
        }
    }
}

/// Relative position bias table computing logarithmic buckets.
#[derive(Debug, Clone)]
pub struct RelativePositionBias {
    /// Learned bias parameters `[num_heads, num_buckets]`.
    pub bias_table: Tensor,
    /// Configuration options.
    pub config: RelativeConfig,
}

impl RelativePositionBias {
    /// Creates a new `RelativePositionBias` module.
    pub fn new(config: RelativeConfig, seed: u64) -> Self {
        let total_weights = config.num_heads * config.num_buckets;
        let mut data = Vec::with_capacity(total_weights);
        for i in 0..config.num_heads {
            for j in 0..config.num_buckets {
                let s = (seed.wrapping_add((i * 1009 + j * 31 + 5) as u64)) as f64;
                data.push((s.sin() * 43758.5453).fract() * 0.1);
            }
        }
        let bias_table = Tensor::from_vec(data, vec![config.num_heads, config.num_buckets]);
        Self { bias_table, config }
    }

    /// Maps a signed relative position $(j - i)$ into a bucket index $0 \le \text{bucket} < \text{num\_buckets}$ (T5 formula).
    pub fn relative_position_bucket(
        relative_position: i64,
        bidirectional: bool,
        num_buckets: usize,
        max_distance: usize,
    ) -> usize {
        let mut n = -relative_position;
        let mut num_b = num_buckets;
        let mut max_exact = num_b / 2;

        let mut bucket = 0usize;

        if bidirectional {
            num_b /= 2;
            if n > 0 {
                bucket += num_b;
            } else {
                n = -n;
            }
            max_exact = num_b / 2;
        } else {
            n = n.max(0);
        }

        let is_small = (n as usize) < max_exact;
        if is_small {
            bucket += n as usize;
        } else {
            let val = max_exact as f64
                + ((n as f64 / max_exact as f64).ln() / (max_distance as f64 / max_exact as f64).ln())
                    * (num_b - max_exact) as f64;
            let val_idx = (val as usize).min(num_b - 1);
            bucket += val_idx;
        }

        bucket.min(num_buckets - 1)
    }

    /// Computes the 3D relative bias matrix `[num_heads, seq_q, seq_k]`.
    pub fn compute_bias(&self, seq_q: usize, seq_k: usize) -> Tensor {
        let mut bias_data = vec![0.0f64; self.config.num_heads * seq_q * seq_k];
        let table = self.bias_table.data();

        for i in 0..seq_q {
            for j in 0..seq_k {
                let rel_pos = (j as i64) - (i as i64);
                let bucket = Self::relative_position_bucket(
                    rel_pos,
                    self.config.bidirectional,
                    self.config.num_buckets,
                    self.config.max_distance,
                );

                for h in 0..self.config.num_heads {
                    let out_idx = h * seq_q * seq_k + i * seq_k + j;
                    let table_idx = h * self.config.num_buckets + bucket;
                    bias_data[out_idx] = table[table_idx];
                }
            }
        }

        Tensor::from_vec(bias_data, vec![self.config.num_heads, seq_q, seq_k])
    }
}

/// Relative Positional Bias Attention Layer.
#[derive(Debug, Clone)]
pub struct RelativeAttention {
    /// Multi-head linear projections.
    pub mha: crate::attention::multi_head::MultiHeadAttention,
    /// Relative bias module.
    pub rel_bias: RelativePositionBias,
}

impl RelativeAttention {
    /// Creates a new `RelativeAttention` layer.
    pub fn new(config: RelativeConfig, seed: u64) -> Self {
        let head_dim = config.hidden_dim / config.num_heads;
        let mha_cfg = crate::attention::multi_head::MhaConfig {
            hidden_dim: config.hidden_dim,
            num_heads: config.num_heads,
            head_dim,
            dropout: 0.0,
            bias: false,
            is_causal: !config.bidirectional,
        };
        let mha = crate::attention::multi_head::MultiHeadAttention::new(mha_cfg, seed);
        let rel_bias = RelativePositionBias::new(config, seed.wrapping_add(500));

        Self { mha, rel_bias }
    }
}

impl Attention for RelativeAttention {
    fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        _mask: &AttentionMask,
    ) -> TransformerResult<Tensor> {
        let seq_q = query.shape()[1];
        let seq_k = key.shape()[1];
        let bias_matrix = self.rel_bias.compute_bias(seq_q, seq_k);
        let combined_mask = AttentionMask::AdditiveBias(bias_matrix);

        self.mha.forward(query, key, value, &combined_mask)
    }

    fn kind(&self) -> AttentionKind {
        AttentionKind::Relative
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown, clippy::excessive_precision, clippy::float_cmp, clippy::len_zero, clippy::all)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::utils::*;
    use crate::ops::*;
    use crate::attention::*;
    use crate::attention::scaled::*;
    use crate::attention::multi_head::*;
    use crate::attention::relative::*;
    use crate::attention::flash_lite::*;
    use crate::attention::multi_query::*;
    use crate::attention::xformers_lite::*;
    use crate::position::*;
    use crate::position::rope::*;
    use crate::position::alibi::*;
    use crate::position::learned::*;
    use crate::embedding_layers::*;
    use crate::ffn::*;
    use crate::encoder::*;
    use crate::encoder::block::*;
    use crate::encoder::layer::*;
    use crate::decoder::*;
    use crate::decoder::layer::*;
    use crate::decoder::cross::*;
    use crate::head::*;
    use crate::kv_cache::*;
    use crate::generate::*;
    use crate::models::*;
    use crate::models::bert_lite::*;
    use crate::models::gpt_lite::*;
    use crate::models::t5_lite::*;
    use crate::models::llama_lite::*;
    use crate::builder::*;
    use brain_core::Tensor;

    #[test]
    fn test_relative_attention_1() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 1 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }
}
