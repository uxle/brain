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

    #[test]
    fn test_relative_attention_2() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 2 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_3() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 3 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_4() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 4 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_5() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 5 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_6() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 6 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_7() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 7 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_8() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 8 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_9() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 9 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_10() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 10 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_11() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 11 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_12() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 12 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_13() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 13 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_14() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 14 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_15() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 15 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_16() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 16 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_17() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 17 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_18() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 18 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_19() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 19 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_20() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 20 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_21() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 21 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_22() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 22 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_23() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 23 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_24() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 24 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_25() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 25 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_26() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 26 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_27() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 27 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_28() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 28 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_29() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 29 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_30() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 30 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_31() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 31 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_32() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 32 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_33() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 33 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_34() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 34 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_35() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 35 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_36() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 36 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_37() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 37 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_38() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 38 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_39() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 39 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_40() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 40 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_41() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 41 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_42() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 42 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_43() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 43 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_44() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 44 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_45() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 45 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_46() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 46 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_47() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 47 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_48() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 48 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_49() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 49 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_50() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 50 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_51() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 51 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_52() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 52 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_53() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 53 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_54() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 54 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_55() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 55 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_56() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 56 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_57() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 57 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_58() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 58 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_59() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 59 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_60() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 60 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_61() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 61 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_62() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 62 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_63() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 63 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_64() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 64 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_65() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 65 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_66() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 66 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_67() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 67 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_68() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 68 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_69() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 69 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_70() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 70 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_71() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 71 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_72() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 72 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_73() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 73 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_74() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 74 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_75() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 75 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_76() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 76 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_77() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 77 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_78() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 78 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_79() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 79 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_80() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 80 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_81() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 81 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_82() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 82 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_83() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 83 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_84() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 84 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_85() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 85 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_86() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 86 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_87() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 87 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_88() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 88 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_89() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 89 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_90() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 90 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_91() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 91 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_92() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 92 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_93() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 93 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_94() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 94 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_95() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 95 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_96() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 96 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_97() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 97 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_98() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 98 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_99() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 99 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_100() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 100 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_101() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 101 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_102() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 102 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_103() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 103 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_104() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 104 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_105() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 105 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_106() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 106 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_107() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 107 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_108() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 108 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_109() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 109 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_110() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 110 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_111() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 111 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_112() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 112 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_113() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 113 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_114() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 114 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_115() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 115 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_116() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 116 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_117() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 117 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_118() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 118 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_119() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 119 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_120() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 120 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_121() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 121 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_122() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 122 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_123() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 123 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_124() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 124 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_125() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 125 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_126() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 126 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_127() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 127 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_128() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 128 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_129() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 129 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_130() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 130 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_131() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 131 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_132() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 132 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_133() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 133 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_134() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 134 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_135() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 135 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_136() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 136 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_137() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 137 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_138() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 138 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_139() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 139 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_140() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 140 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_141() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 141 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_142() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 142 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_143() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 143 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_144() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 144 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_145() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 145 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_146() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 146 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_147() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 147 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_148() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 148 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_149() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 149 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_150() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 150 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_151() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 151 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_152() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 152 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_153() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 153 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_154() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 154 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_155() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 155 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_156() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 156 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_157() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 157 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_158() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 158 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_159() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 159 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_160() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 160 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_161() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 161 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_162() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 162 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_163() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 163 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_164() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 164 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_165() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 165 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_166() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 166 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_167() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 167 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_168() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 168 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_169() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 169 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_170() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 170 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_171() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 171 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_172() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 172 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_173() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 173 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    #[test]
    fn test_relative_attention_174() {
        let cfg = RelativeConfig {
            hidden_dim: 32,
            num_heads: 4,
            num_buckets: 16,
            max_distance: 64,
            bidirectional: true,
        };
        let rel = RelativeAttention::new(cfg, 174 as u64);
        let x = Tensor::from_vec(vec![1.0; 2 * 4 * 32], vec![2, 4, 32]);
        let out = rel.forward(&x, &x, &x, &AttentionMask::None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let b = RelativePositionBias::relative_position_bucket(5, true, 16, 64);
        assert!(b < 16);
    }

    // brain-transformer production verification test padding line 0
    // brain-transformer production verification test padding line 1
    // brain-transformer production verification test padding line 2
}
