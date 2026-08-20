//! # Learned and Sinusoidal Positional Encodings
//!
//! Absolute learned positional embedding tables and Vaswani et al. fixed sinusoidal wave frequency tables.
#![allow(
    missing_docs,
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_comparisons,
    clippy::all
)]

use crate::core::{TransformerError, TransformerResult};
use brain_core::Tensor;

/// Configuration for absolute positional embeddings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PositionConfig {
    /// Maximum context position count.
    pub max_position_embeddings: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
}

impl Default for PositionConfig {
    fn default() -> Self {
        Self {
            max_position_embeddings: 512,
            hidden_dim: 768,
        }
    }
}

/// Fixed sinusoidal positional encoding generator.
pub struct SinusoidalPositionalEmbedding;

impl SinusoidalPositionalEmbedding {
    /// Generates standard sinusoidal positional encoding tensor of shape `[max_len, dim]`.
    pub fn generate(max_len: usize, dim: usize) -> Tensor {
        let mut data = vec![0.0f64; max_len * dim];

        for pos in 0..max_len {
            for i in 0..(dim / 2) {
                let freq = (pos as f64) / (10000.0f64).powf((2 * i) as f64 / dim as f64);
                let sin_idx = pos * dim + (2 * i);
                let cos_idx = pos * dim + (2 * i + 1);
                data[sin_idx] = freq.sin();
                if cos_idx < (pos + 1) * dim {
                    data[cos_idx] = freq.cos();
                }
            }
        }

        Tensor::from_vec(data, vec![max_len, dim])
    }
}

/// Trainable learned positional embedding table.
#[derive(Debug, Clone)]
pub struct LearnedPositionalEmbedding {
    /// Embedding weights `[max_position_embeddings, hidden_dim]`.
    pub weights: Tensor,
    /// Configuration options.
    pub config: PositionConfig,
}

impl LearnedPositionalEmbedding {
    /// Creates a new `LearnedPositionalEmbedding` table.
    pub fn new(config: PositionConfig, seed: u64) -> Self {
        let numel = config.max_position_embeddings * config.hidden_dim;
        let std_dev = 0.02;
        let mut data = Vec::with_capacity(numel);

        for i in 0..config.max_position_embeddings {
            for j in 0..config.hidden_dim {
                let s = (seed.wrapping_add((i * 10007 + j * 37 + 1) as u64)) as f64;
                let val = (s.sin() * 43758.5453).fract() * 2.0 * std_dev - std_dev;
                data.push(val);
            }
        }

        let weights = Tensor::from_vec(
            data,
            vec![config.max_position_embeddings, config.hidden_dim],
        );
        Self { weights, config }
    }

    /// Looks up positional embeddings for sequence slice of length `seq_len` with `pos_offset`.
    pub fn forward(&self, seq_len: usize, pos_offset: usize) -> TransformerResult<Tensor> {
        if pos_offset + seq_len > self.config.max_position_embeddings {
            return Err(TransformerError::ContextLengthExceeded {
                max_len: self.config.max_position_embeddings,
                requested: pos_offset + seq_len,
            });
        }

        let dim = self.config.hidden_dim;
        let start = pos_offset * dim;
        let end = (pos_offset + seq_len) * dim;
        let slice = self.weights.data()[start..end].to_vec();

        Ok(Tensor::from_vec(slice, vec![seq_len, dim]))
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant,
        clippy::needless_range_loop,
        clippy::manual_div_ceil,
        clippy::manual_is_multiple_of,
        clippy::too_many_arguments,
        clippy::doc_markdown,
        clippy::excessive_precision,
        clippy::float_cmp,
        clippy::len_zero,
        clippy::all
    )]
    use super::*;
    use crate::attention::flash_lite::*;
    use crate::attention::multi_head::*;
    use crate::attention::multi_query::*;
    use crate::attention::relative::*;
    use crate::attention::scaled::*;
    use crate::attention::xformers_lite::*;
    use crate::attention::*;
    use crate::builder::*;
    use crate::config::*;
    use crate::core::*;
    use crate::decoder::cross::*;
    use crate::decoder::layer::*;
    use crate::decoder::*;
    use crate::embedding_layers::*;
    use crate::encoder::block::*;
    use crate::encoder::layer::*;
    use crate::encoder::*;
    use crate::ffn::*;
    use crate::generate::*;
    use crate::head::*;
    use crate::kv_cache::*;
    use crate::models::bert_lite::*;
    use crate::models::gpt_lite::*;
    use crate::models::llama_lite::*;
    use crate::models::t5_lite::*;
    use crate::models::*;
    use crate::ops::*;
    use crate::position::alibi::*;
    use crate::position::learned::*;
    use crate::position::rope::*;
    use crate::position::*;
    use crate::utils::*;
    use brain_core::Tensor;

    #[test]
    fn test_learned_sinusoidal_position_1() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig {
            max_position_embeddings: 64,
            hidden_dim: 16,
        };
        let learned = LearnedPositionalEmbedding::new(cfg, 1 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }
}
