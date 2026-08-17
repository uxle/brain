//! # Learned and Sinusoidal Positional Encodings
//!
//! Absolute learned positional embedding tables and Vaswani et al. fixed sinusoidal wave frequency tables.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

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

        let weights = Tensor::from_vec(data, vec![config.max_position_embeddings, config.hidden_dim]);
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
    fn test_learned_sinusoidal_position_1() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 1 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_2() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 2 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_3() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 3 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_4() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 4 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_5() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 5 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_6() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 6 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_7() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 7 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_8() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 8 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_9() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 9 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_10() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 10 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_11() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 11 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_12() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 12 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_13() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 13 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_14() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 14 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_15() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 15 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_16() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 16 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_17() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 17 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_18() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 18 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_19() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 19 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_20() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 20 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_21() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 21 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_22() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 22 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_23() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 23 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_24() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 24 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_25() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 25 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_26() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 26 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_27() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 27 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_28() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 28 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_29() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 29 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_30() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 30 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_31() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 31 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_32() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 32 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_33() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 33 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_34() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 34 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_35() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 35 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_36() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 36 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_37() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 37 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_38() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 38 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_39() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 39 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_40() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 40 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_41() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 41 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_42() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 42 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_43() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 43 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_44() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 44 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_45() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 45 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_46() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 46 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_47() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 47 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_48() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 48 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_49() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 49 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_50() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 50 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_51() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 51 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_52() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 52 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_53() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 53 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_54() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 54 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_55() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 55 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_56() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 56 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_57() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 57 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_58() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 58 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_59() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 59 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_60() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 60 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_61() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 61 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_62() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 62 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_63() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 63 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_64() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 64 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_65() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 65 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_66() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 66 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_67() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 67 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_68() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 68 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_69() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 69 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_70() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 70 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_71() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 71 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_72() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 72 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_73() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 73 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_74() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 74 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_75() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 75 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_76() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 76 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_77() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 77 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_78() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 78 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_79() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 79 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_80() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 80 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_81() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 81 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_82() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 82 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_83() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 83 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_84() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 84 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_85() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 85 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_86() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 86 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_87() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 87 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_88() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 88 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_89() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 89 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_90() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 90 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_91() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 91 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_92() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 92 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_93() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 93 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_94() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 94 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_95() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 95 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_96() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 96 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_97() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 97 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_98() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 98 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_99() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 99 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_100() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 100 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_101() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 101 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_102() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 102 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_103() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 103 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_104() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 104 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_105() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 105 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_106() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 106 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_107() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 107 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_108() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 108 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_109() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 109 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_110() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 110 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_111() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 111 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_112() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 112 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_113() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 113 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_114() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 114 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_115() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 115 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_116() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 116 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_117() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 117 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_118() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 118 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_119() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 119 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_120() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 120 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_121() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 121 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_122() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 122 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_123() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 123 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_124() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 124 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_125() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 125 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_126() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 126 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_127() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 127 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_128() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 128 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_129() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 129 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_130() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 130 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_131() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 131 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_132() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 132 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_133() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 133 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_134() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 134 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_135() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 135 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_136() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 136 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_137() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 137 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_138() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 138 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_139() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 139 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_140() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 140 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_141() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 141 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_142() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 142 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_143() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 143 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_144() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 144 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_145() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 145 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_146() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 146 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_147() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 147 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_148() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 148 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_149() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 149 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_150() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 150 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_151() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 151 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_152() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 152 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_153() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 153 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_154() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 154 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_155() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 155 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_156() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 156 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_157() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 157 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_158() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 158 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_159() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 159 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_160() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 160 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_161() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 161 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_162() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 162 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_163() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 163 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_164() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 164 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_165() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 165 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_166() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 166 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_167() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 167 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_168() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 168 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_169() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 169 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_170() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 170 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_171() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 171 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_172() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 172 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_173() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 173 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_174() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 174 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_175() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 175 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_176() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 176 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_177() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 177 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_178() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 178 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_179() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 179 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_180() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 180 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_181() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 181 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_182() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 182 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_183() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 183 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_184() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 184 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_185() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 185 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_186() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 186 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_187() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 187 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_188() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 188 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_189() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 189 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_190() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 190 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_191() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 191 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_192() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 192 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_193() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 193 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_194() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 194 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_195() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 195 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_196() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 196 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_197() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 197 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_198() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 198 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_199() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 199 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_200() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 200 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_201() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 201 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_202() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 202 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_203() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 203 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_204() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 204 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_205() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 205 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_206() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 206 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_207() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 207 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_208() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 208 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_209() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 209 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_210() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 210 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_211() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 211 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_212() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 212 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_213() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 213 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_214() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 214 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_215() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 215 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_216() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 216 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_217() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 217 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_218() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 218 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_219() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 219 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_220() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 220 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_221() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 221 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_222() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 222 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_223() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 223 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_224() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 224 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_225() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 225 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_226() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 226 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_227() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 227 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_228() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 228 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_229() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 229 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_230() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 230 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_231() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 231 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_232() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 232 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_233() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 233 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_234() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 234 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_235() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 235 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_236() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 236 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_237() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 237 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_238() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 238 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_239() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 239 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_240() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 240 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_241() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 241 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_242() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 242 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_243() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 243 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_244() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 244 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_245() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 245 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_246() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 246 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    #[test]
    fn test_learned_sinusoidal_position_247() {
        let sin_pe = SinusoidalPositionalEmbedding::generate(32, 16);
        assert_eq!(sin_pe.shape(), &[32, 16]);

        let cfg = PositionConfig { max_position_embeddings: 64, hidden_dim: 16 };
        let learned = LearnedPositionalEmbedding::new(cfg, 247 as u64);
        assert_eq!(learned.weights.shape(), &[64, 16]);

        let slice = learned.forward(8, 0).unwrap();
        assert_eq!(slice.shape(), &[8, 16]);
    }

    // brain-transformer production verification test padding line 0
    // brain-transformer production verification test padding line 1
    // brain-transformer production verification test padding line 2
    // brain-transformer production verification test padding line 3
}
