//! # Embedding Layer Representations & Positional Encodings
//!
//! Trainable embedding tables, sinusoidal / learned positional encodings, and Rotary Positional Embedding (RoPE) frequencies.
#![allow(
    missing_docs,
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_comparisons,
    clippy::all
)]

pub mod fasttext;
pub mod pretrained;

use crate::config::EmbeddingConfig;
use crate::core::{TextError, TextResult, TokenId};
use brain_core::Tensor;

/// Trainable word lookup embedding layer.
#[derive(Debug, Clone)]
pub struct WordEmbedding {
    /// 2D Weight tensor of shape `[vocab_size, embedding_dim]`.
    pub weight: Tensor,
    /// Configuration options.
    pub config: EmbeddingConfig,
}

impl WordEmbedding {
    /// Creates a new `WordEmbedding` with randomly initialized normal weights (Xavier-style).
    pub fn new(vocab_size: usize, embedding_dim: usize, padding_idx: Option<usize>) -> Self {
        let std_dev = (1.0 / embedding_dim as f64).sqrt();
        let mut data = Vec::with_capacity(vocab_size * embedding_dim);

        for id in 0..vocab_size {
            if Some(id) == padding_idx {
                data.extend(std::iter::repeat(0.0f64).take(embedding_dim));
            } else {
                for i in 0..embedding_dim {
                    // Deterministic pseudo-random initialization
                    let seed = (id * 10007 + i * 37 + 1) as f64;
                    let val = (seed.sin() * 43758.5453).fract() * 2.0 * std_dev - std_dev;
                    data.push(val);
                }
            }
        }

        let weight = Tensor::from_vec(data, vec![vocab_size, embedding_dim]);
        let mut config = EmbeddingConfig::default();
        config.vocab_size = vocab_size;
        config.embedding_dim = embedding_dim;
        config.padding_idx = padding_idx;

        Self { weight, config }
    }

    /// Looks up embedding vectors for a 1D slice of token IDs, producing a 2D Tensor `[seq_len, embedding_dim]`.
    pub fn forward(&self, input_ids: &[TokenId]) -> Tensor {
        let seq_len = input_ids.len();
        let dim = self.config.embedding_dim;
        let mut output_data = Vec::with_capacity(seq_len * dim);
        let weight_slice = self.weight.data();

        for &id in input_ids {
            if id < self.config.vocab_size {
                let start = id * dim;
                let end = start + dim;
                output_data.extend_from_slice(&weight_slice[start..end]);
            } else {
                output_data.extend(std::iter::repeat(0.0f64).take(dim));
            }
        }

        Tensor::from_vec(output_data, vec![seq_len, dim])
    }

    /// Looks up embedding vectors for a 2D Tensor of input IDs `[batch_size, seq_len]`, returning `[batch_size, seq_len, dim]`.
    pub fn forward_tensor(&self, input_ids: &Tensor) -> TextResult<Tensor> {
        let dims = input_ids.shape();
        if dims.len() != 2 {
            return Err(TextError::DimensionMismatch {
                expected: 2,
                found: dims.len(),
            });
        }
        let batch_size = dims[0];
        let seq_len = dims[1];
        let dim = self.config.embedding_dim;

        let ids_slice = input_ids.data();
        let mut output_data = Vec::with_capacity(batch_size * seq_len * dim);
        let weight_slice = self.weight.data();

        for &raw_id in ids_slice {
            let id = raw_id.max(0.0) as usize;
            if id < self.config.vocab_size {
                let start = id * dim;
                let end = start + dim;
                output_data.extend_from_slice(&weight_slice[start..end]);
            } else {
                output_data.extend(std::iter::repeat(0.0f64).take(dim));
            }
        }

        Ok(Tensor::from_vec(
            output_data,
            vec![batch_size, seq_len, dim],
        ))
    }

    /// Returns a slice view of the embedding vector for a given token ID.
    pub fn get_embedding(&self, id: TokenId) -> Option<&[f64]> {
        if id < self.config.vocab_size {
            let start = id * self.config.embedding_dim;
            let end = start + self.config.embedding_dim;
            Some(&self.weight.data()[start..end])
        } else {
            None
        }
    }
}

/// Factory for positional embedding representations.
pub struct PositionalEmbedding;

impl PositionalEmbedding {
    /// Generates standard sinusoidal positional encodings (Vaswani et al., 2017) of shape `[max_len, dim]`.
    pub fn sinusoidal(max_len: usize, dim: usize) -> Tensor {
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

    /// Creates a learned positional embedding table of shape `[max_len, dim]`.
    pub fn learned(max_len: usize, dim: usize) -> WordEmbedding {
        WordEmbedding::new(max_len, dim, None)
    }

    /// Computes Rotary Position Embedding (RoPE) cosine and sine frequency matrices for attention layers.
    pub fn rotary_frequencies(
        dim: usize,
        max_len: usize,
        theta: f32,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let half_dim = dim / 2;
        let mut cos_table = vec![vec![0.0f32; half_dim]; max_len];
        let mut sin_table = vec![vec![0.0f32; half_dim]; max_len];

        for pos in 0..max_len {
            for i in 0..half_dim {
                let freq = 1.0 / theta.powf((2 * i) as f32 / dim as f32);
                let angle = (pos as f32) * freq;
                cos_table[pos][i] = angle.cos();
                sin_table[pos][i] = angle.sin();
            }
        }

        (cos_table, sin_table)
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
        clippy::len_zero
    )]
    use super::*;
    use crate::analyze::*;
    use crate::builder::*;
    use crate::compute::*;
    use crate::config::*;
    use crate::core::*;
    use crate::embedding::fasttext::*;
    use crate::embedding::pretrained::*;
    use crate::embedding::*;
    use crate::features::*;
    use crate::helper::*;
    use crate::lm::*;
    use crate::ops::*;
    use crate::optimize::*;
    use crate::process::*;
    use crate::similarity::*;
    use crate::text_ops::*;
    use crate::tokenizer::bpe::*;
    use crate::tokenizer::bytelevel::*;
    use crate::tokenizer::char::*;
    use crate::tokenizer::normalizer::*;
    use crate::tokenizer::post::*;
    use crate::tokenizer::pretokenizer::*;
    use crate::tokenizer::sentencepiece::*;
    use crate::tokenizer::trainer::*;
    use crate::tokenizer::wordpiece::*;
    use crate::tokenizer::*;
    use crate::transform::*;
    use crate::utils::*;
    use crate::vocab::*;
    use crate::VERSION;
    use brain_core::Tensor;

    #[test]
    fn test_embedding_layers_1() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 1 % 100]);
        assert_eq!(looked.shape(), &[3, 32]);

        let input_tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let t_looked = emb.forward_tensor(&input_tensor).unwrap();
        assert_eq!(t_looked.shape(), &[2, 2, 32]);

        let pe = PositionalEmbedding::sinusoidal(10, 16);
        assert_eq!(pe.shape(), &[10, 16]);

        let (cos_t, sin_t) = PositionalEmbedding::rotary_frequencies(16, 10, 10000.0);
        assert_eq!(cos_t.len(), 10);
        assert_eq!(sin_t.len(), 10);
    }
}
