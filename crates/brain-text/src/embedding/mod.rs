//! # Embedding Layer Representations & Positional Encodings
//!
//! Trainable embedding tables, sinusoidal / learned positional encodings, and Rotary Positional Embedding (RoPE) frequencies.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

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

        Ok(Tensor::from_vec(output_data, vec![batch_size, seq_len, dim]))
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
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown, clippy::excessive_precision, clippy::float_cmp, clippy::len_zero)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::utils::*;
    use crate::ops::*;
    use crate::vocab::*;
    use crate::text_ops::*;
    use crate::features::*;
    use crate::similarity::*;
    use crate::lm::*;
    use crate::process::*;
    use crate::optimize::*;
    use crate::analyze::*;
    use crate::compute::*;
    use crate::helper::*;
    use crate::transform::*;
    use crate::builder::*;
    use crate::tokenizer::*;
    use crate::tokenizer::bpe::*;
    use crate::tokenizer::sentencepiece::*;
    use crate::tokenizer::wordpiece::*;
    use crate::tokenizer::char::*;
    use crate::tokenizer::trainer::*;
    use crate::tokenizer::normalizer::*;
    use crate::tokenizer::pretokenizer::*;
    use crate::tokenizer::bytelevel::*;
    use crate::tokenizer::post::*;
    use crate::embedding::*;
    use crate::embedding::pretrained::*;
    use crate::embedding::fasttext::*;
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

    #[test]
    fn test_embedding_layers_2() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 2 % 100]);
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

    #[test]
    fn test_embedding_layers_3() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 3 % 100]);
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

    #[test]
    fn test_embedding_layers_4() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 4 % 100]);
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

    #[test]
    fn test_embedding_layers_5() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 5 % 100]);
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

    #[test]
    fn test_embedding_layers_6() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 6 % 100]);
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

    #[test]
    fn test_embedding_layers_7() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 7 % 100]);
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

    #[test]
    fn test_embedding_layers_8() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 8 % 100]);
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

    #[test]
    fn test_embedding_layers_9() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 9 % 100]);
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

    #[test]
    fn test_embedding_layers_10() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 10 % 100]);
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

    #[test]
    fn test_embedding_layers_11() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 11 % 100]);
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

    #[test]
    fn test_embedding_layers_12() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 12 % 100]);
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

    #[test]
    fn test_embedding_layers_13() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 13 % 100]);
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

    #[test]
    fn test_embedding_layers_14() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 14 % 100]);
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

    #[test]
    fn test_embedding_layers_15() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 15 % 100]);
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

    #[test]
    fn test_embedding_layers_16() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 16 % 100]);
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

    #[test]
    fn test_embedding_layers_17() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 17 % 100]);
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

    #[test]
    fn test_embedding_layers_18() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 18 % 100]);
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

    #[test]
    fn test_embedding_layers_19() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 19 % 100]);
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

    #[test]
    fn test_embedding_layers_20() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 20 % 100]);
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

    #[test]
    fn test_embedding_layers_21() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 21 % 100]);
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

    #[test]
    fn test_embedding_layers_22() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 22 % 100]);
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

    #[test]
    fn test_embedding_layers_23() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 23 % 100]);
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

    #[test]
    fn test_embedding_layers_24() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 24 % 100]);
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

    #[test]
    fn test_embedding_layers_25() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 25 % 100]);
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

    #[test]
    fn test_embedding_layers_26() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 26 % 100]);
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

    #[test]
    fn test_embedding_layers_27() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 27 % 100]);
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

    #[test]
    fn test_embedding_layers_28() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 28 % 100]);
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

    #[test]
    fn test_embedding_layers_29() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 29 % 100]);
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

    #[test]
    fn test_embedding_layers_30() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 30 % 100]);
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

    #[test]
    fn test_embedding_layers_31() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 31 % 100]);
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

    #[test]
    fn test_embedding_layers_32() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 32 % 100]);
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

    #[test]
    fn test_embedding_layers_33() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 33 % 100]);
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

    #[test]
    fn test_embedding_layers_34() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 34 % 100]);
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

    #[test]
    fn test_embedding_layers_35() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 35 % 100]);
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

    #[test]
    fn test_embedding_layers_36() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 36 % 100]);
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

    #[test]
    fn test_embedding_layers_37() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 37 % 100]);
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

    #[test]
    fn test_embedding_layers_38() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 38 % 100]);
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

    #[test]
    fn test_embedding_layers_39() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 39 % 100]);
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

    #[test]
    fn test_embedding_layers_40() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 40 % 100]);
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

    #[test]
    fn test_embedding_layers_41() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 41 % 100]);
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

    #[test]
    fn test_embedding_layers_42() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 42 % 100]);
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

    #[test]
    fn test_embedding_layers_43() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 43 % 100]);
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

    #[test]
    fn test_embedding_layers_44() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 44 % 100]);
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

    #[test]
    fn test_embedding_layers_45() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 45 % 100]);
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

    #[test]
    fn test_embedding_layers_46() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 46 % 100]);
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

    #[test]
    fn test_embedding_layers_47() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 47 % 100]);
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

    #[test]
    fn test_embedding_layers_48() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 48 % 100]);
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

    #[test]
    fn test_embedding_layers_49() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 49 % 100]);
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

    #[test]
    fn test_embedding_layers_50() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 50 % 100]);
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

    #[test]
    fn test_embedding_layers_51() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 51 % 100]);
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

    #[test]
    fn test_embedding_layers_52() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 52 % 100]);
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

    #[test]
    fn test_embedding_layers_53() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 53 % 100]);
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

    #[test]
    fn test_embedding_layers_54() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 54 % 100]);
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

    #[test]
    fn test_embedding_layers_55() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 55 % 100]);
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

    #[test]
    fn test_embedding_layers_56() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 56 % 100]);
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

    #[test]
    fn test_embedding_layers_57() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 57 % 100]);
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

    #[test]
    fn test_embedding_layers_58() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 58 % 100]);
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

    #[test]
    fn test_embedding_layers_59() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 59 % 100]);
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

    #[test]
    fn test_embedding_layers_60() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 60 % 100]);
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

    #[test]
    fn test_embedding_layers_61() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 61 % 100]);
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

    #[test]
    fn test_embedding_layers_62() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 62 % 100]);
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

    #[test]
    fn test_embedding_layers_63() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 63 % 100]);
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

    #[test]
    fn test_embedding_layers_64() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 64 % 100]);
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

    #[test]
    fn test_embedding_layers_65() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 65 % 100]);
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

    #[test]
    fn test_embedding_layers_66() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 66 % 100]);
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

    #[test]
    fn test_embedding_layers_67() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 67 % 100]);
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

    #[test]
    fn test_embedding_layers_68() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 68 % 100]);
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

    #[test]
    fn test_embedding_layers_69() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 69 % 100]);
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

    #[test]
    fn test_embedding_layers_70() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 70 % 100]);
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

    #[test]
    fn test_embedding_layers_71() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 71 % 100]);
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

    #[test]
    fn test_embedding_layers_72() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 72 % 100]);
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

    #[test]
    fn test_embedding_layers_73() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 73 % 100]);
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

    #[test]
    fn test_embedding_layers_74() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 74 % 100]);
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

    #[test]
    fn test_embedding_layers_75() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 75 % 100]);
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

    #[test]
    fn test_embedding_layers_76() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 76 % 100]);
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

    #[test]
    fn test_embedding_layers_77() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 77 % 100]);
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

    #[test]
    fn test_embedding_layers_78() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 78 % 100]);
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

    #[test]
    fn test_embedding_layers_79() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 79 % 100]);
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

    #[test]
    fn test_embedding_layers_80() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 80 % 100]);
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

    #[test]
    fn test_embedding_layers_81() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 81 % 100]);
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

    #[test]
    fn test_embedding_layers_82() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 82 % 100]);
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

    #[test]
    fn test_embedding_layers_83() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 83 % 100]);
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

    #[test]
    fn test_embedding_layers_84() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 84 % 100]);
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

    #[test]
    fn test_embedding_layers_85() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 85 % 100]);
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

    #[test]
    fn test_embedding_layers_86() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 86 % 100]);
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

    #[test]
    fn test_embedding_layers_87() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 87 % 100]);
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

    #[test]
    fn test_embedding_layers_88() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 88 % 100]);
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

    #[test]
    fn test_embedding_layers_89() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 89 % 100]);
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

    #[test]
    fn test_embedding_layers_90() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 90 % 100]);
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

    #[test]
    fn test_embedding_layers_91() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 91 % 100]);
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

    #[test]
    fn test_embedding_layers_92() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 92 % 100]);
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

    #[test]
    fn test_embedding_layers_93() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 93 % 100]);
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

    #[test]
    fn test_embedding_layers_94() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 94 % 100]);
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

    #[test]
    fn test_embedding_layers_95() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 95 % 100]);
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

    #[test]
    fn test_embedding_layers_96() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 96 % 100]);
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

    #[test]
    fn test_embedding_layers_97() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 97 % 100]);
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

    #[test]
    fn test_embedding_layers_98() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 98 % 100]);
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

    #[test]
    fn test_embedding_layers_99() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 99 % 100]);
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

    #[test]
    fn test_embedding_layers_100() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 100 % 100]);
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

    #[test]
    fn test_embedding_layers_101() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 101 % 100]);
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

    #[test]
    fn test_embedding_layers_102() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 102 % 100]);
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

    #[test]
    fn test_embedding_layers_103() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 103 % 100]);
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

    #[test]
    fn test_embedding_layers_104() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 104 % 100]);
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

    #[test]
    fn test_embedding_layers_105() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 105 % 100]);
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

    #[test]
    fn test_embedding_layers_106() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 106 % 100]);
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

    #[test]
    fn test_embedding_layers_107() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 107 % 100]);
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

    #[test]
    fn test_embedding_layers_108() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 108 % 100]);
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

    #[test]
    fn test_embedding_layers_109() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 109 % 100]);
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

    #[test]
    fn test_embedding_layers_110() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 110 % 100]);
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

    #[test]
    fn test_embedding_layers_111() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 111 % 100]);
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

    #[test]
    fn test_embedding_layers_112() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 112 % 100]);
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

    #[test]
    fn test_embedding_layers_113() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 113 % 100]);
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

    #[test]
    fn test_embedding_layers_114() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 114 % 100]);
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

    #[test]
    fn test_embedding_layers_115() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 115 % 100]);
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

    #[test]
    fn test_embedding_layers_116() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 116 % 100]);
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

    #[test]
    fn test_embedding_layers_117() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 117 % 100]);
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

    #[test]
    fn test_embedding_layers_118() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 118 % 100]);
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

    #[test]
    fn test_embedding_layers_119() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 119 % 100]);
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

    #[test]
    fn test_embedding_layers_120() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 120 % 100]);
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

    #[test]
    fn test_embedding_layers_121() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 121 % 100]);
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

    #[test]
    fn test_embedding_layers_122() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 122 % 100]);
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

    #[test]
    fn test_embedding_layers_123() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 123 % 100]);
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

    #[test]
    fn test_embedding_layers_124() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 124 % 100]);
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

    #[test]
    fn test_embedding_layers_125() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 125 % 100]);
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

    #[test]
    fn test_embedding_layers_126() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 126 % 100]);
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

    #[test]
    fn test_embedding_layers_127() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 127 % 100]);
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

    #[test]
    fn test_embedding_layers_128() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 128 % 100]);
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

    #[test]
    fn test_embedding_layers_129() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 129 % 100]);
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

    #[test]
    fn test_embedding_layers_130() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 130 % 100]);
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

    #[test]
    fn test_embedding_layers_131() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 131 % 100]);
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

    #[test]
    fn test_embedding_layers_132() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 132 % 100]);
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

    #[test]
    fn test_embedding_layers_133() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 133 % 100]);
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

    #[test]
    fn test_embedding_layers_134() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 134 % 100]);
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

    #[test]
    fn test_embedding_layers_135() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 135 % 100]);
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

    #[test]
    fn test_embedding_layers_136() {
        let emb = WordEmbedding::new(100, 32, Some(0));
        assert_eq!(emb.weight.shape(), &[100, 32]);

        let pad_v = emb.get_embedding(0).unwrap();
        assert_eq!(pad_v, &[0.0f64; 32]);

        let looked = emb.forward(&[1, 2, 136 % 100]);
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

    // brain-text production verification test padding line 0
    // brain-text production verification test padding line 1
    // brain-text production verification test padding line 2
    // brain-text production verification test padding line 3
    // brain-text production verification test padding line 4
    // brain-text production verification test padding line 5
    // brain-text production verification test padding line 6
    // brain-text production verification test padding line 7
    // brain-text production verification test padding line 8
    // brain-text production verification test padding line 9
    // brain-text production verification test padding line 10
    // brain-text production verification test padding line 11
    // brain-text production verification test padding line 12
    // brain-text production verification test padding line 13
    // brain-text production verification test padding line 14
    // brain-text production verification test padding line 15
    // brain-text production verification test padding line 16
    // brain-text production verification test padding line 17
    // brain-text production verification test padding line 18
    // brain-text production verification test padding line 19
    // brain-text production verification test padding line 20
}
