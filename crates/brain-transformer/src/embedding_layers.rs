//! # Transformer Token, Positional, and Segment Embedding Layers
//!
//! Embedding lookup tables, additive positional embeddings, segment token type embeddings, and dropout.
#![allow(
    missing_docs,
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_comparisons,
    clippy::all
)]

use crate::config::{NormType, PositionEncodingType};
use crate::core::{TransformerError, TransformerResult};
use crate::ops::{layer_norm, rms_norm};
use crate::position::learned::SinusoidalPositionalEmbedding;
use brain_core::Tensor;

/// Configuration for transformer embedding layer.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Embedding vector dimension $d_{\text{model}}$.
    pub hidden_dim: usize,
    /// Maximum positional context length.
    pub max_position_embeddings: usize,
    /// Number of token type / segment categories (e.g. 2 for BERT sentence pairs).
    pub type_vocab_size: Option<usize>,
    /// Embedding dropout probability.
    pub dropout: f32,
    /// Positional encoding strategy.
    pub pos_encoding: PositionEncodingType,
    /// Normalization type applied post-embedding sum.
    pub norm_type: Option<NormType>,
    /// Normalization epsilon.
    pub norm_eps: f64,
}

impl Default for EmbConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_dim: 768,
            max_position_embeddings: 2048,
            type_vocab_size: None,
            dropout: 0.0,
            pos_encoding: PositionEncodingType::Rope,
            norm_type: None,
            norm_eps: 1e-6,
        }
    }
}

/// Unified Transformer Embedding Layer: Token + Position + Segment.
#[derive(Debug, Clone)]
pub struct TransformerEmbedding {
    /// Token embedding lookup weights `[vocab_size, hidden_dim]`.
    pub word_embeddings: Tensor,
    /// Optional learned absolute positional embeddings `[max_position_embeddings, hidden_dim]`.
    pub position_embeddings: Option<Tensor>,
    /// Optional token type / segment embeddings `[type_vocab_size, hidden_dim]`.
    pub token_type_embeddings: Option<Tensor>,
    /// Optional LayerNorm / RMSNorm gamma weights `[hidden_dim]`.
    pub norm_gamma: Option<Tensor>,
    /// Optional LayerNorm beta bias `[hidden_dim]`.
    pub norm_beta: Option<Tensor>,
    /// Configuration options.
    pub config: EmbConfig,
}

impl TransformerEmbedding {
    /// Creates a new `TransformerEmbedding` layer with initialized weights.
    pub fn new(config: EmbConfig, seed: u64) -> Self {
        let std_dev = (1.0 / config.hidden_dim as f64).sqrt();
        let num_w = config.vocab_size * config.hidden_dim;
        let mut w_data = Vec::with_capacity(num_w);

        for i in 0..config.vocab_size {
            for j in 0..config.hidden_dim {
                let s = (seed.wrapping_add((i * 10007 + j * 37 + 1) as u64)) as f64;
                let val = (s.sin() * 43758.5453).fract() * 2.0 * std_dev - std_dev;
                w_data.push(val);
            }
        }
        let word_embeddings = Tensor::from_vec(w_data, vec![config.vocab_size, config.hidden_dim]);

        let position_embeddings = match config.pos_encoding {
            PositionEncodingType::Learned => {
                let num_p = config.max_position_embeddings * config.hidden_dim;
                let mut p_data = Vec::with_capacity(num_p);
                for i in 0..config.max_position_embeddings {
                    for j in 0..config.hidden_dim {
                        let s = (seed.wrapping_add((i * 7919 + j * 13 + 500) as u64)) as f64;
                        let val = (s.sin() * 43758.5453).fract() * 2.0 * std_dev - std_dev;
                        p_data.push(val);
                    }
                }
                Some(Tensor::from_vec(
                    p_data,
                    vec![config.max_position_embeddings, config.hidden_dim],
                ))
            }
            PositionEncodingType::Sinusoidal => {
                let sin_pe = SinusoidalPositionalEmbedding::generate(
                    config.max_position_embeddings,
                    config.hidden_dim,
                );
                Some(sin_pe)
            }
            _ => None,
        };

        let token_type_embeddings = if let Some(type_size) = config.type_vocab_size {
            let mut tt_data = vec![0.0f64; type_size * config.hidden_dim];
            for i in 0..type_size {
                for j in 0..config.hidden_dim {
                    let s = (seed.wrapping_add((i * 997 + j * 17 + 100) as u64)) as f64;
                    tt_data[i * config.hidden_dim + j] = (s.sin() * 43758.5453).fract() * 0.02;
                }
            }
            Some(Tensor::from_vec(
                tt_data,
                vec![type_size, config.hidden_dim],
            ))
        } else {
            None
        };

        let norm_gamma = if config.norm_type.is_some() {
            Some(Tensor::ones(vec![config.hidden_dim]))
        } else {
            None
        };
        let norm_beta = if config.norm_type == Some(NormType::LayerNorm) {
            Some(Tensor::zeros(vec![config.hidden_dim]))
        } else {
            None
        };

        Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            norm_gamma,
            norm_beta,
            config,
        }
    }

    /// Performs embedding forward lookup for input IDs of shape `[batch_size, seq_len]`.
    pub fn forward(
        &self,
        input_ids: &[usize],
        batch_size: usize,
        seq_len: usize,
        token_type_ids: Option<&[usize]>,
        past_pos_offset: usize,
    ) -> TransformerResult<Tensor> {
        if input_ids.len() != batch_size * seq_len {
            return Err(TransformerError::DimensionMismatch {
                expected: batch_size * seq_len,
                found: input_ids.len(),
            });
        }

        let dim = self.config.hidden_dim;
        let mut out_data = Vec::with_capacity(batch_size * seq_len * dim);
        let w_slice = self.word_embeddings.data();

        let pos_slice = self.position_embeddings.as_ref().map(|p| p.data());
        let type_slice = self.token_type_embeddings.as_ref().map(|t| t.data());

        for b in 0..batch_size {
            for s in 0..seq_len {
                let token_idx = b * seq_len + s;
                let id = input_ids[token_idx];

                if id >= self.config.vocab_size {
                    return Err(TransformerError::DimensionMismatch {
                        expected: self.config.vocab_size,
                        found: id,
                    });
                }

                let w_offset = id * dim;
                let mut token_vec = w_slice[w_offset..w_offset + dim].to_vec();

                // Add positional embeddings if enabled
                if let Some(pos_data) = pos_slice {
                    let pos = past_pos_offset + s;
                    if pos < self.config.max_position_embeddings {
                        let p_offset = pos * dim;
                        for i in 0..dim {
                            token_vec[i] += pos_data[p_offset + i];
                        }
                    }
                }

                // Add token type embeddings if enabled
                if let (Some(type_ids), Some(t_data)) = (token_type_ids, type_slice) {
                    if token_idx < type_ids.len() {
                        let t_id = type_ids[token_idx];
                        let t_offset = t_id * dim;
                        if t_offset + dim <= t_data.len() {
                            for i in 0..dim {
                                token_vec[i] += t_data[t_offset + i];
                            }
                        }
                    }
                }

                out_data.extend_from_slice(&token_vec);
            }
        }

        let tensor = Tensor::from_vec(out_data, vec![batch_size, seq_len, dim]);

        // Apply post-embedding normalization if configured
        match self.config.norm_type {
            Some(NormType::LayerNorm) => layer_norm(
                &tensor,
                self.norm_gamma.as_ref(),
                self.norm_beta.as_ref(),
                self.config.norm_eps,
            ),
            Some(NormType::RmsNorm) => {
                rms_norm(&tensor, self.norm_gamma.as_ref(), self.config.norm_eps)
            }
            None => Ok(tensor),
        }
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
    fn test_embedding_layers_1() {
        let cfg = EmbConfig {
            vocab_size: 100,
            hidden_dim: 32,
            max_position_embeddings: 64,
            type_vocab_size: Some(2),
            pos_encoding: PositionEncodingType::Learned,
            norm_type: Some(NormType::LayerNorm),
            ..Default::default()
        };
        let emb = TransformerEmbedding::new(cfg, 1 as u64);
        assert_eq!(emb.word_embeddings.shape(), &[100, 32]);
        assert_eq!(emb.position_embeddings.as_ref().unwrap().shape(), &[64, 32]);

        let ids = vec![1, 2, 3, 4];
        let type_ids = vec![0, 0, 1, 1];
        let out = emb.forward(&ids, 2, 2, Some(&type_ids), 0).unwrap();
        assert_eq!(out.shape(), &[2, 2, 32]);
    }
}
