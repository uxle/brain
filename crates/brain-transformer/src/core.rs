//! # Transformer Core Types, Error Handling, and Data Structures
//!
//! Foundational representations for attention masks, sequence batches, cache states, linear layer parameters, and transformer errors.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use brain_core::Tensor;
use std::fmt;

/// Primary error enum for transformer operations, configuration validation, attention mechanisms, and generation.
#[derive(Debug, Clone, PartialEq)]
pub enum TransformerError {
    /// Dimension mismatch during linear projection, attention computation, or concatenation.
    DimensionMismatch { expected: usize, found: usize },
    /// Head dimension not divisible by total hidden dimension.
    InvalidHeadDim { hidden_dim: usize, num_heads: usize },
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Sequence length exceeds maximum supported positional context length.
    ContextLengthExceeded { max_len: usize, requested: usize },
    /// KV-cache index or capacity error during autoregressive generation.
    CacheError(String),
    /// Generation stopped or failed due to invalid token or sampling parameters.
    GenerationError(String),
    /// Empty sequence or tensor provided where non-empty required.
    EmptyInput,
    /// Numerical instability or non-finite float encountered.
    NumericalError(String),
}

impl fmt::Display for TransformerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransformerError::DimensionMismatch { expected, found } => {
                write!(f, "Dimension mismatch: expected {}, found {}", expected, found)
            }
            TransformerError::InvalidHeadDim { hidden_dim, num_heads } => {
                write!(
                    f,
                    "Hidden dimension {} not divisible by {} heads",
                    hidden_dim, num_heads
                )
            }
            TransformerError::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            TransformerError::ContextLengthExceeded { max_len, requested } => {
                write!(
                    f,
                    "Requested context length {} exceeds maximum capacity {}",
                    requested, max_len
                )
            }
            TransformerError::CacheError(msg) => write!(f, "KV-Cache error: {}", msg),
            TransformerError::GenerationError(msg) => write!(f, "Generation error: {}", msg),
            TransformerError::EmptyInput => write!(f, "Input tensor or sequence is empty"),
            TransformerError::NumericalError(msg) => write!(f, "Numerical error: {}", msg),
        }
    }
}

impl std::error::Error for TransformerError {}

/// Standard result type alias for transformer operations.
pub type TransformerResult<T> = Result<T, TransformerError>;

/// Attention mask container supporting boolean masks, additive bias masks, and causal lower-triangular masks.
#[derive(Debug, Clone, PartialEq)]
pub enum AttentionMask {
    /// No mask applied (full bidirectional attention).
    None,
    /// Standard causal lower-triangular mask (prevents attending to future tokens).
    Causal,
    /// 2D Boolean padding mask of shape `[batch_size, seq_len]` where `true` indicates valid token and `false` indicates padding.
    Padding(Tensor),
    /// 2D or 4D additive floating point bias tensor (added directly to raw attention logits before softmax).
    AdditiveBias(Tensor),
    /// Combined causal and padding mask.
    CausalWithPadding(Tensor),
}

impl AttentionMask {
    /// Returns true if no masking is required.
    pub fn is_none(&self) -> bool {
        matches!(self, AttentionMask::None)
    }

    /// Returns true if the mask includes causal lower-triangular constraints.
    pub fn is_causal(&self) -> bool {
        matches!(self, AttentionMask::Causal | AttentionMask::CausalWithPadding(_))
    }
}

/// Linear projection parameter pair: weights matrix `[in_features, out_features]` and optional bias vector `[out_features]`.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearParams {
    /// 2D Weight tensor of shape `[in_features, out_features]`.
    pub weight: Tensor,
    /// Optional 1D bias vector of shape `[out_features]`.
    pub bias: Option<Tensor>,
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
}

impl LinearParams {
    /// Creates a new `LinearParams` container with Xavier-style pseudo-random initialization.
    pub fn new(in_features: usize, out_features: usize, has_bias: bool, seed: u64) -> Self {
        let std_dev = (2.0 / (in_features + out_features) as f64).sqrt();
        let mut w_data = Vec::with_capacity(in_features * out_features);

        for i in 0..in_features {
            for j in 0..out_features {
                let s = (seed.wrapping_add((i * 10007 + j * 37 + 1) as u64)) as f64;
                let val = (s.sin() * 43758.5453).fract() * 2.0 * std_dev - std_dev;
                w_data.push(val);
            }
        }

        let weight = Tensor::from_vec(w_data, vec![in_features, out_features]);
        let bias = if has_bias {
            Some(Tensor::zeros(vec![out_features]))
        } else {
            None
        };

        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    /// Applies linear projection: $Y = XW + b$ for 2D input `[batch_size * seq_len, in_features]` or 3D `[batch, seq, in]`.
    pub fn forward(&self, input: &Tensor) -> TransformerResult<Tensor> {
        let shape = input.shape();
        if shape.is_empty() {
            return Err(TransformerError::EmptyInput);
        }

        let last_dim = *shape.last().unwrap();
        if last_dim != self.in_features {
            return Err(TransformerError::DimensionMismatch {
                expected: self.in_features,
                found: last_dim,
            });
        }

        let total_tokens: usize = shape[..shape.len() - 1].iter().product();
        let in_data = input.data();
        let w_data = self.weight.data();
        let mut out_data = vec![0.0f64; total_tokens * self.out_features];

        for t in 0..total_tokens {
            let in_offset = t * self.in_features;
            let out_offset = t * self.out_features;

            for j in 0..self.out_features {
                let mut sum = 0.0f64;
                for i in 0..self.in_features {
                    sum += in_data[in_offset + i] * w_data[i * self.out_features + j];
                }
                if let Some(ref b) = self.bias {
                    sum += b.data()[j];
                }
                out_data[out_offset + j] = sum;
            }
        }

        let mut out_shape = shape.to_vec();
        *out_shape.last_mut().unwrap() = self.out_features;
        Ok(Tensor::from_vec(out_data, out_shape))
    }
}

/// Batch sequence information helper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchSeq {
    /// Batch size dimension.
    pub batch_size: usize,
    /// Sequence length dimension.
    pub seq_len: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
}

impl BatchSeq {
    /// Creates a new `BatchSeq` description.
    pub fn new(batch_size: usize, seq_len: usize, hidden_dim: usize) -> Self {
        Self {
            batch_size,
            seq_len,
            hidden_dim,
        }
    }

    /// Total number of tokens across the batch: `batch_size * seq_len`.
    pub fn total_tokens(&self) -> usize {
        self.batch_size * self.seq_len
    }

    /// Total number of elements in representation tensor: `batch_size * seq_len * hidden_dim`.
    pub fn total_elements(&self) -> usize {
        self.batch_size * self.seq_len * self.hidden_dim
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
    fn test_core_primitives_1() {
        let lp = LinearParams::new(16, 32, true, 1 as u64);
        assert_eq!(lp.weight.shape(), &[16, 32]);
        assert_eq!(lp.bias.as_ref().unwrap().shape(), &[32]);

        let inp = Tensor::from_vec(vec![1.0; 2 * 4 * 16], vec![2, 4, 16]);
        let out = lp.forward(&inp).unwrap();
        assert_eq!(out.shape(), &[2, 4, 32]);

        let mask_none = AttentionMask::None;
        assert!(mask_none.is_none());
        let mask_causal = AttentionMask::Causal;
        assert!(mask_causal.is_causal());

        let bs = BatchSeq::new(2, 8, 64);
        assert_eq!(bs.total_tokens(), 16);
        assert_eq!(bs.total_elements(), 1024);

        let err = TransformerError::InvalidHeadDim { hidden_dim: 64, num_heads: 7 };
        assert!(err.to_string().contains("not divisible"));
    }
}
