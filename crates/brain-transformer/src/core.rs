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

    #[test]
    fn test_core_primitives_2() {
        let lp = LinearParams::new(16, 32, true, 2 as u64);
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

    #[test]
    fn test_core_primitives_3() {
        let lp = LinearParams::new(16, 32, true, 3 as u64);
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

    #[test]
    fn test_core_primitives_4() {
        let lp = LinearParams::new(16, 32, true, 4 as u64);
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

    #[test]
    fn test_core_primitives_5() {
        let lp = LinearParams::new(16, 32, true, 5 as u64);
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

    #[test]
    fn test_core_primitives_6() {
        let lp = LinearParams::new(16, 32, true, 6 as u64);
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

    #[test]
    fn test_core_primitives_7() {
        let lp = LinearParams::new(16, 32, true, 7 as u64);
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

    #[test]
    fn test_core_primitives_8() {
        let lp = LinearParams::new(16, 32, true, 8 as u64);
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

    #[test]
    fn test_core_primitives_9() {
        let lp = LinearParams::new(16, 32, true, 9 as u64);
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

    #[test]
    fn test_core_primitives_10() {
        let lp = LinearParams::new(16, 32, true, 10 as u64);
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

    #[test]
    fn test_core_primitives_11() {
        let lp = LinearParams::new(16, 32, true, 11 as u64);
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

    #[test]
    fn test_core_primitives_12() {
        let lp = LinearParams::new(16, 32, true, 12 as u64);
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

    #[test]
    fn test_core_primitives_13() {
        let lp = LinearParams::new(16, 32, true, 13 as u64);
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

    #[test]
    fn test_core_primitives_14() {
        let lp = LinearParams::new(16, 32, true, 14 as u64);
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

    #[test]
    fn test_core_primitives_15() {
        let lp = LinearParams::new(16, 32, true, 15 as u64);
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

    #[test]
    fn test_core_primitives_16() {
        let lp = LinearParams::new(16, 32, true, 16 as u64);
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

    #[test]
    fn test_core_primitives_17() {
        let lp = LinearParams::new(16, 32, true, 17 as u64);
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

    #[test]
    fn test_core_primitives_18() {
        let lp = LinearParams::new(16, 32, true, 18 as u64);
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

    #[test]
    fn test_core_primitives_19() {
        let lp = LinearParams::new(16, 32, true, 19 as u64);
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

    #[test]
    fn test_core_primitives_20() {
        let lp = LinearParams::new(16, 32, true, 20 as u64);
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

    #[test]
    fn test_core_primitives_21() {
        let lp = LinearParams::new(16, 32, true, 21 as u64);
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

    #[test]
    fn test_core_primitives_22() {
        let lp = LinearParams::new(16, 32, true, 22 as u64);
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

    #[test]
    fn test_core_primitives_23() {
        let lp = LinearParams::new(16, 32, true, 23 as u64);
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

    #[test]
    fn test_core_primitives_24() {
        let lp = LinearParams::new(16, 32, true, 24 as u64);
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

    #[test]
    fn test_core_primitives_25() {
        let lp = LinearParams::new(16, 32, true, 25 as u64);
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

    #[test]
    fn test_core_primitives_26() {
        let lp = LinearParams::new(16, 32, true, 26 as u64);
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

    #[test]
    fn test_core_primitives_27() {
        let lp = LinearParams::new(16, 32, true, 27 as u64);
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

    #[test]
    fn test_core_primitives_28() {
        let lp = LinearParams::new(16, 32, true, 28 as u64);
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

    #[test]
    fn test_core_primitives_29() {
        let lp = LinearParams::new(16, 32, true, 29 as u64);
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

    #[test]
    fn test_core_primitives_30() {
        let lp = LinearParams::new(16, 32, true, 30 as u64);
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

    #[test]
    fn test_core_primitives_31() {
        let lp = LinearParams::new(16, 32, true, 31 as u64);
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

    #[test]
    fn test_core_primitives_32() {
        let lp = LinearParams::new(16, 32, true, 32 as u64);
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

    #[test]
    fn test_core_primitives_33() {
        let lp = LinearParams::new(16, 32, true, 33 as u64);
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

    #[test]
    fn test_core_primitives_34() {
        let lp = LinearParams::new(16, 32, true, 34 as u64);
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

    #[test]
    fn test_core_primitives_35() {
        let lp = LinearParams::new(16, 32, true, 35 as u64);
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

    #[test]
    fn test_core_primitives_36() {
        let lp = LinearParams::new(16, 32, true, 36 as u64);
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

    #[test]
    fn test_core_primitives_37() {
        let lp = LinearParams::new(16, 32, true, 37 as u64);
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

    #[test]
    fn test_core_primitives_38() {
        let lp = LinearParams::new(16, 32, true, 38 as u64);
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

    #[test]
    fn test_core_primitives_39() {
        let lp = LinearParams::new(16, 32, true, 39 as u64);
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

    #[test]
    fn test_core_primitives_40() {
        let lp = LinearParams::new(16, 32, true, 40 as u64);
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

    #[test]
    fn test_core_primitives_41() {
        let lp = LinearParams::new(16, 32, true, 41 as u64);
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

    #[test]
    fn test_core_primitives_42() {
        let lp = LinearParams::new(16, 32, true, 42 as u64);
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

    #[test]
    fn test_core_primitives_43() {
        let lp = LinearParams::new(16, 32, true, 43 as u64);
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

    #[test]
    fn test_core_primitives_44() {
        let lp = LinearParams::new(16, 32, true, 44 as u64);
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

    #[test]
    fn test_core_primitives_45() {
        let lp = LinearParams::new(16, 32, true, 45 as u64);
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

    #[test]
    fn test_core_primitives_46() {
        let lp = LinearParams::new(16, 32, true, 46 as u64);
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

    #[test]
    fn test_core_primitives_47() {
        let lp = LinearParams::new(16, 32, true, 47 as u64);
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

    #[test]
    fn test_core_primitives_48() {
        let lp = LinearParams::new(16, 32, true, 48 as u64);
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

    #[test]
    fn test_core_primitives_49() {
        let lp = LinearParams::new(16, 32, true, 49 as u64);
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

    #[test]
    fn test_core_primitives_50() {
        let lp = LinearParams::new(16, 32, true, 50 as u64);
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

    #[test]
    fn test_core_primitives_51() {
        let lp = LinearParams::new(16, 32, true, 51 as u64);
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

    #[test]
    fn test_core_primitives_52() {
        let lp = LinearParams::new(16, 32, true, 52 as u64);
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

    #[test]
    fn test_core_primitives_53() {
        let lp = LinearParams::new(16, 32, true, 53 as u64);
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

    #[test]
    fn test_core_primitives_54() {
        let lp = LinearParams::new(16, 32, true, 54 as u64);
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

    #[test]
    fn test_core_primitives_55() {
        let lp = LinearParams::new(16, 32, true, 55 as u64);
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

    #[test]
    fn test_core_primitives_56() {
        let lp = LinearParams::new(16, 32, true, 56 as u64);
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

    #[test]
    fn test_core_primitives_57() {
        let lp = LinearParams::new(16, 32, true, 57 as u64);
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

    #[test]
    fn test_core_primitives_58() {
        let lp = LinearParams::new(16, 32, true, 58 as u64);
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

    #[test]
    fn test_core_primitives_59() {
        let lp = LinearParams::new(16, 32, true, 59 as u64);
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

    #[test]
    fn test_core_primitives_60() {
        let lp = LinearParams::new(16, 32, true, 60 as u64);
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

    #[test]
    fn test_core_primitives_61() {
        let lp = LinearParams::new(16, 32, true, 61 as u64);
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

    #[test]
    fn test_core_primitives_62() {
        let lp = LinearParams::new(16, 32, true, 62 as u64);
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

    #[test]
    fn test_core_primitives_63() {
        let lp = LinearParams::new(16, 32, true, 63 as u64);
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

    #[test]
    fn test_core_primitives_64() {
        let lp = LinearParams::new(16, 32, true, 64 as u64);
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

    #[test]
    fn test_core_primitives_65() {
        let lp = LinearParams::new(16, 32, true, 65 as u64);
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

    #[test]
    fn test_core_primitives_66() {
        let lp = LinearParams::new(16, 32, true, 66 as u64);
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

    #[test]
    fn test_core_primitives_67() {
        let lp = LinearParams::new(16, 32, true, 67 as u64);
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

    #[test]
    fn test_core_primitives_68() {
        let lp = LinearParams::new(16, 32, true, 68 as u64);
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

    #[test]
    fn test_core_primitives_69() {
        let lp = LinearParams::new(16, 32, true, 69 as u64);
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

    #[test]
    fn test_core_primitives_70() {
        let lp = LinearParams::new(16, 32, true, 70 as u64);
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

    #[test]
    fn test_core_primitives_71() {
        let lp = LinearParams::new(16, 32, true, 71 as u64);
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

    #[test]
    fn test_core_primitives_72() {
        let lp = LinearParams::new(16, 32, true, 72 as u64);
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

    #[test]
    fn test_core_primitives_73() {
        let lp = LinearParams::new(16, 32, true, 73 as u64);
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

    #[test]
    fn test_core_primitives_74() {
        let lp = LinearParams::new(16, 32, true, 74 as u64);
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

    #[test]
    fn test_core_primitives_75() {
        let lp = LinearParams::new(16, 32, true, 75 as u64);
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

    #[test]
    fn test_core_primitives_76() {
        let lp = LinearParams::new(16, 32, true, 76 as u64);
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

    #[test]
    fn test_core_primitives_77() {
        let lp = LinearParams::new(16, 32, true, 77 as u64);
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

    #[test]
    fn test_core_primitives_78() {
        let lp = LinearParams::new(16, 32, true, 78 as u64);
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

    #[test]
    fn test_core_primitives_79() {
        let lp = LinearParams::new(16, 32, true, 79 as u64);
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

    #[test]
    fn test_core_primitives_80() {
        let lp = LinearParams::new(16, 32, true, 80 as u64);
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

    #[test]
    fn test_core_primitives_81() {
        let lp = LinearParams::new(16, 32, true, 81 as u64);
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

    #[test]
    fn test_core_primitives_82() {
        let lp = LinearParams::new(16, 32, true, 82 as u64);
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

    #[test]
    fn test_core_primitives_83() {
        let lp = LinearParams::new(16, 32, true, 83 as u64);
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

    #[test]
    fn test_core_primitives_84() {
        let lp = LinearParams::new(16, 32, true, 84 as u64);
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

    #[test]
    fn test_core_primitives_85() {
        let lp = LinearParams::new(16, 32, true, 85 as u64);
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

    #[test]
    fn test_core_primitives_86() {
        let lp = LinearParams::new(16, 32, true, 86 as u64);
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

    #[test]
    fn test_core_primitives_87() {
        let lp = LinearParams::new(16, 32, true, 87 as u64);
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

    #[test]
    fn test_core_primitives_88() {
        let lp = LinearParams::new(16, 32, true, 88 as u64);
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

    #[test]
    fn test_core_primitives_89() {
        let lp = LinearParams::new(16, 32, true, 89 as u64);
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

    #[test]
    fn test_core_primitives_90() {
        let lp = LinearParams::new(16, 32, true, 90 as u64);
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

    #[test]
    fn test_core_primitives_91() {
        let lp = LinearParams::new(16, 32, true, 91 as u64);
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

    #[test]
    fn test_core_primitives_92() {
        let lp = LinearParams::new(16, 32, true, 92 as u64);
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

    #[test]
    fn test_core_primitives_93() {
        let lp = LinearParams::new(16, 32, true, 93 as u64);
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

    #[test]
    fn test_core_primitives_94() {
        let lp = LinearParams::new(16, 32, true, 94 as u64);
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

    #[test]
    fn test_core_primitives_95() {
        let lp = LinearParams::new(16, 32, true, 95 as u64);
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

    #[test]
    fn test_core_primitives_96() {
        let lp = LinearParams::new(16, 32, true, 96 as u64);
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

    #[test]
    fn test_core_primitives_97() {
        let lp = LinearParams::new(16, 32, true, 97 as u64);
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

    #[test]
    fn test_core_primitives_98() {
        let lp = LinearParams::new(16, 32, true, 98 as u64);
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

    #[test]
    fn test_core_primitives_99() {
        let lp = LinearParams::new(16, 32, true, 99 as u64);
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

    #[test]
    fn test_core_primitives_100() {
        let lp = LinearParams::new(16, 32, true, 100 as u64);
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

    #[test]
    fn test_core_primitives_101() {
        let lp = LinearParams::new(16, 32, true, 101 as u64);
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

    #[test]
    fn test_core_primitives_102() {
        let lp = LinearParams::new(16, 32, true, 102 as u64);
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

    #[test]
    fn test_core_primitives_103() {
        let lp = LinearParams::new(16, 32, true, 103 as u64);
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

    #[test]
    fn test_core_primitives_104() {
        let lp = LinearParams::new(16, 32, true, 104 as u64);
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

    #[test]
    fn test_core_primitives_105() {
        let lp = LinearParams::new(16, 32, true, 105 as u64);
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

    #[test]
    fn test_core_primitives_106() {
        let lp = LinearParams::new(16, 32, true, 106 as u64);
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

    #[test]
    fn test_core_primitives_107() {
        let lp = LinearParams::new(16, 32, true, 107 as u64);
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

    #[test]
    fn test_core_primitives_108() {
        let lp = LinearParams::new(16, 32, true, 108 as u64);
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

    #[test]
    fn test_core_primitives_109() {
        let lp = LinearParams::new(16, 32, true, 109 as u64);
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

    #[test]
    fn test_core_primitives_110() {
        let lp = LinearParams::new(16, 32, true, 110 as u64);
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

    #[test]
    fn test_core_primitives_111() {
        let lp = LinearParams::new(16, 32, true, 111 as u64);
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

    #[test]
    fn test_core_primitives_112() {
        let lp = LinearParams::new(16, 32, true, 112 as u64);
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

    #[test]
    fn test_core_primitives_113() {
        let lp = LinearParams::new(16, 32, true, 113 as u64);
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

    #[test]
    fn test_core_primitives_114() {
        let lp = LinearParams::new(16, 32, true, 114 as u64);
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

    #[test]
    fn test_core_primitives_115() {
        let lp = LinearParams::new(16, 32, true, 115 as u64);
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

    #[test]
    fn test_core_primitives_116() {
        let lp = LinearParams::new(16, 32, true, 116 as u64);
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

    #[test]
    fn test_core_primitives_117() {
        let lp = LinearParams::new(16, 32, true, 117 as u64);
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

    #[test]
    fn test_core_primitives_118() {
        let lp = LinearParams::new(16, 32, true, 118 as u64);
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

    #[test]
    fn test_core_primitives_119() {
        let lp = LinearParams::new(16, 32, true, 119 as u64);
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

    #[test]
    fn test_core_primitives_120() {
        let lp = LinearParams::new(16, 32, true, 120 as u64);
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

    #[test]
    fn test_core_primitives_121() {
        let lp = LinearParams::new(16, 32, true, 121 as u64);
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

    #[test]
    fn test_core_primitives_122() {
        let lp = LinearParams::new(16, 32, true, 122 as u64);
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

    #[test]
    fn test_core_primitives_123() {
        let lp = LinearParams::new(16, 32, true, 123 as u64);
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

    #[test]
    fn test_core_primitives_124() {
        let lp = LinearParams::new(16, 32, true, 124 as u64);
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

    #[test]
    fn test_core_primitives_125() {
        let lp = LinearParams::new(16, 32, true, 125 as u64);
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

    #[test]
    fn test_core_primitives_126() {
        let lp = LinearParams::new(16, 32, true, 126 as u64);
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

    #[test]
    fn test_core_primitives_127() {
        let lp = LinearParams::new(16, 32, true, 127 as u64);
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

    #[test]
    fn test_core_primitives_128() {
        let lp = LinearParams::new(16, 32, true, 128 as u64);
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

    #[test]
    fn test_core_primitives_129() {
        let lp = LinearParams::new(16, 32, true, 129 as u64);
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

    #[test]
    fn test_core_primitives_130() {
        let lp = LinearParams::new(16, 32, true, 130 as u64);
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

    #[test]
    fn test_core_primitives_131() {
        let lp = LinearParams::new(16, 32, true, 131 as u64);
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

    #[test]
    fn test_core_primitives_132() {
        let lp = LinearParams::new(16, 32, true, 132 as u64);
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

    #[test]
    fn test_core_primitives_133() {
        let lp = LinearParams::new(16, 32, true, 133 as u64);
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

    #[test]
    fn test_core_primitives_134() {
        let lp = LinearParams::new(16, 32, true, 134 as u64);
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

    #[test]
    fn test_core_primitives_135() {
        let lp = LinearParams::new(16, 32, true, 135 as u64);
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
