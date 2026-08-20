//! # Transformer Architecture Configurations & Hyperparameters
//!
//! Comprehensive configuration structs for encoders, decoders, attention mechanisms, normalizations, activations, and unified transformers.
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

/// Non-linear activation function type for Feed-Forward Networks and Attention projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ActivationType {
    /// Gaussian Error Linear Unit (standard or tanh approximation).
    #[default]
    Gelu,
    /// Rectified Linear Unit: $\max(0, x)$.
    Relu,
    /// Sigmoid Linear Unit / Swish: $x \cdot \sigma(x)$.
    Silu,
    /// Fast approximation to GELU.
    QuickGelu,
    /// SwiGLU gated activation for modern architectures (Llama / PaLM).
    Swiglu,
    /// GEGLU gated activation.
    Geglu,
}

/// Normalization layer placement strategy relative to sub-layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum NormPosition {
    /// Pre-LayerNorm: normalize input before attention and FFN sub-layers (standard in modern transformers).
    #[default]
    PreNorm,
    /// Post-LayerNorm: normalize after residual addition (original Vaswani et al. 2017).
    PostNorm,
    /// Dual / Sandwich Norm: normalize both before sub-layer and after residual connection.
    SandwichNorm,
}

/// Normalization algorithm variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum NormType {
    /// Standard Layer Normalization with mean subtraction and variance division.
    #[default]
    LayerNorm,
    /// Root Mean Square Normalization (RMSNorm) without mean subtraction (Zhang & Sennrich 2019).
    RmsNorm,
}

/// Transformer model architecture family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ModelArch {
    /// Encoder-only bidirectional representation model (BERT style).
    #[default]
    EncoderOnly,
    /// Decoder-only autoregressive causal language model (GPT style).
    DecoderOnly,
    /// Encoder-Decoder sequence-to-sequence model (T5 / BART style).
    EncoderDecoder,
}

/// Positional encoding strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PositionEncodingType {
    /// Rotary Position Embedding (RoPE) applied in attention heads (Su et al. 2021).
    #[default]
    Rope,
    /// Attention with Linear Biases (ALiBi) added to attention matrix (Press et al. 2021).
    Alibi,
    /// Fixed sinusoidal positional embeddings (Vaswani et al. 2017).
    Sinusoidal,
    /// Learned 1D absolute positional embedding table.
    Learned,
    /// Relative positional bias table (T5 / Shaw).
    RelativeBias,
    /// No positional encoding (e.g. for pure continuous graphs or permutation-invariant models).
    None,
}

/// Configuration for multi-head attention layers.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionConfig {
    /// Model hidden dimension (e.g. 768, 1024, 4096).
    pub hidden_dim: usize,
    /// Number of attention query heads (e.g. 12, 16, 32).
    pub num_heads: usize,
    /// Number of key/value heads for Grouped Query Attention (GQA / MQA). If equal to `num_heads`, standard MHA is used.
    pub num_kv_heads: usize,
    /// Head dimension: `hidden_dim / num_heads` by default.
    pub head_dim: usize,
    /// Attention dropout probability.
    pub dropout: f32,
    /// Whether query, key, value projections include additive bias vectors.
    pub bias: bool,
    /// Positional encoding type used in attention computation.
    pub pos_encoding: PositionEncodingType,
    /// RoPE base frequency parameter (theta, typically 10000.0 or 500000.0).
    pub rope_theta: f32,
    /// Maximum context sequence length.
    pub max_seq_len: usize,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        let hidden_dim = 768;
        let num_heads = 12;
        Self {
            hidden_dim,
            num_heads,
            num_kv_heads: num_heads,
            head_dim: hidden_dim / num_heads,
            dropout: 0.0,
            bias: false,
            pos_encoding: PositionEncodingType::Rope,
            rope_theta: 10000.0,
            max_seq_len: 2048,
        }
    }
}

impl AttentionConfig {
    /// Validates attention parameters and checks head dimension divisibility.
    pub fn validate(&self) -> TransformerResult<()> {
        if self.hidden_dim == 0 || self.num_heads == 0 {
            return Err(TransformerError::InvalidConfig(
                "hidden_dim and num_heads must be > 0".into(),
            ));
        }
        if self.hidden_dim % self.num_heads != 0 {
            return Err(TransformerError::InvalidHeadDim {
                hidden_dim: self.hidden_dim,
                num_heads: self.num_heads,
            });
        }
        if self.num_kv_heads == 0 || self.num_heads % self.num_kv_heads != 0 {
            return Err(TransformerError::InvalidConfig(format!(
                "num_heads ({}) must be divisible by num_kv_heads ({})",
                self.num_heads, self.num_kv_heads
            )));
        }
        Ok(())
    }
}

/// Configuration for Feed-Forward Network (FFN) / MLP sub-layers.
#[derive(Debug, Clone, PartialEq)]
pub struct FfnConfig {
    /// Input / output hidden dimension.
    pub hidden_dim: usize,
    /// Intermediate expansion dimension (typically `4 * hidden_dim` or `(8/3) * hidden_dim` for SwiGLU).
    pub intermediate_dim: usize,
    /// Activation function.
    pub activation: ActivationType,
    /// Dropout probability.
    pub dropout: f32,
    /// Whether linear layers include bias terms.
    pub bias: bool,
}

impl Default for FfnConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 768,
            intermediate_dim: 3072,
            activation: ActivationType::Gelu,
            dropout: 0.0,
            bias: false,
        }
    }
}

/// Unified comprehensive configuration for full Transformer models.
#[derive(Debug, Clone, PartialEq)]
pub struct TransformerConfig {
    /// Model architectural family.
    pub arch: ModelArch,
    /// Vocabulary size for input token embeddings and language model output projection.
    pub vocab_size: usize,
    /// Model representation hidden dimension $d_{\text{model}}$.
    pub hidden_dim: usize,
    /// Number of transformer blocks (layers).
    pub num_layers: usize,
    /// Attention sub-layer configuration.
    pub attention: AttentionConfig,
    /// Feed-forward network sub-layer configuration.
    pub ffn: FfnConfig,
    /// Normalization placement (PreNorm / PostNorm).
    pub norm_position: NormPosition,
    /// Normalization algorithm (LayerNorm / RMSNorm).
    pub norm_type: NormType,
    /// Normalization epsilon for numerical stability.
    pub norm_eps: f64,
    /// Maximum sequence length for context window.
    pub max_seq_len: usize,
    /// Tie input token embedding weights with output language modeling head weights.
    pub tie_word_embeddings: bool,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        let hidden_dim = 768;
        let num_heads = 12;
        let attention = AttentionConfig {
            hidden_dim,
            num_heads,
            num_kv_heads: num_heads,
            head_dim: hidden_dim / num_heads,
            dropout: 0.0,
            bias: false,
            pos_encoding: PositionEncodingType::Rope,
            rope_theta: 10000.0,
            max_seq_len: 2048,
        };
        let ffn = FfnConfig {
            hidden_dim,
            intermediate_dim: 3072,
            activation: ActivationType::Gelu,
            dropout: 0.0,
            bias: false,
        };
        Self {
            arch: ModelArch::DecoderOnly,
            vocab_size: 32000,
            hidden_dim,
            num_layers: 12,
            attention,
            ffn,
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            norm_eps: 1e-6,
            max_seq_len: 2048,
            tie_word_embeddings: false,
        }
    }
}

impl TransformerConfig {
    /// Creates a BERT-style bidirectional encoder configuration.
    pub fn bert_base() -> Self {
        Self {
            arch: ModelArch::EncoderOnly,
            vocab_size: 30522,
            hidden_dim: 768,
            num_layers: 12,
            attention: AttentionConfig {
                hidden_dim: 768,
                num_heads: 12,
                num_kv_heads: 12,
                head_dim: 64,
                dropout: 0.1,
                bias: true,
                pos_encoding: PositionEncodingType::Learned,
                rope_theta: 10000.0,
                max_seq_len: 512,
            },
            ffn: FfnConfig {
                hidden_dim: 768,
                intermediate_dim: 3072,
                activation: ActivationType::Gelu,
                dropout: 0.1,
                bias: true,
            },
            norm_position: NormPosition::PostNorm,
            norm_type: NormType::LayerNorm,
            norm_eps: 1e-12,
            max_seq_len: 512,
            tie_word_embeddings: true,
        }
    }

    /// Creates a GPT-2 style causal autoregressive language model configuration.
    pub fn gpt2_small() -> Self {
        Self {
            arch: ModelArch::DecoderOnly,
            vocab_size: 50257,
            hidden_dim: 768,
            num_layers: 12,
            attention: AttentionConfig {
                hidden_dim: 768,
                num_heads: 12,
                num_kv_heads: 12,
                head_dim: 64,
                dropout: 0.1,
                bias: true,
                pos_encoding: PositionEncodingType::Learned,
                rope_theta: 10000.0,
                max_seq_len: 1024,
            },
            ffn: FfnConfig {
                hidden_dim: 768,
                intermediate_dim: 3072,
                activation: ActivationType::Gelu,
                dropout: 0.1,
                bias: true,
            },
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::LayerNorm,
            norm_eps: 1e-5,
            max_seq_len: 1024,
            tie_word_embeddings: true,
        }
    }

    /// Creates a modern LLaMA-style autoregressive language model configuration.
    pub fn llama_7b() -> Self {
        let hidden_dim = 4096;
        let num_heads = 32;
        Self {
            arch: ModelArch::DecoderOnly,
            vocab_size: 32000,
            hidden_dim,
            num_layers: 32,
            attention: AttentionConfig {
                hidden_dim,
                num_heads,
                num_kv_heads: num_heads,
                head_dim: 128,
                dropout: 0.0,
                bias: false,
                pos_encoding: PositionEncodingType::Rope,
                rope_theta: 10000.0,
                max_seq_len: 4096,
            },
            ffn: FfnConfig {
                hidden_dim,
                intermediate_dim: 11008,
                activation: ActivationType::Swiglu,
                dropout: 0.0,
                bias: false,
            },
            norm_position: NormPosition::PreNorm,
            norm_type: NormType::RmsNorm,
            norm_eps: 1e-5,
            max_seq_len: 4096,
            tie_word_embeddings: false,
        }
    }

    /// Validates all configuration parameters and dimensional consistencies.
    pub fn validate(&self) -> TransformerResult<()> {
        if self.vocab_size == 0 {
            return Err(TransformerError::InvalidConfig(
                "vocab_size must be > 0".into(),
            ));
        }
        if self.hidden_dim == 0 {
            return Err(TransformerError::InvalidConfig(
                "hidden_dim must be > 0".into(),
            ));
        }
        if self.num_layers == 0 {
            return Err(TransformerError::InvalidConfig(
                "num_layers must be > 0".into(),
            ));
        }
        if self.hidden_dim != self.attention.hidden_dim || self.hidden_dim != self.ffn.hidden_dim {
            return Err(TransformerError::InvalidConfig(
                "hidden_dim must match between transformer, attention, and ffn configs".into(),
            ));
        }
        self.attention.validate()?;
        Ok(())
    }

    /// Returns a multi-line formatted summary of model architecture parameters and estimated weight counts.
    pub fn summary(&self) -> String {
        let emb_params = self.vocab_size * self.hidden_dim;
        let attn_qkv = self.hidden_dim
            * (self.hidden_dim + 2 * self.attention.num_kv_heads * self.attention.head_dim);
        let attn_out = self.hidden_dim * self.hidden_dim;
        let ffn_params = match self.ffn.activation {
            ActivationType::Swiglu | ActivationType::Geglu => {
                3 * self.hidden_dim * self.ffn.intermediate_dim
            }
            _ => 2 * self.hidden_dim * self.ffn.intermediate_dim,
        };
        let per_layer_params = attn_qkv + attn_out + ffn_params + 2 * self.hidden_dim;
        let total_params = emb_params + self.num_layers * per_layer_params;

        format!(
            "TransformerConfig Summary:\n  Arch: {:?}\n  Layers: {}\n  Hidden Dim: {}\n  Heads: {} (KV: {})\n  Head Dim: {}\n  FFN Dim: {}\n  Activation: {:?}\n  Norm: {:?} ({:?})\n  Max Context: {}\n  Est. Parameters: ~{:.2}M",
            self.arch,
            self.num_layers,
            self.hidden_dim,
            self.attention.num_heads,
            self.attention.num_kv_heads,
            self.attention.head_dim,
            self.ffn.intermediate_dim,
            self.ffn.activation,
            self.norm_type,
            self.norm_position,
            self.max_seq_len,
            total_params as f64 / 1_000_000.0,
        )
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
    fn test_config_pipeline_1() {
        let mut cfg = TransformerConfig::default();
        cfg.vocab_size = 1000 + 1;
        cfg.hidden_dim = 64;
        cfg.num_layers = 2;
        cfg.attention.hidden_dim = 64;
        cfg.attention.num_heads = 4;
        cfg.attention.num_kv_heads = 2;
        cfg.attention.head_dim = 16;
        cfg.ffn.hidden_dim = 64;
        cfg.ffn.intermediate_dim = 128;

        assert!(cfg.validate().is_ok());
        let sum = cfg.summary();
        assert!(sum.contains("TransformerConfig Summary"));

        let bert = TransformerConfig::bert_base();
        assert!(bert.validate().is_ok());

        let gpt = TransformerConfig::gpt2_small();
        assert!(gpt.validate().is_ok());

        let llama = TransformerConfig::llama_7b();
        assert!(llama.validate().is_ok());
    }
}
