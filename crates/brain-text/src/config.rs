//! # Configuration Models for NLP Pipelines
//!
//! Structural configurations for tokenizers, embeddings, sequence processing, and unified pipelines.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::core::{TextError, TextResult};

/// Available tokenizer algorithm architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenizerType {
    /// Byte-Pair Encoding (BPE) algorithm.
    Bpe,
    /// SentencePiece Unigram language model.
    SentencePiece,
    /// WordPiece subword algorithm (BERT style).
    WordPiece,
    /// Character-level tokenizer.
    Char,
    /// Whitespace and punctuation word-level tokenizer.
    Word,
    /// Raw byte-level tokenizer.
    ByteLevel,
}

/// Side on which sequence padding is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PaddingSide {
    /// Pad tokens to the left.
    Left,
    /// Pad tokens to the right (default).
    #[default]
    Right,
}

/// Side on which sequence truncation is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TruncationSide {
    /// Truncate tokens from the left.
    Left,
    /// Truncate tokens from the right (default).
    #[default]
    Right,
}

/// Special tokens configuration.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SpecialTokensConfig {
    /// Padding token string (e.g. `[PAD]` or `<pad>`).
    pub pad_token: Option<String>,
    /// Unknown token string (e.g. `[UNK]` or `<unk>`).
    pub unk_token: Option<String>,
    /// Beginning of sequence token string (e.g. `[BOS]` or `<s>`).
    pub bos_token: Option<String>,
    /// End of sequence token string (e.g. `[EOS]` or `</s>`).
    pub eos_token: Option<String>,
    /// Mask token string (e.g. `[MASK]` or `<mask>`).
    pub mask_token: Option<String>,
    /// Separator token string (e.g. `[SEP]`).
    pub sep_token: Option<String>,
    /// Classification / document token string (e.g. `[CLS]`).
    pub cls_token: Option<String>,
}

impl SpecialTokensConfig {
    /// Creates a standard BERT-style special tokens configuration.
    pub fn bert() -> Self {
        Self {
            pad_token: Some("[PAD]".to_string()),
            unk_token: Some("[UNK]".to_string()),
            bos_token: None,
            eos_token: None,
            mask_token: Some("[MASK]".to_string()),
            sep_token: Some("[SEP]".to_string()),
            cls_token: Some("[CLS]".to_string()),
        }
    }

    /// Creates a standard GPT-2 style special tokens configuration.
    pub fn gpt2() -> Self {
        Self {
            pad_token: Some("<|endoftext|>".to_string()),
            unk_token: Some("<|endoftext|>".to_string()),
            bos_token: Some("<|endoftext|>".to_string()),
            eos_token: Some("<|endoftext|>".to_string()),
            mask_token: None,
            sep_token: None,
            cls_token: None,
        }
    }

    /// Creates a standard T5 / RoBERTa style special tokens configuration.
    pub fn roberta() -> Self {
        Self {
            pad_token: Some("<pad>".to_string()),
            unk_token: Some("<unk>".to_string()),
            bos_token: Some("<s>".to_string()),
            eos_token: Some("</s>".to_string()),
            mask_token: Some("<mask>".to_string()),
            sep_token: Some("</s>".to_string()),
            cls_token: Some("<s>".to_string()),
        }
    }
}

/// Configuration for tokenizer initialization and behavior.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenizerConfig {
    /// Tokenizer algorithm type.
    pub tokenizer_type: TokenizerType,
    /// Target vocabulary size.
    pub vocab_size: usize,
    /// Convert text to lowercase before tokenizing.
    pub lowercase: bool,
    /// Strip accents and diacritics.
    pub strip_accents: bool,
    /// Special tokens definitions.
    pub special_tokens: SpecialTokensConfig,
    /// Maximum allowed sequence length.
    pub max_length: Option<usize>,
    /// Enable automatic truncation to max length.
    pub truncation: bool,
    /// Enable automatic padding to max length.
    pub padding: bool,
    /// Prepend whitespace prefix.
    pub add_prefix_space: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            tokenizer_type: TokenizerType::Bpe,
            vocab_size: 32000,
            lowercase: false,
            strip_accents: false,
            special_tokens: SpecialTokensConfig::gpt2(),
            max_length: Some(512),
            truncation: true,
            padding: false,
            add_prefix_space: false,
        }
    }
}

impl TokenizerConfig {
    /// Validates the tokenizer configuration.
    pub fn validate(&self) -> TextResult<()> {
        if self.vocab_size == 0 {
            return Err(TextError::InvalidConfig("Vocab size cannot be 0".to_string()));
        }
        if let Some(max_len) = self.max_length {
            if max_len == 0 {
                return Err(TextError::InvalidConfig("Max length cannot be 0".to_string()));
            }
        }
        Ok(())
    }
}

/// Configuration for word and positional embedding layers.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingConfig {
    /// Vocabulary size for embedding table.
    pub vocab_size: usize,
    /// Dimension of embedding vectors.
    pub embedding_dim: usize,
    /// Padding token index (zeroed / ignored during gradient computation).
    pub padding_idx: Option<usize>,
    /// Optional maximum vector L2 norm for clipping.
    pub max_norm: Option<f32>,
    /// Scale gradients by inverse frequency.
    pub scale_grad_by_freq: bool,
    /// Sparse gradient updates flag.
    pub sparse: bool,
    /// Whether positional embeddings are learned or fixed sinusoidal.
    pub learned_positional: bool,
    /// Maximum positional context length.
    pub max_position_embeddings: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            embedding_dim: 768,
            padding_idx: Some(0),
            max_norm: None,
            scale_grad_by_freq: false,
            sparse: false,
            learned_positional: false,
            max_position_embeddings: 512,
        }
    }
}

impl EmbeddingConfig {
    /// Validates the embedding configuration.
    pub fn validate(&self) -> TextResult<()> {
        if self.vocab_size == 0 {
            return Err(TextError::InvalidConfig("Vocab size must be > 0".to_string()));
        }
        if self.embedding_dim == 0 {
            return Err(TextError::InvalidConfig("Embedding dim must be > 0".to_string()));
        }
        if self.max_position_embeddings == 0 {
            return Err(TextError::InvalidConfig("Max position embeddings must be > 0".to_string()));
        }
        Ok(())
    }
}

/// Configuration for batch sequence processing and formatting.
#[derive(Debug, Clone, PartialEq)]
pub struct ProcessConfig {
    /// Target maximum sequence length.
    pub max_length: usize,
    /// Pad batch sequences to max_length.
    pub pad_to_max: bool,
    /// Side to pad sequences.
    pub padding_side: PaddingSide,
    /// Side to truncate sequences.
    pub truncation_side: TruncationSide,
    /// Return output as Tensor structures.
    pub return_tensors: bool,
}

impl Default for ProcessConfig {
    fn default() -> Self {
        Self {
            max_length: 512,
            pad_to_max: false,
            padding_side: PaddingSide::Right,
            truncation_side: TruncationSide::Right,
            return_tensors: true,
        }
    }
}

/// Unified comprehensive NLP configuration.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TextConfig {
    /// Tokenizer configuration.
    pub tokenizer: TokenizerConfig,
    /// Embedding configuration.
    pub embedding: EmbeddingConfig,
    /// Sequence processing configuration.
    pub process: ProcessConfig,
}

impl TextConfig {
    /// Creates a standard default TextConfig.
    pub fn new() -> Self {
        Self::default()
    }

    /// Validates all sub-configurations.
    pub fn validate(&self) -> TextResult<()> {
        self.tokenizer.validate()?;
        self.embedding.validate()?;
        if self.process.max_length == 0 {
            return Err(TextError::InvalidConfig("Process max_length must be > 0".to_string()));
        }
        Ok(())
    }

    /// Returns a human-readable textual summary of configuration settings.
    pub fn summary(&self) -> String {
        format!(
            "TextConfig: tokenizer={:?}, vocab={}, emb_dim={}, max_len={}",
            self.tokenizer.tokenizer_type,
            self.tokenizer.vocab_size,
            self.embedding.embedding_dim,
            self.process.max_length,
        )
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
    fn test_config_pipeline_1() {
        let mut cfg = TextConfig::new();
        cfg.tokenizer.vocab_size = 1000 + 1;
        cfg.embedding.vocab_size = 1000 + 1;
        cfg.embedding.embedding_dim = 128 + (1 % 64);
        assert!(cfg.validate().is_ok());
        let sum = cfg.summary();
        assert!(sum.contains("TextConfig"));

        let mut invalid_cfg = cfg.clone();
        invalid_cfg.tokenizer.vocab_size = 0;
        assert!(invalid_cfg.validate().is_err());

        let mut invalid_emb = cfg.clone();
        invalid_emb.embedding.embedding_dim = 0;
        assert!(invalid_emb.validate().is_err());

        let bert_spec = SpecialTokensConfig::bert();
        assert_eq!(bert_spec.cls_token.as_deref(), Some("[CLS]"));
        let gpt_spec = SpecialTokensConfig::gpt2();
        assert_eq!(gpt_spec.eos_token.as_deref(), Some("<|endoftext|>"));
    }
}
