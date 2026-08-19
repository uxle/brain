//! # NLP Core Types & Data Containers
//!
//! Foundational representations for tokens, spans, batches, and NLP error types.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use std::fmt;
use brain_core::Tensor;

/// Token identifier type.
pub type TokenId = usize;

/// Container for a sequence of token identifiers with utility operations.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct TokenIds(pub Vec<TokenId>);

impl TokenIds {
    /// Creates a new empty `TokenIds` container.
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// Creates a `TokenIds` container from a vector of IDs.
    pub fn from_vec(ids: Vec<TokenId>) -> Self {
        Self(ids)
    }

    /// Creates a `TokenIds` container from a slice.
    pub fn from_slice(slice: &[TokenId]) -> Self {
        Self(slice.to_vec())
    }

    /// Returns the number of token IDs.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns true if there are no token IDs.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns a slice view of the token IDs.
    pub fn as_slice(&self) -> &[TokenId] {
        &self.0
    }

    /// Returns a mutable slice view of the token IDs.
    pub fn as_mut_slice(&mut self) -> &mut [TokenId] {
        &mut self.0
    }

    /// Appends a token ID.
    pub fn push(&mut self, id: TokenId) {
        self.0.push(id);
    }

    /// Returns the inner vector of token IDs.
    pub fn to_vec(&self) -> Vec<TokenId> {
        self.0.clone()
    }

    /// Checks if a token ID exists in the sequence.
    pub fn contains_id(&self, id: TokenId) -> bool {
        self.0.contains(&id)
    }

    /// Pads the sequence to a target length with the given pad ID.
    pub fn pad_to(&mut self, target_len: usize, pad_id: TokenId) {
        while self.0.len() < target_len {
            self.0.push(pad_id);
        }
    }

    /// Truncates the sequence to a maximum length.
    pub fn truncate_to(&mut self, max_len: usize) {
        if self.0.len() > max_len {
            self.0.truncate(max_len);
        }
    }
}

impl From<Vec<TokenId>> for TokenIds {
    fn from(v: Vec<TokenId>) -> Self {
        Self(v)
    }
}

impl AsRef<[TokenId]> for TokenIds {
    fn as_ref(&self) -> &[TokenId] {
        &self.0
    }
}

/// Detailed metadata for a single token in text.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenMeta {
    /// String token representation.
    pub token: String,
    /// Numerical token identifier.
    pub id: TokenId,
    /// Byte start offset in original text.
    pub start_offset: usize,
    /// Byte end offset in original text.
    pub end_offset: usize,
    /// Whether this is a special control token.
    pub is_special: bool,
    /// Log-probability or piece score.
    pub score: f32,
}

impl TokenMeta {
    /// Creates a new token metadata record.
    pub fn new(token: impl Into<String>, id: TokenId, start_offset: usize, end_offset: usize) -> Self {
        Self {
            token: token.into(),
            id,
            start_offset,
            end_offset,
            is_special: false,
            score: 0.0,
        }
    }

    /// Sets whether the token is special.
    pub fn with_special(mut self, special: bool) -> Self {
        self.is_special = special;
        self
    }

    /// Sets the score for the token.
    pub fn with_score(mut self, score: f32) -> Self {
        self.score = score;
        self
    }
}

/// Output of a tokenizer encoding operation.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TokenizedOutput {
    /// Token IDs.
    pub ids: Vec<TokenId>,
    /// Token string representations.
    pub tokens: Vec<String>,
    /// Character or byte offsets `(start, end)` in the source string.
    pub offsets: Vec<(usize, usize)>,
    /// Attention mask (`1` for valid token, `0` for padding).
    pub attention_mask: Vec<u8>,
    /// Token type IDs (e.g. `0` for segment A, `1` for segment B).
    pub type_ids: Option<Vec<usize>>,
    /// Mask indicating which tokens are special tokens (`1` for special, `0` for normal).
    pub special_tokens_mask: Vec<u8>,
}

impl TokenizedOutput {
    /// Creates a new `TokenizedOutput` with default masks.
    pub fn new(ids: Vec<TokenId>, tokens: Vec<String>, offsets: Vec<(usize, usize)>) -> Self {
        let len = ids.len();
        let attention_mask = vec![1u8; len];
        let special_tokens_mask = vec![0u8; len];
        Self {
            ids,
            tokens,
            offsets,
            attention_mask,
            type_ids: None,
            special_tokens_mask,
        }
    }

    /// Returns the sequence length in tokens.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Sets the token type IDs.
    pub fn with_type_ids(mut self, type_ids: Vec<usize>) -> Self {
        self.type_ids = Some(type_ids);
        self
    }

    /// Sets the special tokens mask.
    pub fn with_special_tokens_mask(mut self, mask: Vec<u8>) -> Self {
        self.special_tokens_mask = mask;
        self
    }
}

/// Batch of tokenized sequences with uniform padded shapes.
#[derive(Debug, Clone, PartialEq)]
pub struct TextBatch {
    /// Individual tokenized sequences.
    pub sequences: Vec<TokenizedOutput>,
    /// Number of sequences in batch.
    pub batch_size: usize,
    /// Maximum sequence length across batch.
    pub max_length: usize,
    /// Padding token ID used for collation.
    pub pad_id: TokenId,
}

impl TextBatch {
    /// Creates a new `TextBatch` from a list of tokenized outputs.
    pub fn from_outputs(sequences: Vec<TokenizedOutput>, pad_id: TokenId) -> Self {
        let batch_size = sequences.len();
        let max_length = sequences.iter().map(|s| s.ids.len()).max().unwrap_or(0);
        Self {
            sequences,
            batch_size,
            max_length,
            pad_id,
        }
    }

    /// Converts the batch input IDs into a 2D `Tensor` of shape `[batch_size, max_length]`.
    pub fn to_tensor(&self) -> Tensor {
        let mut data = Vec::with_capacity(self.batch_size * self.max_length);
        for seq in &self.sequences {
            for &id in &seq.ids {
                data.push(id as f64);
            }
            for _ in seq.ids.len()..self.max_length {
                data.push(self.pad_id as f64);
            }
        }
        if data.is_empty() {
            Tensor::zeros(vec![0, 0])
        } else {
            Tensor::from_slice(&data, vec![self.batch_size, self.max_length])
        }
    }

    /// Converts the batch attention masks into a 2D `Tensor` of shape `[batch_size, max_length]`.
    pub fn to_attention_mask_tensor(&self) -> Tensor {
        let mut data = Vec::with_capacity(self.batch_size * self.max_length);
        for seq in &self.sequences {
            for &m in &seq.attention_mask {
                data.push(m as f64);
            }
            for _ in seq.attention_mask.len()..self.max_length {
                data.push(0.0);
            }
        }
        if data.is_empty() {
            Tensor::zeros(vec![0, 0])
        } else {
            Tensor::from_slice(&data, vec![self.batch_size, self.max_length])
        }
    }
}

/// Labeled text span (e.g. for NER or extraction).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextSpan {
    /// Start character/byte index.
    pub start: usize,
    /// End character/byte index.
    pub end: usize,
    /// Substring text.
    pub text: String,
    /// Optional categorical entity label.
    pub label: Option<String>,
}

impl TextSpan {
    /// Creates a new text span.
    pub fn new(start: usize, end: usize, text: impl Into<String>) -> Self {
        Self {
            start,
            end,
            text: text.into(),
            label: None,
        }
    }

    /// Sets the label for this span.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// Helper wrapper for vocabulary size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VocabSize(pub usize);

impl VocabSize {
    /// Creates a new vocabulary size.
    pub fn new(size: usize) -> Self {
        Self(size)
    }

    /// Returns the underlying usize.
    pub fn get(&self) -> usize {
        self.0
    }

    /// Checks if a given token ID is within vocabulary bounds.
    pub fn is_valid_id(&self, id: TokenId) -> bool {
        id < self.0
    }
}

/// Error types occurring in NLP, tokenization, embeddings, and text processing.
#[derive(Debug, Clone, PartialEq)]
pub enum TextError {
    /// Invalid or corrupted vocabulary.
    InvalidVocab(String),
    /// Tokenization algorithm failed.
    TokenizationFailed(String),
    /// Decoding IDs back to text failed.
    DecodingFailed(String),
    /// Invalid character or byte offset.
    InvalidOffset(String),
    /// Input text or token sequence is empty when non-empty required.
    EmptyInput,
    /// Token ID is out of vocabulary bounds.
    VocabOutOfBounds { id: usize, vocab_size: usize },
    /// Dimension mismatch in embedding lookup or tensor conversion.
    DimensionMismatch { expected: usize, found: usize },
    /// File I/O or serialization error.
    IoError(String),
    /// Configuration parameter is invalid.
    InvalidConfig(String),
    /// Pretrained model format parsing error.
    PretrainedLoadError(String),
    /// Generic processing error.
    ProcessingError(String),
}

impl fmt::Display for TextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TextError::InvalidVocab(msg) => write!(f, "Invalid vocabulary: {}", msg),
            TextError::TokenizationFailed(msg) => write!(f, "Tokenization failed: {}", msg),
            TextError::DecodingFailed(msg) => write!(f, "Decoding failed: {}", msg),
            TextError::InvalidOffset(msg) => write!(f, "Invalid offset: {}", msg),
            TextError::EmptyInput => write!(f, "Input text or sequence is empty"),
            TextError::VocabOutOfBounds { id, vocab_size } => {
                write!(f, "Token ID {} out of bounds for vocab size {}", id, vocab_size)
            }
            TextError::DimensionMismatch { expected, found } => {
                write!(f, "Dimension mismatch: expected {}, found {}", expected, found)
            }
            TextError::IoError(msg) => write!(f, "I/O error: {}", msg),
            TextError::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            TextError::PretrainedLoadError(msg) => write!(f, "Pretrained loading error: {}", msg),
            TextError::ProcessingError(msg) => write!(f, "Processing error: {}", msg),
        }
    }
}

impl std::error::Error for TextError {}

/// NLP Result type alias.
pub type TextResult<T> = Result<T, TextError>;

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
    fn test_core_functionality_1() {
        let mut ids = TokenIds::new();
        ids.push(1);
        ids.push(1 + 1);
        assert_eq!(ids.len(), 2);
        assert!(ids.contains_id(1));
        assert!(!ids.contains_id(1 + 99999));
        ids.pad_to(5, 0);
        assert_eq!(ids.len(), 5);
        ids.truncate_to(3);
        assert_eq!(ids.len(), 3);

        let meta = TokenMeta::new("token_1", 1, 0, 5)
            .with_special(false)
            .with_score(0.1f32);
        assert_eq!(meta.id, 1);
        assert_eq!(meta.token, "token_1");

        let output = TokenizedOutput::new(vec![1, 1+1], vec!["a".to_string(), "b".to_string()], vec![(0,1), (1,2)]);
        assert_eq!(output.len(), 2);
        assert_eq!(output.attention_mask, vec![1, 1]);

        let batch = TextBatch::from_outputs(vec![output], 0);
        assert_eq!(batch.batch_size, 1);
        assert_eq!(batch.max_length, 2);
        let t = batch.to_tensor();
        assert_eq!(t.shape(), &[1, 2]);

        let vs = VocabSize::new(10000 + 1);
        assert!(vs.is_valid_id(1));
        assert!(!vs.is_valid_id(20000 + 1));

        let err = TextError::VocabOutOfBounds { id: 100, vocab_size: 50 };
        assert!(err.to_string().contains("out of bounds"));
    }
}
