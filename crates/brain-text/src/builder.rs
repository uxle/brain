//! # Fluent Builders for Tokenizers, Embeddings, and NLP Pipelines
//!
//! Fluent API builders for BPE, SentencePiece, WordPiece, character tokenizers, and embedding layers.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::embedding::WordEmbedding;
use crate::tokenizer::bpe::{BpeConfig, BpeTokenizer};
use crate::tokenizer::char::{CharConfig, CharTokenizer, WordConfig, WordTokenizer};
use crate::tokenizer::sentencepiece::{SentencePieceTokenizer, SpConfig};
use crate::tokenizer::trainer::{BpeTrainer, TrainConfig, UnigramTrainer, WordPieceTrainer};
use crate::tokenizer::wordpiece::{WordPieceConfig, WordPieceTokenizer};
use crate::vocab::Vocab;

/// Unified entry point builder for NLP models and tokenizers.
#[derive(Debug, Clone, Default)]
pub struct TextBuilder;

impl TextBuilder {
    /// Creates a new `TextBuilder`.
    pub fn new() -> Self {
        Self
    }

    /// Begins constructing a Byte-Pair Encoding (BPE) tokenizer.
    pub fn bpe(&self) -> BpeBuilder {
        BpeBuilder::new()
    }

    /// Begins constructing a SentencePiece Unigram tokenizer.
    pub fn sentencepiece(&self) -> SpBuilder {
        SpBuilder::new()
    }

    /// Begins constructing a WordPiece tokenizer.
    pub fn wordpiece(&self) -> WordPieceBuilder {
        WordPieceBuilder::new()
    }

    /// Begins constructing a character tokenizer.
    pub fn char_tokenizer(&self) -> CharTokenizerBuilder {
        CharTokenizerBuilder::new()
    }

    /// Begins constructing a word tokenizer.
    pub fn word_tokenizer(&self) -> WordTokenizerBuilder {
        WordTokenizerBuilder::new()
    }

    /// Begins constructing a trainable WordEmbedding layer.
    pub fn embedding(&self, vocab_size: usize, dim: usize) -> EmbeddingBuilder {
        EmbeddingBuilder::new(vocab_size, dim)
    }
}

/// Fluent builder for BPE tokenizers.
#[derive(Debug, Clone, Default)]
pub struct BpeBuilder {
    config: BpeConfig,
}

impl BpeBuilder {
    /// Creates a new `BpeBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets target vocabulary size.
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.config.vocab_size = size;
        self
    }

    /// Sets minimum merge frequency.
    pub fn min_frequency(mut self, freq: usize) -> Self {
        self.config.min_frequency = freq;
        self
    }

    /// Trains and returns a new `BpeTokenizer` from text corpus.
    pub fn train(self, corpus: &[&str]) -> BpeTokenizer {
        let train_cfg = TrainConfig {
            vocab_size: self.config.vocab_size,
            min_frequency: self.config.min_frequency,
            ..Default::default()
        };
        let (vocab, merges) = BpeTrainer::train(corpus, &train_cfg);
        BpeTokenizer::from_vocab_and_merges(vocab, merges, self.config)
    }
}

/// Fluent builder for SentencePiece tokenizers.
#[derive(Debug, Clone, Default)]
pub struct SpBuilder {
    config: SpConfig,
}

impl SpBuilder {
    /// Creates a new `SpBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets target vocabulary size.
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.config.vocab_size = size;
        self
    }

    /// Trains and returns a new `SentencePieceTokenizer` from text corpus.
    pub fn train(self, corpus: &[&str]) -> SentencePieceTokenizer {
        let train_cfg = TrainConfig {
            vocab_size: self.config.vocab_size,
            ..Default::default()
        };
        let (vocab, scores) = UnigramTrainer::train(corpus, &train_cfg);
        SentencePieceTokenizer::from_pieces(vocab, scores, self.config)
    }
}

/// Fluent builder for WordPiece tokenizers.
#[derive(Debug, Clone, Default)]
pub struct WordPieceBuilder {
    config: WordPieceConfig,
}

impl WordPieceBuilder {
    /// Creates a new `WordPieceBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets target vocabulary size.
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.config.vocab_size = size;
        self
    }

    /// Trains and returns a new `WordPieceTokenizer` from text corpus.
    pub fn train(self, corpus: &[&str]) -> WordPieceTokenizer {
        let train_cfg = TrainConfig {
            vocab_size: self.config.vocab_size,
            ..Default::default()
        };
        let vocab = WordPieceTrainer::train(corpus, &train_cfg);
        WordPieceTokenizer::new(vocab, self.config)
    }
}

/// Fluent builder for character tokenizers.
#[derive(Debug, Clone, Default)]
pub struct CharTokenizerBuilder {
    config: CharConfig,
}

impl CharTokenizerBuilder {
    /// Creates a new `CharTokenizerBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets lowercase flag.
    pub fn lowercase(mut self, lower: bool) -> Self {
        self.config.lowercase = lower;
        self
    }

    /// Builds the `CharTokenizer` with the supplied vocabulary.
    pub fn build(self, vocab: Vocab) -> CharTokenizer {
        CharTokenizer::new(vocab, self.config)
    }
}

/// Fluent builder for word tokenizers.
#[derive(Debug, Clone, Default)]
pub struct WordTokenizerBuilder {
    config: WordConfig,
}

impl WordTokenizerBuilder {
    /// Creates a new `WordTokenizerBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets lowercase flag.
    pub fn lowercase(mut self, lower: bool) -> Self {
        self.config.lowercase = lower;
        self
    }

    /// Builds the `WordTokenizer` with the supplied vocabulary.
    pub fn build(self, vocab: Vocab) -> WordTokenizer {
        WordTokenizer::new(vocab, self.config)
    }
}

/// Fluent builder for WordEmbedding layers.
#[derive(Debug, Clone)]
pub struct EmbeddingBuilder {
    vocab_size: usize,
    dim: usize,
    padding_idx: Option<usize>,
}

impl EmbeddingBuilder {
    /// Creates a new `EmbeddingBuilder`.
    pub fn new(vocab_size: usize, dim: usize) -> Self {
        Self {
            vocab_size,
            dim,
            padding_idx: None,
        }
    }

    /// Sets the padding token index.
    pub fn padding_idx(mut self, idx: usize) -> Self {
        self.padding_idx = Some(idx);
        self
    }

    /// Builds and initializes the `WordEmbedding` layer.
    pub fn build(self) -> WordEmbedding {
        WordEmbedding::new(self.vocab_size, self.dim, self.padding_idx)
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
    fn test_builders_suite_1() {
        let corpus = vec!["hello world", "hello neural network_1"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }
}
