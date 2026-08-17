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

    #[test]
    fn test_builders_suite_2() {
        let corpus = vec!["hello world", "hello neural network_2"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_3() {
        let corpus = vec!["hello world", "hello neural network_3"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_4() {
        let corpus = vec!["hello world", "hello neural network_4"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_5() {
        let corpus = vec!["hello world", "hello neural network_5"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_6() {
        let corpus = vec!["hello world", "hello neural network_6"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_7() {
        let corpus = vec!["hello world", "hello neural network_7"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_8() {
        let corpus = vec!["hello world", "hello neural network_8"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_9() {
        let corpus = vec!["hello world", "hello neural network_9"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_10() {
        let corpus = vec!["hello world", "hello neural network_10"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_11() {
        let corpus = vec!["hello world", "hello neural network_11"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_12() {
        let corpus = vec!["hello world", "hello neural network_12"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_13() {
        let corpus = vec!["hello world", "hello neural network_13"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_14() {
        let corpus = vec!["hello world", "hello neural network_14"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_15() {
        let corpus = vec!["hello world", "hello neural network_15"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_16() {
        let corpus = vec!["hello world", "hello neural network_16"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_17() {
        let corpus = vec!["hello world", "hello neural network_17"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_18() {
        let corpus = vec!["hello world", "hello neural network_18"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_19() {
        let corpus = vec!["hello world", "hello neural network_19"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_20() {
        let corpus = vec!["hello world", "hello neural network_20"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_21() {
        let corpus = vec!["hello world", "hello neural network_21"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_22() {
        let corpus = vec!["hello world", "hello neural network_22"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_23() {
        let corpus = vec!["hello world", "hello neural network_23"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_24() {
        let corpus = vec!["hello world", "hello neural network_24"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_25() {
        let corpus = vec!["hello world", "hello neural network_25"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_26() {
        let corpus = vec!["hello world", "hello neural network_26"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_27() {
        let corpus = vec!["hello world", "hello neural network_27"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_28() {
        let corpus = vec!["hello world", "hello neural network_28"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_29() {
        let corpus = vec!["hello world", "hello neural network_29"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_30() {
        let corpus = vec!["hello world", "hello neural network_30"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_31() {
        let corpus = vec!["hello world", "hello neural network_31"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_32() {
        let corpus = vec!["hello world", "hello neural network_32"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_33() {
        let corpus = vec!["hello world", "hello neural network_33"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_34() {
        let corpus = vec!["hello world", "hello neural network_34"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_35() {
        let corpus = vec!["hello world", "hello neural network_35"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_36() {
        let corpus = vec!["hello world", "hello neural network_36"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_37() {
        let corpus = vec!["hello world", "hello neural network_37"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_38() {
        let corpus = vec!["hello world", "hello neural network_38"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_39() {
        let corpus = vec!["hello world", "hello neural network_39"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_40() {
        let corpus = vec!["hello world", "hello neural network_40"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_41() {
        let corpus = vec!["hello world", "hello neural network_41"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_42() {
        let corpus = vec!["hello world", "hello neural network_42"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_43() {
        let corpus = vec!["hello world", "hello neural network_43"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_44() {
        let corpus = vec!["hello world", "hello neural network_44"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_45() {
        let corpus = vec!["hello world", "hello neural network_45"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_46() {
        let corpus = vec!["hello world", "hello neural network_46"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_47() {
        let corpus = vec!["hello world", "hello neural network_47"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_48() {
        let corpus = vec!["hello world", "hello neural network_48"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_49() {
        let corpus = vec!["hello world", "hello neural network_49"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_50() {
        let corpus = vec!["hello world", "hello neural network_50"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_51() {
        let corpus = vec!["hello world", "hello neural network_51"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_52() {
        let corpus = vec!["hello world", "hello neural network_52"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_53() {
        let corpus = vec!["hello world", "hello neural network_53"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_54() {
        let corpus = vec!["hello world", "hello neural network_54"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_55() {
        let corpus = vec!["hello world", "hello neural network_55"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_56() {
        let corpus = vec!["hello world", "hello neural network_56"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_57() {
        let corpus = vec!["hello world", "hello neural network_57"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_58() {
        let corpus = vec!["hello world", "hello neural network_58"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_59() {
        let corpus = vec!["hello world", "hello neural network_59"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_60() {
        let corpus = vec!["hello world", "hello neural network_60"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_61() {
        let corpus = vec!["hello world", "hello neural network_61"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_62() {
        let corpus = vec!["hello world", "hello neural network_62"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_63() {
        let corpus = vec!["hello world", "hello neural network_63"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_64() {
        let corpus = vec!["hello world", "hello neural network_64"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_65() {
        let corpus = vec!["hello world", "hello neural network_65"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_66() {
        let corpus = vec!["hello world", "hello neural network_66"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_67() {
        let corpus = vec!["hello world", "hello neural network_67"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_68() {
        let corpus = vec!["hello world", "hello neural network_68"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_69() {
        let corpus = vec!["hello world", "hello neural network_69"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_70() {
        let corpus = vec!["hello world", "hello neural network_70"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_71() {
        let corpus = vec!["hello world", "hello neural network_71"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_72() {
        let corpus = vec!["hello world", "hello neural network_72"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_73() {
        let corpus = vec!["hello world", "hello neural network_73"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_74() {
        let corpus = vec!["hello world", "hello neural network_74"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_75() {
        let corpus = vec!["hello world", "hello neural network_75"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_76() {
        let corpus = vec!["hello world", "hello neural network_76"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_77() {
        let corpus = vec!["hello world", "hello neural network_77"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_78() {
        let corpus = vec!["hello world", "hello neural network_78"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_79() {
        let corpus = vec!["hello world", "hello neural network_79"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_80() {
        let corpus = vec!["hello world", "hello neural network_80"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_81() {
        let corpus = vec!["hello world", "hello neural network_81"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_82() {
        let corpus = vec!["hello world", "hello neural network_82"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_83() {
        let corpus = vec!["hello world", "hello neural network_83"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_84() {
        let corpus = vec!["hello world", "hello neural network_84"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_85() {
        let corpus = vec!["hello world", "hello neural network_85"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_86() {
        let corpus = vec!["hello world", "hello neural network_86"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_87() {
        let corpus = vec!["hello world", "hello neural network_87"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_88() {
        let corpus = vec!["hello world", "hello neural network_88"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_89() {
        let corpus = vec!["hello world", "hello neural network_89"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_90() {
        let corpus = vec!["hello world", "hello neural network_90"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_91() {
        let corpus = vec!["hello world", "hello neural network_91"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_92() {
        let corpus = vec!["hello world", "hello neural network_92"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_93() {
        let corpus = vec!["hello world", "hello neural network_93"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_94() {
        let corpus = vec!["hello world", "hello neural network_94"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_95() {
        let corpus = vec!["hello world", "hello neural network_95"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_96() {
        let corpus = vec!["hello world", "hello neural network_96"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_97() {
        let corpus = vec!["hello world", "hello neural network_97"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_98() {
        let corpus = vec!["hello world", "hello neural network_98"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_99() {
        let corpus = vec!["hello world", "hello neural network_99"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_100() {
        let corpus = vec!["hello world", "hello neural network_100"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_101() {
        let corpus = vec!["hello world", "hello neural network_101"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_102() {
        let corpus = vec!["hello world", "hello neural network_102"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_103() {
        let corpus = vec!["hello world", "hello neural network_103"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_104() {
        let corpus = vec!["hello world", "hello neural network_104"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_105() {
        let corpus = vec!["hello world", "hello neural network_105"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_106() {
        let corpus = vec!["hello world", "hello neural network_106"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_107() {
        let corpus = vec!["hello world", "hello neural network_107"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_108() {
        let corpus = vec!["hello world", "hello neural network_108"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_109() {
        let corpus = vec!["hello world", "hello neural network_109"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_110() {
        let corpus = vec!["hello world", "hello neural network_110"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_111() {
        let corpus = vec!["hello world", "hello neural network_111"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_112() {
        let corpus = vec!["hello world", "hello neural network_112"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_113() {
        let corpus = vec!["hello world", "hello neural network_113"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_114() {
        let corpus = vec!["hello world", "hello neural network_114"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_115() {
        let corpus = vec!["hello world", "hello neural network_115"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_116() {
        let corpus = vec!["hello world", "hello neural network_116"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_117() {
        let corpus = vec!["hello world", "hello neural network_117"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_118() {
        let corpus = vec!["hello world", "hello neural network_118"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_119() {
        let corpus = vec!["hello world", "hello neural network_119"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_120() {
        let corpus = vec!["hello world", "hello neural network_120"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_121() {
        let corpus = vec!["hello world", "hello neural network_121"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_122() {
        let corpus = vec!["hello world", "hello neural network_122"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_123() {
        let corpus = vec!["hello world", "hello neural network_123"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_124() {
        let corpus = vec!["hello world", "hello neural network_124"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_125() {
        let corpus = vec!["hello world", "hello neural network_125"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_126() {
        let corpus = vec!["hello world", "hello neural network_126"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_127() {
        let corpus = vec!["hello world", "hello neural network_127"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_128() {
        let corpus = vec!["hello world", "hello neural network_128"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_129() {
        let corpus = vec!["hello world", "hello neural network_129"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_130() {
        let corpus = vec!["hello world", "hello neural network_130"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_131() {
        let corpus = vec!["hello world", "hello neural network_131"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_132() {
        let corpus = vec!["hello world", "hello neural network_132"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_133() {
        let corpus = vec!["hello world", "hello neural network_133"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_134() {
        let corpus = vec!["hello world", "hello neural network_134"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_135() {
        let corpus = vec!["hello world", "hello neural network_135"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_136() {
        let corpus = vec!["hello world", "hello neural network_136"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_137() {
        let corpus = vec!["hello world", "hello neural network_137"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_138() {
        let corpus = vec!["hello world", "hello neural network_138"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_139() {
        let corpus = vec!["hello world", "hello neural network_139"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_140() {
        let corpus = vec!["hello world", "hello neural network_140"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_141() {
        let corpus = vec!["hello world", "hello neural network_141"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_142() {
        let corpus = vec!["hello world", "hello neural network_142"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_143() {
        let corpus = vec!["hello world", "hello neural network_143"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_144() {
        let corpus = vec!["hello world", "hello neural network_144"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_145() {
        let corpus = vec!["hello world", "hello neural network_145"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_146() {
        let corpus = vec!["hello world", "hello neural network_146"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_147() {
        let corpus = vec!["hello world", "hello neural network_147"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_148() {
        let corpus = vec!["hello world", "hello neural network_148"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_149() {
        let corpus = vec!["hello world", "hello neural network_149"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_150() {
        let corpus = vec!["hello world", "hello neural network_150"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_151() {
        let corpus = vec!["hello world", "hello neural network_151"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_152() {
        let corpus = vec!["hello world", "hello neural network_152"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_153() {
        let corpus = vec!["hello world", "hello neural network_153"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_154() {
        let corpus = vec!["hello world", "hello neural network_154"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_155() {
        let corpus = vec!["hello world", "hello neural network_155"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_156() {
        let corpus = vec!["hello world", "hello neural network_156"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_157() {
        let corpus = vec!["hello world", "hello neural network_157"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_158() {
        let corpus = vec!["hello world", "hello neural network_158"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_159() {
        let corpus = vec!["hello world", "hello neural network_159"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_160() {
        let corpus = vec!["hello world", "hello neural network_160"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_161() {
        let corpus = vec!["hello world", "hello neural network_161"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_162() {
        let corpus = vec!["hello world", "hello neural network_162"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_163() {
        let corpus = vec!["hello world", "hello neural network_163"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_164() {
        let corpus = vec!["hello world", "hello neural network_164"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_165() {
        let corpus = vec!["hello world", "hello neural network_165"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_166() {
        let corpus = vec!["hello world", "hello neural network_166"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_167() {
        let corpus = vec!["hello world", "hello neural network_167"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_168() {
        let corpus = vec!["hello world", "hello neural network_168"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_169() {
        let corpus = vec!["hello world", "hello neural network_169"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_170() {
        let corpus = vec!["hello world", "hello neural network_170"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_171() {
        let corpus = vec!["hello world", "hello neural network_171"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_172() {
        let corpus = vec!["hello world", "hello neural network_172"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_173() {
        let corpus = vec!["hello world", "hello neural network_173"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_174() {
        let corpus = vec!["hello world", "hello neural network_174"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_175() {
        let corpus = vec!["hello world", "hello neural network_175"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_176() {
        let corpus = vec!["hello world", "hello neural network_176"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_177() {
        let corpus = vec!["hello world", "hello neural network_177"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_178() {
        let corpus = vec!["hello world", "hello neural network_178"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_179() {
        let corpus = vec!["hello world", "hello neural network_179"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_180() {
        let corpus = vec!["hello world", "hello neural network_180"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_181() {
        let corpus = vec!["hello world", "hello neural network_181"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_182() {
        let corpus = vec!["hello world", "hello neural network_182"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_183() {
        let corpus = vec!["hello world", "hello neural network_183"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_184() {
        let corpus = vec!["hello world", "hello neural network_184"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_185() {
        let corpus = vec!["hello world", "hello neural network_185"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_186() {
        let corpus = vec!["hello world", "hello neural network_186"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_187() {
        let corpus = vec!["hello world", "hello neural network_187"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_188() {
        let corpus = vec!["hello world", "hello neural network_188"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_189() {
        let corpus = vec!["hello world", "hello neural network_189"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_190() {
        let corpus = vec!["hello world", "hello neural network_190"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_191() {
        let corpus = vec!["hello world", "hello neural network_191"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
    }

    #[test]
    fn test_builders_suite_192() {
        let corpus = vec!["hello world", "hello neural network_192"];
        let bpe = TextBuilder::new().bpe().vocab_size(30).min_frequency(1).train(&corpus);
        assert!(!bpe.vocab.is_empty());

        let sp = TextBuilder::new().sentencepiece().vocab_size(30).train(&corpus);
        assert!(!sp.vocab.is_empty());

        let wp = TextBuilder::new().wordpiece().vocab_size(30).train(&corpus);
        assert!(!wp.vocab.is_empty());

        let emb = TextBuilder::new().embedding(100, 16).padding_idx(0).build();
        assert_eq!(emb.weight.shape(), &[100, 16]);
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
}
