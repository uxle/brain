//! # Brain-Text: Production-Grade Natural Language Processing Framework
//!
//! High-performance tokenization algorithms (BPE, SentencePiece, WordPiece, Char, Word),
//! trainable & pretrained embeddings (Learned, Sinusoidal, GloVe, Word2Vec, FastText),
//! text feature extractors (TF-IDF, BM25, Bag-of-Words), similarity metrics, and LM preprocessing utilities.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

pub mod analyze;
pub mod builder;
pub mod compute;
pub mod config;
pub mod core;
pub mod embedding;
pub mod features;
pub mod helper;
pub mod r#impl;
pub mod lm;
pub mod ops;
pub mod optimize;
pub mod process;
pub mod similarity;
pub mod text_ops;
pub mod tokenizer;
pub mod transform;
pub mod utils;
pub mod vocab;

// Flat Re-exports of Key Core Structures
pub use analyze::{analyze_text, flesch_reading_ease, flesch_kincaid_grade, gunning_fog_index, TextStats};
pub use builder::TextBuilder;
pub use compute::{bleu_score, corpus_bleu, rouge_n, rouge_l, chrf_score, perplexity, word_error_rate};
pub use config::{EmbeddingConfig, ProcessConfig, SpecialTokensConfig, TextConfig, TokenizerConfig, TokenizerType};
pub use core::{TextBatch, TextError, TextResult, TokenId, TokenIds, TokenMeta, TokenizedOutput, VocabSize};
pub use embedding::fasttext::{FastTextConfig, FastTextEmbedding};
pub use embedding::pretrained::{PretrainedConfig, PretrainedEmbedding};
pub use embedding::{PositionalEmbedding, WordEmbedding};
pub use features::{BagOfWords, Bm25, FeatureNorm, HashingVectorizer, TextFeatureConfig, TfIdf};
pub use helper::{DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, SpanCorruptionHelper, TextAugmenter};
pub use r#impl::{build_embedding_layer, compute_similarity, decode_ids, encode_text, extract_top_keywords_tfidf, train_bpe_tokenizer, train_unigram_tokenizer};
pub use lm::{LmConfig, LmPreprocessor};
pub use ops::{create_attention_mask, create_position_ids, create_token_type_ids, ids_to_tokens, mask_tokens, pack_sequences, pad_sequences, tokens_to_ids, truncate_sequences};
pub use optimize::{prune_vocab, OptimizeConfig, VocabTrie};
pub use process::{clean_text, filter_by_length, pad_and_collate, process_batch_texts, split_into_paragraphs, split_into_sentences, truncate_batch};
pub use similarity::{SimilarityConfig, SimilarityMetric, TextSimilarity};
pub use text_ops::{character_ngrams, collocations, ngrams, shingles, term_frequencies, text_entropy, word_counts};
pub use tokenizer::bpe::{BpeConfig, BpeTokenizer};
pub use tokenizer::bytelevel::{ByteLevelConfig, ByteLevelEncoder};
pub use tokenizer::char::{CharConfig, CharTokenizer, WordConfig, WordTokenizer};
pub use tokenizer::normalizer::{Normalizer, NormalizerConfig};
pub use tokenizer::post::{PostConfig, PostProcessor, TruncationStrategy};
pub use tokenizer::pretokenizer::{PreTokenConfig, PreTokenizer};
pub use tokenizer::sentencepiece::{SentencePieceTokenizer, SpConfig};
pub use tokenizer::trainer::{BpeTrainer, TrainConfig, UnigramTrainer, WordPieceTrainer};
pub use tokenizer::wordpiece::{WordPieceConfig, WordPieceTokenizer};
pub use tokenizer::{Tokenizer, TokenizerError};
pub use transform::{case_transform, censor_words, normalize_punctuation, replace_patterns, transliterate_ascii, CaseKind, TextPipeline};
pub use utils::{byte_decode, byte_encode, damerau_levenshtein_distance, jaccard_similarity, levenshtein_distance, split_ws, unicode_helpers, TextRng};
pub use vocab::{SpecialKind, Vocab, VocabBuilder, VocabConfig};

/// Framework version constant.
pub const VERSION: &str = "0.2.0";

/// Convenient prelude module for single-line imports.
pub mod prelude {
    pub use crate::analyze::*;
    pub use crate::builder::*;
    pub use crate::compute::*;
    pub use crate::config::*;
    pub use crate::core::*;
    pub use crate::embedding::fasttext::*;
    pub use crate::embedding::pretrained::*;
    pub use crate::embedding::*;
    pub use crate::features::*;
    pub use crate::helper::*;
    pub use crate::lm::*;
    pub use crate::ops::*;
    pub use crate::optimize::*;
    pub use crate::process::*;
    pub use crate::r#impl::*;
    pub use crate::similarity::*;
    pub use crate::text_ops::*;
    pub use crate::tokenizer::bpe::*;
    pub use crate::tokenizer::bytelevel::*;
    pub use crate::tokenizer::char::*;
    pub use crate::tokenizer::normalizer::*;
    pub use crate::tokenizer::post::*;
    pub use crate::tokenizer::pretokenizer::*;
    pub use crate::tokenizer::sentencepiece::*;
    pub use crate::tokenizer::trainer::*;
    pub use crate::tokenizer::wordpiece::*;
    pub use crate::tokenizer::*;
    pub use crate::transform::*;
    pub use crate::utils::*;
    pub use crate::vocab::*;
    pub use crate::VERSION;
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
    fn test_crate_prelude_and_exports_1() {
        use crate::prelude::*;
        assert_eq!(VERSION, "0.2.0");
        let cfg = TextConfig::default();
        assert!(cfg.validate().is_ok());

        let mut vocab = Vocab::new();
        vocab.insert("brain_1");
        assert!(vocab.contains("brain_1"));
    }
}
