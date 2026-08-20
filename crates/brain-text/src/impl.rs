//! # High-Level End-to-End NLP Execution Paths
//!
//! Convenient end-to-end APIs for encoding, decoding, training, embeddings, and similarity queries.
#![allow(
    missing_docs,
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_comparisons,
    clippy::all
)]

use crate::builder::TextBuilder;
use crate::core::{TextResult, TokenId, TokenizedOutput};
use crate::embedding::{PositionalEmbedding, WordEmbedding};
use crate::features::{FeatureNorm, TextFeatureConfig, TfIdf};
use crate::similarity::{SimilarityMetric, TextSimilarity};
use crate::tokenizer::bpe::BpeTokenizer;
use crate::tokenizer::sentencepiece::SentencePieceTokenizer;
use crate::tokenizer::Tokenizer;
use brain_core::Tensor;

/// Encodes raw text using any implementation of `Tokenizer`.
pub fn encode_text(text: &str, tokenizer: &dyn Tokenizer) -> TextResult<TokenizedOutput> {
    tokenizer.encode(text)
}

/// Decodes numeric token IDs into string text using any `Tokenizer`.
pub fn decode_ids(ids: &[TokenId], tokenizer: &dyn Tokenizer) -> TextResult<String> {
    tokenizer.decode(ids)
}

/// Trains a BPE tokenizer end-to-end from raw text slices.
pub fn train_bpe_tokenizer(
    corpus: &[&str],
    vocab_size: usize,
    _special_tokens: &[&str],
) -> TextResult<BpeTokenizer> {
    Ok(TextBuilder::new()
        .bpe()
        .vocab_size(vocab_size)
        .train(corpus))
}

/// Trains a SentencePiece unigram tokenizer end-to-end from raw text slices.
pub fn train_unigram_tokenizer(
    corpus: &[&str],
    vocab_size: usize,
    _special_tokens: &[&str],
) -> TextResult<SentencePieceTokenizer> {
    Ok(TextBuilder::new()
        .sentencepiece()
        .vocab_size(vocab_size)
        .train(corpus))
}

/// Builds a standard WordEmbedding lookup table.
pub fn build_embedding_layer(
    vocab_size: usize,
    dim: usize,
    pad_idx: Option<usize>,
) -> WordEmbedding {
    let mut builder = TextBuilder::new().embedding(vocab_size, dim);
    if let Some(p) = pad_idx {
        builder = builder.padding_idx(p);
    }
    builder.build()
}

/// Generates a standard sinusoidal positional embedding tensor.
pub fn create_sinusoidal_positional_embedding(seq_len: usize, dim: usize) -> Tensor {
    PositionalEmbedding::sinusoidal(seq_len, dim)
}

/// Computes similarity between two text strings using a selected metric.
pub fn compute_similarity(text1: &str, text2: &str, metric: SimilarityMetric) -> f64 {
    match metric {
        SimilarityMetric::Levenshtein => TextSimilarity::levenshtein_similarity(text1, text2),
        SimilarityMetric::DamerauLevenshtein => TextSimilarity::damerau_similarity(text1, text2),
        SimilarityMetric::Jaro => TextSimilarity::jaro(text1, text2),
        SimilarityMetric::JaroWinkler => TextSimilarity::jaro_winkler(text1, text2),
        SimilarityMetric::Jaccard => {
            let t1: Vec<String> = text1.split_whitespace().map(|s| s.to_string()).collect();
            let t2: Vec<String> = text2.split_whitespace().map(|s| s.to_string()).collect();
            TextSimilarity::jaccard(&t1, &t2)
        }
        SimilarityMetric::SorensenDice => {
            let t1: Vec<String> = text1.split_whitespace().map(|s| s.to_string()).collect();
            let t2: Vec<String> = text2.split_whitespace().map(|s| s.to_string()).collect();
            TextSimilarity::sorensen_dice(&t1, &t2)
        }
        _ => TextSimilarity::levenshtein_similarity(text1, text2),
    }
}

/// Extracts the top-k highest weighted keywords from a document in a corpus using TF-IDF.
pub fn extract_top_keywords_tfidf(
    docs: &[&str],
    doc_idx: usize,
    top_k: usize,
) -> Vec<(String, f64)> {
    if doc_idx >= docs.len() {
        return Vec::new();
    }

    let corpus_tokens: Vec<Vec<String>> = docs
        .iter()
        .map(|&d| d.split_whitespace().map(|w| w.to_lowercase()).collect())
        .collect();

    let cfg = TextFeatureConfig {
        norm: FeatureNorm::None,
        ..Default::default()
    };
    let mut tfidf = TfIdf::new(cfg);
    tfidf.fit(&corpus_tokens);

    let doc_weights = tfidf.transform(&corpus_tokens[doc_idx]);
    let mut scored: Vec<(String, f64)> = Vec::new();

    for id in 0..tfidf.vocab.len() {
        if id < doc_weights.len() && doc_weights[id] > 0.0 {
            if let Some(tok) = tfidf.vocab.get_token(id) {
                scored.push((tok.to_string(), doc_weights[id]));
            }
        }
    }

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);
    scored
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
        clippy::len_zero
    )]
    use super::*;
    use crate::analyze::*;
    use crate::builder::*;
    use crate::compute::*;
    use crate::config::*;
    use crate::core::*;
    use crate::embedding::fasttext::*;
    use crate::embedding::pretrained::*;
    use crate::embedding::*;
    use crate::features::*;
    use crate::helper::*;
    use crate::lm::*;
    use crate::ops::*;
    use crate::optimize::*;
    use crate::process::*;
    use crate::similarity::*;
    use crate::text_ops::*;
    use crate::tokenizer::bpe::*;
    use crate::tokenizer::bytelevel::*;
    use crate::tokenizer::char::*;
    use crate::tokenizer::normalizer::*;
    use crate::tokenizer::post::*;
    use crate::tokenizer::pretokenizer::*;
    use crate::tokenizer::sentencepiece::*;
    use crate::tokenizer::trainer::*;
    use crate::tokenizer::wordpiece::*;
    use crate::tokenizer::*;
    use crate::transform::*;
    use crate::utils::*;
    use crate::vocab::*;
    use crate::VERSION;
    use brain_core::Tensor;

    #[test]
    fn test_impl_end_to_end_1() {
        let corpus = vec!["deep learning models", "neural networks for nlp_1"];
        let bpe = train_bpe_tokenizer(&corpus, 30, &[]).unwrap();
        let enc = encode_text("deep neural", &bpe).unwrap();
        assert!(!enc.ids.is_empty());

        let dec = decode_ids(&enc.ids, &bpe).unwrap();
        assert!(!dec.is_empty());

        let emb = build_embedding_layer(50, 16, Some(0));
        assert_eq!(emb.weight.shape(), &[50, 16]);

        let pos = create_sinusoidal_positional_embedding(8, 16);
        assert_eq!(pos.shape(), &[8, 16]);

        let sim = compute_similarity("martha", "marhta", SimilarityMetric::DamerauLevenshtein);
        assert!(sim > 0.6);

        let keywords = extract_top_keywords_tfidf(&corpus, 0, 2);
        assert!(!keywords.is_empty());
    }
}
