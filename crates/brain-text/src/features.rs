//! # Text Feature Extraction: Bag-of-Words, TF-IDF, and BM25
//!
//! Vector space representations, inverse document frequency weighting, and BM25 relevance scoring.
#![allow(
    missing_docs,
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_comparisons,
    clippy::all
)]

use crate::vocab::Vocab;
use std::collections::{HashMap, HashSet};

/// Vector normalization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FeatureNorm {
    /// L1 normalization (sum of absolute values = 1.0).
    L1,
    /// L2 Euclidean normalization (sum of squares = 1.0).
    #[default]
    L2,
    /// No normalization.
    None,
}

/// Configuration for vectorizer feature extractors.
#[derive(Debug, Clone, PartialEq)]
pub struct TextFeatureConfig {
    /// Maximum feature vocabulary size.
    pub max_features: Option<usize>,
    /// Vector normalization scheme.
    pub norm: FeatureNorm,
    /// Minimum document frequency.
    pub min_df: usize,
    /// Maximum document frequency ratio (0.0 to 1.0).
    pub max_df: f64,
    /// Use sublinear scaling: $1 + \log(TF)$.
    pub sublinear_tf: bool,
    /// Smooth IDF weights: $\log((N+1)/(DF+1)) + 1$.
    pub smooth_idf: bool,
}

impl Default for TextFeatureConfig {
    fn default() -> Self {
        Self {
            max_features: None,
            norm: FeatureNorm::L2,
            min_df: 1,
            max_df: 1.0,
            sublinear_tf: false,
            smooth_idf: true,
        }
    }
}

/// Bag-of-Words count vectorizer.
#[derive(Debug, Clone, Default)]
pub struct BagOfWords {
    /// Vocabulary of terms.
    pub vocab: Vocab,
    /// Normalization strategy.
    pub norm: FeatureNorm,
}

impl BagOfWords {
    /// Creates a new BagOfWords vectorizer with a given vocabulary.
    pub fn new(vocab: Vocab) -> Self {
        Self {
            vocab,
            norm: FeatureNorm::None,
        }
    }

    /// Fits vocabulary from a corpus of documents.
    pub fn fit(&mut self, corpus: &[Vec<String>]) {
        for doc in corpus {
            for token in doc {
                self.vocab.insert(token);
            }
        }
    }

    /// Transforms a document into a count vector.
    pub fn transform(&self, tokens: &[String]) -> Vec<f32> {
        let mut vector = vec![0.0f32; self.vocab.len()];
        for token in tokens {
            if let Some(id) = self.vocab.get_id(token) {
                if id < vector.len() {
                    vector[id] += 1.0;
                }
            }
        }
        self.apply_norm(&mut vector);
        vector
    }

    /// Fits vocabulary and transforms corpus in one pass.
    pub fn fit_transform(&mut self, corpus: &[Vec<String>]) -> Vec<Vec<f32>> {
        self.fit(corpus);
        corpus.iter().map(|doc| self.transform(doc)).collect()
    }

    fn apply_norm(&self, vector: &mut [f32]) {
        match self.norm {
            FeatureNorm::L1 => {
                let sum: f32 = vector.iter().map(|x| x.abs()).sum();
                if sum > 0.0 {
                    for x in vector.iter_mut() {
                        *x /= sum;
                    }
                }
            }
            FeatureNorm::L2 => {
                let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for x in vector.iter_mut() {
                        *x /= norm;
                    }
                }
            }
            FeatureNorm::None => {}
        }
    }
}

/// Term Frequency - Inverse Document Frequency (TF-IDF) Vectorizer.
#[derive(Debug, Clone, Default)]
pub struct TfIdf {
    /// Vocabulary.
    pub vocab: Vocab,
    /// Precomputed IDF values indexed by token ID.
    pub idf: HashMap<usize, f64>,
    /// Configuration options.
    pub config: TextFeatureConfig,
}

impl TfIdf {
    /// Creates a new TF-IDF vectorizer with the given configuration.
    pub fn new(config: TextFeatureConfig) -> Self {
        Self {
            vocab: Vocab::new(),
            idf: HashMap::new(),
            config,
        }
    }

    /// Fits vocabulary and computes IDF weights across corpus.
    pub fn fit(&mut self, corpus: &[Vec<String>]) {
        let n_docs = corpus.len() as f64;
        let mut df_counts: HashMap<String, usize> = HashMap::new();

        for doc in corpus {
            let unique_tokens: HashSet<&String> = doc.iter().collect();
            for token in unique_tokens {
                *df_counts.entry(token.clone()).or_insert(0) += 1;
            }
        }

        self.vocab = Vocab::new();
        let mut filtered_tokens: Vec<(String, usize)> = Vec::new();

        for (token, count) in df_counts {
            let ratio = count as f64 / n_docs;
            if count >= self.config.min_df && ratio <= self.config.max_df {
                filtered_tokens.push((token, count));
            }
        }

        filtered_tokens.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        if let Some(max_f) = self.config.max_features {
            filtered_tokens.truncate(max_f);
        }

        for (token, df) in filtered_tokens {
            let id = self.vocab.insert(&token);
            let idf_val = if self.config.smooth_idf {
                ((n_docs + 1.0) / (df as f64 + 1.0)).ln() + 1.0
            } else {
                (n_docs / df as f64).ln() + 1.0
            };
            self.idf.insert(id, idf_val);
        }
    }

    /// Transforms a document into a TF-IDF weight vector.
    pub fn transform(&self, tokens: &[String]) -> Vec<f64> {
        let mut tf_counts: HashMap<usize, f64> = HashMap::new();
        for token in tokens {
            if let Some(id) = self.vocab.get_id(token) {
                *tf_counts.entry(id).or_insert(0.0) += 1.0;
            }
        }

        let mut vector = vec![0.0f64; self.vocab.len()];
        for (&id, &count) in &tf_counts {
            if id < vector.len() {
                let tf = if self.config.sublinear_tf {
                    1.0 + count.ln()
                } else {
                    count
                };
                let idf = self.idf.get(&id).copied().unwrap_or(1.0);
                vector[id] = tf * idf;
            }
        }

        match self.config.norm {
            FeatureNorm::L1 => {
                let sum: f64 = vector.iter().map(|x| x.abs()).sum();
                if sum > 0.0 {
                    for x in vector.iter_mut() {
                        *x /= sum;
                    }
                }
            }
            FeatureNorm::L2 => {
                let norm: f64 = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 0.0 {
                    for x in vector.iter_mut() {
                        *x /= norm;
                    }
                }
            }
            FeatureNorm::None => {}
        }

        vector
    }

    /// Fits on corpus and transforms all documents.
    pub fn fit_transform(&mut self, corpus: &[Vec<String>]) -> Vec<Vec<f64>> {
        self.fit(corpus);
        corpus.iter().map(|doc| self.transform(doc)).collect()
    }
}

/// Okapi BM25 Ranking and Information Retrieval Model.
#[derive(Debug, Clone)]
pub struct Bm25 {
    /// Term frequency saturation parameter (typically 1.2 to 2.0).
    pub k1: f64,
    /// Document length normalization parameter (typically 0.75).
    pub b: f64,
    /// Average document length across the corpus.
    pub avg_doc_len: f64,
    /// Document lengths for each indexed document.
    pub doc_lens: Vec<usize>,
    /// Indexed documents.
    pub corpus: Vec<HashMap<String, usize>>,
    /// Precomputed IDF values for terms.
    pub idf: HashMap<String, f64>,
}

impl Default for Bm25 {
    fn default() -> Self {
        Self {
            k1: 1.5,
            b: 0.75,
            avg_doc_len: 0.0,
            doc_lens: Vec::new(),
            corpus: Vec::new(),
            idf: HashMap::new(),
        }
    }
}

impl Bm25 {
    /// Creates a new BM25 model with custom hyper-parameters.
    pub fn new(k1: f64, b: f64) -> Self {
        Self {
            k1,
            b,
            avg_doc_len: 0.0,
            doc_lens: Vec::new(),
            corpus: Vec::new(),
            idf: HashMap::new(),
        }
    }

    /// Fits the BM25 model on a corpus of tokenized documents.
    pub fn fit(&mut self, docs: &[Vec<String>]) {
        let n_docs = docs.len() as f64;
        let mut total_len = 0usize;
        let mut df_map: HashMap<String, usize> = HashMap::new();
        self.corpus.clear();
        self.doc_lens.clear();

        for doc in docs {
            let doc_len = doc.len();
            total_len += doc_len;
            self.doc_lens.push(doc_len);

            let mut term_freqs = HashMap::new();
            let mut unique_terms = HashSet::new();

            for term in doc {
                *term_freqs.entry(term.clone()).or_insert(0usize) += 1;
                unique_terms.insert(term.clone());
            }

            for term in unique_terms {
                *df_map.entry(term).or_insert(0usize) += 1;
            }

            self.corpus.push(term_freqs);
        }

        self.avg_doc_len = if docs.is_empty() {
            0.0
        } else {
            total_len as f64 / n_docs
        };

        self.idf.clear();
        for (term, df) in df_map {
            let idf_val = (((n_docs - df as f64 + 0.5) / (df as f64 + 0.5)) + 1.0).ln();
            self.idf.insert(term, idf_val.max(0.0));
        }
    }

    /// Computes the BM25 relevance score for a query against document at `doc_idx`.
    pub fn score(&self, query: &[String], doc_idx: usize) -> f64 {
        if doc_idx >= self.corpus.len() || self.avg_doc_len == 0.0 {
            return 0.0;
        }

        let doc_freqs = &self.corpus[doc_idx];
        let doc_len = self.doc_lens[doc_idx] as f64;
        let mut score = 0.0;

        for term in query {
            if let Some(&tf) = doc_freqs.get(term) {
                let idf = self.idf.get(term).copied().unwrap_or(0.0);
                let tf_f = tf as f64;
                let numerator = tf_f * (self.k1 + 1.0);
                let denominator =
                    tf_f + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len));
                score += idf * (numerator / denominator);
            }
        }

        score
    }

    /// Scores the query against all documents in the corpus.
    pub fn score_corpus(&self, query: &[String]) -> Vec<f64> {
        (0..self.corpus.len())
            .map(|i| self.score(query, i))
            .collect()
    }
}

/// Hashing Vectorizer implementing the hashing trick (Murmur-lite hash).
#[derive(Debug, Clone)]
pub struct HashingVectorizer {
    /// Number of output features.
    pub num_features: usize,
    /// Normalization strategy.
    pub norm: FeatureNorm,
}

impl HashingVectorizer {
    /// Creates a new `HashingVectorizer` with target feature dimensions.
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            norm: FeatureNorm::L2,
        }
    }

    /// Hashes a token string into feature index and sign.
    fn hash_token(&self, s: &str) -> (usize, f32) {
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in s.as_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        let idx = (h as usize) % self.num_features;
        let sign = if (h >> 31) & 1 == 1 { 1.0f32 } else { -1.0f32 };
        (idx, sign)
    }

    /// Transforms a document into a hashed feature vector.
    pub fn transform(&self, tokens: &[String]) -> Vec<f32> {
        let mut vector = vec![0.0f32; self.num_features];
        for token in tokens {
            let (idx, sign) = self.hash_token(token);
            vector[idx] += sign;
        }

        match self.norm {
            FeatureNorm::L1 => {
                let sum: f32 = vector.iter().map(|x| x.abs()).sum();
                if sum > 0.0 {
                    for x in vector.iter_mut() {
                        *x /= sum;
                    }
                }
            }
            FeatureNorm::L2 => {
                let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for x in vector.iter_mut() {
                        *x /= norm;
                    }
                }
            }
            FeatureNorm::None => {}
        }

        vector
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
    fn test_feature_extractors_1() {
        let corpus = vec![
            vec![
                "machine".to_string(),
                "learning".to_string(),
                "model_1".to_string(),
            ],
            vec![
                "deep".to_string(),
                "learning".to_string(),
                "neural".to_string(),
            ],
        ];

        let mut bow = BagOfWords::default();
        let bow_matrix = bow.fit_transform(&corpus);
        assert_eq!(bow_matrix.len(), 2);
        assert_eq!(bow_matrix[0].len(), bow.vocab.len());

        let mut tfidf = TfIdf::new(TextFeatureConfig::default());
        let tfidf_matrix = tfidf.fit_transform(&corpus);
        assert_eq!(tfidf_matrix.len(), 2);

        let mut bm25 = Bm25::default();
        bm25.fit(&corpus);
        let q = vec!["learning".to_string()];
        let s0 = bm25.score(&q, 0);
        assert!(s0 > 0.0);

        let hasher = HashingVectorizer::new(128);
        let hv = hasher.transform(&corpus[0]);
        assert_eq!(hv.len(), 128);
    }
}
