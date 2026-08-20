//! # FastText Subword N-Gram Embeddings
//!
//! Subword character n-gram hashing for robust representation of out-of-vocabulary (OOV) and morphology-rich words.
#![allow(
    missing_docs,
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_comparisons,
    clippy::all
)]

use crate::similarity::TextSimilarity;
use std::collections::HashMap;

/// Configuration for FastText subword representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FastTextConfig {
    /// Minimum character n-gram length (default 3).
    pub min_n: usize,
    /// Maximum character n-gram length (default 6).
    pub max_n: usize,
    /// Number of hash buckets for subword n-grams (typically 2,000,000; default 10,000 for compact models).
    pub bucket_size: usize,
    /// Embedding vector dimension.
    pub dim: usize,
}

impl Default for FastTextConfig {
    fn default() -> Self {
        Self {
            min_n: 3,
            max_n: 6,
            bucket_size: 10000,
            dim: 100,
        }
    }
}

/// FastText Subword N-Gram Embedding Model.
#[derive(Debug, Clone)]
pub struct FastTextEmbedding {
    /// Exact full word embedding table.
    pub word_vectors: HashMap<String, Vec<f32>>,
    /// Hash bucket table for subword n-grams.
    pub ngram_buckets: Vec<Vec<f32>>,
    /// Configuration options.
    pub config: FastTextConfig,
}

impl FastTextEmbedding {
    /// Creates a new `FastTextEmbedding` model.
    pub fn new(config: FastTextConfig) -> Self {
        let buckets = vec![vec![0.0f32; config.dim]; config.bucket_size];
        Self {
            word_vectors: HashMap::new(),
            ngram_buckets: buckets,
            config,
        }
    }

    /// Computes FNV-1a 64-bit hash of an n-gram string mapped into bucket space.
    pub fn compute_ngram_hash(ngram: &str, bucket_size: usize) -> usize {
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in ngram.as_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        (h as usize) % bucket_size
    }

    /// Extracts all subword character n-grams of lengths `min_n..=max_n` bracketed with `<` and `>`.
    pub fn extract_subwords(&self, word: &str) -> Vec<String> {
        let wrapped = format!("<{}>", word);
        let chars: Vec<char> = wrapped.chars().collect();
        let mut ngrams = Vec::new();

        for n in self.config.min_n..=self.config.max_n {
            if chars.len() >= n {
                for i in 0..=(chars.len() - n) {
                    let gram: String = chars[i..i + n].iter().collect();
                    ngrams.push(gram);
                }
            }
        }

        ngrams
    }

    /// Inserts a trained full-word vector into the embedding dictionary.
    pub fn insert_word(&mut self, word: &str, vector: Vec<f32>) {
        let subwords = self.extract_subwords(word);
        let count = (subwords.len() + 1) as f32;
        let share = vector.iter().map(|&x| x / count).collect::<Vec<f32>>();

        // Accumulate subword shares into buckets
        for sub in subwords {
            let bucket = Self::compute_ngram_hash(&sub, self.config.bucket_size);
            for i in 0..self.config.dim {
                self.ngram_buckets[bucket][i] += share[i];
            }
        }

        self.word_vectors.insert(word.to_string(), vector);
    }

    /// Computes embedding vector for any word (handles known words and unseen OOV words smoothly).
    pub fn get_word_vector(&self, word: &str) -> Vec<f32> {
        let mut vector = vec![0.0f32; self.config.dim];
        let mut count = 0.0f32;

        if let Some(w_vec) = self.word_vectors.get(word) {
            for i in 0..self.config.dim {
                vector[i] += w_vec[i];
            }
            count += 1.0;
        }

        let subwords = self.extract_subwords(word);
        for sub in subwords {
            let bucket = Self::compute_ngram_hash(&sub, self.config.bucket_size);
            let b_vec = &self.ngram_buckets[bucket];
            for i in 0..self.config.dim {
                vector[i] += b_vec[i];
            }
            count += 1.0;
        }

        if count > 0.0 {
            for x in vector.iter_mut() {
                *x /= count;
            }
        }

        vector
    }

    /// Computes most similar candidate words to a given target word query.
    pub fn most_similar(
        &self,
        word: &str,
        candidates: &[&str],
        top_k: usize,
    ) -> Vec<(String, f32)> {
        let target_vec = self.get_word_vector(word);
        let mut scored = Vec::with_capacity(candidates.len());

        for &cand in candidates {
            if cand != word {
                let cand_vec = self.get_word_vector(cand);
                let sim = TextSimilarity::cosine(&target_vec, &cand_vec);
                scored.push((cand.to_string(), sim));
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
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
    fn test_fasttext_embeddings_1() {
        let cfg = FastTextConfig {
            dim: 8,
            bucket_size: 100,
            min_n: 3,
            max_n: 4,
        };
        let mut ft = FastTextEmbedding::new(cfg);
        ft.insert_word("apple", vec![0.1; 8]);
        ft.insert_word("application_1", vec![0.2; 8]);

        let v_known = ft.get_word_vector("apple");
        assert_eq!(v_known.len(), 8);

        // OOV word with shared subwords like "app"
        let v_oov = ft.get_word_vector("applesauce");
        assert_eq!(v_oov.len(), 8);

        let sim = ft.most_similar("apples", &["apple", "banana", "car"], 2);
        assert_eq!(sim.len(), 2);
    }
}
