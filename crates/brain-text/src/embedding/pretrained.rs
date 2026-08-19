//! # Pretrained Embedding Loaders: GloVe & Word2Vec
//!
//! High-throughput parsers for standard GloVe and Word2Vec formats with vector algebra and analogy solvers.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::core::{TextError, TextResult};
use crate::embedding::WordEmbedding;
use crate::similarity::TextSimilarity;
use crate::vocab::Vocab;

/// Initialization strategy for out-of-vocabulary terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnkInit {
    /// Zero initialize OOV vectors.
    #[default]
    Zero,
    /// Random Gaussian initialization.
    RandomNormal,
    /// Average of all known embedding vectors.
    Mean,
}

/// Configuration for loading pretrained word embeddings.
#[derive(Debug, Clone, PartialEq)]
pub struct PretrainedConfig {
    /// Expected vector dimension.
    pub embedding_dim: usize,
    /// Maximum vocabulary limit.
    pub max_vocab: Option<usize>,
    /// Normalize vectors to unit Euclidean length upon loading.
    pub normalize: bool,
    /// Out-of-vocabulary initialization strategy.
    pub unk_init: UnkInit,
}

impl Default for PretrainedConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 100,
            max_vocab: None,
            normalize: false,
            unk_init: UnkInit::Zero,
        }
    }
}

/// Static pretrained word embedding lookup table.
#[derive(Debug, Clone, Default)]
pub struct PretrainedEmbedding {
    /// Vocabulary mapping terms to indices.
    pub vocab: Vocab,
    /// Flat vector storage: `vectors[id]` contains float vector of length `dim`.
    pub vectors: Vec<Vec<f32>>,
    /// Embedding dimension.
    pub dim: usize,
}

impl PretrainedEmbedding {
    /// Loads pretrained word vectors from a GloVe format string (`word 0.1 0.2 ... 0.5\n`).
    pub fn load_glove_str(glove_text: &str, config: &PretrainedConfig) -> TextResult<Self> {
        let mut vocab = Vocab::new();
        let mut vectors = Vec::new();

        for line in glove_text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let mut parts = line.split_whitespace();
            let word = match parts.next() {
                Some(w) => w,
                None => continue,
            };

            let mut vec = Vec::with_capacity(config.embedding_dim);
            for p in parts {
                if let Ok(val) = p.parse::<f32>() {
                    vec.push(val);
                }
            }

            if vec.len() != config.embedding_dim {
                return Err(TextError::PretrainedLoadError(format!(
                    "Dimension mismatch for word '{}': expected {}, found {}",
                    word,
                    config.embedding_dim,
                    vec.len()
                )));
            }

            if config.normalize {
                let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for x in vec.iter_mut() {
                        *x /= norm;
                    }
                }
            }

            vocab.insert(word);
            vectors.push(vec);

            if let Some(max_v) = config.max_vocab {
                if vocab.len() >= max_v {
                    break;
                }
            }
        }

        Ok(Self {
            vocab,
            vectors,
            dim: config.embedding_dim,
        })
    }

    /// Loads pretrained word vectors from Word2Vec text format with header line (`N DIM\nword 0.1 ...`).
    pub fn load_word2vec_text_str(w2v_text: &str, config: &PretrainedConfig) -> TextResult<Self> {
        let mut lines = w2v_text.lines();
        let _header = lines.next().ok_or_else(|| {
            TextError::PretrainedLoadError("Empty Word2Vec file (missing header)".to_string())
        })?;

        let rest: String = lines.collect::<Vec<&str>>().join("\n");
        Self::load_glove_str(&rest, config)
    }

    /// Retrieves the embedding vector for a given word.
    pub fn get_vector(&self, word: &str) -> Option<&[f32]> {
        self.vocab.get_id(word).and_then(|id| {
            if id < self.vectors.len() {
                Some(self.vectors[id].as_slice())
            } else {
                None
            }
        })
    }

    /// Finds the top-k most similar words using cosine similarity.
    pub fn most_similar(&self, word: &str, top_k: usize) -> Vec<(String, f32)> {
        let target_vec = match self.get_vector(word) {
            Some(v) => v,
            None => return Vec::new(),
        };

        let mut similarities = Vec::with_capacity(self.vocab.len());

        for id in 0..self.vocab.len() {
            if let Some(token) = self.vocab.get_token(id) {
                if token != word {
                    let sim = TextSimilarity::cosine(target_vec, &self.vectors[id]);
                    similarities.push((token.to_string(), sim));
                }
            }
        }

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(top_k);
        similarities
    }

    /// Solves vector analogies: $v_{pos1} - v_{neg} + v_{pos2} \approx v_{result}$ (e.g. king - man + woman = queen).
    pub fn analogy(
        &self,
        pos1: &str,
        pos2: &str,
        neg: &str,
        top_k: usize,
    ) -> Vec<(String, f32)> {
        let v_pos1 = match self.get_vector(pos1) {
            Some(v) => v,
            None => return Vec::new(),
        };
        let v_pos2 = match self.get_vector(pos2) {
            Some(v) => v,
            None => return Vec::new(),
        };
        let v_neg = match self.get_vector(neg) {
            Some(v) => v,
            None => return Vec::new(),
        };

        let mut target_vec = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            target_vec[i] = v_pos1[i] - v_neg[i] + v_pos2[i];
        }

        let mut similarities = Vec::with_capacity(self.vocab.len());

        for id in 0..self.vocab.len() {
            if let Some(token) = self.vocab.get_token(id) {
                if token != pos1 && token != pos2 && token != neg {
                    let sim = TextSimilarity::cosine(&target_vec, &self.vectors[id]);
                    similarities.push((token.to_string(), sim));
                }
            }
        }

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(top_k);
        similarities
    }

    /// Converts loaded pretrained embeddings into a trainable `WordEmbedding` layer.
    pub fn to_word_embedding(&self) -> WordEmbedding {
        let mut flat = Vec::with_capacity(self.vocab.len() * self.dim);
        for v in &self.vectors {
            for &x in v {
                flat.push(x as f64);
            }
        }
        let tensor = brain_core::Tensor::from_vec(flat, vec![self.vocab.len(), self.dim]);
        let mut config = crate::config::EmbeddingConfig::default();
        config.vocab_size = self.vocab.len();
        config.embedding_dim = self.dim;

        WordEmbedding {
            weight: tensor,
            config,
        }
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
    fn test_pretrained_embeddings_1() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_1 0.0 0.1 0.2";
        let cfg = PretrainedConfig { embedding_dim: 3, normalize: true, ..Default::default() };
        let p_emb = PretrainedEmbedding::load_glove_str(glove_sample, &cfg).unwrap();
        assert_eq!(p_emb.vocab.len(), 5);

        let kv = p_emb.get_vector("king").unwrap();
        assert_eq!(kv.len(), 3);

        let most_sim = p_emb.most_similar("king", 2);
        assert!(!most_sim.is_empty());

        let anal = p_emb.analogy("king", "woman", "man", 1);
        assert!(!anal.is_empty());

        let layer = p_emb.to_word_embedding();
        assert_eq!(layer.weight.shape(), &[5, 3]);
    }
}
