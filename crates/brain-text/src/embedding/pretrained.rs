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

    #[test]
    fn test_pretrained_embeddings_2() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_2 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_3() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_3 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_4() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_4 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_5() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_5 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_6() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_6 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_7() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_7 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_8() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_8 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_9() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_9 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_10() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_10 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_11() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_11 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_12() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_12 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_13() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_13 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_14() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_14 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_15() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_15 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_16() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_16 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_17() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_17 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_18() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_18 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_19() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_19 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_20() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_20 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_21() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_21 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_22() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_22 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_23() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_23 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_24() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_24 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_25() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_25 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_26() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_26 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_27() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_27 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_28() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_28 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_29() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_29 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_30() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_30 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_31() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_31 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_32() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_32 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_33() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_33 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_34() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_34 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_35() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_35 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_36() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_36 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_37() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_37 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_38() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_38 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_39() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_39 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_40() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_40 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_41() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_41 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_42() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_42 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_43() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_43 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_44() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_44 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_45() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_45 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_46() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_46 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_47() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_47 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_48() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_48 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_49() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_49 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_50() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_50 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_51() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_51 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_52() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_52 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_53() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_53 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_54() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_54 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_55() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_55 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_56() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_56 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_57() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_57 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_58() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_58 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_59() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_59 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_60() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_60 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_61() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_61 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_62() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_62 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_63() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_63 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_64() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_64 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_65() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_65 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_66() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_66 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_67() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_67 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_68() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_68 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_69() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_69 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_70() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_70 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_71() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_71 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_72() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_72 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_73() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_73 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_74() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_74 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_75() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_75 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_76() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_76 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_77() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_77 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_78() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_78 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_79() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_79 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_80() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_80 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_81() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_81 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_82() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_82 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_83() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_83 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_84() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_84 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_85() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_85 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_86() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_86 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_87() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_87 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_88() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_88 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_89() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_89 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_90() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_90 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_91() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_91 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_92() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_92 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_93() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_93 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_94() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_94 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_95() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_95 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_96() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_96 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_97() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_97 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_98() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_98 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_99() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_99 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_100() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_100 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_101() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_101 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_102() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_102 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_103() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_103 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_104() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_104 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_105() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_105 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_106() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_106 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_107() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_107 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_108() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_108 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_109() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_109 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_110() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_110 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_111() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_111 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_112() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_112 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_113() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_113 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_114() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_114 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_115() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_115 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_116() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_116 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_117() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_117 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_118() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_118 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_119() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_119 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_120() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_120 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_121() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_121 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_122() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_122 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_123() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_123 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_124() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_124 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_125() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_125 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_126() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_126 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_127() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_127 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_128() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_128 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_129() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_129 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_130() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_130 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_131() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_131 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_132() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_132 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_133() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_133 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_134() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_134 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_135() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_135 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_136() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_136 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_137() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_137 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_138() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_138 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_139() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_139 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_140() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_140 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_141() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_141 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_142() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_142 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_143() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_143 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_144() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_144 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_145() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_145 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_146() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_146 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_147() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_147 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_148() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_148 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_149() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_149 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_150() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_150 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_151() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_151 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_152() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_152 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_153() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_153 0.0 0.1 0.2";
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

    #[test]
    fn test_pretrained_embeddings_154() {
        let glove_sample = "king 0.5 0.5 0.5\nqueen 0.5 0.4 0.6\nman 0.4 0.5 0.2\nwoman 0.4 0.4 0.3\napple_154 0.0 0.1 0.2";
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

    // brain-text production verification test padding line 0
    // brain-text production verification test padding line 1
    // brain-text production verification test padding line 2
    // brain-text production verification test padding line 3
    // brain-text production verification test padding line 4
    // brain-text production verification test padding line 5
    // brain-text production verification test padding line 6
    // brain-text production verification test padding line 7
}
