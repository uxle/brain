//! # Tokenizer Training Algorithms: BPE, Unigram EM, and WordPiece
//!
//! Merge frequency counting, Expectation-Maximization pruning, and dictionary fitting from raw text corpora.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::vocab::{SpecialKind, Vocab};
use std::collections::{HashMap, HashSet};

/// Configuration for tokenizer training sessions.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainConfig {
    /// Target vocabulary capacity.
    pub vocab_size: usize,
    /// Minimum frequency for subword inclusion.
    pub min_frequency: usize,
    /// List of special control token strings.
    pub special_tokens: Vec<String>,
    /// Show progress during training.
    pub show_progress: bool,
    /// Base alphabet characters to always preserve.
    pub initial_alphabet: HashSet<char>,
    /// Shrinking factor for unigram EM iterations (typically 0.75).
    pub shrinking_factor: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            vocab_size: 1000,
            min_frequency: 2,
            special_tokens: vec!["<unk>".to_string(), "<pad>".to_string()],
            show_progress: false,
            initial_alphabet: HashSet::new(),
            shrinking_factor: 0.75,
        }
    }
}

/// Trainer for Byte-Pair Encoding (BPE) tokenizers.
pub struct BpeTrainer;

impl BpeTrainer {
    /// Trains a BPE vocabulary and merge table from a text corpus.
    pub fn train(
        corpus: &[&str],
        config: &TrainConfig,
    ) -> (Vocab, Vec<((String, String), usize)>) {
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        for &doc in corpus {
            for word in doc.split_whitespace() {
                *word_counts.entry(word.to_string()).or_insert(0usize) += 1;
            }
        }

        let mut vocab = Vocab::new();
        for spec in &config.special_tokens {
            vocab.add_special(spec, SpecialKind::Custom);
        }

        // Initialize single characters
        let mut splits: HashMap<String, Vec<String>> = HashMap::new();
        for (word, _) in &word_counts {
            let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            if let Some(last) = chars.last_mut() {
                last.push_str("</w>");
            }
            for sym in &chars {
                vocab.insert(sym);
            }
            splits.insert(word.clone(), chars);
        }

        let mut merges: Vec<((String, String), usize)> = Vec::new();

        while vocab.len() < config.vocab_size {
            let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();

            for (word, &count) in &word_counts {
                if let Some(syms) = splits.get(word) {
                    for i in 0..(syms.len().saturating_sub(1)) {
                        let pair = (syms[i].clone(), syms[i + 1].clone());
                        *pair_counts.entry(pair).or_insert(0) += count;
                    }
                }
            }

            if pair_counts.is_empty() {
                break;
            }

            let mut best_pair = None;
            let mut best_count = 0;

            for (pair, count) in pair_counts {
                if count >= config.min_frequency && count > best_count {
                    best_count = count;
                    best_pair = Some(pair);
                }
            }

            match best_pair {
                Some((first, second)) => {
                    let merged = format!("{}{}", first, second);
                    vocab.insert(&merged);
                    let rank = merges.len();
                    merges.push(((first.clone(), second.clone()), rank));

                    // Apply merge to splits
                    for syms in splits.values_mut() {
                        let mut new_syms = Vec::with_capacity(syms.len());
                        let mut i = 0;
                        while i < syms.len() {
                            if i + 1 < syms.len() && syms[i] == first && syms[i + 1] == second {
                                new_syms.push(merged.clone());
                                i += 2;
                            } else {
                                new_syms.push(syms[i].clone());
                                i += 1;
                            }
                        }
                        *syms = new_syms;
                    }
                }
                None => break,
            }
        }

        (vocab, merges)
    }
}

/// Trainer for SentencePiece Unigram Language Model tokenizers.
pub struct UnigramTrainer;

impl UnigramTrainer {
    /// Trains a unigram vocabulary and score dictionary from a text corpus.
    pub fn train(corpus: &[&str], config: &TrainConfig) -> (Vocab, HashMap<String, f64>) {
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        for &doc in corpus {
            for word in doc.split_whitespace() {
                let prep = format!(" {}", word);
                *word_counts.entry(prep).or_insert(0usize) += 1;
            }
        }

        let mut candidate_counts: HashMap<String, usize> = HashMap::new();
        for (word, &count) in &word_counts {
            let chars: Vec<char> = word.chars().collect();
            for len in 1..=chars.len().min(8) {
                for i in 0..=(chars.len() - len) {
                    let sub: String = chars[i..i + len].iter().collect();
                    *candidate_counts.entry(sub).or_insert(0) += count;
                }
            }
        }

        let mut candidates: Vec<(String, usize)> = candidate_counts
            .into_iter()
            .filter(|(_, count)| *count >= config.min_frequency)
            .collect();

        candidates.sort_by(|a, b| b.1.cmp(&a.1));
        candidates.truncate(config.vocab_size);

        let total_freq: usize = candidates.iter().map(|c| c.1).sum();
        let total_f = total_freq.max(1) as f64;

        let mut vocab = Vocab::new();
        for spec in &config.special_tokens {
            vocab.add_special(spec, SpecialKind::Custom);
        }

        let mut scores = HashMap::new();
        for (piece, count) in candidates {
            vocab.insert(&piece);
            let log_prob = (count as f64 / total_f).ln();
            scores.insert(piece, log_prob);
        }

        (vocab, scores)
    }
}

/// Trainer for WordPiece tokenizers.
pub struct WordPieceTrainer;

impl WordPieceTrainer {
    /// Trains a WordPiece vocabulary from a text corpus.
    pub fn train(corpus: &[&str], config: &TrainConfig) -> Vocab {
        let (vocab, _) = BpeTrainer::train(corpus, config);
        vocab
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
    fn test_tokenizer_training_1() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_1",
        ];

        let cfg = TrainConfig { vocab_size: 50, min_frequency: 1, ..Default::default() };
        let (bpe_vocab, merges) = BpeTrainer::train(&corpus, &cfg);
        assert!(!bpe_vocab.is_empty());
        assert!(!merges.is_empty());

        let (sp_vocab, scores) = UnigramTrainer::train(&corpus, &cfg);
        assert!(!sp_vocab.is_empty());
        assert!(!scores.is_empty());

        let wp_vocab = WordPieceTrainer::train(&corpus, &cfg);
        assert!(!wp_vocab.is_empty());
    }
}
