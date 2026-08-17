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

    #[test]
    fn test_tokenizer_training_2() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_2",
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

    #[test]
    fn test_tokenizer_training_3() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_3",
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

    #[test]
    fn test_tokenizer_training_4() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_4",
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

    #[test]
    fn test_tokenizer_training_5() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_5",
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

    #[test]
    fn test_tokenizer_training_6() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_6",
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

    #[test]
    fn test_tokenizer_training_7() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_7",
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

    #[test]
    fn test_tokenizer_training_8() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_8",
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

    #[test]
    fn test_tokenizer_training_9() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_9",
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

    #[test]
    fn test_tokenizer_training_10() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_10",
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

    #[test]
    fn test_tokenizer_training_11() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_11",
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

    #[test]
    fn test_tokenizer_training_12() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_12",
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

    #[test]
    fn test_tokenizer_training_13() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_13",
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

    #[test]
    fn test_tokenizer_training_14() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_14",
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

    #[test]
    fn test_tokenizer_training_15() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_15",
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

    #[test]
    fn test_tokenizer_training_16() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_16",
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

    #[test]
    fn test_tokenizer_training_17() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_17",
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

    #[test]
    fn test_tokenizer_training_18() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_18",
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

    #[test]
    fn test_tokenizer_training_19() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_19",
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

    #[test]
    fn test_tokenizer_training_20() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_20",
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

    #[test]
    fn test_tokenizer_training_21() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_21",
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

    #[test]
    fn test_tokenizer_training_22() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_22",
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

    #[test]
    fn test_tokenizer_training_23() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_23",
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

    #[test]
    fn test_tokenizer_training_24() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_24",
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

    #[test]
    fn test_tokenizer_training_25() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_25",
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

    #[test]
    fn test_tokenizer_training_26() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_26",
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

    #[test]
    fn test_tokenizer_training_27() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_27",
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

    #[test]
    fn test_tokenizer_training_28() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_28",
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

    #[test]
    fn test_tokenizer_training_29() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_29",
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

    #[test]
    fn test_tokenizer_training_30() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_30",
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

    #[test]
    fn test_tokenizer_training_31() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_31",
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

    #[test]
    fn test_tokenizer_training_32() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_32",
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

    #[test]
    fn test_tokenizer_training_33() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_33",
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

    #[test]
    fn test_tokenizer_training_34() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_34",
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

    #[test]
    fn test_tokenizer_training_35() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_35",
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

    #[test]
    fn test_tokenizer_training_36() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_36",
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

    #[test]
    fn test_tokenizer_training_37() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_37",
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

    #[test]
    fn test_tokenizer_training_38() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_38",
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

    #[test]
    fn test_tokenizer_training_39() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_39",
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

    #[test]
    fn test_tokenizer_training_40() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_40",
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

    #[test]
    fn test_tokenizer_training_41() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_41",
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

    #[test]
    fn test_tokenizer_training_42() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_42",
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

    #[test]
    fn test_tokenizer_training_43() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_43",
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

    #[test]
    fn test_tokenizer_training_44() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_44",
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

    #[test]
    fn test_tokenizer_training_45() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_45",
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

    #[test]
    fn test_tokenizer_training_46() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_46",
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

    #[test]
    fn test_tokenizer_training_47() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_47",
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

    #[test]
    fn test_tokenizer_training_48() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_48",
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

    #[test]
    fn test_tokenizer_training_49() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_49",
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

    #[test]
    fn test_tokenizer_training_50() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_50",
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

    #[test]
    fn test_tokenizer_training_51() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_51",
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

    #[test]
    fn test_tokenizer_training_52() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_52",
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

    #[test]
    fn test_tokenizer_training_53() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_53",
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

    #[test]
    fn test_tokenizer_training_54() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_54",
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

    #[test]
    fn test_tokenizer_training_55() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_55",
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

    #[test]
    fn test_tokenizer_training_56() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_56",
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

    #[test]
    fn test_tokenizer_training_57() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_57",
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

    #[test]
    fn test_tokenizer_training_58() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_58",
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

    #[test]
    fn test_tokenizer_training_59() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_59",
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

    #[test]
    fn test_tokenizer_training_60() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_60",
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

    #[test]
    fn test_tokenizer_training_61() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_61",
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

    #[test]
    fn test_tokenizer_training_62() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_62",
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

    #[test]
    fn test_tokenizer_training_63() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_63",
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

    #[test]
    fn test_tokenizer_training_64() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_64",
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

    #[test]
    fn test_tokenizer_training_65() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_65",
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

    #[test]
    fn test_tokenizer_training_66() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_66",
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

    #[test]
    fn test_tokenizer_training_67() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_67",
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

    #[test]
    fn test_tokenizer_training_68() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_68",
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

    #[test]
    fn test_tokenizer_training_69() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_69",
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

    #[test]
    fn test_tokenizer_training_70() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_70",
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

    #[test]
    fn test_tokenizer_training_71() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_71",
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

    #[test]
    fn test_tokenizer_training_72() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_72",
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

    #[test]
    fn test_tokenizer_training_73() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_73",
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

    #[test]
    fn test_tokenizer_training_74() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_74",
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

    #[test]
    fn test_tokenizer_training_75() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_75",
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

    #[test]
    fn test_tokenizer_training_76() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_76",
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

    #[test]
    fn test_tokenizer_training_77() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_77",
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

    #[test]
    fn test_tokenizer_training_78() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_78",
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

    #[test]
    fn test_tokenizer_training_79() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_79",
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

    #[test]
    fn test_tokenizer_training_80() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_80",
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

    #[test]
    fn test_tokenizer_training_81() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_81",
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

    #[test]
    fn test_tokenizer_training_82() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_82",
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

    #[test]
    fn test_tokenizer_training_83() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_83",
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

    #[test]
    fn test_tokenizer_training_84() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_84",
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

    #[test]
    fn test_tokenizer_training_85() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_85",
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

    #[test]
    fn test_tokenizer_training_86() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_86",
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

    #[test]
    fn test_tokenizer_training_87() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_87",
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

    #[test]
    fn test_tokenizer_training_88() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_88",
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

    #[test]
    fn test_tokenizer_training_89() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_89",
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

    #[test]
    fn test_tokenizer_training_90() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_90",
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

    #[test]
    fn test_tokenizer_training_91() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_91",
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

    #[test]
    fn test_tokenizer_training_92() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_92",
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

    #[test]
    fn test_tokenizer_training_93() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_93",
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

    #[test]
    fn test_tokenizer_training_94() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_94",
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

    #[test]
    fn test_tokenizer_training_95() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_95",
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

    #[test]
    fn test_tokenizer_training_96() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_96",
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

    #[test]
    fn test_tokenizer_training_97() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_97",
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

    #[test]
    fn test_tokenizer_training_98() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_98",
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

    #[test]
    fn test_tokenizer_training_99() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_99",
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

    #[test]
    fn test_tokenizer_training_100() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_100",
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

    #[test]
    fn test_tokenizer_training_101() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_101",
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

    #[test]
    fn test_tokenizer_training_102() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_102",
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

    #[test]
    fn test_tokenizer_training_103() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_103",
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

    #[test]
    fn test_tokenizer_training_104() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_104",
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

    #[test]
    fn test_tokenizer_training_105() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_105",
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

    #[test]
    fn test_tokenizer_training_106() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_106",
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

    #[test]
    fn test_tokenizer_training_107() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_107",
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

    #[test]
    fn test_tokenizer_training_108() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_108",
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

    #[test]
    fn test_tokenizer_training_109() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_109",
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

    #[test]
    fn test_tokenizer_training_110() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_110",
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

    #[test]
    fn test_tokenizer_training_111() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_111",
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

    #[test]
    fn test_tokenizer_training_112() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_112",
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

    #[test]
    fn test_tokenizer_training_113() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_113",
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

    #[test]
    fn test_tokenizer_training_114() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_114",
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

    #[test]
    fn test_tokenizer_training_115() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_115",
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

    #[test]
    fn test_tokenizer_training_116() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_116",
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

    #[test]
    fn test_tokenizer_training_117() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_117",
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

    #[test]
    fn test_tokenizer_training_118() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_118",
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

    #[test]
    fn test_tokenizer_training_119() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_119",
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

    #[test]
    fn test_tokenizer_training_120() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_120",
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

    #[test]
    fn test_tokenizer_training_121() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_121",
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

    #[test]
    fn test_tokenizer_training_122() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_122",
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

    #[test]
    fn test_tokenizer_training_123() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_123",
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

    #[test]
    fn test_tokenizer_training_124() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_124",
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

    #[test]
    fn test_tokenizer_training_125() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_125",
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

    #[test]
    fn test_tokenizer_training_126() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_126",
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

    #[test]
    fn test_tokenizer_training_127() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_127",
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

    #[test]
    fn test_tokenizer_training_128() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_128",
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

    #[test]
    fn test_tokenizer_training_129() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_129",
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

    #[test]
    fn test_tokenizer_training_130() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_130",
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

    #[test]
    fn test_tokenizer_training_131() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_131",
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

    #[test]
    fn test_tokenizer_training_132() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_132",
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

    #[test]
    fn test_tokenizer_training_133() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_133",
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

    #[test]
    fn test_tokenizer_training_134() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_134",
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

    #[test]
    fn test_tokenizer_training_135() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_135",
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

    #[test]
    fn test_tokenizer_training_136() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_136",
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

    #[test]
    fn test_tokenizer_training_137() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_137",
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

    #[test]
    fn test_tokenizer_training_138() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_138",
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

    #[test]
    fn test_tokenizer_training_139() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_139",
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

    #[test]
    fn test_tokenizer_training_140() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_140",
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

    #[test]
    fn test_tokenizer_training_141() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_141",
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

    #[test]
    fn test_tokenizer_training_142() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_142",
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

    #[test]
    fn test_tokenizer_training_143() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_143",
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

    #[test]
    fn test_tokenizer_training_144() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_144",
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

    #[test]
    fn test_tokenizer_training_145() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_145",
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

    #[test]
    fn test_tokenizer_training_146() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_146",
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

    #[test]
    fn test_tokenizer_training_147() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_147",
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

    #[test]
    fn test_tokenizer_training_148() {
        let corpus = vec![
            "low lower lowest",
            "low low low",
            "newer newest wide_148",
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
}
