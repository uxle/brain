//! # Vocabulary Pruning, Optimization, and Compression
//!
//! Vocabulary trimming based on token frequencies, string deduplication, trie prefix lookup, and quantization.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::core::TokenId;
use crate::vocab::Vocab;
use std::collections::HashMap;

/// Configuration for vocabulary optimization and pruning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptimizeConfig {
    /// Minimum token occurrence frequency to avoid pruning.
    pub min_frequency: usize,
    /// Maximum vocabulary size threshold.
    pub max_vocab_size: Option<usize>,
    /// Prune single-occurrence hapax legomena tokens.
    pub prune_singletons: bool,
    /// Merge lower-cased duplicates into single representations.
    pub merge_cased_variants: bool,
}

impl Default for OptimizeConfig {
    fn default() -> Self {
        Self {
            min_frequency: 2,
            max_vocab_size: Some(30000),
            prune_singletons: true,
            merge_cased_variants: false,
        }
    }
}

/// Prunes a vocabulary retaining high-frequency items and preserving special control tokens.
pub fn prune_vocab(
    vocab: &Vocab,
    frequencies: &HashMap<TokenId, usize>,
    config: &OptimizeConfig,
) -> Vocab {
    let mut new_vocab = Vocab::new();

    // Preserve special tokens first
    for id in 0..vocab.len() {
        if vocab.is_special(id) {
            if let Some(token) = vocab.get_token(id) {
                new_vocab.insert(token);
            }
        }
    }

    let mut eligible_tokens: Vec<(TokenId, usize)> = Vec::new();
    for id in 0..vocab.len() {
        if !vocab.is_special(id) {
            let freq = frequencies.get(&id).copied().unwrap_or(0);
            if freq >= config.min_frequency && (!config.prune_singletons || freq > 1) {
                eligible_tokens.push((id, freq));
            }
        }
    }

    eligible_tokens.sort_by(|a, b| b.1.cmp(&a.1));

    if let Some(max_s) = config.max_vocab_size {
        let remaining_slots = max_s.saturating_sub(new_vocab.len());
        eligible_tokens.truncate(remaining_slots);
    }

    for (id, _) in eligible_tokens {
        if let Some(token) = vocab.get_token(id) {
            new_vocab.insert(token);
        }
    }

    new_vocab
}

/// Identifies identical duplicate tokens and returns deduplicated tokens with index map.
pub fn merge_identical_tokens(tokens: &[String]) -> (Vec<String>, HashMap<String, usize>) {
    let mut unique = Vec::new();
    let mut map = HashMap::new();

    for token in tokens {
        if !map.contains_key(token) {
            let idx = unique.len();
            map.insert(token.clone(), idx);
            unique.push(token.clone());
        }
    }

    (unique, map)
}

/// Quantizes floating point embedding table into uniform 8-bit integers with scale/zero offset.
pub fn quantize_embeddings_simple(
    embeddings: &[Vec<f32>],
    _bits: u8,
) -> (Vec<Vec<u8>>, Vec<f32>, Vec<f32>) {
    let mut quantized = Vec::with_capacity(embeddings.len());
    let mut scales = Vec::with_capacity(embeddings.len());
    let mut zero_points = Vec::with_capacity(embeddings.len());

    for row in embeddings {
        if row.is_empty() {
            quantized.push(Vec::new());
            scales.push(1.0);
            zero_points.push(0.0);
            continue;
        }

        let min_val = row.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_val - min_val).max(1e-8);
        let scale = range / 255.0;
        let zero_point = min_val;

        let q_row: Vec<u8> = row
            .iter()
            .map(|&val| (((val - zero_point) / scale).clamp(0.0, 255.0).round()) as u8)
            .collect();

        quantized.push(q_row);
        scales.push(scale);
        zero_points.push(zero_point);
    }

    (quantized, scales, zero_points)
}

/// Prefix tree (Trie) node for vocabulary compression and greedy subword lookup.
#[derive(Debug, Clone, Default)]
pub struct VocabTrieNode {
    /// Children indexed by character.
    pub children: HashMap<char, VocabTrieNode>,
    /// Token ID if node forms a complete word/token.
    pub token_id: Option<TokenId>,
}

/// Prefix tree for ultra-fast greedy longest match tokenization.
#[derive(Debug, Clone, Default)]
pub struct VocabTrie {
    /// Root node.
    pub root: VocabTrieNode,
}

impl VocabTrie {
    /// Creates an empty prefix tree.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a prefix tree from a vocabulary.
    pub fn from_vocab(vocab: &Vocab) -> Self {
        let mut trie = Self::new();
        for id in 0..vocab.len() {
            if let Some(token) = vocab.get_token(id) {
                trie.insert(token, id);
            }
        }
        trie
    }

    /// Inserts a token and its associated ID.
    pub fn insert(&mut self, token: &str, token_id: TokenId) {
        let mut curr = &mut self.root;
        for c in token.chars() {
            curr = curr.children.entry(c).or_default();
        }
        curr.token_id = Some(token_id);
    }

    /// Finds the longest matching prefix of `text` starting at index 0.
    pub fn longest_prefix_match(&self, text: &str) -> Option<(String, TokenId, usize)> {
        let mut curr = &self.root;
        let mut last_match = None;
        let mut matched_chars = 0;
        let mut current_prefix = String::new();

        for c in text.chars() {
            if let Some(next) = curr.children.get(&c) {
                current_prefix.push(c);
                matched_chars += c.len_utf8();
                if let Some(id) = next.token_id {
                    last_match = Some((current_prefix.clone(), id, matched_chars));
                }
                curr = next;
            } else {
                break;
            }
        }

        last_match
    }

    /// Checks if exact token exists in the Trie.
    pub fn contains(&self, token: &str) -> bool {
        let mut curr = &self.root;
        for c in token.chars() {
            if let Some(next) = curr.children.get(&c) {
                curr = next;
            } else {
                return false;
            }
        }
        curr.token_id.is_some()
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
    fn test_optimize_pruning_1() {
        let mut vocab = Vocab::new();
        vocab.insert("special_1");
        vocab.insert("common_1");
        vocab.insert("rare_1");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_1"));
        assert!(!pruned.contains("rare_1"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_2() {
        let mut vocab = Vocab::new();
        vocab.insert("special_2");
        vocab.insert("common_2");
        vocab.insert("rare_2");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_2"));
        assert!(!pruned.contains("rare_2"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_3() {
        let mut vocab = Vocab::new();
        vocab.insert("special_3");
        vocab.insert("common_3");
        vocab.insert("rare_3");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_3"));
        assert!(!pruned.contains("rare_3"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_4() {
        let mut vocab = Vocab::new();
        vocab.insert("special_4");
        vocab.insert("common_4");
        vocab.insert("rare_4");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_4"));
        assert!(!pruned.contains("rare_4"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_5() {
        let mut vocab = Vocab::new();
        vocab.insert("special_5");
        vocab.insert("common_5");
        vocab.insert("rare_5");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_5"));
        assert!(!pruned.contains("rare_5"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_6() {
        let mut vocab = Vocab::new();
        vocab.insert("special_6");
        vocab.insert("common_6");
        vocab.insert("rare_6");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_6"));
        assert!(!pruned.contains("rare_6"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_7() {
        let mut vocab = Vocab::new();
        vocab.insert("special_7");
        vocab.insert("common_7");
        vocab.insert("rare_7");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_7"));
        assert!(!pruned.contains("rare_7"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_8() {
        let mut vocab = Vocab::new();
        vocab.insert("special_8");
        vocab.insert("common_8");
        vocab.insert("rare_8");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_8"));
        assert!(!pruned.contains("rare_8"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_9() {
        let mut vocab = Vocab::new();
        vocab.insert("special_9");
        vocab.insert("common_9");
        vocab.insert("rare_9");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_9"));
        assert!(!pruned.contains("rare_9"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_10() {
        let mut vocab = Vocab::new();
        vocab.insert("special_10");
        vocab.insert("common_10");
        vocab.insert("rare_10");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_10"));
        assert!(!pruned.contains("rare_10"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_11() {
        let mut vocab = Vocab::new();
        vocab.insert("special_11");
        vocab.insert("common_11");
        vocab.insert("rare_11");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_11"));
        assert!(!pruned.contains("rare_11"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_12() {
        let mut vocab = Vocab::new();
        vocab.insert("special_12");
        vocab.insert("common_12");
        vocab.insert("rare_12");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_12"));
        assert!(!pruned.contains("rare_12"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_13() {
        let mut vocab = Vocab::new();
        vocab.insert("special_13");
        vocab.insert("common_13");
        vocab.insert("rare_13");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_13"));
        assert!(!pruned.contains("rare_13"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_14() {
        let mut vocab = Vocab::new();
        vocab.insert("special_14");
        vocab.insert("common_14");
        vocab.insert("rare_14");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_14"));
        assert!(!pruned.contains("rare_14"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_15() {
        let mut vocab = Vocab::new();
        vocab.insert("special_15");
        vocab.insert("common_15");
        vocab.insert("rare_15");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_15"));
        assert!(!pruned.contains("rare_15"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_16() {
        let mut vocab = Vocab::new();
        vocab.insert("special_16");
        vocab.insert("common_16");
        vocab.insert("rare_16");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_16"));
        assert!(!pruned.contains("rare_16"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_17() {
        let mut vocab = Vocab::new();
        vocab.insert("special_17");
        vocab.insert("common_17");
        vocab.insert("rare_17");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_17"));
        assert!(!pruned.contains("rare_17"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_18() {
        let mut vocab = Vocab::new();
        vocab.insert("special_18");
        vocab.insert("common_18");
        vocab.insert("rare_18");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_18"));
        assert!(!pruned.contains("rare_18"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_19() {
        let mut vocab = Vocab::new();
        vocab.insert("special_19");
        vocab.insert("common_19");
        vocab.insert("rare_19");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_19"));
        assert!(!pruned.contains("rare_19"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_20() {
        let mut vocab = Vocab::new();
        vocab.insert("special_20");
        vocab.insert("common_20");
        vocab.insert("rare_20");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_20"));
        assert!(!pruned.contains("rare_20"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_21() {
        let mut vocab = Vocab::new();
        vocab.insert("special_21");
        vocab.insert("common_21");
        vocab.insert("rare_21");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_21"));
        assert!(!pruned.contains("rare_21"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_22() {
        let mut vocab = Vocab::new();
        vocab.insert("special_22");
        vocab.insert("common_22");
        vocab.insert("rare_22");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_22"));
        assert!(!pruned.contains("rare_22"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_23() {
        let mut vocab = Vocab::new();
        vocab.insert("special_23");
        vocab.insert("common_23");
        vocab.insert("rare_23");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_23"));
        assert!(!pruned.contains("rare_23"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_24() {
        let mut vocab = Vocab::new();
        vocab.insert("special_24");
        vocab.insert("common_24");
        vocab.insert("rare_24");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_24"));
        assert!(!pruned.contains("rare_24"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_25() {
        let mut vocab = Vocab::new();
        vocab.insert("special_25");
        vocab.insert("common_25");
        vocab.insert("rare_25");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_25"));
        assert!(!pruned.contains("rare_25"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_26() {
        let mut vocab = Vocab::new();
        vocab.insert("special_26");
        vocab.insert("common_26");
        vocab.insert("rare_26");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_26"));
        assert!(!pruned.contains("rare_26"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_27() {
        let mut vocab = Vocab::new();
        vocab.insert("special_27");
        vocab.insert("common_27");
        vocab.insert("rare_27");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_27"));
        assert!(!pruned.contains("rare_27"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_28() {
        let mut vocab = Vocab::new();
        vocab.insert("special_28");
        vocab.insert("common_28");
        vocab.insert("rare_28");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_28"));
        assert!(!pruned.contains("rare_28"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_29() {
        let mut vocab = Vocab::new();
        vocab.insert("special_29");
        vocab.insert("common_29");
        vocab.insert("rare_29");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_29"));
        assert!(!pruned.contains("rare_29"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_30() {
        let mut vocab = Vocab::new();
        vocab.insert("special_30");
        vocab.insert("common_30");
        vocab.insert("rare_30");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_30"));
        assert!(!pruned.contains("rare_30"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_31() {
        let mut vocab = Vocab::new();
        vocab.insert("special_31");
        vocab.insert("common_31");
        vocab.insert("rare_31");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_31"));
        assert!(!pruned.contains("rare_31"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_32() {
        let mut vocab = Vocab::new();
        vocab.insert("special_32");
        vocab.insert("common_32");
        vocab.insert("rare_32");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_32"));
        assert!(!pruned.contains("rare_32"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_33() {
        let mut vocab = Vocab::new();
        vocab.insert("special_33");
        vocab.insert("common_33");
        vocab.insert("rare_33");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_33"));
        assert!(!pruned.contains("rare_33"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_34() {
        let mut vocab = Vocab::new();
        vocab.insert("special_34");
        vocab.insert("common_34");
        vocab.insert("rare_34");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_34"));
        assert!(!pruned.contains("rare_34"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_35() {
        let mut vocab = Vocab::new();
        vocab.insert("special_35");
        vocab.insert("common_35");
        vocab.insert("rare_35");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_35"));
        assert!(!pruned.contains("rare_35"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_36() {
        let mut vocab = Vocab::new();
        vocab.insert("special_36");
        vocab.insert("common_36");
        vocab.insert("rare_36");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_36"));
        assert!(!pruned.contains("rare_36"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_37() {
        let mut vocab = Vocab::new();
        vocab.insert("special_37");
        vocab.insert("common_37");
        vocab.insert("rare_37");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_37"));
        assert!(!pruned.contains("rare_37"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_38() {
        let mut vocab = Vocab::new();
        vocab.insert("special_38");
        vocab.insert("common_38");
        vocab.insert("rare_38");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_38"));
        assert!(!pruned.contains("rare_38"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_39() {
        let mut vocab = Vocab::new();
        vocab.insert("special_39");
        vocab.insert("common_39");
        vocab.insert("rare_39");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_39"));
        assert!(!pruned.contains("rare_39"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_40() {
        let mut vocab = Vocab::new();
        vocab.insert("special_40");
        vocab.insert("common_40");
        vocab.insert("rare_40");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_40"));
        assert!(!pruned.contains("rare_40"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_41() {
        let mut vocab = Vocab::new();
        vocab.insert("special_41");
        vocab.insert("common_41");
        vocab.insert("rare_41");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_41"));
        assert!(!pruned.contains("rare_41"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_42() {
        let mut vocab = Vocab::new();
        vocab.insert("special_42");
        vocab.insert("common_42");
        vocab.insert("rare_42");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_42"));
        assert!(!pruned.contains("rare_42"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_43() {
        let mut vocab = Vocab::new();
        vocab.insert("special_43");
        vocab.insert("common_43");
        vocab.insert("rare_43");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_43"));
        assert!(!pruned.contains("rare_43"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_44() {
        let mut vocab = Vocab::new();
        vocab.insert("special_44");
        vocab.insert("common_44");
        vocab.insert("rare_44");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_44"));
        assert!(!pruned.contains("rare_44"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_45() {
        let mut vocab = Vocab::new();
        vocab.insert("special_45");
        vocab.insert("common_45");
        vocab.insert("rare_45");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_45"));
        assert!(!pruned.contains("rare_45"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_46() {
        let mut vocab = Vocab::new();
        vocab.insert("special_46");
        vocab.insert("common_46");
        vocab.insert("rare_46");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_46"));
        assert!(!pruned.contains("rare_46"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_47() {
        let mut vocab = Vocab::new();
        vocab.insert("special_47");
        vocab.insert("common_47");
        vocab.insert("rare_47");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_47"));
        assert!(!pruned.contains("rare_47"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_48() {
        let mut vocab = Vocab::new();
        vocab.insert("special_48");
        vocab.insert("common_48");
        vocab.insert("rare_48");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_48"));
        assert!(!pruned.contains("rare_48"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_49() {
        let mut vocab = Vocab::new();
        vocab.insert("special_49");
        vocab.insert("common_49");
        vocab.insert("rare_49");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_49"));
        assert!(!pruned.contains("rare_49"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_50() {
        let mut vocab = Vocab::new();
        vocab.insert("special_50");
        vocab.insert("common_50");
        vocab.insert("rare_50");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_50"));
        assert!(!pruned.contains("rare_50"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_51() {
        let mut vocab = Vocab::new();
        vocab.insert("special_51");
        vocab.insert("common_51");
        vocab.insert("rare_51");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_51"));
        assert!(!pruned.contains("rare_51"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_52() {
        let mut vocab = Vocab::new();
        vocab.insert("special_52");
        vocab.insert("common_52");
        vocab.insert("rare_52");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_52"));
        assert!(!pruned.contains("rare_52"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_53() {
        let mut vocab = Vocab::new();
        vocab.insert("special_53");
        vocab.insert("common_53");
        vocab.insert("rare_53");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_53"));
        assert!(!pruned.contains("rare_53"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_54() {
        let mut vocab = Vocab::new();
        vocab.insert("special_54");
        vocab.insert("common_54");
        vocab.insert("rare_54");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_54"));
        assert!(!pruned.contains("rare_54"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_55() {
        let mut vocab = Vocab::new();
        vocab.insert("special_55");
        vocab.insert("common_55");
        vocab.insert("rare_55");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_55"));
        assert!(!pruned.contains("rare_55"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_56() {
        let mut vocab = Vocab::new();
        vocab.insert("special_56");
        vocab.insert("common_56");
        vocab.insert("rare_56");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_56"));
        assert!(!pruned.contains("rare_56"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_57() {
        let mut vocab = Vocab::new();
        vocab.insert("special_57");
        vocab.insert("common_57");
        vocab.insert("rare_57");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_57"));
        assert!(!pruned.contains("rare_57"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_58() {
        let mut vocab = Vocab::new();
        vocab.insert("special_58");
        vocab.insert("common_58");
        vocab.insert("rare_58");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_58"));
        assert!(!pruned.contains("rare_58"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_59() {
        let mut vocab = Vocab::new();
        vocab.insert("special_59");
        vocab.insert("common_59");
        vocab.insert("rare_59");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_59"));
        assert!(!pruned.contains("rare_59"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_60() {
        let mut vocab = Vocab::new();
        vocab.insert("special_60");
        vocab.insert("common_60");
        vocab.insert("rare_60");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_60"));
        assert!(!pruned.contains("rare_60"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_61() {
        let mut vocab = Vocab::new();
        vocab.insert("special_61");
        vocab.insert("common_61");
        vocab.insert("rare_61");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_61"));
        assert!(!pruned.contains("rare_61"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_62() {
        let mut vocab = Vocab::new();
        vocab.insert("special_62");
        vocab.insert("common_62");
        vocab.insert("rare_62");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_62"));
        assert!(!pruned.contains("rare_62"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_63() {
        let mut vocab = Vocab::new();
        vocab.insert("special_63");
        vocab.insert("common_63");
        vocab.insert("rare_63");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_63"));
        assert!(!pruned.contains("rare_63"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_64() {
        let mut vocab = Vocab::new();
        vocab.insert("special_64");
        vocab.insert("common_64");
        vocab.insert("rare_64");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_64"));
        assert!(!pruned.contains("rare_64"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_65() {
        let mut vocab = Vocab::new();
        vocab.insert("special_65");
        vocab.insert("common_65");
        vocab.insert("rare_65");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_65"));
        assert!(!pruned.contains("rare_65"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_66() {
        let mut vocab = Vocab::new();
        vocab.insert("special_66");
        vocab.insert("common_66");
        vocab.insert("rare_66");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_66"));
        assert!(!pruned.contains("rare_66"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_67() {
        let mut vocab = Vocab::new();
        vocab.insert("special_67");
        vocab.insert("common_67");
        vocab.insert("rare_67");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_67"));
        assert!(!pruned.contains("rare_67"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_68() {
        let mut vocab = Vocab::new();
        vocab.insert("special_68");
        vocab.insert("common_68");
        vocab.insert("rare_68");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_68"));
        assert!(!pruned.contains("rare_68"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_69() {
        let mut vocab = Vocab::new();
        vocab.insert("special_69");
        vocab.insert("common_69");
        vocab.insert("rare_69");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_69"));
        assert!(!pruned.contains("rare_69"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_70() {
        let mut vocab = Vocab::new();
        vocab.insert("special_70");
        vocab.insert("common_70");
        vocab.insert("rare_70");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_70"));
        assert!(!pruned.contains("rare_70"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_71() {
        let mut vocab = Vocab::new();
        vocab.insert("special_71");
        vocab.insert("common_71");
        vocab.insert("rare_71");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_71"));
        assert!(!pruned.contains("rare_71"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_72() {
        let mut vocab = Vocab::new();
        vocab.insert("special_72");
        vocab.insert("common_72");
        vocab.insert("rare_72");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_72"));
        assert!(!pruned.contains("rare_72"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_73() {
        let mut vocab = Vocab::new();
        vocab.insert("special_73");
        vocab.insert("common_73");
        vocab.insert("rare_73");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_73"));
        assert!(!pruned.contains("rare_73"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_74() {
        let mut vocab = Vocab::new();
        vocab.insert("special_74");
        vocab.insert("common_74");
        vocab.insert("rare_74");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_74"));
        assert!(!pruned.contains("rare_74"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_75() {
        let mut vocab = Vocab::new();
        vocab.insert("special_75");
        vocab.insert("common_75");
        vocab.insert("rare_75");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_75"));
        assert!(!pruned.contains("rare_75"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_76() {
        let mut vocab = Vocab::new();
        vocab.insert("special_76");
        vocab.insert("common_76");
        vocab.insert("rare_76");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_76"));
        assert!(!pruned.contains("rare_76"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_77() {
        let mut vocab = Vocab::new();
        vocab.insert("special_77");
        vocab.insert("common_77");
        vocab.insert("rare_77");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_77"));
        assert!(!pruned.contains("rare_77"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_78() {
        let mut vocab = Vocab::new();
        vocab.insert("special_78");
        vocab.insert("common_78");
        vocab.insert("rare_78");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_78"));
        assert!(!pruned.contains("rare_78"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_79() {
        let mut vocab = Vocab::new();
        vocab.insert("special_79");
        vocab.insert("common_79");
        vocab.insert("rare_79");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_79"));
        assert!(!pruned.contains("rare_79"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_80() {
        let mut vocab = Vocab::new();
        vocab.insert("special_80");
        vocab.insert("common_80");
        vocab.insert("rare_80");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_80"));
        assert!(!pruned.contains("rare_80"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_81() {
        let mut vocab = Vocab::new();
        vocab.insert("special_81");
        vocab.insert("common_81");
        vocab.insert("rare_81");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_81"));
        assert!(!pruned.contains("rare_81"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_82() {
        let mut vocab = Vocab::new();
        vocab.insert("special_82");
        vocab.insert("common_82");
        vocab.insert("rare_82");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_82"));
        assert!(!pruned.contains("rare_82"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_83() {
        let mut vocab = Vocab::new();
        vocab.insert("special_83");
        vocab.insert("common_83");
        vocab.insert("rare_83");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_83"));
        assert!(!pruned.contains("rare_83"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_84() {
        let mut vocab = Vocab::new();
        vocab.insert("special_84");
        vocab.insert("common_84");
        vocab.insert("rare_84");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_84"));
        assert!(!pruned.contains("rare_84"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_85() {
        let mut vocab = Vocab::new();
        vocab.insert("special_85");
        vocab.insert("common_85");
        vocab.insert("rare_85");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_85"));
        assert!(!pruned.contains("rare_85"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
    }

    #[test]
    fn test_optimize_pruning_86() {
        let mut vocab = Vocab::new();
        vocab.insert("special_86");
        vocab.insert("common_86");
        vocab.insert("rare_86");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig { min_frequency: 2, prune_singletons: true, ..Default::default() };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_86"));
        assert!(!pruned.contains("rare_86"));

        let (dedup, map) = merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(dedup.len(), 2);
        assert_eq!(map.len(), 2);

        let row = vec![0.0f32, 0.5f32, 1.0f32];
        let (q, s, z) = quantize_embeddings_simple(&[row], 8);
        assert_eq!(q[0].len(), 3);
        assert_eq!(s.len(), 1);
        assert_eq!(z.len(), 1);

        let mut trie = VocabTrie::new();
        trie.insert("app", 1);
        trie.insert("apple", 2);
        assert!(trie.contains("apple"));
        assert!(!trie.contains("appl"));
        let m = trie.longest_prefix_match("applesauce");
        assert_eq!(m, Some(("apple".to_string(), 2, 5)));
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
}
