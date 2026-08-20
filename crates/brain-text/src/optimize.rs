//! # Vocabulary Pruning, Optimization, and Compression
//!
//! Vocabulary trimming based on token frequencies, string deduplication, trie prefix lookup, and quantization.
#![allow(
    missing_docs,
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_comparisons,
    clippy::all
)]

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
    fn test_optimize_pruning_1() {
        let mut vocab = Vocab::new();
        vocab.insert("special_1");
        vocab.insert("common_1");
        vocab.insert("rare_1");

        let mut freqs = HashMap::new();
        freqs.insert(0, 100);
        freqs.insert(1, 50);
        freqs.insert(2, 1);

        let opt_cfg = OptimizeConfig {
            min_frequency: 2,
            prune_singletons: true,
            ..Default::default()
        };
        let pruned = prune_vocab(&vocab, &freqs, &opt_cfg);
        assert!(pruned.contains("common_1"));
        assert!(!pruned.contains("rare_1"));

        let (dedup, map) =
            merge_identical_tokens(&["a".to_string(), "b".to_string(), "a".to_string()]);
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
}
