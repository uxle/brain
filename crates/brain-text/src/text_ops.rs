//! # N-grams, Shingles, and Text Frequency Analytics
//!
//! Extraction of character and word n-grams, skip-grams, term frequency counters, and statistical metrics.
#![allow(
    missing_docs,
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_comparisons,
    clippy::all
)]

use std::collections::{HashMap, HashSet};

/// Extracts word n-grams of a given order `n` from a slice of tokens.
pub fn ngrams(tokens: &[String], n: usize) -> Vec<Vec<String>> {
    if n == 0 || tokens.len() < n {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(tokens.len() - n + 1);
    for i in 0..=(tokens.len() - n) {
        result.push(tokens[i..i + n].to_vec());
    }
    result
}

/// Extracts character-level n-grams from a string slice.
pub fn character_ngrams(text: &str, n: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if n == 0 || chars.len() < n {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(chars.len() - n + 1);
    for i in 0..=(chars.len() - n) {
        result.push(chars[i..i + n].iter().collect());
    }
    result
}

/// Generates a set of character shingles of length `k`.
pub fn shingles(text: &str, k: usize) -> HashSet<String> {
    character_ngrams(text, k).into_iter().collect()
}

/// Counts word occurrences in a whitespace-separated text string.
pub fn word_counts(text: &str) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for word in text.split_whitespace() {
        let cleaned: String = word.chars().filter(|c| !c.is_ascii_punctuation()).collect();
        if !cleaned.is_empty() {
            *counts.entry(cleaned.to_lowercase()).or_insert(0) += 1;
        }
    }
    counts
}

/// Computes the frequency of each n-gram across a sequence of tokens.
pub fn ngram_freq(tokens: &[String], n: usize) -> HashMap<Vec<String>, usize> {
    let mut freqs = HashMap::new();
    for gram in ngrams(tokens, n) {
        *freqs.entry(gram).or_insert(0) += 1;
    }
    freqs
}

/// Extracts skip-grams with maximum skip distance `k` and order `n`.
pub fn skipgrams(tokens: &[String], n: usize, k: usize) -> Vec<Vec<String>> {
    if n == 0 || tokens.is_empty() {
        return Vec::new();
    }
    if n == 1 {
        return tokens.iter().map(|t| vec![t.clone()]).collect();
    }
    if n == 2 {
        let mut result = Vec::new();
        for i in 0..tokens.len() {
            let limit = (i + k + 2).min(tokens.len());
            for j in (i + 1)..limit {
                result.push(vec![tokens[i].clone(), tokens[j].clone()]);
            }
        }
        return result;
    }
    ngrams(tokens, n)
}

/// Computes bigram collocations filtered by minimum frequency and Pointwise Mutual Information (PMI).
pub fn collocations(
    tokens: &[String],
    min_count: usize,
    min_pmi: f64,
) -> Vec<(String, String, f64)> {
    if tokens.len() < 2 {
        return Vec::new();
    }
    let total_tokens = tokens.len() as f64;
    let total_bigrams = (tokens.len() - 1) as f64;

    let mut unigram_counts: HashMap<String, usize> = HashMap::new();
    for t in tokens {
        *unigram_counts.entry(t.clone()).or_insert(0) += 1;
    }

    let mut bigram_counts: HashMap<(String, String), usize> = HashMap::new();
    for i in 0..(tokens.len() - 1) {
        let pair = (tokens[i].clone(), tokens[i + 1].clone());
        *bigram_counts.entry(pair).or_insert(0) += 1;
    }

    let mut results = Vec::new();
    for ((w1, w2), count) in bigram_counts {
        if count >= min_count {
            let p_w1 = *unigram_counts.get(&w1).unwrap_or(&1) as f64 / total_tokens;
            let p_w2 = *unigram_counts.get(&w2).unwrap_or(&1) as f64 / total_tokens;
            let p_w1_w2 = count as f64 / total_bigrams;
            let pmi = (p_w1_w2 / (p_w1 * p_w2)).ln();
            if pmi >= min_pmi {
                results.push((w1, w2, pmi));
            }
        }
    }
    results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Computes normalized term frequencies (TF) for a token list.
pub fn term_frequencies(tokens: &[String]) -> HashMap<String, f64> {
    let mut counts = HashMap::new();
    for t in tokens {
        *counts.entry(t.clone()).or_insert(0usize) += 1;
    }
    let total = tokens.len() as f64;
    if total == 0.0 {
        return HashMap::new();
    }
    counts
        .into_iter()
        .map(|(k, v)| (k, v as f64 / total))
        .collect()
}

/// Computes Shannon entropy of character distribution in text.
pub fn text_entropy(text: &str) -> f64 {
    if text.is_empty() {
        return 0.0;
    }
    let mut counts = HashMap::new();
    let mut total = 0usize;
    for c in text.chars() {
        *counts.entry(c).or_insert(0usize) += 1;
        total += 1;
    }
    let mut entropy = 0.0;
    let n = total as f64;
    for &count in counts.values() {
        let p = count as f64 / n;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }
    entropy
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
    fn test_text_ops_suite_1() {
        let tokens = vec![
            "the".to_string(),
            "quick".to_string(),
            "brown".to_string(),
            "fox_1".to_string(),
        ];
        let bigrams = ngrams(&tokens, 2);
        assert_eq!(bigrams.len(), 3);
        assert_eq!(bigrams[0], vec!["the", "quick"]);

        let char_grams = character_ngrams("hello_1", 3);
        assert!(!char_grams.is_empty());

        let shing = shingles("banana_1", 2);
        assert!(shing.contains("an"));

        let wc = word_counts("hello world hello 1");
        assert_eq!(wc.get("hello"), Some(&2));

        let freq = ngram_freq(&tokens, 2);
        assert_eq!(
            freq.get(&vec!["the".to_string(), "quick".to_string()]),
            Some(&1)
        );

        let skips = skipgrams(&tokens, 2, 1);
        assert!(!skips.is_empty());

        let tf = term_frequencies(&tokens);
        assert!((tf.get("the").unwrap() - 0.25).abs() < 1e-5);

        let ent = text_entropy("aaaaaabbbbcc_1");
        assert!(ent > 0.0);
    }
}
