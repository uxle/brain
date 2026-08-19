//! # Text Statistics, Lexical Diversity, and Readability Metrics
//!
//! Comprehensive readability scoring formulas (Flesch, Kincaid, Gunning Fog, Coleman-Liau, ARI) and statistical entropy.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use std::collections::HashMap;

/// Comprehensive descriptive statistics for a document.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TextStats {
    /// Total character count including spaces and punctuation.
    pub char_count: usize,
    /// Total alphabetic letter count.
    pub letter_count: usize,
    /// Total digit count.
    pub digit_count: usize,
    /// Total whitespace character count.
    pub space_count: usize,
    /// Total punctuation mark count.
    pub punctuation_count: usize,
    /// Total word count.
    pub word_count: usize,
    /// Total sentence count.
    pub sentence_count: usize,
    /// Average word length in characters.
    pub avg_word_length: f64,
    /// Average sentence length in words.
    pub avg_sentence_length: f64,
    /// Type-Token Ratio (unique words / total words).
    pub type_token_ratio: f64,
}

/// Analyzes text and computes complete descriptive statistics.
pub fn analyze_text(text: &str) -> TextStats {
    let mut stats = TextStats::default();
    stats.char_count = text.chars().count();

    for c in text.chars() {
        if c.is_alphabetic() {
            stats.letter_count += 1;
        } else if c.is_ascii_digit() {
            stats.digit_count += 1;
        } else if c.is_whitespace() {
            stats.space_count += 1;
        } else if c.is_ascii_punctuation() {
            stats.punctuation_count += 1;
        }
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    stats.word_count = words.len();

    let mut unique_words = HashMap::new();
    let mut total_word_len = 0usize;
    for &w in &words {
        let cleaned: String = w.chars().filter(|c| c.is_alphanumeric()).collect();
        if !cleaned.is_empty() {
            total_word_len += cleaned.chars().count();
            *unique_words.entry(cleaned.to_lowercase()).or_insert(0usize) += 1;
        }
    }

    let sentences = crate::process::split_into_sentences(text);
    stats.sentence_count = sentences.len().max(1);

    stats.avg_word_length = if stats.word_count > 0 {
        total_word_len as f64 / stats.word_count as f64
    } else {
        0.0
    };

    stats.avg_sentence_length = if stats.sentence_count > 0 {
        stats.word_count as f64 / stats.sentence_count as f64
    } else {
        0.0
    };

    stats.type_token_ratio = if stats.word_count > 0 {
        unique_words.len() as f64 / stats.word_count as f64
    } else {
        0.0
    };

    stats
}

/// Estimates syllable count in an English word using vowel grouping heuristics.
pub fn count_syllables(word: &str) -> usize {
    let w = word.to_lowercase();
    let chars: Vec<char> = w.chars().filter(|c| c.is_alphabetic()).collect();
    if chars.is_empty() {
        return 0;
    }

    let vowels = ['a', 'e', 'i', 'o', 'u', 'y'];
    let mut count = 0usize;
    let mut prev_vowel = false;

    for &c in &chars {
        let is_vowel = vowels.contains(&c);
        if is_vowel && !prev_vowel {
            count += 1;
        }
        prev_vowel = is_vowel;
    }

    // Adjust for silent e
    if chars.len() > 2 && chars.last() == Some(&'e') && count > 1 {
        count -= 1;
    }

    count.max(1)
}

/// Computes Flesch Reading Ease score: $206.835 - 1.015 \cdot (W/S) - 84.6 \cdot (L/W)$.
pub fn flesch_reading_ease(text: &str) -> f64 {
    let stats = analyze_text(text);
    if stats.word_count == 0 || stats.sentence_count == 0 {
        return 100.0;
    }

    let mut total_syllables = 0usize;
    for word in text.split_whitespace() {
        total_syllables += count_syllables(word);
    }

    let w_s = stats.word_count as f64 / stats.sentence_count as f64;
    let s_w = total_syllables as f64 / stats.word_count as f64;
    206.835 - 1.015 * w_s - 84.6 * s_w
}

/// Computes Flesch-Kincaid Grade Level: $0.39 \cdot (W/S) + 11.8 \cdot (L/W) - 15.59$.
pub fn flesch_kincaid_grade(text: &str) -> f64 {
    let stats = analyze_text(text);
    if stats.word_count == 0 || stats.sentence_count == 0 {
        return 0.0;
    }

    let mut total_syllables = 0usize;
    for word in text.split_whitespace() {
        total_syllables += count_syllables(word);
    }

    let w_s = stats.word_count as f64 / stats.sentence_count as f64;
    let s_w = total_syllables as f64 / stats.word_count as f64;
    (0.39 * w_s + 11.8 * s_w - 15.59).max(0.0)
}

/// Computes Gunning Fog Index: $0.4 \cdot ((W/S) + 100 \cdot (Complex/W))$.
pub fn gunning_fog_index(text: &str) -> f64 {
    let stats = analyze_text(text);
    if stats.word_count == 0 || stats.sentence_count == 0 {
        return 0.0;
    }

    let mut complex_words = 0usize;
    for word in text.split_whitespace() {
        if count_syllables(word) >= 3 {
            complex_words += 1;
        }
    }

    let w_s = stats.word_count as f64 / stats.sentence_count as f64;
    let complex_ratio = complex_words as f64 / stats.word_count as f64;
    0.4 * (w_s + 100.0 * complex_ratio)
}

/// Computes Coleman-Liau Index: $0.0588 \cdot L - 0.296 \cdot S - 15.8$.
pub fn coleman_liau_index(text: &str) -> f64 {
    let stats = analyze_text(text);
    if stats.word_count == 0 {
        return 0.0;
    }

    let l = (stats.letter_count as f64 / stats.word_count as f64) * 100.0;
    let s = (stats.sentence_count as f64 / stats.word_count as f64) * 100.0;
    0.0588 * l - 0.296 * s - 15.8
}

/// Computes Automated Readability Index (ARI): $4.71 \cdot (C/W) + 0.5 \cdot (W/S) - 21.43$.
pub fn automated_readability_index(text: &str) -> f64 {
    let stats = analyze_text(text);
    if stats.word_count == 0 || stats.sentence_count == 0 {
        return 0.0;
    }

    let c_w = stats.letter_count as f64 / stats.word_count as f64;
    let w_s = stats.word_count as f64 / stats.sentence_count as f64;
    (4.71 * c_w + 0.5 * w_s - 21.43).max(0.0)
}

/// Computes Hapax Legomena ratio (proportion of words occurring exactly once).
pub fn hapax_legomena_ratio(words: &[String]) -> f64 {
    if words.is_empty() {
        return 0.0;
    }
    let mut counts = HashMap::new();
    for w in words {
        *counts.entry(w).or_insert(0usize) += 1;
    }
    let hapax_count = counts.values().filter(|&&c| c == 1).count();
    hapax_count as f64 / words.len() as f64
}

/// Computes Yule's K Characteristic for vocabulary richness: $10^4 \cdot \frac{\sum i^2 V_i - N}{N^2}$.
pub fn yules_k_characteristic(words: &[String]) -> f64 {
    if words.is_empty() {
        return 0.0;
    }
    let n = words.len() as f64;
    let mut freqs = HashMap::new();
    for w in words {
        *freqs.entry(w).or_insert(0usize) += 1;
    }

    let mut freq_spectrum = HashMap::new();
    for &f in freqs.values() {
        *freq_spectrum.entry(f).or_insert(0usize) += 1;
    }

    let mut sum_i2_vi = 0.0;
    for (i, v_i) in freq_spectrum {
        sum_i2_vi += (i * i) as f64 * v_i as f64;
    }

    10000.0 * ((sum_i2_vi - n) / (n * n))
}

/// Computes Shannon entropy over character frequencies in a text.
pub fn shannon_entropy_chars(text: &str) -> f64 {
    crate::text_ops::text_entropy(text)
}

/// Computes Shannon entropy over word tokens.
pub fn shannon_entropy_words(words: &[String]) -> f64 {
    if words.is_empty() {
        return 0.0;
    }
    let mut counts = HashMap::new();
    for w in words {
        *counts.entry(w).or_insert(0usize) += 1;
    }
    let total = words.len() as f64;
    let mut entropy = 0.0;
    for &c in counts.values() {
        let p = c as f64 / total;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }
    entropy
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
    fn test_analyze_suite_1() {
        let sample = "The quick brown fox jumps over the lazy dog. This is test 1.";
        let stats = analyze_text(sample);
        assert!(stats.word_count > 0);
        assert!(stats.sentence_count >= 2);

        let ease = flesch_reading_ease(sample);
        assert!(ease > 0.0);

        let fk = flesch_kincaid_grade(sample);
        assert!(fk >= 0.0);

        let fog = gunning_fog_index(sample);
        assert!(fog >= 0.0);

        let cl = coleman_liau_index(sample);
        assert!(cl >= -20.0);

        let ari = automated_readability_index(sample);
        assert!(ari >= 0.0);

        let words: Vec<String> = sample.split_whitespace().map(|s| s.to_string()).collect();
        let h_ratio = hapax_legomena_ratio(&words);
        assert!(h_ratio > 0.0);

        let yule = yules_k_characteristic(&words);
        assert!(yule >= 0.0);

        let ent = shannon_entropy_words(&words);
        assert!(ent > 0.0);
    }
}
