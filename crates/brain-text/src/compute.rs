//! # NLP Evaluation Metrics: BLEU, ROUGE, chrF, WER, CER, and Perplexity
//!
//! Benchmark evaluation metrics for text generation, translation, summarization, and language modeling.
#![allow(
    missing_docs,
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_comparisons,
    clippy::all
)]

use std::collections::HashMap;

/// Computes sentence-level BLEU-N score with optional smoothing.
pub fn bleu_score(reference: &[String], candidate: &[String], max_n: usize, smooth: bool) -> f64 {
    if candidate.is_empty() || reference.is_empty() || max_n == 0 {
        return 0.0;
    }

    let r_len = reference.len() as f64;
    let c_len = candidate.len() as f64;

    // Brevity penalty
    let bp = if c_len > r_len {
        1.0
    } else {
        (1.0 - (r_len / c_len)).exp()
    };

    let mut log_sum = 0.0;

    for n in 1..=max_n {
        let ref_ngrams = crate::text_ops::ngram_freq(reference, n);
        let cand_ngrams = crate::text_ops::ngram_freq(candidate, n);

        let mut clipped_matches = 0usize;
        let mut total_cand = 0usize;

        for (gram, &c_count) in &cand_ngrams {
            total_cand += c_count;
            let r_count = ref_ngrams.get(gram).copied().unwrap_or(0);
            clipped_matches += c_count.min(r_count);
        }

        let p_n = if total_cand == 0 {
            if smooth {
                1e-5
            } else {
                0.0
            }
        } else if clipped_matches == 0 {
            if smooth {
                1.0 / (2.0 * total_cand as f64)
            } else {
                1e-9
            }
        } else {
            clipped_matches as f64 / total_cand as f64
        };

        log_sum += p_n.ln() / (max_n as f64);
    }

    bp * log_sum.exp()
}

/// Computes corpus-level BLEU score across candidate and reference pairs.
pub fn corpus_bleu(references: &[Vec<String>], candidates: &[Vec<String>], max_n: usize) -> f64 {
    if references.len() != candidates.len() || references.is_empty() {
        return 0.0;
    }
    let total: f64 = references
        .iter()
        .zip(candidates.iter())
        .map(|(r, c)| bleu_score(r, c, max_n, true))
        .sum();
    total / references.len() as f64
}

/// Computes ROUGE-N precision, recall, and F1 score for n-grams.
pub fn rouge_n(reference: &[String], candidate: &[String], n: usize) -> (f64, f64, f64) {
    if reference.is_empty() || candidate.is_empty() || n == 0 {
        return (0.0, 0.0, 0.0);
    }

    let ref_ngrams = crate::text_ops::ngram_freq(reference, n);
    let cand_ngrams = crate::text_ops::ngram_freq(candidate, n);

    let mut overlap = 0usize;
    let total_ref: usize = ref_ngrams.values().sum();
    let total_cand: usize = cand_ngrams.values().sum();

    for (gram, &c_count) in &cand_ngrams {
        let r_count = ref_ngrams.get(gram).copied().unwrap_or(0);
        overlap += c_count.min(r_count);
    }

    let precision = if total_cand > 0 {
        overlap as f64 / total_cand as f64
    } else {
        0.0
    };
    let recall = if total_ref > 0 {
        overlap as f64 / total_ref as f64
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        (2.0 * precision * recall) / (precision + recall)
    } else {
        0.0
    };

    (precision, recall, f1)
}

/// Computes ROUGE-L based on Longest Common Subsequence (LCS).
pub fn rouge_l(reference: &[String], candidate: &[String]) -> (f64, f64, f64) {
    let m = reference.len();
    let n = candidate.len();
    if m == 0 || n == 0 {
        return (0.0, 0.0, 0.0);
    }

    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            if reference[i - 1] == candidate[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    let lcs_len = dp[m][n] as f64;
    let precision = lcs_len / n as f64;
    let recall = lcs_len / m as f64;
    let f1 = if precision + recall > 0.0 {
        (2.0 * precision * recall) / (precision + recall)
    } else {
        0.0
    };

    (precision, recall, f1)
}

/// Computes character n-gram F-score (chrF metric).
pub fn chrf_score(reference: &str, candidate: &str, n: usize, beta: f64) -> f64 {
    let ref_grams = crate::text_ops::shingles(reference, n);
    let cand_grams = crate::text_ops::shingles(candidate, n);

    if ref_grams.is_empty() && cand_grams.is_empty() {
        return 1.0;
    }
    if ref_grams.is_empty() || cand_grams.is_empty() {
        return 0.0;
    }

    let intersection = ref_grams.intersection(&cand_grams).count() as f64;
    let precision = intersection / cand_grams.len() as f64;
    let recall = intersection / ref_grams.len() as f64;

    let beta_sq = beta * beta;
    if precision + recall == 0.0 {
        0.0
    } else {
        (1.0 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
    }
}

/// Exact match binary score (1.0 if identical, 0.0 otherwise).
pub fn exact_match_score(reference: &str, candidate: &str) -> f64 {
    if reference.trim() == candidate.trim() {
        1.0
    } else {
        0.0
    }
}

/// F1 token overlap score.
pub fn f1_token_score(reference: &[String], candidate: &[String]) -> f64 {
    let (_p, _r, f1) = rouge_n(reference, candidate, 1);
    f1
}

/// Computes language model perplexity from average cross-entropy losses: $PPL = \exp(\frac{1}{N}\sum Loss_i)$.
pub fn perplexity(cross_entropy_losses: &[f64]) -> f64 {
    if cross_entropy_losses.is_empty() {
        return 1.0;
    }
    let avg_loss: f64 =
        cross_entropy_losses.iter().sum::<f64>() / cross_entropy_losses.len() as f64;
    avg_loss.exp()
}

/// Computes bits per character (BPC) from cross-entropy loss.
pub fn bits_per_character(cross_entropy_loss: f64) -> f64 {
    cross_entropy_loss / 2.0f64.ln()
}

/// Computes Word Error Rate (WER) using Levenshtein distance on words.
pub fn word_error_rate(reference: &[String], hypothesis: &[String]) -> f64 {
    let r_str = reference.join(" ");
    let h_str = hypothesis.join(" ");
    let dist = crate::utils::levenshtein_distance(&r_str, &h_str);
    if reference.is_empty() {
        if hypothesis.is_empty() {
            0.0
        } else {
            1.0
        }
    } else {
        dist as f64 / reference.len().max(1) as f64
    }
}

/// Computes Character Error Rate (CER) using Levenshtein distance on characters.
pub fn character_error_rate(reference: &str, hypothesis: &str) -> f64 {
    let dist = crate::utils::levenshtein_distance(reference, hypothesis);
    if reference.is_empty() {
        if hypothesis.is_empty() {
            0.0
        } else {
            1.0
        }
    } else {
        dist as f64 / reference.chars().count().max(1) as f64
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
    fn test_compute_metrics_1() {
        let r = vec![
            "the".to_string(),
            "cat".to_string(),
            "sat".to_string(),
            "on_1".to_string(),
            "mat".to_string(),
        ];
        let c = vec![
            "the".to_string(),
            "cat".to_string(),
            "sat".to_string(),
            "on_1".to_string(),
            "mat".to_string(),
        ];
        let bleu = bleu_score(&r, &c, 4, false);
        assert!((bleu - 1.0).abs() < 1e-4);

        let (p, rec, f1) = rouge_n(&r, &c, 2);
        assert!((f1 - 1.0).abs() < 1e-4);

        let (lp, lr, lf1) = rouge_l(&r, &c);
        assert!((lf1 - 1.0).abs() < 1e-4);

        let chrf = chrf_score("hello 1", "hello 1", 3, 2.0);
        assert_eq!(chrf, 1.0);

        let em = exact_match_score("test_1", "test_1");
        assert_eq!(em, 1.0);

        let ppl = perplexity(&[0.0]);
        assert!((ppl - 1.0).abs() < 1e-5);

        let bpc = bits_per_character(1.0);
        assert!(bpc > 0.0);

        let wer = word_error_rate(&r, &c);
        assert_eq!(wer, 0.0);
    }
}
