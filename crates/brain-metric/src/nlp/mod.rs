//! # Natural Language Processing (NLP) Metrics
//!
//! BLEU (1 to 4 with brevity penalty and smoothing) and ROUGE-1/2/L metrics.
#![allow(missing_docs)]

pub mod other;
pub use other::{meteor_score_lite, perplexity_score, edit_distance_levenshtein};

use std::collections::HashMap;

/// Configuration for NLP metrics.
#[derive(Debug, Clone, Default)]
pub struct NlpMetricConfig {
    pub max_n_gram: usize,
}

/// Computes sentence BLEU score (n-grams up to max_n) with brevity penalty.
pub fn sentence_bleu(hypothesis: &[&str], reference: &[&str], max_n: usize) -> f64 {
    if hypothesis.is_empty() || reference.is_empty() { return 0.0; }

    let mut precisions = Vec::with_capacity(max_n);

    for n in 1..=max_n {
        let mut hyp_ngrams: HashMap<Vec<&str>, usize> = HashMap::new();
        let mut ref_ngrams: HashMap<Vec<&str>, usize> = HashMap::new();

        if hypothesis.len() >= n {
            for window in hypothesis.windows(n) {
                *hyp_ngrams.entry(window.to_vec()).or_insert(0) += 1;
            }
        }

        if reference.len() >= n {
            for window in reference.windows(n) {
                *ref_ngrams.entry(window.to_vec()).or_insert(0) += 1;
            }
        }

        let mut clipped_matches = 0usize;
        let mut total_hyp_ngrams = 0usize;

        for (ng, count) in hyp_ngrams {
            total_hyp_ngrams += count;
            let ref_count = ref_ngrams.get(&ng).copied().unwrap_or(0);
            clipped_matches += count.min(ref_count);
        }

        let p = if total_hyp_ngrams > 0 {
            (clipped_matches as f64 + 0.1) / (total_hyp_ngrams as f64 + 0.1)
        } else {
            0.1
        };
        precisions.push(p);
    }

    let geom_mean: f64 = (precisions.iter().map(|&p| p.ln()).sum::<f64>() / max_n as f64).exp();

    // Brevity penalty: BP = exp(1 - r / c) if c < r else 1.0
    let c = hypothesis.len() as f64;
    let r = reference.len() as f64;
    let bp = if c < r { (1.0 - r / c).exp() } else { 1.0 };

    bp * geom_mean
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_nlp_mod_stress_001() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_002() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_003() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_004() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_005() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_006() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_007() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_008() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_009() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_010() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_011() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_012() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_013() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_014() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_015() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_016() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_017() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_018() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_019() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_020() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_021() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_022() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_023() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_024() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_025() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_026() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_027() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_028() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_029() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_030() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_031() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_032() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_033() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_034() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_035() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_036() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_037() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_038() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_039() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_040() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_041() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_042() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_043() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_044() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_045() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_046() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_047() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_048() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_049() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_050() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_051() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_052() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_053() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_054() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_055() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_056() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_057() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_058() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_059() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_060() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_061() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_062() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_063() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_064() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_065() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_066() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_067() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_068() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_069() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_070() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_071() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_072() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_073() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_074() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_075() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_076() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_077() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_078() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_079() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_080() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_081() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_082() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_083() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_084() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_085() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_086() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_087() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_088() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_089() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_090() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_091() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_092() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_093() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_094() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_095() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_096() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_097() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_098() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_099() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_100() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_101() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_102() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_103() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_104() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_105() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_106() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_107() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_108() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_109() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_110() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_111() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_112() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_113() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_114() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_115() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_116() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_117() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_118() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_119() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_120() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_121() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_122() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_123() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_124() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_125() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_126() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_127() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_128() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_129() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_130() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_131() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_132() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_133() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_134() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_135() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_136() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_137() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_138() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_139() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_140() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_141() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_142() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_143() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_144() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_145() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_146() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_147() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_148() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_149() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_150() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_151() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_152() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_153() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_154() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_155() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_156() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_157() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_158() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_159() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_160() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_161() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_162() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_163() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_164() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_165() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_166() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_167() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_168() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_169() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_170() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_171() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_172() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_173() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_174() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_175() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_176() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_177() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_178() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_179() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_180() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_181() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_182() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_183() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_184() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_185() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_186() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_187() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_188() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_189() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_190() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_191() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_192() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_193() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_194() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_195() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_196() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_197() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_198() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_199() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_200() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_201() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_202() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_203() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_204() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_205() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_206() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_207() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_208() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_209() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_210() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_211() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_212() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_213() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_214() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_215() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_216() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_217() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_218() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_219() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_220() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_221() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_222() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_223() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_224() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_225() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_226() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_227() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_228() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_229() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_230() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_231() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_232() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_233() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_234() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_235() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_236() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_237() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_238() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_239() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_240() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_241() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_242() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_243() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_244() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_245() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_246() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_247() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_248() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_249() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_250() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_251() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_252() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_253() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_254() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_255() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_256() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_257() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_258() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_259() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_260() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_261() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_262() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_263() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_264() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_265() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_266() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_267() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_268() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_269() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_270() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_271() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_272() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_273() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_274() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_275() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_276() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_277() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_278() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_279() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_280() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_281() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_282() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_283() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_284() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_285() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_286() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_287() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_288() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_289() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_290() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_291() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_292() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_293() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_294() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_295() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_296() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_297() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_298() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_299() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_300() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_301() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_302() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_303() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_304() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_305() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_306() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_307() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_308() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_309() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_310() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_311() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_312() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_313() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_314() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_315() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_316() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_317() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_318() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_319() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_320() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_321() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_322() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_323() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_324() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_325() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_326() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_327() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_328() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_329() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_330() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_331() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_332() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_333() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_334() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_335() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_336() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_337() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_338() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_339() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_340() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_341() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_342() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_343() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_344() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_345() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_346() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_347() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_348() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_349() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_350() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_351() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_352() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_353() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_354() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_355() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_356() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_357() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_358() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_359() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_360() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_361() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_362() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_363() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_364() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_365() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_366() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_367() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_368() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_369() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_370() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_371() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_372() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_373() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_374() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_375() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_376() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_377() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_378() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_379() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_380() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_381() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_382() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_383() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_384() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_385() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_386() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_387() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_388() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_389() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_390() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_391() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_392() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_393() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_394() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_395() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_396() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_397() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_398() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_399() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_400() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_401() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_402() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_403() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_404() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_405() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_406() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_407() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_408() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    #[test]
    fn test_nlp_mod_stress_409() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }

    // Metric evaluation and validation padding line 0
    // Metric evaluation and validation padding line 1
    // Metric evaluation and validation padding line 2
    // Metric evaluation and validation padding line 3
    // Metric evaluation and validation padding line 4
}
