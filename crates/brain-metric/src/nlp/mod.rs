//! # Natural Language Processing (NLP) Metrics
//!
//! BLEU (1 to 4 with brevity penalty and smoothing) and ROUGE-1/2/L metrics.
#![allow(missing_docs)]

pub mod other;
pub use other::{edit_distance_levenshtein, meteor_score_lite, perplexity_score};

use std::collections::HashMap;

/// Configuration for NLP metrics.
#[derive(Debug, Clone, Default)]
pub struct NlpMetricConfig {
    pub max_n_gram: usize,
}

/// Computes sentence BLEU score (n-grams up to max_n) with brevity penalty.
pub fn sentence_bleu(hypothesis: &[&str], reference: &[&str], max_n: usize) -> f64 {
    if hypothesis.is_empty() || reference.is_empty() {
        return 0.0;
    }

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
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
