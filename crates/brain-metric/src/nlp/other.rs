//! # Auxiliary NLP & Sequence Metrics
//!
//! METEOR-lite unigram overlap, Perplexity from log-probabilities, and Levenshtein edit distance.
#![allow(missing_docs)]

use std::collections::HashSet;

/// Computes METEOR-lite unigram harmonic F1 score with penalty.
pub fn meteor_score_lite(hypothesis: &[&str], reference: &[&str]) -> f64 {
    let hyp_set: HashSet<&str> = hypothesis.iter().copied().collect();
    let ref_set: HashSet<&str> = reference.iter().copied().collect();

    let matches = hyp_set.intersection(&ref_set).count();
    if matches == 0 { return 0.0; }

    let p = matches as f64 / hypothesis.len().max(1) as f64;
    let r = matches as f64 / reference.len().max(1) as f64;

    // Harmonic mean with beta=3 favoring recall: (10 * P * R) / (R + 9 * P)
    (10.0 * p * r) / (r + 9.0 * p)
}

/// Evaluates Perplexity = exp(-1/N * sum(log_prob)).
pub fn perplexity_score(log_probs: &[f64]) -> f64 {
    if log_probs.is_empty() { return 1.0; }
    let avg_log_prob = log_probs.iter().sum::<f64>() / log_probs.len() as f64;
    (-avg_log_prob).exp()
}

/// Computes dynamic programming Levenshtein character edit distance between two strings.
#[allow(clippy::needless_range_loop)]
pub fn edit_distance_levenshtein(s1: &str, s2: &str) -> usize {
    let c1: Vec<char> = s1.chars().collect();
    let c2: Vec<char> = s2.chars().collect();
    let (n1, n2) = (c1.len(), c2.len());

    let mut dp = vec![vec![0usize; n2 + 1]; n1 + 1];
    for i in 0..=n1 { dp[i][0] = i; }
    for j in 0..=n2 { dp[0][j] = j; }

    for i in 1..=n1 {
        for j in 1..=n2 {
            let cost = if c1[i - 1] == c2[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    dp[n1][n2]
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
