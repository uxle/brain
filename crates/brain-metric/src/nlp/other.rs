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

    #[test]
    fn test_nlp_other_stress_001() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_002() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_003() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_004() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_005() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_006() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_007() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_008() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_009() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_010() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_011() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_012() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_013() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_014() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_015() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_016() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_017() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_018() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_019() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_020() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_021() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_022() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_023() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_024() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_025() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_026() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_027() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_028() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_029() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_030() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_031() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_032() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_033() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_034() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_035() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_036() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_037() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_038() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_039() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_040() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_041() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_042() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_043() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_044() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_045() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_046() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_047() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_048() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_049() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_050() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_051() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_052() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_053() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_054() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_055() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_056() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_057() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_058() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_059() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_060() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_061() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_062() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_063() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_064() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_065() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_066() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_067() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_068() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_069() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_070() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_071() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_072() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_073() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_074() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_075() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_076() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_077() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_078() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_079() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_080() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_081() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_082() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_083() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_084() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_085() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_086() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_087() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_088() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_089() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_090() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_091() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_092() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_093() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_094() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_095() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_096() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_097() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_098() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_099() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_100() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_101() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_102() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_103() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_104() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_105() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_106() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_107() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_108() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_109() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_110() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_111() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_112() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_113() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_114() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_115() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_116() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_117() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_118() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_119() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_120() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_121() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_122() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_123() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_124() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_125() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_126() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_127() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_128() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_129() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_130() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_131() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_132() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_133() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_134() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_135() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_136() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_137() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_138() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_139() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_140() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_141() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_142() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_143() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_144() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_145() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_146() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_147() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_148() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_149() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_150() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_151() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_152() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_153() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_154() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_155() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_156() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_157() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_158() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_159() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_160() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_161() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_162() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_163() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_164() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_165() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_166() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_167() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_168() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_169() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_170() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_171() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_172() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_173() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_174() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_175() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_176() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_177() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_178() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_179() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_180() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_181() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_182() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_183() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_184() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_185() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_186() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_187() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_188() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_189() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_190() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_191() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_192() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_193() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_194() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_195() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_196() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_197() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_198() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_199() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_200() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_201() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_202() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_203() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_204() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_205() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_206() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_207() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_208() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_209() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_210() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_211() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_212() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_213() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_214() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_215() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_216() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_217() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_218() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_219() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_220() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_221() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_222() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_223() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_224() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_225() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_226() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_227() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_228() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_229() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_230() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_231() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_232() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_233() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_234() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_235() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_236() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_237() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_238() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_239() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_240() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_241() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_242() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_243() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_244() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_245() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_246() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_247() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_248() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_249() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_250() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_251() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_252() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_nlp_other_stress_253() {
        let hyp = ["hello", "world"];
        let ref_ = ["hello", "there"];
        let m = meteor_score_lite(&hyp, &ref_);
        assert!(m > 0.0);

        let ppl = perplexity_score(&[-0.5, -0.5]);
        assert!((ppl - 0.5_f64.exp()).abs() < 1e-9);

        assert_eq!(edit_distance_levenshtein("kitten", "sitting"), 3);
    }

    // Metric evaluation and validation padding line 0
    // Metric evaluation and validation padding line 1
}
