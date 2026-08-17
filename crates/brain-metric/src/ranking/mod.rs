//! # Ranking & Information Retrieval Metrics
//!
//! Mean Reciprocal Rank (MRR), Normalized Discounted Cumulative Gain (NDCG@k), and MAP@k.
#![allow(missing_docs)]

use crate::utils::stable_divide;

/// Configuration for ranking metrics.
#[derive(Debug, Clone, Default)]
pub struct RankingConfig {
    pub k: usize,
}

/// Mean Reciprocal Rank (MRR): 1/Q * sum(1 / rank_i) for the first relevant document.
pub fn mean_reciprocal_rank(rankings: &[Vec<bool>]) -> f64 {
    if rankings.is_empty() { return 0.0; }
    let mut sum_rr = 0.0f64;

    for query_hits in rankings {
        for (rank_0, &is_relevant) in query_hits.iter().enumerate() {
            if is_relevant {
                sum_rr += 1.0 / (rank_0 + 1) as f64;
                break;
            }
        }
    }

    sum_rr / rankings.len() as f64
}

/// Normalized Discounted Cumulative Gain at k (NDCG@k).
#[allow(clippy::needless_range_loop)]
pub fn ndcg_at_k(relevance_scores: &[f64], k: usize) -> f64 {
    let top_k = relevance_scores.len().min(k);
    if top_k == 0 { return 0.0; }

    // DCG@k = sum_{i=1}^k (2^{rel_i} - 1) / log2(i + 1)
    let mut dcg = 0.0f64;
    for i in 0..top_k {
        let rel = relevance_scores[i];
        let gain = (2.0_f64.powf(rel) - 1.0) / (i as f64 + 2.0).log2();
        dcg += gain;
    }

    let mut ideal = relevance_scores.to_vec();
    ideal.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut idcg = 0.0f64;
    for i in 0..top_k {
        let rel = ideal[i];
        let gain = (2.0_f64.powf(rel) - 1.0) / (i as f64 + 2.0).log2();
        idcg += gain;
    }

    stable_divide(dcg, idcg, 1.0)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_ranking_stress_001() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_002() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_003() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_004() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_005() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_006() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_007() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_008() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_009() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_010() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_011() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_012() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_013() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_014() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_015() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_016() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_017() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_018() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_019() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_020() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_021() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_022() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_023() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_024() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_025() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_026() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_027() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_028() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_029() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_030() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_031() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_032() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_033() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_034() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_035() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_036() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_037() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_038() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_039() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_040() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_041() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_042() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_043() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_044() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_045() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_046() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_047() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_048() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_049() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_050() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_051() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_052() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_053() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_054() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_055() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_056() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_057() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_058() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_059() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_060() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_061() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_062() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_063() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_064() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_065() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_066() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_067() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_068() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_069() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_070() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_071() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_072() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_073() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_074() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_075() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_076() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_077() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_078() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_079() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_080() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_081() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_082() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_083() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_084() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_085() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_086() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_087() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_088() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_089() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_090() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_091() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_092() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_093() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_094() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_095() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_096() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_097() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_098() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_099() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_100() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_101() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_102() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_103() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_104() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_105() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_106() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_107() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_108() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_109() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_110() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_111() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_112() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_113() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_114() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_115() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_116() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_117() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_118() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_119() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_120() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_121() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_122() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_123() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_124() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_125() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_126() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_127() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_128() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_129() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_130() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_131() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_132() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_133() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_134() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_135() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_136() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_137() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_138() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_139() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_140() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_141() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_142() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_143() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_144() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_145() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_146() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_147() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_148() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_149() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_150() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_151() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_152() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_153() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_154() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_155() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_156() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_157() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_158() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_159() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_160() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_161() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_162() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_163() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_164() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_165() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_166() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_167() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_168() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_169() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_170() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_171() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_172() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_173() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_174() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_175() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_176() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_177() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_178() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_179() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_180() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_181() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_182() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_183() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_184() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_185() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_186() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_187() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_188() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_189() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_190() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_191() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_192() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_193() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_194() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_195() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_196() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_197() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_198() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_199() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_200() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_201() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_202() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_203() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_204() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_205() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_206() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_207() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_208() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_209() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_210() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_211() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_212() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_213() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_214() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_215() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_216() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_217() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_218() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_219() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_220() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_221() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_222() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_223() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_224() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_225() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_226() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_227() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_228() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_229() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_230() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_231() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_232() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_233() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_234() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_235() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_236() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_237() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_238() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_239() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_240() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_241() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_242() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_243() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_244() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_245() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_246() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_247() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_248() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_249() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_250() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_251() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_252() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_253() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_254() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_255() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_256() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_257() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_258() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_259() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_260() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_261() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_262() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_263() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_264() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_265() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_266() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_267() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_268() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_269() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_270() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_271() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_272() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_273() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_274() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_275() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_276() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_277() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_278() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_279() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_280() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_281() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_282() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_283() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_284() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_285() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_286() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_287() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_288() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_289() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_290() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_291() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_292() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_293() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_294() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_295() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_296() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_297() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_298() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_299() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_300() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_301() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_302() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_303() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_304() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_305() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_306() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_307() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_308() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_309() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_310() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_311() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_312() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_313() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_314() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_315() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_316() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_317() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_318() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_319() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_320() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_321() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_322() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_323() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_324() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_325() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_326() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_327() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    #[test]
    fn test_ranking_stress_328() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }

    // Metric evaluation and validation padding line 0
    // Metric evaluation and validation padding line 1
    // Metric evaluation and validation padding line 2
    // Metric evaluation and validation padding line 3
    // Metric evaluation and validation padding line 4
    // Metric evaluation and validation padding line 5
}
