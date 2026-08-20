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
    if rankings.is_empty() {
        return 0.0;
    }
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
    if top_k == 0 {
        return 0.0;
    }

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
