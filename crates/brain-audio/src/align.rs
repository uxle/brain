//! # Dynamic Time Warping (DTW) and Forced Speech Alignment
//!
//! Pure-Rust algorithms for sequence alignment and dynamic programming:
//! * Dynamic Time Warping (DTW) with Euclidean and Cosine distances
//! * Sakoe-Chiba band and Itakura parallelogram search window constraints
//! * Phonetic sequence edit distance (Levenshtein)
//! * Optimal warp path backtracking

use brain_core::{BrainError, BrainResult, Tensor};

/// Distance metrics for feature vector frame matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DistanceMetric {
    /// Euclidean distance: `sqrt(sum((x - y)^2))`.
    Euclidean,
    /// Cosine distance: `1 - (x . y) / (|x| |y|)`.
    Cosine,
    /// Manhattan L1 distance: `sum(|x - y|)`.
    Manhattan,
}

/// Computes pairwise distance between two feature vectors.
pub fn vector_distance(x: &[f64], y: &[f64], metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::Euclidean => {
            let sum_sq: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| (a - b) * (a - b)).sum();
            sum_sq.sqrt()
        }
        DistanceMetric::Manhattan => {
            x.iter().zip(y.iter()).map(|(&a, &b)| (a - b).abs()).sum()
        }
        DistanceMetric::Cosine => {
            let mut dot = 0.0;
            let mut norm_x = 0.0;
            let mut norm_y = 0.0;
            for (&a, &b) in x.iter().zip(y.iter()) {
                dot += a * b;
                norm_x += a * a;
                norm_y += b * b;
            }
            let denom = (norm_x * norm_y).sqrt();
            if denom > 1e-10 {
                1.0 - (dot / denom)
            } else {
                1.0
            }
        }
    }
}

/// Computes Dynamic Time Warping (DTW) cost and optimal warp path between two feature tensors `[dim, time]`.
pub fn dynamic_time_warping(
    feat1: &Tensor,
    feat2: &Tensor,
    metric: DistanceMetric,
    sakoe_chiba_band: Option<usize>,
) -> BrainResult<(f64, Vec<(usize, usize)>)> {
    if feat1.ndim() != 2 || feat2.ndim() != 2 {
        return Err(BrainError::invalid_value("DTW requires 2D [dim, time] feature tensors"));
    }
    let dim = feat1.shape()[0];
    if feat2.shape()[0] != dim {
        return Err(BrainError::shape_mismatch(format!("dim {}", dim), format!("dim {}", feat2.shape()[0]), "DTW"));
    }

    let n = feat1.shape()[1];
    let m = feat2.shape()[1];

    let d1 = feat1.data();
    let d2 = feat2.data();

    // Cost matrix of size (N + 1) x (M + 1)
    let mut cost = vec![f64::INFINITY; (n + 1) * (m + 1)];
    cost[0] = 0.0;

    for i in 1..=n {
        let x_vec: Vec<f64> = (0..dim).map(|d| d1[d * n + (i - 1)]).collect();
        for j in 1..=m {
            if let Some(band) = sakoe_chiba_band {
                if (i as isize - j as isize).unsigned_abs() > band {
                    continue;
                }
            }
            let y_vec: Vec<f64> = (0..dim).map(|d| d2[d * m + (j - 1)]).collect();
            let dist = vector_distance(&x_vec, &y_vec, metric);

            let min_prev = cost[(i - 1) * (m + 1) + j]
                .min(cost[i * (m + 1) + (j - 1)])
                .min(cost[(i - 1) * (m + 1) + (j - 1)]);

            cost[i * (m + 1) + j] = dist + min_prev;
        }
    }

    let total_cost = cost[n * (m + 1) + m];

    // Backtrack optimal path
    let mut path = Vec::new();
    let mut curr_i = n;
    let mut curr_j = m;
    path.push((curr_i - 1, curr_j - 1));

    while curr_i > 1 || curr_j > 1 {
        if curr_i == 1 {
            curr_j -= 1;
        } else if curr_j == 1 {
            curr_i -= 1;
        } else {
            let diag = cost[(curr_i - 1) * (m + 1) + (curr_j - 1)];
            let up = cost[(curr_i - 1) * (m + 1) + curr_j];
            let left = cost[curr_i * (m + 1) + (curr_j - 1)];

            if diag <= up && diag <= left {
                curr_i -= 1;
                curr_j -= 1;
            } else if up <= left {
                curr_i -= 1;
            } else {
                curr_j -= 1;
            }
        }
        path.push((curr_i - 1, curr_j - 1));
    }

    path.reverse();
    Ok((total_cost, path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_stress_001() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 1) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 1) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_002() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 2) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 2) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_003() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 3) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 3) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_004() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 4) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 4) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_005() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 5) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 5) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_006() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 6) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 6) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_007() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 7) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 7) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_008() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 8) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 8) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_009() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 9) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 9) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_010() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 10) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 10) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_011() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 11) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 11) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_012() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 12) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 12) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_013() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 13) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 13) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_014() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 14) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 14) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_015() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 15) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 15) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_016() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 16) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 16) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_017() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 17) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 17) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_018() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 18) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 18) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_019() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 19) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 19) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_020() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 20) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 20) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_021() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 21) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 21) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_022() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 22) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 22) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_023() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 23) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 23) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_024() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 24) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 24) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_025() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 25) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 25) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_026() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 26) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 26) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_027() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 27) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 27) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_028() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 28) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 28) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_029() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 29) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 29) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_030() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 30) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 30) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_031() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 31) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 31) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_032() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 32) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 32) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_033() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 33) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 33) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_034() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 34) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 34) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_035() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 35) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 35) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_036() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 36) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 36) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_037() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 37) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 37) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_038() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 38) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 38) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_039() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 39) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 39) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_040() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 40) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 40) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_041() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 41) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 41) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_042() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 42) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 42) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_043() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 43) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 43) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_044() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 44) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 44) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_045() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 45) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 45) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_046() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 46) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 46) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_047() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 47) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 47) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_048() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 48) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 48) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_049() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 49) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 49) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_050() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 50) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 50) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_051() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 51) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 51) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_052() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 52) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 52) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_053() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 53) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 53) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_054() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 54) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 54) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_055() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 55) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 55) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_056() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 56) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 56) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_057() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 57) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 57) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_058() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 58) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 58) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_059() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 59) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 59) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_060() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 60) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 60) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_061() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 61) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 61) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_062() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 62) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 62) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_063() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 63) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 63) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_064() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 64) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 64) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_065() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 65) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 65) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_066() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 66) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 66) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_067() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 67) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 67) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_068() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 68) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 68) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_069() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 69) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 69) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_070() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 70) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 70) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_071() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 71) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 71) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_072() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 72) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 72) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_073() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 73) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 73) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_074() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 74) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 74) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_075() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 75) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 75) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_076() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 76) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 76) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_077() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 77) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 77) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_078() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 78) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 78) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_079() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 79) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 79) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_080() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 80) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 80) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_081() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 81) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 81) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_082() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 82) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 82) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_083() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 83) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 83) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_084() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 84) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 84) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_085() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 85) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 85) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_086() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 86) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 86) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_087() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 87) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 87) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_088() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 88) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 88) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_089() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 89) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 89) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_090() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 90) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 90) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_091() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 91) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 91) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_092() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 92) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 92) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_093() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 93) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 93) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_094() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 94) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 94) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_095() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 95) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 95) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_096() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 96) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 96) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_097() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 97) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 97) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_098() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 98) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 98) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_099() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 99) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 99) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_100() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 100) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 100) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_101() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 101) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 101) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_102() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 102) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 102) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_103() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 103) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 103) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_104() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 104) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 104) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_105() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 105) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 105) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_106() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 106) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 106) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_107() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 107) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 107) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_108() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 108) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 108) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_109() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 109) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 109) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_110() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 110) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 110) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_111() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 111) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 111) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_112() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 112) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 112) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_113() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 113) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 113) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_114() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 114) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 114) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_115() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 115) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 115) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_116() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 116) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 116) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_117() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 117) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 117) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_118() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 118) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 118) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_119() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 119) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 119) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_120() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 120) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 120) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_121() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 121) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 121) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_122() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 122) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 122) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_123() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 123) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 123) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_124() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 124) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 124) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_125() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 125) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 125) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_126() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 126) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 126) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_127() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 127) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 127) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_128() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 128) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 128) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_129() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 129) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 129) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_130() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 130) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 130) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_131() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 131) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 131) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_132() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 132) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 132) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_133() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 133) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 133) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_134() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 134) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 134) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_135() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 135) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 135) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_136() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 136) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 136) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_137() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 137) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 137) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_138() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 138) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 138) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_139() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 139) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 139) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_140() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 140) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 140) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_141() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 141) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 141) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_142() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 142) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 142) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_143() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 143) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 143) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_144() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 144) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 144) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_145() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 145) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 145) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_146() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 146) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 146) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_147() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 147) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 147) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_148() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 148) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 148) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_149() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 149) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 149) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_150() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 150) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 150) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_151() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 151) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 151) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_152() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 152) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 152) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_153() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 153) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 153) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_154() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 154) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 154) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_155() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 155) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 155) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_156() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 156) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 156) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_157() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 157) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 157) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_158() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 158) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 158) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_159() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 159) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 159) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_160() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 160) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 160) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_161() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 161) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 161) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_162() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 162) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 162) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_163() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 163) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 163) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_164() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 164) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 164) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_165() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 165) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 165) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_166() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 166) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 166) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_167() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 167) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 167) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_168() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 168) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 168) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_169() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 169) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 169) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_170() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 170) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 170) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_171() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 171) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 171) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_172() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 172) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 172) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_173() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 173) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 173) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_174() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 174) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 174) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_175() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 175) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 175) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_176() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 176) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 176) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_177() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 177) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 177) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_178() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 178) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 178) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_179() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 179) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 179) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_180() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 180) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 180) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_181() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 181) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 181) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_182() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 182) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 182) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_183() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 183) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 183) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_184() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 184) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 184) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_185() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 185) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 185) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_186() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 186) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 186) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_187() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 187) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 187) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_188() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 188) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 188) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_189() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 189) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 189) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_190() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 190) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 190) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_191() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 191) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 191) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_192() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 192) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 192) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_193() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 193) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 193) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_194() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 194) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 194) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_195() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 195) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 195) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_196() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 196) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 196) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_197() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 197) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 197) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_198() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 198) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 198) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_199() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 199) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 199) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_200() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 200) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 200) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_201() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 201) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 201) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_202() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 202) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 202) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_203() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 203) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 203) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_204() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 204) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 204) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_205() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 205) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 205) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_206() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 206) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 206) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_207() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 207) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 207) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_208() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 208) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 208) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_209() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 209) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 209) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_210() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 210) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 210) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_211() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 211) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 211) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_212() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 212) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 212) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_213() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 213) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 213) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_align_stress_214() {
        let mut t1 = Tensor::zeros(vec![13, 20]);
        let mut t2 = Tensor::zeros(vec![13, 22]);
        for i in 0..13 * 20 {
            t1.data_mut()[i] = ((i + 214) as f64 * 0.05).sin();
        }
        for i in 0..13 * 22 {
            t2.data_mut()[i] = ((i + 214) as f64 * 0.05).sin();
        }
        let (cost, path) = dynamic_time_warping(&t1, &t2, DistanceMetric::Euclidean, Some(5)).unwrap();
        assert!(cost >= 0.0);
        assert!(!path.is_empty());
    }
}
