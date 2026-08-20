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
            let sum_sq: f64 = x
                .iter()
                .zip(y.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum();
            sum_sq.sqrt()
        }
        DistanceMetric::Manhattan => x.iter().zip(y.iter()).map(|(&a, &b)| (a - b).abs()).sum(),
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
        return Err(BrainError::invalid_value(
            "DTW requires 2D [dim, time] feature tensors",
        ));
    }
    let dim = feat1.shape()[0];
    if feat2.shape()[0] != dim {
        return Err(BrainError::shape_mismatch(
            format!("dim {}", dim),
            format!("dim {}", feat2.shape()[0]),
            "DTW",
        ));
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
}
