//! # Federated Aggregation Algorithms
//!
//! FedAvg, FedProx, Coordinate-wise Median, and Byzantine-Robust Trimmed-Mean aggregations (McMahan et al., Yin et al.).

use crate::core::ModelDelta;
use brain_core::Tensor;

/// Supported aggregation algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AggregationAlgorithm {
    #[default]
    FedAvg,
    FedProx,
    TrimmedMean,
    Median,
}

/// Aggregates model deltas using standard FedAvg (sample-weighted average).
pub fn fed_avg_aggregate(deltas: &[ModelDelta]) -> Vec<Tensor> {
    if deltas.is_empty() {
        return vec![];
    }
    let total_samples: usize = deltas.iter().map(|d| d.num_samples).sum();
    if total_samples == 0 {
        return deltas[0].weights.clone();
    }
    let n_layers = deltas[0].weights.len();
    (0..n_layers)
        .map(|l| {
            let mut acc = Tensor::zeros(deltas[0].weights[l].shape().to_vec());
            for d in deltas {
                let w = Tensor::scalar(d.num_samples as f64 / total_samples as f64);
                acc = &acc + &(&d.weights[l] * &w);
            }
            acc
        })
        .collect()
}

/// Byzantine-robust Coordinate-wise Trimmed-Mean aggregation.
/// Trims the highest and lowest `trim_fraction` of client values per parameter coordinate.
pub fn trimmed_mean_aggregate(deltas: &[ModelDelta], trim_fraction: f64) -> Vec<Tensor> {
    if deltas.is_empty() {
        return vec![];
    }
    let n_clients = deltas.len();
    if n_clients <= 2 {
        return fed_avg_aggregate(deltas);
    }

    let n_layers = deltas[0].weights.len();
    let k = ((n_clients as f64) * trim_fraction.clamp(0.0, 0.45)).floor() as usize;

    (0..n_layers)
        .map(|l| {
            let shape = deltas[0].weights[l].shape().to_vec();
            let numel = deltas[0].weights[l].numel();
            let mut agg_data = Vec::with_capacity(numel);

            for i in 0..numel {
                let mut vals: Vec<f64> = deltas.iter().map(|d| d.weights[l].data()[i]).collect();
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let remaining = &vals[k..n_clients - k];
                let sum: f64 = remaining.iter().sum();
                let mean = sum / (remaining.len() as f64);
                agg_data.push(mean);
            }

            Tensor::from_slice(&agg_data, shape)
        })
        .collect()
}

/// Coordinate-wise Median aggregation for maximum Byzantine resilience.
pub fn median_aggregate(deltas: &[ModelDelta]) -> Vec<Tensor> {
    if deltas.is_empty() {
        return vec![];
    }
    let n_clients = deltas.len();
    let n_layers = deltas[0].weights.len();

    (0..n_layers)
        .map(|l| {
            let shape = deltas[0].weights[l].shape().to_vec();
            let numel = deltas[0].weights[l].numel();
            let mut agg_data = Vec::with_capacity(numel);

            for i in 0..numel {
                let mut vals: Vec<f64> = deltas.iter().map(|d| d.weights[l].data()[i]).collect();
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let median = if n_clients % 2 == 1 {
                    vals[n_clients / 2]
                } else {
                    (vals[n_clients / 2 - 1] + vals[n_clients / 2]) / 2.0
                };
                agg_data.push(median);
            }

            Tensor::from_slice(&agg_data, shape)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fed_avg_weighted() {
        let d1 = ModelDelta::new(0, vec![Tensor::from_slice(&[10.0, 20.0], vec![2])], 100);
        let d2 = ModelDelta::new(1, vec![Tensor::from_slice(&[30.0, 40.0], vec![2])], 300);

        // 100/400 * 10 + 300/400 * 30 = 2.5 + 22.5 = 25.0
        // 100/400 * 20 + 300/400 * 40 = 5.0 + 30.0 = 35.0
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg[0].data(), &[25.0, 35.0]);
    }

    #[test]
    fn test_trimmed_mean_and_median_byzantine_resilience() {
        // 5 clients, client 4 is an adversarial outlier (+10000.0)
        let mut deltas = Vec::new();
        for i in 0..4 {
            deltas.push(ModelDelta::new(
                i,
                vec![Tensor::from_slice(&[(i + 1) as f64 * 10.0], vec![1])],
                10,
            ));
        }
        deltas.push(ModelDelta::new(
            4,
            vec![Tensor::from_slice(&[10_000.0], vec![1])],
            10,
        ));

        // Median of [10, 20, 30, 40, 10000] is 30.0
        let med = median_aggregate(&deltas);
        assert_eq!(med[0].data()[0], 30.0);

        // Trimmed mean (trim 1 on each end) -> average of [20, 30, 40] = 30.0
        let trim = trimmed_mean_aggregate(&deltas, 0.2);
        assert_eq!(trim[0].data()[0], 30.0);
    }
}
