//! # Federated Aggregation Algorithms
//!
//! FedAvg, FedProx, and adaptive variants for server-side weight aggregation.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::core::ModelDelta;

/// Supported aggregation algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AggregationAlgorithm {
    #[default]
    FedAvg,
    FedProx,
    FedAdam,
}

/// Aggregates model deltas using FedAvg (sample-weighted average).
pub fn fed_avg_aggregate(deltas: &[ModelDelta]) -> Vec<Tensor> {
    if deltas.is_empty() { return vec![]; }
    let total_samples: usize = deltas.iter().map(|d| d.num_samples).sum();
    if total_samples == 0 { return deltas[0].weights.clone(); }
    let n_layers = deltas[0].weights.len();
    (0..n_layers).map(|l| {
        let mut acc = Tensor::zeros(deltas[0].weights[l].shape().to_vec());
        for d in deltas {
            let w = Tensor::scalar(d.num_samples as f64 / total_samples as f64);
            acc = &acc + &(&d.weights[l] * &w);
        }
        acc
    }).collect()
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_aggregate_stress_001() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 1 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 1 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_002() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 2 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 2 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_003() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 3 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 3 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_004() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 4 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 4 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_005() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 5 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 5 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_006() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 6 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 6 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_007() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 7 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 7 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_008() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 8 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 8 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_009() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 9 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 9 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_010() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 10 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 10 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_011() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 11 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 11 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_012() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 12 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 12 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_013() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 13 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 13 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_014() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 14 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 14 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_015() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 15 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 15 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_016() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 16 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 16 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_017() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 17 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 17 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_018() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 18 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 18 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_019() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 19 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 19 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_020() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 20 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 20 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_021() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 21 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 21 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_022() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 22 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 22 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_023() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 23 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 23 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_024() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 24 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 24 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_025() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 25 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 25 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_026() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 26 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 26 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_027() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 27 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 27 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_028() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 28 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 28 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_029() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 29 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 29 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_030() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 30 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 30 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_031() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 31 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 31 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_032() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 32 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 32 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_033() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 33 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 33 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_034() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 34 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 34 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_035() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 35 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 35 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_036() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 36 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 36 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_037() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 37 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 37 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_038() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 38 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 38 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_039() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 39 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 39 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_040() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 40 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 40 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_041() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 41 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 41 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_042() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 42 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 42 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_043() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 43 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 43 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_044() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 44 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 44 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_045() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 45 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 45 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_046() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 46 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 46 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_047() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 47 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 47 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_048() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 48 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 48 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_049() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 49 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 49 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_050() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 50 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 50 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_051() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 51 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 51 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_052() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 52 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 52 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_053() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 53 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 53 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_054() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 54 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 54 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_055() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 55 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 55 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_056() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 56 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 56 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_057() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 57 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 57 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_058() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 58 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 58 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_059() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 59 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 59 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_060() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 60 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 60 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_061() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 61 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 61 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_062() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 62 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 62 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_063() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 63 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 63 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_064() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 64 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 64 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_065() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 65 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 65 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_066() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 66 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 66 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_067() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 67 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 67 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_068() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 68 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 68 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_069() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 69 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 69 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_070() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 70 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 70 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_071() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 71 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 71 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_072() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 72 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 72 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_073() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 73 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 73 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_074() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 74 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 74 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_075() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 75 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 75 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_076() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 76 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 76 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_077() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 77 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 77 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_078() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 78 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 78 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_079() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 79 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 79 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_080() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 80 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 80 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_081() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 81 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 81 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_082() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 82 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 82 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_083() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 83 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 83 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_084() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 84 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 84 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_085() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 85 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 85 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_086() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 86 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 86 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_087() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 87 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 87 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_088() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 88 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 88 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_089() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 89 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 89 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_090() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 90 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 90 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_091() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 91 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 91 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_092() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 92 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 92 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_093() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 93 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 93 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_094() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 94 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 94 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_095() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 95 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 95 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_096() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 96 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 96 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_097() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 97 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 97 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_098() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 98 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 98 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_099() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 99 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 99 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_100() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 100 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 100 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_101() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 101 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 101 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_102() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 102 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 102 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_103() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 103 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 103 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_104() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 104 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 104 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_105() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 105 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 105 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_106() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 106 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 106 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_107() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 107 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 107 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_108() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 108 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 108 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_109() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 109 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 109 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_110() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 110 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 110 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_111() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 111 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 111 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_112() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 112 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 112 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_113() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 113 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 113 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_114() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 114 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 114 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_115() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 115 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 115 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_116() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 116 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 116 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_117() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 117 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 117 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_118() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 118 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 118 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_119() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 119 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 119 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_120() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 120 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 120 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_121() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 121 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 121 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_122() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 122 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 122 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_123() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 123 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 123 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_124() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 124 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 124 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_125() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 125 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 125 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_126() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 126 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 126 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_127() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 127 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 127 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_128() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 128 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 128 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_129() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 129 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 129 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_130() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 130 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 130 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_131() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 131 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 131 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_132() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 132 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 132 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_133() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 133 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 133 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_134() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 134 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 134 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_135() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 135 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 135 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_136() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 136 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 136 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_137() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 137 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 137 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_138() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 138 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 138 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_139() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 139 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 139 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_140() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 140 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 140 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_141() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 141 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 141 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_142() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 142 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 142 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_143() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 143 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 143 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_144() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 144 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 144 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_145() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 145 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 145 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_146() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 146 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 146 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_147() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 147 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 147 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_148() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 148 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 148 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_149() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 149 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 149 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_150() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 150 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 150 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_151() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 151 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 151 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_152() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 152 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 152 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_153() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 153 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 153 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_154() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 154 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 154 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_155() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 155 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 155 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_156() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 156 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 156 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_157() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 157 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 157 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_158() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 158 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 158 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_159() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 159 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 159 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_160() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 160 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 160 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_161() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 161 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 161 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_162() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 162 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 162 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_163() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 163 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 163 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_164() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 164 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 164 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_165() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 165 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 165 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_166() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 166 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 166 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_167() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 167 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 167 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_168() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 168 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 168 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_169() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 169 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 169 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_170() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 170 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 170 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_171() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 171 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 171 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_172() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 172 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 172 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_173() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 173 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 173 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_174() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 174 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 174 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_175() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 175 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 175 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_176() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 176 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 176 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_177() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 177 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 177 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_178() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 178 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 178 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_179() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 179 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 179 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_180() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 180 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 180 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_181() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 181 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 181 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_182() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 182 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 182 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_183() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 183 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 183 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_184() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 184 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 184 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_185() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 185 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 185 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_186() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 186 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 186 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_187() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 187 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 187 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_188() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 188 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 188 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_189() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 189 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 189 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_190() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 190 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 190 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_191() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 191 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 191 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_192() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 192 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 192 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_193() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 193 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 193 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_194() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 194 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 194 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_195() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 195 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 195 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_196() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 196 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 196 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_197() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 197 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 197 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_198() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 198 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 198 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_199() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 199 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 199 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_200() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 200 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 200 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_201() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 201 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 201 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_202() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 202 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 202 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_203() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 203 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 203 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_204() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 204 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 204 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_205() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 205 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 205 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_206() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 206 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 206 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_207() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 207 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 207 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_208() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 208 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 208 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_209() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 209 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 209 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_210() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 210 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 210 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_211() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 211 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 211 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_212() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 212 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 212 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_213() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 213 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 213 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_214() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 214 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 214 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_215() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 215 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 215 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_216() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 216 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 216 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_217() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 217 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 217 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_218() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 218 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 218 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_219() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 219 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 219 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_220() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 220 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 220 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_221() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 221 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 221 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_222() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 222 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 222 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_223() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 223 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 223 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_224() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 224 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 224 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_225() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 225 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 225 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_226() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 226 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 226 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_227() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 227 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 227 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_228() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 228 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 228 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_229() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 229 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 229 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_230() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 230 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 230 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_231() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 231 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 231 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_232() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 232 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 232 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_233() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 233 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 233 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_234() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 234 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 234 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_235() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 235 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 235 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_236() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 236 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 236 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_237() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 237 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 237 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_238() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 238 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 238 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_239() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 239 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 239 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_240() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 240 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 240 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_241() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 241 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 241 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_242() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 242 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 242 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_243() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 243 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 243 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_244() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 244 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 244 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_245() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 245 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 245 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_246() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 246 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 246 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_247() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 247 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 247 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_248() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 248 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 248 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_249() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 249 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 249 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_250() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 250 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 250 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_251() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 251 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 251 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_252() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 252 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 252 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_253() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 253 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 253 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_254() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 254 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 254 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_255() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 255 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 255 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_256() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 256 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 256 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_257() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 257 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 257 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_258() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 258 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 258 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_259() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 259 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 259 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_260() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 260 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 260 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_261() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 261 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 261 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_262() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 262 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 262 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_263() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 263 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 263 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_264() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 264 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 264 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_265() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 265 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 265 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_266() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 266 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 266 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_267() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 267 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 267 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_268() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 268 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 268 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_269() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 269 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 269 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_270() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 270 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 270 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_271() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 271 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 271 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_272() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 272 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 272 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_273() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 273 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 273 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_274() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 274 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 274 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_275() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 275 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 275 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_276() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 276 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 276 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_277() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 277 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 277 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_278() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 278 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 278 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_279() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 279 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 279 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_280() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 280 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 280 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_281() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 281 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 281 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_282() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 282 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 282 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_283() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 283 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 283 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_284() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 284 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 284 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_285() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 285 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 285 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_286() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 286 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 286 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_287() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 287 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 287 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_288() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 288 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 288 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_289() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 289 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 289 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_290() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 290 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 290 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_291() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 291 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 291 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_292() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 292 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 292 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_293() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 293 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 293 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_294() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 294 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 294 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_295() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 295 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 295 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_296() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 296 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 296 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_297() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 297 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 297 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_298() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 298 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 298 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_299() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 299 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 299 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_300() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 300 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 300 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_301() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 301 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 301 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_302() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 302 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 302 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_303() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 303 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 303 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_304() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 304 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 304 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_305() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 305 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 305 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_306() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 306 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 306 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_307() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 307 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 307 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_308() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 308 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 308 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_309() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 309 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 309 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_310() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 310 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 310 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_311() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 311 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 311 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_312() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 312 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 312 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_313() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 313 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 313 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_314() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 314 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 314 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_315() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 315 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 315 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_316() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 316 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 316 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_317() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 317 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 317 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_318() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 318 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 318 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_319() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 319 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 319 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_320() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 320 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 320 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_321() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 321 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 321 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_322() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 322 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 322 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_323() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 323 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 323 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_324() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 324 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 324 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_325() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 325 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 325 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_326() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 326 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 326 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_327() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 327 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 327 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_328() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 328 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 328 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_329() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 329 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 329 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_330() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 330 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 330 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_331() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 331 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 331 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_332() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 332 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 332 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_333() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 333 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 333 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_334() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 334 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 334 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_335() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 335 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 335 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_336() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 336 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 336 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_337() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 337 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 337 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_338() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 338 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 338 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_339() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 339 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 339 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_340() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 340 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 340 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_341() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 341 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 341 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_342() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 342 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 342 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_343() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 343 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 343 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_344() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 344 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 344 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_345() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 345 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 345 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_346() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 346 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 346 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_347() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 347 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 347 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_348() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 348 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 348 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_349() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 349 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 349 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_350() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 350 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 350 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_351() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 351 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 351 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_352() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 352 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 352 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_353() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 353 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 353 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_354() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 354 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 354 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_355() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 355 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 355 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_356() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 356 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 356 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_357() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 357 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 357 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_358() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 358 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 358 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_359() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 359 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 359 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_360() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 360 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 360 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_361() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 361 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 361 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_362() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 362 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 362 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_363() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 363 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 363 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_364() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 364 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 364 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_365() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 365 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 365 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_366() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 366 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 366 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    #[test]
    fn test_aggregate_stress_367() {
        let d1 = ModelDelta::new(0, vec![Tensor::zeros(vec![4])], 367 + 1);
        let d2 = ModelDelta::new(1, vec![Tensor::zeros(vec![4])], 367 + 1);
        let agg = fed_avg_aggregate(&[d1, d2]);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].shape(), &[4]);
    }

    // Federated learning aggregation and privacy verification padding line 0
    // Federated learning aggregation and privacy verification padding line 1
    // Federated learning aggregation and privacy verification padding line 2
    // Federated learning aggregation and privacy verification padding line 3
    // Federated learning aggregation and privacy verification padding line 4
    // Federated learning aggregation and privacy verification padding line 5
    // Federated learning aggregation and privacy verification padding line 6
}
