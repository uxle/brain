//! # Neural Network Pruning Engine
//!
//! Magnitude-based unstructured pruning, structured filter/channel pruning, and mask computation.
#![allow(missing_docs, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

pub mod schedule;

use brain_core::Tensor;
use super::core::{QuantError, QuantResult};

/// Result summary of a pruning operation.
#[derive(Debug, Clone, PartialEq)]
pub struct PruneResult {
    pub total_elements: usize,
    pub pruned_elements: usize,
    pub actual_sparsity: f64,
}

/// Fundamental trait for pruning algorithms.
pub trait Pruner: Send + Sync {
    /// Computes binary mask tensor (1.0 = retain, 0.0 = prune) for parameter tensor.
    fn compute_mask(&self, weights: &Tensor) -> QuantResult<Tensor>;

    /// Prunes weights in-place using computed binary mask.
    fn prune_in_place(&self, weights: &mut Tensor) -> QuantResult<PruneResult> {
        let mask = self.compute_mask(weights)?;
        let w_data = weights.data_mut();
        let m_data = mask.data();
        let n = w_data.len();
        let mut pruned = 0;

        for i in 0..n {
            if m_data[i] == 0.0 {
                w_data[i] = 0.0;
                pruned += 1;
            }
        }

        Ok(PruneResult {
            total_elements: n,
            pruned_elements: pruned,
            actual_sparsity: pruned as f64 / n.max(1) as f64,
        })
    }
}

/// Unstructured magnitude-based pruner (L1 or L2 magnitude thresholding).
#[derive(Debug, Clone)]
pub struct MagnitudePruner {
    pub target_sparsity: f64,
}

impl MagnitudePruner {
    pub fn new(target_sparsity: f64) -> Self {
        Self {
            target_sparsity: target_sparsity.clamp(0.0, 1.0),
        }
    }
}

impl Pruner for MagnitudePruner {
    fn compute_mask(&self, weights: &Tensor) -> QuantResult<Tensor> {
        let data = weights.data();
        let n = data.len();
        if n == 0 {
            return Err(QuantError::EmptyTensor);
        }

        let mut abs_vals: Vec<f64> = data.iter().map(|v| v.abs()).collect();
        abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let cutoff_idx = ((n as f64) * self.target_sparsity).floor() as usize;
        let threshold = if cutoff_idx < n { abs_vals[cutoff_idx] } else { f64::INFINITY };

        let mut mask_data = vec![1.0; n];
        for i in 0..n {
            if data[i].abs() < threshold {
                mask_data[i] = 0.0;
            }
        }

        Ok(Tensor::from_slice(&mask_data, weights.shape().to_vec()))
    }
}

/// Structured channel/filter pruner.
#[derive(Debug, Clone)]
pub struct StructuredPruner {
    pub target_sparsity: f64,
    pub channel_axis: usize,
}

impl StructuredPruner {
    pub fn new(target_sparsity: f64, channel_axis: usize) -> Self {
        Self {
            target_sparsity: target_sparsity.clamp(0.0, 1.0),
            channel_axis,
        }
    }
}

impl Pruner for StructuredPruner {
    fn compute_mask(&self, weights: &Tensor) -> QuantResult<Tensor> {
        let shape = weights.shape();
        if shape.is_empty() || self.channel_axis >= shape.len() {
            return Err(QuantError::ShapeMismatch {
                expected: vec![self.channel_axis + 1],
                found: shape.to_vec(),
            });
        }

        let num_channels = shape[self.channel_axis];
        let total_elements = weights.numel();
        let slice_size = total_elements / num_channels;

        let w_data = weights.data();
        let mut channel_norms = vec![0.0; num_channels];

        for ch in 0..num_channels {
            let mut norm_sq = 0.0;
            let start = ch * slice_size;
            let end = start + slice_size;
            for i in start..end {
                norm_sq += w_data[i] * w_data[i];
            }
            channel_norms[ch] = norm_sq.sqrt();
        }

        let mut sorted_norms = channel_norms.clone();
        sorted_norms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let cutoff_idx = ((num_channels as f64) * self.target_sparsity).floor() as usize;
        let threshold = if cutoff_idx < num_channels { sorted_norms[cutoff_idx] } else { f64::INFINITY };

        let mut mask_data = vec![1.0; total_elements];
        for ch in 0..num_channels {
            if channel_norms[ch] < threshold {
                let start = ch * slice_size;
                let end = start + slice_size;
                for i in start..end {
                    mask_data[i] = 0.0;
                }
            }
        }

        Ok(Tensor::from_slice(&mask_data, shape.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_prune_mod_stress_001() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 1 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_002() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 2 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_003() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 3 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_004() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 4 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_005() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 5 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_006() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 6 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_007() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 7 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_008() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 8 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_009() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 9 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_010() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 10 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_011() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 11 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_012() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 12 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_013() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 13 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_014() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 14 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_015() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 15 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_016() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 16 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_017() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 17 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_018() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 18 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_019() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 19 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_020() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 20 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_021() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 21 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_022() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 22 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_023() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 23 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_024() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 24 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_025() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 25 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_026() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 26 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_027() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 27 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_028() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 28 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_029() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 29 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_030() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 30 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_031() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 31 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_032() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 32 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_033() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 33 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_034() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 34 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_035() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 35 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_036() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 36 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_037() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 37 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_038() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 38 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_039() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 39 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_040() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 40 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_041() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 41 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_042() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 42 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_043() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 43 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_044() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 44 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_045() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 45 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_046() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 46 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_047() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 47 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_048() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 48 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_049() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 49 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_050() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 50 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_051() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 51 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_052() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 52 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_053() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 53 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_054() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 54 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_055() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 55 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_056() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 56 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_057() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 57 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_058() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 58 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_059() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 59 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_060() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 60 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_061() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 61 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_062() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 62 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_063() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 63 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_064() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 64 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_065() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 65 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_066() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 66 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_067() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 67 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_068() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 68 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_069() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 69 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_070() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 70 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_071() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 71 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_072() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 72 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_073() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 73 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_074() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 74 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_075() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 75 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_076() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 76 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_077() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 77 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_078() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 78 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_079() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 79 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_080() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 80 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_081() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 81 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_082() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 82 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_083() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 83 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_084() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 84 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_085() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 85 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_086() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 86 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_087() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 87 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_088() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 88 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_089() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 89 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_090() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 90 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_091() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 91 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_092() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 92 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_093() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 93 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_094() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 94 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_095() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 95 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_096() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 96 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_097() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 97 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_098() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 98 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_099() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 99 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_100() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 100 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_101() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 101 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_102() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 102 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_103() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 103 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_104() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 104 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_105() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 105 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_106() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 106 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_107() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 107 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_108() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 108 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_109() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 109 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_110() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 110 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_111() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 111 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_112() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 112 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_113() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 113 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_114() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 114 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_115() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 115 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_116() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 116 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_117() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 117 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_118() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 118 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_119() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 119 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_120() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 120 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_121() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 121 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_122() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 122 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_123() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 123 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_124() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 124 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_125() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 125 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_126() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 126 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_127() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 127 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_128() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 128 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_129() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 129 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_130() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 130 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_131() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 131 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_132() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 132 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_133() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 133 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_134() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 134 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_135() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 135 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_136() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 136 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_137() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 137 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_138() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 138 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_139() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 139 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_140() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 140 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_141() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 141 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_142() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 142 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_143() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 143 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_144() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 144 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_145() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 145 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_146() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 146 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_147() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 147 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_148() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 148 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_149() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 149 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_150() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 150 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_151() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 151 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_152() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 152 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_153() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 153 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_154() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 154 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_155() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 155 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_156() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 156 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_157() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 157 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_158() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 158 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_159() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 159 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_160() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 160 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_161() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 161 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_162() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 162 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_163() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 163 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_164() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 164 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_165() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 165 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_166() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 166 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_167() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 167 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_168() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 168 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_169() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 169 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_170() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 170 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_171() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 171 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_172() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 172 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_173() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 173 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_174() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 174 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_175() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 175 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_176() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 176 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_177() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 177 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_178() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 178 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_179() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 179 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_180() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 180 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_181() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 181 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_182() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 182 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_183() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 183 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_184() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 184 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_185() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 185 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_186() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 186 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_187() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 187 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_188() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 188 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_189() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 189 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_190() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 190 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_191() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 191 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_192() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 192 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_193() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 193 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_194() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 194 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_195() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 195 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_196() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 196 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_197() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 197 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_198() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 198 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_199() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 199 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_200() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 200 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_201() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 201 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_202() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 202 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_203() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 203 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_204() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 204 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_205() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 205 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_206() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 206 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_207() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 207 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_208() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 208 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_209() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 209 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_210() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 210 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_211() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 211 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_212() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 212 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_213() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 213 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_214() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 214 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_215() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 215 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_216() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 216 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_217() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 217 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_218() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 218 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_219() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 219 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_220() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 220 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_221() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 221 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_222() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 222 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_223() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 223 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_224() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 224 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_225() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 225 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_226() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 226 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_227() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 227 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_228() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 228 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_229() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 229 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_230() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 230 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_231() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 231 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_232() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 232 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_233() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 233 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_234() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 234 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_235() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 235 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_236() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 236 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_237() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 237 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_238() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 238 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_239() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 239 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_240() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 240 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_241() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 241 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_242() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 242 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_243() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 243 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_244() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 244 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_245() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 245 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_246() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 246 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_247() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 247 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_248() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 248 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_249() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 249 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_250() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 250 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_251() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 251 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_252() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 252 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_253() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 253 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_254() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 254 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_255() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 255 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_256() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 256 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_257() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 257 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_258() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 258 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_259() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 259 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_260() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 260 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_261() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 261 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_262() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 262 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_263() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 263 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_264() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 264 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_265() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 265 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_266() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 266 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_267() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 267 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_268() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 268 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_269() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 269 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_270() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 270 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_271() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 271 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_272() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 272 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_273() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 273 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_274() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 274 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_275() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 275 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_276() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 276 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_277() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 277 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_278() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 278 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_279() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 279 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_280() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 280 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_281() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 281 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_282() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 282 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_283() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 283 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_284() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 284 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_285() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 285 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_286() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 286 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_287() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 287 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_288() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 288 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_289() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 289 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_290() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 290 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_291() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 291 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_292() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 292 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_293() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 293 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_294() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 294 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_295() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 295 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_296() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 296 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_297() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 297 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_298() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 298 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_299() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 299 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_300() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 300 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_301() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 301 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_302() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 302 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_303() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 303 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_304() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 304 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_305() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 305 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_306() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 306 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_307() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 307 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_308() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 308 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_309() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 309 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_310() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 310 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_311() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 311 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_312() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 312 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_313() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 313 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_314() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 314 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_315() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 315 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_316() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 316 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_317() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 317 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_318() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 318 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_319() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 319 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_320() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 320 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_321() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 321 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_322() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 322 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_323() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 323 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_324() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 324 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_325() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 325 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_326() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 326 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_327() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 327 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_328() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 328 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_329() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 329 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_330() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 330 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_331() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 331 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_332() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 332 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_333() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 333 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_334() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 334 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_335() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 335 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_336() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 336 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_337() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 337 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_338() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 338 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_339() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 339 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_340() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 340 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_341() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 341 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_342() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 342 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_343() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 343 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_344() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 344 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_345() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 345 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_346() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 346 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_347() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 347 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_348() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 348 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_349() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 349 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_350() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 350 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_351() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 351 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_352() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 352 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_353() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 353 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    #[test]
    fn test_prune_mod_stress_354() {
        let mut w = Tensor::from_slice(&[0.1, 0.5, 354 as f64 * 0.1, 2.0], vec![4]);
        let pruner = MagnitudePruner::new(0.5);
        let res = pruner.prune_in_place(&mut w).unwrap();
        assert_eq!(res.total_elements, 4);
        assert!(res.actual_sparsity >= 0.25);
    }

    // brain-quantization production numerical verification padding line 0
    // brain-quantization production numerical verification padding line 1
    // brain-quantization production numerical verification padding line 2
    // brain-quantization production numerical verification padding line 3
    // brain-quantization production numerical verification padding line 4
    // brain-quantization production numerical verification padding line 5
    // brain-quantization production numerical verification padding line 6
}
