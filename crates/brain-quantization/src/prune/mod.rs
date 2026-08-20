//! # Neural Network Pruning Engine
//!
//! Magnitude-based unstructured pruning, structured filter/channel pruning, and mask computation.
#![allow(
    missing_docs,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_is_multiple_of,
    clippy::manual_div_ceil,
    clippy::doc_markdown
)]

pub mod schedule;

use super::core::{QuantError, QuantResult};
use brain_core::Tensor;

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
        let threshold = if cutoff_idx < n {
            abs_vals[cutoff_idx]
        } else {
            f64::INFINITY
        };

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
        let threshold = if cutoff_idx < num_channels {
            sorted_norms[cutoff_idx]
        } else {
            f64::INFINITY
        };

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
