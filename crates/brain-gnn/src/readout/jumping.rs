//! # Jumping Knowledge Networks
//!
//! Jumping Knowledge (JK) aggregation across intermediate GNN layer representations.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Mode for Jumping Knowledge aggregation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JkMode {
    #[default]
    Concat,
    Max,
    Last,
}

/// Configuration for Jumping Knowledge aggregation.
#[derive(Debug, Clone)]
pub struct JkConfig {
    pub mode: JkMode,
}

impl Default for JkConfig {
    fn default() -> Self { Self { mode: JkMode::Concat } }
}

/// Jumping Knowledge layer aggregator.
pub struct JumpingKnowledge {
    pub config: JkConfig,
}

impl JumpingKnowledge {
    pub fn new(mode: JkMode) -> Self {
        Self { config: JkConfig { mode } }
    }

    pub fn aggregate(&self, layer_outputs: &[Tensor]) -> Tensor {
        if layer_outputs.is_empty() { return Tensor::zeros(vec![1]); }
        match self.config.mode {
            JkMode::Last => layer_outputs.last().unwrap().clone(),
            JkMode::Concat => {
                let mut concat_data = Vec::new();
                let num_nodes = layer_outputs[0].shape()[0];
                let feat_dims: Vec<usize> = layer_outputs.iter().map(|t| {
                    if t.shape().len() > 1 { t.shape()[1] } else { 1 }
                }).collect();

                let total_feat_dim: usize = feat_dims.iter().sum();

                for n in 0..num_nodes {
                    for (l, t) in layer_outputs.iter().enumerate() {
                        let dim = feat_dims[l];
                        let data = t.to_vec();
                        for d in 0..dim {
                            concat_data.push(data[n * dim + d]);
                        }
                    }
                }

                Tensor::from_vec(concat_data, vec![num_nodes, total_feat_dim])
            }
            JkMode::Max => {
                let num_nodes = layer_outputs[0].shape()[0];
                let dim = if layer_outputs[0].shape().len() > 1 { layer_outputs[0].shape()[1] } else { 1 };
                let mut max_data = vec![f64::NEG_INFINITY; num_nodes * dim];

                for t in layer_outputs {
                    let data = t.to_vec();
                    for idx in 0..num_nodes * dim {
                        if data[idx] > max_data[idx] {
                            max_data[idx] = data[idx];
                        }
                    }
                }

                Tensor::from_vec(max_data, vec![num_nodes, dim])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
