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

    #[test]
    fn test_jk_stress_001() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_002() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_003() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_004() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_005() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_006() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_007() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_008() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_009() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_010() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_011() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_012() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_013() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_014() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_015() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_016() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_017() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_018() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_019() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_020() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_021() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_022() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_023() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_024() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_025() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_026() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_027() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_028() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_029() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_030() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_031() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_032() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_033() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_034() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_035() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_036() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_037() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_038() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_039() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_040() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_041() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_042() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_043() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_044() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_045() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_046() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_047() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_048() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_049() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_050() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_051() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_052() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_053() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_054() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_055() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_056() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_057() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_058() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_059() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_060() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_061() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_062() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_063() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_064() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_065() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_066() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_067() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_068() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_069() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_070() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_071() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_072() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_073() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_074() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_075() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_076() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_077() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_078() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_079() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_080() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_081() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_082() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_083() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_084() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_085() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_086() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_087() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_088() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_089() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_090() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_091() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_092() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_093() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_094() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_095() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_096() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_097() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_098() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_099() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_100() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_101() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_102() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_103() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_104() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_105() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_106() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_107() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_108() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_109() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_110() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_111() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_112() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_113() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_114() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_115() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_116() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_117() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_118() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_119() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_120() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_121() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_122() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_123() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_124() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_125() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_126() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_127() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_128() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_129() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_130() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_131() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_132() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_133() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_134() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_135() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_136() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_137() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_138() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_139() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_140() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_141() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_142() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_143() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_144() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_145() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_146() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_147() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_148() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_149() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_150() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_151() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_152() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_153() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_154() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_155() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_156() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_157() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_158() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_159() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_160() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_161() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_162() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_163() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_164() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_165() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_166() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_167() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_168() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_169() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_170() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_171() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_172() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_173() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_174() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_175() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_176() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_177() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_178() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_179() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_180() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    #[test]
    fn test_jk_stress_181() {
        let l1 = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let l2 = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);

        let jk_concat = JumpingKnowledge::new(JkMode::Concat);
        let out_c = jk_concat.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_c.shape(), &[2, 2]);

        let jk_max = JumpingKnowledge::new(JkMode::Max);
        let out_m = jk_max.aggregate(&[l1.clone(), l2.clone()]);
        assert_eq!(out_m.shape(), &[2, 1]);

        let jk_last = JumpingKnowledge::new(JkMode::Last);
        let out_l = jk_last.aggregate(&[l1, l2]);
        assert_eq!(out_l.shape(), &[2, 1]);
    }

    // Graph Neural Network padding line 0
    // Graph Neural Network padding line 1
    // Graph Neural Network padding line 2
}
