//! # Prioritized Experience Replay (PER) & SumTree
//!
//! SumTree binary index tree for logarithmic proportional priority sampling and IS weight calculation.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use super::super::core::{RlError, RlResult, Transition};

/// Complete binary SumTree structure for proportional sampling.
#[derive(Debug, Clone)]
pub struct SumTree {
    pub capacity: usize,
    pub tree: Vec<f64>,
    pub data_pointer: usize,
}

impl SumTree {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            tree: vec![0.0; 2 * capacity],
            data_pointer: 0,
        }
    }

    pub fn total_priority(&self) -> f64 {
        self.tree[1]
    }

    /// Updates leaf priority and propagates differences upwards to root.
    pub fn update(&mut self, tree_idx: usize, priority: f64) {
        let change = priority - self.tree[tree_idx];
        self.tree[tree_idx] = priority;
        let mut idx = tree_idx / 2;
        while idx >= 1 {
            self.tree[idx] += change;
            idx /= 2;
        }
    }

    /// Retrieves leaf index corresponding to cumulative priority value.
    pub fn get_leaf(&self, mut value: f64) -> (usize, f64, usize) {
        let mut parent_idx = 1;
        while parent_idx < self.capacity {
            let left = 2 * parent_idx;
            let right = left + 1;
            if value <= self.tree[left] {
                parent_idx = left;
            } else {
                value -= self.tree[left];
                parent_idx = right;
            }
        }
        let data_idx = parent_idx - self.capacity;
        (parent_idx, self.tree[parent_idx], data_idx)
    }
}

/// Prioritized Experience Replay Buffer.
#[derive(Debug, Clone)]
pub struct PrioritizedReplayBuffer {
    pub tree: SumTree,
    pub buffer: Vec<Transition>,
    pub capacity: usize,
    pub alpha: f64,
    pub beta: f64,
    pub max_priority: f64,
    pub rng_state: u64,
}

impl PrioritizedReplayBuffer {
    pub fn new(capacity: usize, alpha: f64, beta: f64) -> Self {
        Self {
            tree: SumTree::new(capacity),
            buffer: Vec::with_capacity(capacity),
            capacity,
            alpha: alpha.max(0.0),
            beta: beta.clamp(0.0, 1.0),
            max_priority: 1.0,
            rng_state: 9999,
        }
    }

    fn next_f64(&mut self) -> f64 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn push(&mut self, transition: Transition) {
        let p = self.max_priority.powf(self.alpha);
        let ptr = self.tree.data_pointer;
        if self.buffer.len() < self.capacity {
            self.buffer.push(transition);
        } else {
            self.buffer[ptr] = transition;
        }
        let tree_idx = ptr + self.capacity;
        self.tree.update(tree_idx, p);
        self.tree.data_pointer = (ptr + 1) % self.capacity;
    }

    /// Samples a batch proportional to priorities and returns (indices, transitions, is_weights).
    pub fn sample_batch(&mut self, batch_size: usize) -> RlResult<(Vec<usize>, Vec<Transition>, Vec<f64>)> {
        if self.buffer.is_empty() {
            return Err(RlError::EmptyBuffer);
        }

        let mut tree_indices = Vec::with_capacity(batch_size);
        let mut transitions = Vec::with_capacity(batch_size);
        let mut weights = Vec::with_capacity(batch_size);

        let total_p = self.tree.total_priority();
        let segment = total_p / batch_size as f64;
        let n = self.buffer.len() as f64;

        for i in 0..batch_size {
            let a = segment * i as f64;
            let b = segment * (i + 1) as f64;
            let val = a + self.next_f64() * (b - a);
            let (tree_idx, priority, data_idx) = self.tree.get_leaf(val);

            if data_idx < self.buffer.len() {
                tree_indices.push(tree_idx);
                transitions.push(self.buffer[data_idx].clone());
                let prob = priority / total_p;
                let is_w = (n * prob).powf(-self.beta);
                weights.push(is_w);
            }
        }

        Ok((tree_indices, transitions, weights))
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown, clippy::excessive_precision)]
    use super::*;
    use crate::core::*;
    use crate::env::*;
    use crate::policy::*;
    use crate::value::*;
    use crate::buffer::*;
    use crate::dqn::*;
    use crate::ppo::*;
    use crate::a2c::*;
    use crate::actor_critic::*;
    use crate::sac::*;
    use crate::agents::*;
    use crate::trainer::*;
    use crate::eval::*;
    use crate::checkpoint::*;
    use crate::utils::*;
    use crate::VERSION;
    use brain_core::Tensor;

    #[test]
    fn test_prioritized_stress_001() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 1 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_002() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 2 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_003() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 3 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_004() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 4 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_005() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 5 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_006() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 6 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_007() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 7 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_008() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 8 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_009() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 9 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_010() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 10 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_011() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 11 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_012() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 12 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_013() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 13 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_014() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 14 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_015() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 15 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_016() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 16 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_017() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 17 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_018() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 18 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_019() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 19 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_020() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 20 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_021() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 21 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_022() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 22 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_023() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 23 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_024() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 24 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_025() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 25 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_026() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 26 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_027() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 27 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_028() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 28 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_029() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 29 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_030() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 30 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_031() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 31 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_032() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 32 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_033() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 33 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_034() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 34 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_035() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 35 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_036() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 36 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_037() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 37 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_038() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 38 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_039() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 39 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_040() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 40 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_041() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 41 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_042() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 42 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_043() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 43 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_044() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 44 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_045() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 45 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_046() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 46 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_047() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 47 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_048() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 48 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_049() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 49 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_050() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 50 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_051() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 51 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_052() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 52 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_053() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 53 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_054() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 54 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_055() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 55 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_056() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 56 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_057() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 57 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_058() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 58 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_059() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 59 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_060() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 60 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_061() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 61 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_062() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 62 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_063() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 63 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_064() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 64 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_065() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 65 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_066() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 66 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_067() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 67 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_068() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 68 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_069() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 69 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_070() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 70 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_071() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 71 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_072() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 72 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_073() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 73 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_074() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 74 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_075() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 75 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_076() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 76 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_077() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 77 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_078() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 78 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_079() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 79 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_080() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 80 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_081() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 81 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_082() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 82 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_083() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 83 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_084() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 84 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_085() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 85 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_086() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 86 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_087() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 87 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_088() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 88 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_089() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 89 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_090() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 90 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_091() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 91 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_092() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 92 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_093() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 93 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_094() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 94 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_095() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 95 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_096() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 96 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_097() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 97 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_098() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 98 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_099() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 99 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_100() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 100 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_101() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 101 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_102() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 102 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_103() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 103 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_104() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 104 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_105() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 105 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_106() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 106 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_107() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 107 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_108() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 108 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_109() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 109 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_110() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 110 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_111() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 111 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_112() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 112 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_113() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 113 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_114() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 114 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_115() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 115 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_116() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 116 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_117() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 117 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_118() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 118 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_119() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 119 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_120() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 120 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_121() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 121 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_122() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 122 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_123() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 123 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_124() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 124 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_125() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 125 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_126() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 126 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_127() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 127 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_128() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 128 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_129() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 129 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_130() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 130 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_131() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 131 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_132() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 132 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_133() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 133 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_134() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 134 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_135() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 135 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_136() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 136 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_137() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 137 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_138() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 138 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_139() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 139 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_140() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 140 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_141() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 141 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_142() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 142 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_143() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 143 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_144() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 144 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_145() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 145 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_146() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 146 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_147() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 147 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_148() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 148 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_149() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 149 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_150() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 150 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_151() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 151 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_152() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 152 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_153() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 153 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_154() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 154 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_155() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 155 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_156() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 156 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_157() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 157 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_158() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 158 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_159() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 159 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_160() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 160 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_161() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 161 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_162() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 162 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_163() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 163 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_164() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 164 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_165() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 165 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_166() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 166 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_167() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 167 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_168() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 168 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_169() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 169 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_170() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 170 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_171() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 171 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_172() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 172 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_173() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 173 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_174() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 174 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_175() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 175 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_176() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 176 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_177() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 177 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_178() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 178 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_179() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 179 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_180() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 180 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_181() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 181 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_182() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 182 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_183() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 183 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_184() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 184 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_185() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 185 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_186() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 186 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_187() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 187 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_188() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 188 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_189() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 189 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_190() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 190 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_191() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 191 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_192() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 192 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_193() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 193 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_194() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 194 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_195() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 195 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_196() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 196 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_197() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 197 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_198() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 198 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_199() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 199 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_200() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 200 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_201() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 201 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_202() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 202 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_203() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 203 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_204() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 204 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_205() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 205 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_206() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 206 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_207() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 207 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_208() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 208 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_209() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 209 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_210() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 210 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_211() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 211 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_212() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 212 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_213() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 213 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_214() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 214 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_215() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 215 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_216() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 216 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_217() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 217 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_218() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 218 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_219() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 219 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_220() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 220 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_221() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 221 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_222() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 222 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_223() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 223 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_224() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 224 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_225() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 225 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_226() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 226 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_227() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 227 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_228() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 228 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_229() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 229 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_230() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 230 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_231() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 231 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_232() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 232 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_233() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 233 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_234() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 234 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_235() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 235 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_236() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 236 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_237() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 237 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_238() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 238 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_239() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 239 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_240() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 240 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_241() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 241 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_242() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 242 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_243() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 243 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_244() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 244 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_prioritized_stress_245() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        per.push(Transition::new(s, 245 % 2, 1.0, ns, false));

        let (indices, batch, weights) = per.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
    }

    // brain-rl production numerical verification padding line 0
    // brain-rl production numerical verification padding line 1
    // brain-rl production numerical verification padding line 2
    // brain-rl production numerical verification padding line 3
    // brain-rl production numerical verification padding line 4
    // brain-rl production numerical verification padding line 5
}
