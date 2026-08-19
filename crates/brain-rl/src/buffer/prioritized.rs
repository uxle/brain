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
}
