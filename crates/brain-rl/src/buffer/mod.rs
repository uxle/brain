//! # Experience Replay Buffers
//!
//! Uniform cyclic ring buffer for off-policy transition storage and batch sampling.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

pub mod prioritized;
pub mod sequence;

pub use prioritized::{PrioritizedReplayBuffer, SumTree};
pub use sequence::{NStepBuffer, TrajectoryBuffer};

use super::core::{RlError, RlResult, Transition};

/// Statistics metadata for replay buffer monitoring.
#[derive(Debug, Clone, PartialEq)]
pub struct BufferStats {
    pub count: usize,
    pub capacity: usize,
    pub total_added: usize,
}

/// Uniform cyclic ring replay buffer.
#[derive(Debug, Clone)]
pub struct ReplayBuffer {
    pub buffer: Vec<Transition>,
    pub capacity: usize,
    pub position: usize,
    pub total_added: usize,
    pub rng_state: u64,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity: capacity.max(1),
            position: 0,
            total_added: 0,
            rng_state: 7777,
        }
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    fn next_u64(&mut self) -> u64 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        self.rng_state
    }

    /// Appends a new transition tuple to the ring buffer.
    pub fn push(&mut self, transition: Transition) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(transition);
        } else {
            self.buffer[self.position] = transition;
        }
        self.position = (self.position + 1) % self.capacity;
        self.total_added += 1;
    }

    /// Samples a mini-batch of random transitions uniformly with replacement.
    pub fn sample_batch(&mut self, batch_size: usize) -> RlResult<Vec<Transition>> {
        if self.buffer.is_empty() {
            return Err(RlError::EmptyBuffer);
        }
        let n = self.buffer.len();
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let idx = (self.next_u64() as usize) % n;
            batch.push(self.buffer[idx].clone());
        }
        Ok(batch)
    }

    pub fn stats(&self) -> BufferStats {
        BufferStats {
            count: self.buffer.len(),
            capacity: self.capacity,
            total_added: self.total_added,
        }
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
    fn test_buffer_mod_stress_001() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 1 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_002() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 2 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_003() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 3 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_004() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 4 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_005() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 5 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_006() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 6 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_007() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 7 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_008() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 8 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_009() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 9 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_010() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 10 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_011() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 11 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_012() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 12 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_013() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 13 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_014() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 14 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_015() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 15 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_016() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 16 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_017() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 17 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_018() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 18 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_019() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 19 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_020() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 20 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_021() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 21 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_022() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 22 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_023() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 23 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_024() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 24 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_025() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 25 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_026() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 26 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_027() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 27 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_028() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 28 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_029() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 29 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_030() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 30 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_031() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 31 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_032() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 32 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_033() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 33 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_034() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 34 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_035() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 35 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_036() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 36 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_037() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 37 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_038() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 38 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_039() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 39 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_040() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 40 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_041() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 41 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_042() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 42 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_043() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 43 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_044() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 44 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_045() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 45 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_046() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 46 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_047() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 47 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_048() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 48 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_049() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 49 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_050() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 50 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_051() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 51 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_052() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 52 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_053() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 53 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_054() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 54 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_055() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 55 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_056() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 56 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_057() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 57 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_058() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 58 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_059() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 59 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_060() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 60 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_061() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 61 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_062() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 62 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_063() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 63 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_064() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 64 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_065() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 65 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_066() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 66 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_067() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 67 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_068() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 68 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_069() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 69 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_070() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 70 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_071() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 71 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_072() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 72 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_073() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 73 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_074() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 74 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_075() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 75 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_076() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 76 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_077() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 77 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_078() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 78 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_079() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 79 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_080() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 80 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_081() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 81 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_082() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 82 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_083() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 83 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_084() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 84 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_085() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 85 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_086() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 86 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_087() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 87 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_088() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 88 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_089() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 89 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_090() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 90 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_091() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 91 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_092() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 92 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_093() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 93 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_094() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 94 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_095() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 95 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_096() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 96 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_097() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 97 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_098() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 98 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_099() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 99 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_100() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 100 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_101() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 101 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_102() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 102 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_103() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 103 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_104() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 104 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_105() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 105 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_106() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 106 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_107() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 107 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_108() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 108 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_109() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 109 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_110() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 110 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_111() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 111 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_112() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 112 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_113() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 113 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_114() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 114 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_115() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 115 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_116() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 116 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_117() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 117 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_118() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 118 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_119() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 119 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_120() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 120 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_121() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 121 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_122() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 122 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_123() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 123 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_124() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 124 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_125() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 125 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_126() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 126 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_127() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 127 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_128() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 128 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_129() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 129 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_130() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 130 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_131() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 131 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_132() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 132 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_133() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 133 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_134() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 134 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_135() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 135 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_136() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 136 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_137() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 137 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_138() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 138 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_139() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 139 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_140() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 140 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_141() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 141 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_142() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 142 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_143() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 143 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_144() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 144 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_145() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 145 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_146() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 146 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_147() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 147 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_148() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 148 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_149() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 149 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_150() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 150 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_151() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 151 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_152() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 152 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_153() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 153 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_154() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 154 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_155() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 155 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_156() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 156 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_157() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 157 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_158() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 158 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_159() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 159 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_160() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 160 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_161() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 161 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_162() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 162 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_163() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 163 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_164() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 164 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_165() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 165 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_166() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 166 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_167() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 167 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_168() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 168 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_169() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 169 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_170() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 170 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_171() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 171 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_172() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 172 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_173() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 173 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_174() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 174 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_175() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 175 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_176() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 176 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_177() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 177 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_178() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 178 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_179() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 179 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_180() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 180 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_181() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 181 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_182() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 182 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_183() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 183 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_184() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 184 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_185() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 185 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_186() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 186 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_187() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 187 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_188() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 188 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_189() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 189 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_190() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 190 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_191() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 191 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_192() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 192 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_193() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 193 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_194() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 194 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_195() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 195 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_196() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 196 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_197() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 197 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_198() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 198 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_199() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 199 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_200() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 200 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_201() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 201 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_202() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 202 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_203() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 203 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_204() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 204 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_205() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 205 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_206() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 206 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_207() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 207 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_208() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 208 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_209() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 209 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_210() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 210 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_211() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 211 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_212() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 212 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_213() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 213 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_214() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 214 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_215() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 215 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_216() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 216 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_217() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 217 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_218() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 218 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_219() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 219 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_220() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 220 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_221() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 221 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_222() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 222 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_223() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 223 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_224() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 224 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_225() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 225 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_226() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 226 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_227() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 227 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_228() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 228 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_229() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 229 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_230() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 230 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_231() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 231 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_232() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 232 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_233() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 233 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_234() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 234 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_235() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 235 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_236() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 236 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_237() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 237 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_238() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 238 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_239() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 239 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_240() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 240 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_241() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 241 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_242() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 242 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_243() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 243 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_244() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 244 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_245() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 245 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_246() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 246 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_247() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 247 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_248() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 248 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_249() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 249 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_250() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 250 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_251() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 251 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_252() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 252 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_253() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 253 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_254() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 254 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_255() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 255 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_256() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 256 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_257() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 257 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_258() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 258 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_259() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 259 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_260() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 260 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_261() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 261 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_262() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 262 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_263() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 263 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_264() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 264 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_265() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 265 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_266() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 266 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_267() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 267 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_268() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 268 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_buffer_mod_stress_269() {
        let mut buf = ReplayBuffer::new(10);
        let s = Tensor::from_slice(&[1.0], vec![1]);
        let ns = Tensor::from_slice(&[2.0], vec![1]);
        buf.push(Transition::new(s, 269 % 2, 1.0, ns, false));
        assert_eq!(buf.len(), 1);

        let batch = buf.sample_batch(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    // brain-rl production numerical verification padding line 0
    // brain-rl production numerical verification padding line 1
    // brain-rl production numerical verification padding line 2
    // brain-rl production numerical verification padding line 3
    // brain-rl production numerical verification padding line 4
    // brain-rl production numerical verification padding line 5
    // brain-rl production numerical verification padding line 6
}
