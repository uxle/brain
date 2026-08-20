//! # Experience Replay Buffers
//!
//! Uniform cyclic ring buffer for off-policy transition storage and batch sampling.
#![allow(
    missing_docs,
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_is_multiple_of,
    clippy::manual_div_ceil,
    clippy::doc_markdown
)]

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
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant,
        clippy::needless_range_loop,
        clippy::manual_div_ceil,
        clippy::manual_is_multiple_of,
        clippy::too_many_arguments,
        clippy::doc_markdown,
        clippy::excessive_precision
    )]
    use super::*;
    use crate::a2c::*;
    use crate::actor_critic::*;
    use crate::agents::*;
    use crate::buffer::*;
    use crate::checkpoint::*;
    use crate::core::*;
    use crate::dqn::*;
    use crate::env::*;
    use crate::eval::*;
    use crate::policy::*;
    use crate::ppo::*;
    use crate::sac::*;
    use crate::trainer::*;
    use crate::utils::*;
    use crate::value::*;
    use crate::VERSION;
    use brain_core::Tensor;
}
