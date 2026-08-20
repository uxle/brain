//! # Collective Communication Primitives
//!
//! Provides the primary [`CollectiveOp`] trait, allreduce, broadcast, allgather, and reduce-scatter operations.

pub mod allreduce;
pub mod ring;
pub mod tree;

pub use allreduce::{
    execute_allreduce, ring_allreduce_simulate, AllReduceAlgorithm, AllReduceConfig,
};
pub use ring::RingTopology;
pub use tree::TreeTopology;

use brain_core::Tensor;

/// Abstract collective communication operation trait.
pub trait CollectiveOp: Send + Sync {
    fn allreduce(&self, tensor: &Tensor) -> Tensor;
    fn broadcast(&self, tensor: &Tensor, root: usize) -> Tensor;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
