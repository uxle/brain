//! # Selective Activation Checkpointing
//!
//! Configurable policies for deciding which intermediate activations to retain vs recompute.

use crate::value::Value;
use brain_core::BrainResult;

/// Strategy determining which layers or activation sizes are retained in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointPolicy {
    /// Discard all non-leaf forward activations and recompute everything.
    All,
    /// Only checkpoint operations exceeding a certain tensor size in bytes.
    Selective { threshold_elements: usize },
    /// Do not checkpoint (standard autograd behavior).
    None,
}

/// Checkpoints a functional closure `f` with given `inputs`.
pub fn checkpoint<F>(f: F, inputs: &[&Value]) -> BrainResult<Vec<Value>>
where
    F: Fn(&[&Value]) -> BrainResult<Vec<Value>>,
{
    let any_requires_grad = inputs.iter().any(|v| v.requires_grad());
    if !any_requires_grad {
        let detached_inputs: Vec<Value> = inputs.iter().map(|&v| v.detach()).collect();
        let detached_refs: Vec<&Value> = detached_inputs.iter().collect();
        return f(&detached_refs);
    }

    f(inputs)
}
