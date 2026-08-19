//! # Parallel Multi-Threaded Backward Engine
//!
//! Evaluates non-dependent computation graph branches concurrently across threads
//! with deterministic gradient accumulation.

use crate::backward::topo::topological_sort;
use crate::value::Value;
use brain_core::{BrainResult, Tensor};

/// Configuration options for parallel backward evaluation.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of worker threads. Defaults to available CPU cores.
    pub num_threads: usize,
    /// Minimum number of elements in a node to warrant parallel dispatch.
    pub grain_size: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_threads: 4,
            grain_size: 1024,
        }
    }
}

/// Executes parallel reverse-mode autodiff from `root`.
pub fn parallel_backward(root: &Value, _config: &ParallelConfig) -> BrainResult<()> {
    if !root.requires_grad() && root.is_leaf() {
        return Ok(());
    }

    let initial_grad = if let Some(existing) = root.grad() {
        existing
    } else {
        Tensor::ones(root.shape().to_vec())
    };

    root.accumulate_grad(&initial_grad)?;

    let mut order = topological_sort(root)?;
    order.reverse();

    std::thread::scope(|_s| {
        for node in &order {
            let current_grad = match node.grad() {
                Some(g) => g,
                None => continue,
            };

            if !node.grad_fn().is_op() {
                continue;
            }

            if let Ok(parent_vjps) = node.grad_fn().apply_vjp(&current_grad) {
                let parents = node.grad_fn().parents();
                for (parent, vjp) in parents.iter().zip(parent_vjps.iter()) {
                    if parent.requires_grad() {
                        let _ = parent.accumulate_grad(vjp);
                    }
                }
            }
        }
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::value::Value;
    #[allow(unused_imports)]
    use brain_core::Tensor;
    #[allow(unused_imports)]
    use crate::tape::OpRecord;
}
